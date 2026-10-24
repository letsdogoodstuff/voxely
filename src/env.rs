//! Environment lighting. Every sky mode (uniform, atmospheric, image based)
//! is baked into one equirectangular radiance map so the path tracer has a
//! single, importance-sampled code path for it.
//!
//! Map layout: row 0 looks straight up (+Z), columns sweep azimuth from +X
//! towards +Y. `w` of each texel holds its sampling weight, normalised so the
//! solid-angle pdf of a direction is `w / (2 pi^2 sin(theta))`.

use std::f32::consts::PI;
use std::path::Path;

use anyhow::{bail, Context, Result};
use glam::Vec3;

use crate::scene::RenderSettings;

#[derive(Clone)]
pub struct EnvMap {
    pub width: u32,
    pub height: u32,
    /// Radiance rgb + sampling weight.
    pub texels: Vec<[f32; 4]>,
    /// Marginal CDF over rows (`height` entries) followed by each row's
    /// conditional CDF over columns (`width * height` entries).
    pub cdf: Vec<f32>,
}

impl EnvMap {
    pub fn from_radiance(width: u32, height: u32, rgb: &[Vec3]) -> Self {
        let (w, h) = (width as usize, height as usize);
        let lum = |c: Vec3| (c.dot(Vec3::new(0.2126, 0.7152, 0.0722))).max(0.0);
        // Pixel weight: luminance times the solid angle it covers, with a small
        // floor so every direction stays reachable.
        let mut f = vec![0f32; w * h];
        for y in 0..h {
            let s = (PI * (y as f32 + 0.5) / h as f32).sin();
            for x in 0..w {
                f[y * w + x] = (lum(rgb[y * w + x]) + 1e-4) * s;
            }
        }
        let total: f32 = f.iter().sum();
        let mut cdf = vec![0f32; h + w * h];
        let mut acc_rows = 0f32;
        for y in 0..h {
            let row = &f[y * w..(y + 1) * w];
            let row_sum: f32 = row.iter().sum();
            let mut acc = 0f32;
            for x in 0..w {
                acc += row[x];
                cdf[h + y * w + x] = if row_sum > 0.0 { acc / row_sum } else { (x + 1) as f32 / w as f32 };
            }
            acc_rows += row_sum;
            cdf[y] = acc_rows / total;
        }
        let texels = (0..w * h)
            .map(|i| {
                let c = rgb[i];
                [c.x, c.y, c.z, f[i] / total * (w * h) as f32]
            })
            .collect();
        EnvMap { width, height, texels, cdf }
    }

    pub fn uniform(c: Vec3) -> Self {
        Self::from_radiance(8, 4, &[c; 32])
    }

    /// Load an image and scale it; `rotation` is a fraction of a full turn.
    pub fn from_hdr(path: &Path, intensity: f32, rotation: f32) -> Result<Self> {
        let (w, h, px) = load_hdr(path)?;
        let shift = ((rotation.rem_euclid(1.0)) * w as f32).round() as usize;
        let mut rgb = vec![Vec3::ZERO; px.len()];
        for y in 0..h {
            for x in 0..w {
                // Image u increases with azimuth measured clockwise from above,
                // so mirror it to match our counter-clockwise convention.
                let src = y * w + (w - 1 - x + shift) % w;
                rgb[y * w + x] = px[src] * intensity;
            }
        }
        Ok(Self::from_radiance(w as u32, h as u32, &rgb))
    }

    pub fn from_settings(s: &RenderSettings) -> Self {
        match s.env_mode {
            EnvMode::Image => {
                if let Some(path) = &s.ibl_path {
                    match Self::from_hdr(path, s.ibl_intensity, s.ibl_rotation) {
                        Ok(m) => return m,
                        Err(e) => log::warn!("IBL {}: {e:#}", path.display()),
                    }
                }
                Self::uniform(s.sky_color * s.sky_intensity)
            }
            EnvMode::Atmosphere => bake_atmosphere(s, 256, 128),
            EnvMode::Uniform => Self::uniform(s.sky_color * s.sky_intensity),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EnvMode {
    Uniform,
    Atmosphere,
    Image,
}

// ------------------------------------------------------------ Radiance .hdr

fn rgbe(e: [u8; 4]) -> Vec3 {
    if e[3] == 0 {
        return Vec3::ZERO;
    }
    let f = 2f32.powi(e[3] as i32 - 136);
    Vec3::new(e[0] as f32, e[1] as f32, e[2] as f32) * f
}

/// Minimal Radiance RGBE reader (flat and new-style RLE scanlines).
pub fn load_hdr(path: &Path) -> Result<(usize, usize, Vec<Vec3>)> {
    let b = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mut o = 0;
    let mut line = || -> Result<String> {
        let end = b[o..].iter().position(|&c| c == b'\n').context("truncated header")?;
        let s = String::from_utf8_lossy(&b[o..o + end]).into_owned();
        o += end + 1;
        Ok(s)
    };
    if !line()?.starts_with("#?") {
        bail!("not a Radiance file");
    }
    loop {
        let l = line()?;
        if l.is_empty() {
            break;
        }
        if l.starts_with("FORMAT=") && l != "FORMAT=32-bit_rle_rgbe" {
            bail!("unsupported {l}");
        }
    }
    let res = line()?;
    let p: Vec<&str> = res.split_whitespace().collect();
    if p.len() != 4 || p[0] != "-Y" || p[2] != "+X" {
        bail!("unsupported orientation {res}");
    }
    let (h, w): (usize, usize) = (p[1].parse()?, p[3].parse()?);
    let mut out = Vec::with_capacity(w * h);
    let mut scan = vec![[0u8; 4]; w];
    for _ in 0..h {
        let rle = b.get(o..o + 4).context("truncated pixels")?;
        if w >= 8 && w < 32768 && rle[0] == 2 && rle[1] == 2 && rle[2] & 0x80 == 0 {
            o += 4;
            for ch in 0..4 {
                let mut x = 0;
                while x < w {
                    let n = *b.get(o).context("truncated rle")? as usize;
                    o += 1;
                    if n > 128 {
                        let v = *b.get(o).context("truncated rle")?;
                        o += 1;
                        for _ in 0..n - 128 {
                            if x < w {
                                scan[x][ch] = v;
                            }
                            x += 1;
                        }
                    } else {
                        for _ in 0..n {
                            if x < w {
                                scan[x][ch] = *b.get(o).context("truncated rle")?;
                            }
                            o += 1;
                            x += 1;
                        }
                    }
                }
            }
        } else {
            for px in scan.iter_mut() {
                let s = b.get(o..o + 4).context("truncated pixels")?;
                *px = [s[0], s[1], s[2], s[3]];
                o += 4;
            }
        }
        out.extend(scan.iter().map(|&e| rgbe(e)));
    }
    Ok((w, h, out))
}

// ------------------------------------------------------------ atmosphere

const EARTH_R: f32 = 6360e3;
const ATMO_R: f32 = 6420e3;
const H_RAYLEIGH: f32 = 8e3;
const H_MIE: f32 = 1.2e3;

struct Atmo {
    rayleigh: Vec3,
    mie: Vec3,
    ozone: Vec3,
    g: f32,
}

impl Atmo {
    fn new(s: &RenderSettings) -> Self {
        // UI densities map to scattering coefficients around Earth's values
        // (Rayleigh blue ~33e-6/m, Mie ~21e-6/m at the default 0.4).
        Atmo {
            rayleigh: s.rayleigh_color * s.rayleigh_density * 8.3e-5,
            mie: s.mie_color * s.mie_density * 5.0e-5,
            ozone: (Vec3::ONE - s.ozone_color) * s.ozone_density * 2.0e-6,
            g: s.mie_g,
        }
    }

    /// Extinction-weighted optical depth from `p` along `d` to the top.
    fn optical_depth(&self, p: Vec3, d: Vec3, steps: u32) -> Option<Vec3> {
        let t = ray_sphere(p, d, ATMO_R)?;
        if ray_sphere(p, d, EARTH_R).is_some_and(|e| e > 0.0) {
            return None;
        }
        let dt = t / steps as f32;
        let (mut r, mut m, mut o) = (0f32, 0f32, 0f32);
        for i in 0..steps {
            let h = (p + d * (dt * (i as f32 + 0.5))).length() - EARTH_R;
            r += (-h / H_RAYLEIGH).exp() * dt;
            m += (-h / H_MIE).exp() * dt;
            o += ozone_profile(h) * dt;
        }
        Some(self.rayleigh * r + self.mie * 1.1 * m + self.ozone * o)
    }

    /// Single-scattering sky radiance seen from the ground along `d`, per unit
    /// sun irradiance.
    fn sky(&self, d: Vec3, sun: Vec3) -> Vec3 {
        let p = Vec3::new(0.0, 0.0, EARTH_R + 1.0);
        let dir = Vec3::new(d.x, d.y, d.z.max(0.0)).normalize();
        let Some(t) = ray_sphere(p, dir, ATMO_R) else { return Vec3::ZERO };
        let steps = 24;
        let dt = t / steps as f32;
        let mu = dir.dot(sun);
        let phase_r = 3.0 / (16.0 * PI) * (1.0 + mu * mu);
        let g = self.g;
        let phase_m = 3.0 / (8.0 * PI) * ((1.0 - g * g) * (1.0 + mu * mu))
            / ((2.0 + g * g) * (1.0 + g * g - 2.0 * g * mu).powf(1.5));
        let (mut sum_r, mut sum_m) = (Vec3::ZERO, Vec3::ZERO);
        let mut depth = Vec3::ZERO;
        for i in 0..steps {
            let q = p + dir * (dt * (i as f32 + 0.5));
            let h = q.length() - EARTH_R;
            let (hr, hm) = ((-h / H_RAYLEIGH).exp() * dt, (-h / H_MIE).exp() * dt);
            depth += self.rayleigh * hr + self.mie * 1.1 * hm + self.ozone * ozone_profile(h) * dt;
            let Some(light) = self.optical_depth(q, sun, 8) else { continue };
            let att = (-(depth + light)).exp();
            sum_r += att * hr;
            sum_m += att * hm;
        }
        sum_r * self.rayleigh * phase_r + sum_m * self.mie * phase_m
    }

    /// Fraction of sunlight reaching the ground.
    pub fn transmittance(&self, sun: Vec3) -> Vec3 {
        let p = Vec3::new(0.0, 0.0, EARTH_R + 1.0);
        let s = Vec3::new(sun.x, sun.y, sun.z.max(0.01)).normalize();
        self.optical_depth(p, s, 32).map(|d| (-d).exp()).unwrap_or(Vec3::ZERO)
    }
}

fn ozone_profile(h: f32) -> f32 {
    (1.0 - (h - 25e3).abs() / 15e3).max(0.0)
}

fn ray_sphere(p: Vec3, d: Vec3, r: f32) -> Option<f32> {
    let b = p.dot(d);
    let c = p.dot(p) - r * r;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b + disc.sqrt();
    (t > 0.0).then_some(t)
}

/// Color the sun takes on after passing through the atmosphere.
pub fn sun_transmittance(s: &RenderSettings) -> Vec3 {
    if s.env_mode != EnvMode::Atmosphere {
        return Vec3::ONE;
    }
    Atmo::new(s).transmittance(s.sun_dir())
}

pub fn bake_atmosphere(s: &RenderSettings, w: u32, h: u32) -> EnvMap {
    let atmo = Atmo::new(s);
    let sun = s.sun_dir();
    // Scale so the sky brightness tracks the sun intensity slider.
    let k = s.sun_color * s.sun_intensity * 20.0;
    let mut rgb = vec![Vec3::ZERO; (w * h) as usize];
    for y in 0..h {
        let theta = PI * (y as f32 + 0.5) / h as f32;
        for x in 0..w {
            let phi = 2.0 * PI * (x as f32 + 0.5) / w as f32;
            let d = Vec3::new(theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos());
            let mut c = atmo.sky(d, sun) * k;
            if d.z < 0.0 {
                // Below the horizon: fade the horizon color towards dark.
                c *= (1.0 + d.z * 2.0).max(0.2);
            }
            rgb[(y * w + x) as usize] = c;
        }
    }
    EnvMap::from_radiance(w, h, &rgb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_pdf_integrates_to_one() {
        let m = EnvMap::uniform(Vec3::ONE);
        // Integrate pdf over the sphere numerically.
        let n = 400;
        let mut sum = 0.0f64;
        for y in 0..n {
            let theta = PI * (y as f32 + 0.5) / n as f32;
            for x in 0..n {
                let ty = ((theta / PI) * m.height as f32) as usize;
                let tx = (x * m.width as usize) / n;
                let w = m.texels[ty * m.width as usize + tx][3];
                let pdf = w / (2.0 * PI * PI * theta.sin());
                sum += (pdf * theta.sin() * (PI / n as f32) * (2.0 * PI / n as f32)) as f64;
            }
        }
        assert!((sum - 1.0).abs() < 1e-3, "{sum}");
    }

    #[test]
    fn loads_sample_hdr() {
        let p = Path::new(env!("HOME")).join("Downloads/MagicaVoxel-0.99.6.2-macos-10.15/ibl/HDR_041_Path_Env.hdr");
        if !p.exists() {
            return;
        }
        let (w, h, px) = load_hdr(&p).unwrap();
        assert_eq!((w, h), (512, 256));
        assert!(px.iter().all(|c| c.is_finite()));
        assert!(px.iter().any(|c| c.x > 0.0));
    }
}
