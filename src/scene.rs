//! Flattens a `VoxFile` scene graph into a dense voxel volume plus the
//! material table and render settings the GPU path tracer consumes.

use anyhow::{bail, Result};
use glam::{IVec3, Mat3, Vec3};

use std::path::PathBuf;

use crate::env::EnvMode;
use crate::vox::{Dict, Node, VoxFile};

/// Edge length of the coarse occupancy bricks used to skip empty space.
pub const BRICK: i32 = 8;

pub struct Volume {
    pub dims: IVec3,
    /// Color index per cell, x-fastest then y then z.
    pub cells: Vec<u8>,
    /// 1 if the brick may contain a voxel (stays set after erasing, which
    /// only costs a little traversal speed).
    pub bricks: Vec<u8>,
    pub brick_dims: IVec3,
}

impl Volume {
    pub fn new(dims: IVec3) -> Self {
        let brick_dims = (dims + IVec3::splat(BRICK - 1)) / BRICK;
        Volume {
            dims,
            cells: vec![0; (dims.x * dims.y * dims.z) as usize],
            bricks: vec![0; (brick_dims.x * brick_dims.y * brick_dims.z) as usize],
            brick_dims,
        }
    }

    pub fn contains(&self, p: IVec3) -> bool {
        p.cmpge(IVec3::ZERO).all() && p.cmplt(self.dims).all()
    }

    pub fn index(&self, p: IVec3) -> usize {
        (p.x + self.dims.x * (p.y + self.dims.y * p.z)) as usize
    }

    pub fn position(&self, i: usize) -> IVec3 {
        let i = i as i32;
        IVec3::new(i % self.dims.x, (i / self.dims.x) % self.dims.y, i / (self.dims.x * self.dims.y))
    }

    /// Color index at `p`, 0 outside the volume.
    pub fn get(&self, p: IVec3) -> u8 {
        if self.contains(p) { self.cells[self.index(p)] } else { 0 }
    }

    /// Set a cell by index, keeping the brick occupancy conservative.
    pub fn set_index(&mut self, i: usize, c: u8) {
        self.cells[i] = c;
        if c != 0 {
            let b = self.position(i) / BRICK;
            let bi = (b.x + self.brick_dims.x * (b.y + self.brick_dims.y * b.z)) as usize;
            self.bricks[bi] = 1;
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaterialKind {
    Diffuse = 0,
    Metal = 1,
    Glass = 2,
    Emit = 3,
    Blend = 4,
    Media = 5,
}

/// GPU material record; must match `Material` in `pathtrace.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterial {
    /// Linear RGB albedo, alpha from palette.
    pub color: [f32; 4],
    pub kind: u32,
    pub rough: f32,
    pub metal: f32,
    pub ior: f32,
    /// Emitted radiance multiplier (already includes power).
    pub emit: f32,
    pub trans: f32,
    pub density: f32,
    /// Phase function asymmetry for clouds.
    pub g: f32,
}

/// Renderer settings, initialised from the file's `rOBJ` chunks.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderSettings {
    pub sun_pitch: f32,
    pub sun_yaw: f32,
    pub sun_area: f32,
    pub sun_intensity: f32,
    pub sun_color: Vec3,
    pub sun_disk: bool,
    pub sky_intensity: f32,
    pub sky_color: Vec3,
    pub env_mode: EnvMode,
    pub ibl_path: Option<PathBuf>,
    pub ibl_intensity: f32,
    /// Fraction of a full turn.
    pub ibl_rotation: f32,
    /// Atmosphere colors are scattering ratios, not sRGB display colors.
    pub rayleigh_density: f32,
    pub rayleigh_color: Vec3,
    pub mie_density: f32,
    pub mie_color: Vec3,
    pub mie_g: f32,
    pub ozone_density: f32,
    pub ozone_color: Vec3,
    pub fog_density: f32,
    pub fog_color: Vec3,
    pub fog_g: f32,
    pub translucent_shadows: bool,
    pub fov: f32,
    pub projection: Projection,
    pub aperture: f32,
    /// Aperture polygon sides; below 3 means a round aperture.
    pub blades: u32,
    /// Degrees.
    pub blade_rotation: f32,
    /// Linear multiplier.
    pub exposure: f32,
    pub gamma: f32,
    pub aces: bool,
    pub vignette: f32,
    pub ground: bool,
    pub ground_color: Vec3,
    pub bloom: bool,
    pub bloom_mix: f32,
    pub bloom_scale: f32,
    pub bloom_aspect: f32,
    pub bloom_threshold: f32,
    pub max_diffuse: u32,
    pub max_specular: u32,
    pub max_scatter: u32,
    pub clamp: f32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        // Defaults match a fresh MagicaVoxel 0.99.6 scene.
        Self {
            sun_pitch: 50.0,
            sun_yaw: 50.0,
            sun_area: 0.07,
            sun_intensity: 0.6,
            sun_color: Vec3::ONE,
            sun_disk: false,
            sky_intensity: 0.7,
            sky_color: Vec3::ONE,
            env_mode: EnvMode::Uniform,
            ibl_path: None,
            ibl_intensity: 1.0,
            ibl_rotation: 0.0,
            rayleigh_density: 0.4,
            rayleigh_color: Vec3::new(77.0, 153.0, 255.0) / 255.0,
            mie_density: 0.4,
            mie_color: Vec3::ONE,
            mie_g: 0.78,
            ozone_density: 0.0,
            ozone_color: Vec3::new(105.0, 255.0, 110.0) / 255.0,
            fog_density: 0.0,
            fog_color: Vec3::ONE,
            fog_g: 0.0,
            translucent_shadows: true,
            fov: 45.0,
            projection: Projection::Perspective,
            aperture: 0.0,
            blades: 0,
            blade_rotation: 0.0,
            exposure: 1.0,
            gamma: 2.2,
            aces: false,
            vignette: 0.0,
            ground: true,
            ground_color: srgb_to_linear(Vec3::splat(80.0 / 255.0)),
            bloom: false,
            bloom_mix: 0.5,
            bloom_scale: 0.0,
            bloom_aspect: 0.0,
            bloom_threshold: 1.0,
            max_diffuse: 4,
            max_specular: 8,
            max_scatter: 8,
            clamp: 50.0,
        }
    }
}

/// One exposed face of an emissive voxel; must match `Light` in
/// `pathtrace.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLight {
    /// Face corner (minimum on the two in-plane axes), CDF value in `w`.
    pub corner_cdf: [f32; 4],
    /// Normal axis, 1 if the normal points along +axis, material index.
    pub axis_sign_mat: [u32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Projection {
    Perspective = 0,
    Orthographic = 1,
    Stereographic = 2,
    Panorama = 3,
    Cylindrical = 4,
}

pub struct Scene {
    pub volume: Volume,
    /// sRGB palette as stored in the file, indexed by color index.
    pub palette: Vec<[u8; 4]>,
    /// Render chunks from the file, written back unchanged on save.
    pub render_dicts: Vec<Dict>,
    pub materials: Vec<GpuMaterial>,
    pub lights: Vec<GpuLight>,
    /// Sum of the power of all light faces.
    pub light_power: f32,
    pub settings: RenderSettings,
}

fn luminance(c: [f32; 4]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Collect every emissive voxel face that borders air or glass.
pub fn build_lights(v: &Volume, mats: &[GpuMaterial]) -> (Vec<GpuLight>, f32) {
    const MAX_FACES: usize = 1 << 20;
    let power = |idx: u8| {
        let m = &mats[idx as usize];
        if m.kind == MaterialKind::Emit as u32 { m.emit * luminance(m.color) } else { 0.0 }
    };
    let see_through = |p: IVec3| {
        if p.cmplt(IVec3::ZERO).any() || p.cmpge(v.dims).any() {
            return true;
        }
        let c = v.cells[(p.x + v.dims.x * (p.y + v.dims.y * p.z)) as usize];
        c == 0 || mats[c as usize].kind == MaterialKind::Glass as u32
    };
    let mut lights = Vec::new();
    let mut total = 0.0f32;
    for z in 0..v.dims.z {
        for y in 0..v.dims.y {
            for x in 0..v.dims.x {
                let c = v.cells[(x + v.dims.x * (y + v.dims.y * z)) as usize];
                if c == 0 {
                    continue;
                }
                let pw = power(c);
                if pw <= 0.0 {
                    continue;
                }
                let p = IVec3::new(x, y, z);
                for axis in 0..3 {
                    for positive in [false, true] {
                        let mut d = IVec3::ZERO;
                        d[axis] = if positive { 1 } else { -1 };
                        if !see_through(p + d) || lights.len() >= MAX_FACES {
                            continue;
                        }
                        let mut corner = p.as_vec3();
                        if positive {
                            corner[axis] += 1.0;
                        }
                        total += pw;
                        lights.push(GpuLight {
                            corner_cdf: [corner.x, corner.y, corner.z, total],
                            axis_sign_mat: [axis as u32, positive as u32, c as u32, 0],
                        });
                    }
                }
            }
        }
    }
    for l in &mut lights {
        l.corner_cdf[3] /= total;
    }
    (lights, total)
}

pub fn srgb_to_linear(c: Vec3) -> Vec3 {
    let f = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
    Vec3::new(f(c.x), f(c.y), f(c.z))
}

fn num(d: &Dict, k: &str) -> Option<f32> {
    d.get(k).and_then(|s| s.trim().parse().ok())
}

fn vec3(d: &Dict, k: &str) -> Option<Vec3> {
    let v: Vec<f32> = d.get(k)?.split_whitespace().filter_map(|s| s.parse().ok()).collect();
    (v.len() == 3).then(|| Vec3::new(v[0], v[1], v[2]))
}

fn color(d: &Dict, k: &str) -> Option<Vec3> {
    vec3(d, k).map(|c| srgb_to_linear(c / 255.0))
}

fn raw_color(d: &Dict, k: &str) -> Option<Vec3> {
    vec3(d, k).map(|c| c / 255.0)
}

fn flag(d: &Dict, k: &str) -> Option<bool> {
    d.get(k).map(|s| s.trim() == "1")
}

impl RenderSettings {
    pub fn from_vox(f: &VoxFile) -> Self {
        let mut s = Self::default();
        if let Some(d) = f.render.get("_inf") {
            if let Some(a) = vec3_2(d, "_angle") {
                s.sun_pitch = a.0;
                s.sun_yaw = a.1;
            }
            s.sun_area = num(d, "_area").unwrap_or(s.sun_area);
            s.sun_intensity = num(d, "_i").unwrap_or(s.sun_intensity);
            s.sun_color = color(d, "_k").unwrap_or(s.sun_color);
            s.sun_disk = flag(d, "_disk").unwrap_or(s.sun_disk);
        }
        if let Some(d) = f.render.get("_uni") {
            s.sky_intensity = num(d, "_i").unwrap_or(s.sky_intensity);
            s.sky_color = color(d, "_k").unwrap_or(s.sky_color);
        }
        if let Some(d) = f.render.get("_env") {
            // Order of the sky tabs in the UI: uniform, atmosphere, image.
            s.env_mode = match num(d, "_mode").unwrap_or(0.0) as i32 {
                1 => EnvMode::Atmosphere,
                2 => EnvMode::Image,
                _ => EnvMode::Uniform,
            };
        }
        if let Some(d) = f.render.get("_ibl") {
            s.ibl_intensity = num(d, "_i").unwrap_or(s.ibl_intensity);
            s.ibl_rotation = num(d, "_rot").unwrap_or(s.ibl_rotation);
        }
        // Old files split the atmosphere into `_rayleigh` and `_mie`.
        if let Some(d) = f.render.get("_rayleigh") {
            s.rayleigh_density = num(d, "_d").unwrap_or(s.rayleigh_density);
            s.rayleigh_color = raw_color(d, "_k").unwrap_or(s.rayleigh_color);
        }
        if let Some(d) = f.render.get("_mie") {
            s.mie_density = num(d, "_d").unwrap_or(s.mie_density);
            s.mie_color = raw_color(d, "_k").unwrap_or(s.mie_color);
            s.mie_g = num(d, "_g").unwrap_or(s.mie_g);
        }
        if let Some(d) = f.render.get("_atm") {
            s.rayleigh_density = num(d, "_ray_d").unwrap_or(s.rayleigh_density);
            s.rayleigh_color = raw_color(d, "_ray_k").unwrap_or(s.rayleigh_color);
            s.mie_density = num(d, "_mie_d").unwrap_or(s.mie_density);
            s.mie_color = raw_color(d, "_mie_k").unwrap_or(s.mie_color);
            s.mie_g = num(d, "_mie_g").unwrap_or(s.mie_g);
            s.ozone_density = num(d, "_o3_d").unwrap_or(s.ozone_density);
            s.ozone_color = raw_color(d, "_o3_k").unwrap_or(s.ozone_color);
        }
        if let Some(d) = f.render.get("_fog") {
            if flag(d, "_enable") == Some(true) {
                s.fog_density = num(d, "_d").unwrap_or(0.0);
            }
            s.fog_color = color(d, "_k").unwrap_or(s.fog_color);
        }
        if let Some(d) = f.render.get("_fog_uni") {
            s.fog_density = num(d, "_d").unwrap_or(s.fog_density);
            s.fog_color = color(d, "_k").unwrap_or(s.fog_color);
            s.fog_g = num(d, "_g").unwrap_or(s.fog_g);
        }
        // Older files use `_len`, newer ones split it into `_lens` and `_film`.
        for key in ["_len", "_lens"] {
            if let Some(d) = f.render.get(key) {
                s.fov = num(d, "_fov").unwrap_or(s.fov);
                s.blades = num(d, "_blade_n").map(|v| v as u32).unwrap_or(s.blades);
                s.blade_rotation = num(d, "_blade_r").unwrap_or(s.blade_rotation);
                // Projection tabs in the UI: perspective, stereographic, panorama.
                s.projection = match num(d, "_proj").unwrap_or(0.0) as i32 {
                    1 => Projection::Stereographic,
                    2 => Projection::Panorama,
                    _ => Projection::Perspective,
                };
                s.aperture = num(d, "_aperture").or(num(d, "_dof")).unwrap_or(s.aperture);
                // Old files store exposure in stops.
                s.exposure = num(d, "_exp").map(f32::exp2).unwrap_or(s.exposure);
                s.gamma = num(d, "_gam").unwrap_or(s.gamma);
                s.vignette = num(d, "_vig").unwrap_or(s.vignette);
            }
        }
        if let Some(d) = f.render.get("_film") {
            // Newer files store a linear multiplier (inferred, not verified).
            s.exposure = num(d, "_expo").unwrap_or(s.exposure);
            s.gamma = num(d, "_gam").unwrap_or(s.gamma);
            s.vignette = num(d, "_vig").unwrap_or(s.vignette);
            s.aces = flag(d, "_aces").unwrap_or(s.aces);
        }
        if let Some(d) = f.render.get("_bloom") {
            s.bloom_mix = num(d, "_mix").unwrap_or(s.bloom_mix);
            s.bloom_scale = num(d, "_scale").unwrap_or(s.bloom_scale);
            s.bloom_aspect = num(d, "_aspect").unwrap_or(s.bloom_aspect);
            s.bloom_threshold = num(d, "_threshold").unwrap_or(s.bloom_threshold);
        }
        if let Some(d) = f.render.get("_ground") {
            s.ground_color = color(d, "_color").unwrap_or(s.ground_color);
            s.ground = flag(d, "_enable").unwrap_or(s.ground);
        }
        if let Some(d) = f.render.get("_setting") {
            s.ground = flag(d, "_ground").unwrap_or(s.ground);
        }
        // Depth of field is opt-in: files store an aperture even when DOF is off.
        s.aperture = 0.0;
        s
    }

    pub fn sun_dir(&self) -> Vec3 {
        let (p, y) = (self.sun_pitch.to_radians(), self.sun_yaw.to_radians());
        Vec3::new(p.cos() * y.cos(), p.cos() * y.sin(), p.sin()).normalize()
    }
}

fn vec3_2(d: &Dict, k: &str) -> Option<(f32, f32)> {
    let v: Vec<f32> = d.get(k)?.split_whitespace().filter_map(|s| s.parse().ok()).collect();
    (v.len() == 2).then(|| (v[0], v[1]))
}

pub fn material(idx: u32, rgba: [u8; 4], d: Option<&Dict>) -> GpuMaterial {
    let c = srgb_to_linear(Vec3::new(rgba[0] as f32, rgba[1] as f32, rgba[2] as f32) / 255.0);
    let mut m = GpuMaterial {
        color: [c.x, c.y, c.z, rgba[3] as f32 / 255.0],
        kind: MaterialKind::Diffuse as u32,
        rough: 0.1,
        metal: 0.0,
        ior: 1.3,
        emit: 0.0,
        trans: 0.0,
        density: 0.0,
        g: 0.0,
    };
    if idx == 0 {
        return m;
    }
    let Some(d) = d else { return m };
    let weight = num(d, "_weight").unwrap_or(1.0);
    m.rough = num(d, "_rough").unwrap_or(m.rough);
    // `_ri` is the refractive index; `_ior` stores it minus one.
    m.ior = num(d, "_ri").or(num(d, "_ior").map(|v| v + 1.0)).unwrap_or(m.ior);
    m.kind = match d.get("_type").map(String::as_str) {
        Some("_metal") => {
            m.metal = num(d, "_metal").unwrap_or(weight);
            MaterialKind::Metal
        }
        Some("_glass") => {
            m.trans = num(d, "_trans").or(num(d, "_alpha")).unwrap_or(weight);
            MaterialKind::Glass
        }
        Some("_emit") => {
            let flux = num(d, "_flux").unwrap_or(0.0);
            // Power is a log-scale multiplier on the base emission.
            m.emit = num(d, "_emit").unwrap_or(weight) * 10f32.powf(flux * 0.5);
            MaterialKind::Emit
        }
        Some("_blend") => {
            m.metal = num(d, "_metal").unwrap_or(0.0);
            m.trans = num(d, "_trans").or(num(d, "_alpha")).unwrap_or(0.0);
            MaterialKind::Blend
        }
        Some("_media") => {
            // Density is per voxel; scale so the UI's small values read as clouds.
            m.density = num(d, "_d").unwrap_or(0.05) * 10.0;
            m.g = num(d, "_g").or(num(d, "_g0")).unwrap_or(0.0);
            MaterialKind::Media
        }
        _ => MaterialKind::Diffuse,
    } as u32;
    m
}

/// Voxels placed in world space plus the bounds of every model box, so the
/// editable volume covers each model's full size, not just its voxels.
struct Placed {
    voxels: Vec<(IVec3, u8)>,
    min: IVec3,
    max: IVec3,
}

impl Placed {
    fn grow(&mut self, p: IVec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }
}

fn fmt(v: f32) -> String {
    // Short, round-trippable decimals.
    let s = format!("{v:.4}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// The `MATL` dictionary for a material, or `None` for plain diffuse.
pub fn material_dict(m: &GpuMaterial) -> Option<Dict> {
    let mut d = Dict::new();
    let mut set = |k: &str, v: String| {
        d.insert(k.to_string(), v);
    };
    let kind = match m.kind {
        k if k == MaterialKind::Metal as u32 => "_metal",
        k if k == MaterialKind::Glass as u32 => "_glass",
        k if k == MaterialKind::Emit as u32 => "_emit",
        k if k == MaterialKind::Blend as u32 => "_blend",
        k if k == MaterialKind::Media as u32 => "_media",
        _ => return None,
    };
    set("_type", kind.into());
    set("_rough", fmt(m.rough));
    set("_ri", fmt(m.ior));
    set("_ior", fmt(m.ior - 1.0));
    match kind {
        "_metal" => set("_metal", fmt(m.metal)),
        "_glass" => set("_trans", fmt(m.trans)),
        "_blend" => {
            set("_metal", fmt(m.metal));
            set("_trans", fmt(m.trans));
        }
        "_emit" => {
            // Inverse of `material`: emission above 1 moves into power.
            let (e, flux) = if m.emit > 1.0 { (1.0, 2.0 * m.emit.log10()) } else { (m.emit, 0.0) };
            set("_emit", fmt(e));
            set("_flux", fmt(flux));
        }
        "_media" => {
            set("_d", fmt(m.density / 10.0));
            set("_g", fmt(m.g));
        }
        _ => {}
    }
    Some(d)
}

impl Scene {
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let mats: Vec<(u32, Dict)> =
            (1..256u32).filter_map(|i| material_dict(&self.materials[i as usize]).map(|d| (i, d))).collect();
        crate::vox::write(path, self.volume.dims, &self.volume.cells, &self.palette, &mats, &self.render_dicts)
    }

    pub fn from_vox(f: &VoxFile) -> Result<Self> {
        let mut placed = Placed { voxels: Vec::new(), min: IVec3::MAX, max: IVec3::MIN };
        if f.nodes.is_empty() {
            for m in &f.models {
                placed.grow(IVec3::ZERO);
                placed.grow(m.size - IVec3::ONE);
                placed.voxels.extend(m.voxels.iter().map(|v| (IVec3::new(v[0] as i32, v[1] as i32, v[2] as i32), v[3])));
            }
        } else {
            walk(f, 0, Mat3::IDENTITY, Vec3::ZERO, &mut placed, 0);
        }
        if placed.min.cmpgt(placed.max).any() {
            bail!("scene has no visible models");
        }

        let dims = placed.max - placed.min + IVec3::ONE;
        if dims.x as usize * dims.y as usize * dims.z as usize > 1024 * 1024 * 1024 {
            bail!("scene too large for a dense volume: {dims}");
        }
        let mut volume = Volume::new(dims);
        for (p, c) in placed.voxels {
            let i = volume.index(p - placed.min);
            volume.set_index(i, c);
        }
        let render_dicts = f.render_order.iter().filter_map(|k| f.render.get(k).cloned()).collect();
        Ok(Self::assemble(volume, f.palette.clone(), &f.materials, render_dicts, RenderSettings::from_vox(f)))
    }

    /// An empty canvas with the default palette, like MagicaVoxel's new model.
    pub fn empty(dims: IVec3) -> Self {
        Self::assemble(Volume::new(dims), crate::vox::default_palette(), &Default::default(), Vec::new(), RenderSettings::default())
    }

    fn assemble(
        volume: Volume,
        palette: Vec<[u8; 4]>,
        mats: &std::collections::HashMap<u32, Dict>,
        render_dicts: Vec<Dict>,
        settings: RenderSettings,
    ) -> Self {
        let materials: Vec<GpuMaterial> = (0..256u32).map(|i| material(i, palette[i as usize], mats.get(&i))).collect();
        let (lights, light_power) = build_lights(&volume, &materials);
        Scene { volume, palette, render_dicts, materials, lights, light_power, settings }
    }

    /// Rebuild the emissive face list after edits.
    pub fn rebuild_lights(&mut self) {
        (self.lights, self.light_power) = build_lights(&self.volume, &self.materials);
    }

    /// Recompute a material's linear color after its palette entry changed.
    pub fn sync_color(&mut self, idx: usize) {
        let p = self.palette[idx];
        let c = srgb_to_linear(Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32) / 255.0);
        self.materials[idx].color = [c.x, c.y, c.z, p[3] as f32 / 255.0];
    }
}

fn hidden(attrs: &Dict) -> bool {
    attrs.get("_hidden").is_some_and(|v| v == "1")
}

fn walk(f: &VoxFile, id: i32, rot: Mat3, trans: Vec3, out: &mut Placed, depth: u32) {
    if depth > 64 {
        return;
    }
    match f.nodes.get(&id) {
        Some(Node::Transform { attrs, child, layer, rotation, translation }) => {
            let layer_hidden = f.layers.get(layer).is_some_and(hidden);
            if hidden(attrs) || layer_hidden {
                return;
            }
            let r = rot * *rotation;
            let t = rot * translation.as_vec3() + trans;
            walk(f, *child, r, t, out, depth + 1);
        }
        Some(Node::Group { attrs, children }) => {
            if hidden(attrs) {
                return;
            }
            for c in children {
                walk(f, *c, rot, trans, out, depth + 1);
            }
        }
        Some(Node::Shape { attrs, models }) => {
            if hidden(attrs) {
                return;
            }
            for &mi in models {
                let Some(m) = f.models.get(mi as usize) else { continue };
                // Models pivot around their center; integer division matches
                // MagicaVoxel's placement for odd and even sizes.
                let half = (m.size / 2).as_vec3();
                let place = |v: Vec3| (rot * (v + 0.5 - half) + trans).floor().as_ivec3();
                for corner in 0..8 {
                    let c = IVec3::new(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1) * (m.size - IVec3::ONE);
                    out.grow(place(c.as_vec3()));
                }
                for v in &m.voxels {
                    out.voxels.push((place(Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32)), v[3]));
                }
            }
        }
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_reload_round_trips() {
        let mut s = Scene::empty(IVec3::new(300, 20, 10));
        // Voxels in two 256-wide tiles, one emissive, one glass.
        for (p, c) in [(IVec3::new(1, 2, 3), 10u8), (IVec3::new(280, 19, 9), 20), (IVec3::new(0, 0, 0), 30)] {
            let i = s.volume.index(p);
            s.volume.set_index(i, c);
        }
        s.materials[20].kind = MaterialKind::Emit as u32;
        s.materials[20].emit = 31.6;
        s.materials[30].kind = MaterialKind::Glass as u32;
        s.materials[30].trans = 0.8;
        s.palette[10] = [1, 2, 3, 255];

        let path = std::env::temp_dir().join(format!("voxely-roundtrip-{}.vox", std::process::id()));
        s.save(&path).unwrap();
        let back = Scene::from_vox(&crate::vox::VoxFile::load(&path).unwrap()).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.volume.dims, s.volume.dims);
        assert_eq!(back.volume.cells, s.volume.cells);
        assert_eq!(back.palette[10], [1, 2, 3, 255]);
        assert_eq!(back.materials[20].kind, MaterialKind::Emit as u32);
        assert!((back.materials[20].emit - 31.6).abs() < 0.05, "{}", back.materials[20].emit);
        assert_eq!(back.materials[30].kind, MaterialKind::Glass as u32);
        assert!((back.materials[30].trans - 0.8).abs() < 1e-4);
    }
}
