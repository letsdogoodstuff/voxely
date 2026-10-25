//! Voxel art in a picture: the size of one voxel in pixels, found from the
//! regular spacing of the edges between blocks.

use crate::control::surface::Image;

/// The block size (pixels per voxel) in the masked area of a picture, with
/// a confidence 0..1; None when the area shows no regular spacing. Regular
/// texture (a lattice, bricks) repeats too, so whether the picture is voxel
/// art at all is for the caller (the agent looking at it) to judge.
pub fn voxel_pitch(img: &Image, mask: &dyn Fn(usize, usize) -> bool, bbox: [usize; 4]) -> Option<(f32, f32)> {
    let [x0, y0, x1, y1] = bbox;
    let (w, h) = (x1.saturating_sub(x0) + 1, y1.saturating_sub(y0) + 1);
    if w < 24 || h < 24 {
        return None;
    }
    let luma = |x: usize, y: usize| {
        let p = img.px[y * img.w + x];
        0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32
    };
    // Rendered voxel art is shaded and in perspective, so the grid only
    // repeats cleanly over small patches: measure the period in many
    // windows (edge strength between neighboring columns / rows, summed
    // over the window) and take the consensus.
    let win = (w.min(h) / 2).clamp(24, 72);
    let step = win / 2;
    let mut votes: Vec<(f32, f32)> = Vec::new();
    let mut wy = y0;
    while wy + win <= y1 + 1 {
        let mut wx = x0;
        while wx + win <= x1 + 1 {
            let (mut ex, mut ey, mut inside) = (vec![0f32; win], vec![0f32; win], 0usize);
            for y in wy..(wy + win).min(img.h - 1) {
                for x in wx..(wx + win).min(img.w - 1) {
                    if !mask(x, y) || !mask(x + 1, y) || !mask(x, y + 1) {
                        continue;
                    }
                    inside += 1;
                    let c = luma(x, y);
                    ex[x - wx] += (luma(x + 1, y) - c).abs();
                    ey[y - wy] += (luma(x, y + 1) - c).abs();
                }
            }
            if inside * 2 >= win * win {
                votes.extend(period(&ex));
                votes.extend(period(&ey));
            }
            wx += step;
        }
        wy += step;
    }
    let total: f32 = votes.iter().map(|v| v.1).sum();
    if votes.len() < 4 || total <= 0.0 {
        return None;
    }
    // The period with the most support (votes within 12%; a vote at twice
    // a period also supports it, as every other edge can be faint).
    let support = |p: f32| -> f32 {
        votes.iter().map(|&(q, s)| if (q / p - 1.0).abs() < 0.12 { s } else if (q / (2.0 * p) - 1.0).abs() < 0.08 { 0.5 * s } else { 0.0 }).sum()
    };
    let (best, sup) = votes.iter().map(|v| (v.0, support(v.0))).max_by(|a, b| a.1.total_cmp(&b.1))?;
    // Refine: the strength-weighted mean of the votes near it.
    let near: Vec<&(f32, f32)> = votes.iter().filter(|v| (v.0 / best - 1.0).abs() < 0.12).collect();
    let pitch = near.iter().map(|v| v.0 * v.1).sum::<f32>() / near.iter().map(|v| v.1).sum::<f32>().max(1e-6);
    let conf = (sup / total).clamp(0.0, 1.0);
    (conf >= 0.22).then_some((pitch, conf))
}

/// The repeat length of a profile (2.5..48 px) from its autocorrelation:
/// the first strong peak, refined to sub-pixel; with the peak's strength.
fn period(prof: &[f32]) -> Option<(f32, f32)> {
    let n = prof.len();
    if n < 16 {
        return None;
    }
    // Remove the slow trend (shading, the silhouette) so only the
    // block-to-block rhythm is left.
    let k = 9.min(n / 2);
    let hp: Vec<f32> = (0..n)
        .map(|i| {
            let (a, b) = (i.saturating_sub(k), (i + k + 1).min(n));
            prof[i] - prof[a..b].iter().sum::<f32>() / (b - a) as f32
        })
        .collect();
    let ac = |lag: usize| -> f32 { (0..n - lag).map(|i| hp[i] * hp[i + lag]).sum::<f32>() / (n - lag) as f32 };
    let a0 = ac(0);
    if a0 <= 1e-6 {
        return None;
    }
    let max_lag = 48.min(n / 3);
    let r: Vec<f32> = (0..=max_lag + 1).map(|l| ac(l) / a0).collect();
    // The first local maximum above a threshold; prefer the fundamental
    // over its multiples.
    let peak = (3..=max_lag).find(|&l| r[l] > 0.15 && r[l] >= r[l - 1] && r[l] >= r[l + 1])?;
    let (a, b, c) = (r[peak - 1], r[peak], r[peak + 1]);
    let d = a - 2.0 * b + c;
    let off = if d.abs() > 1e-6 { 0.5 * (a - c) / d } else { 0.0 };
    Some((peak as f32 + off.clamp(-0.5, 0.5), b.clamp(0.0, 1.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_block_size_of_voxel_art() {
        // 7-pixel blocks of random shades.
        let (w, h, b) = (210, 140, 7);
        let mut px = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let k = ((x / b) * 7919 + (y / b) * 104_729) % 251;
                let v = (60 + k % 160) as u8;
                px.push([v, v / 2 + 40, 255 - v, 255]);
            }
        }
        let img = Image { w, h, px };
        let (p, conf) = voxel_pitch(&img, &|_, _| true, [0, 0, w - 1, h - 1]).expect("a grid");
        assert!((p - 7.0).abs() < 0.3, "pitch {p}");
        assert!(conf > 0.3, "confidence {conf}");
    }

    #[test]
    fn smooth_pictures_have_no_grid() {
        let (w, h) = (200, 120);
        let px = (0..w * h).map(|i| { let (x, y) = (i % w, i / w); [(x + y) as u8, (x * 2 % 256) as u8, (y * 2) as u8, 255] }).collect();
        let img = Image { w, h, px };
        assert!(voxel_pitch(&img, &|_, _| true, [0, 0, w - 1, h - 1]).is_none());
    }
}
