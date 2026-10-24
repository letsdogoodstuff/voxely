//! Palette tools from MagicaVoxel's palette menu: sort, mask unused,
//! random voxel colors and color-code copy/paste. A palette is 256 RGBA
//! entries; index 0 is empty and voxels use 1..=255.
#![allow(dead_code)]

use std::cmp::Ordering;

use crate::ops::Rng;
use crate::scene::Volume;

/// Hue sectors used to group colors when sorting.
const HUE_BINS: f32 = 24.0;

/// Hue in [0, 1), saturation and value in [0, 1].
fn hsv(c: [u8; 4]) -> (f32, f32, f32) {
    let [r, g, b] = [c[0], c[1], c[2]].map(|x| x as f32 / 255.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let s = if max > 0.0 { d / max } else { 0.0 };
    let h = if d == 0.0 {
        0.0
    } else if max == r {
        ((g - b) / d).rem_euclid(6.0)
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    (h / 6.0, s, max)
}

/// Sort key: chromatic colors first, grouped by hue sector (red, orange,
/// ..., magenta), each sector from bright to dark and then from vivid to
/// pale; near-grays (saturation < 0.15 or value < 0.08) last, bright to
/// dark.
fn order(a: [u8; 4], b: [u8; 4]) -> Ordering {
    let key = |c: [u8; 4]| {
        let (h, s, v) = hsv(c);
        let gray = s < 0.15 || v < 0.08;
        let bin = if gray { 0 } else { ((h * HUE_BINS) as i32).min(HUE_BINS as i32 - 1) };
        (gray, bin, v, s)
    };
    let (ka, kb) = (key(a), key(b));
    ka.0.cmp(&kb.0)
        .then(ka.1.cmp(&kb.1))
        .then(kb.2.total_cmp(&ka.2))
        .then(kb.3.total_cmp(&ka.3))
}

/// Sort palette entries 1..=255. Colors used by `volume` come first, then
/// the unused ones; each group is ordered by hue sector, then value (bright
/// to dark), then saturation (vivid to pale), with near-grays at the end
/// of the group (see `order`). The sort is stable. Returns the new palette
/// and `remap` (old index -> new index, `remap[0] == 0`) to apply to the
/// voxels with `apply_remap`, so `new[remap[c]] == palette[c]`.
pub fn sort(palette: &[[u8; 4]], volume: &Volume) -> (Vec<[u8; 4]>, [u8; 256]) {
    let used = mask_unused(palette, volume);
    let n = palette.len().min(256);
    let mut idx: Vec<usize> = (1..n).collect();
    idx.sort_by(|&a, &b| used[b].cmp(&used[a]).then(order(palette[a], palette[b])));
    let mut remap: [u8; 256] = std::array::from_fn(|i| i as u8);
    let mut out = palette.to_vec();
    for (k, &old) in idx.iter().enumerate() {
        remap[old] = (k + 1) as u8;
        out[k + 1] = palette[old];
    }
    (out, remap)
}

/// Which palette entries the volume uses (MagicaVoxel's "Mask Unused
/// Voxel Colors"); `palette.len()` entries, index 0 always false.
pub fn mask_unused(palette: &[[u8; 4]], volume: &Volume) -> Vec<bool> {
    let mut used = vec![false; palette.len().max(256)];
    for &c in &volume.cells {
        used[c as usize] = true;
    }
    used[0] = false;
    used.truncate(palette.len());
    used
}

/// Replace every voxel's color index `c` with `remap[c]`.
pub fn apply_remap(volume: &Volume, remap: &[u8; 256]) -> Volume {
    Volume::from_cells(volume.dims, volume.cells.iter().map(|&c| if c == 0 { 0 } else { remap[c as usize] }).collect())
}

/// Recolor each solid (and selected, if `sel` is given) voxel with a color
/// picked uniformly from `colors` (MagicaVoxel's "Rand"); empty cells stay
/// empty and color 0 in `colors` is ignored. Deterministic per seed.
pub fn random_colors(v: &Volume, sel: Option<&[bool]>, colors: &[u8], seed: u32) -> Volume {
    let colors: Vec<u8> = colors.iter().copied().filter(|&c| c != 0).collect();
    if colors.is_empty() {
        return v.clone();
    }
    let mut rng = Rng::new(seed);
    let cells = v
        .cells
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            if c != 0 && sel.is_none_or(|s| s[i]) { colors[rng.below(colors.len() as u32) as usize] } else { c }
        })
        .collect();
    Volume::from_cells(v.dims, cells)
}

/// "#RRGGBB" (uppercase) for copying a color.
pub fn color_code(c: [u8; 4]) -> String {
    format!("#{:02X}{:02X}{:02X}", c[0], c[1], c[2])
}

/// Parse a pasted color code: optional "#" or "0x", then RRGGBB, RRGGBBAA
/// or the RGB shorthand. Alpha defaults to 255.
pub fn parse_color_code(s: &str) -> Option<[u8; 4]> {
    let s = s.trim();
    let s = s.strip_prefix('#').or_else(|| s.strip_prefix("0x")).or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    if !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let byte = |i: usize| u8::from_str_radix(&s[i..i + 2], 16).ok();
    match s.len() {
        3 => {
            let d = |i: usize| u8::from_str_radix(&s[i..i + 1], 16).ok().map(|x| x * 17);
            Some([d(0)?, d(1)?, d(2)?, 255])
        }
        6 => Some([byte(0)?, byte(2)?, byte(4)?, 255]),
        8 => Some([byte(0)?, byte(2)?, byte(4)?, byte(6)?]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vox::default_palette;
    use glam::IVec3;

    fn model() -> Volume {
        let mut v = Volume::new(IVec3::new(4, 4, 4));
        for (i, c) in [1u8, 37, 200, 255, 37, 120].into_iter().enumerate() {
            v.set_index(i * 7, c);
        }
        v
    }

    #[test]
    fn sort_preserves_voxel_colors() {
        let pal = default_palette();
        let v = model();
        let (new, remap) = sort(&pal, &v);
        assert_eq!(new.len(), 256);
        assert_eq!(remap[0], 0);
        let mut seen = [false; 256];
        for c in 0..256 {
            assert!(!seen[remap[c] as usize], "remap is a permutation");
            seen[remap[c] as usize] = true;
            assert_eq!(new[remap[c] as usize], pal[c]);
        }
        let r = apply_remap(&v, &remap);
        for i in 0..v.cells.len() {
            assert_eq!(new[r.cells[i] as usize], pal[v.cells[i] as usize]);
            assert_eq!(r.cells[i] == 0, v.cells[i] == 0);
        }
        // The five used colors come first.
        let mut first: Vec<u8> = r.cells.iter().copied().filter(|&c| c != 0).collect();
        first.sort();
        first.dedup();
        assert_eq!(first, vec![1, 2, 3, 4, 5]);
        // Grays sort after chromatic colors among the unused ones.
        let (_, s, v) = hsv(new[255]);
        assert!(s < 0.15 || v < 0.08);
        let (_, s, v) = hsv(new[6]);
        assert!(s >= 0.15 && v >= 0.08);
    }

    #[test]
    fn mask_and_random() {
        let pal = default_palette();
        let v = model();
        let used = mask_unused(&pal, &v);
        assert_eq!(used.len(), 256);
        assert_eq!(used.iter().filter(|&&u| u).count(), 5);
        assert!(used[37] && used[255] && !used[0] && !used[2]);

        let r = random_colors(&v, None, &[9, 10], 7);
        assert_eq!(r.cells, random_colors(&v, None, &[9, 10], 7).cells);
        for i in 0..v.cells.len() {
            if v.cells[i] == 0 {
                assert_eq!(r.cells[i], 0);
            } else {
                assert!(r.cells[i] == 9 || r.cells[i] == 10);
            }
        }
        let mut sel = vec![false; v.cells.len()];
        sel[0] = true;
        let s = random_colors(&v, Some(&sel), &[9], 1);
        assert_eq!(s.cells[0], 9);
        assert_eq!(s.cells[7], 37);
        assert_eq!(random_colors(&v, None, &[], 1).cells, v.cells);
    }

    #[test]
    fn color_codes() {
        assert_eq!(color_code([0x12, 0xab, 0xff, 0x80]), "#12ABFF");
        assert_eq!(parse_color_code("#12ABFF"), Some([0x12, 0xab, 0xff, 255]));
        assert_eq!(parse_color_code(" 12abff "), Some([0x12, 0xab, 0xff, 255]));
        assert_eq!(parse_color_code("0x12abff80"), Some([0x12, 0xab, 0xff, 0x80]));
        assert_eq!(parse_color_code("#f0a"), Some([0xff, 0x00, 0xaa, 255]));
        assert_eq!(parse_color_code("#12ABF"), None);
        assert_eq!(parse_color_code("#12ABFG"), None);
        assert_eq!(parse_color_code("#+2ABFF"), None);
        for c in default_palette() {
            assert_eq!(parse_color_code(&color_code(c)), Some([c[0], c[1], c[2], 255]));
        }
    }
}
