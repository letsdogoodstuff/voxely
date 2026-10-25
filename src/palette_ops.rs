//! Palette tools from MagicaVoxel's palette menu: sort, mask unused,
//! random voxel colors and color-code copy/paste. A palette is 256 RGBA
//! entries; index 0 is empty and voxels use 1..=255.
#![allow(dead_code)]

use std::cmp::Ordering;
use std::path::Path;

use anyhow::{Context, Result};

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
    let order = sort_order(palette, &mask_unused(palette, volume), SortKey::Hue);
    (permute(palette, &order), remap_from_order(&order))
}

/// What `sort_order` orders colors by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortKey {
    /// Hue sectors, each bright to dark (see `order`).
    Hue,
    /// Vivid to pale.
    Saturation,
    /// Bright to dark.
    Value,
}

impl SortKey {
    pub const ALL: [SortKey; 3] = [SortKey::Hue, SortKey::Saturation, SortKey::Value];

    pub fn name(self) -> &'static str {
        match self {
            SortKey::Hue => "hue",
            SortKey::Saturation => "saturation",
            SortKey::Value => "value",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name().eq_ignore_ascii_case(s))
    }
}

/// The order (new index -> old index) that sorts entries 1..=255 by
/// `key`, used colors (`used`) first.
pub fn sort_order(palette: &[[u8; 4]], used: &[bool], key: SortKey) -> Vec<usize> {
    let n = palette.len().min(256);
    let is_used = |i: usize| used.get(i).copied().unwrap_or(false);
    let mut idx: Vec<usize> = (1..n).collect();
    idx.sort_by(|&a, &b| {
        let (pa, pb) = (palette[a], palette[b]);
        let (_, sa, va) = hsv(pa);
        let (_, sb, vb) = hsv(pb);
        is_used(b).cmp(&is_used(a)).then(match key {
            SortKey::Hue => order(pa, pb),
            SortKey::Saturation => sb.total_cmp(&sa).then(order(pa, pb)),
            SortKey::Value => vb.total_cmp(&va).then(order(pa, pb)),
        })
    });
    let mut out = vec![0];
    out.extend(idx);
    out
}

/// `items` with entry `k` taken from `order[k]` (entries past `order`
/// unchanged).
pub fn permute<T: Copy>(items: &[T], order: &[usize]) -> Vec<T> {
    let mut out = items.to_vec();
    for (k, &old) in order.iter().enumerate() {
        out[k] = items[old];
    }
    out
}

/// Old index -> new index for an `order` (new index -> old index).
pub fn remap_from_order(order: &[usize]) -> [u8; 256] {
    let mut remap: [u8; 256] = std::array::from_fn(|i| i as u8);
    for (k, &old) in order.iter().enumerate() {
        remap[old] = k as u8;
    }
    remap
}

/// The order in which entries `a` and `b` trade places (MagicaVoxel's
/// Alt+drag).
pub fn swap_order(a: usize, b: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..256).collect();
    order.swap(a, b);
    order
}

/// The order in which entry `from` is taken out and put back in at `to`,
/// the entries between sliding over one place (MagicaVoxel's Ctrl+drag).
pub fn move_order(from: usize, to: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..256).collect();
    let c = order.remove(from);
    order.insert(to, c);
    order
}

/// Pack the used colors (`used`) into slots 1.., merging exact duplicates
/// (equal `keys`, e.g. the same RGBA and material) into their first copy;
/// then the unused colors, then the freed duplicate slots. Returns the
/// order (new index -> old index) and the voxel remap (old index -> new
/// index), which sends duplicates to their first copy.
pub fn compact<K: std::hash::Hash + Eq>(keys: &[K], used: &[bool]) -> (Vec<usize>, [u8; 256]) {
    let is_used = |i: usize| used.get(i).copied().unwrap_or(false);
    let mut order = vec![0usize];
    let mut dups = Vec::new();
    let mut first: std::collections::HashMap<&K, usize> = Default::default();
    for i in (1..256).filter(|&i| is_used(i)) {
        if first.contains_key(&keys[i]) {
            dups.push(i);
        } else {
            first.insert(&keys[i], order.len());
            order.push(i);
        }
    }
    order.extend((1..256).filter(|&i| !is_used(i)));
    order.extend(&dups);
    let mut remap = remap_from_order(&order);
    for &i in &dups {
        remap[i] = first[&keys[i]] as u8;
    }
    (order, remap)
}

/// sRGB to OKLab, for perceptual blends and distances.
fn oklab(c: [u8; 4]) -> [f32; 3] {
    let lin = |x: u8| {
        let x = x as f32 / 255.0;
        if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) }
    };
    let [r, g, b] = [lin(c[0]), lin(c[1]), lin(c[2])];
    let l = (0.412_221_47 * r + 0.536_332_55 * g + 0.051_445_99 * b).cbrt();
    let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b).cbrt();
    let s = (0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b).cbrt();
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

fn from_oklab(lab: [f32; 3], alpha: u8) -> [u8; 4] {
    let l = lab[0] + 0.396_337_78 * lab[1] + 0.215_803_76 * lab[2];
    let m = lab[0] - 0.105_561_346 * lab[1] - 0.063_854_17 * lab[2];
    let s = lab[0] - 0.089_484_18 * lab[1] - 1.291_485_5 * lab[2];
    let (l, m, s) = (l * l * l, m * m * m, s * s * s);
    let rgb = [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ];
    let enc = |x: f32| {
        let x = x.clamp(0.0, 1.0);
        let x = if x <= 0.003_130_8 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
        (x * 255.0).round() as u8
    };
    [enc(rgb[0]), enc(rgb[1]), enc(rgb[2]), alpha]
}

/// Fill the entries strictly between `a` and `b` with an even blend from
/// one end's color to the other's (in OKLab, so the steps look even), like
/// MagicaVoxel's Ctrl+Alt+Shift drag.
pub fn gradient(palette: &mut [[u8; 4]], a: usize, b: usize) {
    let (lo, hi) = (a.min(b), a.max(b));
    let (ca, cb) = (oklab(palette[lo]), oklab(palette[hi]));
    let (aa, ab) = (palette[lo][3] as f32, palette[hi][3] as f32);
    for i in lo + 1..hi {
        let t = (i - lo) as f32 / (hi - lo) as f32;
        let lab = std::array::from_fn(|k| ca[k] + (cb[k] - ca[k]) * t);
        palette[i] = from_oklab(lab, (aa + (ab - aa) * t).round() as u8);
    }
}

/// For each entry of `old`, the index among `candidates` whose color in
/// `new` is closest (OKLab distance). Entry 0 maps to 0.
pub fn nearest_map(old: &[[u8; 4]], new: &[[u8; 4]], candidates: &[usize]) -> [u8; 256] {
    let labs: Vec<(usize, [f32; 3])> = candidates.iter().map(|&i| (i, oklab(new[i]))).collect();
    std::array::from_fn(|i| {
        if i == 0 || i >= old.len() || labs.is_empty() {
            return i as u8;
        }
        let c = oklab(old[i]);
        let d = |l: &[f32; 3]| (0..3).map(|k| (l[k] - c[k]).powi(2)).sum::<f32>();
        labs.iter().min_by(|x, y| d(&x.1).total_cmp(&d(&y.1))).map_or(i as u8, |x| x.0 as u8)
    })
}

/// Shift a color's hue (in turns), saturation and value (both -1..1).
pub fn adjust(c: [u8; 4], dh: f32, ds: f32, dv: f32) -> [u8; 4] {
    let (h, s, v) = hsv(c);
    let (h, s, v) = ((h + dh).rem_euclid(1.0), (s + ds).clamp(0.0, 1.0), (v + dv).clamp(0.0, 1.0));
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s));
    let (r, g, b) = match i as i32 % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    let u = |x: f32| (x * 255.0).round() as u8;
    [u(r), u(g), u(b), c[3]]
}

/// Palette file extensions `read_palette` understands.
pub const PALETTE_EXTENSIONS: [&str; 5] = ["png", "act", "gpl", "hex", "txt"];

pub fn is_palette_file(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()).is_some_and(|e| PALETTE_EXTENSIONS.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

/// Colors of a palette file in order (the first goes to color index 1):
/// a MagicaVoxel palette image (256x1, pixel k is index k + 1) or any
/// other palette image, such as Lospec's scaled swatches (its distinct
/// colors in reading order); Adobe `.act`; GIMP `.gpl`; or `.hex` /
/// `.txt` with one RRGGBB per line. At most 255 colors.
pub fn read_palette(path: &Path) -> Result<Vec<[u8; 4]>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
    let mut colors = match ext.as_str() {
        "png" => {
            let px = read_png(path)?;
            if px.len() <= 256 {
                px
            } else {
                let mut seen = std::collections::HashSet::new();
                px.into_iter().filter(|c| c[3] > 0 && seen.insert(*c)).collect()
            }
        }
        "act" => {
            let b = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
            anyhow::ensure!(b.len() >= 768, "an .act palette is 768 bytes");
            // Optional trailer: color count (big-endian) and transparent index.
            let n = if b.len() >= 770 { u16::from_be_bytes([b[768], b[769]]) as usize } else { 256 };
            let n = if n == 0 || n > 256 { 256 } else { n };
            b[..n * 3].chunks_exact(3).map(|c| [c[0], c[1], c[2], 255]).collect()
        }
        _ => {
            let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
            text.lines().filter_map(text_color).collect()
        }
    };
    colors.truncate(255);
    anyhow::ensure!(!colors.is_empty(), "no colors in {}", path.display());
    Ok(colors)
}

/// A color from a line of a GIMP palette ("R G B name") or a hex list
/// ("RRGGBB" or "#RRGGBB"); headers and comments give none.
fn text_color(line: &str) -> Option<[u8; 4]> {
    let line = line.trim();
    let nums: Vec<u8> = line.split_whitespace().take(3).map_while(|t| t.parse().ok()).collect();
    if nums.len() == 3 {
        return Some([nums[0], nums[1], nums[2], 255]);
    }
    let code = line.split_whitespace().next()?;
    if code.trim_start_matches('#').len() < 6 {
        return None;
    }
    parse_color_code(code).map(|c| [c[0], c[1], c[2], 255])
}

/// RGBA pixels of a PNG in reading order.
fn read_png(path: &Path) -> Result<Vec<[u8; 4]>> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut decoder = png::Decoder::new(std::io::BufReader::new(file));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::ALPHA);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size().context("palette image too large")?];
    let info = reader.next_frame(&mut buf)?;
    let channels = info.color_type.samples();
    Ok(buf[..info.buffer_size()]
        .chunks_exact(channels)
        .map(|c| match channels {
            4 => [c[0], c[1], c[2], c[3]],
            3 => [c[0], c[1], c[2], 255],
            2 => [c[0], c[0], c[0], c[1]],
            _ => [c[0], c[0], c[0], 255],
        })
        .collect())
}

/// Save a palette as MagicaVoxel's 256x1 palette image (pixel k is color
/// index k + 1).
pub fn write_palette(path: &Path, palette: &[[u8; 4]]) -> Result<()> {
    let mut px = Vec::with_capacity(1024);
    for k in 0..256 {
        px.extend_from_slice(&palette.get(k + 1).copied().unwrap_or([0, 0, 0, 255]));
    }
    crate::save_png(path, 256, 1, &px)
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

    /// `new[remap[c]] == old[c]` for every color: the model looks the same.
    fn keeps_look(old: &[[u8; 4]], order: &[usize]) {
        let new = permute(old, order);
        let remap = remap_from_order(order);
        for c in 0..256 {
            assert_eq!(new[remap[c] as usize], old[c], "color {c}");
        }
    }

    #[test]
    fn swap_move_and_sort_keep_the_look() {
        let pal = default_palette();
        keeps_look(&pal, &swap_order(3, 200));
        keeps_look(&pal, &move_order(10, 4));
        keeps_look(&pal, &move_order(4, 10));
        let m = move_order(2, 5);
        assert_eq!(&m[..7], &[0, 1, 3, 4, 5, 2, 6]);
        let used = mask_unused(&pal, &model());
        for key in SortKey::ALL {
            keeps_look(&pal, &sort_order(&pal, &used, key));
        }
        let by_value = permute(&pal, &sort_order(&pal, &vec![false; 256], SortKey::Value));
        assert!((1..255).all(|i| hsv(by_value[i]).2 >= hsv(by_value[i + 1]).2));
    }

    #[test]
    fn compact_packs_used_colors_and_merges_duplicates() {
        let mut pal = default_palette();
        pal[200] = pal[37];
        let mut used = vec![false; 256];
        for i in [37, 200, 120] {
            used[i] = true;
        }
        let (order, remap) = compact(&pal, &used);
        let new = permute(&pal, &order);
        assert_eq!((remap[37], remap[200], remap[120]), (1, 1, 2));
        for c in 1..256 {
            assert_eq!(new[remap[c] as usize], pal[c]);
        }
        let mut seen = order.clone();
        seen.sort();
        assert_eq!(seen, (0..256).collect::<Vec<_>>(), "order is a permutation");
        // The same color with a different material (here: a flag) stays.
        let keys: Vec<([u8; 4], bool)> = pal.iter().enumerate().map(|(i, c)| (*c, i == 200)).collect();
        let (_, remap) = compact(&keys, &used);
        assert_ne!(remap[37], remap[200]);
    }

    #[test]
    fn gradient_blends_between_the_ends() {
        let mut pal = vec![[0u8; 4]; 256];
        pal[10] = [0, 0, 0, 255];
        pal[14] = [255, 255, 255, 255];
        pal[20] = [1, 2, 3, 4];
        gradient(&mut pal, 14, 10);
        assert_eq!(pal[10], [0, 0, 0, 255]);
        assert_eq!(pal[14], [255, 255, 255, 255]);
        assert_eq!(pal[20], [1, 2, 3, 4]);
        assert!(pal[11][0] < pal[12][0] && pal[12][0] < pal[13][0]);
        assert!(pal[12][0] == pal[12][1] && pal[12][1] == pal[12][2], "grays stay gray");
    }

    #[test]
    fn nearest_and_adjust() {
        let mut new = vec![[0u8; 4]; 256];
        new[1] = [250, 10, 10, 255];
        new[2] = [10, 10, 250, 255];
        let mut old = vec![[0u8; 4]; 256];
        old[5] = [200, 40, 40, 255];
        old[6] = [30, 30, 180, 255];
        let m = nearest_map(&old, &new, &[1, 2]);
        assert_eq!((m[0], m[5], m[6]), (0, 1, 2));
        let c = [200, 40, 40, 255];
        assert_eq!(adjust(c, 0.0, 0.0, 0.0), c);
        assert_eq!(adjust(c, 1.0 / 3.0, 0.0, 0.0), [40, 200, 40, 255]);
        assert_eq!(adjust(c, 0.0, -1.0, 0.0), [200, 200, 200, 255]);
    }

    #[test]
    fn palette_files() {
        let dir = std::env::temp_dir().join(format!("voxely-pal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let hex = dir.join("p.hex");
        std::fs::write(&hex, "ff0000\n00FF00\n\n#0000ff\n").unwrap();
        assert_eq!(read_palette(&hex).unwrap(), vec![[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]]);
        let gpl = dir.join("p.gpl");
        std::fs::write(&gpl, "GIMP Palette\nName: x\nColumns: 4\n#\n 12  34  56\tdark\n255 255 255 white\n").unwrap();
        assert_eq!(read_palette(&gpl).unwrap(), vec![[12, 34, 56, 255], [255, 255, 255, 255]]);
        let act = dir.join("p.act");
        let mut b = vec![0u8; 772];
        b[..6].copy_from_slice(&[1, 2, 3, 4, 5, 6]);
        b[768..770].copy_from_slice(&2u16.to_be_bytes());
        std::fs::write(&act, b).unwrap();
        assert_eq!(read_palette(&act).unwrap(), vec![[1, 2, 3, 255], [4, 5, 6, 255]]);
        let png = dir.join("p.png");
        let pal = default_palette();
        write_palette(&png, &pal).unwrap();
        let back = read_palette(&png).unwrap();
        assert_eq!(back.len(), 255);
        assert!((1..256).all(|i| back[i - 1] == pal[i]));
        // A scaled swatch image gives its distinct colors in order.
        let mut px = Vec::new();
        for _row in 0..4 {
            for c in [[9u8, 9, 9, 255], [9, 9, 9, 255], [7, 7, 7, 255]].iter().cycle().take(90) {
                px.extend_from_slice(c);
            }
        }
        crate::save_png(&png, 90, 4, &px).unwrap();
        assert_eq!(read_palette(&png).unwrap(), vec![[9, 9, 9, 255], [7, 7, 7, 255]]);
        let _ = std::fs::remove_dir_all(dir);
    }
}
