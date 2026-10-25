//! Pictures to voxels: a flat or extruded image, a heightmap terrain, and
//! slice sheets (every layer of a model side by side, as MagicaVoxel's
//! slice export writes them, or a folder of one image per layer).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use glam::IVec3;

use super::Model;
use crate::control::surface::{Image, load_image, resample};

/// How a picture stands in the model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImageMode {
    /// One voxel thick.
    Flat,
    /// This many voxels thick.
    Extrude(u32),
    /// Brighter pixels stand out further, up to this many voxels.
    Relief(u32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageOptions {
    /// Voxels along the picture's longer side; None keeps one voxel per
    /// pixel (scaled down to 256 for big pictures).
    pub size: Option<u32>,
    pub mode: ImageMode,
    /// Standing up facing -y (image up = +z), or lying on the ground
    /// (image up = +y).
    pub standing: bool,
    /// Pixels at least this opaque become voxels.
    pub alpha: u8,
}

impl Default for ImageOptions {
    fn default() -> Self {
        ImageOptions { size: None, mode: ImageMode::Flat, standing: true, alpha: 128 }
    }
}

/// How a heightmap is colored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeightColors {
    /// Sand, grass, forest, rock and snow by height, soil under grass.
    Terrain,
    /// The picture's own colors (a color heightmap or a painted map).
    Picture,
    /// Grays from dark (low) to light (high).
    Gray,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightOptions {
    /// Voxels along the map's longer side; None keeps one voxel per pixel
    /// (scaled down to 512 for big maps).
    pub size: Option<u32>,
    /// Voxels from the lowest to the highest point.
    pub height: u32,
    /// Dark is high instead of low.
    pub invert: bool,
    pub colors: HeightColors,
    /// Fill water up to this many voxels (0: no water).
    pub water: u32,
    /// Fill columns down to the ground; else only the surface, a few
    /// voxels deep (cheaper for big maps).
    pub solid: bool,
}

impl Default for HeightOptions {
    fn default() -> Self {
        HeightOptions { size: None, height: 32, invert: false, colors: HeightColors::Terrain, water: 0, solid: true }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SliceOptions {
    /// Slices across and down the sheet; 0 and 0 finds the layout (a row
    /// with 1 px gaps as MagicaVoxel exports, or a strip of squares).
    pub columns: u32,
    pub rows: u32,
    /// Pixels between slices in a grid.
    pub gap: u32,
    /// The first slice is the top layer instead of the bottom one.
    pub top_first: bool,
}

impl Default for SliceOptions {
    fn default() -> Self {
        SliceOptions { columns: 0, rows: 0, gap: 0, top_first: false }
    }
}

fn name_of(path: &Path) -> String {
    path.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Output size for a picture `w` x `h` with `size` voxels along its
/// longer side (or one per pixel, capped at `cap`).
fn fit(w: usize, h: usize, size: Option<u32>, cap: usize) -> (usize, usize) {
    let long = w.max(h);
    let target = match size {
        Some(s) => (s as usize).clamp(1, crate::ops::MAX_DIM as usize),
        None => long.min(cap),
    };
    if target == long {
        return (w, h);
    }
    let k = target as f64 / long as f64;
    (((w as f64 * k).round() as usize).max(1), ((h as f64 * k).round() as usize).max(1))
}

fn luma(c: [u8; 3]) -> f32 {
    (0.2126 * c[0] as f32 + 0.7152 * c[1] as f32 + 0.0722 * c[2] as f32) / 255.0
}

/// A picture as voxels.
pub fn image(path: &Path, o: &ImageOptions) -> Result<Model> {
    let img = load_image(path)?;
    image_from(&img, o, name_of(path))
}

pub fn image_from(img: &Image, o: &ImageOptions, name: String) -> Result<Model> {
    ensure!(img.w > 0 && img.h > 0, "empty image");
    let (w, h) = fit(img.w, img.h, o.size, 256);
    let px = resample(img, w, h, o.alpha);
    let depth = match o.mode {
        ImageMode::Flat => 1,
        ImageMode::Extrude(d) | ImageMode::Relief(d) => d.clamp(1, crate::ops::MAX_DIM as u32) as i32,
    };
    let mut voxels = Vec::new();
    for (k, c) in px.iter().enumerate() {
        let Some(c) = *c else { continue };
        let (u, v) = ((k % w) as i32, (h - 1 - k / w) as i32);
        let thick = match o.mode {
            ImageMode::Relief(_) => 1 + (luma(c) * (depth - 1) as f32).round() as i32,
            _ => depth,
        };
        for d in 0..thick {
            // Standing: depth runs back along +y from the front face.
            let p = if o.standing { IVec3::new(u, d, v) } else { IVec3::new(u, v, d) };
            voxels.push((p, [c[0], c[1], c[2], 255]));
        }
    }
    let dims = if o.standing { IVec3::new(w as i32, depth, h as i32) } else { IVec3::new(w as i32, h as i32, depth) };
    Ok(Model { name, dims, offset: IVec3::ZERO, voxels })
}

// ---------------------------------------------------------------- heightmaps

/// Terrain colors by height (0..1 of the map's range), from the top down:
/// the surface color, then what's under it.
fn terrain(t: f32, below: i32) -> [u8; 3] {
    const SAND: [u8; 3] = [214, 194, 140];
    const GRASS: [u8; 3] = [92, 148, 62];
    const FOREST: [u8; 3] = [58, 108, 52];
    const ROCK: [u8; 3] = [124, 116, 106];
    const SNOW: [u8; 3] = [236, 240, 244];
    const SOIL: [u8; 3] = [118, 84, 56];
    const STONE: [u8; 3] = [98, 98, 104];
    let top = match t {
        t if t < 0.1 => SAND,
        t if t < 0.45 => GRASS,
        t if t < 0.62 => FOREST,
        t if t < 0.82 => ROCK,
        _ => SNOW,
    };
    match below {
        0 => top,
        _ if top == SAND => if below < 3 { SAND } else { STONE },
        _ if top == GRASS || top == FOREST => if below < 3 { SOIL } else { STONE },
        _ => STONE,
    }
}

const WATER: [u8; 3] = [52, 108, 176];

/// A heightmap picture as terrain: x right, y away (the picture's top is
/// the far side), z up.
pub fn heightmap(path: &Path, o: &HeightOptions) -> Result<Model> {
    let img = load_image(path)?;
    heightmap_from(&img, o, name_of(path))
}

pub fn heightmap_from(img: &Image, o: &HeightOptions, name: String) -> Result<Model> {
    ensure!(img.w > 0 && img.h > 0, "empty image");
    let (w, h) = fit(img.w, img.h, o.size, 512);
    let px = resample(img, w, h, 1);
    let tall = o.height.clamp(1, crate::ops::MAX_DIM as u32) as i32;
    // Heights spread over the map's own range, so dim maps still rise.
    let lum: Vec<Option<f32>> = px.iter().map(|c| c.map(|c| if o.invert { 1.0 - luma(c) } else { luma(c) })).collect();
    let (lo, hi) = lum.iter().flatten().fold((f32::MAX, f32::MIN), |(a, b), &l| (a.min(l), b.max(l)));
    ensure!(lo <= hi, "the heightmap is transparent");
    let span = (hi - lo).max(1e-6);
    let top: Vec<Option<i32>> = lum.iter().map(|l| l.map(|l| ((l - lo) / span * (tall - 1) as f32).round() as i32)).collect();
    let water = (o.water as i32).min(tall);
    let at = |u: i32, v: i32| if u < 0 || v < 0 || u >= w as i32 || v >= h as i32 { None } else { top[v as usize * w + u as usize] };
    let mut voxels = Vec::new();
    for v in 0..h as i32 {
        for u in 0..w as i32 {
            let Some(z) = at(u, v) else { continue };
            // The lowest neighbor sets how deep a surface-only column goes,
            // so slopes have no holes.
            let low = [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().filter_map(|&(du, dv)| at(u + du, v + dv)).min().unwrap_or(z);
            let bottom = if o.solid { 0 } else { (low + 1).min(z - 2).max(0) };
            let c = px[v as usize * w + u as usize].unwrap();
            let y = h as i32 - 1 - v;
            let t = z as f32 / (tall - 1).max(1) as f32;
            for zz in bottom..=z {
                let below = z - zz;
                let col = match o.colors {
                    HeightColors::Terrain => terrain(t, below),
                    HeightColors::Picture => c,
                    HeightColors::Gray => {
                        let g = (40.0 + zz as f32 / (tall - 1).max(1) as f32 * 200.0).round() as u8;
                        // 16 steps keep the palette small.
                        let g = g / 16 * 16 + 8;
                        [g, g, g]
                    }
                };
                voxels.push((IVec3::new(u, y, zz), [col[0], col[1], col[2], 255]));
            }
            for zz in z + 1..water {
                voxels.push((IVec3::new(u, y, zz), [WATER[0], WATER[1], WATER[2], 255]));
            }
        }
    }
    Ok(Model { name, dims: IVec3::new(w as i32, h as i32, tall.max(water)), offset: IVec3::ZERO, voxels })
}

// ---------------------------------------------------------------- slices

/// A slice sheet (or a folder of slice images) as a model: each slice is
/// one z layer seen from above (x right, y up).
pub fn slices(path: &Path, o: &SliceOptions) -> Result<Model> {
    let layers = if path.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(path)
            .with_context(|| format!("reading {}", path.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| super::IMAGES.contains(&e.to_string_lossy().to_ascii_lowercase().as_str())))
            .collect();
        ensure!(!files.is_empty(), "{} has no PNG or JPEG slices", path.display());
        files.sort_by(|a, b| natural(a).cmp(&natural(b)));
        let imgs = files.iter().map(|f| load_image(f)).collect::<Result<Vec<_>>>()?;
        let (w, h) = (imgs[0].w, imgs[0].h);
        ensure!(imgs.iter().all(|i| i.w == w && i.h == h), "the slices in {} differ in size", path.display());
        imgs
    } else {
        cut(&load_image(path)?, o)?
    };
    slices_from(layers, o, name_of(path))
}

/// Sort key that orders "slice2" before "slice10".
fn natural(p: &Path) -> (String, u64) {
    let s = p.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let digits: String = s.chars().rev().take_while(char::is_ascii_digit).collect::<Vec<_>>().into_iter().rev().collect();
    (s[..s.len() - digits.len()].to_string(), digits.parse().unwrap_or(0))
}

fn transparent(c: [u8; 4]) -> bool {
    c[3] < 128
}

/// Cut a sheet into its slices.
fn cut(img: &Image, o: &SliceOptions) -> Result<Vec<Image>> {
    let (w, h) = (img.w, img.h);
    let crop = |x0: usize, y0: usize, sw: usize, sh: usize| Image {
        w: sw,
        h: sh,
        px: (y0..y0 + sh).flat_map(|y| (x0..x0 + sw).map(move |x| img.px[y * w + x])).collect(),
    };
    let (cols, rows, gap) = if o.columns > 0 || o.rows > 0 {
        let (c, r) = (o.columns.max(1) as usize, o.rows.max(1) as usize);
        (c, r, o.gap as usize)
    } else {
        layout(img).context("couldn't tell how the slices are laid out; give the columns and rows")?
    };
    let sw = w.checked_sub(gap * (cols - 1)).map(|x| x / cols).unwrap_or(0);
    let sh = h.checked_sub(gap * (rows - 1)).map(|x| x / rows).unwrap_or(0);
    ensure!(sw > 0 && sh > 0, "a {w}x{h} sheet can't hold {cols} x {rows} slices with {gap} px gaps");
    let mut out = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            out.push(crop(c * (sw + gap), r * (sh + gap), sw, sh));
        }
    }
    Ok(out)
}

/// Columns, rows and gap of a sheet: a row of slices with 1 px clear
/// gaps (MagicaVoxel's slice export), or a strip of square slices.
fn layout(img: &Image) -> Option<(usize, usize, usize)> {
    let (w, h) = (img.w, img.h);
    let clear_col = |x: usize| (0..h).all(|y| transparent(img.px[y * w + x]));
    // Slice width s: columns s, 2s+1, ... are clear and the sheet is
    // n*(s+1)-1 wide. Square slices win, then the narrowest.
    let fits: Vec<usize> = (1..w).filter(|&s| (w + 1) % (s + 1) == 0 && (w + 1) / (s + 1) > 1 && (1..(w + 1) / (s + 1)).all(|k| clear_col(k * (s + 1) - 1))).collect();
    if let Some(&s) = fits.iter().find(|&&s| s == h).or(fits.iter().find(|&&s| s >= 2)) {
        return Some(((w + 1) / (s + 1), 1, 1));
    }
    if w > h && w % h == 0 {
        return Some((w / h, 1, 0));
    }
    if h > w && h % w == 0 {
        return Some((1, h / w, 0));
    }
    None
}

pub fn slices_from(mut layers: Vec<Image>, o: &SliceOptions, name: String) -> Result<Model> {
    ensure!(!layers.is_empty(), "no slices");
    if o.top_first {
        layers.reverse();
    }
    let (w, h) = (layers[0].w, layers[0].h);
    ensure!(w as i32 <= crate::ops::MAX_DIM * 4 && h as i32 <= crate::ops::MAX_DIM * 4, "slices of {w}x{h} are too big");
    // Opaque slices (scans on black, say) use the corner color as empty.
    let any_clear = layers.iter().any(|l| l.px.iter().any(|&c| transparent(c)));
    let empty = (!any_clear).then(|| layers[0].px[0]);
    let is_empty = |c: [u8; 4]| match empty {
        Some(e) => (0..3).all(|k| c[k].abs_diff(e[k]) <= 8),
        None => transparent(c),
    };
    let mut voxels = Vec::new();
    for (z, l) in layers.iter().enumerate() {
        if l.w != w || l.h != h {
            bail!("slice {z} is {}x{}, not {w}x{h}", l.w, l.h);
        }
        for y in 0..h {
            for x in 0..w {
                let c = l.px[y * w + x];
                if !is_empty(c) {
                    voxels.push((IVec3::new(x as i32, (h - 1 - y) as i32, z as i32), [c[0], c[1], c[2], 255]));
                }
            }
        }
    }
    Ok(Model { name, dims: IVec3::new(w as i32, h as i32, layers.len() as i32), offset: IVec3::ZERO, voxels })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: usize, h: usize, f: impl Fn(usize, usize) -> [u8; 4]) -> Image {
        Image { w, h, px: (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| f(x, y)).collect() }
    }

    #[test]
    fn heightmap_rises_with_brightness() {
        // Left column black, right column white.
        let m = img(2, 1, |x, _| if x == 0 { [0, 0, 0, 255] } else { [255; 4] });
        let o = HeightOptions { height: 10, ..Default::default() };
        let t = heightmap_from(&m, &o, "t".into()).unwrap();
        let top = |x: i32| t.voxels.iter().filter(|v| v.0.x == x).map(|v| v.0.z).max().unwrap();
        assert_eq!((top(0), top(1)), (0, 9));
        assert_eq!(t.dims, IVec3::new(2, 1, 10));
        // Solid: every cell under the top is filled.
        assert_eq!(t.voxels.iter().filter(|v| v.0.x == 1).count(), 10);
        let inv = heightmap_from(&m, &HeightOptions { invert: true, ..o }, "t".into()).unwrap();
        assert_eq!(inv.voxels.iter().filter(|v| v.0.x == 0).map(|v| v.0.z).max(), Some(9));
    }

    #[test]
    fn heightmap_water_fills_low_ground() {
        let m = img(2, 1, |x, _| if x == 0 { [0, 0, 0, 255] } else { [255; 4] });
        let t = heightmap_from(&m, &HeightOptions { height: 10, water: 4, ..Default::default() }, "t".into()).unwrap();
        let water: Vec<i32> = t.voxels.iter().filter(|v| v.1[..3] == WATER).map(|v| v.0.z).collect();
        assert_eq!(water, vec![1, 2, 3]);
    }

    #[test]
    fn picture_top_is_far_side() {
        // A red pixel at the image's top row ends up at high y.
        let m = img(1, 2, |_, y| if y == 0 { [255, 0, 0, 255] } else { [0, 0, 255, 255] });
        let t = heightmap_from(&m, &HeightOptions { colors: HeightColors::Picture, height: 1, ..Default::default() }, "t".into()).unwrap();
        let red = t.voxels.iter().find(|v| v.1 == [255, 0, 0, 255]).unwrap();
        assert_eq!(red.0.y, 1);
    }

    #[test]
    fn slice_sheet_round_trips_magicavoxel_layout() {
        // Three 2x2 slices with 1 px gaps: 8 wide. Slice z has a voxel at
        // x = z % 2, y = 1 (row 0).
        let m = img(8, 2, |x, y| {
            let (z, sx) = (x / 3, x % 3);
            if sx < 2 && y == 0 && sx == z % 2 { [10 * z as u8 + 10, 0, 0, 255] } else { [0; 4] }
        });
        let t = slices_from(cut(&m, &SliceOptions::default()).unwrap(), &SliceOptions::default(), "s".into()).unwrap();
        assert_eq!(t.dims, IVec3::new(2, 2, 3));
        let mut v = t.voxels.clone();
        v.sort_by_key(|v| v.0.z);
        assert_eq!(v.iter().map(|v| v.0).collect::<Vec<_>>(), vec![IVec3::new(0, 1, 0), IVec3::new(1, 1, 1), IVec3::new(0, 1, 2)]);
        assert_eq!(v[2].1, [30, 0, 0, 255]);
    }

    #[test]
    fn slice_grid_and_strip() {
        // A 2x2 grid of 3x3 opaque slices on black: the corner is empty.
        let m = img(6, 6, |x, y| if x % 3 == 1 && y % 3 == 1 { [255; 4] } else { [0, 0, 0, 255] });
        let o = SliceOptions { columns: 2, rows: 2, ..Default::default() };
        let t = slices_from(cut(&m, &o).unwrap(), &o, "g".into()).unwrap();
        assert_eq!(t.dims, IVec3::new(3, 3, 4));
        assert_eq!(t.voxels.len(), 4);
        // A strip of three 4x4 squares lays itself out.
        let strip = img(12, 4, |_, _| [9, 9, 9, 255]);
        assert_eq!(layout(&strip), Some((3, 1, 0)));
    }

    #[test]
    fn image_stands_up_facing_front() {
        let m = img(2, 3, |x, y| if x == 0 && y == 0 { [1, 2, 3, 255] } else { [0; 4] });
        let t = image_from(&m, &ImageOptions { mode: ImageMode::Extrude(2), ..Default::default() }, "i".into()).unwrap();
        assert_eq!(t.dims, IVec3::new(2, 2, 3));
        let mut ps: Vec<IVec3> = t.voxels.iter().map(|v| v.0).collect();
        ps.sort_by_key(|p| p.y);
        assert_eq!(ps, vec![IVec3::new(0, 0, 2), IVec3::new(0, 1, 2)]);
    }
}
