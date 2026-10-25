//! Surface work: recolor noise, outlines, paint by normal, palette presets, reference images.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};
use glam::{IVec3, Vec3};
use serde_json::{Value, json};

use super::{Ctx, NEIGHBORS, apply, boolean, color, color_schema, edited, get, int, num, string, text, tool, xyz};
use crate::scene::{Mask, Scene, Volume};

pub(crate) mod presets;

#[derive(Default)]
pub struct State {
    /// Palette slots this session filled (presets, image colors); later
    /// free-slot searches skip them even before voxels use them.
    reserved: Vec<u8>,
}

impl State {
    /// Follow the palette's colors to new indices.
    pub fn remap_colors(&mut self, remap: &[u8; 256]) {
        self.reserved.iter_mut().for_each(|c| *c = remap[*c as usize]);
        self.reserved.dedup();
    }
}

pub const PART_TOOLS: &[&str] = &["vary_colors", "shade_edges", "paint_faces", "voxelize_image"];

pub fn tools() -> Vec<Value> {
    let colors = json!({ "type": "array", "items": { "type": ["integer", "string"] }, "description": "Palette indices or \"#rrggbb\" (nearest palette color)" });
    let pattern = json!({ "type": "string", "enum": ["random", "noise", "dither"],
        "description": "random: per-voxel speckle; noise: smooth 3D patches about `scale` voxels across (natural); dither: even ordered pattern (a checkerboard at 50/50)" });
    let range = json!({ "type": "array", "items": { "type": "integer" }, "minItems": 2, "maxItems": 2, "description": "[first, last] palette indices to choose colors from (e.g. the slots a preset went into); default all" });
    vec![
        tool("vary_colors", "Textured look only, NOT the house style (flat designed colors): use it only when the user asks for a noisy or textured surface. Break up flat colors: replace voxels of `colors` (default: every voxel) with a seeded mix of the `with` colors, so a flat green becomes natural foliage. List `with` in order (dark to light) for noise to grade smoothly; include the original color in `with` to keep some of it. Works on the selection if there is one, else the whole part. Same seed, same result.", json!({
            "colors": colors.clone(),
            "with": colors.clone(),
            "weights": { "type": "array", "items": { "type": "number" }, "description": "Relative share of each `with` color (default equal); the result matches these proportions" },
            "pattern": pattern.clone(),
            "scale": { "type": "number", "description": "noise: feature size in voxels (default 4)" },
            "amount": { "type": "number", "description": "Fraction of the matching voxels to change, 0-1 (default 1)" },
            "seed": { "type": "integer" },
        }), &["with"]),
        tool("shade_edges", "Painted shading, NOT the house style (the renderer lights the model): use it only when the user asks for painted shading. mode edges: darken convex edges and corners (voxels exposed on two or more axes), with highlight: true lightening the top rims instead (edges with an exposed top face). mode corners: only outer corners (exposed on all three axes). mode cavity: ambient occlusion, darkening voxels in crevices and inner corners (few empty cells within `radius`). Shades come from the palette: the nearest darker (or lighter) color of the same hue, unless `color` / `highlight_color` is given. Run it once at the end: running twice shades twice. Works on the selection if there is one, else the whole part.", json!({
            "mode": { "type": "string", "enum": ["edges", "corners", "cavity"] },
            "highlight": { "type": "boolean", "description": "edges: lighten edges with an exposed top face, darken the rest" },
            "amount": { "type": "number", "description": "How far to shift toward black or white when picking palette shades, 0-1 (default 0.2)" },
            "color": color_schema(),
            "highlight_color": color_schema(),
            "only_color": colors.clone(),
            "radius": { "type": "integer", "description": "cavity: neighborhood radius 1-6 (default 2)" },
            "threshold": { "type": "number", "description": "cavity: darken voxels whose open fraction of the neighborhood is below this (default 0.3; a flat surface is about 0.4, an inner corner about 0.2)" },
            "levels": { "type": "integer", "description": "cavity: 1-3 shades, deeper crevices get darker (default 1)" },
            "palette_range": range.clone(),
        }), &["mode"]),
        tool("paint_faces", "Recolor surface voxels by which way they face: lighter grass on top faces, darker undersides, snow on top, moss on the north side. A voxel qualifies when its neighbor in `direction` is empty. Works on the selection if there is one, else the whole part.", json!({
            "direction": { "type": ["string", "array"], "items": { "type": "string" },
                "description": "top/+z, bottom/-z, +x/east, -x/west, +y/north, -y/south, sides (all four), all; or a list of these" },
            "color": color_schema(),
            "only_color": colors.clone(),
            "depth": { "type": "integer", "description": "Paint this many voxels deep from the face (default 1)" },
            "vary_depth": { "type": "boolean", "description": "Let depth vary 1..depth with noise (snow drifts, ragged grass)" },
            "coverage": { "type": "number", "description": "Fraction of qualifying faces to paint, 0-1 (default 1)" },
            "pattern": pattern,
            "scale": { "type": "number", "description": "noise: patch size in voxels (default 3)" },
            "open": { "type": "boolean", "description": "Only faces with nothing at all beyond them in that direction (open sky for top: no snow under overhangs)" },
            "seed": { "type": "integer" },
        }), &["direction", "color"]),
        tool("list_palette_presets", "Well-known open palettes apply_palette_preset can load, with their colors.", json!({}), &[]),
        tool("apply_palette_preset", "Write a well-known palette into palette slots without disturbing colors the scene uses. Returns where it went and an image of the swatches labeled with their indices.", json!({
            "name": { "type": "string", "description": "A name from list_palette_presets, e.g. pico-8, endesga-32" },
            "start_index": { "type": "integer", "description": "First slot to write (default: the first run of free slots)" },
            "overwrite": { "type": "boolean", "description": "Allow replacing colors that voxels use (they change color)" },
        }), &["name"]),
        tool("voxelize_image", "Turn a PNG or JPEG (a sprite, logo or reference) into voxels in the current part, matching each pixel to the nearest palette color. E.g. {path, height: 32} voxelizes a sprite 32 voxels tall, standing up facing -y. The canvas grows if needed; transparent pixels stay empty and existing voxels are kept elsewhere.", json!({
            "path": { "type": "string" },
            "height": { "type": "integer", "description": "Image height in voxels; width follows the aspect ratio" },
            "width": { "type": "integer", "description": "Image width in voxels (when height isn't given)" },
            "plane": { "type": "string", "enum": ["xz", "yz", "xy"], "description": "xz: standing, viewed from -y (default); yz: standing, viewed from +x; xy: lying on the floor, viewed from above" },
            "mode": { "type": "string", "enum": ["flat", "extrude", "heightmap"], "description": "flat: one voxel thick (default); extrude: `depth` thick; heightmap: columns 1..depth tall by brightness" },
            "depth": { "type": "integer", "description": "extrude: thickness (default 4); heightmap: tallest column (default 8)" },
            "invert": { "type": "boolean", "description": "heightmap: dark is tall" },
            "offset": xyz("Part coordinates of the image's minimum corner (default [0, 0, 0])"),
            "alpha_threshold": { "type": "integer", "description": "Pixels with alpha below this are empty, 0-255 (default 128)" },
            "add_colors": { "type": "boolean", "description": "Put the image's main colors into free palette slots first, for exact colors" },
            "max_colors": { "type": "integer", "description": "add_colors: most slots to use (default: every free slot)" },
            "palette_range": range,
        }), &["path"]),
    ]
}

pub fn dispatch(ctx: &mut Ctx, name: &str, args: &Value) -> Option<Result<Vec<Value>>> {
    Some(match name {
        "vary_colors" => vary_colors(ctx, args),
        "shade_edges" => shade_edges(ctx, args),
        "paint_faces" => paint_faces(ctx, args),
        "list_palette_presets" => list_presets(),
        "apply_palette_preset" => apply_preset(ctx, args),
        "voxelize_image" => voxelize_image(ctx, args),
        _ => return None,
    })
}

// ---------------------------------------------------------------- args

/// One color or a list of them.
fn color_list(scene: &Scene, v: &Value) -> Result<Vec<u8>> {
    let list: Vec<u8> = match v.as_array() {
        Some(a) => a.iter().map(|c| color(scene, c)).collect::<Result<_>>()?,
        None => vec![color(scene, v)?],
    };
    ensure!(list.iter().all(|&c| c != 0), "color 0 is empty space");
    Ok(list)
}

fn opt_colors(scene: &Scene, args: &Value, key: &str) -> Result<Option<Vec<u8>>> {
    get(args, key).map(|v| color_list(scene, v)).transpose()
}

fn pattern_arg(args: &Value, default: Pattern) -> Result<Pattern> {
    Ok(match string(args, "pattern") {
        None => default,
        Some("random") => Pattern::Random,
        Some("noise") => Pattern::Noise,
        Some("dither") => Pattern::Dither,
        Some(p) => bail!("unknown pattern {p:?}; use random, noise or dither"),
    })
}

fn seed_arg(args: &Value) -> u32 {
    int(args, "seed").unwrap_or(1) as u32
}

/// Palette indices allowed as matches (`palette_range`, default 1..=255).
fn palette_range(args: &Value) -> Result<Vec<u8>> {
    let Some(v) = get(args, "palette_range") else { return Ok((1..=255).collect()) };
    let a = v.as_array().filter(|a| a.len() == 2).context("palette_range is [first, last]")?;
    let n = |i: usize| a[i].as_u64().filter(|n| (1..=255).contains(n)).context("palette_range indices are 1-255");
    let (lo, hi) = (n(0)?.min(n(1)?), n(0)?.max(n(1)?));
    Ok((lo as u8..=hi as u8).collect())
}

/// Cells a surface tool may change: the selection, or every cell.
fn target(ctx: &Ctx) -> Option<Mask> {
    ctx.ed.editor.selection().cloned()
}

fn in_target(sel: &Option<Mask>, i: usize) -> bool {
    sel.as_ref().is_none_or(|s| s.get(i).unwrap_or(false))
}

/// "12 #aabbcc -> 34 #ddeeff x 50" lines for a recolor.
fn describe(scene: &Scene, old: &Volume, new: &Volume) -> String {
    let mut moves: HashMap<(u8, u8), usize> = HashMap::new();
    for (_, a, b) in old.cells.diff(&new.cells) {
        *moves.entry((a, b)).or_default() += 1;
    }
    let mut list: Vec<_> = moves.into_iter().collect();
    list.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let total: usize = list.iter().map(|m| m.1).sum();
    let hex = |c: u8| super::hex(scene.palette[c as usize]);
    let mut lines: Vec<String> = list.iter().take(24).map(|((a, b), n)| format!("{a} {} -> {b} {} x {n}", hex(*a), hex(*b))).collect();
    if list.len() > 24 {
        lines.push(format!("... and {} more pairs", list.len() - 24));
    }
    format!("{total} voxels recolored:\n{}", lines.join("\n"))
}

// ---------------------------------------------------------------- noise

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Pattern {
    Random,
    Noise,
    Dither,
}

fn hash(p: IVec3, seed: u32) -> u32 {
    let mut h = (p.x as u32).wrapping_mul(0x8da6_b343) ^ (p.y as u32).wrapping_mul(0xd816_3841) ^ (p.z as u32).wrapping_mul(0xcb1a_b31f) ^ seed.wrapping_mul(0x1656_67b1);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

/// White noise in [0, 1).
fn rand01(p: IVec3, seed: u32) -> f32 {
    (hash(p, seed) >> 8) as f32 / (1u32 << 24) as f32
}

fn value_noise(p: Vec3, seed: u32) -> f32 {
    let base = p.floor();
    let f = p - base;
    let s = f * f * (3.0 - 2.0 * f);
    let i = base.as_ivec3();
    let mut acc = 0.0;
    for k in 0..8 {
        let o = IVec3::new(k & 1, (k >> 1) & 1, k >> 2);
        let w = |o: i32, s: f32| if o == 1 { s } else { 1.0 - s };
        acc += w(o.x, s.x) * w(o.y, s.y) * w(o.z, s.z) * rand01(i + o, seed);
    }
    acc
}

/// Smooth 3D noise in [0, 1) with features about `scale` voxels across.
pub fn fbm(p: IVec3, scale: f32, seed: u32) -> f32 {
    let mut q = (p.as_vec3() + 0.5) / scale.max(0.5);
    let (mut sum, mut amp, mut norm) = (0.0, 1.0, 0.0);
    for o in 0..3u32 {
        sum += amp * value_noise(q + Vec3::splat(o as f32 * 17.3), seed.wrapping_add(o.wrapping_mul(0x9e37_79b9)));
        norm += amp;
        amp *= 0.5;
        q *= 2.0;
    }
    sum / norm
}

/// Ordered-dither threshold in [0, 1): a 4x4x4 Bayer-style matrix whose
/// first half is a 3D checkerboard, so faces on every axis dither evenly.
pub fn dither(p: IVec3) -> f32 {
    // Rank of each cell of a 2x2x2 block: the even-parity cells first.
    const BASE: [u32; 8] = [0, 5, 7, 2, 6, 3, 1, 4];
    let cell = |q: IVec3| BASE[(q.x & 1 | (q.y & 1) << 1 | (q.z & 1) << 2) as usize];
    let p = p & IVec3::splat(3);
    ((cell(p) * 8 + cell(p >> 1)) as f32 + 0.5) / 64.0
}

/// A value in [0, 1) per cell for `pattern`. Random and noise values are
/// replaced by their rank, so thresholds and weights give exact shares.
fn pattern_values(v: &Volume, cells: &[usize], pattern: Pattern, scale: f32, seed: u32) -> Vec<f32> {
    let raw: Vec<f32> = cells
        .iter()
        .map(|&i| {
            let p = v.position(i);
            match pattern {
                Pattern::Random => rand01(p, seed),
                Pattern::Noise => fbm(p, scale, seed),
                Pattern::Dither => dither(p),
            }
        })
        .collect();
    if pattern == Pattern::Dither {
        return raw;
    }
    let mut order: Vec<usize> = (0..cells.len()).collect();
    order.sort_by(|&a, &b| raw[a].total_cmp(&raw[b]).then(a.cmp(&b)));
    let mut out = vec![0.0; cells.len()];
    let n = cells.len() as f32;
    for (rank, &k) in order.iter().enumerate() {
        out[k] = (rank as f32 + 0.5) / n;
    }
    out
}

/// Which weighted band a value in [0, 1) falls in.
fn band(cum: &[f32], t: f32) -> usize {
    cum.iter().position(|&c| t < c).unwrap_or(cum.len() - 1)
}

/// Recolor `cells` of `v` with a mix of `with` (shares `weights`), changing
/// a fraction `amount` of them.
pub fn vary(v: &Volume, cells: &[usize], with: &[u8], weights: &[f32], pattern: Pattern, scale: f32, seed: u32, amount: f32) -> Volume {
    let mut out = v.clone();
    if cells.is_empty() || with.is_empty() {
        return out;
    }
    let total: f32 = weights.iter().sum::<f32>().max(1e-6);
    let cum: Vec<f32> = weights.iter().scan(0.0, |s, w| {
        *s += w / total;
        Some(*s)
    }).collect();
    let amount = amount.clamp(0.0, 1.0);
    let (chosen, t): (Vec<usize>, Vec<f32>) = if pattern == Pattern::Dither {
        // The same thresholds pick the cells and their colors.
        cells.iter().map(|&i| (i, dither(v.position(i)))).filter(|&(_, t)| t < amount).map(|(i, t)| (i, t / amount.max(1e-6))).unzip()
    } else {
        let keep: Vec<usize> = if amount >= 1.0 {
            cells.to_vec()
        } else {
            let k = pattern_values(v, cells, pattern, scale, seed ^ 0x5bd1_e995);
            cells.iter().zip(k).filter(|&(_, k)| k < amount).map(|(&i, _)| i).collect()
        };
        let t = pattern_values(v, &keep, pattern, scale, seed);
        (keep, t)
    };
    for (&i, &t) in chosen.iter().zip(&t) {
        out.cells[i] = with[band(&cum, t)];
    }
    out
}

fn vary_colors(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let with = color_list(ctx.scene, get(a, "with").context("missing `with` (colors to mix in)")?)?;
    let weights: Vec<f32> = match get(a, "weights").and_then(Value::as_array) {
        Some(w) => {
            ensure!(w.len() == with.len(), "`weights` needs one number per `with` color");
            w.iter().map(|x| x.as_f64().map(|f| f.max(0.0) as f32).context("weights are numbers")).collect::<Result<_>>()?
        }
        None => vec![1.0; with.len()],
    };
    let only = opt_colors(ctx.scene, a, "colors")?;
    let sel = target(ctx);
    let v = &ctx.scene.volume;
    let cells: Vec<usize> = (0..v.cells.len()).filter(|&i| v.cells[i] != 0 && in_target(&sel, i) && only.as_ref().is_none_or(|o| o.contains(&v.cells[i]))).collect();
    if cells.is_empty() {
        bail!("no voxels to recolor{}", if sel.is_some() { " in the selection" } else { "" });
    }
    let new = vary(v, &cells, &with, &weights, pattern_arg(a, Pattern::Noise)?, num(a, "scale").unwrap_or(4.0), seed_arg(a), num(a, "amount").unwrap_or(1.0));
    let report = describe(ctx.scene, v, &new);
    let changed = apply(ctx, new);
    let mut out = edited(ctx, changed, "Varied colors");
    out.push(text(report));
    Ok(out)
}

// ---------------------------------------------------------------- color

/// sRGB to OKLab (L in 0..1).
pub fn oklab(c: [u8; 4]) -> Vec3 {
    let lin = crate::scene::srgb_to_linear(Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32) / 255.0);
    let (r, g, b) = (lin.x, lin.y, lin.z);
    let l = (0.412_221_47 * r + 0.536_332_55 * g + 0.051_445_995 * b).cbrt();
    let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b).cbrt();
    let s = (0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b).cbrt();
    Vec3::new(
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    )
}

/// Color distance with chroma counted double, so hue holds.
fn lab_dist(a: Vec3, b: Vec3) -> f32 {
    let d = a - b;
    d.x * d.x + 2.0 * (d.y * d.y + d.z * d.z)
}

/// The palette entry in `allowed` nearest to `c`.
fn nearest(palette: &[[u8; 4]], allowed: &[u8], c: Vec3) -> u8 {
    *allowed.iter().min_by(|&&x, &&y| lab_dist(oklab(palette[x as usize]), c).total_cmp(&lab_dist(oklab(palette[y as usize]), c))).unwrap_or(&1)
}

/// The palette color in `allowed` closest to `c` moved `amount` of the way
/// to black (or white), and clearly darker (lighter) than `c`.
pub fn shade_of(palette: &[[u8; 4]], allowed: &[u8], c: u8, lighter: bool, amount: f32) -> Option<u8> {
    let o = oklab(palette[c as usize]);
    // Scaling linear RGB scales all of L, a and b alike in OKLab.
    let goal = if lighter { o + (Vec3::new(1.0, 0.0, 0.0) - o) * amount } else { o * (1.0 - amount) };
    allowed
        .iter()
        .copied()
        .filter(|&i| i != c)
        .map(|i| (i, oklab(palette[i as usize])))
        .filter(|(_, l)| if lighter { l.x > o.x + 0.02 } else { l.x < o.x - 0.02 })
        .min_by(|a, b| lab_dist(a.1, goal).total_cmp(&lab_dist(b.1, goal)))
        .map(|(i, _)| i)
}

// ---------------------------------------------------------------- shading

fn solid(v: &Volume, p: IVec3) -> bool {
    v.get(p) != 0
}

/// Bit k set when the neighbor `NEIGHBORS[k]` (+x, -x, +y, -y, +z, -z) is
/// empty or outside the canvas.
pub fn exposed(v: &Volume, p: IVec3) -> u8 {
    NEIGHBORS.iter().enumerate().fold(0, |m, (k, &d)| if solid(v, p + d) { m } else { m | 1 << k })
}

/// How many axes a voxel is exposed on (0-3).
fn exposed_axes(mask: u8) -> u32 {
    (0..3).filter(|k| mask >> (2 * k) & 3 != 0).count() as u32
}

/// Fraction of the cells within `r` (a cube, minus the center) that are empty.
pub fn openness(v: &Volume, p: IVec3, r: i32) -> f32 {
    let mut empty = 0;
    for z in -r..=r {
        for y in -r..=r {
            for x in -r..=r {
                if (x, y, z) != (0, 0, 0) && !solid(v, p + IVec3::new(x, y, z)) {
                    empty += 1;
                }
            }
        }
    }
    empty as f32 / ((2 * r + 1).pow(3) - 1) as f32
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ShadeMode {
    Edges,
    Corners,
    Cavity,
}

/// How many shades darker (negative) or lighter (positive) each cell
/// should go; 0 for untouched.
pub fn shade_steps(v: &Volume, cells: &[usize], mode: ShadeMode, highlight: bool, radius: i32, threshold: f32, levels: i32) -> Vec<i32> {
    cells
        .iter()
        .map(|&i| {
            let p = v.position(i);
            let m = exposed(v, p);
            if m == 0 {
                return 0;
            }
            match mode {
                ShadeMode::Edges if exposed_axes(m) >= 2 => {
                    if highlight && m & 1 << 4 != 0 { 1 } else { -1 }
                }
                ShadeMode::Corners if exposed_axes(m) == 3 => -1,
                ShadeMode::Cavity => {
                    let open = openness(v, p, radius);
                    -((1..=levels).filter(|&k| open < threshold * (levels - k + 1) as f32 / levels as f32).count() as i32)
                }
                _ => 0,
            }
        })
        .collect()
}

fn shade_edges(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let mode = match string(a, "mode").context("missing `mode` (edges, corners or cavity)")? {
        "edges" => ShadeMode::Edges,
        "corners" => ShadeMode::Corners,
        "cavity" | "ao" => ShadeMode::Cavity,
        m => bail!("unknown mode {m:?}; use edges, corners or cavity"),
    };
    let highlight = boolean(a, "highlight").unwrap_or(false);
    let amount = num(a, "amount").unwrap_or(0.2).clamp(0.02, 1.0);
    let dark = get(a, "color").map(|c| color(ctx.scene, c)).transpose()?;
    let light = get(a, "highlight_color").map(|c| color(ctx.scene, c)).transpose()?;
    let only = opt_colors(ctx.scene, a, "only_color")?;
    let radius = int(a, "radius").unwrap_or(2).clamp(1, 6) as i32;
    let threshold = num(a, "threshold").unwrap_or(0.3);
    let levels = int(a, "levels").unwrap_or(1).clamp(1, 3) as i32;
    let allowed = palette_range(a)?;
    let sel = target(ctx);
    let v = &ctx.scene.volume;
    let cells: Vec<usize> = (0..v.cells.len()).filter(|&i| v.cells[i] != 0 && in_target(&sel, i) && only.as_ref().is_none_or(|o| o.contains(&v.cells[i]))).collect();
    let steps = shade_steps(v, &cells, mode, highlight, radius, threshold, levels);
    let palette = &ctx.scene.palette;
    let mut memo: HashMap<(u8, i32), Option<u8>> = HashMap::new();
    let mut missing: Vec<u8> = Vec::new();
    let mut new = v.clone();
    for (&i, &s) in cells.iter().zip(&steps) {
        if s == 0 {
            continue;
        }
        let c = v.cells[i];
        let to = *memo.entry((c, s)).or_insert_with(|| match (s > 0, dark, light) {
            (false, Some(d), _) => Some(d),
            (true, _, Some(l)) => Some(l),
            _ => (0..s.abs()).try_fold(c, |c, _| shade_of(palette, &allowed, c, s > 0, amount)),
        });
        match to {
            Some(to) => new.cells[i] = to,
            None if !missing.contains(&c) => missing.push(c),
            None => {}
        }
    }
    let mut report = describe(ctx.scene, v, &new);
    if !missing.is_empty() {
        report.push_str(&format!("\nNo darker/lighter palette shade for colors {missing:?}; pass `color` / `highlight_color` or add shades with set_palette_colors."));
    }
    let changed = apply(ctx, new);
    let mut out = edited(ctx, changed, "Shaded");
    out.push(text(report));
    Ok(out)
}

// ---------------------------------------------------------------- paint by face

fn direction(name: &str) -> Result<Vec<IVec3>> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "top" | "up" | "+z" | "z" => vec![IVec3::Z],
        "bottom" | "down" | "under" | "-z" => vec![IVec3::NEG_Z],
        "+x" | "x" | "east" | "right" => vec![IVec3::X],
        "-x" | "west" | "left" => vec![IVec3::NEG_X],
        "+y" | "y" | "north" | "back" => vec![IVec3::Y],
        "-y" | "south" | "front" => vec![IVec3::NEG_Y],
        "sides" | "side" => vec![IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y],
        "all" | "any" => NEIGHBORS.to_vec(),
        d => bail!("unknown direction {d:?}"),
    })
}

pub struct Paint {
    pub dirs: Vec<IVec3>,
    pub color: u8,
    pub depth: i32,
    pub vary_depth: bool,
    pub coverage: f32,
    pub pattern: Pattern,
    pub scale: f32,
    pub open: bool,
    pub seed: u32,
}

/// Paint cells (those `ok` allows) whose neighbor along one of `dirs` is
/// empty, and up to `depth` cells inward.
pub fn paint(v: &Volume, ok: &dyn Fn(usize) -> bool, p: &Paint) -> Volume {
    let mut faces: Vec<(usize, IVec3)> = Vec::new();
    for i in 0..v.cells.len() {
        if v.cells[i] == 0 || !ok(i) {
            continue;
        }
        let at = v.position(i);
        for &d in &p.dirs {
            let clear = |mut q: IVec3| loop {
                q += d;
                if !v.contains(q) {
                    break true;
                }
                if solid(v, q) {
                    break false;
                }
                if !p.open {
                    break true;
                }
            };
            if clear(at) {
                faces.push((i, d));
            }
        }
    }
    let cells: Vec<usize> = faces.iter().map(|f| f.0).collect();
    let t = if p.coverage >= 1.0 { vec![0.0; cells.len()] } else { pattern_values(v, &cells, p.pattern, p.scale, p.seed) };
    let mut out = v.clone();
    for (&(i, d), &t) in faces.iter().zip(&t) {
        if t >= p.coverage {
            continue;
        }
        let at = v.position(i);
        let depth = if p.vary_depth { 1 + (fbm(at, p.scale, p.seed ^ 0x27d4_eb2f) * p.depth as f32) as i32 } else { p.depth };
        for k in 0..depth.min(p.depth).max(1) {
            let q = at - d * k;
            if !solid(v, q) || !ok(v.index(q)) {
                break;
            }
            out.cells[v.index(q)] = p.color;
        }
    }
    out
}

fn paint_faces(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let mut dirs = Vec::new();
    match get(a, "direction").context("missing `direction`")? {
        Value::Array(list) => {
            for d in list {
                dirs.extend(direction(d.as_str().context("directions are strings")?)?);
            }
        }
        d => dirs.extend(direction(d.as_str().context("direction is a string like \"top\"")?)?),
    }
    dirs.dedup();
    let c = super::color_arg(ctx.scene, a, "color")?;
    ensure!(c != 0, "color 0 is empty space; use edit_model clear to erase");
    let only = opt_colors(ctx.scene, a, "only_color")?;
    let coverage = num(a, "coverage").unwrap_or(1.0).clamp(0.0, 1.0);
    let p = Paint {
        dirs,
        color: c,
        depth: int(a, "depth").unwrap_or(1).clamp(1, 256) as i32,
        vary_depth: boolean(a, "vary_depth").unwrap_or(false),
        coverage,
        pattern: pattern_arg(a, Pattern::Noise)?,
        scale: num(a, "scale").unwrap_or(3.0),
        open: boolean(a, "open").unwrap_or(false),
        seed: seed_arg(a),
    };
    let sel = target(ctx);
    let v = &ctx.scene.volume;
    let ok = |i: usize| in_target(&sel, i) && only.as_ref().is_none_or(|o| o.contains(&v.cells[i]));
    let new = paint(v, &ok, &p);
    let report = describe(ctx.scene, v, &new);
    let changed = apply(ctx, new);
    let mut out = edited(ctx, changed, "Painted faces");
    out.push(text(report));
    Ok(out)
}

// ---------------------------------------------------------------- palettes

/// Palette slots used by any part's voxels.
fn used_slots(scene: &Scene) -> [bool; 256] {
    let mut used = [false; 256];
    let w = &scene.world;
    let vols = w.models().into_iter().filter_map(|id| w.node(id).and_then(|n| n.volume()));
    for v in vols.chain(std::iter::once(&scene.volume)) {
        for (_, c) in v.cells.filled() {
            used[c as usize] = true;
        }
    }
    used[0] = true;
    used
}

/// Write `colors` into slots, resetting unused slots to plain diffuse.
fn write_slots(ctx: &mut Ctx, slots: &[u8], colors: &[[u8; 4]], used: &[bool; 256]) {
    for (&i, &c) in slots.iter().zip(colors) {
        let i = i as usize;
        ctx.scene.palette[i] = [c[0], c[1], c[2], 255];
        ctx.ed.palette.look[i] = None;
        if used[i] {
            ctx.scene.sync_color(i);
        } else {
            ctx.scene.materials[i] = crate::scene::material(i as u32, ctx.scene.palette[i], None);
        }
        if !ctx.ed.tools.surface.reserved.contains(&(i as u8)) {
            ctx.ed.tools.surface.reserved.push(i as u8);
        }
    }
    super::materials_changed(ctx);
}

fn list_presets() -> Result<Vec<Value>> {
    let list: Vec<Value> = presets::PRESETS.iter().map(|p| json!({ "name": p.name, "count": p.colors.len(), "by": p.author, "colors": p.colors })).collect();
    Ok(vec![text(serde_json::to_string(&list)?)])
}

fn apply_preset(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let name = string(a, "name").context("missing `name`")?;
    let preset = presets::find(name).ok_or_else(|| anyhow!("no preset {name:?}; have {}", presets::PRESETS.iter().map(|p| p.name).collect::<Vec<_>>().join(", ")))?;
    let n = preset.colors.len();
    let used = used_slots(ctx.scene);
    let reserved = &ctx.ed.tools.surface.reserved;
    let free = |i: usize| !used[i] && !reserved.contains(&(i as u8));
    let overwrite = boolean(a, "overwrite").unwrap_or(false);
    let slots: Vec<u8> = match int(a, "start_index") {
        Some(s) => {
            ensure!(s >= 1 && s as usize + n - 1 <= 255, "start_index must be 1..={} for {n} colors", 256 - n);
            let slots: Vec<u8> = (s as usize..s as usize + n).map(|i| i as u8).collect();
            let busy: Vec<u8> = slots.iter().copied().filter(|&i| used[i as usize]).collect();
            ensure!(busy.is_empty() || overwrite, "slots {busy:?} hold colors voxels use; pass overwrite: true or another start_index");
            slots
        }
        None => match (1..=256 - n).find(|&s| (s..s + n).all(free)) {
            Some(s) => (s..s + n).map(|i| i as u8).collect(),
            None => {
                let scattered: Vec<u8> = (1..256).filter(|&i| free(i)).map(|i| i as u8).take(n).collect();
                ensure!(scattered.len() == n, "only {} free palette slots for {n} colors; pass start_index with overwrite: true", scattered.len());
                scattered
            }
        },
    };
    let colors: Vec<[u8; 4]> = preset.colors.iter().map(|c| super::parse_hex(c)).collect::<Result<_>>()?;
    write_slots(ctx, &slots, &colors, &used);
    let replaced: Vec<u8> = slots.iter().copied().filter(|&i| used[i as usize]).collect();
    let used = used_slots(ctx.scene);
    let reserved = &ctx.ed.tools.surface.reserved;
    let free_left = (1..256).filter(|&i| !used[i] && !reserved.contains(&(i as u8))).count();
    let span = if slots.windows(2).all(|w| w[1] == w[0] + 1) { format!("slots {}-{}", slots[0], slots[n - 1]) } else { format!("slots {slots:?}") };
    let map: Vec<String> = slots.iter().zip(preset.colors).map(|(i, c)| format!("{i}={c}")).collect();
    let mut note = format!(
        "Wrote {} ({n} colors) into {span}; use palette_range [{}, {}] in voxelize_image/shade_edges to stick to it.\n{}\nColors in use: {}. Free slots left: {free_left}.",
        preset.name,
        slots.iter().min().unwrap(),
        slots.iter().max().unwrap(),
        map.join(" "),
        (1..256).filter(|&i| used[i]).count()
    );
    if !replaced.is_empty() {
        note.push_str(&format!(" Replaced colors in use at {replaced:?}."));
    }
    let entries: Vec<(u8, [u8; 4])> = slots.iter().map(|&i| (i, ctx.scene.palette[i as usize])).collect();
    let (w, h, px) = swatches(&entries);
    Ok(vec![text(note), super::image(w, h, &px)?])
}

/// 3x5 digits, three bits per row, top row first.
const DIGITS: [[u8; 5]; 10] = [
    [7, 5, 5, 5, 7],
    [2, 6, 2, 2, 7],
    [7, 1, 7, 4, 7],
    [7, 1, 7, 1, 7],
    [5, 5, 7, 1, 1],
    [7, 4, 7, 1, 7],
    [7, 4, 7, 5, 7],
    [7, 1, 1, 1, 1],
    [7, 5, 7, 5, 7],
    [7, 5, 7, 1, 7],
];

/// A grid of color swatches, each labeled with its palette index.
pub fn swatches(entries: &[(u8, [u8; 4])]) -> (u32, u32, Vec<u8>) {
    const CELL: usize = 48;
    const GAP: usize = 4;
    let cols = entries.len().clamp(1, 8);
    let rows = entries.len().div_ceil(cols).max(1);
    let (w, h) = (cols * (CELL + GAP) + GAP, rows * (CELL + GAP) + GAP);
    let mut px = vec![0u8; w * h * 4];
    for p in px.chunks_exact_mut(4) {
        p.copy_from_slice(&[40, 40, 44, 255]);
    }
    let mut put = |x: usize, y: usize, c: [u8; 4]| px[(y * w + x) * 4..][..4].copy_from_slice(&[c[0], c[1], c[2], 255]);
    for (k, &(index, c)) in entries.iter().enumerate() {
        let (x0, y0) = (GAP + (k % cols) * (CELL + GAP), GAP + (k / cols) * (CELL + GAP));
        for y in 0..CELL {
            for x in 0..CELL {
                put(x0 + x, y0 + y, c);
            }
        }
        let luma = 0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32;
        let ink = if luma > 140.0 { [0, 0, 0, 255] } else { [255, 255, 255, 255] };
        for (j, ch) in index.to_string().bytes().enumerate() {
            let glyph = DIGITS[(ch - b'0') as usize];
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..3 {
                    if bits >> (2 - col) & 1 == 1 {
                        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                            put(x0 + 4 + j * 8 + col * 2 + dx, y0 + 4 + row * 2 + dy, ink);
                        }
                    }
                }
            }
        }
    }
    (w as u32, h as u32, px)
}

// ---------------------------------------------------------------- images

pub struct Image {
    pub w: usize,
    pub h: usize,
    /// RGBA, row 0 at the top.
    pub px: Vec<[u8; 4]>,
}

pub fn load_image(path: &Path) -> Result<Image> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.starts_with(b"\x89PNG") {
        let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
        dec.set_transformations(png::Transformations::EXPAND | png::Transformations::ALPHA | png::Transformations::STRIP_16);
        let mut reader = dec.read_info()?;
        let mut buf = vec![0; reader.output_buffer_size().context("image too large")?];
        let info = reader.next_frame(&mut buf)?;
        let ch = info.color_type.samples();
        let px = buf[..info.buffer_size()]
            .chunks_exact(ch)
            .map(|c| match ch {
                4 => [c[0], c[1], c[2], c[3]],
                3 => [c[0], c[1], c[2], 255],
                2 => [c[0], c[0], c[0], c[1]],
                _ => [c[0], c[0], c[0], 255],
            })
            .collect();
        Ok(Image { w: info.width as usize, h: info.height as usize, px })
    } else if bytes.starts_with(&[0xff, 0xd8]) {
        use zune_jpeg::zune_core::{colorspace::ColorSpace, options::DecoderOptions};
        let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
        let mut dec = zune_jpeg::JpegDecoder::new_with_options(std::io::Cursor::new(bytes), opts);
        let rgb = dec.decode().map_err(|e| anyhow!("decoding JPEG: {e:?}"))?;
        let info = dec.info().context("JPEG has no header")?;
        let px = rgb.chunks_exact(3).map(|c| [c[0], c[1], c[2], 255]).collect();
        Ok(Image { w: info.width as usize, h: info.height as usize, px })
    } else {
        bail!("{} isn't a PNG or JPEG", path.display())
    }
}

/// Shrink or grow to `w` x `h`: each output pixel is empty when most of
/// its source box is transparent, else the box's dominant color (pixel
/// art stays crisp), or its average when no color dominates (photos).
pub fn resample(img: &Image, w: usize, h: usize, alpha_threshold: u8) -> Vec<Option<[u8; 3]>> {
    let mut out = Vec::with_capacity(w * h);
    for v in 0..h {
        let (y0, y1) = (v * img.h / h, ((v + 1) * img.h / h).max(v * img.h / h + 1));
        for u in 0..w {
            let (x0, x1) = (u * img.w / w, ((u + 1) * img.w / w).max(u * img.w / w + 1));
            let mut counts: HashMap<[u8; 3], usize> = HashMap::new();
            let (mut total, mut sum) = (0, [0u64; 3]);
            for y in y0..y1.min(img.h) {
                for x in x0..x1.min(img.w) {
                    total += 1;
                    let p = img.px[y * img.w + x];
                    if p[3] >= alpha_threshold.max(1) {
                        *counts.entry([p[0], p[1], p[2]]).or_default() += 1;
                        for k in 0..3 {
                            sum[k] += p[k] as u64;
                        }
                    }
                }
            }
            let opaque: usize = counts.values().sum();
            if opaque * 2 < total || opaque == 0 {
                out.push(None);
                continue;
            }
            let (&mode, &n) = counts.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0))).unwrap();
            out.push(Some(if n * 3 >= opaque { mode } else { sum.map(|s| (s / opaque as u64) as u8) }));
        }
    }
    out
}

fn voxelize_image(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let path = string(a, "path").context("missing `path`")?;
    let img = load_image(Path::new(path))?;
    ensure!(img.w > 0 && img.h > 0, "empty image");
    let max = crate::ops::MAX_DIM as usize;
    let (w, h, note) = match (int(a, "height"), int(a, "width")) {
        (Some(h), _) => {
            let h = (h.max(1) as usize).min(max);
            ((h * img.w + img.h / 2) / img.h, h, "")
        }
        (None, Some(w)) => {
            let w = (w.max(1) as usize).min(max);
            (w, (w * img.h + img.w / 2) / img.w, "")
        }
        (None, None) if img.w.max(img.h) > 256 => {
            let k = 256.0 / img.w.max(img.h) as f32;
            ((img.w as f32 * k).round() as usize, (img.h as f32 * k).round() as usize, " (scaled down to 256; pass height or width)")
        }
        _ => (img.w, img.h, ""),
    };
    let (w, h) = (w.max(1), h.max(1));
    let mode = string(a, "mode").unwrap_or("flat");
    let depth = match mode {
        "flat" => 1,
        "extrude" => int(a, "depth").unwrap_or(4),
        "heightmap" => int(a, "depth").unwrap_or(8),
        m => bail!("unknown mode {m:?}; use flat, extrude or heightmap"),
    }
    .clamp(1, max as i64) as i32;
    let alpha = int(a, "alpha_threshold").unwrap_or(128).clamp(0, 255) as u8;
    let pixels = resample(&img, w, h, alpha);
    ensure!(pixels.iter().any(Option::is_some), "every pixel is transparent at alpha_threshold {alpha}");

    // Colors: optionally add the main ones to free slots, then match.
    let mut allowed = palette_range(a)?;
    let mut added: Vec<u8> = Vec::new();
    if boolean(a, "add_colors").unwrap_or(false) {
        let used = used_slots(ctx.scene);
        let reserved = &ctx.ed.tools.surface.reserved;
        let free: Vec<u8> = (1..256).rev().filter(|&i| !used[i] && !reserved.contains(&(i as u8))).map(|i| i as u8).collect();
        let limit = int(a, "max_colors").map_or(free.len(), |m| (m.max(0) as usize).min(free.len()));
        let mut counts: HashMap<[u8; 3], usize> = HashMap::new();
        for c in pixels.iter().flatten() {
            *counts.entry(*c).or_default() += 1;
        }
        let mut by_count: Vec<([u8; 3], usize)> = counts.into_iter().collect();
        by_count.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let palette = &ctx.scene.palette;
        let mut picked: Vec<Vec3> = Vec::new();
        let mut colors: Vec<[u8; 4]> = Vec::new();
        for (c, _) in by_count {
            if colors.len() >= limit {
                break;
            }
            let rgba = [c[0], c[1], c[2], 255];
            let lab = oklab(rgba);
            let exact = allowed.iter().any(|&i| palette[i as usize][..3] == c);
            if !exact && picked.iter().all(|&p| lab_dist(p, lab) > 0.0009) {
                picked.push(lab);
                colors.push(rgba);
            }
        }
        added = free[..colors.len()].to_vec();
        write_slots(ctx, &added, &colors, &used);
        allowed.extend(&added);
    }
    let mut memo: HashMap<[u8; 3], u8> = HashMap::new();
    let palette = &ctx.scene.palette;
    let mut index = |c: [u8; 3]| *memo.entry(c).or_insert_with(|| nearest(palette, &allowed, oklab([c[0], c[1], c[2], 255])));

    // Place: image u runs along the plane's first axis, v (up) along its second.
    let (ua, va, da) = match string(a, "plane").unwrap_or("xz") {
        "xz" => (0, 2, 1),
        "yz" => (1, 2, 0),
        "xy" => (0, 1, 2),
        p => bail!("unknown plane {p:?}; use xz, yz or xy"),
    };
    let offset = super::opt_ivec3(a, "offset")?.unwrap_or(IVec3::ZERO).max(IVec3::ZERO);
    let invert = boolean(a, "invert").unwrap_or(false);
    let mut cells: Vec<(IVec3, u8)> = Vec::new();
    for (k, px) in pixels.iter().enumerate() {
        let Some(c) = *px else { continue };
        let (u, v) = ((k % w) as i32, (h - 1 - k / w) as i32);
        let thick = if mode == "heightmap" {
            let lum = (0.2126 * c[0] as f32 + 0.7152 * c[1] as f32 + 0.0722 * c[2] as f32) / 255.0;
            1 + ((if invert { 1.0 - lum } else { lum }) * (depth - 1) as f32).round() as i32
        } else {
            depth
        };
        let ci = index(c);
        for d in 0..thick {
            let mut p = IVec3::ZERO;
            p[ua] = u;
            p[va] = v;
            p[da] = d;
            cells.push((offset + p, ci));
        }
    }
    let mut extent = IVec3::ZERO;
    extent[ua] = w as i32;
    extent[va] = h as i32;
    extent[da] = depth;
    let old = &ctx.scene.volume;
    let dims = old.dims.max(offset + extent);
    ensure!(dims.max_element() <= crate::ops::MAX_DIM, "the image needs a canvas of {dims:?}, over {} per axis", crate::ops::MAX_DIM);
    let mut new = if dims == old.dims {
        old.clone()
    } else {
        let mut n = Volume::new(dims);
        for (i, &c) in old.cells.iter().enumerate() {
            if c != 0 {
                let j = n.index(old.position(i));
                n.cells[j] = c;
            }
        }
        n
    };
    for &(p, c) in &cells {
        let j = new.index(p);
        new.cells[j] = c;
    }
    new.refresh_bricks();
    let grew = dims != old.dims;
    let mut colors: Vec<u8> = cells.iter().map(|c| c.1).collect();
    colors.sort();
    colors.dedup();
    let changed = apply(ctx, new);
    let mut out = edited(ctx, changed, &format!("Voxelized {}x{} image as {w}x{h}{note}", img.w, img.h));
    let mut info = format!("{} voxels, {mode}, at {:?}; uses colors {colors:?}.", cells.len(), offset.to_array());
    if grew {
        info.push_str(&format!(" Canvas grew to {:?}.", dims.to_array()));
    }
    if !added.is_empty() {
        info.push_str(&format!(" Added {} image colors to palette slots {added:?}.", added.len()));
    }
    out.push(text(info));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube(n: i32, lo: i32, hi: i32, c: u8) -> Volume {
        let mut v = Volume::new(IVec3::splat(n));
        for i in 0..v.cells.len() {
            let p = v.position(i);
            if p.cmpge(IVec3::splat(lo)).all() && p.cmple(IVec3::splat(hi)).all() {
                v.cells[i] = c;
            }
        }
        v
    }

    #[test]
    fn vary_is_seeded_and_proportional() {
        let v = cube(16, 0, 15, 1);
        let cells: Vec<usize> = (0..v.cells.len()).collect();
        for pattern in [Pattern::Random, Pattern::Noise, Pattern::Dither] {
            let a = vary(&v, &cells, &[2, 3], &[1.0, 3.0], pattern, 4.0, 7, 1.0);
            let b = vary(&v, &cells, &[2, 3], &[1.0, 3.0], pattern, 4.0, 7, 1.0);
            assert_eq!(a.cells, b.cells, "{pattern:?}");
            let twos = a.cells.iter().filter(|&&c| c == 2).count() as f32 / a.cells.len() as f32;
            assert!((twos - 0.25).abs() < 0.01, "{pattern:?} {twos}");
        }
        let a = vary(&v, &cells, &[2, 3], &[1.0, 1.0], Pattern::Noise, 4.0, 1, 1.0);
        let c = vary(&v, &cells, &[2, 3], &[1.0, 1.0], Pattern::Noise, 4.0, 2, 1.0);
        assert_ne!(a.cells, c.cells);
        let half = vary(&v, &cells, &[2], &[1.0], Pattern::Random, 4.0, 1, 0.5);
        let n = half.cells.iter().filter(|&&c| c == 2).count();
        assert_eq!(n, v.cells.len() / 2);
    }

    #[test]
    fn dither_checkers_every_face() {
        for (a, b) in [(0, 1), (1, 2), (0, 2)] {
            for s in 0..4 {
                for t in 0..4 {
                    let mut p = IVec3::splat(5);
                    p[a] = s;
                    p[b] = t;
                    assert_eq!(dither(p) < 0.5, (p.x + p.y + p.z) % 2 == 0);
                }
            }
        }
        let mut seen: Vec<u32> = (0..64).map(|i| (dither(IVec3::new(i & 3, (i >> 2) & 3, i >> 4)) * 64.0) as u32).collect();
        seen.sort();
        assert_eq!(seen, (0..64).collect::<Vec<_>>());
    }

    #[test]
    fn faces_and_edges() {
        let v = cube(6, 1, 4, 1);
        assert_eq!(exposed(&v, IVec3::new(2, 2, 2)), 0);
        assert_eq!(exposed(&v, IVec3::new(2, 2, 4)), 1 << 4);
        assert_eq!(exposed(&v, IVec3::new(4, 4, 4)), 1 | 1 << 2 | 1 << 4);
        let cells: Vec<usize> = (0..v.cells.len()).filter(|&i| v.cells[i] != 0).collect();
        let edges = shade_steps(&v, &cells, ShadeMode::Edges, false, 2, 0.3, 1);
        // A 4-cube has 8 corners and 12 edges of 2 inner cells each.
        assert_eq!(edges.iter().filter(|&&s| s < 0).count(), 8 + 12 * 2);
        let lit = shade_steps(&v, &cells, ShadeMode::Edges, true, 2, 0.3, 1);
        assert_eq!(lit.iter().filter(|&&s| s > 0).count(), 4 + 4 * 2);
        let corners = shade_steps(&v, &cells, ShadeMode::Corners, false, 2, 0.3, 1);
        assert_eq!(corners.iter().filter(|&&s| s < 0).count(), 8);
        // Top paint: one layer, and three deep.
        let top = |depth| Paint { dirs: vec![IVec3::Z], color: 2, depth, vary_depth: false, coverage: 1.0, pattern: Pattern::Noise, scale: 3.0, open: false, seed: 1 };
        let p = paint(&v, &|_| true, &top(1));
        assert_eq!(p.cells.iter().filter(|&&c| c == 2).count(), 16);
        let p = paint(&v, &|_| true, &top(3));
        assert_eq!(p.cells.iter().filter(|&&c| c == 2).count(), 48);
    }

    #[test]
    fn cavity_finds_inner_corners() {
        // An L: floor plus a wall; the crease is enclosed, the top edge is open.
        let mut v = Volume::new(IVec3::splat(10));
        for i in 0..v.cells.len() {
            let p = v.position(i);
            if p.z <= 2 || p.x <= 2 {
                v.cells[i] = 1;
            }
        }
        let crease = openness(&v, IVec3::new(3, 5, 2), 2);
        let flat = openness(&v, IVec3::new(7, 5, 2), 2);
        assert!(crease < 0.3 && flat > 0.35, "{crease} {flat}");
    }

    #[test]
    fn shades_keep_hue() {
        let mut pal = vec![[0u8, 0, 0, 255]; 256];
        pal[1] = [80, 160, 60, 255]; // green
        pal[2] = [40, 90, 30, 255]; // dark green
        pal[3] = [90, 40, 40, 255]; // dark red, same lightness
        pal[4] = [170, 230, 140, 255]; // light green
        let all: Vec<u8> = (1..=4).collect();
        assert_eq!(shade_of(&pal, &all, 1, false, 0.3), Some(2));
        assert_eq!(shade_of(&pal, &all, 1, true, 0.3), Some(4));
        assert_eq!(nearest(&pal, &all, oklab([85, 150, 70, 255])), 1);
    }

    #[test]
    fn resample_keeps_pixel_art() {
        // A 4x4 image of 2x2 blocks: red, green / transparent, blue.
        let mut px = vec![[0u8; 4]; 16];
        for y in 0..4 {
            for x in 0..4 {
                px[y * 4 + x] = match (x / 2, y / 2) {
                    (0, 0) => [255, 0, 0, 255],
                    (1, 0) => [0, 255, 0, 255],
                    (0, 1) => [0, 0, 0, 0],
                    _ => [0, 0, 255, 255],
                };
            }
        }
        let img = Image { w: 4, h: 4, px };
        assert_eq!(resample(&img, 2, 2, 128), vec![Some([255, 0, 0]), Some([0, 255, 0]), None, Some([0, 0, 255])]);
        assert_eq!(resample(&img, 8, 8, 128).len(), 64);
    }

    #[test]
    fn swatch_image_size() {
        let (w, h, px) = swatches(&[(1, [255, 0, 0, 255]), (200, [0, 0, 0, 255])]);
        assert_eq!((w, h), (2 * 52 + 4, 56));
        assert_eq!(px.len(), (w * h * 4) as usize);
    }
}
