//! Meshes to voxels: OBJ (with .mtl colors and textures, and vertex
//! colors) and glTF / GLB (the default scene's node hierarchy, vertex
//! colors, base color factors and textures).
//!
//! The surface is found with a conservative triangle / box overlap test,
//! so every voxel a triangle touches is set and a closed mesh gives a
//! closed (6-connected watertight) shell. Each surface voxel takes the
//! color at the point of its triangles nearest its center. With `solid`,
//! the outside is flood-filled from a padded border and everything it
//! doesn't reach is filled, colored from the nearest surface voxel along
//! its x row.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use glam::{IVec3, Mat4, Quat, Vec2, Vec3};
use serde_json::Value;

use super::Model;
use crate::control::surface::{Image, load_image};
use crate::gltf_import::Gltf;

pub const EXTENSIONS: &[&str] = &["obj", "glb", "gltf"];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshOptions {
    /// Voxels along the mesh's longest side.
    pub size: u32,
    /// Fill the inside (closed meshes), not just the surface shell.
    pub solid: bool,
    /// The file's up axis: Y (OBJ/glTF convention, the default) or Z.
    pub z_up: bool,
}

impl Default for MeshOptions {
    fn default() -> Self {
        MeshOptions { size: 64, solid: true, z_up: false }
    }
}

/// Colors are kept as display (sRGB) values 0..1 and multiplied together.
#[derive(Clone, Debug)]
struct Material {
    color: [f32; 3],
    /// Index into `Mesh::textures`.
    tex: Option<usize>,
}

const WHITE: Material = Material { color: [1.0; 3], tex: None };

struct Tri {
    p: [Vec3; 3],
    /// Texture coordinates, v down (row 0 at the top of the image).
    uv: Option<[Vec2; 3]>,
    col: Option<[[f32; 3]; 3]>,
    mat: u32,
}

#[derive(Default)]
struct Mesh {
    tris: Vec<Tri>,
    mats: Vec<Material>,
    textures: Vec<Image>,
    /// Problems worth telling the user (e.g. a missing material file).
    warnings: Vec<String>,
}

#[cfg(test)]
pub fn voxelize(path: &Path, opts: &MeshOptions) -> Result<Model> {
    Ok(voxelize_noted(path, opts)?.0)
}

/// Voxelize the mesh at `path`, with warnings about what couldn't be read.
pub fn voxelize_noted(path: &Path, opts: &MeshOptions) -> Result<(Model, Vec<String>)> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let mut mesh = match ext.as_str() {
        "obj" => load_obj(path),
        "glb" | "gltf" => load_gltf(path),
        _ => bail!("{} isn't a mesh voxely reads ({})", path.display(), EXTENSIONS.join(", ")),
    }
    .with_context(|| format!("importing {}", path.display()))?;
    if !opts.z_up {
        // y up to z up, front (+z) to -y: (x, y, z) -> (x, -z, y).
        for t in &mut mesh.tris {
            for p in &mut t.p {
                *p = Vec3::new(p.x, -p.z, p.y);
            }
        }
    }
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("mesh").to_string();
    let model = voxelize_mesh(&mesh, opts, name).with_context(|| format!("voxelizing {}", path.display()))?;
    Ok((model, std::mem::take(&mut mesh.warnings)))
}

// ---------------------------------------------------------------- voxelizing

/// The best hit so far for a surface voxel.
#[derive(Clone, Copy)]
struct Hit {
    d2: f32,
    tri: u32,
    /// Barycentric weights of the triangle's 2nd and 3rd corners.
    b: [f32; 2],
}

fn voxelize_mesh(mesh: &Mesh, opts: &MeshOptions, name: String) -> Result<Model> {
    let size = opts.size.clamp(1, 1024);
    let tris: Vec<&Tri> = mesh.tris.iter().filter(|t| t.p.iter().all(|p| p.is_finite())).collect();
    if tris.is_empty() {
        bail!("the mesh has no triangles");
    }
    let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for t in &tris {
        for p in t.p {
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    let extent = hi - lo;
    let longest = extent.max_element();
    // A hair of margin keeps faces lying exactly on the box's sides inside
    // the end voxels instead of on their boundaries.
    let margin = 1e-3;
    let s = size as f32;
    let exact = if longest > 0.0 { s / longest } else { 1.0 };
    let dims = (extent * exact - Vec3::splat(1e-3)).ceil().as_ivec3().clamp(IVec3::ONE, IVec3::splat(size as i32));
    let scale = if longest > 0.0 { (s - 2.0 * margin) / longest } else { 1.0 };
    // Centered in each axis's voxels.
    let shift = (dims.as_vec3() - extent * scale) * 0.5;
    let (dx, dy) = (dims.x as usize, dims.y as usize);
    let key = |x: usize, y: usize, z: usize| (x + dx * (y + dy * z)) as u32;

    let mut hits: HashMap<u32, Hit> = HashMap::new();
    const EPS: f32 = 1e-4;
    // Voxels must overlap a triangle by a hair, and triangles are tested a
    // little behind their front: a face on a voxel boundary (a voxel mesh's
    // faces all are) then fills the voxel behind it, not the one in front
    // too, and its edges don't reach the voxels beside it.
    const INSET: f32 = 0.02;
    let h = 0.5 - INSET;
    let top = dims - IVec3::ONE;
    for (ti, t) in tris.iter().enumerate() {
        let v = t.p.map(|p| (p - lo) * scale + shift);
        let back = -(v[1] - v[0]).cross(v[2] - v[0]).normalize_or_zero() * (2.0 * INSET);
        let v = v.map(|p| p + back);
        let tmin = v[0].min(v[1]).min(v[2]);
        let tmax = v[0].max(v[1]).max(v[2]);
        let a = (tmin - Vec3::splat(EPS)).floor().as_ivec3().clamp(IVec3::ZERO, top);
        let b = (tmax + Vec3::splat(EPS)).floor().as_ivec3().clamp(IVec3::ZERO, top);
        let e = [v[1] - v[0], v[2] - v[1], v[0] - v[2]];
        let n = e[0].cross(e[1]);
        let axes: [Vec3; 9] = std::array::from_fn(|k| [Vec3::X, Vec3::Y, Vec3::Z][k / 3].cross(e[k % 3]));
        for z in a.z..=b.z {
            for y in a.y..=b.y {
                for x in a.x..=b.x {
                    let c = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                    if !tri_box(&v, c, h, n, &axes) {
                        continue;
                    }
                    let (q, bc) = closest_on_tri(c, &v);
                    let d2 = q.distance_squared(c);
                    let k = key(x as usize, y as usize, z as usize);
                    let hit = Hit { d2, tri: ti as u32, b: bc };
                    hits.entry(k).and_modify(|o| if d2 < o.d2 { *o = hit }).or_insert(hit);
                }
            }
        }
    }

    let colors: HashMap<u32, [u8; 4]> = hits.iter().map(|(&k, hit)| (k, color_at(mesh, tris[hit.tri as usize], hit.b))).collect();
    drop(hits);
    let unkey = |k: u32| {
        let k = k as usize;
        IVec3::new((k % dx) as i32, (k / dx % dy) as i32, (k / (dx * dy)) as i32)
    };

    let mut voxels: Vec<(IVec3, [u8; 4])>;
    if !opts.solid {
        let mut keys: Vec<u32> = colors.keys().copied().collect();
        keys.sort_unstable();
        voxels = keys.into_iter().map(|k| (unkey(k), colors[&k])).collect();
    } else {
        let outside = flood_outside(dims, &colors);
        let (pdx, pdy) = (dx + 2, dy + 2);
        let pkey = |x: usize, y: usize, z: usize| (x + 1) + pdx * ((y + 1) + pdy * (z + 1));
        voxels = Vec::with_capacity(colors.len());
        let mut row: Vec<(usize, [u8; 4])> = Vec::new();
        for z in 0..dims.z as usize {
            for y in 0..dy {
                row.clear();
                row.extend((0..dx).filter_map(|x| Some((x, *colors.get(&key(x, y, z))?))));
                if row.is_empty() {
                    continue;
                }
                let mut next = 0; // first surface voxel at or after x
                for x in row[0].0..=row[row.len() - 1].0 {
                    while next < row.len() && row[next].0 < x {
                        next += 1;
                    }
                    if row[next].0 == x {
                        voxels.push((IVec3::new(x as i32, y as i32, z as i32), row[next].1));
                    } else if !outside.get(pkey(x, y, z)) {
                        // Between row[next - 1] and row[next]: the nearer.
                        let (l, r) = (row[next - 1], row[next]);
                        let c = if x - l.0 <= r.0 - x { l.1 } else { r.1 };
                        voxels.push((IVec3::new(x as i32, y as i32, z as i32), c));
                    }
                }
            }
        }
    }
    Ok(Model { name, dims, offset: IVec3::ZERO, voxels })
}

/// Conservative triangle / box overlap (separating axes, Akenine-Möller):
/// box centered at `c` with half size `h`, triangle `v` with normal `n` and
/// edge-cross axes `axes`.
fn tri_box(v: &[Vec3; 3], c: Vec3, h: f32, n: Vec3, axes: &[Vec3; 9]) -> bool {
    let p = [v[0] - c, v[1] - c, v[2] - c];
    for k in 0..3 {
        let (a, b, cc) = (p[0][k], p[1][k], p[2][k]);
        if a.min(b).min(cc) > h || a.max(b).max(cc) < -h {
            return false;
        }
    }
    if n.dot(p[0]).abs() > h * n.abs().element_sum() {
        return false;
    }
    for a in axes {
        let (p0, p1, p2) = (a.dot(p[0]), a.dot(p[1]), a.dot(p[2]));
        let r = h * a.abs().element_sum();
        if p0.min(p1).min(p2) > r || p0.max(p1).max(p2) < -r {
            return false;
        }
    }
    true
}

/// The point of triangle `t` nearest `p` and its barycentric weights for the
/// 2nd and 3rd corners (Ericson, Real-Time Collision Detection 5.1.5).
fn closest_on_tri(p: Vec3, t: &[Vec3; 3]) -> (Vec3, [f32; 2]) {
    let [a, b, c] = *t;
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [0.0, 0.0]);
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, [v, 0.0]);
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, [0.0, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, [1.0 - w, w]);
    }
    let sum = va + vb + vc;
    if !(sum.abs() > 1e-20) {
        return (a, [0.0, 0.0]);
    }
    let (v, w) = (vb / sum, vc / sum);
    (a + ab * v + ac * w, [v, w])
}

fn color_at(mesh: &Mesh, t: &Tri, b: [f32; 2]) -> [u8; 4] {
    let w = [1.0 - b[0] - b[1], b[0], b[1]];
    let m = mesh.mats.get(t.mat as usize).unwrap_or(&WHITE);
    let mut c = m.color;
    if let (Some(img), Some(uv)) = (m.tex.and_then(|i| mesh.textures.get(i)), t.uv) {
        let uv = uv[0] * w[0] + uv[1] * w[1] + uv[2] * w[2];
        let px = sample(img, uv);
        for k in 0..3 {
            c[k] *= px[k] as f32 / 255.0;
        }
    }
    if let Some(col) = t.col {
        for k in 0..3 {
            c[k] *= col[0][k] * w[0] + col[1][k] * w[1] + col[2][k] * w[2];
        }
    }
    let u = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
    [u(c[0]), u(c[1]), u(c[2]), 255]
}

/// Nearest-texel lookup with wrapping; `uv` has v down.
fn sample(img: &Image, uv: Vec2) -> [u8; 4] {
    if img.w == 0 || img.h == 0 || img.px.len() < img.w * img.h {
        return [255; 4];
    }
    let wrap = |t: f32, n: usize| {
        let f = t - t.floor();
        if f.is_finite() { ((f * n as f32) as usize).min(n - 1) } else { 0 }
    };
    img.px[wrap(uv.y, img.h) * img.w + wrap(uv.x, img.w)]
}

/// A bitset.
struct Bits(Vec<u64>);

impl Bits {
    fn new(n: usize) -> Bits {
        Bits(vec![0; n.div_ceil(64)])
    }
    fn get(&self, i: usize) -> bool {
        self.0[i >> 6] >> (i & 63) & 1 == 1
    }
    fn set(&mut self, i: usize) {
        self.0[i >> 6] |= 1 << (i & 63);
    }
}

/// The cells reachable (6-connected) from outside without crossing the
/// surface, over the box padded by one cell on every side.
fn flood_outside(dims: IVec3, surface: &HashMap<u32, [u8; 4]>) -> Bits {
    let (dx, dy) = (dims.x as usize, dims.y as usize);
    let (px, py, pz) = (dx + 2, dy + 2, dims.z as usize + 2);
    let n = px * py * pz;
    let mut blocked = Bits::new(n);
    for &k in surface.keys() {
        let k = k as usize;
        let (x, y, z) = (k % dx, k / dx % dy, k / (dx * dy));
        blocked.set((x + 1) + px * ((y + 1) + py * (z + 1)));
    }
    let mut out = Bits::new(n);
    let mut stack = vec![0usize];
    out.set(0);
    while let Some(i) = stack.pop() {
        let (x, y, z) = (i % px, i / px % py, i / (px * py));
        let mut go = |j: usize, out: &mut Bits| {
            if !out.get(j) && !blocked.get(j) {
                out.set(j);
                stack.push(j);
            }
        };
        if x > 0 {
            go(i - 1, &mut out);
        }
        if x + 1 < px {
            go(i + 1, &mut out);
        }
        if y > 0 {
            go(i - px, &mut out);
        }
        if y + 1 < py {
            go(i + px, &mut out);
        }
        if z > 0 {
            go(i - px * py, &mut out);
        }
        if z + 1 < pz {
            go(i + px * py, &mut out);
        }
    }
    out
}

// ---------------------------------------------------------------- textures

/// Loads each texture file once; failures are logged and skipped.
#[derive(Default)]
struct TextureCache {
    by_path: HashMap<PathBuf, Option<usize>>,
}

impl TextureCache {
    fn file(&mut self, mesh: &mut Mesh, path: PathBuf) -> Option<usize> {
        if let Some(&i) = self.by_path.get(&path) {
            return i;
        }
        let i = match load_image(&path) {
            Ok(img) => {
                mesh.textures.push(img);
                Some(mesh.textures.len() - 1)
            }
            Err(e) => {
                log::warn!("texture {}: {e:#}", path.display());
                None
            }
        };
        self.by_path.insert(path, i);
        i
    }
}

/// Decode PNG / JPEG bytes (through a temporary file, as `load_image`
/// reads files).
fn image_from_bytes(bytes: &[u8]) -> Result<Image> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let p = std::env::temp_dir().join(format!("voxely-mesh-tex-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    std::fs::write(&p, bytes).with_context(|| format!("writing {}", p.display()))?;
    let img = load_image(&p);
    std::fs::remove_file(&p).ok();
    img
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Some(v) = std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ---------------------------------------------------------------- OBJ

#[derive(Clone, Debug, Default)]
struct ObjMaterial {
    kd: Option<[f32; 3]>,
    map: Option<PathBuf>,
}

fn nums(rest: &str, line: usize) -> Result<Vec<f32>> {
    rest.split_whitespace().map(|t| t.parse::<f32>().with_context(|| format!("line {line}: bad number `{t}`"))).collect()
}

/// A file name from an .obj / .mtl line (after the keyword): options such
/// as `-s 1 1 1` are skipped by taking the last word.
fn file_arg(dir: &Path, rest: &str) -> PathBuf {
    let rest = rest.trim();
    let name = if rest.starts_with('-') { rest.split_whitespace().last().unwrap_or("") } else { rest };
    dir.join(name.replace('\\', "/"))
}

/// The files an `mtllib` line names: quoted names (which may hold
/// spaces), else the whole line if it is one file, else each word.
fn mtl_files(dir: &Path, rest: &str) -> Vec<PathBuf> {
    let rest = rest.trim();
    if rest.contains('"') {
        return rest.split('"').skip(1).step_by(2).filter(|n| !n.trim().is_empty()).map(|n| dir.join(n.replace('\\', "/"))).collect();
    }
    let whole = dir.join(rest.replace('\\', "/"));
    if whole.is_file() {
        return vec![whole];
    }
    rest.split_whitespace().map(|n| dir.join(n.replace('\\', "/"))).collect()
}

fn load_mtl(path: &Path, into: &mut HashMap<String, ObjMaterial>) -> Result<()> {
    let text = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let text = String::from_utf8_lossy(&text);
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut cur: Option<String> = None;
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim();
        let (kw, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        match kw {
            "newmtl" => {
                let name = rest.trim().to_string();
                into.entry(name.clone()).or_default();
                cur = Some(name);
            }
            "Kd" => {
                let Some(m) = cur.as_ref().and_then(|c| into.get_mut(c)) else { continue };
                let v = nums(rest, ln + 1).with_context(|| format!("in {}", path.display()))?;
                if v.len() >= 3 {
                    m.kd = Some([v[0], v[1], v[2]]);
                } else if v.len() == 1 {
                    m.kd = Some([v[0]; 3]);
                }
            }
            "map_Kd" => {
                let Some(m) = cur.as_ref().and_then(|c| into.get_mut(c)) else { continue };
                if !rest.trim().is_empty() {
                    m.map = Some(file_arg(dir, rest));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn load_obj(path: &Path) -> Result<Mesh> {
    let text = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let text = String::from_utf8_lossy(&text);
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut pos: Vec<Vec3> = Vec::new();
    let mut vcol: Vec<Option<[f32; 3]>> = Vec::new();
    let mut uvs: Vec<Vec2> = Vec::new();
    let mut library: HashMap<String, ObjMaterial> = HashMap::new();
    // Material names in order of first use; index 0 is "no material".
    let mut used: Vec<String> = vec![String::new()];
    let mut cur = 0u32;
    let mut mesh = Mesh::default();
    let mut corners: Vec<(usize, Option<usize>)> = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let ln = ln + 1;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (kw, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        match kw {
            "v" => {
                let v = nums(rest, ln)?;
                if v.len() < 3 {
                    bail!("line {ln}: a vertex needs x y z");
                }
                pos.push(Vec3::new(v[0], v[1], v[2]));
                vcol.push((v.len() >= 6).then(|| {
                    let c = [v[3], v[4], v[5]];
                    if c.iter().any(|&x| x > 1.0) { c.map(|x| x / 255.0) } else { c }
                }));
            }
            "vt" => {
                let v = nums(rest, ln)?;
                let (u, w) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
                uvs.push(Vec2::new(u, 1.0 - w));
            }
            "f" => {
                corners.clear();
                for tok in rest.split_whitespace() {
                    let mut parts = tok.split('/');
                    let resolve = |s: Option<&str>, len: usize, what: &str| -> Result<Option<usize>> {
                        let Some(s) = s.filter(|s| !s.is_empty()) else { return Ok(None) };
                        let i: i64 = s.parse().with_context(|| format!("line {ln}: bad {what} index `{s}`"))?;
                        let r = if i > 0 { i - 1 } else { len as i64 + i };
                        if i == 0 || r < 0 || r >= len as i64 {
                            bail!("line {ln}: {what} index {i} is out of range (there are {len})");
                        }
                        Ok(Some(r as usize))
                    };
                    let vi = resolve(parts.next(), pos.len(), "vertex")?.with_context(|| format!("line {ln}: face corner `{tok}` has no vertex"))?;
                    let ti = resolve(parts.next(), uvs.len(), "texture")?;
                    corners.push((vi, ti));
                }
                for k in 1..corners.len().saturating_sub(1) {
                    let c = [corners[0], corners[k], corners[k + 1]];
                    let uv = c.iter().map(|x| x.1.map(|t| uvs[t])).collect::<Option<Vec<_>>>().map(|v| [v[0], v[1], v[2]]);
                    let col = c.iter().map(|x| vcol[x.0]).collect::<Option<Vec<_>>>().map(|v| [v[0], v[1], v[2]]);
                    mesh.tris.push(Tri { p: c.map(|x| pos[x.0]), uv, col, mat: cur });
                }
            }
            "mtllib" => {
                for p in mtl_files(dir, rest) {
                    if let Err(e) = load_mtl(&p, &mut library) {
                        log::warn!("{e:#}");
                        let name = p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned());
                        mesh.warnings.push(format!("material file {name:?} (line {ln}) couldn't be read, so its colors are missing"));
                    }
                }
            }
            "usemtl" => {
                let name = rest.trim();
                cur = match used.iter().position(|u| u == name) {
                    Some(i) => i as u32,
                    None => {
                        used.push(name.to_string());
                        (used.len() - 1) as u32
                    }
                };
            }
            _ => {}
        }
    }
    let mut cache = TextureCache::default();
    for name in &used {
        let m = library.get(name).cloned().unwrap_or_default();
        let tex = m.map.and_then(|p| cache.file(&mut mesh, p));
        // A texture replaces Kd (exporters often write Kd 0.8 beside it).
        let color = if tex.is_some() { [1.0; 3] } else { m.kd.unwrap_or([1.0; 3]) };
        mesh.mats.push(Material { color, tex });
    }
    Ok(mesh)
}

// ---------------------------------------------------------------- glTF

fn to_srgb(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.0031308 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect()).unwrap_or_default()
}

fn node_matrix(n: &Value) -> Mat4 {
    let m = floats(&n["matrix"]);
    if m.len() == 16 {
        return Mat4::from_cols_slice(&m);
    }
    let get = |k: &str, d: Vec<f32>, len: usize| Some(floats(&n[k])).filter(|v| v.len() == len).unwrap_or(d);
    let t = get("translation", vec![0.0; 3], 3);
    let r = get("rotation", vec![0.0, 0.0, 0.0, 1.0], 4);
    let s = get("scale", vec![1.0; 3], 3);
    let q = Quat::from_slice(&r);
    let q = if q.length_squared() > 1e-12 { q.normalize() } else { Quat::IDENTITY };
    Mat4::from_scale_rotation_translation(Vec3::from_slice(&s), q, Vec3::from_slice(&t))
}

fn load_gltf(path: &Path) -> Result<Mesh> {
    let g = Gltf::load(path)?;
    let doc = g.json();
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut mesh = Mesh::default();

    // Images, decoded on first use.
    let mut images: HashMap<usize, Option<usize>> = HashMap::new();
    let mut cache = TextureCache::default();
    let mut image = |mesh: &mut Mesh, i: usize| -> Option<usize> {
        if let Some(&t) = images.get(&i) {
            return t;
        }
        let im = &doc["images"][i];
        let got: Result<Option<usize>> = (|| {
            if let Some(uri) = im["uri"].as_str() {
                if uri.starts_with("data:") {
                    use base64::Engine;
                    let b64 = uri.split_once(',').map_or("", |x| x.1);
                    let bytes = base64::engine::general_purpose::STANDARD.decode(b64).context("bad image data URI")?;
                    let img = image_from_bytes(&bytes)?;
                    mesh.textures.push(img);
                    return Ok(Some(mesh.textures.len() - 1));
                }
                return Ok(cache.file(mesh, dir.join(percent_decode(uri))));
            }
            let view = im["bufferView"].as_u64().context("image has no uri or bufferView")?;
            let img = image_from_bytes(g.view_bytes(view as usize)?)?;
            mesh.textures.push(img);
            Ok(Some(mesh.textures.len() - 1))
        })();
        let t = got.unwrap_or_else(|e| {
            log::warn!("image {i}: {e:#}");
            None
        });
        images.insert(i, t);
        t
    };

    // Materials, plus a white one at the end for primitives without one.
    let mut tex_set: Vec<u64> = Vec::new();
    for m in doc["materials"].as_array().into_iter().flatten() {
        let pbr = &m["pbrMetallicRoughness"];
        let f = floats(&pbr["baseColorFactor"]);
        let color = if f.len() >= 3 { [to_srgb(f[0]), to_srgb(f[1]), to_srgb(f[2])] } else { [1.0; 3] };
        let bt = &pbr["baseColorTexture"];
        let tex = bt["index"].as_u64().and_then(|t| doc["textures"][t as usize]["source"].as_u64()).and_then(|src| image(&mut mesh, src as usize));
        tex_set.push(bt["texCoord"].as_u64().unwrap_or(0));
        mesh.mats.push(Material { color, tex });
    }
    let default_mat = mesh.mats.len() as u32;
    mesh.mats.push(WHITE);

    // The default scene's roots, or every parentless node.
    let nodes = doc["nodes"].as_array().map_or(&[][..], |v| v.as_slice());
    let scene = doc["scene"].as_u64().unwrap_or(0) as usize;
    let roots: Vec<usize> = match doc["scenes"][scene]["nodes"].as_array() {
        Some(r) => r.iter().filter_map(Value::as_u64).map(|x| x as usize).collect(),
        None => {
            let mut child = vec![false; nodes.len()];
            for n in nodes {
                for c in n["children"].as_array().into_iter().flatten().filter_map(Value::as_u64) {
                    if let Some(x) = child.get_mut(c as usize) {
                        *x = true;
                    }
                }
            }
            (0..nodes.len()).filter(|&i| !child[i]).collect()
        }
    };
    let mut stack: Vec<(usize, Mat4, usize)> = roots.into_iter().rev().map(|r| (r, Mat4::IDENTITY, 0)).collect();
    let mut visits = 0usize;
    while let Some((ni, parent, depth)) = stack.pop() {
        let Some(n) = nodes.get(ni) else { continue };
        visits += 1;
        if depth > 256 || visits > 1_000_000 {
            bail!("the node hierarchy is too deep or loops");
        }
        let m = parent * node_matrix(n);
        if let Some(mi) = n["mesh"].as_u64() {
            let prims = doc["meshes"][mi as usize]["primitives"].as_array().map_or(&[][..], |v| v.as_slice());
            for (pi, p) in prims.iter().enumerate() {
                add_primitive(&g, p, m, &tex_set, default_mat, &mut mesh).with_context(|| format!("mesh {mi} primitive {pi}"))?;
            }
        }
        for c in n["children"].as_array().into_iter().flatten().rev().filter_map(Value::as_u64) {
            stack.push((c as usize, m, depth + 1));
        }
    }
    Ok(mesh)
}

fn add_primitive(g: &Gltf, p: &Value, m: Mat4, tex_set: &[u64], default_mat: u32, mesh: &mut Mesh) -> Result<()> {
    let mode = p["mode"].as_u64().unwrap_or(4);
    if !matches!(mode, 4..=6) {
        return Ok(()); // points and lines have no surface
    }
    let attrs = &p["attributes"];
    let Some(pa) = attrs["POSITION"].as_u64() else { return Ok(()) };
    let pos: Vec<Vec3> = g.read_floats(pa as usize).context("POSITION")?.chunks_exact(3).map(|c| m.transform_point3(Vec3::from_slice(c))).collect();
    let n = pos.len();
    let mat = p["material"].as_u64().map_or(default_mat, |x| (x as u32).min(default_mat));
    let set = tex_set.get(mat as usize).copied().unwrap_or(0);
    let uv: Option<Vec<Vec2>> = match attrs[format!("TEXCOORD_{set}")].as_u64() {
        Some(a) => Some(g.read_floats(a as usize).context("TEXCOORD")?.chunks_exact(2).map(Vec2::from_slice).collect()),
        None => None,
    };
    let col: Option<Vec<[f32; 3]>> = match attrs["COLOR_0"].as_u64() {
        Some(a) => {
            let comps = if g.json()["accessors"][a as usize]["type"].as_str() == Some("VEC4") { 4 } else { 3 };
            Some(g.read_floats(a as usize).context("COLOR_0")?.chunks_exact(comps).map(|c| [to_srgb(c[0]), to_srgb(c[1]), to_srgb(c[2])]).collect())
        }
        None => None,
    };
    let idx: Vec<u32> = match p["indices"].as_u64() {
        Some(a) => g.read_uints(a as usize).context("indices")?,
        None => (0..n as u32).collect(),
    };
    let tri = |i: usize| -> Option<[usize; 3]> {
        match mode {
            4 => Some([idx[3 * i], idx[3 * i + 1], idx[3 * i + 2]].map(|x| x as usize)),
            5 if i % 2 == 0 => Some([idx[i], idx[i + 1], idx[i + 2]].map(|x| x as usize)),
            5 => Some([idx[i + 1], idx[i], idx[i + 2]].map(|x| x as usize)),
            _ => Some([idx[0], idx[i + 1], idx[i + 2]].map(|x| x as usize)),
        }
    };
    let count = if mode == 4 { idx.len() / 3 } else { idx.len().saturating_sub(2) };
    mesh.tris.reserve(count);
    for i in 0..count {
        let Some(c) = tri(i) else { continue };
        if c.iter().any(|&x| x >= n) {
            bail!("index {} is out of range (there are {n} vertices)", c.iter().max().unwrap());
        }
        mesh.tris.push(Tri {
            p: c.map(|x| pos[x]),
            uv: uv.as_ref().and_then(|u| Some([*u.get(c[0])?, *u.get(c[1])?, *u.get(c[2])?])),
            col: col.as_ref().and_then(|u| Some([*u.get(c[0])?, *u.get(c[1])?, *u.get(c[2])?])),
            mat,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A temp file removed on drop.
    struct Temp(PathBuf);

    impl Temp {
        fn new(name: &str, contents: &[u8]) -> Temp {
            let p = std::env::temp_dir().join(format!("voxely-mesh-test-{}-{name}", std::process::id()));
            std::fs::write(&p, contents).unwrap();
            Temp(p)
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            std::fs::remove_file(&self.0).ok();
        }
    }

    const CUBE_V: &str = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n";
    const CUBE_F: &str = "f 1 4 3 2\nf 5 6 7 8\nf 1 2 6 5\nf 2 3 7 6\nf 3 4 8 7\nf 4 1 5 8\n";

    fn opts(size: u32, solid: bool) -> MeshOptions {
        MeshOptions { size, solid, z_up: false }
    }

    #[test]
    fn voxel_meshes_come_back_voxel_for_voxel() {
        // An L of voxels, exported as voxely meshes: re-imported at its own
        // size, it has the same voxels (faces on voxel boundaries don't
        // fill the voxels in front of them).
        let mut s = crate::scene::Scene::empty(IVec3::new(3, 2, 3));
        for p in [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]] {
            s.volume.set(IVec3::from(p), 1);
        }
        let dir = std::env::temp_dir().join(format!("voxely-mesh-rt-{}", std::process::id()));
        for f in [crate::export::Format::Obj, crate::export::Format::Gltf] {
            let files = crate::export::export(f, &s, &dir.join("l.x")).unwrap();
            for solid in [true, false] {
                let m = voxelize(&files[0], &opts(3, solid)).unwrap();
                assert_eq!((m.dims, m.voxels.len()), (IVec3::new(3, 2, 3), 6), "{f:?} solid {solid}");
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mtllib_names_quoted_several_and_missing() {
        let dir = Path::new("/d");
        assert_eq!(mtl_files(dir, r#""my colors.mtl" other.mtl"#), vec![dir.join("my colors.mtl")]);
        assert_eq!(mtl_files(dir, "a.mtl b.mtl"), vec![dir.join("a.mtl"), dir.join("b.mtl")]);
        let f = Temp::new("nomtl.obj", format!("mtllib missing.mtl\n{CUBE_V}{CUBE_F}").as_bytes());
        let (_, warnings) = voxelize_noted(&f.0, &opts(2, true)).unwrap();
        assert!(warnings.iter().any(|w| w.contains("missing.mtl")), "{warnings:?}");
    }

    #[test]
    fn unit_cube_solid_and_hollow() {
        let f = Temp::new("cube.obj", format!("{CUBE_V}{CUBE_F}").as_bytes());
        let m = voxelize(&f.0, &opts(8, true)).unwrap();
        assert_eq!(m.dims, IVec3::splat(8));
        assert_eq!(m.voxels.len(), 512);
        assert!(m.voxels.iter().all(|v| v.1 == [255, 255, 255, 255]));
        let m = voxelize(&f.0, &opts(8, false)).unwrap();
        assert_eq!(m.dims, IVec3::splat(8));
        assert_eq!(m.voxels.len(), 8 * 8 * 8 - 6 * 6 * 6);
        assert!(m.voxels.iter().all(|(p, _)| p.min_element() == 0 || p.max_element() == 7));
    }

    #[test]
    fn negative_indices_and_slashes() {
        let faces = "f -8/-1 -5/-1 -6/-1 -7/-1\nf -4//1 -3//1 -2//1 -1//1\nf 1/1/1 2/1/1 6/1/1 5/1/1\nf 2 3 7 6\nf 3 4 8 7\nf 4 1 5 8\n";
        let f = Temp::new("neg.obj", format!("{CUBE_V}vt 0 0\nvn 0 0 1\n{faces}").as_bytes());
        let m = voxelize(&f.0, &opts(8, true)).unwrap();
        assert_eq!(m.voxels.len(), 512);
    }

    #[test]
    fn material_color() {
        let mtl = Temp::new("red.mtl", b"newmtl red\nKd 1 0 0\nnewmtl blue\nKd 0 0 1\n");
        let name = mtl.0.file_name().unwrap().to_str().unwrap();
        let f = Temp::new("red.obj", format!("mtllib {name}\n{CUBE_V}usemtl red\n{CUBE_F}").as_bytes());
        let m = voxelize(&f.0, &opts(8, true)).unwrap();
        assert_eq!(m.voxels.len(), 512);
        assert!(m.voxels.iter().all(|v| v.1 == [255, 0, 0, 255]), "{:?}", m.voxels[0]);
    }

    #[test]
    fn vertex_colors() {
        let v: String = CUBE_V.lines().map(|l| format!("{l} 0 1 0\n")).collect();
        let f = Temp::new("vc.obj", format!("{v}{CUBE_F}").as_bytes());
        let m = voxelize(&f.0, &opts(4, false)).unwrap();
        assert!(m.voxels.iter().all(|v| v.1 == [0, 255, 0, 255]));
    }

    #[test]
    fn texture_sampled_with_v_up() {
        // A 1x2 PNG: top row red, bottom row blue, on a quad facing -y
        // (OBJ z = toward the viewer; v = 1 at the top).
        let mut png_bytes = Vec::new();
        {
            let mut e = png::Encoder::new(&mut png_bytes, 1, 2);
            e.set_color(png::ColorType::Rgba);
            let mut w = e.write_header().unwrap();
            w.write_image_data(&[255, 0, 0, 255, 0, 0, 255, 255]).unwrap();
        }
        let tex = Temp::new("tex.png", &png_bytes);
        let tname = tex.0.file_name().unwrap().to_str().unwrap();
        let mtl = Temp::new("tex.mtl", format!("newmtl t\nKd 0.8 0.8 0.8\nmap_Kd {tname}\n").as_bytes());
        let mname = mtl.0.file_name().unwrap().to_str().unwrap();
        let obj = format!("mtllib {mname}\nusemtl t\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3 4/4\n");
        let f = Temp::new("tex.obj", obj.as_bytes());
        let m = voxelize(&f.0, &opts(4, false)).unwrap();
        // OBJ y (up) is voxely z.
        assert_eq!(m.dims, IVec3::new(4, 1, 4));
        for (p, c) in &m.voxels {
            let want = if p.z >= 2 { [255, 0, 0, 255] } else { [0, 0, 255, 255] };
            assert_eq!(*c, want, "{p}");
        }
    }

    #[test]
    fn tilted_mesh_is_watertight() {
        // A cube rotated off every axis, hollow: the flood from outside
        // must not get in, so solid has more voxels and includes the center.
        let r = Mat4::from_quat(Quat::from_euler(glam::EulerRot::XYZ, 0.37, 0.71, 0.19));
        let v: String = CUBE_V
            .lines()
            .map(|l| {
                let c: Vec<f32> = l[2..].split_whitespace().map(|x| x.parse().unwrap()).collect();
                let p = r.transform_point3(Vec3::new(c[0], c[1], c[2]));
                format!("v {} {} {}\n", p.x, p.y, p.z)
            })
            .collect();
        let f = Temp::new("tilt.obj", format!("{v}{CUBE_F}").as_bytes());
        let hollow = voxelize(&f.0, &opts(40, false)).unwrap();
        let solid = voxelize(&f.0, &opts(40, true)).unwrap();
        assert!(solid.voxels.len() > hollow.voxels.len() * 2, "{} vs {}", solid.voxels.len(), hollow.voxels.len());
        let c = solid.dims / 2;
        assert!(solid.voxels.iter().any(|v| v.0 == c));
    }

    #[test]
    fn bad_input_is_an_error() {
        let f = Temp::new("bad.obj", b"v 0 0 0\nv 1 0 0\nf 1 2 3\n");
        assert!(voxelize(&f.0, &opts(8, true)).is_err());
        let f = Temp::new("bad2.obj", b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 -9\n");
        assert!(voxelize(&f.0, &opts(8, true)).is_err());
        let f = Temp::new("empty.obj", b"# nothing\n");
        assert!(voxelize(&f.0, &opts(8, true)).is_err());
        assert!(voxelize(Path::new("/nonexistent/voxely/x.obj"), &opts(8, true)).is_err());
        let f = Temp::new("bad.gltf", br#"{"asset":{"version":"2.0"},"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],"nodes":[{"mesh":0}],"accessors":[{"bufferView":0,"componentType":5126,"count":1000,"type":"VEC3"}],"bufferViews":[{"buffer":0,"byteLength":12}],"buffers":[{"byteLength":12,"uri":"data:application/octet-stream;base64,AAAAAAAAAAAAAAAA"}]}"#);
        assert!(voxelize(&f.0, &opts(8, true)).is_err());
    }

    /// A glTF with a box 1 wide, 3 tall (glTF y), 1 deep, under a node
    /// that scales by 2 and moves it; red base color.
    fn tall_box_gltf() -> String {
        use base64::Engine;
        let mut data: Vec<u8> = Vec::new();
        let corners: Vec<[f32; 3]> = (0..8).map(|i| [(i & 1) as f32, ((i >> 1) & 1) as f32 * 3.0, ((i >> 2) & 1) as f32]).collect();
        for c in &corners {
            for x in c {
                data.extend(x.to_le_bytes());
            }
        }
        // Six quads of corner bits (x=1, y=2, z=4).
        let quads = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
        let idx: Vec<u16> = quads.iter().flat_map(|q| [q[0], q[1], q[2], q[0], q[2], q[3]]).collect();
        for i in &idx {
            data.extend(i.to_le_bytes());
        }
        let b64 = base64::engine::general_purpose::STANDARD.encode(&data);
        serde_json::json!({
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "children": [1], "scale": [2.0, 2.0, 2.0] }, { "mesh": 0, "translation": [5.0, 0.0, 0.0] }],
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 1, "material": 0 }] }],
            "materials": [{ "pbrMetallicRoughness": { "baseColorFactor": [1.0, 0.0, 0.0, 1.0] } }],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR" }
            ],
            "bufferViews": [{ "buffer": 0, "byteLength": 96 }, { "buffer": 0, "byteOffset": 96, "byteLength": 72 }],
            "buffers": [{ "byteLength": data.len(), "uri": format!("data:application/octet-stream;base64,{b64}") }]
        })
        .to_string()
    }

    #[test]
    fn gltf_y_up_box_stands_along_z() {
        let f = Temp::new("tall.gltf", tall_box_gltf().as_bytes());
        let m = voxelize(&f.0, &opts(12, true)).unwrap();
        assert_eq!(m.dims, IVec3::new(4, 4, 12));
        assert_eq!(m.voxels.len(), 4 * 4 * 12);
        assert!(m.voxels.iter().all(|v| v.1 == [255, 0, 0, 255]));
        // z_up keeps the file's axes: tall along y.
        let m = voxelize(&f.0, &MeshOptions { size: 12, solid: false, z_up: true }).unwrap();
        assert_eq!(m.dims, IVec3::new(4, 12, 4));
    }

    /// `cargo test --release import::mesh::tests::speed -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn speed() {
        // A ~100k-triangle sphere at size 256.
        let (nu, nv) = (250usize, 200usize);
        let mut tris = Vec::new();
        let pt = |i: usize, j: usize| {
            let (th, ph) = (i as f32 / nu as f32 * std::f32::consts::TAU, j as f32 / nv as f32 * std::f32::consts::PI);
            Vec3::new(ph.sin() * th.cos(), ph.sin() * th.sin(), ph.cos())
        };
        for j in 0..nv {
            for i in 0..nu {
                let (a, b, c, d) = (pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1));
                tris.push(Tri { p: [a, b, c], uv: None, col: None, mat: 0 });
                tris.push(Tri { p: [a, c, d], uv: None, col: None, mat: 0 });
            }
        }
        let mesh = Mesh { tris, mats: vec![WHITE], ..Default::default() };
        let t = std::time::Instant::now();
        let m = voxelize_mesh(&mesh, &opts(256, true), "s".into()).unwrap();
        println!("{} tris -> {} voxels in {:?}", mesh.tris.len(), m.voxels.len(), t.elapsed());
    }
}
