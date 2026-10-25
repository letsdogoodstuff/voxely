//! MagicaVoxel's twelve "Export" panel formats, with the fixed defaults of
//! its `config.txt` (`io_iso`, `io_2d`, `io_obj`, ...).
//!
//! Byte layouts of xraw, qb and slab follow what MagicaVoxel 0.99.6 itself
//! writes (checked against its `MV_Plugin_RAW/QB/SLAB::ExportModel`):
//!
//! - **xraw**: indices are the color index as is, 0 = empty; the 256-entry
//!   RGBA palette is indexed by color index with entry 0 = transparent black.
//! - **qb**: Y-up, voxel `(x, y, z)` goes to qb `(x, z, sy-1-y)`, which keeps
//!   the handedness; colors are bytes R, G, B, A (alpha 0 = empty), x fastest.
//! - **slab**: SLAB6 flips y and z (`(x, sy-1-y, sz-1-z)`, a 180° turn about
//!   x), stores `c - 1` (255 = empty) with z fastest, then palette entries
//!   1..=255 followed by entry 0, as 6-bit RGB.
//!
//! Meshes are centered on the model's bounding box and scaled by 0.1. obj is
//! Y-up (`(x, z, -y)`), the PLY formats are Z-up. Faces wind
//! counter-clockwise seen from outside. PLY files are ASCII like
//! MagicaVoxel's (`%g %g %g %d %d %d` vertices, quads for ply/cube,
//! triangles for mc).

// The UI and MCP tool call into this module from elsewhere.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use glam::{IVec3, Vec3};

use crate::scene::{Scene, Volume};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Obj,
    Ply,
    Mc,
    Xraw,
    Slab,
    Qb,
    Vox,
    Iso,
    TwoD,
    Slice,
    Point,
    Cube,
}

impl Format {
    /// In MagicaVoxel's button order.
    pub const ALL: [Format; 12] = [
        Format::Obj,
        Format::Ply,
        Format::Mc,
        Format::Xraw,
        Format::Slab,
        Format::Qb,
        Format::Vox,
        Format::Iso,
        Format::TwoD,
        Format::Slice,
        Format::Point,
        Format::Cube,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Format::Obj => "obj",
            Format::Ply => "ply",
            Format::Mc => "mc",
            Format::Xraw => "xraw",
            Format::Slab => "slab",
            Format::Qb => "qb",
            Format::Vox => "vox",
            Format::Iso => "iso",
            Format::TwoD => "2d",
            Format::Slice => "slice",
            Format::Point => "point",
            Format::Cube => "cube",
        }
    }

    /// Extension of the main file written. Slab writes a SLAB6 `.vox`, so
    /// callers should pick a file name that does not clash with the model.
    pub fn extension(self) -> &'static str {
        match self {
            Format::Obj => "obj",
            Format::Ply | Format::Mc | Format::Point | Format::Cube => "ply",
            Format::Xraw => "xraw",
            Format::Slab | Format::Vox => "vox",
            Format::Qb => "qb",
            Format::Iso | Format::TwoD | Format::Slice => "png",
        }
    }

    /// MagicaVoxel's tooltip (its xraw hint reads "vox : ..."; fixed here).
    pub fn hint(self) -> &'static str {
        match self {
            Format::Obj => "obj : Export obj mesh",
            Format::Ply => "ply : Export ply mesh",
            Format::Mc => "mc : Export Marching Cubes",
            Format::Xraw => "xraw : Export xraw volume",
            Format::Slab => "slab : Export slab6 vox volume",
            Format::Qb => "qb : Export qb volume",
            Format::Vox => "vox : Export vox voxels",
            Format::Iso => "iso : Export isometric pixel sprite",
            Format::TwoD => "2d : Export 2D sprite",
            Format::Slice => "slice : Export slices",
            Format::Point => "point : Export point clouds",
            Format::Cube => "cube : Export cubes",
        }
    }

    /// By `name()`; also accepts MagicaVoxel's command name `pcl` for point.
    pub fn from_name(s: &str) -> Option<Format> {
        let s = s.trim().to_ascii_lowercase();
        if s == "pcl" {
            return Some(Format::Point);
        }
        Format::ALL.into_iter().find(|f| f.name() == s)
    }
}

/// Write `scene` in `format` to `path` (extension is replaced by the
/// format's). Returns every file written (e.g. obj+mtl+png).
pub fn export(format: Format, scene: &Scene, path: &Path) -> Result<Vec<PathBuf>> {
    let path = path.with_extension(format.extension());
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let v = &scene.volume;
    let pal = &scene.palette;
    let mut files = vec![path.clone()];
    match format {
        Format::Obj => files = write_obj(v, pal, &path)?,
        Format::Ply => write_file(&path, ply_mesh(v, pal).as_bytes())?,
        Format::Mc => write_file(&path, marching(v, pal).as_bytes())?,
        Format::Xraw => write_file(&path, &xraw(v, pal))?,
        Format::Slab => write_file(&path, &slab(v, pal))?,
        Format::Qb => {
            let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            write_file(&path, &qb(v, pal, &name))?
        }
        Format::Vox => scene.save(&path)?,
        Format::Iso => save_image(&path, &iso(v, pal))?,
        Format::TwoD => save_image(&path, &sprite_2d(v, pal))?,
        Format::Slice => save_image(&path, &slices(v, pal))?,
        Format::Point => write_file(&path, points(v, pal).as_bytes())?,
        Format::Cube => write_file(&path, cubes(v, pal).as_bytes())?,
    }
    Ok(files)
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

// ---------------------------------------------------------------- settings

const MESH_SCALE: f64 = 0.1;
const CUBE_MARGIN: f32 = 0.1;
const POINT_SCALE: f64 = 1.0;

const ISO_SIZE: i32 = 2;
const ISO_HEIGHT: i32 = 4;
const ISO_VIEWS: i32 = 4;
const ISO_OUTLINE: bool = true;
const ISO_TOP: [f32; 3] = [1.17, 1.15, 1.25];
const ISO_LEFT: [f32; 3] = [0.37, 0.35, 0.55];
const ISO_RIGHT: [f32; 3] = [0.87, 0.85, 0.95];
const ISO_BEVEL: [f32; 3] = [0.13, 0.13, 0.13];

const SPRITE_SIZE: i32 = 16;
const SPRITE_SHADOW: bool = true;

// ---------------------------------------------------------------- helpers

/// The six face directions: +x, -x, +y, -y, +z, -z.
fn dir(n: usize) -> IVec3 {
    let mut d = IVec3::ZERO;
    d[n / 2] = if n % 2 == 0 { 1 } else { -1 };
    d
}

fn rgb(pal: &[[u8; 4]], c: u8) -> [u8; 3] {
    let p = pal[c as usize];
    [p[0], p[1], p[2]]
}

/// Like C's `%g` for the values meshes produce: no exponent, no trailing
/// zeros.
fn num(f: f64) -> String {
    let s = format!("{f:.6}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" || s.is_empty() { "0".into() } else { s.into() }
}

/// Mesh vertex in output space: centered on the model, scaled, and for
/// Y-up mapped to `(x, z, -y)`.
fn place(v: &Volume, p: Vec3, scale: f64, y_up: bool) -> [f64; 3] {
    let q = (p - v.dims.as_vec3() * 0.5).as_dvec3() * scale;
    if y_up { [q.x, q.z, -q.y] } else { [q.x, q.y, q.z] }
}

fn ply_header(out: &mut String, verts: usize, faces: Option<usize>) {
    out.push_str("ply\nformat ascii 1.0\ncomment : voxely (MagicaVoxel compatible)\n");
    let _ = writeln!(out, "element vertex {verts}");
    out.push_str("property float x\nproperty float y\nproperty float z\n");
    out.push_str("property uchar red\nproperty uchar green\nproperty uchar blue\n");
    if let Some(f) = faces {
        let _ = writeln!(out, "element face {f}");
        out.push_str("property list uchar int vertex_index\n");
    }
    out.push_str("end_header\n");
}

fn ply_vertex(out: &mut String, p: [f64; 3], c: [u8; 3]) {
    let _ = writeln!(out, "{} {} {} {} {} {}", num(p[0]), num(p[1]), num(p[2]), c[0], c[1], c[2]);
}

// ---------------------------------------------------------------- greedy mesh

/// A rectangle of exposed voxel faces of one color, corners counter-clockwise
/// seen from outside, in voxel units.
struct Quad {
    /// Index into `dir`.
    normal: usize,
    color: u8,
    corners: [IVec3; 4],
}

/// Exposed faces merged into maximal rectangles per color, plane and
/// direction (io_obj `merge 1`).
fn greedy_quads(v: &Volume) -> Vec<Quad> {
    let d = v.dims;
    let mut quads = Vec::new();
    for n in 0..6 {
        let axis = n / 2;
        let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
        let step = dir(n);
        let (nu, nw) = (d[u] as usize, d[w] as usize);
        let mut mask = vec![0u8; nu * nw];
        for k in 0..d[axis] {
            for j in 0..nw {
                for i in 0..nu {
                    let mut p = IVec3::ZERO;
                    p[axis] = k;
                    p[u] = i as i32;
                    p[w] = j as i32;
                    let c = v.get(p);
                    mask[i + j * nu] = if c != 0 && v.get(p + step) == 0 { c } else { 0 };
                }
            }
            for j in 0..nw {
                let mut i = 0;
                while i < nu {
                    let c = mask[i + j * nu];
                    if c == 0 {
                        i += 1;
                        continue;
                    }
                    let mut width = 1;
                    while i + width < nu && mask[i + width + j * nu] == c {
                        width += 1;
                    }
                    let mut height = 1;
                    'grow: while j + height < nw {
                        for x in i..i + width {
                            if mask[x + (j + height) * nu] != c {
                                break 'grow;
                            }
                        }
                        height += 1;
                    }
                    for y in j..j + height {
                        mask[y * nu + i..y * nu + i + width].fill(0);
                    }
                    let mut b = IVec3::ZERO;
                    b[axis] = k + (n % 2 == 0) as i32;
                    b[u] = i as i32;
                    b[w] = j as i32;
                    let mut du = IVec3::ZERO;
                    du[u] = width as i32;
                    let mut dw = IVec3::ZERO;
                    dw[w] = height as i32;
                    // u x w = +axis, so (b, b+du, b+du+dw, b+dw) faces +axis.
                    let corners = if n % 2 == 0 {
                        [b, b + du, b + du + dw, b + dw]
                    } else {
                        [b, b + dw, b + du + dw, b + du]
                    };
                    quads.push(Quad { normal: n, color: c, corners });
                    i += width;
                }
            }
        }
    }
    quads
}

// ---------------------------------------------------------------- obj

/// obj + mtl + 256x1 palette texture. Texture pixel `k` holds color index
/// `k + 1` (MagicaVoxel's palette image layout; pixel 255 holds entry 0), so
/// a face of color `c` uses `vt ((c - 1) + 0.5) / 256, 0.5`, the center of
/// its pixel. One `vt` per used color, six `vn`, shared vertices, quads.
fn write_obj(v: &Volume, pal: &[[u8; 4]], path: &Path) -> Result<Vec<PathBuf>> {
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
    let mtl_path = path.with_extension("mtl");
    let png_path = path.with_extension("png");
    let quads = greedy_quads(v);

    let mut out = String::new();
    out.push_str("# voxely (MagicaVoxel compatible)\n\n# group\n");
    let _ = writeln!(out, "o {stem}\n\n# material\nmtllib {stem}.mtl\nusemtl palette\n\n# normals");
    for n in 0..6 {
        let d = dir(n);
        let _ = writeln!(out, "vn {} {} {}", d.x, d.z, -d.y);
    }
    out.push_str("\n# texcoords\n");
    let mut used = [false; 256];
    for q in &quads {
        used[q.color as usize] = true;
    }
    let mut tex = [0usize; 256];
    let mut count = 0;
    for c in 1..256 {
        if used[c] {
            count += 1;
            tex[c] = count;
            let _ = writeln!(out, "vt {} 0.5", num((c as f64 - 1.0 + 0.5) / 256.0));
        }
    }
    out.push_str("\n# verts\n");
    let mut index: HashMap<IVec3, usize> = HashMap::new();
    let mut faces = Vec::with_capacity(quads.len());
    for q in &quads {
        let ids = q.corners.map(|p| {
            let next = index.len() + 1;
            *index.entry(p).or_insert_with(|| {
                let o = place(v, p.as_vec3(), MESH_SCALE, true);
                let _ = writeln!(out, "v {} {} {}", num(o[0]), num(o[1]), num(o[2]));
                next
            })
        });
        faces.push((ids, tex[q.color as usize], q.normal + 1));
    }
    out.push_str("\n# faces\n");
    for (ids, t, n) in faces {
        let _ = writeln!(out, "f {}/{t}/{n} {}/{t}/{n} {}/{t}/{n} {}/{t}/{n}", ids[0], ids[1], ids[2], ids[3]);
    }
    write_file(path, out.as_bytes())?;

    let mtl = format!(
        "# voxely (MagicaVoxel compatible)\n\nnewmtl palette\nillum 1\nKa 0.000 0.000 0.000\nKd 1.000 1.000 1.000\nKs 0.000 0.000 0.000\nmap_Kd {stem}.png\n"
    );
    write_file(&mtl_path, mtl.as_bytes())?;

    let mut img = Image::new(256, 1);
    for k in 0..256 {
        img.put(k, 0, pal[(k as usize + 1) % 256]);
    }
    save_image(&png_path, &img)?;
    Ok(vec![path.to_path_buf(), mtl_path, png_path])
}

// ---------------------------------------------------------------- ply meshes

/// Greedy-merged faces as colored quads; vertices are duplicated per face so
/// colors stay flat.
fn ply_mesh(v: &Volume, pal: &[[u8; 4]]) -> String {
    let quads = greedy_quads(v);
    let mut out = String::new();
    ply_header(&mut out, quads.len() * 4, Some(quads.len()));
    for q in &quads {
        for p in q.corners {
            ply_vertex(&mut out, place(v, p.as_vec3(), MESH_SCALE, false), rgb(pal, q.color));
        }
    }
    for i in 0..quads.len() {
        let b = i * 4;
        let _ = writeln!(out, "4 {} {} {} {}", b, b + 1, b + 2, b + 3);
    }
    out
}

/// One cube per voxel not enclosed on all six sides, shrunk by the margin:
/// it spans `[p + margin/2, p + 1 - margin/2]`. Eight shared vertices and six
/// quads per cube.
fn cubes(v: &Volume, pal: &[[u8; 4]]) -> String {
    let mut list = Vec::new();
    for (i, &c) in v.cells.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let p = v.position(i);
        if (0..6).any(|n| v.get(p + dir(n)) == 0) {
            list.push((p, c));
        }
    }
    let mut out = String::new();
    ply_header(&mut out, list.len() * 8, Some(list.len() * 6));
    let lo = CUBE_MARGIN * 0.5;
    let hi = 1.0 - CUBE_MARGIN * 0.5;
    for &(p, c) in &list {
        for k in 0..8 {
            let o = Vec3::new(
                if k & 1 != 0 { hi } else { lo },
                if k & 2 != 0 { hi } else { lo },
                if k & 4 != 0 { hi } else { lo },
            );
            ply_vertex(&mut out, place(v, p.as_vec3() + o, MESH_SCALE, false), rgb(pal, c));
        }
    }
    // Corner bit masks per face, counter-clockwise from outside.
    const FACES: [[usize; 4]; 6] = [
        [1, 3, 7, 5], // +x
        [0, 4, 6, 2], // -x
        [2, 6, 7, 3], // +y
        [0, 1, 5, 4], // -y
        [4, 5, 7, 6], // +z
        [0, 2, 3, 1], // -z
    ];
    for i in 0..list.len() {
        let b = i * 8;
        for f in FACES {
            let _ = writeln!(out, "4 {} {} {} {}", b + f[0], b + f[1], b + f[2], b + f[3]);
        }
    }
    out
}

/// The "Marching Cubes" export, done as naive Surface Nets (no lookup
/// tables): the occupancy samples are voxel centers; every dual cell with
/// mixed corners gets one vertex at the mean of its sign-change edge
/// midpoints, and every solid/empty voxel pair contributes a quad (two
/// triangles) joining the four cells around their shared edge, wound to
/// face from solid to empty. Vertices are shared, so the surface is closed.
/// A vertex takes the color of its nearest solid corner voxel.
fn marching(v: &Volume, pal: &[[u8; 4]]) -> String {
    let d = v.dims;
    let cd = d + IVec3::ONE; // cells c in [-1, d-1]
    let cell = |c: IVec3| ((c.x + 1) + cd.x * ((c.y + 1) + cd.y * (c.z + 1))) as usize;
    let mut ids = vec![u32::MAX; (cd.x * cd.y * cd.z) as usize];
    let mut verts: Vec<(Vec3, u8)> = Vec::new();
    for z in -1..d.z {
        for y in -1..d.y {
            for x in -1..d.x {
                let c = IVec3::new(x, y, z);
                let mut corner = [0u8; 8];
                for (k, cc) in corner.iter_mut().enumerate() {
                    let o = IVec3::new((k & 1) as i32, ((k >> 1) & 1) as i32, ((k >> 2) & 1) as i32);
                    *cc = v.get(c + o);
                }
                let solid = corner.iter().filter(|&&k| k != 0).count();
                if solid == 0 || solid == 8 {
                    continue;
                }
                let mut sum = Vec3::ZERO;
                let mut n = 0;
                for a in 0..8 {
                    for bit in [1, 2, 4] {
                        let b = a | bit;
                        if b != a && (corner[a] != 0) != (corner[b] != 0) {
                            let pa = Vec3::new((a & 1) as f32, ((a >> 1) & 1) as f32, ((a >> 2) & 1) as f32);
                            let pb = Vec3::new((b & 1) as f32, ((b >> 1) & 1) as f32, ((b >> 2) & 1) as f32);
                            sum += (pa + pb) * 0.5;
                            n += 1;
                        }
                    }
                }
                let off = sum / n as f32;
                let mut best = (f32::MAX, 0u8);
                for (k, &cc) in corner.iter().enumerate() {
                    if cc != 0 {
                        let o = Vec3::new((k & 1) as f32, ((k >> 1) & 1) as f32, ((k >> 2) & 1) as f32);
                        let dist = (o - off).length_squared();
                        if dist < best.0 {
                            best = (dist, cc);
                        }
                    }
                }
                ids[cell(c)] = verts.len() as u32;
                verts.push((c.as_vec3() + Vec3::splat(0.5) + off, best.1));
            }
        }
    }
    let mut tris: Vec<[u32; 3]> = Vec::new();
    for axis in 0..3 {
        let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
        let e = dir(axis * 2);
        for k in -1..d[axis] {
            for j in 0..d[w] {
                for i in 0..d[u] {
                    let mut a = IVec3::ZERO;
                    a[axis] = k;
                    a[u] = i;
                    a[w] = j;
                    let sa = v.get(a) != 0;
                    if sa == (v.get(a + e) != 0) {
                        continue;
                    }
                    let at = |du: i32, dw: i32| {
                        let mut c = a;
                        c[u] += du;
                        c[w] += dw;
                        ids[cell(c)]
                    };
                    let mut q = [at(-1, -1), at(0, -1), at(0, 0), at(-1, 0)];
                    if !sa {
                        q.reverse();
                    }
                    tris.push([q[0], q[1], q[2]]);
                    tris.push([q[0], q[2], q[3]]);
                }
            }
        }
    }
    let mut out = String::new();
    ply_header(&mut out, verts.len(), Some(tris.len()));
    for &(p, c) in &verts {
        ply_vertex(&mut out, place(v, p, MESH_SCALE, false), rgb(pal, c));
    }
    for t in tris {
        let _ = writeln!(out, "3 {} {} {}", t[0], t[1], t[2]);
    }
    out
}

/// Voxel centers (sample offset 0), centered on the model, scale 1, Z up.
fn points(v: &Volume, pal: &[[u8; 4]]) -> String {
    let count = v.cells.iter().filter(|&&c| c != 0).count();
    let mut out = String::new();
    ply_header(&mut out, count, None);
    for (i, &c) in v.cells.iter().enumerate() {
        if c != 0 {
            let p = v.position(i).as_vec3() + Vec3::splat(0.5);
            ply_vertex(&mut out, place(v, p, POINT_SCALE, false), rgb(pal, c));
        }
    }
    out
}

// ---------------------------------------------------------------- volumes

fn push_i32(b: &mut Vec<u8>, x: i32) {
    b.extend_from_slice(&x.to_le_bytes());
}

fn xraw(v: &Volume, pal: &[[u8; 4]]) -> Vec<u8> {
    let mut b = Vec::with_capacity(24 + v.cells.len() + 1024);
    b.extend_from_slice(b"XRAW");
    b.extend_from_slice(&[0, 4, 8, 8]); // unsigned, RGBA, 8 bits/channel, 8 bits/index
    for x in v.dims.to_array() {
        push_i32(&mut b, x);
    }
    push_i32(&mut b, 256);
    b.extend_from_slice(&v.cells);
    b.extend_from_slice(&[0, 0, 0, 0]);
    for c in &pal[1..256] {
        b.extend_from_slice(c);
    }
    b
}

fn slab(v: &Volume, pal: &[[u8; 4]]) -> Vec<u8> {
    let d = v.dims;
    let mut b = Vec::with_capacity(12 + v.cells.len() + 768);
    for x in d.to_array() {
        push_i32(&mut b, x);
    }
    for x in 0..d.x {
        for y in (0..d.y).rev() {
            for z in (0..d.z).rev() {
                b.push(v.get(IVec3::new(x, y, z)).wrapping_sub(1));
            }
        }
    }
    for i in (1..256).chain([0]) {
        b.extend_from_slice(&[pal[i][0] >> 2, pal[i][1] >> 2, pal[i][2] >> 2]);
    }
    b
}

fn qb(v: &Volume, pal: &[[u8; 4]], name: &str) -> Vec<u8> {
    let d = v.dims;
    let mut b = Vec::with_capacity(64 + v.cells.len() * 4);
    b.extend_from_slice(&[1, 1, 0, 0]); // version 1.1.0.0
    for x in [0, 1, 0, 0, 1] {
        // color format RGBA, right-handed, uncompressed, no visibility mask, one matrix
        b.extend_from_slice(&(x as u32).to_le_bytes());
    }
    let name = &name.as_bytes()[..name.len().min(127)];
    b.push(name.len() as u8);
    b.extend_from_slice(name);
    for x in [d.x, d.z, d.y] {
        b.extend_from_slice(&(x as u32).to_le_bytes());
    }
    for _ in 0..3 {
        push_i32(&mut b, 0);
    }
    for qz in 0..d.y {
        for qy in 0..d.z {
            for qx in 0..d.x {
                let c = v.get(IVec3::new(qx, d.y - 1 - qz, qy));
                if c == 0 {
                    b.extend_from_slice(&[0, 0, 0, 0]);
                } else {
                    let p = pal[c as usize];
                    b.extend_from_slice(&[p[0], p[1], p[2], 255]);
                }
            }
        }
    }
    b
}

// ---------------------------------------------------------------- images

struct Image {
    w: i32,
    h: i32,
    px: Vec<[u8; 4]>,
}

impl Image {
    fn new(w: i32, h: i32) -> Self {
        Image { w, h, px: vec![[0; 4]; (w.max(1) * h.max(1)) as usize] }
    }

    fn get(&self, x: i32, y: i32) -> [u8; 4] {
        if x < 0 || y < 0 || x >= self.w || y >= self.h { [0; 4] } else { self.px[(x + y * self.w) as usize] }
    }

    fn put(&mut self, x: i32, y: i32, c: [u8; 4]) {
        if x >= 0 && y >= 0 && x < self.w && y < self.h {
            self.px[(x + y * self.w) as usize] = c;
        }
    }

    fn blit(&mut self, src: &Image, ox: i32, oy: i32) {
        for y in 0..src.h {
            for x in 0..src.w {
                self.put(ox + x, oy + y, src.get(x, y));
            }
        }
    }

    /// Dark rim on transparent pixels next to opaque ones.
    fn outline(&mut self) {
        let src = self.px.clone();
        let (w, h) = (self.w, self.h);
        let at = |x: i32, y: i32| if x < 0 || y < 0 || x >= w || y >= h { [0; 4] } else { src[(x + y * w) as usize] };
        for y in 0..self.h {
            for x in 0..self.w {
                if at(x, y)[3] != 0 {
                    continue;
                }
                let n = [(1, 0), (-1, 0), (0, 1), (0, -1)].into_iter().map(|(dx, dy)| at(x + dx, y + dy)).find(|c| c[3] != 0);
                if let Some(c) = n {
                    self.put(x, y, [c[0] / 4, c[1] / 4, c[2] / 4, 255]);
                }
            }
        }
    }
}

fn save_image(path: &Path, img: &Image) -> Result<()> {
    crate::save_png(path, img.w.max(1) as u32, img.h.max(1) as u32, bytemuck::cast_slice(&img.px))
}

fn shade(c: [u8; 4], k: [f32; 3], add: [f32; 3]) -> [u8; 4] {
    let f = |i: usize| ((c[i] as f32 / 255.0 * k[i] + add[i]) * 255.0).round().clamp(0.0, 255.0) as u8;
    [f(0), f(1), f(2), 255]
}

/// Face of an iso voxel sprite a pixel belongs to.
#[derive(Clone, Copy, PartialEq)]
enum IsoFace {
    Top,
    /// +y face, lower left of the diamond.
    Left,
    /// +x face, lower right.
    Right,
}

/// One voxel's sprite: pixel offset, face, and for top pixels the in-plane
/// neighbor directions (indices into `dir`) whose edge the pixel touches.
fn iso_sprite() -> Vec<(i32, i32, IsoFace, Vec<usize>)> {
    let (w, t, h) = ((4 * ISO_SIZE) as f32, (2 * ISO_SIZE) as f32, ISO_HEIGHT as f32);
    // Top face coordinates (a along +x, b along +y), both in [0, 1] inside.
    let ab = |cx: f32, cy: f32| (cy / t + (cx - w / 2.0) / w, cy / t - (cx - w / 2.0) / w);
    let inside = |(a, b): (f32, f32)| (0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&b);
    let mut out = Vec::new();
    for py in 0..(t + h) as i32 {
        for px in 0..w as i32 {
            let (cx, cy) = (px as f32 + 0.5, py as f32 + 0.5);
            if inside(ab(cx, cy)) {
                let mut edges = Vec::new();
                for (dx, dy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                    let (a, b) = ab(cx + dx, cy + dy);
                    for (cond, n) in [(a > 1.0, 0), (a < 0.0, 1), (b > 1.0, 2), (b < 0.0, 3)] {
                        if cond && !edges.contains(&n) {
                            edges.push(n);
                        }
                    }
                }
                out.push((px, py, IsoFace::Top, edges));
            } else if cx < w / 2.0 {
                let top = t / 2.0 + cx * t / w;
                if cy >= top && cy <= top + h {
                    out.push((px, py, IsoFace::Left, Vec::new()));
                }
            } else {
                let top = t - (cx - w / 2.0) * t / w;
                if cy >= top && cy <= top + h {
                    out.push((px, py, IsoFace::Right, Vec::new()));
                }
            }
        }
    }
    out
}

/// Isometric pixel sprite (io_iso): a 2:1 projection where a voxel's top is
/// a `4*size` x `2*size` diamond and its sides are `height` px tall. Screen
/// x = (x - y) * 2*size, screen y = (x + y) * size - z * height, so +x faces
/// are lit with `right` and +y faces with `left`. Voxels are painted in
/// increasing x + y + z. Top-face pixels on an edge whose in-plane neighbor
/// is empty get the bevel highlight; each view gets a 1 px dark outline in
/// a 1 px margin. The four views sit side by side, the first showing the
/// model's -y (front) and -x sides, each next one turned 90° about z; each
/// is `(X+Y)*2*size + 2` wide and `(X+Y)*size + Z*height + 2` tall.
fn iso(v: &Volume, pal: &[[u8; 4]]) -> Image {
    let d = v.dims;
    let (sw, st) = (2 * ISO_SIZE, ISO_SIZE); // half diamond width, half height
    let fw = (d.x + d.y) * sw + 2;
    let fh = (d.x + d.y) * st + d.z * ISO_HEIGHT + 2;
    let sprite = iso_sprite();
    let mut img = Image::new(fw * ISO_VIEWS, fh);
    for view in 0..ISO_VIEWS {
        // Turn the model so the first view looks at its front (-y) side.
        let turn = (view + 2) % 4;
        // Rotated dims and a lookup from rotated to model coordinates.
        let rd = if turn % 2 == 0 { d } else { IVec3::new(d.y, d.x, d.z) };
        let get = |p: IVec3| {
            let (x, y) = match turn {
                0 => (p.x, p.y),
                1 => (p.y, d.y - 1 - p.x),
                2 => (d.x - 1 - p.x, d.y - 1 - p.y),
                _ => (d.x - 1 - p.y, p.x),
            };
            v.get(IVec3::new(x, y, p.z))
        };
        let mut order: Vec<Vec<IVec3>> = vec![Vec::new(); (rd.x + rd.y + rd.z) as usize];
        for z in 0..rd.z {
            for y in 0..rd.y {
                for x in 0..rd.x {
                    let p = IVec3::new(x, y, z);
                    if get(p) != 0 && (get(p + IVec3::X) == 0 || get(p + IVec3::Y) == 0 || get(p + IVec3::Z) == 0) {
                        order[(x + y + z) as usize].push(p);
                    }
                }
            }
        }
        let mut frame = Image::new(fw, fh);
        for p in order.into_iter().flatten() {
            let c = pal[get(p) as usize];
            let ox = 1 + (p.x - p.y + rd.y - 1) * sw;
            let oy = 1 + (p.x + p.y) * st + (rd.z - 1 - p.z) * ISO_HEIGHT;
            let top_open = get(p + IVec3::Z) == 0;
            for (px, py, face, edges) in &sprite {
                let col = match face {
                    IsoFace::Top if top_open => {
                        let bevel = ISO_OUTLINE && edges.iter().any(|&n| get(p + dir(n)) == 0);
                        shade(c, ISO_TOP, if bevel { ISO_BEVEL } else { [0.0; 3] })
                    }
                    IsoFace::Top => continue,
                    IsoFace::Left if get(p + IVec3::Y) == 0 => shade(c, ISO_LEFT, [0.0; 3]),
                    IsoFace::Right if get(p + IVec3::X) == 0 => shade(c, ISO_RIGHT, [0.0; 3]),
                    _ => continue,
                };
                frame.put(ox + px, oy + py, col);
            }
        }
        if ISO_OUTLINE {
            frame.outline();
        }
        img.blit(&frame, view * fw, 0);
    }
    img
}

/// Front sprite (io_2d): looking along +y, x to the right and z up, `size`
/// px per voxel, the frontmost voxel's color. With `shadow`, a voxel is
/// darkened by how far it sits behind the frontmost voxel of the model (up
/// to 45% at the back of the model's depth range). Image is `X*size` by
/// `Z*size`, transparent where the column is empty.
fn sprite_2d(v: &Volume, pal: &[[u8; 4]]) -> Image {
    let d = v.dims;
    let mut front = vec![None; (d.x * d.z) as usize];
    let (mut lo, mut hi) = (i32::MAX, i32::MIN);
    for z in 0..d.z {
        for x in 0..d.x {
            if let Some(y) = (0..d.y).find(|&y| v.get(IVec3::new(x, y, z)) != 0) {
                front[(x + z * d.x) as usize] = Some((y, v.get(IVec3::new(x, y, z))));
                lo = lo.min(y);
                hi = hi.max(y);
            }
        }
    }
    let s = SPRITE_SIZE;
    let mut img = Image::new(d.x * s, d.z * s);
    for z in 0..d.z {
        for x in 0..d.x {
            let Some((y, c)) = front[(x + z * d.x) as usize] else { continue };
            let k = if SPRITE_SHADOW && hi > lo { 1.0 - 0.45 * (y - lo) as f32 / (hi - lo) as f32 } else { 1.0 };
            let col = shade(pal[c as usize], [k; 3], [0.0; 3]);
            for py in 0..s {
                for px in 0..s {
                    img.put(x * s + px, (d.z - 1 - z) * s + py, col);
                }
            }
        }
    }
    img
}

/// Every z layer, bottom first, left to right, 1 px per voxel with a 1 px
/// transparent gap: `Z*X + (Z-1)` wide, `Y` tall. Each slice is seen from
/// above (x right, y up, so row 0 is y = Y-1).
fn slices(v: &Volume, pal: &[[u8; 4]]) -> Image {
    let d = v.dims;
    let mut img = Image::new(d.z * (d.x + 1) - 1, d.y);
    for z in 0..d.z {
        for y in 0..d.y {
            for x in 0..d.x {
                let c = v.get(IVec3::new(x, y, z));
                if c != 0 {
                    let p = pal[c as usize];
                    img.put(z * (d.x + 1) + x, d.y - 1 - y, [p[0], p[1], p[2], 255]);
                }
            }
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3x3x3: a 2x2x1 base of color 1, a column of color 2 rising from its
    /// corner (an L), and a lone voxel of color 3. The enclosed-voxel case
    /// is tested with a full block.
    fn test_scene() -> Scene {
        let mut s = Scene::empty(IVec3::new(3, 3, 3));
        let mut set = |p: IVec3, c: u8| {
            let i = s.volume.index(p);
            s.volume.set_index(i, c);
        };
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            set(IVec3::new(x, y, 0), 1);
        }
        set(IVec3::new(0, 0, 1), 2);
        set(IVec3::new(0, 0, 2), 2);
        set(IVec3::new(2, 2, 2), 3);
        s.palette[1] = [200, 30, 30, 255];
        s.palette[2] = [30, 200, 30, 255];
        s.palette[3] = [30, 30, 200, 255];
        s
    }

    fn voxels(s: &Scene) -> usize {
        s.volume.cells.iter().filter(|&&c| c != 0).count()
    }

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!("voxely-export-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn png_size(p: &Path) -> (u32, u32) {
        let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(p).unwrap()));
        let r = dec.read_info().unwrap();
        (r.info().width, r.info().height)
    }

    /// (vertex count, face count, data lines) of an ASCII PLY.
    fn ply_counts(p: &Path) -> (usize, usize, usize) {
        let text = std::fs::read_to_string(p).unwrap();
        let (head, body) = text.split_once("end_header\n").unwrap();
        let get = |k: &str| {
            head.lines().find_map(|l| l.strip_prefix(k)).map(|n| n.trim().parse().unwrap()).unwrap_or(0)
        };
        (get("element vertex"), get("element face"), body.lines().count())
    }

    #[test]
    fn names_round_trip() {
        for f in Format::ALL {
            assert_eq!(Format::from_name(f.name()), Some(f));
            assert!(f.hint().starts_with(f.name()));
        }
        assert_eq!(Format::from_name("pcl"), Some(Format::Point));
        assert_eq!(Format::from_name("nope"), None);
    }

    #[test]
    fn every_format_writes() {
        let s = test_scene();
        let dir = tmp().join("all");
        for f in Format::ALL {
            let files = export(f, &s, &dir.join(format!("t-{}.x", f.name()))).unwrap();
            for p in &files {
                assert!(p.exists() && std::fs::metadata(p).unwrap().len() > 0, "{f:?} {}", p.display());
            }
            assert_eq!(files[0].extension().unwrap(), f.extension());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn greedy_merges_same_color() {
        // A lone 2x2x1 slab of one color is a box of 6 quads.
        let mut s = Scene::empty(IVec3::new(2, 2, 1));
        s.volume.cells.fill(5);
        assert_eq!(greedy_quads(&s.volume).len(), 6);

        let dir = tmp().join("obj");
        let files = export(Format::Obj, &s, &dir.join("slab.obj")).unwrap();
        assert_eq!(files.len(), 3);
        let text = std::fs::read_to_string(&files[0]).unwrap();
        let count = |p: &str| text.lines().filter(|l| l.starts_with(p)).count();
        assert_eq!(count("v "), 8);
        assert_eq!(count("vn "), 6);
        assert_eq!(count("vt "), 1);
        assert_eq!(count("f "), 6);
        // Color 5 -> pixel 4 -> u = 4.5 / 256.
        assert!(text.contains(&format!("vt {} 0.5", num(4.5 / 256.0))));
        assert_eq!(png_size(&files[2]), (256, 1));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn obj_faces_point_outward() {
        let s = test_scene();
        let dir = tmp().join("obj2");
        let files = export(Format::Obj, &s, &dir.join("m.obj")).unwrap();
        let text = std::fs::read_to_string(&files[0]).unwrap();
        let mut verts = vec![Vec3::ZERO];
        let mut normals = vec![Vec3::ZERO];
        for l in text.lines() {
            let n: Vec<f32> = l.split_whitespace().skip(1).filter_map(|x| x.parse().ok()).collect();
            if l.starts_with("v ") {
                verts.push(Vec3::new(n[0], n[1], n[2]));
            } else if l.starts_with("vn ") {
                normals.push(Vec3::new(n[0], n[1], n[2]));
            } else if l.starts_with("f ") {
                let idx: Vec<Vec<usize>> =
                    l.split_whitespace().skip(1).map(|t| t.split('/').map(|x| x.parse().unwrap()).collect()).collect();
                let (a, b, c) = (verts[idx[0][0]], verts[idx[1][0]], verts[idx[2][0]]);
                let geo = (b - a).cross(c - a).normalize();
                assert!(geo.dot(normals[idx[0][2]]) > 0.99, "{l}");
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ply_files_are_consistent() {
        let s = test_scene();
        let dir = tmp().join("ply");
        let n = voxels(&s);
        for f in [Format::Ply, Format::Mc, Format::Cube, Format::Point] {
            let p = &export(f, &s, &dir.join(f.name())).unwrap()[0];
            let (v, faces, lines) = ply_counts(p);
            assert_eq!(v + faces, lines, "{f:?}");
            match f {
                Format::Point => assert_eq!((v, faces), (n, 0)),
                Format::Cube => assert_eq!((v, faces), (n * 8, n * 6)),
                _ => assert!(v > 0 && faces > 0),
            }
        }
        // An enclosed voxel is skipped by cube.
        let mut full = Scene::empty(IVec3::splat(3));
        full.volume.cells.fill(1);
        let p = &export(Format::Cube, &full, &dir.join("full")).unwrap()[0];
        assert_eq!(ply_counts(p).0, 26 * 8);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn surface_nets_is_closed() {
        let s = test_scene();
        let text = marching(&s.volume, &s.palette);
        let body = text.split_once("end_header\n").unwrap().1;
        let verts: Vec<Vec3> = body
            .lines()
            .filter(|l| l.split_whitespace().count() == 6)
            .map(|l| {
                let n: Vec<f32> = l.split_whitespace().take(3).map(|x| x.parse().unwrap()).collect();
                Vec3::new(n[0], n[1], n[2])
            })
            .collect();
        // Outward winding gives a positive enclosed volume.
        let mut volume = 0.0;
        for l in body.lines().filter(|l| l.starts_with("3 ") && l.split_whitespace().count() == 4) {
            let t: Vec<usize> = l.split_whitespace().skip(1).map(|x| x.parse().unwrap()).collect();
            volume += verts[t[0]].dot(verts[t[1]].cross(verts[t[2]])) / 6.0;
        }
        assert!(volume > 0.0, "{volume}");
        // Every directed edge must be matched by its reverse.
        let mut edges: HashMap<(u32, u32), i32> = HashMap::new();
        for l in body.lines().filter(|l| l.starts_with("3 ") && l.split_whitespace().count() == 4) {
            let t: Vec<u32> = l.split_whitespace().skip(1).map(|x| x.parse().unwrap()).collect();
            for k in 0..3 {
                *edges.entry((t[k], t[(k + 1) % 3])).or_default() += 1;
            }
        }
        for (&(a, b), &n) in &edges {
            assert_eq!(edges.get(&(b, a)).copied().unwrap_or(0), n, "edge {a}-{b}");
        }
        assert!(!body.lines().any(|l| l.contains(&u32::MAX.to_string())));
    }

    #[test]
    fn volume_formats() {
        let s = test_scene();
        let d = s.volume.dims;
        let n = voxels(&s);

        let x = xraw(&s.volume, &s.palette);
        assert_eq!(&x[..8], b"XRAW\x00\x04\x08\x08");
        assert_eq!(i32::from_le_bytes(x[8..12].try_into().unwrap()), 3);
        assert_eq!(x.len(), 24 + 27 + 1024);
        assert_eq!(x[24..51].iter().filter(|&&c| c != 0).count(), n);
        assert_eq!(&x[51 + 4..51 + 8], &[200, 30, 30, 255]);

        let b = slab(&s.volume, &s.palette);
        assert_eq!(b.len(), 12 + 27 + 768);
        assert_eq!(b[12..39].iter().filter(|&&c| c != 255).count(), n);
        // (0,0,0) color 1 -> slab (x=0, y=2, z=2) = index (0*3 + 2)*3 + 2.
        assert_eq!(b[12 + 8], 0);
        assert_eq!(&b[39..42], &[200 >> 2, 30 >> 2, 30 >> 2]);

        let q = qb(&s.volume, &s.palette, "m");
        assert_eq!(&q[..4], &[1, 1, 0, 0]);
        assert_eq!(q[24], 1);
        let sizes: Vec<u32> = (0..3).map(|i| u32::from_le_bytes(q[26 + i * 4..30 + i * 4].try_into().unwrap())).collect();
        assert_eq!(sizes, [d.x as u32, d.z as u32, d.y as u32]);
        let data = &q[26 + 24..];
        assert_eq!(data.len(), 27 * 4);
        assert_eq!(data.chunks(4).filter(|c| c[3] != 0).count(), n);
        // Model (0,0,2) is qb (0, 2, 2).
        let i = (0 + 2 * 3 + 2 * 9) * 4;
        assert_eq!(&data[i..i + 4], &[30, 200, 30, 255]);
    }

    #[test]
    fn vox_round_trips() {
        let s = test_scene();
        let dir = tmp().join("vox");
        let p = &export(Format::Vox, &s, &dir.join("m.txt")).unwrap()[0];
        let back = Scene::from_vox(&crate::vox::VoxFile::load(p).unwrap()).unwrap();
        assert_eq!(voxels(&back), voxels(&s));
        assert_eq!(back.palette[3], s.palette[3]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn image_sizes() {
        let s = test_scene();
        let dir = tmp().join("img");
        let (x, y, z) = (3, 3, 3);
        let size = |f| png_size(&export(f, &s, &dir.join(Format::name(f))).unwrap()[0]);
        let fw = (x + y) * 4 + 2;
        assert_eq!(size(Format::Iso), (fw * 4, (x + y) * 2 + z * 4 + 2));
        assert_eq!(size(Format::TwoD), (x * 16, z * 16));
        assert_eq!(size(Format::Slice), (z * (x + 1) - 1, y));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Writes iso/2d/slice of the knight for eyeballing:
    /// `cargo test --release export::tests::knight -- --ignored`.
    #[test]
    #[ignore]
    fn knight() {
        let home = std::env::var("HOME").unwrap();
        let model = Path::new(&home).join("Downloads/MagicaVoxel-0.99.6.2-macos-10.15/vox/chr_knight.vox");
        let s = crate::open_vox(&model).unwrap();
        let out = std::env::var("VOXELY_EXPORT_DIR").map(PathBuf::from).unwrap_or_else(|_| tmp());
        for f in Format::ALL {
            for p in export(f, &s, &out.join(format!("knight-{}", f.name()))).unwrap() {
                println!("wrote {}", p.display());
            }
        }
    }
}
