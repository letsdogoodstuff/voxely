//! Other voxel formats: Qubicle `.qb`, MagicaVoxel `.xraw`, Ace of Spades
//! `.vxl`, Build/Voxlap `.kvx`, `.binvox`, and Minecraft `.schematic`
//! (MCEdit/Schematica) and `.schem` (Sponge v1-v3).
//!
//! Every reader turns its file's axes into voxely's (z up, a model faces -y)
//! by a rotation, never a mirror, so models keep their handedness:
//!
//! - **qb** is Y-up. Right-handed files (the flag export.rs writes) map qb
//!   `(x, y, z)` to `(x, sz-1-z, y)`; left-handed ones to `(x, z, y)`.
//!   Matrix positions map the same way.
//! - **xraw** is MagicaVoxel's own axes, x fastest.
//! - **vxl** and **kvx** have z pointing down and y south: `(x, y, z)` goes to
//!   `(x, sy-1-y, sz-1-z)` (the same turn export.rs uses for SLAB6).
//! - **binvox** is Y-up and right-handed, y fastest, then z, then x:
//!   `(x, y, z)` goes to `(x, sz-1-z, y)`.
//! - **Minecraft** is Y-up with z south: `(x, y, z)` goes to
//!   `(x, length-1-z, y)`, so north is +y and builds face south (-y).
//!
//! Minecraft blocks have no colors, so names map to representative colors
//! (unknown names get a stable color hashed from the name).

use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use glam::IVec3;

use super::Model;

/// File extensions these readers handle (lowercase, no dot).
pub const EXTENSIONS: &[&str] = &["qb", "xraw", "vxl", "kvx", "binvox", "schematic", "schem"];

/// Largest model accepted per axis.
const MAX_AXIS: i64 = 4096;
/// Largest model (or qb file) accepted, in cells.
const MAX_CELLS: i64 = 200_000_000;
/// Largest decompressed NBT accepted.
const MAX_NBT: u64 = 1 << 30;
/// Color of formats without any (binvox).
const GRAY: [u8; 4] = [200, 200, 200, 255];

/// Read every model in the file (chosen by extension, checked against magic
/// where the format has one).
pub fn load(path: &Path) -> Result<Vec<Model>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mut models = parse(&ext, &bytes).with_context(|| format!("reading {}", path.display()))?;
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    for m in &mut models {
        if m.name.is_empty() {
            m.name = stem.clone();
        }
    }
    Ok(models)
}

/// Same, from bytes (ext without the dot), for tests.
pub fn parse(ext: &str, bytes: &[u8]) -> Result<Vec<Model>> {
    match ext.to_ascii_lowercase().as_str() {
        "qb" => qb(bytes),
        "xraw" => Ok(vec![xraw(bytes)?]),
        "vxl" => Ok(vec![vxl(bytes)?]),
        "kvx" => Ok(vec![kvx(bytes)?]),
        "binvox" => Ok(vec![binvox(bytes)?]),
        "schematic" | "schem" => Ok(vec![schematic(bytes)?]),
        _ => bail!("unsupported voxel format .{ext}"),
    }
}

// ---------------------------------------------------------------- helpers

/// A bounds-checked cursor over a file's bytes.
struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
    what: &'static str,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8], what: &'static str) -> Self {
        Reader { b, pos: 0, what }
    }

    fn left(&self) -> usize {
        self.b.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(n <= self.left(), "{}: file is truncated", self.what);
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn arr<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into()?)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.arr::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.arr()?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.arr()?))
    }

    fn be_u16(&mut self) -> Result<u16> {
        Ok(u16::from_be_bytes(self.arr()?))
    }

    fn be_i32(&mut self) -> Result<i32> {
        Ok(i32::from_be_bytes(self.arr()?))
    }

    fn be_i64(&mut self) -> Result<i64> {
        Ok(i64::from_be_bytes(self.arr()?))
    }
}

/// Model dimensions, rejecting empty or oversized boxes.
fn dims(what: &str, x: i64, y: i64, z: i64) -> Result<IVec3> {
    for n in [x, y, z] {
        ensure!((1..=MAX_AXIS).contains(&n), "{what}: size {x}x{y}x{z} is out of range (1 to {MAX_AXIS} per axis)");
    }
    ensure!(x * y * z <= MAX_CELLS, "{what}: {x}x{y}x{z} is too many cells");
    Ok(IVec3::new(x as i32, y as i32, z as i32))
}

fn to_i32(what: &str, v: i64) -> Result<i32> {
    i32::try_from(v).with_context(|| format!("{what}: position {v} is out of range"))
}

// ---------------------------------------------------------------- qb

/// Qubicle binary: matrices of 32-bit colors (alpha 0 = empty), x fastest,
/// optionally run-length coded per z slice.
fn qb(b: &[u8]) -> Result<Vec<Model>> {
    const CODE: u32 = 2;
    const NEXT_SLICE: u32 = 6;
    let mut r = Reader::new(b, "qb");
    r.take(4)?; // version, 1.1.0.0
    let bgra = match r.u32()? {
        0 => false,
        1 => true,
        x => bail!("qb: unknown color format {x}"),
    };
    let right = match r.u32()? {
        0 => false,
        1 => true,
        x => bail!("qb: unknown z axis orientation {x}"),
    };
    let compressed = r.u32()? != 0;
    r.u32()?; // visibility mask: alpha then holds face flags, still 0 = empty
    let count = r.u32()? as usize;
    ensure!(count <= r.left() / 25, "qb: bad matrix count {count}");
    let mut total = 0;
    let mut models = Vec::with_capacity(count);
    for _ in 0..count {
        let len = r.u8()? as usize;
        let name = String::from_utf8_lossy(r.take(len)?).into_owned();
        let size = [r.u32()?, r.u32()?, r.u32()?].map(i64::from);
        let qd = dims("qb", size[0], size[1], size[2])?;
        total += size[0] * size[1] * size[2];
        ensure!(total <= MAX_CELLS, "qb: too many cells in all");
        let p = [r.i32()?, r.i32()?, r.i32()?].map(i64::from);
        let (sx, sy, sz) = (qd.x as usize, qd.y as usize, qd.z as usize);
        let slice = sx * sy;
        let mut cells = vec![[0u8; 4]; slice * sz];
        if compressed {
            for z in cells.chunks_exact_mut(slice) {
                let mut i = 0;
                loop {
                    let v = r.arr::<4>()?;
                    match u32::from_le_bytes(v) {
                        NEXT_SLICE => break,
                        CODE => {
                            let n = r.u32()? as usize;
                            let c = r.arr::<4>()?;
                            ensure!(n <= slice - i, "qb: a run overflows its slice");
                            z[i..i + n].fill(c);
                            i += n;
                        }
                        _ => {
                            ensure!(i < slice, "qb: a slice overflows");
                            z[i] = v;
                            i += 1;
                        }
                    }
                }
            }
        } else {
            let data = r.take(cells.len() * 4)?;
            for (c, s) in cells.iter_mut().zip(data.chunks_exact(4)) {
                c.copy_from_slice(s);
            }
        }
        let mut voxels = Vec::new();
        for (i, c) in cells.iter().enumerate() {
            if c[3] == 0 {
                continue;
            }
            let (x, y, z) = (i % sx, (i / sx) % sy, i / slice);
            let rgb = if bgra { [c[2], c[1], c[0]] } else { [c[0], c[1], c[2]] };
            let vy = if right { sz - 1 - z } else { z };
            voxels.push((IVec3::new(x as i32, vy as i32, y as i32), [rgb[0], rgb[1], rgb[2], 255]));
        }
        let oy = if right { -(p[2] + sz as i64) } else { p[2] };
        let offset = IVec3::new(to_i32("qb", p[0])?, to_i32("qb", oy)?, to_i32("qb", p[1])?);
        models.push(Model { name, dims: IVec3::new(qd.x, qd.z, qd.y), offset, voxels });
    }
    Ok(models)
}

// ---------------------------------------------------------------- xraw

/// MagicaVoxel's raw volume: indices into a palette, or direct colors.
fn xraw(b: &[u8]) -> Result<Model> {
    let mut r = Reader::new(b, "xraw");
    ensure!(r.take(4)? == b"XRAW", "xraw: not an xraw file");
    let [kind, ch, bits, ibits] = r.arr::<4>()?;
    ensure!(kind == 0 || (kind == 2 && bits == 32), "xraw: unsupported channel type {kind} with {bits} bits");
    ensure!((1..=4).contains(&ch), "xraw: unsupported channel count {ch}");
    ensure!(matches!(bits, 8 | 16 | 32), "xraw: unsupported {bits} bits per channel");
    ensure!(matches!(ibits, 0 | 8 | 16), "xraw: unsupported {ibits} bits per index");
    let (w, h, d) = (r.i32()?, r.i32()?, r.i32()?);
    let dm = dims("xraw", w.into(), h.into(), d.into())?;
    let pal_n = r.i32()?;
    let n = (w as usize) * (h as usize) * (d as usize);
    let bw = bits as usize / 8;
    let csize = ch as usize * bw;
    let color = |s: &[u8]| -> [u8; 4] {
        let mut c = [0, 0, 0, 255];
        for (k, v) in s.chunks_exact(bw).enumerate() {
            c[k] = match bw {
                1 => v[0],
                2 => v[1],
                _ if kind == 2 => (f32::from_le_bytes([v[0], v[1], v[2], v[3]]).clamp(0.0, 1.0) * 255.0).round() as u8,
                _ => v[3],
            };
        }
        if ch == 1 {
            c[1] = c[0];
            c[2] = c[0];
        }
        c
    };
    let pos = |i: usize| IVec3::new((i % w as usize) as i32, (i / w as usize % h as usize) as i32, (i / (w as usize * h as usize)) as i32);
    let mut voxels = Vec::new();
    if ibits == 0 {
        for (i, s) in r.take(n * csize)?.chunks_exact(csize).enumerate() {
            let c = color(s);
            let empty = if ch == 4 { c[3] == 0 } else { c[..3] == [0, 0, 0] };
            if !empty {
                voxels.push((pos(i), [c[0], c[1], c[2], 255]));
            }
        }
    } else {
        ensure!((0..=1 << ibits).contains(&pal_n), "xraw: bad palette size {pal_n}");
        let ib = ibits as usize / 8;
        let idx = r.take(n * ib)?;
        let pal: Vec<[u8; 4]> = r.take(pal_n as usize * csize)?.chunks_exact(csize).map(color).collect();
        // MagicaVoxel writes 0 = empty with a transparent entry 0; the spec
        // otherwise reserves the largest index.
        let zero_empty = ch == 4 && pal.first().is_some_and(|c| c[3] == 0);
        let max = (1usize << ibits) - 1;
        for (i, s) in idx.chunks_exact(ib).enumerate() {
            let k = if ib == 1 { s[0] as usize } else { u16::from_le_bytes([s[0], s[1]]) as usize };
            let Some(c) = pal.get(k) else { continue };
            if (!zero_empty && k == max) || (ch == 4 && c[3] == 0) {
                continue;
            }
            voxels.push((pos(i), [c[0], c[1], c[2], 255]));
        }
    }
    Ok(Model { name: String::new(), dims: dm, offset: IVec3::ZERO, voxels })
}

// ---------------------------------------------------------------- vxl

/// An Ace of Spades map: 512x512 columns of spans, 64 deep, z down. Only the
/// colored (surface) voxels are kept; the file has no colors for the hidden
/// solid inside.
fn vxl(b: &[u8]) -> Result<Model> {
    const N: usize = 512;
    const D: usize = 64;
    let mut voxels = Vec::new();
    let mut pos = 0;
    let get = |p: usize| b.get(p).copied().context("vxl: file is truncated (expected a 512x512x64 Ace of Spades map)");
    let mut push = |x: usize, y: usize, z: usize, p: usize| -> Result<()> {
        let c = [get(p + 2)?, get(p + 1)?, get(p)?, 255];
        voxels.push((IVec3::new(x as i32, (N - 1 - y) as i32, (D - 1 - z) as i32), c));
        Ok(())
    };
    for y in 0..N {
        for x in 0..N {
            loop {
                let (n, s, e) = (get(pos)? as usize, get(pos + 1)? as usize, get(pos + 2)? as usize);
                ensure!(s <= e + 1 && e < D, "vxl: bad span at column ({x}, {y})");
                let top = e + 1 - s;
                for i in 0..top {
                    push(x, y, s + i, pos + 4 + 4 * i)?;
                }
                if n == 0 {
                    pos += 4 * (top + 1);
                    break;
                }
                ensure!(n > top, "vxl: bad span length at column ({x}, {y})");
                let bottom = n - 1 - top;
                let next = pos + 4 * n;
                let end = get(next + 3)? as usize; // where the next span's air starts
                ensure!(bottom <= end && end <= D, "vxl: bad span at column ({x}, {y})");
                for i in 0..bottom {
                    push(x, y, end - bottom + i, pos + 4 + 4 * (top + i))?;
                }
                pos = next;
            }
        }
    }
    ensure!(pos <= b.len(), "vxl: file is truncated");
    ensure!(pos == b.len(), "vxl: {} bytes after the map (only 512x512x64 Ace of Spades maps are supported)", b.len() - pos);
    let dm = IVec3::new(N as i32, N as i32, D as i32);
    Ok(Model { name: String::new(), dims: dm, offset: IVec3::ZERO, voxels })
}

// ---------------------------------------------------------------- kvx

/// Build engine voxels: columns of slabs, z down, 6-bit palette in the last
/// 768 bytes. Only the first (full detail) mip level is read.
fn kvx(b: &[u8]) -> Result<Model> {
    ensure!(b.len() >= 28 + 768, "kvx: file is too short");
    let pal = &b[b.len() - 768..];
    let mut r = Reader::new(&b[..b.len() - 768], "kvx");
    let numbytes = r.i32()?;
    ensure!(numbytes >= 24 && numbytes as usize <= r.left(), "kvx: bad mip size {numbytes}");
    let mip = &b[4..4 + numbytes as usize];
    let mut r = Reader::new(mip, "kvx");
    let (xs, ys, zs) = (r.i32()?, r.i32()?, r.i32()?);
    let dm = dims("kvx", xs.into(), ys.into(), zs.into())?;
    r.take(12)?; // pivot
    let (xs, ys, zs) = (xs as usize, ys as usize, zs as usize);
    let xoff = (0..=xs).map(|_| r.i32().map(i64::from)).collect::<Result<Vec<_>>>()?;
    let xyoff = (0..xs * (ys + 1)).map(|_| r.u16().map(i64::from)).collect::<Result<Vec<_>>>()?;
    let data = &mip[r.pos..];
    let mut voxels = Vec::new();
    for x in 0..xs {
        for y in 0..ys {
            let base = xoff[x] - xoff[0];
            let (a, z) = (base + xyoff[x * (ys + 1) + y], base + xyoff[x * (ys + 1) + y + 1]);
            ensure!(0 <= a && a <= z && z as usize <= data.len(), "kvx: bad column offset at ({x}, {y})");
            let mut col = Reader::new(&data[a as usize..z as usize], "kvx");
            while col.left() > 0 {
                let [top, len, _faces] = col.arr::<3>()?;
                for (i, &c) in col.take(len as usize)?.iter().enumerate() {
                    let vz = top as usize + i;
                    ensure!(vz < zs, "kvx: slab runs past the bottom at ({x}, {y})");
                    let p = &pal[c as usize * 3..c as usize * 3 + 3];
                    let six = |v: u8| (v & 63) << 2 | (v & 63) >> 4;
                    let pos = IVec3::new(x as i32, (ys - 1 - y) as i32, (zs - 1 - vz) as i32);
                    voxels.push((pos, [six(p[0]), six(p[1]), six(p[2]), 255]));
                }
            }
        }
    }
    Ok(Model { name: String::new(), dims: dm, offset: IVec3::ZERO, voxels })
}

// ---------------------------------------------------------------- binvox

/// binvox: a text header, then (value, count) byte pairs over the grid.
fn binvox(b: &[u8]) -> Result<Model> {
    ensure!(b.starts_with(b"#binvox"), "binvox: not a binvox file");
    let mut pos = 0;
    let mut dim = None;
    loop {
        let end = b[pos..].iter().position(|&c| c == b'\n').context("binvox: header is truncated")? + pos;
        ensure!(end < 4096, "binvox: header is too long");
        let line = std::str::from_utf8(&b[pos..end]).context("binvox: bad header")?.trim();
        pos = end + 1;
        if line == "data" {
            break;
        }
        if let Some(rest) = line.strip_prefix("dim ") {
            let v: Vec<i64> = rest.split_whitespace().map(str::parse).collect::<Result<_, _>>().context("binvox: bad dim line")?;
            ensure!(v.len() == 3, "binvox: bad dim line");
            dim = Some([v[0], v[1], v[2]]);
        }
    }
    let [dx, dz, dy] = dim.context("binvox: no dim line")?;
    let dm = dims("binvox", dx, dz, dy)?;
    let (dz, dy) = (dz as usize, dy as usize);
    let n = dm.x as usize * dz * dy;
    let mut voxels = Vec::new();
    let mut i = 0;
    for pair in b[pos..].chunks(2) {
        if i == n {
            break;
        }
        let [v, c] = pair else { bail!("binvox: data is truncated") };
        let c = *c as usize;
        ensure!(c <= n - i, "binvox: data overflows the grid");
        if *v != 0 {
            for k in i..i + c {
                let (y, z, x) = (k % dy, k / dy % dz, k / (dy * dz));
                voxels.push((IVec3::new(x as i32, (dz - 1 - z) as i32, y as i32), GRAY));
            }
        }
        i += c;
    }
    ensure!(i == n, "binvox: data is truncated");
    Ok(Model { name: String::new(), dims: dm, offset: IVec3::ZERO, voxels })
}

// ---------------------------------------------------------------- nbt

/// A Named Binary Tag value (Minecraft's big-endian tree format).
#[derive(Debug)]
enum Nbt {
    Int(i64),
    Float,
    Bytes(Vec<u8>),
    Str,
    List,
    Compound(Vec<(String, Nbt)>),
    Ints,
}

impl Nbt {
    fn get(&self, key: &str) -> Option<&Nbt> {
        match self {
            Nbt::Compound(c) => c.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn int(&self) -> Option<i64> {
        match self {
            Nbt::Int(v) => Some(*v),
            _ => None,
        }
    }
}

/// The file's bytes, gunzipped if they are gzip.
fn inflate(b: &[u8]) -> Result<Cow<'_, [u8]>> {
    if !b.starts_with(&[0x1f, 0x8b]) {
        return Ok(Cow::Borrowed(b));
    }
    let mut out = Vec::new();
    flate2::read::MultiGzDecoder::new(b).take(MAX_NBT + 1).read_to_end(&mut out).context("decompressing gzip")?;
    ensure!(out.len() as u64 <= MAX_NBT, "decompresses to more than 1 GiB");
    Ok(Cow::Owned(out))
}

fn nbt_root(b: &[u8]) -> Result<Nbt> {
    let mut r = Reader::new(b, "nbt");
    ensure!(r.u8()? == 10, "not an NBT file");
    nbt_string(&mut r)?;
    nbt_value(&mut r, 10, 0)
}

fn nbt_string(r: &mut Reader) -> Result<String> {
    let n = r.be_u16()? as usize;
    Ok(String::from_utf8_lossy(r.take(n)?).into_owned())
}

/// An array or list length, checked against the bytes left.
fn nbt_len(r: &mut Reader, elem: usize) -> Result<usize> {
    let n = r.be_i32()?;
    ensure!(n >= 0, "nbt: negative length");
    ensure!((n as usize).saturating_mul(elem) <= r.left(), "nbt: file is truncated");
    Ok(n as usize)
}

/// Strings, lists, floats and int arrays are skipped: schematics only need
/// the numbers, byte arrays and compounds.
fn nbt_value(r: &mut Reader, tag: u8, depth: u32) -> Result<Nbt> {
    ensure!(depth < 512, "nbt: nested too deeply");
    Ok(match tag {
        1 => Nbt::Int(r.u8()? as i8 as i64),
        2 => Nbt::Int(i16::from_be_bytes(r.arr()?) as i64),
        3 => Nbt::Int(r.be_i32()? as i64),
        4 => Nbt::Int(r.be_i64()?),
        5 => {
            r.take(4)?;
            Nbt::Float
        }
        6 => {
            r.take(8)?;
            Nbt::Float
        }
        7 => {
            let n = nbt_len(r, 1)?;
            Nbt::Bytes(r.take(n)?.to_vec())
        }
        8 => {
            nbt_string(r)?;
            Nbt::Str
        }
        9 => {
            let elem = r.u8()?;
            let min = match elem {
                0 => 0,
                1 => 1,
                2 | 8 => 2,
                3 | 5 | 7 | 11 | 12 | 9 => 4,
                4 | 6 => 8,
                _ => 1,
            };
            let n = nbt_len(r, min)?;
            if elem != 0 {
                for _ in 0..n {
                    nbt_value(r, elem, depth + 1)?;
                }
            }
            Nbt::List
        }
        10 => {
            let mut c = Vec::new();
            loop {
                let t = r.u8()?;
                if t == 0 {
                    break;
                }
                let name = nbt_string(r)?;
                c.push((name, nbt_value(r, t, depth + 1)?));
            }
            Nbt::Compound(c)
        }
        11 => {
            let n = nbt_len(r, 4)?;
            r.take(n * 4)?;
            Nbt::Ints
        }
        12 => {
            let n = nbt_len(r, 8)?;
            r.take(n * 8)?;
            Nbt::Ints
        }
        _ => bail!("nbt: unknown tag type {tag}"),
    })
}

/// A little-endian base-128 varint, as Sponge's BlockData stores ids.
fn varint(b: &[u8], pos: &mut usize) -> Result<u32> {
    let mut v = 0u32;
    for shift in (0..35).step_by(7) {
        let byte = *b.get(*pos).context("schematic: block data is truncated")?;
        *pos += 1;
        v |= ((byte & 0x7f) as u32) << shift;
        if byte & 0x80 == 0 {
            return Ok(v);
        }
    }
    bail!("schematic: bad varint in block data")
}

// ---------------------------------------------------------------- schematic

/// MCEdit/Schematica `.schematic` (numeric Blocks/Data) or Sponge `.schem`
/// v1/v2 (Palette + BlockData) and v3 (Schematic/Blocks/{Palette, Data}).
fn schematic(bytes: &[u8]) -> Result<Model> {
    let raw = inflate(bytes).context("schematic")?;
    let root = nbt_root(&raw).context("schematic")?;
    let s = match root.get("Schematic") {
        Some(c @ Nbt::Compound(_)) => c,
        _ => &root,
    };
    // Sponge sizes are unsigned shorts.
    let size = |k: &str| {
        let v = s.get(k).and_then(Nbt::int).with_context(|| format!("schematic: no {k}"))?;
        Ok::<i64, anyhow::Error>(if v < 0 { v & 0xffff } else { v })
    };
    let (w, h, l) = (size("Width")?, size("Height")?, size("Length")?);
    let dm = dims("schematic", w, l, h)?;
    let (w, l) = (w as usize, l as usize);
    let n = dm.x as usize * dm.y as usize * dm.z as usize;
    let pos = |i: usize| IVec3::new((i % w) as i32, (l - 1 - i / w % l) as i32, (i / (w * l)) as i32);
    let mut voxels = Vec::new();

    let (palette, data) = match s.get("Blocks") {
        Some(Nbt::Bytes(blocks)) => {
            ensure!(blocks.len() == n, "schematic: Blocks has {} entries for {n} cells", blocks.len());
            let data = match s.get("Data") {
                Some(Nbt::Bytes(d)) if d.len() == n => Some(d),
                Some(_) => bail!("schematic: Data does not match Blocks"),
                None => None,
            };
            let add = match s.get("AddBlocks") {
                Some(Nbt::Bytes(a)) => a.as_slice(),
                _ => &[],
            };
            let mut cache: HashMap<(u16, u8), Option<[u8; 4]>> = HashMap::new();
            for (i, &id) in blocks.iter().enumerate() {
                let hi = add.get(i >> 1).map_or(0, |&a| if i & 1 == 0 { a & 0x0f } else { a >> 4 });
                let id = id as u16 | (hi as u16) << 8;
                if id == 0 {
                    continue;
                }
                let dv = data.map_or(0, |d| d[i] & 15);
                let c = *cache.entry((id, dv)).or_insert_with(|| block_color(&legacy_name(id, dv)));
                if let Some(c) = c {
                    voxels.push((pos(i), c));
                }
            }
            return Ok(Model { name: String::new(), dims: dm, offset: IVec3::ZERO, voxels });
        }
        Some(b @ Nbt::Compound(_)) => (b.get("Palette"), b.get("Data")),
        Some(_) => bail!("schematic: bad Blocks tag"),
        None => (s.get("Palette"), s.get("BlockData")),
    };
    let Some(Nbt::Compound(palette)) = palette else { bail!("schematic: no block palette") };
    let Some(Nbt::Bytes(data)) = data else { bail!("schematic: no block data") };
    // Per palette id: None = not in the palette, Some(None) = air.
    let mut colors: Vec<Option<Option<[u8; 4]>>> = Vec::new();
    for (name, id) in palette {
        let id = id.int().with_context(|| format!("schematic: bad palette id for {name}"))?;
        ensure!((0..1 << 20).contains(&id), "schematic: palette id {id} is out of range");
        let id = id as usize;
        if colors.len() <= id {
            colors.resize(id + 1, None);
        }
        colors[id] = Some(block_color(name));
    }
    let mut p = 0;
    for i in 0..n {
        let id = varint(data, &mut p)? as usize;
        match colors.get(id) {
            Some(Some(Some(c))) => voxels.push((pos(i), *c)),
            Some(Some(None)) => {}
            _ => bail!("schematic: block id {id} is not in the palette"),
        }
    }
    Ok(Model { name: String::new(), dims: dm, offset: IVec3::ZERO, voxels })
}

// ---------------------------------------------------------------- block colors

const DYES: [&str; 16] = [
    "white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray", "light_gray", "cyan", "purple", "blue", "brown",
    "green", "red", "black",
];

const WOOL: [[u8; 3]; 16] = [
    [234, 236, 237], [241, 118, 20], [190, 69, 180], [58, 175, 217], [249, 198, 40], [112, 185, 26], [238, 141, 172], [63, 68, 72],
    [142, 142, 135], [21, 138, 145], [122, 42, 173], [53, 57, 157], [114, 72, 41], [85, 110, 28], [161, 39, 35], [21, 21, 26],
];

const CONCRETE: [[u8; 3]; 16] = [
    [207, 213, 214], [224, 97, 1], [169, 48, 159], [36, 137, 199], [241, 175, 21], [94, 169, 24], [214, 101, 143], [55, 58, 62],
    [125, 125, 115], [21, 119, 136], [100, 32, 156], [45, 47, 143], [96, 60, 32], [73, 91, 36], [142, 33, 33], [8, 10, 15],
];

const TERRACOTTA: [[u8; 3]; 16] = [
    [210, 178, 161], [162, 84, 38], [150, 88, 109], [113, 109, 138], [186, 133, 35], [104, 118, 53], [162, 78, 79], [58, 42, 36],
    [135, 107, 98], [87, 91, 91], [118, 70, 86], [74, 60, 91], [77, 51, 36], [76, 83, 42], [143, 61, 47], [37, 23, 16],
];

/// (wood, planks, bark, leaves).
const WOODS: &[(&str, [u8; 3], [u8; 3], [u8; 3])] = &[
    ("oak", [162, 130, 78], [109, 85, 50], [72, 140, 40]),
    ("spruce", [115, 85, 49], [58, 37, 16], [60, 95, 60]),
    ("birch", [192, 175, 121], [216, 215, 210], [100, 130, 60]),
    ("jungle", [160, 115, 81], [85, 67, 25], [55, 130, 20]),
    ("acacia", [168, 90, 50], [103, 96, 86], [70, 120, 25]),
    ("dark_oak", [67, 43, 20], [60, 46, 26], [50, 110, 25]),
    ("mangrove", [117, 54, 48], [84, 66, 41], [70, 115, 30]),
    ("cherry", [226, 178, 172], [54, 33, 44], [230, 170, 195]),
    ("pale_oak", [228, 217, 215], [87, 77, 75], [160, 165, 155]),
    ("bamboo", [194, 173, 80], [127, 144, 58], [100, 150, 50]),
    ("crimson", [101, 48, 70], [92, 25, 29], [114, 2, 2]),
    ("warped", [43, 104, 99], [58, 58, 77], [22, 119, 121]),
    ("azalea", [95, 120, 45], [95, 120, 45], [95, 120, 45]),
    ("flowering_azalea", [110, 120, 70], [110, 120, 70], [110, 120, 70]),
];

const COLORS: &[(&str, [u8; 3])] = &[
    ("stone", [125, 125, 125]),
    ("granite", [149, 103, 85]),
    ("polished_granite", [154, 106, 89]),
    ("diorite", [188, 188, 188]),
    ("polished_diorite", [192, 193, 194]),
    ("andesite", [136, 136, 136]),
    ("polished_andesite", [132, 134, 133]),
    ("deepslate", [80, 80, 82]),
    ("cobbled_deepslate", [77, 77, 80]),
    ("polished_deepslate", [72, 72, 73]),
    ("deepslate_bricks", [70, 70, 71]),
    ("deepslate_tiles", [54, 54, 55]),
    ("tuff", [108, 109, 102]),
    ("tuff_bricks", [98, 103, 95]),
    ("calcite", [223, 224, 220]),
    ("dripstone_block", [134, 107, 92]),
    ("bedrock", [85, 85, 85]),
    ("cobblestone", [127, 127, 127]),
    ("mossy_cobblestone", [110, 118, 94]),
    ("stone_bricks", [122, 121, 122]),
    ("mossy_stone_bricks", [115, 121, 105]),
    ("cracked_stone_bricks", [118, 117, 118]),
    ("chiseled_stone_bricks", [119, 118, 119]),
    ("smooth_stone", [158, 158, 158]),
    ("bricks", [150, 97, 83]),
    ("mud", [60, 57, 60]),
    ("mud_bricks", [137, 103, 79]),
    ("packed_mud", [142, 106, 79]),
    ("grass_block", [127, 178, 56]),
    ("grass", [98, 145, 52]),
    ("short_grass", [98, 145, 52]),
    ("tall_grass", [98, 145, 52]),
    ("fern", [88, 135, 50]),
    ("large_fern", [88, 135, 50]),
    ("vine", [70, 120, 30]),
    ("moss_block", [89, 109, 45]),
    ("dirt", [134, 96, 67]),
    ("coarse_dirt", [119, 85, 59]),
    ("rooted_dirt", [144, 103, 76]),
    ("podzol", [91, 63, 24]),
    ("mycelium", [111, 99, 105]),
    ("dirt_path", [148, 122, 65]),
    ("grass_path", [148, 122, 65]),
    ("farmland", [110, 75, 45]),
    ("clay", [160, 166, 179]),
    ("gravel", [131, 127, 126]),
    ("sand", [219, 207, 163]),
    ("red_sand", [190, 102, 33]),
    ("sandstone", [216, 203, 155]),
    ("chiseled_sandstone", [216, 203, 155]),
    ("cut_sandstone", [217, 206, 159]),
    ("smooth_sandstone", [223, 214, 170]),
    ("red_sandstone", [181, 98, 31]),
    ("chiseled_red_sandstone", [181, 98, 31]),
    ("cut_red_sandstone", [189, 101, 32]),
    ("smooth_red_sandstone", [181, 98, 31]),
    ("snow", [249, 254, 254]),
    ("snow_block", [249, 254, 254]),
    ("powder_snow", [248, 253, 253]),
    ("ice", [145, 183, 253]),
    ("frosted_ice", [140, 181, 252]),
    ("packed_ice", [141, 180, 250]),
    ("blue_ice", [116, 167, 253]),
    ("water", [63, 118, 228]),
    ("bubble_column", [63, 118, 228]),
    ("lava", [207, 92, 20]),
    ("magma_block", [142, 63, 31]),
    ("obsidian", [15, 10, 24]),
    ("crying_obsidian", [32, 10, 60]),
    ("glass", [175, 213, 219]),
    ("glass_pane", [175, 213, 219]),
    ("tinted_glass", [44, 38, 46]),
    ("netherrack", [97, 38, 38]),
    ("nether_bricks", [44, 21, 26]),
    ("cracked_nether_bricks", [40, 20, 23]),
    ("chiseled_nether_bricks", [47, 23, 28]),
    ("red_nether_bricks", [69, 7, 9]),
    ("nether_wart_block", [114, 2, 2]),
    ("warped_wart_block", [22, 119, 121]),
    ("shroomlight", [240, 146, 70]),
    ("crimson_nylium", [130, 31, 31]),
    ("warped_nylium", [43, 114, 101]),
    ("soul_sand", [81, 62, 50]),
    ("soul_soil", [75, 57, 46]),
    ("glowstone", [171, 131, 84]),
    ("basalt", [80, 81, 86]),
    ("polished_basalt", [99, 98, 100]),
    ("smooth_basalt", [72, 72, 78]),
    ("blackstone", [42, 36, 41]),
    ("polished_blackstone", [53, 48, 56]),
    ("polished_blackstone_bricks", [48, 42, 49]),
    ("gilded_blackstone", [56, 43, 38]),
    ("end_stone", [219, 222, 158]),
    ("end_stone_bricks", [218, 224, 162]),
    ("purpur_block", [169, 125, 169]),
    ("purpur_pillar", [171, 129, 171]),
    ("prismarine", [99, 156, 151]),
    ("prismarine_bricks", [99, 171, 158]),
    ("dark_prismarine", [51, 91, 75]),
    ("sea_lantern", [172, 199, 190]),
    ("quartz_block", [235, 229, 222]),
    ("smooth_quartz", [235, 229, 222]),
    ("chiseled_quartz_block", [231, 226, 218]),
    ("quartz_pillar", [235, 230, 224]),
    ("quartz_bricks", [234, 229, 221]),
    ("coal_ore", [105, 105, 105]),
    ("iron_ore", [136, 129, 122]),
    ("copper_ore", [124, 125, 120]),
    ("gold_ore", [145, 133, 106]),
    ("redstone_ore", [140, 109, 109]),
    ("emerald_ore", [117, 136, 124]),
    ("lapis_ore", [99, 110, 132]),
    ("diamond_ore", [121, 141, 140]),
    ("deepslate_coal_ore", [74, 74, 76]),
    ("deepslate_iron_ore", [106, 99, 94]),
    ("deepslate_copper_ore", [92, 93, 89]),
    ("deepslate_gold_ore", [115, 102, 78]),
    ("deepslate_redstone_ore", [104, 73, 74]),
    ("deepslate_emerald_ore", [78, 104, 87]),
    ("deepslate_lapis_ore", [79, 90, 115]),
    ("deepslate_diamond_ore", [83, 106, 106]),
    ("nether_gold_ore", [115, 54, 42]),
    ("nether_quartz_ore", [117, 65, 62]),
    ("ancient_debris", [94, 66, 58]),
    ("coal_block", [16, 15, 15]),
    ("iron_block", [220, 220, 220]),
    ("gold_block", [246, 208, 61]),
    ("diamond_block", [98, 237, 228]),
    ("emerald_block", [42, 203, 87]),
    ("lapis_block", [30, 67, 140]),
    ("redstone_block", [175, 24, 5]),
    ("netherite_block", [66, 61, 63]),
    ("raw_iron_block", [166, 135, 107]),
    ("raw_gold_block", [221, 169, 46]),
    ("raw_copper_block", [154, 105, 79]),
    ("amethyst_block", [133, 97, 191]),
    ("budding_amethyst", [132, 96, 186]),
    ("copper", [192, 107, 79]),
    ("copper_block", [192, 107, 79]),
    ("cut_copper", [191, 106, 80]),
    ("exposed_copper", [161, 125, 103]),
    ("exposed_cut_copper", [154, 121, 101]),
    ("weathered_copper", [108, 153, 110]),
    ("weathered_cut_copper", [109, 145, 107]),
    ("oxidized_copper", [82, 162, 132]),
    ("oxidized_cut_copper", [79, 153, 126]),
    ("terracotta", [152, 94, 68]),
    ("hay_block", [166, 136, 38]),
    ("bone_block", [229, 225, 207]),
    ("pumpkin", [198, 118, 24]),
    ("carved_pumpkin", [150, 84, 17]),
    ("jack_o_lantern", [214, 152, 52]),
    ("melon", [111, 145, 30]),
    ("cactus", [85, 127, 43]),
    ("sugar_cane", [148, 192, 101]),
    ("kelp", [87, 140, 45]),
    ("dried_kelp_block", [50, 58, 38]),
    ("bookshelf", [117, 94, 59]),
    ("crafting_table", [119, 73, 42]),
    ("furnace", [110, 110, 110]),
    ("chest", [155, 113, 48]),
    ("trapped_chest", [155, 113, 48]),
    ("barrel", [134, 100, 58]),
    ("tnt", [142, 62, 53]),
    ("sponge", [195, 192, 74]),
    ("wet_sponge", [171, 181, 70]),
    ("slime_block", [111, 192, 91]),
    ("honey_block", [251, 185, 52]),
    ("honeycomb_block", [229, 148, 29]),
    ("redstone_lamp", [95, 54, 30]),
    ("iron_bars", [136, 139, 135]),
    ("torch", [255, 210, 110]),
    ("lantern", [106, 91, 83]),
    ("brown_mushroom_block", [149, 111, 81]),
    ("red_mushroom_block", [200, 46, 45]),
    ("mushroom_stem", [203, 196, 185]),
    ("cobweb", [228, 233, 234]),
    ("ladder", [124, 97, 56]),
    ("lily_pad", [32, 128, 48]),
    ("dandelion", [255, 236, 79]),
    ("poppy", [237, 48, 44]),
    ("dead_bush", [107, 79, 41]),
    ("enchanting_table", [128, 51, 51]),
    ("beacon", [117, 220, 215]),
    ("end_portal_frame", [91, 120, 97]),
    ("note_block", [88, 58, 40]),
    ("jukebox", [93, 64, 47]),
    ("anvil", [68, 68, 68]),
    ("spawner", [36, 46, 62]),
    ("piston", [154, 132, 92]),
    ("sticky_piston", [154, 132, 92]),
    ("observer", [98, 98, 98]),
    ("dispenser", [110, 110, 110]),
    ("dropper", [110, 110, 110]),
    ("hopper", [67, 67, 67]),
    ("target", [226, 170, 158]),
    ("sculk", [12, 29, 36]),
    ("reinforced_deepslate", [80, 83, 80]),
];

/// Representative color of a block (`minecraft:` and `[states]` stripped);
/// None for air.
fn block_color(full: &str) -> Option<[u8; 4]> {
    let name = full.split('[').next().unwrap_or(full);
    let name = name.split_once(':').map_or(name, |(_, n)| n).trim().to_ascii_lowercase();
    if matches!(name.as_str(), "air" | "cave_air" | "void_air" | "structure_void" | "barrier" | "light" | "moving_piston") {
        return None;
    }
    let c = known_color(&name).unwrap_or_else(|| hashed_color(&name));
    Some([c[0], c[1], c[2], 255])
}

/// The table, then the name with shape suffixes (stairs, slabs, walls, ...)
/// or variant prefixes (waxed, polished, ...) taken off.
fn known_color(n: &str) -> Option<[u8; 3]> {
    if let Some(c) = base_color(n) {
        return Some(c);
    }
    const SHAPES: &[&str] = &[
        "_stairs", "_slab", "_wall", "_fence_gate", "_fence", "_pressure_plate", "_button", "_door", "_trapdoor", "_pane", "_carpet",
        "_pillar", "_grate", "_bulb",
    ];
    for s in SHAPES {
        if let Some(base) = n.strip_suffix(s) {
            for cand in [base.to_string(), format!("{base}s"), format!("{base}_planks"), format!("{base}_block")] {
                if let Some(c) = base_color(&cand) {
                    return Some(c);
                }
            }
            return known_color(base);
        }
    }
    for p in ["waxed_", "infested_", "potted_", "chiseled_", "cracked_", "cut_", "polished_", "smooth_", "mossy_"] {
        if let Some(c) = n.strip_prefix(p).and_then(known_color) {
            return Some(c);
        }
    }
    None
}

fn base_color(n: &str) -> Option<[u8; 3]> {
    if let Some(&(_, c)) = COLORS.iter().find(|(k, _)| *k == n) {
        return Some(c);
    }
    for (i, d) in DYES.iter().enumerate() {
        let c = match n.strip_prefix(d).and_then(|r| r.strip_prefix('_')) {
            Some("wool" | "carpet" | "bed" | "banner" | "wall_banner" | "candle" | "stained_glass" | "stained_glass_pane") => WOOL[i],
            Some("concrete" | "shulker_box" | "glazed_terracotta") => CONCRETE[i],
            Some("concrete_powder") => CONCRETE[i].map(|v| v + (255 - v) / 5),
            Some("terracotta" | "stained_hardened_clay") => TERRACOTTA[i],
            _ => continue,
        };
        return Some(c);
    }
    let (stripped, n) = n.strip_prefix("stripped_").map_or((false, n), |r| (true, r));
    for &(w, planks, bark, leaves) in WOODS {
        let c = match n.strip_prefix(w).and_then(|r| r.strip_prefix('_')) {
            Some("log" | "wood" | "stem" | "hyphae" | "block") if stripped => planks,
            Some("log" | "wood" | "stem" | "hyphae" | "block") => bark,
            Some("planks" | "mosaic" | "sign" | "wall_sign" | "hanging_sign" | "wall_hanging_sign") if !stripped => planks,
            Some("leaves" | "sapling") if !stripped => leaves,
            _ => continue,
        };
        return Some(c);
    }
    None
}

/// A stable, moderately saturated color from the name (FNV-1a).
fn hashed_color(name: &str) -> [u8; 3] {
    let mut h: u32 = 0x811c_9dc5;
    for b in name.bytes() {
        h = (h ^ b as u32).wrapping_mul(0x0100_0193);
    }
    let hue = (h % 360) as f32 / 60.0;
    let s = 0.35 + ((h >> 9) % 30) as f32 / 100.0;
    let v = 0.55 + ((h >> 17) % 30) as f32 / 100.0;
    let c = v * s;
    let x = c * (1.0 - (hue % 2.0 - 1.0).abs());
    let (r, g, b) = match hue as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r, g, b].map(|k| ((k + m) * 255.0).round() as u8)
}

/// The modern name of a pre-1.13 numeric block id with its data value.
fn legacy_name(id: u16, data: u8) -> String {
    const OLD_WOODS: [&str; 6] = ["oak", "spruce", "birch", "jungle", "acacia", "dark_oak"];
    const LATE_WOODS: [&str; 5] = ["spruce", "birch", "jungle", "dark_oak", "acacia"];
    let dye = DYES[(data & 15) as usize];
    let wood = |i: u8| OLD_WOODS.get(i as usize).copied().unwrap_or("oak");
    let s = match id {
        0 => "air",
        1 => match data {
            1 => "granite",
            2 => "polished_granite",
            3 => "diorite",
            4 => "polished_diorite",
            5 => "andesite",
            6 => "polished_andesite",
            _ => "stone",
        },
        2 => "grass_block",
        3 => match data {
            1 => "coarse_dirt",
            2 => "podzol",
            _ => "dirt",
        },
        4 => "cobblestone",
        5 | 125 => return format!("{}_planks", wood(data & 7)),
        6 => return format!("{}_sapling", wood(data & 7)),
        7 => "bedrock",
        8 | 9 => "water",
        10 | 11 => "lava",
        12 if data == 1 => "red_sand",
        12 => "sand",
        13 => "gravel",
        14 => "gold_ore",
        15 => "iron_ore",
        16 => "coal_ore",
        17 => return format!("{}_log", wood(data & 3)),
        18 => return format!("{}_leaves", wood(data & 3)),
        19 if data == 1 => "wet_sponge",
        19 => "sponge",
        20 => "glass",
        21 => "lapis_ore",
        22 => "lapis_block",
        23 => "dispenser",
        24 => match data {
            1 => "chiseled_sandstone",
            2 => "cut_sandstone",
            _ => "sandstone",
        },
        25 => "note_block",
        26 => "red_bed",
        27 => "powered_rail",
        28 => "detector_rail",
        29 => "sticky_piston",
        30 => "cobweb",
        31 if data == 2 => "fern",
        31 => "short_grass",
        32 => "dead_bush",
        33 => "piston",
        34 => "piston_head",
        35 => return format!("{dye}_wool"),
        36 => "moving_piston",
        37 => "dandelion",
        38 => "poppy",
        39 => "brown_mushroom",
        40 => "red_mushroom",
        41 => "gold_block",
        42 => "iron_block",
        43 | 44 => match data & 7 {
            1 => "sandstone",
            2 => "oak_planks",
            3 => "cobblestone",
            4 => "bricks",
            5 => "stone_bricks",
            6 => "nether_bricks",
            7 => "quartz_block",
            _ => "smooth_stone",
        },
        45 => "bricks",
        46 => "tnt",
        47 => "bookshelf",
        48 => "mossy_cobblestone",
        49 => "obsidian",
        50 => "torch",
        51 => "fire",
        52 => "spawner",
        53 => "oak_stairs",
        54 => "chest",
        55 => "redstone_wire",
        56 => "diamond_ore",
        57 => "diamond_block",
        58 => "crafting_table",
        59 => "wheat",
        60 => "farmland",
        61 | 62 => "furnace",
        63 | 68 => "oak_sign",
        64 => "oak_door",
        65 => "ladder",
        66 => "rail",
        67 => "cobblestone_stairs",
        69 => "lever",
        70 => "stone_pressure_plate",
        71 => "iron_door",
        72 => "oak_pressure_plate",
        73 | 74 => "redstone_ore",
        75 | 76 => "redstone_torch",
        77 => "stone_button",
        78 => "snow",
        79 => "ice",
        80 => "snow_block",
        81 => "cactus",
        82 => "clay",
        83 => "sugar_cane",
        84 => "jukebox",
        85 => "oak_fence",
        86 => "carved_pumpkin",
        87 => "netherrack",
        88 => "soul_sand",
        89 => "glowstone",
        90 => "nether_portal",
        91 => "jack_o_lantern",
        92 => "cake",
        93 | 94 => "repeater",
        95 => return format!("{dye}_stained_glass"),
        96 => "oak_trapdoor",
        97 => match data {
            1 => "cobblestone",
            2..=5 => "stone_bricks",
            _ => "stone",
        },
        98 => match data {
            1 => "mossy_stone_bricks",
            2 => "cracked_stone_bricks",
            3 => "chiseled_stone_bricks",
            _ => "stone_bricks",
        },
        99 => "brown_mushroom_block",
        100 => "red_mushroom_block",
        101 => "iron_bars",
        102 => "glass_pane",
        103 => "melon",
        104 => "pumpkin_stem",
        105 => "melon_stem",
        106 => "vine",
        107 => "oak_fence_gate",
        108 => "brick_stairs",
        109 => "stone_brick_stairs",
        110 => "mycelium",
        111 => "lily_pad",
        112 => "nether_bricks",
        113 => "nether_brick_fence",
        114 => "nether_brick_stairs",
        115 => "nether_wart",
        116 => "enchanting_table",
        117 => "brewing_stand",
        118 => "cauldron",
        119 => "end_portal",
        120 => "end_portal_frame",
        121 => "end_stone",
        122 => "dragon_egg",
        123 | 124 => "redstone_lamp",
        126 => return format!("{}_slab", wood(data & 7)),
        127 => "cocoa",
        128 => "sandstone_stairs",
        129 => "emerald_ore",
        130 => "ender_chest",
        131 => "tripwire_hook",
        132 => "tripwire",
        133 => "emerald_block",
        134 => "spruce_stairs",
        135 => "birch_stairs",
        136 => "jungle_stairs",
        137 => "command_block",
        138 => "beacon",
        139 if data == 1 => "mossy_cobblestone_wall",
        139 => "cobblestone_wall",
        140 => "flower_pot",
        141 => "carrots",
        142 => "potatoes",
        143 => "oak_button",
        144 => "skeleton_skull",
        145 => "anvil",
        146 => "trapped_chest",
        147 => "light_weighted_pressure_plate",
        148 => "heavy_weighted_pressure_plate",
        149 | 150 => "comparator",
        151 | 178 => "daylight_detector",
        152 => "redstone_block",
        153 => "nether_quartz_ore",
        154 => "hopper",
        155 => match data {
            1 => "chiseled_quartz_block",
            2..=4 => "quartz_pillar",
            _ => "quartz_block",
        },
        156 => "quartz_stairs",
        157 => "activator_rail",
        158 => "dropper",
        159 => return format!("{dye}_terracotta"),
        160 => return format!("{dye}_stained_glass_pane"),
        161 => return format!("{}_leaves", wood(4 + (data & 1))),
        162 => return format!("{}_log", wood(4 + (data & 1))),
        163 => "acacia_stairs",
        164 => "dark_oak_stairs",
        165 => "slime_block",
        166 => "barrier",
        167 => "iron_trapdoor",
        168 => match data {
            1 => "prismarine_bricks",
            2 => "dark_prismarine",
            _ => "prismarine",
        },
        169 => "sea_lantern",
        170 => "hay_block",
        171 => return format!("{dye}_carpet"),
        172 => "terracotta",
        173 => "coal_block",
        174 => "packed_ice",
        175 => match data & 7 {
            1 => "lilac",
            2 => "tall_grass",
            3 => "large_fern",
            4 => "rose_bush",
            5 => "peony",
            _ => "sunflower",
        },
        176 | 177 => "white_banner",
        179 | 181 => "red_sandstone",
        180 => "red_sandstone_stairs",
        182 => "red_sandstone_slab",
        183..=187 => return format!("{}_fence_gate", LATE_WOODS[id as usize - 183]),
        188..=192 => return format!("{}_fence", LATE_WOODS[id as usize - 188]),
        193 => "spruce_door",
        194 => "birch_door",
        195 => "jungle_door",
        196 => "acacia_door",
        197 => "dark_oak_door",
        198 => "end_rod",
        199 => "chorus_plant",
        200 => "chorus_flower",
        201 => "purpur_block",
        202 => "purpur_pillar",
        203 => "purpur_stairs",
        204 | 205 => "purpur_slab",
        206 => "end_stone_bricks",
        207 => "beetroots",
        208 => "dirt_path",
        209 => "end_gateway",
        210 => "repeating_command_block",
        211 => "chain_command_block",
        212 => "frosted_ice",
        213 => "magma_block",
        214 => "nether_wart_block",
        215 => "red_nether_bricks",
        216 => "bone_block",
        217 => "structure_void",
        218 => "observer",
        219..=234 => return format!("{}_shulker_box", DYES[id as usize - 219]),
        235..=250 => return format!("{}_glazed_terracotta", DYES[id as usize - 235]),
        251 => return format!("{dye}_concrete"),
        252 => return format!("{dye}_concrete_powder"),
        255 => "structure_block",
        _ => return format!("legacy_block_{id}"),
    };
    s.to_string()
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::{Format, export};
    use crate::scene::Scene;

    fn sorted(v: Vec<(IVec3, [u8; 4])>) -> Vec<([i32; 3], [u8; 4])> {
        let mut out: Vec<_> = v.into_iter().map(|(p, c)| (p.to_array(), c)).collect();
        out.sort();
        out
    }

    fn one(ext: &str, b: &[u8]) -> Model {
        let mut m = parse(ext, b).unwrap();
        assert_eq!(m.len(), 1);
        m.remove(0)
    }

    /// Every prefix and some single-byte corruptions must error, not panic.
    fn no_panics(ext: &str, b: &[u8]) {
        for n in 0..b.len() {
            let _ = parse(ext, &b[..n]);
        }
        for i in 0..b.len() {
            for v in [0x00, 0x7f, 0xff] {
                let mut c = b.to_vec();
                c[i] = v;
                let _ = parse(ext, &c);
            }
        }
    }

    // A scene like export.rs's tests use.
    fn test_scene() -> Scene {
        let mut s = Scene::empty(IVec3::new(3, 4, 5));
        let mut set = |p: IVec3, c: u8| {
            let i = s.volume.index(p);
            s.volume.set_index(i, c);
        };
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            set(IVec3::new(x, y, 0), 1);
        }
        set(IVec3::new(0, 0, 1), 2);
        set(IVec3::new(0, 3, 4), 2);
        set(IVec3::new(2, 2, 2), 3);
        s.palette[1] = [200, 30, 30, 255];
        s.palette[2] = [30, 200, 30, 255];
        s.palette[3] = [30, 30, 200, 255];
        s
    }

    fn scene_voxels(s: &Scene) -> Vec<([i32; 3], [u8; 4])> {
        sorted(s.volume.filled().map(|(p, c)| (p, s.palette[c as usize])).collect())
    }

    fn round_trip(format: Format) {
        let s = test_scene();
        let dir = std::env::temp_dir().join(format!("voxely-import-{}-{}", format.name(), std::process::id()));
        let path = &export(format, &s, &dir.join("rt.x")).unwrap()[0];
        let m = load(path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].name, "rt");
        assert_eq!(m[0].dims, s.volume.dims);
        assert_eq!(sorted(m[0].voxels.clone()), scene_voxels(&s));
    }

    #[test]
    fn qb_round_trips_through_export() {
        round_trip(Format::Qb);
    }

    #[test]
    fn xraw_round_trips_through_export() {
        round_trip(Format::Xraw);
    }

    struct QbMatrix {
        name: &'static str,
        size: [u32; 3],
        pos: [i32; 3],
        cells: Vec<[u8; 4]>,
    }

    fn qb_bytes(bgra: bool, right: bool, compressed: bool, ms: &[QbMatrix]) -> Vec<u8> {
        let mut b = vec![1, 1, 0, 0];
        for x in [bgra as u32, right as u32, compressed as u32, 0, ms.len() as u32] {
            b.extend(x.to_le_bytes());
        }
        for m in ms {
            b.push(m.name.len() as u8);
            b.extend(m.name.as_bytes());
            for s in m.size {
                b.extend(s.to_le_bytes());
            }
            for p in m.pos {
                b.extend(p.to_le_bytes());
            }
            if compressed {
                for slice in m.cells.chunks((m.size[0] * m.size[1]) as usize) {
                    let mut i = 0;
                    while i < slice.len() {
                        let run = slice[i..].iter().take_while(|&&c| c == slice[i]).count();
                        if run > 1 {
                            b.extend(2u32.to_le_bytes());
                            b.extend((run as u32).to_le_bytes());
                        }
                        b.extend(slice[i]);
                        i += run.max(1);
                    }
                    b.extend(6u32.to_le_bytes());
                }
            } else {
                for c in &m.cells {
                    b.extend(c);
                }
            }
        }
        b
    }

    fn matrices() -> Vec<QbMatrix> {
        let mut a = vec![[0; 4]; 2];
        a[1] = [255, 0, 0, 255]; // qb (1, 0, 0)
        let mut b = vec![[9, 9, 9, 0]; 6]; // alpha 0 is empty whatever the rgb
        // qb (0, 1, 2) in a 1x2x3 matrix: index x + y*1 + z*2 = 5.
        b[5] = [10, 20, 30, 255];
        vec![
            QbMatrix { name: "a", size: [2, 1, 1], pos: [0, 0, 0], cells: a },
            QbMatrix { name: "b", size: [1, 2, 3], pos: [5, 6, 7], cells: b },
        ]
    }

    #[test]
    fn qb_right_handed_matrices() {
        let m = parse("qb", &qb_bytes(false, true, false, &matrices())).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!((m[0].name.as_str(), m[0].dims, m[0].offset), ("a", IVec3::new(2, 1, 1), IVec3::new(0, -1, 0)));
        assert_eq!(m[0].voxels, vec![(IVec3::new(1, 0, 0), [255, 0, 0, 255])]);
        // qb y up -> z; right-handed qb z (2 of 3) -> y = 0; position z 7 -> y -(7+3).
        assert_eq!((m[1].name.as_str(), m[1].dims, m[1].offset), ("b", IVec3::new(1, 3, 2), IVec3::new(5, -10, 6)));
        assert_eq!(m[1].voxels, vec![(IVec3::new(0, 0, 1), [10, 20, 30, 255])]);
    }

    #[test]
    fn qb_compressed_bgra_left_handed() {
        let mut ms = matrices();
        for m in &mut ms {
            for c in &mut m.cells {
                c.swap(0, 2);
            }
        }
        let m = parse("qb", &qb_bytes(true, false, true, &ms)).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].offset, IVec3::new(0, 0, 0));
        assert_eq!(m[0].voxels, vec![(IVec3::new(1, 0, 0), [255, 0, 0, 255])]);
        assert_eq!(m[1].offset, IVec3::new(5, 7, 6));
        assert_eq!(m[1].voxels, vec![(IVec3::new(0, 2, 1), [10, 20, 30, 255])]);
        // Compressed and uncompressed agree.
        let u = parse("qb", &qb_bytes(true, false, false, &ms)).unwrap();
        assert_eq!(u[1].voxels, m[1].voxels);
    }

    #[test]
    fn qb_rejects_bad_files() {
        let good = qb_bytes(false, true, true, &matrices());
        no_panics("qb", &good);
        let huge = qb_bytes(false, true, false, &[QbMatrix { name: "", size: [5000, 1, 1], pos: [0; 3], cells: vec![] }]);
        assert!(parse("qb", &huge).unwrap_err().to_string().contains("out of range"));
        let many = qb_bytes(false, true, false, &[QbMatrix { name: "", size: [4096, 4096, 4096], pos: [0; 3], cells: vec![] }]);
        assert!(parse("qb", &many).is_err());
        assert!(parse("nope", &good).is_err());
    }

    fn xraw_header(ch: u8, bits_index: u8, d: [i32; 3], pal: i32) -> Vec<u8> {
        let mut b = b"XRAW".to_vec();
        b.extend([0, ch, 8, bits_index]);
        for x in d.into_iter().chain([pal]) {
            b.extend(x.to_le_bytes());
        }
        b
    }

    #[test]
    fn xraw_direct_and_spec_palette() {
        // Direct RGBA, 2x1x2: (1, 0, 1) is index 3.
        let mut b = xraw_header(4, 0, [2, 1, 2], 0);
        for i in 0..4 {
            b.extend(if i == 3 { [1, 2, 3, 255] } else { [7, 7, 7, 0] });
        }
        let m = one("xraw", &b);
        assert_eq!(m.dims, IVec3::new(2, 1, 2));
        assert_eq!(m.voxels, vec![(IVec3::new(1, 0, 1), [1, 2, 3, 255])]);
        no_panics("xraw", &b);

        // RGB palette with 255 = empty (the spec's convention), 1x2x1.
        let mut b = xraw_header(3, 8, [1, 2, 1], 256);
        b.extend([255, 4]);
        for i in 0..256 {
            b.extend([i as u8, 0, 100]);
        }
        let m = one("xraw", &b);
        assert_eq!(m.voxels, vec![(IVec3::new(0, 1, 0), [4, 0, 100, 255])]);
    }

    fn vxl_map(special: impl Fn(usize, usize) -> Option<Vec<u8>>) -> Vec<u8> {
        let mut b = Vec::new();
        for y in 0..512 {
            for x in 0..512 {
                match special(x, y) {
                    Some(col) => b.extend(col),
                    None => b.extend([0, 63, 63, 0, 50, 60, 70, 0]), // floor only, BGRA
                }
            }
        }
        b
    }

    #[test]
    fn vxl_columns() {
        // Column (3, 5): a span with top colors at z 10..=11 and two bottom
        // colors ending where the next span's air starts (z 20), then a last
        // span at z 30.
        let b = vxl_map(|x, y| {
            (x == 3 && y == 5).then(|| {
                let mut c = vec![5, 10, 11, 0];
                c.extend([1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 10, 11, 12, 0]);
                c.extend([0, 30, 30, 20, 13, 14, 15, 0]);
                c
            })
        });
        let m = one("vxl", &b);
        assert_eq!(m.dims, IVec3::new(512, 512, 64));
        assert_eq!(m.voxels.len(), 512 * 512 - 1 + 5);
        let at = |p: IVec3| m.voxels.iter().find(|v| v.0 == p).map(|v| v.1);
        // z down -> up, y south -> north.
        assert_eq!(at(IVec3::new(3, 511 - 5, 63 - 10)), Some([3, 2, 1, 255]));
        assert_eq!(at(IVec3::new(3, 511 - 5, 63 - 11)), Some([6, 5, 4, 255]));
        assert_eq!(at(IVec3::new(3, 511 - 5, 63 - 18)), Some([9, 8, 7, 255]));
        assert_eq!(at(IVec3::new(3, 511 - 5, 63 - 19)), Some([12, 11, 10, 255]));
        assert_eq!(at(IVec3::new(3, 511 - 5, 63 - 30)), Some([15, 14, 13, 255]));
        assert_eq!(at(IVec3::new(0, 511, 0)), Some([70, 60, 50, 255]));

        for n in [0, 7, 1000, b.len() - 1] {
            assert!(parse("vxl", &b[..n]).is_err());
        }
        let mut longer = b.clone();
        longer.push(0);
        assert!(parse("vxl", &longer).is_err());
        let mut bad = b.clone();
        bad[2] = 70; // span end past the bottom
        assert!(parse("vxl", &bad).is_err());
    }

    fn kvx_bytes() -> Vec<u8> {
        // 2x3x4 with one slab at column (1, 2): z 1..=2, colors 5 and 6.
        let mut mip = Vec::new();
        for v in [2i32, 3, 4, 0, 0, 0] {
            mip.extend(v.to_le_bytes());
        }
        for v in [28i32, 28, 33] {
            mip.extend(v.to_le_bytes());
        }
        for v in [0u16, 0, 0, 0, 0, 0, 0, 5] {
            mip.extend(v.to_le_bytes());
        }
        mip.extend([1, 2, 0, 5, 6]);
        let mut b = (mip.len() as i32).to_le_bytes().to_vec();
        b.extend(mip);
        let mut pal = vec![0u8; 768];
        pal[15..18].copy_from_slice(&[63, 0, 0]);
        pal[18..21].copy_from_slice(&[0, 63, 32]);
        b.extend(pal);
        b
    }

    #[test]
    fn kvx_slab() {
        let b = kvx_bytes();
        let m = one("kvx", &b);
        assert_eq!(m.dims, IVec3::new(2, 3, 4));
        assert_eq!(
            sorted(m.voxels),
            sorted(vec![(IVec3::new(1, 0, 2), [255, 0, 0, 255]), (IVec3::new(1, 0, 1), [0, 255, 130, 255])])
        );
        no_panics("kvx", &b);
    }

    #[test]
    fn binvox_rle() {
        // dim 2 3 4: x 2, z 3, y 4; voxel x=1, z=2, y=3 is index 1*12 + 2*4 + 3 = 23.
        let mut b = b"#binvox 1\ndim 2 3 4\ntranslate 0 0 0\nscale 1\ndata\n".to_vec();
        b.extend([0, 23, 1, 1]);
        let m = one("binvox", &b);
        assert_eq!(m.dims, IVec3::new(2, 3, 4));
        assert_eq!(m.voxels, vec![(IVec3::new(1, 0, 3), GRAY)]);
        no_panics("binvox", &b);
        let mut over = b.clone();
        *over.last_mut().unwrap() = 2;
        assert!(parse("binvox", &over).is_err());
    }

    // ---- NBT writing for tests

    fn named(b: &mut Vec<u8>, tag: u8, name: &str) {
        b.push(tag);
        b.extend((name.len() as u16).to_be_bytes());
        b.extend(name.as_bytes());
    }

    fn short(b: &mut Vec<u8>, name: &str, v: i16) {
        named(b, 2, name);
        b.extend(v.to_be_bytes());
    }

    fn int(b: &mut Vec<u8>, name: &str, v: i32) {
        named(b, 3, name);
        b.extend(v.to_be_bytes());
    }

    fn bytes(b: &mut Vec<u8>, name: &str, v: &[u8]) {
        named(b, 7, name);
        b.extend((v.len() as i32).to_be_bytes());
        b.extend(v);
    }

    fn gz(b: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        e.write_all(b).unwrap();
        e.finish().unwrap()
    }

    fn sizes(b: &mut Vec<u8>) {
        short(b, "Width", 2);
        short(b, "Height", 3);
        short(b, "Length", 4);
    }

    /// Cell index of Minecraft (x, y, z) in a 2x3x4 (WxHxL) schematic.
    fn mc(x: usize, y: usize, z: usize) -> usize {
        (y * 4 + z) * 2 + x
    }

    #[test]
    fn schematic_legacy_ids() {
        let mut blocks = vec![0u8; 24];
        let mut data = vec![0u8; 24];
        blocks[mc(1, 2, 0)] = 35; // red wool
        data[mc(1, 2, 0)] = 14;
        blocks[mc(0, 0, 3)] = 1; // stone
        blocks[mc(1, 0, 3)] = 17; // spruce log
        data[mc(1, 0, 3)] = 1 | 4;
        let mut b = Vec::new();
        named(&mut b, 10, "Schematic");
        sizes(&mut b);
        named(&mut b, 8, "Materials");
        b.extend(5u16.to_be_bytes());
        b.extend(b"Alpha");
        bytes(&mut b, "Blocks", &blocks);
        bytes(&mut b, "Data", &data);
        named(&mut b, 9, "Entities");
        b.extend([10, 0, 0, 0, 0]);
        b.push(0);
        let b = gz(&b);
        let m = one("schematic", &b);
        assert_eq!(m.dims, IVec3::new(2, 4, 3));
        // Minecraft (x, y, z) -> (x, length-1-z, y).
        assert_eq!(
            sorted(m.voxels),
            sorted(vec![
                (IVec3::new(1, 3, 2), [161, 39, 35, 255]),
                (IVec3::new(0, 0, 0), [125, 125, 125, 255]),
                (IVec3::new(1, 0, 0), [58, 37, 16, 255]),
            ])
        );
        no_panics("schematic", &b[..b.len().min(200)]);
    }

    fn palette(b: &mut Vec<u8>, entries: &[(&str, i32)]) {
        named(b, 10, "Palette");
        for (k, v) in entries {
            int(b, k, *v);
        }
        b.push(0);
    }

    const PALETTE: &[(&str, i32)] = &[("minecraft:air", 0), ("minecraft:stone", 1), ("minecraft:oak_log[axis=y]", 2), ("minecraft:white_wool", 200)];

    fn block_data() -> Vec<u8> {
        let mut ids = vec![0u32; 24];
        ids[mc(1, 2, 0)] = 200;
        ids[mc(0, 1, 3)] = 2;
        ids[mc(1, 1, 1)] = 1;
        let mut out = Vec::new();
        for mut v in ids {
            loop {
                let byte = (v & 0x7f) as u8;
                v >>= 7;
                if v == 0 {
                    out.push(byte);
                    break;
                }
                out.push(byte | 0x80);
            }
        }
        out
    }

    fn expected_sponge() -> Vec<([i32; 3], [u8; 4])> {
        sorted(vec![
            (IVec3::new(1, 3, 2), [234, 236, 237, 255]),
            (IVec3::new(0, 0, 1), [109, 85, 50, 255]),
            (IVec3::new(1, 2, 1), [125, 125, 125, 255]),
        ])
    }

    #[test]
    fn sponge_v2() {
        let mut b = Vec::new();
        named(&mut b, 10, "Schematic");
        int(&mut b, "Version", 2);
        sizes(&mut b);
        int(&mut b, "PaletteMax", 201);
        palette(&mut b, PALETTE);
        bytes(&mut b, "BlockData", &block_data());
        b.push(0);
        let m = one("schem", &gz(&b));
        assert_eq!(m.dims, IVec3::new(2, 4, 3));
        assert_eq!(sorted(m.voxels), expected_sponge());
        // Uncompressed NBT reads too.
        assert_eq!(sorted(one("schem", &b).voxels), expected_sponge());
        no_panics("schem", &b);
    }

    #[test]
    fn sponge_v3() {
        let mut b = Vec::new();
        named(&mut b, 10, "");
        named(&mut b, 10, "Schematic");
        int(&mut b, "Version", 3);
        sizes(&mut b);
        named(&mut b, 10, "Blocks");
        palette(&mut b, PALETTE);
        bytes(&mut b, "Data", &block_data());
        named(&mut b, 9, "BlockEntities");
        b.extend([10, 0, 0, 0, 0]);
        b.extend([0, 0, 0]);
        let m = one("schem", &gz(&b));
        assert_eq!(sorted(m.voxels), expected_sponge());

        // An id missing from the palette is an error.
        let mut bad = b.clone();
        let at = bad.windows(4).position(|w| w == b"Data").unwrap() + 8;
        bad[at] = 77;
        assert!(parse("schem", &bad).is_err());
    }

    #[test]
    fn block_names() {
        assert_eq!(block_color("minecraft:air"), None);
        assert_eq!(block_color("minecraft:cave_air"), None);
        assert_eq!(block_color("minecraft:stone"), Some([125, 125, 125, 255]));
        assert_eq!(block_color("minecraft:grass_block[snowy=false]"), Some([127, 178, 56, 255]));
        assert_eq!(block_color("light_blue_concrete"), Some([36, 137, 199, 255]));
        assert_eq!(block_color("blue_concrete"), Some([45, 47, 143, 255]));
        assert_eq!(block_color("minecraft:stripped_birch_log[axis=x]"), Some([192, 175, 121, 255]));
        assert_eq!(block_color("dark_oak_planks"), Some([67, 43, 20, 255]));
        assert_eq!(block_color("oak_stairs[facing=north]"), block_color("oak_planks"));
        assert_eq!(block_color("stone_brick_slab"), block_color("stone_bricks"));
        assert_eq!(block_color("quartz_stairs"), block_color("quartz_block"));
        assert_eq!(block_color("waxed_oxidized_cut_copper_stairs"), block_color("oxidized_cut_copper"));
        assert_eq!(block_color("blue_ice"), Some([116, 167, 253, 255]));
        let a = block_color("modded:mystery_block").unwrap();
        assert_eq!(block_color("modded:mystery_block"), Some(a));
        assert_ne!(block_color("mystery_block_2"), Some(a));
        // Every legacy id maps to a name that has a color (or is air).
        for id in 1..256u16 {
            for d in 0..16 {
                let n = legacy_name(id, d);
                assert!(!n.is_empty());
                let _ = block_color(&n);
            }
        }
        assert_eq!(legacy_name(5, 5), "dark_oak_planks");
        assert_eq!(legacy_name(162, 1), "dark_oak_log");
        assert_eq!(legacy_name(251, 11), "blue_concrete");
    }
}
