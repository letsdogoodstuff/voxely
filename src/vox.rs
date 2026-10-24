//! MagicaVoxel `.vox` reader (format version 150/200).
//!
//! Parses models, palette, materials (`MATL`), the scene graph
//! (`nTRN`/`nGRP`/`nSHP`), layers (`LAYR`) and render settings (`rOBJ`).

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use glam::{IVec3, Mat3};

pub type Dict = HashMap<String, String>;

#[derive(Debug, Clone)]
pub struct Model {
    pub size: IVec3,
    /// (x, y, z, color index 1..=255)
    pub voxels: Vec<[u8; 4]>,
}

#[derive(Debug, Clone)]
pub enum Node {
    Transform {
        attrs: Dict,
        child: i32,
        layer: i32,
        rotation: Mat3,
        translation: IVec3,
    },
    Group {
        attrs: Dict,
        children: Vec<i32>,
    },
    Shape {
        attrs: Dict,
        models: Vec<i32>,
    },
}

#[derive(Debug, Clone, Default)]
pub struct VoxFile {
    pub models: Vec<Model>,
    /// RGBA, indexed by voxel color index (entry 0 is unused / empty).
    pub palette: Vec<[u8; 4]>,
    /// Material dictionaries, indexed by voxel color index.
    pub materials: HashMap<u32, Dict>,
    pub nodes: HashMap<i32, Node>,
    pub layers: HashMap<i32, Dict>,
    /// Render settings keyed by their `_type` (e.g. `_inf`, `_uni`, `_lens`).
    pub render: HashMap<String, Dict>,
    /// `_type`s of the render chunks in file order.
    pub render_order: Vec<String>,
}

struct Reader<'a> {
    b: &'a [u8],
    o: usize,
}

impl<'a> Reader<'a> {
    fn i32(&mut self) -> Result<i32> {
        let s = self.b.get(self.o..self.o + 4).context("unexpected end of file")?;
        self.o += 4;
        Ok(i32::from_le_bytes(s.try_into().unwrap()))
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self.b.get(self.o..self.o + n).context("unexpected end of file")?;
        self.o += n;
        Ok(s)
    }
    fn string(&mut self) -> Result<String> {
        let n = self.i32()? as usize;
        Ok(String::from_utf8_lossy(self.bytes(n)?).into_owned())
    }
    fn dict(&mut self) -> Result<Dict> {
        let n = self.i32()?;
        let mut d = Dict::new();
        for _ in 0..n {
            let k = self.string()?;
            let v = self.string()?;
            d.insert(k, v);
        }
        Ok(d)
    }
}

/// Decode the packed rotation byte used by `nTRN` frames.
/// Bits 0-1: column of the non-zero entry in row 0, bits 2-3: row 1,
/// bits 4/5/6: sign of rows 0/1/2.
pub fn decode_rotation(r: u8) -> Mat3 {
    let i0 = (r & 3) as usize;
    let i1 = ((r >> 2) & 3) as usize;
    let i2 = 3 - i0 - i1;
    let mut rows = [[0.0f32; 3]; 3];
    rows[0][i0.min(2)] = if r & 0x10 != 0 { -1.0 } else { 1.0 };
    rows[1][i1.min(2)] = if r & 0x20 != 0 { -1.0 } else { 1.0 };
    rows[2][i2.min(2)] = if r & 0x40 != 0 { -1.0 } else { 1.0 };
    // glam is column-major; build from rows via transpose.
    Mat3::from_cols_array_2d(&rows).transpose()
}

impl VoxFile {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&bytes)
    }

    pub fn parse(b: &[u8]) -> Result<Self> {
        if b.len() < 8 || &b[0..4] != b"VOX " {
            bail!("not a .vox file");
        }
        let mut r = Reader { b, o: 8 };
        let mut f = VoxFile { palette: default_palette(), ..Default::default() };
        let mut pending_size: Option<IVec3> = None;

        // MAIN chunk header; its children follow directly.
        r.bytes(4)?;
        r.i32()?;
        r.i32()?;

        while r.o + 12 <= b.len() {
            let id: [u8; 4] = r.bytes(4)?.try_into().unwrap();
            let content = r.i32()? as usize;
            let _children = r.i32()?;
            let end = r.o + content;
            match &id {
                b"SIZE" => {
                    pending_size = Some(IVec3::new(r.i32()?, r.i32()?, r.i32()?));
                }
                b"XYZI" => {
                    let n = r.i32()? as usize;
                    let raw = r.bytes(n * 4)?;
                    let voxels = raw.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
                    let size = pending_size.take().context("XYZI without SIZE")?;
                    f.models.push(Model { size, voxels });
                }
                b"RGBA" => {
                    let raw = r.bytes(256 * 4)?;
                    let mut pal = vec![[0u8; 4]; 256];
                    // Entry k of the chunk maps to color index k + 1.
                    for (k, c) in raw.chunks_exact(4).take(255).enumerate() {
                        pal[k + 1] = [c[0], c[1], c[2], c[3]];
                    }
                    f.palette = pal;
                }
                b"MATL" => {
                    let id = r.i32()? as u32;
                    f.materials.insert(id, r.dict()?);
                }
                b"nTRN" => {
                    let node = r.i32()?;
                    let attrs = r.dict()?;
                    let child = r.i32()?;
                    let _reserved = r.i32()?;
                    let layer = r.i32()?;
                    let frames = r.i32()?;
                    let mut rotation = Mat3::IDENTITY;
                    let mut translation = IVec3::ZERO;
                    for k in 0..frames {
                        let fd = r.dict()?;
                        if k == 0 {
                            if let Some(rv) = fd.get("_r").and_then(|s| s.parse::<u8>().ok()) {
                                rotation = decode_rotation(rv);
                            }
                            if let Some(t) = fd.get("_t") {
                                let v: Vec<i32> = t.split_whitespace().filter_map(|s| s.parse().ok()).collect();
                                if v.len() == 3 {
                                    translation = IVec3::new(v[0], v[1], v[2]);
                                }
                            }
                        }
                    }
                    f.nodes.insert(node, Node::Transform { attrs, child, layer, rotation, translation });
                }
                b"nGRP" => {
                    let node = r.i32()?;
                    let attrs = r.dict()?;
                    let n = r.i32()?;
                    let children = (0..n).map(|_| r.i32()).collect::<Result<_>>()?;
                    f.nodes.insert(node, Node::Group { attrs, children });
                }
                b"nSHP" => {
                    let node = r.i32()?;
                    let attrs = r.dict()?;
                    let n = r.i32()?;
                    let mut models = Vec::new();
                    for _ in 0..n {
                        models.push(r.i32()?);
                        r.dict()?;
                    }
                    f.nodes.insert(node, Node::Shape { attrs, models });
                }
                b"LAYR" => {
                    let id = r.i32()?;
                    let attrs = r.dict()?;
                    f.layers.insert(id, attrs);
                }
                b"rOBJ" => {
                    let d = r.dict()?;
                    if let Some(t) = d.get("_type") {
                        if !f.render.contains_key(t) {
                            f.render_order.push(t.clone());
                        }
                        f.render.insert(t.clone(), d);
                    }
                }
                _ => {}
            }
            r.o = end;
        }
        Ok(f)
    }
}

fn put_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_string(out: &mut Vec<u8>, s: &str) {
    put_i32(out, s.len() as i32);
    out.extend_from_slice(s.as_bytes());
}

fn put_dict(out: &mut Vec<u8>, d: &Dict) {
    // Sorted so output is deterministic.
    let mut keys: Vec<&String> = d.keys().collect();
    keys.sort();
    put_i32(out, keys.len() as i32);
    for k in keys {
        put_string(out, k);
        put_string(out, &d[k]);
    }
}

fn put_chunk(out: &mut Vec<u8>, id: &[u8; 4], content: &[u8], children: &[u8]) {
    out.extend_from_slice(id);
    put_i32(out, content.len() as i32);
    put_i32(out, children.len() as i32);
    out.extend_from_slice(content);
    out.extend_from_slice(children);
}

/// MagicaVoxel models are at most 256 voxels on a side.
const MAX_MODEL: i32 = 256;

/// Write a dense volume as a `.vox` file, split into <= 256^3 models placed
/// by a flat scene graph. `palette` is indexed by color index.
pub fn write(
    path: &std::path::Path,
    dims: IVec3,
    cells: &[u8],
    palette: &[[u8; 4]],
    materials: &[(u32, Dict)],
    render: &[Dict],
) -> Result<()> {
    let tiles = (dims + IVec3::splat(MAX_MODEL - 1)) / MAX_MODEL;
    let mut models: Vec<(IVec3, IVec3, Vec<u8>)> = Vec::new();
    for tz in 0..tiles.z {
        for ty in 0..tiles.y {
            for tx in 0..tiles.x {
                let origin = IVec3::new(tx, ty, tz) * MAX_MODEL;
                let size = (dims - origin).min(IVec3::splat(MAX_MODEL));
                let mut xyzi = Vec::new();
                for z in 0..size.z {
                    for y in 0..size.y {
                        for x in 0..size.x {
                            let p = origin + IVec3::new(x, y, z);
                            let c = cells[(p.x + dims.x * (p.y + dims.y * p.z)) as usize];
                            if c != 0 {
                                xyzi.extend_from_slice(&[x as u8, y as u8, z as u8, c]);
                            }
                        }
                    }
                }
                // Keep at least one model so an empty canvas keeps its size.
                if !xyzi.is_empty() || (models.is_empty() && (tx, ty, tz) == (tiles.x - 1, tiles.y - 1, tiles.z - 1)) {
                    models.push((origin, size, xyzi));
                }
            }
        }
    }

    let mut body = Vec::new();
    for (_, size, xyzi) in &models {
        let mut c = Vec::new();
        put_i32(&mut c, size.x);
        put_i32(&mut c, size.y);
        put_i32(&mut c, size.z);
        put_chunk(&mut body, b"SIZE", &c, &[]);
        let mut c = Vec::new();
        put_i32(&mut c, (xyzi.len() / 4) as i32);
        c.extend_from_slice(xyzi);
        put_chunk(&mut body, b"XYZI", &c, &[]);
    }

    // Scene graph: root transform -> group -> (transform -> shape) per model.
    let empty = Dict::new();
    let transform = |out: &mut Vec<u8>, id: i32, child: i32, layer: i32, frame: &Dict| {
        let mut c = Vec::new();
        put_i32(&mut c, id);
        put_dict(&mut c, &empty);
        put_i32(&mut c, child);
        put_i32(&mut c, -1);
        put_i32(&mut c, layer);
        put_i32(&mut c, 1);
        put_dict(&mut c, frame);
        put_chunk(out, b"nTRN", &c, &[]);
    };
    transform(&mut body, 0, 1, -1, &empty);
    let mut c = Vec::new();
    put_i32(&mut c, 1);
    put_dict(&mut c, &empty);
    put_i32(&mut c, models.len() as i32);
    for k in 0..models.len() as i32 {
        put_i32(&mut c, 2 + 2 * k);
    }
    put_chunk(&mut body, b"nGRP", &c, &[]);
    for (k, (origin, size, _)) in models.iter().enumerate() {
        let k = k as i32;
        // The reader centers each model on its translation.
        let t = *origin + *size / 2;
        let frame: Dict = [("_t".to_string(), format!("{} {} {}", t.x, t.y, t.z))].into();
        transform(&mut body, 2 + 2 * k, 3 + 2 * k, 0, &frame);
        let mut c = Vec::new();
        put_i32(&mut c, 3 + 2 * k);
        put_dict(&mut c, &empty);
        put_i32(&mut c, 1);
        put_i32(&mut c, k);
        put_dict(&mut c, &empty);
        put_chunk(&mut body, b"nSHP", &c, &[]);
    }
    let mut c = Vec::new();
    put_i32(&mut c, 0);
    put_dict(&mut c, &empty);
    put_i32(&mut c, -1);
    put_chunk(&mut body, b"LAYR", &c, &[]);

    // Palette entry k holds color index k + 1.
    let mut c = Vec::with_capacity(1024);
    for k in 0..256 {
        c.extend_from_slice(&if k < 255 { palette[k + 1] } else { [0; 4] });
    }
    put_chunk(&mut body, b"RGBA", &c, &[]);
    for (id, d) in materials {
        let mut c = Vec::new();
        put_i32(&mut c, *id as i32);
        put_dict(&mut c, d);
        put_chunk(&mut body, b"MATL", &c, &[]);
    }
    for d in render {
        let mut c = Vec::new();
        put_dict(&mut c, d);
        put_chunk(&mut body, b"rOBJ", &c, &[]);
    }

    let mut out = b"VOX ".to_vec();
    put_i32(&mut out, 200);
    put_chunk(&mut out, b"MAIN", &[], &body);
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))
}

/// The default palette used when a file has no `RGBA` chunk: a 6x6x6 color
/// cube followed by red, green, blue and gray ramps.
pub fn default_palette() -> Vec<[u8; 4]> {
    let mut pal = vec![[0u8; 4]];
    let levels = [0xff, 0xcc, 0x99, 0x66, 0x33, 0x00];
    for &r in &levels {
        for &g in &levels {
            for &b in &levels {
                if r == 0 && g == 0 && b == 0 {
                    continue;
                }
                pal.push([r, g, b, 0xff]);
            }
        }
    }
    let ramp = [0xee, 0xdd, 0xbb, 0xaa, 0x88, 0x77, 0x55, 0x44, 0x22, 0x11];
    for ch in 0..4 {
        for &v in &ramp {
            let c = match ch {
                0 => [v, 0, 0, 0xff],
                1 => [0, v, 0, 0xff],
                2 => [0, 0, v, 0xff],
                _ => [v, v, v, 0xff],
            };
            pal.push(c);
        }
    }
    pal.truncate(256);
    pal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_palette_has_256_entries() {
        let p = default_palette();
        assert_eq!(p.len(), 256);
        assert_eq!(p[1], [0xff, 0xff, 0xff, 0xff]);
        assert_eq!(p[255], [0x11, 0x11, 0x11, 0xff]);
    }

    #[test]
    fn identity_rotation() {
        // 0b0000100: row0 -> col0, row1 -> col1, row2 -> col2, all positive.
        assert_eq!(decode_rotation(4), Mat3::IDENTITY);
    }
}
