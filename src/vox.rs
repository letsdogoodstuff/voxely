//! MagicaVoxel `.vox` reader (format version 150/200).
//!
//! Parses models, palette, materials (`MATL`), the scene graph
//! (`nTRN`/`nGRP`/`nSHP`), layers (`LAYR`) and render settings (`rOBJ`).

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use glam::{IVec3, Mat3};

use crate::scene::Volume;
use crate::anim::Key;
use crate::world::{Kind, Rot, World, ROOT};

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
        /// The first frame's transform.
        rotation: Mat3,
        translation: IVec3,
        /// Every frame's dictionary (`_f`, `_r`, `_t`), for animation.
        frames: Vec<Dict>,
    },
    Group {
        attrs: Dict,
        children: Vec<i32>,
    },
    Shape {
        attrs: Dict,
        models: Vec<i32>,
        /// Frame each model starts on (`_f`), for flipbook animation.
        model_frames: Vec<u32>,
    },
}

#[derive(Debug, Clone, Default)]
pub struct VoxFile {
    pub models: Vec<Model>,
    /// RGBA, indexed by voxel color index (entry 0 is unused / empty).
    pub palette: Vec<[u8; 4]>,
    /// Material dictionaries, indexed by voxel color index.
    pub materials: HashMap<u32, Dict>,
    /// Palette notes (`NOTE`): a name per palette row, row k holding color
    /// indices 8k + 1 ..= 8k + 8.
    pub notes: Vec<String>,
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

/// A `nTRN` frame's rotation and translation.
pub fn frame_transform(fd: &Dict) -> (Mat3, IVec3) {
    let rotation = fd.get("_r").and_then(|s| s.parse::<u8>().ok()).map_or(Mat3::IDENTITY, decode_rotation);
    let v: Vec<i32> = fd.get("_t").map(|t| t.split_whitespace().filter_map(|s| s.parse().ok()).collect()).unwrap_or_default();
    let translation = if v.len() == 3 { IVec3::new(v[0], v[1], v[2]) } else { IVec3::ZERO };
    (rotation, translation)
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
                b"NOTE" => {
                    let n = r.i32()?.clamp(0, 256);
                    f.notes = (0..n).map(|_| r.string()).collect::<Result<_>>()?;
                }
                b"nTRN" => {
                    let node = r.i32()?;
                    let attrs = r.dict()?;
                    let child = r.i32()?;
                    let _reserved = r.i32()?;
                    let layer = r.i32()?;
                    let frames = r.i32()?;
                    let frames: Vec<Dict> = (0..frames.clamp(0, 100_000)).map(|_| r.dict()).collect::<Result<_>>()?;
                    let (rotation, translation) = frames.first().map_or((Mat3::IDENTITY, IVec3::ZERO), frame_transform);
                    f.nodes.insert(node, Node::Transform { attrs, child, layer, rotation, translation, frames });
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
                    let (mut models, mut model_frames) = (Vec::new(), Vec::new());
                    for k in 0..n {
                        models.push(r.i32()?);
                        let d = r.dict()?;
                        model_frames.push(d.get("_f").and_then(|v| v.trim().parse().ok()).unwrap_or(k as u32));
                    }
                    f.nodes.insert(node, Node::Shape { attrs, models, model_frames });
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

/// voxely's timeline settings, in a render chunk MagicaVoxel ignores.
pub const ANIM_TYPE: &str = "_vx_anim";

/// MagicaVoxel models are at most 256 voxels on a side.
const MAX_MODEL: i32 = 256;

/// Builds the scene graph chunks: node ids are handed out in pre-order and
/// every group or shape is wrapped in a transform, as MagicaVoxel expects.
struct Graph {
    next: i32,
    nodes: Vec<(i32, Vec<u8>)>,
    models: Vec<(IVec3, Vec<u8>)>,
}

impl Graph {
    fn id(&mut self) -> i32 {
        self.next += 1;
        self.next - 1
    }

    #[allow(clippy::too_many_arguments)]
    fn transform(&mut self, id: i32, child: i32, name: &str, hidden: bool, layer: i32, rot: u8, t: IVec3) {
        self.transform_with(id, child, name, hidden, layer, rot, t, Dict::new(), &[]);
    }

    #[allow(clippy::too_many_arguments)]
    /// `keys` (with their translations already offset) replace the single
    /// frame when the node is animated.
    fn transform_with(&mut self, id: i32, child: i32, name: &str, hidden: bool, layer: i32, rot: u8, t: IVec3, mut attrs: Dict, keys: &[Key]) {
        if !name.is_empty() {
            attrs.insert("_name".into(), name.into());
        }
        if hidden {
            attrs.insert("_hidden".into(), "1".into());
        }
        let frame = |rot: u8, t: IVec3| {
            let mut frame = Dict::new();
            if rot != Rot::IDENTITY.to_byte() {
                frame.insert("_r".into(), rot.to_string());
            }
            if t != IVec3::ZERO {
                frame.insert("_t".into(), format!("{} {} {}", t.x, t.y, t.z));
            }
            frame
        };
        let frames: Vec<Dict> = if keys.is_empty() {
            vec![frame(rot, t)]
        } else {
            keys.iter()
                .map(|k| {
                    let mut d = frame(k.rot.to_byte(), k.t);
                    d.insert("_f".into(), k.frame.to_string());
                    if k.tilt != glam::Vec3::ZERO {
                        d.insert("_vx_tilt".into(), format!("{} {} {}", k.tilt.x, k.tilt.y, k.tilt.z));
                    }
                    d
                })
                .collect()
        };
        let mut c = Vec::new();
        put_i32(&mut c, id);
        put_dict(&mut c, &attrs);
        put_i32(&mut c, child);
        put_i32(&mut c, -1);
        put_i32(&mut c, layer);
        put_i32(&mut c, frames.len() as i32);
        for f in &frames {
            put_dict(&mut c, f);
        }
        self.nodes.push((id, chunk(b"nTRN", &c)));
    }

    fn group(&mut self, id: i32, children: &[i32]) {
        let mut c = Vec::new();
        put_i32(&mut c, id);
        put_dict(&mut c, &Dict::new());
        put_i32(&mut c, children.len() as i32);
        for &k in children {
            put_i32(&mut c, k);
        }
        self.nodes.push((id, chunk(b"nGRP", &c)));
    }

    fn shape(&mut self, id: i32, v: &Volume, origin: IVec3, size: IVec3) {
        self.shape_frames(id, &[(0, v)], origin, size);
    }

    /// A shape showing each (first frame, voxels) model in turn.
    fn shape_frames(&mut self, id: i32, frames: &[(u32, &Volume)], origin: IVec3, size: IVec3) {
        let mut c = Vec::new();
        put_i32(&mut c, id);
        put_dict(&mut c, &Dict::new());
        put_i32(&mut c, frames.len() as i32);
        for &(frame, v) in frames {
            put_i32(&mut c, self.models.len() as i32);
            let mut d = Dict::new();
            if frames.len() > 1 {
                d.insert("_f".into(), frame.to_string());
            }
            put_dict(&mut c, &d);
            self.model_data(v, origin, size);
        }
        self.nodes.push((id, chunk(b"nSHP", &c)));
    }

    fn model_data(&mut self, v: &Volume, origin: IVec3, size: IVec3) {
        let mut xyzi = Vec::new();
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let c = v.get(origin + IVec3::new(x, y, z));
                    if c != 0 {
                        xyzi.extend_from_slice(&[x as u8, y as u8, z as u8, c]);
                    }
                }
            }
        }
        self.models.push((size, xyzi));
    }

    /// Write world node `n` as transform `trn` (already allocated), its
    /// translation shifted by `offset` (children of parts are stored in the
    /// part's voxel space, `.vox` groups are centered).
    fn node(&mut self, w: &World, n: usize, trn: i32, offset: IVec3) {
        let Some(node) = w.node(n) else { return };
        let child = self.id();
        let (name, rot, layer, t) = (node.name.as_str(), node.rot.to_byte(), node.layer, node.t + offset);
        let mut extras = Dict::new();
        crate::world::write_extras(node, &mut extras);
        let keys: Vec<Key> = node.keys.iter().map(|k| Key { t: k.t + offset, ..*k }).collect();
        match &node.kind {
            Kind::Group => {
                self.transform_with(trn, child, name, node.hidden, layer, rot, t, extras, &keys);
                self.children(w, child, &[], &node.children, IVec3::ZERO);
            }
            Kind::Model(v) if node.children.is_empty() => {
                self.transform_with(trn, child, name, node.hidden, layer, rot, t, extras, &keys);
                self.model(child, node, v, layer);
            }
            Kind::Model(v) => {
                // A part with attached parts: a group holding the part and
                // them, marked so voxely reads it back as one part.
                extras.insert("_vx_part".into(), "1".into());
                self.transform_with(trn, child, name, node.hidden, layer, rot, t, extras, &keys);
                let (shape_trn, shape) = (self.id(), self.id());
                self.children(w, child, &[shape_trn], &node.children, -(v.dims / 2));
                self.transform(shape_trn, shape, "", false, layer, Rot::IDENTITY.to_byte(), IVec3::ZERO);
                self.model(shape, node, v, layer);
            }
        }
    }

    /// Group `id` holding the transforms `first` plus world nodes `children`.
    fn children(&mut self, w: &World, id: i32, first: &[i32], children: &[usize], offset: IVec3) {
        let ids: Vec<i32> = children.iter().map(|_| self.id()).collect();
        self.group(id, &[first, &ids].concat());
        for (&c, &cid) in children.iter().zip(&ids) {
            self.node(w, c, cid, offset);
        }
    }

    /// A model's voxels as shape `id`, or a group of tiles when it is over
    /// 256 on a side (landing on the same cells).
    fn model(&mut self, id: i32, node: &crate::world::Node, v: &Volume, layer: i32) {
        let frames = node.flip_volumes();
        // Flipbook models share a size in `.vox`; the largest covers them.
        let size = frames.iter().fold(v.dims, |a, (_, f)| a.max(f.dims));
        if size.cmple(IVec3::splat(MAX_MODEL)).all() {
            return self.shape_frames(id, &frames, IVec3::ZERO, size);
        }
        let tiles = (v.dims + IVec3::splat(MAX_MODEL - 1)) / MAX_MODEL;
        let mut placed = Vec::new();
        for tz in 0..tiles.z {
            for ty in 0..tiles.y {
                for tx in 0..tiles.x {
                    let o = IVec3::new(tx, ty, tz) * MAX_MODEL;
                    placed.push((o, (v.dims - o).min(IVec3::splat(MAX_MODEL))));
                }
            }
        }
        let ids: Vec<i32> = placed.iter().map(|_| self.id()).collect();
        self.group(id, &ids);
        for (&(o, size), &tid) in placed.iter().zip(&ids) {
            let shape = self.id();
            self.transform(tid, shape, "", false, layer, Rot::IDENTITY.to_byte(), o + size / 2 - v.dims / 2);
            self.shape(shape, v, o, size);
        }
    }
}

fn chunk(id: &[u8; 4], content: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    put_chunk(&mut out, id, content, &[]);
    out
}

/// Write the scene tree as a `.vox` file. `palette` is indexed by color
/// index. Models over 256 on a side are split into tiles.
pub fn write(
    path: &std::path::Path,
    world: &World,
    palette: &[[u8; 4]],
    materials: &[(u32, Dict)],
    notes: &[String],
    render: &[Dict],
) -> Result<()> {
    let mut g = Graph { next: 0, nodes: Vec::new(), models: Vec::new() };
    // Root transform 0 -> group 1 -> a transform per top-level node.
    let (root, top) = (g.id(), g.id());
    g.transform(root, top, "", false, -1, Rot::IDENTITY.to_byte(), IVec3::ZERO);
    let children = world.node(ROOT).map(|n| n.children().to_vec()).unwrap_or_default();
    let ids: Vec<i32> = children.iter().map(|_| g.id()).collect();
    g.group(top, &ids);
    for (&c, &id) in children.iter().zip(&ids) {
        g.node(world, c, id, IVec3::ZERO);
    }
    g.nodes.sort_by_key(|(id, _)| *id);

    let mut body = Vec::new();
    for (size, xyzi) in &g.models {
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
    for (_, bytes) in &g.nodes {
        body.extend_from_slice(bytes);
    }
    for l in &world.layers {
        let mut attrs = Dict::new();
        if !l.name.is_empty() {
            attrs.insert("_name".into(), l.name.clone());
        }
        if l.hidden {
            attrs.insert("_hidden".into(), "1".into());
        }
        let mut c = Vec::new();
        put_i32(&mut c, l.id);
        put_dict(&mut c, &attrs);
        put_i32(&mut c, -1);
        put_chunk(&mut body, b"LAYR", &c, &[]);
    }

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
    if notes.iter().any(|n| !n.is_empty()) {
        let mut c = Vec::new();
        put_i32(&mut c, notes.len() as i32);
        for n in notes {
            put_string(&mut c, n);
        }
        put_chunk(&mut body, b"NOTE", &c, &[]);
    }
    for d in render.iter().filter(|d| d.get("_type").is_none_or(|t| t != ANIM_TYPE)) {
        let mut c = Vec::new();
        put_dict(&mut c, d);
        put_chunk(&mut body, b"rOBJ", &c, &[]);
    }
    if world.animated() {
        let a = &world.anim;
        let d: Dict = [
            ("_type", ANIM_TYPE.to_string()),
            ("_frames", a.frames.to_string()),
            ("_fps", a.fps.to_string()),
            ("_loop", (a.looping as u8).to_string()),
            ("_smooth", (a.smooth as u8).to_string()),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let mut c = Vec::new();
        put_dict(&mut c, &d);
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
