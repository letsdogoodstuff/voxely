//! Import animation clips from glTF (`.glb` / `.gltf`) onto a voxel rig:
//! free libraries like Quaternius's Universal Animation Library, clips
//! exported from Blender, or voxely's own exports.
//!
//! Retargeting: the source's joints are matched to the rig's parts by name
//! (`rig::bone_of`). For each bone, the source's turn away from its rest
//! pose is applied to the part, after first lining up the two rest poses
//! (so a T-pose clip drives a rig whose arms hang down). The hips also
//! carry the clip's movement, scaled to the rig's size. Parts stay rigid.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use glam::{Mat3, Mat4, Quat, Vec3};
use serde_json::Value;

use crate::anim::{tilt_deg, Key};
use crate::rig::{map_bones, Bone};
use crate::world::{rot_mat, World};

/// A parsed glTF file: the JSON and its buffers.
pub struct Gltf {
    doc: Value,
    buffers: Vec<Vec<u8>>,
}

/// One clip: name and length in seconds.
#[derive(Clone, Debug)]
pub struct Clip {
    pub name: String,
    pub duration: f32,
}

/// A node's local transform.
#[derive(Clone, Copy, Debug)]
struct Trs {
    t: Vec3,
    r: Quat,
    s: Vec3,
}

impl Trs {
    fn mat(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.s, self.r, self.t)
    }
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect()).unwrap_or_default()
}

impl Gltf {
    pub fn load(path: &Path) -> Result<Gltf> {
        let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let (doc, bin): (Value, Option<Vec<u8>>) = if data.starts_with(b"glTF") {
            let u32_at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap()) as usize;
            if data.len() < 20 {
                bail!("truncated .glb");
            }
            let jlen = u32_at(12);
            let doc = serde_json::from_slice(data.get(20..20 + jlen).context("truncated .glb")?)?;
            let b = 20 + jlen;
            let bin = (data.len() >= b + 8 && &data[b + 4..b + 8] == b"BIN\0").then(|| data[b + 8..(b + 8 + u32_at(b)).min(data.len())].to_vec());
            (doc, bin)
        } else {
            (serde_json::from_slice(&data).context("not glTF")?, None)
        };
        let dir = path.parent().unwrap_or(Path::new("."));
        let mut buffers = Vec::new();
        let mut bin = bin;
        for (k, b) in doc["buffers"].as_array().into_iter().flatten().enumerate() {
            let bytes = match b["uri"].as_str() {
                None if k == 0 => bin.take().context("missing GLB binary chunk")?,
                None => bail!("buffer {k} has no data"),
                Some(uri) if uri.starts_with("data:") => {
                    use base64::Engine;
                    let b64 = uri.split_once(',').map_or("", |x| x.1);
                    base64::engine::general_purpose::STANDARD.decode(b64).context("bad data URI")?
                }
                Some(uri) => std::fs::read(dir.join(uri)).with_context(|| format!("reading buffer {uri}"))?,
            };
            buffers.push(bytes);
        }
        Ok(Gltf { doc, buffers })
    }

    /// An accessor's values as floats (normalized integers scaled to -1..1
    /// or 0..1), flattened.
    fn accessor(&self, index: usize) -> Result<Vec<f32>> {
        let a = &self.doc["accessors"][index];
        let count = a["count"].as_u64().context("accessor count")? as usize;
        let comps = match a["type"].as_str() {
            Some("SCALAR") => 1,
            Some("VEC2") => 2,
            Some("VEC3") => 3,
            Some("VEC4") => 4,
            Some("MAT4") => 16,
            t => bail!("accessor type {t:?}"),
        };
        let ct = a["componentType"].as_u64().unwrap_or(5126);
        let size = match ct {
            5126 | 5125 => 4,
            5122 | 5123 => 2,
            _ => 1,
        };
        let Some(view) = a["bufferView"].as_u64() else { return Ok(vec![0.0; count * comps]) };
        let v = &self.doc["bufferViews"][view as usize];
        let buf = self.buffers.get(v["buffer"].as_u64().unwrap_or(0) as usize).context("missing buffer")?;
        let base = v["byteOffset"].as_u64().unwrap_or(0) as usize + a["byteOffset"].as_u64().unwrap_or(0) as usize;
        let stride = v["byteStride"].as_u64().map_or(comps * size, |s| s as usize);
        let mut out = Vec::with_capacity(count * comps);
        for i in 0..count {
            for c in 0..comps {
                let o = base + i * stride + c * size;
                let b = buf.get(o..o + size).context("accessor out of range")?;
                out.push(match ct {
                    5126 => f32::from_le_bytes(b.try_into().unwrap()),
                    5125 => u32::from_le_bytes(b.try_into().unwrap()) as f32,
                    5122 => (i16::from_le_bytes(b.try_into().unwrap()) as f32 / 32767.0).max(-1.0),
                    5123 => u16::from_le_bytes(b.try_into().unwrap()) as f32 / 65535.0,
                    5120 => (b[0] as i8 as f32 / 127.0).max(-1.0),
                    _ => b[0] as f32 / 255.0,
                });
            }
        }
        Ok(out)
    }

    pub fn clips(&self) -> Vec<Clip> {
        self.doc["animations"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
            .map(|(k, a)| {
                let duration = a["samplers"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|s| self.doc["accessors"][s["input"].as_u64()? as usize]["max"][0].as_f64())
                    .fold(0.0f64, f64::max) as f32;
                Clip { name: a["name"].as_str().map_or_else(|| format!("clip {k}"), str::to_string), duration }
            })
            .collect()
    }

    fn rest(&self, node: &Value) -> Trs {
        if let Some(m) = node.get("matrix") {
            let (s, r, t) = Mat4::from_cols_slice(&floats(m)).to_scale_rotation_translation();
            return Trs { t, r, s };
        }
        let f = |k: &str, d: &[f32]| node.get(k).map_or(d.to_vec(), floats);
        let (t, r, s) = (f("translation", &[0.0; 3]), f("rotation", &[0.0, 0.0, 0.0, 1.0]), f("scale", &[1.0; 3]));
        Trs { t: Vec3::from_slice(&t), r: Quat::from_slice(&r).normalize(), s: Vec3::from_slice(&s) }
    }
}

/// A channel: node, which property, keyframe times and values.
struct Channel {
    node: usize,
    path: String,
    times: Vec<f32>,
    values: Vec<f32>,
    step: bool,
    cubic: bool,
}

impl Channel {
    /// The value at time `t` (`n` components).
    fn sample(&self, t: f32, n: usize) -> Vec<f32> {
        let k = self.times.len();
        let at = |i: usize| -> &[f32] {
            // Cubic splines store (in-tangent, value, out-tangent).
            let i = if self.cubic { i * 3 + 1 } else { i };
            &self.values[i * n..(i + 1) * n]
        };
        if k == 0 {
            return vec![0.0; n];
        }
        if t <= self.times[0] {
            return at(0).to_vec();
        }
        if t >= self.times[k - 1] {
            return at(k - 1).to_vec();
        }
        let i = self.times.partition_point(|&x| x <= t).max(1) - 1;
        let (a, b) = (at(i), at(i + 1));
        if self.step {
            return a.to_vec();
        }
        let s = (t - self.times[i]) / (self.times[i + 1] - self.times[i]).max(1e-6);
        if n == 4 {
            let (qa, qb) = (Quat::from_slice(a), Quat::from_slice(b));
            return qa.slerp(qb, s).to_array().to_vec();
        }
        a.iter().zip(b).map(|(x, y)| x + (y - x) * s).collect()
    }
}

/// Rest poses are lined up for bones whose rest directions differ by more
/// than this (radians).
const ALIGN_ABOVE: f32 = 0.45;

/// glTF (y up) to voxel space (z up): (x, y, z) -> (x, -z, y).
fn from_gltf() -> Mat3 {
    Mat3::from_cols(Vec3::X, Vec3::Z, -Vec3::Y)
}

fn rot3(m: Mat4) -> Mat3 {
    let (_, r, _) = m.to_scale_rotation_translation();
    Mat3::from_quat(r)
}

/// What an import did.
pub struct Imported {
    pub clip: String,
    pub frames: u32,
    pub bones: Vec<(Bone, String, String)>,
    pub missing: Vec<Bone>,
}

/// Import `clip` (name or index; default the first) onto the rig `rig` (any
/// part of it; default the first humanoid / four-legged rig), sampled at the
/// timeline's fps. Replaces the matched parts' keys and sets the timeline
/// to the clip's length.
pub fn import(w: &mut World, g: &Gltf, clip: Option<&str>, rig: Option<usize>, root_motion: bool) -> Result<Imported> {
    let clips = g.clips();
    if clips.is_empty() {
        bail!("no animations in this file");
    }
    let ci = match clip {
        None => 0,
        Some(c) => clips
            .iter()
            .position(|x| x.name.eq_ignore_ascii_case(c))
            .or_else(|| c.parse::<usize>().ok().filter(|&i| i < clips.len()))
            .with_context(|| format!("no clip `{c}`; clips: {}", clips.iter().map(|c| c.name.as_str()).collect::<Vec<_>>().join(", ")))?,
    };
    let anim = &g.doc["animations"][ci];
    let mut channels = Vec::new();
    for c in anim["channels"].as_array().into_iter().flatten() {
        let (Some(node), Some(path)) = (c["target"]["node"].as_u64(), c["target"]["path"].as_str()) else { continue };
        let s = &anim["samplers"][c["sampler"].as_u64().unwrap_or(0) as usize];
        let interp = s["interpolation"].as_str().unwrap_or("LINEAR");
        channels.push(Channel {
            node: node as usize,
            path: path.to_string(),
            times: g.accessor(s["input"].as_u64().context("sampler input")? as usize)?,
            values: g.accessor(s["output"].as_u64().context("sampler output")? as usize)?,
            step: interp == "STEP",
            cubic: interp == "CUBICSPLINE",
        });
    }

    // Source nodes: rest transforms, parents, names.
    let nodes = g.doc["nodes"].as_array().cloned().unwrap_or_default();
    let rest: Vec<Trs> = nodes.iter().map(|n| g.rest(n)).collect();
    let mut parent = vec![None; nodes.len()];
    for (i, n) in nodes.iter().enumerate() {
        for c in n["children"].as_array().into_iter().flatten().filter_map(Value::as_u64) {
            if let Some(p) = parent.get_mut(c as usize) {
                *p = Some(i);
            }
        }
    }
    let global = |locals: &[Trs], i: usize| -> Mat4 {
        let mut m = locals[i].mat();
        let mut cur = parent[i];
        while let Some(p) = cur {
            m = locals[p].mat() * m;
            cur = parent[p];
        }
        m
    };
    // Joints: a skin's, or every node; matched to bones by name, in
    // hierarchy order.
    let skin = g.doc["skins"].get(0);
    let joints: Vec<usize> = match skin {
        Some(s) => s["joints"].as_array().into_iter().flatten().filter_map(|j| j.as_u64().map(|j| j as usize)).collect(),
        None => (0..nodes.len()).collect(),
    };
    let depth = |mut i: usize| {
        let mut d = 0;
        while let Some(p) = parent[i] {
            d += 1;
            i = p;
        }
        d
    };
    let mut ordered = joints.clone();
    ordered.sort_by_key(|&j| depth(j));
    let named: Vec<(usize, String)> = ordered.iter().filter_map(|&j| Some((j, nodes[j]["name"].as_str()?.to_string()))).collect();
    let src = map_bones(&named);

    // Rest globals: from the skin's bind matrices when there is a skin.
    let mut rest_global: HashMap<usize, Mat4> = HashMap::new();
    for &j in &joints {
        rest_global.insert(j, global(&rest, j));
    }
    if let Some(s) = skin {
        if let Some(ibm) = s["inverseBindMatrices"].as_u64() {
            let m = g.accessor(ibm as usize)?;
            let mesh_node = nodes.iter().position(|n| n["skin"].as_u64() == Some(0));
            let base = mesh_node.map_or(Mat4::IDENTITY, |n| global(&rest, n));
            for (k, &j) in joints.iter().enumerate() {
                if let Some(cols) = m.get(k * 16..k * 16 + 16) {
                    rest_global.insert(j, base * Mat4::from_cols_slice(cols).inverse());
                }
            }
        }
    }

    // The rig.
    let quad = src.contains_key(&Bone::LegFL);
    let root = crate::rig::rig_root(w, rig, if quad { Bone::LegFL } else { Bone::UpperLegL });
    let dst = crate::rig::find_bones(w, root);
    let pairs: Vec<(Bone, usize, usize)> = {
        let mut v: Vec<(Bone, usize, usize)> = src.iter().filter_map(|(b, &s)| Some((*b, s, *dst.get(b)?))).collect();
        // Parents before children, so each part's parent is posed first.
        v.sort_by_key(|&(_, _, d)| {
            let mut n = 0;
            let mut cur = Some(d);
            while let Some(c) = cur.and_then(|c| w.node(c)).and_then(|n| n.parent) {
                n += 1;
                cur = Some(c);
            }
            n
        });
        v
    };
    if pairs.is_empty() {
        bail!("no joints matched: the clip's joints are {:?}; the rig needs parts named like hips, torso, head, upper_arm_l ...", named.iter().take(12).map(|n| n.1.as_str()).collect::<Vec<_>>());
    }
    let missing: Vec<Bone> = dst.keys().filter(|b| !src.contains_key(b)).copied().collect();

    // Line up rest poses: rotate each part's rest direction onto the
    // source bone's rest direction.
    let c = from_gltf();
    w.set_frame(0);
    // A bone's direction points at the next bone in its chain (torso to
    // head, upper arm to forearm ...), else at its first joint child that
    // isn't an attachment point.
    let bone_at: HashMap<usize, Bone> = src.iter().map(|(&b, &j)| (j, b)).collect();
    let next = |b: Bone| -> &[Bone] {
        use Bone::*;
        match b {
            Hips => &[Torso],
            Torso => &[Torso, Head],
            UpperArmL => &[LowerArmL],
            UpperArmR => &[LowerArmR],
            UpperLegL => &[LowerLegL],
            UpperLegR => &[LowerLegR],
            _ => &[],
        }
    };
    let src_dir = |j: usize| -> Option<Vec3> {
        let p = rest_global[&j].w_axis.truncate();
        let kids: Vec<usize> = nodes[j]["children"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|x| x.as_u64())
            .map(|x| x as usize)
            .filter(|x| rest_global.contains_key(x) && nodes[*x]["extras"]["voxely_socket"].as_bool() != Some(true))
            .collect();
        let chain = bone_at.get(&j).map_or(&[][..], |&b| next(b));
        let child = kids.iter().copied().find(|k| bone_at.get(k).is_some_and(|b| chain.contains(b))).or_else(|| kids.first().copied())?;
        let d = c * (rest_global[&child].w_axis.truncate() - p);
        (d.length() > 1e-4).then(|| d.normalize())
    };
    let dst_dir = |w: &World, id: usize| -> Option<Vec3> {
        let v = w.node(id)?.volume()?;
        let a = w.affine(id);
        let d = a.transform_point3(v.dims.as_vec3() * 0.5) - w.pivot_world(id);
        (d.length() > 0.5).then(|| d.normalize())
    };
    let align: HashMap<usize, Mat3> = pairs
        .iter()
        .map(|&(_, s, d)| {
            // Only big differences (T-pose arms on an arms-down rig); a
            // source's natural spine curve or knee bend isn't copied.
            let q = match (src_dir(s), dst_dir(w, d)) {
                (Some(a), Some(b)) if a.dot(b) < ALIGN_ABOVE.cos() => Quat::from_rotation_arc(b, a),
                _ => Quat::IDENTITY,
            };
            (d, Mat3::from_quat(q))
        })
        .collect();
    let target_rest: HashMap<usize, Mat3> = pairs.iter().map(|&(_, _, d)| (d, Mat3::from(w.affine(d).matrix3))).collect();
    let rest_keys: HashMap<usize, Key> = pairs.iter().filter_map(|&(_, _, d)| Some((d, w.node(d)?.key_now(0)))).collect();

    // Hips movement, scaled from the clip's hip height to the rig's.
    let hips = pairs.iter().find(|p| p.0 == Bone::Hips).map(|p| (p.1, p.2));
    let scale = hips.map_or(1.0, |(s, d)| {
        let src_h = (c * rest_global[&s].w_axis.truncate()).z;
        let dst_h = w.pivot_world(d).z - w.bounds(root).map_or(0.0, |b| b.0.z as f32);
        if src_h > 1e-3 { dst_h / src_h } else { 1.0 }
    });

    let clip_info = &clips[ci];
    let fps = w.anim.fps.max(1.0);
    // Samples from the first key to the last, inclusive.
    let frames = ((clip_info.duration * fps).round() as u32 + 1).max(2);
    let mut posed = w.clone();
    let mut keys: HashMap<usize, Vec<Key>> = HashMap::new();
    for f in 0..frames {
        let t = f as f32 / fps;
        let mut locals = rest.clone();
        for ch in &channels {
            let Some(l) = locals.get_mut(ch.node) else { continue };
            match ch.path.as_str() {
                "translation" => l.t = Vec3::from_slice(&ch.sample(t, 3)),
                "rotation" => l.r = Quat::from_slice(&ch.sample(t, 4)).normalize(),
                "scale" => l.s = Vec3::from_slice(&ch.sample(t, 3)),
                _ => {}
            }
        }
        for &(bone, s, d) in &pairs {
            let now = global(&locals, s);
            // The source's turn from rest, in voxel space.
            let delta = c * (rot3(now) * rot3(rest_global[&s]).transpose()) * c.transpose();
            let want = delta * align[&d] * target_rest[&d];
            let Some(n) = posed.node(d) else { continue };
            let parent_rot = n.parent.map_or(Mat3::IDENTITY, |p| Mat3::from(posed.affine(p).matrix3));
            let r = rest_keys[&d];
            let local = parent_rot.inverse() * want;
            let tilt = tilt_deg(Quat::from_mat3(&(rot_mat(r.rot).transpose() * local)).normalize());
            let mut k = Key { frame: f, rot: r.rot, t: r.t, tilt };
            if bone == Bone::Hips && root_motion {
                let moved = c * (now.w_axis.truncate() - rest_global[&s].w_axis.truncate()) * scale;
                let inv = n.parent.map_or(Mat3::IDENTITY, |p| Mat3::from(posed.affine(p).matrix3).inverse());
                k.t = r.t + (inv * moved).round().as_ivec3();
            }
            if let Some(n) = posed.node_mut(d) {
                (n.t, n.tilt) = (k.t, k.tilt);
            }
            keys.entry(d).or_default().push(k);
        }
    }
    for (d, list) in keys {
        if let Some(n) = w.node_mut(d) {
            n.keys = list;
        }
    }
    w.anim.frames = frames;
    w.anim.smooth = true;
    w.set_frame(0);
    Ok(Imported {
        clip: clip_info.name.clone(),
        frames,
        bones: pairs.iter().map(|&(b, s, d)| (b, nodes[s]["name"].as_str().unwrap_or("").to_string(), w.label(d))).collect(),
        missing,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rig::{find_bones, humanoid};
    use crate::scene::{Scene, Volume};
    use crate::world::ROOT;
    use glam::IVec3;

    fn rig_world() -> World {
        let mut w = World::single(Volume::new(IVec3::ONE));
        let first = w.models()[0];
        w.delete(first);
        humanoid(&mut w, IVec3::ZERO, 1, "", (1, 2, 3));
        w
    }

    /// Bone directions (pivot to part center) on each frame.
    fn directions(w: &mut World, frames: u32) -> Vec<HashMap<Bone, Vec3>> {
        let bones = find_bones(w, ROOT);
        (0..frames)
            .map(|f| {
                w.set_frame(f);
                bones
                    .iter()
                    .filter_map(|(&b, &id)| {
                        let v = w.node(id)?.volume()?;
                        let d = w.affine(id).transform_point3(v.dims.as_vec3() * 0.5) - w.pivot_world(id);
                        (d.length() > 0.5).then(|| (b, d.normalize()))
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn voxely_clip_round_trips_onto_a_bigger_rig() {
        // A clip whose first frame is the rest pose: export it...
        let mut s = Scene::empty(IVec3::ONE);
        s.world = rig_world();
        s.world.anim = crate::anim::Anim { frames: 12, fps: 12.0, looping: true, smooth: true };
        let bones = find_bones(&s.world, ROOT);
        for (bone, axis) in [(Bone::UpperArmR, Vec3::new(-70.0, 0.0, 0.0)), (Bone::UpperLegL, Vec3::new(-30.0, 0.0, 0.0)), (Bone::Head, Vec3::new(0.0, 0.0, 25.0)), (Bone::LowerArmR, Vec3::new(-40.0, 0.0, 0.0))] {
            let id = bones[&bone];
            s.world.set_frame(0);
            s.world.key(id);
            s.world.set_frame(6);
            s.world.set_tilt(id, axis);
            s.world.key(id);
        }
        s.world.set_frame(0);
        s.world_mode();
        let path = std::env::temp_dir().join(format!("voxely-import-{}.glb", std::process::id()));
        crate::gltf::write(&s, &path).unwrap();
        let src = directions(&mut s.world.clone(), 12);

        // ...and import it onto a rig twice the size.
        let g = Gltf::load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(g.clips().len(), 1);
        let mut w = World::single(Volume::new(IVec3::ONE));
        let first = w.models()[0];
        w.delete(first);
        humanoid(&mut w, IVec3::ZERO, 2, "big_", (1, 2, 3));
        w.anim.fps = 12.0;
        let done = import(&mut w, &g, None, None, true).unwrap();
        assert_eq!(done.frames, 12);
        assert!(done.bones.len() >= 10, "{:?}", done.bones.iter().map(|b| &b.1).collect::<Vec<_>>());
        let dst = directions(&mut w, 12);
        for f in 0..12 {
            for (b, d) in &src[f] {
                let got = dst[f][b];
                assert!(got.dot(*d) > 0.99, "frame {f} {b:?}: {got} vs {d}");
            }
        }
    }

    #[test]
    fn t_pose_clips_drive_arms_down_rigs() {
        // A hand-made glTF: a T-posed left arm (pointing +x) that stays
        // put, then lowers to point down.
        let doc = serde_json::json!({
            "asset": { "version": "2.0" },
            "nodes": [
                { "name": "Hips", "children": [1], "translation": [0.0, 1.0, 0.0] },
                { "name": "LeftArm", "children": [2], "translation": [0.2, 0.4, 0.0] },
                { "name": "LeftForeArm", "translation": [0.3, 0.0, 0.0] }
            ],
            "animations": [{ "name": "lower", "channels": [{ "sampler": 0, "target": { "node": 1, "path": "rotation" } }],
                "samplers": [{ "input": 0, "output": 1 }] }],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 2, "type": "SCALAR", "max": [1.0], "min": [0.0] },
                { "bufferView": 1, "componentType": 5126, "count": 2, "type": "VEC4" }
            ],
            "bufferViews": [{ "buffer": 0, "byteLength": 8 }, { "buffer": 0, "byteOffset": 8, "byteLength": 32 }],
            "buffers": [{ "byteLength": 40 }]
        });
        let half = std::f32::consts::FRAC_PI_4;
        // From rest (identity) to -90° about z (glTF), which turns +x down.
        let data: Vec<f32> = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -half.sin(), half.cos()];
        let g = Gltf { doc, buffers: vec![bytemuck::cast_slice(&data).to_vec()] };
        let mut w = rig_world();
        w.anim.fps = 4.0;
        import(&mut w, &g, Some("lower"), None, false).unwrap();
        let arm = find_bones(&w, ROOT)[&Bone::UpperArmL];
        let dir = |w: &World| {
            let v = w.node(arm).unwrap().volume().unwrap();
            (w.affine(arm).transform_point3(v.dims.as_vec3() * 0.5) - w.pivot_world(arm)).normalize()
        };
        // Frame 0 is the clip's T-pose, so the rig's arm (hanging down at
        // rest) is raised to point out (+x, the character's left).
        w.set_frame(0);
        assert!(dir(&w).x > 0.99, "{}", dir(&w));
        // Halfway the clip has lowered it 45°.
        w.set_frame(2);
        let d = dir(&w);
        assert!(d.x > 0.5 && d.z < -0.5, "{d}");
    }
}
