//! glTF 2.0 export (`.glb`, or `.gltf` with an embedded buffer), the format
//! game engines load directly: Godot, Unity (glTFast), Unreal, Three.js,
//! Babylon, Bevy.
//!
//! - Mesh: the same greedy-merged faces as the OBJ export, 1 voxel = 0.1 m,
//!   Y up, centered on the canvas. Each face has its own four vertices so
//!   normals stay flat.
//! - Color: a 256x1 palette texture (pixel k = color index k + 1, like the
//!   OBJ export) sampled with nearest filtering so colors don't bleed.
//! - Materials: one primitive per distinct voxely material, mapped onto
//!   glTF's metallic-roughness model plus KHR_materials_emissive_strength
//!   (emit), KHR_materials_transmission and KHR_materials_ior (glass).
//! - Clouds: glTF has no volumetric scattering, so each cloud material gets
//!   its own node (tagged in `extras` with `voxely_material: "cloud"`,
//!   density, phase and color, so engines can swap in their own fog) and a
//!   material that degrades gracefully: alpha blending with opacity from the
//!   density works everywhere, and KHR_materials_diffuse_transmission plus
//!   KHR_materials_volume give a softer look where supported.
//! - Structure: the scene tree. A root node named after the project, then a
//!   node per group and part with its transform; each node's origin is the
//!   part's pivot so engines turn it the same way. Sockets are empty child
//!   nodes (`extras.voxely_socket`), clouds are child nodes of their part.

use std::path::Path;

use anyhow::{Context, Result};
use glam::{Mat3, Quat, Vec3};
use serde_json::{Value, json};

use crate::export::{dir, greedy_quads};
use crate::scene::{GpuMaterial, MaterialKind, Scene};

const SCALE: f32 = 0.1;
const NEAREST: u32 = 9728;
const CLAMP_TO_EDGE: u32 = 33071;
const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;
const FLOAT: u32 = 5126;
const UNSIGNED_INT: u32 = 5125;

/// glTF material settings shared by every color that maps to them.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Look {
    metallic: f32,
    roughness: f32,
    emissive: f32,
    transmission: f32,
    ior: f32,
}

fn look(m: &GpuMaterial) -> Look {
    let q = |v: f32| (v * 100.0).round() / 100.0;
    let plain = Look { metallic: 0.0, roughness: 1.0, emissive: 0.0, transmission: 0.0, ior: 1.5 };
    let kind = m.kind;
    let l = if kind == MaterialKind::Metal as u32 {
        Look { metallic: m.metal, roughness: m.rough, ..plain }
    } else if kind == MaterialKind::Glass as u32 {
        Look { roughness: m.rough, transmission: m.trans.max(0.01), ior: m.ior, ..plain }
    } else if kind == MaterialKind::Emit as u32 {
        Look { emissive: m.emit.max(0.0), ..plain }
    } else if kind == MaterialKind::Blend as u32 {
        Look { metallic: m.metal, roughness: m.rough, transmission: m.trans, ior: m.ior, ..plain }
    } else {
        plain
    };
    Look { metallic: q(l.metallic), roughness: q(l.roughness), emissive: q(l.emissive), transmission: q(l.transmission), ior: q(l.ior) }
}

/// A cloud (participating media) color and how it is approximated.
#[derive(Clone, Copy, Debug)]
struct Cloud {
    color: u8,
    /// Extinction per voxel, as the path tracer uses it.
    density: f32,
    /// Henyey-Greenstein phase asymmetry.
    phase: f32,
    /// Mean distance through the cloud in voxels: 4 V / S.
    chord: f32,
}

impl Cloud {
    /// Coverage of one look through the cloud: 1 - transmittance over the
    /// mean chord, kept away from fully clear or fully solid.
    fn opacity(&self) -> f32 {
        (1.0 - (-self.density * self.chord).exp()).clamp(0.05, 0.95)
    }
}

/// What a group of faces becomes: a regular surface or a cloud.
#[derive(Clone, Copy, Debug)]
enum Kind {
    Surface(Look),
    Cloud(Cloud),
}

/// Mean chord (4 V / S, in voxels) of the voxels of each color.
fn chords(v: &crate::scene::Volume) -> [f32; 256] {
    let (mut vol, mut area) = ([0u32; 256], [0u32; 256]);
    for (i, &c) in v.cells.iter().enumerate() {
        if c == 0 {
            continue;
        }
        vol[c as usize] += 1;
        let p = v.position(i);
        for n in 0..6 {
            if v.get(p + dir(n)) != c {
                area[c as usize] += 1;
            }
        }
    }
    std::array::from_fn(|c| if area[c] == 0 { 1.0 } else { 4.0 * vol[c] as f32 / area[c] as f32 })
}

fn hex(c: [u8; 4]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

/// Vertex data for one primitive.
#[derive(Default)]
struct Prim {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

/// Builds the binary buffer with 4-byte aligned views.
#[derive(Default)]
struct Bin {
    data: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
}

impl Bin {
    fn view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        while self.data.len() % 4 != 0 {
            self.data.push(0);
        }
        let mut v = json!({ "buffer": 0, "byteOffset": self.data.len(), "byteLength": bytes.len() });
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        self.data.extend_from_slice(bytes);
        self.views.push(v);
        self.views.len() - 1
    }

    fn accessor(&mut self, bytes: &[u8], target: u32, component: u32, count: usize, kind: &str, bounds: Option<(Value, Value)>) -> usize {
        let view = self.view(bytes, Some(target));
        let mut a = json!({ "bufferView": view, "componentType": component, "count": count, "type": kind });
        if let Some((min, max)) = bounds {
            a["min"] = min;
            a["max"] = max;
        }
        self.accessors.push(a);
        self.accessors.len() - 1
    }
}

fn bytes<T: bytemuck::Pod>(v: &[T]) -> &[u8] {
    bytemuck::cast_slice(v)
}

/// Faces of one part, grouped into primitives by material kind.
struct PartMesh {
    /// (index into `kinds`, vertex data)
    prims: Vec<(usize, Prim)>,
}

/// Voxel space (z up, voxels) to glTF (y up, meters): (x, y, z) -> (x, z, -y).
fn to_gltf(v: Vec3) -> [f32; 3] {
    [v.x * SCALE, v.z * SCALE, -v.y * SCALE]
}

fn mat_to_gltf(m: Mat3) -> Mat3 {
    let c = Mat3::from_cols(Vec3::X, -Vec3::Z, Vec3::Y);
    c * m * c.transpose()
}

/// A node's transform as glTF TRS, or a matrix when it mirrors.
fn node_transform(node: &mut Value, translation: Vec3, linear: Mat3) {
    let t = to_gltf(translation);
    let m = mat_to_gltf(linear);
    if m.determinant() < 0.0 {
        let a = glam::Affine3A::from_mat3_translation(m, Vec3::from(t));
        node["matrix"] = json!(glam::Mat4::from(a).to_cols_array());
        return;
    }
    if t != [0.0; 3] {
        node["translation"] = json!(t);
    }
    let q = Quat::from_mat3(&m).normalize();
    if q.angle_between(Quat::IDENTITY) > 1e-5 {
        node["rotation"] = json!(q.to_array());
    }
}

/// A primitive in voxel space (relative to its pivot) turned into glTF
/// space through `flip` (identity or a mirror), with triangles rewound
/// when mirrored so they still face out.
fn mirror(p: &Prim, flip: Mat3) -> Prim {
    let mirrored = flip.determinant() < 0.0;
    let mut indices = p.indices.clone();
    if mirrored {
        indices.chunks_exact_mut(3).for_each(|t| t.swap(1, 2));
    }
    let dir = |v: Vec3| [v.x, v.z, -v.y];
    Prim {
        positions: p.positions.iter().map(|&v| to_gltf(flip * Vec3::from(v))).collect(),
        normals: p.normals.iter().map(|&n| dir(flip * Vec3::from(n))).collect(),
        uvs: p.uvs.clone(),
        indices,
    }
}

/// One primitive's buffers, using material `k`.
fn prim_json(bin: &mut Bin, k: usize, p: &Prim) -> Value {
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for q in &p.positions {
        for i in 0..3 {
            lo[i] = lo[i].min(q[i]);
            hi[i] = hi[i].max(q[i]);
        }
    }
    let pos = bin.accessor(bytes(&p.positions), ARRAY_BUFFER, FLOAT, p.positions.len(), "VEC3", Some((json!(lo), json!(hi))));
    let nrm = bin.accessor(bytes(&p.normals), ARRAY_BUFFER, FLOAT, p.normals.len(), "VEC3", None);
    let uv = bin.accessor(bytes(&p.uvs), ARRAY_BUFFER, FLOAT, p.uvs.len(), "VEC2", None);
    let idx = bin.accessor(bytes(&p.indices), ELEMENT_ARRAY_BUFFER, UNSIGNED_INT, p.indices.len(), "SCALAR", None);
    json!({ "attributes": { "POSITION": pos, "NORMAL": nrm, "TEXCOORD_0": uv }, "indices": idx, "material": k })
}

/// What `add_node` writes into.
struct Out<'a> {
    nodes: Vec<Value>,
    meshes: Vec<Value>,
    bin: &'a mut Bin,
    scene: &'a Scene,
    world: &'a crate::world::World,
    parts: &'a std::collections::HashMap<usize, PartMesh>,
    kinds: &'a [Kind],
    project: &'a str,
    /// One visible model: an unnamed one takes the project's name.
    only: bool,
}

/// Add node `id` (and its subtree), placed relative to its parent's
/// pivot `offset`. Returns its index.
///
/// glTF engines don't all handle mirrored nodes well, so mirrors are baked
/// into meshes instead: `pre` is the mirror the parent baked into its own
/// mesh, which this node's transform is corrected by.
fn add_node(id: usize, offset: Vec3, pre: Mat3, o: &mut Out) -> Option<usize> {
    let (world, kinds) = (o.world, o.kinds);
    let n = world.node(id)?;
    if !world.visible(id) {
        return None;
    }
    let pivot = world.pivot_local(id);
    let linear = crate::world::rot_mat(n.rot) * Mat3::from_quat(n.tilt_quat());
    // The node's local transform with its origin moved to the pivot.
    let half = n.volume().map_or(Vec3::ZERO, |v| (v.dims / 2).as_vec3());
    let translation = n.t.as_vec3() + crate::world::rot_mat(n.rot) * (pivot - half) - offset;
    let label = if o.only && n.name.is_empty() && !n.is_group() { o.project.to_string() } else { world.label(id) };
    let mut node = json!({ "name": label });
    let linear = pre * linear;
    // Mirrored: flip x in the mesh and children so the node is a rotation.
    let flip = if linear.determinant() < 0.0 { Mat3::from_diagonal(Vec3::new(-1.0, 1.0, 1.0)) } else { Mat3::IDENTITY };
    node_transform(&mut node, pre * translation, linear * flip);
    let index = o.nodes.len();
    o.nodes.push(node);
    let mut children = Vec::new();
    if let Some(pm) = o.parts.get(&id) {
        let mut surface = Vec::new();
        let clouds = pm.prims.iter().filter(|(k, _)| matches!(kinds[*k], Kind::Cloud(_))).count();
        for (k, p) in &pm.prims {
            let prim = prim_json(o.bin, *k, &mirror(p, flip));
            match kinds[*k] {
                Kind::Surface(_) => surface.push(prim),
                Kind::Cloud(c) => {
                    let cname = if clouds == 1 { format!("{label}_cloud") } else { format!("{label}_cloud_{}", c.color) };
                    o.meshes.push(json!({ "name": format!("{cname}_mesh"), "primitives": [prim] }));
                    o.nodes.push(json!({ "name": cname, "mesh": o.meshes.len() - 1, "extras": cloud_extras(o.scene, &c) }));
                    children.push(o.nodes.len() - 1);
                }
            }
        }
        if !surface.is_empty() {
            o.meshes.push(json!({ "name": format!("{label}_mesh"), "primitives": surface }));
            o.nodes[index]["mesh"] = json!(o.meshes.len() - 1);
        }
    }
    for s in &n.sockets {
        let mut sock = json!({ "name": s.name, "extras": { "voxely_socket": true } });
        node_transform(&mut sock, flip * (s.pos - pivot), Mat3::IDENTITY);
        o.nodes.push(sock);
        children.push(o.nodes.len() - 1);
    }
    for &c in n.children() {
        // Children sit relative to the group's pivot.
        if let Some(ci) = add_node(c, pivot, flip, o) {
            children.push(ci);
        }
    }
    if !children.is_empty() {
        o.nodes[index]["children"] = json!(children);
    }
    Some(index)
}

/// Write `scene` to `path`: binary `.glb`, or `.gltf` JSON with the buffer
/// embedded as a data URI when the extension is `gltf`.
pub fn write(scene: &Scene, path: &Path) -> Result<()> {
    let embedded = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gltf"));
    let name = path.file_stem().map_or("model".into(), |s| s.to_string_lossy().into_owned());
    let world = scene.synced_world();
    let world = &*world;
    let models: Vec<usize> = world.models().into_iter().filter(|&m| world.visible(m)).collect();

    // Group faces of each part: surfaces by material, clouds by color.
    let mut kinds: Vec<Kind> = Vec::new();
    let mut parts: std::collections::HashMap<usize, PartMesh> = Default::default();
    for &id in &models {
        let Some(v) = world.node(id).and_then(|n| n.volume()) else { continue };
        let pivot = world.pivot_local(id);
        let chord = chords(v);
        let mut prims: Vec<(usize, Prim)> = Vec::new();
        for q in greedy_quads(v) {
            let m = &scene.materials[q.color as usize];
            let kind = if m.kind == MaterialKind::Media as u32 {
                Kind::Cloud(Cloud { color: q.color, density: m.density, phase: m.g, chord: chord[q.color as usize] })
            } else {
                Kind::Surface(look(m))
            };
            let same = |k: &Kind| match (k, &kind) {
                (Kind::Surface(a), Kind::Surface(b)) => a == b,
                (Kind::Cloud(a), Kind::Cloud(b)) => a.color == b.color,
                _ => false,
            };
            let k = kinds.iter().position(same).unwrap_or_else(|| {
                kinds.push(kind);
                kinds.len() - 1
            });
            let slot = prims.iter().position(|(pk, _)| *pk == k).unwrap_or_else(|| {
                prims.push((k, Prim::default()));
                prims.len() - 1
            });
            let p = &mut prims[slot].1;
            let base = p.positions.len() as u32;
            let n = dir(q.normal).as_vec3().to_array();
            let uv = [((q.color as f32 - 1.0) + 0.5) / 256.0, 0.5];
            for c in q.corners {
                // Voxel units relative to the pivot, the node's origin;
                // converted to glTF space per node (see `add_node`).
                p.positions.push((c.as_vec3() - pivot).to_array());
                p.normals.push(n);
                p.uvs.push(uv);
            }
            p.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        parts.insert(id, PartMesh { prims });
    }

    let mut bin = Bin::default();

    // Nodes mirror the scene tree. Each node's origin is its pivot, so
    // engines turn parts the way they turn here; sockets are empty child
    // nodes to attach things to.
    let mut nodes: Vec<Value> = vec![json!({ "name": name })];
    // Center the scene like the other exports.
    let center = world.bounds(crate::world::ROOT).map_or(Vec3::ZERO, |(lo, size)| lo.as_vec3() + size.as_vec3() * 0.5);
    if center != Vec3::ZERO {
        nodes[0]["translation"] = json!(to_gltf(-center));
    }
    let mut out = Out { nodes, meshes: Vec::new(), bin: &mut bin, scene, world, parts: &parts, kinds: &kinds, project: &name, only: models.len() == 1 };
    let mut top = Vec::new();
    for &c in world.node(crate::world::ROOT).map(|n| n.children()).unwrap_or_default() {
        if let Some(i) = add_node(c, Vec3::ZERO, Mat3::IDENTITY, &mut out) {
            top.push(i);
        }
    }
    let Out { mut nodes, meshes, .. } = out;
    if !top.is_empty() {
        nodes[0]["children"] = json!(top);
    }

    // Palette texture.
    let mut png = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut png, 256, 1);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let px: Vec<u8> = (0..256).flat_map(|k| scene.palette[(k + 1) % 256]).collect();
        enc.write_header()?.write_image_data(&px)?;
    }
    let image = bin.view(&png, None);

    let mut used = std::collections::BTreeSet::new();
    let materials: Vec<Value> = kinds
        .iter()
        .enumerate()
        .map(|(k, kind)| {
            let mut ext = serde_json::Map::new();
            let mut m = match kind {
                Kind::Surface(l) => {
                    let mut m = json!({
                        "name": format!("voxel_{k}"),
                        "pbrMetallicRoughness": {
                            "baseColorTexture": { "index": 0 },
                            "metallicFactor": l.metallic,
                            "roughnessFactor": l.roughness,
                        },
                    });
                    if l.emissive > 0.0 {
                        m["emissiveTexture"] = json!({ "index": 0 });
                        m["emissiveFactor"] = json!([1.0, 1.0, 1.0]);
                        if l.emissive > 1.0 {
                            ext.insert("KHR_materials_emissive_strength".into(), json!({ "emissiveStrength": l.emissive }));
                        }
                    }
                    if l.transmission > 0.0 {
                        ext.insert("KHR_materials_transmission".into(), json!({ "transmissionFactor": l.transmission }));
                        ext.insert("KHR_materials_ior".into(), json!({ "ior": l.ior }));
                    }
                    m
                }
                Kind::Cloud(c) => {
                    let rgb = scene.materials[c.color as usize].color;
                    // Everywhere: alpha blending at the cloud's opacity.
                    let m = json!({
                        "name": format!("cloud_{}", c.color),
                        "alphaMode": "BLEND",
                        "pbrMetallicRoughness": {
                            "baseColorTexture": { "index": 0 },
                            "baseColorFactor": [1.0, 1.0, 1.0, c.opacity()],
                            "metallicFactor": 0.0,
                            "roughnessFactor": 1.0,
                        },
                        "extras": cloud_extras(scene, c),
                    });
                    // Where supported: light diffusing through, absorbed
                    // over distance like the cloud's density.
                    ext.insert(
                        "KHR_materials_diffuse_transmission".into(),
                        json!({ "diffuseTransmissionFactor": 1.0, "diffuseTransmissionColorFactor": [rgb[0], rgb[1], rgb[2]] }),
                    );
                    ext.insert(
                        "KHR_materials_volume".into(),
                        json!({
                            "thicknessFactor": c.chord * SCALE,
                            "attenuationDistance": SCALE / c.density.max(1e-3),
                            "attenuationColor": [rgb[0], rgb[1], rgb[2]],
                        }),
                    );
                    m
                }
            };
            if !ext.is_empty() {
                used.extend(ext.keys().cloned());
                m["extensions"] = Value::Object(ext);
            }
            m
        })
        .collect();

    let mut buffer = json!({ "byteLength": bin.data.len() });
    if embedded {
        use base64::Engine;
        buffer["uri"] = json!(format!("data:application/octet-stream;base64,{}", base64::engine::general_purpose::STANDARD.encode(&bin.data)));
    }
    let mut doc = json!({
        "asset": { "version": "2.0", "generator": concat!("voxely ", env!("CARGO_PKG_VERSION")) },
        "scene": 0,
        "scenes": [{ "name": name, "nodes": [0] }],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "textures": [{ "sampler": 0, "source": 0 }],
        "samplers": [{ "magFilter": NEAREST, "minFilter": NEAREST, "wrapS": CLAMP_TO_EDGE, "wrapT": CLAMP_TO_EDGE }],
        "images": [{ "bufferView": image, "mimeType": "image/png", "name": "palette" }],
        "accessors": bin.accessors,
        "bufferViews": bin.views,
        "buffers": [buffer],
    });
    if kinds.is_empty() {
        // An empty model: a valid file with an empty scene.
        for key in ["meshes", "materials"] {
            doc.as_object_mut().unwrap().remove(key);
        }
    }
    if !used.is_empty() {
        doc["extensionsUsed"] = json!(used.into_iter().collect::<Vec<_>>());
    }

    let out = if embedded {
        serde_json::to_vec_pretty(&doc)?
    } else {
        glb(&serde_json::to_vec(&doc)?, &bin.data)
    };
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))
}

/// Metadata so engines can find clouds and replace them with real fog.
fn cloud_extras(scene: &Scene, c: &Cloud) -> Value {
    json!({
        "voxely_material": "cloud",
        "palette_index": c.color,
        "color": hex(scene.palette[c.color as usize]),
        "density": c.density,
        "density_units": "extinction per voxel; 1 voxel = 0.1 m",
        "phase": c.phase,
        "opacity": c.opacity(),
    })
}

/// Pack JSON and binary chunks into a GLB container.
fn glb(json: &[u8], bin: &[u8]) -> Vec<u8> {
    let pad = |n: usize| (4 - n % 4) % 4;
    let (jp, bp) = (pad(json.len()), pad(bin.len()));
    let total = 12 + 8 + json.len() + jp + 8 + bin.len() + bp;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&((json.len() + jp) as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(json);
    out.extend(std::iter::repeat_n(b' ', jp));
    out.extend_from_slice(&((bin.len() + bp) as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(bin);
    out.extend(std::iter::repeat_n(0u8, bp));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;

    fn read_glb(path: &Path) -> (Value, Vec<u8>) {
        let data = std::fs::read(path).unwrap();
        assert_eq!(&data[0..4], b"glTF");
        assert_eq!(u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize, data.len());
        let jlen = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        assert_eq!(&data[16..20], b"JSON");
        let doc: Value = serde_json::from_slice(&data[20..20 + jlen]).unwrap();
        let b = 20 + jlen;
        let blen = u32::from_le_bytes(data[b..b + 4].try_into().unwrap()) as usize;
        assert_eq!(&data[b + 4..b + 8], b"BIN\0");
        (doc, data[b + 8..b + 8 + blen].to_vec())
    }

    #[test]
    fn glb_has_mesh_materials_and_texture() {
        let mut s = Scene::empty(IVec3::new(4, 4, 4));
        // A 2x2x2 cube of color 5 and one emissive voxel of color 9.
        for p in [IVec3::new(0, 0, 0), IVec3::new(1, 0, 0), IVec3::new(0, 1, 0), IVec3::new(1, 1, 0)] {
            s.volume.set(p, 5);
            s.volume.set(p + IVec3::Z, 5);
        }
        s.volume.set(IVec3::new(3, 3, 0), 9);
        s.materials[9].kind = MaterialKind::Emit as u32;
        s.materials[9].emit = 4.0;
        let dir = std::env::temp_dir().join("voxely-gltf-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.glb");
        write(&s, &path).unwrap();
        let (doc, bin) = read_glb(&path);
        // The BIN chunk is padded to 4 bytes past the buffer's length.
        let len = doc["buffers"][0]["byteLength"].as_u64().unwrap() as usize;
        assert!(bin.len() >= len && bin.len() - len < 4);
        let prims = doc["meshes"][0]["primitives"].as_array().unwrap();
        // Diffuse and emissive groups.
        assert_eq!(prims.len(), 2);
        assert_eq!(doc["materials"].as_array().unwrap().len(), 2);
        // The cube merges to 6 quads, the lone voxel is 6 more: 12 quads.
        let verts: u64 = prims.iter().map(|p| doc["accessors"][p["attributes"]["POSITION"].as_u64().unwrap() as usize]["count"].as_u64().unwrap()).sum();
        assert_eq!(verts, 12 * 4);
        let emissive = doc["materials"].as_array().unwrap().iter().find(|m| m.get("emissiveTexture").is_some()).unwrap();
        assert_eq!(emissive["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"], json!(4.0));
        assert_eq!(doc["samplers"][0]["magFilter"], json!(NEAREST));
        // Every view lies inside the buffer and accessors fit their views.
        for a in doc["accessors"].as_array().unwrap() {
            let view = &doc["bufferViews"][a["bufferView"].as_u64().unwrap() as usize];
            let end = view["byteOffset"].as_u64().unwrap() + view["byteLength"].as_u64().unwrap();
            assert!(end as usize <= bin.len());
            let comps = match a["type"].as_str().unwrap() {
                "VEC3" => 3,
                "VEC2" => 2,
                _ => 1,
            };
            assert_eq!(a["count"].as_u64().unwrap() * comps * 4, view["byteLength"].as_u64().unwrap());
        }

        let gltf = dir.join("m.gltf");
        write(&s, &gltf).unwrap();
        let doc: Value = serde_json::from_slice(&std::fs::read(&gltf).unwrap()).unwrap();
        assert!(doc["buffers"][0]["uri"].as_str().unwrap().starts_with("data:application/octet-stream;base64,"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn clouds_get_their_own_node_and_blended_material() {
        let mut s = Scene::empty(IVec3::new(8, 8, 8));
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    s.volume.set(IVec3::new(x, y, z), 7);
                }
            }
        }
        s.volume.set(IVec3::new(6, 6, 0), 2);
        s.materials[7].kind = MaterialKind::Media as u32;
        s.materials[7].density = 0.5;
        let path = std::env::temp_dir().join("voxely-gltf-cloud.glb");
        write(&s, &path).unwrap();
        let (doc, _) = read_glb(&path);
        std::fs::remove_file(&path).ok();

        // Root, model, and the cloud under the model.
        let nodes = doc["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[0]["children"], json!([1]));
        assert_eq!(nodes[1]["children"], json!([2]));
        let cloud = &nodes[2];
        assert_eq!(cloud["name"].as_str().unwrap(), "voxely-gltf-cloud_cloud");
        assert_eq!(cloud["extras"]["voxely_material"], json!("cloud"));
        assert_eq!(cloud["extras"]["palette_index"], json!(7));

        let mesh = &doc["meshes"][cloud["mesh"].as_u64().unwrap() as usize];
        let mat = &doc["materials"][mesh["primitives"][0]["material"].as_u64().unwrap() as usize];
        assert_eq!(mat["alphaMode"], json!("BLEND"));
        // 3x3x3 block: chord 4*27/54 = 2 voxels; opacity 1 - e^-1.
        let alpha = mat["pbrMetallicRoughness"]["baseColorFactor"][3].as_f64().unwrap();
        assert!((alpha - (1.0 - (-1.0f64).exp())).abs() < 1e-3, "{alpha}");
        assert!(mat["extensions"]["KHR_materials_diffuse_transmission"].is_object());
        assert!(mat["extensions"]["KHR_materials_volume"]["thicknessFactor"].as_f64().unwrap() > 0.0);
        let used: Vec<&str> = doc["extensionsUsed"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        assert!(used.contains(&"KHR_materials_volume") && used.contains(&"KHR_materials_diffuse_transmission"));
    }

    #[test]
    fn empty_model_is_valid() {
        let s = Scene::empty(IVec3::splat(3));
        let path = std::env::temp_dir().join("voxely-gltf-empty.glb");
        write(&s, &path).unwrap();
        let (doc, _) = read_glb(&path);
        assert!(doc.get("meshes").is_none());
        std::fs::remove_file(&path).ok();
    }

    /// A node's local matrix.
    fn local(n: &Value) -> glam::Mat4 {
        if let Some(m) = n.get("matrix") {
            let a: Vec<f32> = m.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
            return glam::Mat4::from_cols_slice(&a);
        }
        let f = |k: &str, d: Vec<f32>| n.get(k).map_or(d, |v| v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect());
        let t = f("translation", vec![0.0; 3]);
        let r = f("rotation", vec![0.0, 0.0, 0.0, 1.0]);
        glam::Mat4::from_rotation_translation(Quat::from_array([r[0], r[1], r[2], r[3]]), Vec3::new(t[0], t[1], t[2]))
    }

    #[test]
    fn tree_exports_nodes_pivots_and_sockets() {
        use crate::world::{Rot, Socket, ROOT};
        let mut s = Scene::empty(IVec3::new(4, 2, 6));
        s.volume = crate::ops::full(&s.volume, 3);
        let body = s.editing().unwrap();
        s.world_mode();
        let (arm, gem) = s.change(|w| {
            w.node_mut(body).unwrap().name = "body".into();
            w.node_mut(body).unwrap().sockets.push(Socket { name: "hand".into(), pos: Vec3::new(4.0, 1.0, 5.0) });
            let mut v = crate::scene::Volume::new(IVec3::new(2, 2, 5));
            v = crate::ops::full(&v, 9);
            let arm = w.add_model(ROOT, "arm", v, IVec3::new(4, 0, 1));
            w.node_mut(arm).unwrap().pivot = Some(Vec3::new(1.0, 1.0, 5.0));
            w.reorient(arm, Rot::turn(1, 1));
            w.reorient(arm, Rot::flip(1));
            let g = w.group(&[body, arm]).unwrap();
            w.node_mut(g).unwrap().name = "bot".into();
            // Attached under the mirrored arm, and turned itself.
            let gem = w.add_model(ROOT, "gem", crate::ops::full(&crate::scene::Volume::new(IVec3::new(3, 1, 2)), 5), IVec3::new(9, 0, 0));
            w.reorient(gem, Rot::turn(2, 1));
            w.attach_to(gem, arm, None, None);
            (arm, gem)
        });
        let path = std::env::temp_dir().join(format!("voxely-gltf-tree-{}.glb", std::process::id()));
        write(&s, &path).unwrap();
        let (doc, bin) = read_glb(&path);
        std::fs::remove_file(&path).ok();
        let nodes = doc["nodes"].as_array().unwrap();
        let find = |name: &str| nodes.iter().position(|n| n["name"] == json!(name)).unwrap();
        let (bot, body_i, arm_i, hand, gem_i) = (find("bot"), find("body"), find("arm"), find("hand"), find("gem"));
        assert!(nodes[arm_i]["children"].as_array().unwrap().contains(&json!(gem_i)));
        assert_eq!(nodes[0]["children"], json!([bot]));
        assert_eq!(nodes[bot]["children"], json!([body_i, arm_i]));
        assert!(nodes[body_i]["children"].as_array().unwrap().contains(&json!(hand)));
        assert_eq!(nodes[hand]["extras"]["voxely_socket"], json!(true));

        // Rebuild world positions from the node chain; they must land on
        // the scene's voxels (converted to glTF space).
        let mut global = vec![glam::Mat4::IDENTITY; nodes.len()];
        fn walk(i: usize, parent: glam::Mat4, nodes: &[Value], out: &mut Vec<glam::Mat4>) {
            out[i] = parent * local(&nodes[i]);
            for c in nodes[i]["children"].as_array().into_iter().flatten() {
                walk(c.as_u64().unwrap() as usize, out[i], nodes, out);
            }
        }
        walk(0, glam::Mat4::IDENTITY, nodes, &mut global);
        let w = &s.world;
        let (lo, size) = w.bounds(ROOT).unwrap();
        let center = lo.as_vec3() + size.as_vec3() * 0.5;
        for (id, ni) in [(body, body_i), (arm, arm_i), (gem, gem_i)] {
            let mesh = &doc["meshes"][nodes[ni]["mesh"].as_u64().unwrap() as usize];
            let acc = &doc["accessors"][mesh["primitives"][0]["attributes"]["POSITION"].as_u64().unwrap() as usize];
            let view = &doc["bufferViews"][acc["bufferView"].as_u64().unwrap() as usize];
            let off = view["byteOffset"].as_u64().unwrap() as usize;
            let pts: &[[f32; 3]] = bytemuck::cast_slice(&bin[off..off + view["byteLength"].as_u64().unwrap() as usize]);
            let (mut a, mut b) = (Vec3::MAX, Vec3::MIN);
            for p in pts {
                let q = global[ni].transform_point3(Vec3::from(*p));
                (a, b) = (a.min(q), b.max(q));
            }
            let (plo, psize) = w.bounds(id).unwrap();
            let x = Vec3::from(to_gltf(plo.as_vec3() - center));
            let y = Vec3::from(to_gltf((plo + psize).as_vec3() - center));
            let (ea, eb) = (x.min(y), x.max(y));
            assert!((a - ea).abs().max_element() < 1e-4 && (b - eb).abs().max_element() < 1e-4, "{id}: {a} {b} vs {ea} {eb}");
        }
        // The socket sits where the scene says.
        let sw = w.socket_world(body, "hand").unwrap();
        let got = global[hand].transform_point3(Vec3::ZERO);
        assert!((got - Vec3::from(to_gltf(sw - center))).length() < 1e-4);
        // Mirrors are baked into meshes: every node is a plain rotation,
        // and triangles still wind counter-clockwise seen from outside.
        assert!(nodes.iter().all(|n| n.get("matrix").is_none()));
        let read = |acc: u64| -> &[u8] {
            let a = &doc["accessors"][acc as usize];
            let v = &doc["bufferViews"][a["bufferView"].as_u64().unwrap() as usize];
            let off = v["byteOffset"].as_u64().unwrap() as usize;
            &bin[off..off + v["byteLength"].as_u64().unwrap() as usize]
        };
        for ni in [body_i, arm_i, gem_i] {
            let prim = &doc["meshes"][nodes[ni]["mesh"].as_u64().unwrap() as usize]["primitives"][0];
            let pos: &[[f32; 3]] = bytemuck::cast_slice(read(prim["attributes"]["POSITION"].as_u64().unwrap()));
            let nrm: &[[f32; 3]] = bytemuck::cast_slice(read(prim["attributes"]["NORMAL"].as_u64().unwrap()));
            let idx: &[u32] = bytemuck::cast_slice(read(prim["indices"].as_u64().unwrap()));
            assert!(global[ni].determinant() > 0.0);
            for t in idx.chunks_exact(3) {
                let [a, b, c] = [0, 1, 2].map(|k| Vec3::from(pos[t[k] as usize]));
                assert!((b - a).cross(c - a).dot(Vec3::from(nrm[t[0] as usize])) > 0.0, "inward triangle in node {ni}");
            }
        }
    }
}
