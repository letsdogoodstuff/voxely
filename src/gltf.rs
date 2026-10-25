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
//! - Structure: a root node named after the project with a child node for
//!   the model and one per cloud.

use std::path::Path;

use anyhow::{Context, Result};
use glam::Vec3;
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

/// Write `scene` to `path`: binary `.glb`, or `.gltf` JSON with the buffer
/// embedded as a data URI when the extension is `gltf`.
pub fn write(scene: &Scene, path: &Path) -> Result<()> {
    let embedded = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gltf"));
    let name = path.file_stem().map_or("model".into(), |s| s.to_string_lossy().into_owned());
    let v = &scene.volume;
    let center = v.dims.as_vec3() * 0.5;
    // Voxel space (z up) to glTF (y up, meters).
    let place = |p: Vec3| {
        let q = (p - center) * SCALE;
        [q.x, q.z, -q.y]
    };

    // Group faces: surfaces by material, clouds by color.
    let chord = chords(v);
    let mut kinds: Vec<Kind> = Vec::new();
    let mut prims: Vec<Prim> = Vec::new();
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
        let k = match kinds.iter().position(same) {
            Some(k) => k,
            None => {
                kinds.push(kind);
                prims.push(Prim::default());
                kinds.len() - 1
            }
        };
        let p = &mut prims[k];
        let base = p.positions.len() as u32;
        let d = dir(q.normal).as_vec3();
        let n = [d.x, d.z, -d.y];
        let uv = [((q.color as f32 - 1.0) + 0.5) / 256.0, 0.5];
        for c in q.corners {
            p.positions.push(place(c.as_vec3()));
            p.normals.push(n);
            p.uvs.push(uv);
        }
        p.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    let mut bin = Bin::default();
    // Surfaces share one mesh; each cloud gets its own.
    let mut surface_prims = Vec::new();
    let mut cloud_meshes: Vec<(Cloud, Value)> = Vec::new();
    for (k, p) in prims.iter().enumerate() {
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
        let prim = json!({
            "attributes": { "POSITION": pos, "NORMAL": nrm, "TEXCOORD_0": uv },
            "indices": idx,
            "material": k,
        });
        match kinds[k] {
            Kind::Surface(_) => surface_prims.push(prim),
            Kind::Cloud(c) => cloud_meshes.push((c, prim)),
        }
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

    // Nodes: the root, the surface model, then one node per cloud.
    let mut meshes = Vec::new();
    let mut nodes = vec![json!({ "name": name })];
    if !surface_prims.is_empty() {
        meshes.push(json!({ "name": format!("{name}_mesh"), "primitives": surface_prims }));
        nodes.push(json!({ "name": format!("{name}_model"), "mesh": meshes.len() - 1 }));
    }
    let single = cloud_meshes.len() == 1;
    for (c, prim) in cloud_meshes {
        let node = if single { format!("{name}_cloud") } else { format!("{name}_cloud_{}", c.color) };
        meshes.push(json!({ "name": format!("{node}_mesh"), "primitives": [prim] }));
        nodes.push(json!({ "name": node, "mesh": meshes.len() - 1, "extras": cloud_extras(scene, &c) }));
    }
    if nodes.len() > 1 {
        nodes[0]["children"] = json!((1..nodes.len()).collect::<Vec<_>>());
    }

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

        // Root, model, cloud.
        let nodes = doc["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[0]["children"], json!([1, 2]));
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
}
