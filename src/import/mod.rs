//! Importing models made elsewhere: other voxel formats, meshes,
//! heightmaps, slice sheets and images.

pub mod formats;
pub mod mesh;
pub mod picture;
pub mod window;

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, bail, ensure};
use glam::{IVec3, Vec3};

use crate::scene::{Scene, Volume};
use crate::world::{ROOT, World};

/// Voxels read from another format, in voxely's axes (z up, a model
/// faces -y), with their own colors.
#[derive(Clone, Debug, Default)]
pub struct Model {
    pub name: String,
    pub dims: IVec3,
    /// Where this model's box sits among the file's other models.
    pub offset: IVec3,
    /// Positions inside 0..dims, colors RGBA (alpha 255).
    pub voxels: Vec<(IVec3, [u8; 4])>,
}

/// How a file is read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum How {
    /// Another voxel format (.qb, .vxl, .schem, ...).
    Voxels,
    Mesh(mesh::MeshOptions),
    Image(picture::ImageOptions),
    Heightmap(picture::HeightOptions),
    Slices(picture::SliceOptions),
}

impl How {
    pub fn name(&self) -> &'static str {
        match self {
            How::Voxels => "voxels",
            How::Mesh(_) => "mesh",
            How::Image(_) => "image",
            How::Heightmap(_) => "heightmap",
            How::Slices(_) => "slices",
        }
    }
}

/// Image file extensions (read as an image, a heightmap or slices).
pub const IMAGES: &[&str] = &["png", "jpg", "jpeg"];

/// Every extension the importer reads.
pub fn extensions() -> Vec<&'static str> {
    formats::EXTENSIONS.iter().chain(mesh::EXTENSIONS).chain(IMAGES).copied().collect()
}

fn ext(path: &Path) -> String {
    path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default()
}

/// The usual way to read `path` (images default to a flat picture), or
/// None for files the importer doesn't read.
pub fn default_how(path: &Path) -> Option<How> {
    let e = ext(path);
    let e = e.as_str();
    if formats::EXTENSIONS.contains(&e) {
        Some(How::Voxels)
    } else if mesh::EXTENSIONS.contains(&e) {
        Some(How::Mesh(Default::default()))
    } else if IMAGES.contains(&e) {
        Some(How::Image(Default::default()))
    } else {
        None
    }
}

pub fn is_importable(path: &Path) -> bool {
    default_how(path).is_some()
}

/// Read `path` (a file, or for slices also a folder of images) into
/// models no larger than a part may be, plus warnings about what couldn't
/// be read.
pub fn read(path: &Path, how: &How) -> Result<(Vec<Model>, Vec<String>)> {
    let mut warnings = Vec::new();
    let models = match how {
        How::Voxels => formats::load(path)?,
        How::Mesh(o) => {
            let (m, w) = mesh::voxelize_noted(path, o)?;
            warnings = w;
            vec![m]
        }
        How::Image(o) => vec![picture::image(path, o)?],
        How::Heightmap(o) => vec![picture::heightmap(path, o)?],
        How::Slices(o) => vec![picture::slices(path, o)?],
    };
    let models: Vec<Model> = models.into_iter().filter(|m| !m.voxels.is_empty()).collect();
    ensure!(!models.is_empty(), "{} has no voxels", path.display());
    let total: usize = models.iter().map(|m| m.voxels.len()).sum();
    ensure!(total <= MAX_VOXELS, "{} has {total} voxels; the importer takes up to {MAX_VOXELS}", path.display());
    Ok((models.into_iter().flat_map(split).collect(), warnings))
}

/// Voxels one import may bring in.
pub const MAX_VOXELS: usize = 64 << 20;

/// Cut a model bigger than a part may be into tiles that fit.
fn split(m: Model) -> Vec<Model> {
    let max = crate::ops::MAX_DIM;
    if m.dims.max_element() <= max {
        return vec![m];
    }
    let mut out: HashMap<IVec3, Model> = HashMap::new();
    for (p, c) in m.voxels {
        let t = p / max;
        let tile = out.entry(t).or_insert_with(|| {
            let lo = t * max;
            let name = if m.name.is_empty() { format!("{} {} {}", t.x, t.y, t.z) } else { format!("{} {} {} {}", m.name, t.x, t.y, t.z) };
            Model { name, dims: (m.dims - lo).min(IVec3::splat(max)), offset: m.offset + lo, voxels: Vec::new() }
        });
        tile.voxels.push((p - t * max, c));
    }
    let mut out: Vec<Model> = out.into_values().collect();
    out.sort_by_key(|m| (m.offset.z, m.offset.y, m.offset.x));
    out
}

// ---------------------------------------------------------------- colors

/// Distinct colors and how many voxels use each, most used first.
fn color_counts(models: &[Model]) -> Vec<([u8; 3], usize)> {
    let mut counts: HashMap<[u8; 3], usize> = HashMap::new();
    for m in models {
        for (_, c) in &m.voxels {
            *counts.entry([c[0], c[1], c[2]]).or_default() += 1;
        }
    }
    let mut v: Vec<([u8; 3], usize)> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

/// At most `n` colors standing for `colors` (with counts): all of them
/// when they fit, else median cut weighted by use.
pub fn reduce(colors: &[([u8; 3], usize)], n: usize) -> Vec<[u8; 3]> {
    if colors.len() <= n {
        return colors.iter().map(|c| c.0).collect();
    }
    if n == 0 {
        return Vec::new();
    }
    let mut boxes: Vec<Vec<([u8; 3], usize)>> = vec![colors.to_vec()];
    while boxes.len() < n {
        // Split the box with the widest channel range (times its weight's
        // square root, so busy boxes split first) at its weighted median.
        let score = |b: &Vec<([u8; 3], usize)>| {
            if b.len() < 2 {
                return (0.0, 0);
            }
            let (ch, range) = (0..3)
                .map(|k| {
                    let (lo, hi) = b.iter().fold((255u8, 0u8), |(lo, hi), c| (lo.min(c.0[k]), hi.max(c.0[k])));
                    (k, (hi - lo) as f64)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            let weight: usize = b.iter().map(|c| c.1).sum();
            (range * (weight as f64).sqrt(), ch)
        };
        let (k, (s, ch)) = boxes.iter().map(score).enumerate().max_by(|a, b| a.1.0.total_cmp(&b.1.0)).unwrap();
        if s <= 0.0 {
            break;
        }
        let mut b = boxes.swap_remove(k);
        b.sort_by_key(|c| c.0[ch]);
        let half: usize = b.iter().map(|c| c.1).sum::<usize>() / 2;
        let mut acc = 0;
        let mut cut = 1;
        for (i, c) in b.iter().enumerate() {
            acc += c.1;
            if acc >= half {
                cut = (i + 1).clamp(1, b.len() - 1);
                break;
            }
        }
        let rest = b.split_off(cut);
        boxes.push(b);
        boxes.push(rest);
    }
    boxes
        .iter()
        .map(|b| {
            let w: u64 = b.iter().map(|c| c.1 as u64).sum::<u64>().max(1);
            let mut s = [0u64; 3];
            for c in b {
                for k in 0..3 {
                    s[k] += c.0[k] as u64 * c.1 as u64;
                }
            }
            s.map(|x| ((x + w / 2) / w) as u8)
        })
        .collect()
}

fn lab(c: [u8; 3]) -> Vec3 {
    crate::control::surface::oklab([c[0], c[1], c[2], 255])
}

fn dist(a: Vec3, b: Vec3) -> f32 {
    let d = a - b;
    d.x * d.x + 2.0 * (d.y * d.y + d.z * d.z)
}

/// For each color in `models`, the palette slot to use. Colors already in
/// the palette keep their slot; the rest go into the slots `free` (reduced
/// to fit), or to the nearest palette color. Returns the map and the
/// slots written.
fn assign(palette: &mut [[u8; 4]], models: &[Model], free: &[u8]) -> (HashMap<[u8; 3], u8>, Vec<u8>) {
    let counts = color_counts(models);
    let exact: HashMap<[u8; 3], u8> = (1..256).rev().map(|i| ([palette[i][0], palette[i][1], palette[i][2]], i as u8)).collect();
    let missing: Vec<([u8; 3], usize)> = counts.iter().filter(|c| !exact.contains_key(&c.0)).copied().collect();
    // Slots already holding one of the colors stay as they are.
    let kept: std::collections::HashSet<u8> = counts.iter().filter_map(|c| exact.get(&c.0).copied()).collect();
    let free: Vec<u8> = free.iter().copied().filter(|s| !kept.contains(s)).collect();
    let mut written = Vec::new();
    for (&slot, c) in free.iter().zip(reduce(&missing, free.len())) {
        palette[slot as usize] = [c[0], c[1], c[2], 255];
        written.push(slot);
    }
    let labs: Vec<(u8, Vec3)> = (1..256).map(|i| (i as u8, lab([palette[i][0], palette[i][1], palette[i][2]]))).collect();
    let map = counts
        .iter()
        .map(|&(c, _)| {
            let slot = exact.get(&c).copied().unwrap_or_else(|| {
                let l = lab(c);
                labs.iter().min_by(|a, b| dist(a.1, l).total_cmp(&dist(b.1, l))).map_or(1, |x| x.0)
            });
            (c, slot)
        })
        .collect();
    (map, written)
}

fn volume(m: &Model, map: &HashMap<[u8; 3], u8>) -> Volume {
    let mut v = Volume::new(m.dims.max(IVec3::ONE));
    for &(p, c) in &m.voxels {
        if v.contains(p) {
            v.set(p, map[&[c[0], c[1], c[2]]]);
        }
    }
    v.refresh_bricks();
    v
}

/// A new scene holding `models` as parts, with a palette made from their
/// colors.
pub fn new_scene(models: &[Model]) -> Result<Scene> {
    ensure!(!models.is_empty(), "nothing to import");
    let mut palette = crate::vox::default_palette();
    // Every slot is free: the file's colors replace the default palette
    // (unused slots keep the default's colors).
    let free: Vec<u8> = (1..=255).collect();
    let (map, _) = assign(&mut palette, models, &free);
    let mut world = World::empty();
    let lo = models.iter().map(|m| m.offset).reduce(IVec3::min).unwrap();
    for m in models {
        world.add_model(ROOT, &m.name, volume(m, &map), m.offset - lo);
    }
    let bricks: usize = world.models().iter().filter_map(|&m| world.node(m)?.volume()).map(|v| v.cells.allocated()).sum();
    if bricks > crate::scene::MAX_BRICKS {
        bail!("too large: voxels in {bricks} 8x8x8 bricks (the limit is {})", crate::scene::MAX_BRICKS);
    }
    Ok(Scene::from_world(world, palette))
}

/// Add `models` to `scene` as new parts (grouped under `name` when there
/// are several), right of what's there, as one undo step. Their colors
/// fill unused palette slots, or match the nearest palette colors when
/// `add_colors` is off or the slots run out. Returns the new top node and
/// the palette slots written.
pub fn add_to_scene(scene: &mut Scene, models: &[Model], name: &str, add_colors: bool) -> Result<(usize, Vec<u8>)> {
    ensure!(!models.is_empty(), "nothing to import");
    let before = scene.palette_state();
    let used = scene.used_colors();
    let free: Vec<u8> = if add_colors { (1..256).rev().filter(|&i| !used[i]).map(|i| i as u8).collect() } else { Vec::new() };
    let (map, written) = assign(&mut scene.palette, models, &free);
    for &i in &written {
        scene.materials[i as usize] = crate::scene::material(i as u32, scene.palette[i as usize], None);
    }
    scene.world_mode();
    let w = &scene.world;
    let base = w.bounds(ROOT).map_or(IVec3::ZERO, |(lo, s)| IVec3::new(lo.x + s.x + 2, lo.y, lo.z.max(0)));
    let lo = models.iter().map(|m| m.offset).reduce(IVec3::min).unwrap();
    let vols: Vec<(String, Volume, IVec3)> = models.iter().map(|m| (if m.name.is_empty() { name.to_string() } else { m.name.clone() }, volume(m, &map), base + m.offset - lo)).collect();
    let top = scene.change(|w| {
        let ids: Vec<usize> = vols.into_iter().map(|(n, v, at)| w.add_model(ROOT, &n, v, at)).collect();
        match ids.as_slice() {
            [one] => *one,
            _ => {
                let g = w.group(&ids).unwrap_or(ids[0]);
                if let Some(n) = w.node_mut(g) {
                    n.name = name.to_string();
                }
                g
            }
        }
    });
    scene.picked = vec![top];
    if !written.is_empty() {
        scene.palette_step(before);
    }
    Ok((top, written))
}

/// What an import brought in, for messages.
pub fn summary(models: &[Model]) -> String {
    let voxels: usize = models.iter().map(|m| m.voxels.len()).sum();
    let colors = color_counts(models).len();
    let size = models.iter().map(|m| (m.offset, m.offset + m.dims)).reduce(|a, b| (a.0.min(b.0), a.1.max(b.1))).map_or(IVec3::ZERO, |(lo, hi)| hi - lo);
    let parts = if models.len() == 1 { String::new() } else { format!(" in {} parts", models.len()) };
    format!("{voxels} voxels{parts}, {} x {} x {}, {colors} colors", size.x, size.y, size.z)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(voxels: &[([i32; 3], [u8; 4])]) -> Model {
        let voxels: Vec<(IVec3, [u8; 4])> = voxels.iter().map(|&(p, c)| (IVec3::from(p), c)).collect();
        let dims = voxels.iter().map(|v| v.0 + 1).reduce(IVec3::max).unwrap();
        Model { name: "m".into(), dims, offset: IVec3::ZERO, voxels }
    }

    #[test]
    fn reduce_keeps_few_colors_and_cuts_many() {
        let few = [([1, 2, 3], 5), ([200, 0, 0], 1)];
        assert_eq!(reduce(&few, 4).len(), 2);
        let many: Vec<([u8; 3], usize)> = (0..=255u8).flat_map(|r| [([r, 0, 0], 1), ([0, r, 255], 1)]).collect();
        let out = reduce(&many, 16);
        assert_eq!(out.len(), 16);
        // Both clusters are represented.
        assert!(out.iter().any(|c| c[2] == 0) && out.iter().any(|c| c[2] == 255));
    }

    #[test]
    fn new_scene_uses_exact_colors() {
        let m = model(&[([0, 0, 0], [10, 20, 30, 255]), ([1, 0, 2], [200, 100, 50, 255])]);
        let s = new_scene(&[m]).unwrap();
        let v = &s.volume;
        let a = s.palette[v.get(IVec3::new(0, 0, 0)) as usize];
        let b = s.palette[v.get(IVec3::new(1, 0, 2)) as usize];
        assert_eq!(a, [10, 20, 30, 255]);
        assert_eq!(b, [200, 100, 50, 255]);
    }

    #[test]
    fn adding_fills_free_slots_and_groups_parts() {
        let mut s = Scene::empty(IVec3::splat(4));
        s.volume.set(IVec3::ZERO, 1);
        let before = s.palette[1];
        let mut a = model(&[([0, 0, 0], [1, 2, 3, 255])]);
        let mut b = model(&[([0, 0, 0], before)]);
        a.name = "a".into();
        b.name = "b".into();
        b.offset = IVec3::new(5, 0, 0);
        let (top, written) = add_to_scene(&mut s, &[a, b], "file", true).unwrap();
        assert_eq!(written.len(), 1, "only the new color takes a slot");
        assert_eq!(s.palette[1], before, "colors in use are kept");
        assert!(s.world.node(top).unwrap().is_group());
        assert_eq!(s.world.label(top), "file");
        assert_eq!(s.world.models().len(), 3);
        assert!(s.can_undo_tree());
    }

    #[test]
    fn colors_already_in_the_palette_keep_their_slots() {
        // More colors than slots, white among them: white keeps slot 1 (the
        // default palette's white) and nothing else is written over it.
        let mut voxels: Vec<([i32; 3], [u8; 4])> = (0..300).map(|i| ([i % 20, 0, i / 20], [(i * 7 % 256) as u8, (i * 13 % 256) as u8, 77, 255])).collect();
        voxels.push(([0, 1, 0], [255, 255, 255, 255]));
        let s = new_scene(&[model(&voxels)]).unwrap();
        assert_eq!(s.volume.get(IVec3::new(0, 1, 0)), 1);
        assert_eq!(s.palette[1], [255, 255, 255, 255]);
    }

    #[test]
    fn undoing_an_import_restores_the_palette() {
        let mut s = Scene::empty(IVec3::splat(4));
        s.volume.set(IVec3::ZERO, 1);
        let (before, mats) = s.palette_state();
        add_to_scene(&mut s, &[model(&[([0, 0, 0], [1, 2, 3, 255])])], "file", true).unwrap();
        let after = s.palette.clone();
        assert_ne!(after, before);
        assert!(s.undo_tree(false));
        assert_eq!(s.palette, before);
        assert_eq!(bytemuck::cast_slice::<_, u8>(&s.materials), bytemuck::cast_slice::<_, u8>(&mats));
        assert!(s.undo_tree(true));
        assert_eq!(s.palette, after);
    }

    #[test]
    fn big_models_split_into_parts() {
        let max = crate::ops::MAX_DIM;
        let m = Model { name: "big".into(), dims: IVec3::new(max + 10, 2, 1), offset: IVec3::ZERO, voxels: vec![(IVec3::ZERO, [9; 4]), (IVec3::new(max + 5, 1, 0), [9; 4])] };
        let parts = split(m);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[1].offset, IVec3::new(max, 0, 0));
        assert_eq!(parts[1].voxels[0].0, IVec3::new(5, 1, 0));
        assert_eq!(parts[1].dims, IVec3::new(10, 2, 1));
    }
}
