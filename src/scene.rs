//! Flattens a `VoxFile` scene graph into a voxel volume plus the
//! material table and render settings the GPU path tracer consumes.

use anyhow::{bail, Result};
use glam::{IVec3, Vec3};

use std::path::PathBuf;

use crate::env::EnvMode;
use crate::vox::{Dict, VoxFile};
use crate::world::World;

use crate::grid::Grid;

/// Palette index per cell, stored sparsely: only 8³ bricks that have held
/// a voxel take memory.
pub type Cells = Grid<u8>;

/// A selection or highlight: one flag per cell, stored like `Cells`.
pub type Mask = Grid<bool>;

#[derive(Clone)]
pub struct Volume {
    pub dims: IVec3,
    /// Color index per cell, addressed x-fastest then y then z.
    pub cells: Cells,
}

impl Volume {
    pub fn new(dims: IVec3) -> Self {
        Volume { dims, cells: Grid::new(dims) }
    }

    pub fn contains(&self, p: IVec3) -> bool {
        p.cmpge(IVec3::ZERO).all() && p.cmplt(self.dims).all()
    }

    pub fn index(&self, p: IVec3) -> usize {
        (p.x + self.dims.x * (p.y + self.dims.y * p.z)) as usize
    }

    pub fn position(&self, i: usize) -> IVec3 {
        let i = i as i32;
        IVec3::new(i % self.dims.x, (i / self.dims.x) % self.dims.y, i / (self.dims.x * self.dims.y))
    }

    /// A volume holding `cells` (dense, x-fastest).
    pub fn from_cells(dims: IVec3, cells: Vec<u8>) -> Self {
        Volume { dims, cells: Grid::from_vec(dims, &cells) }
    }

    /// Bricks along each axis.
    pub fn brick_dims(&self) -> IVec3 {
        self.cells.brick_dims()
    }

    /// Free the storage of bricks emptied by erasing (after bulk edits).
    pub fn refresh_bricks(&mut self) {
        self.cells.compact();
    }

    pub fn set(&mut self, p: IVec3, c: u8) {
        if self.contains(p) {
            self.cells.set_at(p, c);
        }
    }

    /// Color index at `p`, 0 outside the volume.
    pub fn get(&self, p: IVec3) -> u8 {
        if self.contains(p) { self.cells.at(p) } else { 0 }
    }

    pub fn set_index(&mut self, i: usize, c: u8) {
        self.cells.set(i, c);
    }

    /// Solid cells as (position, color), brick by brick. Costs time for the
    /// bricks holding voxels, not the whole box.
    pub fn filled(&self) -> impl Iterator<Item = (IVec3, u8)> + '_ {
        self.cells.filled().map(|(i, c)| (self.position(i), c))
    }

    /// Solid cells as (position, color) in dense order (x fastest), for
    /// output whose order should not depend on the storage.
    pub fn filled_sorted(&self) -> Vec<(IVec3, u8)> {
        let mut cells: Vec<(usize, u8)> = self.cells.filled().collect();
        cells.sort_unstable_by_key(|c| c.0);
        cells.into_iter().map(|(i, c)| (self.position(i), c)).collect()
    }

    /// Number of solid cells.
    pub fn count(&self) -> usize {
        self.cells.filled().count()
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaterialKind {
    Diffuse = 0,
    Metal = 1,
    Glass = 2,
    Emit = 3,
    Blend = 4,
    Media = 5,
}

/// GPU material record; must match `Material` in `pathtrace.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterial {
    /// Linear RGB albedo, alpha from palette.
    pub color: [f32; 4],
    pub kind: u32,
    pub rough: f32,
    pub metal: f32,
    pub ior: f32,
    /// Emitted radiance multiplier (already includes power).
    pub emit: f32,
    pub trans: f32,
    pub density: f32,
    /// Phase function asymmetry for clouds.
    pub g: f32,
}

/// Renderer settings, initialised from the file's `rOBJ` chunks.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderSettings {
    pub sun_pitch: f32,
    pub sun_yaw: f32,
    pub sun_area: f32,
    pub sun_intensity: f32,
    pub sun_color: Vec3,
    pub sun_disk: bool,
    pub sky_intensity: f32,
    pub sky_color: Vec3,
    pub env_mode: EnvMode,
    pub ibl_path: Option<PathBuf>,
    pub ibl_intensity: f32,
    /// Fraction of a full turn.
    pub ibl_rotation: f32,
    /// Atmosphere colors are scattering ratios, not sRGB display colors.
    pub rayleigh_density: f32,
    pub rayleigh_color: Vec3,
    pub mie_density: f32,
    pub mie_color: Vec3,
    pub mie_g: f32,
    pub ozone_density: f32,
    pub ozone_color: Vec3,
    pub fog_density: f32,
    pub fog_color: Vec3,
    pub fog_g: f32,
    pub translucent_shadows: bool,
    pub fov: f32,
    pub projection: Projection,
    pub aperture: f32,
    /// Aperture polygon sides; below 3 means a round aperture.
    pub blades: u32,
    /// Degrees.
    pub blade_rotation: f32,
    /// Linear multiplier.
    pub exposure: f32,
    pub gamma: f32,
    pub aces: bool,
    pub vignette: f32,
    pub ground: bool,
    pub ground_color: Vec3,
    pub bloom: bool,
    pub bloom_mix: f32,
    pub bloom_scale: f32,
    pub bloom_aspect: f32,
    pub bloom_threshold: f32,
    pub max_diffuse: u32,
    pub max_specular: u32,
    pub max_scatter: u32,
    pub clamp: f32,
    /// Edge-aware denoising of the accumulated image (display only).
    pub denoise: bool,
    pub denoise_view: DenoiseView,
    /// Lines along silhouette, crease and color-change voxel edges.
    pub show_edges: bool,
    pub edge_color: Vec3,
    /// Fraction of a voxel, 0.01..1.
    pub edge_width: f32,
    /// Grid lines on the ground plane (and voxel faces unless ground only).
    pub show_grid: bool,
    pub grid_color: Vec3,
    /// Voxels between lines, 1..1024.
    pub grid_spacing: u32,
    /// Line width as a fraction of a voxel.
    pub grid_width: f32,
    pub grid_ground_only: bool,
    /// What camera rays that miss everything show (lighting is unaffected).
    pub bg_mode: BgMode,
    pub bg_color: Vec3,
}

/// Which image to show while denoising: the filtered one, the raw
/// accumulation, or original on the left and denoised on the right.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DenoiseView {
    #[default]
    Denoised = 0,
    Original = 1,
    Split = 2,
}

/// The directly visible background.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BgMode {
    /// The sky / environment map.
    #[default]
    Sky = 0,
    /// A solid `bg_color`.
    Color = 1,
    /// Alpha 0 in saved images; a checkerboard in the window.
    Transparent = 2,
}

impl Default for RenderSettings {
    fn default() -> Self {
        // Defaults match a fresh MagicaVoxel 0.99.6 scene.
        Self {
            sun_pitch: 50.0,
            sun_yaw: 50.0,
            sun_area: 0.07,
            sun_intensity: 0.6,
            sun_color: Vec3::ONE,
            sun_disk: false,
            sky_intensity: 0.7,
            sky_color: Vec3::ONE,
            env_mode: EnvMode::Uniform,
            ibl_path: None,
            ibl_intensity: 1.0,
            ibl_rotation: 0.0,
            rayleigh_density: 0.4,
            rayleigh_color: Vec3::new(77.0, 153.0, 255.0) / 255.0,
            mie_density: 0.4,
            mie_color: Vec3::ONE,
            mie_g: 0.78,
            ozone_density: 0.0,
            ozone_color: Vec3::new(105.0, 255.0, 110.0) / 255.0,
            fog_density: 0.0,
            fog_color: Vec3::ONE,
            fog_g: 0.0,
            translucent_shadows: true,
            fov: 45.0,
            projection: Projection::Perspective,
            aperture: 0.0,
            blades: 0,
            blade_rotation: 0.0,
            exposure: 1.0,
            gamma: 2.2,
            aces: false,
            vignette: 0.0,
            ground: true,
            ground_color: srgb_to_linear(Vec3::splat(80.0 / 255.0)),
            bloom: false,
            bloom_mix: 0.5,
            bloom_scale: 0.0,
            bloom_aspect: 0.0,
            bloom_threshold: 1.0,
            max_diffuse: 4,
            max_specular: 8,
            max_scatter: 8,
            clamp: 50.0,
            denoise: false,
            denoise_view: DenoiseView::Denoised,
            show_edges: false,
            edge_color: Vec3::ZERO,
            edge_width: 0.1,
            show_grid: false,
            grid_color: srgb_to_linear(Vec3::splat(40.0 / 255.0)),
            grid_spacing: 1,
            grid_width: 0.05,
            grid_ground_only: false,
            bg_mode: BgMode::Sky,
            bg_color: srgb_to_linear(Vec3::splat(40.0 / 255.0)),
        }
    }
}

/// One exposed face of an emissive voxel; must match `Light` in
/// `pathtrace.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLight {
    /// Face corner (minimum on the two in-plane axes), CDF value in `w`.
    pub corner_cdf: [f32; 4],
    /// Normal axis, 1 if the normal points along +axis, material index.
    pub axis_sign_mat: [u32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Projection {
    Perspective = 0,
    Orthographic = 1,
    Stereographic = 2,
    Panorama = 3,
    Cylindrical = 4,
}

/// What the voxel tools work on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Editing one model, turned to world orientation.
    Edit(usize),
    /// Every visible model flattened; parts are picked and moved whole.
    World,
}

/// Most 8³ bricks of voxels a scene can hold: what the path tracer's brick
/// pool fits in 2048-deep textures (half a billion voxels' worth).
pub const MAX_BRICKS: usize = 1 << 20;

/// Saved trees for undoing whole-part changes.
const HISTORY: usize = 64;

pub struct Scene {
    /// The voxels being worked on: the model being edited, or in World
    /// mode every visible model flattened.
    pub volume: Volume,
    /// The scene tree. While editing, the edited model's voxels live in
    /// `volume` and are written back when leaving edit mode or saving.
    pub world: World,
    pub mode: Mode,
    /// World position of `volume`'s cell 0.
    pub origin: IVec3,
    /// World mode: the model node each cell of `volume` came from.
    pub owners: Grid<u32>,
    /// Edit mode: the other visible models, flattened, and where their
    /// cell 0 sits relative to `volume`.
    pub context: Option<(Volume, IVec3)>,
    /// The picked nodes (World mode), or the edited model.
    pub picked: Vec<usize>,
    history: Vec<World>,
    future: Vec<World>,
    /// Tree undo steps pushed so far (`history` is capped, so its length
    /// doesn't say).
    steps: u64,
    /// Tree steps that also changed the palette (an import): the step's
    /// position, and the palette and materials before and after it.
    palette_steps: Vec<(u64, PaletteState, PaletteState)>,
    /// The palette or materials changed by undo/redo, for the GPU.
    pub palette_dirty: bool,
    /// sRGB palette as stored in the file, indexed by color index.
    pub palette: Vec<[u8; 4]>,
    /// Palette row names (ramps), as in MagicaVoxel's `NOTE` chunk: entry
    /// k names colors 8k + 1 ..= 8k + 8.
    pub notes: Vec<String>,
    /// Render chunks from the file, written back unchanged on save.
    pub render_dicts: Vec<Dict>,
    /// `MATL` dictionaries from the file: written back as they were while
    /// the material is unchanged, else with voxely's keys updated and the
    /// rest kept.
    pub vox_materials: std::collections::HashMap<u32, Dict>,
    /// Cameras and palette order from the file (camera 0 is the view).
    pub vox_extra: crate::vox::Extra,
    /// The view as read from camera 0, to tell whether it has moved.
    pub vox_camera: Option<crate::gpu::Camera>,
    pub materials: Vec<GpuMaterial>,
    pub lights: Vec<GpuLight>,
    /// Sum of the power of all light faces.
    pub light_power: f32,
    pub settings: RenderSettings,
}

/// A palette and its materials.
pub type PaletteState = (Vec<[u8; 4]>, Vec<GpuMaterial>);

fn luminance(c: [f32; 4]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Collect every emissive voxel face that borders air or glass.
pub fn build_lights(v: &Volume, mats: &[GpuMaterial]) -> (Vec<GpuLight>, f32) {
    const MAX_FACES: usize = 1 << 20;
    let power = |idx: u8| {
        let m = &mats[idx as usize];
        if m.kind == MaterialKind::Emit as u32 { m.emit * luminance(m.color) } else { 0.0 }
    };
    let see_through = |p: IVec3| {
        if p.cmplt(IVec3::ZERO).any() || p.cmpge(v.dims).any() {
            return true;
        }
        let c = v.cells[(p.x + v.dims.x * (p.y + v.dims.y * p.z)) as usize];
        c == 0 || mats[c as usize].kind == MaterialKind::Glass as u32
    };
    let mut lights = Vec::new();
    let mut total = 0.0f32;
    // Emissive voxels in dense order, so the light list doesn't depend on
    // how the volume is stored.
    let mut emissive: Vec<(usize, u8)> = v.cells.filled().filter(|&(_, c)| power(c) > 0.0).collect();
    emissive.sort_unstable_by_key(|&(i, _)| i);
    for (i, c) in emissive {
        let pw = power(c);
        let p = v.position(i);
        for axis in 0..3 {
            for positive in [false, true] {
                let mut d = IVec3::ZERO;
                d[axis] = if positive { 1 } else { -1 };
                if !see_through(p + d) || lights.len() >= MAX_FACES {
                    continue;
                }
                let mut corner = p.as_vec3();
                if positive {
                    corner[axis] += 1.0;
                }
                total += pw;
                lights.push(GpuLight {
                    corner_cdf: [corner.x, corner.y, corner.z, total],
                    axis_sign_mat: [axis as u32, positive as u32, c as u32, 0],
                });
            }
        }
    }
    for l in &mut lights {
        l.corner_cdf[3] /= total;
    }
    (lights, total)
}

pub fn srgb_to_linear(c: Vec3) -> Vec3 {
    let f = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
    Vec3::new(f(c.x), f(c.y), f(c.z))
}

fn num(d: &Dict, k: &str) -> Option<f32> {
    d.get(k).and_then(|s| s.trim().parse().ok())
}

fn vec3(d: &Dict, k: &str) -> Option<Vec3> {
    let v: Vec<f32> = d.get(k)?.split_whitespace().filter_map(|s| s.parse().ok()).collect();
    (v.len() == 3).then(|| Vec3::new(v[0], v[1], v[2]))
}

fn color(d: &Dict, k: &str) -> Option<Vec3> {
    vec3(d, k).map(|c| srgb_to_linear(c / 255.0))
}

fn raw_color(d: &Dict, k: &str) -> Option<Vec3> {
    vec3(d, k).map(|c| c / 255.0)
}

fn flag(d: &Dict, k: &str) -> Option<bool> {
    d.get(k).map(|s| s.trim() == "1")
}

/// The render chunk for settings MagicaVoxel has no key for.
const VOXELY: &str = "_voxely";

fn linear_to_srgb(c: Vec3) -> Vec3 {
    let f = |v: f32| if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    Vec3::new(f(c.x), f(c.y), f(c.z))
}

fn rgb_255(c: Vec3) -> String {
    let c = (c.clamp(Vec3::ZERO, Vec3::ONE) * 255.0).round();
    format!("{} {} {}", c.x as u8, c.y as u8, c.z as u8)
}

/// A linear color as `color` reads it back.
fn srgb_255(c: Vec3) -> String {
    rgb_255(linear_to_srgb(c))
}

/// A scattering ratio as `raw_color` reads it back.
fn raw_255(c: Vec3) -> String {
    rgb_255(c)
}

impl RenderSettings {
    pub fn from_vox(f: &VoxFile) -> Self {
        let mut s = Self::default();
        if let Some(d) = f.render.get("_inf") {
            if let Some(a) = vec3_2(d, "_angle") {
                s.sun_pitch = a.0;
                s.sun_yaw = a.1;
            }
            s.sun_area = num(d, "_area").unwrap_or(s.sun_area);
            s.sun_intensity = num(d, "_i").unwrap_or(s.sun_intensity);
            s.sun_color = color(d, "_k").unwrap_or(s.sun_color);
            s.sun_disk = flag(d, "_disk").unwrap_or(s.sun_disk);
        }
        if let Some(d) = f.render.get("_uni") {
            s.sky_intensity = num(d, "_i").unwrap_or(s.sky_intensity);
            s.sky_color = color(d, "_k").unwrap_or(s.sky_color);
        }
        if let Some(d) = f.render.get("_env") {
            // Order of the sky tabs in the UI: uniform, atmosphere, image.
            s.env_mode = match num(d, "_mode").unwrap_or(0.0) as i32 {
                1 => EnvMode::Atmosphere,
                2 => EnvMode::Image,
                _ => EnvMode::Uniform,
            };
        }
        if let Some(d) = f.render.get("_ibl") {
            s.ibl_intensity = num(d, "_i").unwrap_or(s.ibl_intensity);
            s.ibl_rotation = num(d, "_rot").unwrap_or(s.ibl_rotation);
        }
        // Old files split the atmosphere into `_rayleigh` and `_mie`.
        if let Some(d) = f.render.get("_rayleigh") {
            s.rayleigh_density = num(d, "_d").unwrap_or(s.rayleigh_density);
            s.rayleigh_color = raw_color(d, "_k").unwrap_or(s.rayleigh_color);
        }
        if let Some(d) = f.render.get("_mie") {
            s.mie_density = num(d, "_d").unwrap_or(s.mie_density);
            s.mie_color = raw_color(d, "_k").unwrap_or(s.mie_color);
            s.mie_g = num(d, "_g").unwrap_or(s.mie_g);
        }
        if let Some(d) = f.render.get("_atm") {
            s.rayleigh_density = num(d, "_ray_d").unwrap_or(s.rayleigh_density);
            s.rayleigh_color = raw_color(d, "_ray_k").unwrap_or(s.rayleigh_color);
            s.mie_density = num(d, "_mie_d").unwrap_or(s.mie_density);
            s.mie_color = raw_color(d, "_mie_k").unwrap_or(s.mie_color);
            s.mie_g = num(d, "_mie_g").unwrap_or(s.mie_g);
            s.ozone_density = num(d, "_o3_d").unwrap_or(s.ozone_density);
            s.ozone_color = raw_color(d, "_o3_k").unwrap_or(s.ozone_color);
        }
        if let Some(d) = f.render.get("_fog") {
            if flag(d, "_enable") == Some(true) {
                s.fog_density = num(d, "_d").unwrap_or(0.0);
            }
            s.fog_color = color(d, "_k").unwrap_or(s.fog_color);
        }
        if let Some(d) = f.render.get("_fog_uni") {
            s.fog_density = num(d, "_d").unwrap_or(s.fog_density);
            s.fog_color = color(d, "_k").unwrap_or(s.fog_color);
            s.fog_g = num(d, "_g").unwrap_or(s.fog_g);
        }
        // Older files use `_len`, newer ones split it into `_lens` and `_film`.
        for key in ["_len", "_lens"] {
            if let Some(d) = f.render.get(key) {
                s.fov = num(d, "_fov").unwrap_or(s.fov);
                s.blades = num(d, "_blade_n").map(|v| v as u32).unwrap_or(s.blades);
                s.blade_rotation = num(d, "_blade_r").unwrap_or(s.blade_rotation);
                // Projection tabs in the UI: perspective, stereographic, panorama.
                s.projection = match num(d, "_proj").unwrap_or(0.0) as i32 {
                    1 => Projection::Stereographic,
                    2 => Projection::Panorama,
                    _ => Projection::Perspective,
                };
                s.aperture = num(d, "_aperture").or(num(d, "_dof")).unwrap_or(s.aperture);
                // Old files store exposure in stops.
                s.exposure = num(d, "_exp").map(f32::exp2).unwrap_or(s.exposure);
                s.gamma = num(d, "_gam").unwrap_or(s.gamma);
                s.vignette = num(d, "_vig").unwrap_or(s.vignette);
            }
        }
        if let Some(d) = f.render.get("_film") {
            // Newer files store a linear multiplier (inferred, not verified).
            s.exposure = num(d, "_expo").unwrap_or(s.exposure);
            s.gamma = num(d, "_gam").unwrap_or(s.gamma);
            s.vignette = num(d, "_vig").unwrap_or(s.vignette);
            s.aces = flag(d, "_aces").unwrap_or(s.aces);
        }
        if let Some(d) = f.render.get("_bloom") {
            s.bloom_mix = num(d, "_mix").unwrap_or(s.bloom_mix);
            s.bloom_scale = num(d, "_scale").unwrap_or(s.bloom_scale);
            s.bloom_aspect = num(d, "_aspect").unwrap_or(s.bloom_aspect);
            s.bloom_threshold = num(d, "_threshold").unwrap_or(s.bloom_threshold);
        }
        if let Some(d) = f.render.get("_ground") {
            s.ground_color = color(d, "_color").unwrap_or(s.ground_color);
            s.ground = flag(d, "_enable").unwrap_or(s.ground);
        }
        if let Some(d) = f.render.get("_bg") {
            s.bg_color = color(d, "_color").unwrap_or(s.bg_color);
            if flag(d, "_enable") == Some(true) {
                s.bg_mode = BgMode::Color;
            }
        }
        if let Some(d) = f.render.get("_edge") {
            s.edge_color = color(d, "_color").unwrap_or(s.edge_color);
            s.edge_width = num(d, "_width").unwrap_or(s.edge_width);
            s.show_edges = flag(d, "_enable").unwrap_or(s.show_edges);
        }
        if let Some(d) = f.render.get("_grid") {
            s.grid_color = color(d, "_color").unwrap_or(s.grid_color);
            s.grid_spacing = num(d, "_spacing").map(|v| (v as u32).clamp(1, 1024)).unwrap_or(s.grid_spacing);
            s.grid_width = num(d, "_width").unwrap_or(s.grid_width);
            s.show_grid = flag(d, "_enable").unwrap_or(s.show_grid);
        }
        // Newer files keep the display toggles here.
        if let Some(d) = f.render.get("_setting") {
            s.ground = flag(d, "_ground").unwrap_or(s.ground);
            s.show_grid = flag(d, "_grid").unwrap_or(s.show_grid);
            s.show_edges = flag(d, "_edge").unwrap_or(s.show_edges);
            if flag(d, "_bg_c") == Some(true) {
                s.bg_mode = BgMode::Color;
            }
            if flag(d, "_bg_a") == Some(true) {
                s.bg_mode = BgMode::Transparent;
            }
        }
        // Depth of field is opt-in: files store an aperture even when DOF is off.
        s.aperture = 0.0;
        if let Some(d) = f.render.get(VOXELY) {
            s.read_voxely(d);
        }
        s
    }

    /// Settings MagicaVoxel has no key for, saved by voxely.
    fn read_voxely(&mut self, d: &Dict) {
        let s = self;
        s.projection = match num(d, "_proj").map(|v| v as i32) {
            Some(1) => Projection::Orthographic,
            Some(2) => Projection::Stereographic,
            Some(3) => Projection::Panorama,
            Some(4) => Projection::Cylindrical,
            Some(_) => Projection::Perspective,
            None => s.projection,
        };
        s.aperture = num(d, "_dof").unwrap_or(s.aperture);
        s.bloom = flag(d, "_bloom").unwrap_or(s.bloom);
        s.denoise = flag(d, "_denoise").unwrap_or(s.denoise);
        s.denoise_view = match num(d, "_denoise_view").map(|v| v as i32) {
            Some(1) => DenoiseView::Original,
            Some(2) => DenoiseView::Split,
            Some(_) => DenoiseView::Denoised,
            None => s.denoise_view,
        };
        s.bg_mode = match num(d, "_bg").map(|v| v as i32) {
            Some(1) => BgMode::Color,
            Some(2) => BgMode::Transparent,
            Some(_) => BgMode::Sky,
            None => s.bg_mode,
        };
        s.max_diffuse = num(d, "_diffuse").map_or(s.max_diffuse, |v| v as u32);
        s.max_specular = num(d, "_specular").map_or(s.max_specular, |v| v as u32);
        s.max_scatter = num(d, "_scatter").map_or(s.max_scatter, |v| v as u32);
        s.clamp = num(d, "_clamp").unwrap_or(s.clamp);
        s.translucent_shadows = flag(d, "_tr_shadow").unwrap_or(s.translucent_shadows);
        s.grid_ground_only = flag(d, "_grid_ground").unwrap_or(s.grid_ground_only);
        if let Some(p) = d.get("_ibl_path").filter(|p| !p.is_empty()) {
            s.ibl_path = Some(PathBuf::from(p));
        }
    }

    /// Write these settings into a file's render chunks, MagicaVoxel's keys
    /// first, keeping any keys and chunks voxely doesn't know.
    pub fn write_vox(&self, dicts: &mut Vec<Dict>) {
        let s = self;
        let b = |v: bool| if v { "1" } else { "0" }.to_string();
        let n = |v: f32| v.to_string();
        let mut put = |ty: &str, kv: Vec<(&str, String)>| {
            let i = match dicts.iter().position(|d| d.get("_type").is_some_and(|t| t == ty)) {
                Some(i) => i,
                None => {
                    dicts.push(Dict::from([("_type".to_string(), ty.to_string())]));
                    dicts.len() - 1
                }
            };
            for (k, v) in kv {
                dicts[i].insert(k.to_string(), v);
            }
        };
        put("_inf", vec![
            ("_i", n(s.sun_intensity)),
            ("_k", srgb_255(s.sun_color)),
            ("_angle", format!("{} {}", s.sun_pitch, s.sun_yaw)),
            ("_area", n(s.sun_area)),
            ("_disk", b(s.sun_disk)),
        ]);
        put("_uni", vec![("_i", n(s.sky_intensity)), ("_k", srgb_255(s.sky_color))]);
        let mode = match s.env_mode {
            EnvMode::Uniform => 0,
            EnvMode::Atmosphere => 1,
            EnvMode::Image => 2,
        };
        put("_env", vec![("_mode", mode.to_string())]);
        put("_ibl", vec![("_i", n(s.ibl_intensity)), ("_rot", n(s.ibl_rotation))]);
        put("_atm", vec![
            ("_ray_d", n(s.rayleigh_density)),
            ("_ray_k", raw_255(s.rayleigh_color)),
            ("_mie_d", n(s.mie_density)),
            ("_mie_k", raw_255(s.mie_color)),
            ("_mie_g", n(s.mie_g)),
            ("_o3_d", n(s.ozone_density)),
            ("_o3_k", raw_255(s.ozone_color)),
        ]);
        put("_fog_uni", vec![("_d", n(s.fog_density)), ("_k", srgb_255(s.fog_color)), ("_g", n(s.fog_g))]);
        // MagicaVoxel's projection tabs; the rest are voxely's own.
        let proj = match s.projection {
            Projection::Stereographic => 1,
            Projection::Panorama => 2,
            _ => 0,
        };
        let mut lens = vec![("_proj", proj.to_string()), ("_fov", n(s.fov)), ("_blade_n", s.blades.to_string()), ("_blade_r", n(s.blade_rotation))];
        if s.aperture > 0.0 {
            lens.push(("_aperture", n(s.aperture)));
        }
        put("_lens", lens);
        put("_film", vec![("_expo", n(s.exposure)), ("_vig", n(s.vignette)), ("_aces", b(s.aces)), ("_gam", n(s.gamma))]);
        put("_bloom", vec![
            ("_mix", n(s.bloom_mix)),
            ("_scale", n(s.bloom_scale)),
            ("_aspect", n(s.bloom_aspect)),
            ("_threshold", n(s.bloom_threshold)),
        ]);
        put("_ground", vec![("_color", srgb_255(s.ground_color)), ("_enable", b(s.ground))]);
        put("_bg", vec![("_color", srgb_255(s.bg_color)), ("_enable", b(s.bg_mode == BgMode::Color))]);
        put("_edge", vec![("_color", srgb_255(s.edge_color)), ("_width", n(s.edge_width)), ("_enable", b(s.show_edges))]);
        put("_grid", vec![
            ("_color", srgb_255(s.grid_color)),
            ("_spacing", s.grid_spacing.to_string()),
            ("_width", n(s.grid_width)),
            ("_enable", b(s.show_grid)),
        ]);
        put("_setting", vec![
            ("_ground", b(s.ground)),
            ("_grid", b(s.show_grid)),
            ("_edge", b(s.show_edges)),
            ("_bg_c", b(s.bg_mode == BgMode::Color)),
            ("_bg_a", b(s.bg_mode == BgMode::Transparent)),
        ]);
        put(VOXELY, vec![
            ("_proj", (s.projection as u32).to_string()),
            ("_dof", n(s.aperture)),
            ("_bloom", b(s.bloom)),
            ("_denoise", b(s.denoise)),
            ("_denoise_view", (s.denoise_view as u32).to_string()),
            ("_bg", (s.bg_mode as u32).to_string()),
            ("_diffuse", s.max_diffuse.to_string()),
            ("_specular", s.max_specular.to_string()),
            ("_scatter", s.max_scatter.to_string()),
            ("_clamp", n(s.clamp)),
            ("_tr_shadow", b(s.translucent_shadows)),
            ("_grid_ground", b(s.grid_ground_only)),
            ("_ibl_path", s.ibl_path.as_ref().map(|p| p.display().to_string()).unwrap_or_default()),
        ]);
    }

    pub fn sun_dir(&self) -> Vec3 {
        let (p, y) = (self.sun_pitch.to_radians(), self.sun_yaw.to_radians());
        Vec3::new(p.cos() * y.cos(), p.cos() * y.sin(), p.sin()).normalize()
    }
}

fn vec3_2(d: &Dict, k: &str) -> Option<(f32, f32)> {
    let v: Vec<f32> = d.get(k)?.split_whitespace().filter_map(|s| s.parse().ok()).collect();
    (v.len() == 2).then(|| (v[0], v[1]))
}

pub fn material(idx: u32, rgba: [u8; 4], d: Option<&Dict>) -> GpuMaterial {
    let c = srgb_to_linear(Vec3::new(rgba[0] as f32, rgba[1] as f32, rgba[2] as f32) / 255.0);
    let mut m = GpuMaterial {
        color: [c.x, c.y, c.z, rgba[3] as f32 / 255.0],
        kind: MaterialKind::Diffuse as u32,
        rough: 0.1,
        metal: 0.0,
        ior: 1.3,
        emit: 0.0,
        trans: 0.0,
        density: 0.0,
        g: 0.0,
    };
    if idx == 0 {
        return m;
    }
    let Some(d) = d else { return m };
    let weight = num(d, "_weight").unwrap_or(1.0);
    m.rough = num(d, "_rough").unwrap_or(m.rough);
    // `_ri` is the refractive index; `_ior` stores it minus one.
    m.ior = num(d, "_ri").or(num(d, "_ior").map(|v| v + 1.0)).unwrap_or(m.ior);
    m.kind = match d.get("_type").map(String::as_str) {
        Some("_metal") => {
            m.metal = num(d, "_metal").unwrap_or(weight);
            MaterialKind::Metal
        }
        Some("_glass") => {
            m.trans = num(d, "_trans").or(num(d, "_alpha")).unwrap_or(weight);
            MaterialKind::Glass
        }
        Some("_emit") => {
            let flux = num(d, "_flux").unwrap_or(0.0);
            // Power is a log-scale multiplier on the base emission.
            m.emit = num(d, "_emit").unwrap_or(weight) * 10f32.powf(flux * 0.5);
            MaterialKind::Emit
        }
        Some("_blend") => {
            m.metal = num(d, "_metal").unwrap_or(0.0);
            m.trans = num(d, "_trans").or(num(d, "_alpha")).unwrap_or(0.0);
            MaterialKind::Blend
        }
        Some("_media") => {
            // Density is per voxel; scale so the UI's small values read as clouds.
            m.density = num(d, "_d").unwrap_or(0.05) * 10.0;
            m.g = num(d, "_g").or(num(d, "_g0")).unwrap_or(0.0);
            MaterialKind::Media
        }
        _ => MaterialKind::Diffuse,
    } as u32;
    m
}

/// `rCAM` key holding voxely's own orbit camera.
const VX_CAM: &str = "_vx_cam";

fn fmt(v: f32) -> String {
    // Short, round-trippable decimals.
    let s = format!("{v:.4}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// The `MATL` dictionary for a material, or `None` for plain diffuse.
pub fn material_dict(m: &GpuMaterial) -> Option<Dict> {
    let mut d = Dict::new();
    let mut set = |k: &str, v: String| {
        d.insert(k.to_string(), v);
    };
    let kind = match m.kind {
        k if k == MaterialKind::Metal as u32 => "_metal",
        k if k == MaterialKind::Glass as u32 => "_glass",
        k if k == MaterialKind::Emit as u32 => "_emit",
        k if k == MaterialKind::Blend as u32 => "_blend",
        k if k == MaterialKind::Media as u32 => "_media",
        _ => return None,
    };
    set("_type", kind.into());
    set("_rough", fmt(m.rough));
    set("_ri", fmt(m.ior));
    set("_ior", fmt(m.ior - 1.0));
    match kind {
        "_metal" => set("_metal", fmt(m.metal)),
        "_glass" => set("_trans", fmt(m.trans)),
        "_blend" => {
            set("_metal", fmt(m.metal));
            set("_trans", fmt(m.trans));
        }
        "_emit" => {
            // Inverse of `material`: MagicaVoxel's power is a whole number
            // 0..=4, so take the least that keeps emission within 1.
            let flux = (0..=4).find(|&f| m.emit <= 10f32.powf(f as f32 * 0.5) * 1.0001).unwrap_or(4);
            set("_emit", fmt(m.emit / 10f32.powf(flux as f32 * 0.5)));
            set("_flux", flux.to_string());
        }
        "_media" => {
            set("_d", fmt(m.density / 10.0));
            set("_g", fmt(m.g));
        }
        _ => {}
    }
    Some(d)
}

impl Scene {
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let mut ids: Vec<u32> = (1..256).chain(self.vox_materials.keys().copied()).collect();
        ids.sort();
        ids.dedup();
        let mats: Vec<(u32, Dict)> = ids.into_iter().filter_map(|i| self.material_out(i).map(|d| (i, d))).collect();
        let mut render = self.render_dicts.clone();
        self.settings.write_vox(&mut render);
        crate::vox::write_with(path, &self.synced_world(), &self.palette, &mats, &self.notes, &render, &self.vox_extra)
    }

    /// The `MATL` dictionary saved for color `i`.
    fn material_out(&self, i: u32) -> Option<Dict> {
        let orig = self.vox_materials.get(&i);
        let Some(m) = self.materials.get(i as usize).filter(|_| (1..256).contains(&i)) else { return orig.cloned() };
        let Some(orig) = orig else { return material_dict(m) };
        // Unchanged: exactly as read.
        let mut was = material(i, self.palette[i as usize], Some(orig));
        was.color = m.color;
        if bytemuck::bytes_of(&was) == bytemuck::bytes_of(m) {
            return Some(orig.clone());
        }
        let mut d = orig.clone();
        match material_dict(m) {
            Some(new) => d.extend(new),
            None => {
                d.insert("_type".into(), "_diffuse".into());
                d.insert("_rough".into(), fmt(m.rough));
            }
        }
        Some(d)
    }

    /// Camera 0 of the file as an orbit camera around a target in
    /// `volume`'s coordinates, if the file has one voxely wrote.
    pub fn file_camera(&self) -> Option<crate::gpu::Camera> {
        let d = &self.vox_extra.cameras.iter().find(|(id, _)| *id == 0)?.1;
        let v: Vec<f32> = d.get(VX_CAM)?.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        let [yaw, pitch, dist, x, y, z] = v[..] else { return None };
        let target = Vec3::new(x, y, z) - self.origin.as_vec3();
        [yaw, pitch, dist].iter().chain(&target.to_array()).all(|v| v.is_finite()).then_some(crate::gpu::Camera { target, yaw, pitch, dist: dist.max(0.1) })
    }

    /// Store `cam` (target in `volume`'s coordinates) as camera 0, unless
    /// it is still the camera read from the file, which is kept as it was.
    pub fn store_camera(&mut self, cam: &crate::gpu::Camera) {
        if self.vox_camera.as_ref() == Some(cam) {
            return;
        }
        let at = cam.target + self.origin.as_vec3();
        let mut d = self.vox_extra.cameras.iter().find(|(id, _)| *id == 0).map_or_else(Dict::new, |(_, d)| d.clone());
        let ortho = self.settings.projection == Projection::Orthographic;
        let deg = |r: f32| fmt(r.to_degrees());
        for (k, v) in [
            ("_mode", if ortho { "orth".to_string() } else { "pers".to_string() }),
            ("_focus", format!("{} {} {}", fmt(at.x), fmt(at.y), fmt(at.z))),
            // Pitch then yaw, as in the sun's `_angle`.
            ("_angle", format!("{} {} 0", deg(cam.pitch), deg(cam.yaw))),
            ("_radius", fmt(cam.dist)),
            ("_fov", fmt(self.settings.fov)),
            // voxely's exact orbit, in world coordinates.
            (VX_CAM, format!("{} {} {} {} {} {}", cam.yaw, cam.pitch, cam.dist, at.x, at.y, at.z)),
        ] {
            d.insert(k.to_string(), v);
        }
        d.entry("_frustum".into()).or_insert_with(|| "0.4".into());
        self.vox_extra.cameras.retain(|(id, _)| *id != 0);
        self.vox_extra.cameras.insert(0, (0, d));
    }

    pub fn from_vox(f: &VoxFile) -> Result<Self> {
        let world = World::from_vox(f);
        let models = world.models();
        if !models.iter().any(|&m| world.visible(m)) {
            bail!("scene has no visible models");
        }
        // Storage, and the path tracer's brick pool, grow with the bricks
        // holding voxels, not with the models' boxes.
        let bricks: usize = models.iter().filter_map(|&m| world.node(m)?.volume()).map(|v| v.cells.allocated()).sum();
        if bricks > MAX_BRICKS {
            bail!("scene too large: voxels in {bricks} 8x8x8 bricks (the limit is {MAX_BRICKS})");
        }
        let render_dicts = f.render_order.iter().filter(|k| *k != crate::vox::ANIM_TYPE).filter_map(|k| f.render.get(k).cloned()).collect();
        let mut s = Self::assemble(world, f.palette.clone(), &f.materials, render_dicts, RenderSettings::from_vox(f));
        s.notes = f.notes.clone();
        s.vox_extra = f.extra.clone();
        Ok(s)
    }

    /// An empty canvas with the default palette, like MagicaVoxel's new model.
    pub fn empty(dims: IVec3) -> Self {
        Self::assemble(World::single(Volume::new(dims)), crate::vox::default_palette(), &Default::default(), Vec::new(), RenderSettings::default())
    }

    /// A scene of `world`'s models with `palette` and default materials.
    pub fn from_world(world: World, palette: Vec<[u8; 4]>) -> Self {
        Self::assemble(world, palette, &Default::default(), Vec::new(), RenderSettings::default())
    }

    /// Opens in edit mode when there is one model, else in World mode.
    fn assemble(
        world: World,
        palette: Vec<[u8; 4]>,
        mats: &std::collections::HashMap<u32, Dict>,
        render_dicts: Vec<Dict>,
        settings: RenderSettings,
    ) -> Self {
        let materials: Vec<GpuMaterial> = (0..256u32).map(|i| material(i, palette[i as usize], mats.get(&i))).collect();
        let models = world.models();
        let mut s = Scene {
            volume: Volume::new(IVec3::ONE),
            world,
            mode: Mode::World,
            origin: IVec3::ZERO,
            owners: Grid::new(IVec3::ZERO),
            context: None,
            picked: Vec::new(),
            history: Vec::new(),
            future: Vec::new(),
            steps: 0,
            palette_steps: Vec::new(),
            palette_dirty: false,
            palette,
            notes: Vec::new(),
            render_dicts,
            vox_materials: mats.clone(),
            vox_extra: Default::default(),
            vox_camera: None,
            materials,
            lights: Vec::new(),
            light_power: 0.0,
            settings,
        };
        match models.as_slice() {
            [only] => s.view(Mode::Edit(*only)),
            _ => s.view(Mode::World),
        }
        s.rebuild_lights();
        s
    }

    // ---- the scene tree ----

    pub fn editing(&self) -> Option<usize> {
        match self.mode {
            Mode::Edit(id) => Some(id),
            Mode::World => None,
        }
    }

    /// Rebuild `volume` for `mode` from the tree, dropping unsaved edits.
    fn view(&mut self, mode: Mode) {
        let mode = match mode {
            Mode::Edit(id) if self.world.node(id).is_some_and(|n| !n.is_group()) => mode,
            _ => Mode::World,
        };
        self.mode = mode;
        match mode {
            Mode::Edit(id) => {
                let (v, origin) = self.world.checkout(id).expect("a model");
                // A freely rotated part is edited upright and on its own:
                // around the others it would sit at the wrong angle.
                self.context = if self.world.tilted(id) { None } else { self.world.composite(Some(id)).map(|c| (c.volume, c.origin - origin)) };
                (self.volume, self.origin) = (v, origin);
                self.owners = Grid::new(IVec3::ZERO);
                self.picked = vec![id];
            }
            Mode::World => {
                self.context = None;
                match self.world.composite(None) {
                    Some(c) => (self.volume, self.origin, self.owners) = (c.volume, c.origin, c.owners),
                    None => {
                        self.volume = Volume::new(IVec3::ONE);
                        self.owners = Grid::new(IVec3::ONE);
                    }
                }
                let w = &self.world;
                self.picked.retain(|&p| w.node(p).is_some());
            }
        }
    }

    /// The tree with the edited model's current voxels.
    pub fn synced_world(&self) -> std::borrow::Cow<'_, World> {
        match self.mode {
            Mode::Edit(id) => {
                let mut w = self.world.clone();
                w.checkin(id, &self.volume, self.origin);
                std::borrow::Cow::Owned(w)
            }
            Mode::World => std::borrow::Cow::Borrowed(&self.world),
        }
    }

    /// Write the edited model back into the tree (one undo step) if it
    /// changed.
    fn store(&mut self) {
        let Mode::Edit(id) = self.mode else { return };
        let unchanged = self.world.checkout(id).is_some_and(|(v, o)| o == self.origin && v.dims == self.volume.dims && v.cells == self.volume.cells);
        if !unchanged {
            self.remember();
            self.world.checkin(id, &self.volume, self.origin);
        }
    }

    fn remember(&mut self) {
        // Steps being dropped from the redo list take their palettes along.
        let at = self.position();
        self.palette_steps.retain(|s| s.0 <= at);
        self.history.push(self.world.clone());
        if self.history.len() > HISTORY {
            self.history.remove(0);
        }
        self.future.clear();
        self.steps += 1;
    }

    /// Where undo is in the tree's steps: steps pushed minus steps undone.
    fn position(&self) -> u64 {
        self.steps - self.future.len() as u64
    }

    /// The palette and materials, to restore with an undo step.
    pub fn palette_state(&self) -> PaletteState {
        (self.palette.clone(), self.materials.clone())
    }

    /// The last tree step also changed the palette from `before`: undoing
    /// it puts `before` back, redoing it the palette as it is now.
    pub fn palette_step(&mut self, before: PaletteState) {
        let at = self.position();
        self.palette_steps.retain(|s| s.0 != at);
        self.palette_steps.push((at, before, self.palette_state()));
    }

    /// Tree undo steps pushed so far, for `squash_tree`.
    pub fn tree_steps(&self) -> u64 {
        self.steps
    }

    /// Fold every tree change since `tree_steps()` was `steps`, plus the
    /// edited model's unsaved voxels, into one undo step back to `before`.
    /// Callers reset the voxel editor (its history is now in the tree).
    pub fn squash_tree(&mut self, steps: u64, before: World) {
        self.store();
        let n = (self.steps - steps) as usize;
        if n == 0 {
            return;
        }
        self.history.truncate(self.history.len().saturating_sub(n));
        self.remember();
        *self.history.last_mut().unwrap() = before;
    }

    /// Edit model `id` (or the model of a group's first model).
    pub fn edit(&mut self, id: usize) -> bool {
        let id = match self.world.node(id) {
            Some(n) if n.is_group() => {
                let mut first = None;
                self.world.visit(id, &mut |m, n| {
                    if first.is_none() && !n.is_group() {
                        first = Some(m);
                    }
                });
                match first {
                    Some(m) => m,
                    None => return false,
                }
            }
            Some(_) => id,
            None => return false,
        };
        self.store();
        self.view(Mode::Edit(id));
        true
    }

    /// Replace the edited model's canvas with `volume`, whose cell 0 sits at
    /// cell `at` of the current one, as one tree undo step: the voxels,
    /// pivot, sockets and attached parts stay where they are in the world.
    /// Callers reset the voxel editor. False outside edit mode.
    pub fn recanvas(&mut self, volume: &Volume, at: IVec3) -> bool {
        let Mode::Edit(id) = self.mode else { return false };
        let origin = self.origin + at;
        // Quiet: nothing moved, so no animation keys.
        self.change_quiet(|w| w.checkin(id, volume, origin));
        true
    }

    /// Leave edit mode for the World view.
    pub fn world_mode(&mut self) {
        self.store();
        self.view(Mode::World);
    }

    /// Change the tree as one undo step, then refresh the view.
    pub fn change<R>(&mut self, f: impl FnOnce(&mut World) -> R) -> R {
        self.begin_change();
        self.apply(f)
    }

    pub fn can_undo_tree(&self) -> bool {
        !self.history.is_empty()
    }

    pub fn can_redo_tree(&self) -> bool {
        !self.future.is_empty()
    }

    /// Step the tree back (or forward). The view is rebuilt, so callers
    /// reset the voxel editor.
    pub fn undo_tree(&mut self, forward: bool) -> bool {
        self.store();
        let at = self.position();
        let (from, to) = if forward { (&mut self.future, &mut self.history) } else { (&mut self.history, &mut self.future) };
        let Some(w) = from.pop() else { return false };
        to.push(std::mem::replace(&mut self.world, w));
        // The step's palette change, if it made one.
        let step = if forward { at + 1 } else { at };
        if let Some((_, before, after)) = self.palette_steps.iter().find(|s| s.0 == step) {
            (self.palette, self.materials) = if forward { after.clone() } else { before.clone() };
            self.palette_dirty = true;
        }
        self.view(self.mode);
        true
    }

    /// World mode: cells of the picked nodes, for highlighting.
    pub fn highlight(&self) -> Option<Mask> {
        if self.mode != Mode::World || self.picked.is_empty() {
            return None;
        }
        let mut ids = std::collections::HashSet::new();
        for &p in &self.picked {
            self.world.visit(p, &mut |i, _| {
                ids.insert(i as u32);
            });
        }
        let mut mask = Mask::new(self.owners.dims());
        for (i, o) in self.owners.filled() {
            if ids.contains(&o) {
                mask.set(i, true);
            }
        }
        Some(mask)
    }

    /// Start a tree change that `apply` continues (one undo step, e.g. a
    /// drag).
    pub fn begin_change(&mut self) {
        self.store();
        self.remember();
    }

    /// Change the tree without a new undo step, then refresh the view.
    pub fn apply<R>(&mut self, f: impl FnOnce(&mut World) -> R) -> R {
        let r = f(&mut self.world);
        // When animating, moves and turns record keys on this frame.
        if let Some(before) = self.history.last() {
            self.world.autokey(before);
        }
        self.view(self.mode);
        r
    }

    /// Change the tree as one undo step without recording keys (for
    /// changes to the animation itself).
    pub fn change_quiet<R>(&mut self, f: impl FnOnce(&mut World) -> R) -> R {
        self.begin_change();
        let r = f(&mut self.world);
        self.view(self.mode);
        r
    }

    /// Show another frame of the animation (edits so far are kept).
    pub fn set_frame(&mut self, frame: u32) {
        self.store();
        self.world.set_frame(frame);
        self.view(self.mode);
    }

    /// Recolor every model through `remap` (palette sorting, swapping,
    /// packing), including flipbook drawings and the tree's undo steps, so
    /// undoing later brings back the same colors.
    pub fn remap(&mut self, remap: &[u8]) {
        let fix = |v: &mut Volume| v.cells.update_filled(|c| *c = remap[*c as usize]);
        fix(&mut self.volume);
        if let Some((ctx, _)) = &mut self.context {
            fix(ctx);
        }
        for w in std::iter::once(&mut self.world).chain(&mut self.history).chain(&mut self.future) {
            for n in w.nodes.iter_mut().flatten() {
                if let crate::world::Kind::Model(v) = &mut n.kind {
                    fix(v);
                }
                n.flip.iter_mut().filter_map(|s| s.volume.as_mut()).for_each(fix);
            }
        }
    }

    /// Which palette entries any model, drawing or the edited part uses.
    pub fn used_colors(&self) -> Vec<bool> {
        let mut used = vec![false; 256];
        let mut mark = |v: &Volume| v.cells.filled().for_each(|(_, c)| used[c as usize] = true);
        mark(&self.volume);
        for n in self.world.nodes.iter().flatten() {
            if let crate::world::Kind::Model(v) = &n.kind {
                mark(v);
            }
            n.flip.iter().filter_map(|s| s.volume.as_ref()).for_each(&mut mark);
        }
        used[0] = false;
        used
    }

    /// Every visible model flattened (for exports and rendering).
    pub fn flat(&self) -> std::borrow::Cow<'_, Volume> {
        match &self.context {
            Some((ctx, offset)) => {
                let lo = offset.min(IVec3::ZERO);
                let hi = (*offset + ctx.dims).max(self.volume.dims);
                let mut out = Volume::new(hi - lo);
                for (v, at) in [(ctx, *offset - lo), (&self.volume, -lo)] {
                    for (p, c) in v.filled() {
                        let j = out.index(p + at);
                        out.set_index(j, c);
                    }
                }
                std::borrow::Cow::Owned(out)
            }
            None => std::borrow::Cow::Borrowed(&self.volume),
        }
    }

    /// Where `flat()`'s cell 0 sits relative to `volume`.
    pub fn flat_offset(&self) -> IVec3 {
        self.context.as_ref().map_or(IVec3::ZERO, |(_, o)| o.min(IVec3::ZERO))
    }

    /// Rebuild the emissive face list after edits.
    pub fn rebuild_lights(&mut self) {
        (self.lights, self.light_power) = build_lights(&self.flat(), &self.materials);
    }

    /// Recompute a material's linear color after its palette entry changed.
    pub fn sync_color(&mut self, idx: usize) {
        let p = self.palette[idx];
        let c = srgb_to_linear(Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32) / 255.0);
        self.materials[idx].color = [c.x, c.y, c.z, p[3] as f32 / 255.0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_reload_round_trips() {
        let mut s = Scene::empty(IVec3::new(300, 20, 10));
        // Voxels in two 256-wide tiles, one emissive, one glass.
        for (p, c) in [(IVec3::new(1, 2, 3), 10u8), (IVec3::new(280, 19, 9), 20), (IVec3::new(0, 0, 0), 30)] {
            let i = s.volume.index(p);
            s.volume.set_index(i, c);
        }
        s.materials[20].kind = MaterialKind::Emit as u32;
        s.materials[20].emit = 31.6;
        s.materials[30].kind = MaterialKind::Glass as u32;
        s.materials[30].trans = 0.8;
        s.palette[10] = [1, 2, 3, 255];

        let path = std::env::temp_dir().join(format!("voxely-roundtrip-{}.vox", std::process::id()));
        s.save(&path).unwrap();
        let back = Scene::from_vox(&crate::vox::VoxFile::load(&path).unwrap()).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(back.volume.dims, s.volume.dims);
        assert_eq!(back.volume.cells, s.volume.cells);
        // The 256-wide tiles come back as the one part they were.
        assert_eq!(back.world.models().len(), 1);
        assert_eq!(back.editing(), back.world.models().first().copied());
        assert_eq!(back.palette[10], [1, 2, 3, 255]);
        assert_eq!(back.materials[20].kind, MaterialKind::Emit as u32);
        assert!((back.materials[20].emit - 31.6).abs() < 0.05, "{}", back.materials[20].emit);
        assert_eq!(back.materials[30].kind, MaterialKind::Glass as u32);
        assert!((back.materials[30].trans - 0.8).abs() < 1e-4);
    }

    /// Every chunk of type `id` in a `.vox` file, raw.
    fn chunks(path: &std::path::Path, id: &[u8; 4]) -> Vec<Vec<u8>> {
        let b = std::fs::read(path).unwrap();
        let (mut o, mut out) = (20, Vec::new());
        while o + 12 <= b.len() {
            let n = i32::from_le_bytes(b[o + 4..o + 8].try_into().unwrap()) as usize;
            if &b[o..o + 4] == id {
                out.push(b[o + 12..o + 12 + n].to_vec());
            }
            o += 12 + n;
        }
        out
    }

    #[test]
    fn magicavoxel_chunks_save_back_unchanged() {
        let dir = std::env::temp_dir();
        let (a, b) = (dir.join(format!("voxely-chunks-a-{}.vox", std::process::id())), dir.join(format!("voxely-chunks-b-{}.vox", std::process::id())));
        let mut v = Volume::new(IVec3::splat(4));
        v.set(IVec3::ONE, 3);
        let mut palette = crate::vox::default_palette();
        palette[0] = [4, 62, 76, 255];
        let d = |kv: &[(&str, &str)]| -> Dict { kv.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect() };
        // Keys voxely doesn't use, in MagicaVoxel's order, and its emission.
        let mats = vec![
            (1, d(&[("_type", "_diffuse"), ("_weight", "1"), ("_rough", "0.1"), ("_spec", "0.5"), ("_ior", "0.3"), ("_att", "0"), ("_g0", "-0.5"), ("_flux", "0"), ("_ldr", "0")])),
            (2, d(&[("_type", "_metal"), ("_rough", "0.2"), ("_metal", "0.8"), ("_sp", "0.6"), ("_ior", "0.3")])),
            (3, d(&[("_type", "_emit"), ("_emit", "0.4"), ("_flux", "3"), ("_ldr", "0.5")])),
            (4, d(&[("_type", "_media"), ("_d", "0.03"), ("_g", "0.2"), ("_media_type", "_scatter")])),
        ];
        let extra = crate::vox::Extra {
            cameras: vec![(0, d(&[("_mode", "pers"), ("_focus", "1 2 3"), ("_angle", "30 40 0"), ("_radius", "90"), ("_frustum", "0.4"), ("_fov", "45")])), (1, d(&[("_mode", "orth"), ("_radius", "50")]))],
            imap: Some((0..=255u8).rev().collect()),
        };
        crate::vox::write_with(&a, &World::single(v), &palette, &mats, &[], &[], &extra).unwrap();
        let s = Scene::from_vox(&crate::vox::VoxFile::load(&a).unwrap()).unwrap();
        s.save(&b).unwrap();
        for id in [b"MATL", b"rCAM", b"IMAP", b"RGBA"] {
            assert_eq!(chunks(&a, id), chunks(&b, id), "{}", String::from_utf8_lossy(id));
        }
        std::fs::remove_file(&a).ok();
        std::fs::remove_file(&b).ok();
    }

    #[test]
    fn edited_emission_saves_magicavoxel_power() {
        let mut s = Scene::empty(IVec3::splat(4));
        s.volume.set(IVec3::ZERO, 5);
        s.materials[5].kind = MaterialKind::Emit as u32;
        s.materials[5].emit = 30.0;
        let d = material_dict(&s.materials[5]).unwrap();
        // A whole-number power, emission within 1, as MagicaVoxel's sliders.
        assert_eq!(d["_flux"], "3");
        assert!(d["_emit"].parse::<f32>().unwrap() <= 1.0);
        let back = save_and_load(&s);
        assert!((back.materials[5].emit - 30.0).abs() < 0.01, "{}", back.materials[5].emit);
    }

    #[test]
    fn camera_is_saved_and_restored() {
        let mut s = Scene::empty(IVec3::splat(4));
        s.volume.set(IVec3::ZERO, 5);
        let cam = crate::gpu::Camera { target: Vec3::new(1.5, 2.0, -3.0), yaw: 0.3, pitch: 0.7, dist: 42.0 };
        s.store_camera(&cam);
        let mut back = save_and_load(&s);
        assert_eq!(back.file_camera(), Some(cam));
        // An unmoved camera keeps the file's own dictionary.
        back.vox_camera = back.file_camera();
        back.vox_extra.cameras[0].1.insert("_custom".into(), "1".into());
        let before = back.vox_extra.cameras.clone();
        back.store_camera(&cam);
        assert_eq!(back.vox_extra.cameras, before);
    }

    #[test]
    fn ramp_names_round_trip() {
        let mut s = Scene::empty(IVec3::splat(4));
        s.volume.set_index(0, 3);
        s.notes = vec![String::new(); 32];
        s.notes[0] = "skin".into();
        s.notes[31] = "glow".into();
        let back = save_and_load(&s);
        assert_eq!(back.notes, s.notes);
        // No names, no NOTE chunk.
        s.notes.clear();
        assert!(save_and_load(&s).notes.is_empty());
    }

    #[test]
    fn remap_reaches_tree_history() {
        let mut s = Scene::empty(IVec3::splat(4));
        s.volume.set_index(0, 3);
        s.store();
        s.volume.set_index(1, 4);
        // The undo step holds the model before this edit: just color 3.
        s.store();
        let mut remap: [u8; 256] = std::array::from_fn(|i| i as u8);
        remap[3] = 9;
        remap[4] = 10;
        s.remap(&remap);
        assert_eq!(s.volume.cells.to_vec()[..2], [9, 10]);
        assert!(s.used_colors()[9] && !s.used_colors()[3]);
        let old = s.history.last().unwrap();
        let cells: Vec<u8> = old.nodes.iter().flatten().filter_map(|n| n.volume()).flat_map(|v| v.cells.iter().copied()).filter(|&c| c != 0).collect();
        assert_eq!(cells, vec![9], "the undo step was recolored too");
    }

    fn save_and_load(s: &Scene) -> Scene {
        let path = std::env::temp_dir().join(format!("voxely-settings-{}-{:?}.vox", std::process::id(), std::thread::current().id()));
        s.save(&path).unwrap();
        let back = Scene::from_vox(&crate::vox::VoxFile::load(&path).unwrap()).unwrap();
        std::fs::remove_file(&path).ok();
        back
    }

    #[test]
    fn render_settings_round_trip() {
        let mut scene = Scene::empty(IVec3::splat(4));
        let d = RenderSettings::default();
        // Every field away from its default.
        scene.settings = RenderSettings {
            sun_pitch: 12.5,
            sun_yaw: 200.0,
            sun_area: 0.3,
            sun_intensity: 1.7,
            sun_color: Vec3::new(1.0, 0.5, 0.25),
            sun_disk: true,
            sky_intensity: 1.3,
            sky_color: Vec3::new(0.2, 0.4, 0.8),
            env_mode: EnvMode::Image,
            ibl_path: Some(PathBuf::from("/some/sky.hdr")),
            ibl_intensity: 2.5,
            ibl_rotation: 0.25,
            rayleigh_density: 1.1,
            rayleigh_color: Vec3::new(0.3, 0.6, 0.9),
            mie_density: 0.9,
            mie_color: Vec3::new(0.9, 0.8, 0.7),
            mie_g: 0.6,
            ozone_density: 0.5,
            ozone_color: Vec3::new(0.4, 0.9, 0.5),
            fog_density: 0.8,
            fog_color: Vec3::new(0.7, 0.7, 0.9),
            fog_g: -0.3,
            translucent_shadows: false,
            fov: 70.0,
            projection: Projection::Cylindrical,
            aperture: 0.75,
            blades: 6,
            blade_rotation: 30.0,
            exposure: 2.5,
            gamma: 2.8,
            aces: true,
            vignette: 0.4,
            ground: false,
            ground_color: Vec3::new(0.1, 0.2, 0.3),
            bloom: true,
            bloom_mix: 0.7,
            bloom_scale: 0.3,
            bloom_aspect: 0.6,
            bloom_threshold: 2.0,
            max_diffuse: 9,
            max_specular: 12,
            max_scatter: 6,
            clamp: 300.0,
            denoise: true,
            denoise_view: DenoiseView::Split,
            show_edges: true,
            edge_color: Vec3::new(1.0, 0.0, 0.0),
            edge_width: 0.4,
            show_grid: true,
            grid_color: Vec3::new(0.0, 1.0, 0.0),
            grid_spacing: 8,
            grid_width: 0.2,
            grid_ground_only: true,
            bg_mode: BgMode::Transparent,
            bg_color: Vec3::new(0.5, 0.0, 0.5),
        };
        let s = &scene.settings;
        macro_rules! all_changed {
            ($($f:ident),*) => {$( assert_ne!(s.$f, d.$f, stringify!($f)); )*};
        }
        all_changed!(sun_pitch, sun_yaw, sun_area, sun_intensity, sun_color, sun_disk, sky_intensity, sky_color, env_mode, ibl_path,
            ibl_intensity, ibl_rotation, rayleigh_density, rayleigh_color, mie_density, mie_color, mie_g, ozone_density, ozone_color,
            fog_density, fog_color, fog_g, translucent_shadows, fov, projection, aperture, blades, blade_rotation, exposure, gamma, aces,
            vignette, ground, ground_color, bloom, bloom_mix, bloom_scale, bloom_aspect, bloom_threshold, max_diffuse, max_specular,
            max_scatter, clamp, denoise, denoise_view, show_edges, edge_color, edge_width, show_grid, grid_color, grid_spacing,
            grid_width, grid_ground_only, bg_mode, bg_color);

        let mut back = save_and_load(&scene).settings;
        // Colors are stored as 8-bit; compare them loosely, the rest exactly.
        for (got, want) in [
            (&mut back.sun_color, s.sun_color),
            (&mut back.sky_color, s.sky_color),
            (&mut back.rayleigh_color, s.rayleigh_color),
            (&mut back.mie_color, s.mie_color),
            (&mut back.ozone_color, s.ozone_color),
            (&mut back.fog_color, s.fog_color),
            (&mut back.ground_color, s.ground_color),
            (&mut back.edge_color, s.edge_color),
            (&mut back.grid_color, s.grid_color),
            (&mut back.bg_color, s.bg_color),
        ] {
            assert!(got.abs_diff_eq(want, 0.01), "{got} vs {want}");
            *got = want;
        }
        assert_eq!(&back, s);
    }

    #[test]
    fn edited_magicavoxel_file_keeps_changes() {
        // Open a file MagicaVoxel wrote, change settings, save, reopen.
        let path = std::path::Path::new(env!("HOME")).join("Downloads/MagicaVoxel-0.99.6.2-macos-10.15/vox/monu10.vox");
        if !path.exists() {
            eprintln!("no MagicaVoxel samples; skipping");
            return;
        }
        let mut scene = Scene::from_vox(&crate::vox::VoxFile::load(&path).unwrap()).unwrap();
        let before = scene.settings.clone();
        scene.settings.sun_pitch = 17.0;
        scene.settings.exposure = 3.5;
        scene.settings.fog_density = 0.6;
        scene.settings.show_grid = true;
        let back = save_and_load(&scene).settings;
        assert_eq!((back.sun_pitch, back.exposure, back.fog_density, back.show_grid), (17.0, 3.5, 0.6, true));
        // Untouched settings keep the file's values.
        assert_eq!((back.sun_yaw, back.fov, back.env_mode, back.gamma), (before.sun_yaw, before.fov, before.env_mode, before.gamma));
    }

    #[test]
    fn color_and_off_settings_round_trip() {
        // The other display modes, and settings switched back off.
        let mut scene = Scene::empty(IVec3::splat(4));
        scene.settings.bg_mode = BgMode::Color;
        scene.settings.projection = Projection::Panorama;
        scene.settings.env_mode = EnvMode::Atmosphere;
        let back = save_and_load(&scene).settings;
        assert_eq!(back.bg_mode, BgMode::Color);
        assert_eq!(back.projection, Projection::Panorama);
        assert_eq!(back.env_mode, EnvMode::Atmosphere);
        assert_eq!(back.aperture, 0.0);
        assert!(back.ground && !back.show_grid && !back.show_edges && !back.bloom);
    }

    #[test]
    fn magicavoxel_display_keys_load() {
        // Grid, edge and background as MagicaVoxel 0.99.6 writes them.
        let mut f = crate::vox::VoxFile::default();
        let dict = |kv: &[(&str, &str)]| kv.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect::<Dict>();
        f.render.insert("_grid".into(), dict(&[("_type", "_grid"), ("_color", "108 108 108"), ("_spacing", "16"), ("_width", "0.1"), ("_enable", "0")]));
        f.render.insert("_edge".into(), dict(&[("_type", "_edge"), ("_color", "0 0 0"), ("_width", "0.2"), ("_enable", "0")]));
        f.render.insert("_setting".into(), dict(&[("_type", "_setting"), ("_grid", "1"), ("_edge", "1"), ("_bg_c", "1")]));
        let s = RenderSettings::from_vox(&f);
        assert!(s.show_grid && s.show_edges);
        assert_eq!(s.grid_spacing, 16);
        assert!((s.edge_width - 0.2).abs() < 1e-6);
        assert_eq!(s.bg_mode, BgMode::Color);
    }

    /// The pre-tree flattener, kept as the reference for placement.
    fn reference(f: &VoxFile) -> (IVec3, Vec<(IVec3, u8)>) {
        use crate::vox::Node;
        use glam::Mat3;
        fn walk(f: &VoxFile, id: i32, rot: Mat3, trans: Vec3, out: &mut Vec<(IVec3, u8)>, lo: &mut IVec3) {
            match f.nodes.get(&id) {
                Some(Node::Transform { child, rotation, translation, .. }) => {
                    walk(f, *child, rot * *rotation, rot * translation.as_vec3() + trans, out, lo)
                }
                Some(Node::Group { children, .. }) => children.iter().for_each(|c| walk(f, *c, rot, trans, out, lo)),
                Some(Node::Shape { models, .. }) => {
                    let m = &f.models[models[0] as usize];
                    let half = (m.size / 2).as_vec3();
                    let place = |v: Vec3| (rot * (v + 0.5 - half) + trans).floor().as_ivec3();
                    for corner in 0..8 {
                        let c = IVec3::new(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1) * (m.size - IVec3::ONE);
                        *lo = lo.min(place(c.as_vec3()));
                    }
                    for v in &m.voxels {
                        out.push((place(Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32)), v[3]));
                    }
                }
                None => {}
            }
        }
        let (mut out, mut lo) = (Vec::new(), IVec3::MAX);
        walk(f, 0, Mat3::IDENTITY, Vec3::ZERO, &mut out, &mut lo);
        (lo, out)
    }

    /// root -> group [ a, "arm" (turned, moved) -> group [ "hand" (flipped) ] ]
    fn nested_file() -> VoxFile {
        use crate::vox::{Model, Node};
        let model = |size: IVec3, n: u8| Model {
            size,
            voxels: (0..size.x * size.y * size.z)
                .map(|i| [(i % size.x) as u8, ((i / size.x) % size.y) as u8, (i / (size.x * size.y)) as u8, n + (i % 7) as u8])
                .collect(),
        };
        let trn = |name: &str, child: i32, r: u8, t: IVec3| Node::Transform {
            attrs: if name.is_empty() { Dict::new() } else { [("_name".to_string(), name.to_string())].into() },
            child,
            layer: 0,
            rotation: crate::vox::decode_rotation(r),
            translation: t,
            frames: vec![[("_r".to_string(), r.to_string()), ("_t".to_string(), format!("{} {} {}", t.x, t.y, t.z))].into()],
        };
        let shape = |m: i32| Node::Shape { attrs: Dict::new(), models: vec![m], model_frames: vec![0] };
        let group = |c: Vec<i32>| Node::Group { attrs: Dict::new(), children: c };
        // Rotation bytes: 4 identity, 17 = turn about z, 36 = flip y.
        let nodes = [
            (0, trn("", 1, 4, IVec3::ZERO)),
            (1, group(vec![2, 4])),
            (2, trn("a", 3, 4, IVec3::new(0, 0, 2))),
            (3, shape(0)),
            (4, trn("arm", 5, 17, IVec3::new(9, -4, 3))),
            (5, group(vec![6])),
            (6, trn("hand", 7, 36, IVec3::new(2, 1, 0))),
            (7, shape(1)),
        ];
        VoxFile {
            models: vec![model(IVec3::new(4, 3, 5), 1), model(IVec3::new(3, 2, 2), 20)],
            palette: crate::vox::default_palette(),
            nodes: nodes.into_iter().collect(),
            layers: [(0, Dict::new())].into(),
            ..Default::default()
        }
    }

    #[test]
    fn nested_tree_places_like_magicavoxel() {
        let f = nested_file();
        let (lo, voxels) = reference(&f);
        let s = Scene::from_vox(&f).unwrap();
        assert_eq!(s.mode, Mode::World);
        assert_eq!(s.origin, lo);
        let mut expect = Volume::new(s.volume.dims);
        for (p, c) in voxels {
            expect.set(p - lo, c);
        }
        assert_eq!(s.volume.cells, expect.cells);

        // Saving and reloading keeps the tree and the placement.
        let path = std::env::temp_dir().join(format!("voxely-tree-{}.vox", std::process::id()));
        s.save(&path).unwrap();
        let back = Scene::from_vox(&crate::vox::VoxFile::load(&path).unwrap()).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.volume.cells, s.volume.cells);
        assert_eq!(back.origin, s.origin);
        let names: Vec<String> = back.world.models().iter().map(|&m| back.world.label(m)).collect();
        assert_eq!(names, ["a", "hand"]);
        let hand = back.world.find("hand").unwrap();
        assert_eq!(back.world.label(back.world.node(hand).unwrap().parent.unwrap()), "arm");
    }

    #[test]
    fn editing_a_part_writes_back_in_place() {
        let mut s = Scene::from_vox(&nested_file()).unwrap();
        let hand = s.world.find("hand").unwrap();
        let before = s.volume.clone();
        assert!(s.edit(hand));
        // The hand is turned and flipped; edit it in world orientation.
        let (lo, dims) = s.world.bounds(hand).unwrap();
        assert_eq!((s.origin, s.volume.dims), (lo, dims));
        let ctx = s.context.as_ref().unwrap();
        assert!(ctx.0.cells.iter().any(|&c| c != 0));
        assert_eq!(s.flat().cells.iter().filter(|&&c| c != 0).count(), before.cells.iter().filter(|&&c| c != 0).count());
        let i = s.volume.index(IVec3::ZERO);
        s.volume.set_index(i, 200);
        s.world_mode();
        assert_eq!(s.volume.get(lo - s.origin), 200);
        // One undo step restores the old part.
        assert!(s.undo_tree(false));
        assert_eq!(s.volume.cells, before.cells);
        assert!(s.undo_tree(true));
        assert_eq!(s.volume.get(lo - s.origin), 200);
    }
}
