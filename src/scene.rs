//! Flattens a `VoxFile` scene graph into a dense voxel volume plus the
//! material table and render settings the GPU path tracer consumes.

use anyhow::{bail, Result};
use glam::{IVec3, Vec3};

use std::path::PathBuf;

use crate::env::EnvMode;
use crate::vox::{Dict, VoxFile};
use crate::world::World;

/// Edge length of the coarse occupancy bricks used to skip empty space.
pub const BRICK: i32 = 8;

#[derive(Clone)]
pub struct Volume {
    pub dims: IVec3,
    /// Color index per cell, x-fastest then y then z.
    pub cells: Vec<u8>,
    /// 1 if the brick may contain a voxel (stays set after erasing, which
    /// only costs a little traversal speed).
    pub bricks: Vec<u8>,
    pub brick_dims: IVec3,
}

impl Volume {
    pub fn new(dims: IVec3) -> Self {
        let brick_dims = (dims + IVec3::splat(BRICK - 1)) / BRICK;
        Volume {
            dims,
            cells: vec![0; (dims.x * dims.y * dims.z) as usize],
            bricks: vec![0; (brick_dims.x * brick_dims.y * brick_dims.z) as usize],
            brick_dims,
        }
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

    /// A volume holding `cells` (x-fastest), with bricks computed.
    pub fn from_cells(dims: IVec3, cells: Vec<u8>) -> Self {
        let mut v = Volume::new(dims);
        v.cells = cells;
        v.refresh_bricks();
        v
    }

    /// Recompute brick occupancy exactly (after bulk edits).
    pub fn refresh_bricks(&mut self) {
        self.bricks.fill(0);
        for i in 0..self.cells.len() {
            if self.cells[i] != 0 {
                let b = self.position(i) / BRICK;
                let bi = (b.x + self.brick_dims.x * (b.y + self.brick_dims.y * b.z)) as usize;
                self.bricks[bi] = 1;
            }
        }
    }

    pub fn set(&mut self, p: IVec3, c: u8) {
        if self.contains(p) {
            let i = self.index(p);
            self.set_index(i, c);
        }
    }

    /// Color index at `p`, 0 outside the volume.
    pub fn get(&self, p: IVec3) -> u8 {
        if self.contains(p) { self.cells[self.index(p)] } else { 0 }
    }

    /// Set a cell by index, keeping the brick occupancy conservative.
    pub fn set_index(&mut self, i: usize, c: u8) {
        self.cells[i] = c;
        if c != 0 {
            let b = self.position(i) / BRICK;
            let bi = (b.x + self.brick_dims.x * (b.y + self.brick_dims.y * b.z)) as usize;
            self.bricks[bi] = 1;
        }
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
    pub owners: Vec<u32>,
    /// Edit mode: the other visible models, flattened, and where their
    /// cell 0 sits relative to `volume`.
    pub context: Option<(Volume, IVec3)>,
    /// The picked nodes (World mode), or the edited model.
    pub picked: Vec<usize>,
    history: Vec<World>,
    future: Vec<World>,
    /// sRGB palette as stored in the file, indexed by color index.
    pub palette: Vec<[u8; 4]>,
    /// Render chunks from the file, written back unchanged on save.
    pub render_dicts: Vec<Dict>,
    pub materials: Vec<GpuMaterial>,
    pub lights: Vec<GpuLight>,
    /// Sum of the power of all light faces.
    pub light_power: f32,
    pub settings: RenderSettings,
}

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
    for z in 0..v.dims.z {
        for y in 0..v.dims.y {
            for x in 0..v.dims.x {
                let c = v.cells[(x + v.dims.x * (y + v.dims.y * z)) as usize];
                if c == 0 {
                    continue;
                }
                let pw = power(c);
                if pw <= 0.0 {
                    continue;
                }
                let p = IVec3::new(x, y, z);
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
        if let Some(d) = f.render.get("_setting") {
            s.ground = flag(d, "_ground").unwrap_or(s.ground);
        }
        // Depth of field is opt-in: files store an aperture even when DOF is off.
        s.aperture = 0.0;
        s
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
            // Inverse of `material`: emission above 1 moves into power.
            let (e, flux) = if m.emit > 1.0 { (1.0, 2.0 * m.emit.log10()) } else { (m.emit, 0.0) };
            set("_emit", fmt(e));
            set("_flux", fmt(flux));
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
        let mats: Vec<(u32, Dict)> =
            (1..256u32).filter_map(|i| material_dict(&self.materials[i as usize]).map(|d| (i, d))).collect();
        crate::vox::write(path, &self.synced_world(), &self.palette, &mats, &self.render_dicts)
    }

    pub fn from_vox(f: &VoxFile) -> Result<Self> {
        let world = World::from_vox(f);
        let models = world.models();
        if !models.iter().any(|&m| world.visible(m)) {
            bail!("scene has no visible models");
        }
        let cells: u64 = models.iter().filter_map(|&m| world.node(m)?.volume()).map(|v| v.cells.len() as u64).sum();
        if cells > 1024 * 1024 * 1024 {
            bail!("scene too large");
        }
        let render_dicts = f.render_order.iter().filter(|k| *k != crate::vox::ANIM_TYPE).filter_map(|k| f.render.get(k).cloned()).collect();
        Ok(Self::assemble(world, f.palette.clone(), &f.materials, render_dicts, RenderSettings::from_vox(f)))
    }

    /// An empty canvas with the default palette, like MagicaVoxel's new model.
    pub fn empty(dims: IVec3) -> Self {
        Self::assemble(World::single(Volume::new(dims)), crate::vox::default_palette(), &Default::default(), Vec::new(), RenderSettings::default())
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
            owners: Vec::new(),
            context: None,
            picked: Vec::new(),
            history: Vec::new(),
            future: Vec::new(),
            palette,
            render_dicts,
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
                self.owners.clear();
                self.picked = vec![id];
            }
            Mode::World => {
                self.context = None;
                match self.world.composite(None) {
                    Some(c) => (self.volume, self.origin, self.owners) = (c.volume, c.origin, c.owners),
                    None => {
                        self.volume = Volume::new(IVec3::ONE);
                        self.owners = vec![0];
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
        self.history.push(self.world.clone());
        if self.history.len() > HISTORY {
            self.history.remove(0);
        }
        self.future.clear();
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
        let (from, to) = if forward { (&mut self.future, &mut self.history) } else { (&mut self.history, &mut self.future) };
        let Some(w) = from.pop() else { return false };
        to.push(std::mem::replace(&mut self.world, w));
        self.view(self.mode);
        true
    }

    /// World mode: cells of the picked nodes, for highlighting.
    pub fn highlight(&self) -> Option<Vec<bool>> {
        if self.mode != Mode::World || self.picked.is_empty() {
            return None;
        }
        let mut ids = std::collections::HashSet::new();
        for &p in &self.picked {
            self.world.visit(p, &mut |i, _| {
                ids.insert(i as u32);
            });
        }
        Some(self.owners.iter().map(|o| ids.contains(o)).collect())
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

    /// Recolor every model through `remap` (palette sorting).
    pub fn remap(&mut self, remap: &[u8]) {
        let fix = |v: &mut Volume| v.cells.iter_mut().filter(|c| **c != 0).for_each(|c| *c = remap[*c as usize]);
        fix(&mut self.volume);
        if let Some((ctx, _)) = &mut self.context {
            fix(ctx);
        }
        for n in self.world.nodes.iter_mut().flatten() {
            if let crate::world::Kind::Model(v) = &mut n.kind {
                fix(v);
            }
        }
    }

    /// Every visible model flattened (for exports and rendering).
    pub fn flat(&self) -> std::borrow::Cow<'_, Volume> {
        match &self.context {
            Some((ctx, offset)) => {
                let lo = offset.min(IVec3::ZERO);
                let hi = (*offset + ctx.dims).max(self.volume.dims);
                let mut out = Volume::new(hi - lo);
                for (v, at) in [(ctx, *offset - lo), (&self.volume, -lo)] {
                    for (i, &c) in v.cells.iter().enumerate() {
                        if c != 0 {
                            let j = out.index(v.position(i) + at);
                            out.set_index(j, c);
                        }
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
        assert_eq!(back.palette[10], [1, 2, 3, 255]);
        assert_eq!(back.materials[20].kind, MaterialKind::Emit as u32);
        assert!((back.materials[20].emit - 31.6).abs() < 0.05, "{}", back.materials[20].emit);
        assert_eq!(back.materials[30].kind, MaterialKind::Glass as u32);
        assert!((back.materials[30].trans - 0.8).abs() < 1e-4);
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
