//! Flattens a `VoxFile` scene graph into a dense voxel volume plus the
//! material table and render settings the GPU path tracer consumes.

use anyhow::{bail, Result};
use glam::{IVec3, Mat3, Vec3};

use std::path::PathBuf;

use crate::env::EnvMode;
use crate::vox::{Dict, Node, VoxFile};

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

pub struct Scene {
    pub volume: Volume,
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

/// Voxels placed in world space plus the bounds of every model box, so the
/// editable volume covers each model's full size, not just its voxels.
struct Placed {
    voxels: Vec<(IVec3, u8)>,
    min: IVec3,
    max: IVec3,
}

impl Placed {
    fn grow(&mut self, p: IVec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }
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
        let mut render = self.render_dicts.clone();
        self.settings.write_vox(&mut render);
        crate::vox::write(path, self.volume.dims, &self.volume.cells, &self.palette, &mats, &render)
    }

    pub fn from_vox(f: &VoxFile) -> Result<Self> {
        let mut placed = Placed { voxels: Vec::new(), min: IVec3::MAX, max: IVec3::MIN };
        if f.nodes.is_empty() {
            for m in &f.models {
                placed.grow(IVec3::ZERO);
                placed.grow(m.size - IVec3::ONE);
                placed.voxels.extend(m.voxels.iter().map(|v| (IVec3::new(v[0] as i32, v[1] as i32, v[2] as i32), v[3])));
            }
        } else {
            walk(f, 0, Mat3::IDENTITY, Vec3::ZERO, &mut placed, 0);
        }
        if placed.min.cmpgt(placed.max).any() {
            bail!("scene has no visible models");
        }

        let dims = placed.max - placed.min + IVec3::ONE;
        if dims.x as usize * dims.y as usize * dims.z as usize > 1024 * 1024 * 1024 {
            bail!("scene too large for a dense volume: {dims}");
        }
        let mut volume = Volume::new(dims);
        for (p, c) in placed.voxels {
            let i = volume.index(p - placed.min);
            volume.set_index(i, c);
        }
        let render_dicts = f.render_order.iter().filter_map(|k| f.render.get(k).cloned()).collect();
        Ok(Self::assemble(volume, f.palette.clone(), &f.materials, render_dicts, RenderSettings::from_vox(f)))
    }

    /// An empty canvas with the default palette, like MagicaVoxel's new model.
    pub fn empty(dims: IVec3) -> Self {
        Self::assemble(Volume::new(dims), crate::vox::default_palette(), &Default::default(), Vec::new(), RenderSettings::default())
    }

    fn assemble(
        volume: Volume,
        palette: Vec<[u8; 4]>,
        mats: &std::collections::HashMap<u32, Dict>,
        render_dicts: Vec<Dict>,
        settings: RenderSettings,
    ) -> Self {
        let materials: Vec<GpuMaterial> = (0..256u32).map(|i| material(i, palette[i as usize], mats.get(&i))).collect();
        let (lights, light_power) = build_lights(&volume, &materials);
        Scene { volume, palette, render_dicts, materials, lights, light_power, settings }
    }

    /// Rebuild the emissive face list after edits.
    pub fn rebuild_lights(&mut self) {
        (self.lights, self.light_power) = build_lights(&self.volume, &self.materials);
    }

    /// Recompute a material's linear color after its palette entry changed.
    pub fn sync_color(&mut self, idx: usize) {
        let p = self.palette[idx];
        let c = srgb_to_linear(Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32) / 255.0);
        self.materials[idx].color = [c.x, c.y, c.z, p[3] as f32 / 255.0];
    }
}

fn hidden(attrs: &Dict) -> bool {
    attrs.get("_hidden").is_some_and(|v| v == "1")
}

fn walk(f: &VoxFile, id: i32, rot: Mat3, trans: Vec3, out: &mut Placed, depth: u32) {
    if depth > 64 {
        return;
    }
    match f.nodes.get(&id) {
        Some(Node::Transform { attrs, child, layer, rotation, translation }) => {
            let layer_hidden = f.layers.get(layer).is_some_and(hidden);
            if hidden(attrs) || layer_hidden {
                return;
            }
            let r = rot * *rotation;
            let t = rot * translation.as_vec3() + trans;
            walk(f, *child, r, t, out, depth + 1);
        }
        Some(Node::Group { attrs, children }) => {
            if hidden(attrs) {
                return;
            }
            for c in children {
                walk(f, *c, rot, trans, out, depth + 1);
            }
        }
        Some(Node::Shape { attrs, models }) => {
            if hidden(attrs) {
                return;
            }
            for &mi in models {
                let Some(m) = f.models.get(mi as usize) else { continue };
                // Models pivot around their center; integer division matches
                // MagicaVoxel's placement for odd and even sizes.
                let half = (m.size / 2).as_vec3();
                let place = |v: Vec3| (rot * (v + 0.5 - half) + trans).floor().as_ivec3();
                for corner in 0..8 {
                    let c = IVec3::new(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1) * (m.size - IVec3::ONE);
                    out.grow(place(c.as_vec3()));
                }
                for v in &m.voxels {
                    out.voxels.push((place(Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32)), v[3]));
                }
            }
        }
        None => {}
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
}
