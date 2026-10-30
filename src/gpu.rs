//! wgpu host side of the path tracer: device setup, scene upload,
//! progressive accumulation, bloom and tone-mapped output.

use anyhow::{Context, Result};
use glam::Vec3;

use crate::env::{self, EnvMap};
use crate::modeler::Modeler;
use crate::scene::{BgMode, DenoiseView, GpuLight, GpuMaterial, Mask, Projection, RenderSettings, Scene, Volume};

const COMMON_WGSL: &str = include_str!("shaders/common.wgsl");
const PATHTRACE_WGSL: &str = include_str!("shaders/pathtrace.wgsl");
const DISPLAY_WGSL: &str = include_str!("shaders/display.wgsl");
const DENOISE_WGSL: &str = include_str!("shaders/denoise.wgsl");
/// À-trous passes (strides 1, 2, 4, ...). Odd, so the result lands in `den_b`.
const DENOISE_PASSES: u32 = 5;
/// Bits of `Params::aux[0]`, matching FLAG_* in common.wgsl.
const FLAG_DENOISE: u32 = 1;
const FLAG_EDGES: u32 = 2;
const FLAG_GRID: u32 = 4;
const FLAG_GRID_GROUND_ONLY: u32 = 8;
const FLAG_BG_HIDDEN: u32 = 16;
/// Frames after loading during which a view resize re-frames the camera.
const REFRAME_FRAMES: u32 = 3;

/// Orbit camera around a target point, Z up like MagicaVoxel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
}

impl Camera {
    /// A three-quarter view that fits the whole canvas in a view with the
    /// given vertical field of view (degrees) and aspect ratio.
    pub fn framing(dims: Vec3, fov: f32, aspect: f32) -> Self {
        let half_v = fov.to_radians() * 0.5;
        let half_h = (half_v.tan() * aspect).atan();
        let radius = dims.length() * 0.5;
        Camera {
            target: dims * 0.5,
            yaw: 45f32.to_radians() + std::f32::consts::PI,
            pitch: 25f32.to_radians(),
            dist: radius / half_v.min(half_h).sin() * 1.05,
        }
    }

    pub fn eye(&self) -> Vec3 {
        let d = Vec3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
        );
        self.target + d * self.dist
    }

    /// (forward, right, up)
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let fwd = (self.target - self.eye()).normalize();
        let right = fwd.cross(Vec3::Z).normalize();
        let up = right.cross(fwd);
        (fwd, right, up)
    }
}

/// Must match `Params` in common.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    cam_pos: [f32; 4],
    cam_fwd: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    sun_dir: [f32; 4],
    sun_col: [f32; 4],
    fog: [f32; 4],
    ground_col: [f32; 4],
    film: [f32; 4],
    misc: [f32; 4],
    bloom: [f32; 4],
    dims: [u32; 4],
    frame: [u32; 4],
    env: [u32; 4],
    lens: [f32; 4],
    view: [f32; 4],
    aux: [u32; 4],
    bg_col: [f32; 4],
    edge_col: [f32; 4],
    grid_col: [f32; 4],
}

fn storage_buffer(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, data: &[u8]) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: data.len().max(16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, data);
    buf
}

fn scratch_buffer(device: &wgpu::Device, label: &str, size: u64, extra: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE | extra,
        mapped_at_creation: false,
    })
}

fn write_3d(queue: &wgpu::Queue, tex: &wgpu::Texture, dims: [u32; 3], texel: u32, data: &[u8]) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        data,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(dims[0] * texel), rows_per_image: Some(dims[1]) },
        wgpu::Extent3d { width: dims[0], height: dims[1], depth_or_array_layers: dims[2] },
    );
}

fn upload_3d(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, dims: [u32; 3], format: wgpu::TextureFormat, data: &[u8]) -> wgpu::Texture {
    let size = wgpu::Extent3d { width: dims[0], height: dims[1], depth_or_array_layers: dims[2] };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    write_3d(queue, &tex, dims, (data.len() / (dims[0] * dims[1] * dims[2]) as usize) as u32, data);
    tex
}

/// Bricks per row and per column of a layer of the path tracer's brick
/// pool; must match `POOL_W` in `pathtrace.wgsl`.
const POOL_W: u32 = 64;

/// A volume in the path tracer's sparse layout: a brick map (0 for an
/// empty 8³ brick, else its pool slot plus one) and a pool texture of the
/// non-empty bricks, `POOL_W` x `POOL_W` bricks per layer.
pub struct TraceLayout {
    pub map: Vec<u32>,
    pub map_dims: [u32; 3],
    pub pool: Vec<u8>,
    pub pool_dims: [u32; 3],
}

impl TraceLayout {
    pub fn new(v: &Volume) -> Self {
        let bd = v.brick_dims();
        let map_dims = [bd.x.max(1) as u32, bd.y.max(1) as u32, bd.z.max(1) as u32];
        let mut map = vec![0u32; (map_dims[0] * map_dims[1] * map_dims[2]) as usize];
        let bricks: Vec<_> = v.cells.bricks().filter(|(_, cells)| cells.iter().any(|&c| c != 0)).collect();
        let layers = (bricks.len() as u32).div_ceil(POOL_W * POOL_W).max(1);
        let pool_dims = [POOL_W * 8, POOL_W * 8, layers * 8];
        let mut pool = vec![0u8; (pool_dims[0] * pool_dims[1] * pool_dims[2]) as usize];
        for (slot, (b, cells)) in bricks.iter().enumerate() {
            let slot = slot as u32;
            map[(b.x as u32 + map_dims[0] * (b.y as u32 + map_dims[1] * b.z as u32)) as usize] = slot + 1;
            let at = [slot % POOL_W * 8, slot / POOL_W % POOL_W * 8, slot / (POOL_W * POOL_W) * 8];
            for z in 0..8 {
                for y in 0..8 {
                    let row = (at[0] + pool_dims[0] * (at[1] + y + pool_dims[1] * (at[2] + z))) as usize;
                    pool[row..row + 8].copy_from_slice(&cells[(8 * (y + 8 * z)) as usize..][..8]);
                }
            }
        }
        TraceLayout { map, map_dims, pool, pool_dims }
    }

    /// Whether the GPU can hold this layout.
    pub fn fits(&self, max_dim: u32) -> bool {
        self.map_dims.iter().chain(&self.pool_dims).all(|&d| d <= max_dim)
    }
}

pub async fn request_device(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
) -> Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: surface,
            ..Default::default()
        })
        .await
        .context("no GPU adapter")?;
    log::info!("adapter: {:?}", adapter.get_info().name);
    let limits = adapter.limits();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("voxely"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                max_buffer_size: limits.max_buffer_size,
                max_texture_dimension_3d: limits.max_texture_dimension_3d,
                max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage.min(16),
                ..wgpu::Limits::default()
            },
            ..Default::default()
        })
        .await?;
    Ok((adapter, device, queue))
}

struct Pipelines {
    trace: wgpu::ComputePipeline,
    bloom_down: wgpu::ComputePipeline,
    bloom_h: wgpu::ComputePipeline,
    bloom_v: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    denoise_init: wgpu::ComputePipeline,
    denoise_atrous: wgpu::ComputePipeline,
    display: Option<wgpu::RenderPipeline>,
}

/// Which optional per-pixel buffers are allocated (full size); the rest
/// are 16-byte placeholders so disabled features cost no memory.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Features {
    denoise: bool,
    overlay: bool,
}

impl Features {
    fn wanted(s: &RenderSettings) -> Self {
        Features { denoise: s.denoise && s.denoise_view != DenoiseView::Original, overlay: s.show_edges || s.show_grid }
    }
}

/// Resolution-dependent buffers and every bind group (which reference them).
struct Targets {
    width: u32,
    height: u32,
    out: wgpu::Buffer,
    /// The radiance sum (kept for tests that poison it).
    #[cfg_attr(not(test), allow(dead_code))]
    accum: wgpu::Buffer,
    trace_bg: wgpu::BindGroup,
    bloom_down_bg: wgpu::BindGroup,
    bloom_h_bg: wgpu::BindGroup,
    bloom_v_bg: wgpu::BindGroup,
    resolve_bg: wgpu::BindGroup,
    display_bg: Option<wgpu::BindGroup>,
    features: Features,
    denoise_init_bg: wgpu::BindGroup,
    denoise_bgs: Vec<wgpu::BindGroup>,
}

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub settings: RenderSettings,
    pub camera: Camera,
    /// Samples accumulated since the last reset.
    pub frame: u32,
    dims: [u32; 3],
    light_count: u32,
    light_power: f32,
    params: wgpu::Buffer,
    /// The path tracer's voxels (see `TraceLayout`): the pool of non-empty
    /// bricks and the brick map into it.
    vox_tex: wgpu::Texture,
    brick_tex: wgpu::Texture,
    brick_dims: [u32; 3],
    pool_dims: [u32; 3],
    vox_view: wgpu::TextureView,
    brick_view: wgpu::TextureView,
    materials: wgpu::Buffer,
    lights: wgpu::Buffer,
    env_texels: wgpu::Buffer,
    env_cdf: wgpu::Buffer,
    env_size: (u32, u32),
    /// Settings the current environment map was baked from.
    env_key: String,
    /// Sun color after atmospheric extinction.
    sun_tint: Vec3,
    /// Top-left of the image within the window, in pixels.
    offset: (u32, u32),
    /// Where the path tracer's volume (every model) sits relative to the
    /// edited one, which the camera and picking use.
    trace_offset: glam::IVec3,
    /// World position of the volume's cell 0 at the last upload; when it
    /// moves (switching between models or World mode) the camera follows
    /// so the view stays put.
    frame_origin: glam::IVec3,
    pipelines: Pipelines,
    targets: Targets,
    /// Rasterized editing view (window renders only).
    pub modeler: Option<Modeler>,
    /// Color format the modeler draws in.
    modeler_format: Option<wgpu::TextureFormat>,
    /// Show the modeler instead of the path tracer (F1).
    pub modeling: bool,
    /// Canvas to re-frame if the view is resized within the next few
    /// frames (the panels are laid out after a scene loads).
    reframe: Option<(Vec3, u32)>,
}

fn compute_pipeline(device: &wgpu::Device, module: &wgpu::ShaderModule, entry: &str) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn bind_group(device: &wgpu::Device, label: &str, layout: &wgpu::BindGroupLayout, entries: &[(u32, wgpu::BindingResource)]) -> wgpu::BindGroup {
    let entries: Vec<_> = entries
        .iter()
        .map(|(binding, resource)| wgpu::BindGroupEntry { binding: *binding, resource: resource.clone() })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries: &entries })
}

impl Renderer {
    /// `surface_format` is `Some` when rendering to a window.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        scene: &Scene,
        surface_format: Option<wgpu::TextureFormat>,
        width: u32,
        height: u32,
    ) -> Self {
        let v = &scene.volume;
        let dims = [v.dims.x as u32, v.dims.y as u32, v.dims.z as u32];
        let layout = TraceLayout::new(v);
        let vox_tex = upload_3d(&device, &queue, "voxel pool", layout.pool_dims, wgpu::TextureFormat::R8Uint, &layout.pool);
        let brick_tex = upload_3d(&device, &queue, "brick map", layout.map_dims, wgpu::TextureFormat::R32Uint, bytemuck::cast_slice(&layout.map));
        let vox_view = vox_tex.create_view(&Default::default());
        let brick_view = brick_tex.create_view(&Default::default());
        let materials = storage_buffer(&device, &queue, "materials", bytemuck::cast_slice(&scene.materials));
        let lights_data: &[GpuLight] = if scene.lights.is_empty() { &[GpuLight { corner_cdf: [0.0; 4], axis_sign_mat: [0; 4] }] } else { &scene.lights };
        let lights = storage_buffer(&device, &queue, "lights", bytemuck::cast_slice(lights_data));
        log::info!("{} emissive faces", scene.lights.len());
        let modeler = surface_format.map(|f| (Modeler::new(&device, f, v, &scene.palette), f));

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let trace_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pathtrace"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON_WGSL}\n{PATHTRACE_WGSL}").into()),
        });
        let post_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("display"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON_WGSL}\n{DISPLAY_WGSL}").into()),
        });
        let denoise_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("denoise"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON_WGSL}\n{DENOISE_WGSL}").into()),
        });
        let pipelines = Pipelines {
            trace: compute_pipeline(&device, &trace_module, "main"),
            bloom_down: compute_pipeline(&device, &post_module, "bloom_down"),
            bloom_h: compute_pipeline(&device, &post_module, "bloom_h"),
            bloom_v: compute_pipeline(&device, &post_module, "bloom_v"),
            resolve: compute_pipeline(&device, &post_module, "resolve"),
            denoise_init: compute_pipeline(&device, &denoise_module, "denoise_init"),
            denoise_atrous: compute_pipeline(&device, &denoise_module, "denoise_atrous"),
            display: surface_format.map(|format| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("display"),
                    layout: None,
                    vertex: wgpu::VertexState {
                        module: &post_module,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &post_module,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(format.into())],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
            }),
        };

        // Placeholders until the first environment bake.
        let env_texels = storage_buffer(&device, &queue, "env", &[0; 16]);
        let env_cdf = storage_buffer(&device, &queue, "env_cdf", &[0; 16]);
        let features = Features::wanted(&scene.settings);
        let targets = Self::placeholder_targets(&device, &pipelines, &params, &vox_view, &brick_view, &materials, &lights, &env_texels, &env_cdf, width, height, features);
        let mut r = Renderer {
            device,
            queue,
            settings: scene.settings.clone(),
            camera: Camera::framing(v.dims.as_vec3(), scene.settings.fov, width as f32 / height.max(1) as f32),
            reframe: Some((v.dims.as_vec3(), REFRAME_FRAMES)),
            frame: 0,
            dims,
            light_count: scene.lights.len() as u32,
            light_power: scene.light_power,
            params,
            vox_tex,
            brick_tex,
            brick_dims: layout.map_dims,
            pool_dims: layout.pool_dims,
            vox_view,
            brick_view,
            materials,
            lights,
            env_texels,
            env_cdf,
            env_size: (1, 1),
            env_key: String::new(),
            sun_tint: Vec3::ONE,
            offset: (0, 0),
            trace_offset: glam::IVec3::ZERO,
            frame_origin: scene.origin,
            pipelines,
            targets,
            modeling: modeler.is_some(),
            modeler_format: modeler.as_ref().map(|m| m.1),
            modeler: modeler.map(|m| m.0),
        };
        r.update_env();
        if scene.context.is_some() {
            r.upload_scene(scene, None);
        }
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn placeholder_targets(
        device: &wgpu::Device,
        p: &Pipelines,
        params: &wgpu::Buffer,
        vox_view: &wgpu::TextureView,
        brick_view: &wgpu::TextureView,
        materials: &wgpu::Buffer,
        lights: &wgpu::Buffer,
        env_texels: &wgpu::Buffer,
        env_cdf: &wgpu::Buffer,
        width: u32,
        height: u32,
        features: Features,
    ) -> Targets {
        let pixels = (width * height) as u64;
        let bloom_pixels = (width.div_ceil(4) * height.div_ceil(4)) as u64;
        let none = wgpu::BufferUsages::empty();
        let accum = scratch_buffer(device, "accum", pixels * 16, wgpu::BufferUsages::COPY_DST);
        let out = scratch_buffer(device, "out_rgba", pixels * 4, wgpu::BufferUsages::COPY_SRC);
        let bloom_a = scratch_buffer(device, "bloom_a", bloom_pixels * 16, none);
        let bloom_b = scratch_buffer(device, "bloom_b", bloom_pixels * 16, none);
        let den_pixels = if features.denoise { pixels } else { 0 };
        let aov_albedo = scratch_buffer(device, "aov_albedo", den_pixels * 16, none);
        let aov_normal = scratch_buffer(device, "aov_normal", den_pixels * 16, none);
        let den_a = scratch_buffer(device, "denoise_a", den_pixels * 16, none);
        let den_b = scratch_buffer(device, "denoise_b", den_pixels * 16, none);
        let overlay = scratch_buffer(device, "overlay", if features.overlay { pixels * 16 } else { 0 }, none);

        let trace_bg = bind_group(
            device,
            "trace",
            &p.trace.get_bind_group_layout(0),
            &[
                (0, params.as_entire_binding()),
                (1, wgpu::BindingResource::TextureView(vox_view)),
                (2, wgpu::BindingResource::TextureView(brick_view)),
                (3, materials.as_entire_binding()),
                (4, accum.as_entire_binding()),
                (5, env_texels.as_entire_binding()),
                (6, env_cdf.as_entire_binding()),
                (7, lights.as_entire_binding()),
                (8, aov_albedo.as_entire_binding()),
                (9, aov_normal.as_entire_binding()),
                (10, overlay.as_entire_binding()),
            ],
        );
        let denoise_init_bg = bind_group(
            device,
            "denoise_init",
            &p.denoise_init.get_bind_group_layout(0),
            &[(0, params.as_entire_binding()), (1, accum.as_entire_binding()), (2, aov_albedo.as_entire_binding()), (5, den_a.as_entire_binding())],
        );
        let denoise_bgs = (0..DENOISE_PASSES)
            .map(|i| {
                let info = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("denoise_pass"),
                    size: 16,
                    usage: wgpu::BufferUsages::UNIFORM,
                    mapped_at_creation: true,
                });
                let data = [1u32 << i, (i + 1 == DENOISE_PASSES) as u32, 0, 0];
                info.slice(..).get_mapped_range_mut().expect("mapped at creation").copy_from_slice(bytemuck::bytes_of(&data));
                info.unmap();
                let (src, dst) = if i % 2 == 0 { (&den_a, &den_b) } else { (&den_b, &den_a) };
                bind_group(
                    device,
                    "denoise_atrous",
                    &p.denoise_atrous.get_bind_group_layout(0),
                    &[
                        (0, params.as_entire_binding()),
                        (2, aov_albedo.as_entire_binding()),
                        (3, aov_normal.as_entire_binding()),
                        (4, src.as_entire_binding()),
                        (5, dst.as_entire_binding()),
                        (6, info.as_entire_binding()),
                    ],
                )
            })
            .collect();
        let post = |pipe_layout: wgpu::BindGroupLayout, label: &str, bindings: &[u32]| {
            let all = [
                (0, params.as_entire_binding()),
                (1, accum.as_entire_binding()),
                (2, out.as_entire_binding()),
                (3, bloom_a.as_entire_binding()),
                (4, bloom_b.as_entire_binding()),
                (5, den_b.as_entire_binding()),
                (6, overlay.as_entire_binding()),
            ];
            let picked: Vec<_> = all.into_iter().filter(|(b, _)| bindings.contains(b)).collect();
            bind_group(device, label, &pipe_layout, &picked)
        };
        Targets {
            width,
            height,
            bloom_down_bg: post(p.bloom_down.get_bind_group_layout(0), "bloom_down", &[0, 1, 3, 5]),
            bloom_h_bg: post(p.bloom_h.get_bind_group_layout(0), "bloom_h", &[0, 3, 4]),
            bloom_v_bg: post(p.bloom_v.get_bind_group_layout(0), "bloom_v", &[0, 3, 4]),
            resolve_bg: post(p.resolve.get_bind_group_layout(0), "resolve", &[0, 1, 2, 3, 5, 6]),
            display_bg: p.display.as_ref().map(|d| post(d.get_bind_group_layout(0), "display", &[0, 1, 3, 5, 6])),
            out,
            accum,
            trace_bg,
            features,
            denoise_init_bg,
            denoise_bgs,
        }
    }

    fn rebuild_targets(&mut self, width: u32, height: u32) {
        self.targets = Self::placeholder_targets(
            &self.device,
            &self.pipelines,
            &self.params,
            &self.vox_view,
            &self.brick_view,
            &self.materials,
            &self.lights,
            &self.env_texels,
            &self.env_cdf,
            width,
            height,
            Features::wanted(&self.settings),
        );
        self.reset();
    }

    /// Render into the window rectangle at (`x`, `y`) of size `w` x `h`.
    pub fn set_viewport(&mut self, x: u32, y: u32, w: u32, h: u32) {
        self.offset = (x, y);
        self.resize(w, h);
        if let Some((dims, _)) = self.reframe {
            self.fit_camera(dims);
        }
    }

    fn fit_camera(&mut self, dims: Vec3) {
        let (w, h) = self.size();
        self.camera = Camera::framing(dims, self.settings.fov, w as f32 / h.max(1) as f32);
        self.reset();
    }

    /// Point the camera at a canvas of size `dims`, fitting it in the view.
    pub fn frame(&mut self, dims: Vec3) {
        self.fit_camera(dims);
        self.reframe = Some((dims, REFRAME_FRAMES));
    }

    /// Use an explicit camera, cancelling any pending re-frame.
    pub fn set_camera(&mut self, camera: Camera) {
        self.camera = camera;
        self.reframe = None;
        self.reset();
    }

    pub fn viewport(&self) -> (u32, u32, u32, u32) {
        (self.offset.0, self.offset.1, self.targets.width, self.targets.height)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if width != self.targets.width || height != self.targets.height {
            self.rebuild_targets(width, height);
        }
    }

    /// Re-upload voxels after edits, reallocating if the size changed.
    /// `sel` marks selected cells for the modeler.
    pub fn upload_volume(&mut self, v: &Volume, sel: Option<&Mask>) {
        if let Some(m) = self.modeler.as_mut() {
            m.upload_volume(&self.device, v, sel);
            m.set_context(&self.device, None);
        }
        self.trace_offset = glam::IVec3::ZERO;
        self.upload_trace(v);
    }

    /// Upload a scene's voxels: the edited model, with the other models
    /// as dimmed context, for the modeler; every model for the path tracer.
    pub fn upload_scene(&mut self, scene: &Scene, sel: Option<&Mask>) {
        if scene.origin != self.frame_origin {
            self.camera.target += (self.frame_origin - scene.origin).as_vec3();
            self.frame_origin = scene.origin;
        }
        if scene.context.is_none() {
            return self.upload_volume(&scene.volume, sel);
        }
        if let Some(m) = self.modeler.as_mut() {
            m.upload_volume(&self.device, &scene.volume, sel);
            m.set_context(&self.device, scene.context.as_ref());
        }
        self.trace_offset = scene.flat_offset();
        self.upload_trace(&scene.flat());
    }

    fn upload_trace(&mut self, v: &Volume) {
        let dims = [v.dims.x as u32, v.dims.y as u32, v.dims.z as u32];
        let layout = TraceLayout::new(v);
        if !layout.fits(self.device.limits().max_texture_dimension_3d) {
            log::warn!("scene too large for the path tracer ({:?} voxels, {} bricks)", dims, layout.pool.len() / 512);
            return;
        }
        if layout.map_dims != self.brick_dims || layout.pool_dims != self.pool_dims {
            self.vox_tex = upload_3d(&self.device, &self.queue, "voxel pool", layout.pool_dims, wgpu::TextureFormat::R8Uint, &layout.pool);
            self.brick_tex = upload_3d(&self.device, &self.queue, "brick map", layout.map_dims, wgpu::TextureFormat::R32Uint, bytemuck::cast_slice(&layout.map));
            self.vox_view = self.vox_tex.create_view(&Default::default());
            self.brick_view = self.brick_tex.create_view(&Default::default());
            (self.brick_dims, self.pool_dims) = (layout.map_dims, layout.pool_dims);
            self.dims = dims;
            self.rebuild_targets(self.targets.width, self.targets.height);
        } else {
            write_3d(&self.queue, &self.vox_tex, self.pool_dims, 1, &layout.pool);
            write_3d(&self.queue, &self.brick_tex, self.brick_dims, 4, bytemuck::cast_slice(&layout.map));
            self.dims = dims;
        }
        self.reset();
    }

    pub fn upload_materials(&mut self, mats: &[GpuMaterial]) {
        self.queue.write_buffer(&self.materials, 0, bytemuck::cast_slice(mats));
        self.reset();
    }

    pub fn set_lights(&mut self, lights: &[GpuLight], power: f32) {
        let data: &[GpuLight] = if lights.is_empty() { &[GpuLight { corner_cdf: [0.0; 4], axis_sign_mat: [0; 4] }] } else { lights };
        self.lights = storage_buffer(&self.device, &self.queue, "lights", bytemuck::cast_slice(data));
        self.light_count = lights.len() as u32;
        self.light_power = power;
        self.rebuild_targets(self.targets.width, self.targets.height);
    }

    /// World-space ray through pixel (`x`, `y`) of the image, matching the
    /// shader for perspective and orthographic projections.
    pub fn camera_ray(&self, x: f32, y: f32) -> (Vec3, Vec3) {
        let (w, h) = (self.targets.width as f32, self.targets.height as f32);
        let (x, y) = (x - self.offset.0 as f32, y - self.offset.1 as f32);
        let ndc = (x / w * 2.0 - 1.0, 1.0 - y / h * 2.0);
        let (fwd, right, up) = self.camera.basis();
        let tan_half = (self.settings.fov.to_radians() * 0.5).tan();
        let aspect = w / h;
        let eye = self.camera.eye();
        if self.settings.projection == Projection::Orthographic {
            let half = self.camera.dist * tan_half;
            return (eye + (right * ndc.0 * aspect + up * ndc.1) * half, fwd);
        }
        (eye, (fwd + right * ndc.0 * tan_half * aspect + up * ndc.1 * tan_half).normalize())
    }

    /// Pixel position of a world point, if it is in front of the camera.
    pub fn project(&self, p: Vec3) -> Option<(f32, f32)> {
        let (w, h) = (self.targets.width as f32, self.targets.height as f32);
        let (x, y) = project_point(&self.camera, self.settings.fov, self.settings.projection, w, h, p)?;
        Some((x + self.offset.0 as f32, y + self.offset.1 as f32))
    }

    /// Restart accumulation (after the camera or settings change).
    pub fn reset(&mut self) {
        self.frame = 0;
    }

    /// Re-bake the environment map if any setting it depends on changed.
    fn update_env(&mut self) {
        let s = &self.settings;
        let key = format!(
            "{:?}|{:?}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
            s.env_mode,
            s.ibl_path,
            s.ibl_intensity,
            s.ibl_rotation,
            s.sky_color,
            s.sky_intensity,
            s.rayleigh_density,
            s.rayleigh_color,
            s.mie_density,
            s.mie_color,
            s.mie_g,
            s.ozone_density,
            s.ozone_color,
            s.sun_color,
            s.sun_intensity,
            if s.env_mode == env::EnvMode::Atmosphere { format!("{}/{}", s.sun_pitch, s.sun_yaw) } else { String::new() },
        );
        if key == self.env_key {
            return;
        }
        let t = std::time::Instant::now();
        let map = EnvMap::from_settings(s);
        self.sun_tint = env::sun_transmittance(s);
        log::debug!("environment {}x{} baked in {:.1?}", map.width, map.height, t.elapsed());
        self.env_texels = storage_buffer(&self.device, &self.queue, "env", bytemuck::cast_slice(&map.texels));
        self.env_cdf = storage_buffer(&self.device, &self.queue, "env_cdf", bytemuck::cast_slice(&map.cdf));
        self.env_size = (map.width, map.height);
        self.env_key = key;
        self.rebuild_targets(self.targets.width, self.targets.height);
    }

    fn write_params(&self) {
        let s = &self.settings;
        let c = &self.camera;
        let (fwd, right, up) = c.basis();
        let eye = c.eye() - self.trace_offset.as_vec3();
        let t = &self.targets;
        let sun_dir = s.sun_dir();
        let sun = s.sun_color * s.sun_intensity * 3.0 * self.sun_tint;
        let fog_range = Vec3::new(self.dims[0] as f32, self.dims[1] as f32, self.dims[2] as f32).length() * 4.0;
        let bloom_mix = if s.bloom { s.bloom_mix } else { 0.0 };
        let p = Params {
            cam_pos: [eye.x, eye.y, eye.z, (s.fov.to_radians() * 0.5).tan()],
            // Lens radius scales with the focus distance so the blur amount
            // reads the same at any zoom.
            cam_fwd: [fwd.x, fwd.y, fwd.z, s.aperture * c.dist * 0.05],
            cam_right: [right.x, right.y, right.z, c.dist],
            cam_up: [up.x, up.y, up.z, t.width as f32 / t.height as f32],
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, s.sun_area.max(1e-4).cos()],
            sun_col: [sun.x, sun.y, sun.z, s.sun_disk as u32 as f32],
            fog: [s.fog_color.x, s.fog_color.y, s.fog_color.z, s.fog_density * 0.01],
            ground_col: [s.ground_color.x, s.ground_color.y, s.ground_color.z, s.ground as u32 as f32],
            film: [s.exposure, s.gamma, s.aces as u32 as f32, s.vignette],
            misc: [s.clamp, self.light_power.max(1e-6), s.fog_g, fog_range],
            bloom: [bloom_mix, s.bloom_scale, s.bloom_aspect, s.bloom_threshold],
            dims: [self.dims[0], self.dims[1], self.dims[2], s.translucent_shadows as u32],
            frame: [self.frame, t.width, t.height, s.max_diffuse | (s.max_specular << 8) | (s.max_scatter << 16)],
            env: [self.env_size.0, self.env_size.1, self.light_count, 0],
            lens: [s.projection as u32 as f32, s.blades as f32, s.blade_rotation.to_radians(), 0.0],
            view: [self.offset.0 as f32, self.offset.1 as f32, 0.0, 0.0],
            aux: [self.flags(), s.denoise_view as u32, s.bg_mode as u32, s.grid_spacing.clamp(1, 1024)],
            bg_col: [s.bg_color.x, s.bg_color.y, s.bg_color.z, 0.0],
            edge_col: [s.edge_color.x, s.edge_color.y, s.edge_color.z, s.edge_width.clamp(0.01, 1.0)],
            grid_col: [s.grid_color.x, s.grid_color.y, s.grid_color.z, s.grid_width.clamp(0.001, 1.0)],
        };
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
    }

    /// True when the denoiser runs on the displayed image.
    pub fn denoising(&self) -> bool {
        let s = &self.settings;
        self.targets.features.denoise && s.denoise && s.denoise_view != DenoiseView::Original
    }

    fn flags(&self) -> u32 {
        let s = &self.settings;
        let overlay = self.targets.features.overlay;
        let mut f = 0;
        if self.denoising() {
            f |= FLAG_DENOISE;
        }
        if overlay && s.show_edges {
            f |= FLAG_EDGES;
        }
        if overlay && s.show_grid {
            f |= FLAG_GRID;
        }
        if s.grid_ground_only {
            f |= FLAG_GRID_GROUND_ONLY;
        }
        if s.bg_mode != BgMode::Sky {
            f |= FLAG_BG_HIDDEN;
        }
        f
    }

    pub fn size(&self) -> (u32, u32) {
        (self.targets.width, self.targets.height)
    }

    /// Submit `samples` path-tracing passes.
    fn trace(&mut self, samples: u32) {
        self.update_env();
        if Features::wanted(&self.settings) != self.targets.features {
            self.rebuild_targets(self.targets.width, self.targets.height);
        }
        let groups = (self.targets.width.div_ceil(8), self.targets.height.div_ceil(8));
        for _ in 0..samples {
            // Each sample needs its own frame index in the uniform buffer, so
            // submit one pass per sample.
            self.write_params();
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pathtrace"), timestamp_writes: None });
                pass.set_pipeline(&self.pipelines.trace);
                pass.set_bind_group(0, &self.targets.trace_bg, &[]);
                pass.dispatch_workgroups(groups.0, groups.1, 1);
            }
            self.queue.submit([enc.finish()]);
            self.frame += 1;
        }
    }

    /// Record the denoiser passes if it is on.
    fn encode_denoise(&self, enc: &mut wgpu::CommandEncoder) {
        if !self.denoising() {
            return;
        }
        let t = &self.targets;
        let groups = (t.width.div_ceil(8), t.height.div_ceil(8));
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("denoise"), timestamp_writes: None });
        pass.set_pipeline(&self.pipelines.denoise_init);
        pass.set_bind_group(0, &t.denoise_init_bg, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, 1);
        pass.set_pipeline(&self.pipelines.denoise_atrous);
        for bg in &t.denoise_bgs {
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
    }

    /// Record the bloom passes if bloom is on.
    fn encode_bloom(&self, enc: &mut wgpu::CommandEncoder) {
        if !self.settings.bloom || self.settings.bloom_mix <= 0.0 {
            return;
        }
        let t = &self.targets;
        let groups = (t.width.div_ceil(4).div_ceil(8), t.height.div_ceil(4).div_ceil(8));
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("bloom"), timestamp_writes: None });
        for (pipe, bg) in [
            (&self.pipelines.bloom_down, &t.bloom_down_bg),
            (&self.pipelines.bloom_h, &t.bloom_h_bg),
            (&self.pipelines.bloom_v, &t.bloom_v_bg),
        ] {
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
    }

    /// Refresh only the modeler's selection highlight.
    pub fn upload_selection(&mut self, v: &Volume, sel: Option<&Mask>) {
        if let Some(m) = self.modeler.as_mut() {
            m.upload_volume(&self.device, v, sel);
        }
    }

    /// Show `v` (and dimmed `context`) in the modeler only, leaving the
    /// path tracer's voxels alone: screenshots of something other than the
    /// edited part. Call again with the scene's own to restore.
    pub fn modeler_show(&mut self, v: &Volume, context: Option<&(Volume, glam::IVec3)>, sel: Option<&Mask>) {
        if let Some(m) = self.modeler.as_mut() {
            m.upload_volume(&self.device, v, sel);
            m.set_context(&self.device, context);
        }
    }

    /// Voxels about to be placed, in the modeler (see `Modeler::set_ghost`).
    pub fn set_ghost(&mut self, dims: glam::IVec3, cells: &[(glam::IVec3, u8)]) {
        if let Some(m) = self.modeler.as_mut() {
            m.set_ghost(&self.device, dims, cells);
        }
    }

    /// Mark cells in the modeler (see `Modeler::set_highlight`).
    pub fn set_highlight(&mut self, dims: glam::IVec3, cells: &[(glam::IVec3, bool)]) {
        if let Some(m) = self.modeler.as_mut() {
            m.set_highlight(&self.device, dims, cells);
        }
    }

    /// Show a different scene: voxels, materials, lights and settings.
    pub fn load_scene(&mut self, scene: &Scene) {
        self.settings = scene.settings.clone();
        self.upload_scene(scene, None);
        self.upload_materials(&scene.materials);
        self.set_palette(&scene.palette);
        self.set_lights(&scene.lights, scene.light_power);
        self.frame(scene.volume.dims.as_vec3());
        self.reset();
    }

    pub fn set_palette(&mut self, palette: &[[u8; 4]]) {
        if let Some(m) = self.modeler.as_mut() {
            m.set_palette(palette);
        }
    }

    /// True while the path tracer is shown and still refining.
    pub fn converging(&self) -> bool {
        !self.modeling && self.frame < 4096
    }

    /// Draw the modeler, or accumulate samples and draw the tone-mapped
    /// path-traced image, into `view` (of size `target`).
    pub fn render_to_view(&mut self, view: &wgpu::TextureView, target: (u32, u32), samples: u32) {
        self.reframe = self.reframe.and_then(|(d, n)| (n > 1).then_some((d, n - 1)));
        if self.modeling {
            if let Some(m) = self.modeler.as_mut() {
                let viewport = (self.offset.0, self.offset.1, self.targets.width, self.targets.height);
                m.draw(&self.device, &self.queue, view, target, viewport, &self.camera, self.settings.fov, self.settings.projection);
                return;
            }
        }
        self.trace(samples);
        self.write_params();
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.encode_denoise(&mut enc);
        self.encode_bloom(&mut enc);
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("display"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let (x, y) = self.offset;
            pass.set_viewport(x as f32, y as f32, self.targets.width as f32, self.targets.height as f32, 0.0, 1.0);
            pass.set_pipeline(self.pipelines.display.as_ref().expect("no display pipeline"));
            pass.set_bind_group(0, self.targets.display_bg.as_ref().unwrap(), &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([enc.finish()]);
    }

    /// Render `samples` samples offscreen and return RGBA8 pixels.
    pub fn render_image(&mut self, samples: u32) -> Result<Vec<u8>> {
        self.render_image_counted(samples, &std::sync::atomic::AtomicU32::new(0))
    }

    /// `render_image`, adding each batch of samples to `done` once the GPU
    /// has traced it (for progress reports).
    pub fn render_image_counted(&mut self, samples: u32, done: &std::sync::atomic::AtomicU32) -> Result<Vec<u8>> {
        self.reset();
        let t0 = std::time::Instant::now();
        let mut left = samples;
        while left > 0 {
            let n = left.min(16);
            self.trace(n);
            self.device.poll(wgpu::PollType::wait_indefinitely())?;
            done.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
            left -= n;
        }
        let pixels = self.read_pixels()?;
        let t = &self.targets;
        log::info!("{samples} spp at {}x{} in {:.2?}", t.width, t.height, t0.elapsed());
        Ok(pixels)
    }

    /// Draw the modeler view at `width` x `height` offscreen with the
    /// current camera and read it back as RGBA8.
    pub fn modeler_image(&mut self, width: u32, height: u32) -> Result<Vec<u8>> {
        let (camera, projection) = (self.camera, self.settings.projection);
        self.modeler_image_from(width, height, &camera, projection)
    }

    /// Like `modeler_image` from another camera, leaving the renderer's
    /// own camera (the one the window shows) alone.
    pub fn modeler_image_from(&mut self, width: u32, height: u32, camera: &Camera, projection: Projection) -> Result<Vec<u8>> {
        let (Some(m), Some(format)) = (self.modeler.as_mut(), self.modeler_format) else {
            anyhow::bail!("this renderer has no modeler view");
        };
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("modeler_image"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        // Agents' views show the model alone, without the window's reference
        // or World mode's model boxes.
        let plane = m.reference_plane.take();
        let boxes = m.boxes.as_mut().map(std::mem::take);
        m.draw(&self.device, &self.queue, &view, (width, height), (0, 0, width, height), camera, self.settings.fov, projection);
        m.reference_plane = plane;
        if let Some(b) = boxes {
            m.boxes = Some(b);
        }
        let bgra = matches!(format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
        read_texture(&self.device, &self.queue, &tex, width, height, bgra)
    }

    /// Tone map the current accumulation and read it back as RGBA8.
    pub fn read_pixels(&self) -> Result<Vec<u8>> {
        self.write_params();
        let t = &self.targets;
        let size = (t.width * t.height * 4) as u64;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.encode_denoise(&mut enc);
        self.encode_bloom(&mut enc);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("resolve"), timestamp_writes: None });
            pass.set_pipeline(&self.pipelines.resolve);
            pass.set_bind_group(0, &t.resolve_bg, &[]);
            pass.dispatch_workgroups(t.width.div_ceil(8), t.height.div_ceil(8), 1);
        }
        enc.copy_buffer_to_buffer(&t.out, 0, &readback, 0, size);
        self.queue.submit([enc.finish()]);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        let pixels = slice.get_mapped_range()?.to_vec();
        Ok(pixels)
    }
}

/// Pixel position of world point `p` in a `w` x `h` image seen from
/// `camera`, if it is in front of it. Every projection but orthographic
/// counts as perspective, like the modeler.
pub fn project_point(camera: &Camera, fov: f32, projection: Projection, w: f32, h: f32, p: Vec3) -> Option<(f32, f32)> {
    let (fwd, right, up) = camera.basis();
    let tan_half = (fov.to_radians() * 0.5).tan();
    let aspect = w / h;
    let d = p - camera.eye();
    let z = d.dot(fwd);
    let (x, y) = if projection == Projection::Orthographic {
        let half = camera.dist * tan_half;
        (d.dot(right) / (half * aspect), d.dot(up) / half)
    } else {
        if z <= 1e-3 {
            return None;
        }
        (d.dot(right) / (z * tan_half * aspect), d.dot(up) / (z * tan_half))
    };
    Some(((x + 1.0) * 0.5 * w, (1.0 - y) * 0.5 * h))
}

/// Read an 8-bit color texture back as RGBA8 (swapping channels for BGRA).
pub fn read_texture(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture, w: u32, h: u32, bgra: bool) -> Result<Vec<u8>> {
    let row = (w * 4).next_multiple_of(256);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("texture_readback"),
        size: (row * h) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |res| res.expect("map failed"));
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let data = slice.get_mapped_range()?;
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for px in data[(y * row) as usize..(y * row + w * 4) as usize].chunks_exact(4) {
            rgba.extend_from_slice(&if bgra { [px[2], px[1], px[0], 255] } else { [px[0], px[1], px[2], 255] });
        }
    }
    Ok(rgba)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::MaterialKind;
    use glam::IVec3;

    /// Pixels at most `level` in every channel.
    fn dark(px: &[u8], level: u8) -> usize {
        px.chunks_exact(4).filter(|p| p[0] <= level && p[1] <= level && p[2] <= level).count()
    }

    #[test]
    fn denoiser_contains_bad_samples() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let (_a, device, queue) = pollster::block_on(request_device(&instance, None)).expect("gpu");
        let mut scene = Scene::empty(IVec3::splat(16));
        for x in 4..12 {
            for y in 4..12 {
                for z in 0..6 {
                    scene.volume.set(IVec3::new(x, y, z), 10);
                }
            }
        }
        scene.volume.set(IVec3::new(7, 2, 1), 20);
        scene.materials[20].kind = MaterialKind::Emit as u32;
        scene.materials[20].emit = 20.0;
        scene.rebuild_lights();
        // An odd size, so workgroups overhang both edges.
        let (w, h) = (101, 77);
        let mut r = Renderer::new(device, queue, &scene, None, w, h);
        r.settings.denoise = true;
        r.settings.aces = true;
        r.settings.bg_mode = BgMode::Color;
        r.settings.bg_color = Vec3::splat(0.86);
        r.set_lights(&scene.lights, scene.light_power);
        r.set_camera(Camera::framing(scene.volume.dims.as_vec3(), r.settings.fov, w as f32 / h as f32));
        let clean = r.render_image(16).unwrap();
        assert_eq!(dark(&clean, 2), 0, "a light scene renders without black pixels");

        // One NaN sample in the middle used to black out a 125 px block.
        let i = (h / 2 * w + w / 2) as u64 * 16;
        r.queue.write_buffer(&r.targets.accum, i, bytemuck::bytes_of(&[f32::NAN, f32::INFINITY, 0.5f32]));
        let poisoned = r.read_pixels().unwrap();
        assert_eq!(dark(&poisoned, 2), 0, "a bad sample spread through the denoiser");
    }
}
