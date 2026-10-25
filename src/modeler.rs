//! The modeler view: a rasterized mesh of exposed voxel faces, shaded like
//! MagicaVoxel's editor (its GLSL ships unencrypted in the binary), with
//! per-vertex ambient occlusion and a ground grid under the volume.

use glam::{IVec3, Mat4, Vec3};

use crate::gpu::Camera;
use crate::scene::{Mask, Projection, Volume};

const MODELER_WGSL: &str = include_str!("shaders/modeler.wgsl");
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SHADOW_SIZE: u32 = 2048;
/// Direction the modeler's light comes from. Tilted off every axis so no
/// face is lit by the ambient term alone.
const LIGHT: Vec3 = Vec3::new(0.42, 0.55, 0.72);
/// `kind` of the ground quad's vertices; voxel faces use 0..6.
const FLOOR: u32 = 6;
/// Vertex `data` bit marking context (other models) faces.
const CONTEXT: u32 = 1 << 23;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// Corner position; w = corner within the face (bit 0 along the
    /// face's first tangent, bit 1 along the second).
    pub pos: [u16; 4],
    /// Bits 0-2 face kind (+x -x +y -y +z -z floor), 8-15 color index,
    /// 16-17 ambient occlusion (0 = fully occluded, 3 = open), 18 selected,
    /// 19-22 which face edges are outline edges (tangent 1 at 0 / 1,
    /// tangent 2 at 0 / 1).
    pub data: u32,
}

fn pack(kind: u32, color: u8, ao: u32) -> u32 {
    kind | (color as u32) << 8 | ao << 16
}

/// Exposed faces of every voxel as triangles, plus the ground quad.
pub fn build_mesh(v: &Volume, sel: Option<&Mask>) -> Vec<Vertex> {
    let mut out = Vec::new();
    let solid = |p: IVec3| v.get(p) != 0;
    for (i, c) in v.cells.filled() {
        let p = v.position(i);
        let selected = (sel.is_some_and(|s| s[i]) as u32) << 18;
        for axis in 0..3 {
            for (side, sign) in [(0u32, 1), (1, -1)] {
                let mut n = IVec3::ZERO;
                n[axis] = sign;
                if solid(p + n) {
                    continue;
                }
                let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
                let mut eu = IVec3::ZERO;
                eu[u] = 1;
                let mut ew = IVec3::ZERO;
                ew[w] = 1;
                // Corners counter-clockwise seen from outside.
                let mut corners = [(0, 0), (1, 0), (1, 1), (0, 1)];
                if sign < 0 {
                    corners.reverse();
                }
                let base = p + if sign > 0 { n } else { IVec3::ZERO };
                let front = p + n;
                // An edge is an outline unless the surface continues flat
                // in the same color past it.
                let mut edges = 0;
                for (bit, d) in [(0, -eu), (1, eu), (2, -ew), (3, ew)] {
                    let q = p + d;
                    if !(v.get(q) == c && !solid(q + n)) {
                        edges |= 1 << bit;
                    }
                }
                let edges = edges << 19;
                let verts = corners.map(|(a, b)| {
                    let (du, dw) = (eu * (2 * a - 1), ew * (2 * b - 1));
                    let (s1, s2, k) = (solid(front + du), solid(front + dw), solid(front + du + dw));
                    let ao = if s1 && s2 { 0 } else { 3 - (s1 as u32 + s2 as u32 + k as u32) };
                    let q = base + eu * a + ew * b;
                    let corner = (a | b << 1) as u16;
                    (Vertex { pos: [q.x as u16, q.y as u16, q.z as u16, corner], data: pack(axis as u32 * 2 + side, c, ao) | selected | edges }, ao)
                });
                // Split along the brighter diagonal so AO interpolates evenly.
                let order = if verts[0].1 + verts[2].1 >= verts[1].1 + verts[3].1 { [0, 1, 2, 0, 2, 3] } else { [1, 2, 3, 1, 3, 0] };
                out.extend(order.map(|k| verts[k].0));
            }
        }
    }
    let (x, y) = (v.dims.x as u16, v.dims.y as u16);
    let floor = |x, y| Vertex { pos: [x, y, 0, 0], data: pack(FLOOR, 0, 3) };
    out.extend([floor(0, 0), floor(x, 0), floor(x, y), floor(0, 0), floor(x, y), floor(0, y)]);
    out
}

/// Display options from MagicaVoxel's modeler "Display" panel.
#[derive(Clone, Copy, Debug)]
pub struct ModelerOptions {
    pub ao: bool,
    /// Outline voxel edges where the surface changes [Edge].
    pub edge: bool,
    pub grid: bool,
    pub constant: bool,
    pub frame: bool,
    /// Outline of the volume bounds (MagicaVoxel's `rm_show bo`).
    pub bounds: bool,
    /// Cast shadows from the modeler light [SW].
    pub shadow: bool,
    /// Draw voxels as wireframe outlines [WR].
    pub wire: bool,
    pub background: Vec3,
    pub ambient: Vec3,
}

impl Default for ModelerOptions {
    fn default() -> Self {
        ModelerOptions {
            ao: true,
            edge: false,
            grid: false,
            constant: false,
            frame: false,
            bounds: true,
            shadow: false,
            wire: false,
            background: Vec3::new(0.17, 0.17, 0.17),
            // `ambient_color` from MagicaVoxel's config.txt.
            ambient: Vec3::new(0.19, 0.18, 0.26),
        }
    }
}

/// Must match `U` in modeler.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    light: [f32; 4],
    ambient: [f32; 4],
    grid: [f32; 4],
    floor: [f32; 4],
    opts: [f32; 4],
    /// rgb outline color, w = outlines on.
    edge: [f32; 4],
    /// Bounding box line color (with alpha).
    line: [f32; 4],
    /// Light-space transform for the shadow map.
    shadow_vp: [[f32; 4]; 4],
    /// x shadows on, y wireframe on.
    view: [f32; 4],
    /// Offset of the context mesh (other models).
    context: [f32; 4],
    /// rgb color context fades toward, w = how far.
    ghost: [f32; 4],
    /// Highlighted cells that hold voxels: rgb, w = opacity where visible.
    mark: [f32; 4],
    /// Highlighted empty cells: rgb, w = opacity where hidden.
    mark_empty: [f32; 4],
    palette: [[f32; 4]; 256],
}

pub struct Modeler {
    pub options: ModelerOptions,
    pipeline: wgpu::RenderPipeline,
    /// Same as `pipeline` without back-face culling, for wireframe.
    wire_pipeline: wgpu::RenderPipeline,
    /// Depth-only pass from the light into `shadow_view`.
    shadow_pipeline: wgpu::RenderPipeline,
    shadow_view: wgpu::TextureView,
    uniforms: wgpu::Buffer,
    /// Uniforms plus the shadow map, for the voxel passes.
    bind_group: wgpu::BindGroup,
    /// Uniforms only, for passes that render into the shadow map or draw
    /// lines.
    plain_bind_group: wgpu::BindGroup,
    vertices: wgpu::Buffer,
    /// Bounding box edges, depth-tested so voxels hide the back edges.
    line_pipeline: wgpu::RenderPipeline,
    bounds: wgpu::Buffer,
    vertex_count: u32,
    /// The other models while editing one, drawn dimmed.
    context: Option<(wgpu::Buffer, u32, Vec3)>,
    /// Faces of highlighted cells (color 1 solid, 2 empty), drawn over
    /// everything, faint where hidden.
    highlight: Option<(wgpu::Buffer, u32)>,
    mark_pipeline: wgpu::RenderPipeline,
    mark_xray_pipeline: wgpu::RenderPipeline,
    palette: [[f32; 4]; 256],
    dims: Vec3,
    depth: Option<(wgpu::TextureView, (u32, u32))>,
}

impl Modeler {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, volume: &Volume, palette: &[[u8; 4]]) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("modeler"),
            source: wgpu::ShaderSource::Wgsl(MODELER_WGSL.into()),
        });
        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let plain_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("modeler_plain"), entries: &[uniform_entry] });
        let shaded_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("modeler_shaded"),
            entries: &[
                uniform_entry,
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let layout = |bgl: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(bgl)], immediate_size: 0 })
        };
        let (plain_pl, shaded_pl) = (layout(&plain_layout), layout(&shaded_layout));
        let vertex_layout = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Uint16x4, 1 => Uint32],
        })];
        let depth = |write: bool, compare: wgpu::CompareFunction| {
            Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(write),
                depth_compare: Some(compare),
                stencil: Default::default(),
                bias: Default::default(),
            })
        };
        let voxels = |label: &str, cull: Option<wgpu::Face>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&shaded_pl),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &vertex_layout },
                primitive: wgpu::PrimitiveState { cull_mode: cull, ..Default::default() },
                depth_stencil: depth(true, wgpu::CompareFunction::Less),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline = voxels("modeler", Some(wgpu::Face::Back));
        let wire_pipeline = voxels("modeler_wire", None);
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("modeler_shadow"),
            layout: Some(&plain_pl),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_shadow"), compilation_options: Default::default(), buffers: &vertex_layout },
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            }),
            multisample: Default::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("modeler_lines"),
            layout: Some(&plain_pl),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_line"), compilation_options: Default::default(), buffers: &vertex_layout },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
            depth_stencil: depth(false, wgpu::CompareFunction::LessEqual),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_line"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let mark = |label: &str, entry: &str, compare: wgpu::CompareFunction| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&plain_pl),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_mark"), compilation_options: Default::default(), buffers: &vertex_layout },
                primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..Default::default() },
                depth_stencil: depth(false, compare),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let mark_pipeline = mark("modeler_mark", "fs_mark", wgpu::CompareFunction::LessEqual);
        let mark_xray_pipeline = mark("modeler_mark_xray", "fs_mark_xray", wgpu::CompareFunction::Always);
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("modeler_uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow_view = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("modeler_shadow"),
                size: wgpu::Extent3d { width: SHADOW_SIZE, height: SHADOW_SIZE, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("modeler_shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("modeler"),
            layout: &shaded_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&shadow_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        let plain_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("modeler_plain"),
            layout: &plain_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }],
        });
        let (vertices, vertex_count) = vertex_buffer(device, volume, None);
        let mut m = Modeler {
            options: ModelerOptions::default(),
            pipeline,
            wire_pipeline,
            shadow_pipeline,
            shadow_view,
            uniforms,
            bind_group,
            plain_bind_group,
            vertices,
            line_pipeline,
            bounds: bounds_buffer(device, volume.dims),
            vertex_count,
            context: None,
            highlight: None,
            mark_pipeline,
            mark_xray_pipeline,
            palette: [[0.0; 4]; 256],
            dims: volume.dims.as_vec3(),
            depth: None,
        };
        m.set_palette(palette);
        m
    }

    pub fn upload_volume(&mut self, device: &wgpu::Device, volume: &Volume, sel: Option<&Mask>) {
        (self.vertices, self.vertex_count) = vertex_buffer(device, volume, sel);
        if volume.dims.as_vec3() != self.dims {
            self.bounds = bounds_buffer(device, volume.dims);
        }
        self.dims = volume.dims.as_vec3();
    }

    /// Show `context` (voxels and the offset of their cell 0) dimmed
    /// around the edited volume, or nothing.
    pub fn set_context(&mut self, device: &wgpu::Device, context: Option<&(Volume, IVec3)>) {
        self.context = context.and_then(|(v, offset)| {
            let mut mesh = build_mesh(v, None);
            mesh.truncate(mesh.len().saturating_sub(6));
            if mesh.is_empty() {
                return None;
            }
            for m in &mut mesh {
                m.data |= CONTEXT;
            }
            Some((mesh_buffer(device, &mesh), mesh.len() as u32, offset.as_vec3()))
        });
    }

    /// Highlight `cells` (with whether each holds a voxel) of a volume of
    /// size `dims`, or nothing if empty.
    pub fn set_highlight(&mut self, device: &wgpu::Device, dims: IVec3, cells: &[(IVec3, bool)]) {
        self.highlight = highlight_mesh(dims, cells).map(|mesh| (mesh_buffer(device, &mesh), mesh.len() as u32));
    }

    /// The modeler shows palette colors as-is, like MagicaVoxel.
    pub fn set_palette(&mut self, palette: &[[u8; 4]]) {
        for (dst, src) in self.palette.iter_mut().zip(palette) {
            *dst = src.map(|c| c as f32 / 255.0);
        }
    }

    /// Draw into `view` (of size `target`), inside the `viewport` rectangle.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        target: (u32, u32),
        viewport: (u32, u32, u32, u32),
        camera: &Camera,
        fov: f32,
        projection: Projection,
    ) {
        if self.depth.as_ref().is_none_or(|(_, size)| *size != target) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("modeler_depth"),
                size: wgpu::Extent3d { width: target.0, height: target.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth = Some((tex.create_view(&Default::default()), target));
        }
        let (x, y, w, h) = viewport;
        let o = &self.options;
        let light = LIGHT.normalize();
        let u = Uniforms {
            view_proj: view_proj(camera, fov, projection, w as f32 / h as f32, self.dims).to_cols_array_2d(),
            light: light.extend(0.0).to_array(),
            ambient: o.ambient.extend(0.0).to_array(),
            grid: [0.0, 0.0, 0.0, 0.75],
            floor: (o.background * 1.25 + 0.02).extend(10.0).to_array(),
            opts: [o.grid as u32 as f32, o.ao as u32 as f32, o.constant as u32 as f32, o.frame as u32 as f32],
            edge: [0.0, 0.0, 0.0, o.edge as u32 as f32],
            line: [1.0, 1.0, 1.0, 0.45],
            shadow_vp: shadow_view_proj(self.dims).to_cols_array_2d(),
            view: [o.shadow as u32 as f32, o.wire as u32 as f32, 0.0, 0.0],
            context: self.context.as_ref().map_or(Vec3::ZERO, |c| c.2).extend(0.0).to_array(),
            ghost: o.background.extend(0.6).to_array(),
            mark: [1.0, 0.15, 0.85, 0.55],
            mark_empty: [1.0, 0.85, 0.1, 0.3],
            palette: self.palette,
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));
        let bg = o.background;
        let mut enc = device.create_command_encoder(&Default::default());
        if o.shadow {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("modeler_shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(0, &self.plain_bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertices.slice(..));
            // Voxels only: the ground quad (last 6 vertices) receives shadows.
            pass.draw(0..self.vertex_count - 6, 0..1);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("modeler"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: bg.x as f64, g: bg.y as f64, b: bg.z as f64, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth.as_ref().unwrap().0,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
            pass.set_pipeline(if o.wire { &self.wire_pipeline } else { &self.pipeline });
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertices.slice(..));
            // The ground quad (the last 6 vertices) shows with grid lines.
            let n = if o.grid || o.frame { self.vertex_count } else { self.vertex_count - 6 };
            pass.draw(0..n, 0..1);
            if let Some((buf, count, _)) = &self.context {
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..*count, 0..1);
            }
            if let Some((buf, count)) = &self.highlight {
                pass.set_bind_group(0, &self.plain_bind_group, &[]);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.set_pipeline(&self.mark_xray_pipeline);
                pass.draw(0..*count, 0..1);
                pass.set_pipeline(&self.mark_pipeline);
                pass.draw(0..*count, 0..1);
            }
            if o.bounds {
                pass.set_pipeline(&self.line_pipeline);
                pass.set_bind_group(0, &self.plain_bind_group, &[]);
                pass.set_vertex_buffer(0, self.bounds.slice(..));
                pass.draw(0..24, 0..1);
            }
        }
        queue.submit([enc.finish()]);
    }
}

/// Outer faces of the highlighted cells inside `dims`: color 1 where the
/// cell holds a voxel, 2 where it's empty (e.g. just erased).
fn highlight_mesh(dims: IVec3, cells: &[(IVec3, bool)]) -> Option<Vec<Vertex>> {
    let mut v = Volume::new(dims);
    let mut any = false;
    for &(p, solid) in cells {
        if v.contains(p) {
            let i = v.index(p);
            v.set_index(i, if solid { 1 } else { 2 });
            any = true;
        }
    }
    if !any {
        return None;
    }
    let mut mesh = build_mesh(&v, None);
    mesh.truncate(mesh.len() - 6);
    Some(mesh)
}

/// The 12 edges of the canvas box as a line list.
fn bounds_buffer(device: &wgpu::Device, dims: IVec3) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    let corner = |i: u32| {
        let p = IVec3::new((i & 1) as i32, ((i >> 1) & 1) as i32, (i >> 2) as i32) * dims;
        Vertex { pos: [p.x as u16, p.y as u16, p.z as u16, 0], data: 0 }
    };
    let mut lines = Vec::with_capacity(24);
    for a in 0..8u32 {
        for bit in [1, 2, 4] {
            if a & bit == 0 {
                lines.extend([corner(a), corner(a | bit)]);
            }
        }
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("modeler_bounds"),
        contents: bytemuck::cast_slice(&lines),
        usage: wgpu::BufferUsages::VERTEX,
    })
}

fn vertex_buffer(device: &wgpu::Device, volume: &Volume, sel: Option<&Mask>) -> (wgpu::Buffer, u32) {
    let mesh = build_mesh(volume, sel);
    (mesh_buffer(device, &mesh), mesh.len() as u32)
}

fn mesh_buffer(device: &wgpu::Device, mesh: &[Vertex]) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("modeler_mesh"),
        size: std::mem::size_of_val(mesh) as u64,
        usage: wgpu::BufferUsages::VERTEX,
        mapped_at_creation: true,
    });
    buf.slice(..).get_mapped_range_mut().expect("mapped at creation").copy_from_slice(bytemuck::cast_slice(mesh));
    buf.unmap();
    buf
}

/// Right-handed view matrix looking from `eye` along `fwd`.
fn look(eye: Vec3, fwd: Vec3) -> Mat4 {
    let f = fwd.normalize();
    let s = f.cross(Vec3::Z).normalize();
    let u = s.cross(f);
    Mat4::from_cols(
        Vec3::new(s.x, u.x, -f.x).extend(0.0),
        Vec3::new(s.y, u.y, -f.y).extend(0.0),
        Vec3::new(s.z, u.z, -f.z).extend(0.0),
        Vec3::new(-s.dot(eye), -u.dot(eye), f.dot(eye)).extend(1.0),
    )
}

/// Orthographic projection from the light covering the whole canvas.
fn shadow_view_proj(dims: Vec3) -> Mat4 {
    let radius = dims.length() * 0.5 + 1.0;
    let center = dims * 0.5;
    let dir = LIGHT.normalize();
    let view = look(center + dir * radius * 2.0, -dir);
    let (near, far) = (radius, radius * 3.0);
    let proj = Mat4::from_cols(
        glam::Vec4::new(1.0 / radius, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, 1.0 / radius, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, -1.0 / (far - near), 0.0),
        glam::Vec4::new(0.0, 0.0, -near / (far - near), 1.0),
    );
    proj * view
}

/// Same camera model as the path tracer's perspective and ortho rays,
/// with wgpu's 0..1 clip depth.
fn view_proj(camera: &Camera, fov: f32, projection: Projection, aspect: f32, dims: Vec3) -> Mat4 {
    let eye = camera.eye();
    let (f, s, u) = camera.basis();
    let view = Mat4::from_cols(
        Vec3::new(s.x, u.x, -f.x).extend(0.0),
        Vec3::new(s.y, u.y, -f.y).extend(0.0),
        Vec3::new(s.z, u.z, -f.z).extend(0.0),
        Vec3::new(-s.dot(eye), -u.dot(eye), f.dot(eye)).extend(1.0),
    );
    let reach = camera.dist + dims.length() * 4.0 + 100.0;
    let tan_half = (fov.to_radians() * 0.5).tan();
    let proj = if projection == Projection::Orthographic {
        // Clip depth spans -reach..reach in front of and behind the eye.
        let half = camera.dist * tan_half;
        let rd = 1.0 / (-2.0 * reach);
        Mat4::from_cols(
            glam::Vec4::new(1.0 / (half * aspect), 0.0, 0.0, 0.0),
            glam::Vec4::new(0.0, 1.0 / half, 0.0, 0.0),
            glam::Vec4::new(0.0, 0.0, rd, 0.0),
            glam::Vec4::new(0.0, 0.0, -rd * reach, 1.0),
        )
    } else {
        let near = (camera.dist * 0.01).clamp(0.01, 1.0);
        let r = reach / (near - reach);
        Mat4::from_cols(
            glam::Vec4::new(1.0 / (tan_half * aspect), 0.0, 0.0, 0.0),
            glam::Vec4::new(0.0, 1.0 / tan_half, 0.0, 0.0),
            glam::Vec4::new(0.0, 0.0, r, -1.0),
            glam::Vec4::new(0.0, 0.0, r * near, 0.0),
        )
    };
    proj * view
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_voxel_has_six_open_faces() {
        let mut v = Volume::new(IVec3::new(3, 3, 3));
        let i = v.index(IVec3::ONE);
        v.set_index(i, 7);
        let mesh = build_mesh(&v, None);
        assert_eq!(mesh.len(), 6 * 6 + 6);
        assert!(mesh[..36].iter().all(|m| (m.data >> 16) & 3 == 3 && (m.data >> 8) & 255 == 7));
    }

    #[test]
    fn highlight_marks_solid_and_empty_cells() {
        let dims = IVec3::splat(4);
        assert!(highlight_mesh(dims, &[(IVec3::splat(9), true)]).is_none());
        let mesh = highlight_mesh(dims, &[(IVec3::ZERO, true), (IVec3::X, false)]).unwrap();
        // Two touching cubes: 10 faces, no ground quad.
        assert_eq!(mesh.len(), 10 * 6);
        assert!(mesh.iter().any(|m| (m.data >> 8) & 255 == 2) && mesh.iter().any(|m| (m.data >> 8) & 255 == 1));
    }

    #[test]
    fn hidden_faces_are_culled_and_corners_darken() {
        let mut v = Volume::new(IVec3::new(4, 4, 4));
        for p in [IVec3::new(1, 1, 0), IVec3::new(2, 1, 0), IVec3::new(1, 1, 1)] {
            let i = v.index(p);
            v.set_index(i, 1);
        }
        let mesh = build_mesh(&v, None);
        // 3 cubes, 18 faces, minus 2 shared pairs.
        assert_eq!(mesh.len(), (18 - 4) * 6 + 6);
        // The top of (2,1,0) touches the cube above (1,1,1) along one edge.
        assert!(mesh.iter().any(|m| m.data & 7 == 4 && (m.data >> 16) & 3 < 3));
    }
}
