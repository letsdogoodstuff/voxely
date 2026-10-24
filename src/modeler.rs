//! The modeler view: a rasterized mesh of exposed voxel faces, shaded like
//! MagicaVoxel's editor (its GLSL ships unencrypted in the binary), with
//! per-vertex ambient occlusion and a ground grid under the volume.

use glam::{IVec3, Mat4, Vec3};

use crate::gpu::Camera;
use crate::scene::{Projection, Volume};

const MODELER_WGSL: &str = include_str!("shaders/modeler.wgsl");
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// `kind` of the ground quad's vertices; voxel faces use 0..6.
const FLOOR: u32 = 6;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// Corner position; w unused.
    pub pos: [u16; 4],
    /// Bits 0-2 face kind (+x -x +y -y +z -z floor), 8-15 color index,
    /// 16-17 ambient occlusion (0 = fully occluded, 3 = open).
    pub data: u32,
}

fn pack(kind: u32, color: u8, ao: u32) -> u32 {
    kind | (color as u32) << 8 | ao << 16
}

/// Exposed faces of every voxel as triangles, plus the ground quad.
pub fn build_mesh(v: &Volume) -> Vec<Vertex> {
    let mut out = Vec::new();
    let solid = |p: IVec3| v.get(p) != 0;
    for (i, &c) in v.cells.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let p = v.position(i);
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
                let verts = corners.map(|(a, b)| {
                    let (du, dw) = (eu * (2 * a - 1), ew * (2 * b - 1));
                    let (s1, s2, k) = (solid(front + du), solid(front + dw), solid(front + du + dw));
                    let ao = if s1 && s2 { 0 } else { 3 - (s1 as u32 + s2 as u32 + k as u32) };
                    let q = base + eu * a + ew * b;
                    (Vertex { pos: [q.x as u16, q.y as u16, q.z as u16, 0], data: pack(axis as u32 * 2 + side, c, ao) }, ao)
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
    pub grid: bool,
    pub constant: bool,
    pub frame: bool,
    /// Outline of the volume bounds (MagicaVoxel's `rm_show bo`).
    pub bounds: bool,
    pub background: Vec3,
    pub ambient: Vec3,
}

impl Default for ModelerOptions {
    fn default() -> Self {
        ModelerOptions {
            ao: true,
            grid: false,
            constant: false,
            frame: true,
            bounds: true,
            background: Vec3::new(0.32, 0.32, 0.34),
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
    palette: [[f32; 4]; 256],
}

pub struct Modeler {
    pub options: ModelerOptions,
    pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertices: wgpu::Buffer,
    vertex_count: u32,
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
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("modeler"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Uint16x4, 1 => Uint32],
                })],
            },
            primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("modeler_uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("modeler"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }],
        });
        let (vertices, vertex_count) = vertex_buffer(device, volume);
        let mut m = Modeler {
            options: ModelerOptions::default(),
            pipeline,
            uniforms,
            bind_group,
            vertices,
            vertex_count,
            palette: [[0.0; 4]; 256],
            dims: volume.dims.as_vec3(),
            depth: None,
        };
        m.set_palette(palette);
        m
    }

    pub fn upload_volume(&mut self, device: &wgpu::Device, volume: &Volume) {
        (self.vertices, self.vertex_count) = vertex_buffer(device, volume);
        self.dims = volume.dims.as_vec3();
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
        // Tilted off every axis so no face is lit by the ambient term alone.
        let light = Vec3::new(0.42, 0.55, 0.72).normalize();
        let u = Uniforms {
            view_proj: view_proj(camera, fov, projection, w as f32 / h as f32, self.dims).to_cols_array_2d(),
            light: light.extend(0.0).to_array(),
            ambient: o.ambient.extend(0.0).to_array(),
            grid: [0.0, 0.0, 0.0, 0.75],
            floor: (o.background * 0.8).extend(10.0).to_array(),
            opts: [o.grid as u32 as f32, o.ao as u32 as f32, o.constant as u32 as f32, o.frame as u32 as f32],
            palette: self.palette,
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));
        let bg = o.background;
        let mut enc = device.create_command_encoder(&Default::default());
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertices.slice(..));
            pass.draw(0..self.vertex_count, 0..1);
        }
        queue.submit([enc.finish()]);
    }
}

fn vertex_buffer(device: &wgpu::Device, volume: &Volume) -> (wgpu::Buffer, u32) {
    let mesh = build_mesh(volume);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("modeler_mesh"),
        size: std::mem::size_of_val(mesh.as_slice()) as u64,
        usage: wgpu::BufferUsages::VERTEX,
        mapped_at_creation: true,
    });
    buf.slice(..).get_mapped_range_mut().expect("mapped at creation").copy_from_slice(bytemuck::cast_slice(&mesh));
    buf.unmap();
    (buf, mesh.len() as u32)
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
        let mesh = build_mesh(&v);
        assert_eq!(mesh.len(), 6 * 6 + 6);
        assert!(mesh[..36].iter().all(|m| m.data >> 16 == 3 && (m.data >> 8) & 255 == 7));
    }

    #[test]
    fn hidden_faces_are_culled_and_corners_darken() {
        let mut v = Volume::new(IVec3::new(4, 4, 4));
        for p in [IVec3::new(1, 1, 0), IVec3::new(2, 1, 0), IVec3::new(1, 1, 1)] {
            let i = v.index(p);
            v.set_index(i, 1);
        }
        let mesh = build_mesh(&v);
        // 3 cubes, 18 faces, minus 2 shared pairs.
        assert_eq!(mesh.len(), (18 - 4) * 6 + 6);
        // The top of (2,1,0) touches the cube above (1,1,1) along one edge.
        assert!(mesh.iter().any(|m| m.data & 7 == 4 && m.data >> 16 < 3));
    }
}
