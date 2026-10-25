// Modeler view. The voxel shading follows MagicaVoxel's editor shader:
// palette color, optional grid and frame lines, then
// color * (ambient + diffuse), with back-lit faces kept at 80%.

struct U {
    view_proj: mat4x4<f32>,
    light: vec4<f32>,
    ambient: vec4<f32>,
    // rgb grid and frame line color on the walls.
    line: vec4<f32>,
    // Grid then frame lines: x / z darkness on voxel faces, y / w opacity
    // on the walls.
    faces: vec4<f32>,
    // x grid, y ambient occlusion, z constant shading, w frame lines.
    opts: vec4<f32>,
    // rgb outline color, w = outlines on.
    edge: vec4<f32>,
    // xyz where frame lines are measured from, w their spacing.
    frame: vec4<f32>,
    // Light-space transform for the shadow map.
    shadow_vp: mat4x4<f32>,
    // x shadows on, y outline width in pixels, zw one pixel in clip units.
    view: vec4<f32>,
    // Camera position (w = 1), or direction toward the camera (w = 0, ortho).
    eye: vec4<f32>,
    // Offset of context faces (other models while editing one).
    context: vec4<f32>,
    // rgb color context fades toward, w = how far.
    ghost: vec4<f32>,
    // Highlighted cells holding voxels: rgb, w = opacity where visible.
    mark: vec4<f32>,
    // Highlighted empty cells: rgb, w = opacity where hidden.
    mark_empty: vec4<f32>,
    palette: array<vec4<f32>, 256>,
}

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) pos: vec3<f32>,
    @location(1) ao: f32,
    @location(2) @interpolate(flat) data: u32,
    // Position within the face, 0..1 along each tangent.
    @location(3) local: vec2<f32>,
}

const NORMALS = array<vec3<f32>, 6>(
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, -1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, -1.0),
);
const AXES = array<vec3<f32>, 3>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));

const CONTEXT = 1u << 23u;

@vertex
fn vs_main(@location(0) pos: vec4<u32>, @location(1) data: u32) -> VsOut {
    return vertex(pos, data);
}

fn vertex(pos: vec4<u32>, data: u32) -> VsOut {
    var o: VsOut;
    o.pos = vec3<f32>(pos.xyz);
    if (data & CONTEXT) != 0u {
        o.pos += u.context.xyz;
    }
    o.clip = u.view_proj * vec4<f32>(o.pos, 1.0);
    o.ao = f32((data >> 16u) & 3u) / 3.0;
    o.local = vec2<f32>(f32(pos.w & 1u), f32((pos.w >> 1u) & 1u));
    o.data = data;
    return o;
}

// Depth-only pass from the light.
@vertex
fn vs_shadow(@location(0) pos: vec4<u32>, @location(1) data: u32) -> @builtin(position) vec4<f32> {
    return u.shadow_vp * vec4<f32>(vec3<f32>(pos.xyz), 1.0);
}

// Fraction of light reaching `p` (4-tap PCF).
fn lit(p: vec3<f32>) -> f32 {
    let l = u.shadow_vp * vec4<f32>(p, 1.0);
    let uv = l.xy * vec2<f32>(0.5, -0.5) + 0.5;
    let texel = 1.0 / vec2<f32>(textureDimensions(shadow_map));
    var sum = 0.0;
    for (var i = 0; i < 4; i++) {
        let o = (vec2<f32>(f32(i & 1), f32(i >> 1u)) - 0.5) * texel;
        sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + o, l.z - 0.0015);
    }
    return sum * 0.25;
}

fn screen_grad(v: vec2<f32>) -> vec2<f32> {
    return max(abs(dpdx(v)), abs(dpdy(v)));
}

// 0 on a line of the given spacing, 1 more than `width` pixels from it.
fn lines(uv: vec2<f32>, grad: vec2<f32>, spacing: f32, width: f32) -> f32 {
    let d = abs(uv - floor(uv / spacing + 0.5) * spacing) / max(grad, vec2<f32>(1e-6));
    return smoothstep(0.0, width, min(d.x, d.y));
}

// A face's coordinates along its two tangents.
fn face_uv(p: vec3<f32>, n: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(dot(p, n.zxy), dot(p, n.yzx));
}

// Whether a face with normal `n` at `p` faces the camera.
fn facing(p: vec3<f32>, n: vec3<f32>) -> bool {
    let v = select(u.eye.xyz, u.eye.xyz - p, u.eye.w > 0.0);
    return dot(v, n) > 0.0;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let kind = in.data & 7u;
    let n = NORMALS[kind];
    let uv = face_uv(in.pos, n);
    let fuv = uv - face_uv(u.frame.xyz, n);
    // Derivatives must be taken in uniform control flow.
    let grad = screen_grad(uv);
    let lgrad = max(fwidth(in.local), vec2<f32>(1e-6));

    var c = u.palette[(in.data >> 8u) & 255u].rgb;
    if u.opts.x > 0.0 {
        c *= mix(u.faces.x, 1.0, lines(uv, grad, 1.0, 1.0));
    }
    if u.opts.w > 0.0 {
        c *= mix(u.faces.z, 1.0, lines(fuv, grad, u.frame.w, 1.5));
    }
    if u.opts.z <= 0.0 {
        let d = dot(u.light.xyz, n);
        c *= u.ambient.rgb + (max(d, 0.0) + max(-d, 0.0) * 0.8) * 1.3;
        // Shadows from the light darken the lit faces, like MagicaVoxel's.
        if u.view.x > 0.0 && d > 0.0 {
            c *= mix(0.72, 1.0, lit(in.pos + n * 0.02));
        }
    }
    if u.opts.y > 0.0 {
        c *= mix(0.45, 1.0, in.ao);
    }
    if u.edge.w > 0.0 {
        // Distance in pixels to the nearest outline edge of this face: a
        // silhouette or convex edge, or a crease whose other face shows.
        let e = (in.data >> 19u) & 15u;
        let concave = (in.data >> 24u) & 15u;
        let axis = kind / 2u;
        let t1 = AXES[(axis + 1u) % 3u];
        let t2 = AXES[(axis + 2u) % 3u];
        let dirs = array<vec3<f32>, 4>(-t1, t1, -t2, t2);
        let dist = array<f32, 4>(in.local.x / lgrad.x, (1.0 - in.local.x) / lgrad.x, in.local.y / lgrad.y, (1.0 - in.local.y) / lgrad.y);
        var d = 1e6;
        for (var i = 0u; i < 4u; i++) {
            let bit = 1u << i;
            if (e & bit) != 0u && ((concave & bit) == 0u || facing(in.pos, -dirs[i])) {
                d = min(d, dist[i]);
            }
        }
        c = mix(u.edge.rgb, c, smoothstep(u.view.y - 0.5, u.view.y + 0.5, d));
    }
    if (in.data & CONTEXT) != 0u {
        // Other models: shown for reference, faded toward the background.
        c = mix(c, u.ghost.rgb, u.ghost.w);
    }
    if (in.data & (1u << 18u)) != 0u {
        // Selected, like MagicaVoxel: the voxel keeps its color, a little
        // brighter, with a thin white outline.
        c = mix(c, vec3<f32>(1.0), 0.12);
        let d = abs(uv - floor(uv + 0.5)) / max(grad, vec2<f32>(1e-6));
        c = mix(vec3<f32>(1.0), c, smoothstep(0.4, 1.4, min(d.x, d.y)));
    }
    return vec4<f32>(c, 1.0);
}

// The canvas walls facing the camera from behind the model: faint lines
// every voxel [Grid] and brighter ones every few [Frame].
@fragment
fn fs_wall(in: VsOut) -> @location(0) vec4<f32> {
    let n = NORMALS[in.data & 7u];
    let uv = face_uv(in.pos, n);
    let fuv = uv - face_uv(u.frame.xyz, n);
    let grad = screen_grad(uv);
    let grid = (1.0 - lines(uv, grad, 1.0, 1.0)) * u.opts.x * u.faces.y;
    let frame = (1.0 - lines(fuv, grad, u.frame.w, 1.5)) * u.opts.w * u.faces.w;
    return vec4<f32>(u.line.rgb, max(grid, frame));
}

struct LineOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) color: vec3<f32>,
}

// Box edges (rgb in `data`), drawn a few times a fraction of a pixel apart
// for thicker lines. Pulled slightly toward the camera so edges lying on
// voxel faces still show; voxels in front hide the rest.
@vertex
fn vs_line(@location(0) pos: vec4<u32>, @location(1) data: u32, @builtin(instance_index) i: u32) -> LineOut {
    var o: LineOut;
    o.clip = u.view_proj * vec4<f32>(vec3<f32>(pos.xyz), 1.0);
    let shift = (vec2<f32>(f32(i & 1u), f32(i >> 1u)) - 0.5) * 1.5;
    o.clip += vec4<f32>(shift * u.view.zw * o.clip.w, -2e-4 * o.clip.w, 0.0);
    o.color = vec3<f32>(vec3<u32>(data, data >> 8u, data >> 16u) & vec3<u32>(255u)) / 255.0;
    return o;
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}

// Highlighted cells (color 1 solid, 2 empty), pulled toward the camera so
// they cover the voxel faces they sit on.
@vertex
fn vs_mark(@location(0) pos: vec4<u32>, @location(1) data: u32) -> VsOut {
    var o = vertex(pos, data);
    o.clip.z -= 3e-4 * o.clip.w;
    return o;
}

fn mark_color(in: VsOut) -> vec3<f32> {
    if ((in.data >> 8u) & 255u) == 2u {
        return u.mark_empty.rgb;
    }
    return u.mark.rgb;
}

// Where visible: a strong tint with each cell outlined.
@fragment
fn fs_mark(in: VsOut) -> @location(0) vec4<f32> {
    let n = NORMALS[in.data & 7u];
    let uv = face_uv(in.pos, n);
    let edge = 1.0 - lines(uv, screen_grad(uv), 1.0, 3.0);
    let c = mark_color(in);
    return vec4<f32>(mix(c, c * 0.35, edge), max(u.mark.w, edge));
}

// Through other voxels: a faint tint, so hidden edits still show.
@fragment
fn fs_mark_xray(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(mark_color(in), u.mark_empty.w);
}
