// Modeler view. The voxel shading follows MagicaVoxel's editor shader:
// palette color, optional per-voxel grid lines, then
// color * (ambient + diffuse), with back-lit faces kept at 80%.

struct U {
    view_proj: mat4x4<f32>,
    light: vec4<f32>,
    ambient: vec4<f32>,
    // rgb line color, w = minimum line visibility.
    grid: vec4<f32>,
    // rgb ground color, w = frame line spacing.
    floor: vec4<f32>,
    // x grid, y ambient occlusion, z constant shading, w frame lines.
    opts: vec4<f32>,
    palette: array<vec4<f32>, 256>,
}

@group(0) @binding(0) var<uniform> u: U;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) pos: vec3<f32>,
    @location(1) ao: f32,
    @location(2) @interpolate(flat) data: u32,
}

const NORMALS = array<vec3<f32>, 7>(
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, -1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, -1.0),
    vec3<f32>(0.0, 0.0, 1.0),
);

@vertex
fn vs_main(@location(0) pos: vec4<u32>, @location(1) data: u32) -> VsOut {
    var o: VsOut;
    o.pos = vec3<f32>(pos.xyz);
    o.clip = u.view_proj * vec4<f32>(o.pos, 1.0);
    o.ao = f32((data >> 16u) & 3u) / 3.0;
    o.data = data;
    return o;
}

fn screen_grad(v: vec2<f32>) -> vec2<f32> {
    return max(abs(dpdx(v)), abs(dpdy(v)));
}

// 0 on a line of the given spacing, 1 away from it.
fn lines(uv: vec2<f32>, grad: vec2<f32>, spacing: f32) -> f32 {
    let d = abs(uv - floor(uv / spacing + 0.5) * spacing) / max(grad, vec2<f32>(1e-6));
    return smoothstep(0.0, 3.0, min(d.x, d.y));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let kind = in.data & 7u;
    let n = NORMALS[kind];
    let uv = vec2<f32>(dot(in.pos, n.zxy), dot(in.pos, n.yzx));
    // Derivatives must be taken in uniform control flow.
    let grad = screen_grad(uv);

    if kind == 6u {
        var c = u.floor.rgb;
        c = mix(c * 0.85, c, lines(uv, grad, 1.0));
        if u.opts.w > 0.0 {
            c = mix(c * 0.6, c, lines(uv, grad, u.floor.w));
        }
        return vec4<f32>(c, 1.0);
    }

    var c = u.palette[(in.data >> 8u) & 255u].rgb;
    if u.opts.x > 0.0 {
        let g = min(lines(uv, grad, 1.0) + u.grid.w, 1.0);
        c = mix(u.grid.rgb, c, g);
    }
    if u.opts.z <= 0.0 {
        let d = dot(u.light.xyz, n);
        c *= u.ambient.rgb + (max(d, 0.0) + max(-d, 0.0) * 0.8) * 1.3;
    }
    if u.opts.y > 0.0 {
        c *= mix(0.45, 1.0, in.ao);
    }
    return vec4<f32>(c, 1.0);
}
