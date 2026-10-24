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
    // rgb outline color, w = outlines on.
    edge: vec4<f32>,
    // Bounding box line color.
    line: vec4<f32>,
    // Light-space transform for the shadow map.
    shadow_vp: mat4x4<f32>,
    // x shadows on, y wireframe on.
    view: vec4<f32>,
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
    let lgrad = max(fwidth(in.local), vec2<f32>(1e-6));

    var shade = 1.0;
    if u.view.x > 0.0 && dot(u.light.xyz, n) > 0.0 {
        shade = mix(0.55, 1.0, lit(in.pos + n * 0.02));
    }

    if kind == 6u {
        var c = u.floor.rgb * shade;
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
        c *= u.ambient.rgb + (max(d, 0.0) * shade + max(-d, 0.0) * 0.8) * 1.3;
    }
    if u.opts.y > 0.0 {
        c *= mix(0.45, 1.0, in.ao);
    }
    if u.view.y > 0.0 {
        // Wireframe: keep only a thin border around each voxel face.
        let d = min(in.local, 1.0 - in.local) / lgrad;
        if min(d.x, d.y) > 1.0 {
            discard;
        }
    }
    if u.edge.w > 0.0 {
        // Distance in pixels to each outline edge of this face.
        let e = (in.data >> 19u) & 15u;
        let d0 = in.local / lgrad;
        let d1 = (1.0 - in.local) / lgrad;
        var d = 1e6;
        if (e & 1u) != 0u { d = min(d, d0.x); }
        if (e & 2u) != 0u { d = min(d, d1.x); }
        if (e & 4u) != 0u { d = min(d, d0.y); }
        if (e & 8u) != 0u { d = min(d, d1.y); }
        c = mix(u.edge.rgb, c, smoothstep(0.5, 1.5, d) * 0.7 + 0.3);
    }
    if (in.data & (1u << 18u)) != 0u {
        // Selected: tint toward the selection color and outline each voxel.
        c = mix(c, vec3<f32>(0.25, 0.65, 1.0), 0.45);
        c = mix(vec3<f32>(0.1, 0.35, 0.9), c, lines(uv, grad, 1.0));
    }
    return vec4<f32>(c, 1.0);
}

// Bounding box edges. Pulled slightly toward the camera so edges lying on
// voxel faces still show; voxels in front hide the rest.
@vertex
fn vs_line(@location(0) pos: vec4<u32>, @location(1) data: u32) -> @builtin(position) vec4<f32> {
    var c = u.view_proj * vec4<f32>(vec3<f32>(pos.xyz), 1.0);
    c.z -= 2e-4 * c.w;
    return c;
}

@fragment
fn fs_line() -> @location(0) vec4<f32> {
    return u.line;
}
