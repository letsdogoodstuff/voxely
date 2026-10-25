// A reference picture on a plane in the modeler: one textured quad.

struct U {
    view_proj: mat4x4<f32>,
    // Bottom-left corner, then the picture's right and up edges.
    corner: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    // x = opacity.
    opts: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var pic: texture_2d<f32>;
@group(0) @binding(2) var pic_sampler: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let t = quad[i];
    let p = u.corner.xyz + u.right.xyz * t.x + u.up.xyz * t.y;
    var out: VsOut;
    out.clip = u.view_proj * vec4<f32>(p, 1.0);
    // Texture rows run top down.
    out.uv = vec2<f32>(t.x, 1.0 - t.y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(pic, pic_sampler, in.uv);
    if c.a < 0.02 {
        discard;
    }
    return vec4<f32>(c.rgb, c.a * u.opts.x);
}
