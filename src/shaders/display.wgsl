// Post-processing: choice of original / denoised image, bloom (threshold,
// quarter-res downsample, separable Gaussian blur), tone mapping, the
// background and edge/grid lines, either to the window (vs/fs) or into a
// packed RGBA8 buffer for headless renders (resolve).

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> accum: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out_rgba: array<u32>;
@group(0) @binding(3) var<storage, read_write> bloom_a: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> bloom_b: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> denoised: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> overlay: array<vec4<f32>>;

fn bloom_size() -> vec2<u32> {
    return (P.frame.yz + 3u) / 4u;
}

fn split_view() -> bool {
    return (P.aux.x & FLAG_DENOISE) != 0u && P.aux.y == VIEW_SPLIT;
}

// Averaged linear radiance (premultiplied by coverage) shown at `px`.
fn average(px: vec2<u32>) -> vec3<f32> {
    let i = px.y * P.frame.y + px.x;
    if (P.aux.x & FLAG_DENOISE) != 0u && !(split_view() && px.x < P.frame.y / 2u) {
        return denoised[i].rgb;
    }
    return accum[i].rgb * inv_samples(P);
}

@compute @workgroup_size(8, 8, 1)
fn bloom_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bs = bloom_size();
    if gid.x >= bs.x || gid.y >= bs.y {
        return;
    }
    var sum = vec3<f32>(0.0);
    for (var y = 0u; y < 4u; y++) {
        for (var x = 0u; x < 4u; x++) {
            let px = min(gid.xy * 4u + vec2<u32>(x, y), P.frame.yz - 1u);
            let c = average(px) * P.film.x;
            sum += max(c - vec3<f32>(P.bloom.w), vec3<f32>(0.0));
        }
    }
    bloom_a[gid.y * bs.x + gid.x] = vec4<f32>(sum / 16.0, 1.0);
}

fn blur(gid: vec2<u32>, dir: vec2<i32>, sigma: f32, horizontal: bool) -> vec3<f32> {
    let bs = bloom_size();
    let radius = i32(ceil(sigma * 3.0));
    // Stride so very wide kernels stay at 64 taps per side.
    let stride = max(1, radius / 64);
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var i = -radius; i <= radius; i += stride) {
        let p = clamp(vec2<i32>(gid) + dir * i, vec2<i32>(0), vec2<i32>(bs) - 1);
        let w = exp(-f32(i * i) / (2.0 * sigma * sigma));
        let idx = u32(p.y) * bs.x + u32(p.x);
        if horizontal {
            sum += bloom_a[idx].rgb * w;
        } else {
            sum += bloom_b[idx].rgb * w;
        }
        wsum += w;
    }
    return sum / wsum;
}

fn bloom_sigma() -> vec2<f32> {
    // Radius as a fraction of image height, stretched horizontally by aspect.
    let base = (0.01 + P.bloom.y * 0.1) * f32(bloom_size().y);
    return vec2<f32>(base * (1.0 + P.bloom.z * 4.0), base);
}

@compute @workgroup_size(8, 8, 1)
fn bloom_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bs = bloom_size();
    if gid.x >= bs.x || gid.y >= bs.y {
        return;
    }
    bloom_b[gid.y * bs.x + gid.x] = vec4<f32>(blur(gid.xy, vec2<i32>(1, 0), max(bloom_sigma().x, 0.5), true), 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn bloom_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bs = bloom_size();
    if gid.x >= bs.x || gid.y >= bs.y {
        return;
    }
    bloom_a[gid.y * bs.x + gid.x] = vec4<f32>(blur(gid.xy, vec2<i32>(0, 1), max(bloom_sigma().y, 0.5), false), 1.0);
}

fn bloom_at(uv: vec2<f32>) -> vec3<f32> {
    let bs = bloom_size();
    let p = uv * vec2<f32>(bs) - 0.5;
    let i = vec2<i32>(floor(p));
    let f = p - floor(p);
    let mx = vec2<i32>(bs) - 1;
    let a = bloom_a[u32(clamp(i.y, 0, mx.y)) * bs.x + u32(clamp(i.x, 0, mx.x))].rgb;
    let b = bloom_a[u32(clamp(i.y, 0, mx.y)) * bs.x + u32(clamp(i.x + 1, 0, mx.x))].rgb;
    let c = bloom_a[u32(clamp(i.y + 1, 0, mx.y)) * bs.x + u32(clamp(i.x, 0, mx.x))].rgb;
    let d = bloom_a[u32(clamp(i.y + 1, 0, mx.y)) * bs.x + u32(clamp(i.x + 1, 0, mx.x))].rgb;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn encode(c: vec3<f32>) -> vec3<f32> {
    return pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / P.film.y));
}

// Display color with straight alpha.
fn final_color(px: vec2<u32>, uv: vec2<f32>) -> vec4<f32> {
    let i = px.y * P.frame.y + px.x;
    let ic = inv_samples(P);
    var alpha = clamp(accum[i].w * ic, 0.0, 1.0);
    var c = average(px) / max(alpha, 1e-4);
    if P.bloom.x > 0.0 {
        // Bloom was computed after exposure; undo it so tonemap applies once.
        c += bloom_at(uv) * P.bloom.x / max(P.film.x, 1e-6);
    }
    var rgb = tonemap(P, c, uv);
    if P.aux.z == BG_COLOR {
        rgb = mix(encode(P.bg_col.rgb), rgb, alpha);
        alpha = 1.0;
    }
    if (P.aux.x & (FLAG_EDGES | FLAG_GRID)) != 0u {
        let o = overlay[i] * ic;
        let cov = clamp(o.w, 0.0, 1.0);
        if cov > 0.0 {
            let line = encode(o.rgb / o.w);
            let a = cov + alpha * (1.0 - cov);
            rgb = (line * cov + rgb * alpha * (1.0 - cov)) / a;
            alpha = a;
        }
    }
    let mid = P.frame.y / 2u;
    if split_view() && (px.x == mid || px.x + 1u == mid) {
        rgb = vec3<f32>(1.0);
        alpha = 1.0;
    }
    return vec4<f32>(rgb, alpha);
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VsOut;
    o.pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let pos = in.pos.xy - P.view.xy;
    let px = min(vec2<u32>(max(pos, vec2<f32>(0.0))), P.frame.yz - 1u);
    let uv = pos / vec2<f32>(P.frame.yz);
    let c = final_color(px, uv);
    // Transparent background over a checkerboard.
    let check = ((px.x >> 3u) + (px.y >> 3u)) & 1u;
    let bg = vec3<f32>(select(0.4, 0.6, check == 1u));
    return vec4<f32>(mix(bg, c.rgb, c.a), 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.frame.y || gid.y >= P.frame.z {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(P.frame.yz);
    out_rgba[gid.y * P.frame.y + gid.x] = pack4x8unorm(final_color(gid.xy, uv));
}
