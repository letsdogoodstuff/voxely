// Edge-aware à-trous wavelet denoiser (after SVGF, Schied et al. 2017)
// over the current accumulation. Radiance is divided by the first-hit
// albedo, filtered with a 5x5 B3-spline kernel at growing strides whose
// taps are weighted by normal, depth, albedo and luminance similarity
// (the last scaled by the estimated variance of the mean), and multiplied
// back by the albedo in the final pass.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> accum: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> aov_albedo: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> aov_normal: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> dst: array<vec4<f32>>;
// x: tap stride in pixels, y: 1 on the last pass.
@group(0) @binding(6) var<uniform> pass_info: vec4<u32>;

const SIGMA_L: f32 = 4.0;
const SIGMA_N: f32 = 16.0;
const SIGMA_A: f32 = 4.0;

fn albedo_at(i: u32) -> vec3<f32> {
    return max(aov_albedo[i].rgb * inv_samples(P), vec3<f32>(DEMOD_EPS));
}

// Demodulated radiance and the variance of its mean.
@compute @workgroup_size(8, 8, 1)
fn denoise_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= P.frame.y || gid.y >= P.frame.z {
        return;
    }
    let i = gid.y * P.frame.y + gid.x;
    let ic = inv_samples(P);
    let d = accum[i].rgb * ic / albedo_at(i);
    let l = luminance(d);
    let var_mean = max(aov_albedo[i].w * ic - l * l, 0.0) * ic;
    // A non-finite input would spread over every pass's footprint.
    dst[i] = select(vec4<f32>(0.0), vec4<f32>(d, var_mean), finite4(vec4<f32>(d, var_mean)));
}

fn depth_at(p: vec2<i32>) -> f32 {
    let q = clamp(p, vec2<i32>(0), vec2<i32>(P.frame.yz) - 1);
    return aov_normal[u32(q.y) * P.frame.y + u32(q.x)].w * inv_samples(P);
}

@compute @workgroup_size(8, 8, 1)
fn denoise_atrous(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<i32>(P.frame.yz);
    let p = vec2<i32>(gid.xy);
    if p.x >= size.x || p.y >= size.y {
        return;
    }
    let ic = inv_samples(P);
    let stride = i32(pass_info.x);
    let i = gid.y * P.frame.y + gid.x;
    let center = src[i];
    let nd = aov_normal[i] * ic;
    let a_p = albedo_at(i);
    let l_p = luminance(center.rgb);

    // Depth slope per pixel, from the smoother of the two one-sided
    // differences so discontinuities don't inflate it.
    let z = nd.w;
    let gx = min(abs(depth_at(p + vec2<i32>(1, 0)) - z), abs(z - depth_at(p - vec2<i32>(1, 0))));
    let gy = min(abs(depth_at(p + vec2<i32>(0, 1)) - z), abs(z - depth_at(p - vec2<i32>(0, 1))));

    // Variance prefiltered with a 3x3 Gaussian.
    var v = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = clamp(p + vec2<i32>(dx, dy), vec2<i32>(0), size - 1);
            let k = select(0.5, 0.25, dx != 0) * select(0.5, 0.25, dy != 0);
            v += src[u32(q.y) * P.frame.y + u32(q.x)].w * k;
        }
    }
    let sigma_l = SIGMA_L * sqrt(v) + 1e-6;

    var kernel = array<f32, 3>(0.375, 0.25, 0.0625);
    var csum = vec3<f32>(0.0);
    var vsum = 0.0;
    var wsum = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let q = p + vec2<i32>(dx, dy) * stride;
            if q.x < 0 || q.y < 0 || q.x >= size.x || q.y >= size.y {
                continue;
            }
            let j = u32(q.y) * P.frame.y + u32(q.x);
            let s = src[j];
            if !finite4(s) {
                continue;
            }
            var w = kernel[abs(dx)] * kernel[abs(dy)];
            if dx != 0 || dy != 0 {
                let nq = aov_normal[j] * ic;
                let dn = nd.xyz - nq.xyz;
                let da = a_p - albedo_at(j);
                let dz = abs(z - nq.w) / (gx * f32(abs(dx * stride)) + gy * f32(abs(dy * stride)) + 0.002 * z + 1e-3);
                let dl = abs(l_p - luminance(s.rgb)) / sigma_l;
                w *= exp(-dot(dn, dn) * SIGMA_N - dot(da, da) * SIGMA_A - dz - dl);
            }
            csum += s.rgb * w;
            vsum += s.w * w * w;
            wsum += w;
        }
    }
    var out = vec4<f32>(0.0);
    if wsum > 0.0 {
        out = vec4<f32>(csum / wsum, vsum / (wsum * wsum));
    }
    if pass_info.y == 1u {
        out = vec4<f32>(out.rgb * a_p, out.w);
    }
    dst[i] = select(vec4<f32>(0.0), out, finite4(out));
}
