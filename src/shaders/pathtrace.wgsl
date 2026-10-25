// Progressive voxel path tracer. One sample per pixel per dispatch,
// accumulated into `accum`.
//
// Light transport: Lambertian, GGX metal/plastic and rough dielectric
// surfaces; homogeneous fog; direct lighting from the sun, the environment
// map and emissive voxel faces, the latter two combined with BSDF sampling
// by multiple importance sampling (power heuristic).

struct Material {
    color: vec4<f32>,
    kind: u32,
    rough: f32,
    metal: f32,
    ior: f32,
    emit: f32,
    trans: f32,
    density: f32,
    g: f32,
}

struct Light {
    corner_cdf: vec4<f32>,
    axis_sign_mat: vec4<u32>,
}

const KIND_DIFFUSE: u32 = 0u;
const KIND_METAL: u32 = 1u;
const KIND_GLASS: u32 = 2u;
const KIND_EMIT: u32 = 3u;
const KIND_BLEND: u32 = 4u;
const KIND_MEDIA: u32 = 5u;

const PI: f32 = 3.14159265;
const EPS: f32 = 1e-3;
const BIG: f32 = 1e30;

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var vox: texture_3d<u32>;
@group(0) @binding(2) var bricks: texture_3d<u32>;
@group(0) @binding(3) var<storage, read> mats: array<Material, 256>;
@group(0) @binding(4) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> env: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> env_cdf: array<f32>;
@group(0) @binding(7) var<storage, read> lights: array<Light>;
// Denoiser AOVs (summed like `accum`): first non-glass hit albedo and the
// squared luminance of the demodulated sample; normal and path depth.
@group(0) @binding(8) var<storage, read_write> aov_albedo: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> aov_normal: array<vec4<f32>>;
// Edge / grid lines: rgb * coverage, coverage.
@group(0) @binding(10) var<storage, read_write> overlay: array<vec4<f32>>;

// ---------------------------------------------------------------- random

var<private> rng: u32;

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn rand() -> f32 {
    rng = pcg(rng);
    return f32(rng >> 8u) / 16777216.0;
}

fn rand2() -> vec2<f32> {
    return vec2<f32>(rand(), rand());
}

// ---------------------------------------------------------------- sampling

fn onb(n: vec3<f32>) -> mat3x3<f32> {
    // Duff et al. 2017, branchless orthonormal basis.
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return mat3x3<f32>(t, bt, n);
}

fn cosine_dir(n: vec3<f32>) -> vec3<f32> {
    let u = rand2();
    let r = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    let l = vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
    return onb(n) * l;
}

fn cone_dir(axis: vec3<f32>, cos_max: f32) -> vec3<f32> {
    let u = rand2();
    let c = 1.0 - u.x * (1.0 - cos_max);
    let s = sqrt(max(0.0, 1.0 - c * c));
    let phi = 2.0 * PI * u.y;
    return onb(axis) * vec3<f32>(s * cos(phi), s * sin(phi), c);
}

// Sample a GGX microfacet normal around `n` (distribution of normals).
fn ggx_normal(n: vec3<f32>, rough: f32) -> vec3<f32> {
    let a = max(rough * rough, 1e-4);
    let u = rand2();
    let cos2 = (1.0 - u.x) / (1.0 + (a * a - 1.0) * u.x);
    let c = sqrt(cos2);
    let s = sqrt(max(0.0, 1.0 - cos2));
    let phi = 2.0 * PI * u.y;
    return onb(n) * vec3<f32>(s * cos(phi), s * sin(phi), c);
}

fn smith_g1(ndv: f32, a: f32) -> f32 {
    let a2 = a * a;
    return 2.0 * ndv / (ndv + sqrt(a2 + (1.0 - a2) * ndv * ndv));
}

fn schlick(f0: vec3<f32>, c: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - clamp(c, 0.0, 1.0), 5.0);
}

fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {
    // eta = n_incident / n_transmitted
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if sin2_t >= 1.0 {
        return 1.0;
    }
    let cos_t = sqrt(1.0 - sin2_t);
    let rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5 * (rs * rs + rp * rp);
}

fn hg_phase(cos_t: f32, g: f32) -> f32 {
    let d = 1.0 + g * g - 2.0 * g * cos_t;
    return (1.0 - g * g) / (4.0 * PI * d * sqrt(d));
}

// Sample a direction from the Henyey-Greenstein phase function around `wo`.
fn hg_sample(wo: vec3<f32>, g: f32) -> vec3<f32> {
    let u = rand2();
    var c: f32;
    if abs(g) < 1e-3 {
        c = 1.0 - 2.0 * u.x;
    } else {
        let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u.x);
        c = (1.0 + g * g - s * s) / (2.0 * g);
    }
    let st = sqrt(max(0.0, 1.0 - c * c));
    let phi = 2.0 * PI * u.y;
    return onb(wo) * vec3<f32>(st * cos(phi), st * sin(phi), c);
}

fn power_heuristic(a: f32, b: f32) -> f32 {
    // Clamped so the squares stay finite: light pdfs grow with distance
    // squared, and a grazing ray can hit the ground billions of voxels away.
    let a2 = min(a * a, 1e36);
    let b2 = min(b * b, 1e36);
    return a2 / max(a2 + b2, 1e-20);
}

// ---------------------------------------------------------------- environment

fn env_texel(d: vec3<f32>) -> vec4<f32> {
    let w = P.env.x;
    let h = P.env.y;
    let theta = acos(clamp(d.z, -1.0, 1.0));
    var phi = atan2(d.y, d.x);
    if phi < 0.0 {
        phi += 2.0 * PI;
    }
    let x = min(u32(phi / (2.0 * PI) * f32(w)), w - 1u);
    let y = min(u32(theta / PI * f32(h)), h - 1u);
    return env[y * w + x];
}

// Bilinear lookup, used where the map is seen directly (camera rays).
fn env_smooth(d: vec3<f32>) -> vec3<f32> {
    let w = P.env.x;
    let h = P.env.y;
    let theta = acos(clamp(d.z, -1.0, 1.0));
    var phi = atan2(d.y, d.x);
    if phi < 0.0 {
        phi += 2.0 * PI;
    }
    let p = vec2<f32>(phi / (2.0 * PI) * f32(w), theta / PI * f32(h)) - 0.5;
    let f = fract(p);
    let x0 = u32(i32(floor(p.x)) + i32(w)) % w;
    let x1 = (x0 + 1u) % w;
    let y0 = u32(clamp(i32(floor(p.y)), 0, i32(h) - 1));
    let y1 = min(y0 + 1u, h - 1u);
    let a = mix(env[y0 * w + x0].rgb, env[y0 * w + x1].rgb, f.x);
    let b = mix(env[y1 * w + x0].rgb, env[y1 * w + x1].rgb, f.x);
    return mix(a, b, f.y);
}

fn env_pdf(d: vec3<f32>, texel: vec4<f32>) -> f32 {
    let s = max(sqrt(max(0.0, 1.0 - d.z * d.z)), 1e-4);
    return texel.w / (2.0 * PI * PI * s);
}

// First index in cdf[start, start + n) whose value is >= u.
fn search(start: u32, n: u32, u: f32) -> u32 {
    var lo = 0u;
    var hi = n - 1u;
    while lo < hi {
        let mid = (lo + hi) / 2u;
        if env_cdf[start + mid] < u {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

struct EnvSample {
    dir: vec3<f32>,
    radiance: vec3<f32>,
    pdf: f32,
}

fn sample_env() -> EnvSample {
    let w = P.env.x;
    let h = P.env.y;
    let y = search(0u, h, rand());
    let x = search(h + y * w, w, rand());
    let theta = PI * (f32(y) + rand()) / f32(h);
    let phi = 2.0 * PI * (f32(x) + rand()) / f32(w);
    var s: EnvSample;
    s.dir = vec3<f32>(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
    let t = env[y * w + x];
    s.radiance = t.rgb;
    s.pdf = env_pdf(s.dir, t);
    return s;
}

// ---------------------------------------------------------------- tracing

struct Hit {
    hit: bool,
    t: f32,
    n: vec3<f32>,
    v: u32,
}

fn axis_normal(axis: i32, step: vec3<i32>) -> vec3<f32> {
    var n = vec3<f32>(0.0);
    n[axis] = -f32(step[axis]);
    return n;
}

fn min_axis(v: vec3<f32>) -> i32 {
    if v.x <= v.y && v.x <= v.z {
        return 0;
    }
    if v.y <= v.z {
        return 1;
    }
    return 2;
}

// March from `ro` along `rd` until the cell value differs from `inside`
// (0 = travelling through air). Empty 8^3 bricks are skipped in air.
// The returned normal always faces against `rd`.
fn trace(ro: vec3<f32>, rd_in: vec3<f32>, inside: u32, tmax: f32) -> Hit {
    var h: Hit;
    h.hit = false;
    h.t = tmax;

    // Avoid exact zero components so the slab math stays finite.
    let rd = select(rd_in, vec3<f32>(1e-7), abs(rd_in) < vec3<f32>(1e-7));
    let inv = 1.0 / rd;
    let dimsi = vec3<i32>(P.dims.xyz);
    let dims = vec3<f32>(dimsi);
    let step = select(vec3<i32>(-1), vec3<i32>(1), rd > vec3<f32>(0.0));

    let ta = (vec3<f32>(0.0) - ro) * inv;
    let tb = (dims - ro) * inv;
    let tnear = min(ta, tb);
    let tfar = max(ta, tb);
    var t = max(max(tnear.x, tnear.y), max(tnear.z, 0.0));
    let tend = min(min(tfar.x, tfar.y), min(tfar.z, tmax));
    if t >= tend {
        return h;
    }

    var axis: i32;
    if t > 0.0 {
        if tnear.x >= tnear.y && tnear.x >= tnear.z {
            axis = 0;
        } else if tnear.y >= tnear.z {
            axis = 1;
        } else {
            axis = 2;
        }
    } else {
        let a = abs(rd);
        axis = select(select(2, 1, a.y >= a.z), 0, a.x >= a.y && a.x >= a.z);
    }
    var n = axis_normal(axis, step);
    var cell = clamp(vec3<i32>(floor(ro + rd * t)), vec3<i32>(0), dimsi - 1);

    for (var i = 0; i < 4096; i++) {
        if inside == 0u && textureLoad(bricks, cell >> vec3<u32>(3u), 0).r == 0u {
            // Jump to the exit face of this empty brick.
            let b0 = vec3<f32>((cell >> vec3<u32>(3u)) << vec3<u32>(3u));
            let bexit = (select(b0, b0 + 8.0, rd > vec3<f32>(0.0)) - ro) * inv;
            axis = min_axis(bexit);
            t = bexit[axis];
            if t >= tend {
                return h;
            }
            // Stay inside the brick on the other axes to absorb rounding error.
            let bi = vec3<i32>(b0);
            cell = clamp(vec3<i32>(floor(ro + rd * t)), bi, bi + 7);
            cell[axis] = select(bi[axis] - 1, bi[axis] + 8, step[axis] > 0);
            if any(cell < vec3<i32>(0)) || any(cell >= dimsi) {
                return h;
            }
            n = axis_normal(axis, step);
            continue;
        }

        let v = textureLoad(vox, cell, 0).r;
        if v != inside {
            h.hit = true;
            h.t = t;
            h.n = n;
            h.v = v;
            return h;
        }

        let tn = (vec3<f32>(cell + max(step, vec3<i32>(0))) - ro) * inv;
        axis = min_axis(tn);
        t = tn[axis];
        cell[axis] += step[axis];
        n = axis_normal(axis, step);
        if any(cell < vec3<i32>(0)) || any(cell >= dimsi) || t >= tmax {
            if inside != 0u && t < tmax {
                // Leaving the volume from inside a solid: exit into air.
                h.hit = true;
                h.t = t;
                h.n = n;
                h.v = 0u;
            }
            return h;
        }
    }
    return h;
}

fn ground_t(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    if P.ground_col.w < 0.5 || rd.z >= 0.0 || ro.z <= 0.0 {
        return BIG;
    }
    return -ro.z / rd.z;
}

fn is_translucent(m: Material) -> bool {
    return m.kind == KIND_GLASS || (m.kind == KIND_BLEND && m.trans > 0.0);
}

// Transmittance from `p` towards `l` over `tmax`. Glass tints the light
// instead of blocking it when translucent shadows are on.
fn shadow(p: vec3<f32>, l: vec3<f32>, tmax: f32) -> vec3<f32> {
    if ground_t(p, l) < tmax {
        return vec3<f32>(0.0);
    }
    var T = vec3<f32>(1.0);
    var o = p;
    var inside = 0u;
    var remaining = tmax;
    let tr = (P.dims.w & 1u) != 0u;
    for (var i = 0; i < 12; i++) {
        let h = trace(o, l, inside, remaining);
        if !h.hit {
            return T;
        }
        if inside == 0u {
            let m = mats[h.v];
            if m.kind != KIND_MEDIA {
                if !tr || !is_translucent(m) {
                    return vec3<f32>(0.0);
                }
                T *= m.color.rgb * select(m.trans, 1.0, m.kind == KIND_GLASS);
            }
            inside = h.v;
            o += l * h.t - h.n * EPS;
        } else {
            let m = mats[inside];
            if m.kind == KIND_MEDIA {
                T *= exp(-m.density * h.t);
            }
            // Leaving the translucent solid (or touching another solid,
            // which the next iteration tests from air).
            inside = 0u;
            o += l * h.t + select(-h.n, h.n, h.v != 0u) * EPS;
        }
        remaining -= h.t;
        if remaining <= 0.0 || max(T.x, max(T.y, T.z)) < 1e-3 {
            return T;
        }
    }
    return vec3<f32>(0.0);
}

// ---------------------------------------------------------------- lights

fn light_face_power(m: Material) -> f32 {
    return m.emit * luminance(m.color.rgb);
}

fn search_light(u: f32) -> u32 {
    var lo = 0u;
    var hi = P.env.z - 1u;
    while lo < hi {
        let mid = (lo + hi) / 2u;
        if lights[mid].corner_cdf.w < u {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Solid-angle pdf of having sampled the emissive face hit at distance `t`.
fn light_pdf(m: Material, t: f32, cos_l: f32) -> f32 {
    return light_face_power(m) / P.misc.y * t * t / max(cos_l, 1e-4);
}

// Direct light from emissive voxels for a surface with normal `n`
// (`n` = 0 for a fog scattering event, which uses the phase function).
fn sample_lights(p: vec3<f32>, n: vec3<f32>, wo: vec3<f32>, g: f32) -> vec3<f32> {
    if P.env.z == 0u {
        return vec3<f32>(0.0);
    }
    let li = lights[search_light(rand())];
    let axis = i32(li.axis_sign_mat.x);
    var q = li.corner_cdf.xyz;
    q[(axis + 1) % 3] += rand();
    q[(axis + 2) % 3] += rand();
    var ln = vec3<f32>(0.0);
    ln[axis] = select(-1.0, 1.0, li.axis_sign_mat.y == 1u);

    let d = q - p;
    let dist = length(d);
    let l = d / dist;
    let cos_l = dot(-l, ln);
    if cos_l <= 0.0 {
        return vec3<f32>(0.0);
    }
    let surface = dot(n, n) > 0.0;
    var f: f32;
    var pdf_bsdf: f32;
    if surface {
        let cos_s = dot(n, l);
        if cos_s <= 0.0 {
            return vec3<f32>(0.0);
        }
        f = cos_s / PI;
        pdf_bsdf = f;
    } else {
        f = hg_phase(dot(wo, l), g);
        pdf_bsdf = f;
    }
    let m = mats[li.axis_sign_mat.z];
    // The light face must be the first thing along the ray (clouds and,
    // with translucent shadows, glass in between only attenuate).
    let vis = shadow(p, l, dist - 0.01);
    if max(vis.x, max(vis.y, vis.z)) <= 0.0 {
        return vec3<f32>(0.0);
    }
    let pdf_l = light_pdf(m, dist, cos_l);
    return vis * m.color.rgb * m.emit * f * power_heuristic(pdf_l, pdf_bsdf) / pdf_l;
}

// Direct sun light. Returns irradiance-weighted BSDF value.
fn sample_sun(p: vec3<f32>, n: vec3<f32>, wo: vec3<f32>, g: f32) -> vec3<f32> {
    let l = cone_dir(P.sun_dir.xyz, P.sun_dir.w);
    var f: f32;
    if dot(n, n) > 0.0 {
        f = max(dot(n, l), 0.0) / PI;
    } else {
        f = hg_phase(dot(wo, l), g);
    }
    if f <= 0.0 {
        return vec3<f32>(0.0);
    }
    return P.sun_col.rgb * f * shadow(p, l, BIG);
}

fn sample_env_light(p: vec3<f32>, n: vec3<f32>, wo: vec3<f32>, g: f32) -> vec3<f32> {
    let s = sample_env();
    var f: f32;
    if dot(n, n) > 0.0 {
        f = max(dot(n, s.dir), 0.0) / PI;
    } else {
        f = hg_phase(dot(wo, s.dir), g);
    }
    if f <= 0.0 || s.pdf <= 0.0 {
        return vec3<f32>(0.0);
    }
    let vis = shadow(p, s.dir, BIG);
    return s.radiance * f * vis * power_heuristic(s.pdf, f) / s.pdf;
}

// All direct light at a surface (normal `n`) or volume (`n` = 0, phase `g`).
fn direct(p: vec3<f32>, n: vec3<f32>, wo: vec3<f32>, g: f32) -> vec3<f32> {
    return sample_sun(p, n, wo, g) + sample_env_light(p, n, wo, g) + sample_lights(p, n, wo, g);
}

// ---------------------------------------------------------------- integrator

// Per-sample outputs besides radiance, set by `radiance`.
var<private> aov_done: bool;
var<private> aov_a: vec3<f32>;
var<private> aov_n: vec3<f32>;
var<private> aov_t: f32;
// Primary hit: 0 nothing, 1 voxel, 2 ground; distance and face normal.
var<private> prim_hit: u32;
var<private> prim_t: f32;
var<private> prim_n: vec3<f32>;
// 0 where a camera ray sees the (hidden) background.
var<private> alpha: f32;

fn record_aov(a: vec3<f32>, n: vec3<f32>, t: f32) {
    if !aov_done {
        aov_done = true;
        aov_a = a;
        aov_n = n;
        aov_t = t;
    }
}

fn radiance(ro_in: vec3<f32>, rd_in: vec3<f32>) -> vec3<f32> {
    var ro = ro_in;
    var rd = rd_in;
    var thr = vec3<f32>(1.0);
    var L = vec3<f32>(0.0);
    var inside = 0u;
    var n_diffuse = 0u;
    var n_specular = 0u;
    var n_scatter = 0u;
    // Pdf of the BSDF sample that produced `rd`; 0 after delta-like events.
    var last_pdf = 0.0;
    // Distance travelled so far, for the depth AOV.
    var path_t = 0.0;
    let max_diffuse = P.frame.w & 0xffu;
    let max_specular = (P.frame.w >> 8u) & 0xffu;
    let max_scatter = (P.frame.w >> 16u) & 0xffu;

    for (var bounce = 0u; bounce < 48u; bounce++) {
        let h = trace(ro, rd, inside, BIG);
        let tg = select(BIG, ground_t(ro, rd), inside == 0u);
        let t_surf = min(select(BIG, h.t, h.hit), tg);
        if bounce == 0u {
            prim_t = t_surf;
            prim_hit = select(select(1u, 2u, tg < h.t), 0u, t_surf >= BIG);
            prim_n = select(h.n, vec3<f32>(0.0, 0.0, 1.0), tg < h.t);
        }

        // Inside a cloud voxel: free-flight sample against the exit distance.
        if inside != 0u && mats[inside].kind == KIND_MEDIA {
            let cm = mats[inside];
            let tf = -log(1.0 - rand()) / max(cm.density, 1e-4);
            if tf < h.t {
                n_scatter++;
                if n_scatter > max_scatter {
                    break;
                }
                let p = ro + rd * tf;
                record_aov(thr * cm.color.rgb, vec3<f32>(0.0), path_t + tf);
                path_t += tf;
                thr *= cm.color.rgb;
                L += thr * direct(p, vec3<f32>(0.0), rd, cm.g);
                let nd = hg_sample(rd, cm.g);
                last_pdf = hg_phase(dot(rd, nd), cm.g);
                ro = p;
                rd = nd;
                continue;
            }
            if h.v == 0u {
                // Leave the cloud without refraction.
                ro += rd * h.t - h.n * EPS;
                path_t += h.t;
                inside = 0u;
                continue;
            }
            // Otherwise the cloud touches a solid: shade it below as from air.
        }

        // Homogeneous fog in air, limited to a range around the scene.
        if inside == 0u && P.fog.w > 0.0 {
            let tf = -log(1.0 - rand()) / P.fog.w;
            if tf < min(t_surf, P.misc.w) {
                n_scatter++;
                if n_scatter > max_scatter {
                    break;
                }
                let p = ro + rd * tf;
                record_aov(thr * P.fog.rgb, vec3<f32>(0.0), path_t + tf);
                path_t += tf;
                thr *= P.fog.rgb;
                let none = vec3<f32>(0.0);
                L += thr * direct(p, none, rd, P.misc.z);
                let nd = hg_sample(rd, P.misc.z);
                last_pdf = hg_phase(dot(rd, nd), P.misc.z);
                ro = p;
                rd = nd;
                continue;
            }
        }

        if t_surf >= BIG {
            if bounce == 0u && (P.aux.x & FLAG_BG_HIDDEN) != 0u {
                // The background is composited later; light still comes
                // from the sky for every other ray.
                alpha = 0.0;
                record_aov(vec3<f32>(1.0), vec3<f32>(0.0), 0.0);
                break;
            }
            record_aov(thr, vec3<f32>(0.0), 0.0);
            let tex = env_texel(rd);
            var w = 1.0;
            if last_pdf > 0.0 {
                w = power_heuristic(last_pdf, env_pdf(rd, tex));
            }
            let c = select(tex.rgb, env_smooth(rd), bounce == 0u);
            L += thr * c * w;
            if bounce == 0u && P.sun_col.w > 0.5 && dot(rd, P.sun_dir.xyz) > P.sun_dir.w {
                L += thr * P.sun_col.rgb / max(2.0 * PI * (1.0 - P.sun_dir.w), 1e-4);
            }
            break;
        }

        var n: vec3<f32>;
        var p: vec3<f32>;
        var m: Material;
        var hit_v = h.v;
        if tg < h.t {
            p = ro + rd * tg;
            n = vec3<f32>(0.0, 0.0, 1.0);
            m.color = vec4<f32>(P.ground_col.rgb, 1.0);
            m.kind = KIND_DIFFUSE;
            hit_v = 0u;
        } else {
            p = ro + rd * h.t;
            n = h.n;
            // Exiting a solid: shade with the material we were inside.
            m = mats[select(h.v, inside, inside != 0u && h.v == 0u)];
            if inside != 0u && h.v != 0u {
                // A solid touching another solid: treat as entering it from air.
                inside = 0u;
            }
        }
        let albedo = m.color.rgb;
        path_t += t_surf;

        if m.kind == KIND_EMIT && inside == 0u {
            var w = 1.0;
            if last_pdf > 0.0 {
                w = power_heuristic(last_pdf, light_pdf(m, h.t, dot(-rd, n)));
            }
            L += thr * albedo * m.emit * w;
        }

        // Pick the lobe for blended materials.
        var kind = m.kind;
        if kind == KIND_BLEND {
            let r = rand();
            kind = select(select(KIND_DIFFUSE, KIND_GLASS, r < m.metal + m.trans), KIND_METAL, r < m.metal);
        }
        if kind == KIND_GLASS && inside == 0u && m.trans > 0.0 && m.trans < 1.0 && rand() >= m.trans {
            kind = KIND_DIFFUSE;
        }
        if kind == KIND_EMIT {
            kind = KIND_DIFFUSE;
        }
        if kind != KIND_GLASS && kind != KIND_MEDIA {
            // Glass is looked through: the AOVs come from what lies behind.
            record_aov(thr * albedo, n, path_t);
        }
        if kind == KIND_MEDIA {
            // Enter the cloud; scattering happens on the next iteration.
            ro = p - n * EPS;
            inside = hit_v;
            continue;
        }

        if kind == KIND_GLASS {
            n_specular++;
            if n_specular > max_specular {
                break;
            }
            last_pdf = 0.0;
            let entering = inside == 0u;
            let eta = select(m.ior, 1.0 / m.ior, entering);
            let hn = ggx_normal(n, m.rough * 0.5);
            let cos_i = abs(dot(-rd, hn));
            let f = fresnel_dielectric(cos_i, eta);
            if rand() < f {
                rd = reflect(rd, hn);
                if dot(rd, n) <= 0.0 {
                    rd = reflect(rd, n);
                }
                ro = p + n * EPS;
            } else {
                var t_dir = refract(rd, hn, eta);
                if dot(t_dir, t_dir) == 0.0 || dot(t_dir, n) >= 0.0 {
                    t_dir = refract(rd, n, eta);
                }
                if dot(t_dir, t_dir) == 0.0 {
                    rd = reflect(rd, n);
                    ro = p + n * EPS;
                    continue;
                }
                rd = normalize(t_dir);
                ro = p - n * EPS;
                if entering {
                    thr *= albedo;
                    inside = hit_v;
                } else {
                    inside = 0u;
                }
            }
            continue;
        }

        // Opaque surfaces from here on: diffuse, metal, plastic.
        var spec = false;
        var f0 = vec3<f32>(0.04);
        if kind == KIND_METAL {
            if rand() < m.metal {
                spec = true;
                f0 = albedo;
            } else {
                let r0 = (m.ior - 1.0) / (m.ior + 1.0);
                let fr = schlick(vec3<f32>(r0 * r0), dot(-rd, n)).x;
                if rand() < fr {
                    spec = true;
                    f0 = vec3<f32>(1.0);
                }
            }
        }

        if spec {
            n_specular++;
            if n_specular > max_specular {
                break;
            }
            last_pdf = 0.0;
            let a = max(m.rough * m.rough, 1e-4);
            let hn = ggx_normal(n, m.rough);
            let v = -rd;
            let l = reflect(rd, hn);
            let ndl = dot(n, l);
            let ndv = max(dot(n, v), 1e-4);
            if ndl <= 0.0 {
                break;
            }
            let vdh = max(dot(v, hn), 0.0);
            let ndh = max(dot(n, hn), 1e-4);
            let g = smith_g1(ndv, a) * smith_g1(ndl, a);
            thr *= schlick(f0, vdh) * g * vdh / (ndh * ndv);
            rd = l;
            ro = p + n * EPS;
        } else {
            n_diffuse++;
            if n_diffuse > max_diffuse {
                break;
            }
            ro = p + n * EPS;
            thr *= albedo;
            L += thr * direct(ro, n, rd, 0.0);
            rd = cosine_dir(n);
            last_pdf = max(dot(n, rd), 1e-4) / PI;
        }

        // Russian roulette once paths get long.
        if bounce > 3u {
            let q = clamp(max(thr.x, max(thr.y, thr.z)), 0.05, 0.95);
            if rand() > q {
                break;
            }
            thr /= q;
        }
    }
    record_aov(vec3<f32>(1.0), vec3<f32>(0.0), 0.0);
    let lum = max(L.x, max(L.y, L.z));
    if lum > P.misc.x {
        L *= P.misc.x / lum;
    }
    return L;
}

const PROJ_PERSPECTIVE: f32 = 0.0;
const PROJ_ORTHO: f32 = 1.0;
const PROJ_STEREO: f32 = 2.0;
const PROJ_PANO: f32 = 3.0;
const PROJ_CYLINDER: f32 = 4.0;

// Camera ray direction for a point in normalized device coordinates.
fn camera_dir(ndc: vec2<f32>) -> vec3<f32> {
    let f = P.cam_fwd.xyz;
    let r = P.cam_right.xyz;
    let u = P.cam_up.xyz;
    let tan_half = P.cam_pos.w;
    let aspect = P.cam_up.w;
    let proj = P.lens.x;
    if proj == PROJ_STEREO {
        // r = 2 tan(theta / 2), scaled so the vertical edge is fov / 2.
        let half_fov = atan(tan_half);
        let q = vec2<f32>(ndc.x * aspect, ndc.y) * 2.0 * tan(half_fov * 0.5);
        let rl = length(q);
        let theta = 2.0 * atan(rl * 0.5);
        let d2 = select(vec2<f32>(0.0), q / rl, rl > 0.0);
        return normalize(f * cos(theta) + (r * d2.x + u * d2.y) * sin(theta));
    }
    if proj == PROJ_PANO {
        // Full equirectangular panorama around the camera.
        let phi = ndc.x * PI;
        let theta = ndc.y * PI * 0.5;
        let h = f * cos(phi) + r * sin(phi);
        let hf = normalize(vec3<f32>(h.x, h.y, 0.0) + vec3<f32>(1e-6, 0.0, 0.0));
        return normalize(hf * cos(theta) + vec3<f32>(0.0, 0.0, 1.0) * sin(theta));
    }
    if proj == PROJ_CYLINDER {
        let phi = ndc.x * atan(tan_half * aspect);
        return normalize(f * cos(phi) + r * sin(phi) + u * ndc.y * tan_half);
    }
    if proj == PROJ_ORTHO {
        return f;
    }
    return normalize(f + r * ndc.x * tan_half * aspect + u * ndc.y * tan_half);
}

// Point on the unit aperture: a disk, or a regular polygon with
// `P.lens.y` blades rotated by `P.lens.z` radians.
fn aperture_sample() -> vec2<f32> {
    let n = P.lens.y;
    if n < 3.0 {
        let u = rand2();
        let r = sqrt(u.x);
        let phi = 2.0 * PI * u.y;
        return vec2<f32>(cos(phi), sin(phi)) * r;
    }
    // Pick a triangle fan sector, then a uniform point inside it.
    let k = floor(rand() * n);
    let a0 = P.lens.z + 2.0 * PI * k / n;
    let a1 = a0 + 2.0 * PI / n;
    var u = rand2();
    if u.x + u.y > 1.0 {
        u = vec2<f32>(1.0) - u;
    }
    return vec2<f32>(cos(a0), sin(a0)) * u.x + vec2<f32>(cos(a1), sin(a1)) * u.y;
}

// ---------------------------------------------------------------- overlays

fn vox_at(c: vec3<i32>) -> u32 {
    if any(c < vec3<i32>(0)) || any(c >= vec3<i32>(P.dims.xyz)) {
        return 0u;
    }
    return textureLoad(vox, c, 0).r;
}

// World-space size of a pixel at distance `t`.
fn footprint(t: f32) -> f32 {
    let h = f32(P.frame.z);
    if P.lens.x == PROJ_ORTHO {
        return 2.0 * P.cam_right.w * P.cam_pos.w / h;
    }
    return t * 2.0 * P.cam_pos.w / h;
}

// Coverage of a line `half_width` either side of distance 0, widened to
// at least a pixel (`fp`) and dimmed to match so it neither vanishes nor
// aliases when small.
fn line_cov(d: f32, half_width: f32, fp: f32) -> f32 {
    let hw = max(half_width, fp * 0.5);
    return select(0.0, min(1.0, half_width / hw), d < hw);
}

fn normal_axis(n: vec3<f32>) -> i32 {
    return select(select(2, 1, abs(n.y) > 0.5), 0, abs(n.x) > 0.5);
}

// Lines on the edges of the face at `p` (normal `n`) where the surface
// changes: an empty or differently colored in-plane neighbor, or a solid
// cell beyond it (a concave crease).
fn edge_cov(p: vec3<f32>, n: vec3<f32>, fp: f32) -> f32 {
    let a = normal_axis(n);
    let s = vec3<i32>(round(n));
    let base = floor(p - n * 0.5);
    let cell = vec3<i32>(base);
    let v = vox_at(cell);
    if v == 0u {
        return 0.0;
    }
    let f = p - base;
    let w = P.edge_col.w;
    var cov = 0.0;
    for (var k = 1; k <= 2; k++) {
        let u = (a + k) % 3;
        for (var side = 0; side < 2; side++) {
            let c = line_cov(select(f[u], 1.0 - f[u], side == 1), w, fp);
            if c <= cov {
                continue;
            }
            var o = vec3<i32>(0);
            o[u] = select(-1, 1, side == 1);
            if vox_at(cell + o + s) != 0u || vox_at(cell + o) != v {
                cov = c;
            }
        }
    }
    return cov;
}

fn grid_line(x: f32, sp: f32, fp: f32) -> f32 {
    return line_cov(abs(x - round(x / sp) * sp), P.grid_col.w * 0.5, fp);
}

fn grid_cov(p: vec3<f32>, n: vec3<f32>, fp: f32) -> f32 {
    let sp = f32(max(P.aux.w, 1u));
    // Fade out where lines would be closer than a couple of pixels.
    let fade = 1.0 - smoothstep(0.25, 0.5, fp / sp);
    if fade <= 0.0 {
        return 0.0;
    }
    let a = normal_axis(n);
    return max(grid_line(p[(a + 1) % 3], sp, fp), grid_line(p[(a + 2) % 3], sp, fp)) * fade;
}

// Edge and grid lines for the primary ray: rgb * coverage, coverage.
fn overlay_sample(ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    let flags = P.aux.x;
    var e = 0.0;
    var g = 0.0;
    if prim_hit == 1u {
        let p = ro + rd * prim_t;
        let fp = footprint(prim_t) / max(abs(dot(rd, prim_n)), 0.1);
        if (flags & FLAG_EDGES) != 0u {
            e = edge_cov(p, prim_n, fp);
        }
        if (flags & FLAG_GRID) != 0u && (flags & FLAG_GRID_GROUND_ONLY) == 0u {
            g = grid_cov(p, prim_n, fp);
        }
    }
    if (flags & FLAG_GRID) != 0u && abs(rd.z) > 1e-6 {
        // The ground plane, drawn even when the ground itself is off.
        let tp = -ro.z / rd.z;
        if tp > 0.0 && (prim_hit != 1u || tp < prim_t) {
            let fp = footprint(tp) / max(abs(rd.z), 0.1);
            g = max(g, grid_cov(ro + rd * tp, vec3<f32>(0.0, 0.0, 1.0), fp));
        }
    }
    let cov = e + (1.0 - e) * g;
    return vec4<f32>(P.edge_col.rgb * e + P.grid_col.rgb * g * (1.0 - e), cov);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = P.frame.y;
    let hgt = P.frame.z;
    if gid.x >= w || gid.y >= hgt {
        return;
    }
    let idx = gid.y * w + gid.x;
    rng = pcg(idx ^ pcg(P.frame.x * 9781u + 1u));

    // Jittered pixel position for anti-aliasing, y up.
    let px = (vec2<f32>(gid.xy) + rand2()) / vec2<f32>(f32(w), f32(hgt));
    let ndc = vec2<f32>(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0);
    var ro = P.cam_pos.xyz;
    var rd = camera_dir(ndc);
    if P.lens.x == PROJ_ORTHO {
        let half = P.cam_right.w * P.cam_pos.w;
        ro += (P.cam_right.xyz * ndc.x * P.cam_up.w + P.cam_up.xyz * ndc.y) * half;
    }

    // Thin lens depth of field.
    if P.cam_fwd.w > 0.0 {
        let focus = ro + rd * (P.cam_right.w / dot(rd, P.cam_fwd.xyz));
        let a = aperture_sample() * P.cam_fwd.w;
        ro += P.cam_right.xyz * a.x + P.cam_up.xyz * a.y;
        rd = normalize(focus - ro);
    }

    aov_done = false;
    prim_hit = 0u;
    prim_t = BIG;
    alpha = 1.0;
    var L = radiance(ro, rd);
    // Drop a sample that went non-finite: one NaN would black out the
    // pixel, and the denoiser would spread it over a whole block.
    if !finite3(L) {
        L = vec3<f32>(0.0);
    }
    // w accumulates coverage; the sample count is `P.frame.x`.
    let c = vec4<f32>(L, alpha);
    let first = P.frame.x == 0u;
    if first {
        accum[idx] = c;
    } else {
        accum[idx] += c;
    }
    let flags = P.aux.x;
    if (flags & FLAG_DENOISE) != 0u {
        let l = luminance(L / max(aov_a, vec3<f32>(DEMOD_EPS)));
        var a = vec4<f32>(aov_a, l * l);
        var n = vec4<f32>(aov_n, aov_t);
        if !finite4(a) || !finite4(n) {
            a = vec4<f32>(1.0, 1.0, 1.0, 0.0);
            n = vec4<f32>(0.0);
        }
        if first {
            aov_albedo[idx] = a;
            aov_normal[idx] = n;
        } else {
            aov_albedo[idx] += a;
            aov_normal[idx] += n;
        }
    }
    if (flags & (FLAG_EDGES | FLAG_GRID)) != 0u {
        let o = overlay_sample(ro, rd);
        if first {
            overlay[idx] = o;
        } else {
            overlay[idx] += o;
        }
    }
}
