// Shared between the path tracer and the display/resolve passes.
// Must match `Params` in gpu.rs.
struct Params {
    cam_pos: vec4<f32>,    // xyz, tan(fov / 2)
    cam_fwd: vec4<f32>,    // xyz, lens radius
    cam_right: vec4<f32>,  // xyz, focus distance
    cam_up: vec4<f32>,     // xyz, aspect
    sun_dir: vec4<f32>,    // xyz, cos(sun half-angle)
    sun_col: vec4<f32>,    // irradiance rgb, sun disk flag
    fog: vec4<f32>,        // albedo rgb, density per voxel
    ground_col: vec4<f32>, // albedo rgb, ground flag
    film: vec4<f32>,       // exposure (linear), gamma, aces flag, vignette
    misc: vec4<f32>,       // energy clamp, total light power, fog phase g, fog range
    bloom: vec4<f32>,      // mix, radius, aspect, threshold
    dims: vec4<u32>,       // volume dims xyz, flags (bit 0: translucent shadows)
    frame: vec4<u32>,      // frame index, width, height, depths (diffuse | specular << 8 | scatter << 16)
    env: vec4<u32>,        // env map width, height, light face count, unused
    lens: vec4<f32>,       // projection, aperture blades, blade rotation (rad), unused
    view: vec4<f32>,       // viewport offset in the window (pixels), unused
    aux: vec4<u32>,        // FLAG_* bits, denoise view, background mode, grid spacing
    bg_col: vec4<f32>,     // solid background rgb (linear), unused
    edge_col: vec4<f32>,   // edge line rgb (linear), width (fraction of a voxel)
    grid_col: vec4<f32>,   // grid line rgb (linear), width (fraction of a voxel)
}

const FLAG_DENOISE: u32 = 1u;
const FLAG_EDGES: u32 = 2u;
const FLAG_GRID: u32 = 4u;
const FLAG_GRID_GROUND_ONLY: u32 = 8u;
const FLAG_BG_HIDDEN: u32 = 16u;

const VIEW_DENOISED: u32 = 0u;
const VIEW_ORIGINAL: u32 = 1u;
const VIEW_SPLIT: u32 = 2u;

const BG_SKY: u32 = 0u;
const BG_COLOR: u32 = 1u;
const BG_TRANSPARENT: u32 = 2u;

// Floor on albedo when dividing it out of radiance for denoising.
const DEMOD_EPS: f32 = 0.02;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// 1 / number of accumulated samples (valid in the post passes).
fn inv_samples(p: Params) -> f32 {
    return 1.0 / f32(max(p.frame.x, 1u));
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    // Narkowicz 2015 fit of the ACES filmic curve.
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// `c` is the averaged linear radiance.
fn tonemap(p: Params, c_in: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    var c = c_in * p.film.x;
    if p.film.w > 0.0 {
        let d = uv - vec2<f32>(0.5);
        c *= mix(1.0, smoothstep(0.8, 0.2, length(d)), p.film.w);
    }
    if p.film.z > 0.5 {
        c = aces(c);
    }
    return pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / p.film.y));
}
