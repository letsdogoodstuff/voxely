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
