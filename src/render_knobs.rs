//! Checks that every render-mode setting and material slider changes the
//! path-traced image (and in the expected direction where there is one).
//! Needs a GPU. Set `VOXELY_KNOB_DIR` to also save each before/after pair.

use glam::{IVec3, Vec3};

use crate::env::EnvMode;
use crate::gpu::{self, Renderer};
use crate::scene::{BgMode, DenoiseView, GpuMaterial, MaterialKind, Projection, RenderSettings, Scene};

const W: u32 = 128;
const H: u32 = 96;
const SPP: u32 = 24;

// Palette slots for the test blocks.
const DIFFUSE: usize = 1;
const METAL: usize = 2;
const GLASS: usize = 3;
const EMIT: usize = 4;
const BLEND: usize = 5;
const CLOUD: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Expect {
    Changed,
    Brighter,
    Darker,
    /// Some pixels become fully transparent.
    Transparent,
}

type Edit = Box<dyn Fn(&mut RenderSettings, &mut [GpuMaterial])>;

struct Case {
    name: &'static str,
    setup: Edit,
    change: Edit,
    expect: Expect,
}

fn case(name: &'static str, expect: Expect, setup: impl Fn(&mut RenderSettings, &mut [GpuMaterial]) + 'static, change: impl Fn(&mut RenderSettings, &mut [GpuMaterial]) + 'static) -> Case {
    Case { name, setup: Box::new(setup), change: Box::new(change), expect }
}

fn none(_: &mut RenderSettings, _: &mut [GpuMaterial]) {}

fn ibl() -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(env!("HOME")).join("Downloads/MagicaVoxel-0.99.6.2-macos-10.15/ibl/HDR_041_Path_Env.hdr");
    p.exists().then_some(p)
}

fn cases() -> Vec<Case> {
    use Expect::*;
    let atmo = |s: &mut RenderSettings, _: &mut [GpuMaterial]| {
        s.env_mode = EnvMode::Atmosphere;
        s.projection = Projection::Panorama;
    };
    let fog = |s: &mut RenderSettings, _: &mut [GpuMaterial]| s.fog_density = 1.0;
    let bloom = |s: &mut RenderSettings, _: &mut [GpuMaterial]| s.bloom = true;
    let dof = |s: &mut RenderSettings, _: &mut [GpuMaterial]| s.aperture = 1.5;
    let edges = |s: &mut RenderSettings, _: &mut [GpuMaterial]| s.show_edges = true;
    // A 1-voxel grid fades out at this small size (lines closer than a
    // few pixels are hidden to avoid moire), so use a coarser one.
    let grid = |s: &mut RenderSettings, _: &mut [GpuMaterial]| {
        s.show_grid = true;
        s.grid_spacing = 4;
    };
    // The default camera looks down at the ground; without it the
    // background shows above the model.
    let no_ground = |s: &mut RenderSettings, _: &mut [GpuMaterial]| s.ground = false;
    let mut v = vec![
        // Light: sun
        case("sun pitch", Changed, none, |s, _| s.sun_pitch = 20.0),
        case("sun yaw", Changed, none, |s, _| s.sun_yaw = 200.0),
        case("sun area", Changed, none, |s, _| s.sun_area = 0.6),
        case("sun intensity", Brighter, none, |s, _| s.sun_intensity = 2.0),
        case("sun color", Changed, none, |s, _| s.sun_color = Vec3::new(1.0, 0.3, 0.1)),
        case("sun disk", Brighter, |s, _| s.projection = Projection::Panorama, |s, _| s.sun_disk = true),
        // Light: uniform sky
        case("sky intensity", Brighter, none, |s, _| s.sky_intensity = 2.0),
        case("sky color", Changed, none, |s, _| s.sky_color = Vec3::new(0.2, 0.4, 1.0)),
        // Light: atmosphere (seen in a panorama so the sky itself shows)
        case("sky mode atmosphere", Changed, none, |s, _| s.env_mode = EnvMode::Atmosphere),
        case("rayleigh density", Changed, atmo, |s, _| s.rayleigh_density = 1.5),
        case("rayleigh color", Changed, atmo, |s, _| s.rayleigh_color = Vec3::new(1.0, 0.3, 0.3)),
        case("mie density", Changed, atmo, |s, _| s.mie_density = 1.5),
        case("mie phase", Changed, atmo, |s, _| s.mie_g = 0.55),
        case("mie color", Changed, atmo, |s, _| s.mie_color = Vec3::new(1.0, 0.5, 0.2)),
        case("ozone density", Changed, atmo, |s, _| s.ozone_density = 1.5),
        case("ozone color", Changed, move |s, m| {
            atmo(s, m);
            s.ozone_density = 1.0;
        }, |s, _| s.ozone_color = Vec3::new(1.0, 0.2, 0.8)),
        case("sun pitch (atmosphere sky)", Changed, atmo, |s, _| s.sun_pitch = 5.0),
        // Light: fog
        case("fog density", Changed, none, fog),
        case("fog color", Changed, fog, |s, _| s.fog_color = Vec3::new(1.0, 0.2, 0.2)),
        case("fog phase", Changed, fog, |s, _| s.fog_g = 0.8),
        // Sample
        case("max diffuse", Darker, none, |s, _| s.max_diffuse = 0),
        case("max specular", Changed, none, |s, _| s.max_specular = 5),
        case("max scatter", Changed, fog, |s, _| s.max_scatter = 5),
        case("energy clamp", Darker, none, |s, _| s.clamp = 1.0),
        case("translucent shadow", Changed, none, |s, _| s.translucent_shadows = false),
        // Display
        case("ground off", Changed, none, |s, _| s.ground = false),
        case("ground color", Changed, none, |s, _| s.ground_color = Vec3::new(0.8, 0.1, 0.1)),
        case("bg color mode", Changed, no_ground, |s, _| s.bg_mode = BgMode::Color),
        case("bg color", Changed, move |s, m| {
            no_ground(s, m);
            s.bg_mode = BgMode::Color;
        }, |s, _| s.bg_color = Vec3::new(1.0, 0.0, 1.0)),
        case("bg transparent", Transparent, no_ground, |s, _| s.bg_mode = BgMode::Transparent),
        case("edges", Changed, none, edges),
        case("edge color", Changed, edges, |s, _| s.edge_color = Vec3::new(1.0, 0.0, 0.0)),
        case("edge width", Changed, edges, |s, _| s.edge_width = 0.5),
        case("grid", Changed, none, grid),
        case("grid color", Changed, grid, |s, _| s.grid_color = Vec3::new(0.0, 1.0, 0.0)),
        case("grid spacing", Changed, grid, |s, _| s.grid_spacing = 8),
        case("grid width", Changed, grid, |s, _| s.grid_width = 0.6),
        case("grid ground only", Changed, grid, |s, _| s.grid_ground_only = true),
        // Camera: lens
        case("projection ortho", Changed, none, |s, _| s.projection = Projection::Orthographic),
        case("projection stereographic", Changed, none, |s, _| s.projection = Projection::Stereographic),
        case("projection panorama", Changed, none, |s, _| s.projection = Projection::Panorama),
        case("projection cylindrical", Changed, none, |s, _| s.projection = Projection::Cylindrical),
        case("field of view", Changed, none, |s, _| s.fov = 90.0),
        case("aperture", Changed, none, dof),
        case("blades", Changed, dof, |s, _| s.blades = 5),
        case("blade rotation", Changed, move |s, m| {
            dof(s, m);
            s.blades = 5;
        }, |s, _| s.blade_rotation = 36.0),
        // Camera: film
        case("exposure", Brighter, none, |s, _| s.exposure = 3.0),
        case("vignette", Darker, none, |s, _| s.vignette = 1.0),
        case("gamma", Brighter, none, |s, _| s.gamma = 4.0),
        case("aces", Changed, none, |s, _| s.aces = true),
        case("bloom", Changed, none, bloom),
        case("bloom mix", Changed, bloom, |s, _| s.bloom_mix = 1.0),
        case("bloom size", Changed, bloom, |s, _| s.bloom_scale = 1.0),
        case("bloom aspect", Changed, bloom, |s, _| s.bloom_aspect = 1.0),
        case("bloom threshold", Changed, bloom, |s, _| s.bloom_threshold = 0.2),
        // Image
        case("denoise", Changed, none, |s, _| s.denoise = true),
        case("denoise view original", Changed, |s, _| s.denoise = true, |s, _| s.denoise_view = DenoiseView::Original),
        case("denoise view split", Changed, |s, _| s.denoise = true, |s, _| s.denoise_view = DenoiseView::Split),
        // Materials
        case("diffuse -> metal", Changed, none, |_, m| m[DIFFUSE].kind = MaterialKind::Metal as u32),
        case("diffuse -> glass", Changed, none, |_, m| m[DIFFUSE].kind = MaterialKind::Glass as u32),
        case("diffuse -> emit", Brighter, none, |_, m| {
            m[DIFFUSE].kind = MaterialKind::Emit as u32;
            m[DIFFUSE].emit = 2.0;
        }),
        case("diffuse -> cloud", Changed, none, |_, m| {
            m[DIFFUSE].kind = MaterialKind::Media as u32;
            m[DIFFUSE].density = 1.0;
        }),
        case("metal roughness", Changed, none, |_, m| m[METAL].rough = 0.9),
        // IOR only shapes the non-metallic share of a metal.
        case("metal ior", Changed, |_, m| m[METAL].metal = 0.5, |_, m| m[METAL].ior = 2.8),
        case("metal metallic", Changed, none, |_, m| m[METAL].metal = 0.0),
        case("glass roughness", Changed, none, |_, m| m[GLASS].rough = 0.9),
        case("glass ior", Changed, none, |_, m| m[GLASS].ior = 2.5),
        case("glass transparency", Changed, none, |_, m| m[GLASS].trans = 0.2),
        case("blend roughness", Changed, none, |_, m| m[BLEND].rough = 0.9),
        case("blend ior", Changed, none, |_, m| m[BLEND].ior = 2.5),
        case("blend metallic", Changed, none, |_, m| m[BLEND].metal = 1.0),
        case("blend transparency", Changed, none, |_, m| m[BLEND].trans = 0.9),
        case("emission", Brighter, none, |_, m| m[EMIT].emit = 20.0),
        case("cloud density", Changed, none, |_, m| m[CLOUD].density = 6.0),
        case("cloud phase", Changed, none, |_, m| m[CLOUD].g = 0.8),
    ];
    if let Some(p) = ibl() {
        let with = move |s: &mut RenderSettings, _: &mut [GpuMaterial]| {
            s.env_mode = EnvMode::Image;
            s.ibl_path = Some(p.clone());
            s.projection = Projection::Panorama;
        };
        v.push(case("sky mode ibl", Changed, none, with.clone()));
        v.push(case("ibl intensity", Brighter, with.clone(), |s, _| s.ibl_intensity = 3.0));
        v.push(case("ibl rotation", Changed, with, |s, _| s.ibl_rotation = 0.25));
    } else {
        eprintln!("no IBL image found; skipping IBL cases");
    }
    v
}

/// Six blocks, one per material kind, with the defaults the UI starts from.
fn test_scene() -> Scene {
    let mut scene = Scene::empty(IVec3::new(32, 24, 12));
    let blocks = [
        (DIFFUSE, IVec3::new(2, 2, 0), 6),
        (METAL, IVec3::new(12, 2, 0), 6),
        (GLASS, IVec3::new(22, 2, 0), 6),
        (EMIT, IVec3::new(26, 14, 0), 3),
        (BLEND, IVec3::new(2, 14, 0), 6),
        (CLOUD, IVec3::new(12, 14, 0), 6),
    ];
    for (c, o, n) in blocks {
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    scene.volume.set(o + IVec3::new(x, y, z), c as u8);
                }
            }
        }
    }
    let m = &mut scene.materials;
    m[METAL].kind = MaterialKind::Metal as u32;
    m[METAL].metal = 1.0;
    m[GLASS].kind = MaterialKind::Glass as u32;
    m[GLASS].trans = 1.0;
    m[EMIT].kind = MaterialKind::Emit as u32;
    m[EMIT].emit = 3.0;
    m[BLEND].kind = MaterialKind::Blend as u32;
    m[BLEND].metal = 0.3;
    m[BLEND].trans = 0.3;
    m[CLOUD].kind = MaterialKind::Media as u32;
    m[CLOUD].density = 1.0;
    scene.rebuild_lights();
    scene
}

fn render(r: &mut Renderer, scene: &mut Scene, s: &RenderSettings, mats: &[GpuMaterial]) -> Vec<u8> {
    r.settings = s.clone();
    scene.materials.copy_from_slice(mats);
    scene.rebuild_lights();
    r.upload_materials(&scene.materials);
    r.set_lights(&scene.lights, scene.light_power);
    r.render_image(SPP).expect("render")
}

fn mean_luma(px: &[u8]) -> f64 {
    px.chunks_exact(4).map(|p| 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64).sum::<f64>() / (px.len() / 4) as f64
}

/// Mean absolute difference per channel (0-255) and fraction of pixels changed.
fn diff(a: &[u8], b: &[u8]) -> (f64, f64) {
    let sum: u64 = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y) as u64).sum();
    let changed = a.chunks_exact(4).zip(b.chunks_exact(4)).filter(|(x, y)| x != y).count();
    (sum as f64 / a.len() as f64, changed as f64 / (a.len() / 4) as f64)
}

#[test]
#[ignore = "needs a GPU; run with --ignored"]
fn every_render_knob_changes_the_image() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let (_adapter, device, queue) = pollster::block_on(gpu::request_device(&instance, None)).expect("gpu");
    let mut scene = test_scene();
    let mut r = Renderer::new(device, queue, &scene, None, W, H);
    let out = std::env::var_os("VOXELY_KNOB_DIR").map(std::path::PathBuf::from);
    let (base_s, base_m) = (scene.settings.clone(), scene.materials.clone());

    // Same settings twice must give the same image, or "changed" means nothing.
    let a = render(&mut r, &mut scene, &base_s, &base_m);
    let b = render(&mut r, &mut scene, &base_s, &base_m);
    assert_eq!(diff(&a, &b).1, 0.0, "path tracer is not deterministic");

    let mut failures = Vec::new();
    for c in cases() {
        let (mut s0, mut m0) = (base_s.clone(), base_m.clone());
        (c.setup)(&mut s0, &mut m0);
        let (mut s1, mut m1) = (s0.clone(), m0.clone());
        (c.change)(&mut s1, &mut m1);
        let before = render(&mut r, &mut scene, &s0, &m0);
        let after = render(&mut r, &mut scene, &s1, &m1);
        // Going back must restore the image exactly (no stale GPU state).
        let again = render(&mut r, &mut scene, &s0, &m0);
        let (mad, frac) = diff(&before, &after);
        let (dl, restored) = (mean_luma(&after) - mean_luma(&before), diff(&before, &again).1 == 0.0);
        let ok = restored
            && frac > 0.0
            && match c.expect {
                Expect::Changed => true,
                Expect::Brighter => dl > 0.0,
                Expect::Darker => dl < 0.0,
                Expect::Transparent => after.chunks_exact(4).any(|p| p[3] == 0) && before.chunks_exact(4).all(|p| p[3] == 255),
            };
        println!("{:<28} {:>4} mad {mad:7.3}  changed {:5.1}%  dluma {dl:+7.2}  restored {restored}", c.name, if ok { "ok" } else { "FAIL" }, frac * 100.0);
        if !ok {
            failures.push(c.name);
        }
        if let Some(dir) = &out {
            let slug = c.name.replace([' ', '>', '(', ')'], "_");
            crate::save_png(&dir.join(format!("{slug}-a.png")), W, H, &before).unwrap();
            crate::save_png(&dir.join(format!("{slug}-b.png")), W, H, &after).unwrap();
        }
    }
    assert!(failures.is_empty(), "knobs with no or wrong effect: {failures:?}");
}
