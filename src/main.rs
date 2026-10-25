mod agents;
mod app;
mod control;
mod edit;
mod env;
mod export;
mod gltf;
mod gpu;
mod mcp;
mod modeler;
mod ops;
mod palette_ops;
#[cfg(test)]
mod render_knobs;
mod scene;
mod ui;
mod vox;
mod xs;

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use winit::event_loop::{ControlFlow, EventLoop};

use gpu::Renderer;
use scene::Scene;

const USAGE: &str = "\
usage: voxely <model.vox> [options]
       voxely --new [<w>x<h>x<d>] [options]
       voxely mcp [--headless] [--assets <MagicaVoxel folder>]
                        MCP server on stdio for agents; drives the open
                        window if there is one, otherwise runs headless

  --new [size]         start from a solid block (default 40x40x40)
  --out <file.vox>     where Ctrl+S saves (default export/<name>.vox)
  --xs <shader.txt>    apply a MagicaVoxel voxel shader to the model first
  --xs-args <a,b,..>   argument values for --xs (default: the file's)

  --render <out.png>   render offscreen and exit
  --spp <n>            samples per pixel for --render (default 256)
  --size <w>x<h>       image size for --render (default 960x640)
  --sky <mode>         uniform | atmo | ibl (default: from the file)
  --ibl <file.hdr>     environment image (default: first .hdr in ../ibl)
  --sun <pitch>,<yaw>  sun angles in degrees
  --fog <density>      fog density (MagicaVoxel units)
  --bloom              enable bloom
  --proj <mode>        pers | ortho | stereo | pano | cyl
  --dof <aperture>     depth of field aperture (0 = off), focus on target
  --blades <n>         aperture polygon sides
  --denoise            denoise the image
  --denoise-view <v>   denoised | original | split
  --edges [r,g,b]      draw voxel edges (sRGB 0-255, default black)
  --edge-width <f>     edge width as a fraction of a voxel
  --grid [n]           draw a grid every n voxels (default 1)
  --grid-ground        grid on the ground only
  --no-ground          hide the ground plane
  --bg-color <r,g,b>   solid background color (sRGB 0-255)
  --bg-transparent     transparent background (PNG alpha 0)
  --cam <yaw>,<pitch>,<dist>[,<x>,<y>,<z>]
                       camera orbit in degrees/voxels, optional target

editor controls (MagicaVoxel layout; Cmd works for Ctrl on macOS):
  left click/drag  use brush       right drag  orbit      wheel  zoom
  middle drag or space + left drag  pan                   F12    reset camera
  W/A/S/D/Q/E move camera          Z + drag  zoom         X + click  focus point
  5 focus selection  Ctrl+5 snap to axis    7/8 save/load camera
  numpad 0-9 load camera slot, Ctrl+numpad saves it
  T attach  R erase  G paint       V voxel  F face  B box  L line
  M marquee select  N region select  (Shift adds, Shift+Alt removes)
  Ctrl+drag move selection         Ctrl+A/D/I select all/none/invert
  Ctrl+C/X/V copy/cut/paste        U full  I fill  Delete clear
  F5/F6/F7 mirror x/y/z            1 / 2  previous/next color
  Ctrl+Z undo  Ctrl+Y redo         Ctrl+S save  Ctrl+Shift+S save as
  Ctrl+O open  Ctrl+P new          Ctrl+G grid  Ctrl+F frame  Ctrl+B bounds
  F1 model/render                  Tab  hide panels
  arrows sun angle                 6  save a PNG snapshot to export/";

#[derive(Default)]
struct Args {
    model: PathBuf,
    render: Option<PathBuf>,
    spp: u32,
    size: (u32, u32),
    sky: Option<env::EnvMode>,
    ibl: Option<PathBuf>,
    sun: Option<(f32, f32)>,
    fog: Option<f32>,
    bloom: bool,
    cam: Option<Vec<f32>>,
    /// Debug: save the viewer with its UI to this PNG and exit.
    ui_shot: Option<PathBuf>,
    /// Show the path tracer instead of the modeler in `--ui-shot`.
    pathtrace: bool,
    proj: Option<scene::Projection>,
    dof: Option<f32>,
    blades: Option<u32>,
    new: Option<glam::IVec3>,
    out: Option<PathBuf>,
    xs: Option<PathBuf>,
    xs_args: Vec<f32>,
    denoise: bool,
    denoise_view: Option<scene::DenoiseView>,
    edges: Option<Option<glam::Vec3>>,
    edge_width: Option<f32>,
    grid: Option<u32>,
    grid_ground: bool,
    no_ground: bool,
    bg: Option<(scene::BgMode, glam::Vec3)>,
}

/// "r,g,b" in sRGB 0-255 to linear.
fn parse_rgb(v: &str) -> Result<glam::Vec3> {
    let c: Vec<f32> = v.split(',').map(str::parse).collect::<Result<_, _>>()?;
    if c.len() != 3 {
        bail!("color needs r,g,b");
    }
    Ok(scene::srgb_to_linear(glam::Vec3::new(c[0], c[1], c[2]) / 255.0))
}

fn parse_args() -> Result<Args> {
    let it = std::env::args().skip(1);
    let mut model = None;
    let mut args = Args { spp: 256, size: (960, 640), ..Default::default() };
    let mut it = it.peekable();
    while let Some(a) = it.next() {
        // An optional value: the next argument unless it is an option.
        let opt = |it: &mut std::iter::Peekable<_>| -> Option<String> { it.next_if(|v: &String| !v.is_empty() && v.chars().all(|c| c.is_ascii_digit() || c == ',' || c == '.')) };
        if a == "--edges" {
            args.edges = Some(opt(&mut it).map(|v| parse_rgb(&v)).transpose()?);
            continue;
        }
        if a == "--grid" {
            args.grid = Some(opt(&mut it).map(|v| v.parse()).transpose()?.unwrap_or(1));
            continue;
        }
        let mut val = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--denoise" => args.denoise = true,
            "--denoise-view" => {
                use scene::DenoiseView::*;
                args.denoise_view = Some(match val()?.as_str() {
                    "denoised" => Denoised,
                    "original" => Original,
                    "split" => Split,
                    other => bail!("unknown denoise view {other}"),
                })
            }
            "--edge-width" => args.edge_width = Some(val()?.parse()?),
            "--grid-ground" => args.grid_ground = true,
            "--no-ground" => args.no_ground = true,
            "--bg-color" => args.bg = Some((scene::BgMode::Color, parse_rgb(&val()?)?)),
            "--bg-transparent" => args.bg = Some((scene::BgMode::Transparent, glam::Vec3::ZERO)),
            "--render" => args.render = Some(PathBuf::from(val()?)),
            "--spp" => args.spp = val()?.parse()?,
            "--size" => {
                let v = val()?;
                let (w, h) = v.split_once('x').context("--size needs WxH")?;
                args.size = (w.parse()?, h.parse()?);
            }
            "--sky" => {
                args.sky = Some(match val()?.as_str() {
                    "uniform" => env::EnvMode::Uniform,
                    "atmo" | "atmosphere" => env::EnvMode::Atmosphere,
                    "ibl" | "image" => env::EnvMode::Image,
                    other => bail!("unknown sky mode {other}"),
                })
            }
            "--ibl" => args.ibl = Some(PathBuf::from(val()?)),
            "--sun" => {
                let v = val()?;
                let (p, y) = v.split_once(',').context("--sun needs pitch,yaw")?;
                args.sun = Some((p.parse()?, y.parse()?));
            }
            "--fog" => args.fog = Some(val()?.parse()?),
            "--bloom" => args.bloom = true,
            "--pathtrace" => args.pathtrace = true,
            "--ui-shot" => args.ui_shot = Some(PathBuf::from(val()?)),
            "--proj" => {
                use scene::Projection::*;
                args.proj = Some(match val()?.as_str() {
                    "pers" => Perspective,
                    "ortho" => Orthographic,
                    "stereo" | "sg" => Stereographic,
                    "pano" => Panorama,
                    "cyl" => Cylindrical,
                    other => bail!("unknown projection {other}"),
                })
            }
            "--dof" => args.dof = Some(val()?.parse()?),
            "--blades" => args.blades = Some(val()?.parse()?),
            "--out" => args.out = Some(PathBuf::from(val()?)),
            "--xs" => args.xs = Some(PathBuf::from(val()?)),
            "--xs-args" => args.xs_args = val()?.split(',').map(str::parse).collect::<Result<_, _>>()?,
            "--new" => args.new = Some(glam::IVec3::splat(40)),
            "--cam" => {
                let v: Vec<f32> = val()?.split(',').map(str::parse).collect::<Result<_, _>>()?;
                if v.len() != 3 && v.len() != 6 {
                    bail!("--cam needs yaw,pitch,dist[,x,y,z]");
                }
                args.cam = Some(v);
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            _ if a.starts_with("--") => bail!("unknown option {a}\n\n{USAGE}"),
            // A size right after --new.
            _ if args.new.is_some() && model.is_none() && a.split('x').count() == 3 && a.split('x').all(|v| v.parse::<i32>().is_ok()) => {
                let v: Vec<i32> = a.split('x').map(|v| v.parse().unwrap()).collect();
                if v.iter().any(|&n| !(1..=1024).contains(&n)) {
                    bail!("--new size must be 1..1024 per side");
                }
                args.new = Some(glam::IVec3::new(v[0], v[1], v[2]));
            }
            _ => model = Some(PathBuf::from(a)),
        }
    }
    args.model = match (model, args.new) {
        (Some(m), _) => m,
        (None, Some(_)) => PathBuf::from("project.vox"),
        (None, None) => bail!("{USAGE}"),
    };
    Ok(args)
}

/// Environment images: the `ibl` folder next to the model's `vox` folder
/// (MagicaVoxel's layout), then the workspace's and MagicaVoxel's own.
pub fn ibl_files(model: &Path) -> Vec<PathBuf> {
    let dirs = [
        model.parent().and_then(Path::parent).map(|d| d.join("ibl")),
        Some(app::workspace_dir().join("ibl")),
        app::magicavoxel_dir().map(|d| d.join("ibl")),
    ];
    let mut files: Vec<PathBuf> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for dir in dirs.into_iter().flatten() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        let mut hdrs: Vec<PathBuf> = rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("hdr")))
            .collect();
        hdrs.sort();
        files.extend(hdrs.into_iter().filter(|p| seen.insert(std::fs::canonicalize(p).unwrap_or_else(|_| p.clone()))));
    }
    files
}

/// The default environment image for a model.
pub fn find_ibl(model: &Path) -> Option<PathBuf> {
    ibl_files(model).into_iter().next()
}

/// The image a file saved, if it is still there, else the default.
fn ibl_for(saved: Option<PathBuf>, model: &Path) -> Option<PathBuf> {
    saved.filter(|p| p.is_file()).or_else(|| find_ibl(model))
}

/// Load a `.vox` file for the editor's Open command.
pub fn open_vox(path: &Path) -> Result<Scene> {
    let mut scene = Scene::from_vox(&vox::VoxFile::load(path)?)?;
    scene.settings.ibl_path = ibl_for(scene.settings.ibl_path.take(), path);
    Ok(scene)
}

fn load_scene(args: &Args) -> Result<Scene> {
    let path = args.model.as_path();
    let t = std::time::Instant::now();
    let mut scene = match args.new {
        Some(dims) => {
            // A solid block to carve into, like MagicaVoxel's New.
            let mut scene = Scene::empty(dims);
            scene.volume = ops::full(&scene.volume, 1);
            scene
        }
        None => Scene::from_vox(&vox::VoxFile::load(path)?)?,
    };
    let s = &mut scene.settings;
    s.ibl_path = args.ibl.clone().or_else(|| ibl_for(s.ibl_path.take(), path));
    if let Some(m) = args.sky {
        s.env_mode = m;
    }
    if let Some((p, y)) = args.sun {
        s.sun_pitch = p;
        s.sun_yaw = y;
    }
    if let Some(f) = args.fog {
        s.fog_density = f;
    }
    s.bloom |= args.bloom;
    if let Some(p) = args.proj {
        s.projection = p;
    }
    if let Some(a) = args.dof {
        s.aperture = a;
    }
    if let Some(b) = args.blades {
        s.blades = b;
    }
    s.denoise |= args.denoise;
    if let Some(v) = args.denoise_view {
        s.denoise_view = v;
    }
    if let Some(c) = args.edges {
        s.show_edges = true;
        s.edge_color = c.unwrap_or(s.edge_color);
    }
    if let Some(w) = args.edge_width {
        s.edge_width = w;
    }
    if let Some(n) = args.grid {
        s.show_grid = true;
        s.grid_spacing = n;
    }
    s.grid_ground_only |= args.grid_ground;
    if args.no_ground {
        s.ground = false;
    }
    if let Some((mode, c)) = args.bg {
        s.bg_mode = mode;
        if mode == scene::BgMode::Color {
            s.bg_color = c;
        }
    }
    let d = scene.volume.dims;
    let filled = scene.volume.cells.iter().filter(|&&c| c != 0).count();
    log::info!("loaded {} ({}x{}x{}, {filled} voxels) in {:.1?}", path.display(), d.x, d.y, d.z, t.elapsed());
    Ok(scene)
}

pub fn save_png(path: &Path, w: u32, h: u32, rgba: &[u8]) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(file, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(rgba)?;
    Ok(())
}

fn apply_cam(r: &mut Renderer, args: &Args) {
    if let Some(c) = &args.cam {
        let mut cam = r.camera;
        cam.yaw = c[0].to_radians();
        cam.pitch = c[1].to_radians();
        cam.dist = c[2];
        if c.len() == 6 {
            cam.target = glam::Vec3::new(c[3], c[4], c[5]);
        }
        r.set_camera(cam);
    }
}

fn render_headless(args: &Args, scene: &Scene) -> Result<()> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let (_adapter, device, queue) = pollster::block_on(gpu::request_device(&instance, None))?;
    let (w, h) = args.size;
    let mut r = Renderer::new(device, queue, scene, None, w, h);
    apply_cam(&mut r, args);
    let pixels = r.render_image(args.spp)?;
    let out = args.render.as_ref().unwrap();
    save_png(out, w, h, &pixels)?;
    println!("wrote {}", out.display());
    Ok(())
}

/// Run a voxel shader over the loaded model (color index 1, no selection).
fn apply_xs(scene: &mut Scene, path: &Path, values: &[f32]) -> Result<()> {
    let shader = xs::XsShader::load(path)?;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let (_adapter, device, queue) = pollster::block_on(gpu::request_device(&instance, None))?;
    let t = std::time::Instant::now();
    scene.volume = xs::run(&device, &queue, &shader, values, &scene.volume, &scene.palette, 1, &[])?;
    let filled = scene.volume.cells.iter().filter(|&&c| c != 0).count();
    log::info!("xs {}: {filled} voxels in {:.1?}", shader.name, t.elapsed());
    Ok(())
}

/// Debug aid: render the scene plus the editor and settings panels
/// offscreen, as the window would show them, and save a PNG. The brush
/// preview is shown for the voxel under the image center.
fn ui_snapshot(args: &Args, scene: &mut Scene, out_path: &Path) -> Result<()> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let (_adapter, device, queue) = pollster::block_on(gpu::request_device(&instance, None))?;
    let (w, h) = args.size;
    let format = wgpu::TextureFormat::Bgra8Unorm;
    let mut egui_renderer = egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());
    let mut r = Renderer::new(device, queue, scene, Some(format), w, h);
    apply_cam(&mut r, args);
    r.modeling = !args.pathtrace;
    let tex = r.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ui_snapshot"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    let ctx = egui::Context::default();
    let ppp = 2.0;
    let panel = ui::PanelState { ibl_files: list_ibl(&args.model, scene.settings.ibl_path.as_ref()) };
    let mut ed = app::EditState::new(1, Some(args.model.clone()), PathBuf::new());
    ed.enabled = r.modeling;
    let mut out = None;
    // A few passes so layout and fonts settle; textures upload every pass.
    // The 3D view is fitted between the panels, then rendered.
    for pass in 0..3 {
        if pass == 2 {
            r.render_to_view(&view, (w, h), args.spp);
            let (x, y, vw, vh) = r.viewport();
            let (ro, rd) = r.camera_ray((x + vw / 2) as f32, (y + vh / 2) as f32);
            ed.hover = edit::pick(&scene.volume, ro, rd);
        }
        let mut raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32 / ppp, h as f32 / ppp))),
            ..Default::default()
        };
        raw.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(ppp);
        let o = ctx.run_ui(raw, |ui| {
            let a = app::build_ui(ui, &mut r, scene, &mut ed, &panel, true);
            if pass < 2 {
                if let Some(rect) = a.viewport {
                    app::fit_viewport(&mut r, rect, ppp, (w, h));
                }
            }
        });
        for (id, deltas) in &o.textures_delta.set {
            for d in deltas {
                egui_renderer.update_texture(&r.device, &r.queue, *id, d);
            }
        }
        out = Some(o);
    }
    let mut out = out.unwrap();
    out.textures_delta.set.clear();
    app::paint_egui(&r.device, &r.queue, &mut egui_renderer, &ctx, &view, [w, h], out);

    let row = (w * 4).next_multiple_of(256);
    let buf = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ui_readback"),
        size: (row * h) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = r.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    r.queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |res| res.expect("map failed"));
    r.device.poll(wgpu::PollType::wait_indefinitely())?;
    let data = slice.get_mapped_range()?;
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for px in data[(y * row) as usize..(y * row + w * 4) as usize].chunks_exact(4) {
            rgba.extend_from_slice(&[px[2], px[1], px[0], 255]);
        }
    }
    drop(data);
    save_png(out_path, w, h, &rgba)?;
    println!("wrote {}", out_path.display());
    Ok(())
}

/// Every environment image the IBL list offers, plus the chosen one.
fn list_ibl(model: &Path, current: Option<&PathBuf>) -> Vec<PathBuf> {
    let mut files = ibl_files(model);
    if let Some(c) = current {
        if !files.contains(c) {
            files.push(c.clone());
        }
    }
    files
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("voxely=info,wgpu=warn")).init();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().is_some_and(|a| a == "mcp") {
        return mcp::run(&argv[1..]);
    }
    let args = parse_args()?;
    let mut scene = load_scene(&args)?;
    if let Some(p) = &args.xs {
        apply_xs(&mut scene, p, &args.xs_args)?;
    }
    if args.render.is_some() {
        return render_headless(&args, &scene);
    }
    if let Some(p) = &args.ui_shot {
        return ui_snapshot(&args, &mut scene, p);
    }
    let source = args.new.is_none().then(|| args.model.clone());
    let save_path = args.out.clone().unwrap_or_else(|| app::default_save_path(source.as_deref()));
    println!("Ctrl+S saves to {}", save_path.display());
    let panel = ui::PanelState { ibl_files: list_ibl(&args.model, scene.settings.ibl_path.as_ref()) };
    let event_loop = EventLoop::<app::UserEvent>::with_user_event().build()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    // Agents connect through `voxely mcp`, which talks to this socket.
    app::serve_control(event_loop.create_proxy());
    let mut app = app::App::new(scene, panel, source, save_path);
    let result = event_loop.run_app(&mut app);
    let _ = std::fs::remove_file(mcp::socket_path());
    result?;
    Ok(())
}
