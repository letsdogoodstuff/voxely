//! The interactive window: modeler and path-traced views, editor and
//! settings panels, brush overlay, and MagicaVoxel-style mouse and
//! keyboard controls.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use glam::{IVec3, Vec2, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::edit::{self, Editor, Pick, SelMode, Tool};
use crate::gpu::{self, Camera, Renderer};
use crate::ops;
use crate::scene::{MaterialKind, Projection, Scene};
use crate::ui::{self, Command, Op};

/// Size of a new project, like MagicaVoxel's default.
const NEW_SIZE: i32 = 40;
/// Camera slots: 0-9 on the number pad, plus the quick slot (7 / 8).
const QUICK_SLOT: usize = 10;

/// Editing state shared by the window and the offscreen UI snapshot.
pub struct EditState {
    pub editor: Editor,
    pub hover: Option<Pick>,
    /// Editing happens in the modeler; in the renderer the left mouse
    /// button orbits instead (F1 switches).
    pub enabled: bool,
    /// The file the project was opened from, if any.
    pub source: Option<PathBuf>,
    pub save_path: PathBuf,
    /// `.vox` files next to the source, for the Project list.
    pub projects: Vec<PathBuf>,
    /// Palette images in MagicaVoxel's `palette` folder.
    pub palettes: Vec<PathBuf>,
    pub project_tab: ui::ProjectTab,
    /// Free camera: rotate around the eye instead of the target.
    pub free_look: bool,
    pub shaders: ui::ShaderState,
    pub light_tab: ui::LightTab,
    pub matter_tab: ui::MatterTab,
    pub material_clipboard: Option<crate::scene::GpuMaterial>,
    pub image: ui::ImageSettings,
    pub transform_text: ui::TransformText,
    pub palette: ui::PaletteState,
    /// Canvas size being typed into the Canvas panel.
    pub canvas: IVec3,
    canvas_for: IVec3,
    /// Marquee drag corners in window pixels.
    pub marquee: Option<(Vec2, Vec2)>,
    cameras: [Option<Camera>; QUICK_SLOT + 1],
}

impl EditState {
    pub fn new(color: u8, source: Option<PathBuf>, save_path: PathBuf) -> Self {
        let projects = list_projects(source.as_deref());
        let palettes = list_palettes(source.as_deref());
        EditState {
            palettes,
            project_tab: ui::ProjectTab::default(),
            free_look: false,
            light_tab: ui::LightTab::default(),
            matter_tab: ui::MatterTab::default(),
            material_clipboard: None,
            image: ui::ImageSettings::default(),
            transform_text: ui::TransformText::default(),
            palette: ui::PaletteState::default(),
            shaders: ui::ShaderState { files: list_shaders(source.as_deref()), current: None, values: Vec::new(), error: None },
            editor: Editor::new(color),
            hover: None,
            enabled: true,
            source,
            save_path,
            projects,
            canvas: IVec3::ZERO,
            canvas_for: IVec3::ZERO,
            marquee: None,
            cameras: [None; QUICK_SLOT + 1],
        }
    }

    fn name(&self) -> String {
        self.save_path.file_stem().map_or("project".into(), |n| n.to_string_lossy().into_owned())
    }
}

/// Sorted `.vox` files in the directory of `source`.
fn list_projects(source: Option<&Path>) -> Vec<PathBuf> {
    let Some(dir) = source.and_then(Path::parent) else { return Vec::new() };
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("vox")))
        .collect();
    out.sort();
    out
}

/// MagicaVoxel keeps palette images in a `palette` folder next to `vox`.
fn list_palettes(source: Option<&Path>) -> Vec<PathBuf> {
    let Some(dir) = source.and_then(Path::parent).and_then(Path::parent) else { return Vec::new() };
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir.join("palette"))
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("png")))
        .collect();
    out.sort();
    out
}

/// MagicaVoxel keeps voxel shaders in a `shader` folder next to `vox`.
fn list_shaders(source: Option<&Path>) -> Vec<PathBuf> {
    source.and_then(Path::parent).and_then(Path::parent).map_or_else(Vec::new, |d| crate::xs::list_shaders(&d.join("shader")))
}

/// Give color `i` material `m`, keeping its color.
fn set_material(r: &mut Renderer, scene: &mut Scene, i: u8, mut m: crate::scene::GpuMaterial) {
    let i = i as usize;
    m.color = scene.materials[i].color;
    scene.materials[i] = m;
    r.upload_materials(&scene.materials);
    scene.rebuild_lights();
    r.set_lights(&scene.lights, scene.light_power);
}

/// Render the final image offscreen at the Image size and save it.
fn render_final(r: &Renderer, scene: &Scene, image: ui::ImageSettings) {
    let mut out = Renderer::new(r.device.clone(), r.queue.clone(), scene, None, image.width, image.height);
    out.settings = r.settings.clone();
    out.set_camera(r.camera);
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let name = format!("export/voxely-render-{secs}.png");
    match out.render_image(image.spp).and_then(|px| crate::save_png(Path::new(&name), image.width, image.height, &px)) {
        Ok(()) => eprintln!("rendered {name} ({} spp)", image.spp),
        Err(e) => eprintln!("render failed: {e:#}"),
    }
}

/// Run the chosen shader over the volume.
fn shader_output(r: &Renderer, scene: &Scene, ed: &mut EditState) -> Option<crate::scene::Volume> {
    let (_, shader) = ed.shaders.current.as_ref()?;
    let sels: Vec<u8> = (1..256).filter(|&i| ed.palette.selected[i]).map(|i| i as u8).collect();
    match crate::xs::run(&r.device, &r.queue, shader, &ed.shaders.values, &scene.volume, &scene.palette, ed.editor.color, &sels) {
        Ok(v) => {
            ed.shaders.error = None;
            Some(v)
        }
        Err(e) => {
            ed.shaders.error = Some(format!("{e:#}"));
            None
        }
    }
}

/// Before a Code brush stroke, compute what the shader would make.
pub fn prepare_code_brush(r: &Renderer, scene: &Scene, ed: &mut EditState) {
    ed.editor.code = if ed.editor.brush == edit::Brush::Code { shader_output(r, scene, ed) } else { None };
}

/// Colors of a palette image: pixel k is color index k + 1.
pub(crate) fn load_palette(path: &Path) -> Result<Vec<[u8; 4]>> {
    let mut decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::ALPHA);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size().context("palette image too large")?];
    let info = reader.next_frame(&mut buf)?;
    let px = &buf[..info.buffer_size()];
    let channels = info.color_type.samples();
    let colors: Vec<[u8; 4]> = px
        .chunks_exact(channels)
        .take(255)
        .map(|c| match channels {
            4 => [c[0], c[1], c[2], c[3]],
            3 => [c[0], c[1], c[2], 255],
            2 => [c[0], c[0], c[0], c[1]],
            _ => [c[0], c[0], c[0], 255],
        })
        .collect();
    anyhow::ensure!(!colors.is_empty(), "empty palette image");
    Ok(colors)
}

/// Snap to the nearest isometric view (orthographic, 45 degree turns).
pub(crate) fn iso_camera(r: &mut Renderer) {
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};
    let cam = &mut r.camera;
    cam.yaw = ((cam.yaw - FRAC_PI_4) / FRAC_PI_2).round() * FRAC_PI_2 + FRAC_PI_4;
    cam.pitch = (1.0 / 2f32.sqrt()).atan();
    r.settings.projection = Projection::Orthographic;
    r.reset();
}

fn is_iso(r: &Renderer) -> bool {
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};
    let k = (r.camera.yaw - FRAC_PI_4) / FRAC_PI_2;
    r.settings.projection == Projection::Orthographic && (k - k.round()).abs() < 1e-3 && (r.camera.pitch - (1.0 / 2f32.sqrt()).atan()).abs() < 1e-3
}

/// Where Ctrl+S writes a project by default: `export/<name>.vox`, so the
/// original (which saving flattens) is not overwritten.
pub fn default_save_path(source: Option<&Path>) -> PathBuf {
    let name = source.and_then(Path::file_name).map_or("project.vox".into(), |n| n.to_string_lossy().into_owned());
    PathBuf::from("export").join(name)
}

/// Push voxel edits to the GPU, refreshing emissive faces if any exist.
pub fn sync_volume(r: &mut Renderer, scene: &mut Scene, editor: &mut Editor) {
    r.upload_volume(&scene.volume, editor.selection());
    editor.sel_dirty = false;
    let any_emit = scene.materials.iter().any(|m| m.kind == MaterialKind::Emit as u32 && m.emit > 0.0);
    if any_emit || !scene.lights.is_empty() {
        scene.rebuild_lights();
        r.set_lights(&scene.lights, scene.light_power);
    }
}

pub fn save(scene: &Scene, path: &Path) {
    let result = path.parent().map_or(Ok(()), std::fs::create_dir_all).map_err(anyhow::Error::from).and_then(|_| scene.save(path));
    match result {
        Ok(()) => eprintln!("saved {}", path.display()),
        Err(e) => eprintln!("save failed: {e:#}"),
    }
}

pub fn snapshot(r: &Renderer) {
    let (w, h) = r.size();
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let name = format!("export/voxely-{secs}.png");
    match r.read_pixels().and_then(|px| crate::save_png(Path::new(&name), w, h, &px)) {
        Ok(()) => eprintln!("saved {name}"),
        Err(e) => eprintln!("snapshot failed: {e:#}"),
    }
}

/// Bounding box of solid cells (selected ones, if a selection is given).
fn solid_bounds(scene: &Scene, sel: Option<&[bool]>) -> Option<(Vec3, Vec3)> {
    let v = &scene.volume;
    let mut b: Option<(IVec3, IVec3)> = None;
    for (i, &c) in v.cells.iter().enumerate() {
        if c != 0 && sel.is_none_or(|s| s[i]) {
            let p = v.position(i);
            b = Some(b.map_or((p, p), |(lo, hi)| (lo.min(p), hi.max(p))));
        }
    }
    b.map(|(lo, hi)| (lo.as_vec3(), (hi + IVec3::ONE).as_vec3()))
}

/// Point the project, palette and shader lists at the folders around
/// `model` (a file in MagicaVoxel's vox folder). Lists that come up empty
/// keep what was found before.
pub(crate) fn set_asset_lists(ed: &mut EditState, model: &Path) {
    ed.projects = list_projects(Some(model));
    let palettes = list_palettes(Some(model));
    if !palettes.is_empty() {
        ed.palettes = palettes;
    }
    let shaders = list_shaders(Some(model));
    if !shaders.is_empty() {
        ed.shaders.files = shaders;
    }
}

pub(crate) fn open_project(path: PathBuf, r: &mut Renderer, scene: &mut Scene, ed: &mut EditState) -> Result<()> {
    *scene = crate::open_vox(&path).with_context(|| format!("opening {}", path.display()))?;
    ed.save_path = default_save_path(Some(&path));
    set_asset_lists(ed, &path);
    ed.source = Some(path);
    loaded(r, scene, ed);
    Ok(())
}

/// Show a newly opened or created scene.
pub(crate) fn loaded(r: &mut Renderer, scene: &mut Scene, ed: &mut EditState) {
    ed.editor.reset();
    ed.hover = None;
    r.load_scene(scene);
    ed.editor.sel_dirty = false;
    eprintln!("Ctrl+S saves to {}", ed.save_path.display());
}

fn dialog_dir(ed: &EditState) -> PathBuf {
    ed.source.as_deref().and_then(Path::parent).map_or_else(|| PathBuf::from("."), Path::to_path_buf)
}

pub(crate) fn apply_op(op: Op, scene: &Scene, editor: &Editor) -> Option<crate::scene::Volume> {
    let v = &scene.volume;
    let (sel, c) = (editor.selection(), editor.color);
    Some(match op {
        Op::Full => ops::full(v, c),
        Op::Fill => ops::fill(v, sel, c),
        Op::Clear => ops::clear(v, sel),
        Op::Reverse => ops::reverse(v, c),
        Op::Double => match ops::double(v) {
            Some(d) => d,
            None => {
                eprintln!("2X: the canvas would exceed {} voxels per side", ops::MAX_DIM);
                return None;
            }
        },
        Op::Half => ops::half(v),
        Op::Flip(a) => ops::flip(v, a),
        Op::Fit => ops::fit(v)?,
        Op::Loop(a, d) => ops::loop_shift(v, sel, a, d),
        Op::Mirror(a) => ops::mirror(v, a),
        Op::Shape(k) => ops::shape(v, k, c),
        Op::Diagonal(a) => ops::diagonal(v, a),
        Op::Maze => ops::maze(v, c, 4),
        Op::Perlin => ops::perlin(v, c, seed()),
        Op::Rotate(a) => ops::rotate(v, a),
        Op::Hull => ops::hull(v),
        Op::Flood => ops::flood(v, c),
        Op::Dilate => ops::dilate(v),
        Op::Erode => ops::erode(v),
    })
}

/// A seed that changes every call, for Perlin and Rand.
fn seed() -> u32 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.subsec_nanos() | 1)
}

/// Selected palette colors, or the current color if none are selected.
fn selected_colors(ed: &EditState) -> Vec<u8> {
    let sel: Vec<u8> = (1..256).filter(|&i| ed.palette.selected[i]).map(|i| i as u8).collect();
    if sel.is_empty() { vec![ed.editor.color] } else { sel }
}

/// Apply a Scale / Shear / Rotate / Repeat text like "xyz 2".
pub(crate) fn transform(kind: ui::TransformKind, text: &str, v: &crate::scene::Volume) -> Option<crate::scene::Volume> {
    let Some((axes, nums)) = ops::parse_axes_args(text) else {
        eprintln!("expected [axes] [values], e.g. \"xyz 2\", got {text:?}");
        return None;
    };
    let first = axes.iter().position(|&a| a)?;
    let n = |k: usize| nums.get(k).copied();
    match kind {
        ui::TransformKind::Scale => ops::scale(v, axes, n(0)?),
        ui::TransformKind::Shear => Some(ops::shear(v, first, n(0)?, n(1).unwrap_or(n(0)?))),
        ui::TransformKind::Rotate => Some(ops::rotate_angle(v, first, n(0)?)),
        ui::TransformKind::Repeat => ops::repeat(v, axes, n(0)?.max(1.0) as u32),
    }
}

/// Run a button or hotkey command.
pub fn run_command(cmd: Command, r: &mut Renderer, scene: &mut Scene, ed: &mut EditState) {
    let dims = scene.volume.dims;
    let changed = match cmd {
        Command::New => {
            let ibl = r.settings.ibl_path.clone();
            // Like MagicaVoxel, a new project is a solid block of the
            // current color to carve into.
            *scene = Scene::empty(IVec3::splat(NEW_SIZE));
            scene.volume = ops::full(&scene.volume, ed.editor.color);
            scene.settings.ibl_path = ibl;
            ed.source = None;
            ed.save_path = default_save_path(None);
            loaded(r, scene, ed);
            false
        }
        Command::Open => {
            if let Some(p) = rfd::FileDialog::new().add_filter("MagicaVoxel", &["vox"]).set_directory(dialog_dir(ed)).pick_file() {
                if let Err(e) = open_project(p, r, scene, ed) {
                    eprintln!("open failed: {e:#}");
                }
            }
            false
        }
        Command::OpenPath(p) => {
            if let Err(e) = open_project(p, r, scene, ed) {
                eprintln!("open failed: {e:#}");
            }
            false
        }
        Command::Save => {
            save(scene, &ed.save_path);
            false
        }
        Command::SaveAs => {
            let dir = ed.save_path.parent().filter(|d| d.is_dir()).map_or_else(|| dialog_dir(ed), Path::to_path_buf);
            let dialog = rfd::FileDialog::new().add_filter("MagicaVoxel", &["vox"]).set_directory(dir).set_file_name(format!("{}.vox", ed.name()));
            if let Some(p) = dialog.save_file() {
                ed.save_path = p.with_extension("vox");
                save(scene, &ed.save_path);
            }
            false
        }
        Command::Undo => ed.editor.undo(&mut scene.volume),
        Command::Redo => ed.editor.redo(&mut scene.volume),
        Command::Op(op) => match apply_op(op, scene, &ed.editor) {
            Some(new) => ed.editor.commit(&mut scene.volume, new),
            None => false,
        },
        Command::Resize(d) => {
            let new = ops::resize(&scene.volume, d);
            ed.editor.commit(&mut scene.volume, new)
        }
        Command::SelectAll => {
            ed.editor.select_all(&scene.volume);
            false
        }
        Command::SelectNone => {
            ed.editor.select_none();
            false
        }
        Command::SelectInvert => {
            ed.editor.select_invert(&scene.volume);
            false
        }
        Command::Copy => {
            ed.editor.copy(&scene.volume);
            false
        }
        Command::Cut => ed.editor.cut(&mut scene.volume),
        Command::Paste => ed.editor.paste(&mut scene.volume),
        Command::ResetCamera => {
            r.frame(dims.as_vec3());
            false
        }
        Command::Focus => {
            let (lo, hi) = solid_bounds(scene, ed.editor.selection()).unwrap_or((Vec3::ZERO, dims.as_vec3()));
            r.camera.target = (lo + hi) * 0.5;
            r.reset();
            false
        }
        Command::Snapshot => {
            snapshot(r);
            false
        }
        Command::LoadPattern(p) => {
            match crate::open_vox(&p) {
                Ok(s) => match ops::fit(&s.volume) {
                    Some(volume) => {
                        let name = p.file_stem().map_or(String::new(), |n| n.to_string_lossy().into_owned());
                        ed.editor.pattern = Some(edit::Pattern { name, volume });
                        ed.editor.brush = edit::Brush::Pattern;
                    }
                    None => eprintln!("pattern {}: empty model", p.display()),
                },
                Err(e) => eprintln!("pattern {}: {e:#}", p.display()),
            }
            false
        }
        Command::SelectShader(p) => {
            match crate::xs::XsShader::load(&p) {
                Ok(sh) => {
                    ed.shaders.values = sh.args.iter().map(|a| a.value).collect();
                    ed.shaders.current = Some((p, sh));
                    ed.shaders.error = None;
                    ed.editor.brush = edit::Brush::Code;
                }
                Err(e) => ed.shaders.error = Some(format!("{e:#}")),
            }
            false
        }
        Command::RunShader => match shader_output(r, scene, ed) {
            Some(out) => {
                // With a selection, only selected voxels change.
                let new = match ed.editor.selection() {
                    Some(sel) => crate::scene::Volume::from_cells(
                        scene.volume.dims,
                        scene.volume.cells.iter().zip(&out.cells).enumerate().map(|(i, (&a, &b))| if sel[i] { b } else { a }).collect(),
                    ),
                    None => out,
                };
                ed.editor.commit(&mut scene.volume, new)
            }
            None => false,
        },
        Command::PasteMaterial => {
            if let Some(m) = ed.material_clipboard {
                set_material(r, scene, ed.editor.color, m);
            }
            false
        }
        Command::ResetMaterial => {
            let i = ed.editor.color;
            set_material(r, scene, i, crate::scene::material(i as u32, scene.palette[i as usize], None));
            false
        }
        Command::Transform(kind, text) => match transform(kind, &text, &scene.volume) {
            Some(new) => ed.editor.commit(&mut scene.volume, new),
            None => false,
        },
        Command::PaletteSort => {
            let (palette, remap) = crate::palette_ops::sort(&scene.palette, &scene.volume);
            let new = crate::palette_ops::apply_remap(&scene.volume, &remap);
            // Materials follow their colors to the new indices.
            let mut materials = scene.materials.clone();
            for (old, &to) in remap.iter().enumerate() {
                materials[to as usize] = scene.materials[old];
            }
            scene.palette = palette;
            scene.materials = materials;
            ed.editor.color = remap[ed.editor.color as usize];
            let old_sel = ed.palette.selected.clone();
            for (old, &to) in remap.iter().enumerate() {
                ed.palette.selected[to as usize] = old_sel[old];
            }
            r.upload_materials(&scene.materials);
            r.set_palette(&scene.palette);
            // Sorting isn't undoable as one step with the palette, so the
            // voxel remap is committed without history.
            scene.volume = new;
            true
        }
        Command::PaletteRandom => {
            let colors = selected_colors(ed);
            let new = crate::palette_ops::random_colors(&scene.volume, ed.editor.selection(), &colors, seed());
            ed.editor.commit(&mut scene.volume, new)
        }
        Command::RenderImage => {
            render_final(r, scene, ed.image);
            false
        }
        Command::PatternFromSelection => {
            if ed.editor.pattern_from_selection(&scene.volume) {
                ed.editor.brush = edit::Brush::Pattern;
            }
            false
        }
        Command::LoadPalette(p) => {
            match load_palette(&p) {
                Ok(colors) => {
                    for (k, c) in colors.into_iter().enumerate() {
                        scene.palette[k + 1] = c;
                        scene.sync_color(k + 1);
                    }
                    r.upload_materials(&scene.materials);
                    r.set_palette(&scene.palette);
                    scene.rebuild_lights();
                    r.set_lights(&scene.lights, scene.light_power);
                }
                Err(e) => eprintln!("palette {}: {e:#}", p.display()),
            }
            false
        }
    };
    if changed {
        sync_volume(r, scene, &mut ed.editor);
        // Keep the model in view when the canvas changes size.
        let new = scene.volume.dims;
        if new != dims {
            r.camera.target = new.as_vec3() * 0.5;
            r.camera.dist *= new.as_vec3().length() / dims.as_vec3().length();
        }
    }
    if ed.editor.sel_dirty {
        r.upload_selection(&scene.volume, ed.editor.selection());
        ed.editor.sel_dirty = false;
    }
}

fn outline(p: &egui::Painter, r: &Renderer, ppp: f32, lo: Vec3, hi: Vec3, stroke: egui::Stroke) {
    let corner = |i: u32| Vec3::new(if i & 1 != 0 { hi.x } else { lo.x }, if i & 2 != 0 { hi.y } else { lo.y }, if i & 4 != 0 { hi.z } else { lo.z });
    let pts: Vec<Option<egui::Pos2>> = (0..8).map(|i| r.project(corner(i)).map(|(x, y)| egui::pos2(x / ppp, y / ppp))).collect();
    for a in 0..8u32 {
        for bit in [1, 2, 4] {
            let b = a | bit;
            if b != a {
                if let (Some(pa), Some(pb)) = (pts[a as usize], pts[b as usize]) {
                    p.line_segment([pa, pb], stroke);
                }
            }
        }
    }
}

/// Model bounds, the marquee, and the cells the current brush would touch.
fn draw_overlay(p: &egui::Painter, r: &Renderer, scene: &Scene, ed: &EditState, ppp: f32) {
    // Only draw over the 3D view, not the panels.
    let (x, y, w, h) = r.viewport();
    let p = &p.with_clip_rect(egui::Rect::from_min_size(egui::pos2(x as f32, y as f32) / ppp, egui::vec2(w as f32, h as f32) / ppp));
    draw_gizmo(p, r, ppp);
    // Brush previews show only in the modeler. (The bounding box is drawn
    // by the modeler itself, depth-tested.)
    if !r.modeling || !ed.enabled {
        return;
    }
    if let Some((a, b)) = ed.marquee {
        let rect = egui::Rect::from_two_pos(egui::pos2(a.x, a.y) / ppp, egui::pos2(b.x, b.y) / ppp);
        p.rect_filled(rect, 0.0, egui::Color32::from_rgba_unmultiplied(80, 160, 255, 30));
        p.rect_stroke(rect, 0.0, egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 160, 255)), egui::StrokeKind::Inside);
        return;
    }
    let cells = ed.editor.preview(&scene.volume, ed.hover);
    if cells.is_empty() {
        return;
    }
    let color = match ed.editor.tool {
        Tool::Attach => egui::Color32::WHITE,
        Tool::Erase => egui::Color32::from_rgb(255, 80, 80),
        Tool::Paint => {
            let c = scene.palette[ed.editor.color as usize];
            egui::Color32::from_rgb(c[0], c[1], c[2])
        }
        Tool::Marquee | Tool::Region | Tool::Move => egui::Color32::from_rgb(80, 160, 255),
        Tool::RegionRemove => egui::Color32::from_rgb(255, 80, 80),
        Tool::RegionReplace => {
            let c = scene.palette[ed.editor.color as usize];
            egui::Color32::from_rgb(c[0], c[1], c[2])
        }
        Tool::Pick => egui::Color32::YELLOW,
    };
    let stroke = egui::Stroke::new(1.5, color);
    if cells.len() <= 64 {
        for c in &cells {
            outline(p, r, ppp, c.as_vec3(), c.as_vec3() + Vec3::ONE, stroke);
        }
    } else {
        let lo = cells.iter().fold(IVec3::MAX, |a, c| a.min(*c));
        let hi = cells.iter().fold(IVec3::MIN, |a, c| a.max(*c));
        outline(p, r, ppp, lo.as_vec3(), (hi + IVec3::ONE).as_vec3(), stroke);
    }
}

/// Orientation cube in the bottom-right corner of the view, with the X, Y
/// and Z edges from the origin corner in red, green and blue.
fn draw_gizmo(p: &egui::Painter, r: &Renderer, ppp: f32) {
    let (x, y, w, h) = r.viewport();
    let size = 22.0;
    let center = egui::pos2((x + w) as f32 / ppp - 50.0, (y + h) as f32 / ppp - 50.0);
    let (_, right, up) = r.camera.basis();
    let screen = |v: Vec3| center + egui::vec2(v.dot(right), -v.dot(up)) * size;
    let corner = |i: u32| Vec3::new((i & 1) as f32, ((i >> 1) & 1) as f32, (i >> 2) as f32) - 0.5;
    let axis = [egui::Color32::from_rgb(230, 60, 60), egui::Color32::from_rgb(60, 200, 60), egui::Color32::from_rgb(70, 110, 255)];
    for a in 0..8u32 {
        for (k, bit) in [1u32, 2, 4].into_iter().enumerate() {
            let b = a | bit;
            if b != a {
                let color = if a == 0 { axis[k] } else { egui::Color32::from_white_alpha(140) };
                p.line_segment([screen(corner(a)), screen(corner(b))], egui::Stroke::new(if a == 0 { 2.0 } else { 1.0 }, color));
            }
        }
    }
}

/// Fit the renderer to the area between the panels (in points).
pub fn fit_viewport(r: &mut Renderer, rect: egui::Rect, ppp: f32, window: (u32, u32)) {
    let x = (rect.min.x * ppp).round().clamp(0.0, window.0 as f32 - 1.0) as u32;
    let y = (rect.min.y * ppp).round().clamp(0.0, window.1 as f32 - 1.0) as u32;
    let w = ((rect.width() * ppp).round() as u32).clamp(1, window.0 - x);
    let h = ((rect.height() * ppp).round() as u32).clamp(1, window.1 - y);
    if r.viewport() != (x, y, w, h) {
        r.set_viewport(x, y, w, h);
    }
}

/// Lay out both panels and the overlay. Returns the commands asked for,
/// which the caller runs after the frame (they may open dialogs).
pub fn build_ui(
    ui: &mut egui::Ui,
    r: &mut Renderer,
    scene: &mut Scene,
    ed: &mut EditState,
    panel: &ui::PanelState,
    show: bool,
) -> ui::Actions {
    let ppp = ui.ctx().pixels_per_point();
    let (spp, size) = (r.frame, r.size());
    let mut actions = ui::Actions::default();
    let mut edit = ui::EditActions::default();
    let before = r.settings.clone();
    let dims = scene.volume.dims;
    if ed.canvas_for != dims {
        (ed.canvas, ed.canvas_for) = (dims, dims);
    }
    let cmds = &mut actions.commands;
    if show {
        egui::Panel::top("top").show(ui, |ui| {
            let was = r.modeling;
            let mut modeling = r.modeling;
            ui::top_bar(ui, &mut modeling, &ed.name(), &ed.editor, cmds);
            if r.modeler.is_some() && modeling != was {
                r.modeling = modeling;
                ed.enabled = modeling;
                r.reset();
            }
        });
        egui::Panel::bottom("status").show(ui, |ui| {
            let hover = ed.hover.filter(|h| !h.on_ground()).map(|h| (h.cell, scene.volume.get(h.cell)));
            let hint = match (r.modeling, ed.editor.tool) {
                (false, _) => "Rotate [RButton/LButton] : Move [MButton] : Zoom [Wheel]",
                (_, Tool::Marquee) => "Select [LButton drag] : Add [Shift] : Remove [Shift+Alt] : Rotate [RButton]",
                (_, Tool::Region) => "Select region [LButton] : Add [Shift] : Remove [Shift+Alt] : Rotate [RButton]",
                (_, Tool::RegionRemove) => "Remove region [LButton] : Rotate [RButton] : Pan [MButton]",
                (_, Tool::RegionReplace) => "Recolor region [LButton] : Pick color [Alt+LButton] : Rotate [RButton]",
                (_, Tool::Move) => "Move [LButton drag] : Rotate [RButton] : Pan [MButton]",
                (_, Tool::Pick) => "Pick color [LButton] : Rotate [RButton] : Pan [MButton]",
                _ => "Draw [LButton] : Move selection [Ctrl+LButton] : Rotate [RButton] : Pan [MButton]",
            };
            let iso = is_iso(r);
            let options = r.modeler.as_mut().filter(|_| r.modeling).map(|m| &mut m.options);
            let status = ui::Status { options, hover, projection: &mut r.settings.projection, iso, free: &mut ed.free_look, hint };
            if ui::status_bar(ui, status, cmds) {
                iso_camera(r);
            }
        });
        egui::Panel::left("palette").exact_size(160.0).resizable(false).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                edit = ui::palette_panel(ui, &mut ed.editor, scene, &mut ed.palette, &ed.palettes, cmds);
            });
        });
        // The Project column shows in both modes, like MagicaVoxel.
        egui::Panel::right("project").exact_size(150.0).resizable(false).show(ui, |ui| {
            let current = ed.source.clone().unwrap_or_default();
            let pattern = ed.editor.pattern.as_ref().map(|p| p.name.clone());
            ui::project_panel(ui, &mut ed.project_tab, &ed.projects, &current, pattern.as_deref(), &mut ed.shaders, cmds);
        });
        if r.modeling {
            egui::Panel::left("brush").exact_size(190.0).resizable(false).show(ui, |ui| ui::brush_panel(ui, &mut ed.editor, cmds));
            if let Some(m) = r.modeler.as_mut() {
                egui::Panel::right("edit").exact_size(160.0).resizable(false).show(ui, |ui| ui::edit_panel(ui, &ed.editor, &mut m.options, &mut ed.transform_text, cmds));
            }
            egui::Panel::top("viewbar").show(ui, |ui| ui::view_bar(ui, dims, &mut ed.canvas, cmds));
        } else {
            // MagicaVoxel's render layout: Light / Sample / Display on the
            // left, Matter / Camera / Image on the right.
            egui::Panel::left("light").default_size(250.0).show(ui, |ui| ui::light_panel(ui, &mut ed.light_tab, &mut r.settings, panel));
            egui::Panel::right("matter").default_size(250.0).show(ui, |ui| {
                let state = ui::MatterState { tab: &mut ed.matter_tab, clipboard: &mut ed.material_clipboard, image: &mut ed.image, spp, size };
                let a = ui::matter_panel(ui, &mut r.settings, &ed.editor, scene, state, cmds);
                edit.materials_changed |= a.materials_changed;
            });
        }
    }
    actions.viewport = Some(ui.available_rect_before_wrap());
    draw_overlay(&ui.ctx().layer_painter(egui::LayerId::background()), r, scene, ed, ppp);

    if r.settings != before {
        r.reset();
    }
    if edit.materials_changed {
        r.upload_materials(&scene.materials);
        r.set_palette(&scene.palette);
        scene.rebuild_lights();
        r.set_lights(&scene.lights, scene.light_power);
    }
    actions
}

/// Paint one egui frame on top of `view`.
pub fn paint_egui(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    egui_renderer: &mut egui_wgpu::Renderer,
    ctx: &egui::Context,
    view: &wgpu::TextureView,
    size_in_pixels: [u32; 2],
    out: egui::FullOutput,
) {
    let jobs = ctx.tessellate(out.shapes, out.pixels_per_point);
    let screen = egui_wgpu::ScreenDescriptor { size_in_pixels, pixels_per_point: out.pixels_per_point };
    for (id, deltas) in &out.textures_delta.set {
        for delta in deltas {
            egui_renderer.update_texture(device, queue, *id, delta);
        }
    }
    let mut enc = device.create_command_encoder(&Default::default());
    let extra = egui_renderer.update_buffers(device, queue, &mut enc, &jobs, &screen);
    {
        let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        egui_renderer.render(&mut pass.forget_lifetime(), &jobs, &screen);
    }
    queue.submit(extra.into_iter().chain([enc.finish()]));
    for id in &out.textures_delta.free {
        egui_renderer.free_texture(id);
    }
}

struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

#[derive(Default)]
struct Mouse {
    left: bool,
    right: bool,
    middle: bool,
    /// Held keys that change what a left drag does: Space pans, Z zooms,
    /// X focuses on the clicked point.
    space: bool,
    zoom: bool,
    focus: bool,
    /// Fractional brush-size change from Ctrl+Alt+drag.
    size_drag: f32,
    pos: Option<(f64, f64)>,
}

/// Events sent to the window from other threads.
pub enum UserEvent {
    /// A tool call from `voxely mcp` and where to send the result.
    Control { tool: String, args: serde_json::Value, reply: std::sync::mpsc::Sender<serde_json::Value> },
}

/// Listen on the control socket and forward each request to the window.
/// Requests are one JSON line `{"tool", "args"}`; replies are one line.
pub fn serve_control(proxy: winit::event_loop::EventLoopProxy<UserEvent>) {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixListener;
    let path = crate::mcp::socket_path();
    // The newest window takes over the socket.
    let _ = std::fs::remove_file(&path);
    let listener = match UnixListener::bind(&path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("control socket {}: {e}", path.display());
            return;
        }
    };
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let proxy = proxy.clone();
            std::thread::spawn(move || {
                let Ok(mut writer) = stream.try_clone() else { return };
                for line in BufReader::new(stream).lines().map_while(Result::ok) {
                    let reply = match serde_json::from_str::<serde_json::Value>(&line) {
                        Ok(req) => {
                            let (tx, rx) = std::sync::mpsc::channel();
                            let tool = req["tool"].as_str().unwrap_or("").to_string();
                            let args = req.get("args").cloned().unwrap_or(serde_json::json!({}));
                            if proxy.send_event(UserEvent::Control { tool, args, reply: tx }).is_err() {
                                break;
                            }
                            rx.recv().unwrap_or_else(|_| serde_json::json!({ "content": [{ "type": "text", "text": "the voxely window closed" }], "isError": true }))
                        }
                        Err(e) => serde_json::json!({ "content": [{ "type": "text", "text": format!("bad request: {e}") }], "isError": true }),
                    };
                    if writeln!(writer, "{reply}").is_err() {
                        break;
                    }
                }
            });
        }
    });
}

pub struct App {
    scene: Scene,
    gfx: Option<Gfx>,
    mouse: Mouse,
    modifiers: ModifiersState,
    panel: ui::PanelState,
    show_panel: bool,
    edit: EditState,
}

impl App {
    pub fn new(scene: Scene, panel: ui::PanelState, source: Option<PathBuf>, save_path: PathBuf) -> Self {
        // Start on the most used color in the model, like picking it up.
        let mut counts = [0usize; 256];
        for &c in &scene.volume.cells {
            counts[c as usize] += 1;
        }
        let color = (1..256).max_by_key(|&i| counts[i]).filter(|&i| counts[i] > 0).unwrap_or(1) as u8;
        App {
            scene,
            gfx: None,
            mouse: Mouse::default(),
            modifiers: ModifiersState::empty(),
            panel,
            show_panel: true,
            edit: EditState::new(color, source, save_path),
        }
    }

    fn init(&mut self, el: &ActiveEventLoop) -> Result<()> {
        let window = Arc::new(
            el.create_window(
                Window::default_attributes()
                    .with_title(format!("voxely — {}", self.edit.name()))
                    .with_inner_size(winit::dpi::LogicalSize::new(1280, 800)),
            )?,
        );
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(
            el.owned_display_handle(),
        )));
        let surface = instance.create_surface(window.clone())?;
        let (adapter, device, queue) = pollster::block_on(gpu::request_device(&instance, Some(&surface)))?;
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .context("surface not supported by adapter")?;
        // We apply gamma ourselves, so prefer a non-sRGB format.
        let caps = surface.get_capabilities(&adapter);
        if let Some(f) = caps.formats.iter().find(|f| !f.is_srgb()) {
            config.format = *f;
        }
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);
        let egui_renderer = egui_wgpu::Renderer::new(&device, config.format, egui_wgpu::RendererOptions::default());
        let renderer = Renderer::new(device, queue, &self.scene, Some(config.format), config.width, config.height);
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        self.gfx = Some(Gfx { window, surface, config, renderer, egui_ctx, egui_state, egui_renderer });
        Ok(())
    }

    fn ray(&self) -> Option<(Vec3, Vec3)> {
        let (g, (x, y)) = (self.gfx.as_ref()?, self.mouse.pos?);
        Some(g.renderer.camera_ray(x as f32, y as f32))
    }

    fn command(&mut self, cmd: Command) {
        if let Some(g) = self.gfx.as_mut() {
            run_command(cmd, &mut g.renderer, &mut self.scene, &mut self.edit);
            g.window.request_redraw();
        }
    }

    /// Upload the volume after an edit.
    fn sync(&mut self) {
        if let Some(g) = self.gfx.as_mut() {
            sync_volume(&mut g.renderer, &mut self.scene, &mut self.edit.editor);
        }
    }

    fn update_hover(&mut self) {
        self.edit.hover = self.ray().and_then(|(ro, rd)| edit::pick(&self.scene.volume, ro, rd));
    }

    fn sel_mode(&self) -> SelMode {
        match (self.modifiers.shift_key(), self.modifiers.alt_key()) {
            (true, true) => SelMode::Remove,
            (true, false) => SelMode::Add,
            _ => SelMode::Replace,
        }
    }

    fn key(&mut self, event: &KeyEvent) {
        let cmd = self.modifiers.control_key() || self.modifiers.super_key();
        let shift = self.modifiers.shift_key();
        // Number pad: load camera slot, Ctrl+number saves it.
        if let PhysicalKey::Code(code) = event.physical_key {
            const PAD: [KeyCode; 10] = [
                KeyCode::Numpad0, KeyCode::Numpad1, KeyCode::Numpad2, KeyCode::Numpad3, KeyCode::Numpad4,
                KeyCode::Numpad5, KeyCode::Numpad6, KeyCode::Numpad7, KeyCode::Numpad8, KeyCode::Numpad9,
            ];
            if let Some(slot) = PAD.iter().position(|&k| k == code) {
                self.camera_slot(slot, cmd);
                return;
            }
        }
        let c = match event.logical_key.as_ref() {
            Key::Character(c) => c.to_ascii_lowercase(),
            _ => String::new(),
        };
        let modeling = self.gfx.as_ref().is_some_and(|g| g.renderer.modeling);
        if cmd && !modeling && matches!(c.as_str(), "c" | "v" | "r") {
            // The renderer's material clipboard.
            match c.as_str() {
                "c" => {
                    let i = self.edit.editor.color as usize;
                    self.edit.material_clipboard = Some(self.scene.materials[i]);
                }
                "v" => self.command(Command::PasteMaterial),
                _ => self.command(Command::ResetMaterial),
            }
            self.redraw_soon();
            return;
        }
        if cmd {
            let command = match c.as_str() {
                "z" if shift => Some(Command::Redo),
                "z" => Some(Command::Undo),
                "y" => Some(Command::Redo),
                "s" if shift => Some(Command::SaveAs),
                "s" => Some(Command::Save),
                "o" => Some(Command::Open),
                "p" => Some(Command::New),
                "a" => Some(Command::SelectAll),
                "d" => Some(Command::SelectNone),
                "i" => Some(Command::SelectInvert),
                "c" => Some(Command::Copy),
                "x" => Some(Command::Cut),
                "v" => Some(Command::Paste),
                "5" => {
                    self.face_camera();
                    None
                }
                "g" | "f" | "b" | "e" => {
                    if let Some(m) = self.gfx.as_mut().and_then(|g| g.renderer.modeler.as_mut()) {
                        let o = &mut m.options;
                        match c.as_str() {
                            "g" => o.grid ^= true,
                            "f" => o.frame ^= true,
                            "e" => o.edge ^= true,
                            _ => o.bounds ^= true,
                        }
                    }
                    None
                }
                _ => None,
            };
            if let Some(command) = command {
                self.command(command);
            }
            self.redraw_soon();
            return;
        }
        let Some(g) = self.gfx.as_mut() else { return };
        let ed = &mut self.edit.editor;
        let r = &mut g.renderer;
        let step = self.scene.volume.dims.max_element() as f32 * 0.05;
        let (fwd, right, _) = r.camera.basis();
        let flat = Vec3::new(fwd.x, fwd.y, 0.0).normalize_or_zero();
        let mut command = None;
        match event.logical_key.as_ref() {
            Key::Named(NamedKey::Delete | NamedKey::Backspace) => command = Some(Command::Op(Op::Clear)),
            Key::Named(NamedKey::F5) => ed.mirror[0] ^= true,
            Key::Named(NamedKey::F6) => ed.mirror[1] ^= true,
            Key::Named(NamedKey::F7) => ed.mirror[2] ^= true,
            Key::Named(NamedKey::F1) => {
                r.modeling ^= true;
                self.edit.enabled = r.modeling;
                r.reset();
            }
            Key::Named(NamedKey::F12) => command = Some(Command::ResetCamera),
            Key::Named(NamedKey::Tab) => self.show_panel ^= true,
            Key::Named(NamedKey::Space) => self.mouse.space = true,
            Key::Named(NamedKey::ArrowLeft) => r.settings.sun_yaw -= 10.0,
            Key::Named(NamedKey::ArrowRight) => r.settings.sun_yaw += 10.0,
            Key::Named(NamedKey::ArrowUp) => r.settings.sun_pitch = (r.settings.sun_pitch + 5.0).min(90.0),
            Key::Named(NamedKey::ArrowDown) => r.settings.sun_pitch = (r.settings.sun_pitch - 5.0).max(-90.0),
            _ => match c.as_str() {
                // MagicaVoxel's model-editor hotkeys.
                "v" => ed.brush = edit::Brush::Voxel,
                "f" => ed.brush = edit::Brush::Face,
                "b" => ed.brush = edit::Brush::Box,
                "l" => ed.brush = edit::Brush::Geo,
                "c" => ed.brush = edit::Brush::Code,
                "p" => ed.brush = edit::Brush::Pattern,
                // Pattern rotation, or voxel brush size.
                "-" | "=" | "+" if ed.brush == edit::Brush::Pattern => ed.rotate_pattern(2, if c == "-" { 3 } else { 1 }),
                "9" | "0" if ed.brush == edit::Brush::Pattern => ed.rotate_pattern(0, if c == "9" { 3 } else { 1 }),
                "-" => ed.opts.size = (ed.opts.size - 1).max(1),
                "=" | "+" => ed.opts.size = (ed.opts.size + 1).min(64),
                "t" => ed.tool = Tool::Attach,
                "r" => ed.tool = Tool::Erase,
                "g" => ed.tool = Tool::Paint,
                "m" => ed.tool = Tool::Marquee,
                "n" => ed.tool = Tool::Region,
                "u" => command = Some(Command::Op(Op::Full)),
                "i" => command = Some(Command::Op(Op::Fill)),
                "1" => ed.color = ed.color.saturating_sub(1).max(1),
                "2" => ed.color = ed.color.saturating_add(1),
                "5" => command = Some(Command::Focus),
                "6" => command = Some(Command::Snapshot),
                "7" => self.edit.cameras[QUICK_SLOT] = Some(r.camera),
                "8" => {
                    if let Some(cam) = self.edit.cameras[QUICK_SLOT] {
                        r.camera = cam;
                    }
                }
                // Camera movement.
                "w" => r.camera.target += flat * step,
                "s" => r.camera.target -= flat * step,
                "a" => r.camera.target -= right * step,
                "d" => r.camera.target += right * step,
                "q" => r.camera.target.z -= step,
                "e" => r.camera.target.z += step,
                "z" => self.mouse.zoom = true,
                "x" => self.mouse.focus = true,
                _ => return,
            },
        }
        // Camera and sun changes need a fresh accumulation.
        g.renderer.reset();
        if let Some(command) = command {
            self.command(command);
        }
        self.update_hover();
        self.redraw_soon();
    }

    fn key_released(&mut self, key: &Key) {
        match key.as_ref() {
            Key::Named(NamedKey::Space) => self.mouse.space = false,
            Key::Character(c) if c.eq_ignore_ascii_case("z") => self.mouse.zoom = false,
            Key::Character(c) if c.eq_ignore_ascii_case("x") => self.mouse.focus = false,
            _ => {}
        }
    }

    fn camera_slot(&mut self, slot: usize, store: bool) {
        let Some(g) = self.gfx.as_mut() else { return };
        if store {
            self.edit.cameras[slot] = Some(g.renderer.camera);
        } else if let Some(cam) = self.edit.cameras[slot] {
            g.renderer.camera = cam;
            g.renderer.reset();
        }
        self.redraw_soon();
    }

    /// Snap the view to the nearest axis [Ctrl+5].
    fn face_camera(&mut self) {
        let Some(g) = self.gfx.as_mut() else { return };
        let cam = &mut g.renderer.camera;
        let quarter = std::f32::consts::FRAC_PI_2;
        cam.yaw = (cam.yaw / quarter).round() * quarter;
        cam.pitch = if cam.pitch.abs() > quarter * 0.5 { 1.55 * cam.pitch.signum() } else { 0.0 };
        g.renderer.reset();
    }

    fn redraw_soon(&self) {
        if let Some(g) = self.gfx.as_ref() {
            g.window.request_redraw();
        }
    }

    fn mouse_moved(&mut self, x: f64, y: f64) {
        let last = self.mouse.pos.replace((x, y));
        let Some(g) = self.gfx.as_mut() else { return };
        let (dx, dy) = last.map(|(lx, ly)| ((x - lx) as f32, (y - ly) as f32)).unwrap_or((0.0, 0.0));
        let cam = &mut g.renderer.camera;
        let editing = self.edit.enabled && !self.mouse.space && !self.mouse.zoom;
        let ctrl_alt = (self.modifiers.control_key() || self.modifiers.super_key()) && self.modifiers.alt_key();
        if let Some((_, b)) = self.edit.marquee.as_mut() {
            *b = Vec2::new(x as f32, y as f32);
        } else if self.mouse.left && ctrl_alt && !self.edit.editor.active() {
            // Ctrl+Alt+drag resizes the voxel brush.
            let o = &mut self.edit.editor.opts;
            self.mouse.size_drag += dx * 0.1;
            let step = self.mouse.size_drag.trunc();
            self.mouse.size_drag -= step;
            o.size = (o.size + step as i32).clamp(1, 64);
        } else if self.mouse.left && self.mouse.zoom {
            cam.dist = (cam.dist * (dy * 0.01).exp()).max(1.0);
            g.renderer.reset();
        } else if self.mouse.right || (self.mouse.left && !editing && !self.edit.editor.active()) && !self.mouse.space {
            let eye = cam.eye();
            if self.edit.free_look {
                // Look around from where the camera is.
                cam.yaw -= dx * 0.003;
                cam.pitch = (cam.pitch + dy * 0.003).clamp(-1.55, 1.55);
                cam.target = eye - (cam.eye() - cam.target);
            } else {
                cam.yaw -= dx * 0.005;
                cam.pitch = (cam.pitch + dy * 0.005).clamp(-1.55, 1.55);
            }
            g.renderer.reset();
        } else if self.mouse.middle || (self.mouse.left && self.mouse.space) {
            let (_, right, up) = cam.basis();
            let k = cam.dist * 0.0015;
            cam.target += (-right * dx + up * dy) * k;
            g.renderer.reset();
        } else if self.mouse.left && self.edit.editor.active() {
            let (ro, rd) = g.renderer.camera_ray(x as f32, y as f32);
            let hit = edit::pick(&self.scene.volume, ro, rd);
            if self.edit.editor.drag(&mut self.scene.volume, ro, rd, hit) {
                sync_volume(&mut g.renderer, &mut self.scene, &mut self.edit.editor);
            }
        }
        self.update_hover();
        self.redraw_soon();
    }

    /// Select every voxel whose center falls inside the marquee.
    fn finish_marquee(&mut self, a: Vec2, b: Vec2) {
        let Some(g) = self.gfx.as_ref() else { return };
        let (lo, hi) = (a.min(b), a.max(b));
        let v = &self.scene.volume;
        let visible_only = self.edit.editor.opts.visible_only;
        let cells: Vec<usize> = (0..v.cells.len())
            .filter(|&i| v.cells[i] != 0)
            .filter(|&i| {
                let p = v.position(i);
                let Some((x, y)) = g.renderer.project(p.as_vec3() + 0.5) else { return false };
                if x < lo.x || x > hi.x || y < lo.y || y > hi.y {
                    return false;
                }
                // Visible: the camera ray through the voxel hits it first.
                !visible_only || {
                    let (ro, rd) = g.renderer.camera_ray(x, y);
                    edit::pick(v, ro, rd).is_some_and(|h| h.cell == p)
                }
            })
            .collect();
        let mode = self.sel_mode();
        self.edit.editor.select(&self.scene.volume, cells, mode);
    }

    fn mouse_button(&mut self, button: MouseButton, down: bool) {
        match button {
            MouseButton::Left => {
                self.mouse.left = down;
                let cmd = self.modifiers.control_key() || self.modifiers.super_key();
                let pos = self.mouse.pos.map(|(x, y)| Vec2::new(x as f32, y as f32));
                if down && self.mouse.focus {
                    // X+click: orbit around the clicked point.
                    self.update_hover();
                    if let (Some(h), Some(g)) = (self.edit.hover, self.gfx.as_mut()) {
                        g.renderer.camera.target = h.cell.as_vec3() + 0.5 + h.normal.as_vec3() * 0.5;
                        g.renderer.reset();
                    }
                } else if down && self.edit.enabled && !self.mouse.space && !self.mouse.zoom {
                    self.update_hover();
                    let hover = self.edit.hover;
                    self.edit.editor.sel_mode = self.sel_mode();
                    let tool = self.edit.editor.tool;
                    let alt = self.modifiers.alt_key();
                    let alt_only = alt && !self.modifiers.shift_key() && !cmd;
                    if cmd && alt {
                        // Ctrl+Alt+drag resizes the brush (see mouse_moved).
                    } else if alt_only {
                        // Alt+click picks a voxel's color with any tool.
                        if let Some(h) = hover.filter(|h| !h.on_ground()) {
                            self.edit.editor.color = self.scene.volume.get(h.cell);
                        }
                    } else if (cmd && self.edit.editor.selected_count() > 0) || tool == Tool::Move {
                        if let Some(hit) = hover {
                            let shift = self.modifiers.shift_key();
                            self.edit.editor.begin_move(&self.scene.volume, hit, shift);
                        }
                    } else if tool == Tool::Marquee && self.edit.editor.opts.marquee == edit::MarqueeMode::Rect {
                        self.edit.marquee = pos.map(|p| (p, p));
                    } else if let Some(hit) = hover {
                        if let Some(g) = self.gfx.as_ref() {
                            prepare_code_brush(&g.renderer, &self.scene, &mut self.edit);
                        }
                        let (ro, rd) = self.ray().unwrap_or((Vec3::ZERO, Vec3::NEG_Z));
                        if self.edit.editor.begin(&mut self.scene.volume, hit, ro, rd) {
                            self.sync();
                        }
                    } else if self.edit.editor.tool == Tool::Region && self.edit.editor.sel_mode == SelMode::Replace {
                        self.edit.editor.select_none();
                    }
                } else if !down {
                    if let Some((a, b)) = self.edit.marquee.take() {
                        self.finish_marquee(a, b);
                    } else if self.edit.editor.active() && self.edit.editor.end(&mut self.scene.volume) {
                        self.sync();
                    }
                }
                self.update_hover();
                self.redraw_soon();
            }
            MouseButton::Right => self.mouse.right = down,
            MouseButton::Middle => self.mouse.middle = down,
            _ => {}
        }
    }

    fn redraw(&mut self) {
        let Some(g) = self.gfx.as_mut() else { return };
        if self.edit.editor.sel_dirty {
            g.renderer.upload_selection(&self.scene.volume, self.edit.editor.selection());
            self.edit.editor.sel_dirty = false;
        }
        let frame = match g.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                g.surface.configure(&g.renderer.device, &g.config);
                g.window.request_redraw();
                return;
            }
            _ => {
                g.window.request_redraw();
                return;
            }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        // Keep refining until the image converges, then idle.
        let samples = if g.renderer.converging() { 1 } else { 0 };
        g.renderer.render_to_view(&view, (g.config.width, g.config.height), samples);

        let raw = g.egui_state.take_egui_input(&g.window);
        let (scene, ed, panel, show) = (&mut self.scene, &mut self.edit, &self.panel, self.show_panel);
        let renderer = &mut g.renderer;
        let mut actions = ui::Actions::default();
        let out = g.egui_ctx.run_ui(raw, |ui| actions = build_ui(ui, renderer, scene, ed, panel, show));
        g.egui_state.handle_platform_output(&g.window, out.platform_output.clone());
        paint_egui(&g.renderer.device, &g.renderer.queue, &mut g.egui_renderer, &g.egui_ctx, &view, [g.config.width, g.config.height], out);
        g.renderer.queue.present(frame);
        if let Some(rect) = actions.viewport {
            let ppp = g.egui_ctx.pixels_per_point();
            fit_viewport(&mut g.renderer, rect, ppp, (g.config.width, g.config.height));
        }
        let converging = g.renderer.converging();
        for cmd in actions.commands {
            self.command(cmd);
        }

        let Some(g) = self.gfx.as_ref() else { return };
        let ed = &self.edit.editor;
        let name = self.edit.name();
        let title = if g.renderer.modeling {
            let sel = match ed.selected_count() {
                0 => String::new(),
                n => format!(" — {n} selected"),
            };
            format!("voxely — {name} — Model — {:?} {:?}{sel}", ed.tool, ed.brush)
        } else {
            format!("voxely — {name} — Render — {} spp", g.renderer.frame)
        };
        g.window.set_title(&title);
        if converging {
            g.window.request_redraw();
        }
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn user_event(&mut self, _el: &ActiveEventLoop, event: UserEvent) {
        let UserEvent::Control { tool, args, reply } = event;
        let result = match self.gfx.as_mut() {
            Some(g) => {
                let mut ctx = crate::control::Ctx { r: &mut g.renderer, scene: &mut self.scene, ed: &mut self.edit, live: true };
                let result = crate::control::call(&mut ctx, &tool, &args);
                g.window.request_redraw();
                result
            }
            None => serde_json::json!({ "content": [{ "type": "text", "text": "the voxely window is still starting" }], "isError": true }),
        };
        let _ = reply.send(result);
    }

    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_none() {
            if let Err(e) = self.init(el) {
                eprintln!("failed to start renderer: {e:#}");
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let Some(g) = self.gfx.as_mut() {
            let resp = g.egui_state.on_window_event(&g.window, &event);
            if resp.repaint {
                g.window.request_redraw();
            }
            let pointer_on_ui = g.egui_ctx.egui_wants_pointer_input();
            let press = matches!(event, WindowEvent::MouseInput { state: ElementState::Pressed, .. } | WindowEvent::MouseWheel { .. });
            let typing = g.egui_ctx.egui_wants_keyboard_input() && matches!(event, WindowEvent::KeyboardInput { .. });
            if (resp.consumed && !matches!(event, WindowEvent::CursorMoved { .. } | WindowEvent::MouseInput { .. })) || (pointer_on_ui && press) || typing {
                return;
            }
        }
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(size) => {
                if let Some(g) = self.gfx.as_mut() {
                    g.config.width = size.width.max(1);
                    g.config.height = size.height.max(1);
                    g.surface.configure(&g.renderer.device, &g.config);
                    g.renderer.set_viewport(0, 0, g.config.width, g.config.height);
                    g.window.request_redraw();
                }
            }
            WindowEvent::ModifiersChanged(m) => self.modifiers = m.state(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    self.key(&event);
                } else {
                    self.key_released(&event.logical_key);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => self.mouse_button(button, state == ElementState::Pressed),
            WindowEvent::CursorMoved { position, .. } => self.mouse_moved(position.x, position.y),
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(g) = self.gfx.as_mut() {
                    let d = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y * 0.1,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.005,
                    };
                    g.renderer.camera.dist = (g.renderer.camera.dist * (1.0 - d)).max(1.0);
                    g.renderer.reset();
                    g.window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }
}
