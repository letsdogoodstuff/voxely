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
use crate::outline::TreeOp;
use crate::scene::{Mask, MaterialKind, Mode, Projection, Scene};
use crate::ui::{self, Command, Op};
use crate::world::{Rot, ROOT};

/// Size of a new project, like MagicaVoxel's default.
const NEW_SIZE: i32 = 40;
/// Size of a new model made in World mode.
const NEW_PART: i32 = 20;
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
    /// The user's projects: `.vox` files in the workspace folder.
    pub projects: Vec<PathBuf>,
    /// MagicaVoxel's sample models (its `vox` folder), when known.
    pub samples: Vec<PathBuf>,
    /// When `projects` was last read from disk.
    scanned: std::time::Instant,
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
    /// Result of the last export or import, shown under the Export buttons.
    pub export_status: Option<String>,
    /// MCP agent control of this window.
    pub agent: crate::agents::AgentState,
    pub prefs: crate::prefs::Prefs,
    pub prefs_open: bool,
    /// The interface scale in use (the chosen one, unless the window is
    /// too narrow for it).
    pub ui_scale: f32,
    pub transform_text: ui::TransformText,
    pub palette: ui::PaletteState,
    /// Canvas size being typed into the Canvas panel.
    pub canvas: IVec3,
    canvas_for: IVec3,
    /// Marquee drag corners in window pixels.
    pub marquee: Option<(Vec2, Vec2)>,
    pub outline: crate::outline::OutlineState,
    /// World mode: parts being dragged.
    pub world_drag: Option<WorldDrag>,
    /// World mode: last click, for double-click to edit.
    last_click: Option<(std::time::Instant, usize)>,
    /// World mode: nodes copied with Ctrl+C.
    pub part_clipboard: Vec<usize>,
    pub timeline: crate::timeline::TimelineState,
    cameras: [Option<Camera>; QUICK_SLOT + 1],
    /// State for agent tools (symmetry, checkpoints, last edit, ...).
    pub tools: crate::control::ToolState,
    /// A file picked for import, while its Import window is open.
    pub import: Option<crate::import::window::Pending>,
    /// The reference picture shown behind the model.
    pub backdrop: crate::backdrop::Backdrop,
    /// A project waiting for its delete to be confirmed.
    pub confirm_delete: Option<PathBuf>,
    /// What the last action did, or why it failed (the status bar).
    pub notice: Option<Notice>,
    /// Unsaved changes since the last save or open.
    pub dirty: Dirty,
    /// A command that would drop unsaved changes, waiting for Save /
    /// Don't save / Cancel.
    pub unsaved_ask: Option<Command>,
    /// Don't save was chosen: run the next guarded command anyway.
    discard: bool,
    /// The window should close.
    pub quit: bool,
    /// Last folder each kind of file dialog was used in.
    dialog_dirs: std::collections::HashMap<&'static str, PathBuf>,
}

/// A status-bar message. Errors stay until dismissed or the next action;
/// other messages fade after a few seconds.
pub struct Notice {
    pub text: String,
    pub error: bool,
    pub at: std::time::Instant,
}

/// What unsaved-change tracking compares: the undo histories, the colors
/// and materials, and the render settings that are saved with the file.
#[derive(Clone, PartialEq)]
pub struct Sig {
    history: [u64; 6],
    look: u64,
    settings: crate::scene::RenderSettings,
}

impl Sig {
    pub fn of(settings: &crate::scene::RenderSettings, scene: &Scene, ed: &EditState) -> Sig {
        use std::hash::{Hash, Hasher};
        let (tree_back, tree_fwd) = scene.tree_depth();
        let (undo, redo) = ed.editor.history_lens();
        let history = [scene.tree_steps(), tree_back as u64, tree_fwd as u64, undo as u64, redo as u64, ed.palette.history.len() as u64];
        let mut h = std::hash::DefaultHasher::new();
        scene.palette.hash(&mut h);
        bytemuck::cast_slice::<_, u8>(&scene.materials).hash(&mut h);
        scene.notes.hash(&mut h);
        // View-only settings don't make the project unsaved.
        let mut settings = settings.clone();
        settings.projection = Projection::Perspective;
        (settings.denoise, settings.denoise_view) = (false, Default::default());
        Sig { history, look: h.finish(), settings }
    }
}

/// Unsaved-change tracking: the scene as last seen, and how many changes
/// were seen in all and as of the last save. Undoing back to the saved
/// state still counts as a change.
#[derive(Default)]
pub struct Dirty {
    last: Option<Sig>,
    changes: u64,
    saved: u64,
}

impl Dirty {
    pub fn is_dirty(&self) -> bool {
        self.changes != self.saved
    }

    /// Look at the scene now. Returns true if its undo history moved (an
    /// edit, an undo or a redo) since the last look.
    pub fn see(&mut self, sig: Sig) -> bool {
        let Some(last) = &self.last else {
            self.last = Some(sig);
            return false;
        };
        if *last == sig {
            return false;
        }
        let edit = last.history != sig.history;
        self.changes += 1;
        self.last = Some(sig);
        edit
    }

    /// The scene was just saved, opened or made.
    pub fn clean(&mut self, sig: Sig) {
        self.last = Some(sig);
        self.saved = self.changes;
    }

    /// Changes not on disk (e.g. a new project from an import).
    pub fn touch(&mut self) {
        self.changes += 1;
    }
}

/// Parts dragged across the plane of the face first clicked.
pub struct WorldDrag {
    ids: Vec<usize>,
    /// World cell first clicked and the plane's axis.
    anchor: IVec3,
    axis: usize,
    moved: IVec3,
}

impl EditState {
    pub fn new(color: u8, source: Option<PathBuf>, save_path: PathBuf) -> Self {
        // A model outside MagicaVoxel's folders (a workspace project, say)
        // has none of its assets around it; list the install's instead.
        let install = magicavoxel_dir().map(|d| d.join("vox").join("_.vox"));
        let or_install = |found: Vec<PathBuf>, list: fn(Option<&Path>) -> Vec<PathBuf>| if found.is_empty() { list(install.as_deref()) } else { found };
        let samples = or_install(list_samples(source.as_deref()), list_samples);
        let palettes = or_install(list_palettes(source.as_deref()), list_palettes);
        EditState {
            samples,
            scanned: std::time::Instant::now(),
            palettes,
            project_tab: ui::ProjectTab::default(),
            free_look: false,
            light_tab: ui::LightTab::default(),
            matter_tab: ui::MatterTab::default(),
            material_clipboard: None,
            image: ui::ImageSettings::default(),
            export_status: None,
            agent: crate::agents::AgentState::default(),
            prefs: crate::prefs::Prefs::default(),
            prefs_open: false,
            ui_scale: 1.0,
            transform_text: ui::TransformText::default(),
            palette: ui::PaletteState::default(),
            shaders: ui::ShaderState { files: or_install(list_shaders(source.as_deref()), list_shaders), current: None, values: Vec::new(), error: None },
            editor: Editor::new(color),
            hover: None,
            enabled: true,
            source,
            save_path,
            projects: list_vox(&workspace_dir()),
            canvas: IVec3::ZERO,
            canvas_for: IVec3::ZERO,
            marquee: None,
            outline: Default::default(),
            world_drag: None,
            last_click: None,
            part_clipboard: Vec::new(),
            timeline: Default::default(),
            cameras: [None; QUICK_SLOT + 1],
            tools: Default::default(),
            import: None,
            backdrop: Default::default(),
            confirm_delete: None,
            notice: None,
            dirty: Dirty::default(),
            unsaved_ask: None,
            discard: false,
            quit: false,
            dialog_dirs: Default::default(),
        }
    }

    /// Show what an action did in the status bar (and the terminal).
    pub fn say(&mut self, text: impl Into<String>) {
        let text = text.into();
        eprintln!("{text}");
        self.notice = Some(Notice { text, error: false, at: std::time::Instant::now() });
    }

    /// Show why an action failed; it stays until dismissed or the next
    /// action.
    pub fn fail(&mut self, text: impl Into<String>) {
        let text = text.into();
        eprintln!("{text}");
        self.notice = Some(Notice { text, error: true, at: std::time::Instant::now() });
    }

    /// Where a `kind` file dialog starts: where it was last used, else
    /// `fallback`.
    fn dialog_dir(&self, kind: &'static str, fallback: PathBuf) -> PathBuf {
        self.dialog_dirs.get(kind).filter(|d| d.is_dir()).cloned().unwrap_or(fallback)
    }

    /// Remember the folder of a file picked in a `kind` dialog.
    fn picked(&mut self, kind: &'static str, path: &Path) {
        let dir = if path.is_dir() { Some(path) } else { path.parent() };
        if let Some(d) = dir {
            self.dialog_dirs.insert(kind, d.to_path_buf());
        }
    }

    pub(crate) fn name(&self) -> String {
        self.save_path.file_stem().map_or("project".into(), |n| n.to_string_lossy().into_owned())
    }

    /// Re-read the workspace's project list.
    pub fn rescan(&mut self) {
        self.projects = list_vox(&workspace_dir());
        self.scanned = std::time::Instant::now();
    }

    /// Re-read the project list if it's more than a second old, so
    /// projects saved elsewhere (e.g. by an agent) show up.
    fn rescan_if_stale(&mut self) {
        if self.scanned.elapsed() > std::time::Duration::from_secs(1) {
            self.rescan();
        }
    }
}

/// Where projects are saved and listed: `$VOXELY_WORKSPACE`, or
/// `~/Documents/voxely`.
pub fn workspace_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("VOXELY_WORKSPACE") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join("Documents").join("voxely")
}

/// MagicaVoxel's install folder (its models, palettes, shaders and
/// environment images): `$VOXELY_ASSETS`, or the unzipped download.
pub fn magicavoxel_dir() -> Option<PathBuf> {
    std::env::var_os("VOXELY_ASSETS").map(PathBuf::from).or_else(|| {
        let home = std::env::var_os("HOME")?;
        let guess = PathBuf::from(home).join("Downloads/MagicaVoxel-0.99.6.2-macos-10.15");
        guess.join("vox").is_dir().then_some(guess)
    })
}

/// The workspace file for a project name.
pub fn project_path(name: &str) -> PathBuf {
    workspace_dir().join(format!("{name}.vox"))
}

/// Sample models: the `.vox` files next to `source`, unless that is the
/// workspace itself.
fn list_samples(source: Option<&Path>) -> Vec<PathBuf> {
    match source.and_then(Path::parent) {
        Some(dir) if !same_dir(dir, &workspace_dir()) => list_vox(dir),
        _ => Vec::new(),
    }
}

fn same_dir(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// Sorted `.vox` files in `dir`.
pub fn list_vox(dir: &Path) -> Vec<PathBuf> {
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

/// A new image path in the workspace's export folder, e.g.
/// `export/castle-render-1761300000.png`.
fn image_path(project: &str, kind: &str) -> PathBuf {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    workspace_dir().join("export").join(format!("{project}-{kind}-{secs}.png"))
}

/// Render the final image offscreen at the Image size and save it to the
/// export folder.
fn render_final(r: &Renderer, scene: &Scene, image: ui::ImageSettings, project: &str) -> String {
    let mut out = Renderer::new(r.device.clone(), r.queue.clone(), scene, None, image.width, image.height);
    out.settings = r.settings.clone();
    out.set_camera(r.camera);
    let path = image_path(project, "render");
    match out.render_image(image.spp).and_then(|px| crate::save_png(&path, image.width, image.height, &px)) {
        Ok(()) => format!("Rendered {} ({} spp)", path.display(), image.spp),
        Err(e) => format!("Render failed: {e:#}"),
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

/// Where Ctrl+S writes a project by default. Projects already in the
/// workspace save in place; anything else (a sample, a file elsewhere, a
/// new project) gets a copy in the workspace, so originals are never
/// overwritten by saving's flattening.
pub fn default_save_path(source: Option<&Path>) -> PathBuf {
    if let Some(src) = source {
        if src.parent().is_some_and(|d| same_dir(d, &workspace_dir())) {
            return src.to_path_buf();
        }
    }
    let stem = source.and_then(Path::file_stem).map_or("project".into(), |n| n.to_string_lossy().into_owned());
    if source.is_some() {
        return project_path(&stem);
    }
    // A new project: the first free "project", "project-2", ...
    (1..).map(|k| if k == 1 { project_path(&stem) } else { project_path(&format!("{stem}-{k}")) }).find(|p| !p.exists()).unwrap()
}

/// Default file name for an export, e.g. `castle.obj`. Formats that share
/// an extension get a suffix so they don't overwrite each other.
pub fn export_file_name(project: &str, format: crate::export::Format) -> String {
    use crate::export::Format;
    match format {
        Format::Obj | Format::Xraw | Format::Qb | Format::Vox | Format::Ply => format!("{project}.{}", format.extension()),
        _ => format!("{project}-{}.{}", format.name(), format.extension()),
    }
}

pub fn export_summary(files: &[PathBuf]) -> String {
    let names: Vec<String> = files.iter().filter_map(|f| f.file_name()).map(|n| n.to_string_lossy().into_owned()).collect();
    match files.first().and_then(|f| f.parent()) {
        Some(dir) => format!("Exported {} to {}", names.join(", "), dir.display()),
        None => "Exported".into(),
    }
}

/// Save to `path`, with the renderer's current settings, and make it the
/// current project.
pub fn save_project(r: &Renderer, scene: &mut Scene, ed: &mut EditState, path: PathBuf) -> Result<PathBuf> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    scene.settings = r.settings.clone();
    scene.save(&path)?;
    ed.save_path = path.clone();
    ed.source = Some(path.clone());
    ed.rescan();
    ed.tools.inspect.mark_saved(&path);
    ed.dirty.clean(Sig::of(&r.settings, scene, ed));
    Ok(path)
}

/// Push voxel edits to the GPU, refreshing emissive faces if any exist.
pub fn sync_volume(r: &mut Renderer, scene: &mut Scene, editor: &mut Editor) {
    let highlight = scene.highlight();
    r.upload_scene(scene, highlight.as_ref().or(editor.selection()));
    editor.sel_dirty = false;
    let any_emit = scene.materials.iter().any(|m| m.kind == MaterialKind::Emit as u32 && m.emit > 0.0);
    if any_emit || !scene.lights.is_empty() {
        scene.rebuild_lights();
        r.set_lights(&scene.lights, scene.light_power);
    }
}

/// Refresh the modeler's highlight: selected voxels, or picked parts.
pub fn sync_selection(r: &mut Renderer, scene: &Scene, editor: &mut Editor) {
    let highlight = scene.highlight();
    r.upload_selection(&scene.volume, highlight.as_ref().or(editor.selection()));
    editor.sel_dirty = false;
}

/// Run a scene-tree change. Returns whether the view needs uploading.
pub(crate) fn tree_op(op: TreeOp, scene: &mut Scene, ed: &mut EditState) -> bool {
    let picked = scene.picked.clone();
    let w = &scene.world;
    let top: Vec<usize> = w.node(ROOT).map(|n| n.children().to_vec()).unwrap_or_default();
    match op {
        TreeOp::Pick(id, add) => {
            if !add {
                scene.picked = vec![id];
            } else if let Some(k) = picked.iter().position(|&p| p == id) {
                scene.picked.remove(k);
            } else {
                scene.picked.push(id);
            }
            return true;
        }
        TreeOp::PickAll => {
            scene.picked = top;
            return true;
        }
        TreeOp::PickNone => {
            scene.picked.clear();
            return true;
        }
        TreeOp::PickInvert => {
            scene.picked = top.into_iter().filter(|t| !picked.contains(t)).collect();
            return true;
        }
        _ => {}
    }
    keep_history(scene, &mut ed.editor);
    let w = &scene.world;
    match op {
        TreeOp::New => {
            let near = picked.first().copied().unwrap_or(ROOT);
            // Beside the picked one (or everything), on the ground.
            let at = picked.first().and_then(|&p| w.bounds_all(p)).or_else(|| w.reach()).map_or(IVec3::ZERO, |(lo, size)| IVec3::new(lo.x + size.x + 2, lo.y, 0));
            let parent = w.node(near).and_then(|n| n.parent).unwrap_or(ROOT);
            let id = scene.change(|w| w.add_model(parent, "", crate::scene::Volume::new(IVec3::splat(NEW_PART)), at));
            scene.picked = vec![id];
        }
        TreeOp::Delete => {
            let mut left = scene.world.clone();
            picked.iter().for_each(|&p| {
                left.delete(p);
            });
            if picked.is_empty() || left.models().is_empty() {
                return false;
            }
            scene.change(|w| picked.iter().for_each(|&p| {
                w.delete(p);
            }));
        }
        TreeOp::Duplicate => {
            let copies = scene.change(|w| {
                picked
                    .iter()
                    .filter_map(|&p| {
                        let width = w.bounds_all(p)?.1.x;
                        let c = w.duplicate(p)?;
                        // Beside the original, so it's easy to see.
                        w.shift(c, IVec3::new(width + 1, 0, 0));
                        Some(c)
                    })
                    .collect::<Vec<_>>()
            });
            scene.picked = copies;
        }
        TreeOp::Group if !picked.is_empty() => {
            if let Some(g) = scene.change(|w| w.group(&picked)) {
                scene.picked = vec![g];
            }
        }
        TreeOp::Ungroup => {
            let groups: Vec<usize> = picked.iter().copied().filter(|&p| w.node(p).is_some_and(|n| n.is_group())).collect();
            if groups.is_empty() {
                return false;
            }
            let children: Vec<usize> = groups.iter().flat_map(|&g| w.node(g).map(|n| n.children().to_vec()).unwrap_or_default()).collect();
            scene.change(|w| groups.iter().for_each(|&g| {
                w.ungroup(g);
            }));
            scene.picked = children;
        }
        TreeOp::Hide(hide) => {
            scene.change(|w| {
                if hide {
                    for &p in &picked {
                        if let Some(n) = w.node_mut(p) {
                            n.hidden = true;
                        }
                    }
                } else {
                    w.nodes.iter_mut().flatten().for_each(|n| n.hidden = false);
                    w.layers.iter_mut().for_each(|l| l.hidden = false);
                }
            });
        }
        TreeOp::ToggleHidden(id) => scene.change(|w| {
            if let Some(n) = w.node_mut(id) {
                n.hidden ^= true;
            }
        }),
        TreeOp::Move(d) => scene.change(|w| picked.iter().for_each(|&p| w.translate(p, d))),
        TreeOp::Turn(axis, q) => scene.change(|w| picked.iter().for_each(|&p| w.reorient(p, Rot::turn(axis, q)))),
        TreeOp::Flip(axis) => scene.change(|w| picked.iter().for_each(|&p| w.reorient(p, Rot::flip(axis)))),
        TreeOp::Attach(id, to, socket) => {
            if !scene.change(|w| w.attach_to(id, to, socket.as_deref(), None)) {
                return false;
            }
        }
        TreeOp::Detach(id) => {
            scene.change(|w| w.detach(id));
        }
        TreeOp::SetAnim(a) => {
            let frame = scene.world.frame;
            scene.change_quiet(|w| {
                w.anim = a;
                w.set_frame(frame);
            });
        }
        TreeOp::Key(id) => scene.change_quiet(|w| w.key(id)),
        TreeOp::DeleteKey(id) => {
            let frame = scene.world.frame;
            scene.change_quiet(|w| w.remove_key(id, frame));
        }
        TreeOp::SplitFrame(id) => {
            scene.change_quiet(|w| w.split_frame(id));
        }
        TreeOp::MergeFrame(id) => {
            scene.change_quiet(|w| w.merge_frame(id));
        }
        TreeOp::AddRig(quad) => {
            let at = w.reach().map_or(IVec3::ZERO, |(lo, size)| IVec3::new(lo.x + size.x + 3, lo.y, 0));
            let colors = rig_colors(&scene.palette);
            let root = scene.change_quiet(|w| if quad { crate::rig::quadruped(w, at, 1, "", colors) } else { crate::rig::humanoid(w, at, 1, "", colors) });
            scene.picked = vec![root];
        }
        TreeOp::Motion(name, frames) => {
            let rig = scene.picked.first().copied();
            let on = crate::timeline::rig_label(&scene.world, rig, name.starts_with("quad"));
            // Tried on a copy first, so a failure changes nothing.
            let res = crate::rig::apply_motion(&mut scene.world.clone(), rig, &name, frames, 1.0).and_then(|_| scene.change_quiet(|w| crate::rig::apply_motion(w, rig, &name, frames, 1.0)));
            match res {
                Ok(a) if a.bones.is_empty() => {
                    ed.fail(format!("{name}: no rig parts found on {on} (name parts hips, torso, head, upper_arm_l ...)"));
                }
                Ok(a) => {
                    ed.timeline.open = true;
                    let parts: Vec<&str> = a.bones.iter().map(|(_, l)| l.as_str()).collect();
                    let mut msg = format!("{name} on {on}: keyed {} parts ({})", parts.len(), parts.join(", "));
                    if !a.missing.is_empty() {
                        let missing: Vec<&str> = a.missing.iter().map(|b| crate::rig::bone_name(*b)).collect();
                        msg.push_str(&format!("; no part for {}", missing.join(", ")));
                    }
                    ed.say(msg);
                }
                Err(e) => ed.fail(e),
            }
        }
        TreeOp::Extras(id, x, begin) => {
            let set = |w: &mut crate::world::World| {
                if let Some(n) = w.node_mut(id) {
                    n.pivot = x.pivot;
                    n.sockets = x.sockets.clone();
                }
                w.set_tilt_world(id, x.tilt);
            };
            if begin {
                scene.change(set);
            } else {
                scene.apply(set);
            }
        }
        TreeOp::Rename(id, name) => {
            // Names can't be empty; a taken one gets a number.
            if name.trim().is_empty() {
                return false;
            }
            scene.change(|w| {
                let name = w.unique_name(&name, Some(id));
                if let Some(n) = w.node_mut(id) {
                    n.name = name;
                }
            })
        }
        _ => return false,
    }
    ed.editor.reset();
    true
}

/// Before a change that resets the voxel editor (tree changes, switching
/// part or frame): move this edit session's voxel edits into the tree's
/// history, one undo step each, so none get merged.
pub(crate) fn keep_history(scene: &mut Scene, editor: &mut Editor) {
    if scene.editing().is_some() && editor.can_undo() {
        let past = editor.take_history(&scene.volume, Scene::KEPT_STEPS);
        scene.keep_steps(past);
    }
}

/// Undo (or redo) one step: voxel edits in this edit session first, then
/// whole-part changes.
pub(crate) fn history_step(scene: &mut Scene, ed: &mut EditState, forward: bool) -> bool {
    let e = &mut ed.editor;
    if scene.editing().is_some() && (if forward { e.can_redo() } else { e.can_undo() }) {
        return if forward { e.redo(&mut scene.volume) } else { e.undo(&mut scene.volume) };
    }
    let done = scene.undo_tree(forward);
    if done {
        ed.editor.reset();
    }
    done
}

/// Palette colors closest to a rig's clothes, skin and dark parts.
pub(crate) fn rig_colors(palette: &[[u8; 4]]) -> crate::rig::Colors {
    let near = |c: [i32; 3]| (1..256).min_by_key(|&i| (0..3).map(|k| (palette[i][k] as i32 - c[k]).pow(2)).sum::<i32>()).unwrap_or(1) as u8;
    (near([90, 120, 196]), near([232, 195, 158]), near([60, 60, 80]))
}

/// Switch between editing a model and the World view.
pub(crate) fn set_mode(scene: &mut Scene, ed: &mut EditState, edit: Option<usize>) -> bool {
    keep_history(scene, &mut ed.editor);
    let ok = match edit {
        Some(id) => scene.edit(id),
        None => {
            scene.world_mode();
            true
        }
    };
    ed.editor.reset();
    ed.world_drag = None;
    ok
}

/// Save the path-traced view as it is to the export folder.
pub fn snapshot(r: &Renderer, project: &str) -> String {
    let (w, h) = r.size();
    let path = image_path(project, "snapshot");
    match r.read_pixels().and_then(|px| crate::save_png(&path, w, h, &px)) {
        Ok(()) => format!("Saved {}", path.display()),
        Err(e) => format!("Snapshot failed: {e:#}"),
    }
}

/// Bounding box of solid cells (selected ones, if a selection is given).
fn solid_bounds(scene: &Scene, sel: Option<&Mask>) -> Option<(Vec3, Vec3)> {
    let v = &scene.volume;
    let mut b: Option<(IVec3, IVec3)> = None;
    for (i, _) in v.cells.filled() {
        if sel.is_none_or(|s| s[i]) {
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
    let samples = list_samples(Some(model));
    if !samples.is_empty() {
        ed.samples = samples;
    }
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

/// Import `path` read `how`: as a new project (named after the file), or
/// as new parts of this scene (one undo step). Returns what came in.
pub(crate) fn import_file(path: &Path, how: &crate::import::How, new_project: bool, add_colors: bool, r: &mut Renderer, scene: &mut Scene, ed: &mut EditState) -> Result<String> {
    let models = crate::import::read(path, how)?;
    let stem = path.file_stem().map_or("import".into(), |n| n.to_string_lossy().into_owned());
    let what = crate::import::summary(&models);
    if new_project {
        let ibl = r.settings.ibl_path.clone();
        *scene = crate::import::new_scene(&models)?;
        scene.settings.ibl_path = ibl;
        ed.source = None;
        ed.save_path = (1..).map(|k| if k == 1 { project_path(&stem) } else { project_path(&format!("{stem}-{k}")) }).find(|p| !p.exists()).unwrap();
        loaded(r, scene, ed);
        // Not on disk yet.
        ed.dirty.touch();
        Ok(format!("Imported {} as {} ({what}); Ctrl+S saves it", path.display(), ed.name()))
    } else {
        keep_history(scene, &mut ed.editor);
        let (top, written) = crate::import::add_to_scene(scene, &models, &stem, add_colors)?;
        ed.editor.reset();
        ed.world_drag = None;
        scene.settings = r.settings.clone();
        scene.rebuild_lights();
        r.load_scene(scene);
        let colors = if written.is_empty() { String::new() } else { format!("; {} colors added to the palette", written.len()) };
        Ok(format!("Imported {} as `{}` ({what}{colors})", path.display(), scene.world.label(top)))
    }
}

/// Show `path` as the reference picture (None: no picture).
fn set_backdrop(r: &mut Renderer, ed: &mut EditState, path: Option<PathBuf>) {
    let picture = match path.as_deref().map(crate::backdrop::load).transpose() {
        Ok(p) => p,
        Err(e) => {
            ed.export_status = Some(format!("Reference: {e:#}"));
            None
        }
    };
    if let Some(m) = r.modeler.as_mut() {
        m.set_reference(&r.device, &r.queue, picture.as_ref().map(|(w, h, px)| (*w, *h, px.as_slice())));
    }
    let b = &mut ed.backdrop;
    match (picture, path) {
        (Some((w, h, _)), Some(p)) => {
            (b.size, b.path, b.show, b.open) = ((w, h), Some(p), true, true);
        }
        _ => (b.size, b.path) = ((0, 0), None),
    }
}

/// Ask before deleting `ed.confirm_delete`.
fn confirm_delete(ctx: &egui::Context, ed: &mut EditState, cmds: &mut Vec<Command>) {
    let Some(path) = ed.confirm_delete.clone() else { return };
    let name = path.file_stem().map_or(String::new(), |n| n.to_string_lossy().into_owned());
    let mut keep = true;
    egui::Modal::new(egui::Id::unique("confirm_delete")).show(ctx, |ui| {
        ui.set_width(300.0);
        ui.heading(format!("Delete \u{201c}{name}\u{201d}?"));
        ui.add_space(4.0);
        ui.label(format!("{} and its saved references move to the Trash.", path.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned())));
        if ed.source.as_deref() == Some(path.as_path()) {
            ui.weak("It's the open project: it stays open here, and saving writes it again.");
        }
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            let delete = egui::Button::new(egui::RichText::new("Move to Trash").color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(170, 40, 40));
            if ui.add(delete).clicked() {
                cmds.push(Command::DeleteProjectConfirmed(path.clone()));
            }
            if ui.button("Cancel").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                keep = false;
            }
        });
    });
    if !keep {
        ed.confirm_delete = None;
    }
}

/// Ask whether to save before a command that drops unsaved changes.
fn confirm_unsaved(ctx: &egui::Context, ed: &EditState, cmds: &mut Vec<Command>) {
    let Some(next) = &ed.unsaved_ask else { return };
    let what = match next {
        Command::Quit => "closing",
        Command::New => "starting a new project",
        Command::Import(_) => "importing",
        _ => "opening another project",
    };
    egui::Modal::new(egui::Id::unique("confirm_unsaved")).show(ctx, |ui| {
        ui.set_width(320.0);
        ui.heading(format!("Save changes to \u{201c}{}\u{201d}?", ed.name()));
        ui.add_space(4.0);
        ui.label(format!("Your changes will be lost if you don't save them before {what}."));
        ui.weak(format!("Save writes {}", ed.save_path.display()));
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button("Save").clicked() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                cmds.push(Command::Unsaved(ui::UnsavedChoice::Save));
            }
            if ui.button("Don't save").clicked() {
                cmds.push(Command::Unsaved(ui::UnsavedChoice::Discard));
            }
            if ui.button("Cancel").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                cmds.push(Command::Unsaved(ui::UnsavedChoice::Cancel));
            }
        });
    });
}

/// Move a project (and its saved references) to the Trash. The scene
/// stays open if it's the current one; saving writes it again.
fn delete_project(path: &Path, ed: &mut EditState) -> Result<String> {
    anyhow::ensure!(path.parent().is_some_and(|d| same_dir(d, &workspace_dir())), "only projects in the workspace can be deleted");
    let name = path.file_stem().map_or(String::new(), |n| n.to_string_lossy().into_owned());
    trash::delete(path).with_context(|| format!("moving {} to the Trash", path.display()))?;
    let refs = workspace_dir().join("refs").join(&name);
    if refs.is_dir() {
        let _ = trash::delete(&refs);
    }
    ed.rescan();
    let current = ed.source.as_deref() == Some(path);
    if current {
        ed.source = None;
    }
    Ok(if current { format!("Moved {name} to the Trash; it's still open here, and saving writes it again") } else { format!("Moved {name} to the Trash") })
}

/// Show a newly opened or created scene.
pub(crate) fn loaded(r: &mut Renderer, scene: &mut Scene, ed: &mut EditState) {
    ed.editor.reset();
    // Palette undo and remembered colors belong to the old scene.
    ed.palette.history.clear();
    ed.palette.look = vec![None; 256];
    ed.palette.previewing = None;
    ed.hover = None;
    ed.tools.build.scene_changed();
    r.load_scene(scene);
    ed.editor.sel_dirty = false;
    ed.timeline.play(false);
    ed.timeline.open |= scene.world.animated();
    ed.dirty.clean(Sig::of(&r.settings, scene, ed));
    eprintln!("Ctrl+S saves to {}", ed.save_path.display());
}

/// The project's folder (the workspace for a new one).
fn dialog_dir(ed: &EditState) -> PathBuf {
    ed.source.as_deref().and_then(Path::parent).filter(|d| d.is_dir()).map_or_else(workspace_dir, Path::to_path_buf)
}

/// The window file dialogs belong to.
static DIALOG_PARENT: std::sync::OnceLock<Arc<Window>> = std::sync::OnceLock::new();

/// A file dialog attached to the window (a sheet on macOS), so it can't
/// open behind it.
pub fn file_dialog() -> rfd::FileDialog {
    let d = rfd::FileDialog::new();
    match DIALOG_PARENT.get() {
        Some(w) => d.set_parent(w.as_ref()),
        None => d,
    }
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

/// Save to `path`, saying so (or why not) in the status bar. Returns
/// whether it saved.
fn save_and_say(r: &Renderer, scene: &mut Scene, ed: &mut EditState, path: PathBuf) -> bool {
    match save_project(r, scene, ed, path) {
        Ok(p) => {
            ed.say(format!("Saved {}", p.display()));
            true
        }
        Err(e) => {
            ed.fail(format!("Couldn't save: {e:#}"));
            false
        }
    }
}

/// Run a button or hotkey command.
pub fn run_command(cmd: Command, r: &mut Renderer, scene: &mut Scene, ed: &mut EditState) {
    let dims = scene.volume.dims;
    // In World mode the voxel commands act on whole parts, or not at all.
    let cmd = match (scene.mode, cmd) {
        (Mode::Edit(_), cmd) => cmd,
        (_, Command::SelectAll) => Command::Tree(TreeOp::PickAll),
        (_, Command::SelectNone) => Command::Tree(TreeOp::PickNone),
        (_, Command::SelectInvert) => Command::Tree(TreeOp::PickInvert),
        (_, Command::Op(Op::Clear)) => Command::Tree(TreeOp::Delete),
        (_, Command::Copy) => {
            ed.part_clipboard = scene.picked.clone();
            return;
        }
        (_, Command::Paste) => {
            scene.picked = ed.part_clipboard.iter().copied().filter(|&p| scene.world.node(p).is_some()).collect();
            Command::Tree(TreeOp::Duplicate)
        }
        (_, Command::Op(_) | Command::Resize(_) | Command::Cut | Command::RunShader | Command::Transform(..) | Command::PaletteRandom | Command::PatternFromSelection) => {
            ed.fail("Voxel edits need edit mode: pick a model and press Tab");
            return;
        }
        (_, cmd) => cmd,
    };
    // An error stays up until the next action.
    if !matches!(cmd, Command::Frame(_) | Command::Unsaved(_)) && ed.notice.as_ref().is_some_and(|n| n.error) {
        ed.notice = None;
    }
    // Ask before dropping unsaved changes.
    let replaces = match &cmd {
        Command::New | Command::Open | Command::OpenPath(_) | Command::Quit => true,
        Command::Import(crate::import::window::ImportCmd::Run) => ed.import.as_ref().is_some_and(|i| i.new_project),
        _ => false,
    };
    if replaces && ed.dirty.is_dirty() && !ed.discard {
        ed.unsaved_ask = Some(cmd);
        return;
    }
    let reshape = matches!(cmd, Command::Op(_) | Command::Resize(_) | Command::Transform(..));
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
            let dir = ed.dialog_dir("open", workspace_dir());
            if let Some(p) = file_dialog().add_filter("MagicaVoxel", &["vox"]).set_directory(dir).pick_file() {
                ed.picked("open", &p);
                run_command(Command::OpenPath(p), r, scene, ed);
            }
            false
        }
        Command::OpenPath(p) => {
            match open_project(p.clone(), r, scene, ed) {
                Ok(()) => ed.say(format!("Opened {}", p.display())),
                Err(e) => ed.fail(format!("Couldn't open: {e:#}")),
            }
            false
        }
        Command::Save => {
            let path = ed.save_path.clone();
            save_and_say(r, scene, ed, path);
            false
        }
        Command::SaveAs => {
            let dir = ed.save_path.parent().filter(|d| d.is_dir()).map_or_else(workspace_dir, Path::to_path_buf);
            let dialog = file_dialog().add_filter("MagicaVoxel", &["vox"]).set_directory(dir).set_file_name(format!("{}.vox", ed.name()));
            if let Some(p) = dialog.save_file() {
                save_and_say(r, scene, ed, p.with_extension("vox"));
            }
            false
        }
        Command::Quit => {
            ed.quit = true;
            false
        }
        Command::Unsaved(choice) => {
            let Some(next) = ed.unsaved_ask.take() else { return };
            match choice {
                ui::UnsavedChoice::Save => {
                    let path = ed.save_path.clone();
                    if save_and_say(r, scene, ed, path) {
                        run_command(next, r, scene, ed);
                    }
                }
                ui::UnsavedChoice::Discard => {
                    ed.discard = true;
                    run_command(next, r, scene, ed);
                    ed.discard = false;
                }
                ui::UnsavedChoice::Cancel => {}
            }
            false
        }
        Command::Undo => history_step(scene, ed, false),
        Command::Redo => history_step(scene, ed, true),
        Command::Preferences => {
            ed.prefs_open ^= true;
            false
        }
        Command::ToggleWorld => match scene.mode {
            Mode::Edit(_) => set_mode(scene, ed, None),
            Mode::World => match scene.picked.first().copied().or_else(|| scene.world.models().first().copied()) {
                Some(id) => set_mode(scene, ed, Some(id)),
                None => false,
            },
        },
        Command::EditPart(id) => set_mode(scene, ed, Some(id)),
        Command::ImportAnimation(None) => {
            let dir = ed.dialog_dir("animation", dialog_dir(ed));
            if let Some(p) = file_dialog().add_filter("glTF", &["glb", "gltf"]).set_directory(dir).pick_file() {
                ed.picked("animation", &p);
                match crate::gltf_import::Gltf::load(&p) {
                    Ok(g) => {
                        let names: Vec<String> = g.clips().into_iter().map(|c| c.name).collect();
                        let file = p.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
                        if names.is_empty() {
                            ed.fail(format!("{file} has no animations"));
                        } else {
                            ed.say(format!("{file}: {} clips; pick one and press Use clip", names.len()));
                        }
                        ed.timeline.clips = Some((p, names));
                        ed.timeline.clip = 0;
                    }
                    Err(e) => ed.fail(format!("Couldn't read animations: {e:#}")),
                }
            }
            false
        }
        Command::ImportAnimation(Some((path, k))) => {
            let rig = scene.picked.first().copied();
            keep_history(scene, &mut ed.editor);
            let on = crate::timeline::rig_label(&scene.world, rig, false);
            let res = crate::gltf_import::Gltf::load(&path).and_then(|g| scene.change_quiet(|w| crate::gltf_import::import(w, &g, Some(&k.to_string()), rig, true)));
            match res {
                Ok(done) => {
                    let parts: Vec<&str> = done.bones.iter().map(|b| b.2.as_str()).collect();
                    let mut msg = format!("{} on {on}: {} frames onto {} parts ({})", done.clip, done.frames, parts.len(), parts.join(", "));
                    if !done.missing.is_empty() {
                        let missing: Vec<&str> = done.missing.iter().map(|b| crate::rig::bone_name(*b)).collect();
                        msg.push_str(&format!("; no part for {}", missing.join(", ")));
                    }
                    ed.say(msg);
                }
                Err(e) => ed.fail(format!("Couldn't use the clip: {e:#}")),
            }
            ed.editor.reset();
            true
        }
        Command::ImportPick => {
            let dir = ed.dialog_dir("import", dialog_dir(ed));
            let dialog = file_dialog()
                .add_filter("Everything voxely imports", &crate::import::extensions())
                .add_filter("Voxel formats", crate::import::formats::EXTENSIONS)
                .add_filter("Meshes", crate::import::mesh::EXTENSIONS)
                .add_filter("Pictures (image, heightmap, slices)", crate::import::IMAGES)
                .set_directory(dir);
            if let Some(p) = dialog.pick_file() {
                ed.picked("import", &p);
                ed.import = crate::import::window::Pending::new(p);
            }
            false
        }
        Command::ImportPath(p) => {
            ed.import = crate::import::window::Pending::new(p);
            false
        }
        Command::Import(crate::import::window::ImportCmd::Pick(folder)) => {
            let dialog = file_dialog().set_directory(ed.dialog_dir("import", dialog_dir(ed)));
            let picked = if folder { dialog.pick_folder() } else { dialog.add_filter("Everything voxely imports", &crate::import::extensions()).pick_file() };
            if let Some(p) = picked {
                ed.picked("import", &p);
                let keep = ed.import.as_ref().map(|i| (i.new_project, i.add_colors, i.how));
                ed.import = crate::import::window::Pending::new(p);
                if let (Some(i), Some((new, add, how))) = (ed.import.as_mut(), keep) {
                    (i.new_project, i.add_colors) = (new, add);
                    // A folder keeps the slice settings chosen so far.
                    if folder && matches!(how, crate::import::How::Slices(_)) {
                        i.how = how;
                    }
                }
            }
            false
        }
        Command::Import(crate::import::window::ImportCmd::Run) => {
            let Some(p) = ed.import.take() else { return };
            match import_file(&p.path, &p.how, p.new_project, p.add_colors, r, scene, ed) {
                Ok(msg) => {
                    ed.say(msg.clone());
                    ed.export_status = Some(msg);
                }
                Err(e) => ed.import = Some(crate::import::window::Pending { error: Some(format!("{e:#}")), ..p }),
            }
            false
        }
        Command::ToggleReference => {
            ed.backdrop.open ^= true;
            false
        }
        Command::Reference(cmd) => {
            use crate::backdrop::RefCmd;
            let path = match cmd {
                RefCmd::Pick => {
                    let p = file_dialog().add_filter("Pictures", crate::import::IMAGES).set_directory(ed.dialog_dir("picture", dialog_dir(ed))).pick_file();
                    if let Some(p) = &p {
                        ed.picked("picture", p);
                    }
                    p
                }
                RefCmd::Load(ref p) => Some(p.clone()),
                RefCmd::Agent(v) => ed.backdrop.agent(v),
                RefCmd::Clear => None,
            };
            if cmd == RefCmd::Clear {
                set_backdrop(r, ed, None);
            } else if let Some(p) = path {
                set_backdrop(r, ed, Some(p));
                // A new picture starts as tall as the canvas; an agent's
                // view keeps its pinned placement.
                if matches!(cmd, RefCmd::Pick | RefCmd::Load(_)) {
                    ed.backdrop.fit(scene.volume.dims, scene.origin);
                }
            }
            ed.backdrop.save_current();
            false
        }
        Command::DeleteProject(p) => {
            ed.confirm_delete = Some(p);
            false
        }
        Command::DeleteProjectConfirmed(p) => {
            ed.confirm_delete = None;
            match delete_project(&p, ed) {
                Ok(msg) => ed.say(msg),
                Err(e) => ed.fail(format!("Couldn't delete {}: {e:#}", p.display())),
            }
            false
        }
        Command::Frame(f) => {
            if f == scene.world.frame {
                false
            } else {
                let e = &mut ed.editor;
                if !scene.set_frame(f, |v| e.undo(v)) {
                    ed.editor.reset();
                }
                true
            }
        }
        Command::Tree(op) => tree_op(op, scene, ed),
        Command::Op(op) => match apply_op(op, scene, &ed.editor) {
            Some(new) => ed.editor.commit(&mut scene.volume, new),
            None => {
                if op == Op::Double {
                    ed.fail(format!("2X: the canvas would exceed {} voxels per side", ops::MAX_DIM));
                }
                false
            }
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
            let picked = scene.picked.iter().filter(|_| scene.mode == Mode::World).filter_map(|&p| scene.world.bounds(p)).reduce(|(a, s), (b, t)| {
                let lo = a.min(b);
                (lo, (a + s).max(b + t) - lo)
            });
            let picked = picked.map(|(lo, size)| ((lo - scene.origin).as_vec3(), (lo - scene.origin + size).as_vec3()));
            let (lo, hi) = picked.or_else(|| solid_bounds(scene, ed.editor.selection())).unwrap_or((Vec3::ZERO, dims.as_vec3()));
            r.camera.target = (lo + hi) * 0.5;
            r.reset();
            false
        }
        Command::Snapshot => {
            let msg = snapshot(r, &ed.name());
            ed.say(msg.clone());
            ed.export_status = Some(msg);
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
                    None => ed.fail(format!("Pattern {}: the model is empty", p.display())),
                },
                Err(e) => ed.fail(format!("Pattern {}: {e:#}", p.display())),
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
            None => {
                ed.fail(format!("Expected [axes] [values], e.g. \"xyz 2\", got {text:?}"));
                false
            }
        },
        Command::PaletteRandom => {
            let colors = selected_colors(ed);
            let new = crate::palette_ops::random_colors(&scene.volume, ed.editor.selection(), &colors, seed());
            ed.editor.commit(&mut scene.volume, new)
        }
        Command::Export(format) => {
            let dir = workspace_dir().join("export");
            let _ = std::fs::create_dir_all(&dir);
            let dialog = file_dialog()
                .add_filter(format.name(), &[format.extension()])
                .set_directory(ed.dialog_dir("export", dir))
                .set_file_name(export_file_name(&ed.name(), format));
            if let Some(path) = dialog.save_file() {
                ed.picked("export", &path);
                scene.settings = r.settings.clone();
                let msg = match crate::export::export(format, scene, &path) {
                    Ok(files) => {
                        let msg = export_summary(&files);
                        ed.say(msg.clone());
                        msg
                    }
                    Err(e) => {
                        let msg = format!("{} export failed: {e:#}", format.name());
                        ed.fail(msg.clone());
                        msg
                    }
                };
                ed.export_status = Some(msg);
            }
            false
        }
        Command::RenderImage => {
            let msg = render_final(r, scene, ed.image, &ed.name());
            ed.say(msg.clone());
            ed.export_status = Some(msg);
            false
        }
        Command::PatternFromSelection => {
            if ed.editor.pattern_from_selection(&scene.volume) {
                ed.editor.brush = edit::Brush::Pattern;
            }
            false
        }
        Command::Palette(cmd) => match crate::palette_lib::apply(cmd, r, scene, ed) {
            Ok((msg, changed)) => {
                ed.palette.say(msg);
                changed
            }
            Err(e) => {
                ed.fail(format!("Palette: {e:#}"));
                ed.palette.say(format!("{e:#}"));
                false
            }
        },
        Command::PalettePick(matched) => {
            let dir = ed.dialog_dir("palette", dialog_dir(ed));
            if let Some(path) = file_dialog().add_filter("Palette", &crate::palette_ops::PALETTE_EXTENSIONS).set_directory(dir).pick_file() {
                ed.picked("palette", &path);
                run_command(Command::Palette(crate::palette_lib::Cmd::Load(crate::palette_lib::Source::File(path), matched)), r, scene, ed);
            }
            false
        }
    };
    if changed {
        sync_volume(r, scene, &mut ed.editor);
        // Keep the model in view when the canvas changes size.
        let new = scene.volume.dims;
        if reshape && new != dims {
            r.camera.target = new.as_vec3() * 0.5;
            r.camera.dist *= new.as_vec3().length() / dims.as_vec3().length();
        }
    }
    if ed.editor.sel_dirty {
        sync_selection(r, scene, &mut ed.editor);
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
    for &id in &scene.picked {
        draw_markers(p, r, scene, ppp, id);
    }
    if scene.mode == Mode::World {
        let painter = p;
        // Picked parts' boxes.
        let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(80, 160, 255));
        for &id in &scene.picked {
            if let Some((lo, size)) = scene.world.bounds(id) {
                let lo = lo - scene.origin;
                outline(painter, r, ppp, lo.as_vec3(), (lo + size).as_vec3(), stroke);
            }
        }
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

/// A node's pivot (orange cross) and sockets (green dots with names).
fn draw_markers(p: &egui::Painter, r: &Renderer, scene: &Scene, ppp: f32, id: usize) {
    let w = &scene.world;
    let Some(n) = w.node(id) else { return };
    // Into the volume's frame: the edited part's own coordinates, or world.
    let at = |local: Vec3| {
        let v = if scene.editing() == Some(id) { w.to_part(id, local) } else { w.affine(id).transform_point3(local) - scene.origin.as_vec3() };
        r.project(v).map(|(x, y)| egui::pos2(x / ppp, y / ppp))
    };
    let orange = egui::Color32::from_rgb(255, 160, 40);
    if let Some(c) = at(w.pivot_local(id)) {
        let s = egui::Stroke::new(2.0, orange);
        p.line_segment([c - egui::vec2(6.0, 0.0), c + egui::vec2(6.0, 0.0)], s);
        p.line_segment([c - egui::vec2(0.0, 6.0), c + egui::vec2(0.0, 6.0)], s);
    }
    let green = egui::Color32::from_rgb(90, 220, 120);
    for sock in &n.sockets {
        if let Some(c) = at(sock.pos) {
            p.circle(c, 4.0, green, egui::Stroke::new(1.0, egui::Color32::BLACK));
            p.text(c + egui::vec2(7.0, -7.0), egui::Align2::LEFT_BOTTOM, &sock.name, egui::FontId::proportional(12.0), green);
        }
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
            let editing = scene.editing().is_some();
            let history = (editing && ed.editor.can_undo() || scene.can_undo_tree(), editing && ed.editor.can_redo() || scene.can_redo_tree());
            let name = if ed.dirty.is_dirty() { format!("{} \u{2022}", ed.name()) } else { ed.name() };
            ui::top_bar(ui, &mut modeling, &name, history, cmds);
            if r.modeler.is_some() && modeling != was {
                r.modeling = modeling;
                ed.enabled = modeling;
                r.reset();
            }
        });
        egui::Panel::bottom("status").show(ui, |ui| {
            let hover = ed.hover.filter(|h| !h.on_ground()).map(|h| (h.cell, scene.volume.get(h.cell)));
            let world = scene.mode == Mode::World;
            let hint = match (r.modeling, ed.editor.tool) {
                (false, _) => "Rotate [RButton/LButton] : Move [MButton] : Zoom [Wheel]",
                _ if world => "Pick [LButton] : Add [Shift] : Move [LButton drag, arrows] : Edit [Tab, double-click] : Rotate view [RButton]",
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
            let notice = &mut ed.notice;
            let status = ui::Status { options, hover, projection: &mut r.settings.projection, iso, free: &mut ed.free_look, agent: &mut ed.agent, timeline: &mut ed.timeline.open, hint, notice };
            if ui::status_bar(ui, status, cmds) {
                iso_camera(r);
            }
        });
        let quick = crate::palette_lib::quick(ed);
        egui::Panel::left("palette").exact_size(160.0).resizable(false).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                edit = ui::palette_panel(ui, &mut ed.editor, scene, &mut ed.palette, &quick, cmds);
            });
        });
        // The Project column shows in both modes, like MagicaVoxel.
        egui::Panel::right("project").exact_size(165.0).resizable(false).show(ui, |ui| {
            let current = ed.source.clone().unwrap_or_default();
            let pattern = ed.editor.pattern.as_ref().map(|p| p.name.clone());
            ed.rescan_if_stale();
            ui::project_panel(ui, &mut ed.project_tab, &ed.projects, &ed.samples, &current, pattern.as_deref(), &mut ed.shaders, ed.export_status.as_deref(), cmds);
        });
        if r.modeling {
            let editing = scene.editing().is_some();
            egui::Panel::left("brush").exact_size(190.0).resizable(false).show(ui, |ui| {
                ui.add_enabled_ui(editing, |ui| ui::brush_panel(ui, &mut ed.editor, cmds));
            });
            if !editing {
                egui::Panel::right("world").default_size(180.0).size_range(160.0..=320.0).resizable(true).show(ui, |ui| crate::outline::panel(ui, scene, &mut ed.outline, cmds));
            } else if let Some(m) = r.modeler.as_mut() {
                egui::Panel::right("edit").exact_size(160.0).resizable(false).show(ui, |ui| ui::edit_panel(ui, &ed.editor, &mut m.options, &mut ed.transform_text, cmds));
            }
            let part = scene.editing().map(|id| {
                let label = scene.world.label(id);
                if scene.world.tilted(id) { format!("{label} (upright, alone)") } else { label }
            });
            egui::Panel::top("viewbar").show(ui, |ui| ui::view_bar(ui, dims, &mut ed.canvas, part.as_deref(), scene.world.models().len(), cmds));
        } else {
            // MagicaVoxel's render layout: Light / Sample / Display on the
            // left, Matter / Camera / Image on the right.
            egui::Panel::left("light").default_size(250.0).max_size(320.0).show(ui, |ui| ui::light_panel(ui, &mut ed.light_tab, &mut r.settings, panel));
            egui::Panel::right("matter").default_size(250.0).max_size(320.0).show(ui, |ui| {
                let state = ui::MatterState { tab: &mut ed.matter_tab, clipboard: &mut ed.material_clipboard, image: &mut ed.image, spp, size };
                let a = ui::matter_panel(ui, &mut r.settings, &ed.editor, scene, state, cmds);
                edit.materials_changed |= a.materials_changed;
            });
        }
        // Docked under the 3D view (after the side panels, so it spans
        // only the view) instead of floating over the model.
        if ed.timeline.open {
            egui::Panel::bottom("timeline").show(ui, |ui| crate::timeline::panel(ui, scene, &mut ed.timeline, cmds));
        }
        if ed.agent.open {
            egui::Panel::bottom("agents").resizable(true).default_size(190.0).show(ui, |ui| crate::agents::panel(ui, &mut ed.agent));
        }
    }
    // Autosave shows what's in use (an agent may have changed it).
    let mut autosave = ed.tools.inspect.autosave_config();
    if ui::preferences_window(ui.ctx(), &mut ed.prefs_open, &mut ed.prefs, ed.ui_scale, &mut autosave) {
        ed.tools.inspect.set_autosave_config(autosave.0, autosave.1);
        (ed.prefs.autosave, ed.prefs.autosave_every) = autosave;
        ed.prefs.save();
    }
    crate::palette_lib::window(ui.ctx(), ed, cmds);
    crate::palette_lib::handle_drops(ui.ctx(), ed, cmds);
    crate::import::window::window(ui.ctx(), &mut ed.import, cmds);
    // The reference picture follows the project.
    let (open, project) = (ed.backdrop.open, ed.name());
    if let Some(path) = ed.backdrop.sync_project(&project) {
        let show = ed.backdrop.show;
        set_backdrop(r, ed, path);
        (ed.backdrop.open, ed.backdrop.show) = (open, show);
    }
    crate::backdrop::window(ui.ctx(), &mut ed.backdrop, scene.volume.dims, scene.origin, cmds);
    if let Some(m) = r.modeler.as_mut() {
        m.reference_plane = ed.backdrop.plane(scene.volume.dims, scene.origin);
    }
    confirm_delete(ui.ctx(), ed, cmds);
    confirm_unsaved(ui.ctx(), ed, cmds);
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
    crate::palette_lib::sync_preview(r, scene, ed, edit.materials_changed);
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
/// Returns the socket file's inode, for `release_control`.
pub fn serve_control(proxy: winit::event_loop::EventLoopProxy<UserEvent>) -> Option<u64> {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixListener;
    let path = crate::mcp::socket_path();
    // The newest window takes over the socket.
    let _ = std::fs::remove_file(&path);
    let listener = match UnixListener::bind(&path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("control socket {}: {e}", path.display());
            return None;
        }
    };
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    let inode = std::os::unix::fs::MetadataExt::ino(&std::fs::metadata(&path).ok()?);
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
    Some(inode)
}

/// Remove the control socket on close, unless a newer window has taken it
/// over (it's then a different file).
pub fn release_control(inode: Option<u64>) {
    remove_if_ours(&crate::mcp::socket_path(), inode);
}

fn remove_if_ours(path: &Path, inode: Option<u64>) {
    let ours = std::fs::symlink_metadata(path).is_ok_and(|m| Some(std::os::unix::fs::MetadataExt::ino(&m)) == inode);
    if ours {
        let _ = std::fs::remove_file(path);
    }
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
        for (_, c) in scene.volume.cells.filled() {
            counts[c as usize] += 1;
        }
        let color = (1..256).max_by_key(|&i| counts[i]).filter(|&i| counts[i] > 0).unwrap_or(1) as u8;
        let mut edit = EditState::new(color, source, save_path);
        edit.timeline.open = scene.world.animated();
        edit.prefs = crate::prefs::Prefs::load();
        edit.tools.inspect.set_autosave_config(edit.prefs.autosave, edit.prefs.autosave_every);
        App {
            scene,
            gfx: None,
            mouse: Mouse::default(),
            modifiers: ModifiersState::empty(),
            panel,
            show_panel: true,
            edit,
        }
    }

    fn init(&mut self, el: &ActiveEventLoop) -> Result<()> {
        let window = Arc::new(
            el.create_window(
                Window::default_attributes()
                    .with_title(format!("voxely — {}", self.edit.name()))
                    .with_inner_size(winit::dpi::LogicalSize::new(1280, 800))
                    .with_min_inner_size(winit::dpi::LogicalSize::new(crate::prefs::MIN_WINDOW.0, crate::prefs::MIN_WINDOW.1)),
            )?,
        );
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(
            el.owned_display_handle(),
        )));
        let _ = DIALOG_PARENT.set(window.clone());
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
        // Ctrl+= / Ctrl+- / Ctrl+0 step the saved interface scale instead.
        egui_ctx.options_mut(|o| o.zoom_with_keyboard = false);
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
            // A dropdown left open over a file dialog or the next action
            // would be stale.
            if !matches!(cmd, Command::Frame(_)) {
                egui::Popup::close_all(&g.egui_ctx);
            }
            run_command(cmd, &mut g.renderer, &mut self.scene, &mut self.edit);
            g.window.request_redraw();
        }
    }

    /// Note changes since the last look: the window title's unsaved mark,
    /// and GUI edits count toward autosave (`gui`; agents' edits are
    /// counted by the control tools).
    fn track_changes(&mut self, gui: bool) {
        let Some(g) = self.gfx.as_mut() else { return };
        let sig = Sig::of(&g.renderer.settings, &self.scene, &self.edit);
        if self.edit.dirty.see(sig) && gui {
            let mut ctx = crate::control::Ctx { r: &mut g.renderer, scene: &mut self.scene, ed: &mut self.edit, live: true };
            match crate::control::inspect::count_gui_edit(&mut ctx) {
                Some(Ok(msg)) => self.edit.say(msg),
                Some(Err(msg)) => self.edit.fail(msg),
                None => {}
            }
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
        let world = self.scene.mode == Mode::World;
        if cmd && world {
            // MagicaVoxel's World-editor hotkeys.
            let op = match c.as_str() {
                "n" => Some(TreeOp::New),
                "r" if shift => Some(TreeOp::Ungroup),
                "r" => Some(TreeOp::Group),
                "h" => Some(TreeOp::Hide(!shift)),
                _ => None,
            };
            if let Some(op) = op {
                self.command(Command::Tree(op));
                return;
            }
        }
        if matches!(event.logical_key.as_ref(), Key::Named(NamedKey::Tab)) {
            // Tab switches edit / World mode; Ctrl+Tab hides the panels.
            if cmd {
                self.show_panel ^= true;
                self.redraw_soon();
            } else if self.gfx.as_ref().is_some_and(|g| g.renderer.modeling) {
                self.command(Command::ToggleWorld);
            }
            return;
        }
        if world && !cmd && self.gfx.as_ref().is_some_and(|g| g.renderer.modeling) {
            let mv = |x, y, z| Some(TreeOp::Move(IVec3::new(x, y, z)));
            let op = match event.logical_key.as_ref() {
                Key::Named(NamedKey::ArrowLeft) => mv(-1, 0, 0),
                Key::Named(NamedKey::ArrowRight) => mv(1, 0, 0),
                Key::Named(NamedKey::ArrowDown) => mv(0, -1, 0),
                Key::Named(NamedKey::ArrowUp) => mv(0, 1, 0),
                Key::Named(NamedKey::PageDown) => mv(0, 0, -1),
                Key::Named(NamedKey::PageUp) => mv(0, 0, 1),
                _ => match c.as_str() {
                    "=" | "+" => Some(TreeOp::Turn(2, -1)),
                    "-" => Some(TreeOp::Turn(2, 1)),
                    "9" => Some(TreeOp::Turn(0, -1)),
                    "0" => Some(TreeOp::Turn(0, 1)),
                    _ => None,
                },
            };
            if let Some(op) = op {
                if !self.scene.picked.is_empty() {
                    self.command(Command::Tree(op));
                }
                return;
            }
        }
        if cmd {
            let command = match c.as_str() {
                "z" if shift => Some(Command::Redo),
                "z" => Some(Command::Undo),
                "y" => Some(Command::Redo),
                "s" if shift => Some(Command::SaveAs),
                "s" => Some(Command::Save),
                "o" if shift => Some(Command::ImportPick),
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
                // Interface scale, like a browser's zoom.
                "=" | "+" | "-" | "0" => {
                    self.edit.prefs.step_scale(match c.as_str() {
                        "-" => -1,
                        "0" => 0,
                        _ => 1,
                    });
                    self.edit.prefs.save();
                    None
                }
                "," => Some(Command::Preferences),
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
            Key::Named(NamedKey::Space) => self.mouse.space = true,
            Key::Named(NamedKey::ArrowLeft) => r.settings.sun_yaw -= 10.0,
            Key::Named(NamedKey::ArrowRight) => r.settings.sun_yaw += 10.0,
            Key::Named(NamedKey::ArrowUp) => r.settings.sun_pitch = (r.settings.sun_pitch + 5.0).min(90.0),
            Key::Named(NamedKey::ArrowDown) => r.settings.sun_pitch = (r.settings.sun_pitch - 5.0).max(-90.0),
            _ => match c.as_str() {
                // Animation frames.
                "," | "." => {
                    let a = &self.scene.world.anim;
                    let f = self.scene.world.frame;
                    command = Some(Command::Frame(if c == "," { f.checked_sub(1).unwrap_or(a.frames - 1) } else { (f + 1) % a.frames.max(1) }));
                }
                "/" => self.edit.timeline.play(!self.edit.timeline.playing),
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
        } else if let Some(d) = self.edit.world_drag.as_mut().filter(|_| self.mouse.left) {
            let (ro, rd) = g.renderer.camera_ray(x as f32, y as f32);
            let origin = self.scene.origin;
            if let Some(c) = edit::plane_cell(ro, rd, d.axis, d.anchor[d.axis] - origin[d.axis]) {
                let delta = c + origin - d.anchor;
                if delta != d.moved {
                    if d.moved == IVec3::ZERO {
                        self.scene.begin_change();
                    }
                    let (step, ids) = (delta - d.moved, d.ids.clone());
                    d.moved = delta;
                    self.scene.apply(|w| ids.iter().for_each(|&id| w.translate(id, step)));
                    sync_volume(&mut g.renderer, &mut self.scene, &mut self.edit.editor);
                }
            }
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
        let cells: Vec<usize> = v
            .cells
            .filled()
            .map(|(i, _)| i)
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

    /// World mode click: pick the part under the cursor (its top-level
    /// object; Alt picks the model itself) and start dragging it.
    /// Double-click edits the model.
    fn world_press(&mut self) {
        let hit = self.edit.hover.filter(|h| !h.on_ground());
        let owner = hit.and_then(|h| self.scene.owners.get(self.scene.volume.index(h.cell))).filter(|&o| o != 0);
        let Some((hit, model)) = hit.zip(owner.map(|o| o as usize)) else {
            if !self.modifiers.shift_key() {
                self.command(Command::Tree(TreeOp::PickNone));
            }
            return;
        };
        let now = std::time::Instant::now();
        if self.edit.last_click.is_some_and(|(t, m)| m == model && now.duration_since(t).as_millis() < 400) {
            self.edit.last_click = None;
            self.command(Command::EditPart(model));
            return;
        }
        self.edit.last_click = Some((now, model));
        let w = &self.scene.world;
        let mut node = model;
        if !self.modifiers.alt_key() {
            while let Some(p) = w.node(node).and_then(|n| n.parent).filter(|&p| p != ROOT) {
                node = p;
            }
        }
        let shift = self.modifiers.shift_key();
        if shift || !self.scene.picked.contains(&node) {
            self.command(Command::Tree(TreeOp::Pick(node, shift)));
        }
        if self.scene.picked.contains(&node) {
            let axis = hit.normal.abs().max_position();
            self.edit.world_drag = Some(WorldDrag { ids: self.scene.picked.clone(), anchor: hit.cell + self.scene.origin, axis, moved: IVec3::ZERO });
        }
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
                } else if down && self.edit.enabled && !self.mouse.space && !self.mouse.zoom && self.scene.mode == Mode::World {
                    self.update_hover();
                    self.world_press();
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
                    if self.edit.world_drag.take().is_some() {
                        self.edit.editor.reset();
                    }
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
        if let Some(f) = self.edit.timeline.tick(&self.scene) {
            self.command(Command::Frame(f));
        }
        self.track_changes(true);
        let Some(g) = self.gfx.as_mut() else { return };
        if self.edit.editor.sel_dirty {
            sync_selection(&mut g.renderer, &self.scene, &mut self.edit.editor);
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

        // The chosen interface scale, or less if the window is too narrow.
        let width = g.window.inner_size().to_logical::<f32>(g.window.scale_factor()).width;
        self.edit.ui_scale = crate::prefs::effective_scale(self.edit.prefs.ui_scale, width);
        if (g.egui_ctx.zoom_factor() - self.edit.ui_scale).abs() > 1e-4 {
            g.egui_ctx.set_zoom_factor(self.edit.ui_scale);
            // Applies from the next frame on; draw that one too.
            g.window.request_redraw();
        }
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
        self.track_changes(true);

        let Some(g) = self.gfx.as_ref() else { return };
        let ed = &self.edit.editor;
        let name = if self.edit.dirty.is_dirty() { format!("{} \u{2022}", self.edit.name()) } else { self.edit.name() };
        let title = if g.renderer.modeling {
            let sel = match ed.selected_count() {
                0 => String::new(),
                n => format!(" — {n} selected"),
            };
            match self.scene.editing() {
                Some(id) if self.scene.world.models().len() > 1 => format!("voxely — {name} — {} — {:?} {:?}{sel}", self.scene.world.label(id), ed.tool, ed.brush),
                Some(_) => format!("voxely — {name} — Model — {:?} {:?}{sel}", ed.tool, ed.brush),
                None => format!("voxely — {name} — World — {} picked", self.scene.picked.len()),
            }
        } else {
            format!("voxely — {name} — Render — {} spp", g.renderer.frame)
        };
        g.window.set_title(&title);
        if converging || self.edit.timeline.playing {
            g.window.request_redraw();
        }
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn user_event(&mut self, _el: &ActiveEventLoop, event: UserEvent) {
        let UserEvent::Control { tool, args, reply } = event;
        let result = match self.gfx.as_mut() {
            Some(g) if !self.edit.agent.allow => {
                let result = crate::agents::AgentState::paused_reply();
                self.edit.agent.record(&tool, &result);
                g.window.request_redraw();
                result
            }
            Some(_) => {
                self.track_changes(true);
                let g = self.gfx.as_mut().unwrap();
                let mut ctx = crate::control::Ctx { r: &mut g.renderer, scene: &mut self.scene, ed: &mut self.edit, live: true };
                let result = crate::mcp::call_guarded(&mut ctx, &tool, &args);
                self.edit.agent.record(&tool, &result);
                g.window.request_redraw();
                self.track_changes(false);
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
            // Escape closes any open dropdown, wherever the focus is.
            if let WindowEvent::KeyboardInput { event: KeyEvent { logical_key: Key::Named(NamedKey::Escape), state: ElementState::Pressed, .. }, .. } = &event {
                if egui::Popup::is_any_open(&g.egui_ctx) {
                    egui::Popup::close_all(&g.egui_ctx);
                    g.window.request_redraw();
                    return;
                }
            }
            let typing = g.egui_ctx.egui_wants_keyboard_input() && matches!(event, WindowEvent::KeyboardInput { .. });
            if (resp.consumed && !matches!(event, WindowEvent::CursorMoved { .. } | WindowEvent::MouseInput { .. })) || (pointer_on_ui && press) || typing {
                return;
            }
        }
        match event {
            // With unsaved changes, ask first (Quit sets `quit`).
            WindowEvent::CloseRequested if self.gfx.is_some() => self.command(Command::Quit),
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
        if self.edit.quit {
            el.exit();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(scene: &Scene, ed: &EditState) -> Sig {
        Sig::of(&scene.settings, scene, ed)
    }

    #[test]
    fn unsaved_changes_are_tracked() {
        let mut scene = Scene::empty(IVec3::splat(4));
        let mut ed = EditState::new(1, None, PathBuf::from("x.vox"));
        ed.dirty.clean(sig(&scene, &ed));
        assert!(!ed.dirty.is_dirty());
        assert!(!ed.dirty.see(sig(&scene, &ed)), "nothing changed");
        // A tree change is an edit (it counts toward autosave).
        scene.change(|w| w.anim.fps = 5.0);
        assert!(ed.dirty.see(sig(&scene, &ed)));
        assert!(ed.dirty.is_dirty());
        assert!(!ed.dirty.see(sig(&scene, &ed)), "seen once");
        ed.dirty.clean(sig(&scene, &ed));
        assert!(!ed.dirty.is_dirty());
        // Undo is a change too, and it stays unsaved.
        scene.undo_tree(false);
        assert!(ed.dirty.see(sig(&scene, &ed)));
        assert!(ed.dirty.is_dirty());
        ed.dirty.clean(sig(&scene, &ed));
        // A color change is unsaved but not an undo step.
        scene.palette[3] = [1, 2, 3, 255];
        assert!(!ed.dirty.see(sig(&scene, &ed)));
        assert!(ed.dirty.is_dirty());
        ed.dirty.clean(sig(&scene, &ed));
        // Switching the view's projection isn't a change to the project.
        scene.settings.projection = Projection::Orthographic;
        ed.dirty.see(sig(&scene, &ed));
        assert!(!ed.dirty.is_dirty());
        // Voxel edits in edit mode are.
        let full = ops::full(&scene.volume, 2);
        ed.editor.commit(&mut scene.volume, full);
        assert!(ed.dirty.see(sig(&scene, &ed)));
        assert!(ed.dirty.is_dirty());
    }

    #[test]
    fn errors_stay_and_messages_replace_them() {
        let mut ed = EditState::new(1, None, PathBuf::from("x.vox"));
        ed.fail("Couldn't save: disk full");
        assert!(ed.notice.as_ref().is_some_and(|n| n.error));
        ed.say("Saved x.vox");
        assert!(ed.notice.as_ref().is_some_and(|n| !n.error && n.text == "Saved x.vox"));
    }

    #[test]
    fn closing_an_older_window_keeps_a_newer_socket() {
        let dir = std::env::temp_dir().join(format!("voxely-sock-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.sock");
        let _ = std::fs::remove_file(&path);
        let old = std::os::unix::net::UnixListener::bind(&path).unwrap();
        let old_ino = std::os::unix::fs::MetadataExt::ino(&std::fs::metadata(&path).unwrap());
        // A newer window takes over the path.
        std::fs::remove_file(&path).unwrap();
        let new = std::os::unix::net::UnixListener::bind(&path).unwrap();
        let new_ino = std::os::unix::fs::MetadataExt::ino(&std::fs::metadata(&path).unwrap());
        remove_if_ours(&path, Some(old_ino));
        assert!(path.exists());
        remove_if_ours(&path, Some(new_ino));
        assert!(!path.exists());
        drop((old, new));
        let _ = std::fs::remove_dir_all(dir);
    }
}
