//! The panels: file and brush controls with the palette on the left; in
//! the modeler, MagicaVoxel's Edit / Selection / Canvas / Project / Display
//! panels on the right, and in the renderer its render settings.

use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};

use egui::{CollapsingHeader, ComboBox, Slider, Ui};
use glam::{IVec3, Vec3};

use crate::edit::{Brush, Editor, GeoShape, MarqueeMode, Tool, XformMode};
use crate::env::EnvMode;
use crate::modeler::ModelerOptions;
use crate::scene::{BgMode, DenoiseView, GpuMaterial, MaterialKind, Projection, RenderSettings, Scene};

/// Whole-model edits (see `ops`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Full,
    Fill,
    Clear,
    Reverse,
    Double,
    Half,
    Flip(usize),
    Rotate(usize),
    /// Shrink the canvas to the model.
    Fit,
    /// Shift by one voxel along an axis, wrapping around.
    Loop(usize, i32),
    Mirror(usize),
    Diagonal(usize),
    Shape(crate::ops::Shape),
    Maze,
    Perlin,
    Hull,
    Flood,
    Dilate,
    Erode,
    /// Pattern edits: each voxel becomes the pattern / keep the voxels under
    /// the tiled pattern / color them by it.
    Fractal,
    Mask,
    Tex,
}

impl Op {
    /// The button's name, for messages.
    pub fn name(self) -> &'static str {
        match self {
            Op::Full => "Full",
            Op::Fill => "Fill",
            Op::Clear => "Clear",
            Op::Reverse => "Rev",
            Op::Double => "2X",
            Op::Half => "1/2",
            Op::Flip(_) => "Flip",
            Op::Rotate(_) => "Rot",
            Op::Fit => "Fit",
            Op::Loop(..) => "Loop",
            Op::Mirror(_) => "Mirror",
            Op::Diagonal(_) => "Diagonal",
            Op::Shape(_) => "Shape",
            Op::Maze => "Maze",
            Op::Perlin => "Perlin",
            Op::Hull => "Hull",
            Op::Flood => "Flood",
            Op::Dilate => "Dilate",
            Op::Erode => "Erode",
            Op::Fractal => "Fractal",
            Op::Mask => "Mask",
            Op::Tex => "Tex",
        }
    }
}

/// Something a button or hotkey asks the app to do.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    New,
    Open,
    OpenPath(PathBuf),
    Save,
    SaveAs,
    Undo,
    Redo,
    Op(Op),
    SelectAll,
    SelectNone,
    SelectInvert,
    Copy,
    Cut,
    Paste,
    Resize(IVec3),
    ResetCamera,
    /// Center the camera on the selection, or the model [5].
    Focus,
    Snapshot,
    /// A palette-wide edit: load, save, sort, swap, blend, ...
    Palette(crate::palette_lib::Cmd),
    /// Use a model file as the Pattern brush's pattern.
    LoadPattern(PathBuf),
    /// Make the selection the pattern.
    PatternFromSelection,
    /// Choose the voxel shader for the Code brush and Run.
    SelectShader(PathBuf),
    /// Run the chosen voxel shader over the model (or the selection).
    RunShader,
    /// Import (true) or Export the shader's arg values as a text file.
    ShaderArgs(bool),
    /// Paste the copied material onto the selected color.
    PasteMaterial,
    ResetMaterial,
    /// Render the final image at the Image size and save it.
    RenderImage,
    /// Scale / Shear / Rotate / Repeat with MagicaVoxel's "axes values"
    /// text, e.g. "xyz 2".
    Transform(TransformKind, String),
    /// Recolor voxels randomly with the selected palette colors.
    PaletteRandom,
    /// Export in one of MagicaVoxel's formats (asks where to save).
    Export(crate::export::Format),
    /// Switch between editing a model and the World view [Tab].
    ToggleWorld,
    /// Show or hide the Preferences window [Ctrl+,].
    Preferences,
    /// Edit this node's model (a group's first model).
    EditPart(usize),
    Tree(crate::outline::TreeOp),
    /// Show this animation frame.
    Frame(u32),
    /// Pick a glTF file to import clips from (None), or import clip k.
    ImportAnimation(Option<(PathBuf, usize)>),
    /// Pick a file to import (another voxel format, a mesh, a picture).
    ImportPick,
    /// Open the Import window for this file.
    ImportPath(PathBuf),
    Import(crate::import::window::ImportCmd),
    /// Show or hide the Reference window.
    ToggleReference,
    Reference(crate::backdrop::RefCmd),
    /// Ask before moving this project file to the Trash.
    DeleteProject(PathBuf),
    /// Asked and confirmed: move it to the Trash.
    DeleteProjectConfirmed(PathBuf),
    /// Pick a palette file to load (`true`: keeping the model's look).
    PalettePick(bool),
    /// Close the window.
    Quit,
    /// The answer to "Save changes?" before a command that drops them.
    Unsaved(UnsavedChoice),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnsavedChoice {
    Save,
    Discard,
    Cancel,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TransformKind {
    Scale,
    Shear,
    Rotate,
    Repeat,
}

/// Text in the Edit panel's transform fields.
#[derive(Clone, Debug)]
pub struct TransformText {
    pub scale: String,
    pub shear: String,
    pub rotate: String,
    pub repeat: String,
    pub tab: EditTab,
}

impl Default for TransformText {
    fn default() -> Self {
        TransformText { scale: "xyz 2".into(), shear: "z 0.2 0.2".into(), rotate: "z 45".into(), repeat: "xy 2".into(), tab: EditTab::All }
    }
}

/// Palette state beyond the current color: the multi-color selection and
/// display options.
pub struct PaletteState {
    /// Selected palette colors (Shift/Ctrl+click), used by Rand and shaders.
    pub selected: Vec<bool>,
    /// Dim colors no voxel uses.
    pub mask_unused: bool,
    /// Show RGB sliders instead of HSV.
    pub rgb: bool,
    /// Color code being typed.
    pub code: String,
    /// The swatch being dragged onto another.
    pub drag: Option<usize>,
    /// Colors drawn with lately, newest first.
    pub recent: Vec<u8>,
    /// The undo step of the color drag going on (see
    /// `palette_lib::drag_step`).
    pub step: Option<(u64, usize)>,
    /// The swatch whose color is being edited, so one slider drag is one
    /// palette undo step.
    pub editing: Option<usize>,
    /// Hue / saturation / value shift being dragged, and the colors it
    /// started from.
    pub shift: [f32; 3],
    pub shift_base: Option<Vec<(usize, [u8; 4])>>,
    /// Loading a palette recolors the model's colors to the nearest new
    /// ones instead of replacing them by index (see `Scene::look`).
    pub match_colors: bool,
    pub library_open: bool,
    /// The palette hovered this frame, shown on the model.
    pub preview: Option<crate::palette_lib::Source>,
    /// What the renderer is showing instead of the real palette.
    pub previewing: Option<(String, bool)>,
    /// Colors of library palettes by key (`None` if unreadable).
    pub cache: std::collections::HashMap<String, Option<Vec<[u8; 4]>>>,
    /// Palettes saved in the workspace, and when they were listed.
    pub saved: Vec<PathBuf>,
    pub saved_scanned: Option<std::time::Instant>,
    pub save_name: String,
    /// The last palette edit's result, and when.
    pub status: Option<(String, std::time::Instant)>,
}

impl Default for PaletteState {
    fn default() -> Self {
        PaletteState {
            selected: vec![false; 256],
            mask_unused: false,
            rgb: false,
            code: String::new(),
            drag: None,
            recent: Vec::new(),
            step: None,
            editing: None,
            shift: [0.0; 3],
            shift_base: None,
            match_colors: true,
            library_open: false,
            preview: None,
            previewing: None,
            cache: Default::default(),
            saved: Vec::new(),
            saved_scanned: None,
            save_name: String::new(),
            status: None,
        }
    }
}

impl PaletteState {
    /// A library palette's colors, read once.
    pub fn cache_colors(&mut self, key: &str, src: &crate::palette_lib::Source) -> Option<Vec<[u8; 4]>> {
        self.cache.entry(key.to_string()).or_insert_with(|| src.colors().ok()).clone()
    }

    pub fn say(&mut self, msg: String) {
        self.status = Some((msg, std::time::Instant::now()));
    }
}

/// The chosen voxel shader and its argument values.
pub struct ShaderState {
    pub files: Vec<PathBuf>,
    pub current: Option<(PathBuf, crate::xs::XsShader)>,
    pub values: Vec<f32>,
    /// Passes per run, each over the last one's output [1 - 64].
    pub iterations: u32,
    /// Last compile or run error.
    pub error: Option<String>,
}

/// Which list the Project column shows, like MagicaVoxel's tabs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum ProjectTab {
    #[default]
    Project,
    Pattern,
    Shader,
}

/// Things the panels ask the app to do this frame.
#[derive(Default)]
pub struct Actions {
    pub commands: Vec<Command>,
    /// Area left between the panels for the 3D view, in points.
    pub viewport: Option<egui::Rect>,
}

pub struct PanelState {
    pub ibl_files: Vec<PathBuf>,
}

fn slider(ui: &mut Ui, label: &str, v: &mut f32, range: RangeInclusive<f32>) {
    ui.add(Slider::new(v, range).text(label));
}

fn slider_u(ui: &mut Ui, label: &str, v: &mut u32, range: RangeInclusive<u32>) {
    ui.add(Slider::new(v, range).text(label));
}

fn color(ui: &mut Ui, label: &str, c: &mut Vec3) {
    ui.horizontal(|ui| {
        let mut rgb = c.to_array();
        if ui.color_edit_button_rgb(&mut rgb).changed() {
            *c = Vec3::from_array(rgb);
        }
        ui.label(label);
    });
}

/// Which settings the left render column shows, like MagicaVoxel's
/// Light / Sample / Display tabs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum LightTab {
    #[default]
    All,
    Light,
    Sample,
    Display,
}

/// Which settings the right render column shows: Matter / Camera / Image.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum MatterTab {
    #[default]
    All,
    Matter,
    Camera,
    Image,
}

/// Final-render image settings.
#[derive(Clone, Copy, Debug)]
pub struct ImageSettings {
    pub width: u32,
    pub height: u32,
    pub spp: u32,
}

impl Default for ImageSettings {
    fn default() -> Self {
        ImageSettings { width: 1280, height: 720, spp: 256 }
    }
}

/// Left render column: lighting, sampling and display settings.
pub fn light_panel(ui: &mut Ui, tab: &mut LightTab, s: &mut RenderSettings, state: &PanelState) {
    ui.horizontal(|ui| {
        ui.selectable_value(tab, LightTab::All, "All");
        ui.selectable_value(tab, LightTab::Light, "Light");
        ui.selectable_value(tab, LightTab::Sample, "Sample");
        ui.selectable_value(tab, LightTab::Display, "Display");
    });
    ui.separator();
    let show = |t: LightTab| *tab == LightTab::All || *tab == t;
    let (light, sample, display) = (show(LightTab::Light), show(LightTab::Sample), show(LightTab::Display));
    egui::ScrollArea::vertical().show(ui, |ui| {
        if light {
            CollapsingHeader::new("Sun").default_open(true).show(ui, |ui| {
                slider(ui, "Pitch", &mut s.sun_pitch, -90.0..=90.0);
                slider(ui, "Yaw", &mut s.sun_yaw, 0.0..=360.0);
                slider(ui, "Area", &mut s.sun_area, 0.0..=1.0);
                slider(ui, "Intensity", &mut s.sun_intensity, 0.0..=5.0);
                color(ui, "Color", &mut s.sun_color);
                ui.checkbox(&mut s.sun_disk, "Show sun disk");
            });
            CollapsingHeader::new("Sky").default_open(true).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut s.env_mode, EnvMode::Uniform, "Uniform");
                    ui.selectable_value(&mut s.env_mode, EnvMode::Atmosphere, "Atmo");
                    ui.selectable_value(&mut s.env_mode, EnvMode::Image, "IBL");
                });
                match s.env_mode {
                    EnvMode::Uniform => {
                        slider(ui, "Intensity", &mut s.sky_intensity, 0.0..=5.0);
                        color(ui, "Color", &mut s.sky_color);
                    }
                    EnvMode::Atmosphere => {
                        slider(ui, "Rayleigh", &mut s.rayleigh_density, 0.0..=2.0);
                        color(ui, "Rayleigh color", &mut s.rayleigh_color);
                        slider(ui, "Mie", &mut s.mie_density, 0.0..=2.0);
                        slider(ui, "Mie phase", &mut s.mie_g, 0.5..=0.99);
                        color(ui, "Mie color", &mut s.mie_color);
                        slider(ui, "Ozone", &mut s.ozone_density, 0.0..=2.0);
                        color(ui, "Ozone color", &mut s.ozone_color);
                    }
                    EnvMode::Image => {
                        let name = |p: &PathBuf| p.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                        let current = s.ibl_path.as_ref().map(name).unwrap_or_else(|| "none".into());
                        ComboBox::from_label("Image").selected_text(current).show_ui(ui, |ui| {
                            for f in &state.ibl_files {
                                ui.selectable_value(&mut s.ibl_path, Some(f.clone()), name(f));
                            }
                        });
                        slider(ui, "Intensity", &mut s.ibl_intensity, 0.0..=5.0);
                        slider(ui, "Rotation", &mut s.ibl_rotation, 0.0..=1.0);
                    }
                }
            });
            CollapsingHeader::new("Fog").show(ui, |ui| {
                slider(ui, "Density", &mut s.fog_density, 0.0..=5.0);
                slider(ui, "Phase", &mut s.fog_g, -0.9..=0.9);
                color(ui, "Color", &mut s.fog_color);
            });
        }
        if sample {
            CollapsingHeader::new("Sample").default_open(true).show(ui, |ui| {
                slider_u(ui, "Diffuse", &mut s.max_diffuse, 0..=15);
                slider_u(ui, "Specular", &mut s.max_specular, 5..=15);
                slider_u(ui, "Scatter", &mut s.max_scatter, 5..=15);
                slider(ui, "Energy", &mut s.clamp, 1.0..=5000.0);
                ui.checkbox(&mut s.translucent_shadows, "TR-Shadow").on_hover_text("Translucent shadow");
            });
        }
        if display {
            CollapsingHeader::new("Display").default_open(true).show(ui, |ui| {
                ui.checkbox(&mut s.ground, "Ground");
                color(ui, "Ground color", &mut s.ground_color);
                ui.horizontal(|ui| {
                    ui.label("BG");
                    ui.selectable_value(&mut s.bg_mode, BgMode::Sky, "Sky");
                    ui.selectable_value(&mut s.bg_mode, BgMode::Color, "Color");
                    ui.selectable_value(&mut s.bg_mode, BgMode::Transparent, "Transparent");
                });
                if s.bg_mode == BgMode::Color {
                    color(ui, "Background", &mut s.bg_color);
                }
                ui.checkbox(&mut s.show_edges, "Edge");
                if s.show_edges {
                    color(ui, "Edge color", &mut s.edge_color);
                    slider(ui, "Width", &mut s.edge_width, 0.01..=1.0);
                }
                ui.checkbox(&mut s.show_grid, "Grid");
                if s.show_grid {
                    color(ui, "Grid color", &mut s.grid_color);
                    ui.add(Slider::new(&mut s.grid_spacing, 1..=1024).logarithmic(true).text("Spacing"));
                    slider(ui, "Width", &mut s.grid_width, 0.01..=1.0);
                    ui.checkbox(&mut s.grid_ground_only, "Only on ground");
                }
            });
        }
    });
}

/// What the right render column asks for.
pub struct MatterState<'a> {
    pub tab: &'a mut MatterTab,
    pub clipboard: &'a mut Option<GpuMaterial>,
    pub image: &'a mut ImageSettings,
    pub spp: u32,
    pub size: (u32, u32),
}

/// Right render column: material of the selected color, camera and image.
pub fn matter_panel(ui: &mut Ui, s: &mut RenderSettings, e: &Editor, scene: &mut Scene, m: MatterState, cmds: &mut Vec<Command>) -> EditActions {
    let mut a = EditActions::default();
    let tab = m.tab;
    ui.horizontal(|ui| {
        ui.selectable_value(tab, MatterTab::All, "All");
        ui.selectable_value(tab, MatterTab::Matter, "Matter");
        ui.selectable_value(tab, MatterTab::Camera, "Camera");
        ui.selectable_value(tab, MatterTab::Image, "Image");
    });
    ui.separator();
    let show = |t: MatterTab| *tab == MatterTab::All || *tab == t;
    let (matter, camera, image) = (show(MatterTab::Matter), show(MatterTab::Camera), show(MatterTab::Image));
    egui::ScrollArea::vertical().show(ui, |ui| {
        if matter {
            CollapsingHeader::new("Material").default_open(true).show(ui, |ui| {
                let i = e.color as usize;
                ui.label(format!("Color #{i}"));
                let before = scene.materials[i];
                material_editor(ui, &mut scene.materials[i]);
                ui.horizontal(|ui| {
                    if ui.button("Copy").on_hover_text("Copy material [Ctrl+C]").clicked() {
                        *m.clipboard = Some(scene.materials[i]);
                    }
                    if ui.add_enabled(m.clipboard.is_some(), egui::Button::new("Paste")).on_hover_text("Paste material [Ctrl+V]").clicked() {
                        cmds.push(Command::PasteMaterial);
                    }
                    if ui.button("Reset").on_hover_text("Reset to diffuse [Ctrl+R]").clicked() {
                        cmds.push(Command::ResetMaterial);
                    }
                });
                if bytemuck::bytes_of(&before) != bytemuck::bytes_of(&scene.materials[i]) {
                    a.materials_changed = true;
                }
            });
        }
        if camera {
            CollapsingHeader::new("Lens").default_open(true).show(ui, |ui| {
                ComboBox::from_label("Projection").selected_text(format!("{:?}", s.projection)).show_ui(ui, |ui| {
                    for p in [Projection::Perspective, Projection::Orthographic, Projection::Stereographic, Projection::Panorama, Projection::Cylindrical] {
                        ui.selectable_value(&mut s.projection, p, format!("{p:?}"));
                    }
                });
                slider(ui, "Field of view", &mut s.fov, 1.0..=170.0);
                slider(ui, "Aperture", &mut s.aperture, 0.0..=2.0);
                slider_u(ui, "Blades", &mut s.blades, 0..=8);
                slider(ui, "Blade rotation", &mut s.blade_rotation, -180.0..=180.0);
            });
            CollapsingHeader::new("Film").default_open(true).show(ui, |ui| {
                slider(ui, "Exposure", &mut s.exposure, 0.0..=8.0);
                slider(ui, "Vignette", &mut s.vignette, 0.0..=1.0);
                slider(ui, "Gamma", &mut s.gamma, 1.0..=5.0);
                ui.checkbox(&mut s.aces, "ACES tone mapping");
                ui.checkbox(&mut s.bloom, "Bloom");
                if s.bloom {
                    slider(ui, "Mix", &mut s.bloom_mix, 0.0..=1.0);
                    slider(ui, "Size", &mut s.bloom_scale, 0.0..=1.0);
                    slider(ui, "Aspect", &mut s.bloom_aspect, 0.0..=1.0);
                    slider(ui, "Threshold", &mut s.bloom_threshold, 0.0..=5.0);
                }
            });
        }
        if image {
            CollapsingHeader::new("Image").default_open(true).show(ui, |ui| {
                ui.label(format!("View: {} spp · {}×{}", m.spp, m.size.0, m.size.1));
                ui.checkbox(&mut s.denoise, "Denoise");
                if s.denoise {
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut s.denoise_view, DenoiseView::Original, "L").on_hover_text("Show original image");
                        ui.selectable_value(&mut s.denoise_view, DenoiseView::Denoised, "R").on_hover_text("Show denoised image");
                        ui.selectable_value(&mut s.denoise_view, DenoiseView::Split, "L|R").on_hover_text("Show comparison");
                    });
                }
                ui.horizontal(|ui| {
                    ui.label("Size");
                    ui.add(egui::DragValue::new(&mut m.image.width).range(1..=12000)).on_hover_text("Image width [1 - 12000]");
                    ui.add(egui::DragValue::new(&mut m.image.height).range(1..=12000)).on_hover_text("Image height [1 - 12000]");
                });
                ui.horizontal(|ui| {
                    ui.label("Samples");
                    ui.add(egui::DragValue::new(&mut m.image.spp).range(1..=65536));
                });
                ui.horizontal(|ui| {
                    command(ui, cmds, "Render", "Final render to export/", Command::RenderImage);
                    command(ui, cmds, "Snapshot", "Save the view as it is [6]", Command::Snapshot);
                });
            });
        }
    });
    a
}

// ------------------------------------------------------------ editor panel


/// What the editor panel changed this frame.
#[derive(Default)]
pub struct EditActions {
    /// Palette or material edits that need uploading.
    pub materials_changed: bool,
}

fn command(ui: &mut Ui, cmds: &mut Vec<Command>, label: &str, hint: &str, cmd: Command) {
    let b = ui.button(label);
    let b = if hint.is_empty() { b } else { b.on_hover_text(hint) };
    if b.clicked() {
        cmds.push(cmd);
    }
}

const KINDS: [(MaterialKind, &str); 6] = [
    (MaterialKind::Diffuse, "Diffuse"),
    (MaterialKind::Metal, "Metal"),
    (MaterialKind::Glass, "Glass"),
    (MaterialKind::Emit, "Emit"),
    (MaterialKind::Blend, "Blend"),
    (MaterialKind::Media, "Cloud"),
];

fn material_editor(ui: &mut Ui, m: &mut GpuMaterial) {
    let name = KINDS.iter().find(|(k, _)| *k as u32 == m.kind).map(|(_, n)| *n).unwrap_or("?");
    ComboBox::from_label("Material").selected_text(name).show_ui(ui, |ui| {
        for (k, n) in KINDS {
            ui.selectable_value(&mut m.kind, k as u32, n);
        }
    });
    let kind = m.kind;
    let is = |k: MaterialKind| kind == k as u32;
    if is(MaterialKind::Metal) || is(MaterialKind::Glass) || is(MaterialKind::Blend) {
        slider(ui, "Roughness", &mut m.rough, 0.0..=1.0);
        slider(ui, "IOR", &mut m.ior, 1.0..=3.0);
    }
    if is(MaterialKind::Metal) || is(MaterialKind::Blend) {
        slider(ui, "Metallic", &mut m.metal, 0.0..=1.0);
    }
    if is(MaterialKind::Glass) || is(MaterialKind::Blend) {
        slider(ui, "Transparency", &mut m.trans, 0.0..=1.0);
    }
    if is(MaterialKind::Emit) {
        ui.add(Slider::new(&mut m.emit, 0.0..=100.0).logarithmic(true).text("Emission"));
    }
    if is(MaterialKind::Media) {
        slider(ui, "Density", &mut m.density, 0.0..=10.0);
        slider(ui, "Phase", &mut m.g, -0.9..=0.9);
    }
}

fn heading(ui: &mut Ui, text: &str) {
    ui.vertical_centered(|ui| ui.label(egui::RichText::new(text).strong()));
    ui.separator();
}

/// Top bar: mode tabs, project name, history and file commands.
/// `history` is whether undo and redo are possible.
pub fn top_bar(ui: &mut Ui, modeling: &mut bool, name: &str, history: (bool, bool), cmds: &mut Vec<Command>) {
    ui.horizontal(|ui| {
        ui.selectable_value(modeling, true, egui::RichText::new("Model").size(16.0)).on_hover_text("F1");
        ui.selectable_value(modeling, false, egui::RichText::new("Render").size(16.0)).on_hover_text("F1");
        ui.add_space(24.0);
        ui.label(egui::RichText::new(name).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            command(ui, cmds, "Preferences", "Preferences [Ctrl+,]", Command::Preferences);
            ui.separator();
            command(ui, cmds, "Save As", "Save as [Ctrl+Shift+S]", Command::SaveAs);
            command(ui, cmds, "Save", "Save [Ctrl+S]", Command::Save);
            command(ui, cmds, "Import", "Import another voxel format, a mesh, a heightmap, slices or a picture [Ctrl+Shift+O]", Command::ImportPick);
            command(ui, cmds, "Open", "Open [Ctrl+O]", Command::Open);
            command(ui, cmds, "New", "New [Ctrl+P]", Command::New);
            ui.separator();
            if ui.add_enabled(history.1, egui::Button::new("Redo")).on_hover_text("Redo [Ctrl+Y]").clicked() {
                cmds.push(Command::Redo);
            }
            if ui.add_enabled(history.0, egui::Button::new("Undo")).on_hover_text("Undo [Ctrl+Z]").clicked() {
                cmds.push(Command::Undo);
            }
        });
    });
}

/// The Preferences window. `autosave` is (on, edits between saves).
/// Returns true when a preference changed.
pub fn preferences_window(ctx: &egui::Context, open: &mut bool, prefs: &mut crate::prefs::Prefs, in_use: f32, autosave: &mut (bool, usize)) -> bool {
    let before = (prefs.clone(), *autosave);
    // Under the Preferences button, at the right of the top bar.
    let window = egui::Window::new("Preferences").open(open).collapsible(false).resizable(false);
    window.anchor(egui::Align2::RIGHT_TOP, [-8.0, 40.0]).show(ctx, |ui| {
        ui.label(egui::RichText::new("Interface scale").strong());
        ui.horizontal_wrapped(|ui| {
            for s in crate::prefs::SCALES {
                ui.selectable_value(&mut prefs.ui_scale, s, format!("{}%", (s * 100.0).round()));
            }
        });
        ui.label(egui::RichText::new("Bigger [Ctrl+=]  Smaller [Ctrl+-]  100% [Ctrl+0]").weak().small());
        if in_use < prefs.ui_scale - 1e-3 {
            ui.label(format!("Showing {}% so the panels fit this window; widen it for more.", (in_use * 100.0).round()));
        }
        ui.add_space(6.0);
        ui.label(egui::RichText::new("Autosave").strong());
        ui.horizontal(|ui| {
            ui.checkbox(&mut autosave.0, "Save the project every");
            ui.add_enabled(autosave.0, egui::DragValue::new(&mut autosave.1).range(1..=100_000));
            ui.label("edits");
        });
        ui.label(egui::RichText::new("Counts your edits and agents' alike; saves to the project's file").weak().small());
    });
    (prefs.clone(), *autosave) != before
}

/// What the bottom bar shows and changes.
pub struct Status<'a> {
    pub options: Option<&'a mut ModelerOptions>,
    /// Hovered voxel and its color index.
    pub hover: Option<(IVec3, u8)>,
    /// Shown instead of the hovered voxel while a transform gizmo is up.
    pub xform: Option<String>,
    pub projection: &'a mut Projection,
    pub iso: bool,
    /// Free camera: rotating turns the view around the eye.
    pub free: &'a mut bool,
    pub agent: &'a mut crate::agents::AgentState,
    /// The animation timeline is showing.
    pub timeline: &'a mut bool,
    pub hint: &'a str,
    /// The last action's message, shown instead of the hint.
    pub notice: &'a mut Option<crate::app::Notice>,
}

/// How long a message (not an error) stays in the status bar.
const NOTICE_SECS: u64 = 8;

/// Bottom bar: view toggles, hovered voxel, camera modes and a hint.
/// Returns true when the isometric view was asked for.
pub fn status_bar(ui: &mut Ui, s: Status, cmds: &mut Vec<Command>) -> bool {
    let mut iso = false;
    ui.horizontal(|ui| {
        if let Some(o) = s.options {
            ui.toggle_value(&mut o.others, "BO").on_hover_text("Display Background Objects [Ctrl+B]");
            ui.toggle_value(&mut o.edge, "Edge").on_hover_text("Display Edges [Ctrl+E]");
            ui.toggle_value(&mut o.grid, "Grid").on_hover_text("Display Grids [Ctrl+G]");
            ui.toggle_value(&mut o.frame, "Frame").on_hover_text("Display Frame [Ctrl+F]");
            ui.toggle_value(&mut o.ao, "AO").on_hover_text("Show Ambient Occlusion");
            ui.toggle_value(&mut o.shadow, "SW").on_hover_text("Display Shadow [Ctrl+W]");
            ui.toggle_value(&mut o.wire, "WR").on_hover_text("Display Wireframes [Ctrl+M]");
            ui.toggle_value(&mut o.canvas_box, "BB").on_hover_text("Display the canvas Bounding Box while editing");
            ui.separator();
        }
        let field = |v: Option<i32>| v.map_or("-".to_string(), |v| v.to_string());
        let (p, i) = match s.hover {
            Some((p, i)) => (Some(p), Some(i as i32)),
            None => (None, None),
        };
        match s.xform {
            Some(text) => ui.monospace(format!("{text:<28}")),
            None => ui.monospace(format!(
                "x:{:<4} y:{:<4} z:{:<4} i:{:<4}",
                field(p.map(|p| p.x)),
                field(p.map(|p| p.y)),
                field(p.map(|p| p.z)),
                field(i)
            )),
        };
        ui.separator();
        let pers = *s.projection == Projection::Perspective;
        if ui.selectable_label(pers && !*s.free, "Pers").on_hover_text("Perspective, orbiting the target").clicked() {
            *s.projection = Projection::Perspective;
            *s.free = false;
        }
        if ui.selectable_label(pers && *s.free, "Free").on_hover_text("Perspective, looking around from the eye [WASD/QE to move]").clicked() {
            *s.projection = Projection::Perspective;
            *s.free = true;
        }
        if ui.selectable_label(!pers && !s.iso && *s.projection == Projection::Orthographic, "Orth").clicked() {
            *s.projection = Projection::Orthographic;
        }
        iso = ui.selectable_label(s.iso, "Iso").on_hover_text("Isometric view").clicked();
        command(ui, cmds, "Reset", "Reset camera [F12]", Command::ResetCamera);
        command(ui, cmds, "Snap", "Snapshot [6]", Command::Snapshot);
        ui.separator();
        ui.toggle_value(s.timeline, "Anim").on_hover_text("Animation timeline");
        crate::agents::badge(ui, s.agent);
        ui.separator();
        if s.notice.as_ref().is_some_and(|n| !n.error && n.at.elapsed().as_secs() >= NOTICE_SECS) {
            *s.notice = None;
        }
        match s.notice {
            Some(n) => {
                if n.error && ui.small_button("x").on_hover_text("Dismiss").clicked() {
                    *s.notice = None;
                } else {
                    let text = egui::RichText::new(&n.text);
                    let text = if n.error { text.color(egui::Color32::from_rgb(255, 120, 120)) } else { text };
                    ui.add(egui::Label::new(text).truncate()).on_hover_text(&n.text);
                    if !n.error {
                        ui.ctx().request_repaint_after(std::time::Duration::from_secs(NOTICE_SECS));
                    }
                }
            }
            None => {
                ui.add(egui::Label::new(egui::RichText::new(s.hint).weak()).truncate()).on_hover_text(s.hint);
            }
        }
    });
    iso
}

/// Left column: quick palettes, recent colors, the swatches (drag one onto
/// another to swap, move, copy or blend; right-click for more), the color
/// editor and the ramp tools.
pub fn palette_panel(ui: &mut Ui, e: &mut Editor, scene: &mut Scene, pal: &mut PaletteState, quick: &[crate::palette_lib::Source], cmds: &mut Vec<Command>) -> EditActions {
    use crate::palette_lib::{self as lib, Cmd};
    let mut a = EditActions::default();
    heading(ui, "Palette");
    ui.horizontal(|ui| {
        // Four quick buttons, Lib and the menu share the column's width.
        ui.spacing_mut().item_spacing.x = 3.0;
        ui.spacing_mut().button_padding.x = 4.0;
        for (k, src) in quick.iter().enumerate() {
            let look = if pal.match_colors { ", keeping the model's look" } else { "" };
            let r = ui.small_button(k.to_string()).on_hover_text(format!("Load {}{look}", src.name()));
            if r.hovered() {
                pal.preview = Some(src.clone());
            }
            if r.clicked() {
                cmds.push(Command::Palette(Cmd::Load(src.clone(), pal.match_colors)));
            }
        }
        if ui.selectable_label(pal.library_open, "Lib").on_hover_text("Palette library: favorites, saved palettes, MagicaVoxel's and presets").clicked() {
            pal.library_open ^= true;
        }
        ui.menu_button("...", |ui| {
            ui.menu_button("Sort", |ui| {
                for k in crate::palette_ops::SortKey::ALL {
                    if ui.button(format!("By {}", k.name())).clicked() {
                        cmds.push(Command::Palette(Cmd::Sort(k)));
                        ui.close();
                    }
                }
            });
            if ui.button("Pack").on_hover_text("Used colors first from slot 1, duplicates merged; the model looks the same").clicked() {
                cmds.push(Command::Palette(Cmd::Pack));
                ui.close();
            }
            ui.checkbox(&mut pal.mask_unused, "Mask").on_hover_text("Mask unused voxel colors");
            if ui.button("Rand").on_hover_text("Random voxel colors from the selected palette colors").clicked() {
                cmds.push(Command::PaletteRandom);
                ui.close();
            }
            if ui.button("Inverse").on_hover_text("Inverse color selection").clicked() {
                for s in pal.selected.iter_mut().skip(1) {
                    *s = !*s;
                }
                ui.close();
            }
            if ui.button("Clear selection").clicked() {
                pal.selected.fill(false);
                ui.close();
            }
            ui.separator();
            if ui.add_enabled(crate::palette_lib::can_undo(scene, e), egui::Button::new("Undo palette")).on_hover_text("Undo the last palette change").clicked() {
                cmds.push(Command::Palette(Cmd::Undo));
                ui.close();
            }
        });
    });
    let swatch = 14.0;
    let paint = |ui: &Ui, rect: egui::Rect, c: [u8; 4]| ui.painter().rect_filled(rect, 1.0, egui::Color32::from_rgb(c[0], c[1], c[2]));
    // Recent colors, newest first.
    if pal.recent.first() != Some(&e.color) {
        let c = e.color;
        pal.recent.retain(|&x| x != c);
        pal.recent.insert(0, c);
        pal.recent.truncate(8);
    }
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 1.0;
        for c in pal.recent.clone() {
            let (rect, resp) = ui.allocate_exact_size(egui::vec2(swatch, swatch), egui::Sense::click());
            paint(ui, rect, scene.palette[c as usize]);
            if resp.clicked() {
                e.color = c;
            }
            resp.on_hover_text(format!("#{c} · recent"));
        }
    });
    let used = if pal.mask_unused { crate::palette_ops::mask_unused(&scene.palette, &scene.volume) } else { Vec::new() };
    // 8 columns like MagicaVoxel, which lists colors bottom-up: index 1 at
    // the bottom left, 255 near the top right. Each row is a ramp.
    let mut rects = [egui::Rect::NOTHING; 256];
    egui::Grid::new("palette").spacing([1.0, 1.0]).min_col_width(swatch).show(ui, |ui| {
        for k in 0..256usize {
            let (row, col) = (31 - k / 8, k % 8);
            let i = row * 8 + col + 1;
            if i > 255 {
                ui.allocate_exact_size(egui::vec2(swatch, swatch), egui::Sense::hover());
                ui.end_row();
                continue;
            }
            let p = scene.palette[i];
            let mut fill = egui::Color32::from_rgb(p[0], p[1], p[2]);
            if used.get(i) == Some(&false) {
                fill = fill.gamma_multiply(0.15);
            }
            let (rect, resp) = ui.allocate_exact_size(egui::vec2(swatch, swatch), egui::Sense::click_and_drag());
            rects[i] = rect;
            ui.painter().rect_filled(rect, 1.0, fill);
            if pal.selected[i] {
                ui.painter().rect_stroke(rect.shrink(1.0), 1.0, egui::Stroke::new(1.5, egui::Color32::from_rgb(80, 160, 255)), egui::StrokeKind::Inside);
            }
            if e.color as usize == i {
                ui.painter().rect_stroke(rect.expand(1.0), 1.0, egui::Stroke::new(2.0, egui::Color32::WHITE), egui::StrokeKind::Outside);
            }
            if resp.drag_started() {
                pal.drag = Some(i);
            }
            if resp.clicked() {
                // Shift/Ctrl+click toggles the color selection.
                let m = ui.input(|i| i.modifiers);
                if m.shift || m.command {
                    pal.selected[i] ^= true;
                } else {
                    e.color = i as u8;
                }
            }
            resp.context_menu(|ui| swatch_menu(ui, i, e, scene, pal, cmds));
            let ramp = scene.notes.get((i - 1) / 8).filter(|n| !n.is_empty()).map(|n| format!(" · {n}")).unwrap_or_default();
            resp.on_hover_text(format!("#{i}{ramp}\nShift+click: select · Right-click: more\nDrag onto another: swap · Ctrl: move · Ctrl+Shift: copy · Shift: blend"));
            if col == 7 {
                ui.end_row();
            }
        }
    });
    if let Some(from) = pal.drag {
        let pos = ui.ctx().pointer_interact_pos();
        let target = pos.and_then(|p| (1..256).find(|&i| rects[i].contains(p))).filter(|&t| t != from);
        let (cmd, verb) = drag_command(ui.input(|i| i.modifiers), from, target.unwrap_or(from));
        if ui.input(|i| i.pointer.primary_down()) {
            let painter = ui.ctx().layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::unique("palette_drag")));
            if let Some(p) = pos {
                painter.line_segment([rects[from].center(), p], egui::Stroke::new(2.0, egui::Color32::WHITE));
                let label = match target {
                    Some(t) => format!("{verb} {from} to {t}"),
                    None => verb.to_string(),
                };
                painter.text(p + egui::vec2(12.0, -12.0), egui::Align2::LEFT_BOTTOM, label, egui::FontId::proportional(13.0), egui::Color32::WHITE);
            }
            if let Some(t) = target {
                painter.rect_stroke(rects[t].expand(1.5), 1.0, egui::Stroke::new(2.0, egui::Color32::YELLOW), egui::StrokeKind::Outside);
            }
        } else {
            if target.is_some() {
                cmds.push(Command::Palette(cmd));
            }
            pal.drag = None;
        }
    }
    ui.separator();
    let i = e.color as usize;
    // One slider drag (or one typed code) is one palette undo step.
    if !ui.input(|i| i.pointer.any_down()) {
        pal.editing = None;
    }
    let mut rgb = [scene.palette[i][0], scene.palette[i][1], scene.palette[i][2]];
    let before = rgb;
    ui.spacing_mut().slider_width = 60.0;
    ui.horizontal(|ui| {
        ui.color_edit_button_srgb(&mut rgb);
        ui.label(format!("#{i}"));
        // One button that switches, so both fit the narrow column.
        let (mode, other) = if pal.rgb { ("RGB", "HSV") } else { ("HSV", "RGB") };
        if ui.button(mode).on_hover_text(format!("Sliders in {mode}; click for {other}")).clicked() {
            pal.rgb ^= true;
        }
    });
    if pal.rgb {
        for (k, n) in ["R", "G", "B"].iter().enumerate() {
            ui.add(Slider::new(&mut rgb[k], 0..=255).text(*n));
        }
    } else {
        let mut hsv = egui::ecolor::Hsva::from_srgb(rgb);
        let mut changed = false;
        changed |= ui.add(Slider::new(&mut hsv.h, 0.0..=1.0).text("H")).changed();
        changed |= ui.add(Slider::new(&mut hsv.s, 0.0..=1.0).text("S")).changed();
        changed |= ui.add(Slider::new(&mut hsv.v, 0.0..=1.0).text("V")).changed();
        if changed {
            rgb = hsv.to_srgb();
        }
    }
    ui.horizontal(|ui| {
        if ui.small_button("Copy").on_hover_text("Copy color code").clicked() {
            ui.ctx().copy_text(crate::palette_ops::color_code(scene.palette[i]));
        }
        let r = ui.add(egui::TextEdit::singleline(&mut pal.code).hint_text("#RRGGBB").desired_width(70.0)).on_hover_text("Paste a color code, then Enter");
        if r.lost_focus() && ui.input(|k| k.key_pressed(egui::Key::Enter)) {
            if let Some(c) = crate::palette_ops::parse_color_code(&pal.code) {
                rgb = [c[0], c[1], c[2]];
            }
        }
    });
    if rgb != before {
        lib::drag_step(scene, e, &mut pal.step, pal.editing != Some(i));
        pal.editing = Some(i);
        scene.look[i] = None;
        scene.palette[i] = [rgb[0], rgb[1], rgb[2], scene.palette[i][3]];
        scene.sync_color(i);
        scene.palette_step_update(None);
        a.materials_changed = true;
    }
    a.materials_changed |= ramp_tools(ui, i, e, scene, pal, cmds);
    if let Some((msg, t)) = &pal.status {
        if t.elapsed().as_secs() < 6 {
            ui.add(egui::Label::new(egui::RichText::new(msg).weak().small()).wrap());
        }
    }
    a
}

/// What dropping swatch `from` on `to` does with these modifiers, and its
/// name: swap (plain or Alt, MagicaVoxel's Alt+drag), move (Ctrl), copy
/// (Ctrl+Shift) or blend the slots between (Shift, or MagicaVoxel's
/// Ctrl+Alt+Shift).
fn drag_command(m: egui::Modifiers, from: usize, to: usize) -> (crate::palette_lib::Cmd, &'static str) {
    use crate::palette_lib::Cmd;
    if m.shift && (m.alt || !m.command) {
        (Cmd::Gradient(from, to), "Blend")
    } else if m.command && m.shift {
        (Cmd::Copy(from, to), "Copy")
    } else if m.command {
        (Cmd::Move(from, to), "Move")
    } else {
        (Cmd::Swap(from, to), "Swap")
    }
}

/// The first and last color of the row (ramp) holding color `i`.
fn ramp_of(i: usize) -> (usize, usize) {
    let lo = (i - 1) / 8 * 8 + 1;
    (lo, (lo + 7).min(255))
}

/// Edit the name of the ramp holding color `i`.
fn ramp_name(ui: &mut Ui, i: usize, scene: &mut Scene) {
    let row = (i - 1) / 8;
    let (lo, hi) = ramp_of(i);
    let mut name = scene.notes.get(row).cloned().unwrap_or_default();
    let r = ui.add(egui::TextEdit::singleline(&mut name).hint_text(format!("ramp {lo}-{hi}")).desired_width(100.0));
    if r.on_hover_text("Name this row of colors (saved in the .vox like MagicaVoxel's palette notes)").changed() {
        if scene.notes.len() < 32 {
            scene.notes.resize(32, String::new());
        }
        scene.notes[row] = name;
    }
}

/// The ramp row under the color editor: its name, Blend, and a hue /
/// saturation / value shift for the selected colors (or the ramp).
/// Returns whether colors changed.
fn ramp_tools(ui: &mut Ui, i: usize, e: &mut Editor, scene: &mut Scene, pal: &mut PaletteState, cmds: &mut Vec<Command>) -> bool {
    use crate::palette_lib::{self as lib, Cmd};
    let (lo, hi) = ramp_of(i);
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("Ramp");
        ramp_name(ui, i, scene);
    });
    let picked: Vec<usize> = (1..256).filter(|&k| pal.selected[k]).collect();
    let (targets, what) = if picked.is_empty() { ((lo..=hi).collect::<Vec<_>>(), format!("ramp {lo}-{hi}")) } else { (picked.clone(), format!("{} selected", picked.len())) };
    ui.horizontal(|ui| {
        if ui.small_button("Blend").on_hover_text(format!("Blend {lo}-{hi} evenly from the first color to the last")).clicked() {
            cmds.push(Command::Palette(Cmd::Gradient(lo, hi)));
        }
        ui.label(egui::RichText::new(format!("Shift {what}")).weak().small());
    });
    let mut changed = false;
    let mut done = false;
    for (k, n) in ["H", "S", "V"].into_iter().enumerate() {
        let range = if k == 0 { -0.5..=0.5 } else { -1.0..=1.0 };
        let r = ui.add(Slider::new(&mut pal.shift[k], range).text(n).show_value(false)).on_hover_text(match k {
            0 => "Turn the hue",
            1 => "More or less saturated",
            _ => "Lighter or darker",
        });
        changed |= r.changed();
        done |= r.drag_stopped() || (r.changed() && !r.dragged());
    }
    if changed {
        let fresh = pal.shift_base.as_ref().is_none_or(|b| !b.iter().map(|x| x.0).eq(targets.iter().copied()));
        lib::drag_step(scene, e, &mut pal.step, fresh);
        if fresh {
            lib::forget(&mut scene.look, targets.iter().copied());
            pal.shift_base = Some(targets.iter().map(|&t| (t, scene.palette[t])).collect());
        }
        let [dh, ds, dv] = pal.shift;
        for &(t, c) in pal.shift_base.iter().flatten() {
            scene.palette[t] = crate::palette_ops::adjust(c, dh, ds, dv);
            scene.sync_color(t);
        }
        scene.palette_step_update(None);
    }
    if done {
        pal.shift = [0.0; 3];
        pal.shift_base = None;
    }
    changed
}

/// Right-click menu of swatch `i`.
fn swatch_menu(ui: &mut Ui, i: usize, e: &mut Editor, scene: &mut Scene, pal: &mut PaletteState, cmds: &mut Vec<Command>) {
    use crate::palette_lib::Cmd;
    let code = crate::palette_ops::color_code(scene.palette[i]);
    ui.label(egui::RichText::new(format!("#{i}  {code}")).strong());
    let mut item = |ui: &mut Ui, label: &str, cmd: Cmd| {
        if ui.button(label).clicked() {
            cmds.push(Command::Palette(cmd));
            ui.close();
        }
    };
    item(ui, "Select its voxels", Cmd::SelectVoxels(i, crate::edit::SelMode::Replace));
    item(ui, "Add its voxels to the selection", Cmd::SelectVoxels(i, crate::edit::SelMode::Add));
    item(ui, "Deselect its voxels", Cmd::SelectVoxels(i, crate::edit::SelMode::Remove));
    item(ui, "Delete its voxels", Cmd::RemoveVoxels(i));
    ui.separator();
    if ui.button("Draw with it").clicked() {
        e.color = i as u8;
        ui.close();
    }
    if ui.button(if pal.selected[i] { "Unselect swatch" } else { "Select swatch" }).clicked() {
        pal.selected[i] ^= true;
        ui.close();
    }
    if ui.button("Copy color code").clicked() {
        ui.ctx().copy_text(code);
        ui.close();
    }
    ui.separator();
    ramp_name(ui, i, scene);
    let (lo, hi) = ramp_of(i);
    item(ui, &format!("Blend ramp {lo}-{hi}"), Cmd::Gradient(lo, hi));
}

fn axis_row(ui: &mut Ui, cmds: &mut Vec<Command>, hint: &str, held: bool, op: fn(usize) -> Op) {
    ui.horizontal(|ui| {
        for (axis, n) in ["X", "Y", "Z"].iter().enumerate() {
            if held {
                command_repeat(ui, cmds, n, hint, Command::Op(op(axis)));
            } else {
                command(ui, cmds, n, hint, Command::Op(op(axis)));
            }
        }
    });
}

/// A button that repeats while held, like MagicaVoxel's Loop, Mirror,
/// Diagonal, Dilate and Erode buttons.
fn command_repeat(ui: &mut Ui, cmds: &mut Vec<Command>, label: &str, hint: &str, cmd: Command) {
    const DELAY: f64 = 0.4;
    const EVERY: f64 = 0.1;
    let b = ui.button(label);
    let b = if hint.is_empty() { b } else { b.on_hover_text(hint) };
    let id = b.id.with("repeat");
    let now = ui.input(|i| i.time);
    let state: Option<(f64, u32)> = ui.data(|d| d.get_temp(id));
    if b.is_pointer_button_down_on() {
        let (start, fired) = state.unwrap_or((now, 0));
        let due = ((now - start - DELAY) / EVERY).floor() as i64 + 1;
        let fired = if due > fired as i64 {
            cmds.push(cmd);
            fired + 1
        } else {
            fired
        };
        ui.data_mut(|d| d.insert_temp(id, (start, fired)));
        ui.ctx().request_repaint();
    } else {
        if state.is_some() {
            ui.data_mut(|d| d.remove::<(f64, u32)>(id));
        }
        // A plain click; a hold already fired.
        if b.clicked() && state.is_none_or(|(_, fired)| fired == 0) {
            cmds.push(cmd);
        }
    }
}

/// Which sections the Edit panel shows (MagicaVoxel's All / Transform /
/// Modify tabs).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum EditTab {
    #[default]
    All,
    Transform,
    Modify,
}

/// Right column in the modeler: MagicaVoxel's Edit panel. With a
/// selection, the edits act on the selected voxels (see `app::apply_edit`).
pub fn edit_panel(ui: &mut Ui, e: &Editor, text: &mut TransformText, cmds: &mut Vec<Command>) {
    use crate::ops::Shape;
    heading(ui, "Edit");
    ui.horizontal(|ui| {
        ui.selectable_value(&mut text.tab, EditTab::All, "All").on_hover_text("Show All Options");
        ui.selectable_value(&mut text.tab, EditTab::Transform, "Trans").on_hover_text("Show Transform Options");
        ui.selectable_value(&mut text.tab, EditTab::Modify, "Mod").on_hover_text("Show Modification Options");
    });
    let tab = text.tab;
    egui::ScrollArea::vertical().show(ui, |ui| {
        if tab == EditTab::All {
            CollapsingHeader::new("Tool").default_open(true).show(ui, |ui| {
                egui::Grid::new("tool_ops").num_columns(2).show(ui, |ui| {
                    command(ui, cmds, "Full", "Full Volume [U]", Command::Op(Op::Full));
                    command(ui, cmds, "Fill", "Fill Voxels with Same Color [I]", Command::Op(Op::Fill));
                    ui.end_row();
                    command(ui, cmds, "Clear", "Clear Voxels [Backspace/Delete]", Command::Op(Op::Clear));
                    command(ui, cmds, "Rev", "Reverse Empty Space", Command::Op(Op::Reverse));
                    ui.end_row();
                    command(ui, cmds, "2X", "Double Size", Command::Op(Op::Double));
                    command(ui, cmds, "1/2", "Half Size", Command::Op(Op::Half));
                    ui.end_row();
                });
            });
            CollapsingHeader::new("Select").default_open(true).show(ui, |ui| {
                ui.weak(match e.selected_count() {
                    0 => "nothing selected".to_string(),
                    n => format!("{n} selected"),
                });
                egui::Grid::new("select_ops").num_columns(2).show(ui, |ui| {
                    command(ui, cmds, "All", "Select All [Ctrl+A]", Command::SelectAll);
                    command(ui, cmds, "Inv", "Select Inverse [Ctrl+I]", Command::SelectInvert);
                    ui.end_row();
                    command(ui, cmds, "None", "Select None [Ctrl+D]", Command::SelectNone);
                    ui.end_row();
                    command(ui, cmds, "Copy", "Copy Voxels [Ctrl+C]", Command::Copy);
                    command(ui, cmds, "Cut", "Cut Voxels [Ctrl+X]", Command::Cut);
                    ui.end_row();
                    if ui.add_enabled(e.has_clipboard(), egui::Button::new("Paste")).on_hover_text("Paste Voxels [Ctrl+V]").clicked() {
                        cmds.push(Command::Paste);
                    }
                    ui.end_row();
                });
            });
            ui.separator();
        }
        if tab != EditTab::Modify {
            CollapsingHeader::new("Rot").show(ui, |ui| axis_row(ui, cmds, "Rot : Rotate Model 90 Degrees", false, Op::Rotate));
            CollapsingHeader::new("Flip").show(ui, |ui| axis_row(ui, cmds, "Flip : Flip Model", false, Op::Flip));
            CollapsingHeader::new("Loop").show(ui, |ui| {
                for dir in [1, -1] {
                    ui.horizontal(|ui| {
                        for (axis, n) in ["X", "Y", "Z"].iter().enumerate() {
                            let label = format!("{}{n}", if dir > 0 { "+" } else { "-" });
                            command_repeat(ui, cmds, &label, "Loop : Move Model", Command::Op(Op::Loop(axis, dir)));
                        }
                    });
                }
            });
            for (label, hint, kind) in [
                ("Scale", "Scale Model : [Axis] [Scale]", TransformKind::Scale),
                ("Shear", "Shear Model : [Axis] [Scale]", TransformKind::Shear),
                ("Rotate", "Rotate Model : [Axis] [Angle]", TransformKind::Rotate),
                ("Repeat", "Repeat Model : [Axis] [Scale]", TransformKind::Repeat),
            ] {
                let field = match kind {
                    TransformKind::Scale => &mut text.scale,
                    TransformKind::Shear => &mut text.shear,
                    TransformKind::Rotate => &mut text.rotate,
                    TransformKind::Repeat => &mut text.repeat,
                };
                CollapsingHeader::new(label).show(ui, |ui| {
                    let r = ui.add(egui::TextEdit::singleline(field).desired_width(110.0)).on_hover_text(format!("{hint} : apply [Enter]"));
                    if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        cmds.push(Command::Transform(kind, field.clone()));
                    }
                });
            }
        }
        if tab == EditTab::All {
            ui.separator();
        }
        if tab != EditTab::Transform {
            CollapsingHeader::new("Mirror").show(ui, |ui| axis_row(ui, cmds, "Mirror", true, Op::Mirror));
            CollapsingHeader::new("Diagonal").show(ui, |ui| axis_row(ui, cmds, "Diagonal", true, Op::Diagonal));
            CollapsingHeader::new("Shape").show(ui, |ui| {
                egui::Grid::new("shape_ops").num_columns(2).show(ui, |ui| {
                    command(ui, cmds, "Elli", "Elli : Ellipsoid", Command::Op(Op::Shape(Shape::Ellipsoid)));
                    command(ui, cmds, "Cyli", "Cyli : Cylinder", Command::Op(Op::Shape(Shape::Cylinder)));
                    ui.end_row();
                    command(ui, cmds, "Pyra", "Pyra : Pyramid", Command::Op(Op::Shape(Shape::Pyramid)));
                    command(ui, cmds, "Cone", "Cone : Cone", Command::Op(Op::Shape(Shape::Cone)));
                    ui.end_row();
                    command(ui, cmds, "Maze", "Maze : Maze", Command::Op(Op::Maze));
                    command(ui, cmds, "Perlin", "Perlin : Perlin Noise", Command::Op(Op::Perlin));
                    ui.end_row();
                    command(ui, cmds, "Fractal", "Fractal : Fractal Model with Patterns", Command::Op(Op::Fractal));
                    ui.end_row();
                });
            });
            CollapsingHeader::new("Modify").show(ui, |ui| {
                egui::Grid::new("modify_ops").num_columns(2).show(ui, |ui| {
                    command(ui, cmds, "Hull", "Hull : Remove All Invisible Voxels", Command::Op(Op::Hull));
                    command(ui, cmds, "Flood", "Flood : Flood Fill All Invisible Space", Command::Op(Op::Flood));
                    ui.end_row();
                    command(ui, cmds, "Mask", "Mask : Mask Model with Patterns", Command::Op(Op::Mask));
                    command(ui, cmds, "Tex", "Tex : Texture Model with Patterns", Command::Op(Op::Tex));
                    ui.end_row();
                    command_repeat(ui, cmds, "Dilate", "Dilat : Dilation", Command::Op(Op::Dilate));
                    command_repeat(ui, cmds, "Erode", "Erode : Erosion", Command::Op(Op::Erode));
                    ui.end_row();
                });
            });
        }
    });
}

/// Which sections the Brush panel shows (MagicaVoxel's All / Brush /
/// Display tabs).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum BrushTab {
    #[default]
    All,
    Brush,
    Display,
}

/// Second left column in the modeler, laid out like MagicaVoxel's: mode
/// and tool grids, mirror and axis rows, then fold-out sections for the
/// current tool and mode and the modeler display options.
pub fn brush_panel(ui: &mut Ui, e: &mut Editor, display: Option<&mut ModelerOptions>, cmds: &mut Vec<Command>) {
    heading(ui, "Brush");
    let tab_id = ui.make_persistent_id("brush_tab");
    let mut tab: BrushTab = ui.data(|d| d.get_temp(tab_id)).unwrap_or_default();
    let w = ui.available_width();
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        let cw = ((w - 16.0) / 3.0).floor();
        let s = egui::vec2(cw, 18.0);
        ui.horizontal(|ui| {
            cell(ui, s, &mut tab, BrushTab::All, "All", "Show All Options");
            icon_cell(ui, s, &mut tab, BrushTab::Brush, Icon::Brush, "Show Brush Options");
            icon_cell(ui, s, &mut tab, BrushTab::Display, Icon::Gear, "Show Display Options");
        });
    });
    ui.data_mut(|d| d.insert_temp(tab_id, tab));
    if tab != BrushTab::Display {
        brush_grids(ui, e);
    }
    ui.add_space(4.0);
    egui::ScrollArea::vertical().show(ui, |ui| {
        if tab != BrushTab::Display {
            brush_options(ui, e, cmds);
        }
        if tab != BrushTab::Brush {
            if let Some(o) = display {
                display_options(ui, o);
            }
        }
    });
}

/// Size of one cell in a grid of `n` columns across the panel.
fn cell_size(ui: &Ui, n: f32, h: f32) -> egui::Vec2 {
    egui::vec2(((ui.available_width() - 2.0 * (n - 1.0)) / n).floor(), h)
}

/// A selectable cell: `v` becomes `val` when clicked.
fn cell<T: PartialEq + Copy>(ui: &mut Ui, size: egui::Vec2, v: &mut T, val: T, label: impl Into<egui::WidgetText>, hint: &str) {
    if ui.add_sized(size, egui::Button::selectable(*v == val, label)).on_hover_text(hint).clicked() {
        *v = val;
    }
}

/// The Brush panel's icons, drawn with the painter since the bundled fonts
/// lack most of these shapes.
#[derive(Clone, Copy)]
enum Icon {
    Brush,
    Gear,
    Dot,
    Star,
    Square,
    Extrude,
    Cube,
    Move,
    Marquee,
    Wand,
    Dropper,
    Trash,
    Bucket,
    Mirror,
    Axis,
    Line,
}

/// Draws `icon` centered at `c`, about `2 * r` across.
fn paint_icon(p: &egui::Painter, c: egui::Pos2, r: f32, col: egui::Color32, icon: Icon) {
    use egui::{vec2, Shape, Stroke};
    let s = Stroke::new(1.5, col);
    let at = |x: f32, y: f32| c + vec2(x, y) * r;
    let line = |a: (f32, f32), b: (f32, f32)| p.line_segment([at(a.0, a.1), at(b.0, b.1)], s);
    // An arrow from `a` to `b` with a head at `b`.
    let arrow = |a: (f32, f32), b: (f32, f32)| {
        line(a, b);
        let d = vec2(b.0 - a.0, b.1 - a.1).normalized() * 0.45;
        let n = d.rot90();
        line(b, (b.0 - d.x + n.x, b.1 - d.y + n.y));
        line(b, (b.0 - d.x - n.x, b.1 - d.y - n.y));
    };
    match icon {
        Icon::Brush => {
            p.line_segment([at(0.9, -0.9), at(-0.1, 0.1)], Stroke::new(2.0, col));
            p.circle_filled(at(-0.45, 0.45), r * 0.4, col);
        }
        Icon::Gear => {
            for k in 0..8 {
                let a = k as f32 * std::f32::consts::FRAC_PI_4;
                let (y, x) = a.sin_cos();
                p.line_segment([at(x * 0.5, y * 0.5), at(x * 0.95, y * 0.95)], Stroke::new(2.5, col));
            }
            p.circle_stroke(c, r * 0.6, Stroke::new(2.0, col));
        }
        Icon::Dot => {
            p.circle_filled(c, r * 0.6, col);
        }
        Icon::Star => {
            let pt = |k: usize| {
                let a = -std::f32::consts::FRAC_PI_2 + k as f32 * std::f32::consts::PI / 5.0;
                let d = if k % 2 == 0 { 1.0 } else { 0.42 };
                at(a.cos() * d, a.sin() * d + 0.08)
            };
            for k in 0..10 {
                p.add(Shape::convex_polygon(vec![c + vec2(0.0, 0.08 * r), pt(k), pt(k + 1)], col, Stroke::NONE));
            }
        }
        Icon::Square => {
            p.rect_filled(egui::Rect::from_center_size(c, vec2(1.0, 1.0) * r), 0.0, col);
        }
        Icon::Extrude => {
            line((-0.9, 0.8), (0.9, 0.8));
            arrow((0.0, 0.5), (0.0, -0.9));
        }
        Icon::Cube => {
            let h = |x: f32, y: f32| at(x * 0.87, y);
            let top = vec![h(0.0, -1.0), h(1.0, -0.5), h(0.0, 0.0), h(-1.0, -0.5)];
            p.add(Shape::convex_polygon(top, col, Stroke::NONE));
            p.add(Shape::closed_line(vec![h(0.0, -1.0), h(1.0, -0.5), h(1.0, 0.5), h(0.0, 1.0), h(-1.0, 0.5), h(-1.0, -0.5)], s));
            p.line_segment([h(0.0, 0.0), h(0.0, 1.0)], s);
        }
        Icon::Move => {
            for (x, y) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
                arrow((x * 0.15, y * 0.15), (x * 0.9, y * 0.9));
            }
        }
        Icon::Marquee => {
            for (x, y) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
                line((x * 0.85, y * 0.85), (x * 0.4, y * 0.85));
                line((x * 0.85, y * 0.85), (x * 0.85, y * 0.4));
            }
        }
        Icon::Wand => {
            p.line_segment([at(-0.9, 0.9), at(0.3, -0.3)], Stroke::new(2.0, col));
            for (x, y) in [(0.55, -0.85), (0.85, -0.55), (0.2, -0.85), (0.85, -0.2)] {
                p.circle_filled(at(x, y), 1.2, col);
            }
        }
        Icon::Dropper => {
            p.line_segment([at(-0.85, 0.85), at(0.2, -0.2)], Stroke::new(2.5, col));
            p.circle_filled(at(0.45, -0.45), r * 0.4, col);
        }
        Icon::Trash => {
            p.add(Shape::convex_polygon(vec![at(-0.6, -0.45), at(0.6, -0.45), at(0.5, 0.95), at(-0.5, 0.95)], col, Stroke::NONE));
            line((-0.85, -0.7), (0.85, -0.7));
            line((-0.25, -0.95), (0.25, -0.95));
        }
        Icon::Bucket => {
            p.add(Shape::closed_line(vec![at(-0.2, -0.95), at(0.6, -0.15), at(0.0, 0.45), at(-0.8, -0.35)], s));
            p.add(Shape::convex_polygon(vec![at(-0.8, -0.35), at(0.6, -0.15), at(0.0, 0.45)], col, Stroke::NONE));
            p.circle_filled(at(0.75, 0.6), r * 0.25, col);
        }
        Icon::Mirror => {
            line((0.0, -0.9), (0.0, 0.9));
            p.add(Shape::convex_polygon(vec![at(-0.9, -0.5), at(-0.25, 0.0), at(-0.9, 0.5)], col, Stroke::NONE));
            p.add(Shape::convex_polygon(vec![at(0.9, -0.5), at(0.25, 0.0), at(0.9, 0.5)], col, Stroke::NONE));
        }
        Icon::Axis => {
            arrow((-0.2, 0.2), (0.85, -0.85));
            arrow((0.2, -0.2), (-0.85, 0.85));
        }
        Icon::Line => {
            line((-0.85, 0.85), (0.85, -0.85));
        }
    }
}

/// An empty selectable button with `icon` drawn on it.
fn icon_button(ui: &mut Ui, size: egui::Vec2, selected: bool, icon: Icon, hint: &str) -> egui::Response {
    let b = ui.add_sized(size, egui::Button::selectable(selected, "")).on_hover_text(hint);
    let col = ui.style().interact_selectable(&b, selected).fg_stroke.color;
    let r = (size.y * 0.32).min(size.x * 0.32);
    paint_icon(ui.painter(), b.rect.center(), r, col, icon);
    b
}

/// A selectable icon cell: `v` becomes `val` when clicked.
fn icon_cell<T: PartialEq + Copy>(ui: &mut Ui, size: egui::Vec2, v: &mut T, val: T, icon: Icon, hint: &str) {
    if icon_button(ui, size, *v == val, icon, hint).clicked() {
        *v = val;
    }
}

/// The fixed top of the Brush panel: modes, tools, mirror and axis.
fn brush_grids(ui: &mut Ui, e: &mut Editor) {
    let grid = |ui: &mut Ui, id: &str, add: &mut dyn FnMut(&mut Ui)| {
        egui::Frame::group(ui.style()).inner_margin(2.0).show(ui, |ui| {
            ui.spacing_mut().item_spacing = egui::vec2(2.0, 2.0);
            ui.push_id(id, |ui| add(ui));
        });
    };
    grid(ui, "modes", &mut |ui| {
        let s = cell_size(ui, 3.0, 22.0);
        let b = &mut e.brush;
        ui.horizontal(|ui| {
            icon_cell(ui, s, b, Brush::Geo, Icon::Dot, "Geometry mode [L]: line, square, circle");
            cell(ui, s, b, Brush::Code, "</>", "Voxel shader mode [C]");
            icon_cell(ui, s, b, Brush::Pattern, Icon::Star, "Pattern mode [P] : Rot [-/+/0/9] : Move [Dir/Page/Home]");
        });
        ui.horizontal(|ui| {
            icon_cell(ui, s, b, Brush::Voxel, Icon::Square, "Voxel mode [V] : Resize [-/+/Ctrl+Alt+Drag]");
            icon_cell(ui, s, b, Brush::Face, Icon::Extrude, "Face mode [F] : Extrude [Drag]");
            icon_cell(ui, s, b, Brush::Box, Icon::Cube, "Box mode [B]");
        });
    });
    grid(ui, "tools", &mut |ui| {
        let s = cell_size(ui, 1.0, 26.0);
        let t = &mut e.tool;
        cell(ui, s, t, Tool::Attach, "Attach", "Attach [T]");
        cell(ui, s, t, Tool::Erase, "Erase", "Erase [R]");
        cell(ui, s, t, Tool::Paint, "Paint", "Paint [G]");
    });
    grid(ui, "select_tools", &mut |ui| {
        let s = cell_size(ui, 3.0, 22.0);
        let t = &mut e.tool;
        ui.horizontal(|ui| {
            icon_cell(ui, s, t, Tool::Move, Icon::Move, "Transform: move, scale, wrap or rotate the selection, or the model");
            icon_cell(ui, s, t, Tool::Marquee, Icon::Marquee, "Marquee select [M] : Add [Shift] : Remove [Shift+Alt]");
            icon_cell(ui, s, t, Tool::Region, Icon::Wand, "Region select [N] : Add [Shift] : Remove [Shift+Alt]");
        });
        ui.horizontal(|ui| {
            icon_cell(ui, s, t, Tool::Pick, Icon::Dropper, "Pick voxel color [Alt+LButton]");
            icon_cell(ui, s, t, Tool::RegionRemove, Icon::Trash, "Region remove");
            icon_cell(ui, s, t, Tool::RegionReplace, Icon::Bucket, "Region replace with the current color");
        });
    });
    grid(ui, "mirror_axis", &mut |ui| {
        let s = cell_size(ui, 4.0, 20.0);
        // The row's icon turns the whole row off.
        let row = |ui: &mut Ui, icon: Icon, hint: &str, on: &mut [bool; 3], keys: bool| {
            ui.horizontal(|ui| {
                if icon_button(ui, s, false, icon, &format!("{hint} : Off [Click]")).clicked() {
                    *on = [false; 3];
                }
                for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
                    let tip = if keys { format!("{hint} {n} [F{}]", 5 + i) } else { format!("{hint} {n}") };
                    if ui.add_sized(s, egui::Button::selectable(on[i], *n)).on_hover_text(tip).clicked() {
                        on[i] ^= true;
                    }
                }
            });
        };
        row(ui, Icon::Mirror, "Mirror mode", &mut e.mirror, true);
        row(ui, Icon::Axis, "Axis mode: extend the brush through the canvas", &mut e.opts.axis, false);
    });
}

/// A fold-out section of the Brush panel, open by default, with optional
/// controls on the right of its header.
fn brush_section(ui: &mut Ui, title: &str, right: impl FnOnce(&mut Ui), body: impl FnOnce(&mut Ui)) {
    let id = ui.make_persistent_id(("brush_section", title));
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true)
        .show_header(ui, |ui| {
            ui.label(egui::RichText::new(title.to_uppercase()).small().strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), right);
        })
        .body(body);
}

/// A framed box of choices, `n` to a row.
fn choice_box(ui: &mut Ui, add: impl FnOnce(&mut Ui)) {
    egui::Frame::group(ui.style()).inner_margin(2.0).show(ui, |ui| {
        ui.spacing_mut().item_spacing = egui::vec2(2.0, 2.0);
        ui.set_width(ui.available_width());
        add(ui);
    });
}

/// A row of two choices for `v`, each half the box wide.
fn pick2<T: PartialEq + Copy>(ui: &mut Ui, v: &mut T, a: (T, &str, &str), b: (T, &str, &str)) {
    let s = cell_size(ui, 2.0, 20.0);
    ui.horizontal(|ui| {
        cell(ui, s, v, a.0, a.1, a.2);
        cell(ui, s, v, b.0, b.1, b.2);
    });
}

/// The sections for the current tool and brush mode.
fn brush_options(ui: &mut Ui, e: &mut Editor, cmds: &mut Vec<Command>) {
    match e.tool {
        Tool::Move => brush_section(ui, "Transform", |_| {}, |ui| {
            choice_box(ui, |ui| {
                let s = cell_size(ui, 4.0, 20.0);
                let x = &mut e.opts.xform;
                ui.horizontal(|ui| {
                    cell(ui, s, x, XformMode::Move, "Move", "Move/Loop [Ctrl+LButton] : Normal axis [Shift]");
                    cell(ui, s, x, XformMode::Scale, "Scale", "Scale : Commit [Enter] : Cancel [Esc] : Two-sided [Shift]");
                    cell(ui, s, x, XformMode::Wrap, "Wrap", "Wrap/Crop : Commit [Enter] : Cancel [Esc] : Two-sided [Shift]");
                    cell(ui, s, x, XformMode::Rotate, "Rot", "Rotate : Commit [Enter] : Cancel [Esc] : Snap [Shift]");
                });
            });
            if e.opts.xform == XformMode::Wrap {
                ui.horizontal(|ui| {
                    ui.label("Spacing");
                    for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
                        ui.add(egui::DragValue::new(&mut e.opts.wrap_space[i]).range(0..=256)).on_hover_text(format!("Wrap spacing : {n}"));
                    }
                });
            }
            if e.xform_pending() {
                ui.weak("Commit [Enter] : Cancel [Esc]");
            }
        }),
        Tool::Marquee => brush_section(ui, "Marquee", |_| {}, |ui| {
            choice_box(ui, |ui| {
                let s = cell_size(ui, 3.0, 20.0);
                let m = &mut e.opts.marquee;
                ui.horizontal(|ui| {
                    cell(ui, s, m, MarqueeMode::Rect, "Rect", "Rect select");
                    cell(ui, s, m, MarqueeMode::Brush, "Brush", "Brush select (voxel brush size)");
                    cell(ui, s, m, MarqueeMode::Box, "Box", "Box select");
                });
            });
            if e.opts.marquee == MarqueeMode::Rect {
                ui.checkbox(&mut e.opts.visible_only, "Visible only").on_hover_text("Skip voxels hidden behind others");
            }
        }),
        Tool::Region | Tool::RegionRemove | Tool::RegionReplace => brush_section(ui, "Region", |_| {}, |ui| {
            let r = &mut e.region;
            choice_box(ui, |ui| {
                pick2(ui, &mut r.same_color, (true, "Same", "Same palette color, anywhere"), (false, "Connected", "Connected region"));
                if !r.same_color {
                    pick2(ui, &mut r.by_color, (true, "Color", "Connect through the same color"), (false, "Geo", "Connect through any voxel"));
                    pick2(ui, &mut r.diagonal, (false, "4", "Connect through faces"), (true, "8", "Also connect through edges and corners"));
                }
            });
        }),
        _ => {}
    }
    let o = &mut e.opts;
    match e.brush {
        Brush::Voxel => brush_section(
            ui,
            "Voxel",
            |ui| {
                ui.add(egui::DragValue::new(&mut o.size).range(1..=64)).on_hover_text("Voxel size [1 - 64] : Change size [-/+/Ctrl+Alt+Drag]");
            },
            |ui| {
                choice_box(ui, |ui| {
                    let s = cell_size(ui, 2.0, 20.0);
                    ui.horizontal(|ui| {
                        icon_cell(ui, s, &mut o.sphere, false, Icon::Square, "Cube shape");
                        icon_cell(ui, s, &mut o.sphere, true, Icon::Dot, "Sphere shape");
                    });
                    pick2(ui, &mut o.flat, (true, "2D", "Flat on the face under the cursor"), (false, "3D", "Solid"));
                });
            },
        ),
        Brush::Face => brush_section(ui, "Face", |_| {}, |ui| {
            choice_box(ui, |ui| {
                pick2(ui, &mut o.face_voxel_color, (false, "Palette", "Use palette color"), (true, "Voxel", "Use voxel color"));
                pick2(ui, &mut o.face_by_color, (true, "Color", "Color connectivity"), (false, "Geo", "Geometry connectivity"));
                pick2(ui, &mut o.face_diagonal, (false, "4", "4 neighbor searching"), (true, "8", "8 neighbor searching"));
            });
        }),
        Brush::Geo => brush_section(ui, "Geometry", |_| {}, |ui| {
            choice_box(ui, |ui| {
                let s = cell_size(ui, 3.0, 22.0);
                ui.horizontal(|ui| {
                    icon_cell(ui, s, &mut o.geo, GeoShape::Line, Icon::Line, "Line");
                    icon_cell(ui, s, &mut o.geo, GeoShape::Square, Icon::Square, "Square");
                    icon_cell(ui, s, &mut o.geo, GeoShape::Circle, Icon::Dot, "Circle");
                });
            });
            if o.geo == GeoShape::Line {
                ui.checkbox(&mut o.project, "Project").on_hover_text("Project the line on the surface");
            } else {
                ui.checkbox(&mut o.fill, "Fill").on_hover_text("Fill center");
                ui.checkbox(&mut o.even, "Even").on_hover_text("Even size");
            }
        }),
        Brush::Pattern => {
            let right = |ui: &mut Ui| {
                if icon_button(ui, egui::vec2(18.0, 16.0), false, Icon::Marquee, "Create pattern from selection").clicked() {
                    cmds.push(Command::PatternFromSelection);
                }
            };
            brush_section(ui, "Pattern", right, |ui| {
                match &e.pattern {
                    Some(p) => {
                        let d = p.volume.dims;
                        ui.label(format!("{} ({}×{}×{})", p.name, d.x, d.y, d.z));
                    }
                    None => {
                        ui.weak("Pick a model in Project > Pattern");
                    }
                }
                choice_box(ui, |ui| {
                    pick2(ui, &mut e.opts.pattern_palette, (true, "Palette", "Use palette color"), (false, "Voxel", "Use the pattern's colors"));
                });
                let off = e.opts.pattern_offset;
                if off != IVec3::ZERO {
                    ui.horizontal(|ui| {
                        ui.weak(format!("Offset {} {} {}", off.x, off.y, off.z)).on_hover_text("Move [Dir/Page] : Reset [Home]");
                        if ui.small_button("Reset").on_hover_text("[Home]").clicked() {
                            e.opts.pattern_offset = IVec3::ZERO;
                        }
                    });
                }
                ui.horizontal(|ui| {
                    ui.label("Rot");
                    if ui.small_button("Z").on_hover_text("[-/+]").clicked() {
                        e.rotate_pattern(2, 1);
                    }
                    if ui.small_button("X").on_hover_text("[9]").clicked() {
                        e.rotate_pattern(0, 1);
                    }
                    if ui.small_button("Y").on_hover_text("[0]").clicked() {
                        e.rotate_pattern(1, 1);
                    }
                });
            });
        }
        Brush::Code | Brush::Box => {}
    }
}

/// MagicaVoxel's Display section: how the modeler draws the scene.
fn display_options(ui: &mut Ui, o: &mut ModelerOptions) {
    brush_section(ui, "Display", |_| {}, |ui| {
        ui.checkbox(&mut o.ao, "AO").on_hover_text("Ambient occlusion");
        ui.checkbox(&mut o.constant, "Constant").on_hover_text("Constant shading");
        ui.checkbox(&mut o.fade, "Fade").on_hover_text("Fade Background Objects");
        let mut bg = o.background.to_array();
        ui.horizontal(|ui| {
            ui.color_edit_button_rgb(&mut bg);
            ui.label("Background");
        });
        o.background = bg.into();
        ui.checkbox(&mut o.frame, "Frame").on_hover_text("Frame grid every 10 voxels [Ctrl+F]");
    });
}

/// Far-right column in the modeler: project, pattern and shader lists.
#[allow(clippy::too_many_arguments)]
pub fn project_panel(
    ui: &mut Ui,
    tab: &mut ProjectTab,
    projects: &[PathBuf],
    samples: &[PathBuf],
    // The opened file's own folder (name, files), when it is elsewhere.
    folder: Option<(&str, &[PathBuf])>,
    current: &Path,
    pattern: Option<&str>,
    shaders: &mut ShaderState,
    export_status: Option<&str>,
    cmds: &mut Vec<Command>,
) {
    heading(ui, "Project");
    // MagicaVoxel's Export box, docked at the bottom of the column.
    egui::Panel::bottom("export").resizable(false).show(ui, |ui| {
        ui.spacing_mut().indent = 6.0;
        ui.spacing_mut().button_padding = egui::vec2(2.0, 1.0);
        CollapsingHeader::new("Export").default_open(true).show(ui, |ui| {
            // glTF first: voxely's format for game engines.
            let gltf = crate::export::Format::Gltf;
            if ui.add_sized([ui.available_width(), 20.0], egui::Button::new("glTF (.glb)")).on_hover_text(gltf.hint()).clicked() {
                cmds.push(Command::Export(gltf));
            }
            // MagicaVoxel's formats, three per row, sized to fit the column.
            let w = ((ui.available_width() - 8.0) / 3.0).floor();
            egui::Grid::new("export_formats").num_columns(3).spacing([4.0, 4.0]).show(ui, |ui| {
                for (k, f) in crate::export::Format::ALL.into_iter().enumerate() {
                    if ui.add_sized([w, 20.0], egui::Button::new(f.name())).on_hover_text(f.hint()).clicked() {
                        cmds.push(Command::Export(f));
                    }
                    if k % 3 == 2 {
                        ui.end_row();
                    }
                }
            });
            if let Some(status) = export_status {
                ui.add(egui::Label::new(egui::RichText::new(status).weak().small()).wrap());
            }
        });
    });
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        ui.selectable_value(tab, ProjectTab::Project, "Proj").on_hover_text("Project list");
        ui.selectable_value(tab, ProjectTab::Pattern, "Patt").on_hover_text("Pattern list");
        ui.selectable_value(tab, ProjectTab::Shader, "Shdr").on_hover_text("Shader list");
    });
    egui::ScrollArea::vertical().show(ui, |ui| {
        match *tab {
            ProjectTab::Project | ProjectTab::Pattern => {
                let pattern_tab = *tab == ProjectTab::Pattern;
                let mut sections = vec![("My projects", projects), ("Samples", samples)];
                sections.extend(folder);
                for (title, list) in sections {
                    if list.is_empty() && title != "My projects" {
                        continue;
                    }
                    // Fold-out sections like the Edit column's.
                    CollapsingHeader::new(title).id_salt((title, pattern_tab)).default_open(true).show(ui, |ui| {
                        if list.is_empty() {
                            ui.add(egui::Label::new(egui::RichText::new("Saved projects appear here").weak()).wrap());
                        }
                        for p in list {
                            let name = p.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                            if pattern_tab {
                                if ui.selectable_label(pattern == Some(name.as_str()), name).clicked() {
                                    cmds.push(Command::LoadPattern(p.clone()));
                                }
                            } else {
                                let item = ui.selectable_label(p == current, name);
                                let mine = title == "My projects";
                                let tip = if mine { format!("{}\nRight-click to delete", p.display()) } else { p.display().to_string() };
                                let item = item.on_hover_text(tip);
                                if item.clicked() {
                                    cmds.push(Command::OpenPath(p.clone()));
                                }
                                if mine {
                                    item.context_menu(|ui| {
                                        if ui.button("Open").clicked() {
                                            cmds.push(Command::OpenPath(p.clone()));
                                            ui.close();
                                        }
                                        if ui.button("Delete…").clicked() {
                                            cmds.push(Command::DeleteProject(p.clone()));
                                            ui.close();
                                        }
                                    });
                                }
                            }
                        }
                    });
                }
            }
            ProjectTab::Shader => {
                if shaders.files.is_empty() {
                    ui.add(egui::Label::new(egui::RichText::new("No shader folder found next to the model's folder").weak()).wrap());
                }
                let chosen = shaders.current.as_ref().map(|(p, _)| p.clone());
                for p in &shaders.files {
                    let name = p.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    if ui.selectable_label(chosen.as_ref() == Some(p), name).clicked() {
                        cmds.push(Command::SelectShader(p.clone()));
                    }
                }
                if let Some((_, sh)) = &shaders.current {
                    ui.separator();
                    // Fit the slider and its value box in the narrow column.
                    ui.spacing_mut().slider_width = (ui.available_width() - 56.0).max(40.0);
                    for (arg, value) in sh.args.iter().zip(shaders.values.iter_mut()) {
                        ui.label(&arg.name);
                        ui.add(Slider::new(value, arg.min..=arg.max).step_by(arg.step.max(0.0) as f64).max_decimals(arg.precision as usize));
                    }
                    ui.horizontal(|ui| {
                        ui.label("Iteration").on_hover_text("Iteration number");
                        ui.add(egui::DragValue::new(&mut shaders.iterations).range(1..=64)).on_hover_text("Run the shader this many times, each over the last result [1 - 64]");
                    });
                    ui.horizontal(|ui| {
                        command(ui, cmds, "Run", "Execute shader on the model (or the selection)", Command::RunShader);
                        if let Some((p, _)) = &shaders.current {
                            command(ui, cmds, "Refresh", "Refresh shader: reload the file", Command::SelectShader(p.clone()));
                        }
                    });
                    ui.horizontal(|ui| {
                        command(ui, cmds, "Import", "Import args", Command::ShaderArgs(true));
                        command(ui, cmds, "Export", "Export args", Command::ShaderArgs(false));
                    });
                    ui.add(egui::Label::new(egui::RichText::new("Or drag a box with the Code brush [C]").weak()).wrap());
                }
                if let Some(e) = &shaders.error {
                    ui.add(egui::Label::new(egui::RichText::new(e).color(egui::Color32::from_rgb(255, 120, 120))).wrap());
                }
            }
        }
    });
}

/// Strip above the 3D view: canvas size and fit.
/// `part` names the model being edited (None in World mode).
pub fn view_bar(ui: &mut Ui, dims: IVec3, canvas: &mut IVec3, part: Option<&str>, models: usize, cmds: &mut Vec<Command>) {
    ui.horizontal(|ui| {
        // MagicaVoxel's model / world editor switch.
        if ui.selectable_label(part.is_some(), "Edit").on_hover_text("Edit the picked model's voxels [Tab]").clicked() && part.is_none() {
            cmds.push(Command::ToggleWorld);
        }
        if ui.selectable_label(part.is_none(), "World").on_hover_text("Arrange whole models and groups [Tab]").clicked() && part.is_some() {
            cmds.push(Command::ToggleWorld);
        }
        let Some(part) = part else {
            ui.weak(format!("{models} model{}", if models == 1 { "" } else { "s" }));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                command(ui, cmds, "Focus", "Center the view on the picked objects [5]", Command::Focus);
                command(ui, cmds, "Ref", "A reference picture to model over", Command::ToggleReference);
            });
            return;
        };
        if models > 1 {
            ui.label(egui::RichText::new(part).strong()).on_hover_text("Parts with a free rotation are edited upright and on their own");
        }
        ui.separator();
        command(ui, cmds, "Fit", "Shrink the canvas to the model", Command::Op(Op::Fit));
        ui.separator();
        for axis in 0..3 {
            ui.add(egui::DragValue::new(&mut canvas[axis]).range(1..=crate::ops::MAX_DIM));
        }
        if ui.add_enabled(*canvas != dims, egui::Button::new("Resize")).on_hover_text("Change the canvas size").clicked() {
            cmds.push(Command::Resize(*canvas));
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            command(ui, cmds, "Focus", "Center the view on the selection or model [5]", Command::Focus);
            command(ui, cmds, "Ref", "A reference picture to model over", Command::ToggleReference);
        });
    });
}
