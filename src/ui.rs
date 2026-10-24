//! The panels: file and brush controls with the palette on the left; in
//! the modeler, MagicaVoxel's Edit / Selection / Canvas / Project / Display
//! panels on the right, and in the renderer its render settings.

use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};

use egui::{CollapsingHeader, ComboBox, Slider, Ui};
use glam::{IVec3, Vec3};

use crate::edit::{Brush, Editor, GeoShape, MarqueeMode, Tool};
use crate::env::EnvMode;
use crate::modeler::ModelerOptions;
use crate::scene::{GpuMaterial, MaterialKind, Projection, RenderSettings, Scene};

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
    Shape(crate::ops::Shape),
    Hull,
    Flood,
    Dilate,
    Erode,
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
    /// Replace the palette with a 256x1 palette image.
    LoadPalette(PathBuf),
    /// Use a model file as the Pattern brush's pattern.
    LoadPattern(PathBuf),
    /// Make the selection the pattern.
    PatternFromSelection,
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

pub fn settings_panel(ui: &mut Ui, s: &mut RenderSettings, state: &PanelState, spp: u32, size: (u32, u32), cmds: &mut Vec<Command>) {
    ui.horizontal(|ui| {
        ui.label(format!("{spp} spp · {}×{}", size.0, size.1));
        command(ui, cmds, "Snapshot", "6", Command::Snapshot);
    });
    ui.separator();

    egui::ScrollArea::vertical().show(ui, |ui| {
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
                ui.selectable_value(&mut s.env_mode, EnvMode::Atmosphere, "Atmosphere");
                ui.selectable_value(&mut s.env_mode, EnvMode::Image, "Image");
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

        CollapsingHeader::new("Lens").default_open(true).show(ui, |ui| {
            ComboBox::from_label("Projection").selected_text(format!("{:?}", s.projection)).show_ui(ui, |ui| {
                for p in [
                    Projection::Perspective,
                    Projection::Orthographic,
                    Projection::Stereographic,
                    Projection::Panorama,
                    Projection::Cylindrical,
                ] {
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

        CollapsingHeader::new("Sample").show(ui, |ui| {
            slider_u(ui, "Diffuse depth", &mut s.max_diffuse, 0..=15);
            slider_u(ui, "Specular depth", &mut s.max_specular, 5..=15);
            slider_u(ui, "Scatter depth", &mut s.max_scatter, 5..=15);
            slider(ui, "Clamp energy", &mut s.clamp, 1.0..=5000.0);
            ui.checkbox(&mut s.translucent_shadows, "Translucent shadows");
        });

        CollapsingHeader::new("Display").show(ui, |ui| {
            ui.checkbox(&mut s.ground, "Ground");
            color(ui, "Ground color", &mut s.ground_color);
        });
    });
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
pub fn top_bar(ui: &mut Ui, modeling: &mut bool, name: &str, e: &Editor, cmds: &mut Vec<Command>) {
    ui.horizontal(|ui| {
        ui.selectable_value(modeling, true, egui::RichText::new("Model").size(16.0)).on_hover_text("F1");
        ui.selectable_value(modeling, false, egui::RichText::new("Render").size(16.0)).on_hover_text("F1");
        ui.add_space(24.0);
        ui.label(egui::RichText::new(name).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            command(ui, cmds, "Save As", "Ctrl+Shift+S", Command::SaveAs);
            command(ui, cmds, "Save", "Ctrl+S", Command::Save);
            command(ui, cmds, "Open", "Ctrl+O", Command::Open);
            command(ui, cmds, "New", "Ctrl+P", Command::New);
            ui.separator();
            if ui.add_enabled(e.can_redo(), egui::Button::new("Redo")).on_hover_text("Ctrl+Y").clicked() {
                cmds.push(Command::Redo);
            }
            if ui.add_enabled(e.can_undo(), egui::Button::new("Undo")).on_hover_text("Ctrl+Z").clicked() {
                cmds.push(Command::Undo);
            }
        });
    });
}

/// What the bottom bar shows and changes.
pub struct Status<'a> {
    pub options: Option<&'a mut ModelerOptions>,
    /// Hovered voxel and its color index.
    pub hover: Option<(IVec3, u8)>,
    pub projection: &'a mut Projection,
    pub iso: bool,
    /// Free camera: rotating turns the view around the eye.
    pub free: &'a mut bool,
    pub hint: &'a str,
}

/// Bottom bar: view toggles, hovered voxel, camera modes and a hint.
/// Returns true when the isometric view was asked for.
pub fn status_bar(ui: &mut Ui, s: Status, cmds: &mut Vec<Command>) -> bool {
    let mut iso = false;
    ui.horizontal(|ui| {
        if let Some(o) = s.options {
            ui.toggle_value(&mut o.bounds, "BO").on_hover_text("Bounding box [Ctrl+B]");
            ui.toggle_value(&mut o.edge, "Edge").on_hover_text("Voxel outlines [Ctrl+E]");
            ui.toggle_value(&mut o.grid, "Grid").on_hover_text("Grid lines [Ctrl+G]");
            ui.toggle_value(&mut o.frame, "Frame").on_hover_text("Ground frame lines [Ctrl+F]");
            ui.toggle_value(&mut o.ao, "AO").on_hover_text("Ambient occlusion");
            ui.toggle_value(&mut o.shadow, "SW").on_hover_text("Shadow");
            ui.toggle_value(&mut o.wire, "WR").on_hover_text("Wireframe");
            ui.separator();
        }
        let field = |v: Option<i32>| v.map_or("-".to_string(), |v| v.to_string());
        let (p, i) = match s.hover {
            Some((p, i)) => (Some(p), Some(i as i32)),
            None => (None, None),
        };
        ui.monospace(format!(
            "x:{:<4} y:{:<4} z:{:<4} i:{:<4}",
            field(p.map(|p| p.x)),
            field(p.map(|p| p.y)),
            field(p.map(|p| p.z)),
            field(i)
        ));
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
        ui.separator();
        ui.weak(s.hint);
    });
    iso
}

/// Left column: palette pages, swatches, the color editor, and in the
/// renderer the material of the selected color.
pub fn palette_panel(ui: &mut Ui, e: &mut Editor, scene: &mut Scene, modeling: bool, pages: &[PathBuf], cmds: &mut Vec<Command>) -> EditActions {
    let mut a = EditActions::default();
    heading(ui, "Palette");
    if !pages.is_empty() {
        ui.horizontal(|ui| {
            for (k, p) in pages.iter().enumerate() {
                let name = p.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                if ui.small_button(k.to_string()).on_hover_text(format!("Load palette {name}")).clicked() {
                    cmds.push(Command::LoadPalette(p.clone()));
                }
            }
        });
    }
    // 8 columns like MagicaVoxel, color index 1..=255.
    let swatch = 14.0;
    egui::Grid::new("palette").spacing([1.0, 1.0]).min_col_width(swatch).show(ui, |ui| {
        for i in 1..=255usize {
            let p = scene.palette[i];
            let fill = egui::Color32::from_rgb(p[0], p[1], p[2]);
            let (rect, resp) = ui.allocate_exact_size(egui::vec2(swatch, swatch), egui::Sense::click());
            ui.painter().rect_filled(rect, 1.0, fill);
            if e.color as usize == i {
                ui.painter().rect_stroke(rect.expand(1.0), 1.0, egui::Stroke::new(2.0, egui::Color32::WHITE), egui::StrokeKind::Outside);
            }
            if resp.clicked() {
                e.color = i as u8;
            }
            resp.on_hover_text(format!("#{i}"));
            if i % 8 == 0 {
                ui.end_row();
            }
        }
    });
    ui.separator();
    let i = e.color as usize;
    ui.horizontal(|ui| {
        let mut rgb = [scene.palette[i][0], scene.palette[i][1], scene.palette[i][2]];
        if ui.color_edit_button_srgb(&mut rgb).changed() {
            scene.palette[i] = [rgb[0], rgb[1], rgb[2], scene.palette[i][3]];
            scene.sync_color(i);
            a.materials_changed = true;
        }
        ui.label(format!("#{i}"));
    });
    if !modeling {
        let before = scene.materials[i];
        material_editor(ui, &mut scene.materials[i]);
        if bytemuck::bytes_of(&before) != bytemuck::bytes_of(&scene.materials[i]) {
            a.materials_changed = true;
        }
    }
    a
}

/// Second left column in the modeler: brush, tool, selection tools and
/// mirroring.
pub fn brush_panel(ui: &mut Ui, e: &mut Editor, cmds: &mut Vec<Command>) {
    heading(ui, "Brush");
    egui::Grid::new("brushes").num_columns(3).spacing([4.0, 4.0]).show(ui, |ui| {
        ui.selectable_value(&mut e.brush, Brush::Geo, "Geo").on_hover_text("Geometry mode [L]: line, square, circle");
        ui.selectable_value(&mut e.brush, Brush::Code, "Code").on_hover_text("Voxel shader mode [C]");
        ui.selectable_value(&mut e.brush, Brush::Pattern, "Pattern").on_hover_text("Pattern mode [P] : Rot [-/+/0/9]");
        ui.end_row();
        ui.selectable_value(&mut e.brush, Brush::Voxel, "Voxel").on_hover_text("Voxel mode [V] : Resize [-/+/Ctrl+Alt+Drag]");
        ui.selectable_value(&mut e.brush, Brush::Face, "Face").on_hover_text("Face mode [F] : Extrude [Drag]");
        ui.selectable_value(&mut e.brush, Brush::Box, "Box").on_hover_text("Box mode [B]");
        ui.end_row();
    });
    ui.separator();
    ui.vertical_centered_justified(|ui| {
        ui.selectable_value(&mut e.tool, Tool::Attach, "Attach").on_hover_text("T");
        ui.selectable_value(&mut e.tool, Tool::Erase, "Erase").on_hover_text("R");
        ui.selectable_value(&mut e.tool, Tool::Paint, "Paint").on_hover_text("G");
    });
    ui.separator();
    egui::Grid::new("select_tools").num_columns(3).spacing([4.0, 4.0]).show(ui, |ui| {
        ui.selectable_value(&mut e.tool, Tool::Move, "Move").on_hover_text("Drag the selection, or the model : Normal axis [Shift]");
        ui.selectable_value(&mut e.tool, Tool::Marquee, "Select").on_hover_text("Marquee select [M] : Add [Shift] : Remove [Shift+Alt]");
        ui.selectable_value(&mut e.tool, Tool::Region, "Region").on_hover_text("Region select [N] : Add [Shift] : Remove [Shift+Alt]");
        ui.end_row();
        ui.selectable_value(&mut e.tool, Tool::Pick, "Pick").on_hover_text("Pick voxel color [Alt+LButton]");
        ui.selectable_value(&mut e.tool, Tool::RegionRemove, "Remove").on_hover_text("Region remove");
        ui.selectable_value(&mut e.tool, Tool::RegionReplace, "Replace").on_hover_text("Region replace with the current color");
        ui.end_row();
    });
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("Mirror").on_hover_text("Mirror mode");
        for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
            ui.toggle_value(&mut e.mirror[i], *n).on_hover_text(format!("F{}", 5 + i));
        }
    });
    ui.horizontal(|ui| {
        ui.label("Axis").on_hover_text("Axis mode: extend the brush through the canvas");
        for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
            ui.toggle_value(&mut e.opts.axis[i], *n);
        }
    });
    ui.separator();
    brush_options(ui, e, cmds);
    if matches!(e.tool, Tool::Region | Tool::RegionRemove | Tool::RegionReplace) {
        ui.separator();
        ui.label("Region");
        let r = &mut e.region;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut r.same_color, true, "Color").on_hover_text("Same palette color, anywhere");
            ui.selectable_value(&mut r.same_color, false, "Connected").on_hover_text("Connected region");
        });
        if !r.same_color {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut r.by_color, true, "Color").on_hover_text("Connect through the same color");
                ui.selectable_value(&mut r.by_color, false, "Geo").on_hover_text("Connect through any voxel");
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut r.diagonal, false, "4").on_hover_text("Connect through faces");
                ui.selectable_value(&mut r.diagonal, true, "8").on_hover_text("Also connect through edges and corners");
            });
        }
    }
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("Mirror");
        for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
            ui.toggle_value(&mut e.mirror[i], *n).on_hover_text(format!("F{}", 5 + i));
        }
    });
}

fn axis_row(ui: &mut Ui, cmds: &mut Vec<Command>, op: fn(usize) -> Op) {
    ui.horizontal(|ui| {
        for (axis, n) in ["X", "Y", "Z"].iter().enumerate() {
            command(ui, cmds, n, "", Command::Op(op(axis)));
        }
    });
}

/// Right column in the modeler: MagicaVoxel's Edit panel.
pub fn edit_panel(ui: &mut Ui, e: &Editor, o: &mut ModelerOptions, cmds: &mut Vec<Command>) {
    use crate::ops::Shape;
    heading(ui, "Edit");
    egui::ScrollArea::vertical().show(ui, |ui| {
        CollapsingHeader::new("Tool").default_open(true).show(ui, |ui| {
            egui::Grid::new("tool_ops").num_columns(2).show(ui, |ui| {
                command(ui, cmds, "Full", "Fill the whole canvas [U]", Command::Op(Op::Full));
                command(ui, cmds, "Fill", "Recolor voxels with the current color [I]", Command::Op(Op::Fill));
                ui.end_row();
                command(ui, cmds, "Clear", "Delete voxels [Delete]", Command::Op(Op::Clear));
                command(ui, cmds, "Rev", "Swap empty and solid space", Command::Op(Op::Reverse));
                ui.end_row();
                command(ui, cmds, "2X", "Double size", Command::Op(Op::Double));
                command(ui, cmds, "1/2", "Half size", Command::Op(Op::Half));
                ui.end_row();
            });
        });
        CollapsingHeader::new("Select").default_open(true).show(ui, |ui| {
            ui.weak(match e.selected_count() {
                0 => "nothing selected".to_string(),
                n => format!("{n} selected"),
            });
            egui::Grid::new("select_ops").num_columns(2).show(ui, |ui| {
                command(ui, cmds, "All", "Ctrl+A", Command::SelectAll);
                command(ui, cmds, "Inv", "Ctrl+I", Command::SelectInvert);
                ui.end_row();
                command(ui, cmds, "None", "Ctrl+D", Command::SelectNone);
                ui.end_row();
                command(ui, cmds, "Copy", "Ctrl+C", Command::Copy);
                command(ui, cmds, "Cut", "Ctrl+X", Command::Cut);
                ui.end_row();
                if ui.add_enabled(e.has_clipboard(), egui::Button::new("Paste")).on_hover_text("Ctrl+V").clicked() {
                    cmds.push(Command::Paste);
                }
                ui.end_row();
            });
        });
        CollapsingHeader::new("Rot").show(ui, |ui| axis_row(ui, cmds, Op::Rotate));
        CollapsingHeader::new("Flip").show(ui, |ui| axis_row(ui, cmds, Op::Flip));
        CollapsingHeader::new("Loop").show(ui, |ui| {
            for (axis, n) in ["X", "Y", "Z"].iter().enumerate() {
                ui.horizontal(|ui| {
                    command(ui, cmds, &format!("-{n}"), "", Command::Op(Op::Loop(axis, -1)));
                    command(ui, cmds, &format!("+{n}"), "", Command::Op(Op::Loop(axis, 1)));
                });
            }
        });
        CollapsingHeader::new("Mirror").show(ui, |ui| axis_row(ui, cmds, Op::Mirror));
        CollapsingHeader::new("Shape").show(ui, |ui| {
            ui.horizontal(|ui| {
                command(ui, cmds, "Sphere", "Add an ellipsoid filling the canvas", Command::Op(Op::Shape(Shape::Ellipsoid)));
                command(ui, cmds, "Cyli", "Add a cylinder", Command::Op(Op::Shape(Shape::Cylinder)));
                command(ui, cmds, "Cone", "Add a cone", Command::Op(Op::Shape(Shape::Cone)));
            });
        });
        CollapsingHeader::new("Modify").show(ui, |ui| {
            egui::Grid::new("modify_ops").num_columns(2).show(ui, |ui| {
                command(ui, cmds, "Hull", "Remove all invisible voxels", Command::Op(Op::Hull));
                command(ui, cmds, "Flood", "Fill all enclosed space", Command::Op(Op::Flood));
                ui.end_row();
                command(ui, cmds, "Dilate", "Grow by one voxel", Command::Op(Op::Dilate));
                command(ui, cmds, "Erode", "Shrink by one voxel", Command::Op(Op::Erode));
                ui.end_row();
            });
        });
        CollapsingHeader::new("View").show(ui, |ui| {
            ui.checkbox(&mut o.constant, "Constant shading");
            let mut bg = o.background.to_array();
            ui.horizontal(|ui| {
                ui.color_edit_button_rgb(&mut bg);
                ui.label("Background");
            });
            o.background = bg.into();
        });
    });
}

fn pick2<T: PartialEq + Copy>(ui: &mut Ui, v: &mut T, a: (T, &str, &str), b: (T, &str, &str)) {
    ui.horizontal(|ui| {
        ui.selectable_value(v, a.0, a.1).on_hover_text(a.2);
        ui.selectable_value(v, b.0, b.1).on_hover_text(b.2);
    });
}

/// Options for the current brush or selection tool.
fn brush_options(ui: &mut Ui, e: &mut Editor, cmds: &mut Vec<Command>) {
    if e.tool == Tool::Marquee {
        ui.label("Marquee");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut e.opts.marquee, MarqueeMode::Rect, "Rect").on_hover_text("Rect select");
            ui.selectable_value(&mut e.opts.marquee, MarqueeMode::Brush, "Brush").on_hover_text("Brush select (voxel brush size)");
            ui.selectable_value(&mut e.opts.marquee, MarqueeMode::Box, "Box").on_hover_text("Box select");
        });
        if e.opts.marquee == MarqueeMode::Rect {
            ui.checkbox(&mut e.opts.visible_only, "Visible only").on_hover_text("Skip voxels hidden behind others");
        }
        if e.opts.marquee != MarqueeMode::Brush {
            return;
        }
    }
    let o = &mut e.opts;
    match e.brush {
        Brush::Voxel => {
            ui.horizontal(|ui| {
                ui.label("Size");
                ui.add(egui::DragValue::new(&mut o.size).range(1..=64)).on_hover_text("Voxel size [1 - 64] : Change size [-/+/Ctrl+Alt+Drag]");
            });
            pick2(ui, &mut o.sphere, (false, "Cube", "Cube shape"), (true, "Sphere", "Sphere shape"));
            pick2(ui, &mut o.flat, (true, "2D", "Flat on the face under the cursor"), (false, "3D", "Solid"));
        }
        Brush::Face => {
            pick2(ui, &mut o.face_voxel_color, (false, "Palette", "Use palette color"), (true, "Voxel", "Use voxel color"));
            pick2(ui, &mut o.face_by_color, (true, "Color", "Color connectivity"), (false, "Geo", "Geometry connectivity"));
            pick2(ui, &mut o.face_diagonal, (false, "4", "4 neighbor searching"), (true, "8", "8 neighbor searching"));
        }
        Brush::Geo => {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut o.geo, GeoShape::Line, "Line");
                ui.selectable_value(&mut o.geo, GeoShape::Square, "Square");
                ui.selectable_value(&mut o.geo, GeoShape::Circle, "Circle");
            });
            if o.geo == GeoShape::Line {
                ui.checkbox(&mut o.project, "Project").on_hover_text("Project the line on the surface");
            } else {
                ui.checkbox(&mut o.fill, "Fill").on_hover_text("Fill center");
                ui.checkbox(&mut o.even, "Even").on_hover_text("Even size");
            }
        }
        Brush::Pattern => {
            match &e.pattern {
                Some(p) => {
                    let d = p.volume.dims;
                    ui.label(format!("{} ({}×{}×{})", p.name, d.x, d.y, d.z));
                }
                None => {
                    ui.weak("Pick a model in Project › Pattern");
                }
            }
            pick2(ui, &mut e.opts.pattern_palette, (true, "Palette", "Use palette color"), (false, "Voxel", "Use the pattern's colors"));
            ui.horizontal(|ui| {
                command(ui, cmds, "From selection", "Create pattern from selection", Command::PatternFromSelection);
            });
            ui.horizontal(|ui| {
                ui.label("Rot");
                if ui.small_button("Z").on_hover_text("[-/+]").clicked() {
                    e.rotate_pattern(2, 1);
                }
                if ui.small_button("X").on_hover_text("[9/0]").clicked() {
                    e.rotate_pattern(0, 1);
                }
                if ui.small_button("Y").clicked() {
                    e.rotate_pattern(1, 1);
                }
            });
        }
        Brush::Code => {
            ui.weak("Drag a box to fill it with the shader chosen in Project › Shader");
        }
        Brush::Box => {}
    }
}

/// Far-right column in the modeler: project, pattern and shader lists.
pub fn project_panel(ui: &mut Ui, tab: &mut ProjectTab, projects: &[PathBuf], current: &Path, pattern: Option<&str>, cmds: &mut Vec<Command>) {
    heading(ui, "Project");
    ui.horizontal(|ui| {
        ui.selectable_value(tab, ProjectTab::Project, "Proj").on_hover_text("Project list");
        ui.selectable_value(tab, ProjectTab::Pattern, "Patt").on_hover_text("Pattern list");
        ui.selectable_value(tab, ProjectTab::Shader, "Shdr").on_hover_text("Shader list");
    });
    egui::ScrollArea::vertical().show(ui, |ui| {
        match *tab {
            ProjectTab::Project | ProjectTab::Pattern => {
                if projects.is_empty() {
                    ui.weak("Open a .vox file to list its folder here");
                }
                for p in projects {
                    let name = p.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    if *tab == ProjectTab::Project {
                        if ui.selectable_label(p == current, name).clicked() {
                            cmds.push(Command::OpenPath(p.clone()));
                        }
                    } else if ui.selectable_label(pattern == Some(name.as_str()), name).clicked() {
                        cmds.push(Command::LoadPattern(p.clone()));
                    }
                }
            }
            ProjectTab::Shader => {
                ui.weak("Shaders coming soon");
            }
        }
    });
}

/// Strip above the 3D view: canvas size and fit.
pub fn view_bar(ui: &mut Ui, dims: IVec3, canvas: &mut IVec3, cmds: &mut Vec<Command>) {
    ui.horizontal(|ui| {
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
        });
    });
}
