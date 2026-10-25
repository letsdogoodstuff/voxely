//! The Import window: how to read the chosen file, then where it goes.

use std::path::PathBuf;

use super::picture::{self, ImageMode};
use super::{How, mesh};
use crate::ui::Command;

/// A file picked for import, with the choices made for it so far.
pub struct Pending {
    pub path: PathBuf,
    pub how: How,
    /// Replace the scene with a new project instead of adding parts.
    pub new_project: bool,
    pub add_colors: bool,
    pub error: Option<String>,
}

impl Pending {
    pub fn new(path: PathBuf) -> Option<Pending> {
        let how = if path.is_dir() { How::Slices(Default::default()) } else { super::default_how(&path)? };
        Some(Pending { path, how, new_project: false, add_colors: true, error: None })
    }
}

/// Requests from the Import window, run after the frame.
#[derive(Clone, Debug, PartialEq)]
pub enum ImportCmd {
    /// Ask for a file (or, true, a folder of slices).
    Pick(bool),
    /// Read the pending file into the scene.
    Run,
}

/// Show the Import window for `pending`; closing it drops the import.
pub fn window(ctx: &egui::Context, pending: &mut Option<Pending>, cmds: &mut Vec<Command>) {
    let Some(p) = pending.as_mut() else { return };
    let mut open = true;
    egui::Window::new("Import").open(&mut open).resizable(false).collapsible(false).default_width(260.0).show(ctx, |ui| {
        let name = p.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        ui.label(egui::RichText::new(name).strong()).on_hover_text(p.path.display().to_string());
        let picture_file = super::IMAGES.iter().any(|e| p.path.extension().is_some_and(|x| x.eq_ignore_ascii_case(e)));
        if picture_file || p.path.is_dir() {
            ui.horizontal(|ui| {
                let kind = p.how.name();
                if picture_file {
                    if ui.selectable_label(kind == "image", "Picture").on_hover_text("A flat or extruded picture").clicked() && kind != "image" {
                        p.how = How::Image(Default::default());
                    }
                    if ui.selectable_label(kind == "heightmap", "Heightmap").on_hover_text("Terrain: brighter is higher").clicked() && kind != "heightmap" {
                        p.how = How::Heightmap(Default::default());
                    }
                }
                if ui.selectable_label(kind == "slices", "Slices").on_hover_text("Every layer side by side, bottom first, seen from above (MagicaVoxel's slice export), or a folder of one image per layer").clicked() && kind != "slices" {
                    p.how = How::Slices(Default::default());
                }
            });
        }
        ui.separator();
        options(ui, &mut p.how);
        ui.separator();
        ui.horizontal(|ui| {
            ui.radio_value(&mut p.new_project, false, "Add as parts");
            ui.radio_value(&mut p.new_project, true, "New project");
        });
        if !p.new_project {
            ui.checkbox(&mut p.add_colors, "Add its colors to free palette slots").on_hover_text("Off: match the palette's nearest colors");
        }
        if let Some(e) = &p.error {
            ui.add(egui::Label::new(egui::RichText::new(e).color(egui::Color32::from_rgb(255, 120, 120))).wrap());
        }
        ui.horizontal(|ui| {
            if ui.button("Import").clicked() {
                cmds.push(Command::Import(ImportCmd::Run));
            }
            if ui.button("Other file…").clicked() {
                cmds.push(Command::Import(ImportCmd::Pick(false)));
            }
            if matches!(p.how, How::Slices(_)) && ui.button("Folder…").on_hover_text("A folder of slice images, bottom layer first by name").clicked() {
                cmds.push(Command::Import(ImportCmd::Pick(true)));
            }
        });
    });
    if !open {
        *pending = None;
    }
}

fn size_row(ui: &mut egui::Ui, label: &str, size: &mut Option<u32>, default: u32) {
    ui.horizontal(|ui| {
        let mut on = size.is_some();
        ui.checkbox(&mut on, label).on_hover_text("Off: one voxel per pixel");
        if on {
            let mut s = size.unwrap_or(default);
            ui.add(egui::DragValue::new(&mut s).range(1..=crate::ops::MAX_DIM as u32).suffix(" vox"));
            *size = Some(s);
        } else {
            *size = None;
        }
    });
}

fn options(ui: &mut egui::Ui, how: &mut How) {
    match how {
        How::Voxels => {
            ui.weak("Models, colors and positions are read from the file.");
        }
        How::Mesh(o) => mesh_options(ui, o),
        How::Image(o) => {
            size_row(ui, "Longer side", &mut o.size, 64);
            let (mut kind, mut depth) = match o.mode {
                ImageMode::Flat => (0, 4),
                ImageMode::Extrude(d) => (1, d),
                ImageMode::Relief(d) => (2, d),
            };
            ui.horizontal(|ui| {
                ui.radio_value(&mut kind, 0, "Flat");
                ui.radio_value(&mut kind, 1, "Extrude");
                ui.radio_value(&mut kind, 2, "Relief").on_hover_text("Brighter pixels stand out further");
            });
            if kind > 0 {
                ui.add(egui::Slider::new(&mut depth, 1..=64).text("Depth"));
            }
            o.mode = match kind {
                0 => ImageMode::Flat,
                1 => ImageMode::Extrude(depth),
                _ => ImageMode::Relief(depth),
            };
            ui.horizontal(|ui| {
                ui.radio_value(&mut o.standing, true, "Standing");
                ui.radio_value(&mut o.standing, false, "On the ground");
            });
        }
        How::Heightmap(o) => {
            size_row(ui, "Longer side", &mut o.size, 128);
            ui.add(egui::Slider::new(&mut o.height, 1..=256).text("Height"));
            ui.add(egui::Slider::new(&mut o.water, 0..=o.height).text("Water"));
            ui.horizontal(|ui| {
                ui.radio_value(&mut o.colors, picture::HeightColors::Terrain, "Terrain");
                ui.radio_value(&mut o.colors, picture::HeightColors::Picture, "Picture");
                ui.radio_value(&mut o.colors, picture::HeightColors::Gray, "Gray");
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut o.invert, "Dark is high");
                ui.checkbox(&mut o.solid, "Solid").on_hover_text("Fill columns to the ground; off keeps a surface a few voxels deep");
            });
        }
        How::Slices(o) => {
            ui.horizontal(|ui| {
                ui.label("Grid");
                ui.add(egui::DragValue::new(&mut o.columns).range(0..=4096)).on_hover_text("Slices across (0 and 0: find the layout)");
                ui.label("x");
                ui.add(egui::DragValue::new(&mut o.rows).range(0..=4096)).on_hover_text("Slices down");
                ui.label("gap");
                ui.add(egui::DragValue::new(&mut o.gap).range(0..=64).suffix(" px"));
            });
            ui.checkbox(&mut o.top_first, "First slice is the top layer");
        }
    }
}

fn mesh_options(ui: &mut egui::Ui, o: &mut mesh::MeshOptions) {
    ui.add(egui::Slider::new(&mut o.size, 4..=512).text("Longest side"));
    ui.horizontal(|ui| {
        ui.checkbox(&mut o.solid, "Solid").on_hover_text("Fill the inside of closed meshes");
        ui.checkbox(&mut o.z_up, "Z is up").on_hover_text("Off: Y is up, as in glTF and most OBJ files");
    });
}
