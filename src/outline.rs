//! The World panel (World mode): the scene tree like MagicaVoxel's
//! outliner, plus buttons for MagicaVoxel's World-editor commands.

use egui::{CollapsingHeader, RichText, Ui};
use glam::IVec3;

use crate::scene::{Mode, Scene};
use crate::ui::Command;
use crate::world::{World, ROOT};

/// A change to the scene tree. Most act on the picked nodes.
#[derive(Clone, Debug, PartialEq)]
pub enum TreeOp {
    /// A new empty model next to the picked one [Ctrl+N].
    New,
    Delete,
    Duplicate,
    Group,
    Ungroup,
    /// Hide the picked nodes [Ctrl+H], or show everything [Ctrl+Shift+H].
    Hide(bool),
    ToggleHidden(usize),
    /// Move by whole voxels [arrows, PageUp/PageDown].
    Move(IVec3),
    /// Quarter turns about a world axis [+/- for z, 9/0 for x].
    Turn(usize, i32),
    Flip(usize),
    /// Pick a node; `true` adds to (or removes from) the picked set.
    Pick(usize, bool),
    PickAll,
    PickNone,
    PickInvert,
    Rename(usize, String),
}

/// The rename field's text and which node it belongs to.
#[derive(Default)]
pub struct OutlineState {
    pub rename: Option<(usize, String)>,
}

fn op(ui: &mut Ui, cmds: &mut Vec<Command>, label: &str, hint: &str, enabled: bool, o: TreeOp) {
    if ui.add_enabled(enabled, egui::Button::new(label)).on_hover_text(hint).clicked() {
        cmds.push(Command::Tree(o));
    }
}

fn row(ui: &mut Ui, w: &World, id: usize, picked: &[usize], editing: Option<usize>, cmds: &mut Vec<Command>) {
    let Some(n) = w.node(id) else { return };
    ui.horizontal(|ui| {
        let mut shown = !n.hidden;
        if ui.checkbox(&mut shown, "").on_hover_text("Show").changed() {
            cmds.push(Command::Tree(TreeOp::ToggleHidden(id)));
        }
        let mut text = RichText::new(w.label(id));
        if n.hidden {
            text = text.weak();
        }
        if Some(id) == editing || n.is_group() {
            text = text.strong();
        }
        let r = ui
            .add(egui::Button::selectable(picked.contains(&id), text).truncate())
            .on_hover_text("Click to pick, Shift+click to add, double-click to edit");
        if r.double_clicked() {
            cmds.push(Command::EditPart(id));
        } else if r.clicked() {
            cmds.push(Command::Tree(TreeOp::Pick(id, ui.input(|i| i.modifiers.shift))));
        }
    });
    if n.is_group() {
        ui.indent(("children", id), |ui| {
            for &c in n.children() {
                row(ui, w, c, picked, editing, cmds);
            }
        });
    }
}

/// The World panel.
pub fn panel(ui: &mut Ui, scene: &Scene, state: &mut OutlineState, cmds: &mut Vec<Command>) {
    let w = &scene.world;
    let picked = &scene.picked;
    let any = !picked.is_empty();
    ui.vertical_centered(|ui| ui.label(RichText::new("World").strong()));
    ui.separator();
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.set_max_width(ui.available_width());
        CollapsingHeader::new("Objects").default_open(true).show(ui, |ui| {
            egui::ScrollArea::vertical().id_salt("tree").max_height(260.0).show(ui, |ui| {
                for &c in w.node(ROOT).map(|n| n.children()).unwrap_or_default() {
                    row(ui, w, c, picked, scene.editing(), cmds);
                }
            });
        });
        CollapsingHeader::new("Object").default_open(true).show(ui, |ui| {
            egui::Grid::new("tree_ops").num_columns(2).show(ui, |ui| {
                op(ui, cmds, "New", "New empty model [Ctrl+N]", true, TreeOp::New);
                op(ui, cmds, "Copy", "Duplicate [Ctrl+C, Ctrl+V]", any, TreeOp::Duplicate);
                ui.end_row();
                op(ui, cmds, "Group", "Group the picked objects [Ctrl+R]", any, TreeOp::Group);
                let group = picked.iter().any(|&p| w.node(p).is_some_and(|n| n.is_group()));
                op(ui, cmds, "Ungroup", "Dissolve the picked groups [Ctrl+Shift+R]", group, TreeOp::Ungroup);
                ui.end_row();
                op(ui, cmds, "Hide", "Hide the picked objects [Ctrl+H]", any, TreeOp::Hide(true));
                op(ui, cmds, "Show all", "Show every object [Ctrl+Shift+H]", true, TreeOp::Hide(false));
                ui.end_row();
                op(ui, cmds, "Delete", "Delete the picked objects [Delete]", any, TreeOp::Delete);
                if ui.add_enabled(picked.len() == 1, egui::Button::new("Edit")).on_hover_text("Edit this model's voxels [Tab, double-click]").clicked() {
                    cmds.push(Command::EditPart(picked[0]));
                }
                ui.end_row();
            });
        });
        CollapsingHeader::new("Transform").default_open(true).show(ui, |ui| {
            ui.add_enabled_ui(any, |ui| {
                ui.label("Rotate 90°");
                ui.horizontal(|ui| {
                    for (axis, name) in ["X", "Y", "Z"].into_iter().enumerate() {
                        op(ui, cmds, name, &format!("Turn about {name} (hold Shift: the other way)"), true, TreeOp::Turn(axis, if ui.input(|i| i.modifiers.shift) { -1 } else { 1 }));
                    }
                });
                ui.label("Flip");
                ui.horizontal(|ui| {
                    for (axis, name) in ["X", "Y", "Z"].into_iter().enumerate() {
                        op(ui, cmds, name, &format!("Mirror along {name}"), true, TreeOp::Flip(axis));
                    }
                });
                ui.label("Move");
                ui.horizontal(|ui| {
                    for (axis, name) in ["X", "Y", "Z"].into_iter().enumerate() {
                        let mut d = IVec3::ZERO;
                        d[axis] = 1;
                        let keys = ["Left / Right", "Down / Up", "PageDown / PageUp"][axis];
                        ui.vertical(|ui| {
                            op(ui, cmds, &format!("+{name}"), &format!("Move one voxel [{keys}]"), true, TreeOp::Move(d));
                            op(ui, cmds, &format!("-{name}"), &format!("Move one voxel [{keys}]"), true, TreeOp::Move(-d));
                        });
                    }
                });
            });
        });
        if let [id] = picked.as_slice() {
            let id = *id;
            CollapsingHeader::new("Info").default_open(true).show(ui, |ui| {
                let text = match &mut state.rename {
                    Some((for_id, text)) if *for_id == id => text,
                    other => {
                        *other = Some((id, w.node(id).map(|n| n.name.clone()).unwrap_or_default()));
                        &mut other.as_mut().unwrap().1
                    }
                };
                let r = ui.add(egui::TextEdit::singleline(text).hint_text(w.label(id)).desired_width(f32::INFINITY));
                if r.lost_focus() && w.node(id).is_some_and(|n| n.name != *text) {
                    cmds.push(Command::Tree(TreeOp::Rename(id, text.clone())));
                }
                if let Some((lo, size)) = w.bounds(id) {
                    ui.weak(format!("at {} {} {}", lo.x, lo.y, lo.z));
                    ui.weak(format!("size {} {} {}", size.x, size.y, size.z));
                }
            });
        }
        if scene.mode == Mode::World {
            ui.add_space(6.0);
            ui.weak("Drag a part to move it. Tab edits the picked model.");
        }
    });
}
