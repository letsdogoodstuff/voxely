//! The World panel (World mode): the scene tree like MagicaVoxel's
//! outliner, plus buttons for MagicaVoxel's World-editor commands.

use egui::{CollapsingHeader, RichText, Ui};
use glam::{IVec3, Vec3};

use crate::scene::{Mode, Scene};
use crate::ui::Command;
use crate::world::{Socket, World, ROOT};

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
    /// Attach a node to another so it follows it, snapping onto one of
    /// that part's sockets if given.
    Attach(usize, usize, Option<String>),
    /// Stop following; stays where it is.
    Detach(usize),
    /// Timeline settings.
    SetAnim(crate::anim::Anim),
    /// Key a node's pose on the current frame, or remove that key.
    Key(usize),
    DeleteKey(usize),
    /// Flipbook: own voxels from the current frame on, or drop them.
    SplitFrame(usize),
    MergeFrame(usize),
    /// Add a rig template (true: four-legged).
    AddRig(bool),
    /// Key a motion preset onto the rig over this many frames.
    Motion(String, u32),
    /// New pivot, free rotation and sockets for a node. `true` starts an
    /// undo step (false continues a drag).
    Extras(usize, Extras, bool),
}

/// A node's voxely-only settings, in its own coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Extras {
    pub pivot: Option<Vec3>,
    pub tilt: Vec3,
    pub sockets: Vec<Socket>,
}

impl Extras {
    pub fn of(n: &crate::world::Node) -> Extras {
        Extras { pivot: n.pivot, tilt: n.tilt, sockets: n.sockets.clone() }
    }
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
    if !n.children().is_empty() {
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
                // Rigs nest deep; keep names readable.
                ui.spacing_mut().indent = 8.0;
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
        CollapsingHeader::new("Rig").default_open(false).show(ui, |ui| {
            ui.weak("A character of blocks with joints, ready for the motions in the timeline. Redraw each part to taste.");
            ui.horizontal_wrapped(|ui| {
                op(ui, cmds, "+ Humanoid", "Hips, torso, head, two-piece arms and legs", true, TreeOp::AddRig(false));
                op(ui, cmds, "+ Four-legged", "Body, head, tail and four legs", true, TreeOp::AddRig(true));
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
            attach_ui(ui, w, id, cmds);
            extras_ui(ui, w, id, cmds);
        }
        if scene.mode == Mode::World {
            ui.add_space(6.0);
            ui.weak("Drag a part to move it. Tab edits the picked model.");
        }
    });
}

/// Three drag fields; returns (changed, started a new edit).
fn vec3_ui(ui: &mut Ui, v: &mut Vec3, speed: f64, suffix: &str) -> (bool, bool) {
    let (mut changed, mut begin) = (false, false);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        for k in 0..3 {
            let r = ui.add(egui::DragValue::new(&mut v[k]).speed(speed).max_decimals(1).suffix(suffix));
            if r.changed() {
                changed = true;
                begin |= !r.dragged() || r.drag_started();
            }
        }
    });
    (changed, begin)
}

/// Pivot, free rotation and sockets of node `id`. Positions show in part
/// coordinates (what the voxel tools use while editing it).
fn extras_ui(ui: &mut Ui, w: &World, id: usize, cmds: &mut Vec<Command>) {
    let Some(n) = w.node(id) else { return };
    let mut x = Extras::of(n);
    let (mut changed, mut begin) = (false, false);
    let mut track = |(c, b): (bool, bool)| {
        changed |= c;
        begin |= b;
    };
    CollapsingHeader::new("Pivot").default_open(true).show(ui, |ui| {
        let mut p = w.to_part(id, w.pivot_local(id));
        let r = vec3_ui(ui, &mut p, 0.25, "");
        if r.0 {
            x.pivot = Some(w.from_part(id, p));
        }
        track(r);
        ui.horizontal(|ui| {
            if ui.button("Center").on_hover_text("Turn about the middle").clicked() {
                x.pivot = None;
                track((true, true));
            }
            if let Some((_, size)) = w.bounds(id).filter(|_| !n.is_group() && !n.tilted()) {
                if ui.button("Bottom").on_hover_text("Turn about the middle of the bottom face").clicked() {
                    x.pivot = Some(w.from_part(id, Vec3::new(size.x as f32 * 0.5, size.y as f32 * 0.5, 0.0)));
                    track((true, true));
                }
            }
        });
    });
    CollapsingHeader::new("Rotation").default_open(true).show(ui, |ui| {
        let r = vec3_ui(ui, &mut x.tilt, 1.0, "°");
        track(r);
        if n.tilted() && ui.button("Straighten").on_hover_text("Remove the free rotation").clicked() {
            x.tilt = Vec3::ZERO;
            track((true, true));
        }
    });
    CollapsingHeader::new("Sockets").default_open(true).show(ui, |ui| {
        let mut remove = None;
        for (k, sock) in x.sockets.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                let r = ui.add(egui::TextEdit::singleline(&mut sock.name).desired_width(ui.available_width() - 34.0));
                if r.changed() {
                    track((true, true));
                }
                if ui.small_button("x").on_hover_text("Remove").clicked() {
                    remove = Some(k);
                }
            });
            let mut p = w.to_part(id, sock.pos);
            let r = vec3_ui(ui, &mut p, 0.25, "");
            if r.0 {
                sock.pos = w.from_part(id, p);
            }
            track(r);
        }
        if let Some(k) = remove {
            x.sockets.remove(k);
            track((true, true));
        }
        if ui.button("+ Socket").on_hover_text("A named attachment point, for attaching parts here and for game engines").clicked() {
            let name = (1..).map(|i| format!("socket{i}")).find(|s| !x.sockets.iter().any(|o| &o.name == s)).unwrap();
            x.sockets.push(Socket { name, pos: w.pivot_local(id) });
            track((true, true));
        }
    });
    if changed && x != Extras::of(n) {
        cmds.push(Command::Tree(TreeOp::Extras(id, x, begin)));
    }
}

/// What a node is attached to, and snapping onto that part's sockets.
fn attach_ui(ui: &mut Ui, w: &World, id: usize, cmds: &mut Vec<Command>) {
    let Some(parent) = w.node(id).and_then(|n| n.parent) else { return };
    CollapsingHeader::new("Attached to").default_open(true).show(ui, |ui| {
        let current = if parent == ROOT { "nothing".to_string() } else { w.label(parent) };
        egui::ComboBox::from_id_salt(("attach", id)).selected_text(current).width(ui.available_width()).show_ui(ui, |ui| {
            if ui.selectable_label(parent == ROOT, "nothing").clicked() && parent != ROOT {
                cmds.push(Command::Tree(TreeOp::Detach(id)));
            }
            // Anything not inside this node.
            let mut inside = Vec::new();
            w.visit(id, &mut |i, _| inside.push(i));
            w.visit(ROOT, &mut |i, _| {
                if i != ROOT && !inside.contains(&i) && ui.selectable_label(i == parent, w.label(i)).clicked() && i != parent {
                    cmds.push(Command::Tree(TreeOp::Attach(id, i, None)));
                }
            });
        })
        .response
        .on_hover_text("Attached parts follow their parent's moves and turns");
        let sockets = w.node(parent).map(|n| n.sockets.clone()).unwrap_or_default();
        if parent != ROOT && !sockets.is_empty() {
            ui.horizontal_wrapped(|ui| {
                ui.weak("snap to");
                for s in sockets {
                    if ui.small_button(&s.name).on_hover_text("Move this part's pivot onto the socket").clicked() {
                        cmds.push(Command::Tree(TreeOp::Attach(id, parent, Some(s.name.clone()))));
                    }
                }
            });
        }
    });
}
