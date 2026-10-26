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
    /// Put nodes inside another (they follow it from then on), keeping
    /// where they are in the world.
    PutInside(Vec<usize>, usize),
    /// Work inside a node in World mode (`ROOT`: the whole world).
    Enter(usize),
    /// Move a node by whole voxels from the Transform fields. `true`
    /// starts an undo step (false continues a drag).
    Place(usize, IVec3, bool),
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

/// A node's voxely-only settings, in its own coordinates (the free
/// rotation about world axes, see `World::tilt_world`).
#[derive(Clone, Debug, PartialEq)]
pub struct Extras {
    pub pivot: Option<Vec3>,
    pub tilt: Vec3,
    pub sockets: Vec<Socket>,
}

impl Extras {
    pub fn of(w: &World, id: usize) -> Option<Extras> {
        let n = w.node(id)?;
        Some(Extras { pivot: n.pivot, tilt: w.tilt_world(id), sockets: n.sockets.clone() })
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

/// Tree rows being dragged: the picked ones, or the row grabbed.
struct Dragged(Vec<usize>);

/// Highlight `r` while rows that could go inside `to` are held over it,
/// and put them inside when they're dropped there.
fn drop_target(ui: &Ui, r: &egui::Response, w: &World, to: usize, cmds: &mut Vec<Command>) {
    let fits = |d: &Dragged| d.0.iter().any(|&i| !w.within(i, to) && w.node(i).and_then(|n| n.parent) != Some(to));
    if r.dnd_hover_payload::<Dragged>().is_some_and(|d| fits(&d)) {
        ui.painter().rect_stroke(r.rect, 3.0, ui.visuals().selection.stroke, egui::StrokeKind::Inside);
    }
    if let Some(d) = r.dnd_release_payload::<Dragged>().filter(|d| fits(d)) {
        cmds.push(Command::Tree(TreeOp::PutInside(d.0.clone(), to)));
    }
}

fn row(ui: &mut Ui, w: &World, id: usize, picked: &[usize], editing: Option<usize>, cmds: &mut Vec<Command>) {
    let Some(n) = w.node(id) else { return };
    let inner = !n.children().is_empty();
    ui.horizontal(|ui| {
        let mut shown = !n.hidden;
        if ui.checkbox(&mut shown, "").on_hover_text("Show").changed() {
            cmds.push(Command::Tree(TreeOp::ToggleHidden(id)));
        }
        // Shortened in the middle so a side suffix (_l / _r) stays readable.
        let label = w.label(id);
        let font = egui::TextStyle::Button.resolve(ui.style());
        let room = ui.available_width() - 2.0 * ui.spacing().button_padding.x;
        let shown = fit_middle(ui, &label, &font, room);
        let hint = if inner {
            "Click to pick, Shift+click to add, double-click to go inside\nDrag onto another to put it inside"
        } else {
            "Click to pick, Shift+click to add, double-click to edit\nDrag onto another to put it inside"
        };
        let hint = if shown != label { format!("{label}\n{hint}") } else { hint.to_string() };
        let mut text = RichText::new(shown);
        if n.hidden {
            text = text.weak();
        }
        if Some(id) == editing || n.is_group() {
            text = text.strong();
        }
        let r = ui.add(egui::Button::selectable(picked.contains(&id), text).sense(egui::Sense::click_and_drag())).on_hover_text(hint);
        r.dnd_set_drag_payload(Dragged(if picked.contains(&id) { picked.to_vec() } else { vec![id] }));
        drop_target(ui, &r, w, id, cmds);
        if r.double_clicked() {
            cmds.push(if inner { Command::Tree(TreeOp::Enter(id)) } else { Command::EditPart(id) });
        } else if r.clicked() {
            cmds.push(Command::Tree(TreeOp::Pick(id, ui.input(|i| i.modifiers.shift))));
        }
    });
    if inner {
        ui.indent(("children", id), |ui| {
            for &c in n.children() {
                row(ui, w, c, picked, editing, cmds);
            }
        });
    }
}

/// `text` shortened in the middle ("upper_a…m_l") to fit `width` points
/// in `font`, keeping its start and its last word.
pub fn fit_middle(ui: &Ui, text: &str, font: &egui::FontId, width: f32) -> String {
    let fits = |s: &str| ui.ctx().fonts_mut(|f| f.layout_no_wrap(s.to_string(), font.clone(), egui::Color32::WHITE).size().x) <= width;
    if fits(text) {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    let last_word = chars.iter().rev().position(|c| matches!(c, '_' | ' ' | '-' | '.')).map_or(0, |k| k + 1);
    for keep in (1..chars.len()).rev() {
        let tail = (keep * 2 / 5).max(if last_word <= 4 { last_word } else { 0 }).clamp(1, keep);
        let s: String = chars[..keep - tail].iter().chain(['\u{2026}'].iter()).chain(&chars[chars.len() - tail..]).collect();
        if fits(&s) {
            return s;
        }
    }
    "\u{2026}".into()
}

/// The World panel.
pub fn panel(ui: &mut Ui, scene: &Scene, state: &mut OutlineState, cmds: &mut Vec<Command>) {
    let w = &scene.world;
    let picked = &scene.picked;
    let any = !picked.is_empty();
    let inside = scene.inside;
    ui.vertical_centered(|ui| ui.label(RichText::new("World").strong()));
    ui.separator();
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.set_max_width(ui.available_width());
        if inside != ROOT {
            // Working inside an object: a way back out.
            let up = w.node(inside).and_then(|n| n.parent).unwrap_or(ROOT);
            let to = if up == ROOT { "the whole world".to_string() } else { w.label(up) };
            ui.horizontal(|ui| {
                let r = ui.button(RichText::new("\u{2191} Up").monospace()).on_hover_text(format!("Back out to {to} [Esc]\nDrop rows here to take them out"));
                drop_target(ui, &r, w, up, cmds);
                if r.clicked() {
                    cmds.push(Command::Tree(TreeOp::Enter(up)));
                }
                ui.label(RichText::new(w.label(inside)).strong());
            });
        }
        CollapsingHeader::new("Objects").default_open(true).show(ui, |ui| {
            let tall = (ui.ctx().content_rect().height() * 0.4).max(200.0);
            egui::ScrollArea::vertical().id_salt("tree").max_height(tall).show(ui, |ui| {
                // Rigs nest deep; keep names readable.
                ui.spacing_mut().indent = 8.0;
                for &c in w.node(inside).map(|n| n.children()).unwrap_or_default() {
                    row(ui, w, c, picked, scene.editing(), cmds);
                }
                // Dropping below the rows brings things to this level.
                let (rect, r) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 28.0), egui::Sense::hover());
                if egui::DragAndDrop::has_payload_of_type::<Dragged>(ui.ctx()) {
                    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "drop here: this level", egui::TextStyle::Small.resolve(ui.style()), ui.visuals().weak_text_color());
                }
                drop_target(ui, &r, w, inside, cmds);
            });
        });
        CollapsingHeader::new("Object").default_open(true).show(ui, |ui| {
            egui::Grid::new("tree_ops").num_columns(2).show(ui, |ui| {
                op(ui, cmds, "New", "New empty model [Ctrl+N]", true, TreeOp::New);
                op(ui, cmds, "Copy", "Duplicate [Ctrl+C, Ctrl+V]", any, TreeOp::Duplicate);
                ui.end_row();
                op(ui, cmds, "Group", "Put the picked objects inside a new empty one [Ctrl+R]", any, TreeOp::Group);
                let group = picked.iter().any(|&p| w.node(p).is_some_and(|n| n.is_group()));
                op(ui, cmds, "Ungroup", "Take everything out of the picked groups and remove them [Ctrl+Shift+R]", group, TreeOp::Ungroup);
                ui.end_row();
                op(ui, cmds, "Hide", "Hide the picked objects [Ctrl+H]", any, TreeOp::Hide(true));
                op(ui, cmds, "Show all", "Show every object [Ctrl+Shift+H]", true, TreeOp::Hide(false));
                ui.end_row();
                op(ui, cmds, "Delete", "Delete the picked objects [Delete]", any, TreeOp::Delete);
                if ui.add_enabled(picked.len() == 1, egui::Button::new("Edit")).on_hover_text("Edit this model's voxels [Tab, double-click]").clicked() {
                    cmds.push(Command::EditPart(picked[0]));
                }
                ui.end_row();
                let inner = picked.len() == 1 && w.node(picked[0]).is_some_and(|n| !n.children().is_empty());
                let hint = "Work on what's inside it; the rest dims [double-click its row; Esc comes back out]";
                op(ui, cmds, "Go inside", hint, inner, TreeOp::Enter(picked.first().copied().unwrap_or(ROOT)));
                ui.end_row();
            });
        });
        // Open while the scene has no rig yet, to show it can make one.
        let has_rig = w.node(ROOT).is_some_and(|r| r.children().iter().any(|&t| !crate::rig::find_bones(w, t).is_empty()));
        CollapsingHeader::new("Rig").default_open(!has_rig).show(ui, |ui| {
            ui.weak("A character of blocks with joints, ready for the motions in the timeline. Redraw each part to taste.");
            ui.horizontal_wrapped(|ui| {
                op(ui, cmds, "+ Humanoid", "Hips, torso, head, two-piece arms and legs", true, TreeOp::AddRig(false));
                op(ui, cmds, "+ Four-legged", "Body, head, tail and four legs", true, TreeOp::AddRig(true));
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
                if let Some((_, size)) = w.bounds(id) {
                    ui.weak(format!("size {} {} {}", size.x, size.y, size.z));
                }
                let parent = w.node(id).and_then(|n| n.parent).unwrap_or(ROOT);
                if parent != ROOT {
                    ui.weak(format!("inside {}", w.label(parent)));
                }
                socket_snap(ui, w, id, parent, cmds);
            });
        }
        if any {
            transform_ui(ui, w, picked, cmds);
        }
        if let [id] = picked.as_slice() {
            sockets_ui(ui, w, *id, cmds);
        }
        if scene.mode == Mode::World {
            ui.add_space(6.0);
            ui.add(egui::Label::new(RichText::new("Drag a part to move it. Drag a row onto another to put it inside. Tab edits the picked model.").weak()).wrap());
        }
    });
}

/// Three drag fields; returns (changed, started a new edit).
fn vec3_ui(ui: &mut Ui, v: &mut Vec3, speed: f64, suffix: &str) -> (bool, bool) {
    let (mut changed, mut begin) = (false, false);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        // Three equal fields across the column, so none is cut off.
        let w = ((ui.available_width() - 4.0) / 3.0).floor();
        for k in 0..3 {
            let r = ui.add_sized([w, ui.spacing().interact_size.y], egui::DragValue::new(&mut v[k]).speed(speed).max_decimals(1).suffix(suffix));
            if r.changed() {
                changed = true;
                begin |= !r.dragged() || r.drag_started();
            }
        }
    });
    (changed, begin)
}

/// `vec3_ui` for whole voxels.
fn ivec3_ui(ui: &mut Ui, v: &mut IVec3) -> (bool, bool) {
    let (mut changed, mut begin) = (false, false);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        let w = ((ui.available_width() - 4.0) / 3.0).floor();
        for k in 0..3 {
            let r = ui.add_sized([w, ui.spacing().interact_size.y], egui::DragValue::new(&mut v[k]).speed(0.1));
            if r.changed() {
                changed = true;
                begin |= !r.dragged() || r.drag_started();
            }
        }
    });
    (changed, begin)
}

/// One Transform block: position, rotation, quarter turns and flips, and
/// the pivot everything turns about. Fields show for a single object;
/// the buttons act on every picked one.
fn transform_ui(ui: &mut Ui, w: &World, picked: &[usize], cmds: &mut Vec<Command>) {
    CollapsingHeader::new("Transform").default_open(true).show(ui, |ui| {
        let one = match picked {
            [id] => w.node(*id).map(|n| (*id, n)),
            _ => None,
        };
        if let Some((lo, _)) = one.and_then(|(id, _)| w.bounds(id)) {
            ui.label("Position").on_hover_text("Where its box starts in the world, in voxels [arrows, PageUp/PageDown]");
            let mut p = lo;
            let (changed, begin) = ivec3_ui(ui, &mut p);
            if let (true, Some((id, _))) = (changed && p != lo, one) {
                cmds.push(Command::Tree(TreeOp::Place(id, p - lo, begin)));
            }
        }
        let mut x = one.and_then(|(id, _)| Extras::of(w, id));
        let (mut changed, mut begin) = (false, false);
        if let Some(x) = &mut x {
            ui.label("Rotation").on_hover_text("Free rotation about world axes, turning about the pivot");
            let r = vec3_ui(ui, &mut x.tilt, 1.0, "°");
            (changed, begin) = (r.0, r.1);
        }
        let shift = ui.input(|i| i.modifiers.shift);
        ui.horizontal(|ui| {
            ui.label("Turn 90°");
            for (axis, name) in ["X", "Y", "Z"].into_iter().enumerate() {
                op(ui, cmds, name, &format!("Turn about {name} (hold Shift: the other way)"), true, TreeOp::Turn(axis, if shift { -1 } else { 1 }));
            }
        });
        ui.horizontal(|ui| {
            ui.label("Flip");
            for (axis, name) in ["X", "Y", "Z"].into_iter().enumerate() {
                op(ui, cmds, name, &format!("Mirror along {name}"), true, TreeOp::Flip(axis));
            }
        });
        let (Some((id, n)), Some(x)) = (one, &mut x) else { return };
        if n.tilted() && ui.button("Straighten").on_hover_text("Remove the free rotation").clicked() {
            x.tilt = Vec3::ZERO;
            (changed, begin) = (true, true);
        }
        ui.label("Pivot").on_hover_text("The point it turns about, and where it joins what it's inside. In the part's own voxels.");
        let mut p = w.to_part(id, w.pivot_local(id));
        let r = vec3_ui(ui, &mut p, 0.25, "");
        if r.0 {
            x.pivot = Some(w.from_part(id, p));
        }
        (changed, begin) = (changed | r.0, begin | r.1);
        ui.horizontal(|ui| {
            if ui.button("Center").on_hover_text("Turn about the middle").clicked() {
                x.pivot = None;
                (changed, begin) = (true, true);
            }
            if let Some((_, size)) = w.bounds(id).filter(|_| !n.is_group() && !n.tilted()) {
                if ui.button("Bottom").on_hover_text("Turn about the middle of the bottom face").clicked() {
                    x.pivot = Some(w.from_part(id, Vec3::new(size.x as f32 * 0.5, size.y as f32 * 0.5, 0.0)));
                    (changed, begin) = (true, true);
                }
            }
        });
        if changed && Some(&*x) != Extras::of(w, id).as_ref() {
            cmds.push(Command::Tree(TreeOp::Extras(id, x.clone(), begin)));
        }
    });
}

/// Named attachment points of node `id`.
fn sockets_ui(ui: &mut Ui, w: &World, id: usize, cmds: &mut Vec<Command>) {
    let Some(mut x) = Extras::of(w, id) else { return };
    let (mut changed, mut begin) = (false, false);
    CollapsingHeader::new("Sockets").default_open(false).show(ui, |ui| {
        let mut remove = None;
        for (k, sock) in x.sockets.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                let r = ui.add(egui::TextEdit::singleline(&mut sock.name).desired_width(ui.available_width() - 34.0));
                if r.changed() {
                    (changed, begin) = (true, true);
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
            (changed, begin) = (changed | r.0, begin | r.1);
        }
        if let Some(k) = remove {
            x.sockets.remove(k);
            (changed, begin) = (true, true);
        }
        if ui.button("+ Socket").on_hover_text("A named attachment point, for snapping parts here and for game engines").clicked() {
            let name = (1..).map(|i| format!("socket{i}")).find(|s| !x.sockets.iter().any(|o| &o.name == s)).unwrap();
            x.sockets.push(Socket { name, pos: w.pivot_local(id) });
            (changed, begin) = (true, true);
        }
    });
    if changed && Some(&x) != Extras::of(w, id).as_ref() {
        cmds.push(Command::Tree(TreeOp::Extras(id, x, begin)));
    }
}

/// Buttons snapping node `id` onto the sockets of what it's inside.
fn socket_snap(ui: &mut Ui, w: &World, id: usize, parent: usize, cmds: &mut Vec<Command>) {
    let sockets = w.node(parent).map(|n| n.sockets.clone()).unwrap_or_default();
    if parent == ROOT || sockets.is_empty() {
        return;
    }
    ui.horizontal_wrapped(|ui| {
        ui.weak("snap to");
        for s in sockets {
            if ui.small_button(&s.name).on_hover_text("Move this part's pivot onto the socket").clicked() {
                cmds.push(Command::Tree(TreeOp::Attach(id, parent, Some(s.name.clone()))));
            }
        }
    });
}
