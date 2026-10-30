//! MagicaVoxel's Layer panel, as many layers as you like. Each row is the
//! layer's color dot (click: show or hide), its name (click: new objects go
//! on it; double-click: rename) and a `<` that marks, and sets, the layer of
//! the picked objects. The pencil renames them all; + adds one; right-click
//! a row for its color and the rest.

use egui::{Color32, Sense, Ui};

use crate::outline::TreeOp;
use crate::ui::Command;
use crate::world::World;

#[derive(Default)]
pub struct LayerState {
    /// Every name field's text while the pencil is on.
    names: Option<Vec<String>>,
    /// One layer being renamed (double-click), and whether its field still
    /// needs focus.
    renaming: Option<(i32, String, bool)>,
    /// The layer whose color is being dragged (one undo step per drag).
    coloring: Option<i32>,
}

pub fn panel(ui: &mut Ui, w: &World, picked: &[usize], state: &mut LayerState, cmds: &mut Vec<Command>) {
    let tree = |o: TreeOp| Command::Tree(o);
    // The layer the picked objects are on, when they share one.
    let mut on = picked.iter().filter_map(|&p| w.node(p)).map(|n| n.layer);
    let first = on.next();
    let on = first.filter(|&f| on.all(|l| l == f));
    let editing = state.names.is_some();
    ui.horizontal(|ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("+").on_hover_text("New layer").clicked() {
                cmds.push(tree(TreeOp::LayerAdd));
            }
            if pencil(ui, editing).on_hover_text("Edit Layer Names").clicked() {
                // Closing keeps what was typed (unchanged names are no-ops).
                match state.names.take() {
                    Some(names) => cmds.extend(w.layers.iter().zip(names).map(|(l, n)| tree(TreeOp::LayerName(l.id, n)))),
                    None => state.names = Some(w.layers.iter().map(|l| l.label()).collect()),
                }
            }
        });
    });
    // Layers added while the pencil is on get fields too.
    if let Some(names) = &mut state.names {
        names.resize_with(w.layers.len(), String::new);
    }
    if !ui.input(|i| i.pointer.any_down()) {
        state.coloring = None;
    }
    let h = ui.spacing().interact_size.y;
    let rows = (ui.ctx().content_rect().height() * 0.35).max(8.0 * (h + 1.0));
    egui::ScrollArea::vertical().id_salt("layers").max_height(rows).show(ui, |ui| {
        ui.spacing_mut().item_spacing = egui::vec2(1.0, 1.0);
        for (k, l) in w.layers.iter().enumerate() {
            let count = w.nodes.iter().skip(1).flatten().filter(|n| n.layer == l.id && !n.is_group()).count();
            ui.horizontal(|ui| {
                let well = ui.visuals().widgets.inactive.weak_bg_fill;
                // The color dot: filled when shown, a ring when hidden.
                let (cell, r) = ui.allocate_exact_size(egui::vec2(h + 4.0, h), Sense::click());
                ui.painter().rect_filled(cell, 0.0, if r.hovered() { ui.visuals().widgets.hovered.weak_bg_fill } else { well });
                let [cr, cg, cb] = l.color;
                let color = Color32::from_rgb(cr, cg, cb);
                if l.hidden {
                    ui.painter().circle_stroke(cell.center(), h * 0.2, egui::Stroke::new(1.5, color.gamma_multiply(0.6)));
                } else {
                    ui.painter().circle_filled(cell.center(), h * 0.22, color);
                }
                if r.on_hover_text("Set Layer Visibility").clicked() {
                    cmds.push(tree(TreeOp::LayerShow(l.id)));
                }
                // `<` on the right, the name filling what's left.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (cell, r) = ui.allocate_exact_size(egui::vec2(h * 0.8, h), if picked.is_empty() { Sense::hover() } else { Sense::click() });
                    ui.painter().rect_filled(cell, 0.0, if r.hovered() && !picked.is_empty() { ui.visuals().widgets.hovered.weak_bg_fill } else { well });
                    if on == Some(l.id) {
                        ui.painter().text(cell.center(), egui::Align2::CENTER_CENTER, "<", egui::TextStyle::Button.resolve(ui.style()), ui.visuals().strong_text_color());
                    }
                    if r.on_hover_text("Set Layer for Selected Objects").clicked() {
                        cmds.push(tree(TreeOp::LayerSet(l.id)));
                    }
                    let size = egui::vec2(ui.available_width(), h);
                    let field = match (&mut state.names, &mut state.renaming) {
                        (Some(names), _) => Some((&mut names[k], false)),
                        (None, Some((id, text, focus))) if *id == l.id => Some((text, std::mem::take(focus))),
                        _ => None,
                    };
                    if let Some((text, focus)) = field {
                        let r = ui.add_sized(size, egui::TextEdit::singleline(text).hint_text(l.id.to_string()));
                        if focus {
                            r.request_focus();
                        }
                        if r.lost_focus() && !ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                            cmds.push(tree(TreeOp::LayerName(l.id, text.clone())));
                        }
                        if r.lost_focus() && !editing {
                            state.renaming = None;
                        }
                        return;
                    }
                    let current = w.current_layer == l.id;
                    let (cell, r) = ui.allocate_exact_size(size, Sense::click());
                    let fill = if current {
                        ui.visuals().selection.bg_fill
                    } else if r.hovered() {
                        ui.visuals().widgets.hovered.weak_bg_fill
                    } else {
                        well
                    };
                    ui.painter().rect_filled(cell, 0.0, fill);
                    let text_color = if current {
                        ui.visuals().selection.stroke.color
                    } else if l.hidden {
                        ui.visuals().weak_text_color()
                    } else {
                        ui.visuals().strong_text_color()
                    };
                    let font = egui::TextStyle::Button.resolve(ui.style());
                    let label = crate::outline::fit_middle(ui, &l.label(), &font, cell.width() - 12.0);
                    ui.painter().text(cell.left_center() + egui::vec2(6.0, 0.0), egui::Align2::LEFT_CENTER, label, font, text_color);
                    let hint = format!("{} · {count} object{}\nClick: Set Layer for New Objects\nDouble-click: rename · Right-click: more", l.label(), if count == 1 { "" } else { "s" });
                    let r = r.on_hover_text(hint);
                    if r.double_clicked() {
                        state.renaming = Some((l.id, l.label(), true));
                    } else if r.clicked() {
                        cmds.push(tree(TreeOp::LayerCurrent(l.id)));
                    }
                    r.context_menu(|ui| {
                        ui.horizontal(|ui| {
                            let mut c = l.color;
                            if egui::color_picker::color_edit_button_srgb(ui, &mut c).changed() {
                                cmds.push(tree(TreeOp::LayerColor(l.id, c, state.coloring != Some(l.id))));
                                state.coloring = Some(l.id);
                            }
                            ui.label("Color");
                        });
                        if ui.button("Rename").clicked() {
                            state.renaming = Some((l.id, l.label(), true));
                            ui.close();
                        }
                        if ui.add_enabled(count > 0, egui::Button::new("Pick objects on this layer")).clicked() {
                            cmds.push(tree(TreeOp::LayerPick(l.id)));
                            ui.close();
                        }
                        if ui.add_enabled(!picked.is_empty(), egui::Button::new("Put picked objects here")).clicked() {
                            cmds.push(tree(TreeOp::LayerSet(l.id)));
                            ui.close();
                        }
                        ui.separator();
                        if ui.button("Sort layers by name").clicked() {
                            cmds.push(tree(TreeOp::LayerSort));
                            ui.close();
                        }
                        let to = w.layers.iter().find(|o| o.id != l.id).map(|o| o.label()).unwrap_or_default();
                        let hint = if count > 0 { format!("Its objects move to layer {to}") } else { String::new() };
                        if ui.add_enabled(w.layers.len() > 1, egui::Button::new("Delete layer")).on_hover_text(hint).clicked() {
                            cmds.push(tree(TreeOp::LayerDelete(l.id)));
                            ui.close();
                        }
                    });
                });
            });
        }
    });
}

/// A small pencil button, lit while `on`.
fn pencil(ui: &mut Ui, on: bool) -> egui::Response {
    let s = ui.spacing().interact_size.y;
    let (rect, r) = ui.allocate_exact_size(egui::vec2(s, s), Sense::click());
    let v = ui.style().interact_selectable(&r, on);
    if on || r.hovered() {
        ui.painter().rect_filled(rect, 3.0, v.weak_bg_fill);
    }
    let c = rect.center();
    let (a, b) = (c + egui::vec2(-s * 0.22, s * 0.22), c + egui::vec2(s * 0.22, -s * 0.22));
    ui.painter().line_segment([a + egui::vec2(s * 0.06, -s * 0.06), b], egui::Stroke::new(s * 0.14, v.fg_stroke.color));
    ui.painter().line_segment([a, a + egui::vec2(s * 0.05, -s * 0.05)], egui::Stroke::new(s * 0.06, v.fg_stroke.color));
    r
}
