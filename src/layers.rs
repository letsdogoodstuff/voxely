//! MagicaVoxel's Layer panel: a list of layers, as many as you like. Each
//! row has a visibility switch, the layer's color, its name (click: new
//! objects go on it; double-click: rename) and an arrow that puts the picked
//! objects on it. Right-click a row for the rest; + adds a layer.

use egui::{RichText, Ui};

use crate::outline::TreeOp;
use crate::ui::Command;
use crate::world::World;

#[derive(Default)]
pub struct LayerState {
    /// The layer being renamed, the field's text, and whether the field
    /// still needs focus.
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
    ui.horizontal(|ui| {
        ui.weak("Click: new objects go here");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("+").on_hover_text("New layer").clicked() {
                cmds.push(tree(TreeOp::LayerAdd));
            }
        });
    });
    if !ui.input(|i| i.pointer.any_down()) {
        state.coloring = None;
    }
    let tag = egui::vec2(22.0, ui.spacing().interact_size.y);
    let rows = (ui.ctx().content_rect().height() * 0.35).max(8.0 * (tag.y + ui.spacing().item_spacing.y));
    egui::ScrollArea::vertical().id_salt("layers").max_height(rows).show(ui, |ui| {
        for l in &w.layers {
            let count = w.nodes.iter().skip(1).flatten().filter(|n| n.layer == l.id && !n.is_group()).count();
            ui.horizontal(|ui| {
                let mut shown = !l.hidden;
                if ui.checkbox(&mut shown, "").on_hover_text("Set Layer Visibility").changed() {
                    cmds.push(tree(TreeOp::LayerShow(l.id)));
                }
                let mut color = l.color;
                ui.spacing_mut().interact_size.x = tag.y;
                if egui::color_picker::color_edit_button_srgb(ui, &mut color).on_hover_text("Layer color").changed() {
                    cmds.push(tree(TreeOp::LayerColor(l.id, color, state.coloring != Some(l.id))));
                    state.coloring = Some(l.id);
                }
                // The arrow on the right, the name filling what's left.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let r = ui.add_enabled(!picked.is_empty(), egui::Button::selectable(on == Some(l.id), RichText::new("\u{2190}").monospace()).min_size(tag));
                    if r.on_hover_text("Set Layer for Selected Objects").clicked() {
                        cmds.push(tree(TreeOp::LayerSet(l.id)));
                    }
                    let size = [ui.available_width(), tag.y];
                    match &mut state.renaming {
                        Some((id, text, focus)) if *id == l.id => {
                            let r = ui.add_sized(size, egui::TextEdit::singleline(text).hint_text(l.id.to_string()));
                            if std::mem::take(focus) {
                                r.request_focus();
                            }
                            if r.lost_focus() {
                                if !ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                                    cmds.push(tree(TreeOp::LayerName(l.id, text.clone())));
                                }
                                state.renaming = None;
                            }
                        }
                        _ => {
                            let mut text = RichText::new(l.label());
                            if l.hidden {
                                text = text.weak();
                            }
                            let hint = format!("{count} object{}\nClick: Set Layer for New Objects\nDouble-click: rename · Right-click: more", if count == 1 { "" } else { "s" });
                            let r = ui.add_sized(size, egui::Button::selectable(w.current_layer == l.id, text)).on_hover_text(hint);
                            if r.double_clicked() {
                                state.renaming = Some((l.id, l.label(), true));
                            } else if r.clicked() {
                                cmds.push(tree(TreeOp::LayerCurrent(l.id)));
                            }
                            r.context_menu(|ui| {
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
                        }
                    }
                });
            });
        }
    });
}
