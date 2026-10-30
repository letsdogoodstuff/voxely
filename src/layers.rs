//! MagicaVoxel's Layer panel (`o_layer.ui`): one row per layer with a
//! visibility switch, the layer's name (click: new objects go on it) and an
//! arrow that puts the picked objects on it. Rename swaps the names for text
//! fields.

use egui::{RichText, Ui};

use crate::outline::TreeOp;
use crate::ui::Command;
use crate::world::World;

/// The name fields' text while editing names.
#[derive(Default)]
pub struct LayerState {
    names: Option<Vec<String>>,
}

pub fn panel(ui: &mut Ui, w: &World, picked: &[usize], state: &mut LayerState, cmds: &mut Vec<Command>) {
    // The layer the picked objects are on, when they share one.
    let mut on = picked.iter().filter_map(|&p| w.node(p)).map(|n| n.layer);
    let first = on.next();
    let on = first.filter(|&f| on.all(|l| l == f));
    let editing = state.names.is_some();
    ui.horizontal(|ui| {
        ui.weak(if editing { "Type a name, Enter to keep it" } else { "Click a name: new objects go there" });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.selectable_label(editing, if editing { "Done" } else { "Rename" }).on_hover_text("Edit Layer Names").clicked() {
                // Closing keeps what was typed (unchanged names are no-ops).
                match state.names.take() {
                    Some(names) => cmds.extend(w.layers.iter().zip(names).map(|(l, n)| Command::Tree(TreeOp::LayerName(l.id, n)))),
                    None => state.names = Some(w.layers.iter().map(|l| l.label()).collect()),
                }
            }
        });
    });
    let tag = egui::vec2(22.0, ui.spacing().interact_size.y);
    for (k, l) in w.layers.iter().enumerate() {
        ui.horizontal(|ui| {
            let mut shown = !l.hidden;
            if ui.checkbox(&mut shown, "").on_hover_text("Set Layer Visibility").changed() {
                cmds.push(Command::Tree(TreeOp::LayerShow(l.id)));
            }
            // Layer color, as on the World-mode boxes.
            let [r, g, b] = crate::modeler::LAYER_COLORS[l.id.rem_euclid(8) as usize];
            let (swatch, _) = ui.allocate_exact_size(egui::vec2(4.0, tag.y - 6.0), egui::Sense::hover());
            ui.painter().rect_filled(swatch, 1.0, egui::Color32::from_rgb(r, g, b));
            // The arrow on the right, the name filling what's left.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let r = ui.add_enabled(!picked.is_empty(), egui::Button::selectable(on == Some(l.id), RichText::new("\u{2190}").monospace()).min_size(tag));
                if r.on_hover_text("Set Layer for Selected Objects").clicked() {
                    cmds.push(Command::Tree(TreeOp::LayerSet(l.id)));
                }
                let size = [ui.available_width(), tag.y];
                match &mut state.names {
                    Some(names) => {
                        let r = ui.add_sized(size, egui::TextEdit::singleline(&mut names[k]).hint_text(l.id.to_string()));
                        if r.lost_focus() {
                            cmds.push(Command::Tree(TreeOp::LayerName(l.id, names[k].clone())));
                        }
                    }
                    None => {
                        let mut text = RichText::new(l.label());
                        if l.hidden {
                            text = text.weak();
                        }
                        let r = ui.add_sized(size, egui::Button::selectable(w.current_layer == l.id, text));
                        if r.on_hover_text("Set Layer for New Objects").clicked() {
                            cmds.push(Command::Tree(TreeOp::LayerCurrent(l.id)));
                        }
                    }
                }
            });
        });
    }
}
