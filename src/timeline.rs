//! The animation timeline under the 3D view: play, scrub, frame count and
//! speed, and keys / flipbook frames for the picked part.

use std::time::Instant;

use egui::{Color32, RichText, Sense, Ui};

use crate::anim::Anim;
use crate::outline::TreeOp;
use crate::scene::Scene;
use crate::ui::Command;

#[derive(Default)]
pub struct TimelineState {
    pub open: bool,
    pub playing: bool,
    /// The motion preset picked in the timeline.
    pub motion: usize,
    /// When the shown frame last advanced.
    last: Option<Instant>,
}

impl TimelineState {
    pub fn play(&mut self, on: bool) {
        self.playing = on;
        self.last = None;
    }

    /// While playing: the next frame, once it's due.
    pub fn tick(&mut self, scene: &Scene) -> Option<u32> {
        if !self.playing {
            return None;
        }
        let a = &scene.world.anim;
        let now = Instant::now();
        let last = *self.last.get_or_insert(now);
        if now.duration_since(last).as_secs_f32() < 1.0 / a.fps.max(0.5) {
            return None;
        }
        self.last = Some(now);
        let next = scene.world.frame + 1;
        if next < a.frames {
            Some(next)
        } else if a.looping {
            Some(0)
        } else {
            self.playing = false;
            None
        }
    }
}

const KEY: Color32 = Color32::from_rgb(255, 190, 60);
const FLIP: Color32 = Color32::from_rgb(110, 200, 255);

pub fn panel(ui: &mut Ui, scene: &Scene, st: &mut TimelineState, cmds: &mut Vec<Command>) {
    let w = &scene.world;
    let a = w.anim.clone();
    let frame = w.frame;
    let part = scene.picked.first().copied().filter(|_| scene.picked.len() == 1);
    ui.horizontal(|ui| {
        ui.label(RichText::new("Animation").strong());
        if ui.button("|<").on_hover_text("First frame").clicked() {
            cmds.push(Command::Frame(0));
        }
        if ui.button("<").on_hover_text("Previous frame [,]").clicked() {
            cmds.push(Command::Frame(frame.checked_sub(1).unwrap_or(a.frames - 1)));
        }
        let label = if st.playing { "Pause" } else { "Play" };
        if ui.add_enabled(a.frames > 1, egui::Button::new(label)).on_hover_text("Play / pause [/]").clicked() {
            st.play(!st.playing);
        }
        if ui.button(">").on_hover_text("Next frame [.]").clicked() {
            cmds.push(Command::Frame((frame + 1) % a.frames.max(1)));
        }
        ui.label(format!("frame {frame} of 0-{}", a.frames - 1));
        ui.separator();
        let mut next = a.clone();
        ui.label("Frames");
        ui.add(egui::DragValue::new(&mut next.frames).range(1..=10_000));
        ui.label("FPS");
        ui.add(egui::DragValue::new(&mut next.fps).range(0.5..=120.0).speed(0.2).max_decimals(1));
        ui.checkbox(&mut next.looping, "Loop");
        ui.checkbox(&mut next.smooth, "Smooth").on_hover_text("Blend between keys; off holds each key until the next, like MagicaVoxel");
        if next != a {
            cmds.push(Command::Tree(TreeOp::SetAnim(next)));
        }
        ui.separator();
        match part {
            Some(id) => {
                let n = w.node(id);
                let keyed = n.is_some_and(|n| n.keys.iter().any(|k| k.frame == frame));
                ui.label(RichText::new(w.label(id)).strong());
                if ui.button("Key").on_hover_text("Record this part's pose on this frame (moving it while animating does this too)").clicked() {
                    cmds.push(Command::Tree(TreeOp::Key(id)));
                }
                if ui.add_enabled(keyed, egui::Button::new("Remove key")).clicked() {
                    cmds.push(Command::Tree(TreeOp::DeleteKey(id)));
                }
                let model = n.is_some_and(|n| !n.is_group());
                let split = n.is_some_and(|n| n.flip.iter().any(|s| s.frame == frame && frame > 0));
                if model && !split && ui.button("New drawing here").on_hover_text("Flipbook: from this frame on, the part shows its own copy of the voxels to redraw").clicked() {
                    cmds.push(Command::Tree(TreeOp::SplitFrame(id)));
                }
                if split && ui.button("Remove drawing").on_hover_text("Go back to the previous drawing from this frame").clicked() {
                    cmds.push(Command::Tree(TreeOp::MergeFrame(id)));
                }
            }
            None => {
                ui.weak("Pick one part to key it");
            }
        }
        ui.separator();
        let motions = crate::rig::MOTIONS;
        egui::ComboBox::from_id_salt("motion").selected_text(motions[st.motion.min(motions.len() - 1)].0).show_ui(ui, |ui| {
            for (k, (name, about)) in motions.iter().enumerate() {
                ui.selectable_value(&mut st.motion, k, *name).on_hover_text(*about);
            }
        });
        if ui.button("Apply motion").on_hover_text("Key this motion onto the rig's parts (found by name: hips, torso, head, upper_arm_l ...), replacing their keys").clicked() {
            let frames = if a.frames > 1 { a.frames } else { 24 };
            cmds.push(Command::Tree(TreeOp::Motion(motions[st.motion].0.to_string(), frames)));
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("Close").clicked() {
                st.open = false;
            }
        });
    });
    // The frame strip: click or drag to scrub. Keys of the picked part are
    // diamonds, its flipbook drawings squares.
    let height = 22.0;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), height), Sense::click_and_drag());
    let p = ui.painter_at(rect);
    let n = a.frames.max(1);
    let cell = rect.width() / n as f32;
    let x = |f: u32| rect.left() + cell * (f as f32 + 0.5);
    p.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    let step = ((40.0 / cell).ceil() as u32).max(1);
    for f in 0..n {
        let tall = f % step == 0;
        let top = if tall { rect.top() + 3.0 } else { rect.bottom() - 6.0 };
        p.line_segment([egui::pos2(x(f), top), egui::pos2(x(f), rect.bottom())], (1.0, Color32::from_gray(70)));
        if tall && cell * step as f32 > 18.0 {
            p.text(egui::pos2(x(f) + 2.0, rect.top() + 1.0), egui::Align2::LEFT_TOP, f.to_string(), egui::FontId::proportional(9.0), Color32::from_gray(120));
        }
    }
    p.rect_filled(egui::Rect::from_center_size(egui::pos2(x(frame), rect.center().y), egui::vec2(cell.max(2.0), height)), 0.0, Color32::from_rgba_unmultiplied(80, 160, 255, 90));
    if let Some(id) = part {
        for (f, flip) in w.marks(id) {
            let c = egui::pos2(x(f), rect.center().y + 3.0);
            if flip {
                p.rect_filled(egui::Rect::from_center_size(c, egui::vec2(7.0, 7.0)), 1.0, FLIP);
            } else {
                let d = 4.5;
                p.add(egui::Shape::convex_polygon(vec![c + egui::vec2(0.0, -d), c + egui::vec2(d, 0.0), c + egui::vec2(0.0, d), c + egui::vec2(-d, 0.0)], KEY, egui::Stroke::NONE));
            }
        }
    }
    if let Some(pos) = resp.interact_pointer_pos().filter(|_| resp.clicked() || resp.dragged()) {
        let f = (((pos.x - rect.left()) / cell).floor().max(0.0) as u32).min(n - 1);
        if f != frame {
            cmds.push(Command::Frame(f));
        }
    }
}

/// A one-line summary of timeline settings.
pub fn describe(a: &Anim) -> String {
    format!("{} frames at {} fps{}{}", a.frames, a.fps, if a.looping { ", looping" } else { "" }, if a.smooth { ", smooth" } else { ", stepped" })
}
