//! The animation timeline under the 3D view: play, scrub, frame count and
//! speed, and keys / flipbook frames for the picked part.

use std::time::Instant;

use egui::{Color32, RichText, Sense, Ui};

use crate::anim::Anim;
use crate::outline::TreeOp;
use crate::rig::Bone;
use crate::scene::Scene;
use crate::ui::Command;
use crate::world::{World, ROOT};

#[derive(Default)]
pub struct TimelineState {
    pub open: bool,
    pub playing: bool,
    /// The motion preset picked in the timeline.
    pub motion: usize,
    /// A glTF file whose clips can be imported, and the clip picked.
    pub clips: Option<(std::path::PathBuf, Vec<String>)>,
    pub clip: usize,
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
    ui.horizontal_wrapped(|ui| {
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
    });
    // Second row: keys for the picked part, motions and imports.
    ui.horizontal_wrapped(|ui| {
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
        let rig = scene.picked.first().copied();
        // Only the motions made for this kind of rig.
        let fits: Vec<usize> = match rig_kinds(w, rig) {
            Some(quad) => (0..motions.len()).filter(|&k| motions[k].0.starts_with("quad") == quad).collect(),
            None => (0..motions.len()).collect(),
        };
        if !fits.contains(&st.motion) {
            st.motion = fits[0];
        }
        egui::ComboBox::from_id_salt("motion").selected_text(motions[st.motion].0).show_ui(ui, |ui| {
            for &k in &fits {
                let (name, about) = motions[k];
                ui.selectable_value(&mut st.motion, k, name).on_hover_text(about);
            }
        });
        let name = motions[st.motion].0;
        let on = rig_label(w, rig, name.starts_with("quad"));
        let tip = format!("Key this motion onto {on}'s parts (found by name: hips, torso, head, upper_arm_l ...), replacing their keys. Pick a part of another rig to animate that one.");
        if ui.button("Apply motion").on_hover_text(tip).clicked() {
            let frames = if a.frames > 1 { a.frames } else { 24 };
            cmds.push(Command::Tree(TreeOp::Motion(name.to_string(), frames)));
        }
        ui.add(egui::Label::new(RichText::new(format!("on: {on}")).weak()).extend());
        ui.separator();
        if ui.button("Import...").on_hover_text("Animation clips from a .glb / .gltf file (e.g. Quaternius's free Universal Animation Library), retargeted onto the rig").clicked() {
            cmds.push(Command::ImportAnimation(None));
        }
        if let Some((path, names)) = &st.clips {
            let pick = st.clip.min(names.len().saturating_sub(1));
            egui::ComboBox::from_id_salt("clip").selected_text(names.get(pick).map_or("", String::as_str)).width(150.0).show_ui(ui, |ui| {
                for (k, n) in names.iter().enumerate() {
                    ui.selectable_value(&mut st.clip, k, n);
                }
            });
            let on = rig_label(w, rig, false);
            if ui.button("Use clip").on_hover_text(format!("Retarget this clip onto {on}")).clicked() {
                cmds.push(Command::ImportAnimation(Some((path.clone(), pick))));
            }
            ui.add(egui::Label::new(RichText::new(format!("on: {on}")).weak()).extend());
        }
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

/// Top-level objects that are rigs (have parts named like bones).
fn rigs(w: &World) -> Vec<usize> {
    let tops = w.node(ROOT).map(|n| n.children().to_vec()).unwrap_or_default();
    tops.into_iter().filter(|&t| !crate::rig::find_bones(w, t).is_empty()).collect()
}

/// Whether `top` is a four-legged rig (true) or a humanoid (false).
fn is_quad(w: &World, top: usize) -> bool {
    crate::rig::find_bones(w, top).keys().any(|b| matches!(b, Bone::LegFL | Bone::LegFR | Bone::LegBL | Bone::LegBR))
}

/// Which kind of rig a motion would go onto: the picked part's, or the
/// scene's when all its rigs are one kind. None: either.
fn rig_kinds(w: &World, picked: Option<usize>) -> Option<bool> {
    if let Some(p) = picked {
        let top = crate::rig::top_of(w, p);
        return (!crate::rig::find_bones(w, top).is_empty()).then(|| is_quad(w, top));
    }
    let kinds: Vec<bool> = rigs(w).into_iter().map(|t| is_quad(w, t)).collect();
    kinds.first().copied().filter(|&k| kinds.iter().all(|&o| o == k))
}

/// The rig a motion (`quad`: a four-legged one) goes onto, as
/// `rig::apply_motion` picks it: its name, and its number when the scene
/// has several rigs.
pub fn rig_label(w: &World, picked: Option<usize>, quad: bool) -> String {
    let root = crate::rig::rig_root(w, picked, if quad { Bone::LegFL } else { Bone::UpperLegL });
    if root == ROOT {
        return "no rig".into();
    }
    let all = rigs(w);
    match all.iter().position(|&t| t == root) {
        Some(k) if all.len() > 1 => format!("{} (rig {})", w.label(root), k + 1),
        Some(_) => w.label(root),
        None => format!("{} (not a rig)", w.label(root)),
    }
}

/// A one-line summary of timeline settings.
pub fn describe(a: &Anim) -> String {
    format!("{} frames at {} fps{}{}", a.frames, a.fps, if a.looping { ", looping" } else { "" }, if a.smooth { ", smooth" } else { ", stepped" })
}
