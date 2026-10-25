//! The window's reference picture: an image on a plane beside the model
//! (behind it as seen from the chosen view) to model over, kept with the
//! project. Agents' pinned reference views can be shown the same way.

use std::path::{Path, PathBuf};

use glam::{IVec3, Vec3};
use serde_json::{Value, json};

use crate::modeler::RefPlane;
use crate::ui::Command;

/// Which side the picture is seen from, as agents' reference views.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum View {
    #[default]
    Front,
    Back,
    Left,
    Right,
    Top,
}

impl View {
    pub const ALL: [View; 5] = [View::Front, View::Back, View::Left, View::Right, View::Top];

    pub fn key(self) -> &'static str {
        match self {
            View::Front => "front",
            View::Back => "back",
            View::Left => "left",
            View::Right => "right",
            View::Top => "top",
        }
    }

    fn parse(s: &str) -> Option<View> {
        View::ALL.into_iter().find(|v| v.key() == s)
    }

    /// Horizontal and vertical axes, and whether screen right runs toward
    /// the horizontal axis's negative end (as screenshot's views).
    fn axes(self) -> (usize, usize, bool) {
        match self {
            View::Front => (0, 2, false),
            View::Back => (0, 2, true),
            View::Right => (1, 2, false),
            View::Left => (1, 2, true),
            View::Top => (0, 1, false),
        }
    }
}

/// Requests from the Reference window, run after the frame.
#[derive(Clone, Debug, PartialEq)]
pub enum RefCmd {
    /// Ask for a picture file.
    Pick,
    /// Use this picture.
    Load(PathBuf),
    /// Show an agent's reference view of this project.
    Agent(View),
    Clear,
}

#[derive(Default)]
pub struct Backdrop {
    pub open: bool,
    pub show: bool,
    pub path: Option<PathBuf>,
    /// Picture size in pixels.
    pub size: (u32, u32),
    pub view: View,
    /// Voxels tall (top view: deep along y).
    pub height: f32,
    /// World position of the picture's bottom center along the view's
    /// horizontal and vertical axes.
    pub origin: [f32; 2],
    pub opacity: f32,
    /// Over the voxels instead of behind them.
    pub on_top: bool,
    pub flip: bool,
    /// The project these settings belong to.
    project: Option<String>,
    /// Agent reference views saved for this project.
    pub agent_views: Vec<View>,
}

fn dir(project: &str) -> PathBuf {
    crate::app::workspace_dir().join("refs").join(project)
}

impl Backdrop {
    /// The plane to draw for a view of `dims` voxels whose cell 0 is at
    /// world `origin`, or None when hidden.
    pub fn plane(&self, dims: IVec3, origin: IVec3) -> Option<RefPlane> {
        if !self.show || self.path.is_none() || self.size.1 == 0 {
            return None;
        }
        let (h, v, mirrored) = self.view.axes();
        let width = self.height * self.size.0 as f32 / self.size.1 as f32;
        let o = origin.as_vec3();
        let (ch, cv) = (self.origin[0] - o[h], self.origin[1] - o[v]);
        let mut right = Vec3::ZERO;
        right[h] = if mirrored != self.flip { -width } else { width };
        let mut up = Vec3::ZERO;
        up[v] = self.height;
        let mut corner = Vec3::ZERO;
        corner[h] = ch - right[h] / 2.0;
        corner[v] = cv;
        // Behind the canvas as seen from the view (the top view lies just
        // above the ground).
        let d = 3 - h - v;
        let far = match self.view {
            View::Front | View::Left => dims[d] as f32 + 1.0,
            View::Back | View::Right => -1.0,
            View::Top => 0.02,
        };
        corner[d] = far;
        Some(RefPlane { corner, right, up, opacity: self.opacity, on_top: self.on_top })
    }

    /// Size and place the picture to stand over a view of `dims` at world
    /// `origin`: as tall as the canvas, centered, on the ground.
    pub fn fit(&mut self, dims: IVec3, origin: IVec3) {
        let (h, v, _) = self.view.axes();
        self.height = dims[v].max(1) as f32;
        self.origin = [origin[h] as f32 + dims[h] as f32 / 2.0, origin[v] as f32];
    }

    /// Switch to `project`'s saved reference (saving the last one's).
    /// Returns the picture to upload when it changed (None inside: none).
    pub fn sync_project(&mut self, project: &str) -> Option<Option<PathBuf>> {
        if self.project.as_deref() == Some(project) {
            return None;
        }
        if let Some(old) = self.project.take() {
            self.save(&old);
        }
        self.project = Some(project.to_string());
        let open = self.open;
        *self = Backdrop { open, project: Some(project.to_string()), opacity: 0.5, show: true, ..Default::default() };
        self.agent_views = View::ALL.into_iter().filter(|v| dir(project).join(format!("{}.json", v.key())).exists()).collect();
        let meta: Option<Value> = std::fs::read_to_string(dir(project).join("window.json")).ok().and_then(|s| serde_json::from_str(&s).ok());
        if let Some(m) = meta {
            let f = |k: &str, d: f64| m[k].as_f64().unwrap_or(d) as f32;
            self.path = m["path"].as_str().map(PathBuf::from).filter(|p| p.exists());
            self.view = m["view"].as_str().and_then(View::parse).unwrap_or_default();
            self.height = f("height", 32.0);
            self.origin = [f("h", 0.0), f("v", 0.0)];
            self.opacity = f("opacity", 0.5);
            self.show = m["show"].as_bool().unwrap_or(true);
            self.on_top = m["on_top"].as_bool().unwrap_or(false);
            self.flip = m["flip"].as_bool().unwrap_or(false);
        }
        Some(self.path.clone())
    }

    /// Remember the settings for the current project.
    pub fn save_current(&self) {
        if let Some(p) = &self.project {
            self.save(p);
        }
    }

    fn save(&self, project: &str) {
        let path = dir(project).join("window.json");
        let Some(pic) = &self.path else {
            let _ = std::fs::remove_file(path);
            return;
        };
        let meta = json!({
            "path": pic.display().to_string(), "view": self.view.key(), "height": self.height, "h": self.origin[0], "v": self.origin[1],
            "opacity": self.opacity, "show": self.show, "on_top": self.on_top, "flip": self.flip,
        });
        let _ = std::fs::create_dir_all(dir(project));
        let _ = std::fs::write(path, serde_json::to_string_pretty(&meta).unwrap_or_default());
    }

    /// An agent's saved view: its picture, and its pinned (or traced)
    /// scale and place when it has one.
    pub fn agent(&mut self, view: View) -> Option<PathBuf> {
        let project = self.project.clone()?;
        let d = dir(&project);
        let meta: Value = serde_json::from_str(&std::fs::read_to_string(d.join(format!("{}.json", view.key()))).ok()?).ok()?;
        let pair = |k: &str| meta[k].as_array().and_then(|a| Some([a.first()?.as_f64()? as f32, a.get(1)?.as_f64()? as f32]));
        self.view = view;
        self.flip = false;
        if let Some((h, o)) = meta["height"].as_f64().zip(pair("origin")).or(meta["traced_height"].as_f64().zip(pair("traced_origin"))) {
            self.height = h as f32;
            self.origin = o;
        }
        Some(d.join(format!("{}.png", view.key())))
    }
}

/// The Reference window.
pub fn window(ctx: &egui::Context, b: &mut Backdrop, dims: IVec3, origin: IVec3, cmds: &mut Vec<Command>) {
    let mut open = b.open;
    let before = (b.view, b.height, b.origin, b.opacity, b.show, b.on_top, b.flip);
    egui::Window::new("Reference").open(&mut open).resizable(false).default_width(220.0).show(ctx, |ui| {
        ui.horizontal(|ui| {
            if ui.button("Load picture…").on_hover_text("A PNG or JPEG to model over (or drop one on the window with Shift)").clicked() {
                cmds.push(Command::Reference(RefCmd::Pick));
            }
            if b.path.is_some() && ui.button("Remove").clicked() {
                cmds.push(Command::Reference(RefCmd::Clear));
            }
        });
        if !b.agent_views.is_empty() {
            ui.horizontal_wrapped(|ui| {
                ui.label("Agent's:");
                for v in b.agent_views.clone() {
                    if ui.small_button(v.key()).on_hover_text("Show the reference view an agent set for this project, at its pinned scale").clicked() {
                        cmds.push(Command::Reference(RefCmd::Agent(v)));
                    }
                }
            });
        }
        let Some(path) = &b.path else {
            ui.weak("No picture. It shows on a plane behind the model, as seen from the view you pick.");
            return;
        };
        ui.add(egui::Label::new(egui::RichText::new(path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()).weak()).truncate()).on_hover_text(path.display().to_string());
        ui.horizontal(|ui| {
            ui.checkbox(&mut b.show, "Show");
            ui.checkbox(&mut b.on_top, "Over model").on_hover_text("Draw over the voxels instead of behind them");
            ui.checkbox(&mut b.flip, "Flip");
        });
        egui::ComboBox::from_label("Seen from").selected_text(b.view.key()).show_ui(ui, |ui| {
            for v in View::ALL {
                ui.selectable_value(&mut b.view, v, v.key());
            }
        });
        ui.add(egui::Slider::new(&mut b.opacity, 0.05..=1.0).text("Opacity"));
        egui::Grid::new("ref_place").num_columns(2).show(ui, |ui| {
            ui.label("Height");
            ui.add(egui::DragValue::new(&mut b.height).range(1.0..=4096.0).speed(0.25).suffix(" vox"));
            ui.end_row();
            ui.label("Across");
            ui.add(egui::DragValue::new(&mut b.origin[0]).speed(0.25));
            ui.end_row();
            ui.label("Up");
            ui.add(egui::DragValue::new(&mut b.origin[1]).speed(0.25));
            ui.end_row();
        });
        if ui.button("Fit to canvas").on_hover_text("As tall as the canvas, centered on it").clicked() {
            b.fit(dims, origin);
        }
    });
    b.open = open;
    if before != (b.view, b.height, b.origin, b.opacity, b.show, b.on_top, b.flip) {
        b.save_current();
    }
}

/// Load `path` as the reference picture (rows top down, RGBA).
pub fn load(path: &Path) -> anyhow::Result<(u32, u32, Vec<u8>)> {
    let img = crate::control::surface::load_image(path)?;
    // Big photos are shrunk to fit a texture comfortably.
    let max = 2048usize;
    let (w, h) = if img.w.max(img.h) > max {
        let k = max as f32 / img.w.max(img.h) as f32;
        (((img.w as f32 * k) as usize).max(1), ((img.h as f32 * k) as usize).max(1))
    } else {
        (img.w, img.h)
    };
    let mut out = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            out.extend_from_slice(&img.px[(y * img.h / h) * img.w + x * img.w / w]);
        }
    }
    Ok((w as u32, h as u32, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backdrop(view: View) -> Backdrop {
        Backdrop { show: true, path: Some("x.png".into()), size: (2, 1), view, height: 10.0, origin: [5.0, 0.0], opacity: 1.0, ..Default::default() }
    }

    #[test]
    fn front_picture_stands_behind_the_canvas() {
        let p = backdrop(View::Front).plane(IVec3::new(10, 8, 10), IVec3::ZERO).unwrap();
        assert_eq!(p.corner, Vec3::new(-5.0, 9.0, 0.0));
        assert_eq!(p.right, Vec3::new(20.0, 0.0, 0.0));
        assert_eq!(p.up, Vec3::new(0.0, 0.0, 10.0));
    }

    #[test]
    fn back_and_left_views_run_the_other_way() {
        let p = backdrop(View::Back).plane(IVec3::splat(10), IVec3::ZERO).unwrap();
        assert_eq!((p.corner.x, p.right.x, p.corner.y), (15.0, -20.0, -1.0));
        let p = backdrop(View::Left).plane(IVec3::splat(10), IVec3::ZERO).unwrap();
        assert_eq!((p.corner.y, p.right.y, p.corner.x), (15.0, -20.0, 11.0));
    }

    #[test]
    fn placement_is_in_world_coordinates() {
        // Editing a part whose cell 0 sits at world x 4: the picture
        // stays put in the world.
        let p = backdrop(View::Front).plane(IVec3::splat(10), IVec3::new(4, 0, 0)).unwrap();
        assert_eq!(p.corner.x, -9.0);
    }
}
