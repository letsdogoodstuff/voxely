//! Transform gizmo on screen, like MagicaVoxel's: arrows along the axes
//! (Move, Scale, Wrap) and rings around them (Rotate). The editor says
//! which gizmo to show and on what box (`Editor::gizmo`); this lays out
//! its handles in screen pixels, picks the one under the mouse and draws.

use glam::{Vec2, Vec3};

use crate::edit::{Gizmo, GizmoKind, Handle};
use crate::gpu::Renderer;

/// Arrow length past the box, and ring margin round it, in points.
const ARROW_PT: f32 = 70.0;
const RING_PT: f32 = 30.0;
/// How close (in points) the mouse must be to grab a handle.
const PICK_PT: f32 = 7.0;

fn unit(axis: usize) -> Vec3 {
    let mut d = Vec3::ZERO;
    d[axis] = 1.0;
    d
}

/// Each handle's polyline on screen, in physical pixels (`ppp` pixels per
/// point). Arrows run base to tip.
pub fn shapes(g: &Gizmo, r: &Renderer, ppp: f32) -> Vec<(Handle, Vec<Vec2>)> {
    let proj = |p: Vec3| r.project(p).map(|(x, y)| Vec2::new(x, y));
    let c = (g.lo + g.hi) * 0.5;
    let half = (g.hi - g.lo) * 0.5;
    let (_, right, _) = r.camera.basis();
    let (Some(pc), Some(pr)) = (proj(c), proj(c + right)) else { return Vec::new() };
    // World units per point at the box center, so handles keep their size
    // on screen as the camera zooms.
    let per_pt = ppp / (pr - pc).length().max(1e-3);
    let line = |pts: Vec<Vec3>| pts.into_iter().map(proj).collect::<Option<Vec<Vec2>>>();
    let mut out = Vec::new();
    for axis in 0..3 {
        let d = unit(axis);
        match g.kind {
            GizmoKind::Move | GizmoKind::Faces => {
                for sign in [-1, 1] {
                    let s = sign as f32;
                    let base = if g.kind == GizmoKind::Move { c } else { c + d * half[axis] * s };
                    let tip = c + d * (half[axis] + ARROW_PT * per_pt) * s;
                    if let Some(pts) = line(vec![base, tip]) {
                        out.push((Handle::Arrow { axis, sign }, pts));
                    }
                }
            }
            GizmoKind::Rings => {
                let radius = half.length() + RING_PT * per_pt;
                let (u, w) = (unit((axis + 1) % 3), unit((axis + 2) % 3));
                let pts = (0..=64).map(|k| {
                    let t = k as f32 / 64.0 * std::f32::consts::TAU;
                    c + (u * t.cos() + w * t.sin()) * radius
                });
                if let Some(pts) = line(pts.collect()) {
                    out.push((Handle::Ring { axis }, pts));
                }
            }
        }
    }
    out
}

fn segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-6)).clamp(0.0, 1.0);
    (a + ab * t).distance(p)
}

/// The handle under `mouse` (physical pixels), if any.
pub fn pick(g: &Gizmo, r: &Renderer, mouse: Vec2, ppp: f32) -> Option<Handle> {
    shapes(g, r, ppp)
        .into_iter()
        .filter_map(|(h, pts)| {
            let d = pts.windows(2).map(|s| segment_distance(mouse, s[0], s[1])).fold(f32::MAX, f32::min);
            (d <= PICK_PT * ppp).then_some((h, d))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(h, _)| h)
}

/// Draw the handles, `lit` highlighted (hovered or being dragged).
pub fn draw(p: &egui::Painter, g: &Gizmo, r: &Renderer, ppp: f32, lit: Option<Handle>) {
    const AXIS: [egui::Color32; 3] = [egui::Color32::from_rgb(170, 30, 30), egui::Color32::from_rgb(125, 135, 30), egui::Color32::from_rgb(40, 100, 160)];
    let lit_color = egui::Color32::from_rgb(255, 185, 40);
    for (h, pts) in shapes(g, r, ppp) {
        let axis = match h {
            Handle::Arrow { axis, .. } | Handle::Ring { axis } => axis,
        };
        let color = if lit == Some(h) { lit_color } else { AXIS[axis] };
        let pts: Vec<egui::Pos2> = pts.iter().map(|q| egui::pos2(q.x / ppp, q.y / ppp)).collect();
        p.add(egui::Shape::line(pts.clone(), egui::Stroke::new(2.5, color)));
        if let (Handle::Arrow { .. }, [.., a, b]) = (h, pts.as_slice()) {
            // Arrowhead at the tip.
            let dir = (*b - *a).normalized();
            if dir.is_finite() {
                let side = egui::vec2(-dir.y, dir.x);
                let head = vec![*b + dir * 4.0, *b - dir * 9.0 + side * 5.0, *b - dir * 9.0 - side * 5.0];
                p.add(egui::Shape::convex_polygon(head, color, egui::Stroke::NONE));
            }
        }
    }
}
