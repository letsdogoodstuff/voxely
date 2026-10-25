//! Looking at the model: multi-view screenshots, overlays, edit highlights.

use anyhow::{Result, bail};
use glam::{IVec3, Vec3};
use serde_json::{Value, json};

use super::{Ctx, boolean, get, int, string, text, tool};
use crate::gpu::{Camera, project_point};
use crate::scene::Projection;

pub(super) mod draw;
use draw::{BLUE, Canvas, GREEN, RED, Rgb, WHITE, text_size};

pub const PART_TOOLS: &[&str] = &[];

pub fn tools() -> Vec<Value> {
    vec![tool(
        "screenshot",
        "Fast image of the model in the editor's modeler view, plus a text summary (part, size, voxel count, bounds, top colors, views, last edit). \
         views: \"single\" (default: the current camera, as the window shows it), \"quad\" (2x2: front, right, top orthographic and a 3/4 perspective), \
         \"all\" (front, back, left, right, top, iso), or a list of names from current, front, back, left, right, top, bottom, iso, 3/4. \
         scope \"scene\" (default when the project has more than one part) shows every visible part at full color, framed together, with rulers \
         in world coordinates; scope \"part\" (default with one part) frames the part being edited with the other parts dimmed, rulers in the part's \
         own coordinates (what the voxel tools use). Either way the window's mode and camera are left as they were. \
         focus zooms named views onto a box {from, to} (inclusive cells, in the coordinates the rulers use) or onto a part or group by name, \
         e.g. to inspect a face, a hand or a car's wheel. \
         With overlay (default on for named views) each tile gets a title, an x/y/z axis gizmo (red x, green y, blue z) and rulers labeled in voxel \
         coordinates: on front/back/left/right/top/bottom along the image edges, otherwise along the framed box's edges. Ticks mark cell centers. \
         highlight_last_edit marks the cells the last edit changed: magenta where they hold voxels, yellow where they're now empty, faint through other voxels.",
        json!({
            "width": { "type": "integer", "description": "Whole image width (default 800 for one view, 512 per tile otherwise)" },
            "height": { "type": "integer", "description": "Whole image height (default 600 for one view, 384 per tile otherwise)" },
            "views": { "type": ["string", "array"], "items": { "type": "string" }, "description": "\"single\", \"quad\", \"all\", one view name, or a list of view names" },
            "overlay": { "type": "boolean", "description": "Axis gizmo, rulers and titles (default: on unless the only view is the current camera)" },
            "grid_step": { "type": "integer", "description": "Voxels between labeled ruler ticks (default 8; widened if labels would crowd)" },
            "fit": { "type": "string", "enum": ["canvas", "voxels"], "description": "What named views frame: the whole canvas (default; in scene scope the box around all parts) or just the solid voxels" },
            "scope": { "type": "string", "enum": ["scene", "part"], "description": "\"scene\": all parts, world coordinates; \"part\": the edited part, its own coordinates, others dimmed. Default: scene if there is more than one part" },
            "focus": {
                "description": "Zoom named views onto a part or group name, or a box {from: [x, y, z], to: [x, y, z]} in the rulers' coordinates (world in scene scope, the part's own in part scope)",
                "anyOf": [
                    { "type": "string" },
                    { "type": "object", "properties": { "from": { "type": "array", "items": { "type": "integer" }, "minItems": 3, "maxItems": 3 }, "to": { "type": "array", "items": { "type": "integer" }, "minItems": 3, "maxItems": 3 } }, "required": ["from", "to"] },
                ],
            },
            "highlight_last_edit": { "type": "boolean", "description": "Mark the cells changed by the last edit, if it was to the part shown" },
        }),
        &[],
    )]
}

pub fn dispatch(ctx: &mut Ctx, name: &str, args: &Value) -> Option<Result<Vec<Value>>> {
    match name {
        "screenshot" => Some(screenshot(ctx, args)),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum View {
    Current,
    Front,
    Back,
    Left,
    Right,
    Top,
    Bottom,
    Iso,
    ThreeQuarter,
}

impl View {
    fn parse(s: &str) -> Option<View> {
        Some(match s.to_ascii_lowercase().as_str() {
            "current" | "camera" | "single" => View::Current,
            "front" => View::Front,
            "back" => View::Back,
            "left" => View::Left,
            "right" | "side" => View::Right,
            "top" => View::Top,
            "bottom" => View::Bottom,
            "iso" | "isometric" => View::Iso,
            "3/4" | "three_quarter" | "three-quarter" | "perspective" => View::ThreeQuarter,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            View::Current => "CURRENT",
            View::Front => "FRONT -Y",
            View::Back => "BACK +Y",
            View::Left => "LEFT -X",
            View::Right => "RIGHT +X",
            View::Top => "TOP +Z",
            View::Bottom => "BOTTOM -Z",
            View::Iso => "ISO",
            View::ThreeQuarter => "3/4",
        }
    }

    /// Camera yaw and pitch (degrees) and whether it's orthographic, for
    /// the named views. The eye sits on the side the name says (front: -y).
    fn orient(self) -> Option<(f32, f32, bool)> {
        // Straight up/down would make the camera basis degenerate.
        const UP: f32 = 89.99;
        Some(match self {
            View::Current => return None,
            View::Front => (-90.0, 0.0, true),
            View::Back => (90.0, 0.0, true),
            View::Right => (0.0, 0.0, true),
            View::Left => (180.0, 0.0, true),
            View::Top => (-90.0, UP, true),
            View::Bottom => (-90.0, -UP, true),
            View::Iso => (225.0, 35.264, true),
            View::ThreeQuarter => (225.0, 25.0, false),
        })
    }
}

fn parse_views(v: Option<&Value>) -> Result<Vec<View>> {
    let one = |s: &str| -> Result<Vec<View>> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "quad" => vec![View::Front, View::Right, View::Top, View::ThreeQuarter],
            "turnaround" => vec![View::Front, View::Right, View::Back, View::Left],
            "all" => vec![View::Front, View::Back, View::Left, View::Right, View::Top, View::Iso],
            _ => vec![View::parse(s).ok_or_else(|| anyhow::anyhow!("unknown view {s:?}"))?],
        })
    };
    match v {
        None => Ok(vec![View::Current]),
        Some(Value::String(s)) => one(s),
        Some(Value::Array(a)) if !a.is_empty() && a.len() <= 16 => {
            let mut out = Vec::new();
            for x in a {
                out.extend(one(x.as_str().ok_or_else(|| anyhow::anyhow!("views are strings"))?)?);
            }
            Ok(out)
        }
        _ => bail!("`views` is \"single\", \"quad\", \"all\", \"turnaround\", a view name, or a list of 1-16 view names"),
    }
}

/// One tile: the camera it's drawn from and the inner rectangle the model
/// is framed in (the rest is margin for rulers).
struct Shot {
    camera: Camera,
    projection: Projection,
    inner: (f32, f32, f32, f32),
}

/// Frame the box `lo..hi` in a `w` x `h` tile from a named view's angle.
fn frame_view(yaw: f32, pitch: f32, ortho: bool, lo: Vec3, hi: Vec3, fov: f32, w: f32, h: f32, inner: (f32, f32, f32, f32)) -> Shot {
    let center = (lo + hi) * 0.5;
    let mut camera = Camera { target: center, yaw: yaw.to_radians(), pitch: pitch.to_radians(), dist: 1.0 };
    let tan_half = (fov.to_radians() * 0.5).tan();
    if !ortho {
        let half_v = fov.to_radians() * 0.5;
        let half_h = (half_v.tan() * w / h).atan();
        camera.dist = ((hi - lo).length() * 0.5).max(1.0) / half_v.min(half_h).sin() * 1.1;
        return Shot { camera, projection: Projection::Perspective, inner: (0.0, 0.0, w, h) };
    }
    let (_, right, up) = camera.basis();
    let (mut mu, mut mv) = (0.5f32, 0.5f32);
    for i in 0..8 {
        let c = Vec3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        ) - center;
        mu = mu.max(c.dot(right).abs());
        mv = mv.max(c.dot(up).abs());
    }
    let (ix, iy, iw, ih) = inner;
    // World units per pixel, with a little air around the box.
    let s = (2.0 * mu / iw).max(2.0 * mv / ih) * 1.06;
    camera.dist = s * h * 0.5 / tan_half;
    let (cx, cy) = (ix + iw * 0.5, iy + ih * 0.5);
    camera.target = center - right * (cx - w * 0.5) * s + up * (cy - h * 0.5) * s;
    Shot { camera, projection: Projection::Orthographic, inner }
}

/// A camera at `yaw`/`pitch` (radians) fitting the box `lo..hi` in a `w` x
/// `h` image, leaving `margin` of the image free on every side. Unlike
/// `frame_view`'s bounding sphere, perspective fits the box's corners.
fn fit_box(yaw: f32, pitch: f32, ortho: bool, lo: Vec3, hi: Vec3, fov: f32, w: f32, h: f32, margin: f32) -> Camera {
    let center = (lo + hi) * 0.5;
    let mut camera = Camera { target: center, yaw, pitch, dist: 1.0 };
    let (fwd, right, up) = camera.basis();
    let tan_half = (fov.to_radians() * 0.5).tan();
    let (kx, ky) = (tan_half * w / h * (1.0 - 2.0 * margin), tan_half * (1.0 - 2.0 * margin));
    let corners = (0..8).map(|i| {
        Vec3::new(if i & 1 == 0 { lo.x } else { hi.x }, if i & 2 == 0 { lo.y } else { hi.y }, if i & 4 == 0 { lo.z } else { hi.z }) - center
    });
    if ortho {
        // Half the image height in world units is dist * tan_half.
        let half = corners.fold(0.5f32, |m, c| m.max(c.dot(right).abs() * tan_half / kx).max(c.dot(up).abs() * tan_half / ky));
        camera.dist = half / tan_half;
        return camera;
    }
    // A corner at depth dist + z projects to x / (dist + z).
    camera.dist = corners.fold(1.0f32, |d, c| {
        let z = c.dot(fwd);
        d.max(c.dot(right).abs() / kx - z).max(c.dot(up).abs() / ky - z).max(1.0 - z)
    });
    camera
}

/// The path-traced voxels' box in camera coordinates (cell 0 of the edited
/// volume at the origin), or None if there are none.
fn traced_bounds(scene: &crate::scene::Scene) -> Option<(Vec3, Vec3)> {
    let flat = scene.flat();
    let (lo, hi) = super::bounds(&flat)?;
    let off = scene.flat_offset();
    Some(((lo + off).as_vec3(), (hi + off + IVec3::ONE).as_vec3()))
}

/// `render` with `views` or `frame`: path trace from cameras fitted to the
/// whole scene and tile the images, labeled.
pub fn render_framed(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let views = match get(a, "views") {
        Some(v) => parse_views(Some(v))?,
        None => vec![View::Current],
    };
    let projection = match string(a, "projection") {
        None => None,
        Some("orthographic" | "ortho") => Some(Projection::Orthographic),
        Some("perspective") => Some(Projection::Perspective),
        Some(p) => bail!("unknown projection {p:?}; use \"orthographic\" or \"perspective\""),
    };
    let n = views.len() as u32;
    let cols = if n <= 4 { n } else { (n as f32).sqrt().ceil() as u32 };
    let rows = n.div_ceil(cols);
    let (dw, dh) = if n == 1 { (960, 640) } else { (480 * cols, 600 * rows) };
    let width = int(a, "width").map_or(dw, |v| v as u32).clamp(16 * cols, 8192);
    let height = int(a, "height").map_or(dh, |v| v as u32).clamp(16 * rows, 8192);
    let (tw, th) = (width / cols, height / rows);
    let spp = int(a, "samples").unwrap_or(128).clamp(1, 65536) as u32;
    let (lo, hi) = traced_bounds(ctx.scene).ok_or_else(|| anyhow::anyhow!("nothing to render: the scene has no voxels"))?;

    let t = std::time::Instant::now();
    let mut r = crate::gpu::Renderer::new(ctx.r.device.clone(), ctx.r.queue.clone(), ctx.scene, None, tw, th);
    r.settings = ctx.r.settings.clone();
    let fov = r.settings.fov;
    let mut out = Canvas::new(tw * cols, th * rows, [0, 0, 0, 255].repeat((tw * cols * th * rows) as usize));
    let mut cams = Vec::new();
    for (i, &view) in views.iter().enumerate() {
        let (yaw, pitch, ortho) = match view.orient() {
            Some((y, p, o)) => (y.to_radians(), p.to_radians(), o),
            None => (ctx.r.camera.yaw, ctx.r.camera.pitch, ctx.r.settings.projection == Projection::Orthographic),
        };
        let proj = projection.unwrap_or(if view == View::Current { ctx.r.settings.projection } else if ortho { Projection::Orthographic } else { Projection::Perspective });
        let cam = fit_box(yaw, pitch, proj == Projection::Orthographic, lo, hi, fov, tw as f32, th as f32, 0.06);
        r.settings.projection = proj;
        r.set_camera(cam);
        let mut tile = Canvas::new(tw, th, r.render_image(spp)?);
        if n > 1 {
            let sc = if tw.min(th) >= 280 { 2 } else { 1 };
            tile.text(8, 8, view.name(), sc, WHITE);
        }
        out.blit(&tile, (i as u32 % cols * tw) as i32, (i as u32 / cols * th) as i32);
        // In world coordinates, as set_camera takes them.
        let t = cam.target + ctx.scene.origin.as_vec3();
        cams.push(format!(
            "{}: yaw {:.0}, pitch {:.0}, distance {:.1}, target [{:.1}, {:.1}, {:.1}], {:?}",
            if view == View::Current { "camera" } else { view.name() },
            cam.yaw.to_degrees(),
            cam.pitch.to_degrees(),
            cam.dist,
            t.x,
            t.y,
            t.z,
            proj
        ));
    }
    for c in 1..cols {
        out.rect((c * tw) as i32 - 1, 0, 2, out.h as i32, [0, 0, 0], 0.8);
    }
    for row in 1..rows {
        out.rect(0, (row * th) as i32 - 1, out.w as i32, 2, [0, 0, 0], 0.8);
    }
    let o = ctx.scene.origin.as_vec3();
    let mut note = format!(
        "Rendered {n} view{} at {tw}x{th} each, {spp} spp, in {:.1?}, framed on the voxels [{:.0}, {:.0}, {:.0}]..[{:.0}, {:.0}, {:.0}] world (exclusive). \
         Cameras (set_camera takes these to keep one; the window's camera is unchanged): {}.",
        if n == 1 { "" } else { "s" },
        t.elapsed(),
        lo.x + o.x,
        lo.y + o.y,
        lo.z + o.z,
        hi.x + o.x,
        hi.y + o.y,
        hi.z + o.z,
        cams.join("; ")
    );
    if let Some(p) = string(a, "save_path") {
        let path = std::path::Path::new(p);
        crate::save_png(path, out.w, out.h, &out.px)?;
        note.push_str(&format!(" Saved {}.", std::fs::canonicalize(path).unwrap_or(path.into()).display()));
    }
    Ok(vec![super::image(out.w, out.h, &out.px)?, text(note)])
}

fn axis_color(a: usize) -> Rgb {
    [RED, GREEN, BLUE][a]
}

const AXIS: [&str; 3] = ["X", "Y", "Z"];

/// Everything an overlay needs to know about the scene.
struct Frame {
    lo: Vec3,
    hi: Vec3,
    /// Added to cell indices for labels (world coordinates: the drawn
    /// volume's origin).
    offset: IVec3,
    step: i32,
    fov: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Scope {
    /// Every visible part, world coordinates.
    Scene,
    /// The edited part with the rest dimmed, its own coordinates.
    Part,
}

fn parse_scope(ctx: &Ctx, a: &Value) -> Result<Scope> {
    Ok(match string(a, "scope") {
        None if ctx.scene.world.models().len() > 1 => Scope::Scene,
        None | Some("part") => Scope::Part,
        Some("scene") => Scope::Scene,
        Some(s) => bail!("unknown scope {s:?}; use \"scene\" or \"part\""),
    })
}

/// The `focus` box in drawn-volume cells (min, max exclusive). `world0` is
/// the drawn volume's cell 0 in the world, `offset` what labels add.
fn parse_focus(ctx: &Ctx, a: &Value, world0: IVec3, offset: IVec3) -> Result<Option<(Vec3, Vec3)>> {
    let (lo, hi) = match get(a, "focus") {
        None | Some(Value::Null) => return Ok(None),
        Some(v @ Value::String(_)) => {
            let id = super::find_part(ctx.scene, Some(v))?;
            let (lo, size) = ctx.scene.synced_world().bounds(id).ok_or_else(|| anyhow::anyhow!("`{}` has no voxels to focus on", ctx.scene.world.label(id)))?;
            (lo - world0, lo - world0 + size)
        }
        Some(v @ Value::Object(_)) => {
            let (from, to) = (super::ivec3(v, "from")?, super::ivec3(v, "to")?);
            (from.min(to) - offset, from.max(to) - offset + IVec3::ONE)
        }
        Some(_) => bail!("`focus` is a part name or {{from: [x, y, z], to: [x, y, z]}}"),
    };
    Ok(Some((lo.as_vec3(), hi.as_vec3())))
}

fn screenshot(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let views = parse_views(get(a, "views"))?;
    let n = views.len() as u32;
    let cols = (n as f32).sqrt().ceil() as u32;
    let rows = n.div_ceil(cols);
    let (dw, dh) = if n == 1 { (800, 600) } else { (512 * cols, 384 * rows) };
    let width = int(a, "width").map_or(dw, |v| v as u32).clamp(16 * cols, 4096);
    let height = int(a, "height").map_or(dh, |v| v as u32).clamp(16 * rows, 4096);
    let (tw, th) = (width / cols, height / rows);
    let overlay = boolean(a, "overlay").unwrap_or(views != [View::Current]);
    let step = int(a, "grid_step").unwrap_or(8).clamp(1, 1024) as i32;
    let fit_voxels = match string(a, "fit").unwrap_or("canvas") {
        "canvas" => false,
        "voxels" => true,
        f => bail!("unknown fit {f:?}; use \"canvas\" or \"voxels\""),
    };

    // Scene scope while a part is edited: draw every part flattened (World
    // mode's view) instead of the part with the rest dimmed.
    let scope = parse_scope(ctx, a)?;
    let whole = if scope == Scope::Scene && ctx.scene.editing().is_some() { ctx.scene.synced_world().composite(None) } else { None };
    let (vol, world0) = match &whole {
        Some(c) => (&c.volume, c.origin),
        None => (&ctx.scene.volume, ctx.scene.origin),
    };
    let offset = if whole.is_some() || ctx.scene.editing().is_none() { world0 } else { IVec3::ZERO };
    let solid = super::bounds(vol);
    let focus = parse_focus(ctx, a, world0, offset)?;
    let (lo, hi) = match (focus, solid) {
        (Some(f), _) => f,
        (None, Some((l, h))) if fit_voxels => (l.as_vec3(), (h + IVec3::ONE).as_vec3()),
        _ => (Vec3::ZERO, vol.dims.as_vec3()),
    };
    let frame = Frame { lo, hi, offset, step, fov: ctx.r.settings.fov };
    let dims = vol.dims;
    // The edited part's cells in the drawn volume.
    let part_at = ctx.scene.origin - world0;

    // Highlight the last edit if it was to what's shown.
    let mut notes = Vec::new();
    let mut highlighted = false;
    if boolean(a, "highlight_last_edit").unwrap_or(false) {
        match &ctx.ed.tools.last_edit {
            None => notes.push("No edit to highlight yet.".to_string()),
            Some(e) if e.part != ctx.scene.editing() => notes.push("The last edit was to another part; nothing highlighted.".to_string()),
            Some(e) if e.resized => notes.push("The last edit resized the canvas; nothing highlighted.".to_string()),
            Some(_) if whole.is_some() && ctx.scene.editing().is_some_and(|id| ctx.scene.world.tilted(id)) => {
                notes.push("The edited part is freely rotated, so its cells don't line up with the scene view; use scope \"part\" to highlight.".to_string())
            }
            Some(e) => {
                let cells: Vec<(IVec3, bool)> = e.cells.iter().map(|&p| (p + part_at, vol.get(p + part_at) != 0)).collect();
                ctx.r.set_highlight(dims, &cells);
                highlighted = !cells.is_empty();
            }
        }
    }

    let mut out = Canvas::new(tw * cols, th * rows, [24, 24, 24, 255].repeat((tw * cols * th * rows) as usize));
    let mut names = Vec::new();
    if let Some(c) = &whole {
        ctx.r.modeler_show(&c.volume, None, None);
    }
    // World mode's pick highlight would tint the parts; agents see real colors.
    let result = super::unhighlighted(ctx, |ctx| -> Result<()> {
        for (i, &view) in views.iter().enumerate() {
            let (w, h) = (tw as f32, th as f32);
            let sc = if tw.min(th) >= 280 { 2 } else { 1 };
            let shot = match view.orient() {
                None => {
                    // The window's camera, moved with what's drawn (scene scope draws from another cell 0).
                    let mut camera = ctx.r.camera;
                    camera.target += part_at.as_vec3();
                    Shot { camera, projection: ctx.r.settings.projection, inner: (0.0, 0.0, w, h) }
                }
                Some((yaw, pitch, ortho)) => {
                    let inner = if overlay { margins(&frame, sc, w, h) } else { (4.0, 4.0, w - 8.0, h - 8.0) };
                    frame_view(yaw, pitch, ortho, lo, hi, frame.fov, w, h, inner)
                }
            };
            let px = ctx.r.modeler_image_from(tw, th, &shot.camera, shot.projection)?;
            let mut tile = Canvas::new(tw, th, px);
            let title = if overlay { overlay_tile(&mut tile, &shot, view, &frame, sc) } else { view.name().to_string() };
            if !overlay && n > 1 {
                tile.text(6, 6, &title, sc, WHITE);
            }
            names.push(title);
            out.blit(&tile, (i as u32 % cols * tw) as i32, (i as u32 / cols * th) as i32);
        }
        Ok(())
    });
    if highlighted {
        ctx.r.set_highlight(dims, &[]);
    }
    if whole.is_some() {
        ctx.r.modeler_show(&ctx.scene.volume, ctx.scene.context.as_ref(), ctx.ed.editor.selection());
    }
    result?;
    // Tile borders.
    for c in 1..cols {
        out.rect((c * tw) as i32 - 1, 0, 2, out.h as i32, [0, 0, 0], 0.8);
    }
    for r in 1..rows {
        out.rect(0, (r * th) as i32 - 1, out.w as i32, 2, [0, 0, 0], 0.8);
    }

    let what = Shown { volume: whole.as_ref().map_or(&ctx.scene.volume, |c| &c.volume), whole: whole.is_some(), focus: focus.is_some(), fit_voxels: fit_voxels && solid.is_some() };
    let mut summary = summary(ctx, &what, &names, &frame, views != [View::Current], highlighted);
    for n in notes {
        summary.push('\n');
        summary.push_str(&n);
    }
    Ok(vec![super::image(out.w, out.h, &out.px)?, text(summary)])
}

/// The inner rectangle of an orthographic tile, leaving room for the
/// title and rulers.
fn margins(f: &Frame, sc: i32, w: f32, h: f32) -> (f32, f32, f32, f32) {
    let widest = [f.lo, f.hi].iter().flat_map(|v| v.to_array()).zip([0, 1, 2, 0, 1, 2]).map(|(v, k)| (v as i32 + f.offset[k]).to_string().len()).max().unwrap_or(2);
    let lh = (7 * sc) as f32;
    let left = text_size(&"0".repeat(widest), sc).0 as f32 + 12.0;
    let (top, bottom) = (lh + 12.0, lh + 12.0);
    (left, top, (w - left - 10.0).max(8.0), (h - top - bottom).max(8.0))
}

/// Draw the title, rulers and gizmo on one tile; returns its title.
fn overlay_tile(t: &mut Canvas, shot: &Shot, view: View, f: &Frame, sc: i32) -> String {
    let cam = &shot.camera;
    let (fwd, right, up) = cam.basis();
    let (w, h) = (t.w as f32, t.h as f32);
    let proj = |p: Vec3| project_point(cam, f.fov, shot.projection, w, h, p);
    let aligned = shot.projection == Projection::Orthographic && (0..3).any(|k| fwd[k].abs() > 0.999);
    let name = view.name();
    let mut hint = String::new();
    if aligned {
        let ah = (0..3).max_by(|&a, &b| right[a].abs().total_cmp(&right[b].abs())).unwrap();
        let av = (0..3).max_by(|&a, &b| up[a].abs().total_cmp(&up[b].abs())).unwrap();
        let sign = |v: f32| if v > 0.0 { "+" } else { "-" };
        hint = format!(": {}{} RIGHT, {}{} UP", sign(right[ah]), AXIS[ah], sign(up[av]), AXIS[av]);
        edge_rulers(t, shot, f, sc, ah, av, &proj);
    } else {
        box_rulers(t, cam, f, sc, &proj);
    }
    // The name at full size; the axis hint smaller if it would crowd the gizmo.
    t.text(6, 6, name, sc, WHITE);
    let (nw, _) = text_size(name, sc);
    let room = t.w as i32 - 6 - nw - 16 - 50 * sc;
    let hs = if text_size(&hint, sc).0 <= room { sc } else { 1 };
    t.text(6 + nw, 6 + (sc - hs) * 7, &hint, hs, WHITE);
    gizmo(t, fwd, right, up, sc);
    format!("{name}{hint}")
}

/// Major tick spacing along an axis `extent` voxels long: `step` (halved
/// until there are about three labels), then doubled until labels `room`
/// pixels wide fit `px_per_voxel` apart.
fn tick_step(step: i32, extent: i32, px_per_voxel: f32, room: f32) -> i32 {
    let mut s = step.max(1);
    while s > 1 && extent < 3 * s {
        s /= 2;
    }
    while (s as f32) * px_per_voxel < room && s < 1 << 20 {
        s *= 2;
    }
    s
}

/// Rulers along the bottom (axis `ah`) and left (axis `av`) of an
/// axis-aligned orthographic tile, with faint grid lines on the background.
fn edge_rulers(t: &mut Canvas, shot: &Shot, f: &Frame, sc: i32, ah: usize, av: usize, proj: &dyn Fn(Vec3) -> Option<(f32, f32)>) {
    let (ix, iy, iw, ih) = shot.inner;
    let bg = t.get(1, t.h as i32 - 2).unwrap_or([43, 43, 43]);
    let is_bg = |c: Rgb| (0..3).all(|k| (c[k] as i32 - bg[k] as i32).abs() <= 2);
    let center = (f.lo + f.hi) * 0.5;
    let at = |axis: usize, i: i32| {
        let mut p = center;
        p[axis] = i as f32 + 0.5;
        proj(p).unwrap_or((0.0, 0.0))
    };
    let (lo_i, hi_i) = (f.lo.as_ivec3(), f.hi.as_ivec3() - IVec3::ONE);
    let digits = |axis: usize| [lo_i[axis], hi_i[axis]].iter().map(|v| (v + f.offset[axis]).to_string().len()).max().unwrap_or(1);
    for (axis, horizontal) in [(ah, true), (av, false)] {
        let ppv = {
            let (a, b) = (at(axis, 0), at(axis, 1));
            ((b.0 - a.0).abs() + (b.1 - a.1).abs()).max(1e-3)
        };
        let room = if horizontal { text_size(&"0".repeat(digits(axis)), sc).0 as f32 + 8.0 } else { (7 * sc) as f32 + 6.0 };
        let major = tick_step(f.step, hi_i[axis] - lo_i[axis] + 1, ppv, room);
        let color = axis_color(axis);
        for i in lo_i[axis]..=hi_i[axis] {
            let value = i + f.offset[axis];
            let (x, y) = at(axis, i);
            let is_major = value.rem_euclid(major) == 0;
            if !is_major && ppv < 4.0 {
                continue;
            }
            if horizontal {
                let base = iy + ih;
                let len = if is_major { 6.0 } else { 3.0 };
                t.line((x, base), (x, base + len), 1.0, color, 1.0);
                if is_major {
                    for yy in iy as i32..base as i32 {
                        if t.get(x as i32, yy).is_some_and(is_bg) {
                            t.blend(x as i32, yy, WHITE, 0.10);
                        }
                    }
                    let s = value.to_string();
                    let (lw, _) = text_size(&s, sc);
                    t.text(x as i32 - lw / 2, (base + 8.0) as i32, &s, sc, color);
                }
            } else {
                let len = if is_major { 6.0 } else { 3.0 };
                t.line((ix - len, y), (ix, y), 1.0, color, 1.0);
                if is_major {
                    for xx in ix as i32..(ix + iw) as i32 {
                        if t.get(xx, y as i32).is_some_and(is_bg) {
                            t.blend(xx, y as i32, WHITE, 0.10);
                        }
                    }
                    let s = value.to_string();
                    let (lw, lh) = text_size(&s, sc);
                    t.text((ix - 8.0) as i32 - lw, y as i32 - lh / 2, &s, sc, color);
                }
            }
        }
    }
}

/// Rulers along three edges of the framed box (one per axis) for views
/// that aren't axis-aligned: the two floor edges nearest the camera and
/// the leftmost vertical edge.
fn box_rulers(t: &mut Canvas, cam: &Camera, f: &Frame, sc: i32, proj: &dyn Fn(Vec3) -> Option<(f32, f32)>) {
    let (lo, hi) = (f.lo, f.hi);
    let center = (lo + hi) * 0.5;
    let eye = cam.eye();
    let near = |k: usize| if eye[k] < center[k] { lo[k] } else { hi[k] };
    let Some(c2) = proj(center) else { return };
    // The vertical edge furthest left on screen.
    let corners = [(lo.x, lo.y), (hi.x, lo.y), (lo.x, hi.y), (hi.x, hi.y)];
    let left = corners
        .iter()
        .filter_map(|&(x, y)| proj(Vec3::new(x, y, lo.z)).map(|p| (p.0, (x, y))))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|c| c.1);
    let Some((zx, zy)) = left else { return };
    let edges: [(usize, Vec3); 3] = [(0, Vec3::new(0.0, near(1), lo.z)), (1, Vec3::new(near(0), 0.0, lo.z)), (2, Vec3::new(zx, zy, 0.0))];
    for (axis, base) in edges {
        let color = axis_color(axis);
        let point = |v: f32| {
            let mut p = base;
            p[axis] = v;
            proj(p)
        };
        let (Some(a), Some(b)) = (point(lo[axis]), point(hi[axis])) else { continue };
        t.line(a, b, 1.0, color, 0.8);
        let len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt().max(1e-3);
        let dir = ((b.0 - a.0) / len, (b.1 - a.1) / len);
        // Normal pointing away from the box's middle.
        let mut nrm = (-dir.1, dir.0);
        let mid = ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5);
        if (mid.0 - c2.0) * nrm.0 + (mid.1 - c2.1) * nrm.1 < 0.0 {
            nrm = (-nrm.0, -nrm.1);
        }
        let (lo_i, hi_i) = (lo[axis] as i32, hi[axis] as i32 - 1);
        let ppv = len / (hi[axis] - lo[axis]).max(1.0);
        let digits = [lo_i, hi_i].iter().map(|v| (v + f.offset[axis]).to_string().len()).max().unwrap_or(1);
        let room = text_size(&"0".repeat(digits), sc).0 as f32 * dir.0.abs() + (7 * sc) as f32 * dir.1.abs() + 8.0;
        let major = tick_step(f.step, hi_i - lo_i + 1, ppv, room);
        for i in lo_i..=hi_i {
            let value = i + f.offset[axis];
            let is_major = value.rem_euclid(major) == 0;
            if !is_major && ppv < 5.0 {
                continue;
            }
            let Some(p) = point(i as f32 + 0.5) else { continue };
            let tick = if is_major { 6.0 } else { 3.0 };
            t.line(p, (p.0 + nrm.0 * tick, p.1 + nrm.1 * tick), 1.0, color, 1.0);
            // The vertical edge's first label would sit on the floor edges' ends.
            if is_major && !(axis == 2 && i == lo_i) {
                let s = value.to_string();
                let (lw, lh) = text_size(&s, sc);
                let reach = 9.0 + (nrm.0.abs() * lw as f32 + nrm.1.abs() * lh as f32) * 0.5;
                let (cx, cy) = (p.0 + nrm.0 * reach, p.1 + nrm.1 * reach);
                t.text(cx as i32 - lw / 2, cy as i32 - lh / 2, &s, sc, color);
            }
        }
        // Name the axis past the high end.
        let (lw, lh) = text_size(AXIS[axis], sc);
        let reach = 10.0 + (lw.max(lh) as f32) * 0.5;
        t.text((b.0 + dir.0 * reach) as i32 - lw / 2, (b.1 + dir.1 * reach) as i32 - lh / 2, AXIS[axis], sc, color);
    }
}

/// An x/y/z axis gizmo in the top-right corner, farthest axis first.
fn gizmo(t: &mut Canvas, fwd: Vec3, right: Vec3, up: Vec3, sc: i32) {
    let len = 16.0 * sc as f32;
    let (cx, cy) = (t.w as f32 - len - 10.0 - 4.0 * sc as f32, 12.0 + len + 5.0 * sc as f32);
    let mut axes: Vec<usize> = (0..3).collect();
    let unit = |k: usize| Vec3::AXES[k];
    axes.sort_by(|&a, &b| unit(b).dot(fwd).total_cmp(&unit(a).dot(fwd)));
    t.rect(cx as i32 - 1, cy as i32 - 1, 3, 3, WHITE, 1.0);
    for k in axes {
        let d = (unit(k).dot(right), -unit(k).dot(up));
        let color = axis_color(k);
        let (lw, lh) = text_size(AXIS[k], sc);
        if d.0.hypot(d.1) < 0.2 {
            // Pointing at or away from the viewer: a ring of the axis color.
            let sign = if unit(k).dot(fwd) < 0.0 { "+" } else { "-" };
            let s = format!("{sign}{}", AXIS[k]);
            t.rect(cx as i32 - 3, cy as i32 - 3, 7, 7, color, 0.9);
            t.text(cx as i32 + 6, cy as i32 + 6, &s, sc, color);
            continue;
        }
        let end = (cx + d.0 * len, cy + d.1 * len);
        t.line((cx, cy), end, 2.0, color, 1.0);
        let reach = 4.0 + lw.max(lh) as f32 * 0.6;
        let n = d.0.hypot(d.1);
        t.text((end.0 + d.0 / n * reach) as i32 - lw / 2, (end.1 + d.1 / n * reach) as i32 - lh / 2, AXIS[k], sc, color);
    }
}

/// What a screenshot drew, for its summary.
struct Shown<'a> {
    volume: &'a crate::scene::Volume,
    /// Every part while one is edited (scene scope).
    whole: bool,
    focus: bool,
    fit_voxels: bool,
}

/// A compact text description of what the screenshot shows.
fn summary(ctx: &Ctx, what: &Shown, names: &[String], f: &Frame, framed: bool, highlighted: bool) -> String {
    let scene = &*ctx.scene;
    let v = what.volume;
    let d = v.dims;
    let mut lines = Vec::new();
    match scene.editing() {
        Some(id) if what.whole => lines.push(format!(
            "Scene view: every visible part ({}) at full color, spanning {}x{}x{} from {:?}. Coordinates below are world coordinates. \
             The part being edited, `{}` (id {id}), starts at {:?} world (its own coordinates = world - that); scope \"part\" shows it alone.",
            scene.world.models().len(),
            d.x,
            d.y,
            d.z,
            f.offset.to_array(),
            scene.world.label(id),
            scene.origin.to_array()
        )),
        Some(id) => lines.push(format!(
            "Part `{}` (id {id}, {} parts in scene): size {}x{}x{}, at {:?} world. Coordinates below are the part's own (0..size-1).",
            scene.world.label(id),
            scene.world.models().len(),
            d.x,
            d.y,
            d.z,
            scene.origin.to_array()
        )),
        None => lines.push(format!(
            "World mode, {} parts: scene spans {}x{}x{} from {:?}. Coordinates below are world coordinates.",
            scene.world.models().len(),
            d.x,
            d.y,
            d.z,
            scene.origin.to_array()
        )),
    }
    let mut counts = [0usize; 256];
    for (_, c) in v.cells.filled() {
        counts[c as usize] += 1;
    }
    let total: usize = counts[1..].iter().sum();
    let o = f.offset;
    let b = super::bounds(v).map_or("empty".to_string(), |(lo, hi)| format!("bounds {:?}..{:?} inclusive", (lo + o).to_array(), (hi + o).to_array()));
    lines.push(format!("{total} voxels, {b}."));
    let mut used: Vec<usize> = (1..256).filter(|&i| counts[i] > 0).collect();
    used.sort_by_key(|&i| std::cmp::Reverse(counts[i]));
    if !used.is_empty() {
        let top: Vec<String> = used.iter().take(6).map(|&i| format!("{i} {} x{}", super::hex(scene.palette[i]), counts[i])).collect();
        let more = if used.len() > 6 { format!(" (+{} more)", used.len() - 6) } else { String::new() };
        lines.push(format!("Colors: {}{more}.", top.join(", ")));
    }
    let cam = &ctx.r.camera;
    let target = cam.target + scene.origin.as_vec3();
    let views = names.join(" | ");
    let framing = if framed {
        let framed = match (what.focus, what.fit_voxels, what.whole || scene.editing().is_none()) {
            (true, _, _) => "focus box",
            (false, true, _) => "solid voxels",
            (false, false, true) => "box around all parts",
            (false, false, false) => "canvas",
        };
        format!(" Named views frame the {framed} {:?}..{:?} (exclusive); ruler ticks every {} at cell centers.", (f.lo.as_ivec3() + o).to_array(), (f.hi.as_ivec3() + o).to_array(), f.step)
    } else {
        String::new()
    };
    lines.push(format!(
        "Views: {views}.{framing} Window camera (unchanged): yaw {:.0}, pitch {:.0}, distance {:.1}, target [{:.1}, {:.1}, {:.1}] world, {:?}.",
        cam.yaw.to_degrees(),
        cam.pitch.to_degrees(),
        cam.dist,
        target.x,
        target.y,
        target.z,
        ctx.r.settings.projection
    ));
    if let Some(e) = &ctx.ed.tools.last_edit {
        let mut part = e.part.map_or("World".to_string(), |id| format!("`{}`", scene.world.label(id)));
        if e.part != scene.editing() || (what.whole && e.part.is_some()) {
            part.push_str(" (its own coordinates)");
        }
        let what = if e.resized {
            "resized the canvas".to_string()
        } else {
            let b = e.bounds.map_or(String::new(), |(lo, hi)| format!(", bounds {:?}..{:?}", lo.to_array(), hi.to_array()));
            format!("+{} added, -{} removed, {} recolored{b}", e.added, e.removed, e.recolored)
        };
        let mark = if highlighted { " (highlighted: magenta = holds voxels, yellow = now empty)" } else { "" };
        lines.push(format!("Last edit in {part}: {what}{mark}."));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_parse() {
        assert_eq!(parse_views(None).unwrap(), vec![View::Current]);
        assert_eq!(parse_views(Some(&json!("quad"))).unwrap().len(), 4);
        assert_eq!(parse_views(Some(&json!(["front", "3/4", "top"]))).unwrap(), vec![View::Front, View::ThreeQuarter, View::Top]);
        assert!(parse_views(Some(&json!("sideways"))).is_err());
    }

    #[test]
    fn named_views_face_the_right_way() {
        let cam = |v: View| {
            let (yaw, pitch, _) = v.orient().unwrap();
            Camera { target: Vec3::ZERO, yaw: yaw.to_radians(), pitch: pitch.to_radians(), dist: 1.0 }.basis()
        };
        let close = |a: Vec3, b: Vec3| (a - b).length() < 1e-3;
        let (fwd, right, up) = cam(View::Front);
        assert!(close(fwd, Vec3::Y) && close(right, Vec3::X) && close(up, Vec3::Z));
        let (fwd, right, _) = cam(View::Right);
        assert!(close(fwd, -Vec3::X) && close(right, Vec3::Y));
        let (fwd, right, up) = cam(View::Top);
        assert!(close(fwd, -Vec3::Z) && close(right, Vec3::X) && close(up, Vec3::Y));
    }

    #[test]
    fn ortho_framing_puts_the_box_in_the_inner_rect() {
        let (lo, hi) = (Vec3::ZERO, Vec3::new(40.0, 20.0, 30.0));
        let inner = (50.0, 30.0, 400.0, 300.0);
        let (yaw, pitch, _) = View::Front.orient().unwrap();
        let shot = frame_view(yaw, pitch, true, lo, hi, 45.0, 500.0, 360.0, inner);
        let p = |v: Vec3| project_point(&shot.camera, 45.0, shot.projection, 500.0, 360.0, v).unwrap();
        let (a, b) = (p(Vec3::new(0.0, 0.0, 0.0)), p(Vec3::new(40.0, 0.0, 30.0)));
        // x grows to the right, z up; both corners inside the inner rect.
        assert!(b.0 > a.0 && b.1 < a.1);
        for (x, y) in [a, b] {
            assert!(x >= 50.0 && x <= 450.0 && y >= 30.0 && y <= 330.0, "{x} {y}");
        }
        // Centered in the inner rect.
        assert!(((a.0 + b.0) * 0.5 - 250.0).abs() < 0.5 && ((a.1 + b.1) * 0.5 - 180.0).abs() < 0.5);
    }

    #[test]
    fn turnaround_is_four_sides() {
        assert_eq!(parse_views(Some(&json!("turnaround"))).unwrap(), vec![View::Front, View::Right, View::Back, View::Left]);
    }

    #[test]
    fn fitted_cameras_keep_the_box_in_frame_snugly() {
        let (lo, hi) = (Vec3::new(-3.0, 2.0, 0.0), Vec3::new(17.0, 12.0, 40.0));
        let (w, h, margin) = (900.0, 1100.0, 0.06);
        for ortho in [true, false] {
            for (yaw, pitch) in [(225.0f32, 25.0f32), (-90.0, 0.0), (10.0, 60.0)] {
                let cam = fit_box(yaw.to_radians(), pitch.to_radians(), ortho, lo, hi, 45.0, w, h, margin);
                let proj = if ortho { Projection::Orthographic } else { Projection::Perspective };
                let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                for i in 0..8 {
                    let c = Vec3::new(if i & 1 == 0 { lo.x } else { hi.x }, if i & 2 == 0 { lo.y } else { hi.y }, if i & 4 == 0 { lo.z } else { hi.z });
                    let (x, y) = project_point(&cam, 45.0, proj, w, h, c).expect("in front of the camera");
                    (min, max) = (min.min(Vec3::new(x, y, 0.0)), max.max(Vec3::new(x, y, 0.0)));
                }
                let (mx, my) = (w * margin - 0.5, h * margin - 0.5);
                assert!(min.x >= mx && min.y >= my && max.x <= w - mx && max.y <= h - my, "{ortho} {yaw} {pitch}: {min} {max}");
                // Touching the margin on at least one side, not lost in the middle.
                let slack = (min.x - w * margin).min(min.y - h * margin).min(w - w * margin - max.x).min(h - h * margin - max.y);
                assert!(slack < 2.0, "{ortho} {yaw} {pitch}: slack {slack}");
            }
        }
    }

    #[test]
    fn tick_step_widens_when_crowded() {
        assert_eq!(tick_step(8, 40, 10.0, 30.0), 8);
        assert_eq!(tick_step(8, 40, 2.0, 30.0), 16);
        assert_eq!(tick_step(8, 40, 1.0, 30.0), 32);
        // Short axes get finer steps.
        assert_eq!(tick_step(8, 6, 40.0, 30.0), 2);
    }
}
