//! Building: symmetry, more primitives, tubes, extruded polygons, stamps.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use glam::{DVec3, IVec3, Mat3, Vec2, Vec3};
use serde_json::{Value, json};

use super::{Ctx, NEIGHBORS, apply, arr, boolean, color, color_arg, color_schema, count, edited, get, ivec3, num, opt_ivec3, string, text, tool, xyz};
use crate::scene::{Mask, Volume};

#[derive(Default)]
pub struct State {
    /// Twice the symmetry plane's coordinate per axis (so half cells are
    /// whole numbers), by part id. Missing: the part's center.
    centers: HashMap<usize, [Option<i32>; 3]>,
    /// The tool being run, so whole-model ops aren't mirrored.
    tool: Option<String>,
    /// The axes the last committed edit was mirrored across.
    mirrored: [bool; 3],
    /// Mirror copies of that edit that fell off the canvas.
    mirror_lost: usize,
}

impl State {
    /// Forget moved symmetry planes (another scene was loaded; its part ids
    /// mean other parts).
    pub fn scene_changed(&mut self) {
        self.centers.clear();
    }
}

pub const PART_TOOLS: &[&str] = &["draw_tube", "extrude_polygon", "place_stamp"];

/// Edits that move or regenerate the whole model; mirroring their changes
/// would garble them.
const UNMIRRORED: &[&str] = &[
    "edit_model", "transform_model", "resize_canvas", "run_shader", "move_selection",
    // Surface passes: per-voxel noise and images shouldn't be forced symmetric.
    "vary_colors", "shade_edges", "paint_faces", "voxelize_image", "paint_from_reference",
];

pub const SHAPES: &[&str] =
    &["box", "ellipsoid", "sphere", "cylinder", "cone", "pyramid", "frustum", "torus", "capsule", "rounded_box", "wedge"];

pub fn tools() -> Vec<Value> {
    let axis = json!({ "type": "string", "enum": ["x", "y", "z"] });
    let uv = json!({ "type": "array", "items": { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2 } });
    vec![
        tool("set_symmetry", "Mirror every voxel edit across the part's center plane(s), like a symmetric sculpting mode, and report the current state. While on, set_voxels, fill_box, draw_shape, draw_line, draw_tube, extrude_polygon, place_stamp and run_script edits are mirrored (each still one undo step); edit_model, transform_model, run_shader and move_selection aren't. The axes are one setting shared with the window's Mirror X/Y/Z brush toggles: they stay on when you switch parts, and each part mirrors about its own center unless `center` moved its plane. Call with no arguments to see the state.", json!({
            "x": { "type": "boolean" }, "y": { "type": "boolean" }, "z": { "type": "boolean" },
            "axes": { "type": "array", "items": axis.clone(), "description": "The axes to mirror across, replacing x/y/z; [] turns symmetry off" },
            "center": { "type": ["array", "string"], "items": { "type": ["number", "null"] }, "description": "[x, y, z] plane coordinates in the edited part's cells (whole or .5; null keeps that axis centered), or \"reset\". Default: the part's center ((size-1)/2)" },
        }), &[]),
        tool("draw_tube", "Draw a round tube along a smooth curve through points: tails, tentacles, horns, cables, branches, handles. The radius can taper from start to end.", json!({
            "points": { "type": "array", "items": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 }, "minItems": 2, "description": "[[x, y, z], ...] cells the curve goes through (or its control points for bezier), at least 2" },
            "radius": { "type": "number", "description": "Radius in voxels at the start (default 1; 0.5 is one voxel thick)" },
            "end_radius": { "type": "number", "description": "Radius at the end (default: radius), e.g. 0.5 for a pointed tail or horn" },
            "curve": { "type": "string", "enum": ["smooth", "bezier", "straight"], "description": "smooth (default): Catmull-Rom through every point; bezier: points are control points (passes through the first and last); straight: a polyline" },
            "closed": { "type": "boolean", "description": "Join the last point back to the first (a ring)" },
            "color": color_schema(),
        }), &["points", "color"]),
        tool("extrude_polygon", "Fill a 2D polygon and extrude it through a range of layers: signs, letters, gear and cog outlines, roof and arch profiles, floor plans. Points are cells on the outline (outline cells are included); the polygon may be concave.", json!({
            "points": { "type": "array", "items": { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2 }, "minItems": 2, "description": "[[u, v], ...] outline cells. axis z: u = x, v = y; axis y: u = x, v = z; axis x: u = y, v = z (as in get_slice)" },
            "axis": { "type": "string", "enum": ["x", "y", "z"], "description": "The direction it's extruded along (default z: the polygon lies flat and extrudes upward)" },
            "depth": { "type": ["array", "integer"], "items": { "type": "integer" }, "minItems": 2, "maxItems": 2, "description": "[from, to] layers along axis, inclusive, or one layer" },
            "holes": { "type": "array", "items": uv, "description": "Polygons to leave empty, each [[u, v], ...] (their outline cells are empty too)" },
            "hollow": { "type": "boolean", "description": "Only the outer shell" },
            "color": color_schema(),
        }), &["points", "depth", "color"]),
        tool("save_stamp", "Save voxels as a named reusable stamp (kept in the workspace's stamps folder across sessions): a box of the part being edited, or a whole part cropped to its voxels. Place copies with place_stamp.", json!({
            "name": { "type": "string", "description": "Letters, digits, spaces, - and _; an existing stamp of that name is replaced" },
            "from": xyz("Box corner (default: the part's whole voxel bounds)"), "to": xyz("Opposite corner, inclusive"),
            "part": { "type": ["string", "integer"], "description": "Part to copy from without switching to it (default: the part being edited)" },
        }), &["name"]),
        tool("list_stamps", "Saved stamps with their sizes and voxel counts.", json!({}), &[]),
        tool("delete_stamp", "Delete a saved stamp.", json!({ "name": { "type": "string" } }), &["name"]),
        tool("place_stamp", "Place a saved stamp, or any .vox file (like the window's Pattern brush), into the part being edited. Only its solid cells are written; empty cells leave what's there. Colors map by RGB to the current palette. A file's visible models are flattened into one stamp, cropped to its voxels; import_file brings a .vox in as separate parts instead.", json!({
            "name": { "type": "string", "description": "A saved stamp (see list_stamps)" },
            "path": { "type": "string", "description": "Or a .vox file to stamp, e.g. another project's" },
            "at": xyz("Where the stamp's anchor goes"),
            "anchor": { "type": "string", "enum": ["min", "center", "bottom"], "description": "min (default): its minimum corner; center; bottom: the middle of its bottom (low z) face" },
            "rotate": { "type": ["integer", "array"], "items": { "type": "integer" }, "description": "Counter-clockwise quarter turns: n about z, or [x, y, z] turns applied x then y then z" },
            "flip": { "type": ["string", "array"], "items": axis.clone(), "description": "Axis or axes to mirror the stamp along (before rotating)" },
            "replace": { "type": "object", "description": "Recolor: {\"<stamp color index>\": color, ...}" },
            "color": color_schema(),
        }), &["at"]),
    ]
}

pub fn dispatch(ctx: &mut Ctx, name: &str, args: &Value) -> Option<Result<Vec<Value>>> {
    ctx.ed.tools.build.tool = Some(name.to_string());
    let a = args;
    Some(match name {
        "set_symmetry" => set_symmetry(ctx, a),
        "draw_tube" => (|| {
            let pts = points3(a)?;
            let c = color_arg(ctx.scene, a, "color")?;
            let r0 = num(a, "radius").unwrap_or(1.0);
            let r1 = num(a, "end_radius").unwrap_or(r0);
            for (key, r) in [("radius", r0), ("end_radius", r1)] {
                if !(0.5..=256.0).contains(&r) {
                    bail!("`{key}` must be 0.5-256 voxels (0.5 is one voxel thick), not {r}");
                }
            }
            let curve = string(a, "curve").unwrap_or("smooth");
            let path = sample_curve(&pts, curve, boolean(a, "closed").unwrap_or(false))?;
            let len = path.windows(2).map(|w| w[0].distance(w[1])).sum::<f32>();
            let new = tube_volume(&ctx.scene.volume, &path, r0, r1, c);
            let what = format!("Drew tube through {} points, {len:.0} voxels long", pts.len());
            let changed = apply(ctx, new);
            Ok(edited(ctx, changed, &what))
        })(),
        "extrude_polygon" => (|| {
            let ax = super::axis(a)?;
            let outline = points2(get(a, "points").context("missing `points`")?)?;
            if outline.len() < 2 {
                bail!("`points` needs at least 2 [u, v] points");
            }
            let holes = match get(a, "holes").and_then(Value::as_array) {
                Some(h) => h.iter().map(points2).collect::<Result<Vec<_>>>()?,
                None => Vec::new(),
            };
            let (d0, d1) = match get(a, "depth").context("missing `depth`")? {
                Value::Array(d) if d.len() == 2 => {
                    let n = |v: &Value| v.as_f64().map(|f| f.round() as i32).context("`depth` holds integers");
                    (n(&d[0])?, n(&d[1])?)
                }
                v => v.as_f64().map(|f| (f.round() as i32, f.round() as i32)).context("`depth` is [from, to] or one layer")?,
            };
            let c = color_arg(ctx.scene, a, "color")?;
            let new = extrude_volume(&ctx.scene.volume, ax, &outline, &holes, d0.min(d1), d0.max(d1), c, boolean(a, "hollow").unwrap_or(false));
            let what = format!("Extruded polygon of {} points", outline.len());
            let changed = apply(ctx, new);
            Ok(edited(ctx, changed, &what))
        })(),
        "save_stamp" => save_stamp(ctx, a),
        "list_stamps" => list_stamps(),
        "delete_stamp" => (|| {
            let name = stamp_name(a)?;
            std::fs::remove_file(stamp_path(&name)).map_err(|_| anyhow!("no stamp named {name:?}; see list_stamps"))?;
            Ok(vec![text(format!("Deleted stamp {name:?}"))])
        })(),
        "place_stamp" => place_stamp(ctx, a),
        _ => return None,
    })
}

// ---------------------------------------------------------------- symmetry

/// Called on every edit before it's committed: mirrors the edit's changed
/// cells while symmetry is on.
pub fn before_commit(ctx: &mut Ctx, new: Volume) -> Volume {
    let tool = ctx.ed.tools.build.tool.take();
    let axes = ctx.ed.editor.mirror;
    ctx.ed.tools.build.mirrored = [false; 3];
    let Some(id) = ctx.scene.editing() else { return new };
    if !axes.iter().any(|&on| on) || tool.as_deref().is_some_and(|t| UNMIRRORED.contains(&t)) {
        return new;
    }
    ctx.ed.tools.build.mirrored = axes;
    let planes = planes(ctx, id);
    let (out, lost) = mirror_edit(&ctx.scene.volume, new, axes, planes);
    ctx.ed.tools.build.mirror_lost = lost;
    out
}

/// Twice each axis's mirror plane coordinate for part `id`.
fn planes(ctx: &Ctx, id: usize) -> IVec3 {
    let dims = ctx.scene.volume.dims;
    let set = ctx.ed.tools.build.centers.get(&id).copied().unwrap_or_default();
    IVec3::from_array(std::array::from_fn(|k| set[k].unwrap_or(dims[k] - 1)))
}

/// Copy every changed cell of `new` (vs `old`) to its mirror images across
/// the planes (given doubled: cell p mirrors to `planes - p`) on `axes`.
/// When several cells of one mirror set changed differently, the highest
/// color wins, so adding beats erasing.
pub fn mirror_edit(old: &Volume, new: Volume, axes: [bool; 3], planes: IVec3) -> (Volume, usize) {
    if old.dims != new.dims {
        return (new, 0);
    }
    let mut lost = 0;
    let mut out = new.clone();
    let mut changed: Vec<usize> = old.cells.diff(&new.cells).into_iter().map(|c| c.0).collect();
    changed.sort_unstable();
    let mut done = Mask::new(new.dims);
    let mut orbit = Vec::with_capacity(8);
    for i in changed {
        if done[i] {
            continue;
        }
        orbit.clear();
        orbit.push(new.position(i));
        for k in 0..3 {
            if axes[k] {
                for j in 0..orbit.len() {
                    let mut m = orbit[j];
                    m[k] = planes[k] - m[k];
                    orbit.push(m);
                }
            }
        }
        let n = orbit.len();
        orbit.retain(|&p| new.contains(p));
        lost += n - orbit.len();
        let idx: Vec<usize> = orbit.iter().map(|&p| new.index(p)).collect();
        let c = idx.iter().filter(|&&j| old.cells[j] != new.cells[j]).map(|&j| new.cells[j]).max().unwrap_or(0);
        for j in idx {
            done.set(j, true);
            out.set_index(j, c);
        }
    }
    (out, lost)
}

/// " (mirrored xz)" when the edit just made was mirrored, else "".
pub(super) fn mirrored_note(ctx: &Ctx) -> String {
    let m = ctx.ed.tools.build.mirrored;
    if !m.iter().any(|&on| on) {
        return String::new();
    }
    let axes: String = ["x", "y", "z"].iter().zip(m).filter(|(_, on)| *on).map(|(n, _)| *n).collect();
    match ctx.ed.tools.build.mirror_lost {
        0 => format!(" (mirrored {axes})"),
        1 => format!(" (mirrored {axes}; 1 mirror copy fell outside the canvas, so that edit isn't symmetric there)"),
        n => format!(" (mirrored {axes}; {n} mirror copies fell outside the canvas, so that edit isn't symmetric there)"),
    }
}

fn set_symmetry(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let names = ["x", "y", "z"];
    if let Some(list) = get(a, "axes") {
        let list = list.as_array().context("`axes` is a list like [\"x\"]")?;
        let mut m = [false; 3];
        for v in list {
            let k = v.as_str().and_then(|s| names.iter().position(|n| n.eq_ignore_ascii_case(s))).with_context(|| format!("unknown axis {v}"))?;
            m[k] = true;
        }
        ctx.ed.editor.mirror = m;
    }
    for (k, n) in names.iter().enumerate() {
        if let Some(on) = boolean(a, n) {
            ctx.ed.editor.mirror[k] = on;
        }
    }
    let id = ctx.scene.editing();
    match get(a, "center") {
        Some(Value::String(s)) if s == "reset" || s == "center" => {
            if let Some(id) = id {
                ctx.ed.tools.build.centers.remove(&id);
            }
        }
        Some(Value::Array(c)) if c.len() == 3 => {
            let id = id.context("`center` applies to the part being edited; call edit_part first")?;
            let mut set = [None; 3];
            for k in 0..3 {
                if let Some(f) = c[k].as_f64() {
                    set[k] = Some((f * 2.0).round() as i32);
                } else if !c[k].is_null() {
                    bail!("`center` is [x, y, z] numbers (null keeps an axis centered)");
                }
            }
            ctx.ed.tools.build.centers.insert(id, set);
        }
        Some(_) => bail!("`center` is [x, y, z] or \"reset\""),
        None => {}
    }
    let m = ctx.ed.editor.mirror;
    if !m.iter().any(|&on| on) {
        return Ok(vec![text("Symmetry off.")]);
    }
    let planes: Vec<String> = match id {
        Some(id) => {
            let p = planes(ctx, id);
            (0..3).filter(|&k| m[k]).map(|k| format!("{} = {}", names[k], p[k] as f32 / 2.0)).collect()
        }
        None => (0..3).filter(|&k| m[k]).map(|k| names[k].to_string()).collect(),
    };
    let part = id.map_or_else(|| "each part (World mode now)".to_string(), |id| format!("part `{}`", ctx.scene.world.label(id)));
    Ok(vec![text(format!(
        "Symmetry on: edits are mirrored across {} in {part}. It stays on for other parts (about their own centers) until turned off; the window's Mirror toggles show it too.",
        planes.join(" and ")
    ))])
}

// ---------------------------------------------------------------- solids

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Box,
    Ellipsoid,
    /// Round cross-section whose radius scales to `top` at the +axis end:
    /// cylinder, cone, frustum.
    Round,
    /// Square cross-section, likewise: pyramid, truncated pyramid.
    Square,
    Torus,
    Capsule,
    RoundedBox,
    Wedge,
}

/// A primitive inscribed in a box of cells, optionally rotated about the
/// box center.
pub struct Solid {
    kind: Kind,
    axis: usize,
    /// Wedge: the axis it's tall along.
    up: usize,
    center: Vec3,
    half: Vec3,
    /// Torus tube radius or rounded box corner radius, in voxels.
    radius: f32,
    /// Cross-section scale at the +axis end (1 = straight).
    top: f32,
    /// World to box rotation.
    inv: Option<Mat3>,
    /// Cells that can be inside, inclusive.
    lo: IVec3,
    hi: IVec3,
}

impl Solid {
    /// One of `SHAPES` in the box between corners `a` and `b`, with no
    /// options.
    pub fn new(shape: &str, axis: usize, a: IVec3, b: IVec3) -> Self {
        Self::build(shape, axis, a, b, &Value::Null).expect("a known shape")
    }

    /// A shape from draw_shape's arguments (`top_scale`, `radius`, `up`,
    /// `rotation`; `axis` defaults per shape).
    pub fn build(shape: &str, axis: usize, a: IVec3, b: IVec3, args: &Value) -> Result<Self> {
        let (kind, top) = match shape {
            "box" => (Kind::Box, 1.0),
            "ellipsoid" | "sphere" => (Kind::Ellipsoid, 1.0),
            "cylinder" => (Kind::Round, 1.0),
            "cone" => (Kind::Round, 0.0),
            "frustum" => (Kind::Round, 0.5),
            "pyramid" => (Kind::Square, 0.0),
            "torus" => (Kind::Torus, 1.0),
            "capsule" => (Kind::Capsule, 1.0),
            "rounded_box" => (Kind::RoundedBox, 1.0),
            "wedge" => (Kind::Wedge, 1.0),
            _ => bail!("unknown shape {shape:?}; one of {}", SHAPES.join(", ")),
        };
        let top = num(args, "top_scale").unwrap_or(top);
        if top < 0.0 {
            bail!("`top_scale` must be 0 or more, not {top}");
        }
        let up = super::axis_named(args, "up")?;
        // A wedge rises toward +axis; with no axis given, along x (or y if up is x).
        let axis = if kind == Kind::Wedge && get(args, "axis").is_none() { if up == 0 { 1 } else { 0 } } else { axis };
        if kind == Kind::Wedge && axis == up {
            bail!("a wedge's `axis` (the way it rises) must differ from `up`");
        }
        let (lo, hi) = (a.min(b), a.max(b));
        let center = (lo.as_vec3() + hi.as_vec3() + 1.0) * 0.5;
        let half = (hi - lo + IVec3::ONE).as_vec3() * 0.5;
        let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
        // An explicit radius must fit the box; the defaults are fitted.
        let (min, max, default) = match kind {
            Kind::Torus => (0.5, (half[u] + half[w]) * 0.5, half[axis]),
            Kind::RoundedBox => (0.0, half.min_element(), (half.min_element() * 0.5).max(1.0)),
            _ if get(args, "radius").is_some() => bail!("`radius` is for rounded_box (corner radius) and torus (tube radius) only"),
            _ => (0.0, 0.0, 0.0),
        };
        let radius = match num(args, "radius") {
            Some(r) if !(min..=max).contains(&r) => bail!("`radius` must be {min}-{max} for a {shape} this size, not {r}"),
            Some(r) => r,
            None => default.clamp(min, max),
        };
        let mut s = Solid { kind, axis, up, center, half, radius, top, inv: None, lo, hi };
        if let Some(r) = get(args, "rotation") {
            let d = super::vec3_value(r).context("`rotation` is [x, y, z] degrees")?;
            if d != Vec3::ZERO {
                let m = Mat3::from_rotation_z(d.z.to_radians()) * Mat3::from_rotation_y(d.y.to_radians()) * Mat3::from_rotation_x(d.x.to_radians());
                s.inv = Some(m.transpose());
                let ext = m.abs() * half;
                s.lo = (center - ext).floor().as_ivec3();
                s.hi = (center + ext).ceil().as_ivec3() - 1;
            }
        }
        Ok(s)
    }

    /// The cells that can be inside, inclusive.
    pub fn bounds(&self) -> (IVec3, IVec3) {
        (self.lo, self.hi)
    }

    /// Whether the cell `p` (its center) is inside.
    pub fn has_cell(&self, p: IVec3) -> bool {
        if p.cmplt(self.lo).any() || p.cmpgt(self.hi).any() {
            return false;
        }
        let mut d = p.as_vec3() + 0.5 - self.center;
        if let Some(m) = self.inv {
            d = m * d;
        }
        let h = self.half;
        if d.abs().cmpgt(h + 1e-4).any() {
            return false;
        }
        let q = d / h;
        let (a, u, w) = (self.axis, (self.axis + 1) % 3, (self.axis + 2) % 3);
        // 0 at the base (low end of the axis), 1 at the tip.
        let t = (q[a] + 1.0) * 0.5;
        let scale = 1.0 + (self.top - 1.0) * t;
        match self.kind {
            Kind::Box => true,
            Kind::Ellipsoid => q.length_squared() <= 1.0,
            Kind::Round => (q[u] * q[u] + q[w] * q[w]).sqrt() <= scale,
            Kind::Square => q[u].abs().max(q[w].abs()) <= scale,
            Kind::Torus => {
                let ring = (h[u] + h[w]) * 0.5;
                let rho = (q[u] * q[u] + q[w] * q[w]).sqrt() * ring;
                let r = self.radius;
                let (e, f) = ((rho - (ring - r)) / r, q[a]);
                e * e + f * f <= 1.0
            }
            Kind::Capsule => {
                let cap = ((h[u] + h[w]) * 0.5).min(h[a]);
                let s = h[a] - cap;
                let e = (d[a] - d[a].clamp(-s, s)) / cap;
                q[u] * q[u] + q[w] * q[w] + e * e <= 1.0
            }
            Kind::RoundedBox => {
                let r = self.radius;
                let k = d.abs() - (h - r);
                k.max(Vec3::ZERO).length() + k.max_element().min(0.0) <= r + 1e-4
            }
            Kind::Wedge => t >= (q[self.up] + 1.0) * 0.5 - 1e-4,
        }
    }
}

/// `v` with every cell between `lo` and `hi` where `inside` holds set to
/// `c` (only the shell if `hollow`).
fn fill_where(v: &Volume, lo: IVec3, hi: IVec3, c: u8, hollow: bool, inside: impl Fn(IVec3) -> bool) -> Volume {
    let mut out = v.clone();
    let (lo, hi) = (lo.max(IVec3::ZERO), hi.min(v.dims - 1));
    for z in lo.z..=hi.z {
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let p = IVec3::new(x, y, z);
                if inside(p) && (!hollow || NEIGHBORS.iter().any(|&n| !inside(p + n))) {
                    out.set(p, c);
                }
            }
        }
    }
    out
}

/// Fill `s` with `c`.
pub fn fill_solid(v: &Volume, s: &Solid, c: u8, hollow: bool) -> Volume {
    fill_where(v, s.lo, s.hi, c, hollow, |p| s.has_cell(p))
}

// ---------------------------------------------------------------- tubes

/// draw_line's brush: a ball `thick` voxels across, centered on a cell
/// when odd and between cells (toward +x+y+z) when even.
pub(super) fn line_brush(thick: i32) -> Vec<IVec3> {
    let lo = -(thick - 1) / 2;
    let center = Vec3::splat(if thick % 2 == 0 { 0.5 } else { 0.0 });
    let r = thick as f32 * 0.5 + 1e-3;
    let mut out = Vec::new();
    for z in lo..lo + thick {
        for y in lo..lo + thick {
            for x in lo..lo + thick {
                let o = IVec3::new(x, y, z);
                if (o.as_vec3() - center).length() <= r {
                    out.push(o);
                }
            }
        }
    }
    out
}

/// The part of the segment `a`..`b` inside the box `lo`..`hi`, or None.
fn clip_segment(a: DVec3, b: DVec3, lo: DVec3, hi: DVec3) -> Option<(DVec3, DVec3)> {
    let d = b - a;
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for k in 0..3 {
        if d[k] == 0.0 {
            if a[k] < lo[k] || a[k] > hi[k] {
                return None;
            }
            continue;
        }
        let (mut u, mut v) = ((lo[k] - a[k]) / d[k], (hi[k] - a[k]) / d[k]);
        if u > v {
            std::mem::swap(&mut u, &mut v);
        }
        (t0, t1) = (t0.max(u), t1.min(v));
        if t0 > t1 {
            return None;
        }
    }
    Some((a + d * t0, a + d * t1))
}

/// The cells draw_line paints from `from` to `to`: `brush` stepped one
/// cell at a time (no gaps) along the line, first clipped to the box
/// `lo`..=`hi` (plus the brush's reach). Cells may repeat.
pub(super) fn line_cells(lo: IVec3, hi: IVec3, from: IVec3, to: IVec3, brush: &[IVec3], mut f: impl FnMut(IVec3)) {
    let reach = brush.iter().map(|o| o.abs().max_element()).max().unwrap_or(0) as f64;
    let (lo, hi) = (lo.as_dvec3() - reach, hi.as_dvec3() + reach);
    let Some((a, b)) = clip_segment(from.as_dvec3(), to.as_dvec3(), lo, hi) else { return };
    let d = b - a;
    let steps = d.abs().max_element().ceil().max(1.0) as i64;
    for i in 0..=steps {
        let p = (a + d * (i as f64 / steps as f64)).round().as_ivec3();
        brush.iter().for_each(|&o| f(p + o));
    }
}

fn points3(a: &Value) -> Result<Vec<Vec3>> {
    let list = get(a, "points").and_then(Value::as_array).context("`points` must be [[x, y, z], ...]")?;
    let pts = list.iter().map(|p| super::vec3_value(p).context("each point is [x, y, z]")).collect::<Result<Vec<_>>>()?;
    if pts.len() < 2 {
        bail!("`points` needs at least 2 points");
    }
    Ok(pts)
}

/// A dense polyline (in cell coordinates) along the curve through `pts`.
fn sample_curve(pts: &[Vec3], curve: &str, closed: bool) -> Result<Vec<Vec3>> {
    let mut pts = pts.to_vec();
    if closed && pts.first() != pts.last() {
        pts.push(pts[0]);
    }
    let per = |len: f32| ((len * 3.0).ceil() as usize).clamp(1, 4096);
    let mut out = vec![pts[0]];
    match curve {
        "straight" => out = pts,
        "smooth" => {
            let n = pts.len();
            let at = |i: isize| {
                if closed {
                    pts[(i.rem_euclid(n as isize - 1)) as usize]
                } else {
                    pts[i.clamp(0, n as isize - 1) as usize]
                }
            };
            for i in 0..n - 1 {
                let (p0, p1, p2, p3) = (at(i as isize - 1), at(i as isize), at(i as isize + 1), at(i as isize + 2));
                let steps = per(p1.distance(p2));
                for s in 1..=steps {
                    let t = s as f32 / steps as f32;
                    let (t2, t3) = (t * t, t * t * t);
                    out.push(0.5 * (2.0 * p1 + (p2 - p0) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3));
                }
            }
        }
        "bezier" => {
            let steps = per(pts.windows(2).map(|w| w[0].distance(w[1])).sum());
            for s in 1..=steps {
                let t = s as f32 / steps as f32;
                let mut q = pts.clone();
                while q.len() > 1 {
                    q = q.windows(2).map(|w| w[0].lerp(w[1], t)).collect();
                }
                out.push(q[0]);
            }
        }
        c => bail!("unknown curve {c:?}; smooth, bezier or straight"),
    }
    Ok(out)
}

/// A tube along `path` whose radius goes from `r0` to `r1` by length.
fn tube_volume(v: &Volume, path: &[Vec3], r0: f32, r1: f32, c: u8) -> Volume {
    let (r0, r1) = (r0.clamp(0.5, 256.0), r1.clamp(0.5, 256.0));
    let total = path.windows(2).map(|w| w[0].distance(w[1])).sum::<f32>().max(1e-6);
    let mut out = v.clone();
    let mut along = 0.0;
    let path: Vec<Vec3> = path.iter().map(|p| *p + 0.5).collect();
    let single = [path[0], path[0]];
    let segs: Vec<&[Vec3]> = if path.len() == 1 { vec![&single[..]] } else { path.windows(2).collect() };
    for w in segs {
        let (a, b) = (w[0], w[1]);
        let len = a.distance(b);
        let (ra, rb) = (r0 + (r1 - r0) * along / total, r0 + (r1 - r0) * (along + len) / total);
        along += len;
        let r = ra.max(rb);
        let lo = (a.min(b) - r).floor().as_ivec3().max(IVec3::ZERO);
        let hi = (a.max(b) + r).ceil().as_ivec3().min(v.dims - 1);
        let ab = b - a;
        let l2 = ab.length_squared();
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let p = IVec3::new(x, y, z);
                    let q = p.as_vec3() + 0.5;
                    let t = if l2 > 0.0 { ((q - a).dot(ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
                    if q.distance(a + ab * t) <= ra + (rb - ra) * t + 1e-3 {
                        out.set(p, c);
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------- polygons

fn points2(v: &Value) -> Result<Vec<Vec2>> {
    let list = v.as_array().context("a polygon is [[u, v], ...]")?;
    list.iter()
        .map(|p| {
            let a = p.as_array().filter(|a| a.len() == 2).context("each polygon point is [u, v]")?;
            let n = |i: usize| a[i].as_f64().map(|f| f as f32).context("polygon points hold numbers");
            Ok(Vec2::new(n(0)?, n(1)?))
        })
        .collect()
}

/// Whether `p` is inside polygon `poly` (even-odd) or within half a cell of
/// its outline.
fn in_polygon(poly: &[Vec2], p: Vec2) -> bool {
    let n = poly.len();
    let mut inside = false;
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        let ab = b - a;
        let t = if ab.length_squared() > 0.0 { ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
        if p.distance(a + ab * t) <= 0.5 + 1e-4 {
            return true;
        }
        if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x) {
            inside = !inside;
        }
    }
    inside
}

/// The in-plane axes (u, v) for polygons extruded along `axis`.
fn plane_axes(axis: usize) -> (usize, usize) {
    match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

#[allow(clippy::too_many_arguments)]
fn extrude_volume(v: &Volume, axis: usize, outline: &[Vec2], holes: &[Vec<Vec2>], d0: i32, d1: i32, c: u8, hollow: bool) -> Volume {
    let (u, w) = plane_axes(axis);
    let (mut lo, mut hi) = (IVec3::ZERO, v.dims - 1);
    let min = outline.iter().fold(Vec2::splat(f32::MAX), |m, p| m.min(*p)).floor().as_ivec2();
    let max = outline.iter().fold(Vec2::splat(f32::MIN), |m, p| m.max(*p)).ceil().as_ivec2();
    (lo[u], lo[w], lo[axis]) = (min.x.max(0), min.y.max(0), d0.max(0));
    (hi[u], hi[w], hi[axis]) = (max.x.min(v.dims[u] - 1), max.y.min(v.dims[w] - 1), d1.min(v.dims[axis] - 1));
    // The 2D mask once, then extrude.
    let (mw, mh) = ((hi[u] - lo[u] + 1).max(0), (hi[w] - lo[w] + 1).max(0));
    let mut mask = vec![false; (mw * mh) as usize];
    for j in 0..mh {
        for i in 0..mw {
            let p = Vec2::new((lo[u] + i) as f32, (lo[w] + j) as f32);
            mask[(i + j * mw) as usize] = in_polygon(outline, p) && !holes.iter().any(|h| h.len() >= 2 && in_polygon(h, p));
        }
    }
    let inside = |p: IVec3| {
        let (i, j) = (p[u] - lo[u], p[w] - lo[w]);
        (d0..=d1).contains(&p[axis]) && (0..mw).contains(&i) && (0..mh).contains(&j) && mask[(i + j * mw) as usize]
    };
    fill_where(v, lo, hi, c, hollow, inside)
}

// ---------------------------------------------------------------- stamps

fn stamp_dir() -> PathBuf {
    crate::app::workspace_dir().join("stamps")
}

fn stamp_path(name: &str) -> PathBuf {
    stamp_dir().join(format!("{name}.vox"))
}

fn stamp_name(a: &Value) -> Result<String> {
    let name = string(a, "name").context("missing `name`")?.trim();
    if name.is_empty() || name.len() > 80 || !name.chars().all(|c| c.is_alphanumeric() || " -_".contains(c)) {
        bail!("stamp names use letters, digits, spaces, - and _");
    }
    Ok(name.to_string())
}

/// Write `v` with its palette as a single-model .vox.
pub fn write_stamp(path: &std::path::Path, v: &Volume, palette: &[[u8; 4]]) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    crate::vox::write(path, &crate::world::World::single(v.clone()), palette, &[], &[], &[])
}

/// A stamp's voxels and the palette they were saved with.
pub fn read_stamp(path: &std::path::Path) -> Result<(Volume, Vec<[u8; 4]>)> {
    let f = crate::vox::VoxFile::load(path)?;
    let w = crate::world::World::from_vox(&f);
    // Any .vox can be a stamp: its visible models flattened into one,
    // cropped to its voxels.
    let c = w.composite(None).context("the file has no visible model")?;
    let v = crate::ops::fit(&c.volume).context("the file has no voxels")?;
    Ok((v, f.palette))
}

/// The cells of `v` between `lo` and `hi` (clamped), as their own volume.
fn crop(v: &Volume, lo: IVec3, hi: IVec3) -> Volume {
    let (lo, hi) = (lo.max(IVec3::ZERO), hi.min(v.dims - 1));
    let mut out = Volume::new((hi - lo + 1).max(IVec3::ONE));
    for (p, c) in v.filled() {
        if p.cmpge(lo).all() && p.cmple(hi).all() {
            out.set(p - lo, c);
        }
    }
    out
}

fn save_stamp(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let name = stamp_name(a)?;
    let (src, label) = match get(a, "part") {
        Some(p) => {
            let id = super::find_part(ctx.scene, Some(p))?;
            let w = ctx.scene.synced_world();
            let (v, _) = w.checkout(id).with_context(|| format!("`{}` is a group; pass one of its parts", w.label(id)))?;
            (v, w.label(id))
        }
        None => (ctx.scene.volume.clone(), ctx.scene.editing().map_or_else(|| "the world".into(), |id| ctx.scene.world.label(id))),
    };
    let (lo, hi) = match (opt_ivec3(a, "from")?, opt_ivec3(a, "to")?) {
        (Some(f), Some(t)) => (f.min(t), f.max(t)),
        (None, None) => super::bounds(&src).with_context(|| format!("`{label}` has no voxels"))?,
        _ => bail!("give both `from` and `to`, or neither"),
    };
    // The box on the canvas: placing the stamp at its min corner puts the
    // voxels back where they were.
    let (lo, hi) = (lo.max(IVec3::ZERO), hi.min(src.dims - 1));
    if lo.cmpgt(hi).any() {
        bail!("the box lies outside `{label}`'s canvas; coordinates run [0, 0, 0]..{:?}", arr(src.dims - 1));
    }
    let v = crop(&src, lo, hi);
    let n = count(&v);
    if n == 0 {
        bail!("no voxels between {:?} and {:?}", arr(lo), arr(hi));
    }
    let path = stamp_path(&name);
    let existed = path.exists();
    write_stamp(&path, &v, &ctx.scene.palette)?;
    let d = v.dims;
    Ok(vec![text(format!(
        "{} stamp {name:?}: {n} voxels, {}x{}x{}, from {:?}..{:?} of `{label}` (place_stamp at {:?} puts it back). Saved at {}",
        if existed { "Replaced" } else { "Saved" },
        d.x, d.y, d.z, arr(lo), arr(hi), arr(lo), path.display()
    ))])
}

fn list_stamps() -> Result<Vec<Value>> {
    let list: Vec<Value> = crate::app::list_vox(&stamp_dir())
        .iter()
        .map(|p| {
            let name = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            match read_stamp(p) {
                Ok((v, _)) => json!({ "name": name, "size": arr(v.dims), "voxels": count(&v) }),
                Err(e) => json!({ "name": name, "error": format!("{e:#}") }),
            }
        })
        .collect();
    Ok(vec![text(serde_json::to_string_pretty(&json!({ "folder": stamp_dir().display().to_string(), "stamps": list }))?)])
}

/// A list of axes: "x" or ["x", "z"].
fn axes_arg(v: Option<&Value>) -> Result<Vec<usize>> {
    let one = |v: &Value| match v.as_str() {
        Some("x" | "X") => Ok(0),
        Some("y" | "Y") => Ok(1),
        Some("z" | "Z") => Ok(2),
        _ => bail!("unknown axis {v}"),
    };
    match v {
        None => Ok(Vec::new()),
        Some(Value::Array(a)) => a.iter().map(one).collect(),
        Some(v) => Ok(vec![one(v)?]),
    }
}

/// The stamp's turns as one rotation: n about z, or [x, y, z] turns.
fn turns_arg(v: Option<&Value>) -> Result<crate::world::Rot> {
    use crate::world::Rot;
    let q = match v {
        None => [0, 0, 0],
        Some(Value::Array(a)) if a.len() == 3 => {
            let n = |i: usize| a[i].as_i64().map(|n| n as i32).context("`rotate` holds integers");
            [n(0)?, n(1)?, n(2)?]
        }
        Some(v) => [0, 0, v.as_i64().context("`rotate` is quarter turns: n or [x, y, z]")? as i32],
    };
    Ok(Rot::turn(2, q[2]).mul(&Rot::turn(1, q[1])).mul(&Rot::turn(0, q[0])))
}

/// Stamp cells after flips and turns, from the transformed box's minimum
/// corner, with that box's size.
pub fn orient(v: &Volume, flips: &[usize], rot: crate::world::Rot) -> (Vec<(IVec3, u8)>, IVec3) {
    let d = v.dims;
    let size = rot.extent(d);
    let corners = [rot.apply(IVec3::ZERO), rot.apply(d - 1)];
    let lo = corners[0].min(corners[1]);
    let cells = v
        .cells
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != 0)
        .map(|(i, &c)| {
            let mut p = v.position(i);
            for &k in flips {
                p[k] = d[k] - 1 - p[k];
            }
            (rot.apply(p) - lo, c)
        })
        .collect();
    (cells, size)
}

fn place_stamp(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let (name, path) = match string(a, "path") {
        Some(p) => {
            let path = PathBuf::from(p);
            if !path.is_file() {
                bail!("no file at {}", path.display());
            }
            (path.file_stem().map_or(String::new(), |n| n.to_string_lossy().into_owned()), path)
        }
        None if get(a, "name").is_none() => bail!("give a stamp `name` or a .vox `path`"),
        None => {
            let name = stamp_name(a)?;
            let path = stamp_path(&name);
            if !path.exists() {
                bail!("no stamp named {name:?}; see list_stamps");
            }
            (name, path)
        }
    };
    let (v, pal) = read_stamp(&path)?;
    let at = ivec3(a, "at")?;
    let flips = axes_arg(get(a, "flip"))?;
    let rot = turns_arg(get(a, "rotate"))?;
    // Saved colors to current palette indices: the same index if it still
    // holds the same color, else the nearest by RGB.
    let mut map: [u8; 256] = std::array::from_fn(|i| i as u8);
    for i in 1..256 {
        if pal.get(i).is_some_and(|c| c[..3] != ctx.scene.palette[i][..3]) {
            map[i] = color(ctx.scene, &json!(super::hex(pal[i])))?;
        }
    }
    if let Some(r) = get(a, "replace") {
        let r = r.as_object().context("`replace` is {\"<stamp color index>\": color}")?;
        for (k, c) in r {
            let i: u8 = k.parse().map_err(|_| anyhow!("`replace` keys are stamp color indices 1-255, not {k:?}"))?;
            map[i as usize] = color(ctx.scene, c)?;
        }
    }
    if let Some(c) = get(a, "color") {
        let c = color(ctx.scene, c)?;
        map[1..].fill(c);
    }
    let (cells, size) = orient(&v, &flips, rot);
    let lo = at
        - match string(a, "anchor").unwrap_or("min") {
            "min" => IVec3::ZERO,
            "center" => size / 2,
            "bottom" => IVec3::new(size.x / 2, size.y / 2, 0),
            other => bail!("unknown anchor {other:?}; min, center or bottom"),
        };
    let mut new = ctx.scene.volume.clone();
    let mut skipped = 0;
    for (p, c) in &cells {
        let q = lo + *p;
        if new.contains(q) {
            new.set(q, map[*c as usize]);
        } else {
            skipped += 1;
        }
    }
    let mut what = format!("Placed stamp {name:?} ({}x{}x{}) at {:?}", size.x, size.y, size.z, arr(lo));
    if skipped > 0 {
        what.push_str(&format!("; {skipped} of its voxels fell outside the canvas"));
    }
    let changed = apply(ctx, new);
    Ok(edited(ctx, changed, &what))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(dims: IVec3, from: IVec3, to: IVec3, thick: i32) -> Volume {
        let mut v = Volume::new(dims);
        line_cells(IVec3::ZERO, dims - 1, from, to, &line_brush(thick), |p| v.set(p, 1));
        v
    }

    #[test]
    fn line_thickness_is_its_width() {
        let dims = IVec3::splat(20);
        for t in 1..=4 {
            let v = line(dims, IVec3::new(0, 10, 10), IVec3::new(19, 10, 10), t);
            let across = (0..20).filter(|&y| v.get(IVec3::new(5, y, 10)) != 0).count();
            let up = (0..20).filter(|&z| v.get(IVec3::new(5, 10, z)) != 0).count();
            assert_eq!((across, up), (t as usize, t as usize), "thickness {t}");
        }
    }

    #[test]
    fn huge_lines_are_clipped_without_gaps() {
        let dims = IVec3::splat(20);
        let t = std::time::Instant::now();
        let v = line(dims, IVec3::new(-1_000_000_000, 3, 0), IVec3::new(1_000_000_000, 3, 0), 1);
        assert!(t.elapsed().as_secs_f32() < 0.5);
        assert_eq!(n(&v), 20);
        // A steep diagonal from far away still steps one cell at a time.
        let v = line(dims, IVec3::new(-50, -50, 0), IVec3::new(50, 50, 0), 1);
        assert_eq!(n(&v), 20);
        assert!((0..20).all(|k| v.get(IVec3::new(k, k, 0)) != 0));
        assert_eq!(n(&line(dims, IVec3::new(30, 30, 30), IVec3::new(40, 40, 40), 3)), 0);
    }

    #[test]
    fn mirror_copies_off_the_canvas_are_counted() {
        let old = Volume::new(IVec3::new(10, 4, 4));
        let mut new = old.clone();
        new.set(IVec3::new(1, 0, 0), 2);
        // Plane at x = 2 (twice: 4): x 1 mirrors to 3, on the canvas.
        assert_eq!(mirror_edit(&old, new.clone(), [true, false, false], IVec3::new(4, 0, 0)).1, 0);
        new.set(IVec3::new(8, 0, 0), 2);
        // x 8 mirrors to -4.
        assert_eq!(mirror_edit(&old, new, [true, false, false], IVec3::new(4, 0, 0)).1, 1);
    }

    fn n(v: &Volume) -> usize {
        count(v)
    }

    fn shape(name: &str, axis: usize, a: IVec3, b: IVec3, args: Value) -> Volume {
        let v = Volume::new(IVec3::splat(24));
        fill_solid(&v, &Solid::build(name, axis, a, b, &args).unwrap(), 1, false)
    }

    #[test]
    fn primitives_fill_expected_cells() {
        let (o, e) = (IVec3::ZERO, IVec3::splat(9));
        // A frustum sits between a cone and a cylinder.
        let cyl = n(&shape("cylinder", 2, o, e, json!({})));
        let fr = n(&shape("frustum", 2, o, e, json!({})));
        let cone = n(&shape("cone", 2, o, e, json!({})));
        assert!(cone < fr && fr < cyl, "{cone} {fr} {cyl}");
        // A torus has a hole in the middle and fills its ring.
        let t = shape("torus", 2, o, IVec3::new(15, 15, 3), json!({}));
        assert_eq!(t.get(IVec3::new(8, 8, 2)), 0);
        assert_eq!(t.get(IVec3::new(2, 8, 2)), 1);
        // Rounded box: corners cut, faces full.
        let rb = shape("rounded_box", 2, o, e, json!({ "radius": 3 }));
        assert_eq!(rb.get(o), 0);
        assert_eq!(rb.get(IVec3::new(5, 5, 0)), 1);
        assert!(n(&rb) < 1000 && n(&rb) > 800);
        // Capsule: round ends, full middle.
        let c = shape("capsule", 2, o, IVec3::new(5, 5, 15), json!({}));
        assert_eq!(c.get(IVec3::new(0, 0, 8)), 0);
        assert_eq!(c.get(IVec3::new(2, 2, 15)), 1);
        assert_eq!(c.get(IVec3::new(2, 0, 8)), 1);
        assert_eq!(c.get(IVec3::new(0, 2, 15)), 0);
        // Wedge: half a box, tall at +x.
        let w = shape("wedge", 2, o, e, json!({}));
        assert_eq!(n(&w), 550);
        assert_eq!(w.get(IVec3::new(9, 0, 9)), 1);
        assert_eq!(w.get(IVec3::new(0, 0, 9)), 0);
    }

    #[test]
    fn rotation_tilts_a_cylinder() {
        let (a, b) = (IVec3::new(9, 9, 2), IVec3::new(13, 13, 21));
        let up = shape("cylinder", 2, a, b, json!({}));
        let tilted = shape("cylinder", 2, a, b, json!({ "rotation": [0, 90, 0] }));
        // Lying along x through the same center, about the same size.
        assert_eq!(tilted.get(IVec3::new(2, 11, 11)), 1);
        assert_eq!(tilted.get(IVec3::new(11, 11, 21)), 0);
        assert!(n(&up).abs_diff(n(&tilted)) * 10 < n(&up), "{} {}", n(&up), n(&tilted));
    }

    #[test]
    fn symmetry_mirrors_changed_cells() {
        let old = Volume::new(IVec3::new(10, 6, 4));
        let mut new = old.clone();
        new.set(IVec3::new(1, 2, 3), 5);
        let (m, _) = mirror_edit(&old, new, [true, false, false], IVec3::new(9, 5, 3));
        assert_eq!(m.get(IVec3::new(8, 2, 3)), 5);
        assert_eq!(n(&m), 2);
        // Two axes: four copies; an off-center plane at x = 3.
        let mut new = old.clone();
        new.set(IVec3::new(1, 0, 0), 7);
        let (m, _) = mirror_edit(&old, new, [true, true, false], IVec3::new(6, 5, 3));
        assert_eq!(n(&m), 4);
        assert_eq!(m.get(IVec3::new(5, 5, 0)), 7);
        // Erasing mirrors too.
        let (erased, _) = mirror_edit(&m, Volume::from_cells(m.dims, { let mut c = m.cells.to_vec(); c[m.index(IVec3::new(1, 0, 0))] = 0; c }), [true, true, false], IVec3::new(6, 5, 3));
        assert_eq!(n(&erased), 0);
    }

    #[test]
    fn tube_and_polygon() {
        let v = Volume::new(IVec3::splat(32));
        let path = sample_curve(&[Vec3::new(2.0, 2.0, 2.0), Vec3::new(16.0, 28.0, 10.0), Vec3::new(29.0, 4.0, 20.0)], "smooth", false).unwrap();
        let t = tube_volume(&v, &path, 3.0, 0.5, 1);
        assert_eq!(t.get(IVec3::new(2, 2, 2)), 1);
        assert_eq!(t.get(IVec3::new(29, 4, 20)), 1);
        assert_eq!(t.get(IVec3::new(16, 28, 10)), 1);
        // A thick start, a thin end.
        assert_eq!(t.get(IVec3::new(4, 2, 2)), 1);
        assert_eq!(t.get(IVec3::new(29, 4, 22)), 0);
        // An L-shaped floor plan, 3 layers, with a square hole.
        let l = [Vec2::new(0.0, 0.0), Vec2::new(9.0, 0.0), Vec2::new(9.0, 4.0), Vec2::new(4.0, 4.0), Vec2::new(4.0, 9.0), Vec2::new(0.0, 9.0)];
        let p = extrude_volume(&v, 2, &l, &[], 0, 2, 1, false);
        assert_eq!(n(&p), (10 * 5 + 5 * 5) * 3);
        let hole = vec![Vec2::new(1.0, 1.0), Vec2::new(2.0, 1.0), Vec2::new(2.0, 2.0), Vec2::new(1.0, 2.0)];
        let h = extrude_volume(&v, 2, &l, &[hole], 0, 0, 1, false);
        assert_eq!(n(&h), 75 - 4);
    }

    #[test]
    fn any_vox_file_stamps_as_one_cropped_model() {
        use crate::world::{ROOT, World};
        let dir = std::env::temp_dir().join(format!("voxely-stamp-file-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("scene.vox");
        // Two models with empty margins, 4 apart on x.
        let mut a = Volume::new(IVec3::new(4, 4, 4));
        a.set(IVec3::new(1, 1, 0), 2);
        let mut b = Volume::new(IVec3::new(4, 4, 4));
        b.set(IVec3::new(1, 1, 0), 5);
        let mut w = World::empty();
        w.add_model(ROOT, "a", a, IVec3::ZERO);
        w.add_model(ROOT, "b", b, IVec3::new(4, 0, 0));
        crate::vox::write(&path, &w, &crate::vox::default_palette(), &[], &[], &[]).unwrap();
        let (v, _) = read_stamp(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(v.dims, IVec3::new(5, 1, 1));
        assert_eq!(v.get(IVec3::new(0, 0, 0)), 2);
        assert_eq!(v.get(IVec3::new(4, 0, 0)), 5);
    }

    #[test]
    fn stamps_round_trip_and_turn() {
        let dir = std::env::temp_dir().join(format!("voxely-stamp-{}", std::process::id()));
        let path = dir.join("s.vox");
        let mut v = Volume::new(IVec3::new(3, 2, 1));
        v.set(IVec3::new(0, 0, 0), 4);
        v.set(IVec3::new(2, 1, 0), 9);
        let pal = crate::vox::default_palette();
        write_stamp(&path, &v, &pal).unwrap();
        let (back, p2) = read_stamp(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(back.dims, v.dims);
        assert_eq!(back.cells, v.cells);
        assert_eq!(p2[..256], pal[..256]);
        // A quarter turn about z: 3x2 becomes 2x3.
        let (cells, size) = orient(&v, &[], crate::world::Rot::turn(2, 1));
        assert_eq!(size, IVec3::new(2, 3, 1));
        assert!(cells.iter().all(|(p, _)| p.cmpge(IVec3::ZERO).all() && p.cmplt(size).all()));
        let (flipped, _) = orient(&v, &[0], crate::world::Rot::IDENTITY);
        assert!(flipped.contains(&(IVec3::new(2, 0, 0), 4)));
    }
}
