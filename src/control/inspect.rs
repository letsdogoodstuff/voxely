//! Inspecting and safeguarding: edit reports, slice/projection maps,
//! checkpoints, unsaved-edit tracking and autosave.

use std::cell::Cell;
use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use glam::IVec3;
use serde_json::{Value, json};

use super::{Ctx, arr, boolean, get, int, string, text, tool, xyz};
use crate::app;
use crate::scene::{GpuMaterial, Volume};
use crate::world::World;

/// Edit reports mention unsaved edits from this many on.
const UNSAVED_NOTE: usize = 10;
const AUTOSAVE_EVERY: usize = 25;
/// Autosaves at most this often (a script can commit many edits a second).
const AUTOSAVE_GAP: Duration = Duration::from_secs(5);
const MAX_CHECKPOINTS: usize = 16;
/// Default cap on a map's width and height, in cells.
const MAP_CELLS: i64 = 128;
/// Refuse maps longer than this many characters.
const MAP_CHARS: usize = 400_000;
/// Cells examined when counting how much of a shape fell off the canvas.
const SPILL_WORK: u64 = 32 << 20;

/// Tools after which the scene is replaced; unsaved edits are autosaved
/// first.
const REPLACES_SCENE: [&str; 4] = ["open_project", "open_file", "new_project", "new_scene"];

pub struct State {
    /// Committed edits since the project was last saved.
    unsaved: usize,
    /// The save path `unsaved` counts toward (a new path restarts it).
    project: Option<PathBuf>,
    autosave: bool,
    every: usize,
    last_autosave: Option<Instant>,
    /// An autosave message for the next report.
    note: Cell<Option<String>>,
    /// `last_edit` is from this tool call and not yet reported.
    fresh: Cell<bool>,
    checkpoints: Vec<Checkpoint>,
}

impl Default for State {
    fn default() -> Self {
        State {
            unsaved: 0,
            project: None,
            autosave: true,
            every: AUTOSAVE_EVERY,
            last_autosave: None,
            note: Cell::new(None),
            fresh: Cell::new(false),
            checkpoints: Vec::new(),
        }
    }
}

impl State {
    /// The project was just saved to `path` (called by `app::save_project`).
    pub fn mark_saved(&mut self, path: &Path) {
        self.unsaved = 0;
        self.project = Some(path.to_path_buf());
    }
}

/// A whole-scene snapshot.
#[derive(Clone)]
struct Checkpoint {
    name: String,
    world: World,
    editing: Option<usize>,
    palette: Vec<[u8; 4]>,
    materials: Vec<GpuMaterial>,
    render_dicts: Vec<crate::vox::Dict>,
    source: Option<PathBuf>,
    save_path: PathBuf,
    taken: Instant,
}

pub const PART_TOOLS: &[&str] = &["get_projection"];

pub fn tools() -> Vec<Value> {
    let axis = json!({ "type": "string", "enum": ["x", "y", "z"] });
    vec![
        tool("get_projection", "A text height map: for axis z, each (x, y) column shows the topmost voxel's color index and its z, '.' where the column is empty (for x/y, the first voxel seen from that side). The quickest way to see a part's overall layout. Rows and every 5th column are labeled with coordinates; big canvases are downsampled (see max_cells), so crop with from/to to zoom in.", json!({
            "axis": axis,
            "from_side": { "type": "string", "enum": ["max", "min"], "description": "Look from the high end of the axis (default; for z, from above) or the low end" },
            "show": { "type": "string", "enum": ["both", "color", "height"], "description": "Which maps to return (default both)" },
            "from": xyz("Crop corner, inclusive (also limits the depth searched)"), "to": xyz("Other crop corner"),
            "max_cells": { "type": "integer", "description": "Max map width/height before downsampling (default 128)" },
        }), &[]),
        tool("checkpoint", "Save a named in-memory snapshot of the whole scene (every part and group, palette and materials) before trying something risky; restore returns to it in one step. Without `name`, lists checkpoints. Reusing a name replaces it; the newest 16 are kept until voxely exits (they aren't saved to disk).", json!({
            "name": { "type": "string" },
        }), &[]),
        tool("restore", "Return the whole scene to a checkpoint. Undo reverts a restore's part and voxel changes, but not the palette and materials it brought back.", json!({
            "name": { "type": "string", "description": "A checkpoint name (checkpoint without a name lists them)" },
        }), &["name"]),
        tool("set_autosave", "Autosave saves the project (as save_file without a path does) after every `every` voxel edits, and before open_project/open_file/new_project/new_scene replace a scene with unsaved edits. On by default, every 25 edits. Edit reports show the unsaved count once it reaches 10.", json!({
            "enabled": { "type": "boolean" },
            "every": { "type": "integer", "description": "Edits between autosaves (default 25)" },
        }), &[]),
    ]
}

pub fn dispatch(ctx: &mut Ctx, name: &str, args: &Value) -> Option<Result<Vec<Value>>> {
    let st = &ctx.ed.tools.inspect;
    st.fresh.set(false);
    st.note.set(None);
    if REPLACES_SCENE.contains(&name) && st.autosave && st.unsaved > 0 && st.project.as_ref() == Some(&ctx.ed.save_path) {
        autosave(ctx, "before replacing the scene");
    }
    Some(match name {
        "get_projection" => projection(&ctx.scene.volume, &ctx.scene.palette, args).map(|s| vec![text(s)]),
        "checkpoint" => checkpoint(ctx, args),
        "restore" => restore(ctx, args),
        "set_autosave" => set_autosave(ctx, args),
        _ => return None,
    })
}

/// Called after every committed edit (e.g. to count edits for autosave).
pub fn after_commit(ctx: &mut Ctx) {
    ctx.ed.tools.inspect.fresh.set(true);
    count_edit(ctx);
}

/// Count one unsaved edit, autosaving when due.
fn count_edit(ctx: &mut Ctx) {
    let st = &mut ctx.ed.tools.inspect;
    if st.project.as_ref() != Some(&ctx.ed.save_path) {
        st.project = Some(ctx.ed.save_path.clone());
        st.unsaved = 0;
    }
    st.unsaved += 1;
    if st.autosave && st.unsaved >= st.every && st.last_autosave.is_none_or(|t| t.elapsed() >= AUTOSAVE_GAP) {
        autosave(ctx, "");
    }
}

fn autosave(ctx: &mut Ctx, when: &str) {
    let n = ctx.ed.tools.inspect.unsaved;
    let path = ctx.ed.save_path.clone();
    let when = if when.is_empty() { String::new() } else { format!(" {when}") };
    let note = match app::save_project(ctx.scene, ctx.ed, path.clone()) {
        Ok(p) => format!("Autosaved {} to {}{when}.", plural(n, "edit"), p.display()),
        Err(e) => format!("Autosave to {} failed ({e:#}); call save_file with a path to keep this work.", path.display()),
    };
    let st = &mut ctx.ed.tools.inspect;
    st.last_autosave = Some(Instant::now());
    st.note.set(Some(note));
}

/// The autosave message and unsaved count that end a report.
fn tail(ctx: &Ctx) -> String {
    let st = &ctx.ed.tools.inspect;
    let mut s = String::new();
    if let Some(note) = st.note.take() {
        s.push(' ');
        s.push_str(&note);
    }
    let unsaved = if st.project.as_ref() == Some(&ctx.ed.save_path) { st.unsaved } else { 0 };
    if unsaved >= UNSAVED_NOTE {
        let auto = if st.autosave { format!("autosave every {}", st.every) } else { "autosave is off; call save_file".into() };
        s.push_str(&format!(" Unsaved: {unsaved} edits ({auto})."));
    }
    s
}

// ---------------------------------------------------------------- reports

/// "1 edit", "2 edits"
fn plural(n: usize, what: &str) -> String {
    format!("{} {what}{}", thousands(n), if n == 1 { "" } else { "s" })
}

fn thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// "Part `body` 40x40x40, 5,120 voxels" (or "World ..." in World mode).
fn scope(ctx: &Ctx) -> String {
    let v = &ctx.scene.volume;
    let d = v.dims;
    let what = match ctx.scene.editing() {
        Some(id) => format!("Part `{}`", ctx.scene.world.label(id)),
        None => "World".into(),
    };
    format!("{what} {}x{}x{}, {}", d.x, d.y, d.z, plural(super::count(v), "voxel"))
}

/// The summary after an edit: what changed (from `last_edit` when this call
/// committed one), why nothing did if known, the part's size and count,
/// then `notes` as separate lines.
pub(super) fn report(ctx: &Ctx, changed: bool, what: &str, why: Option<String>, notes: Vec<String>) -> Vec<Value> {
    let fresh = ctx.ed.tools.inspect.fresh.replace(false);
    let mut s = what.to_string();
    if changed {
        match ctx.ed.tools.last_edit.as_ref().filter(|_| fresh) {
            Some(e) if e.resized => s.push_str(": canvas resized"),
            Some(e) => {
                s.push_str(&format!(": +{} -{} voxels, {} recolored", thousands(e.added), thousands(e.removed), thousands(e.recolored)));
                if let Some((lo, hi)) = e.bounds {
                    s.push_str(&format!("; changed {:?}..{:?}", arr(lo), arr(hi)));
                }
            }
            None => {}
        }
    } else {
        s.push_str(": nothing changed");
        if let Some(why) = why {
            s.push_str(&format!(" ({why})"));
        }
    }
    s.push_str(&format!(". {}.", scope(ctx)));
    s.push_str(&tail(ctx));
    let mut out = vec![text(s)];
    out.extend(notes.into_iter().map(text));
    out
}

/// `report` for fill_box / draw_shape / draw_line (`shape` "line" for
/// lines), reading the tool's arguments to tell when the shape spilled past
/// the canvas.
pub(super) fn shape_report(ctx: &Ctx, changed: bool, what: &str, shape: &str, a: &Value) -> Vec<Value> {
    let dims = ctx.scene.volume.dims;
    let color = get(a, "color").and_then(|c| super::color(ctx.scene, c).ok()).unwrap_or(1);
    let spill = spill(dims, shape, a);
    let mut notes = Vec::new();
    if let Some(sp) = spill.as_ref().filter(|s| s.outside > 0 && s.touches && color != 0) {
        let n = if sp.exact { format!("{} cells", thousands(sp.outside)) } else { "Part of the shape".into() };
        notes.push(format!("{n} fell outside the canvas ({}); resize_canvas or move the shape.", sides(dims, sp.sides)));
    }
    let why = (!changed).then(|| match &spill {
        Some(sp) if !sp.touches => format!("it lies entirely outside the canvas {}x{}x{}; coordinates run 0..{}", dims.x, dims.y, dims.z, fmt_max(dims)),
        _ if color == 0 => "that area was already empty".into(),
        _ => format!("every cell it covers already has color {color}"),
    });
    report(ctx, changed, what, why, notes)
}

fn fmt_max(dims: IVec3) -> String {
    format!("{:?}", arr(dims - IVec3::ONE))
}

/// "z > 39, x < 0"
fn sides(dims: IVec3, sides: [[bool; 2]; 3]) -> String {
    let names = ["x", "y", "z"];
    let mut out = Vec::new();
    for k in 0..3 {
        if sides[k][0] {
            out.push(format!("{} < 0", names[k]));
        }
        if sides[k][1] {
            out.push(format!("{} > {}", names[k], dims[k] - 1));
        }
    }
    out.join(", ")
}

/// How much of a shape tool's intended cells fell off the canvas.
#[derive(Debug, PartialEq)]
struct Spill {
    outside: usize,
    /// Which sides it spilled past: [axis][low, high].
    sides: [[bool; 2]; 3],
    /// False when the count was cut short (then `sides` is from the box).
    exact: bool,
    /// The shape's box overlaps the canvas.
    touches: bool,
}

fn mark(dims: IVec3, sides: &mut [[bool; 2]; 3], p: IVec3) {
    for k in 0..3 {
        sides[k][0] |= p[k] < 0;
        sides[k][1] |= p[k] >= dims[k];
    }
}

fn spill(dims: IVec3, shape: &str, a: &Value) -> Option<Spill> {
    let (from, to) = (super::ivec3(a, "from").ok()?, super::ivec3(a, "to").ok()?);
    let mut sp = Spill { outside: 0, sides: [[false; 2]; 3], exact: true, touches: false };
    let canvas = |p: IVec3| p.cmpge(IVec3::ZERO).all() && p.cmplt(dims).all();
    if shape == "line" {
        // Mirrors draw_line's rasterization.
        let thick = int(a, "thickness").unwrap_or(1).clamp(1, 64) as i32;
        let d = to - from;
        let steps = d.abs().max_element().max(1);
        let k = (thick - 1) / 2 + 1;
        if steps as u64 * ((2 * k + 1) as u64).pow(3) > SPILL_WORK {
            return None;
        }
        let r = thick as f32 * 0.5;
        let mut cells = HashSet::new();
        for i in 0..=steps {
            let p = (from.as_vec3() + d.as_vec3() * (i as f32 / steps as f32)).round().as_ivec3();
            for z in -k..=k {
                for y in -k..=k {
                    for x in -k..=k {
                        let o = IVec3::new(x, y, z);
                        if thick == 1 && o != IVec3::ZERO || o.as_vec3().length() > r {
                            continue;
                        }
                        cells.insert(p + o);
                    }
                }
            }
        }
        for p in cells {
            if canvas(p) {
                sp.touches = true;
            } else {
                sp.outside += 1;
                mark(dims, &mut sp.sides, p);
            }
        }
        return Some(sp);
    }
    let solid = super::build::Solid::build(shape, super::axis(a).ok()?, from, to, a).ok()?;
    let hollow = boolean(a, "hollow").unwrap_or(false);
    let (lo, hi) = solid.bounds();
    sp.touches = hi.cmpge(IVec3::ZERO).all() && lo.cmplt(dims).all();
    let inside = |p: IVec3| solid.has_cell(p);
    let drawn = |p: IVec3| inside(p) && (!hollow || super::NEIGHBORS.iter().any(|&n| !inside(p + n)));
    let mut work = 0u64;
    'rows: for z in lo.z..=hi.z {
        for y in lo.y..=hi.y {
            // Only the parts of the row outside the canvas.
            let row_in = (0..dims.y).contains(&y) && (0..dims.z).contains(&z);
            let spans = if row_in { [(lo.x, hi.x.min(-1)), (lo.x.max(dims.x), hi.x)] } else { [(lo.x, hi.x), (1, 0)] };
            for (x0, x1) in spans {
                for x in x0..=x1 {
                    work += 1;
                    if work > SPILL_WORK {
                        sp.exact = false;
                        break 'rows;
                    }
                    let p = IVec3::new(x, y, z);
                    if drawn(p) {
                        sp.outside += 1;
                        mark(dims, &mut sp.sides, p);
                    }
                }
            }
        }
    }
    if !sp.exact {
        // Too big to count: say which sides the box crosses.
        sp.sides = [[false; 2]; 3];
        mark(dims, &mut sp.sides, lo);
        mark(dims, &mut sp.sides, hi);
        sp.outside = 1;
    }
    Some(sp)
}

// ---------------------------------------------------------------- maps

const AXES: [&str; 3] = ["x", "y", "z"];

/// The in-plane axes for a map along `ax`: columns, rows.
fn plane(ax: usize) -> (usize, usize) {
    match ax {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

/// The `from`/`to` crop clamped to the canvas (default: all of it).
fn crop(v: &Volume, a: &Value) -> Result<(IVec3, IVec3)> {
    let max = v.dims - IVec3::ONE;
    let from = super::opt_ivec3(a, "from")?.unwrap_or(IVec3::ZERO);
    let to = super::opt_ivec3(a, "to")?.unwrap_or(max);
    let (lo, hi) = (from.min(to), from.max(to));
    if hi.cmplt(IVec3::ZERO).any() || lo.cmpgt(max).any() {
        bail!("the crop {:?}..{:?} lies outside the canvas; coordinates run [0, 0, 0]..{:?}", arr(lo), arr(hi), arr(max));
    }
    Ok((lo.max(IVec3::ZERO), hi.min(max)))
}

/// Downsampling step so a `nu` x `nw` map fits `max_cells`.
fn step(a: &Value, nu: i32, nw: i32) -> i32 {
    let cap = int(a, "max_cells").unwrap_or(MAP_CELLS).clamp(8, 1024) as i32;
    ((nu.max(nw) + cap - 1) / cap).max(1)
}

/// Cell width for numbers up to `max` (3 keeps the classic slice layout).
fn cell_width(max: u32) -> usize {
    (max.to_string().len() + 1).max(3)
}

/// A line labeling every 5th column with its coordinate.
fn ruler(name: &str, lo: i32, cols: usize, step: i32, width: usize) -> String {
    let mut line: Vec<char> = format!("{name:>4} ").chars().collect();
    let prefix = line.len();
    line.resize(prefix + cols * width, ' ');
    for k in (0..cols).step_by(5) {
        let label: Vec<char> = (lo + k as i32 * step).to_string().chars().collect();
        let end = prefix + (k + 1) * width;
        let start = end.saturating_sub(label.len());
        line[start..end].copy_from_slice(&label);
    }
    line.into_iter().collect::<String>().trim_end().to_string()
}

/// Map rows over columns `u` and rows `w` (top = high), one cell per
/// `step` block, labeled by the block's low corner.
#[allow(clippy::too_many_arguments)]
fn map_rows(out: &mut String, (u, w): (usize, usize), lo: IVec3, hi: IVec3, step: i32, width: usize, mut cell: impl FnMut(i32, i32) -> Option<u32>) {
    let cols = ((hi[u] - lo[u]) / step + 1) as usize;
    let rows = (hi[w] - lo[w]) / step + 1;
    out.push_str(&ruler(AXES[u], lo[u], cols, step, width));
    out.push('\n');
    for k in (0..rows).rev() {
        let row = lo[w] + k * step;
        let mut line = format!("{row:>4} ");
        for c in 0..cols as i32 {
            match cell(lo[u] + c * step, row) {
                Some(n) => line.push_str(&format!("{n:>width$}")),
                None => line.push_str(&format!("{:>width$}", ".")),
            }
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
}

fn legend(used: &BTreeSet<u8>, palette: &[[u8; 4]]) -> String {
    let list: Vec<String> = used.iter().map(|&c| format!("{c}={}", super::hex(palette[c as usize]))).collect();
    format!("colors: {}", list.join(" "))
}

/// get_slice: one layer, or a range, as text.
pub(super) fn slice(v: &Volume, palette: &[[u8; 4]], a: &Value) -> Result<String> {
    let ax = super::axis(a)?;
    let (u, w) = plane(ax);
    let n = v.dims[ax];
    let name = AXES[ax];
    let (first, last) = match (int(a, "index"), int(a, "from_index"), int(a, "to_index")) {
        (Some(i), None, None) => (i, i),
        (i, f, t) if f.is_some() || t.is_some() => (f.or(i).unwrap_or(0), t.or(i).unwrap_or(n as i64 - 1)),
        _ => bail!("give `index` for one layer, or `from_index`/`to_index` for a range; {name} runs 0..{}", n - 1),
    };
    let (first, last) = (first.min(last), first.max(last));
    for i in [first, last] {
        if i < 0 || i >= n as i64 {
            let d = v.dims;
            bail!("{name} index {i} is outside 0..{} (canvas {}x{}x{}); use 0..{}, or resize_canvas to grow it", n - 1, d.x, d.y, d.z, n - 1);
        }
    }
    let (first, last) = (first as i32, last as i32);
    let (mut lo, mut hi) = crop(v, a)?;
    let cropped = lo[u] != 0 || lo[w] != 0 || hi[u] != v.dims[u] - 1 || hi[w] != v.dims[w] - 1;
    (lo[ax], hi[ax]) = (first, last);
    let step = step(a, hi[u] - lo[u] + 1, hi[w] - lo[w] + 1);
    // Colors in range, for the cell width and to spot empty layers.
    let mut layer_used = vec![false; (last - first + 1) as usize];
    let mut max_color = 0u32;
    for z in lo.z..=hi.z {
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let p = IVec3::new(x, y, z);
                let c = v.get(p);
                if c != 0 {
                    layer_used[(p[ax] - first) as usize] = true;
                    max_color = max_color.max(c as u32);
                }
            }
        }
    }
    let width = cell_width(max_color);
    let cols = ((hi[u] - lo[u]) / step + 1) as usize;
    let rows = ((hi[w] - lo[w]) / step + 1) as usize;
    let shown = if first == last { 1 } else { layer_used.iter().filter(|&&b| b).count() };
    let size = shown * (rows + 2) * (6 + cols * width);
    if size > MAP_CHARS {
        bail!("that's about {} KB of text; narrow from_index/to_index, crop with from/to, or lower max_cells", size / 1000);
    }
    let mut out = if first == last { format!("Slice {name}={first}") } else { format!("Slices {name}={first}..{last}") };
    out.push_str(&format!(": columns {} {}..{}, rows {} from {} (top) down to {}", AXES[u], lo[u], hi[u], AXES[w], hi[w], lo[w]));
    if step > 1 {
        out.push_str(&format!("; downsampled: each cell is the most common color in a {step}x{step} block, labeled by its low corner"));
    }
    if cropped && step == 1 && first == last {
        out.push_str(" (cropped)");
    }
    out.push('\n');
    let mut used = BTreeSet::new();
    let mut empty_from: Option<i32> = None;
    for layer in first..=last {
        let multi = first != last;
        if multi && !layer_used[(layer - first) as usize] {
            empty_from.get_or_insert(layer);
            if layer < last {
                continue;
            }
        }
        if let Some(e) = empty_from.take() {
            let end = if layer_used[(layer - first) as usize] { layer - 1 } else { layer };
            out.push_str(&if e == end { format!("{name}={e}: empty\n") } else { format!("{name}={e}..{end}: empty\n") });
            if !layer_used[(layer - first) as usize] {
                continue;
            }
        }
        if multi {
            out.push_str(&format!("{name}={layer}\n"));
        }
        let at = |cu: i32, cw: i32| -> u8 {
            let mut p = IVec3::ZERO;
            p[ax] = layer;
            if step == 1 {
                p[u] = cu;
                p[w] = cw;
                return v.get(p);
            }
            let mut counts = [0u32; 256];
            for dw in 0..step.min(hi[w] - cw + 1) {
                for du in 0..step.min(hi[u] - cu + 1) {
                    p[u] = cu + du;
                    p[w] = cw + dw;
                    counts[v.get(p) as usize] += 1;
                }
            }
            (1..256).max_by_key(|&c| (counts[c], std::cmp::Reverse(c))).filter(|&c| counts[c] > 0).map_or(0, |c| c as u8)
        };
        map_rows(&mut out, (u, w), lo, hi, step, width, |cu, cw| {
            let c = at(cu, cw);
            (c != 0).then(|| {
                used.insert(c);
                c as u32
            })
        });
    }
    out.push_str(&legend(&used, palette));
    Ok(out)
}

/// get_projection: the first voxel seen along an axis, as color and depth
/// maps.
pub(super) fn projection(v: &Volume, palette: &[[u8; 4]], a: &Value) -> Result<String> {
    let ax = super::axis(a)?;
    let (u, w) = plane(ax);
    let name = AXES[ax];
    let from_max = match string(a, "from_side").unwrap_or("max") {
        "max" => true,
        "min" => false,
        s => bail!("unknown from_side {s:?}; use \"max\" (look from high {name}) or \"min\""),
    };
    let (show_color, show_height) = match string(a, "show").unwrap_or("both") {
        "both" => (true, true),
        "color" => (true, false),
        "height" => (false, true),
        s => bail!("unknown show {s:?}; use \"both\", \"color\" or \"height\""),
    };
    let (lo, hi) = crop(v, a)?;
    let (nu, nw) = (hi[u] - lo[u] + 1, hi[w] - lo[w] + 1);
    // First hit per full-resolution column: (depth, color).
    let mut hits: Vec<Option<(i32, u8)>> = vec![None; (nu * nw) as usize];
    for cw in 0..nw {
        for cu in 0..nu {
            let mut p = IVec3::ZERO;
            p[u] = lo[u] + cu;
            p[w] = lo[w] + cw;
            let depths: Box<dyn Iterator<Item = i32>> = if from_max { Box::new((lo[ax]..=hi[ax]).rev()) } else { Box::new(lo[ax]..=hi[ax]) };
            for d in depths {
                p[ax] = d;
                let c = v.get(p);
                if c != 0 {
                    hits[(cu + cw * nu) as usize] = Some((d, c));
                    break;
                }
            }
        }
    }
    let step = step(a, nu, nw);
    // Per block: the nearest hit to the viewer.
    let block = |bu: i32, bw: i32| -> Option<(i32, u8)> {
        let mut best: Option<(i32, u8)> = None;
        for cw in bw - lo[w]..(bw - lo[w] + step).min(nw) {
            for cu in bu - lo[u]..(bu - lo[u] + step).min(nu) {
                if let Some(h) = hits[(cu + cw * nu) as usize] {
                    if best.is_none_or(|b| if from_max { h.0 > b.0 } else { h.0 < b.0 }) {
                        best = Some(h);
                    }
                }
            }
        }
        best
    };
    let cols = ((nu - 1) / step + 1) as usize;
    let rows = ((nw - 1) / step + 1) as usize;
    let maps = show_color as usize + show_height as usize;
    let size = maps * (rows + 2) * (6 + cols * 5);
    if size > MAP_CHARS {
        bail!("that's about {} KB of text; crop with from/to or lower max_cells", size / 1000);
    }
    let side = if from_max { format!("+{name} side (from {name}={})", hi[ax]) } else { format!("-{name} side (from {name}={})", lo[ax]) };
    let mut out = format!("Projection along {name}, seen from the {side}: columns {} {}..{}, rows {} from {} (top) down to {}", AXES[u], lo[u], hi[u], AXES[w], hi[w], lo[w]);
    if step > 1 {
        out.push_str(&format!("; downsampled: each cell is the nearest voxel in a {step}x{step} block, labeled by its low corner"));
    }
    out.push_str(&format!(". '.' = no voxel in that column ({name} {}..{}).\n", lo[ax], hi[ax]));
    let mut used = BTreeSet::new();
    if show_color {
        out.push_str("Color (palette index of the first voxel seen):\n");
        let max = hits.iter().flatten().map(|h| h.1 as u32).max().unwrap_or(0);
        map_rows(&mut out, (u, w), lo, hi, step, cell_width(max), |cu, cw| {
            block(cu, cw).map(|(_, c)| {
                used.insert(c);
                c as u32
            })
        });
    }
    if show_height {
        out.push_str(&format!("Height ({name} of the first voxel seen):\n"));
        let max = hits.iter().flatten().map(|h| h.0 as u32).max().unwrap_or(0);
        map_rows(&mut out, (u, w), lo, hi, step, cell_width(max), |cu, cw| block(cu, cw).map(|(d, _)| d as u32));
    }
    if show_color {
        out.push_str(&legend(&used, palette));
    } else {
        out.pop();
    }
    Ok(out)
}

// ---------------------------------------------------------------- checkpoints

fn ago(t: Instant) -> String {
    let s = t.elapsed().as_secs();
    match s {
        0..60 => format!("{s}s ago"),
        60..3600 => format!("{}m ago", s / 60),
        _ => format!("{}h {}m ago", s / 3600, s / 60 % 60),
    }
}

/// "3 parts, 5,120 voxels"
fn describe(world: &World) -> String {
    let models = world.models();
    let voxels: usize = models.iter().filter_map(|&m| world.node(m)?.volume()).map(super::count).sum();
    format!("{}, {}", plural(models.len(), "part"), plural(voxels, "voxel"))
}

fn list(st: &State) -> String {
    if st.checkpoints.is_empty() {
        return "No checkpoints. checkpoint {\"name\": \"...\"} saves one.".into();
    }
    let items: Vec<String> = st.checkpoints.iter().rev().map(|c| format!("`{}` ({}, {})", c.name, ago(c.taken), describe(&c.world))).collect();
    format!("Checkpoints, newest first: {}", items.join("; "))
}

fn take(ctx: &Ctx, name: &str) -> Checkpoint {
    Checkpoint {
        name: name.to_string(),
        world: ctx.scene.synced_world().into_owned(),
        editing: ctx.scene.editing(),
        palette: ctx.scene.palette.clone(),
        materials: ctx.scene.materials.clone(),
        render_dicts: ctx.scene.render_dicts.clone(),
        source: ctx.ed.source.clone(),
        save_path: ctx.ed.save_path.clone(),
        taken: Instant::now(),
    }
}

fn checkpoint(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let Some(name) = string(a, "name").map(str::trim).filter(|n| !n.is_empty()) else {
        return Ok(vec![text(list(&ctx.ed.tools.inspect))]);
    };
    let cp = take(ctx, name);
    let what = describe(&cp.world);
    let st = &mut ctx.ed.tools.inspect;
    let replaced = st.checkpoints.iter().position(|c| c.name == name).map(|i| st.checkpoints.remove(i)).is_some();
    st.checkpoints.push(cp);
    let mut msg = format!("Checkpoint `{name}` {}: {what}. restore {{\"name\": \"{name}\"}} returns to it.", if replaced { "replaced" } else { "saved" });
    if st.checkpoints.len() > MAX_CHECKPOINTS {
        let old = st.checkpoints.remove(0);
        msg.push_str(&format!(" Dropped the oldest, `{}` (only {MAX_CHECKPOINTS} are kept).", old.name));
    }
    Ok(vec![text(msg)])
}

/// Put a checkpoint's scene in place (the tree change is one undo step).
fn put_back(ctx: &mut Ctx, cp: Checkpoint) {
    let scene = &mut *ctx.scene;
    scene.change(|w| *w = cp.world);
    scene.palette = cp.palette;
    scene.materials = cp.materials;
    scene.render_dicts = cp.render_dicts;
    ctx.ed.source = cp.source;
    ctx.ed.save_path = cp.save_path;
    app::set_mode(ctx.scene, ctx.ed, cp.editing);
}

fn restore(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let name = string(a, "name").map(str::trim).context("missing `name`; checkpoint without a name lists them")?;
    let st = &ctx.ed.tools.inspect;
    let Some(cp) = st.checkpoints.iter().find(|c| c.name == name).cloned() else {
        let names: Vec<String> = st.checkpoints.iter().map(|c| format!("`{}`", c.name)).collect();
        if names.is_empty() {
            bail!("no checkpoint named `{name}`; there are none yet (checkpoint {{\"name\": \"...\"}} saves one)");
        }
        bail!("no checkpoint named `{name}`; checkpoints: {}", names.join(", "));
    };
    let (taken, other) = (cp.taken, cp.save_path != ctx.ed.save_path);
    put_back(ctx, cp);
    super::tree_changed(ctx);
    super::materials_changed(ctx);
    count_edit(ctx);
    let mut msg = format!("Restored `{name}` (taken {}): {}", ago(taken), describe(&ctx.scene.world));
    if let Some(id) = ctx.scene.editing() {
        msg.push_str(&format!("; editing `{}`", ctx.scene.world.label(id)));
    }
    if other {
        msg.push_str(&format!("; it came from another project, so saving now goes to {}", ctx.ed.save_path.display()));
    }
    msg.push_str(". undo reverts the parts and voxels (not the palette and materials).");
    msg.push_str(&tail(ctx));
    Ok(vec![text(msg)])
}

fn set_autosave(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let st = &mut ctx.ed.tools.inspect;
    if let Some(n) = get(a, "every") {
        let n = n.as_i64().filter(|n| (1..=100_000).contains(n)).context("`every` must be a whole number of edits, 1..=100000 (e.g. 25)")?;
        st.every = n as usize;
    }
    if let Some(on) = boolean(a, "enabled") {
        st.autosave = on;
    }
    let unsaved = if st.project.as_ref() == Some(&ctx.ed.save_path) { st.unsaved } else { 0 };
    let msg = if st.autosave {
        format!("Autosave on: every {} edits, to {}. {} now.", st.every, ctx.ed.save_path.display(), plural(unsaved, "unsaved edit"))
    } else {
        format!("Autosave off. {} now; call save_file to keep them.", plural(unsaved, "unsaved edit"))
    };
    Ok(vec![text(msg)])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: Value) -> Value {
        v
    }

    #[test]
    fn spill_counts_cells_past_the_canvas() {
        let dims = IVec3::splat(40);
        // A box poking 2 layers past the top: 10x10x2 cells outside.
        let sp = spill(dims, "box", &args(json!({ "from": [0, 0, 30], "to": [9, 9, 41] }))).unwrap();
        assert_eq!(sp.outside, 200);
        assert_eq!(sides(dims, sp.sides), "z > 39");
        assert!(sp.exact && sp.touches);
        // A sphere inscribed in the canvas loses nothing.
        let sp = spill(dims, "ellipsoid", &args(json!({ "from": [0, 0, 0], "to": [39, 39, 39] }))).unwrap();
        assert_eq!(sp.outside, 0);
        // Entirely outside.
        let sp = spill(dims, "box", &args(json!({ "from": [50, 0, 0], "to": [55, 5, 5] }))).unwrap();
        assert!(!sp.touches && sp.outside == 216);
        // A line from inside to x = -5.
        let sp = spill(dims, "line", &args(json!({ "from": [5, 0, 0], "to": [-5, 0, 0] }))).unwrap();
        assert_eq!(sp.outside, 5);
        assert_eq!(sides(dims, sp.sides), "x < 0");
    }

    #[test]
    fn huge_shapes_spill_without_counting() {
        let sp = spill(IVec3::splat(40), "ellipsoid", &args(json!({ "from": [-5000, -5000, 0], "to": [5000, 5000, 10] }))).unwrap();
        assert!(!sp.exact);
        assert_eq!(sides(IVec3::splat(40), sp.sides), "x < 0, x > 39, y < 0, y > 39");
    }

    fn sample() -> Volume {
        let mut v = Volume::new(IVec3::new(12, 8, 6));
        v.set(IVec3::new(1, 2, 0), 3);
        v.set(IVec3::new(1, 2, 4), 7);
        v.set(IVec3::new(10, 7, 2), 12);
        v
    }

    #[test]
    fn single_slice_keeps_the_classic_layout() {
        let pal = vec![[0u8; 4]; 256];
        let s = slice(&sample(), &pal, &json!({ "axis": "z", "index": 0 })).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines[0], "Slice z=0: columns x 0..11, rows y from 7 (top) down to 0");
        assert_eq!(lines[1], "   x   0              5             10");
        assert_eq!(lines[2], "   7   .  .  .  .  .  .  .  .  .  .  .  .");
        assert_eq!(lines[7], "   2   .  3  .  .  .  .  .  .  .  .  .  .");
        assert_eq!(*lines.last().unwrap(), "colors: 3=#000000");
    }

    #[test]
    fn slice_ranges_collapse_empty_layers() {
        let pal = vec![[0u8; 4]; 256];
        let s = slice(&sample(), &pal, &json!({ "axis": "z", "from_index": 0, "to_index": 5 })).unwrap();
        assert!(s.starts_with("Slices z=0..5"));
        assert!(s.contains("\nz=0\n") && s.contains("\nz=1: empty\n") && s.contains("\nz=2\n") && s.contains("\nz=3: empty\n"));
        assert!(s.contains("\nz=4\n") && s.contains("\nz=5: empty\n"));
        assert!(s.ends_with("colors: 3=#000000 7=#000000 12=#000000"));
        let e = slice(&sample(), &pal, &json!({ "axis": "z", "index": 9 })).unwrap_err().to_string();
        assert!(e.contains("resize_canvas"), "{e}");
    }

    #[test]
    fn projection_shows_top_voxels_and_heights() {
        let pal = vec![[0u8; 4]; 256];
        let s = projection(&sample(), &pal, &json!({ "axis": "z" })).unwrap();
        // The column at (1, 2) shows color 7 from z = 4, not 3 from z = 0.
        let color_row = s.lines().find(|l| l.starts_with("   2 ")).unwrap();
        assert!(color_row.starts_with("   2   .  7  ."), "{color_row}");
        let height_row = s.lines().filter(|l| l.starts_with("   2 ")).nth(1).unwrap();
        assert!(height_row.starts_with("   2   .  4  ."), "{height_row}");
        let from_below = projection(&sample(), &pal, &json!({ "axis": "z", "from_side": "min", "show": "color" })).unwrap();
        assert!(from_below.lines().any(|l| l.starts_with("   2   .  3  .")));
    }

    #[test]
    fn big_maps_downsample() {
        let mut v = Volume::new(IVec3::new(300, 300, 4));
        v.set(IVec3::new(299, 299, 1), 5);
        let pal = vec![[0u8; 4]; 256];
        let s = projection(&v, &pal, &json!({ "show": "height" })).unwrap();
        assert!(s.contains("3x3 block"));
        assert!(s.lines().count() < 110);
        assert!(s.lines().any(|l| l.starts_with(" 297 ") && l.ends_with(" 1")));
    }

    #[test]
    fn rulers_and_thousands() {
        assert_eq!(ruler("x", 0, 11, 1, 3), "   x   0              5             10");
        assert_eq!(ruler("y", 0, 6, 4, 3), "   y   0             20");
        assert_eq!(thousands(5120), "5,120");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_234_567), "1,234,567");
    }

    #[test]
    fn checkpoint_round_trips_the_tree() {
        let mut scene = crate::scene::Scene::empty(IVec3::splat(8));
        scene.volume.set(IVec3::ONE, 9);
        let world = scene.synced_world().into_owned();
        let before = describe(&world);
        assert_eq!(before, "1 part, 1 voxel");
        scene.volume.set(IVec3::ZERO, 9);
        scene.change(|w| *w = world);
        assert_eq!(describe(&scene.world), before);
        // The restore is a tree undo step.
        assert!(scene.undo_tree(false));
        assert_eq!(describe(&scene.world), "1 part, 2 voxels");
    }
}
