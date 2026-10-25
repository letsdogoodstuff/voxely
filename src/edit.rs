//! Voxel editing, modelled on MagicaVoxel's brushes: Attach / Erase /
//! Paint tools combined with Voxel / Face / Box / Geometry / Pattern / Code
//! brushes, axis mode and mirroring, selection (region and marquee), moving and copy/paste of
//! selected voxels, and undo/redo per stroke or whole-model edit.

use std::collections::{HashMap, HashSet, VecDeque};

use glam::{IVec3, Quat, Vec2, Vec3};

use crate::ops;
use crate::scene::{Mask, Volume};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tool {
    Attach,
    Erase,
    Paint,
    /// Drag a screen rectangle to select [M].
    Marquee,
    /// Click to select a region (see `ops::RegionOptions`) [N].
    Region,
    /// Click to delete a region.
    RegionRemove,
    /// Click to recolor a region with the current color.
    RegionReplace,
    /// Drag the selection, or the whole model if nothing is selected.
    Move,
    /// Click a voxel to pick up its color.
    Pick,
}

/// How a selection gesture combines with the current selection:
/// plain replaces, Shift adds, Shift+Alt removes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelMode {
    Replace,
    Add,
    Remove,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Brush {
    /// The voxel brush (sized, cube or sphere); drag to draw [V].
    Voxel,
    /// The connected face region under the cursor [F].
    Face,
    /// Box between press and release [B].
    Box,
    /// Line, square or circle between press and release [L].
    Geo,
    /// Stamp a pattern model [P].
    Pattern,
    /// Box region filled with a voxel shader's output [C].
    Code,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GeoShape {
    Line,
    Square,
    Circle,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MarqueeMode {
    /// Screen rectangle.
    Rect,
    /// Paint the selection with the voxel brush.
    Brush,
    /// 3D box from the pressed voxel to the one under the cursor.
    Box,
}

/// The Move tool's Transform options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum XformMode {
    /// Drag the selection; voxels pushed off the canvas loop around.
    Move,
    /// Drag a face arrow to stretch : Commit [Enter] : Cancel [Esc] : Two-sided [Shift].
    Scale,
    /// Drag a face arrow to repeat or crop : Commit [Enter] : Cancel [Esc] : Two-sided [Shift].
    Wrap,
    /// Drag a ring to turn about its axis : Commit [Enter] : Cancel [Esc] : Snap [Shift].
    Rotate,
}

/// A transform gizmo handle, like MagicaVoxel's: an arrow along an axis
/// (`sign` says which end) for Move, Scale and Wrap, or the ring around an
/// axis for Rotate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Handle {
    Arrow { axis: usize, sign: i32 },
    Ring { axis: usize },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GizmoKind {
    /// Arrows through the box center: drag the selection along an axis.
    Move,
    /// An arrow out of each face of the box: Scale and Wrap.
    Faces,
    /// A ring around each axis through the box center: Rotate.
    Rings,
}

/// The gizmo to show: its kind and the box it works on (world units).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gizmo {
    pub kind: GizmoKind,
    pub lo: Vec3,
    pub hi: Vec3,
}

/// Brush settings from MagicaVoxel's brush option panels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrushOptions {
    /// Voxel brush size [1 - 64].
    pub size: i32,
    /// Sphere (or circle, in 2D) rather than cube.
    pub sphere: bool,
    /// 2D: the footprint lies in the plane of the face under the cursor.
    pub flat: bool,
    /// Face brush: new voxels take the color of the face they grow from.
    pub face_voxel_color: bool,
    /// Face brush: connect through the same color, or any voxel ("Geo").
    pub face_by_color: bool,
    /// Face brush: also connect diagonally (8 rather than 4 neighbors).
    pub face_diagonal: bool,
    pub geo: GeoShape,
    /// Geometry line follows the surface under it.
    pub project: bool,
    /// Square and circle include their center cell ("Fill Center").
    pub fill: bool,
    /// Square and circle have an even width (centered on a cell corner).
    pub even: bool,
    /// Axis mode: extend brushes through the whole canvas along these axes.
    pub axis: [bool; 3],
    pub marquee: MarqueeMode,
    /// Rect select only picks voxels visible from the camera.
    pub visible_only: bool,
    /// Pattern brush paints with the palette color, not its own colors.
    pub pattern_palette: bool,
    /// Pattern brush offset [arrows / PageUp / PageDown, Home resets].
    pub pattern_offset: IVec3,
    pub xform: XformMode,
    /// Wrap: empty cells between the copies along each axis [0 - 256].
    pub wrap_space: [i32; 3],
}

impl Default for BrushOptions {
    fn default() -> Self {
        BrushOptions {
            size: 1,
            sphere: false,
            flat: false,
            face_voxel_color: false,
            face_by_color: true,
            face_diagonal: false,
            geo: GeoShape::Line,
            project: true,
            fill: true,
            even: false,
            axis: [false; 3],
            marquee: MarqueeMode::Rect,
            visible_only: false,
            pattern_palette: false,
            pattern_offset: IVec3::ZERO,
            xform: XformMode::Move,
            wrap_space: [0; 3],
        }
    }
}

/// A model stamped by the Pattern brush.
#[derive(Clone)]
pub struct Pattern {
    pub name: String,
    pub volume: Volume,
}

/// A ray hit: the voxel that was hit (z = -1 for the ground plane) and the
/// face normal it was hit through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pick {
    pub cell: IVec3,
    pub normal: IVec3,
}

impl Pick {
    /// The plane of the face that was hit: (axis, coordinate).
    pub fn plane(&self) -> (usize, f32) {
        let axis = axis_of(self.normal);
        (axis, (self.cell[axis] + (self.normal[axis] > 0) as i32) as f32)
    }

    pub fn on_ground(&self) -> bool {
        self.cell.z < 0
    }
}

/// Cast a ray through the volume (and the ground plane under it).
pub fn pick(v: &Volume, ro: Vec3, rd: Vec3) -> Option<Pick> {
    let dims = v.dims.as_vec3();
    let rd = Vec3::select(rd.abs().cmplt(Vec3::splat(1e-7)), Vec3::splat(1e-7), rd);
    let inv = 1.0 / rd;
    let ta = -ro * inv;
    let tb = (dims - ro) * inv;
    let tnear = ta.min(tb);
    let tfar = ta.max(tb);
    let t0 = tnear.max_element().max(0.0);
    let t1 = tfar.min_element();
    if t0 < t1 {
        let step = IVec3::select(rd.cmpgt(Vec3::ZERO), IVec3::ONE, IVec3::NEG_ONE);
        let mut axis = if t0 > 0.0 { tnear.max_position() } else { rd.abs().max_position() };
        let mut cell = (ro + rd * t0).floor().as_ivec3().clamp(IVec3::ZERO, v.dims - IVec3::ONE);
        loop {
            if v.get(cell) != 0 {
                let mut normal = IVec3::ZERO;
                normal[axis] = -step[axis];
                return Some(Pick { cell, normal });
            }
            let next = (cell + step.max(IVec3::ZERO)).as_vec3();
            let tn = (next - ro) * inv;
            axis = tn.min_position();
            cell[axis] += step[axis];
            if !v.contains(cell) {
                break;
            }
        }
    }
    // The ground plane spans the volume's footprint at z = 0.
    if rd.z < 0.0 && ro.z > 0.0 {
        let p = ro + rd * (-ro.z / rd.z);
        let c = IVec3::new(p.x.floor() as i32, p.y.floor() as i32, -1);
        if c.x >= 0 && c.y >= 0 && c.x < v.dims.x && c.y < v.dims.y {
            return Some(Pick { cell: c, normal: IVec3::Z });
        }
    }
    None
}

/// Cell under the ray on the axis plane `axis = coord` (cell coordinates).
pub fn plane_cell(ro: Vec3, rd: Vec3, axis: usize, coord: i32) -> Option<IVec3> {
    if rd[axis].abs() < 1e-6 {
        return None;
    }
    let t = (coord as f32 + 0.5 - ro[axis]) / rd[axis];
    if t <= 0.0 {
        return None;
    }
    let mut c = (ro + rd * t).floor().as_ivec3();
    c[axis] = coord;
    Some(c)
}

/// Signed offset along `axis` from `p0` to the point on that axis line
/// closest to the ray (None when the ray runs along the axis).
fn axis_offset(p0: Vec3, axis: usize, ro: Vec3, rd: Vec3) -> Option<f32> {
    let mut dir = Vec3::ZERO;
    dir[axis] = 1.0;
    let rd = rd.normalize_or_zero();
    let w = p0 - ro;
    let (b, d, e) = (dir.dot(rd), dir.dot(w), rd.dot(w));
    let denom = 1.0 - b * b;
    (denom >= 1e-6).then(|| (b * e - d) / denom)
}

/// Where the ray crosses the plane `axis = coord` (world units).
pub fn plane_point(ro: Vec3, rd: Vec3, axis: usize, coord: f32) -> Option<Vec3> {
    if rd[axis].abs() < 1e-6 {
        return None;
    }
    let t = (coord - ro[axis]) / rd[axis];
    (t > 0.0).then(|| ro + rd * t)
}

/// The `size` box of `v` at `lo` as a volume of its own (0 outside `v`).
pub fn sub_volume(v: &Volume, lo: IVec3, size: IVec3) -> Volume {
    let mut out = Volume::new(size);
    for p in cuboid(IVec3::ZERO, size - IVec3::ONE) {
        let c = v.get(lo + p);
        if c != 0 {
            out.set(p, c);
        }
    }
    out
}

/// The pattern after (wrapping) `current` in `files` by `step`, matched
/// by file stem; the first or last one when there is no current pattern
/// [3 / 4].
pub fn step_pattern<'a>(files: &[&'a std::path::Path], current: Option<&str>, step: i32) -> Option<&'a std::path::Path> {
    if files.is_empty() {
        return None;
    }
    let n = files.len() as i32;
    let at = current.and_then(|c| files.iter().position(|f| f.file_stem().is_some_and(|s| s.to_string_lossy() == c)));
    let i = match at {
        Some(i) => (i as i32 + step).rem_euclid(n),
        None if step < 0 => n - 1,
        None => 0,
    };
    Some(files[i as usize])
}

fn axis_of(n: IVec3) -> usize {
    n.abs().max_position()
}

/// The cell a tool acts on for a pick.
fn target(tool: Tool, p: Pick) -> Option<IVec3> {
    match tool {
        Tool::Attach => Some(p.cell + p.normal),
        _ if p.on_ground() => None,
        _ => Some(p.cell),
    }
}

/// Connected region of same-colored cells on the picked face plane that
/// are exposed in the normal direction.
fn face_region(v: &Volume, p: Pick, by_color: bool, diagonal: bool) -> Vec<IVec3> {
    const LIMIT: usize = 1 << 20;
    let axis = axis_of(p.normal);
    let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
    let color = v.get(p.cell);
    let inside = |q: IVec3| {
        if p.on_ground() {
            // Ground: the empty floor cells of the bottom layer.
            q.x >= 0 && q.y >= 0 && q.x < v.dims.x && q.y < v.dims.y && v.get(q + p.normal) == 0
        } else {
            let c = v.get(q);
            (if by_color { c == color } else { c != 0 }) && v.get(q + p.normal) == 0
        }
    };
    let mut steps = vec![(1, 0), (-1, 0), (0, 1), (0, -1)];
    if diagonal {
        steps.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)]);
    }
    let mut seen = HashSet::from([p.cell]);
    let mut queue = VecDeque::from([p.cell]);
    let mut out = Vec::new();
    while let Some(q) = queue.pop_front() {
        out.push(q);
        if out.len() >= LIMIT {
            break;
        }
        for &(da, db) in &steps {
            let mut n = q;
            n[a] += da;
            n[b] += db;
            if !seen.contains(&n) && inside(n) {
                seen.insert(n);
                queue.push_back(n);
            }
        }
    }
    out
}

fn line(a: IVec3, b: IVec3) -> Vec<IVec3> {
    let d = b - a;
    let n = d.abs().max_element().max(1);
    (0..=n).map(|i| (a.as_vec3() + d.as_vec3() * (i as f32 / n as f32)).round().as_ivec3()).collect()
}

fn cuboid(a: IVec3, b: IVec3) -> Vec<IVec3> {
    let (lo, hi) = (a.min(b), a.max(b));
    let mut out = Vec::new();
    for z in lo.z..=hi.z {
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                out.push(IVec3::new(x, y, z));
            }
        }
    }
    out
}

struct Stroke {
    /// cell index -> (before, after)
    changes: HashMap<usize, (u8, u8)>,
    /// First target and the plane drags are locked to.
    anchor: IVec3,
    axis: usize,
    current: IVec3,
    /// Normal of the face the stroke started on.
    normal: IVec3,
    /// Camera rays at press and now (for lines projected on the surface).
    ray0: (Vec3, Vec3),
    ray1: (Vec3, Vec3),
    /// Face brush: the region being extruded by dragging.
    extrude: Option<Extrude>,
}

/// Face brush drag [Extrude: Drag]: more layers the further the cursor
/// goes along the face normal.
struct Extrude {
    /// First layer's cells with their colors.
    base: Vec<(IVec3, u8)>,
    /// Center of the pressed face, and the offset along the normal the
    /// press ray was at.
    origin: Vec3,
    t0: f32,
    layers: i32,
}

/// A pending Scale / Wrap / Rotate of the selection (or the whole model),
/// shown in the volume until committed [Enter] or cancelled [Esc].
struct Xform {
    mode: XformMode,
    before: Volume,
    /// The voxels being transformed, in a box of their own at `lo`.
    src: Volume,
    lo: IVec3,
    /// Scale and Wrap: the target box (min corner, size).
    box_lo: IVec3,
    box_size: IVec3,
    /// Rotate: the turn about the source box's center, and the current
    /// drag's angle in degrees.
    rot: Quat,
    angle: f32,
    /// Nothing was selected, so don't select the result.
    whole: bool,
    drag: Option<XformDrag>,
}

/// A drag on one face of the transform box.
struct XformDrag {
    axis: usize,
    /// +1 or -1: which face along `axis` was grabbed.
    sign: i32,
    /// Shift: Scale and Wrap move both faces, Rotate snaps.
    shift: bool,
    /// The grabbed point and its offset along the axis at press.
    origin: Vec3,
    t0: f32,
    box_lo: IVec3,
    box_size: IVec3,
    rot: Quat,
}

/// Rotate turns in whole degrees, or in steps of this many with Shift
/// (measured in MagicaVoxel).
const ROTATE_SNAP_DEG: f32 = 5.0;

impl Xform {
    /// Draw the transformed voxels over `before` into `v`; returns the
    /// cells they cover.
    fn render(&self, v: &mut Volume, space: [i32; 3]) -> Mask {
        *v = self.before.clone();
        for (p, _) in self.src.filled() {
            v.set(self.lo + p, 0);
        }
        let mut mask = Mask::new(v.dims);
        let dims = v.dims;
        let clip = |lo: IVec3, hi: IVec3| cuboid(lo.max(IVec3::ZERO), hi.min(dims - IVec3::ONE));
        let n = self.src.dims;
        let cells: Vec<(IVec3, u8)> = match self.mode {
            XformMode::Scale => clip(self.box_lo, self.box_lo + self.box_size - IVec3::ONE)
                .into_iter()
                .map(|q| {
                    // Nearest source cell to the target cell's center.
                    let rel = q - self.box_lo;
                    (q, self.src.get((rel * 2 + IVec3::ONE) * n / (self.box_size * 2)))
                })
                .collect(),
            XformMode::Wrap => {
                // Copies repeat every size + spacing cells, anchored where
                // the voxels were.
                let period = n + IVec3::from_array(space).max(IVec3::ZERO);
                clip(self.box_lo, self.box_lo + self.box_size - IVec3::ONE)
                    .into_iter()
                    .map(|q| {
                        let off = (q - self.lo).rem_euclid(period);
                        (q, if off.cmplt(n).all() { self.src.get(off) } else { 0 })
                    })
                    .collect()
            }
            XformMode::Rotate => {
                let center = self.lo.as_vec3() + n.as_vec3() * 0.5;
                let half = n.as_vec3() * 0.5;
                let corners = (0..8).map(|k| {
                    let s = Vec3::new(if k & 1 == 0 { -1.0 } else { 1.0 }, if k & 2 == 0 { -1.0 } else { 1.0 }, if k & 4 == 0 { -1.0 } else { 1.0 });
                    center + self.rot * (half * s)
                });
                let (lo, hi) = corners.fold((Vec3::MAX, Vec3::MIN), |(lo, hi), c| (lo.min(c), hi.max(c)));
                let inv = self.rot.inverse();
                clip((lo + 1e-3).floor().as_ivec3(), (hi - 1e-3).floor().as_ivec3())
                    .into_iter()
                    .map(|q| {
                        let s = center + inv * (q.as_vec3() + 0.5 - center);
                        (q, self.src.get((s + 1e-3).floor().as_ivec3() - self.lo))
                    })
                    .collect()
            }
            XformMode::Move => Vec::new(),
        };
        for (q, c) in cells {
            if c != 0 {
                let i = v.index(q);
                v.set_index(i, c);
                mask[i] = true;
            }
        }
        v.refresh_bricks();
        mask
    }
}

/// One undoable edit.
enum Entry {
    /// (cell index, before, after)
    Cells(Vec<(usize, u8, u8)>),
    /// The whole volume before an edit that changed its size; undo swaps.
    Swap(Volume),
}

/// Selected voxels being dragged [Ctrl+drag].
struct Move {
    before: Volume,
    cells: Vec<(IVec3, u8)>,
    anchor: IVec3,
    axis: usize,
    /// Shift: move along the face normal instead of across the face.
    along_normal: bool,
    delta: IVec3,
    /// Moving everything (nothing was selected), so don't select it.
    whole: bool,
    /// Started from a gizmo arrow (the gizmo stays up while dragging), at
    /// this offset along the axis from `anchor`.
    handle: bool,
    t0: f32,
}

pub struct Editor {
    pub tool: Tool,
    pub brush: Brush,
    pub color: u8,
    pub mirror: [bool; 3],
    pub sel_mode: SelMode,
    pub region: ops::RegionOptions,
    pub opts: BrushOptions,
    pub pattern: Option<Pattern>,
    /// The Code brush's shader output for its box, and the box's corner.
    pub code: Option<(IVec3, Volume)>,
    /// Set when the selection changes, so the view can refresh.
    pub sel_dirty: bool,
    undo: Vec<Entry>,
    redo: Vec<Entry>,
    stroke: Option<Stroke>,
    moving: Option<Move>,
    xform: Option<Xform>,
    /// A Code brush result, selected and still movable with the gizmo
    /// arrows; the next stroke lets go of it.
    floating: bool,
    /// Selected cells per volume cell; empty when nothing is selected.
    sel: Mask,
    sel_count: usize,
    /// Copied voxels at their original positions.
    clipboard: Vec<(IVec3, u8)>,
}

impl Editor {
    pub fn new(color: u8) -> Self {
        Editor {
            tool: Tool::Attach,
            brush: Brush::Voxel,
            color,
            mirror: [false; 3],
            sel_mode: SelMode::Replace,
            region: ops::RegionOptions::default(),
            opts: BrushOptions::default(),
            pattern: None,
            code: None,
            sel_dirty: false,
            undo: Vec::new(),
            redo: Vec::new(),
            stroke: None,
            moving: None,
            xform: None,
            floating: false,
            sel: Mask::new(IVec3::ZERO),
            sel_count: 0,
            clipboard: Vec::new(),
        }
    }

    pub fn active(&self) -> bool {
        self.stroke.is_some() || self.moving.is_some() || self.xform.as_ref().is_some_and(|x| x.drag.is_some())
    }

    /// Forget history and selection (after opening another model).
    pub fn reset(&mut self) {
        let (tool, brush, color, mirror, region, opts) = (self.tool, self.brush, self.color, self.mirror, self.region, self.opts);
        let (clipboard, pattern) = (std::mem::take(&mut self.clipboard), self.pattern.take());
        *self = Editor { tool, brush, mirror, region, opts, clipboard, pattern, ..Editor::new(color) };
        self.sel_dirty = true;
    }

    // ---- selection ----

    pub fn selection(&self) -> Option<&Mask> {
        (self.sel_count > 0).then_some(&self.sel)
    }

    pub fn selected_count(&self) -> usize {
        self.sel_count
    }

    pub fn has_clipboard(&self) -> bool {
        !self.clipboard.is_empty()
    }

    fn set_selection(&mut self, v: &Volume, mask: Mask) {
        self.sel_count = mask.filled().filter(|&(i, _)| v.cells[i] != 0).count();
        self.sel = if self.sel_count > 0 { mask } else { Mask::new(IVec3::ZERO) };
        self.sel_dirty = true;
    }

    /// Combine solid cells `cells` with the selection using `mode`.
    pub fn select(&mut self, v: &Volume, cells: impl IntoIterator<Item = usize>, mode: SelMode) {
        self.floating = false;
        let mut mask = match (mode, self.sel.dims() == v.dims) {
            (SelMode::Replace, _) | (_, false) => Mask::new(v.dims),
            _ => std::mem::replace(&mut self.sel, Mask::new(IVec3::ZERO)),
        };
        for i in cells {
            if v.cells[i] != 0 {
                mask.set(i, mode != SelMode::Remove);
            }
        }
        self.set_selection(v, mask);
    }

    pub fn select_all(&mut self, v: &Volume) {
        self.floating = false;
        let mut mask = Mask::new(v.dims);
        for (i, _) in v.cells.filled() {
            mask.set(i, true);
        }
        self.set_selection(v, mask);
    }

    pub fn select_none(&mut self) {
        self.floating = false;
        if self.sel_count > 0 {
            self.sel = Mask::new(IVec3::ZERO);
            self.sel_count = 0;
            self.sel_dirty = true;
        }
    }

    pub fn select_invert(&mut self, v: &Volume) {
        self.floating = false;
        let had = self.selection().cloned();
        let mut mask = Mask::new(v.dims);
        for (i, _) in v.cells.filled() {
            if !had.as_ref().is_some_and(|s| s[i]) {
                mask.set(i, true);
            }
        }
        self.set_selection(v, mask);
    }

    // ---- whole-model edits and history ----

    /// Replace the volume with `new` as one undoable edit. Returns whether
    /// anything changed.
    pub fn commit(&mut self, v: &mut Volume, new: Volume) -> bool {
        self.commit_xform(v);
        if new.dims == v.dims {
            let diff = v.cells.diff(&new.cells);
            if diff.is_empty() {
                return false;
            }
            *v = new;
            self.undo.push(Entry::Cells(diff));
        } else {
            let old = std::mem::replace(v, new);
            self.undo.push(Entry::Swap(old));
        }
        self.redo.clear();
        self.fit_selection(v);
        true
    }

    /// `commit`, then make `sel` the selection when given (an edit that
    /// worked on the selected voxels selects its result).
    pub fn commit_select(&mut self, v: &mut Volume, new: Volume, sel: Option<Mask>) -> bool {
        let changed = self.commit(v, new);
        if let Some(mask) = sel.filter(|m| m.dims() == v.dims) {
            self.set_selection(v, mask);
        }
        changed
    }

    /// Drop the selection if it no longer matches the volume.
    fn fit_selection(&mut self, v: &Volume) {
        if self.sel_count > 0 {
            if self.sel.dims() != v.dims {
                self.select_none();
            } else {
                let mask = std::mem::replace(&mut self.sel, Mask::new(IVec3::ZERO));
                self.set_selection(v, mask);
            }
        }
    }

    fn step(v: &mut Volume, entry: Entry, forward: bool) -> Entry {
        match entry {
            Entry::Cells(cells) => {
                for &(i, old, new) in &cells {
                    v.set_index(i, if forward { new } else { old });
                }
                Entry::Cells(cells)
            }
            Entry::Swap(mut other) => {
                std::mem::swap(v, &mut other);
                Entry::Swap(other)
            }
        }
    }

    // ---- clipboard ----

    /// Copy the selection, or the whole model when nothing is selected.
    pub fn copy(&mut self, v: &Volume) {
        let sel = self.selection();
        self.clipboard = v
            .cells
            .iter()
            .enumerate()
            .filter(|&(i, &c)| c != 0 && sel.is_none_or(|s| s[i]))
            .map(|(i, &c)| (v.position(i), c))
            .collect();
    }

    pub fn cut(&mut self, v: &mut Volume) -> bool {
        self.copy(v);
        let new = ops::clear(v, self.selection());
        self.select_none();
        self.commit(v, new)
    }

    /// Paste where the voxels were copied from and select them, ready to
    /// drag into place.
    pub fn paste(&mut self, v: &mut Volume) -> bool {
        let mut new = v.clone();
        let mut mask = Mask::new(v.dims);
        for &(p, c) in &self.clipboard {
            if v.contains(p) {
                let i = v.index(p);
                new.set_index(i, c);
                mask[i] = true;
            }
        }
        let changed = self.commit(v, new);
        self.set_selection(v, mask);
        changed
    }

    /// Make the selection (or the whole model) the Pattern brush's pattern.
    pub fn pattern_from_selection(&mut self, v: &Volume) -> bool {
        let sel = self.selection();
        let cells: Vec<(IVec3, u8)> =
            v.cells.iter().enumerate().filter(|&(i, &c)| c != 0 && sel.is_none_or(|s| s[i])).map(|(i, &c)| (v.position(i), c)).collect();
        let Some(lo) = cells.iter().map(|c| c.0).reduce(IVec3::min) else { return false };
        let hi = cells.iter().map(|c| c.0).fold(lo, IVec3::max);
        let mut volume = Volume::new(hi - lo + IVec3::ONE);
        for (p, c) in cells {
            volume.set(p - lo, c);
        }
        self.pattern = Some(Pattern { name: "selection".into(), volume });
        true
    }

    /// Turn the pattern 90 degrees about `axis` [-/+ and 9/0].
    pub fn rotate_pattern(&mut self, axis: usize, times: i32) {
        if let Some(p) = self.pattern.as_mut() {
            for _ in 0..times.rem_euclid(4) {
                p.volume = ops::rotate(&p.volume, axis);
            }
        }
    }

    // ---- moving the selection ----

    /// Ctrl+press (or the Move tool): start dragging the selected voxels,
    /// or all of them when nothing is selected, along the plane of the face
    /// under the cursor.
    pub fn begin_move(&mut self, v: &Volume, hit: Pick, along_normal: bool) -> bool {
        self.commit_xform(v);
        let sel = self.selection();
        let cells: Vec<(IVec3, u8)> =
            v.cells.iter().enumerate().filter(|&(i, &c)| c != 0 && sel.is_none_or(|s| s[i])).map(|(i, &c)| (v.position(i), c)).collect();
        if cells.is_empty() {
            return false;
        }
        let whole = sel.is_none();
        self.moving = Some(Move { before: v.clone(), cells, anchor: hit.cell, axis: axis_of(hit.normal), along_normal, delta: IVec3::ZERO, whole, handle: false, t0: 0.0 });
        true
    }

    fn drag_move(&mut self, v: &mut Volume, ro: Vec3, rd: Vec3) -> bool {
        let Some(m) = self.moving.as_mut() else { return false };
        let delta = if m.along_normal {
            // Closest point on the normal line through the grabbed voxel.
            let Some(t) = axis_offset(m.anchor.as_vec3() + 0.5, m.axis, ro, rd) else { return false };
            let mut delta = IVec3::ZERO;
            delta[m.axis] = (t - m.t0).round() as i32;
            delta
        } else {
            let Some(cell) = plane_cell(ro, rd, m.axis, m.anchor[m.axis]) else { return false };
            cell - m.anchor
        };
        if delta == m.delta {
            return false;
        }
        m.delta = delta;
        *v = m.before.clone();
        for &(p, _) in &m.cells {
            v.set(p, 0);
        }
        // Move/Loop: voxels pushed off the canvas come back on the other side.
        let mut mask = Mask::new(v.dims);
        for &(p, c) in &m.cells {
            let i = v.index((p + delta).rem_euclid(v.dims));
            v.set_index(i, c);
            mask[i] = true;
        }
        v.refresh_bricks();
        if !m.whole {
            self.set_selection(v, mask);
        }
        true
    }

    fn end_move(&mut self, v: &mut Volume) -> bool {
        let Some(m) = self.moving.take() else { return false };
        let moved = std::mem::replace(v, m.before);
        let mask = std::mem::replace(&mut self.sel, Mask::new(IVec3::ZERO));
        let floating = self.floating;
        let changed = self.commit(v, moved);
        if !m.whole {
            self.set_selection(v, mask);
        }
        self.floating = floating;
        if changed && floating {
            // Placing a Code brush result: one undo step with the stroke.
            self.merge_last_two();
        }
        changed
    }

    /// Fold the newest undo step into the one before it.
    fn merge_last_two(&mut self) {
        let n = self.undo.len();
        if n < 2 || !matches!((&self.undo[n - 2], &self.undo[n - 1]), (Entry::Cells(_), Entry::Cells(_))) {
            return;
        }
        let (Some(Entry::Cells(b)), Some(Entry::Cells(a))) = (self.undo.pop(), self.undo.pop()) else { return };
        let mut cells: HashMap<usize, (u8, u8)> = a.into_iter().map(|(i, x, y)| (i, (x, y))).collect();
        for (i, x, y) in b {
            cells.entry(i).and_modify(|e| e.1 = y).or_insert((x, y));
        }
        let mut entries: Vec<(usize, u8, u8)> = cells.into_iter().filter(|(_, (x, y))| x != y).map(|(i, (x, y))| (i, x, y)).collect();
        entries.sort_unstable_by_key(|e| e.0);
        self.undo.push(Entry::Cells(entries));
    }

    // ---- Scale / Wrap / Rotate ----

    /// The gizmo for the current tool, like MagicaVoxel's: Scale and Wrap
    /// show arrows on the faces of the selection's box (or the model's),
    /// Rotate rings round its center, and Marquee select (or a Code brush
    /// result being placed) arrows to move the selection.
    pub fn gizmo(&self, v: &Volume) -> Option<Gizmo> {
        let moving = self.moving.as_ref().is_some_and(|m| m.handle);
        if (self.tool == Tool::Marquee || (self.floating && self.brush == Brush::Code)) && self.stroke.is_none() && (self.moving.is_none() || moving) {
            let (lo, hi) = self.solid_bounds(v, true)?;
            return Some(Gizmo { kind: GizmoKind::Move, lo, hi });
        }
        let kind = match (self.tool, self.opts.xform) {
            (Tool::Move, XformMode::Scale | XformMode::Wrap) => GizmoKind::Faces,
            (Tool::Move, XformMode::Rotate) => GizmoKind::Rings,
            _ => return None,
        };
        let (lo, hi) = match &self.xform {
            Some(x) if kind == GizmoKind::Faces => (x.box_lo.as_vec3(), (x.box_lo + x.box_size).as_vec3()),
            Some(x) => (x.lo.as_vec3(), (x.lo + x.src.dims).as_vec3()),
            None => self.solid_bounds(v, false)?,
        };
        Some(Gizmo { kind, lo, hi })
    }

    /// Box around the selected voxels (or all of them, unless `selected`).
    fn solid_bounds(&self, v: &Volume, selected: bool) -> Option<(Vec3, Vec3)> {
        let sel = self.selection();
        if selected && sel.is_none() {
            return None;
        }
        let mut cells = v.cells.filled().map(|(i, _)| i).filter(|&i| sel.is_none_or(|s| s[i])).map(|i| v.position(i));
        let first = cells.next()?;
        let (lo, hi) = cells.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p)));
        Some((lo.as_vec3(), (hi + IVec3::ONE).as_vec3()))
    }

    /// Press on a gizmo handle (see `gizmo`). Move arrows drag the selection
    /// along their axis; Scale and Wrap arrows move their face of the box
    /// (both faces with Shift); Rotate rings turn about their axis (in 5
    /// degree steps with Shift). Scale, Wrap and Rotate stay pending across
    /// drags until committed [Enter] or cancelled [Esc].
    pub fn begin_handle(&mut self, v: &Volume, handle: Handle, ro: Vec3, rd: Vec3, shift: bool) -> bool {
        let Some(g) = self.gizmo(v) else { return false };
        if g.kind == GizmoKind::Move {
            let Handle::Arrow { axis, .. } = handle else { return false };
            let center = ((g.lo + g.hi) * 0.5).floor().as_ivec3();
            let hit = Pick { cell: center, normal: IVec3::ZERO };
            let floating = self.floating;
            if !self.begin_move(v, Pick { normal: [IVec3::X, IVec3::Y, IVec3::Z][axis], ..hit }, true) {
                return false;
            }
            self.floating = floating;
            if let Some(m) = self.moving.as_mut() {
                m.handle = true;
                m.t0 = axis_offset(center.as_vec3() + 0.5, axis, ro, rd).unwrap_or(0.0);
            }
            return true;
        }
        let mode = self.opts.xform;
        if self.xform.as_ref().is_some_and(|x| x.mode != mode) {
            self.commit_xform(v);
        }
        if self.xform.is_none() {
            let sel = self.selection();
            let cells: Vec<(IVec3, u8)> = v.filled().filter(|&(p, _)| sel.is_none_or(|s| s[v.index(p)])).collect();
            let Some(lo) = cells.iter().map(|c| c.0).reduce(IVec3::min) else { return false };
            let hi = cells.iter().map(|c| c.0).fold(lo, IVec3::max);
            let mut src = Volume::new(hi - lo + IVec3::ONE);
            for (p, c) in cells {
                src.set(p - lo, c);
            }
            let size = src.dims;
            self.xform = Some(Xform { mode, before: v.clone(), src, lo, box_lo: lo, box_size: size, rot: Quat::IDENTITY, angle: 0.0, whole: sel.is_none(), drag: None });
        }
        let x = self.xform.as_mut().unwrap();
        let (axis, sign, origin) = match handle {
            Handle::Arrow { axis, sign } if mode != XformMode::Rotate => {
                let (lo, size) = (x.box_lo.as_vec3(), x.box_size.as_vec3());
                let mut origin = lo + size * 0.5;
                origin[axis] = if sign > 0 { lo[axis] + size[axis] } else { lo[axis] };
                (axis, sign.signum(), origin)
            }
            Handle::Ring { axis } if mode == XformMode::Rotate => {
                // The point grabbed on the ring's plane.
                let center = x.lo.as_vec3() + x.src.dims.as_vec3() * 0.5;
                let Some(p) = plane_point(ro, rd, axis, center[axis]) else { return false };
                (axis, 1, p)
            }
            _ => return false,
        };
        let t0 = axis_offset(origin, axis, ro, rd).unwrap_or(0.0);
        x.drag = Some(XformDrag { axis, sign, shift, origin, t0, box_lo: x.box_lo, box_size: x.box_size, rot: x.rot });
        true
    }

    fn drag_xform(&mut self, v: &mut Volume, ro: Vec3, rd: Vec3) -> bool {
        let space = self.opts.wrap_space;
        let Some(x) = self.xform.as_mut() else { return false };
        let Some(d) = x.drag.as_ref() else { return false };
        let a = d.axis;
        if x.mode == XformMode::Rotate {
            let center = x.lo.as_vec3() + x.src.dims.as_vec3() * 0.5;
            let Some(p) = plane_point(ro, rd, a, d.origin[a]) else { return false };
            let (u, w) = ((a + 1) % 3, (a + 2) % 3);
            let (p0, p1) = (d.origin - center, p - center);
            if p0.length_squared() < 1e-6 || p1.length_squared() < 1e-6 {
                return false;
            }
            let deg = (p0[u] * p1[w] - p0[w] * p1[u]).atan2(p0[u] * p1[u] + p0[w] * p1[w]).to_degrees();
            let deg = if d.shift { (deg / ROTATE_SNAP_DEG).trunc() * ROTATE_SNAP_DEG } else { deg.round() };
            x.angle = deg;
            let angle = deg.to_radians();
            let mut dir = Vec3::ZERO;
            dir[a] = 1.0;
            let rot = Quat::from_axis_angle(dir, angle) * d.rot;
            if rot.dot(x.rot).abs() > 1.0 - 1e-7 {
                return false;
            }
            x.rot = rot;
        } else {
            let Some(t) = axis_offset(d.origin, a, ro, rd) else { return false };
            let size0 = d.box_size[a];
            let mut o = (t - d.t0).round() as i32 * d.sign;
            let (mut lo, mut size) = (d.box_lo, d.box_size);
            if d.shift {
                // Two-sided: both faces move, keeping at least one cell.
                o = o.max(-((size0 - 1) / 2));
                lo[a] -= o;
                size[a] += 2 * o;
            } else {
                o = o.max(1 - size0);
                if d.sign < 0 {
                    lo[a] -= o;
                }
                size[a] += o;
            }
            if (lo, size) == (x.box_lo, x.box_size) {
                return false;
            }
            (x.box_lo, x.box_size) = (lo, size);
        }
        let mask = x.render(v, space);
        if !x.whole {
            self.set_selection(v, mask);
        }
        true
    }

    /// Status bar text while a gizmo is up, like MagicaVoxel's: the box
    /// size for Scale and Wrap, the angle for Rotate.
    pub fn xform_status(&self, v: &Volume) -> Option<String> {
        let g = self.gizmo(v)?;
        let s = (g.hi - g.lo).as_ivec3();
        match g.kind {
            GizmoKind::Faces => Some(format!("x.{} y.{} z.{}", s.x, s.y, s.z)),
            GizmoKind::Rings => Some(format!("angle:{}", self.xform.as_ref().map_or(0.0, |x| x.angle))),
            GizmoKind::Move => None,
        }
    }

    /// A Scale, Wrap or Rotate is waiting for Enter or Esc.
    pub fn xform_pending(&self) -> bool {
        self.xform.is_some()
    }

    /// Keep the pending transform as one undoable edit [Enter]. `v` holds
    /// it already. Returns whether anything changed.
    pub fn commit_xform(&mut self, v: &Volume) -> bool {
        let Some(x) = self.xform.take() else { return false };
        let diff = x.before.cells.diff(&v.cells);
        if diff.is_empty() {
            return false;
        }
        self.undo.push(Entry::Cells(diff));
        self.redo.clear();
        true
    }

    /// Drop the pending transform, putting the voxels back [Esc].
    pub fn cancel_xform(&mut self, v: &mut Volume) -> bool {
        let Some(x) = self.xform.take() else { return false };
        *v = x.before;
        if !x.whole {
            let cells: Vec<usize> = x.src.filled().map(|(p, _)| v.index(x.lo + p)).collect();
            self.select(v, cells, SelMode::Replace);
        }
        true
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty() || self.xform.is_some()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    fn mirrored(&self, v: &Volume, p: IVec3) -> Vec<IVec3> {
        let mut out = vec![p];
        for axis in 0..3 {
            if self.mirror[axis] {
                let flipped: Vec<IVec3> = out
                    .iter()
                    .map(|q| {
                        let mut m = *q;
                        m[axis] = v.dims[axis] - 1 - q[axis];
                        m
                    })
                    .collect();
                out.extend(flipped);
            }
        }
        out
    }

    /// The voxel brush's footprint at `c` (size, cube or sphere, 2D or 3D).
    /// When attaching in 3D the footprint rests on the surface instead of
    /// sinking into it.
    fn stamp(&self, c: IVec3, normal: IVec3) -> Vec<IVec3> {
        let o = &self.opts;
        let n = o.size.max(1);
        if n == 1 {
            return vec![c];
        }
        let axis = axis_of(normal);
        let (lo, hi) = (-(n - 1) / 2, n / 2);
        let center = (lo + hi) as f32 * 0.5;
        let mut out = Vec::new();
        for z in lo..=hi {
            for y in lo..=hi {
                for x in lo..=hi {
                    let mut d = IVec3::new(x, y, z);
                    if o.flat && d[axis] != 0 {
                        continue;
                    }
                    if o.sphere {
                        let mut q = d.as_vec3() - center;
                        if o.flat {
                            q[axis] = 0.0;
                        }
                        if q.length() > n as f32 * 0.5 {
                            continue;
                        }
                    }
                    if self.tool == Tool::Attach && !o.flat && normal != IVec3::ZERO {
                        d[axis] += normal[axis] * -lo;
                    }
                    out.push(c + d);
                }
            }
        }
        out
    }

    /// Extend cells through the whole canvas along the Axis-mode axes.
    fn lofted(&self, v: &Volume, cells: Vec<IVec3>) -> Vec<IVec3> {
        let mut out = cells;
        for axis in 0..3 {
            if self.opts.axis[axis] {
                let mut seen = HashSet::new();
                out = out
                    .iter()
                    .flat_map(|&c| {
                        (0..v.dims[axis]).map(move |k| {
                            let mut q = c;
                            q[axis] = k;
                            q
                        })
                    })
                    .filter(|q| seen.insert(*q))
                    .collect();
            }
        }
        out
    }

    /// Cells after the voxel stamp, Axis mode and mirroring, with colors.
    fn expand(&self, v: &Volume, cells: &[(IVec3, u8)], normal: IVec3) -> Vec<(IVec3, u8)> {
        let mut out = Vec::new();
        for &(c, color) in cells {
            let stamped = if self.brush == Brush::Voxel || (self.tool == Tool::Marquee && self.opts.marquee == MarqueeMode::Brush) {
                self.stamp(c, normal)
            } else {
                vec![c]
            };
            for q in self.lofted(v, stamped) {
                out.extend(self.mirrored(v, q).into_iter().map(|m| (m, color)));
            }
        }
        out
    }

    /// Apply the tool to cells in the current color; returns whether
    /// anything changed.
    fn apply(&mut self, v: &mut Volume, cells: &[IVec3]) -> bool {
        let color = self.color;
        let colored: Vec<(IVec3, u8)> = cells.iter().map(|&c| (c, color)).collect();
        self.apply_colored(v, &colored)
    }

    /// Apply the tool to cells, attaching or painting each with its own
    /// color. Marquee strokes select instead.
    fn apply_colored(&mut self, v: &mut Volume, cells: &[(IVec3, u8)]) -> bool {
        let normal = self.stroke.as_ref().map_or(IVec3::Z, |s| s.normal);
        let all = self.expand(v, cells, normal);
        if self.tool == Tool::Marquee {
            let idx: Vec<usize> = all.iter().filter(|(c, _)| v.contains(*c)).map(|(c, _)| v.index(*c)).collect();
            let mode = if self.sel_mode == SelMode::Remove { SelMode::Remove } else { SelMode::Add };
            self.select(v, idx, mode);
            return false;
        }
        let tool = self.tool;
        // Like MagicaVoxel, brushes leave selected voxels alone (region
        // tools and the Edit panel work on them instead).
        let sel = if self.sel_count > 0 && tool != Tool::Attach { Some(&self.sel) } else { None };
        let Some(stroke) = self.stroke.as_mut() else { return false };
        let mut changed = false;
        for (c, color) in all {
            if !v.contains(c) {
                continue;
            }
            let i = v.index(c);
            if sel.is_some_and(|s| s[i]) {
                continue;
            }
            let old = v.cells[i];
            let new = match tool {
                Tool::Attach if old == 0 => color,
                Tool::Erase => 0,
                Tool::Paint if old != 0 => color,
                _ => old,
            };
            if new != old {
                v.set_index(i, new);
                stroke.changes.entry(i).or_insert((old, new)).1 = new;
                changed = true;
            }
        }
        changed
    }

    /// Cells of the Face brush region, with the color each new voxel takes.
    fn face_cells(&self, v: &Volume, hit: Pick) -> Vec<(IVec3, u8)> {
        let region = face_region(v, hit, self.opts.face_by_color, self.opts.face_diagonal);
        match self.tool {
            Tool::Attach => region
                .iter()
                .map(|&c| (c + hit.normal, if self.opts.face_voxel_color && !hit.on_ground() { v.get(c) } else { self.color }))
                .collect(),
            _ if hit.on_ground() => Vec::new(),
            _ => region.into_iter().map(|c| (c, self.color)).collect(),
        }
    }

    /// The Pattern brush's voxels placed at target `t`, extending away from
    /// the surface along `normal` and centered in the other two axes.
    fn pattern_cells(&self, t: IVec3, normal: IVec3) -> Vec<(IVec3, u8)> {
        let Some(p) = self.pattern.as_ref() else { return Vec::new() };
        let axis = axis_of(normal);
        let mut origin = t - p.volume.dims / 2;
        let outward = if self.tool == Tool::Attach { normal[axis] } else { -normal[axis] };
        origin[axis] = if outward >= 0 { t[axis] } else { t[axis] - p.volume.dims[axis] + 1 };
        origin += self.opts.pattern_offset;
        p.volume
            .cells
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != 0)
            .map(|(i, &c)| (origin + p.volume.position(i), if self.opts.pattern_palette { self.color } else { c }))
            .collect()
    }

    /// Geometry-mode shape between the anchor and current cell on the
    /// stroke plane.
    fn geo_cells(&self, v: &Volume, s: &Stroke) -> Vec<IVec3> {
        let o = &self.opts;
        let (a, b, axis) = (s.anchor, s.current, s.axis);
        let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
        let at = |du: i32, dw: i32| {
            let mut q = a;
            q[u] += du;
            q[w] += dw;
            q
        };
        match o.geo {
            GeoShape::Line if o.project => {
                // Follow the surface: cast rays between the press and
                // release rays.
                let ((ro0, rd0), (ro1, rd1)) = (s.ray0, s.ray1);
                let steps = ((b - a).abs().max_element() * 3).clamp(2, 4096);
                let mut out = Vec::new();
                for k in 0..=steps {
                    let f = k as f32 / steps as f32;
                    let (ro, rd) = (ro0.lerp(ro1, f), rd0.lerp(rd1, f).normalize());
                    if let Some(t) = pick(v, ro, rd).and_then(|h| target(self.tool, h)) {
                        if out.last() != Some(&t) {
                            out.push(t);
                        }
                    }
                }
                out
            }
            GeoShape::Line => line(a, b),
            // Like MagicaVoxel: centered on the press cell; Even centers on
            // its lower corner instead. Fill off leaves out only the center
            // cell (the 2x2 around the corner when Even).
            GeoShape::Square | GeoShape::Circle => {
                let (du, dw) = (b[u] - a[u], b[w] - a[w]);
                let even = o.even as i32;
                let center = |x: i32, y: i32| (x == 0 || x == -even) && (y == 0 || y == -even);
                // Square: half side = the larger drag offset. Circle: radius
                // = the drag length rounded up, less half a cell, measured
                // between cell centers.
                let h = du.abs().max(dw.abs());
                let n = ((((du * du + dw * dw) as f32).sqrt() - 1e-4).ceil() as i32).max(1);
                let r = n as f32 - 0.5 + 0.5 * even as f32;
                let k = if o.geo == GeoShape::Square { h } else { n };
                let c = 0.5 * even as f32;
                let mut out = Vec::new();
                for y in -k - even..=k {
                    for x in -k - even..=k {
                        let inside = o.geo == GeoShape::Square || Vec2::new(x as f32 + c, y as f32 + c).length() <= r + 1e-4;
                        if inside && (o.fill || !center(x, y)) {
                            out.push(at(x, y));
                        }
                    }
                }
                out
            }
        }
    }

    /// Mouse press with the camera ray. Returns whether the volume changed.
    pub fn begin(&mut self, v: &mut Volume, hit: Pick, ro: Vec3, rd: Vec3) -> bool {
        self.commit_xform(v);
        if std::mem::take(&mut self.floating) {
            self.select_none();
        }
        match self.tool {
            Tool::Region => {
                if !hit.on_ground() {
                    let cells = ops::region(v, hit.cell, self.region);
                    self.select(v, cells, self.sel_mode);
                } else if self.sel_mode == SelMode::Replace {
                    self.select_none();
                }
                return false;
            }
            Tool::RegionRemove | Tool::RegionReplace => {
                if hit.on_ground() {
                    return false;
                }
                let color = if self.tool == Tool::RegionRemove { 0 } else { self.color };
                let mut new = v.clone();
                for i in ops::region(v, hit.cell, self.region) {
                    new.set_index(i, color);
                }
                return self.commit(v, new);
            }
            Tool::Pick => {
                if !hit.on_ground() {
                    self.color = v.get(hit.cell);
                }
                return false;
            }
            Tool::Marquee if self.opts.marquee == MarqueeMode::Rect => return false,
            Tool::Move => return false,
            _ => {}
        }
        // Marquee brush/box select acts on voxels like erase/paint would.
        let as_tool = if self.tool == Tool::Marquee { Tool::Paint } else { self.tool };
        let Some(t) = target(as_tool, hit) else { return false };
        if self.tool == Tool::Marquee && self.sel_mode == SelMode::Replace {
            self.select_none();
        }
        let axis = axis_of(hit.normal);
        self.stroke = Some(Stroke { changes: HashMap::new(), anchor: t, axis, current: t, normal: hit.normal, ray0: (ro, rd), ray1: (ro, rd), extrude: None });
        if self.tool == Tool::Marquee {
            return if self.opts.marquee == MarqueeMode::Brush { self.apply(v, &[t]) } else { false };
        }
        match self.brush {
            Brush::Voxel => self.apply(v, &[t]),
            Brush::Face => {
                let cells = self.face_cells(v, hit);
                let origin = hit.cell.as_vec3() + 0.5 + hit.normal.as_vec3() * 0.5;
                let t0 = axis_offset(origin, axis, ro, rd).unwrap_or(0.0);
                let changed = self.apply_colored(v, &cells);
                if matches!(self.tool, Tool::Attach | Tool::Erase) {
                    self.stroke.as_mut().unwrap().extrude = Some(Extrude { base: cells, origin, t0, layers: 1 });
                }
                changed
            }
            Brush::Pattern => {
                let cells = self.pattern_cells(t, hit.normal);
                self.apply_colored(v, &cells)
            }
            // Applied on release; the overlay shows the preview meanwhile.
            Brush::Box | Brush::Geo | Brush::Code => false,
        }
    }

    /// Mouse drag with the camera ray. Returns whether the volume changed.
    pub fn drag(&mut self, v: &mut Volume, ro: Vec3, rd: Vec3, hit: Option<Pick>) -> bool {
        if self.moving.is_some() {
            return self.drag_move(v, ro, rd);
        }
        if self.xform.as_ref().is_some_and(|x| x.drag.is_some()) {
            return self.drag_xform(v, ro, rd);
        }
        let Some(s) = self.stroke.as_mut() else { return false };
        s.ray1 = (ro, rd);
        if s.extrude.is_some() {
            return self.drag_extrude(v, ro, rd);
        }
        let (anchor, axis) = (s.anchor, s.axis);
        let marquee = (self.tool == Tool::Marquee).then_some(self.opts.marquee);
        // Attach draws on the plane of the first voxel; erase, paint and
        // brush select follow whatever surface is under the cursor. Like
        // MagicaVoxel, the Box and Code boxes span from the pressed cell to
        // the one under the cursor (in depth too), or stay on the press
        // plane over empty space; so does Box select.
        let planar = self.brush == Brush::Geo || (self.brush == Brush::Voxel && self.tool == Tool::Attach);
        let boxed = matches!(self.brush, Brush::Box | Brush::Code);
        let cell = match marquee {
            Some(MarqueeMode::Box) => hit.and_then(|h| target(Tool::Paint, h)).or_else(|| plane_cell(ro, rd, axis, anchor[axis])),
            Some(_) => hit.and_then(|h| target(Tool::Paint, h)),
            None if planar => plane_cell(ro, rd, axis, anchor[axis]),
            None if boxed => hit.and_then(|h| target(self.tool, h)).or_else(|| plane_cell(ro, rd, axis, anchor[axis])),
            None => hit.and_then(|h| target(self.tool, h)),
        };
        let Some(cell) = cell else { return false };
        let cell = cell.clamp(IVec3::ZERO, v.dims - IVec3::ONE);
        let s = self.stroke.as_mut().unwrap();
        if cell == s.current {
            return false;
        }
        let prev = s.current;
        s.current = cell;
        if marquee == Some(MarqueeMode::Brush) {
            return self.apply(v, &line(prev, cell));
        }
        match self.brush {
            // Fill the gap from the previous cell so fast drags stay connected.
            Brush::Voxel if marquee.is_none() => self.apply(v, &line(prev, cell)),
            _ => false,
        }
    }

    /// Face brush drag: attach (or erase) as many layers of the region as
    /// the cursor is away from the face along its normal, at least one.
    fn drag_extrude(&mut self, v: &mut Volume, ro: Vec3, rd: Vec3) -> bool {
        let s = self.stroke.as_mut().unwrap();
        let (axis, normal) = (s.axis, s.normal);
        let x = s.extrude.as_ref().unwrap();
        let Some(t) = axis_offset(x.origin, axis, ro, rd) else { return false };
        // Layers = how far the cursor is out along the face normal, like
        // MagicaVoxel (Erase digs that many in); dragging inward keeps one.
        let out = (t - x.t0) * normal[axis].signum() as f32;
        let layers = (out.round() as i32).clamp(1, v.dims[axis].max(1));
        if layers == x.layers {
            return false;
        }
        // Redo the stroke from the volume before it.
        for (i, (before, _)) in s.changes.drain() {
            v.set_index(i, before);
        }
        let x = s.extrude.as_mut().unwrap();
        x.layers = layers;
        let step = if self.tool == Tool::Attach { normal } else { -normal };
        let cells: Vec<(IVec3, u8)> = (0..layers).flat_map(|k| x.base.iter().map(move |&(c, color)| (c + step * k, color))).collect();
        self.apply_colored(v, &cells);
        v.refresh_bricks();
        true
    }

    /// The Code brush's box so far, clipped to the canvas, as (min corner,
    /// size). Like MagicaVoxel, the shader runs with this box as its whole
    /// volume; put the output in `code` before release.
    pub fn code_box(&self, v: &Volume) -> Option<(IVec3, IVec3)> {
        let s = self.stroke.as_ref().filter(|_| self.brush == Brush::Code && self.tool != Tool::Marquee)?;
        let top = v.dims - IVec3::ONE;
        let (lo, hi) = (s.anchor.min(s.current).max(IVec3::ZERO), s.anchor.max(s.current).min(top));
        hi.cmpge(lo).all().then(|| (lo, hi - lo + IVec3::ONE))
    }

    /// Mouse release: commit the stroke to the undo history.
    pub fn end(&mut self, v: &mut Volume) -> bool {
        if self.moving.is_some() {
            return self.end_move(v);
        }
        if let Some(x) = self.xform.as_mut().filter(|x| x.drag.is_some()) {
            // Scale / Wrap / Rotate stay pending until Enter or Esc.
            x.drag = None;
            return false;
        }
        let pending = self.stroke.as_ref().map(|s| self.pending_cells(v, s));
        let code = self.code.take().filter(|_| self.brush == Brush::Code);
        let changed = match pending {
            Some(cells) if !cells.is_empty() => {
                if self.tool == Tool::Marquee {
                    // Box select: only voxels inside the box.
                    let solid: Vec<IVec3> = cells.into_iter().filter(|&c| v.get(c) != 0).collect();
                    self.apply(v, &solid)
                } else if let Some((lo, out)) = code {
                    // The shader's voxels: attached into empty cells,
                    // painted over or erased from solid ones.
                    let cells: Vec<(IVec3, u8)> = out.filled().map(|(p, c)| (lo + p, c)).collect();
                    let changed = self.apply_colored(v, &cells);
                    // Like MagicaVoxel, the result stays selected with the
                    // move arrows up, to place it before the next stroke.
                    let placed: Vec<usize> = self.stroke.as_ref().map_or(Vec::new(), |s| s.changes.iter().filter(|(_, c)| c.1 != 0).map(|(&i, _)| i).collect());
                    if changed && !placed.is_empty() {
                        self.select(v, placed, SelMode::Replace);
                        self.floating = true;
                    }
                    changed
                } else {
                    self.apply(v, &cells)
                }
            }
            _ => false,
        };
        if let Some(s) = self.stroke.take() {
            let mut entries: Vec<(usize, u8, u8)> = s.changes.into_iter().filter(|(_, (a, b))| a != b).map(|(i, (a, b))| (i, a, b)).collect();
            if !entries.is_empty() {
                entries.sort_unstable_by_key(|e| e.0);
                self.undo.push(Entry::Cells(entries));
                self.redo.clear();
            }
        }
        changed
    }

    /// Cells a drag-and-release brush will fill when released.
    fn pending_cells(&self, v: &Volume, s: &Stroke) -> Vec<IVec3> {
        if self.tool == Tool::Marquee {
            return if self.opts.marquee == MarqueeMode::Box { cuboid(s.anchor, s.current) } else { Vec::new() };
        }
        match self.brush {
            Brush::Box | Brush::Code => cuboid(s.anchor, s.current),
            Brush::Geo => self.geo_cells(v, s),
            _ => Vec::new(),
        }
    }

    /// Entries in the undo history.
    pub fn history_len(&self) -> usize {
        self.undo.len()
    }

    /// Entries in the undo and redo histories.
    pub fn history_lens(&self) -> (usize, usize) {
        (self.undo.len(), self.redo.len())
    }

    /// Recolor the undo and redo history, the brush color, the copied
    /// voxels and the pattern through `remap` after the palette's colors
    /// moved to new indices.
    pub fn remap_colors(&mut self, remap: &[u8; 256]) {
        let m = |c: &mut u8| *c = remap[*c as usize];
        for e in self.undo.iter_mut().chain(&mut self.redo) {
            match e {
                Entry::Cells(diff) => diff.iter_mut().for_each(|(_, a, b)| {
                    m(a);
                    m(b);
                }),
                Entry::Swap(v) => v.cells.update_filled(m),
            }
        }
        m(&mut self.color);
        self.clipboard.iter_mut().for_each(|(_, c)| m(c));
        if let Some(p) = &mut self.pattern {
            p.volume.cells.update_filled(m);
        }
        if let Some(x) = &mut self.xform {
            x.before.cells.update_filled(m);
            x.src.cells.update_filled(m);
        }
    }

    /// Take the history so it can move into the tree's: the volume before
    /// each of the last `max` edits, oldest first, and the volume after
    /// each of up to `max` undone edits, next redo first (`v` is the volume
    /// now).
    pub fn take_history(&mut self, v: &Volume, max: usize) -> (Vec<Volume>, Vec<Volume>) {
        self.commit_xform(v);
        let walk = |list: Vec<Entry>, forward: bool| -> Vec<Volume> {
            let mut cur = v.clone();
            list.into_iter()
                .rev()
                .take(max)
                .map(|e| {
                    Self::step(&mut cur, e, forward);
                    cur.clone()
                })
                .collect()
        };
        let mut past = walk(std::mem::take(&mut self.undo), false);
        past.reverse();
        (past, walk(std::mem::take(&mut self.redo), true))
    }

    /// Fold the edits made since the history had `len` entries into one,
    /// given the volume as it was then.
    pub fn squash(&mut self, v: &mut Volume, len: usize, before: Volume) {
        if self.undo.len() <= len + 1 {
            return;
        }
        self.undo.truncate(len);
        let after = std::mem::replace(v, before);
        self.commit(v, after);
    }

    pub fn undo(&mut self, v: &mut Volume) -> bool {
        self.floating = false;
        if self.cancel_xform(v) {
            return true;
        }
        let Some(entry) = self.undo.pop() else { return false };
        let entry = Self::step(v, entry, false);
        self.redo.push(entry);
        self.fit_selection(v);
        true
    }

    pub fn redo(&mut self, v: &mut Volume) -> bool {
        self.floating = false;
        self.commit_xform(v);
        let Some(entry) = self.redo.pop() else { return false };
        let entry = Self::step(v, entry, true);
        self.undo.push(entry);
        self.fit_selection(v);
        true
    }

    /// Cells to outline: the pending shape, or what a click would touch.
    pub fn preview(&self, v: &Volume, hover: Option<Pick>) -> Vec<IVec3> {
        let normal = hover.map_or(IVec3::Z, |h| h.normal);
        let cells: Vec<IVec3> = match (&self.stroke, self.tool) {
            (Some(s), _) => {
                let pending = self.pending_cells(v, s);
                if pending.is_empty() {
                    return Vec::new();
                }
                pending
            }
            (None, Tool::Move) => return Vec::new(),
            (None, Tool::Marquee) if self.opts.marquee == MarqueeMode::Rect => return Vec::new(),
            (None, Tool::Marquee) => hover.and_then(|h| target(Tool::Paint, h)).into_iter().collect(),
            (None, Tool::Pick) => hover.filter(|h| !h.on_ground()).map(|h| h.cell).into_iter().collect(),
            // Regions can be scattered all over the model, so like
            // MagicaVoxel only the voxel under the cursor is outlined.
            (None, Tool::Region | Tool::RegionRemove | Tool::RegionReplace) => hover.filter(|h| !h.on_ground()).map(|h| h.cell).into_iter().collect(),
            (None, _) => match (self.brush, hover) {
                (Brush::Face, Some(h)) => self.face_cells(v, h).into_iter().map(|(c, _)| c).collect(),
                (Brush::Pattern, Some(h)) => {
                    target(self.tool, h).map(|t| self.pattern_cells(t, h.normal).into_iter().map(|(c, _)| c).collect()).unwrap_or_default()
                }
                (_, Some(h)) => target(self.tool, h).into_iter().collect(),
                (_, None) => Vec::new(),
            },
        };
        let colored: Vec<(IVec3, u8)> = cells.into_iter().map(|c| (c, 0)).collect();
        // Pending shapes were already placed; only the hover cursor gets the
        // voxel stamp.
        let all = if self.stroke.is_some() {
            colored.iter().flat_map(|&(c, _)| self.lofted(v, vec![c])).flat_map(|c| self.mirrored(v, c)).collect()
        } else {
            self.expand(v, &colored, normal).into_iter().map(|(c, _)| c).collect::<Vec<_>>()
        };
        all.into_iter().filter(|&c| v.contains(c)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vol() -> Volume {
        Volume::new(IVec3::new(8, 8, 8))
    }

    #[test]
    fn remap_colors_reaches_history_clipboard_and_pattern() {
        let mut v = vol();
        let mut e = Editor::new(3);
        let mut painted = v.clone();
        painted.set_index(0, 3);
        e.commit(&mut v, painted);
        e.copy(&v);
        e.pattern = Some(Pattern { name: "p".into(), volume: v.clone() });
        let mut remap: [u8; 256] = std::array::from_fn(|i| i as u8);
        remap[3] = 7;
        v.cells.update_filled(|c| *c = remap[*c as usize]);
        e.remap_colors(&remap);
        assert_eq!(e.color, 7);
        assert_eq!(e.pattern.as_ref().unwrap().volume.cells[0], 7);
        // Paste brings back the remapped color...
        v.set_index(0, 0);
        e.paste(&mut v);
        assert_eq!(v.cells[0], 7);
        // ...and undoing the first edit leaves it empty, redoing gives 7.
        e.select_none();
        e.undo(&mut v);
        e.undo(&mut v);
        assert_eq!(v.cells[0], 0);
        e.redo(&mut v);
        assert_eq!(v.cells[0], 7);
    }

    #[test]
    fn pick_ground_then_attach_and_undo() {
        let mut v = vol();
        let mut e = Editor::new(5);
        let hit = pick(&v, Vec3::new(2.5, 3.5, 20.0), Vec3::new(0.0, 0.0, -1.0)).unwrap();
        assert_eq!(hit, Pick { cell: IVec3::new(2, 3, -1), normal: IVec3::Z });
        assert!(e.begin(&mut v, hit, Vec3::ZERO, Vec3::NEG_Z));
        e.end(&mut v);
        assert_eq!(v.get(IVec3::new(2, 3, 0)), 5);

        // Now the ray hits the new voxel's top face.
        let hit = pick(&v, Vec3::new(2.5, 3.5, 20.0), Vec3::new(0.0, 0.0, -1.0)).unwrap();
        assert_eq!(hit, Pick { cell: IVec3::new(2, 3, 0), normal: IVec3::Z });

        assert!(e.undo(&mut v));
        assert_eq!(v.get(IVec3::new(2, 3, 0)), 0);
        assert!(e.redo(&mut v));
        assert_eq!(v.get(IVec3::new(2, 3, 0)), 5);
    }

    #[test]
    fn box_brush_fills_on_release() {
        let mut v = vol();
        let mut e = Editor::new(7);
        e.brush = Brush::Box;
        let ro = Vec3::new(1.5, 1.5, 20.0);
        let down = Vec3::new(0.0, 0.0, -1.0);
        let hit = pick(&v, ro, down).unwrap();
        assert!(!e.begin(&mut v, hit, Vec3::ZERO, Vec3::NEG_Z));
        let ro2 = Vec3::new(3.5, 2.5, 20.0);
        let hit2 = pick(&v, ro2, down);
        e.drag(&mut v, ro2, down, hit2);
        assert!(e.end(&mut v));
        assert_eq!(v.cells.iter().filter(|&&c| c == 7).count(), 3 * 2);
    }

    #[test]
    fn mirror_x() {
        let mut v = vol();
        let mut e = Editor::new(3);
        e.mirror[0] = true;
        let hit = pick(&v, Vec3::new(1.5, 1.5, 20.0), Vec3::new(0.0, 0.0, -1.0)).unwrap();
        e.begin(&mut v, hit, Vec3::ZERO, Vec3::NEG_Z);
        e.end(&mut v);
        assert_eq!(v.get(IVec3::new(1, 1, 0)), 3);
        assert_eq!(v.get(IVec3::new(6, 1, 0)), 3);
    }

    #[test]
    fn resize_undo_swaps_whole_volume() {
        let mut v = vol();
        v.set(IVec3::new(1, 1, 1), 4);
        let mut e = Editor::new(1);
        let bigger = ops::resize(&v, IVec3::new(10, 10, 10));
        assert!(e.commit(&mut v, bigger));
        assert_eq!(v.dims, IVec3::splat(10));
        assert!(e.undo(&mut v));
        assert_eq!(v.dims, IVec3::splat(8));
        assert_eq!(v.get(IVec3::new(1, 1, 1)), 4);
        assert!(e.redo(&mut v));
        assert_eq!(v.get(IVec3::new(2, 2, 1)), 4);
    }

    #[test]
    fn squash_folds_edits_into_one_undo() {
        let mut v = vol();
        let mut e = Editor::new(1);
        let before = v.clone();
        for k in 0..3 {
            let mut new = v.clone();
            new.set(IVec3::new(k, 0, 0), 2);
            assert!(e.commit(&mut v, new));
        }
        let bigger = ops::resize(&v, IVec3::new(10, 10, 10));
        assert!(e.commit(&mut v, bigger));
        e.squash(&mut v, 0, before);
        assert_eq!((e.history_len(), v.dims), (1, IVec3::splat(10)));
        assert!(e.undo(&mut v));
        assert_eq!(v.dims, IVec3::splat(8));
        assert!(v.cells.iter().all(|&c| c == 0));
        assert!(!e.can_undo());
    }

    #[test]
    fn move_selection_and_paste() {
        let mut v = vol();
        v.set(IVec3::new(1, 1, 0), 4);
        let mut e = Editor::new(1);
        e.select_all(&v);
        let hit = pick(&v, Vec3::new(1.5, 1.5, 20.0), Vec3::NEG_Z).unwrap();
        assert!(e.begin_move(&v, hit, false));
        e.drag(&mut v, Vec3::new(3.5, 1.5, 20.0), Vec3::NEG_Z, None);
        assert!(e.end(&mut v));
        assert_eq!((v.get(IVec3::new(1, 1, 0)), v.get(IVec3::new(3, 1, 0))), (0, 4));
        assert_eq!(e.selected_count(), 1);

        e.copy(&v);
        e.undo(&mut v);
        assert_eq!(v.get(IVec3::new(1, 1, 0)), 4);
        assert!(e.paste(&mut v));
        assert_eq!(v.get(IVec3::new(3, 1, 0)), 4);
        assert_eq!(e.selected_count(), 1);
    }

    #[test]
    fn selection_is_left_alone_by_brushes() {
        let mut v = vol();
        v.set(IVec3::new(1, 1, 0), 4);
        v.set(IVec3::new(2, 1, 0), 4);
        let mut e = Editor::new(9);
        e.select(&v, [v.index(IVec3::new(1, 1, 0))], SelMode::Replace);
        e.tool = Tool::Paint;
        e.brush = Brush::Face;
        let hit = pick(&v, Vec3::new(2.5, 1.5, 20.0), Vec3::NEG_Z).unwrap();
        e.begin(&mut v, hit, Vec3::ZERO, Vec3::NEG_Z);
        e.end(&mut v);
        // Measured in MagicaVoxel: painting over a selection changes only
        // the voxels outside it.
        assert_eq!((v.get(IVec3::new(1, 1, 0)), v.get(IVec3::new(2, 1, 0))), (4, 9));
    }

    fn count(v: &Volume, c: u8) -> usize {
        v.cells.iter().filter(|&&x| x == c).count()
    }

    fn click(e: &mut Editor, v: &mut Volume, x: f32, y: f32) {
        let (ro, rd) = (Vec3::new(x, y, 20.0), Vec3::NEG_Z);
        let hit = pick(v, ro, rd).unwrap();
        e.begin(v, hit, ro, rd);
        e.end(v);
    }

    #[test]
    fn brush_size_shape_and_axis() {
        let mut v = Volume::new(IVec3::splat(16));
        let mut e = Editor::new(2);
        e.opts.size = 3;
        click(&mut e, &mut v, 8.5, 8.5);
        // A 3x3x3 cube resting on the ground.
        assert_eq!(count(&v, 2), 27);
        assert_eq!(v.get(IVec3::new(8, 8, 2)), 2);

        let mut v = Volume::new(IVec3::splat(16));
        e.opts.flat = true;
        click(&mut e, &mut v, 8.5, 8.5);
        assert_eq!(count(&v, 2), 9);

        let mut v = Volume::new(IVec3::splat(16));
        e.opts = BrushOptions { size: 5, sphere: true, flat: true, ..Default::default() };
        click(&mut e, &mut v, 8.5, 8.5);
        assert_eq!(count(&v, 2), 21);

        let mut v = Volume::new(IVec3::splat(16));
        e.opts = BrushOptions { axis: [false, false, true], ..Default::default() };
        click(&mut e, &mut v, 3.5, 3.5);
        assert_eq!(count(&v, 2), 16);
    }

    #[test]
    fn geo_square_and_circle() {
        let mut e = Editor::new(4);
        e.brush = Brush::Geo;
        let down = Vec3::NEG_Z;
        // Measured in MagicaVoxel: a 2-cell drag makes a 5x5 square or a
        // 3x3 circle; Fill off leaves out only the center cell.
        for (shape, fill, expect) in [(GeoShape::Square, true, 25), (GeoShape::Square, false, 24), (GeoShape::Circle, true, 9), (GeoShape::Circle, false, 8)] {
            let mut v = Volume::new(IVec3::splat(16));
            e.opts.geo = shape;
            e.opts.fill = fill;
            let ro = Vec3::new(8.5, 8.5, 20.0);
            let hit = pick(&v, ro, down).unwrap();
            e.begin(&mut v, hit, ro, down);
            let ro2 = Vec3::new(10.5, 8.5, 20.0);
            let hit2 = pick(&v, ro2, down);
            e.drag(&mut v, ro2, down, hit2);
            e.end(&mut v);
            assert_eq!(count(&v, 4), expect, "{shape:?} fill {fill}");
        }
    }

    /// Geo shape cell count for a drag of (dx, dy) on the floor.
    fn geo_count(shape: GeoShape, fill: bool, even: bool, dx: f32, dy: f32) -> usize {
        let mut e = Editor::new(4);
        e.brush = Brush::Geo;
        (e.opts.geo, e.opts.fill, e.opts.even) = (shape, fill, even);
        let mut v = Volume::new(IVec3::splat(24));
        let down = Vec3::NEG_Z;
        let ro = Vec3::new(12.5, 12.5, 30.0);
        let hit = pick(&v, ro, down).unwrap();
        e.begin(&mut v, hit, ro, down);
        let ro2 = ro + Vec3::new(dx, dy, 0.0);
        let hit2 = pick(&v, ro2, down);
        e.drag(&mut v, ro2, down, hit2);
        e.end(&mut v);
        count(&v, 4)
    }

    #[test]
    fn geo_shapes_match_magicavoxel() {
        use GeoShape::{Circle, Square};
        // (shape, fill, even, drag, cells), counted in MagicaVoxel 0.99.6.
        for (shape, fill, even, (dx, dy), expect) in [
            (Circle, true, false, (1.0, 0.0), 1),
            (Circle, true, false, (1.0, 1.0), 9),
            (Circle, true, false, (3.0, 0.0), 21),
            (Circle, true, false, (2.0, 1.0), 21),
            (Circle, true, false, (2.0, 2.0), 21),
            (Circle, true, false, (3.0, 1.0), 37),
            (Circle, true, false, (4.0, 0.0), 37),
            (Circle, false, false, (4.0, 0.0), 36),
            (Circle, true, true, (3.0, 0.0), 32),
            (Circle, true, true, (4.0, 0.0), 52),
            (Circle, false, true, (3.0, 0.0), 28),
            (Square, true, false, (3.0, 0.0), 49),
            (Square, true, false, (3.0, -3.0), 49),
            (Square, false, false, (3.0, 0.0), 48),
            (Square, true, true, (3.0, 0.0), 64),
            (Square, false, true, (3.0, 0.0), 60),
        ] {
            assert_eq!(geo_count(shape, fill, even, dx, dy), expect, "{shape:?} fill {fill} even {even} drag ({dx}, {dy})");
        }
    }

    #[test]
    fn pattern_stamps_on_surface() {
        let mut v = Volume::new(IVec3::splat(16));
        let mut e = Editor::new(1);
        let mut p = Volume::new(IVec3::new(2, 2, 3));
        p.set(IVec3::ZERO, 7);
        p.set(IVec3::new(1, 1, 2), 8);
        e.pattern = Some(Pattern { name: "p".into(), volume: p });
        e.brush = Brush::Pattern;
        click(&mut e, &mut v, 8.5, 8.5);
        assert_eq!(v.get(IVec3::new(7, 7, 0)), 7);
        assert_eq!(v.get(IVec3::new(8, 8, 2)), 8);
    }

    #[test]
    fn move_along_normal() {
        let mut v = vol();
        v.set(IVec3::new(2, 2, 0), 4);
        let mut e = Editor::new(1);
        let hit = Pick { cell: IVec3::new(2, 2, 0), normal: IVec3::Z };
        assert!(e.begin_move(&v, hit, true));
        // A horizontal ray passing 3 voxels above the voxel.
        e.drag(&mut v, Vec3::new(-10.0, 2.5, 3.5), Vec3::X, None);
        e.end(&mut v);
        assert_eq!(v.get(IVec3::new(2, 2, 3)), 4);
        assert_eq!(v.get(IVec3::new(2, 2, 0)), 0);
    }

    #[test]
    fn face_erase_removes_top_layer() {
        let mut v = vol();
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..2 {
                    let i = v.index(IVec3::new(x, y, z));
                    v.set_index(i, 9);
                }
            }
        }
        let mut e = Editor::new(9);
        e.tool = Tool::Erase;
        e.brush = Brush::Face;
        let hit = pick(&v, Vec3::new(1.5, 1.5, 20.0), Vec3::new(0.0, 0.0, -1.0)).unwrap();
        e.begin(&mut v, hit, Vec3::ZERO, Vec3::NEG_Z);
        e.end(&mut v);
        assert_eq!(v.cells.iter().filter(|&&c| c == 9).count(), 16);
    }

    fn slab(v: &mut Volume, depth: i32, c: u8) {
        for p in cuboid(IVec3::ZERO, IVec3::new(3, 3, depth - 1)) {
            v.set(p, c);
        }
    }

    /// A horizontal ray along +x through height z.
    fn level_ray(z: f32) -> (Vec3, Vec3) {
        (Vec3::new(-10.0, 1.5, z), Vec3::X)
    }

    #[test]
    fn face_drag_extrudes_layers_in_one_undo() {
        let mut v = vol();
        slab(&mut v, 1, 9);
        let mut e = Editor::new(9);
        e.brush = Brush::Face;
        let (ro, rd) = (Vec3::new(1.5, 1.5, 20.0), Vec3::NEG_Z);
        let hit = pick(&v, ro, rd).unwrap();
        assert!(e.begin(&mut v, hit, ro, rd));
        assert_eq!(count(&v, 9), 32);
        // Three cells above the face: three layers, then back down to one.
        let (ro, rd) = level_ray(4.0);
        assert!(e.drag(&mut v, ro, rd, None));
        assert_eq!(count(&v, 9), 16 * 4);
        let (ro, rd) = level_ray(1.2);
        assert!(e.drag(&mut v, ro, rd, None));
        assert_eq!(count(&v, 9), 32);
        let (ro, rd) = level_ray(3.0);
        e.drag(&mut v, ro, rd, None);
        e.end(&mut v);
        assert_eq!(count(&v, 9), 48);
        assert!(e.undo(&mut v));
        assert_eq!(count(&v, 9), 16);
        assert!(!e.can_undo());
    }

    #[test]
    fn face_drag_erases_layers() {
        let mut v = vol();
        slab(&mut v, 4, 9);
        let mut e = Editor::new(9);
        e.brush = Brush::Face;
        e.tool = Tool::Erase;
        let (ro, rd) = (Vec3::new(1.5, 1.5, 20.0), Vec3::NEG_Z);
        let hit = pick(&v, ro, rd).unwrap();
        e.begin(&mut v, hit, ro, rd);
        // Two cells out from the face: MagicaVoxel digs two layers in.
        let (ro2, rd2) = level_ray(6.0);
        e.drag(&mut v, ro2, rd2, None);
        e.end(&mut v);
        assert_eq!(count(&v, 9), 32);
        assert_eq!((v.get(IVec3::new(0, 0, 1)), v.get(IVec3::new(0, 0, 2))), (9, 0));
        // Dragging into the model keeps one layer.
        e.undo(&mut v);
        e.begin(&mut v, hit, ro, rd);
        let (ro2, rd2) = level_ray(2.0);
        e.drag(&mut v, ro2, rd2, None);
        e.end(&mut v);
        assert_eq!(count(&v, 9), 48);
    }

    #[test]
    fn pattern_offset_moves_the_stamp() {
        let mut v = Volume::new(IVec3::splat(16));
        let mut e = Editor::new(1);
        let mut p = Volume::new(IVec3::ONE);
        p.set(IVec3::ZERO, 7);
        e.pattern = Some(Pattern { name: "p".into(), volume: p });
        e.brush = Brush::Pattern;
        e.opts.pattern_offset = IVec3::new(2, -1, 3);
        click(&mut e, &mut v, 8.5, 8.5);
        assert_eq!(v.get(IVec3::new(10, 7, 3)), 7);
        assert_eq!(count(&v, 7), 1);
    }

    #[test]
    fn step_pattern_wraps_through_the_list() {
        let files = [std::path::Path::new("/a/x.vox"), std::path::Path::new("/b/y.vox"), std::path::Path::new("/b/z.vox")];
        let stem = |p: Option<&std::path::Path>| p.and_then(|p| p.file_stem()).map(|s| s.to_string_lossy().into_owned());
        assert_eq!(stem(step_pattern(&files, Some("y"), 1)).as_deref(), Some("z"));
        assert_eq!(stem(step_pattern(&files, Some("z"), 1)).as_deref(), Some("x"));
        assert_eq!(stem(step_pattern(&files, Some("x"), -1)).as_deref(), Some("z"));
        assert_eq!(stem(step_pattern(&files, None, 1)).as_deref(), Some("x"));
        assert_eq!(stem(step_pattern(&files, Some("selection"), -1)).as_deref(), Some("z"));
        assert!(step_pattern(&[], None, 1).is_none());
    }

    #[test]
    fn move_loops_around_the_canvas() {
        let mut v = vol();
        v.set(IVec3::new(6, 1, 0), 4);
        let mut e = Editor::new(1);
        e.select_all(&v);
        let hit = Pick { cell: IVec3::new(6, 1, 0), normal: IVec3::Z };
        assert!(e.begin_move(&v, hit, false));
        e.drag(&mut v, Vec3::new(9.5, 1.5, 20.0), Vec3::NEG_Z, None);
        e.end(&mut v);
        assert_eq!((v.get(IVec3::new(1, 1, 0)), count(&v, 4)), (4, 1));
        assert_eq!(e.selected_count(), 1);
    }

    /// A bar of `len` voxels along x from `x0` (y = 3, z = 0).
    fn bar(x0: i32, len: i32) -> Volume {
        let mut v = vol();
        for x in x0..x0 + len {
            v.set(IVec3::new(x, 3, 0), 5);
        }
        v
    }

    fn row(v: &Volume) -> Vec<i32> {
        (0..8).filter(|&x| v.get(IVec3::new(x, 3, 0)) != 0).collect()
    }

    /// Drag the +X arrow of the Scale / Wrap gizmo, whose box ends at
    /// `x_end`, to `x`.
    fn drag_x_face(e: &mut Editor, v: &mut Volume, x_end: i32, x: f32, shift: bool) {
        e.tool = Tool::Move;
        let face = (x_end + 1) as f32;
        assert!(e.begin_handle(v, Handle::Arrow { axis: 0, sign: 1 }, Vec3::new(face, 3.5, 20.0), Vec3::NEG_Z, shift));
        e.drag(v, Vec3::new(x, 3.5, 20.0), Vec3::NEG_Z, None);
        e.end(v);
    }

    #[test]
    fn scale_stretches_and_waits_for_enter() {
        let mut v = bar(2, 2);
        let mut e = Editor::new(1);
        e.tool = Tool::Move;
        e.opts.xform = XformMode::Scale;
        drag_x_face(&mut e, &mut v, 3, 6.0, false);
        assert_eq!(row(&v), vec![2, 3, 4, 5]);
        assert!(e.xform_pending());
        // Another drag keeps building on the pending scale; Shift is two-sided.
        drag_x_face(&mut e, &mut v, 5, 7.0, true);
        assert_eq!(row(&v), vec![1, 2, 3, 4, 5, 6]);
        assert!(e.commit_xform(&v));
        assert!(!e.xform_pending());
        assert!(e.undo(&mut v));
        assert_eq!(row(&v), vec![2, 3]);
        assert!(!e.can_undo());
    }

    #[test]
    fn scale_cancel_and_undo_restore() {
        let mut v = bar(2, 2);
        let mut e = Editor::new(1);
        e.select_all(&v);
        e.opts.xform = XformMode::Scale;
        drag_x_face(&mut e, &mut v, 3, 6.0, false);
        assert_eq!(e.selected_count(), 4);
        assert!(e.cancel_xform(&mut v));
        assert_eq!((row(&v), e.selected_count()), (vec![2, 3], 2));
        // Undo while pending cancels too.
        drag_x_face(&mut e, &mut v, 3, 5.0, false);
        assert_eq!(row(&v), vec![2, 3, 4]);
        assert!(e.undo(&mut v));
        assert_eq!(row(&v), vec![2, 3]);
        // A brush stroke commits the pending transform first.
        drag_x_face(&mut e, &mut v, 3, 5.0, false);
        e.select_none();
        e.tool = Tool::Attach;
        click(&mut e, &mut v, 0.5, 0.5);
        assert!(!e.xform_pending());
        assert_eq!(row(&v), vec![2, 3, 4]);
        e.undo(&mut v);
        e.undo(&mut v);
        assert_eq!((row(&v), v.get(IVec3::ZERO)), (vec![2, 3], 0));
    }

    #[test]
    fn wrap_repeats_with_spacing_and_crops() {
        let mut v = bar(0, 2);
        let mut e = Editor::new(1);
        e.opts.xform = XformMode::Wrap;
        e.opts.wrap_space = [1, 0, 0];
        drag_x_face(&mut e, &mut v, 1, 6.0, false);
        assert_eq!(row(&v), vec![0, 1, 3, 4]);
        e.commit_xform(&v);
        let mut v = bar(0, 3);
        drag_x_face(&mut e, &mut v, 2, 2.0, false);
        assert_eq!(row(&v), vec![0, 1]);
    }

    #[test]
    fn rotate_turns_about_the_box_center() {
        let mut v = bar(2, 3);
        let mut e = Editor::new(1);
        e.tool = Tool::Move;
        e.opts.xform = XformMode::Rotate;
        // Grab the z ring at x = 5.5 and swing it a quarter turn.
        assert!(e.begin_handle(&v, Handle::Ring { axis: 2 }, Vec3::new(5.5, 3.5, 20.0), Vec3::NEG_Z, false));
        e.drag(&mut v, Vec3::new(3.5, 5.5, 20.0), Vec3::NEG_Z, None);
        e.end(&mut v);
        let cells: Vec<IVec3> = v.filled().map(|(p, _)| p).collect();
        assert_eq!(cells, vec![IVec3::new(3, 2, 0), IVec3::new(3, 3, 0), IVec3::new(3, 4, 0)]);
        assert!(e.commit_xform(&v));
    }

    #[test]
    fn code_brush_uses_its_box_as_the_volume() {
        let mut v = vol();
        v.set(IVec3::new(2, 2, 0), 3);
        let mut e = Editor::new(1);
        e.brush = Brush::Code;
        let (ro, rd) = (Vec3::new(1.5, 1.5, 20.0), Vec3::NEG_Z);
        let hit = pick(&v, ro, rd).unwrap();
        e.begin(&mut v, hit, ro, rd);
        let ro2 = Vec3::new(4.5, 3.5, 20.0);
        let hit2 = pick(&v, ro2, rd);
        e.drag(&mut v, ro2, rd, hit2);
        let (lo, size) = e.code_box(&v).unwrap();
        assert_eq!((lo, size), (IVec3::new(1, 1, 0), IVec3::new(4, 3, 1)));
        // The shader sees the box's own voxels at local coordinates.
        let sub = sub_volume(&v, lo, size);
        assert_eq!((sub.get(IVec3::new(1, 1, 0)), sub.count()), (3, 1));
        // Its output lands back in the box; empty output cells change nothing.
        let mut out = Volume::new(size);
        out.set(IVec3::ZERO, 8);
        out.set(IVec3::new(3, 2, 0), 8);
        out.set(IVec3::new(1, 1, 0), 8);
        e.code = Some((lo, out));
        assert!(e.end(&mut v));
        assert_eq!((v.get(IVec3::new(1, 1, 0)), v.get(IVec3::new(4, 3, 0)), v.get(IVec3::new(2, 2, 0))), (8, 8, 3));
        assert_eq!(v.count(), 3);
    }

    #[test]
    fn rotate_turns_in_whole_degrees_or_5_with_shift() {
        for (shift, expect) in [(false, 37.0), (true, 35.0)] {
            let mut v = bar(2, 3);
            let mut e = Editor::new(1);
            e.tool = Tool::Move;
            e.opts.xform = XformMode::Rotate;
            // Center (3.5, 3.5); grab at angle 0 and swing to about 37.4 degrees.
            assert!(e.begin_handle(&v, Handle::Ring { axis: 2 }, Vec3::new(5.5, 3.5, 20.0), Vec3::NEG_Z, shift));
            let a = 37.4f32.to_radians();
            e.drag(&mut v, Vec3::new(3.5 + 2.0 * a.cos(), 3.5 + 2.0 * a.sin(), 20.0), Vec3::NEG_Z, None);
            assert_eq!(e.xform_status(&v).unwrap(), format!("angle:{expect}"));
        }
    }

    #[test]
    fn marquee_gizmo_arrow_moves_the_selection_along_its_axis() {
        let mut v = bar(2, 2);
        let mut e = Editor::new(1);
        e.tool = Tool::Marquee;
        assert!(e.gizmo(&v).is_none());
        e.select_all(&v);
        let g = e.gizmo(&v).unwrap();
        assert_eq!((g.kind, g.lo, g.hi), (GizmoKind::Move, Vec3::new(2.0, 3.0, 0.0), Vec3::new(4.0, 4.0, 1.0)));
        // Grab the +x arrow at x = 6 and pull it to x = 8: two cells right,
        // whatever the ray does across the arrow.
        assert!(e.begin_handle(&v, Handle::Arrow { axis: 0, sign: 1 }, Vec3::new(6.0, 3.5, 20.0), Vec3::NEG_Z, false));
        e.drag(&mut v, Vec3::new(8.0, 6.5, 20.0), Vec3::NEG_Z, None);
        e.end(&mut v);
        assert_eq!((row(&v), e.selected_count()), (vec![4, 5], 2));
    }

    #[test]
    fn code_result_stays_movable_and_undoes_with_its_stroke() {
        let mut v = vol();
        let mut e = Editor::new(1);
        e.brush = Brush::Code;
        let rd = Vec3::NEG_Z;
        let ro = Vec3::new(1.5, 1.5, 20.0);
        let hit = pick(&v, ro, rd).unwrap();
        e.begin(&mut v, hit, ro, rd);
        let ro2 = Vec3::new(2.5, 1.5, 20.0);
        let hit2 = pick(&v, ro2, rd);
        e.drag(&mut v, ro2, rd, hit2);
        let (lo, size) = e.code_box(&v).unwrap();
        let mut out = Volume::new(size);
        out.set(IVec3::ZERO, 8);
        out.set(IVec3::X, 8);
        e.code = Some((lo, out));
        assert!(e.end(&mut v));
        // The result is selected with the move arrows up.
        assert_eq!((e.selected_count(), e.gizmo(&v).map(|g| g.kind)), (2, Some(GizmoKind::Move)));
        let g = e.gizmo(&v).unwrap();
        let top = Vec3::new(2.0, 2.0, g.hi.z + 1.0);
        assert!(e.begin_handle(&v, Handle::Arrow { axis: 2, sign: 1 }, top + Vec3::new(0.0, -20.0, 0.0), Vec3::Y, false));
        e.drag(&mut v, top + Vec3::new(0.0, -20.0, 3.0), Vec3::Y, None);
        e.end(&mut v);
        assert_eq!((v.get(IVec3::new(1, 1, 3)), v.get(IVec3::new(1, 1, 0))), (8, 0));
        // One undo takes back the stroke and the move.
        assert!(e.undo(&mut v));
        assert_eq!((v.count(), e.can_undo()), (0, false));
        // The next stroke lets go of the result.
        let mut v = vol();
        e.redo(&mut v);
        click(&mut e, &mut v, 6.5, 6.5);
        assert!(e.gizmo(&v).is_none());
    }

    #[test]
    fn box_spans_from_the_pressed_voxel_to_the_one_under_the_cursor() {
        // A step: a wall at y = 3 (x 0-7) and one closer at y = 1 (x 5-7),
        // seen from -y.
        let mut v = vol();
        for x in 0..8 {
            for z in 0..4 {
                v.set(IVec3::new(x, 3, z), 2);
                if x >= 5 {
                    v.set(IVec3::new(x, 1, z), 2);
                }
            }
        }
        let mut e = Editor::new(1);
        e.brush = Brush::Box;
        e.tool = Tool::Paint;
        let rd = Vec3::Y;
        let ro = Vec3::new(1.5, -20.0, 0.5);
        let hit = pick(&v, ro, rd).unwrap();
        e.begin(&mut v, hit, ro, rd);
        let ro2 = Vec3::new(6.5, -20.0, 2.5);
        let hit2 = pick(&v, ro2, rd);
        e.drag(&mut v, ro2, rd, hit2);
        e.end(&mut v);
        // x 1-6, y 1-3, z 0-2: the wall's cells x 1-6 and the step's x 5-6.
        assert_eq!(count(&v, 1), 6 * 3 + 2 * 3);
    }
}
