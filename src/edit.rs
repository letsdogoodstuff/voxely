//! Voxel editing, modelled on MagicaVoxel's brushes: Attach / Erase /
//! Paint tools combined with Voxel / Face / Box / Geometry / Pattern / Code
//! brushes, axis mode and mirroring, selection (region and marquee), moving and copy/paste of
//! selected voxels, and undo/redo per stroke or whole-model edit.

use std::collections::{HashMap, HashSet, VecDeque};

use glam::{IVec3, Vec3};

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
    /// 3D box dragged on a plane.
    Box,
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
    /// Square and circle are filled rather than outlines.
    pub fill: bool,
    /// Square and circle have an even width (centered between cells).
    pub even: bool,
    /// Axis mode: extend brushes through the whole canvas along these axes.
    pub axis: [bool; 3],
    pub marquee: MarqueeMode,
    /// Rect select only picks voxels visible from the camera.
    pub visible_only: bool,
    /// Pattern brush paints with the palette color, not its own colors.
    pub pattern_palette: bool,
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
    /// The Code brush's shader output for the current volume.
    pub code: Option<Volume>,
    /// Set when the selection changes, so the view can refresh.
    pub sel_dirty: bool,
    undo: Vec<Entry>,
    redo: Vec<Entry>,
    stroke: Option<Stroke>,
    moving: Option<Move>,
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
            sel: Mask::new(IVec3::ZERO),
            sel_count: 0,
            clipboard: Vec::new(),
        }
    }

    pub fn active(&self) -> bool {
        self.stroke.is_some() || self.moving.is_some()
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
        let mut mask = Mask::new(v.dims);
        for (i, _) in v.cells.filled() {
            mask.set(i, true);
        }
        self.set_selection(v, mask);
    }

    pub fn select_none(&mut self) {
        if self.sel_count > 0 {
            self.sel = Mask::new(IVec3::ZERO);
            self.sel_count = 0;
            self.sel_dirty = true;
        }
    }

    pub fn select_invert(&mut self, v: &Volume) {
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
        let sel = self.selection();
        let cells: Vec<(IVec3, u8)> =
            v.cells.iter().enumerate().filter(|&(i, &c)| c != 0 && sel.is_none_or(|s| s[i])).map(|(i, &c)| (v.position(i), c)).collect();
        if cells.is_empty() {
            return false;
        }
        let whole = sel.is_none();
        self.moving = Some(Move { before: v.clone(), cells, anchor: hit.cell, axis: axis_of(hit.normal), along_normal, delta: IVec3::ZERO, whole });
        true
    }

    fn drag_move(&mut self, v: &mut Volume, ro: Vec3, rd: Vec3) -> bool {
        let Some(m) = self.moving.as_mut() else { return false };
        let delta = if m.along_normal {
            // Closest point on the normal line through the grabbed voxel.
            let mut dir = Vec3::ZERO;
            dir[m.axis] = 1.0;
            let p0 = m.anchor.as_vec3() + 0.5;
            let w = p0 - ro;
            let (b, d, e) = (dir.dot(rd), dir.dot(w), rd.dot(w));
            let denom = 1.0 - b * b;
            if denom < 1e-6 {
                return false;
            }
            let mut delta = IVec3::ZERO;
            delta[m.axis] = ((b * e - d) / denom).round() as i32;
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
        let mut mask = Mask::new(v.dims);
        for &(p, c) in &m.cells {
            let q = p + delta;
            if v.contains(q) {
                let i = v.index(q);
                v.set_index(i, c);
                mask[i] = true;
            }
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
        let changed = self.commit(v, moved);
        if !m.whole {
            self.set_selection(v, mask);
        }
        changed
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
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
        // A selection masks erasing and painting.
        let sel = if self.sel_count > 0 && tool != Tool::Attach { Some(&self.sel) } else { None };
        let code = if self.brush == Brush::Code { self.code.as_ref().filter(|c| c.dims == v.dims) } else { None };
        let Some(stroke) = self.stroke.as_mut() else { return false };
        let mut changed = false;
        for (c, color) in all {
            if !v.contains(c) {
                continue;
            }
            let i = v.index(c);
            if sel.is_some_and(|s| !s[i]) {
                continue;
            }
            let old = v.cells[i];
            let new = match (code, tool) {
                // The shader's output replaces the brushed cells.
                (Some(code), Tool::Attach) if old == 0 => code.cells[i],
                (Some(code), Tool::Paint) if old != 0 && code.cells[i] != 0 => code.cells[i],
                (Some(code), Tool::Erase) if code.cells[i] != 0 => 0,
                (Some(_), _) => old,
                (None, Tool::Attach) if old == 0 => color,
                (None, Tool::Erase) => 0,
                (None, Tool::Paint) if old != 0 => color,
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
            GeoShape::Square => {
                let h = (b[u] - a[u]).abs().max((b[w] - a[w]).abs());
                let (lo, hi) = (-h, h - o.even as i32);
                let mut out = Vec::new();
                for dw in lo..=hi {
                    for du in lo..=hi {
                        if o.fill || du == lo || du == hi || dw == lo || dw == hi {
                            out.push(at(du, dw));
                        }
                    }
                }
                out
            }
            GeoShape::Circle => {
                let d = Vec3::new((b[u] - a[u]) as f32, (b[w] - a[w]) as f32, 0.0);
                let r = d.length() + 0.5;
                let c = if o.even { 0.5 } else { 0.0 };
                let k = r.ceil() as i32 + 1;
                let mut out = Vec::new();
                for dw in -k..=k {
                    for du in -k..=k {
                        let dist = Vec3::new(du as f32 - c, dw as f32 - c, 0.0).length();
                        if dist <= r && (o.fill || dist > r - 1.0) {
                            out.push(at(du, dw));
                        }
                    }
                }
                out
            }
        }
    }

    /// Mouse press with the camera ray. Returns whether the volume changed.
    pub fn begin(&mut self, v: &mut Volume, hit: Pick, ro: Vec3, rd: Vec3) -> bool {
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
        self.stroke = Some(Stroke { changes: HashMap::new(), anchor: t, axis, current: t, normal: hit.normal, ray0: (ro, rd), ray1: (ro, rd) });
        if self.tool == Tool::Marquee {
            return if self.opts.marquee == MarqueeMode::Brush { self.apply(v, &[t]) } else { false };
        }
        match self.brush {
            Brush::Voxel => self.apply(v, &[t]),
            Brush::Face => {
                let cells = self.face_cells(v, hit);
                self.apply_colored(v, &cells)
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
        let Some(s) = self.stroke.as_mut() else { return false };
        s.ray1 = (ro, rd);
        let (anchor, axis) = (s.anchor, s.axis);
        let marquee = (self.tool == Tool::Marquee).then_some(self.opts.marquee);
        // Attach draws on the plane of the first voxel; erase, paint and
        // brush select follow whatever surface is under the cursor.
        let planar = matches!(self.brush, Brush::Box | Brush::Geo | Brush::Code) || (self.brush == Brush::Voxel && self.tool == Tool::Attach);
        let cell = match marquee {
            Some(MarqueeMode::Box) => plane_cell(ro, rd, axis, anchor[axis]),
            Some(_) => hit.and_then(|h| target(Tool::Paint, h)),
            None if planar => plane_cell(ro, rd, axis, anchor[axis]),
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

    /// Mouse release: commit the stroke to the undo history.
    pub fn end(&mut self, v: &mut Volume) -> bool {
        if self.moving.is_some() {
            return self.end_move(v);
        }
        let pending = self.stroke.as_ref().map(|s| self.pending_cells(v, s));
        let changed = match pending {
            Some(cells) if !cells.is_empty() => {
                if self.tool == Tool::Marquee {
                    // Box select: only voxels inside the box.
                    let solid: Vec<IVec3> = cells.into_iter().filter(|&c| v.get(c) != 0).collect();
                    self.apply(v, &solid)
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
    }

    /// Take the history so it can move into the tree's: the volume before
    /// each of the last `max` edits, oldest first, and the volume after
    /// each of up to `max` undone edits, next redo first (`v` is the volume
    /// now).
    pub fn take_history(&mut self, v: &Volume, max: usize) -> (Vec<Volume>, Vec<Volume>) {
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
        let Some(entry) = self.undo.pop() else { return false };
        let entry = Self::step(v, entry, false);
        self.redo.push(entry);
        self.fit_selection(v);
        true
    }

    pub fn redo(&mut self, v: &mut Volume) -> bool {
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
    fn selection_masks_paint() {
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
        assert_eq!((v.get(IVec3::new(1, 1, 0)), v.get(IVec3::new(2, 1, 0))), (9, 4));
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
        for (shape, fill, expect) in [(GeoShape::Square, true, 25), (GeoShape::Square, false, 16), (GeoShape::Circle, true, 21)] {
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
}
