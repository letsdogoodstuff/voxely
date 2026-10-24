//! Voxel editing, modelled on MagicaVoxel's brushes: Attach / Erase /
//! Paint tools combined with Voxel / Face / Box / Line brushes, axis
//! mirroring, and undo/redo per stroke.

use std::collections::{HashMap, HashSet, VecDeque};

use glam::{IVec3, Vec3};

use crate::scene::Volume;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tool {
    Attach,
    Erase,
    Paint,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Brush {
    /// One voxel per click; drag to draw.
    Voxel,
    /// The connected same-color face region under the cursor.
    Face,
    /// Box between press and release.
    Box,
    /// Straight line between press and release.
    Line,
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
fn face_region(v: &Volume, p: Pick) -> Vec<IVec3> {
    const LIMIT: usize = 1 << 20;
    let axis = axis_of(p.normal);
    let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
    let color = v.get(p.cell);
    let inside = |q: IVec3| {
        if p.on_ground() {
            // Ground: the empty floor cells of the bottom layer.
            q.x >= 0 && q.y >= 0 && q.x < v.dims.x && q.y < v.dims.y && v.get(q + p.normal) == 0
        } else {
            v.get(q) == color && v.get(q + p.normal) == 0
        }
    };
    let mut seen = HashSet::from([p.cell]);
    let mut queue = VecDeque::from([p.cell]);
    let mut out = Vec::new();
    while let Some(q) = queue.pop_front() {
        out.push(q);
        if out.len() >= LIMIT {
            break;
        }
        for (ax, d) in [(a, 1), (a, -1), (b, 1), (b, -1)] {
            let mut n = q;
            n[ax] += d;
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
}

pub struct Editor {
    pub tool: Tool,
    pub brush: Brush,
    pub color: u8,
    pub mirror: [bool; 3],
    undo: Vec<Vec<(usize, u8, u8)>>,
    redo: Vec<Vec<(usize, u8, u8)>>,
    stroke: Option<Stroke>,
}

impl Editor {
    pub fn new(color: u8) -> Self {
        Editor { tool: Tool::Attach, brush: Brush::Voxel, color, mirror: [false; 3], undo: Vec::new(), redo: Vec::new(), stroke: None }
    }

    pub fn active(&self) -> bool {
        self.stroke.is_some()
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

    /// Apply the tool to cells; returns whether anything changed.
    fn apply(&mut self, v: &mut Volume, cells: &[IVec3]) -> bool {
        let all: Vec<IVec3> = cells.iter().flat_map(|&c| self.mirrored(v, c)).collect();
        let (tool, color) = (self.tool, self.color);
        let Some(stroke) = self.stroke.as_mut() else { return false };
        let mut changed = false;
        for c in all {
            if !v.contains(c) {
                continue;
            }
            let i = v.index(c);
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

    /// Mouse press. Returns whether the volume changed.
    pub fn begin(&mut self, v: &mut Volume, hit: Pick) -> bool {
        let Some(t) = target(self.tool, hit) else { return false };
        let axis = axis_of(hit.normal);
        self.stroke = Some(Stroke { changes: HashMap::new(), anchor: t, axis, current: t });
        match self.brush {
            Brush::Voxel => self.apply(v, &[t]),
            Brush::Face => {
                let region = face_region(v, hit);
                let cells: Vec<IVec3> = match self.tool {
                    Tool::Attach => region.iter().map(|&c| c + hit.normal).collect(),
                    _ if hit.on_ground() => Vec::new(),
                    _ => region,
                };
                self.apply(v, &cells)
            }
            // Applied on release; the overlay shows the preview meanwhile.
            Brush::Box | Brush::Line => false,
        }
    }

    /// Mouse drag with the camera ray. Returns whether the volume changed.
    pub fn drag(&mut self, v: &mut Volume, ro: Vec3, rd: Vec3, hit: Option<Pick>) -> bool {
        let Some(s) = self.stroke.as_ref() else { return false };
        let (anchor, axis) = (s.anchor, s.axis);
        // Attach draws on the plane of the first voxel; erase and paint
        // follow whatever surface is under the cursor.
        let cell = match (self.brush, self.tool) {
            (Brush::Voxel, Tool::Attach) | (Brush::Box | Brush::Line, _) => plane_cell(ro, rd, axis, anchor[axis]),
            _ => hit.and_then(|h| target(self.tool, h)),
        };
        let Some(cell) = cell else { return false };
        let cell = cell.clamp(IVec3::ZERO, v.dims - IVec3::ONE);
        let s = self.stroke.as_mut().unwrap();
        if cell == s.current {
            return false;
        }
        let prev = s.current;
        s.current = cell;
        match self.brush {
            // Fill the gap from the previous cell so fast drags stay connected.
            Brush::Voxel => self.apply(v, &line(prev, cell)),
            _ => false,
        }
    }

    /// Mouse release: commit the stroke to the undo history.
    pub fn end(&mut self, v: &mut Volume) -> bool {
        let changed = match (self.brush, self.stroke.as_ref()) {
            (Brush::Box, Some(s)) => {
                let cells = cuboid(s.anchor, s.current);
                self.apply(v, &cells)
            }
            (Brush::Line, Some(s)) => {
                let cells = line(s.anchor, s.current);
                self.apply(v, &cells)
            }
            _ => false,
        };
        if let Some(s) = self.stroke.take() {
            let mut entries: Vec<(usize, u8, u8)> = s.changes.into_iter().filter(|(_, (a, b))| a != b).map(|(i, (a, b))| (i, a, b)).collect();
            if !entries.is_empty() {
                entries.sort_unstable_by_key(|e| e.0);
                self.undo.push(entries);
                self.redo.clear();
            }
        }
        changed
    }

    pub fn undo(&mut self, v: &mut Volume) -> bool {
        let Some(entries) = self.undo.pop() else { return false };
        for &(i, old, _) in &entries {
            v.set_index(i, old);
        }
        self.redo.push(entries);
        true
    }

    pub fn redo(&mut self, v: &mut Volume) -> bool {
        let Some(entries) = self.redo.pop() else { return false };
        for &(i, _, new) in &entries {
            v.set_index(i, new);
        }
        self.undo.push(entries);
        true
    }

    /// Cells to outline: the pending box/line, or what a click would touch.
    pub fn preview(&self, v: &Volume, hover: Option<Pick>) -> Vec<IVec3> {
        let cells = match (&self.stroke, self.brush) {
            (Some(s), Brush::Box) => cuboid(s.anchor, s.current),
            (Some(s), Brush::Line) => line(s.anchor, s.current),
            (Some(_), _) => Vec::new(),
            (None, Brush::Face) => match hover {
                Some(h) => {
                    let r = face_region(v, h);
                    if self.tool == Tool::Attach { r.iter().map(|&c| c + h.normal).collect() } else if h.on_ground() { Vec::new() } else { r }
                }
                None => Vec::new(),
            },
            (None, _) => hover.and_then(|h| target(self.tool, h)).into_iter().collect(),
        };
        cells.into_iter().flat_map(|c| self.mirrored(v, c)).filter(|&c| v.contains(c)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vol() -> Volume {
        Volume::new(IVec3::new(8, 8, 8))
    }

    #[test]
    fn pick_ground_then_attach_and_undo() {
        let mut v = vol();
        let mut e = Editor::new(5);
        let hit = pick(&v, Vec3::new(2.5, 3.5, 20.0), Vec3::new(0.0, 0.0, -1.0)).unwrap();
        assert_eq!(hit, Pick { cell: IVec3::new(2, 3, -1), normal: IVec3::Z });
        assert!(e.begin(&mut v, hit));
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
        assert!(!e.begin(&mut v, hit));
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
        e.begin(&mut v, hit);
        e.end(&mut v);
        assert_eq!(v.get(IVec3::new(1, 1, 0)), 3);
        assert_eq!(v.get(IVec3::new(6, 1, 0)), 3);
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
        e.begin(&mut v, hit);
        e.end(&mut v);
        assert_eq!(v.cells.iter().filter(|&&c| c == 9).count(), 16);
    }
}
