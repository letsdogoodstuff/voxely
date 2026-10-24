//! Whole-model edits from MagicaVoxel's modeler "Edit" panel. Each takes
//! the current volume and returns the edited one; `sel` limits an edit to
//! selected cells where that makes sense.

use std::collections::VecDeque;

use glam::IVec3;

use crate::scene::Volume;

/// Largest canvas edge the tools will create.
pub const MAX_DIM: i32 = 1024;

const NEIGHBORS: [IVec3; 6] = [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::Z, IVec3::NEG_Z];

fn selected(sel: Option<&[bool]>, i: usize) -> bool {
    sel.is_none_or(|s| s[i])
}

fn map(v: &Volume, f: impl Fn(usize, u8) -> u8) -> Volume {
    Volume::from_cells(v.dims, v.cells.iter().enumerate().map(|(i, &c)| f(i, c)).collect())
}

/// Fill every empty cell [U].
pub fn full(v: &Volume, color: u8) -> Volume {
    map(v, |_, c| if c == 0 { color } else { c })
}

/// Recolor voxels [I].
pub fn fill(v: &Volume, sel: Option<&[bool]>, color: u8) -> Volume {
    map(v, |i, c| if c != 0 && selected(sel, i) { color } else { c })
}

/// Remove voxels [Delete].
pub fn clear(v: &Volume, sel: Option<&[bool]>) -> Volume {
    map(v, |i, c| if selected(sel, i) { 0 } else { c })
}

/// Swap empty and solid space.
pub fn reverse(v: &Volume, color: u8) -> Volume {
    map(v, |_, c| if c == 0 { color } else { 0 })
}

/// Scale up by two in every axis.
pub fn double(v: &Volume) -> Option<Volume> {
    let dims = v.dims * 2;
    if dims.max_element() > MAX_DIM {
        return None;
    }
    let mut out = Volume::new(dims);
    for (i, &c) in v.cells.iter().enumerate() {
        if c != 0 {
            let p = v.position(i) * 2;
            for k in 0..8 {
                out.set(p + IVec3::new(k & 1, (k >> 1) & 1, k >> 2), c);
            }
        }
    }
    Some(out)
}

/// Scale down by two; each 2x2x2 block keeps its most common color, so
/// thin details survive.
pub fn half(v: &Volume) -> Volume {
    let dims = (v.dims + IVec3::ONE) / 2;
    let mut out = Volume::new(dims);
    for i in 0..out.cells.len() {
        let p = out.position(i);
        let mut counts: Vec<(u8, u32)> = Vec::new();
        for k in 0..8 {
            let c = v.get(p * 2 + IVec3::new(k & 1, (k >> 1) & 1, k >> 2));
            if c != 0 {
                match counts.iter_mut().find(|e| e.0 == c) {
                    Some(e) => e.1 += 1,
                    None => counts.push((c, 1)),
                }
            }
        }
        if let Some(&(c, _)) = counts.iter().max_by_key(|e| e.1) {
            out.set_index(i, c);
        }
    }
    out
}

pub fn flip(v: &Volume, axis: usize) -> Volume {
    let mut out = Volume::new(v.dims);
    for (i, &c) in v.cells.iter().enumerate() {
        if c != 0 {
            let mut p = v.position(i);
            p[axis] = v.dims[axis] - 1 - p[axis];
            out.set(p, c);
        }
    }
    out
}

/// Rotate 90 degrees about `axis`; the canvas turns with the model.
pub fn rotate(v: &Volume, axis: usize) -> Volume {
    let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
    let mut dims = v.dims;
    dims[u] = v.dims[w];
    dims[w] = v.dims[u];
    let mut out = Volume::new(dims);
    for (i, &c) in v.cells.iter().enumerate() {
        if c != 0 {
            let p = v.position(i);
            let mut q = p;
            q[u] = v.dims[w] - 1 - p[w];
            q[w] = p[u];
            out.set(q, c);
        }
    }
    out
}

/// Empty cells connected to the outside of the canvas.
fn outside_air(v: &Volume) -> Vec<bool> {
    let mut out = vec![false; v.cells.len()];
    let mut queue = VecDeque::new();
    for (i, &c) in v.cells.iter().enumerate() {
        let p = v.position(i);
        let edge = p.cmpeq(IVec3::ZERO).any() || p.cmpeq(v.dims - IVec3::ONE).any();
        if c == 0 && edge {
            out[i] = true;
            queue.push_back(p);
        }
    }
    while let Some(p) = queue.pop_front() {
        for n in NEIGHBORS {
            let q = p + n;
            if v.contains(q) {
                let j = v.index(q);
                if !out[j] && v.cells[j] == 0 {
                    out[j] = true;
                    queue.push_back(q);
                }
            }
        }
    }
    out
}

/// Remove voxels that cannot be seen from outside.
pub fn hull(v: &Volume) -> Volume {
    let air = outside_air(v);
    let open = |p: IVec3| !v.contains(p) || air[v.index(p)];
    map(v, |i, c| {
        let p = v.position(i);
        if c != 0 && NEIGHBORS.iter().any(|&n| open(p + n)) { c } else { 0 }
    })
}

/// Fill enclosed empty space.
pub fn flood(v: &Volume, color: u8) -> Volume {
    let air = outside_air(v);
    map(v, |i, c| if c == 0 && !air[i] { color } else { c })
}

/// Grow by one voxel; new cells take a neighbor's color.
pub fn dilate(v: &Volume) -> Volume {
    map(v, |i, c| {
        if c != 0 {
            return c;
        }
        let p = v.position(i);
        NEIGHBORS.iter().map(|&n| v.get(p + n)).find(|&n| n != 0).unwrap_or(0)
    })
}

/// Shrink by one voxel.
pub fn erode(v: &Volume) -> Volume {
    map(v, |i, c| {
        let p = v.position(i);
        if c != 0 && NEIGHBORS.iter().all(|&n| v.get(p + n) != 0) { c } else { 0 }
    })
}

/// Change the canvas size, keeping the model centered in x and y and on
/// the floor.
pub fn resize(v: &Volume, dims: IVec3) -> Volume {
    let dims = dims.clamp(IVec3::ONE, IVec3::splat(MAX_DIM));
    let shift = IVec3::new((dims.x - v.dims.x) / 2, (dims.y - v.dims.y) / 2, 0);
    let mut out = Volume::new(dims);
    for (i, &c) in v.cells.iter().enumerate() {
        if c != 0 {
            out.set(v.position(i) + shift, c);
        }
    }
    out
}

/// Shrink the canvas to the model's bounds (MagicaVoxel's crop/fit).
pub fn fit(v: &Volume) -> Option<Volume> {
    let mut b: Option<(IVec3, IVec3)> = None;
    for (i, &c) in v.cells.iter().enumerate() {
        if c != 0 {
            let p = v.position(i);
            b = Some(b.map_or((p, p), |(lo, hi)| (lo.min(p), hi.max(p))));
        }
    }
    let (lo, hi) = b?;
    let mut out = Volume::new(hi - lo + IVec3::ONE);
    for (i, &c) in v.cells.iter().enumerate() {
        if c != 0 {
            out.set(v.position(i) - lo, c);
        }
    }
    Some(out)
}

/// Shift by one voxel along `axis`, wrapping around the canvas.
pub fn loop_shift(v: &Volume, sel: Option<&[bool]>, axis: usize, dir: i32) -> Volume {
    let mut out = match sel {
        Some(_) => clear(v, sel),
        None => Volume::new(v.dims),
    };
    for (i, &c) in v.cells.iter().enumerate() {
        if c != 0 && selected(sel, i) {
            let mut p = v.position(i);
            p[axis] = (p[axis] + dir).rem_euclid(v.dims[axis]);
            out.set(p, c);
        }
    }
    out
}

/// Make the model symmetric: add its reflection across the middle of `axis`.
pub fn mirror(v: &Volume, axis: usize) -> Volume {
    let flipped = flip(v, axis);
    map(v, |i, c| if c != 0 { c } else { flipped.cells[i] })
}

/// Solids the Shape panel can add, inscribed in the canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Ellipsoid,
    Cylinder,
    Cone,
}

pub fn shape(v: &Volume, kind: Shape, color: u8) -> Volume {
    let half = v.dims.as_vec3() * 0.5;
    map(v, |i, c| {
        if c != 0 {
            return c;
        }
        // Cell center relative to the canvas center, scaled to [-1, 1].
        let q = (v.position(i).as_vec3() + 0.5 - half) / half;
        let r2 = q.x * q.x + q.y * q.y;
        let inside = match kind {
            Shape::Ellipsoid => r2 + q.z * q.z <= 1.0,
            Shape::Cylinder => r2 <= 1.0,
            Shape::Cone => r2.sqrt() <= (1.0 - q.z) * 0.5,
        };
        if inside { color } else { 0 }
    })
}

/// How the Region tools grow from the clicked voxel, after MagicaVoxel's
/// Region options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionOptions {
    /// Every voxel of the clicked color, connected or not (the default).
    pub same_color: bool,
    /// When connected: only through voxels of the same color, rather than
    /// any solid voxel ("Geo").
    pub by_color: bool,
    /// When connected: also step across edges and corners (8-neighbor
    /// rather than 4-neighbor searching).
    pub diagonal: bool,
}

impl Default for RegionOptions {
    fn default() -> Self {
        RegionOptions { same_color: true, by_color: true, diagonal: false }
    }
}

/// Cells of the region containing the voxel at `start`.
pub fn region(v: &Volume, start: IVec3, o: RegionOptions) -> Vec<usize> {
    let color = v.get(start);
    if color == 0 {
        return Vec::new();
    }
    if o.same_color {
        return (0..v.cells.len()).filter(|&i| v.cells[i] == color).collect();
    }
    let steps: Vec<IVec3> = if o.diagonal {
        (0..27).map(|k| IVec3::new(k % 3, (k / 3) % 3, k / 9) - IVec3::ONE).filter(|d| *d != IVec3::ZERO).collect()
    } else {
        NEIGHBORS.to_vec()
    };
    let joins = |c: u8| if o.by_color { c == color } else { c != 0 };
    let mut seen = vec![false; v.cells.len()];
    let mut queue = VecDeque::from([start]);
    seen[v.index(start)] = true;
    let mut out = Vec::new();
    while let Some(p) = queue.pop_front() {
        out.push(v.index(p));
        for &n in &steps {
            let q = p + n;
            if v.contains(q) {
                let j = v.index(q);
                if !seen[j] && joins(v.cells[j]) {
                    seen[j] = true;
                    queue.push_back(q);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_cube(n: i32, pad: i32) -> Volume {
        let mut v = Volume::new(IVec3::splat(n + 2 * pad));
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    v.set(IVec3::new(x, y, z) + pad, 1);
                }
            }
        }
        v
    }

    fn count(v: &Volume) -> usize {
        v.cells.iter().filter(|&&c| c != 0).count()
    }

    #[test]
    fn hull_and_flood_are_inverse_on_a_cube() {
        let v = solid_cube(4, 1);
        let h = hull(&v);
        assert_eq!(count(&h), 64 - 8);
        assert_eq!(count(&flood(&h, 2)), 64);
    }

    #[test]
    fn rotate_four_times_is_identity() {
        let mut v = Volume::new(IVec3::new(3, 5, 2));
        v.set(IVec3::new(0, 4, 1), 7);
        v.set(IVec3::new(2, 0, 0), 3);
        for axis in 0..3 {
            let r = rotate(&v, axis);
            assert_ne!(r.dims, v.dims);
            let back = rotate(&rotate(&rotate(&r, axis), axis), axis);
            assert_eq!(back.dims, v.dims);
            assert_eq!(back.cells, v.cells);
        }
    }

    #[test]
    fn double_then_half_round_trips() {
        let v = solid_cube(2, 1);
        let d = double(&v).unwrap();
        assert_eq!(d.dims, v.dims * 2);
        assert_eq!(count(&d), 8 * 8);
        assert_eq!(half(&d).cells, v.cells);
    }

    #[test]
    fn dilate_erode_and_resize() {
        let v = solid_cube(3, 2);
        assert_eq!(count(&erode(&v)), 1);
        assert_eq!(count(&dilate(&v)), 27 + 6 * 9);
        let r = resize(&v, IVec3::new(9, 9, 4));
        assert_eq!(r.dims, IVec3::new(9, 9, 4));
        assert_eq!(count(&r), 18);
    }

    #[test]
    fn fit_loop_mirror_shape() {
        let mut v = Volume::new(IVec3::new(10, 10, 10));
        v.set(IVec3::new(2, 3, 4), 1);
        v.set(IVec3::new(4, 3, 5), 2);
        let f = fit(&v).unwrap();
        assert_eq!(f.dims, IVec3::new(3, 1, 2));
        assert_eq!(f.get(IVec3::new(2, 0, 1)), 2);

        let l = loop_shift(&f, None, 0, 1);
        assert_eq!(l.get(IVec3::new(0, 0, 1)), 2);
        assert_eq!(l.get(IVec3::new(1, 0, 0)), 1);

        let m = mirror(&v, 0);
        assert_eq!(m.get(IVec3::new(7, 3, 4)), 1);
        assert_eq!(count(&m), 4);

        let s = shape(&Volume::new(IVec3::splat(8)), Shape::Ellipsoid, 3);
        assert!(count(&s) > 200 && count(&s) < 300);
    }

    #[test]
    fn region_follows_color() {
        let mut v = solid_cube(2, 0);
        v.set(IVec3::new(1, 1, 1), 5);
        let connected = RegionOptions { same_color: false, ..Default::default() };
        assert_eq!(region(&v, IVec3::ZERO, connected).len(), 7);
        assert_eq!(region(&v, IVec3::ZERO, RegionOptions { by_color: false, ..connected }).len(), 8);

        // Two same-colored voxels touching only at a corner.
        let mut d = Volume::new(IVec3::splat(4));
        d.set(IVec3::ZERO, 3);
        d.set(IVec3::ONE, 3);
        assert_eq!(region(&d, IVec3::ZERO, connected).len(), 1);
        assert_eq!(region(&d, IVec3::ZERO, RegionOptions { diagonal: true, ..connected }).len(), 2);
        assert_eq!(region(&d, IVec3::ZERO, RegionOptions::default()).len(), 2);
    }
}
