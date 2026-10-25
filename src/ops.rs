//! Whole-model edits from MagicaVoxel's modeler "Edit" panel. Each takes
//! the current volume and returns the edited one; `sel` limits an edit to
//! selected cells where that makes sense.

use std::collections::VecDeque;

use glam::IVec3;

use crate::scene::{Mask, Volume};

/// Largest canvas edge the tools will create.
pub const MAX_DIM: i32 = 1024;

const NEIGHBORS: [IVec3; 6] = [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::Z, IVec3::NEG_Z];

fn selected(sel: Option<&Mask>, i: usize) -> bool {
    sel.is_none_or(|s| s[i])
}

fn map(v: &Volume, mut f: impl FnMut(usize, u8) -> u8) -> Volume {
    let mut out = Volume::new(v.dims);
    for (i, &c) in v.cells.iter().enumerate() {
        out.set_index(i, f(i, c));
    }
    out
}

/// Fill every empty cell [U].
pub fn full(v: &Volume, color: u8) -> Volume {
    map(v, |_, c| if c == 0 { color } else { c })
}

/// Recolor voxels [I].
pub fn fill(v: &Volume, sel: Option<&Mask>, color: u8) -> Volume {
    map(v, |i, c| if c != 0 && selected(sel, i) { color } else { c })
}

/// Remove voxels [Delete].
pub fn clear(v: &Volume, sel: Option<&Mask>) -> Volume {
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
    for (p, c) in v.filled() {
        for k in 0..8 {
            out.set(p * 2 + IVec3::new(k & 1, (k >> 1) & 1, k >> 2), c);
        }
    }
    Some(out)
}

/// Scale down by two; each 2x2x2 block keeps its first voxel in x, then
/// y, then z order, like MagicaVoxel.
pub fn half(v: &Volume) -> Volume {
    let dims = (v.dims + IVec3::ONE) / 2;
    let mut out = Volume::new(dims);
    for i in 0..out.cells.len() {
        let p = out.position(i);
        if let Some(c) = (0..8).map(|k| v.get(p * 2 + IVec3::new(k & 1, (k >> 1) & 1, k >> 2))).find(|&c| c != 0) {
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

/// Change the canvas size (see `resize_shift` for where the model goes).
pub fn resize(v: &Volume, dims: IVec3) -> Volume {
    let dims = dims.clamp(IVec3::ONE, IVec3::splat(MAX_DIM));
    let shift = resize_shift(v, dims);
    let mut out = Volume::new(dims);
    for (i, &c) in v.cells.iter().enumerate() {
        if c != 0 {
            out.set(v.position(i) + shift, c);
        }
    }
    out
}

/// Where `resize` puts the old cell 0 in the new canvas: the old canvas
/// centered in x and y, on the same floor. Along an axis where that would
/// cut voxels off, the model is centered instead (x, y) or set on the
/// floor (z).
pub fn resize_shift(v: &Volume, dims: IVec3) -> IVec3 {
    let dims = dims.clamp(IVec3::ONE, IVec3::splat(MAX_DIM));
    let mut shift = IVec3::new((dims.x - v.dims.x) / 2, (dims.y - v.dims.y) / 2, 0);
    let mut b: Option<(IVec3, IVec3)> = None;
    for (p, _) in v.filled() {
        b = Some(b.map_or((p, p), |(lo, hi)| (lo.min(p), hi.max(p))));
    }
    if let Some((lo, hi)) = b {
        for k in 0..3 {
            if lo[k] + shift[k] < 0 || hi[k] + shift[k] >= dims[k] {
                shift[k] = if k == 2 { -lo[k] } else { (dims[k] - (hi[k] - lo[k] + 1)).div_euclid(2) - lo[k] };
            }
        }
    }
    shift
}

/// Shrink the canvas to the model's bounds (MagicaVoxel's crop/fit).
pub fn fit(v: &Volume) -> Option<Volume> {
    crop(v).map(|(out, _)| out)
}

/// `fit`, plus the old cell the new cell 0 was (the crop's min corner).
pub fn crop(v: &Volume) -> Option<(Volume, IVec3)> {
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
    Some((out, lo))
}

/// Shift by one voxel along `axis`, wrapping around the canvas.
pub fn loop_shift(v: &Volume, sel: Option<&Mask>, axis: usize, dir: i32) -> Volume {
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
    /// Square base on the floor, apex at the top of the canvas.
    #[allow(dead_code)]
    Pyramid,
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
            Shape::Pyramid => q.x.abs().max(q.y.abs()) <= (1.0 - q.z) * 0.5,
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
        let mut cells: Vec<usize> = v.cells.filled().filter(|&(_, c)| c == color).map(|(i, _)| i).collect();
        cells.sort_unstable();
        return cells;
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

/// Tiny deterministic xorshift32 generator (no `rand` dependency).
#[allow(dead_code)]
pub struct Rng(u32);

#[allow(dead_code)]
impl Rng {
    pub fn new(seed: u32) -> Self {
        // Scramble so nearby seeds diverge quickly; xorshift must not be 0.
        let s = seed.wrapping_mul(0x9E37_79B9) ^ 0x2545_F491;
        let mut r = Rng(if s == 0 { 0x1234_5678 } else { s });
        for _ in 0..4 {
            r.next();
        }
        r
    }

    pub fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Uniform in `0..n` (n > 0).
    pub fn below(&mut self, n: u32) -> u32 {
        ((self.next() as u64 * n as u64) >> 32) as u32
    }
}

/// Parse MagicaVoxel's axis text fields such as "xyz 2", "z 0.2 0.2",
/// "z 45" or "xy 2": one or more axis letters (case-insensitive, any
/// order) followed by at least one number. Returns the axes mask and the
/// numbers.
#[allow(dead_code)]
pub fn parse_axes_args(s: &str) -> Option<([bool; 3], Vec<f32>)> {
    let s = s.trim();
    let split = s.find(|c: char| !c.is_ascii_alphabetic()).unwrap_or(s.len());
    let (letters, rest) = s.split_at(split);
    let mut axes = [false; 3];
    for c in letters.chars() {
        match c.to_ascii_lowercase() {
            'x' => axes[0] = true,
            'y' => axes[1] = true,
            'z' => axes[2] = true,
            _ => return None,
        }
    }
    if !axes.iter().any(|&a| a) {
        return None;
    }
    let nums = rest
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<f32>().ok().filter(|n| n.is_finite()))
        .collect::<Option<Vec<f32>>>()?;
    if nums.is_empty() {
        return None;
    }
    Some((axes, nums))
}

/// Scale along `axes` by `factor` with nearest-neighbor resampling; the
/// canvas is resized to match. None if an edge would round to 0 or
/// exceed `MAX_DIM`.
#[allow(dead_code)]
pub fn scale(v: &Volume, axes: [bool; 3], factor: f32) -> Option<Volume> {
    if !(factor.is_finite() && factor > 0.0) {
        return None;
    }
    let mut dims = v.dims;
    for a in 0..3 {
        if axes[a] {
            let d = (v.dims[a] as f64 * factor as f64).round();
            if d < 1.0 || d > MAX_DIM as f64 {
                return None;
            }
            dims[a] = d as i32;
        }
    }
    let mut out = Volume::new(dims);
    for i in 0..out.cells.len() {
        let q = out.position(i);
        // Source cell under the destination cell's center.
        let mut p = IVec3::ZERO;
        for a in 0..3 {
            p[a] = (((2 * q[a] + 1) as i64 * v.dims[a] as i64) / (2 * dims[a] as i64)) as i32;
        }
        let c = v.get(p);
        if c != 0 {
            out.set_index(i, c);
        }
    }
    Some(out)
}

/// Shear the two axes other than `axis` as a function of the `axis`
/// coordinate: for `axis` = z, x += a*z and y += b*z (the next axis in
/// x→y→z order takes `a`, the one after takes `b`). Each slice moves as a
/// whole, so no holes appear; the canvas grows to fit (up to `MAX_DIM`).
#[allow(dead_code)]
pub fn shear(v: &Volume, axis: usize, a: f32, b: f32) -> Volume {
    let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
    let n = v.dims[axis];
    let off = |k: f32, t: i32| (k * t as f32).round() as i32;
    let range = |k: f32| {
        let (lo, hi) = (off(k, 0), off(k, n - 1));
        (lo.min(hi), lo.max(hi))
    };
    let (ru, rw) = (range(a), range(b));
    let mut dims = v.dims;
    dims[u] = (v.dims[u] + ru.1 - ru.0).min(MAX_DIM);
    dims[w] = (v.dims[w] + rw.1 - rw.0).min(MAX_DIM);
    let mut out = Volume::new(dims);
    for (i, &c) in v.cells.iter().enumerate() {
        if c != 0 {
            let p = v.position(i);
            let mut q = p;
            q[u] += off(a, p[axis]) - ru.0;
            q[w] += off(b, p[axis]) - rw.0;
            out.set(q, c);
        }
    }
    out
}

/// Rotate by an arbitrary angle about the axis through the canvas center,
/// in the same direction as `rotate` for positive angles (so 90 degrees
/// matches `rotate` exactly). Each destination cell samples the nearest
/// source cell (inverse mapping, so no holes); the canvas grows to hold
/// the rotated canvas bounds (up to `MAX_DIM`).
#[allow(dead_code)]
pub fn rotate_angle(v: &Volume, axis: usize, degrees: f32) -> Volume {
    let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
    let t = (degrees as f64).to_radians();
    let snap = |x: f64| {
        let r = x.round();
        if (x - r).abs() < 1e-9 { r } else { x }
    };
    let (s, c) = (snap(t.sin()), snap(t.cos()));
    let (du, dw) = (v.dims[u] as f64, v.dims[w] as f64);
    let extent = |x: f64| ((x - 1e-6).ceil() as i32).clamp(1, MAX_DIM);
    let mut dims = v.dims;
    dims[u] = extent(c.abs() * du + s.abs() * dw);
    dims[w] = extent(s.abs() * du + c.abs() * dw);
    let (nu, nw) = (dims[u] as f64, dims[w] as f64);
    let mut out = Volume::new(dims);
    for i in 0..out.cells.len() {
        let q = out.position(i);
        let x = q[u] as f64 + 0.5 - nu * 0.5;
        let y = q[w] as f64 + 0.5 - nw * 0.5;
        // Inverse rotation back into the source canvas.
        let mut p = q;
        p[u] = (c * x + s * y + du * 0.5).floor() as i32;
        p[w] = (-s * x + c * y + dw * 0.5).floor() as i32;
        let col = v.get(p);
        if col != 0 {
            out.set_index(i, col);
        }
    }
    out
}

/// Tile the model `count` times along each of `axes` (count = total
/// copies, so "xy 2" gives a 2x2 grid). None if count is 0 or the canvas
/// would exceed `MAX_DIM`.
#[allow(dead_code)]
pub fn repeat(v: &Volume, axes: [bool; 3], count: u32) -> Option<Volume> {
    if count == 0 {
        return None;
    }
    let mut dims = v.dims;
    for a in 0..3 {
        if axes[a] {
            let d = v.dims[a] as i64 * count as i64;
            if d > MAX_DIM as i64 {
                return None;
            }
            dims[a] = d as i32;
        }
    }
    let mut out = Volume::new(dims);
    for i in 0..out.cells.len() {
        let c = v.get(out.position(i) % v.dims);
        if c != 0 {
            out.set_index(i, c);
        }
    }
    Some(out)
}

/// Diagonal symmetry, MagicaVoxel's "Diagonal X/Y/Z". For `axis`, take the
/// other two axes (u, w) = (axis+1, axis+2) mod 3 and union the model with
/// its transpose across the plane u = w (which contains `axis`); e.g.
/// Diagonal Z adds the copy with x and y swapped, making the model
/// symmetric about the x = y diagonal seen from above. The canvas becomes
/// square in u and w (edge = the larger of the two), anchored at the
/// origin so the diagonal runs through the (0, 0) corner. Existing voxels
/// win where the copy overlaps them.
#[allow(dead_code)]
pub fn diagonal(v: &Volume, axis: usize) -> Volume {
    let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
    let n = v.dims[u].max(v.dims[w]);
    let mut dims = v.dims;
    dims[u] = n;
    dims[w] = n;
    let mut out = Volume::new(dims);
    for (i, &c) in v.cells.iter().enumerate() {
        if c != 0 {
            let p = v.position(i);
            let mut q = p;
            q[u] = p[w];
            q[w] = p[u];
            if out.get(q) == 0 {
                out.set(q, c);
            }
        }
    }
    for (i, &c) in v.cells.iter().enumerate() {
        if c != 0 {
            out.set(v.position(i), c);
        }
    }
    out
}

/// A maze on the floor (x-y plane): a grid of cells `edge` voxels apart
/// (passages `edge - 1` wide, walls one voxel thick; `edge` is clamped to
/// at least 2), carved with a randomized depth-first search so every
/// passage is reachable, with an entrance on the -x side of the first
/// cell and an exit on the +x side of the last. Walls are extruded the
/// full canvas height and centered in x and y; they fill only empty
/// cells. Deterministic for a given canvas size and edge.
#[allow(dead_code)]
pub fn maze(v: &Volume, color: u8, edge: u32) -> Volume {
    let e = edge.clamp(2, MAX_DIM as u32) as i32;
    let (nx, ny) = ((v.dims.x - 1) / e, (v.dims.y - 1) / e);
    if nx < 1 || ny < 1 {
        return v.clone();
    }
    let (wx, wy) = (nx * e + 1, ny * e + 1);
    let mut wall: Vec<bool> = (0..wx * wy).map(|k| (k % wx) % e == 0 || (k / wx) % e == 0).collect();
    // Knock down the wall between neighboring cells.
    let join = |wall: &mut [bool], (ax, ay): (i32, i32), (bx, by): (i32, i32)| {
        for k in 1..e {
            let (x, y) = if ax != bx { (ax.max(bx) * e, ay * e + k) } else { (ax * e + k, ay.max(by) * e) };
            wall[(x + wx * y) as usize] = false;
        }
    };
    let mut rng = Rng::new((nx as u32) << 20 ^ (ny as u32) << 8 ^ e as u32);
    let mut seen = vec![false; (nx * ny) as usize];
    let mut stack = vec![(0, 0)];
    seen[0] = true;
    while let Some(&(cx, cy)) = stack.last() {
        let mut options = [(0, 0); 4];
        let mut k = 0;
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (x, y) = (cx + dx, cy + dy);
            if x >= 0 && y >= 0 && x < nx && y < ny && !seen[(x + nx * y) as usize] {
                options[k] = (x, y);
                k += 1;
            }
        }
        if k == 0 {
            stack.pop();
            continue;
        }
        let next = options[rng.below(k as u32) as usize];
        seen[(next.0 + nx * next.1) as usize] = true;
        join(&mut wall, (cx, cy), next);
        stack.push(next);
    }
    join(&mut wall, (-1, 0), (0, 0));
    join(&mut wall, (nx - 1, ny - 1), (nx, ny - 1));
    let shift = IVec3::new((v.dims.x - wx) / 2, (v.dims.y - wy) / 2, 0);
    map(v, |i, c| {
        if c != 0 {
            return c;
        }
        let p = v.position(i) - shift;
        let inside = p.x >= 0 && p.y >= 0 && p.x < wx && p.y < wy;
        if inside && wall[(p.x + wx * p.y) as usize] { color } else { 0 }
    })
}

/// Ken Perlin's improved gradient noise with a seeded permutation.
struct Perlin {
    perm: [u8; 512],
}

impl Perlin {
    fn new(seed: u32) -> Self {
        let mut p: [u8; 256] = std::array::from_fn(|i| i as u8);
        let mut rng = Rng::new(seed);
        for i in (1..256).rev() {
            p.swap(i, rng.below(i as u32 + 1) as usize);
        }
        Perlin { perm: std::array::from_fn(|i| p[i & 255]) }
    }

    fn noise(&self, x: f32, y: f32, z: f32) -> f32 {
        fn fade(t: f32) -> f32 {
            t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
        }
        fn grad(h: u8, x: f32, y: f32, z: f32) -> f32 {
            let h = h & 15;
            let u = if h < 8 { x } else { y };
            let v = if h < 4 { y } else if h == 12 || h == 14 { x } else { z };
            (if h & 1 == 0 { u } else { -u }) + (if h & 2 == 0 { v } else { -v })
        }
        let lerp = |t: f32, a: f32, b: f32| a + t * (b - a);
        let (fx, fy, fz) = (x.floor(), y.floor(), z.floor());
        let (xi, yi, zi) = ((fx as i32 & 255) as usize, (fy as i32 & 255) as usize, (fz as i32 & 255) as usize);
        let (x, y, z) = (x - fx, y - fy, z - fz);
        let (u, v, w) = (fade(x), fade(y), fade(z));
        let p = &self.perm;
        let a = p[xi] as usize + yi;
        let (aa, ab) = (p[a] as usize + zi, p[a + 1] as usize + zi);
        let b = p[xi + 1] as usize + yi;
        let (ba, bb) = (p[b] as usize + zi, p[b + 1] as usize + zi);
        lerp(
            w,
            lerp(
                v,
                lerp(u, grad(p[aa], x, y, z), grad(p[ba], x - 1.0, y, z)),
                lerp(u, grad(p[ab], x, y - 1.0, z), grad(p[bb], x - 1.0, y - 1.0, z)),
            ),
            lerp(
                v,
                lerp(u, grad(p[aa + 1], x, y, z - 1.0), grad(p[ba + 1], x - 1.0, y, z - 1.0)),
                lerp(u, grad(p[ab + 1], x, y - 1.0, z - 1.0), grad(p[bb + 1], x - 1.0, y - 1.0, z - 1.0)),
            ),
        )
    }
}

/// Perlin-noise terrain filling empty cells: four octaves of 3D gradient
/// noise (base period a third of the larger horizontal canvas edge, at
/// least 8 voxels) added to a falloff with height, so the result is solid
/// near the floor, has rolling hills with occasional overhangs around
/// mid-height and is empty near the top. Deterministic for a given seed.
#[allow(dead_code)]
pub fn perlin(v: &Volume, color: u8, seed: u32) -> Volume {
    let noise = Perlin::new(seed);
    let period = (v.dims.x.max(v.dims.y) as f32 / 3.0).max(8.0);
    let f = 1.0 / period;
    // Random offset so seeds also differ away from the lattice origin.
    let mut rng = Rng::new(seed ^ 0xA5A5_5A5A);
    let o = [0, 1, 2].map(|_| rng.below(1 << 16) as f32 / 256.0);
    map(v, |i, c| {
        if c != 0 {
            return c;
        }
        let p = v.position(i).as_vec3() + 0.5;
        let (mut sum, mut amp, mut freq) = (0.0, 1.0, f);
        for _ in 0..4 {
            sum += amp * noise.noise(p.x * freq + o[0], p.y * freq + o[1], p.z * freq + o[2]);
            amp *= 0.5;
            freq *= 2.0;
        }
        let height = p.z / v.dims.z as f32;
        if sum * 0.5 + 0.45 - height > 0.0 { color } else { 0 }
    })
}

// ---- MagicaVoxel's Edit panel, as the GUI buttons do it ----
//
// The functions above keep the behavior the MCP tools document. The ones
// below match MagicaVoxel 0.99.6 exactly, for the Edit panel: with a
// selection, an edit acts on the selected voxels as a floating layer (they
// are lifted out, edited, and put back over the rest, and the result is the
// new selection).

/// Every cell becomes `color`: MagicaVoxel's Full [U] overwrites the model.
pub fn full_volume(v: &Volume, color: u8) -> Volume {
    map(v, |_, _| color)
}

/// Mirror: the low half along `axis` is copied, reflected, over the high
/// half (the middle slice of an odd canvas stays).
pub fn mirror_half(v: &Volume, axis: usize) -> Volume {
    let n = v.dims[axis];
    map(v, |i, c| {
        let mut p = v.position(i);
        if p[axis] > n - 1 - p[axis] {
            p[axis] = n - 1 - p[axis];
            v.get(p)
        } else {
            c
        }
    })
}

/// Mirror a lifted selection: it gets its reflection appended on the high
/// side of `axis`, doubling it.
pub fn mirror_extend(v: &Volume, axis: usize) -> Option<Volume> {
    let mut dims = v.dims;
    dims[axis] *= 2;
    if dims[axis] > MAX_DIM {
        return None;
    }
    let mut out = Volume::new(dims);
    for (p, c) in v.filled() {
        let mut q = p;
        q[axis] = dims[axis] - 1 - p[axis];
        out.set(p, c);
        out.set(q, c);
    }
    Some(out)
}

/// Diagonal: with (u, w) the other two axes after `axis`, the canvas
/// becomes square in u and w and the cells with w > u take their
/// transposed cell, so the side below the u = w diagonal is copied over
/// the side above it.
pub fn diagonal_half(v: &Volume, axis: usize) -> Volume {
    let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
    let n = v.dims[u].max(v.dims[w]);
    let mut dims = v.dims;
    dims[u] = n;
    dims[w] = n;
    let mut out = Volume::new(dims);
    for i in 0..out.cells.len() {
        let mut p = out.position(i);
        if p[w] > p[u] {
            (p[u], p[w]) = (p[w], p[u]);
        }
        let c = v.get(p);
        if c != 0 {
            out.set_index(i, c);
        }
    }
    out
}

const BOX: [IVec3; 26] = {
    let mut out = [IVec3::ZERO; 26];
    let mut k = 0;
    let mut i = 0;
    while i < 27 {
        if i != 13 {
            out[k] = IVec3::new(i % 3 - 1, (i / 3) % 3 - 1, i / 9 - 1);
            k += 1;
        }
        i += 1;
    }
    out
};

/// Dilation: every empty cell touching a voxel (also across edges and
/// corners) gets `color`.
pub fn dilate_box(v: &Volume, color: u8) -> Volume {
    map(v, |i, c| {
        let p = v.position(i);
        if c != 0 || BOX.iter().all(|&n| v.get(p + n) == 0) { c } else { color }
    })
}

/// Erosion: a voxel stays only if all 26 cells around it are solid (the
/// outside of the canvas is empty).
pub fn erode_box(v: &Volume) -> Volume {
    map(v, |i, c| {
        let p = v.position(i);
        if c != 0 && BOX.iter().all(|&n| v.get(p + n) != 0) { c } else { 0 }
    })
}

/// Fractal: every voxel becomes a copy of the pattern in its own color, so
/// the canvas grows by the pattern's size. None past `MAX_DIM`.
pub fn fractal(v: &Volume, pattern: &Volume) -> Option<Volume> {
    let dims = v.dims * pattern.dims;
    if dims.max_element() > MAX_DIM {
        return None;
    }
    let mut out = Volume::new(dims);
    for (p, c) in v.filled() {
        for (q, _) in pattern.filled() {
            out.set(p * pattern.dims + q, c);
        }
    }
    Some(out)
}

/// The pattern's cell over `p` when it is tiled from the canvas origin.
fn tiled(pattern: &Volume, p: IVec3) -> u8 {
    pattern.get(IVec3::new(p.x.rem_euclid(pattern.dims.x), p.y.rem_euclid(pattern.dims.y), p.z.rem_euclid(pattern.dims.z)))
}

/// Mask: keep only the voxels under the tiled pattern's solid cells.
pub fn mask_pattern(v: &Volume, pattern: &Volume) -> Volume {
    map(v, |i, c| if c != 0 && tiled(pattern, v.position(i)) != 0 { c } else { 0 })
}

/// Tex: voxels under the tiled pattern's solid cells take its colors.
pub fn texture(v: &Volume, pattern: &Volume) -> Volume {
    map(v, |i, c| match tiled(pattern, v.position(i)) {
        t if c != 0 && t != 0 => t,
        _ => c,
    })
}

/// The selected voxels alone, in a canvas the size of `v`.
pub fn layer(v: &Volume, sel: &Mask) -> Volume {
    map(v, |i, c| if sel[i] { c } else { 0 })
}

/// `v` with its selected voxels taken away and `out`'s voxels put over the
/// rest; `out`'s voxels are the new selection.
pub fn merge(v: &Volume, sel: &Mask, out: &Volume) -> (Volume, Mask) {
    let mut mask = Mask::new(v.dims);
    let new = map(v, |i, c| match out.cells[i] {
        0 if sel[i] => 0,
        0 => c,
        o => {
            mask.set(i, true);
            o
        }
    });
    (new, mask)
}

/// Every voxel of `v`, as a selection.
pub fn layer_mask(v: &Volume) -> Mask {
    let mut m = Mask::new(v.dims);
    for (i, _) in v.cells.filled() {
        m.set(i, true);
    }
    m
}

/// The selected voxels cut to their bounding box, and its min corner.
pub fn lift(v: &Volume, sel: &Mask) -> Option<(Volume, IVec3)> {
    crop(&layer(v, sel))
}

/// Where a box of `new` cells goes to stay centered on a box of `old` cells
/// at `lo`, rounded like MagicaVoxel.
pub fn centered(lo: IVec3, old: IVec3, new: IVec3) -> IVec3 {
    lo + (old - IVec3::ONE) / 2 - (new - IVec3::ONE) / 2
}

/// `v` without its selection, with `sub` pasted over it at `at` (clipped
/// to the canvas); the pasted voxels are the new selection.
pub fn place(v: &Volume, sel: &Mask, sub: &Volume, at: IVec3) -> (Volume, Mask) {
    let mut out = Volume::new(v.dims);
    for (p, c) in sub.filled() {
        out.set(p + at, c);
    }
    merge(v, sel, &out)
}

/// `sub` at `at` in a canvas grown to hold both it and the old canvas of
/// `dims` (the old cell 0 lands at the returned offset). None past `MAX_DIM`.
pub fn place_grow(dims: IVec3, sub: &Volume, at: IVec3) -> Option<(Volume, IVec3)> {
    let lo = at.min(IVec3::ZERO);
    let hi = (at + sub.dims).max(dims);
    if (hi - lo).max_element() > MAX_DIM {
        return None;
    }
    let mut out = Volume::new(hi - lo);
    for (p, c) in sub.filled() {
        out.set(p + at - lo, c);
    }
    Some((out, -lo))
}

/// Run a box-shaped edit (rotate, flip, 2X, ...) on the lifted selection
/// and put the result back centered where it was.
pub fn on_lifted(v: &Volume, sel: &Mask, f: impl FnOnce(&Volume) -> Option<Volume>) -> Option<(Volume, Mask)> {
    let (sub, lo) = lift(v, sel)?;
    let new = f(&sub)?;
    Some(place(v, sel, &new, centered(lo, sub.dims, new.dims)))
}

/// Scale the model's voxels about their center, nearest cell; the canvas
/// grows when they no longer fit but never shrinks. None if an edge would
/// round to 0 or the canvas would exceed `MAX_DIM`.
pub fn scale_content(v: &Volume, axes: [bool; 3], factor: f32) -> Option<Volume> {
    let (sub, lo) = crop(v)?;
    let new = scale_floor(&sub, axes, factor)?;
    place_grow(v.dims, &new, centered(lo, sub.dims, new.dims)).map(|(out, _)| out)
}

/// `scale` with MagicaVoxel's sampling: destination cell q takes source
/// cell floor(q * old / new).
pub fn scale_floor(v: &Volume, axes: [bool; 3], factor: f32) -> Option<Volume> {
    let dims = scale(&Volume::new(v.dims), axes, factor)?.dims;
    let mut out = Volume::new(dims);
    for i in 0..out.cells.len() {
        let q = out.position(i);
        let p = (q.as_i64vec3() * v.dims.as_i64vec3() / dims.as_i64vec3()).as_ivec3();
        let c = v.get(p);
        if c != 0 {
            out.set_index(i, c);
        }
    }
    Some(out)
}

/// Shear in place: for `axis` = z, x += floor(a*z) and y += floor(b*z)
/// (the next axes in x→y→z order take a and b). Only `sel`'s voxels move
/// when given; the canvas keeps its size and voxels pushed out are lost.
pub fn shear_clip(v: &Volume, sel: Option<&Mask>, axis: usize, a: f32, b: f32) -> Volume {
    let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
    let mut out = match sel {
        Some(_) => clear(v, sel),
        None => Volume::new(v.dims),
    };
    for (i, &c) in v.cells.iter().enumerate() {
        if c != 0 && selected(sel, i) {
            let p = v.position(i);
            let mut q = p;
            q[u] += (a * p[axis] as f32).floor() as i32;
            q[w] += (b * p[axis] as f32).floor() as i32;
            if v.contains(q) {
                out.set(q, c);
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
        // Kept on the old floor its top would be cut, so it moves down.
        assert_eq!(count(&r), 27);
        // Too small for the model: centered on it, the rest cut off.
        let r = resize(&v, IVec3::new(1, 9, 9));
        assert_eq!((count(&r), r.get(IVec3::new(0, 3, 2))), (9, 1));
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

#[cfg(test)]
mod transform_tests {
    use super::*;

    fn count(v: &Volume) -> usize {
        v.cells.iter().filter(|&&c| c != 0).count()
    }

    fn sample() -> Volume {
        let mut v = Volume::new(IVec3::new(5, 3, 4));
        v.set(IVec3::new(0, 0, 0), 1);
        v.set(IVec3::new(4, 2, 3), 2);
        v.set(IVec3::new(2, 1, 1), 3);
        v.set(IVec3::new(1, 2, 0), 4);
        v
    }

    #[test]
    fn parser_cases() {
        assert_eq!(parse_axes_args("xyz 2"), Some(([true; 3], vec![2.0])));
        assert_eq!(parse_axes_args("z 0.2 0.2"), Some(([false, false, true], vec![0.2, 0.2])));
        assert_eq!(parse_axes_args("  Z 45 "), Some(([false, false, true], vec![45.0])));
        assert_eq!(parse_axes_args("xy 2"), Some(([true, true, false], vec![2.0])));
        assert_eq!(parse_axes_args("z-30"), Some(([false, false, true], vec![-30.0])));
        assert_eq!(parse_axes_args("x 1,2"), Some(([true, false, false], vec![1.0, 2.0])));
        assert_eq!(parse_axes_args("2"), None);
        assert_eq!(parse_axes_args("xw 2"), None);
        assert_eq!(parse_axes_args("x"), None);
        assert_eq!(parse_axes_args("x abc"), None);
        assert_eq!(parse_axes_args(""), None);
    }

    #[test]
    fn scale_matches_double_and_shrinks() {
        let v = sample();
        let s = scale(&v, [true; 3], 2.0).unwrap();
        assert_eq!(s.dims, v.dims * 2);
        assert_eq!(s.cells, double(&v).unwrap().cells);
        let z = scale(&s, [false, false, true], 0.5).unwrap();
        assert_eq!(z.dims, IVec3::new(10, 6, 4));
        assert_eq!(count(&z), count(&v) * 4);
        let back = scale(&s, [true; 3], 0.5).unwrap();
        assert_eq!(back.cells, v.cells);
        assert!(scale(&v, [true; 3], 0.01).is_none());
        assert!(scale(&v, [true; 3], 1000.0).is_none());
        assert!(scale(&v, [true; 3], -1.0).is_none());
    }

    #[test]
    fn shear_moves_slices() {
        let mut v = Volume::new(IVec3::new(1, 1, 6));
        for z in 0..6 {
            v.set(IVec3::new(0, 0, z), 1);
        }
        let s = shear(&v, 2, 1.0, -0.4);
        assert_eq!(s.dims, IVec3::new(6, 3, 6));
        assert_eq!(count(&s), 6);
        for z in 0..6 {
            let y = 2 + (-0.4 * z as f32).round() as i32;
            assert_eq!(s.get(IVec3::new(z, y, z)), 1, "z={z}");
        }
        assert_eq!(shear(&v, 2, 0.0, 0.0).cells, v.cells);
    }

    #[test]
    fn rotate_angle_matches_rotate() {
        let v = sample();
        for axis in 0..3 {
            let r = rotate_angle(&v, axis, 90.0);
            let q = rotate(&v, axis);
            assert_eq!(r.dims, q.dims);
            assert_eq!(r.cells, q.cells);
            assert_eq!(rotate_angle(&v, axis, 360.0).cells, v.cells);
            assert_eq!(rotate_angle(&v, axis, 0.0).cells, v.cells);
        }
        let r = rotate_angle(&v, 2, -90.0);
        assert_eq!(r.cells, rotate(&rotate(&rotate(&v, 2), 2), 2).cells);
    }

    #[test]
    fn rotate_angle_45_grows_canvas_without_holes() {
        let mut v = Volume::new(IVec3::new(10, 10, 2));
        v.cells.fill(1);
        let r = rotate_angle(&v, 2, 45.0);
        assert_eq!(r.dims, IVec3::new(15, 15, 2));
        let n = count(&r) as f32;
        assert!((n - 200.0).abs() < 30.0, "{n}");
        assert_eq!(r.get(IVec3::new(7, 7, 0)), 1);
        assert_eq!(r.get(IVec3::new(0, 0, 0)), 0);
    }

    #[test]
    fn repeat_tiles() {
        let v = sample();
        let r = repeat(&v, [true, true, false], 2).unwrap();
        assert_eq!(r.dims, IVec3::new(10, 6, 4));
        assert_eq!(count(&r), count(&v) * 4);
        assert_eq!(r.get(IVec3::new(9, 5, 3)), 2);
        assert_eq!(r.get(IVec3::new(5, 3, 0)), 1);
        let z = repeat(&v, [false, false, true], 3).unwrap();
        assert_eq!(z.dims.z, 12);
        assert_eq!(count(&z), count(&v) * 3);
        assert_eq!(repeat(&v, [true; 3], 1).unwrap().cells, v.cells);
        assert!(repeat(&v, [true; 3], 0).is_none());
        assert!(repeat(&v, [true; 3], 300).is_none());
    }

    #[test]
    fn diagonal_transposes_other_axes() {
        let mut v = Volume::new(IVec3::new(4, 2, 1));
        v.set(IVec3::new(3, 0, 0), 5);
        v.set(IVec3::new(1, 1, 0), 6);
        let d = diagonal(&v, 2);
        assert_eq!(d.dims, IVec3::new(4, 4, 1));
        assert_eq!(d.get(IVec3::new(3, 0, 0)), 5);
        assert_eq!(d.get(IVec3::new(0, 3, 0)), 5);
        assert_eq!(d.get(IVec3::new(1, 1, 0)), 6);
        assert_eq!(count(&d), 3);
        // The result is symmetric, so applying it again changes nothing.
        assert_eq!(diagonal(&d, 2).cells, d.cells);
        let x = diagonal(&sample(), 0);
        assert_eq!(x.dims, IVec3::new(5, 4, 4));
        for (i, &c) in x.cells.iter().enumerate() {
            let p = x.position(i);
            assert_eq!(c != 0, x.get(IVec3::new(p.x, p.z, p.y)) != 0);
        }
    }

    #[test]
    fn pyramid_shape() {
        let s = shape(&Volume::new(IVec3::splat(10)), Shape::Pyramid, 1);
        let layer = |z: i32| (0..100).filter(|&k| s.get(IVec3::new(k % 10, k / 10, z)) != 0).count();
        assert_eq!(layer(0), 100);
        assert!(layer(5) < layer(2));
        assert!(layer(8) > 0);
        assert_eq!(layer(9), 0);
    }

    #[test]
    fn maze_is_connected_and_deterministic() {
        let v = Volume::new(IVec3::new(21, 17, 3));
        let m = maze(&v, 9, 4);
        assert_eq!(m.cells, maze(&v, 9, 4).cells);
        let (wx, wy) = (21, 17);
        for k in 0..wx * wy {
            let c = m.get(IVec3::new(k % wx, k / wx, 0));
            assert_eq!(c, m.get(IVec3::new(k % wx, k / wx, 2)));
        }
        // 5x4 cells of edge 4 fill the 21x17 floor exactly.
        let floor: Vec<bool> = (0..wx * wy).map(|k| m.get(IVec3::new(k % wx, k / wx, 0)) == 0).collect();
        let open = floor.iter().filter(|&&f| f).count();
        // 20 rooms of 3x3, plus 19 internal openings and the entrance and
        // exit, each 3 voxels: a perfect maze (spanning tree).
        assert_eq!(open, 20 * 9 + 21 * 3);
        let start = (0..wy).find(|&y| floor[(y * wx) as usize]).unwrap();
        let mut seen = vec![false; floor.len()];
        let mut stack = vec![(0, start)];
        seen[(start * wx) as usize] = true;
        let mut reached = 0;
        while let Some((x, y)) = stack.pop() {
            reached += 1;
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let (a, b) = (x + dx, y + dy);
                if a >= 0 && b >= 0 && a < wx && b < wy {
                    let j = (a + wx * b) as usize;
                    if floor[j] && !seen[j] {
                        seen[j] = true;
                        stack.push((a, b));
                    }
                }
            }
        }
        assert_eq!(reached, open);
        assert_eq!(maze(&Volume::new(IVec3::new(2, 2, 2)), 1, 4).cells, vec![0; 8]);
    }

    #[test]
    fn perlin_terrain() {
        let v = Volume::new(IVec3::new(32, 32, 24));
        let a = perlin(&v, 3, 1);
        assert_eq!(a.cells, perlin(&v, 3, 1).cells);
        assert_ne!(a.cells, perlin(&v, 3, 2).cells);
        let n = count(&a);
        assert!(n > v.cells.len() / 5 && n < v.cells.len() * 4 / 5, "{n}");
        let layer = |z: i32| (0..1024).filter(|&k| a.get(IVec3::new(k % 32, k / 32, z)) != 0).count();
        assert!(layer(0) > layer(23));
        assert!(a.cells.iter().all(|&c| c == 0 || c == 3));
    }
}

/// The Edit panel functions against results read off MagicaVoxel 0.99.6.2.
#[cfg(test)]
mod mv_tests {
    use super::*;

    fn vol(dims: [i32; 3], cells: &[([i32; 3], u8)]) -> Volume {
        let mut v = Volume::new(IVec3::from(dims));
        for &(p, c) in cells {
            v.set(IVec3::from(p), c);
        }
        v
    }

    fn cells(v: &Volume) -> Vec<([i32; 3], u8)> {
        let mut out: Vec<_> = v.filled().map(|(p, c)| (p.to_array(), c)).collect();
        out.sort();
        out
    }

    /// An L of color 2 on a floor of color 1, selected (MagicaVoxel's
    /// region select on the L).
    fn floor_and_l() -> (Volume, Mask) {
        let mut v = Volume::new(IVec3::new(8, 6, 3));
        for y in 0..6 {
            for x in 0..8 {
                v.set(IVec3::new(x, y, 0), 1);
            }
        }
        let mut sel = Mask::new(v.dims);
        for p in [[1, 1, 1], [2, 1, 1], [3, 1, 1], [1, 2, 1]] {
            v.set(IVec3::from(p), 2);
            sel.set(v.index(IVec3::from(p)), true);
        }
        (v, sel)
    }

    /// The color-2 cells of the z = `z` layer.
    fn l_cells(v: &Volume, z: i32) -> Vec<[i32; 2]> {
        let mut out: Vec<_> = v.filled().filter(|&(p, c)| c == 2 && p.z == z).map(|(p, _)| [p.x, p.y]).collect();
        out.sort();
        out
    }

    #[test]
    fn half_keeps_each_blocks_first_voxel() {
        // Blocks along x: the first voxel in x, then y, then z order wins
        // over the most common color.
        let v = vol(
            [10, 2, 2],
            &[
                ([1, 0, 0], 1), ([0, 1, 0], 2), ([0, 0, 1], 3),
                ([2, 1, 0], 2), ([2, 0, 1], 3),
                ([5, 0, 0], 1), ([4, 0, 1], 3),
                ([7, 0, 0], 1), ([6, 1, 0], 2), ([6, 1, 1], 2), ([7, 1, 1], 2),
                ([9, 1, 1], 1), ([8, 1, 1], 2), ([9, 0, 1], 3),
            ],
        );
        let h = half(&v);
        assert_eq!(h.dims, IVec3::new(5, 1, 1));
        assert_eq!(h.cells, vec![1, 2, 1, 1, 3]);
    }

    #[test]
    fn full_overwrites_and_mirror_copies_the_low_half() {
        let v = vol([5, 3, 1], &[([0, 0, 0], 1), ([1, 1, 0], 2), ([2, 0, 0], 3), ([2, 2, 0], 3), ([4, 1, 0], 4), ([3, 2, 0], 4)]);
        assert!(full_volume(&v, 9).cells.iter().all(|&c| c == 9));
        // The high half is replaced; the middle column stays.
        assert_eq!(cells(&mirror_half(&v, 0)), vec![([0, 0, 0], 1), ([1, 1, 0], 2), ([2, 0, 0], 3), ([2, 2, 0], 3), ([3, 1, 0], 2), ([4, 0, 0], 1)]);
        assert_eq!(
            cells(&mirror_half(&v, 1)),
            vec![([0, 0, 0], 1), ([0, 2, 0], 1), ([1, 1, 0], 2), ([2, 0, 0], 3), ([2, 2, 0], 3), ([4, 1, 0], 4)]
        );
    }

    #[test]
    fn diagonal_copies_below_the_diagonal_over_above() {
        let v = vol([5, 3, 1], &[([0, 0, 0], 1), ([1, 1, 0], 2), ([2, 0, 0], 3), ([2, 2, 0], 3), ([4, 1, 0], 4), ([3, 2, 0], 4), ([0, 1, 0], 5)]);
        let d = diagonal_half(&v, 2);
        assert_eq!(d.dims, IVec3::new(5, 5, 1));
        // (0, 1) is above the diagonal, so it takes (1, 0), which is empty.
        assert_eq!(
            cells(&d),
            vec![([0, 0, 0], 1), ([0, 2, 0], 3), ([1, 1, 0], 2), ([1, 4, 0], 4), ([2, 0, 0], 3), ([2, 2, 0], 3), ([2, 3, 0], 4), ([3, 2, 0], 4), ([4, 1, 0], 4)]
        );
    }

    #[test]
    fn dilate_and_erode_use_the_whole_box() {
        let d = dilate_box(&vol([7, 7, 3], &[([3, 3, 0], 1)]), 5);
        assert_eq!(d.count(), 18);
        assert_eq!((d.get(IVec3::new(2, 2, 1)), d.get(IVec3::new(3, 3, 0))), (5, 1));
        let mut cube = Volume::new(IVec3::splat(5));
        for i in 0..cube.cells.len() {
            let p = cube.position(i);
            if p.cmpge(IVec3::ONE).all() && p.cmple(IVec3::splat(3)).all() && p != IVec3::ONE {
                cube.set_index(i, 1);
            }
        }
        // A missing corner neighbor is enough to erode the center.
        assert_eq!(erode_box(&cube).count(), 0);
        assert_eq!(erode_box(&full_volume(&cube, 1)).count(), 27);
    }

    #[test]
    fn selection_edits_work_on_the_lifted_voxels() {
        let (v, sel) = floor_and_l();
        let rot = on_lifted(&v, &sel, |s| Some(rotate(s, 2))).unwrap().0;
        assert_eq!(l_cells(&rot, 1), vec![[2, 0], [3, 0], [3, 1], [3, 2]]);
        assert_eq!(rot.get(IVec3::new(1, 1, 0)), 1, "the floor stays");

        let (dbl, mask) = on_lifted(&v, &sel, double).unwrap();
        assert_eq!((l_cells(&dbl, 0), l_cells(&dbl, 1).len()), (vec![], 16));
        assert_eq!(l_cells(&dbl, 2), l_cells(&dbl, 1));
        assert_eq!(l_cells(&dbl, 1)[0], [0, 0]);
        assert_eq!(mask.filled().count(), 32);

        let hlf = on_lifted(&v, &sel, |s| Some(half(s))).unwrap().0;
        assert_eq!(l_cells(&hlf, 1), vec![[2, 1], [3, 1]]);

        let rep = on_lifted(&v, &sel, |s| repeat(s, [true, false, false], 2)).unwrap().0;
        assert_eq!(l_cells(&rep, 1), vec![[0, 1], [0, 2], [1, 1], [2, 1], [3, 1], [3, 2], [4, 1], [5, 1]]);

        let sc = on_lifted(&v, &sel, |s| scale_floor(s, [true, false, false], 2.0)).unwrap().0;
        assert_eq!(l_cells(&sc, 1), vec![[0, 1], [0, 2], [1, 1], [1, 2], [2, 1], [3, 1], [4, 1], [5, 1]]);

        let dia = on_lifted(&v, &sel, |s| Some(diagonal_half(s, 2))).unwrap().0;
        assert_eq!(l_cells(&dia, 1), vec![[1, 0], [1, 1], [1, 2], [2, 0], [3, 0]]);

        // Mirror grows the selection on its high side instead.
        let (sub, lo) = lift(&v, &sel).unwrap();
        let mir = place(&v, &sel, &mirror_extend(&sub, 0).unwrap(), lo).0;
        assert_eq!(l_cells(&mir, 1), vec![[1, 1], [1, 2], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [6, 2]]);
    }

    #[test]
    fn layer_edits_replace_the_selection() {
        let (v, sel) = floor_and_l();
        // Rev with a selection: the selected voxels go, everything else
        // becomes the color and is selected.
        let (rev, mask) = merge(&v, &sel, &reverse(&layer(&v, &sel), 7));
        assert_eq!(rev.get(IVec3::new(1, 1, 1)), 0);
        assert_eq!((rev.get(IVec3::new(0, 0, 0)), rev.count()), (7, 8 * 6 * 3 - 4));
        assert_eq!(mask.filled().count(), rev.count());
    }

    #[test]
    fn pattern_edits() {
        let mut menger = Volume::new(IVec3::splat(3));
        for i in 0..27 {
            let p = menger.position(i);
            if p.cmpeq(IVec3::ONE).bitmask().count_ones() < 2 {
                menger.set_index(i, 9);
            }
        }
        let f = fractal(&menger, &menger).unwrap();
        assert_eq!((f.dims, f.count()), (IVec3::splat(9), 400));
        let block = full_volume(&Volume::new(IVec3::new(7, 5, 4)), 1);
        let m = mask_pattern(&block, &menger);
        assert_eq!((m.get(IVec3::new(1, 1, 0)), m.get(IVec3::new(4, 4, 0)), m.get(IVec3::new(0, 0, 0))), (0, 0, 1));
        let t = texture(&block, &menger);
        assert_eq!((t.get(IVec3::new(1, 1, 0)), t.get(IVec3::new(0, 0, 0))), (1, 9));
    }

    #[test]
    fn shear_and_scale_like_magicavoxel() {
        let v = vol([8, 3, 4], &[([2, 1, 0], 1), ([2, 1, 1], 2), ([2, 1, 2], 3), ([2, 1, 3], 4)]);
        // Offsets round down; the canvas keeps its size.
        let s = shear_clip(&v, None, 2, 0.5, 0.0);
        assert_eq!(cells(&s), vec![([2, 1, 0], 1), ([2, 1, 1], 2), ([3, 1, 2], 3), ([3, 1, 3], 4)]);
        let s = shear_clip(&v, None, 2, -1.0, 0.0);
        assert_eq!(cells(&s), vec![([0, 1, 2], 3), ([1, 1, 1], 2), ([2, 1, 0], 1)]);
        // Scaling grows the canvas only when the voxels don't fit.
        let probe = full_volume(&Volume::new(IVec3::new(6, 4, 3)), 1);
        assert_eq!(scale_content(&probe, [true; 3], 2.0).unwrap().dims, IVec3::new(12, 8, 6));
        let shrunk = scale_content(&probe, [true, false, false], 0.5).unwrap();
        assert_eq!((shrunk.dims, shrunk.count()), (probe.dims, 3 * 4 * 3));
    }
}
