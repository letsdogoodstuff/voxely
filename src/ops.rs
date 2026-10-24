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
