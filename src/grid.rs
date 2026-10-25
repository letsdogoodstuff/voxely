//! Sparse cell storage: a table of 8³ bricks over a box, each brick either
//! absent (every cell the default value) or a block in a pool. Cells keep
//! their dense index (x fastest, then y, then z), so code can index a grid
//! like a flat array, but empty space costs 4 bytes per brick instead of a
//! value per cell.

use std::fmt;
use std::ops::{Index, IndexMut};

use glam::IVec3;

/// Edge length of a brick.
pub const BRICK: i32 = 8;
const B: usize = BRICK as usize;
/// Cells in a brick.
pub const BRICK_CELLS: usize = B * B * B;

#[derive(Clone)]
pub struct Grid<T> {
    dims: IVec3,
    bdims: IVec3,
    /// Per brick: 0 when absent, else its block in `pool` plus one.
    slots: Vec<u32>,
    /// Blocks of `BRICK_CELLS` cells, x fastest within a brick.
    pool: Vec<T>,
    /// The default value; absent bricks read as this.
    zero: T,
    /// A brick row of the default value, for iterating absent bricks.
    zrow: [T; B],
}

impl<T: Copy + Default + PartialEq> Grid<T> {
    pub fn new(dims: IVec3) -> Self {
        let dims = dims.max(IVec3::ZERO);
        let bdims = (dims + IVec3::splat(BRICK - 1)) / BRICK;
        Grid { dims, bdims, slots: vec![0; (bdims.x * bdims.y * bdims.z) as usize], pool: Vec::new(), zero: T::default(), zrow: [T::default(); B] }
    }

    /// A grid holding `cells` (dense, x fastest).
    pub fn from_vec(dims: IVec3, cells: &[T]) -> Self {
        let mut g = Grid::new(dims);
        assert_eq!(cells.len(), g.len(), "cell count doesn't match the grid size");
        // Copy brick-row segments, skipping empty ones.
        let (dx, dy) = (dims.x as usize, dims.y as usize);
        for z in 0..dims.z {
            for y in 0..dims.y {
                let row = &cells[dx * (y as usize + dy * z as usize)..][..dx];
                for x0 in (0..dims.x).step_by(B) {
                    let seg = &row[x0 as usize..(x0 as usize + B).min(dx)];
                    if seg.iter().all(|&c| c == g.zero) {
                        continue;
                    }
                    let (b, o) = g.locate_at(IVec3::new(x0, y, z));
                    let s = match g.slots[b] {
                        0 => g.alloc(b),
                        s => s as usize - 1,
                    };
                    g.pool[s * BRICK_CELLS + o..][..seg.len()].copy_from_slice(seg);
                }
            }
        }
        g
    }

    pub fn dims(&self) -> IVec3 {
        self.dims
    }

    /// Bricks along each axis.
    pub fn brick_dims(&self) -> IVec3 {
        self.bdims
    }

    /// Number of cells (as if dense).
    pub fn len(&self) -> usize {
        (self.dims.x * self.dims.y * self.dims.z) as usize
    }

    /// Bytes of storage in use: the brick table plus the pool.
    pub fn memory(&self) -> usize {
        self.slots.len() * 4 + self.pool.len() * std::mem::size_of::<T>()
    }

    /// Number of bricks with storage.
    pub fn allocated(&self) -> usize {
        self.pool.len() / BRICK_CELLS
    }

    /// Brick table index and offset within the brick of dense index `i`.
    #[inline]
    fn locate(&self, i: usize) -> (usize, usize) {
        let i = i as i32;
        let (x, y, z) = (i % self.dims.x, (i / self.dims.x) % self.dims.y, i / (self.dims.x * self.dims.y));
        self.locate_at(IVec3::new(x, y, z))
    }

    /// Brick table index and offset within the brick of cell `p` (inside).
    #[inline]
    fn locate_at(&self, p: IVec3) -> (usize, usize) {
        let b = ((p.x >> 3) + self.bdims.x * ((p.y >> 3) + self.bdims.y * (p.z >> 3))) as usize;
        (b, ((p.x & 7) | (p.y & 7) << 3 | (p.z & 7) << 6) as usize)
    }

    /// The value at cell `p`, which must be inside the grid.
    #[inline]
    pub fn at(&self, p: IVec3) -> T {
        let (b, o) = self.locate_at(p);
        match self.slots[b] {
            0 => self.zero,
            s => self.pool[(s as usize - 1) * BRICK_CELLS + o],
        }
    }

    /// Set cell `p`, which must be inside the grid.
    #[inline]
    pub fn set_at(&mut self, p: IVec3, v: T) {
        let (b, o) = self.locate_at(p);
        self.write(b, o, v);
    }

    #[inline]
    fn write(&mut self, b: usize, o: usize, v: T) {
        let s = self.slots[b];
        if s != 0 {
            self.pool[(s as usize - 1) * BRICK_CELLS + o] = v;
        } else if v != self.zero {
            let s = self.alloc(b);
            self.pool[s * BRICK_CELLS + o] = v;
        }
    }

    /// The value at dense index `i`, or `None` past the end.
    #[inline]
    pub fn get(&self, i: usize) -> Option<T> {
        (i < self.len()).then(|| self[i])
    }

    /// Whether any cell holds `v` (for the default: any cell at all).
    pub fn contains(&self, v: &T) -> bool {
        if *v == self.zero { self.iter().any(|c| c == v) } else { self.pool.contains(v) }
    }

    /// Set dense index `i`. Writing the default into an absent brick
    /// allocates nothing.
    #[inline]
    pub fn set(&mut self, i: usize, v: T) {
        let (b, o) = self.locate(i);
        self.write(b, o, v);
    }

    /// Give brick `b` storage, returning its block.
    fn alloc(&mut self, b: usize) -> usize {
        self.pool.resize(self.pool.len() + BRICK_CELLS, self.zero);
        let s = self.pool.len() / BRICK_CELLS - 1;
        self.slots[b] = s as u32 + 1;
        s
    }

    /// Set every cell to `v`.
    pub fn fill(&mut self, v: T) {
        self.slots.fill(0);
        self.pool.clear();
        if v != self.zero {
            for b in 0..self.slots.len() {
                self.alloc(b);
            }
            self.pool.fill(v);
        }
    }

    /// Release bricks holding only the default value.
    pub fn compact(&mut self) {
        let mut pool = Vec::new();
        for s in self.slots.iter_mut().filter(|s| **s != 0) {
            let block = &self.pool[(*s as usize - 1) * BRICK_CELLS..][..BRICK_CELLS];
            if block.iter().all(|&c| c == self.zero) {
                *s = 0;
            } else {
                pool.extend_from_slice(block);
                *s = (pool.len() / BRICK_CELLS) as u32;
            }
        }
        self.pool = pool;
    }

    /// Change every non-default cell in place.
    pub fn update_filled(&mut self, mut f: impl FnMut(&mut T)) {
        let zero = self.zero;
        self.pool.iter_mut().filter(|c| **c != zero).for_each(|c| f(c));
    }

    /// Cells that differ from `other` (same size) as (dense index, value
    /// here, value there), brick by brick. Skips bricks neither stores.
    pub fn diff(&self, other: &Self) -> Vec<(usize, T, T)> {
        assert_eq!(self.dims, other.dims, "diff of grids of different sizes");
        let mut out = Vec::new();
        let (dx, dxy) = (self.dims.x as usize, (self.dims.x * self.dims.y) as usize);
        for b in 0..self.slots.len() {
            let (sa, sb) = (self.slots[b], other.slots[b]);
            if sa == 0 && sb == 0 {
                continue;
            }
            let block = |g: &Self, s: u32, o: usize| if s == 0 { g.zero } else { g.pool[(s as usize - 1) * BRICK_CELLS + o] };
            let bi = b as i32;
            let base = IVec3::new(bi % self.bdims.x, (bi / self.bdims.x) % self.bdims.y, bi / (self.bdims.x * self.bdims.y)) * BRICK;
            let lim = (self.dims - base).min(IVec3::splat(BRICK));
            for z in 0..lim.z as usize {
                for y in 0..lim.y as usize {
                    for x in 0..lim.x as usize {
                        let o = x + B * (y + B * z);
                        let (va, vb) = (block(self, sa, o), block(other, sb, o));
                        if va != vb {
                            out.push((base.x as usize + x + dx * (base.y as usize + y) + dxy * (base.z as usize + z), va, vb));
                        }
                    }
                }
            }
        }
        out
    }

    /// Cells as a dense vector.
    pub fn to_vec(&self) -> Vec<T> {
        self.iter().copied().collect()
    }

    /// Every cell in dense order (absent bricks read as the default).
    pub fn iter(&self) -> Iter<'_, T> {
        Iter { g: self, run: &[], x: 0, y: 0, z: 0 }
    }

    /// Non-default cells as (dense index, value), brick by brick (not in
    /// dense order). Costs time for the bricks with storage, not the box.
    pub fn filled(&self) -> impl Iterator<Item = (usize, T)> + '_ {
        self.bricks().flat_map(move |(b, block)| {
            let base = b * BRICK;
            let (dx, dxy) = (self.dims.x as usize, (self.dims.x * self.dims.y) as usize);
            let lim = (self.dims - base).min(IVec3::splat(BRICK));
            let (bx, by, bz) = (base.x as usize, base.y as usize, base.z as usize);
            (0..lim.z as usize).flat_map(move |z| (0..lim.y as usize).flat_map(move |y| (0..lim.x as usize).map(move |x| (x, y, z)))).filter_map(
                move |(x, y, z)| {
                    let v = block[x + B * (y + B * z)];
                    (v != self.zero).then(|| (bx + x + dx * (by + y) + dxy * (bz + z), v))
                },
            )
        })
    }

    /// Bricks with storage: brick coordinates and their cells (x fastest
    /// within the brick; cells past the grid's edge stay default).
    pub fn bricks(&self) -> impl Iterator<Item = (IVec3, &[T])> + '_ {
        self.slots.iter().enumerate().filter(|(_, s)| **s != 0).map(move |(b, &s)| {
            let b = b as i32;
            let bx = self.bdims.x;
            let at = IVec3::new(b % bx, (b / bx) % self.bdims.y, b / (bx * self.bdims.y));
            (at, &self.pool[(s as usize - 1) * BRICK_CELLS..][..BRICK_CELLS])
        })
    }

    /// Whether the brick at brick coordinates `b` has storage.
    pub fn brick_allocated(&self, b: IVec3) -> bool {
        b.cmpge(IVec3::ZERO).all() && b.cmplt(self.bdims).all() && self.slots[(b.x + self.bdims.x * (b.y + self.bdims.y * b.z)) as usize] != 0
    }
}

impl<T: Copy + Default + PartialEq> Index<usize> for Grid<T> {
    type Output = T;

    #[inline]
    fn index(&self, i: usize) -> &T {
        assert!(i < self.len(), "cell {i} out of range ({})", self.len());
        let (b, o) = self.locate(i);
        match self.slots[b] {
            0 => &self.zero,
            s => &self.pool[(s as usize - 1) * BRICK_CELLS + o],
        }
    }
}

/// Writing through an index allocates the cell's brick, even for the
/// default value; use `set` to avoid that.
impl<T: Copy + Default + PartialEq> IndexMut<usize> for Grid<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        assert!(i < self.len(), "cell {i} out of range ({})", self.len());
        let (b, o) = self.locate(i);
        let s = match self.slots[b] {
            0 => self.alloc(b),
            s => s as usize - 1,
        };
        &mut self.pool[s * BRICK_CELLS + o]
    }
}

impl<T: Copy + Default + PartialEq> PartialEq for Grid<T> {
    /// Same size and cells (however they are stored).
    fn eq(&self, other: &Self) -> bool {
        if self.dims != other.dims {
            return false;
        }
        fn block<T>(g: &Grid<T>, b: usize) -> Option<&[T]> {
            match g.slots[b] {
                0 => None,
                s => Some(&g.pool[(s as usize - 1) * BRICK_CELLS..][..BRICK_CELLS]),
            }
        }
        (0..self.slots.len()).all(|b| match (block(self, b), block(other, b)) {
            (None, None) => true,
            (Some(a), Some(c)) => a == c,
            (Some(a), None) | (None, Some(a)) => a.iter().all(|&v| v == self.zero),
        })
    }
}

impl<T: Copy + Default + PartialEq> PartialEq<Vec<T>> for Grid<T> {
    fn eq(&self, other: &Vec<T>) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl<T: Copy + Default + PartialEq + fmt::Debug> fmt::Debug for Grid<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.len() <= 4096 {
            f.debug_list().entries(self.iter()).finish()
        } else {
            write!(f, "Grid {{ dims: {}, bricks: {} }}", self.dims, self.allocated())
        }
    }
}

/// Cells in dense order.
pub struct Iter<'a, T> {
    g: &'a Grid<T>,
    /// The rest of the current row segment (within one brick).
    run: &'a [T],
    /// Start of the next segment.
    x: i32,
    y: i32,
    z: i32,
}

impl<'a, T: Copy + Default + PartialEq> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        loop {
            if let Some((first, rest)) = self.run.split_first() {
                self.run = rest;
                return Some(first);
            }
            self.run = self.segment()?;
        }
    }

    fn fold<A, F: FnMut(A, &'a T) -> A>(mut self, init: A, mut f: F) -> A {
        let mut acc = init;
        loop {
            for c in self.run {
                acc = f(acc, c);
            }
            match self.segment() {
                Some(run) => self.run = run,
                None => return acc,
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let g = self.g;
        let done = (self.x + g.dims.x * (self.y + g.dims.y * self.z)) as usize;
        let n = g.len().saturating_sub(done) + self.run.len();
        (n, Some(n))
    }
}

impl<T: Copy + Default + PartialEq> ExactSizeIterator for Iter<'_, T> {}

impl<'a, T: Copy + Default + PartialEq> Iter<'a, T> {
    /// The next row segment (up to a brick's width), or `None` at the end.
    #[inline]
    fn segment(&mut self) -> Option<&'a [T]> {
        let g = self.g;
        if self.z >= g.dims.z || g.dims.x == 0 || g.dims.y == 0 {
            return None;
        }
        let n = (BRICK - (self.x & 7)).min(g.dims.x - self.x) as usize;
        let (b, o) = g.locate_at(IVec3::new(self.x, self.y, self.z));
        let run = match g.slots[b] {
            0 => &g.zrow[..n],
            s => &g.pool[(s as usize - 1) * BRICK_CELLS + o..][..n],
        };
        self.x += n as i32;
        if self.x >= g.dims.x {
            self.x = 0;
            self.y += 1;
            if self.y >= g.dims.y {
                self.y = 0;
                self.z += 1;
            }
        }
        Some(run)
    }
}

impl<'a, T: Copy + Default + PartialEq> IntoIterator for &'a Grid<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(g: &Grid<u8>) -> Vec<u8> {
        (0..g.len()).map(|i| g[i]).collect()
    }

    #[test]
    fn reads_and_writes_like_a_flat_array() {
        // Not a multiple of the brick size, so edge bricks are partial.
        let dims = IVec3::new(19, 10, 13);
        let mut g = Grid::<u8>::new(dims);
        let mut want = vec![0u8; g.len()];
        for k in 0..400usize {
            let i = (k * 7919) % want.len();
            let v = (k % 5) as u8;
            g.set(i, v);
            want[i] = v;
        }
        assert_eq!(dense(&g), want);
        assert_eq!(g.to_vec(), want);
        assert!(g == want);
        assert_eq!(g.iter().len(), want.len());
        let mut filled: Vec<(usize, u8)> = g.filled().collect();
        filled.sort();
        let expect: Vec<(usize, u8)> = want.iter().enumerate().filter(|(_, c)| **c != 0).map(|(i, &c)| (i, c)).collect();
        assert_eq!(filled, expect);
        assert_eq!(Grid::from_vec(dims, &want), g);
    }

    #[test]
    fn empty_space_costs_no_bricks() {
        let mut g = Grid::<u8>::new(IVec3::splat(1024));
        assert_eq!(g.allocated(), 0);
        assert!(g.memory() <= 128 * 128 * 128 * 4);
        g.set(0, 0);
        assert_eq!(g.allocated(), 0, "writing empty allocates nothing");
        g.set(5, 3);
        g.set(g.len() - 1, 4);
        assert_eq!(g.allocated(), 2);
        g.set(5, 0);
        g.compact();
        assert_eq!(g.allocated(), 1);
        assert_eq!(g[g.len() - 1], 4);
    }

    #[test]
    fn equality_ignores_storage() {
        let dims = IVec3::new(9, 9, 9);
        let mut a = Grid::<u8>::new(dims);
        let b = Grid::<u8>::new(dims);
        a.set(100, 1);
        a.set(100, 0);
        assert!(a == b, "an allocated empty brick equals an absent one");
        a.set(3, 2);
        assert!(a != b);
        assert!(Grid::<u8>::new(IVec3::new(9, 9, 8)) != b);
    }

    #[test]
    fn diff_lists_changes() {
        let dims = IVec3::new(17, 3, 9);
        let mut a = Grid::<u8>::new(dims);
        let mut b = a.clone();
        a.set(400, 1);
        b.set(2, 7);
        b.set(400, 1);
        b.set(401, 2);
        let mut d = a.diff(&b);
        d.sort();
        assert_eq!(d, vec![(2, 0, 7), (401, 0, 2)]);
        assert!(a.diff(&a).is_empty());
    }

    #[test]
    fn fill_and_update() {
        let mut g = Grid::<u8>::new(IVec3::new(10, 1, 1));
        g.fill(3);
        assert!(g.iter().all(|&c| c == 3));
        g.set(4, 0);
        g.update_filled(|c| *c += 1);
        assert_eq!(g.to_vec(), vec![4, 4, 4, 4, 0, 4, 4, 4, 4, 4]);
        g.fill(0);
        assert_eq!(g.allocated(), 0);
        assert!(g.contains(&0) && !g.contains(&4));
    }
}
