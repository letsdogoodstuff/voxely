//! The scene tree, like MagicaVoxel's World editor: groups and models
//! ("parts"), each with a name, a hidden flag, a layer and a 90° rotation /
//! flip plus an integer translation. Models keep their own voxel grids.
//!
//! Placement follows the `.vox` scene graph: a model of size `s` sits
//! centered on its translation, cell `v` landing at
//! `floor(R * (v + 0.5 - s / 2) + t)` (with integer `s / 2`), and transforms
//! compose down the tree. Rotations are signed permutations, so placement
//! is exact integer math and a model can be turned to world orientation and
//! back without loss.
//!
//! On top of that (voxely only, kept in `.vox` attributes MagicaVoxel
//! ignores): a pivot that turns happen about, a free rotation ("tilt") that
//! is resampled when shown or exported, and named sockets (attachment
//! points) for building from parts and for game engines.

use glam::{Affine3A, EulerRot, IVec3, Mat3, Quat, Vec3};

use crate::scene::Volume;
use crate::vox::{Node as VoxNode, VoxFile};

/// A 90° rotation and/or flip: a signed permutation matrix (`m[row][col]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rot {
    pub m: [[i32; 3]; 3],
}

impl Rot {
    pub const IDENTITY: Rot = Rot { m: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] };

    pub fn apply(&self, v: IVec3) -> IVec3 {
        let r = |i: usize| self.m[i][0] * v.x + self.m[i][1] * v.y + self.m[i][2] * v.z;
        IVec3::new(r(0), r(1), r(2))
    }

    /// `self * o`: apply `o` first.
    pub fn mul(&self, o: &Rot) -> Rot {
        let mut m = [[0; 3]; 3];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..3).map(|k| self.m[i][k] * o.m[k][j]).sum();
            }
        }
        Rot { m }
    }

    /// The inverse (rotations are orthogonal).
    pub fn inverse(&self) -> Rot {
        let mut m = [[0; 3]; 3];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = self.m[j][i];
            }
        }
        Rot { m }
    }

    /// A quarter turn `quarters` times counter-clockwise about `axis`
    /// (right-handed).
    pub fn turn(axis: usize, quarters: i32) -> Rot {
        let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
        let mut r = Rot::IDENTITY;
        for _ in 0..quarters.rem_euclid(4) {
            let mut q = Rot { m: [[0; 3]; 3] };
            q.m[axis][axis] = 1;
            q.m[b][a] = 1;
            q.m[a][b] = -1;
            r = q.mul(&r);
        }
        r
    }

    /// Mirror along `axis`.
    pub fn flip(axis: usize) -> Rot {
        let mut r = Rot::IDENTITY;
        r.m[axis][axis] = -1;
        r
    }

    /// Size of a box of `size` after rotating (always positive).
    pub fn extent(&self, size: IVec3) -> IVec3 {
        self.apply(size).abs()
    }

    /// The `.vox` rotation byte: bits 0-1 / 2-3 hold the column of row 0 /
    /// row 1's non-zero entry, bits 4-6 the signs of rows 0-2.
    pub fn to_byte(&self) -> u8 {
        let col = |i: usize| (0..3).find(|&j| self.m[i][j] != 0).unwrap_or(i) as u8;
        let neg = |i: usize| (self.m[i][col(i) as usize] < 0) as u8;
        col(0) | col(1) << 2 | neg(0) << 4 | neg(1) << 5 | neg(2) << 6
    }

    pub fn from_byte(b: u8) -> Rot {
        Rot::from_mat3(crate::vox::decode_rotation(b))
    }

    pub fn from_mat3(m: Mat3) -> Rot {
        let rows = m.transpose().to_cols_array_2d();
        let mut r = Rot { m: [[0; 3]; 3] };
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] = rows[i][j].round() as i32;
            }
        }
        // Anything that isn't a signed permutation becomes identity.
        let valid = (0..3).all(|i| r.m[i].iter().filter(|v| **v != 0).count() == 1)
            && (0..3).all(|j| (0..3).filter(|&i| r.m[i][j] != 0).count() == 1);
        if valid { r } else { Rot::IDENTITY }
    }
}

/// A node's accumulated transform from the root.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Place {
    pub rot: Rot,
    pub t: IVec3,
}

impl Place {
    pub const IDENTITY: Place = Place { rot: Rot::IDENTITY, t: IVec3::ZERO };

    /// A child placed by `rot` and `t` inside this one.
    pub fn then(&self, rot: Rot, t: IVec3) -> Place {
        Place { rot: self.rot.mul(&rot), t: self.rot.apply(t) + self.t }
    }

    /// World cell of cell `v` of a model of `size`.
    pub fn cell(&self, size: IVec3, v: IVec3) -> IVec3 {
        // 2 * (v + 0.5 - size/2) is odd per axis, so this stays exact.
        (self.rot.apply(v * 2 + 1 - (size / 2) * 2) + self.t * 2 - 1) / 2
    }

    /// Model cell that lands on world cell `w` (inverse of `cell`).
    pub fn local(&self, size: IVec3, w: IVec3) -> IVec3 {
        (self.rot.inverse().apply(w * 2 + 1 - self.t * 2) + (size / 2) * 2 - 1) / 2
    }

    /// World box (min corner, size) of a model of `size`.
    pub fn bounds(&self, size: IVec3) -> (IVec3, IVec3) {
        let a = self.cell(size, IVec3::ZERO);
        let b = self.cell(size, size - 1);
        (a.min(b), (a - b).abs() + 1)
    }
}

#[derive(Clone)]
pub enum Kind {
    Group(Vec<usize>),
    Model(Volume),
}

/// A named attachment point in a node's own coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Socket {
    pub name: String,
    pub pos: Vec3,
}

#[derive(Clone)]
pub struct Node {
    pub name: String,
    pub hidden: bool,
    /// `.vox` layer id, or -1.
    pub layer: i32,
    pub rot: Rot,
    pub t: IVec3,
    pub parent: Option<usize>,
    pub kind: Kind,
    /// Where turns happen, in the node's own coordinates (a model's voxel
    /// space, `0..size`). Default: a model's center; a group's box center.
    pub pivot: Option<Vec3>,
    /// Free rotation about the pivot after `rot`: degrees about x, then y,
    /// then z.
    pub tilt: Vec3,
    pub sockets: Vec<Socket>,
}

impl Node {
    fn new(name: &str, parent: Option<usize>, layer: i32, kind: Kind) -> Node {
        Node { name: name.to_string(), hidden: false, layer, rot: Rot::IDENTITY, t: IVec3::ZERO, parent, kind, pivot: None, tilt: Vec3::ZERO, sockets: Vec::new() }
    }

    pub fn tilted(&self) -> bool {
        self.tilt != Vec3::ZERO
    }

    pub fn tilt_quat(&self) -> Quat {
        let r = self.tilt * (std::f32::consts::PI / 180.0);
        Quat::from_euler(EulerRot::ZYX, r.z, r.y, r.x)
    }

    /// Half the model size as `.vox` placement uses it (integer division).
    fn half(&self) -> Vec3 {
        self.volume().map_or(Vec3::ZERO, |v| (v.dims / 2).as_vec3())
    }

    /// The node's transform into its parent (continuous).
    pub fn local_affine(&self) -> Affine3A {
        let r = rot_mat(self.rot);
        let half = self.half();
        if !self.tilted() {
            return Affine3A::from_translation(self.t.as_vec3()) * Affine3A::from_mat3(r) * Affine3A::from_translation(-half);
        }
        let pivot = self.pivot.unwrap_or_else(|| self.volume().map_or(Vec3::ZERO, |v| v.dims.as_vec3() * 0.5));
        Affine3A::from_translation(self.t.as_vec3())
            * Affine3A::from_mat3(r)
            * Affine3A::from_translation(pivot - half)
            * Affine3A::from_quat(self.tilt_quat())
            * Affine3A::from_translation(-pivot)
    }

    pub fn is_group(&self) -> bool {
        matches!(self.kind, Kind::Group(_))
    }

    pub fn children(&self) -> &[usize] {
        match &self.kind {
            Kind::Group(c) => c,
            Kind::Model(_) => &[],
        }
    }

    pub fn volume(&self) -> Option<&Volume> {
        match &self.kind {
            Kind::Model(v) => Some(v),
            Kind::Group(_) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub id: i32,
    pub name: String,
    pub hidden: bool,
}

/// The tree. Node 0 is the root group; deleted nodes leave `None` so ids
/// stay stable.
#[derive(Clone)]
pub struct World {
    pub nodes: Vec<Option<Node>>,
    pub layers: Vec<Layer>,
}

pub const ROOT: usize = 0;

/// Every model of a composite, flattened: voxels in world space plus which
/// model each cell came from.
pub struct Composite {
    pub volume: Volume,
    /// World position of cell 0.
    pub origin: IVec3,
    /// Model node id per cell (0 = empty; the root is never a model).
    pub owners: Vec<u32>,
}

impl World {
    fn root_group() -> Node {
        Node::new("", None, -1, Kind::Group(Vec::new()))
    }

    /// One model filling the box `[0, size)`.
    pub fn single(volume: Volume) -> World {
        let mut w = World { nodes: vec![Some(Self::root_group())], layers: vec![Layer { id: 0, name: String::new(), hidden: false }] };
        w.add_model(ROOT, "", volume, IVec3::ZERO);
        w
    }

    pub fn node(&self, id: usize) -> Option<&Node> {
        self.nodes.get(id).and_then(Option::as_ref)
    }

    pub fn node_mut(&mut self, id: usize) -> Option<&mut Node> {
        self.nodes.get_mut(id).and_then(Option::as_mut)
    }

    /// The display name: the node's name or a default like `model 3`.
    pub fn label(&self, id: usize) -> String {
        match self.node(id) {
            Some(n) if !n.name.is_empty() => n.name.clone(),
            Some(n) if n.is_group() => format!("group {id}"),
            Some(_) => format!("model {id}"),
            None => String::new(),
        }
    }

    /// Model ids in tree order.
    pub fn models(&self) -> Vec<usize> {
        let mut out = Vec::new();
        self.visit(ROOT, &mut |id, n| {
            if !n.is_group() {
                out.push(id);
            }
        });
        out
    }

    /// Pre-order walk.
    pub fn visit(&self, id: usize, f: &mut impl FnMut(usize, &Node)) {
        if let Some(n) = self.node(id) {
            f(id, n);
            for &c in n.children() {
                self.visit(c, f);
            }
        }
    }

    /// Find a node by name (case-insensitive) or `#id` / plain id.
    pub fn find(&self, key: &str) -> Option<usize> {
        let k = key.trim().trim_start_matches('#');
        if let Ok(id) = k.parse::<usize>() {
            if self.node(id).is_some() {
                return Some(id);
            }
        }
        (0..self.nodes.len()).find(|&id| self.node(id).is_some() && self.label(id).eq_ignore_ascii_case(key.trim()))
    }

    pub fn place(&self, id: usize) -> Place {
        let Some(n) = self.node(id) else { return Place::IDENTITY };
        let parent = n.parent.map_or(Place::IDENTITY, |p| self.place(p));
        parent.then(n.rot, n.t)
    }

    fn layer_hidden(&self, layer: i32) -> bool {
        self.layers.iter().any(|l| l.id == layer && l.hidden)
    }

    /// Shown: neither it nor a parent is hidden, nor on a hidden layer.
    pub fn visible(&self, id: usize) -> bool {
        let mut cur = Some(id);
        while let Some(c) = cur {
            let Some(n) = self.node(c) else { return false };
            if n.hidden || self.layer_hidden(n.layer) {
                return false;
            }
            cur = n.parent;
        }
        true
    }

    /// World box (min, size) of a node; groups cover their models.
    pub fn bounds(&self, id: usize) -> Option<(IVec3, IVec3)> {
        let n = self.node(id)?;
        match &n.kind {
            Kind::Model(v) if self.tilted(id) => {
                let a = self.affine(id);
                let (mut lo, mut hi) = (Vec3::MAX, Vec3::MIN);
                for c in 0..8 {
                    let p = a.transform_point3((IVec3::new(c & 1, (c >> 1) & 1, c >> 2) * v.dims).as_vec3());
                    (lo, hi) = (lo.min(p), hi.max(p));
                }
                let lo = lo.floor().as_ivec3();
                Some((lo, hi.ceil().as_ivec3() - lo))
            }
            Kind::Model(v) => Some(self.place(id).bounds(v.dims)),
            Kind::Group(children) => {
                let mut b: Option<(IVec3, IVec3)> = None;
                for &c in children {
                    if let Some((lo, size)) = self.bounds(c) {
                        let hi = lo + size;
                        b = Some(b.map_or((lo, hi), |(a, z)| (a.min(lo), z.max(hi))));
                    }
                }
                b.map(|(lo, hi)| (lo, hi - lo))
            }
        }
    }

    /// Flatten visible models (except `skip`) into one volume covering
    /// their boxes. Later models win where they overlap.
    pub fn composite(&self, skip: Option<usize>) -> Option<Composite> {
        let models: Vec<usize> = self.models().into_iter().filter(|&m| Some(m) != skip && self.visible(m)).collect();
        let mut b: Option<(IVec3, IVec3)> = None;
        for &m in &models {
            let (lo, size) = self.bounds(m)?;
            let hi = lo + size;
            b = Some(b.map_or((lo, hi), |(a, z)| (a.min(lo), z.max(hi))));
        }
        let (lo, hi) = b?;
        let mut volume = Volume::new(hi - lo);
        let mut owners = vec![0u32; volume.cells.len()];
        for m in models {
            let place = self.place(m);
            let v = self.node(m)?.volume()?;
            if self.tilted(m) {
                // Free rotation: sample the model at each world cell center.
                let inv = self.affine(m).inverse();
                let (blo, bsize) = self.bounds(m)?;
                for z in 0..bsize.z {
                    for y in 0..bsize.y {
                        for x in 0..bsize.x {
                            let w = blo + IVec3::new(x, y, z);
                            let c = v.get(inv.transform_point3(w.as_vec3() + 0.5).floor().as_ivec3());
                            if c != 0 {
                                let j = volume.index(w - lo);
                                volume.set_index(j, c);
                                owners[j] = m as u32;
                            }
                        }
                    }
                }
                continue;
            }
            for (i, &c) in v.cells.iter().enumerate() {
                if c != 0 {
                    let w = place.cell(v.dims, v.position(i)) - lo;
                    let j = volume.index(w);
                    volume.set_index(j, c);
                    owners[j] = m as u32;
                }
            }
        }
        Some(Composite { volume, origin: lo, owners })
    }

    /// A model's voxels turned to world orientation, with the world
    /// position of its cell 0.
    pub fn checkout(&self, id: usize) -> Option<(Volume, IVec3)> {
        let v = self.node(id)?.volume()?;
        let place = self.place(id);
        let (lo, size) = place.bounds(v.dims);
        let mut out = Volume::new(size);
        for (i, &c) in v.cells.iter().enumerate() {
            if c != 0 {
                let j = out.index(place.cell(v.dims, v.position(i)) - lo);
                out.set_index(j, c);
            }
        }
        Some((out, lo))
    }

    /// Store `volume` (world orientation, cell 0 at world `origin`) back
    /// into model `id`, keeping its rotation and moving its translation so
    /// the box stays where it was drawn (even if the size changed).
    pub fn checkin(&mut self, id: usize, volume: &Volume, origin: IVec3) {
        let Some(n) = self.node(id) else { return };
        let parent = n.parent.map_or(Place::IDENTITY, |p| self.place(p));
        let rot = parent.rot.mul(&n.rot);
        let size = rot.inverse().extent(volume.dims);
        // Where the box lands with no translation, then shift it into place.
        let at_zero = Place { rot, t: IVec3::ZERO };
        let t = origin - at_zero.bounds(size).0;
        let place = Place { rot, t };
        let mut local = Volume::new(size);
        for (i, &c) in volume.cells.iter().enumerate() {
            if c != 0 {
                let j = local.index(place.local(size, origin + volume.position(i)));
                local.set_index(j, c);
            }
        }
        let node_t = parent.rot.inverse().apply(t - parent.t);
        if let Some(n) = self.node_mut(id) {
            n.t = node_t;
            n.kind = Kind::Model(local);
        }
    }

    fn push(&mut self, parent: usize, node: Node) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Some(node));
        if let Some(Node { kind: Kind::Group(c), .. }) = self.node_mut(parent) {
            c.push(id);
        }
        id
    }

    /// Nearest group at or above `id` (the root if none).
    pub fn group_of(&self, id: usize) -> usize {
        match self.node(id) {
            Some(n) if n.is_group() => id,
            Some(n) => n.parent.unwrap_or(ROOT),
            None => ROOT,
        }
    }

    /// Add an unrotated model under `parent` whose box starts at world `origin`.
    pub fn add_model(&mut self, parent: usize, name: &str, volume: Volume, origin: IVec3) -> usize {
        let parent = self.group_of(parent);
        let pp = self.place(parent);
        let layer = self.layers.first().map_or(-1, |l| l.id);
        let node = Node { rot: pp.rot.inverse(), ..Node::new(name, Some(parent), layer, Kind::Model(Volume::new(IVec3::ONE))) };
        let id = self.push(parent, node);
        self.checkin(id, &volume, origin);
        id
    }

    /// Copy a node (and everything under it) next to the original.
    pub fn duplicate(&mut self, id: usize) -> Option<usize> {
        let parent = self.node(id)?.parent?;
        let copy = self.copy_into(id, parent);
        if let Some(n) = self.node_mut(copy) {
            if !n.name.is_empty() {
                n.name = format!("{} copy", n.name);
            }
        }
        Some(copy)
    }

    fn copy_into(&mut self, id: usize, parent: usize) -> usize {
        let Some(mut n) = self.node(id).cloned() else { return ROOT };
        let children = n.children().to_vec();
        n.parent = Some(parent);
        if n.is_group() {
            n.kind = Kind::Group(Vec::new());
        }
        let copy = self.push(parent, n);
        for c in children {
            self.copy_into(c, copy);
        }
        copy
    }

    /// Remove a node and everything under it (never the root).
    pub fn delete(&mut self, id: usize) -> bool {
        if id == ROOT || self.node(id).is_none() {
            return false;
        }
        let mut doomed = Vec::new();
        self.visit(id, &mut |i, _| doomed.push(i));
        if let Some(p) = self.node(id).and_then(|n| n.parent) {
            if let Some(Node { kind: Kind::Group(c), .. }) = self.node_mut(p) {
                c.retain(|&x| x != id);
            }
        }
        for i in doomed {
            self.nodes[i] = None;
        }
        true
    }

    /// Move nodes into a new group under the first one's parent. World
    /// placement doesn't change.
    pub fn group(&mut self, ids: &[usize]) -> Option<usize> {
        let ids: Vec<usize> = ids.iter().copied().filter(|&i| i != ROOT && self.node(i).is_some()).collect();
        let parent = self.node(*ids.first()?)?.parent?;
        let g = Node::new("", Some(parent), self.layers.first().map_or(-1, |l| l.id), Kind::Group(Vec::new()));
        let gid = self.push(parent, g);
        for id in ids {
            if self.is_ancestor(id, gid) {
                continue;
            }
            self.reparent(id, gid);
        }
        Some(gid)
    }

    fn is_ancestor(&self, a: usize, mut b: usize) -> bool {
        while let Some(p) = self.node(b).and_then(|n| n.parent) {
            if p == a {
                return true;
            }
            b = p;
        }
        false
    }

    /// Move `id` under group `to`, keeping its world placement.
    pub fn reparent(&mut self, id: usize, to: usize) -> bool {
        if id == ROOT || id == to || self.is_ancestor(id, to) || !self.node(to).is_some_and(Node::is_group) {
            return false;
        }
        let world = self.place(id);
        let Some(old) = self.node(id).and_then(|n| n.parent) else { return false };
        if let Some(Node { kind: Kind::Group(c), .. }) = self.node_mut(old) {
            c.retain(|&x| x != id);
        }
        if let Some(Node { kind: Kind::Group(c), .. }) = self.node_mut(to) {
            c.push(id);
        }
        let pp = self.place(to);
        let inv = pp.rot.inverse();
        if let Some(n) = self.node_mut(id) {
            n.parent = Some(to);
            n.rot = inv.mul(&world.rot);
            n.t = inv.apply(world.t - pp.t);
        }
        true
    }

    /// Dissolve a group, moving its children to its parent.
    pub fn ungroup(&mut self, id: usize) -> bool {
        let Some(n) = self.node(id) else { return false };
        let (Some(parent), true) = (n.parent, n.is_group()) else { return false };
        for c in n.children().to_vec() {
            self.reparent(c, parent);
        }
        self.delete(id)
    }

    /// Move a node by `delta` world cells.
    pub fn translate(&mut self, id: usize, delta: IVec3) {
        let inv = self.node(id).and_then(|n| n.parent).map_or(Rot::IDENTITY, |p| self.place(p).rot.inverse());
        if let Some(n) = self.node_mut(id) {
            n.t += inv.apply(delta);
        }
    }

    /// Apply a world-space rotation or flip `q` to a node in place, about
    /// the center of its box.
    pub fn reorient(&mut self, id: usize, q: Rot) {
        if self.node(id).is_some_and(|n| n.pivot.is_some()) {
            // About the chosen pivot.
            let before = self.pivot_world(id);
            let Some(parent) = self.node(id).and_then(|n| n.parent) else { return };
            let pr = self.place(parent).rot;
            if let Some(n) = self.node_mut(id) {
                n.rot = pr.inverse().mul(&q).mul(&pr).mul(&n.rot);
            }
            let shift = (before - self.pivot_world(id)).round().as_ivec3();
            self.translate(id, shift);
            return;
        }
        let Some((lo, size)) = self.bounds(id) else { return };
        let center2 = lo * 2 + size;
        let Some(parent) = self.node(id).and_then(|n| n.parent) else { return };
        let pr = self.place(parent).rot;
        if let Some(n) = self.node_mut(id) {
            // World rotation becomes q * parent * rot.
            n.rot = pr.inverse().mul(&q).mul(&pr).mul(&n.rot);
        }
        if let Some((lo, size)) = self.bounds(id) {
            let shift = (center2 - (lo * 2 + size)).div_euclid(IVec3::splat(2));
            self.translate(id, shift);
        }
    }
}

/// A signed permutation as a float matrix.
pub fn rot_mat(r: Rot) -> Mat3 {
    let c = |j: usize| Vec3::new(r.m[0][j] as f32, r.m[1][j] as f32, r.m[2][j] as f32);
    Mat3::from_cols(c(0), c(1), c(2))
}

// ---- pivots, free rotation, sockets ----

impl World {
    /// Any free rotation on this node or above it.
    pub fn tilted(&self, id: usize) -> bool {
        let mut cur = Some(id);
        while let Some(n) = cur.and_then(|c| self.node(c)) {
            if n.tilted() {
                return true;
            }
            cur = n.parent;
        }
        false
    }

    /// The node's own coordinates to world space, free rotation included.
    pub fn affine(&self, id: usize) -> Affine3A {
        let Some(n) = self.node(id) else { return Affine3A::IDENTITY };
        n.parent.map_or(Affine3A::IDENTITY, |p| self.affine(p)) * n.local_affine()
    }

    /// The node's pivot in world space.
    pub fn pivot_world(&self, id: usize) -> Vec3 {
        let Some(n) = self.node(id) else { return Vec3::ZERO };
        match (n.pivot, n.volume()) {
            (Some(p), _) => self.affine(id).transform_point3(p),
            (None, Some(v)) => self.affine(id).transform_point3(v.dims.as_vec3() * 0.5),
            (None, None) => self.bounds(id).map_or(Vec3::ZERO, |(lo, size)| lo.as_vec3() + size.as_vec3() * 0.5),
        }
    }

    /// The pivot in the node's own coordinates (see `Node::pivot`).
    pub fn pivot_local(&self, id: usize) -> Vec3 {
        let Some(n) = self.node(id) else { return Vec3::ZERO };
        match (n.pivot, n.volume()) {
            (Some(p), _) => p,
            (None, Some(v)) => v.dims.as_vec3() * 0.5,
            (None, None) => self.affine(id).inverse().transform_point3(self.pivot_world(id)),
        }
    }

    /// Part coordinates, the frame voxel editing uses: a model's voxel
    /// space turned to world orientation (free rotation ignored), with its
    /// box starting at 0. Groups use world coordinates.
    pub fn to_part(&self, id: usize, local: Vec3) -> Vec3 {
        let Some(v) = self.node(id).and_then(Node::volume) else { return self.affine(id).transform_point3(local) };
        let place = self.place(id);
        let w = rot_mat(place.rot) * (local - (v.dims / 2).as_vec3()) + place.t.as_vec3();
        w - place.bounds(v.dims).0.as_vec3()
    }

    /// Inverse of `to_part`.
    pub fn from_part(&self, id: usize, p: Vec3) -> Vec3 {
        let Some(v) = self.node(id).and_then(Node::volume) else { return self.affine(id).inverse().transform_point3(p) };
        let place = self.place(id);
        let w = p + place.bounds(v.dims).0.as_vec3() - place.t.as_vec3();
        rot_mat(place.rot).transpose() * w + (v.dims / 2).as_vec3()
    }

    /// Set the free rotation (degrees), turning about the pivot.
    pub fn set_tilt(&mut self, id: usize, deg: Vec3) {
        if self.node(id).is_some_and(|n| n.is_group() && n.pivot.is_none()) {
            // Fix a group's pivot first so it doesn't drift as it turns.
            let p = self.pivot_local(id);
            if let Some(n) = self.node_mut(id) {
                n.pivot = Some(p);
            }
        }
        if let Some(n) = self.node_mut(id) {
            n.tilt = Vec3::new(wrap_deg(deg.x), wrap_deg(deg.y), wrap_deg(deg.z));
        }
    }

    /// Move a node so its pivot lands on `target` (world).
    pub fn attach(&mut self, id: usize, target: Vec3) {
        let d = target - self.pivot_world(id);
        self.translate(id, d.round().as_ivec3());
    }

    /// A socket's world position, by `part:socket` or a node and name.
    pub fn socket_world(&self, id: usize, name: &str) -> Option<Vec3> {
        let s = self.node(id)?.sockets.iter().find(|s| s.name.eq_ignore_ascii_case(name))?;
        Some(self.affine(id).transform_point3(s.pos))
    }
}

fn wrap_deg(d: f32) -> f32 {
    let w = (d + 180.0).rem_euclid(360.0) - 180.0;
    if w.abs() < 1e-4 { 0.0 } else { w }
}

/// voxely's own transform attributes (MagicaVoxel ignores them).
pub fn write_extras(n: &Node, attrs: &mut crate::vox::Dict) {
    let v3 = |v: Vec3| format!("{} {} {}", v.x, v.y, v.z);
    if let Some(p) = n.pivot {
        attrs.insert("_vx_pivot".into(), v3(p));
    }
    if n.tilted() {
        attrs.insert("_vx_tilt".into(), v3(n.tilt));
    }
    if !n.sockets.is_empty() {
        let list: Vec<String> = n.sockets.iter().map(|s| format!("{}:{}", s.name.replace(['|', ':'], "_"), v3(s.pos))).collect();
        attrs.insert("_vx_sockets".into(), list.join("|"));
    }
}

fn read_extras(n: &mut Node, attrs: &crate::vox::Dict) {
    let v3 = |s: &str| {
        let v: Vec<f32> = s.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        (v.len() == 3).then(|| Vec3::new(v[0], v[1], v[2]))
    };
    n.pivot = attrs.get("_vx_pivot").and_then(|s| v3(s));
    n.tilt = attrs.get("_vx_tilt").and_then(|s| v3(s)).unwrap_or(Vec3::ZERO);
    n.sockets = attrs
        .get("_vx_sockets")
        .map(|s| s.split('|').filter_map(|e| e.split_once(':')).filter_map(|(name, p)| Some(Socket { name: name.to_string(), pos: v3(p)? })).collect())
        .unwrap_or_default();
}

// ---- .vox ----

impl World {
    pub fn from_vox(f: &VoxFile) -> World {
        let mut w = World { nodes: vec![Some(Self::root_group())], layers: Vec::new() };
        let mut ids: Vec<i32> = f.layers.keys().copied().collect();
        ids.sort();
        for id in ids {
            let d = &f.layers[&id];
            w.layers.push(Layer { id, name: d.get("_name").cloned().unwrap_or_default(), hidden: d.get("_hidden").is_some_and(|v| v == "1") });
        }
        if w.layers.is_empty() {
            w.layers.push(Layer { id: 0, name: String::new(), hidden: false });
        }
        if f.nodes.is_empty() {
            // Old files: every model at the origin.
            for m in &f.models {
                w.add_model(ROOT, "", model_volume(m), IVec3::ZERO);
            }
            return w;
        }
        // The root transform's child is the top group (or a lone shape).
        match f.nodes.get(&0) {
            Some(VoxNode::Transform { child, .. }) => match f.nodes.get(child) {
                Some(VoxNode::Group { children, .. }) => {
                    for &c in children {
                        w.read_node(f, c, ROOT, 0);
                    }
                }
                _ => w.read_node(f, 0, ROOT, 0),
            },
            _ => {}
        }
        w
    }

    /// Read transform node `trn` and what it wraps.
    fn read_node(&mut self, f: &VoxFile, trn: i32, parent: usize, depth: u32) {
        if depth > 64 {
            return;
        }
        let Some(VoxNode::Transform { attrs, child, layer, rotation, translation }) = f.nodes.get(&trn) else { return };
        let name = attrs.get("_name").cloned().unwrap_or_default();
        let hidden = attrs.get("_hidden").is_some_and(|v| v == "1");
        let mut base = Node { hidden, rot: Rot::from_mat3(*rotation), t: *translation, ..Node::new(&name, Some(parent), *layer, Kind::Group(Vec::new())) };
        read_extras(&mut base, attrs);
        match f.nodes.get(child) {
            Some(VoxNode::Group { attrs: gattrs, children }) => {
                let mut n = base;
                n.hidden |= gattrs.get("_hidden").is_some_and(|v| v == "1");
                let id = self.push(parent, n);
                for &c in children {
                    self.read_node(f, c, id, depth + 1);
                }
            }
            Some(VoxNode::Shape { attrs: sattrs, models }) => {
                // Animated shapes keep their first frame.
                let Some(m) = models.first().and_then(|&m| f.models.get(m as usize)) else { return };
                let mut n = base;
                n.hidden |= sattrs.get("_hidden").is_some_and(|v| v == "1");
                n.kind = Kind::Model(model_volume(m));
                self.push(parent, n);
            }
            _ => {}
        }
    }
}

fn model_volume(m: &crate::vox::Model) -> Volume {
    let mut v = Volume::new(m.size.max(IVec3::ONE));
    for c in &m.voxels {
        v.set(IVec3::new(c[0] as i32, c[1] as i32, c[2] as i32), c[3]);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vol(dims: IVec3, cells: &[(IVec3, u8)]) -> Volume {
        let mut v = Volume::new(dims);
        for &(p, c) in cells {
            v.set(p, c);
        }
        v
    }

    #[test]
    fn rotation_bytes_round_trip() {
        for b in 0..128u8 {
            let r = Rot::from_byte(b);
            // Only bytes naming a real permutation are meaningful.
            if (b & 3) != ((b >> 2) & 3) && (b & 3) < 3 && ((b >> 2) & 3) < 3 {
                assert_eq!(r.to_byte(), b, "byte {b}");
            }
        }
        assert_eq!(Rot::IDENTITY.to_byte(), 4);
        assert_eq!(Rot::turn(2, 4), Rot::IDENTITY);
        assert_eq!(Rot::turn(0, 1).mul(&Rot::turn(0, 1).inverse()), Rot::IDENTITY);
    }

    #[test]
    fn placement_matches_vox_convention() {
        // Even and odd sizes, rotated, round-trip through local().
        for size in [IVec3::new(4, 3, 2), IVec3::new(5, 5, 1)] {
            for r in [Rot::IDENTITY, Rot::turn(2, 1), Rot::turn(0, 3).mul(&Rot::flip(1))] {
                let p = Place { rot: r, t: IVec3::new(7, -3, 2) };
                for z in 0..size.z {
                    for y in 0..size.y {
                        for x in 0..size.x {
                            let v = IVec3::new(x, y, z);
                            assert_eq!(p.local(size, p.cell(size, v)), v);
                        }
                    }
                }
            }
        }
        // Unrotated: the box is [t - size/2, t - size/2 + size).
        let p = Place { rot: Rot::IDENTITY, t: IVec3::new(10, 10, 10) };
        assert_eq!(p.bounds(IVec3::new(4, 5, 1)), (IVec3::new(8, 8, 10), IVec3::new(4, 5, 1)));
    }

    #[test]
    fn checkout_and_checkin_are_lossless() {
        let v = vol(IVec3::new(3, 2, 1), &[(IVec3::new(0, 0, 0), 1), (IVec3::new(2, 1, 0), 2)]);
        let mut w = World::single(v.clone());
        let m = w.models()[0];
        w.reorient(m, Rot::turn(2, 1));
        let (world_vol, origin) = w.checkout(m).unwrap();
        assert_eq!(world_vol.dims, IVec3::new(2, 3, 1));
        w.checkin(m, &world_vol, origin);
        assert_eq!(w.node(m).unwrap().volume().unwrap().cells, v.cells);
        assert_eq!(w.checkout(m).unwrap().1, origin);
    }

    #[test]
    fn grown_model_keeps_its_corner() {
        let mut w = World::single(Volume::new(IVec3::new(4, 4, 4)));
        let m = w.models()[0];
        w.translate(m, IVec3::new(5, 0, 0));
        let (_, origin) = w.checkout(m).unwrap();
        w.checkin(m, &Volume::new(IVec3::new(7, 4, 3)), origin);
        assert_eq!(w.bounds(m), Some((origin, IVec3::new(7, 4, 3))));
    }

    #[test]
    fn composite_places_models_and_owners() {
        let mut w = World::single(vol(IVec3::splat(2), &[(IVec3::ZERO, 5)]));
        let b = w.add_model(ROOT, "b", vol(IVec3::splat(2), &[(IVec3::ONE, 6)]), IVec3::new(4, 0, 0));
        let c = w.composite(None).unwrap();
        assert_eq!(c.origin, IVec3::ZERO);
        assert_eq!(c.volume.dims, IVec3::new(6, 2, 2));
        assert_eq!(c.volume.get(IVec3::new(5, 1, 1)), 6);
        assert_eq!(c.owners[c.volume.index(IVec3::new(5, 1, 1))], b as u32);
        w.node_mut(b).unwrap().hidden = true;
        assert_eq!(w.composite(None).unwrap().volume.dims, IVec3::splat(2));
    }

    #[test]
    fn grouping_keeps_world_placement() {
        let mut w = World::single(vol(IVec3::new(3, 1, 1), &[(IVec3::ZERO, 1)]));
        let a = w.models()[0];
        let b = w.add_model(ROOT, "b", vol(IVec3::ONE, &[(IVec3::ZERO, 2)]), IVec3::new(5, 5, 0));
        let before = w.composite(None).unwrap();
        let g = w.group(&[a, b]).unwrap();
        w.reorient(g, Rot::turn(2, 1));
        w.reorient(g, Rot::turn(2, 3));
        assert_eq!(w.composite(None).unwrap().volume.cells, before.volume.cells);
        w.ungroup(g);
        assert_eq!(w.node(ROOT).unwrap().children(), &[a, b]);
        assert_eq!(w.composite(None).unwrap().volume.cells, before.volume.cells);
        let d = w.duplicate(b).unwrap();
        assert_eq!(w.label(d), "b copy");
        assert!(w.delete(d) && w.node(d).is_none());
    }

    fn solid(w: &World) -> Vec<IVec3> {
        let c = w.composite(None).unwrap();
        let mut out: Vec<IVec3> = (0..c.volume.cells.len()).filter(|&i| c.volume.cells[i] != 0).map(|i| c.volume.position(i) + c.origin).collect();
        out.sort_by_key(|p| (p.z, p.y, p.x));
        out
    }

    #[test]
    fn free_quarter_turn_matches_exact_turn() {
        let v = vol(IVec3::new(4, 2, 3), &[(IVec3::ZERO, 1), (IVec3::new(3, 1, 2), 2), (IVec3::new(1, 0, 2), 3)]);
        let mut a = World::single(v.clone());
        let mut b = World::single(v);
        let m = a.models()[0];
        a.reorient(m, Rot::turn(2, 1));
        b.set_tilt(m, Vec3::new(0.0, 0.0, 90.0));
        assert!(b.tilted(m));
        // Same shape; the free turn is about the exact center, the 90° one
        // keeps the box on whole cells, so allow a shift of under a voxel.
        let (sa, sb) = (solid(&a), solid(&b));
        assert_eq!(sa.len(), sb.len());
        let d = sb[0] - sa[0];
        assert!(d.abs().max_element() <= 1);
        assert!(sa.iter().zip(&sb).all(|(p, q)| *q - *p == d));
        // A 45° tilt keeps voxels (roughly) and grows the box.
        b.set_tilt(m, Vec3::new(0.0, 0.0, 45.0));
        let (_, size) = b.bounds(m).unwrap();
        assert!(size.x >= 4 && size.y >= 4, "{size}");
        assert!(!solid(&b).is_empty());
    }

    #[test]
    fn turns_happen_about_the_pivot_and_attach_snaps() {
        let mut w = World::single(Volume::new(IVec3::new(2, 2, 6)));
        let arm = w.models()[0];
        // Pivot at the top center (the shoulder).
        w.node_mut(arm).unwrap().pivot = Some(Vec3::new(1.0, 1.0, 6.0));
        let shoulder = w.pivot_world(arm);
        w.reorient(arm, Rot::turn(0, 1));
        assert_eq!(w.pivot_world(arm), shoulder);
        // The arm now sticks out sideways from the shoulder.
        let (lo, size) = w.bounds(arm).unwrap();
        assert_eq!(size, IVec3::new(2, 6, 2));
        assert!(lo.z as f32 <= shoulder.z && shoulder.z <= (lo.z + size.z) as f32);

        let body = w.add_model(ROOT, "body", Volume::new(IVec3::splat(4)), IVec3::new(10, 0, 0));
        w.node_mut(body).unwrap().sockets.push(Socket { name: "shoulder_r".into(), pos: Vec3::new(4.0, 2.0, 3.0) });
        let s = w.socket_world(body, "SHOULDER_R").unwrap();
        w.attach(arm, s);
        assert_eq!(w.pivot_world(arm), s);
    }

    #[test]
    fn part_coordinates_follow_the_edit_view() {
        let mut w = World::single(Volume::new(IVec3::new(4, 2, 1)));
        let m = w.models()[0];
        w.translate(m, IVec3::new(5, 5, 0));
        w.reorient(m, Rot::turn(2, 1));
        let (v, _) = w.checkout(m).unwrap();
        // Local x runs along world y after the turn.
        let p = w.to_part(m, Vec3::new(4.0, 0.0, 0.0));
        assert!(p.cmpge(Vec3::ZERO).all() && p.cmple(v.dims.as_vec3()).all(), "{p}");
        assert!((w.from_part(m, p) - Vec3::new(4.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn extras_round_trip_through_vox() {
        let mut w = World::single(vol(IVec3::splat(3), &[(IVec3::ONE, 4)]));
        let m = w.models()[0];
        {
            let n = w.node_mut(m).unwrap();
            n.name = "gun".into();
            n.pivot = Some(Vec3::new(0.5, 1.5, 0.0));
            n.tilt = Vec3::new(0.0, 15.0, -30.0);
            n.sockets = vec![Socket { name: "muzzle".into(), pos: Vec3::new(3.0, 1.5, 1.5) }, Socket { name: "grip".into(), pos: Vec3::new(1.0, 1.0, 0.0) }];
        }
        let path = std::env::temp_dir().join(format!("voxely-extras-{}.vox", std::process::id()));
        crate::vox::write(&path, &w, &crate::vox::default_palette(), &[], &[]).unwrap();
        let back = World::from_vox(&VoxFile::load(&path).unwrap());
        std::fs::remove_file(&path).ok();
        let n = back.node(back.find("gun").unwrap()).unwrap();
        assert_eq!(n.pivot, Some(Vec3::new(0.5, 1.5, 0.0)));
        assert_eq!(n.tilt, Vec3::new(0.0, 15.0, -30.0));
        assert_eq!(n.sockets, w.node(m).unwrap().sockets);
    }
}
