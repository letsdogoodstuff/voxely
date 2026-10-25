//! Animation, stored the way MagicaVoxel stores it: a node's transform can
//! have keyframes (`nTRN` frames), and a part can show a different voxel
//! model from some frames on (`nSHP` models with `_f`, a flipbook).
//!
//! The tree's live fields (`Node::t`, `rot`, `tilt` and the part's volume)
//! always hold the current frame, so drawing, editing and exporting work on
//! a pose like any other. `World::set_frame` re-poses; editing a keyed part
//! (or any part once there is more than one frame) records a key.

use glam::{EulerRot, IVec3, Mat3, Quat, Vec3};

use crate::scene::Volume;
use crate::world::{rot_mat, Kind, Node, Rot, World};

/// Timeline settings.
#[derive(Clone, Debug, PartialEq)]
pub struct Anim {
    /// Number of frames (at least 1).
    pub frames: u32,
    pub fps: f32,
    pub looping: bool,
    /// Blend between keys; otherwise each key holds until the next.
    pub smooth: bool,
}

impl Default for Anim {
    fn default() -> Self {
        Anim { frames: 1, fps: 10.0, looping: true, smooth: true }
    }
}

/// A node's transform on one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Key {
    pub frame: u32,
    pub rot: Rot,
    pub t: IVec3,
    /// Free rotation, degrees (see `Node::tilt`).
    pub tilt: Vec3,
}

/// A flipbook model shown from `frame` until the next slot.
#[derive(Clone)]
pub struct Slot {
    pub frame: u32,
    /// `None` for the slot being shown: its voxels are the part's volume.
    pub volume: Option<Volume>,
}

fn proper(r: Rot) -> Mat3 {
    let m = rot_mat(r);
    if m.determinant() < 0.0 { -m } else { m }
}

pub fn tilt_quat(deg: Vec3) -> Quat {
    let r = deg * (std::f32::consts::PI / 180.0);
    Quat::from_euler(EulerRot::ZYX, r.z, r.y, r.x)
}

pub fn tilt_deg(q: Quat) -> Vec3 {
    let (z, y, x) = q.to_euler(EulerRot::ZYX);
    let d = Vec3::new(x, y, z) * (180.0 / std::f32::consts::PI);
    d.map(|v| if v.abs() < 1e-3 { 0.0 } else { (v * 1000.0).round() / 1000.0 })
}

impl Node {
    pub fn key_now(&self, frame: u32) -> Key {
        Key { frame, rot: self.rot, t: self.t, tilt: self.tilt }
    }

    /// The pivot used to blend turns (in the node's own coordinates).
    fn blend_pivot(&self) -> Vec3 {
        self.pivot.unwrap_or_else(|| self.volume().map_or(Vec3::ZERO, |v| v.dims.as_vec3() * 0.5))
    }

    /// Blend keys `a` -> `b` by `s`, turning about the pivot so joints stay
    /// together mid-turn.
    fn blend(&self, a: &Key, b: &Key, s: f32) -> Key {
        if s <= 0.0 {
            return *a;
        }
        if s >= 1.0 {
            return *b;
        }
        let half = self.volume().map_or(Vec3::ZERO, |v| (v.dims / 2).as_vec3());
        let pivot = self.blend_pivot() - half;
        let at = |k: &Key| k.t.as_vec3() + rot_mat(k.rot) * pivot;
        let p = at(a).lerp(at(b), s);
        let near = if s < 0.5 { a } else { b };
        if rot_mat(a.rot).determinant() * rot_mat(b.rot).determinant() < 0.0 {
            // A mirror can't be blended; it flips halfway.
            return Key { frame: a.frame, ..*near };
        }
        let qa = Quat::from_mat3(&proper(a.rot)) * tilt_quat(a.tilt);
        let qb = Quat::from_mat3(&proper(b.rot)) * tilt_quat(b.tilt);
        let q = qa.slerp(qb, s);
        let rot = near.rot;
        let tilt = tilt_deg(Quat::from_mat3(&proper(rot)).inverse() * q);
        let t = (p - rot_mat(rot) * pivot).round().as_ivec3();
        Key { frame: a.frame, rot, t, tilt }
    }

    /// The pose on `frame`, from the keys.
    pub fn pose(&self, frame: u32, anim: &Anim) -> Option<Key> {
        let keys = &self.keys;
        let (first, last) = (keys.first()?, keys.last()?);
        let smooth = anim.smooth;
        let span = |a: u32, b: u32| (b - a).max(1) as f32;
        if frame <= first.frame && !(anim.looping && smooth && frame < first.frame && keys.len() > 1) {
            return Some(*first);
        }
        if let Some(i) = keys.iter().position(|k| k.frame > frame) {
            if i > 0 {
                let (a, b) = (&keys[i - 1], &keys[i]);
                let s = if smooth { (frame - a.frame) as f32 / span(a.frame, b.frame) } else { 0.0 };
                return Some(self.blend(a, b, s));
            }
        }
        // After the last key (or before the first): hold, or wrap around.
        if anim.looping && smooth && keys.len() > 1 {
            let n = anim.frames.max(last.frame + 1);
            let len = n - last.frame + first.frame;
            let into = if frame >= last.frame { frame - last.frame } else { frame + n - last.frame };
            return Some(self.blend(last, first, into as f32 / len.max(1) as f32));
        }
        Some(*last)
    }

    /// Insert or replace a key.
    pub fn set_key(&mut self, k: Key) {
        match self.keys.binary_search_by_key(&k.frame, |x| x.frame) {
            Ok(i) => self.keys[i] = k,
            Err(i) => self.keys.insert(i, k),
        }
    }

    /// Flipbook slot index for `frame`.
    fn slot_for(&self, frame: u32) -> usize {
        self.flip.iter().rposition(|s| s.frame <= frame).unwrap_or(0)
    }

    /// Every flipbook model as (first frame, voxels), or just the part's.
    pub fn flip_volumes(&self) -> Vec<(u32, &Volume)> {
        let Kind::Model(current) = &self.kind else { return Vec::new() };
        if self.flip.is_empty() {
            return vec![(0, current)];
        }
        self.flip.iter().map(|s| (s.frame, s.volume.as_ref().unwrap_or(current))).collect()
    }

    /// Show flipbook slot `want`.
    fn show_slot(&mut self, want: usize) {
        if want == self.flip_active || want >= self.flip.len() {
            return;
        }
        let Kind::Model(v) = &mut self.kind else { return };
        let Some(next) = self.flip[want].volume.take() else { return };
        let shown = std::mem::replace(v, next);
        self.flip[self.flip_active].volume = Some(shown);
        self.flip_active = want;
    }
}

impl World {
    /// Re-pose every node for `frame`.
    pub fn set_frame(&mut self, frame: u32) {
        let frame = frame.min(self.anim.frames.saturating_sub(1));
        self.frame = frame;
        let anim = self.anim.clone();
        for n in self.nodes.iter_mut().flatten() {
            if let Some(k) = n.pose(frame, &anim) {
                (n.rot, n.t, n.tilt) = (k.rot, k.t, k.tilt);
            }
            if !n.flip.is_empty() {
                let want = n.slot_for(frame);
                n.show_slot(want);
            }
        }
    }

    pub fn animated(&self) -> bool {
        self.anim.frames > 1 || self.nodes.iter().flatten().any(|n| !n.keys.is_empty() || !n.flip.is_empty())
    }

    /// Key every node whose own transform changed since `before` (same
    /// parent), when animating. A node's first key off frame 0 also keys
    /// frame 0 with where it was, so the move animates from there.
    pub fn autokey(&mut self, before: &World) {
        let frame = self.frame;
        let animating = self.anim.frames > 1;
        for id in 0..self.nodes.len() {
            let (Some(n), Some(old)) = (self.node(id), before.node(id)) else { continue };
            if n.parent != old.parent || (n.rot, n.t, n.tilt) == (old.rot, old.t, old.tilt) || !(animating || !n.keys.is_empty()) {
                continue;
            }
            let (now, was) = (n.key_now(frame), old.key_now(0));
            let first = n.keys.is_empty();
            let n = self.node_mut(id).expect("checked");
            if first && frame != 0 {
                n.set_key(was);
            }
            n.set_key(now);
        }
    }

    /// Key `id` as it is now on the current frame.
    pub fn key(&mut self, id: usize) {
        let frame = self.frame;
        if let Some(n) = self.node_mut(id) {
            let k = n.key_now(frame);
            n.set_key(k);
        }
    }

    /// Remove `id`'s key on `frame`; returns whether there was one.
    pub fn remove_key(&mut self, id: usize, frame: u32) -> bool {
        let Some(n) = self.node_mut(id) else { return false };
        let before = n.keys.len();
        n.keys.retain(|k| k.frame != frame);
        let removed = n.keys.len() != before;
        let f = self.frame;
        self.set_frame(f);
        removed
    }

    /// Give part `id` its own voxels from the current frame on (a flipbook
    /// frame), starting as a copy of what it shows now.
    pub fn split_frame(&mut self, id: usize) -> bool {
        let frame = self.frame;
        let Some(n) = self.node_mut(id) else { return false };
        let Kind::Model(v) = &n.kind else { return false };
        if n.flip.is_empty() {
            n.flip.push(Slot { frame: 0, volume: None });
            n.flip_active = 0;
        }
        if n.flip.iter().any(|s| s.frame == frame) {
            return false;
        }
        let copy = v.clone();
        let at = n.flip.partition_point(|s| s.frame < frame);
        n.flip.insert(at, Slot { frame, volume: Some(copy) });
        if at <= n.flip_active {
            n.flip_active += 1;
        }
        n.show_slot(at);
        true
    }

    /// Drop the flipbook frame starting on the current frame (not frame 0).
    pub fn merge_frame(&mut self, id: usize) -> bool {
        let frame = self.frame;
        let Some(n) = self.node_mut(id) else { return false };
        let Some(i) = n.flip.iter().position(|s| s.frame == frame).filter(|&i| i > 0) else { return false };
        n.show_slot(i - 1);
        n.flip.remove(i);
        if n.flip_active > i {
            n.flip_active -= 1;
        }
        if n.flip.len() == 1 {
            n.flip.clear();
            n.flip_active = 0;
        }
        true
    }

    /// Frames that hold a key or flipbook frame for `id` (for the timeline).
    pub fn marks(&self, id: usize) -> Vec<(u32, bool)> {
        let Some(n) = self.node(id) else { return Vec::new() };
        let mut out: Vec<(u32, bool)> = n.keys.iter().map(|k| (k.frame, false)).collect();
        out.extend(n.flip.iter().skip(1).map(|s| (s.frame, true)));
        out
    }

    /// Change the frame count, keeping keys (ones past the end stay but
    /// don't play).
    pub fn set_frames(&mut self, frames: u32) {
        self.anim.frames = frames.clamp(1, 10_000);
        let f = self.frame;
        self.set_frame(f);
    }

    /// Convert `id`'s keys after moving it from parent frame `from` to `to`.
    pub(crate) fn rebase_keys(&mut self, id: usize, from: crate::world::Place, to: crate::world::Place) {
        let inv = to.rot.inverse();
        if let Some(n) = self.node_mut(id) {
            for k in &mut n.keys {
                let (wr, wt) = (from.rot.mul(&k.rot), from.rot.apply(k.t) + from.t);
                k.rot = inv.mul(&wr);
                k.t = inv.apply(wt - to.t);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::ROOT;

    fn arm() -> (World, usize) {
        let mut w = World::single(Volume::new(IVec3::new(2, 2, 6)));
        let m = w.models()[0];
        w.node_mut(m).unwrap().pivot = Some(Vec3::new(1.0, 1.0, 6.0));
        w.anim.frames = 9;
        (w, m)
    }

    #[test]
    fn keys_blend_about_the_pivot() {
        let (mut w, m) = arm();
        let shoulder = w.pivot_world(m);
        let before = w.clone();
        w.set_frame(8);
        w.set_tilt(m, Vec3::new(90.0, 0.0, 0.0));
        w.autokey(&before);
        // Moving on frame 8 keyed frame 0 (where it was) and frame 8.
        assert_eq!(w.node(m).unwrap().keys.iter().map(|k| k.frame).collect::<Vec<_>>(), [0, 8]);
        w.anim.looping = false;
        w.set_frame(4);
        let t = w.node(m).unwrap().tilt;
        assert!((t.x - 45.0).abs() < 0.01, "{t}");
        // Halfway through the swing the shoulder hasn't moved.
        assert!((w.pivot_world(m) - shoulder).abs().max_element() <= 0.5);
        w.set_frame(0);
        assert_eq!(w.node(m).unwrap().tilt, Vec3::ZERO);
        // Stepped: holds frame 0's pose until frame 8.
        w.anim.smooth = false;
        w.set_frame(7);
        assert_eq!(w.node(m).unwrap().tilt, Vec3::ZERO);
        w.set_frame(8);
        assert_eq!(w.node(m).unwrap().tilt.x, 90.0);
    }

    #[test]
    fn quarter_turn_keys_blend_through_the_middle() {
        let (mut w, m) = arm();
        w.key(m);
        w.set_frame(8);
        w.reorient(m, Rot::turn(0, 1));
        w.key(m);
        w.anim.looping = false;
        w.set_frame(4);
        let n = w.node(m).unwrap();
        // Halfway: one of the two exact orientations plus a 45° tilt.
        let full = proper(n.rot) * Mat3::from_quat(tilt_quat(n.tilt));
        let want = Mat3::from_quat(Quat::from_rotation_x(std::f32::consts::FRAC_PI_4));
        assert!((full - want).abs().to_cols_array().iter().all(|v| *v < 1e-3), "{full} vs {want}");
    }

    #[test]
    fn looping_wraps_from_the_last_key_to_the_first() {
        let (mut w, m) = arm();
        w.key(m);
        w.set_frame(4);
        w.set_tilt(m, Vec3::new(40.0, 0.0, 0.0));
        w.key(m);
        // Frames 4..9 wrap back toward frame 0's pose.
        w.set_frame(6);
        let t = w.node(m).unwrap().tilt.x;
        assert!(t > 0.0 && t < 40.0, "{t}");
    }

    #[test]
    fn flipbook_frames_swap_voxels() {
        let (mut w, m) = arm();
        w.set_frame(3);
        assert!(w.split_frame(m));
        if let Kind::Model(v) = &mut w.node_mut(m).unwrap().kind {
            v.set(IVec3::ZERO, 7);
        }
        let shown = |w: &World| w.node(m).unwrap().volume().unwrap().get(IVec3::ZERO);
        assert_eq!(shown(&w), 7);
        w.set_frame(0);
        assert_eq!(shown(&w), 0);
        w.set_frame(8);
        assert_eq!(shown(&w), 7);
        assert_eq!(w.marks(m), [(3, true)]);
        w.set_frame(3);
        assert!(w.merge_frame(m));
        assert!(w.node(m).unwrap().flip.is_empty());
        assert_eq!(shown(&w), 0);
        let _ = ROOT;
    }

    #[test]
    fn animation_round_trips_through_vox() {
        let (mut w, m) = arm();
        w.anim = Anim { frames: 12, fps: 8.0, looping: false, smooth: true };
        w.key(m);
        w.set_frame(6);
        w.set_tilt(m, Vec3::new(30.0, 0.0, 0.0));
        w.translate(m, IVec3::new(2, 0, 1));
        w.key(m);
        w.set_frame(9);
        w.split_frame(m);
        if let Kind::Model(v) = &mut w.node_mut(m).unwrap().kind {
            v.set(IVec3::ONE, 9);
        }
        let path = std::env::temp_dir().join(format!("voxely-anim-{}.vox", std::process::id()));
        crate::vox::write(&path, &w, &crate::vox::default_palette(), &[], &[], &[]).unwrap();
        let back = World::from_vox(&crate::vox::VoxFile::load(&path).unwrap());
        std::fs::remove_file(&path).ok();
        assert_eq!(back.anim, w.anim);
        let bm = back.models()[0];
        assert_eq!(back.node(bm).unwrap().keys, w.node(m).unwrap().keys);
        assert_eq!(back.marks(bm), w.marks(m));
        for f in [0, 3, 6, 9, 11] {
            let (mut a, mut b) = (w.clone(), back.clone());
            a.set_frame(f);
            b.set_frame(f);
            let (na, nb) = (a.node(m).unwrap(), b.node(bm).unwrap());
            assert_eq!((na.t, na.rot, na.tilt), (nb.t, nb.rot, nb.tilt), "frame {f}");
            assert_eq!(na.volume().unwrap().cells, nb.volume().unwrap().cells, "frame {f}");
        }
    }
}
