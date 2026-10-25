//! Rigs and motions for blocky characters: a humanoid and a four-legged
//! template built from parts (each a rigid block turning at its joint), a
//! name matcher that finds bones in any rig (voxely's names, Mixamo's,
//! Unreal's, Blender's ...), and procedural motions that key them.
//!
//! Conventions: z up, the character faces -y, its left is +x. Bones hang
//! from their pivot, so for a limb pointing down a negative turn about x
//! swings it forward; for one pointing up (torso, head) a positive turn
//! leans it forward.

use std::collections::HashMap;
use std::f32::consts::TAU;

use glam::{IVec3, Vec3};

use crate::anim::{tilt_deg, tilt_quat, Key};
use crate::scene::Volume;
use crate::world::{Socket, World, ROOT};

/// The bones motions and imports know about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bone {
    Hips,
    Torso,
    Head,
    UpperArmL,
    LowerArmL,
    UpperArmR,
    LowerArmR,
    UpperLegL,
    LowerLegL,
    UpperLegR,
    LowerLegR,
    /// Four-legged: front left / right, back left / right legs, tail.
    LegFL,
    LegFR,
    LegBL,
    LegBR,
    Tail,
}

use Bone::*;

/// Which bone a part or joint name means, if any.
pub fn bone_of(name: &str) -> Option<Bone> {
    let n = name.to_ascii_lowercase().replace(['.', '-', ' ', ':'], "_");
    // Strip rig prefixes like "mixamorig_".
    let n = n.rsplit_once("rig_").map_or(n.as_str(), |(_, b)| b).to_string();
    let words: Vec<&str> = n.split('_').filter(|w| !w.is_empty()).collect();
    let has = |w: &str| words.contains(&w) || n.contains(w);
    let left = words.iter().any(|w| *w == "l" || *w == "left") || n.starts_with("left") || n.ends_with("_l");
    let right = words.iter().any(|w| *w == "r" || *w == "right") || n.starts_with("right") || n.ends_with("_r");
    let side = |l: Bone, r: Bone| if left { Some(l) } else if right { Some(r) } else { None };
    let front = words.iter().any(|w| matches!(*w, "front" | "f" | "fl" | "fr")) || n.contains("front");
    let back = words.iter().any(|w| matches!(*w, "back" | "b" | "rear" | "hind" | "bl" | "br")) || n.contains("hind") || n.contains("rear");
    if (front || back) && has("leg") {
        let (l, r) = (n.contains("fl") || n.contains("bl") || left, n.contains("fr") || n.contains("br") || right);
        return match (front, l, r) {
            (true, true, _) => Some(LegFL),
            (true, _, true) => Some(LegFR),
            (false, true, _) => Some(LegBL),
            (false, _, true) => Some(LegBR),
            _ => None,
        };
    }
    if has("tail") {
        return Some(Tail);
    }
    if has("head") {
        return Some(Head);
    }
    // ("root" is usually the armature above the hips, so it isn't one.)
    if has("hips") || has("pelvis") || n == "hip" {
        return Some(Hips);
    }
    if has("forearm") || has("lowerarm") || has("lower_arm") || has("elbow") {
        return side(LowerArmL, LowerArmR);
    }
    if has("upperarm") || has("upper_arm") || has("arm") {
        return side(UpperArmL, UpperArmR);
    }
    if has("thigh") || has("upleg") || has("upperleg") || has("upper_leg") {
        return side(UpperLegL, UpperLegR);
    }
    if has("shin") || has("calf") || has("lowerleg") || has("lower_leg") || has("knee") {
        return side(LowerLegL, LowerLegR);
    }
    if has("leg") {
        // A plain "leg" is the upper leg unless the rig also has an upper.
        return side(UpperLegL, UpperLegR);
    }
    if has("torso") || has("chest") || has("spine") || has("body") {
        return Some(Torso);
    }
    None
}

/// Bones found among the nodes under `root` (see `map_bones`).
pub fn find_bones(w: &World, root: usize) -> HashMap<Bone, usize> {
    let mut named = Vec::new();
    w.visit(root, &mut |id, n| {
        if id != ROOT && !n.name.is_empty() {
            named.push((id, n.name.clone()));
        }
    });
    map_bones(&named)
}

/// Bones among `(id, name)` pairs in hierarchy order: the first match
/// wins, and a plain "leg" next to an "upleg" / "thigh" is the lower leg.
pub fn map_bones(named: &[(usize, String)]) -> HashMap<Bone, usize> {
    let mut out = HashMap::new();
    let mut plain_legs = Vec::new();
    for (id, name) in named {
        if let Some(b) = bone_of(name) {
            let lower = name.to_ascii_lowercase();
            if matches!(b, UpperLegL | UpperLegR) && !["up", "thigh", "upper"].iter().any(|k| lower.contains(k)) {
                plain_legs.push((b, *id));
                continue;
            }
            out.entry(b).or_insert(*id);
        }
    }
    for (b, id) in plain_legs {
        let lower = if b == UpperLegL { LowerLegL } else { LowerLegR };
        if out.contains_key(&b) { out.entry(lower).or_insert(id) } else { out.entry(b).or_insert(id) };
    }
    out
}

/// The top-level object a node belongs to.
pub fn top_of(w: &World, mut id: usize) -> usize {
    while let Some(p) = w.node(id).and_then(|n| n.parent).filter(|&p| p != ROOT) {
        id = p;
    }
    id
}

/// The rig to animate: `rig`'s top-level object, or the first top-level
/// object that has `need`.
pub fn rig_root(w: &World, rig: Option<usize>, need: Bone) -> usize {
    match rig {
        Some(id) => top_of(w, id),
        None => {
            let tops = w.node(ROOT).map(|n| n.children().to_vec()).unwrap_or_default();
            tops.into_iter().find(|&t| find_bones(w, t).contains_key(&need)).unwrap_or(ROOT)
        }
    }
}

// ---- templates ----

fn block(size: IVec3, color: u8) -> Volume {
    let mut v = Volume::new(size);
    v.cells.fill(color);
    v.refresh_bricks();
    v
}

struct Builder<'a> {
    w: &'a mut World,
    s: i32,
    prefix: String,
}

impl Builder<'_> {
    fn sz(&self, v: IVec3) -> IVec3 {
        (v * self.s).max(IVec3::ONE)
    }
    /// A part at `at` (template units), pivot at `pivot` (its own units),
    /// attached to `parent` (or top level).
    fn part(&mut self, name: &str, at: IVec3, size: IVec3, color: u8, pivot: Vec3, parent: Option<usize>) -> usize {
        let full = format!("{}{name}", self.prefix);
        let id = self.w.add_model(ROOT, &full, block(self.sz(size), color), at * self.s);
        if let Some(n) = self.w.node_mut(id) {
            n.pivot = Some(pivot * self.s as f32);
        }
        if let Some(p) = parent {
            self.w.reparent(id, p);
        }
        id
    }
}

/// Palette colors for template parts: (main, skin, dark).
pub type Colors = (u8, u8, u8);

/// Add a humanoid made of blocks with its feet at `origin`: hips, torso,
/// head, two-piece arms and legs, joints at the pivots. Returns the hips.
pub fn humanoid(w: &mut World, origin: IVec3, scale: i32, prefix: &str, (main, skin, dark): Colors) -> usize {
    let mut b = Builder { w, s: scale.max(1), prefix: prefix.to_string() };
    let o = origin / b.s;
    let p = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
    let hips = b.part("hips", o + IVec3::new(0, 0, 10), IVec3::new(8, 4, 3), dark, p(4.0, 2.0, 1.5), None);
    let torso = b.part("torso", o + IVec3::new(0, 0, 13), IVec3::new(8, 4, 7), main, p(4.0, 2.0, 0.0), Some(hips));
    b.part("head", o + IVec3::new(1, -1, 20), IVec3::new(6, 6, 6), skin, p(3.0, 3.0, 0.0), Some(torso));
    for (side, x) in [("l", 8), ("r", -3)] {
        let up = b.part(&format!("upper_arm_{side}"), o + IVec3::new(x, 0, 15), IVec3::new(3, 3, 5), main, p(1.5, 1.5, 5.0), Some(torso));
        b.part(&format!("lower_arm_{side}"), o + IVec3::new(x, 0, 10), IVec3::new(3, 3, 5), skin, p(1.5, 1.5, 5.0), Some(up));
        let lx = if side == "l" { 4 } else { 1 };
        let ul = b.part(&format!("upper_leg_{side}"), o + IVec3::new(lx, 0, 5), IVec3::new(3, 4, 5), dark, p(1.5, 2.0, 5.0), Some(hips));
        b.part(&format!("lower_leg_{side}"), o + IVec3::new(lx, 0, 0), IVec3::new(3, 4, 5), dark, p(1.5, 2.0, 5.0), Some(ul));
    }
    let w = b.w;
    if let Some(n) = w.node_mut(torso) {
        n.sockets.push(Socket { name: "back".into(), pos: Vec3::new(4.0, 4.0, 4.0) * scale as f32 });
    }
    for side in ["l", "r"] {
        if let Some(id) = w.find(&format!("{prefix}lower_arm_{side}")) {
            if let Some(n) = w.node_mut(id) {
                n.sockets.push(Socket { name: format!("hand_{side}"), pos: Vec3::new(1.5, 1.5, 0.0) * scale as f32 });
            }
        }
    }
    hips
}

/// Add a four-legged animal with its feet at `origin`, facing -y. Returns
/// the body.
pub fn quadruped(w: &mut World, origin: IVec3, scale: i32, prefix: &str, (main, skin, dark): Colors) -> usize {
    let mut b = Builder { w, s: scale.max(1), prefix: prefix.to_string() };
    let o = origin / b.s;
    let p = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
    let body = b.part("body", o + IVec3::new(0, 0, 5), IVec3::new(6, 12, 5), main, p(3.0, 6.0, 2.5), None);
    b.part("head", o + IVec3::new(1, -5, 8), IVec3::new(4, 5, 4), skin, p(2.0, 5.0, 1.0), Some(body));
    b.part("tail", o + IVec3::new(2, 12, 8), IVec3::new(2, 5, 2), dark, p(1.0, 0.0, 1.0), Some(body));
    for (name, x, y) in [("leg_front_l", 4, 0), ("leg_front_r", 0, 0), ("leg_back_l", 4, 10), ("leg_back_r", 0, 10)] {
        b.part(name, o + IVec3::new(x, y, 0), IVec3::new(2, 2, 5), dark, p(1.0, 1.0, 5.0), Some(body));
    }
    body
}

// ---- motions ----

/// The motions `apply_motion` knows, with a one-line description.
pub const MOTIONS: &[(&str, &str)] = &[
    ("idle", "Breathing and a slight sway"),
    ("walk", "A walk cycle"),
    ("run", "A run cycle, leaning forward"),
    ("jump", "Crouch, leap and land"),
    ("attack", "Right arm wind-up and strike"),
    ("wave", "Right arm raised, waving"),
    ("death", "Falls over backwards"),
    ("quad_walk", "Four-legged walk"),
    ("quad_idle", "Four-legged idle: head bob and tail wag"),
];

/// A bone's offset from its rest pose at phase `t` (0..1 over the cycle):
/// turn in degrees about its own x, y, z, and a move in voxels (root only).
type Pose = HashMap<Bone, (Vec3, Vec3)>;

fn motion(name: &str, t: f32, amount: f32) -> Option<Pose> {
    let a = amount;
    let s = (t * TAU).sin();
    let c = (t * TAU).cos();
    let s2 = (t * 2.0 * TAU).sin();
    let mut p: Pose = HashMap::new();
    let mut turn = |b: Bone, d: Vec3| {
        p.entry(b).or_insert((Vec3::ZERO, Vec3::ZERO)).0 += d * a;
    };
    let x = |v: f32| Vec3::new(v, 0.0, 0.0);
    let y = |v: f32| Vec3::new(0.0, v, 0.0);
    let z = |v: f32| Vec3::new(0.0, 0.0, v);
    let mut lift = Vec3::ZERO;
    match name {
        "idle" => {
            turn(Torso, x(2.0 * s));
            turn(Head, y(4.0 * (t * TAU + 1.0).sin()) + x(-2.0 * s));
            turn(UpperArmL, y(-4.0 - 2.0 * s));
            turn(UpperArmR, y(4.0 + 2.0 * s));
            lift.z = -0.6 * (0.5 - 0.5 * c);
        }
        "walk" | "run" => {
            let run = name == "run";
            let (leg, knee, arm) = if run { (50.0, 80.0, 55.0) } else { (28.0, 40.0, 25.0) };
            turn(UpperLegL, x(-leg * s));
            turn(UpperLegR, x(leg * s));
            turn(LowerLegL, x(knee * c.max(0.0)));
            turn(LowerLegR, x(knee * (-c).max(0.0)));
            turn(UpperArmL, x(arm * s));
            turn(UpperArmR, x(-arm * s));
            turn(LowerArmL, x(if run { -70.0 } else { -15.0 }));
            turn(LowerArmR, x(if run { -70.0 } else { -15.0 }));
            turn(Torso, z(4.0 * s) + x(if run { 12.0 } else { 2.0 }));
            turn(Head, x(if run { -8.0 } else { 0.0 }));
            lift.z = if run { -1.5 } else { -0.8 } * (0.5 - 0.5 * (2.0 * t * TAU).cos());
        }
        "jump" => {
            // 0-0.25 crouch, 0.25-0.6 up and down, 0.6-0.8 land, then stand.
            let crouch = |u: f32| (u * std::f32::consts::PI).sin().max(0.0);
            let (down, air) = if t < 0.25 {
                (crouch(t / 0.25 * 0.5), 0.0)
            } else if t < 0.6 {
                (0.0, ((t - 0.25) / 0.35 * std::f32::consts::PI).sin())
            } else if t < 0.85 {
                (crouch((t - 0.6) / 0.25), 0.0)
            } else {
                (0.0, 0.0)
            };
            turn(UpperLegL, x(-50.0 * down - 15.0 * air));
            turn(UpperLegR, x(-50.0 * down - 15.0 * air));
            turn(LowerLegL, x(80.0 * down + 20.0 * air));
            turn(LowerLegR, x(80.0 * down + 20.0 * air));
            turn(Torso, x(20.0 * down));
            turn(UpperArmL, x(35.0 * down - 150.0 * air));
            turn(UpperArmR, x(35.0 * down - 150.0 * air));
            lift.z = -3.0 * down + 8.0 * air;
        }
        "attack" => {
            // Wind up (0-0.4), strike (0.4-0.55), recover.
            let (wind, hit) = if t < 0.4 {
                ((t / 0.4).min(1.0), 0.0)
            } else if t < 0.55 {
                (1.0 - (t - 0.4) / 0.15, (t - 0.4) / 0.15)
            } else {
                (0.0, 1.0 - (t - 0.55) / 0.45)
            };
            turn(UpperArmR, x(-160.0 * wind - 40.0 * hit));
            turn(LowerArmR, x(-30.0 * wind));
            turn(Torso, z(-20.0 * wind + 25.0 * hit) + x(10.0 * hit));
            turn(UpperArmL, x(15.0 * wind));
            turn(UpperLegL, x(-15.0 * hit));
            turn(LowerLegL, x(15.0 * hit));
        }
        "wave" => {
            let up = (t / 0.15).min(1.0) * (((1.0 - t) / 0.15).min(1.0));
            turn(UpperArmR, y(160.0 * up));
            turn(LowerArmR, y(-30.0 * up * (0.5 + 0.5 * (t * 3.0 * TAU).sin())));
            turn(Head, z(-8.0 * up));
        }
        "death" => {
            let f = (t / 0.6).min(1.0);
            let f = f * f;
            turn(Hips, x(-85.0 * f));
            turn(UpperArmL, y(-60.0 * f));
            turn(UpperArmR, y(60.0 * f));
            turn(Head, x(-20.0 * f));
            lift.z = -6.0 * f;
        }
        "quad_walk" => {
            turn(LegFL, x(-25.0 * s));
            turn(LegBR, x(-25.0 * s));
            turn(LegFR, x(25.0 * s));
            turn(LegBL, x(25.0 * s));
            turn(Head, x(5.0 * s2));
            turn(Tail, z(15.0 * s));
            lift.z = -0.6 * (0.5 - 0.5 * (2.0 * t * TAU).cos());
        }
        "quad_idle" => {
            turn(Tail, z(30.0 * (t * 2.0 * TAU).sin()));
            turn(Head, x(6.0 * s) + z(5.0 * (t * TAU + 1.0).sin()));
        }
        _ => return None,
    }
    // The move goes on the root: hips, or the body of a four-legged rig.
    let root = if name.starts_with("quad") { Torso } else { Hips };
    p.entry(root).or_insert((Vec3::ZERO, Vec3::ZERO)).1 += lift * a;
    Some(p)
}

/// What applying a motion did.
pub struct Applied {
    pub bones: Vec<(Bone, String)>,
    pub missing: Vec<Bone>,
    pub keys: usize,
}

/// Key `name` onto the rig in `w` over `frames` frames from frame 0,
/// replacing the keys of the bones it moves. The rest pose is each bone's
/// pose on frame 0.
///
/// `rig` is any part of the rig to animate (its top-level object is used);
/// without one, the first top-level object with the limbs the motion needs.
pub fn apply_motion(w: &mut World, rig: Option<usize>, name: &str, frames: u32, amount: f32) -> Result<Applied, String> {
    motion(name, 0.0, 1.0).ok_or_else(|| format!("unknown motion `{name}`; try {}", MOTIONS.iter().map(|m| m.0).collect::<Vec<_>>().join(", ")))?;
    let quad = name.starts_with("quad");
    let root = rig_root(w, rig, if quad { LegFL } else { UpperLegL });
    let mut bones = find_bones(w, root);
    if quad {
        // A four-legged rig's body plays the torso / root.
        if let Some(&b) = bones.get(&Torso).or(bones.get(&Hips)) {
            bones.insert(Torso, b);
        }
    }
    let frames = frames.max(2);
    if w.anim.frames < frames {
        w.anim.frames = frames;
    }
    w.anim.smooth = true;
    w.anim.looping = !matches!(name, "death" | "jump" | "attack");
    w.set_frame(0);
    let rest: HashMap<Bone, Key> = bones.iter().filter_map(|(&b, &id)| Some((b, w.node(id)?.key_now(0)))).collect();
    let used: Vec<Bone> = motion(name, 0.25, 1.0).unwrap().keys().copied().chain(motion(name, 0.6, 1.0).unwrap().keys().copied()).collect();
    let mut applied = Applied { bones: Vec::new(), missing: Vec::new(), keys: 0 };
    let step = (frames / 12).max(1);
    let mut sample: Vec<u32> = (0..frames).step_by(step as usize).collect();
    if !w.anim.looping && sample.last() != Some(&(frames - 1)) {
        sample.push(frames - 1);
    }
    let looping = w.anim.looping;
    let mut done = Vec::new();
    for b in used {
        if done.contains(&b) {
            continue;
        }
        done.push(b);
        let (Some(&id), Some(r)) = (bones.get(&b), rest.get(&b)) else {
            applied.missing.push(b);
            continue;
        };
        let Some(n) = w.node_mut(id) else { continue };
        n.keys.clear();
        for &f in &sample {
            // Looping cycles end where they start, so sample a full turn.
            let t = if looping { f as f32 / frames as f32 } else { f as f32 / (frames - 1) as f32 };
            let (turn, lift) = motion(name, t, amount).unwrap().get(&b).copied().unwrap_or_default();
            let q = tilt_quat(r.tilt) * tilt_quat(turn);
            let k = Key { frame: f, rot: r.rot, t: r.t + lift.round().as_ivec3(), tilt: tilt_deg(q) };
            n.set_key(k);
            applied.keys += 1;
        }
        let label = w.label(id);
        applied.bones.push((b, label));
    }
    w.set_frame(0);
    Ok(applied)
}

/// Which part each motion moves, by bone (for messages).
pub fn bone_name(b: Bone) -> &'static str {
    match b {
        Hips => "hips",
        Torso => "torso",
        Head => "head",
        UpperArmL => "upper_arm_l",
        LowerArmL => "lower_arm_l",
        UpperArmR => "upper_arm_r",
        LowerArmR => "lower_arm_r",
        UpperLegL => "upper_leg_l",
        LowerLegL => "lower_leg_l",
        UpperLegR => "upper_leg_r",
        LowerLegR => "lower_leg_r",
        LegFL => "leg_front_l",
        LegFR => "leg_front_r",
        LegBL => "leg_back_l",
        LegBR => "leg_back_r",
        Tail => "tail",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_from_other_rigs_map_to_bones() {
        for (name, bone) in [
            ("mixamorig:LeftArm", UpperArmL),
            ("mixamorig:LeftForeArm", LowerArmL),
            ("mixamorig:RightUpLeg", UpperLegR),
            ("upperarm_l", UpperArmL),
            ("lowerarm_r", LowerArmR),
            ("thigh_l", UpperLegL),
            ("calf_r", LowerLegR),
            ("pelvis", Hips),
            ("spine_02", Torso),
            ("upper_arm.L", UpperArmL),
            ("shin.R", LowerLegR),
            ("Head", Head),
            ("leg_front_l", LegFL),
            ("hind_leg_r", LegBR),
            ("tail", Tail),
            ("DEF-hips", Hips),
            ("DEF-spine.001", Torso),
            ("DEF-upper_arm.L", UpperArmL),
            ("DEF-forearm.R", LowerArmR),
            ("DEF-thigh.L", UpperLegL),
            ("DEF-shin.R", LowerLegR),
        ] {
            assert_eq!(bone_of(name), Some(bone), "{name}");
        }
        assert_eq!(bone_of("sword"), None);
        // Collarbones aren't arms.
        assert_eq!(bone_of("mixamorig:LeftShoulder"), None);
        for other in ["root", "DEF-f_index.01.L", "DEF-hand.L", "DEF-neck", "DEF-toe.R"] {
            assert_eq!(bone_of(other), None, "{other}");
        }
    }

    #[test]
    fn humanoid_template_is_found_and_walks() {
        let mut w = World::single(Volume::new(IVec3::ONE));
        let first = w.models()[0];
        w.delete(first);
        let hips = humanoid(&mut w, IVec3::ZERO, 1, "", (1, 2, 3));
        let bones = find_bones(&w, ROOT);
        assert_eq!(bones.len(), 11, "{bones:?}");
        assert_eq!(bones[&Hips], hips);
        // Arms hang from the torso, forearms from the arms.
        assert_eq!(w.node(bones[&LowerArmL]).unwrap().parent, Some(bones[&UpperArmL]));
        let foot = w.bounds(bones[&LowerLegL]).unwrap();
        assert_eq!(foot.0.z, 0);

        let a = apply_motion(&mut w, None, "walk", 24, 1.0).unwrap();
        assert!(a.missing.is_empty() && a.keys > 0);
        assert_eq!(w.anim.frames, 24);
        // A quarter of the way through, the left leg is forward (its foot
        // toward -y) and the right one back.
        w.set_frame(6);
        let foot_y = |w: &World, b: Bone| w.affine(bones[&b]).transform_point3(Vec3::new(1.5, 2.0, 0.0)).y;
        assert!(foot_y(&w, LowerLegL) < foot_y(&w, LowerLegR) - 1.0);
        // The loop ends where it started.
        w.set_frame(0);
        let start = foot_y(&w, LowerLegL);
        w.set_frame(23);
        assert!((foot_y(&w, LowerLegL) - start).abs() < 2.0);
        assert!(apply_motion(&mut w, None, "moonwalk", 24, 1.0).is_err());
    }

    #[test]
    fn quadruped_template_walks() {
        let mut w = World::single(Volume::new(IVec3::ONE));
        let first = w.models()[0];
        w.delete(first);
        // Next to a humanoid: the motion finds the four-legged rig's parts.
        humanoid(&mut w, IVec3::ZERO, 1, "", (1, 2, 3));
        let body = quadruped(&mut w, IVec3::new(20, 0, 0), 2, "dog_", (1, 2, 3));
        let a = apply_motion(&mut w, None, "quad_walk", 16, 1.0).unwrap();
        assert!(a.missing.is_empty(), "{:?}", a.missing);
        assert!(a.bones.iter().all(|(_, name)| name.starts_with("dog_")), "{:?}", a.bones);
        // Picking the dog's tail also picks the dog.
        let tail = w.find("dog_tail").unwrap();
        let a = apply_motion(&mut w, Some(tail), "quad_idle", 16, 1.0).unwrap();
        assert!(a.bones.iter().any(|(_, name)| name == "dog_head") && w.node(body).is_some());
    }
}
