//! Model health for `get_scene_info health`: pieces that hang in the air
//! (not face-connected to the rest), parts overlapping each other, empty
//! parts, and whether the model stands on the ground.

use std::collections::BTreeMap;

use glam::IVec3;
use serde_json::{Value, json};

use super::arr;
use crate::scene::{Scene, Volume};
use crate::world::World;

/// Floating pieces and overlaps listed in full; the rest are counted.
const LIST: usize = 12;

const SIDES: [IVec3; 6] = [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::Z, IVec3::NEG_Z];

/// One face-connected piece of the composite.
struct Piece {
    voxels: usize,
    lo: IVec3,
    hi: IVec3,
    parts: BTreeMap<u32, usize>,
}

/// Face-connected pieces of `v`, largest first. `owner` names the model at
/// each cell index.
fn pieces(v: &Volume, owner: impl Fn(usize) -> u32) -> Vec<Piece> {
    let mut seen = crate::scene::Mask::new(v.dims);
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for (start, _) in v.filled() {
        if seen.at(start) {
            continue;
        }
        seen.set_at(start, true);
        stack.push(start);
        let mut p = Piece { voxels: 0, lo: start, hi: start, parts: BTreeMap::new() };
        while let Some(c) = stack.pop() {
            p.voxels += 1;
            p.lo = p.lo.min(c);
            p.hi = p.hi.max(c);
            *p.parts.entry(owner(v.index(c))).or_default() += 1;
            for d in SIDES {
                let n = c + d;
                if v.get(n) != 0 && !seen.at(n) {
                    seen.set_at(n, true);
                    stack.push(n);
                }
            }
        }
        out.push(p);
    }
    out.sort_by(|a, b| b.voxels.cmp(&a.voxels));
    out
}

/// Cells two visible models both fill, per pair of models.
fn overlaps(w: &World, models: &[usize], comp: &crate::world::Composite) -> BTreeMap<(usize, usize), usize> {
    let mut out: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for &m in models {
        let Some(own) = w.composite_where(|x| x == m) else { continue };
        for (p, _) in own.volume.filled() {
            let q = p + own.origin - comp.origin;
            let other = comp.owners.at(q) as usize;
            if other != 0 && other != m {
                *out.entry((m.min(other), m.max(other))).or_default() += 1;
            }
        }
    }
    // The composite keeps the later model in a shared cell, so each cell
    // is counted once: from the model it hides.
    out
}

fn plural(n: usize, what: &str) -> String {
    format!("{n} {what}{}", if n == 1 { "" } else { "s" })
}

pub fn report(scene: &Scene) -> Value {
    let w = scene.synced_world();
    let models: Vec<usize> = w.models().into_iter().filter(|&m| w.visible(m)).collect();
    let empty: Vec<String> = models.iter().filter(|&&m| w.node(m).and_then(|n| n.volume()).is_none_or(|v| v.count() == 0)).map(|&m| w.label(m)).collect();
    let Some(comp) = w.composite(None) else {
        return json!({ "summary": "The scene has no visible voxels." });
    };
    let name = |id: u32| w.label(id as usize);
    let all = pieces(&comp.volume, |i| comp.owners.get(i).unwrap_or(0));
    if all.is_empty() {
        return json!({ "summary": "The scene has no visible voxels.", "empty_parts": empty });
    }
    let main = &all[0];
    let floating: Vec<Value> = all[1..]
        .iter()
        .take(LIST)
        .map(|p| {
            json!({
                "voxels": p.voxels,
                "parts": p.parts.keys().map(|&id| name(id)).collect::<Vec<_>>(),
                "from": arr(p.lo + comp.origin),
                "to": arr(p.hi + comp.origin),
            })
        })
        .collect();
    let over = overlaps(&w, &models, &comp);
    let overlap_list: Vec<Value> = over.iter().take(LIST).map(|(&(a, b), &n)| json!({ "parts": [w.label(a), w.label(b)], "cells": n })).collect();
    let ground = comp.origin.z + main.lo.z;

    let mut summary = Vec::new();
    match all.len() - 1 {
        0 => summary.push("All voxels are one connected piece.".to_string()),
        n => summary.push(format!(
            "{n} piece{} not attached to the main model ({}): each touches the rest only at an edge or corner, or not at all.",
            if n == 1 { " is" } else { "s are" },
            plural(all[1..].iter().map(|p| p.voxels).sum::<usize>(), "voxel")
        )),
    }
    if !over.is_empty() {
        summary.push(format!(
            "{} of parts overlap ({}); where they do, the later part hides the earlier one.",
            plural(over.len(), "pair"),
            plural(over.values().sum::<usize>(), "cell")
        ));
    }
    if ground != 0 {
        summary.push(format!("The main model's lowest voxel is at z = {ground}, not on the ground (z = 0)."));
    }
    if !empty.is_empty() {
        summary.push(format!("{} part{} empty.", empty.len(), if empty.len() == 1 { " is" } else { "s are" }));
    }
    json!({
        "summary": summary.join(" "),
        "main_piece": {
            "voxels": main.voxels,
            "from": arr(main.lo + comp.origin),
            "to": arr(main.hi + comp.origin),
        },
        "floating": floating,
        "floating_count": all.len() - 1,
        "overlaps": overlap_list,
        "overlap_pairs": over.len(),
        "lowest_z": ground,
        "empty_parts": empty,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vol(dims: [i32; 3], cells: &[[i32; 3]]) -> Volume {
        let mut v = Volume::new(IVec3::from_array(dims));
        for &c in cells {
            v.set(IVec3::from_array(c), 1);
        }
        v
    }

    #[test]
    fn faces_connect_but_edges_and_corners_do_not() {
        let v = vol([4, 4, 4], &[[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 2, 1]]);
        let p = pieces(&v, |_| 1);
        assert_eq!(p.iter().map(|p| p.voxels).collect::<Vec<_>>(), vec![2, 1, 1]);
    }

    #[test]
    fn one_scene_reports_floating_overlap_and_ground() {
        let mut scene = Scene::empty(IVec3::new(8, 8, 8));
        scene.volume = vol([8, 8, 8], &[[0, 0, 2], [1, 0, 2], [5, 5, 6]]);
        let r = report(&scene);
        assert_eq!(r["floating_count"], 1);
        assert_eq!(r["lowest_z"], 2);
        assert_eq!(r["floating"][0]["from"], json!([5, 5, 6]));
        assert!(r["summary"].as_str().unwrap().contains("not on the ground"));
    }
}
