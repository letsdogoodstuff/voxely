//! Arranging scenes and detailing surfaces: drop parts onto what's below
//! them, project a small pixel pattern onto a curved surface (faces,
//! emblems, carved holes), hollow a model into a shell.

use std::collections::{HashMap, VecDeque};

use anyhow::{Context, Result, bail};
use glam::IVec3;
use serde_json::{Value, json};

use super::{Ctx, NEIGHBORS, apply, color, edited, find_part, get, int, string, text, tool, tree_changed, xyz};
use crate::scene::Volume;

pub const PART_TOOLS: &[&str] = &["project_pattern"];

pub fn tools() -> Vec<Value> {
    let side = json!({ "type": "string", "enum": ["front", "back", "left", "right", "top", "bottom"],
        "description": "The side it's seen from, as in screenshot's views: front = from -y (default), back = from +y, left = from -x, right = from +x, top = from +z, bottom = from -z. The pattern reads unmirrored from that side." });
    vec![
        tool("drop_part", "Move parts or groups straight down (or up) until they rest on what's under them: the ground, a table, a headstone. Each one lands on the highest surface beneath its footprint, so it never sinks in; with nothing beneath, it sits on z = 0. Parts are dropped in the order given, so a stack can be built bottom first.", json!({
            "parts": { "type": "array", "items": { "type": ["string", "integer"] }, "minItems": 1, "description": "Parts or groups to drop" },
            "onto": { "type": ["string", "integer"], "description": "Only land on this part or group (default: any other visible part below it)" },
        }), &["parts"]),
        tool("project_pattern", "Draw a small pixel pattern onto the surface of the part, as seen from one side: each pattern cell is shot straight in from that side and changes the first voxel it hits, so it wraps around curved shapes. For faces (eyes, mouths, a jack-o'-lantern's carved face), emblems, buttons, windows and signs. Color 0 carves: with depth \"through\" it cuts a hole through the wall to the hollow inside.", json!({
            "rows": { "type": "array", "items": { "type": "string" }, "minItems": 1, "description": "The pattern, top row first, one character per cell; \".\" and \" \" leave the cell alone. E.g. [\"#.#\", \"...\", \"###\"]" },
            "colors": { "type": "object", "description": "What each character does: {\"#\": \"#1a1828\", \"o\": 12, \"x\": 0}; 0 carves the cell out" },
            "side": side,
            "center": xyz("Where the pattern's middle goes, in the part's own coordinates; only the two coordinates across the side are used (default: the part's center)"),
            "depth": { "type": ["integer", "string"], "description": "How many voxels deep each cell paints or carves from the surface (default 1), or \"through\": the whole wall, up to the hollow behind it (walls up to 8 thick; cells that skim along a wall go as deep as the others)" },
            "scale": { "type": "integer", "minimum": 1, "maximum": 16, "description": "Voxels per pattern cell (default 1)" },
            "raise": { "type": "boolean", "description": "Add the colors one voxel out from the surface instead of replacing it (raised buttons, rivets); carving ignores it" },
        }), &["rows", "colors"]),
    ]
}

pub fn dispatch(ctx: &mut Ctx, name: &str, args: &Value) -> Option<Result<Vec<Value>>> {
    Some(match name {
        "drop_part" => drop_parts(ctx, args),
        "project_pattern" => project_pattern(ctx, args),
        _ => return None,
    })
}

// ---------------------------------------------------------------- drop

fn drop_parts(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let ids = get(a, "parts").and_then(Value::as_array).context("missing `parts`")?.iter().map(|p| find_part(ctx.scene, Some(p))).collect::<Result<Vec<_>>>()?;
    let onto = match get(a, "onto") {
        Some(p) => {
            let id = find_part(ctx.scene, Some(p))?;
            let mut set = Vec::new();
            ctx.scene.world.visit(id, &mut |i, _| set.push(i));
            Some(set)
        }
        None => None,
    };
    let mut lines = Vec::new();
    for id in ids {
        let world = ctx.scene.synced_world().into_owned();
        let label = world.label(id);
        let mut own = Vec::new();
        world.visit(id, &mut |i, _| own.push(i));
        // Lowest solid cell per world column of what's being dropped.
        let mut low: HashMap<(i32, i32), i32> = HashMap::new();
        let mut top = i32::MIN;
        for &m in &own {
            if !world.visible(m) {
                continue;
            }
            let Some((v, origin)) = world.checkout(m) else { continue };
            for (p, _) in v.filled() {
                let w = origin + p;
                low.entry((w.x, w.y)).and_modify(|z| *z = (*z).min(w.z)).or_insert(w.z);
                top = top.max(w.z);
            }
        }
        if low.is_empty() {
            bail!("`{label}` has no visible voxels to drop");
        }
        let others = world.composite_where(|m| !own.contains(&m) && onto.as_ref().is_none_or(|s| s.contains(&m)));
        // How far each column could fall before touching something under it.
        let mut best: Option<(i32, usize)> = None;
        if let Some(c) = &others {
            for (&(x, y), &z0) in &low {
                let col = IVec3::new(x, y, 0) - c.origin;
                if col.x < 0 || col.y < 0 || col.x >= c.volume.dims.x || col.y >= c.volume.dims.y {
                    continue;
                }
                // The highest solid cell below this thing's top (things
                // floating above it, like a moon, don't count).
                let hi = (top - c.origin.z).min(c.volume.dims.z - 1);
                for z in (0..=hi).rev() {
                    let p = IVec3::new(col.x, col.y, z);
                    if c.volume.get(p) != 0 {
                        let dz = c.origin.z + z + 1 - z0;
                        if best.is_none_or(|(b, _)| dz > b) {
                            best = Some((dz, c.owners.get(c.volume.index(p)).unwrap_or(0) as usize));
                        }
                        break;
                    }
                }
            }
        }
        let (dz, on) = match best {
            Some((dz, owner)) => (dz, format!("`{}`", world.label(owner))),
            None if onto.is_some() => bail!("nothing of `{}` is under `{label}`", world.label(find_part(ctx.scene, get(a, "onto"))?)),
            None => (-low.values().min().copied().unwrap_or(0), "the floor (z = 0)".to_string()),
        };
        if dz != 0 {
            ctx.scene.change(|w| w.translate(id, IVec3::new(0, 0, dz)));
            tree_changed(ctx);
        }
        let lo = ctx.scene.world.bounds(id).map_or(IVec3::ZERO, |b| b.0);
        lines.push(format!("`{label}` {} onto {on}, now at {:?}", match dz {
            0 => "already rests".to_string(),
            d if d < 0 => format!("dropped {}", -d),
            d => format!("raised {d}"),
        }, lo.to_array()));
    }
    Ok(vec![text(lines.join("\n"))])
}

// ---------------------------------------------------------------- pattern

/// (into the surface, pattern right, pattern up) for a side.
fn side_axes(side: &str) -> Result<(IVec3, IVec3, IVec3)> {
    Ok(match side {
        "front" => (IVec3::Y, IVec3::X, IVec3::Z),
        "back" => (IVec3::NEG_Y, IVec3::NEG_X, IVec3::Z),
        "left" => (IVec3::X, IVec3::NEG_Y, IVec3::Z),
        "right" => (IVec3::NEG_X, IVec3::Y, IVec3::Z),
        "top" => (IVec3::NEG_Z, IVec3::X, IVec3::Y),
        "bottom" => (IVec3::Z, IVec3::X, IVec3::NEG_Y),
        s => bail!("unknown side {s:?}; use front, back, left, right, top or bottom"),
    })
}

/// The thickest wall "through" cuts; a longer run of solid runs along a
/// wall rather than across it.
const MAX_WALL: i32 = 8;

enum Depth {
    Cells(i32),
    Through,
}

fn project_pattern(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let rows: Vec<Vec<char>> = get(a, "rows").and_then(Value::as_array).context("missing `rows`")?.iter().map(|r| r.as_str().map(|s| s.chars().collect()).context("`rows` holds strings")).collect::<Result<_>>()?;
    let mut legend = HashMap::new();
    for (k, v) in get(a, "colors").and_then(Value::as_object).context("missing `colors`, e.g. {\"#\": 1}")? {
        let mut chars = k.chars();
        let (Some(ch), None) = (chars.next(), chars.next()) else { bail!("`colors` keys are single characters, not {k:?}") };
        legend.insert(ch, color(ctx.scene, v).with_context(|| format!("`colors` {k:?}"))?);
    }
    for (i, row) in rows.iter().enumerate() {
        if let Some(ch) = row.iter().find(|&&c| c != '.' && c != ' ' && !legend.contains_key(&c)) {
            bail!("row {i} uses {ch:?}, which `colors` doesn't define");
        }
    }
    let (into, right, up) = side_axes(string(a, "side").unwrap_or("front"))?;
    let depth = match get(a, "depth") {
        None => Depth::Cells(1),
        Some(Value::String(s)) if s == "through" => Depth::Through,
        Some(v) => match v.as_i64() {
            Some(n) if (1..=1024).contains(&n) => Depth::Cells(n as i32),
            _ => bail!("`depth` is a number of voxels (1 or more) or \"through\", not {v}"),
        },
    };
    let scale = int(a, "scale").unwrap_or(1).clamp(1, 16) as i32;
    let raise = super::boolean(a, "raise").unwrap_or(false);
    let old = &ctx.scene.volume;
    let dims = old.dims;
    let center = super::opt_ivec3(a, "center")?.unwrap_or(dims / 2);
    let h = rows.len() as i32;
    let w = rows.iter().map(Vec::len).max().unwrap_or(0) as i32;
    let mut new = old.clone();
    let (mut changed, mut missed) = (0, 0);
    // Where a ray starts: the canvas face it enters through.
    let entry = |p: IVec3| -> IVec3 {
        let mut p = p;
        for k in 0..3 {
            if into[k] > 0 {
                p[k] = 0;
            } else if into[k] < 0 {
                p[k] = dims[k] - 1;
            }
        }
        p
    };
    // The first solid cell each pattern cell's ray hits, with its color.
    let mut hits = Vec::new();
    for (j, row) in rows.iter().enumerate() {
        for (i, &ch) in row.iter().enumerate() {
            let Some(&c) = legend.get(&ch) else { continue };
            for sj in 0..scale {
                for si in 0..scale {
                    // Pattern cell (i, j) from the middle, top row up.
                    let du = (i as i32 - w / 2) * scale + si - (scale - 1) / 2;
                    let dv = (h / 2 - j as i32) * scale - sj + (scale - 1) / 2;
                    let mut p = entry(center + right * du + up * dv);
                    while old.contains(p) && old.get(p) == 0 {
                        p += into;
                    }
                    if old.contains(p) {
                        hits.push((p, c));
                    } else {
                        missed += 1;
                    }
                }
            }
        }
    }
    // "through": each ray's run of solid cells to the hollow behind the
    // wall. A ray that doesn't reach a hollow within MAX_WALL is running
    // along a wall (a floor, a curved side), not through it; it's cut as
    // deep as the walls the other rays went through, not into a trench.
    let run = |p: IVec3| -> Option<i32> {
        (1..=MAX_WALL).find(|&n| {
            let q = p + into * n;
            !old.contains(q) || old.get(q) == 0
        })
    };
    let mut runs: Vec<i32> = match depth {
        Depth::Through => hits.iter().filter_map(|&(p, _)| run(p)).collect(),
        Depth::Cells(_) => Vec::new(),
    };
    runs.sort_unstable();
    let wall = runs.get(runs.len() / 2).copied().unwrap_or(1);
    let mut shallow = 0;
    for (p, c) in hits {
        if c != 0 && raise {
            let q = p - into;
            if old.contains(q) {
                new.set(q, c);
                changed += 1;
            } else {
                missed += 1;
            }
            continue;
        }
        let n = match depth {
            Depth::Cells(d) => d,
            Depth::Through => run(p).unwrap_or_else(|| {
                shallow += 1;
                wall
            }),
        };
        for k in 0..n {
            let q = p + into * k;
            if !old.contains(q) || old.get(q) == 0 {
                break;
            }
            new.set(q, c);
            changed += 1;
        }
    }
    let mut what = format!("Projected a {w}x{h} pattern");
    if missed > 0 {
        what.push_str(&format!("; {missed} cells missed the model (nothing behind them)"));
    }
    if shallow > 0 {
        what.push_str(&format!("; {shallow} cells ran along a wall instead of through it, so they were cut {wall} deep like the rest"));
    }
    let committed = changed > 0 && apply(ctx, new);
    Ok(edited(ctx, committed, &what))
}

// ---------------------------------------------------------------- hollow

/// Keep only the voxels within `thickness` of the outside (empty space
/// connected to the canvas edge), so a closed shape becomes a shell. With
/// `inner`, the shell's innermost layer (next to the new hollow) takes
/// that color: the lit inside of a lantern, the flesh inside a fruit.
pub fn hollow_volume(v: &Volume, thickness: i32, inner: Option<u8>) -> Volume {
    let dims = v.dims;
    let n = (dims.x * dims.y * dims.z) as usize;
    // Distance (in face steps) from the outside; 0 for outside cells.
    let mut dist = vec![i32::MAX; n];
    let mut queue = VecDeque::new();
    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                let p = IVec3::new(x, y, z);
                let edge = x == 0 || y == 0 || z == 0 || x == dims.x - 1 || y == dims.y - 1 || z == dims.z - 1;
                if edge {
                    let i = v.index(p);
                    // A solid canvas-edge cell touches the outside.
                    dist[i] = if v.get(p) == 0 { 0 } else { 1 };
                    queue.push_back(p);
                }
            }
        }
    }
    // Empty cells spread distance 0 (they're outside too); solids add one.
    while let Some(p) = queue.pop_front() {
        let d = dist[v.index(p)];
        for o in NEIGHBORS {
            let q = p + o;
            if !v.contains(q) {
                continue;
            }
            let j = v.index(q);
            let nd = if v.get(q) == 0 { if d == 0 { 0 } else { continue } } else { d + 1 };
            if nd < dist[j] {
                dist[j] = nd;
                if nd == 0 {
                    queue.push_front(q);
                } else {
                    queue.push_back(q);
                }
            }
        }
    }
    let mut out = v.clone();
    for (p, _) in v.filled() {
        if dist[v.index(p)] > thickness {
            out.set(p, 0);
        }
    }
    if let Some(c) = inner {
        for (p, _) in out.filled_sorted() {
            if NEIGHBORS.iter().any(|&o| v.contains(p + o) && v.get(p + o) != 0 && out.get(p + o) == 0) {
                out.set(p, c);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ball(r: i32) -> Volume {
        let s = 2 * r + 3;
        let mut v = Volume::new(IVec3::splat(s));
        let c = IVec3::splat(r + 1);
        for z in 0..s {
            for y in 0..s {
                for x in 0..s {
                    let p = IVec3::new(x, y, z);
                    if (p - c).length_squared() <= r * r {
                        v.set(p, 4);
                    }
                }
            }
        }
        v
    }

    #[test]
    fn hollow_keeps_a_shell_and_colors_its_inside() {
        let v = ball(6);
        let h = hollow_volume(&v, 2, Some(7));
        let c = IVec3::splat(7);
        assert_eq!(h.get(c), 0, "the middle is hollow");
        assert_eq!(h.get(c + IVec3::new(6, 0, 0)), 4, "the outer skin stays");
        assert_eq!(h.get(c + IVec3::new(5, 0, 0)), 7, "the inner layer takes the inside color");
        assert_eq!(h.get(c + IVec3::new(4, 0, 0)), 0);
        assert!(h.count() < v.count());
    }

    #[test]
    fn hollow_leaves_thin_shapes_alone() {
        let mut v = Volume::new(IVec3::new(5, 5, 1));
        for x in 0..5 {
            v.set(IVec3::new(x, 2, 0), 3);
        }
        assert_eq!(hollow_volume(&v, 1, Some(9)).count(), 5);
    }

    #[test]
    fn sides_read_unmirrored() {
        // Right of the pattern is image right when seen from that side.
        for (side, cam) in [("front", IVec3::NEG_Y), ("back", IVec3::Y), ("left", IVec3::NEG_X), ("right", IVec3::X), ("top", IVec3::Z), ("bottom", IVec3::NEG_Z)] {
            let (into, right, up) = side_axes(side).unwrap();
            assert_eq!(into, -cam, "{side}");
            // right x up points back at the viewer.
            assert_eq!(right.cross(up), cam, "{side}");
        }
    }
}
