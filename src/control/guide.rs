//! Voxel-art craft guidance for agents: the `get_guide` tool and the MCP
//! prompts (workflow templates clients show as commands).

use anyhow::{Result, anyhow};
use serde_json::{Value, json};

use super::{Ctx, string, text, tool};

pub const PART_TOOLS: &[&str] = &[];

/// Topics for `get_guide`: (name, one-line summary, body).
const TOPICS: &[(&str, &str, &str)] = &[
    ("craft", "Core voxel-art rules: scale, silhouette, blocking, color, checking your work", CRAFT),
    ("character", "Characters and creatures: proportions, faces, limbs, rigging for games", CHARACTER),
    ("prop", "Props, weapons and vehicles", PROP),
    ("building", "Buildings, rooms and environments", BUILDING),
    ("color", "Choosing palettes and shading by hand", COLOR),
    ("rendering", "Materials, lighting and final renders", RENDERING),
    ("game_export", "Making models that work in game engines (glTF, pivots, sockets)", GAME_EXPORT),
];

const CRAFT: &str = "\
VOXEL CRAFT RULES
1. Pick a scale first and stick to it. Typical: a human-sized character is 16-32 voxels tall (24 is a \
good default); a stylized/chibi one has a head 1/3 of its height. Props are sized relative to that \
character (a sword about 2/3 of character height at 24 tall is ~14-16 voxels). Write the scale down \
and size every part from it.
2. Silhouette first. Block out the whole model with big boxes and shapes in 1-2 flat colors before any \
detail. Check it from the front, side and top (screenshot views quad). If the silhouette doesn't read as \
the thing, details won't save it.
3. Big -> medium -> small. Large forms, then secondary forms (limbs, roof, wheels), then details (eyes, \
trim, rivets). Don't detail one area while others are still blocks.
4. Separate parts for separate pieces (body, head, arms, legs, wheels, doors, weapon): create_part each, \
so they can be posed, mirrored, reused and exported with pivots.
5. Use symmetry for symmetric things (set_symmetry) instead of drawing both halves by hand; break it \
deliberately at the end (a scar, an asymmetric strap) for life.
6. Odd vs even widths matter: a 1-voxel-wide nose or sword needs an odd-width face/blade to sit \
centered. Decide early.
7. Color: 3-5 shades per material (shadow, base, light, maybe highlight and deep shadow), not dozens of \
random colors. Light comes from above (and slightly front-left): tops lighter, undersides and crevices \
darker. Put exact colors in the palette with set_palette_colors before drawing; '#rrggbb' only picks \
the nearest existing palette color.
8. Readability beats realism at low resolution: exaggerate eyes, hands, weapon heads, wheels; drop \
anything smaller than a voxel.
9. Check your work after every stage: screenshot with views quad (front/side/top/3-4), with \
highlight_last_edit when an edit might have landed wrong. Read coordinates off the rulers instead of \
guessing. get_slice / get_projection give exact text maps.
10. Save often (save_file) and checkpoint before risky changes; a live window can reload from disk.
11. For anything repetitive (rows of windows, fence posts, bricks, scattered rocks) use run_script \
with a loop instead of hundreds of calls.
12. Finish with surface work: color variation on large flat areas, darkened edges/crevices, lighter top \
faces. Then materials (metal, glass, emission) and a render.";

const CHARACTER: &str = "\
CHARACTERS AND CREATURES
- Proportions (24 tall, realistic-ish): head 4-5, torso 7-8, legs 10-11, arms reach mid-thigh. \
Chibi/cute: head 8-10 of 24, short limbs. Heroic: head 1/8 of height, broad shoulders.
- Build as parts: torso (root), head, upper/lower arms or whole arms, legs, hands if large enough. Set \
pivots at joints (shoulder, hip, neck) with set_pivot, and sockets where things attach (hand_r, \
hand_l, head, back) with set_socket. Attach with transform_part attach_to so the skeleton follows.
- Model in a neutral pose (T or A pose, arms slightly out) so parts don't overlap; pose afterwards by \
turning parts about their pivots.
- Faces at small scale: eyes 1-2 voxels, often a darker color plus a 1-voxel highlight; a mouth line \
or nothing; brows sell expression. Inset eyes by 1 voxel for depth on bigger heads.
- Creatures: start from a reference animal's silhouette, then exaggerate the defining feature (tusks, \
horns, ears, shoulder mass). Goblins: big head, big ears, skinny limbs. Orcs/trolls: huge shoulders, \
small head, long arms. Yetis: round mass, fur as stepped edges, small face.
- Skin/fur/scale colors: pick base, shadow and highlight by hand per creature. Shadows usually shift \
cooler and highlights warmer; human skin shadows go warm red-brown, green skins go teal in shadow, \
white fur goes blue-lavender in shadow.
- Clothing and armor as color bands and 1-voxel raised edges; belts and straps break up big areas.";

const PROP: &str = "\
PROPS, WEAPONS AND VEHICLES
- Size props relative to a character's scale (see craft). Weapons: grip ~4-5 voxels at 24-tall scale; \
put a socket at the grip so it can attach to a hand socket.
- Readable shapes: exaggerate the business end (blade, head, muzzle). Edges in a lighter color sell \
sharpness/metal.
- Vehicles: body as one part, wheels/turrets/doors as separate parts with pivots at axles and hinges; \
duplicate wheels with modify_part duplicate rather than redrawing. Keep wheels a consistent odd \
diameter so axles center.
- Use draw_shape (cylinder, torus, capsule, rounded box) and draw_tube for cables and curves; tilt with \
rotation instead of stair-stepping by hand.
- Mechanical detail: panel lines (1-voxel darker grooves), bolts on corners, emissive lights, glass \
windows (set_material glass).";

const BUILDING: &str = "\
BUILDINGS AND ENVIRONMENTS
- Decide a grid unit: e.g. 1 voxel = 10 cm means doors ~20 tall, floors ~30 apart, a person 17-18 tall. \
Keep every element on that grid.
- Block out masses (walls, roof, chimney), then openings (doors, windows inset by 1), then trim \
(sills, beams, corner stones), then texture (brick/plank variation, moss, cracks).
- Walls are rarely flat single colors: vary bricks/planks slightly; darken bottom rows (dirt) and \
corners; lighten top edges.
- Roofs: pyramids/wedges or stepped layers; overhang by 1-2 voxels so they cast shadow.
- Repetition (windows, fence posts, shingles, tiles): run_script loops or save_stamp/place_stamp.
- Terrain: build height with layered shapes or perlin, then paint_faces top for grass/snow and \
vary colors. Scatter props (rocks, bushes) as stamps with variation.
- Interiors: leave one wall or the roof off (hide a part) for the camera, and add emissive lights.";

const COLOR: &str = "\
COLOR AND SHADING
- Start from a coherent palette (apply_palette_preset, or set_palette_colors with your own) instead \
of hex-matching random colors to whatever is in the palette.
- Per material, choose 3-5 shades by hand: deep shadow, shadow, base, light, highlight. Shift hue as \
well as brightness: shadows cooler/more saturated, highlights warmer/less saturated; darkening alone \
looks muddy. Metals have high contrast; stone/cloth low; skin and fur depend on the creature.
- Limit the palette: a whole character often uses 12-20 colors. Reuse shades across materials.
- Hand shading in voxels: tops one step lighter, undersides one step darker, crevices and contact \
points (armpits, under roofs, ground contact) darkest. Outline edges sparingly.
- Large flat areas look fake: add subtle 2-3 shade variation (vary_colors) at low contrast.
- Check colors in a screenshot and a render; the modeler view is flatter than the path tracer.";

const RENDERING: &str = "\
MATERIALS, LIGHTING, RENDERS
- Materials are per palette color (set_material): metal (roughness 0.2-0.5), glass (transparency \
0.7-0.95, ior 1.5), emit (emission 2-10 for lamps/eyes/screens), cloud for smoke/fog. Give glowing \
and glass parts their own palette colors.
- Lighting: a low sun (sun_pitch 30-50) at a three-quarter angle gives readable form shadows; raise \
sky intensity to soften. Night scenes: dim sky, strong emissive lights, maybe fog.
- Camera: three-quarter view (yaw ~35-45, pitch ~25-35) shows three faces; orthographic or isometric \
for game-asset previews.
- Do quick low-sample renders (samples 32-64) to iterate, then a final render (256+, denoise on).";

const GAME_EXPORT: &str = "\
GAME-READY MODELS
- One part per moving piece; set pivots where they rotate (set_pivot) and sockets where things attach \
(set_socket). These become glTF node origins and empty nodes engines can parent to.
- Keep a sensible origin: characters' root pivot at the feet center (pivot \"bottom\"), props at the \
grip or base.
- Hide internal voxels (edit_model hull) to cut triangles; keep parts under 256 per axis.
- Export with export format gltf (.glb). Test the pose by turning parts about their pivots first.";

/// Workflow prompts: (name, description, arguments [(name, description, required)], template).
/// `{arg}` in the template is replaced with the argument.
const PROMPTS: &[(&str, &str, &[(&str, &str, bool)], &str)] = &[
    (
        "build_character",
        "Build a voxel character or creature step by step",
        &[("subject", "What to build, e.g. \"a goblin archer\"", true), ("height", "Height in voxels (default 24)", false)],
        "Build {subject} in voxely as a voxel character about {height} voxels tall.\n\
First call get_guide for topics craft and character and follow them. Work in stages and check each \
with a screenshot (views quad): 1) new_project, set a palette (3-5 shades per material), plan parts \
and proportions and say the plan; 2) block out every part as flat-color shapes with symmetry on; \
3) refine silhouette; 4) details (face, hands, gear); 5) shading and surface variation; 6) pivots, \
sockets, attachments; 7) materials, a render, save_file. Save after every stage.",
    ),
    (
        "build_prop",
        "Build a voxel prop, weapon or vehicle",
        &[("subject", "What to build, e.g. \"a steampunk pistol\"", true), ("scale", "Size or the character scale it must fit (default: 24-voxel-tall character)", false)],
        "Build {subject} in voxely at a scale fitting {scale}.\n\
First call get_guide for topics craft and prop and follow them. Stages, each checked with a \
screenshot (views quad): palette, block-out, separate moving parts with pivots/sockets, detail, \
shading, materials, render, save_file.",
    ),
    (
        "build_environment",
        "Build a voxel building, room or scene",
        &[("subject", "What to build, e.g. \"a mossy stone watchtower\"", true), ("size", "Canvas size, e.g. 64x64x64 (default: choose one)", false)],
        "Build {subject} in voxely (size: {size}).\n\
First call get_guide for topics craft and building and follow them. Decide the grid unit and say it; \
block out masses, openings, trim, then texture and surface variation; use run_script or stamps for \
repetition; light it (get_guide rendering), render, save_file. Check each stage with a screenshot.",
    ),
    (
        "review_model",
        "Critique the current voxel model and fix its weakest points",
        &[],
        "Review the current voxely model like an art director. Take a screenshot with views quad and a \
quick render, call get_guide craft, then list the 3-5 biggest problems (silhouette, proportions, \
color count and shading, flat areas, floating or broken voxels, scale consistency) and fix them one \
at a time, checking after each. checkpoint first so you can roll back.",
    ),
];

pub fn tools() -> Vec<Value> {
    let names: Vec<&str> = TOPICS.iter().map(|t| t.0).collect();
    let summary: Vec<String> = TOPICS.iter().map(|t| format!("{}: {}", t.0, t.1)).collect();
    vec![tool(
        "get_guide",
        &format!("Voxel-art craft guidance to read before and while building. Topics: {}.", summary.join("; ")),
        json!({ "topic": { "type": "string", "enum": names, "description": "Default: craft" } }),
        &[],
    )]
}

pub fn dispatch(_ctx: &mut Ctx, name: &str, args: &Value) -> Option<Result<Vec<Value>>> {
    (name == "get_guide").then(|| {
        let topic = string(args, "topic").unwrap_or("craft");
        let (_, _, body) = TOPICS.iter().find(|t| t.0 == topic).ok_or_else(|| {
            anyhow!("no guide topic {topic:?}; topics: {}", TOPICS.iter().map(|t| t.0).collect::<Vec<_>>().join(", "))
        })?;
        Ok(vec![text(*body)])
    })
}

/// The MCP `prompts/list` result.
pub fn prompts() -> Value {
    let list: Vec<Value> = PROMPTS
        .iter()
        .map(|(name, description, args, _)| {
            let args: Vec<Value> = args.iter().map(|(n, d, r)| json!({ "name": n, "description": d, "required": r })).collect();
            json!({ "name": name, "description": description, "arguments": args })
        })
        .collect();
    json!({ "prompts": list })
}

/// The MCP `prompts/get` result.
pub fn prompt(params: &Value) -> Result<Value> {
    let name = string(params, "name").unwrap_or("");
    let (_, description, args, template) =
        PROMPTS.iter().find(|p| p.0 == name).ok_or_else(|| anyhow!("no prompt named {name:?}"))?;
    let mut body = template.to_string();
    for (arg, _, _) in args.iter() {
        let value = params.get("arguments").and_then(|a| a.get(*arg)).and_then(Value::as_str).filter(|s| !s.trim().is_empty());
        let value = value.unwrap_or(match *arg {
            "height" => "24",
            "scale" => "a 24-voxel-tall character",
            "size" => "your choice",
            _ => "something of your choice",
        });
        body = body.replace(&format!("{{{arg}}}"), value);
    }
    Ok(json!({ "description": description, "messages": [{ "role": "user", "content": { "type": "text", "text": body } }] }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_fill_arguments() {
        let p = prompt(&json!({ "name": "build_character", "arguments": { "subject": "a yeti" } })).unwrap();
        let text = p["messages"][0]["content"]["text"].as_str().unwrap();
        assert!(text.contains("a yeti") && text.contains("24 voxels") && !text.contains('{'));
        for (name, ..) in PROMPTS {
            assert!(prompt(&json!({ "name": name })).is_ok());
        }
    }
}
