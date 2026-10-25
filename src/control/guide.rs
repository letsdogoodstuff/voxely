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
    ("reference", "Building from a reference image or photo: measuring, profiles, comparing", REFERENCE),
    ("animal", "Animals and creatures with organic bodies", ANIMAL),
    ("vehicle", "Cars, ships, aircraft and other vehicles", VEHICLE),
    ("world", "Landscapes, terrain and large scenes", WORLD),
    ("scripting", "run_script patterns: util shapes, noise, Volume helpers", SCRIPTING),
];

const REFERENCE: &str = "\
BUILDING FROM A REFERENCE IMAGE (any subject)
- Measure, don't eyeball. analyze_image gives the image's blobs (each view on a sheet, with text captions \
flagged), each blob's width profile and its dominant colors. set_reference stores a view (front, back, left, \
right, top) with the project.
- Pick the size first: the model's height (or length for vehicles/animals, footprint for buildings) in voxels. \
Bigger references with fine detail need more voxels: faces, text and trim need ~40+ voxels of height to read.
- reference_profile turns the silhouette into voxel rows: for each z the filled x spans. Build straight from \
those numbers (a run_script loop over the rows, or shapes sized to the spans). Gaps between spans are real \
(between legs, under a car, between towers).
- Views: front = seen from -y (the subject's left is image right), back = from +y, left/right = from -x/+x, \
top = from +z. A sheet's SIDE view is one of left/right: check which way the subject faces in it.
- compare_reference projects the model exactly and reports silhouette IoU, row-by-row width errors and color \
mismatches, plus a diff image (green = reference only, magenta = model only). Fix the largest findings first \
and re-run; aim for 85%+ IoU on the main view before details, then compare again after shading for color.
- Depth isn't in a front view: take it from the side/top references, or from what the subject is (a torso is \
~0.6 of its width deep, a car ~0.45 of its length wide).
- paint_from_reference colors the visible voxels straight from a view (nearest palette color, speckle cleaned): \
run it on the finished shape, sides and back first, front last. Set the palette first (3-5 shades per \
material), because it can only pick colors you have. Pinned references stay put in world coordinates.
- Colors: map dominant colors to materials, then give each material 3-5 hand-picked shades around the sampled \
color (shadow, base, light). The photo's lighting is baked in, so sample from mid-tones, not highlights.";

const ANIMAL: &str = "\
ANIMALS
- Start from the body mass as one smooth shape (run_script: util.smoothUnion of ellipsoids for chest, belly \
and hips), then neck and head, then legs as tapered capsules (util.capsule with two radii), then tail, ears, \
muzzle. Smooth unions keep joins organic instead of boxy.
- Proportions come from the species: measure them from a side-view reference (reference_profile view left or \
right). Quadrupeds: legs set at the corners of the ribcage and hips, not the middle; the back is rarely flat.
- create_rig template quadruped gives named, jointed parts (body, head, tail, legs) ready for apply_motion.
- Fur and feathers: vary_colors with noise on the base coat, darker along the spine and lighter underneath \
(paint_faces bottom with a light shade), stepped edges for tufts. Patterns (stripes, spots) with a run_script \
using util.noise thresholds.
- Eyes are small (1-2 voxels) but placed exactly; a 1-voxel highlight makes them alive.";

const VEHICLE: &str = "\
VEHICLES
- Measure from a side view first (length, wheelbase, height, wheel size), then front/top for width. Keep wheels \
an odd diameter so axles center.
- Body as one part with the cabin/greenhouse inset; wheels, doors, turrets, propellers as separate parts with \
pivots at axles and hinges (duplicate wheels with modify_part duplicate).
- Panels: 1-voxel grooves in a darker shade for door seams and hood lines; chrome trim in a light metal color; \
glass as its own palette color with set_material glass; head/tail lights emissive.
- Use draw_shape rounded_box, capsule and wedge for body curves, rotation for raked windshields and spoilers, \
and util.box with round in scripts for soft car bodies.";

const WORLD: &str = "\
WORLDS AND LANDSCAPES
- Pick the canvas from the scene: a diorama 64-128 wide, a level 256+. Decide the grid unit (how tall a door \
or a person is) and keep everything on it.
- Terrain from a height function in run_script: layered util.noise at two or three scales, flattened where \
paths and buildings go; then paint_faces top for grass/snow, sides for rock/dirt layers, and vary_colors for \
natural variation.
- Water: a flat layer at sea level with a glass or blend material; rivers carved into the height function.
- Scatter trees, rocks and props as stamps (save_stamp once, place_stamp many times with rotation and color \
variation) or procedurally with util.hash so placements are repeatable.
- Big scenes stay manageable as parts: terrain, buildings, vegetation, props, each its own part or group.";

const SCRIPTING: &str = "\
SCRIPTING PATTERNS (run_script)
- One script per stage beats hundreds of calls. v = voxely.getVolume({part}); edit v; voxely.setVolume(v, {part}).
- Volume helpers: v.fillBox(a, b, c), v.fillSdf(shape, color, lo, hi), v.mirrorX(), v.bounds().
- util shapes are signed distance functions (negative inside): sphere, ellipsoid, box (with round), capsule \
(two radii for taper), cylinder; combine with union, smoothUnion(k, ...) for organic joins, subtract, intersect, \
mirrorX(shape, cx) for symmetry, displace(shape, amp, scale) for ragged edges (cloth hems, rock, terrain).
- color may be a function of (x, y, z, d): pick shades from util.noise for variation, or from d for rims.
- util.hash(x, y, z, seed) and util.noise(x, y, z, scale, seed) are repeatable, so a rerun gives the same model.
- Shapes that cross the mirror plane: draw them whole (box from x 18 to 30), don't mirror a half-box; a \
half-box's rounded edge leaves a 1-voxel groove down the middle.
- Example: v.fillSdf(util.smoothUnion(2, util.ellipsoid([20,16,20],[8,6,10]), util.sphere([20,16,33],6)), 14)";

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
- create_rig (template humanoid or quadruped) gives named, jointed parts to reshape, ready for animation.
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

/// Workflow prompts: (name, description, template). Each takes one free-text
/// `request` argument (clients split typed text unpredictably across several
/// arguments, so everything given is joined into it); `{request}` in the
/// template is replaced with it.
const PROMPTS: &[(&str, &str, &str)] = &[
    (
        "build_from_reference",
        "Build anything (character, creature, vehicle, building, scene) from a reference image and measure it against the image",
        "Build this in voxely: {request}\n\
Work from the reference image(s) and measure, don't eyeball. Call get_guide for topics reference and craft, \
plus the subject's topic (character, animal, vehicle, prop, building or world). Then:\n\
1) analyze_image on the reference; say which blob is which view; set_reference for each view (front, back, \
left/right, top) with the project's target size.\n\
2) Write a short spec before drawing: overall size in voxels; a proportions table read from reference_profile \
(rows/heights of each major section and their widths); the parts list; colors from analyze_image's dominant \
colors mapped to materials (3-5 shades each, set with set_palette_colors); anything asymmetric and which side it's on \
(front view: the subject's left is image right).\n\
3) new_project, palette, then block out every part from the profile numbers (run_script with util shapes is \
fastest), symmetry on for symmetric subjects.\n\
4) compare_reference for every referenced view; fix the biggest findings; repeat until the silhouette IoU stops \
improving (aim for 85%+ before details).\n\
5) Details, then color: paint_from_reference for each view (sides and back first, the front last) projects \
the reference's colors onto the visible voxels with your palette; then fix key features by hand (eyes, \
logos), shade what no view covers (vary_colors, shade_edges), and check with screenshot views quad.\n\
6) compare_reference again (shape and color) and fix what regressed. For things that move, set pivots and \
sockets and attach parts.\n\
7) Materials, render views turnaround, save_file. Report the final compare_reference scores per view. Save after \
every stage.",
    ),
    (
        "build_character",
        "Build a voxel character or creature step by step",
        "Build this in voxely as a voxel character: {request}\n\
Call get_guide for topics craft and character (or animal) first and follow them. If there's a reference image, \
follow the build_from_reference workflow instead (analyze_image, set_reference, reference_profile, \
compare_reference). For a humanoid or quadruped, create_rig gives named, jointed parts to reshape. Stages, \
each checked with screenshot views quad: 1) new_project, palette (3-5 shades per material), plan parts and \
proportions and say the plan; 2) block out every part as flat-color shapes with symmetry on; 3) refine the \
silhouette; 4) details (face, hands, gear); 5) shading and surface variation; 6) pivots, sockets, attachments; \
7) materials, a render, save_file. Save after every stage.",
    ),
    (
        "build_prop",
        "Build a voxel prop, weapon or vehicle",
        "Build this in voxely: {request}\n\
Call get_guide for topics craft and prop (or vehicle) first and follow them. If there's a reference image, \
follow the build_from_reference workflow (analyze_image, set_reference, compare_reference). Stages, each checked \
with screenshot views quad: palette, block-out, separate moving parts with pivots/sockets, detail, shading, \
materials, render, save_file.",
    ),
    (
        "build_environment",
        "Build a voxel building, room, landscape or scene",
        "Build this in voxely: {request}\n\
Call get_guide for topics craft and building (or world) first and follow them. Decide the grid unit and say \
it; block out masses, openings, trim, then texture and surface variation; use run_script (util noise and \
shapes) or stamps for repetition and terrain; light it (get_guide rendering), render, save_file. Check each \
stage with a screenshot. With a reference image, use analyze_image / set_reference / compare_reference.",
    ),
    (
        "review_model",
        "Critique the current voxel model and fix its weakest points",
        "Review the current voxely model like an art director{request}. Take a screenshot with views quad and a \
quick render, call get_guide craft, and if references are set run compare_reference for each. List the 3-5 \
biggest problems (silhouette, proportions, color count and shading, flat areas, floating or broken voxels, \
scale consistency) and fix them one at a time, checking after each. checkpoint first so you can roll back.",
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
        .map(|(name, description, _)| {
            let arg = json!({ "name": "request", "description": "What to build, in your own words (mention any attached reference image)", "required": false });
            json!({ "name": name, "description": description, "arguments": [arg] })
        })
        .collect();
    json!({ "prompts": list })
}

/// The MCP `prompts/get` result. Every argument given is joined into the
/// request, in case the client split the text across arguments.
pub fn prompt(params: &Value) -> Result<Value> {
    let name = string(params, "name").unwrap_or("");
    let (_, description, template) = PROMPTS.iter().find(|p| p.0 == name).ok_or_else(|| anyhow!("no prompt named {name:?}"))?;
    let given: Vec<&str> = params
        .get("arguments")
        .and_then(Value::as_object)
        .map(|a| {
            let mut v: Vec<(&String, &str)> = a.iter().filter_map(|(k, v)| v.as_str().map(|s| (k, s.trim()))).filter(|(_, s)| !s.is_empty()).collect();
            // `request` first, then any others in a stable order.
            v.sort_by_key(|(k, _)| (k.as_str() != "request", (*k).clone()));
            v.into_iter().map(|(_, s)| s).collect()
        })
        .unwrap_or_default();
    let request = given.join(" ");
    let body = if name == "review_model" {
        template.replace("{request}", &if request.is_empty() { String::new() } else { format!(", focusing on: {request}") })
    } else {
        template.replace("{request}", if request.is_empty() { "something of your choice" } else { &request })
    };
    Ok(json!({ "description": description, "messages": [{ "role": "user", "content": { "type": "text", "text": body } }] }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_take_the_whole_request() {
        let p = prompt(&json!({ "name": "build_character", "arguments": { "request": "a yeti with a lantern" } })).unwrap();
        let text = p["messages"][0]["content"]["text"].as_str().unwrap();
        assert!(text.contains("a yeti with a lantern") && !text.contains("{request}"));
        // Text split across unknown arguments is joined back together.
        let p = prompt(&json!({ "name": "build_prop", "arguments": { "request": "a red", "extra": "sports car" } })).unwrap();
        assert!(p["messages"][0]["content"]["text"].as_str().unwrap().contains("a red sports car"));
        for (name, ..) in PROMPTS {
            assert!(!prompt(&json!({ "name": name })).unwrap()["messages"][0]["content"]["text"].as_str().unwrap().contains('{'));
        }
    }
}
