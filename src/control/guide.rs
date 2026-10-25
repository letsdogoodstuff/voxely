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
    ("image", "Any picture or photo to voxels: whole scenes, single photos, which workflow to use", IMAGE),
    ("reference", "Building from a turnaround sheet or straight-on views: measuring, profiles, comparing", REFERENCE),
    ("animal", "Animals and creatures with organic bodies", ANIMAL),
    ("vehicle", "Cars, ships, aircraft and other vehicles", VEHICLE),
    ("world", "Landscapes, terrain and large scenes", WORLD),
    ("scripting", "run_script patterns: util shapes, noise, Volume helpers", SCRIPTING),
];

const IMAGE: &str = "\
PICTURES TO VOXELS (any picture a user sends)
- The picture is the design reference; you build the model as a voxel artist would: designed parts, \
flat colors, hand-placed details (get_guide craft). Never project the picture's colors onto voxels \
or build straight from a depth map: that copies the picture's lighting as speckle and lumps.
- First decide what the user wants: the WHOLE SCENE (ground, buildings, props, characters, background) \
or ONE SUBJECT. A full scene with no instruction means the whole scene: build every object.
- Scale: if the picture is voxel art, match its block size, one picture block = one voxel. Measure it: \
analyze_image reports the block size per subject (pixels per block), so a subject that is 880 px tall \
at 7 px per block is about 126 voxels tall. Otherwise pick the house scale (characters 32-48 tall) and \
size everything else from the main subject.
- Read before you build: write a short spec of every object: its parts, proportions (as fractions of \
its height, read off the picture), each material's flat color (hex of its clean mid-tone) and the \
details that make it recognizable (a skull pauldron, runes on a blade, a gem on a crown). For a \
turnaround sheet, analyze_image / set_reference / reference_profile give exact silhouette rows to \
size parts from (get_guide reference).
- Then build each object as its own part, big forms first, then the designed details, checking with \
screenshot views quad. For scenes, read the layout off the picture (where each object sits and how big \
it is relative to the main subject) and build every object by hand at those places.
- Hidden sides: invent them the way the subject is built (mirror symmetric things, continue materials \
around) and say what you invented.
- Judge by eye: render it next to the picture and fix what doesn't look like it. Scores help with \
proportions, but a model can score well and still look wrong.";

const REFERENCE: &str = "\
BUILDING FROM A TURNAROUND SHEET OR STRAIGHT-ON VIEWS
- analyze_image finds the views (with captions flagged), set_reference stores each view (front, back, \
left, right, top). Pick the size first: for voxel-art sheets match the sheet's block size (one sheet \
block = one voxel); otherwise the house scale.
- reference_profile turns a view's silhouette into voxel rows: the filled spans per height. Use them to \
size each part (where the shoulders are, how wide the cape flares, where the boots start); then design \
the parts as clean shapes in flat colors (get_guide craft). Gaps between spans are real.
- Views: front = seen from -y (the subject's left is image right), back = from +y, left/right = from \
-x/+x, top = from +z. A sheet's SIDE view is left or right: check which way the subject faces.
- compare_reference checks proportions (silhouette IoU, row-by-row width errors) with a diff image; fix \
big proportion errors, but don't chase the number with lumpy edges: clean shapes that read beat a \
traced outline.
- Colors: pick each material's flat mid-tone from the sheet and design the details by hand. \
paint_from_reference copies the sheet's shading as speckle; skip it unless the user wants that.";

const ANIMAL: &str = "\
ANIMALS
- Start from the body mass as chunky stacked boxes (chest, belly, hips), stepped at the edges so it reads \
rounded, then neck and head, then legs as tapered capsules (util.capsule with two radii), then tail, ears, \
muzzle. Smooth unions keep joins organic instead of boxy.
- Proportions come from the species: measure them from a side-view reference (reference_profile view left or \
right). Quadrupeds: legs set at the corners of the ribcage and hips, not the middle; the back is rarely flat.
- create_rig template quadruped gives named, jointed parts (body, head, tail, legs) ready for apply_motion.
- Fur and feathers: a flat base coat with designed markings (a lighter belly or muzzle, darker stripes or \
spots placed on purpose) and stepped edges for tufts.
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
paths and buildings go; then paint_faces top for grass/snow and sides for rock/dirt layers, in flat colors.
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
- color may be a function of (x, y, z, d), e.g. a flat accent color for rims (d near 0); keep colors flat.
- util.hash(x, y, z, seed) and util.noise(x, y, z, scale, seed) are repeatable, so a rerun gives the same model.
- Shapes that cross the mirror plane: draw them whole (box from x 18 to 30), don't mirror a half-box; a \
half-box's rounded edge leaves a 1-voxel groove down the middle.
- Example: v.fillSdf(util.smoothUnion(2, util.ellipsoid([20,16,20],[8,6,10]), util.sphere([20,16,33],6)), 14)";

const CRAFT: &str = "\
VOXEL CRAFT RULES (the house style: clean, designed voxel art like MagicaVoxel showcase pieces)
1. Design it, don't sculpt or trace it. Decide every part, color and detail on purpose, the way a \
voxel artist would: chunky boxy forms, clean straight runs, stepped curves only where the shape needs \
one, and small details placed voxel by voxel (eyes, gems, buckles, stitching, rivets, trim lines).
2. Pick a scale first and stick to it. Designing from your head: a character 32-48 voxels tall (a chibi \
head is about 1/3 of that), props sized from the character. From a voxel-art picture: match the \
picture's block size, one picture block = one voxel (measure it, see get_guide image). Write the scale \
down and size every part from it.
3. Flat colors. Each material gets ONE base color, plus at most one deliberate accent shade (a seam, \
trim, the underside of a brim, a pupil). Never noise, speckle or random variation: the path tracer \
does the lighting and shadows, so painted shading only makes models muddy. A whole character is \
usually 8-16 colors. Put exact colors in the palette with set_palette_colors before drawing.
4. Silhouette first: block out the whole model in its flat colors, check front, side and top \
(screenshot views quad); if it doesn't read as the thing, details won't save it.
5. Big -> medium -> small: large forms, then limbs/roof/wheels, then designed details. Don't detail \
one area while others are still blocks.
6. Separate parts for separate pieces (body, head, arms, legs, wheels, doors, weapon) with create_part, \
so they can be posed, reused and exported with pivots.
7. Symmetry for symmetric things (set_symmetry), broken deliberately at the end (a scar, one pauldron).
8. Odd vs even widths: a 1-voxel nose or sword needs an odd-width face or blade to sit centered.
9. Readability beats realism: exaggerate eyes, hands, weapon heads, wheels; drop anything smaller than \
a voxel; give the face real care (eyes with a pupil and a highlight, brows, a mouth line).
10. Check after every stage with a screenshot (views quad) and read coordinates off the rulers. get_slice \
/ get_projection give exact text maps. Save often (save_file).
11. Use fill_box / draw_shape / set_voxels for designed work and run_script for exact repetition \
(rows of windows, bricks, fence posts) or to place a designed layout from arrays. Don't reach for noise, \
smooth blobs or automatic shading tools unless the user asks for that look.
12. Finish with materials (metal, glass, emit for eyes, gems, lamps) and a render.";

const CHARACTER: &str = "\
CHARACTERS AND CREATURES
- Proportions: chibi/stylized (the house default): head about 1/3 of the height, chunky body, short \
limbs; realistic: head 1/7-1/8. Designing from your head, 32-48 voxels tall.
- Faces get the most care: eyes 2-3 voxels with a colored iris, a dark pupil and a 1-voxel highlight; \
brows; a 1-2 voxel mouth or nose hint. A face reads at a glance or the model fails.
- Build as parts: torso (root), head, arms, legs, hands if big enough, gear. Pivots at joints \
(set_pivot), sockets where things attach (set_socket), attach with transform_part attach_to. \
create_rig gives a jointed humanoid or quadruped to reshape.
- Clothing and armor as flat color bands with designed details: a belt with a buckle, straps, a \
1-voxel raised trim, a gem, a skull emblem, stitching lines. Hair as chunky stepped locks in one or \
two colors.
- Creatures: exaggerate the defining feature (tusks, horns, ears, shoulder mass). Goblins: big head, \
big ears, skinny limbs. Orcs: huge shoulders, small head. Skin colors are flat; any creature color works.";

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
COLOR
- Flat, designed color is the house style: one base color per material plus at most one accent shade \
used on purpose (a seam line, trim, the inside of a hood, a pupil). Let the renderer shade the forms.
- Pick a coherent palette first (set_palette_colors with your own picks, or apply_palette_preset) and \
keep a character to about 8-16 colors.
- Contrast makes it read: dark leather next to pale bone, gold trim on gray plate, bright eyes in a \
dark face. Separate neighboring materials by value, not by noise.
- Taking colors from a picture: pick each material's clean mid-tone (not its highlights or shadows, \
which are the picture's lighting) and use it flat.
- Do NOT use vary_colors, shade_edges or paint_from_reference unless the user asks \
for a textured or painted look: they add speckle and baked shading that muddy the model.";

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
        "build_from_image",
        "Turn any picture into voxels: a whole scene, a photo of something, or a character sheet",
        "Turn this picture into voxels in voxely: {request}\n\
Call get_guide for topics image and craft first and follow the house style: designed parts, flat colors, \
hand-placed details. Decide (and say) whether the user wants the whole scene or one subject; a full scene \
with no instruction means the whole scene.\n\
1) Measure: analyze_image. If it is voxel art, use its block \
size as the scale (one picture block = one voxel).\n\
2) Spec: list every object with its parts, proportions, flat colors (hex) and recognizable details. \
new_project and set_palette_colors.\n\
3) Build each object as its own part: block out in flat colors, check with screenshot views quad, then \
design the details (face, trim, emblems). Scenes: place objects where the picture puts them.\n\
4) Render next to the picture and fix what doesn't look like it. Materials (emit for eyes, lava, gems). \
save_file after every stage.",
    ),
    (
        "build_from_reference",
        "Build anything from a turnaround sheet or straight-on views",
        "Build this in voxely: {request}\n\
Call get_guide for topics reference and craft (plus the subject's topic) and follow the house style. \
analyze_image and set_reference for each view; size the model (voxel-art sheets: match the block size); \
write a spec (parts, proportions from reference_profile, flat colors, details); block out the parts in \
flat colors; check proportions with compare_reference; design the details by hand; render turnaround; \
save_file after every stage.",
    ),
    (
        "build_character",
        "Build a voxel character or creature step by step",
        "Build this in voxely as a voxel character: {request}\n\
Call get_guide for topics craft and character (or animal) and follow the house style: chunky designed \
forms, flat colors, a face with real care. If there's a picture, follow build_from_image. Stages, each \
checked with screenshot views quad: 1) new_project, palette (one flat color per material plus accents), \
plan parts and proportions and say the plan; 2) block out every part in flat colors with symmetry on; \
3) refine the silhouette; 4) design the details (face, hands, gear, trim); 5) pivots, sockets, \
attachments; 6) materials, a render, save_file. Save after every stage.",
    ),
    (
        "build_prop",
        "Build a voxel prop, weapon or vehicle",
        "Build this in voxely: {request}\n\
Call get_guide for topics craft and prop (or vehicle) and follow the house style. Stages, each checked \
with screenshot views quad: flat palette, block-out, separate moving parts with pivots/sockets, designed \
details, materials, render, save_file.",
    ),
    (
        "build_environment",
        "Build a voxel building, room, landscape or scene",
        "Build this in voxely: {request}\n\
Call get_guide for topics craft and building (or world) and follow the house style. Decide the grid \
unit and say it; block out masses, openings, trim in flat colors, then designed details; run_script or \
stamps for exact repetition; light it (get_guide rendering), render, save_file. Check each stage with \
a screenshot. With a picture, follow build_from_image.",
    ),
    (
        "review_model",
        "Critique the current voxel model and fix its weakest points",
        "Review the current voxely model like an art director{request}. Take a screenshot with views quad and a \
quick render, read get_guide craft. List the 3-5 biggest problems (silhouette, proportions, muddy or \
noisy color, a face that doesn't read, missing details, floating voxels) and fix them one at a time, \
checking after each. checkpoint first so you can roll back.",
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
