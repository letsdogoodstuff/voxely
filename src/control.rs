//! Remote control: the tools an agent can call (over MCP, see `mcp.rs`)
//! and their implementation against a scene, editor and renderer. The same
//! code runs inside the window (live, so a person can watch) and headless.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use glam::{IVec3, Vec3};
use serde_json::{Value, json};

use crate::app::{self, EditState};
use crate::edit::SelMode;
use crate::gpu::{Camera, Renderer};
use crate::ops;
use crate::scene::{BgMode, MaterialKind, Projection, Scene, Volume, srgb_to_linear};
use crate::ui::{Op, TransformKind};

// Tool families kept in their own files. Each has `tools()` (definitions),
// `PART_TOOLS` (tools that take a `part`), `dispatch` (None if the name
// isn't one of its tools) and a `State` kept in `EditState::tools`.
mod build;
pub mod guide;
mod inspect;
mod surface;
mod view;

/// Per-session state for the tool families, plus a record of the last edit.
#[derive(Default)]
pub struct ToolState {
    pub last_edit: Option<LastEdit>,
    pub build: build::State,
    pub inspect: inspect::State,
    pub surface: surface::State,
    pub view: view::State,
}

/// What the most recent committed edit changed, in the edited part's
/// coordinates.
#[derive(Clone, Debug, Default)]
pub struct LastEdit {
    /// The part that was edited (None in World mode).
    pub part: Option<usize>,
    /// The canvas was resized, so cells can't be compared one to one.
    pub resized: bool,
    pub added: usize,
    pub removed: usize,
    pub recolored: usize,
    /// Bounds of the changed cells, inclusive.
    pub bounds: Option<(IVec3, IVec3)>,
    /// Every changed cell (empty when resized).
    pub cells: Vec<IVec3>,
}

impl LastEdit {
    fn diff(part: Option<usize>, old: &Volume, new: &Volume) -> Self {
        let mut e = LastEdit { part, resized: old.dims != new.dims, ..Default::default() };
        if e.resized {
            return e;
        }
        for (i, (&a, &b)) in old.cells.iter().zip(&new.cells).enumerate() {
            if a == b {
                continue;
            }
            match (a, b) {
                (0, _) => e.added += 1,
                (_, 0) => e.removed += 1,
                _ => e.recolored += 1,
            }
            let p = old.position(i);
            e.bounds = Some(e.bounds.map_or((p, p), |(lo, hi)| (lo.min(p), hi.max(p))));
            e.cells.push(p);
        }
        e
    }
}

/// Everything a tool call can touch.
pub struct Ctx<'a> {
    pub r: &'a mut Renderer,
    pub scene: &'a mut Scene,
    pub ed: &'a mut EditState,
    /// True inside the window, false when running headless.
    pub live: bool,
}

/// Guidance sent to the client when it connects.
pub const INSTRUCTIONS: &str = "\
voxely is a MagicaVoxel-style voxel editor and path tracer. Coordinates are integer voxel cells \
[x, y, z] with z up, from 0 to size-1 on each axis. Colors are palette indices 1-255 (0 = empty) or \
\"#rrggbb\" strings, which pick the nearest palette color (use set_palette_colors to define exact colors). \
Projects live in a workspace folder and show in the window's Project list; use new_project to start \
a named one, list_projects / open_project to flip between them, and save_file to save. \
Typical flow: new_project -> draw with fill_box / draw_shape / draw_line / set_voxels -> look with \
screenshot (fast) -> set_material and set_render_settings -> render (path traced) -> save_file. \
For loops, procedural builds or anything needing many calls, use run_script: one JavaScript program \
that calls every tool as voxely.<camelCaseName>({...}) and can edit a part's whole grid as a Uint8Array. \
Every edit is undoable. If the voxely window is open, edits appear in it live. \
Before building something, read get_guide (topic craft, plus character / prop / building as fits): \
pick a scale, block out the silhouette first, use a few shades per material, and check each stage \
with a screenshot.

Scenes are a tree of parts (models) and groups, like MagicaVoxel's World editor. Voxel tools work on one part at a time, in that part's own coordinates (0..size-1): the part being edited (see get_scene_info), or the one named by a tool's `part` argument. Build separate pieces (body, head, arms, wheels, props) as separate parts with create_part, place them with transform_part (move, 90° rotate, flip) and group_parts, and inspect the tree with list_parts. Give parts pivots (set_pivot: a shoulder, a hinge) \
and named sockets (set_socket: hand, muzzle, seat), then attach parts with transform_part attach_to (\"arm:hand\") so they follow, like a skeleton; \
both carry into glTF exports for game engines. Part positions are world voxel coordinates of the part's minimum corner; a new project's first part spans [0, size).";

fn tool(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": { "type": "object", "properties": properties, "required": required },
    })
}

const XYZ: &str = "[x, y, z] integers";
const COLOR: &str = "Palette index 1-255, or \"#rrggbb\" (nearest palette color)";

fn xyz(desc: &str) -> Value {
    json!({ "type": "array", "items": { "type": "integer" }, "minItems": 3, "maxItems": 3, "description": desc })
}

fn color_schema() -> Value {
    json!({ "type": ["integer", "string"], "description": COLOR })
}

/// Every tool's definition (see `tools`).
fn base_tools() -> Value {
    let axis = json!({ "type": "string", "enum": ["x", "y", "z"] });
    json!([
        tool("get_scene_info", "Canvas size, voxel count and bounds, colors in use, file, selection, camera and mode.", json!({}), &[]),
        tool("new_project", "Create a named project in the workspace (it appears in the window's Project list) and make it current. Empty unless `fill` is given.", json!({
            "name": { "type": "string", "description": "Letters, digits, spaces, - and _" },
            "size": xyz("Canvas size, default [40, 40, 40]; up to 1024 per axis"),
            "fill": color_schema(),
            "overwrite": { "type": "boolean", "description": "Replace an existing project with this name" },
        }), &["name"]),
        tool("list_projects", "The workspace's projects (newest first) and MagicaVoxel's sample models.", json!({}), &[]),
        tool("open_project", "Open a project or sample by name.", json!({ "name": { "type": "string" } }), &["name"]),
        tool("new_scene", "Start a new unnamed project (saved as project, project-2, ... in the workspace). Empty unless `fill` is given.", json!({
            "size": xyz("Canvas size, default [40, 40, 40]; up to 1024 per axis"),
            "fill": color_schema(),
        }), &[]),
        tool("open_file", "Open a MagicaVoxel .vox file, keeping its models and groups as parts.", json!({ "path": { "type": "string" } }), &["path"]),
        tool("save_file", "Save the project as .vox. Without `path`, saves the current project in the workspace.", json!({ "path": { "type": "string" } }), &[]),
        tool("set_voxels", "Set individual voxels. Color 0 erases.", json!({
            "voxels": { "type": "array", "description": "[[x, y, z, color], ...]; color is an index or \"#rrggbb\"", "items": { "type": "array", "items": { "type": ["integer", "string"] } } },
        }), &["voxels"]),
        tool("fill_box", "Fill (or with color 0, clear) the box between two corners, inclusive.", json!({
            "from": xyz(XYZ), "to": xyz(XYZ), "color": color_schema(),
            "hollow": { "type": "boolean", "description": "Only the outer shell" },
        }), &["from", "to", "color"]),
        tool("draw_shape", "Draw a solid shape inscribed in the box between two corners (inclusive), optionally rotated about the box center.", json!({
            "shape": { "type": "string", "enum": build::SHAPES,
                "description": "frustum: a cylinder narrowing to top_scale; torus: a ring around axis, tube as thick as the box along axis (or `radius`); capsule: a cylinder with round ends along axis; rounded_box: a box with corners rounded by `radius`; wedge: a ramp rising toward +axis (default x), full height along `up` (default z) at the +axis end" },
            "from": xyz(XYZ), "to": xyz(XYZ), "color": color_schema(),
            "axis": { "type": "string", "enum": ["x", "y", "z"], "description": "The shape's long axis (default z): cylinders, cones, frustums and pyramids point toward +axis, a torus's hole runs along it" },
            "top_scale": { "type": "number", "description": "cylinder/cone/frustum/pyramid: size of the +axis end relative to the base (cylinder 1, cone and pyramid 0, frustum 0.5); >1 widens" },
            "radius": { "type": "number", "description": "rounded_box: corner radius in voxels; torus: tube radius" },
            "up": { "type": "string", "enum": ["x", "y", "z"], "description": "wedge: the axis it's tall along (default z)" },
            "rotation": { "type": "array", "items": { "type": "number" }, "description": "[x, y, z] degrees about the box center, applied x then y then z, e.g. a tilted cylinder or a leaning cone; the shape may extend past from/to" },
            "hollow": { "type": "boolean", "description": "Only the outer shell" },
        }), &["shape", "from", "to", "color"]),
        tool("draw_line", "Draw a straight line of voxels between two points.", json!({
            "from": xyz(XYZ), "to": xyz(XYZ), "color": color_schema(),
            "thickness": { "type": "integer", "description": "Diameter in voxels (default 1)" },
        }), &["from", "to", "color"]),
        tool("get_voxels", "List solid voxels, optionally within a box.", json!({
            "from": xyz(XYZ), "to": xyz(XYZ), "limit": { "type": "integer", "description": "Max voxels returned (default 5000)" },
        }), &[]),
        tool("get_slice", "A text map of one layer: color indices, '.' for empty. For axis z, rows run from high y (top) to low y.", json!({
            "axis": axis.clone(), "index": { "type": "integer" },
        }), &["index"]),
        tool("edit_model", "MagicaVoxel's whole-model tools. fill/clear apply to the selection if there is one.", json!({
            "op": { "type": "string", "enum": ["full", "fill", "clear", "reverse", "double", "half", "flip", "rotate", "fit", "loop", "mirror", "diagonal", "hull", "flood", "dilate", "erode", "maze", "perlin"],
                "description": "full: fill empty space; fill: recolor voxels; reverse: swap empty/solid; double/half: scale 2x/0.5x; flip/rotate(90 deg)/mirror(make symmetric)/diagonal/loop(shift 1 with wrap) need `axis`; fit: shrink canvas to model; hull: remove hidden voxels; flood: fill enclosed space; dilate/erode: grow/shrink 1; maze/perlin: generate" },
            "axis": axis.clone(),
            "direction": { "type": "integer", "description": "loop: -1 or 1" },
            "color": color_schema(),
        }), &["op"]),
        tool("transform_model", "Scale, shear, rotate by any angle, or repeat the model. The canvas grows to fit.", json!({
            "kind": { "type": "string", "enum": ["scale", "shear", "rotate", "repeat"] },
            "args": { "type": "string", "description": "Axes then values, like MagicaVoxel: scale \"xyz 2\", shear \"z 0.2 0.2\", rotate \"z 45\", repeat \"xy 2\"" },
        }), &["kind", "args"]),
        tool("resize_canvas", "Change the canvas size, keeping the model centered in x/y and on the floor.", json!({ "size": xyz("New size") }), &["size"]),
        tool("list_shaders", "MagicaVoxel voxel shaders available to run_shader, with their arguments.", json!({}), &[]),
        tool("run_shader", "Run a MagicaVoxel voxel shader (GLSL `float map(vec3 v)` returning a color index) over the model, or the selection.", json!({
            "name": { "type": "string", "description": "A shader from list_shaders" },
            "source": { "type": "string", "description": "Or inline GLSL defining float map(vec3 v); built-ins: i_volume_size, i_color_index, voxel(v), palette(i), color_sel(k)" },
            "args": { "type": ["array", "object"], "items": { "type": "number" }, "description": "Argument values: an array in order, or an object by arg var name" },
            "color": color_schema(),
        }), &[]),
        tool("get_palette", "Palette colors as hex with their materials. Default: only colors in use.", json!({ "all": { "type": "boolean" } }), &[]),
        tool("set_palette_colors", "Set exact palette colors.", json!({
            "colors": { "type": "array", "items": { "type": "object", "properties": { "index": { "type": "integer" }, "color": { "type": "string" } } }, "description": "[{\"index\": 1, \"color\": \"#ff8800\"}, ...]" },
        }), &["colors"]),
        tool("load_palette", "Load a palette image (256x1 PNG), by path or by MagicaVoxel palette page number 0-3.", json!({
            "path": { "type": "string" }, "page": { "type": "integer" },
        }), &[]),
        tool("set_material", "Set the render material of a palette color.", json!({
            "color": color_schema(),
            "type": { "type": "string", "enum": ["diffuse", "metal", "glass", "emit", "blend", "cloud"] },
            "roughness": { "type": "number" }, "metallic": { "type": "number" }, "ior": { "type": "number", "description": "Index of refraction, e.g. 1.5" },
            "transparency": { "type": "number" }, "emission": { "type": "number", "description": "Emit strength, e.g. 1-20" },
            "density": { "type": "number", "description": "Cloud density" },
        }), &["color"]),
        tool("select", "Change the selection, which limits fill/clear/run_shader/move_selection.", json!({
            "mode": { "type": "string", "enum": ["all", "none", "invert", "box", "color", "region"],
                "description": "box: solid voxels between from/to; color: every voxel of `color`; region: voxels of the same color connected to `at`" },
            "from": xyz(XYZ), "to": xyz(XYZ), "at": xyz(XYZ), "color": color_schema(),
            "combine": { "type": "string", "enum": ["replace", "add", "remove"] },
        }), &["mode"]),
        tool("move_selection", "Move the selected voxels (or the whole model if none) by an offset.", json!({ "offset": xyz("[dx, dy, dz]") }), &["offset"]),
        tool("undo", "Undo edits.", json!({ "steps": { "type": "integer" } }), &[]),
        tool("redo", "Redo edits.", json!({ "steps": { "type": "integer" } }), &[]),
        tool("set_camera", "Aim the camera (used by screenshot, render and the window). Angles in degrees.", json!({
            "frame": { "type": "boolean", "description": "Reset to a three-quarter view of the whole canvas first" },
            "yaw": { "type": "number", "description": "Orbit angle around z" },
            "pitch": { "type": "number", "description": "Elevation, -89 to 89" },
            "distance": { "type": "number" },
            "target": { "type": "array", "items": { "type": "number" } },
            "projection": { "type": "string", "enum": ["perspective", "orthographic", "isometric", "stereographic", "panorama"] },
            "fov": { "type": "number" },
        }), &[]),
        tool("set_render_settings", "Lighting, sky, film and display settings for render. Colors are \"#rrggbb\".", json!({
            "sun_pitch": { "type": "number" }, "sun_yaw": { "type": "number" }, "sun_intensity": { "type": "number" },
            "sun_area": { "type": "number" }, "sun_color": { "type": "string" },
            "sky": { "type": "string", "enum": ["uniform", "atmosphere", "image"] },
            "sky_intensity": { "type": "number" }, "sky_color": { "type": "string" },
            "exposure": { "type": "number" }, "gamma": { "type": "number" }, "aces": { "type": "boolean" },
            "bloom": { "type": "boolean" }, "denoise": { "type": "boolean" },
            "fog_density": { "type": "number" }, "fog_color": { "type": "string" },
            "ground": { "type": "boolean" }, "ground_color": { "type": "string" },
            "background": { "type": "string", "enum": ["sky", "color", "transparent"] }, "background_color": { "type": "string" },
            "edges": { "type": "boolean" }, "grid": { "type": "boolean" },
            "aperture": { "type": "number", "description": "Depth of field, 0 = off" },
        }), &[]),
        tool("screenshot", "Fast image of the model in the editor's modeler view from the current camera.", json!({
            "width": { "type": "integer" }, "height": { "type": "integer" },
        }), &[]),
        tool("render", "Path-traced image from the current camera and render settings.", json!({
            "width": { "type": "integer" }, "height": { "type": "integer" },
            "samples": { "type": "integer", "description": "Samples per pixel (default 128; more is smoother and slower)" },
            "save_path": { "type": "string", "description": "Also save the PNG here" },
        }), &[]),
        tool("export", "Export the model: gltf (.glb, best for game engines), or MagicaVoxel's formats: obj/ply/mc/cube meshes, xraw/slab/qb/vox volumes, iso/2d/slice sprite images, point clouds.", json!({
            "format": { "type": "string", "enum": ["gltf", "obj", "ply", "mc", "xraw", "slab", "qb", "vox", "iso", "2d", "slice", "point", "cube"],
                "description": "gltf: a .glb for game engines (Godot, Unity, Unreal, Three.js, Bevy) with metal/roughness, emission and glass materials; use a .gltf path for JSON" },
            "path": { "type": "string", "description": "Output file (default: <workspace>/export/<project>.<ext>)" },
        }), &["format"]),
        tool("run_script", "Run a JavaScript program (QuickJS) inside voxely: one call instead of many. Every tool is voxely.<camelCase name>(args) with the same arguments, e.g. voxely.fillBox({from: [0, 0, 0], to: [9, 9, 0], color: \"#88aa44\"}); it returns parsed JSON or text and throws on errors. For bulk edits, v = voxely.getVolume({part}) gives a part's grid in its own coordinates (default: the part being edited): v.size and v.cells (a Uint8Array of palette indices, x fastest, then y, then z) with v.get(x, y, z), v.set(x, y, z, c) and v.forEach(fn); voxely.setVolume(v, {part}) writes it back (new Volume([x, y, z]) starts empty and resizes the part). voxely.color(\"#rrggbb\") gives the nearest palette index. Returns console.log output and the last expression's value. The whole script is one undo step. No file or network access.", json!({
            "source": { "type": "string", "description": "JavaScript source" },
            "path": { "type": "string", "description": "Or a .js file to run" },
            "args": { "description": "Any JSON value, available to the script as `args`" },
            "timeout": { "type": "number", "description": "Seconds before the script is stopped (default 120)" },
        }), &[]),
        tool("set_view_mode", "Show the modeler or the live path tracer in the window.", json!({
            "mode": { "type": "string", "enum": ["model", "render"] },
        }), &["mode"]),
        tool("list_parts", "The scene tree: every part (model) and group with its id, name, position (world min corner), size and children.", json!({}), &[]),
        tool("edit_part", "Edit a part's voxels (the voxel tools then work in its coordinates), or pass \"world\" for World mode, where the window shows every part and you arrange them.", json!({
            "part": { "type": ["string", "integer"], "description": "Part name or id, or \"world\"" },
        }), &["part"]),
        tool("create_part", "Add a new empty part (a separate model) and start editing it.", json!({
            "name": { "type": "string" },
            "size": xyz("Size in voxels, up to 256 per axis for MagicaVoxel compatibility"),
            "position": xyz("World position of its minimum corner (default: beside the existing parts)"),
            "parent": { "type": ["string", "integer"], "description": "Group to put it in (default: the top level)" },
            "fill": color_schema(),
        }), &["name", "size"]),
        tool("transform_part", "Move, rotate or flip a part or group as a whole. 90° turns and flips are exact and turn about its pivot (default: its center), around world axes. free_rotation adds any angle (shown resampled; edit the part upright). attach_to attaches it to another part (snapping its pivot or own socket onto that part's socket) so it follows that part from then on, like a sword in a hand.", json!({
            "part": { "type": ["string", "integer"] },
            "move": xyz("[dx, dy, dz] offset"),
            "position": xyz("New world position of its minimum corner"),
            "rotate_axis": axis,
            "quarter_turns": { "type": "integer", "description": "Counter-clockwise quarter turns about rotate_axis (negative for clockwise)" },
            "flip_axis": axis,
            "free_rotation": { "type": "array", "items": { "type": "number" }, "description": "[x, y, z] degrees about the pivot, applied x then y then z; replaces any previous free rotation. [0, 0, 0] straightens." },
            "attach_to": { "type": "string", "description": "\"part:socket\" (e.g. \"body:shoulder_r\") to snap onto a socket and attach, or \"part\" to attach where it is. Attached parts follow their parent's moves and turns (and nest in glTF exports)." },
            "link": { "type": "boolean", "description": "With attach_to: false only lines it up without attaching (default true)" },
            "attach_socket": { "type": "string", "description": "With attach_to: this part's own socket to line up (default: its pivot). It becomes the pivot if none was set, so later turns keep it attached." },
        }), &["part"]),
        tool("set_pivot", "Set the point a part or group turns about (and its origin in glTF exports), in the part's own coordinates (the ones the voxel tools use while editing it; world coordinates for groups). E.g. a shoulder for an arm, the hinge for a door.", json!({
            "part": { "type": ["string", "integer"] },
            "pivot": { "type": ["array", "string"], "items": { "type": "number" }, "description": "[x, y, z], or \"center\" (default), or \"bottom\" (middle of the bottom face)" },
        }), &["part", "pivot"]),
        tool("set_socket", "Add, move or remove a named attachment point (socket) on a part, in its own coordinates. Other parts attach to sockets with transform_part attach_to; glTF exports them as empty nodes engines can parent things to (hand, muzzle, seat ...).", json!({
            "part": { "type": ["string", "integer"] },
            "name": { "type": "string" },
            "position": { "type": "array", "items": { "type": "number" }, "description": "[x, y, z]; omit to remove the socket" },
        }), &["part", "name"]),
        tool("modify_part", "Rename, duplicate, delete, hide, show or ungroup a part or group.", json!({
            "part": { "type": ["string", "integer"] },
            "action": { "type": "string", "enum": ["rename", "duplicate", "delete", "hide", "show", "ungroup", "detach"], "description": "detach: stop following what it's attached to (it stays where it is)" },
            "name": { "type": "string", "description": "New name (rename) or the copy's name (duplicate)" },
            "offset": xyz("duplicate: where the copy goes relative to the original (default: just beside it on x)"),
        }), &["part", "action"]),
        tool("group_parts", "Put parts and groups into a new group, so they move and turn together.", json!({
            "parts": { "type": "array", "items": { "type": ["string", "integer"] } },
            "name": { "type": "string" },
        }), &["parts"]),
    ])
}

/// Tools that change or read one part's voxels; they take an optional
/// `part` to edit first.
const PART_TOOLS: &[&str] = &[
    "set_voxels", "fill_box", "draw_shape", "draw_line", "get_voxels", "get_slice", "edit_model", "transform_model",
    "resize_canvas", "run_shader", "select", "move_selection",
];

fn is_part_tool(name: &str) -> bool {
    [PART_TOOLS, build::PART_TOOLS, guide::PART_TOOLS, inspect::PART_TOOLS, surface::PART_TOOLS, view::PART_TOOLS].iter().any(|l| l.contains(&name))
}

/// The tool list for MCP `tools/list`, with the shared `part` argument.
pub fn tools() -> Value {
    let mut list = base_tools();
    let extra = [build::tools(), guide::tools(), inspect::tools(), surface::tools(), view::tools()];
    list.as_array_mut().unwrap().extend(extra.into_iter().flatten());
    for t in list.as_array_mut().into_iter().flatten() {
        if is_part_tool(t["name"].as_str().unwrap_or_default()) {
            t["inputSchema"]["properties"]["part"] = json!({ "type": ["string", "integer"], "description": "Part to work on (name or id); default: the part being edited" });
        }
    }
    list
}

/// Run one tool and build the MCP `tools/call` result.
pub fn call(ctx: &mut Ctx, name: &str, args: &Value) -> Value {
    match dispatch(ctx, name, args) {
        Ok(content) => json!({ "content": content, "isError": false }),
        Err(e) => json!({ "content": [text(format!("{e:#}"))], "isError": true }),
    }
}

fn text(s: impl Into<String>) -> Value {
    json!({ "type": "text", "text": s.into() })
}

fn image(w: u32, h: u32, rgba: &[u8]) -> Result<Value> {
    let png = encode_png(w, h, rgba)?;
    Ok(json!({ "type": "image", "data": base64::engine::general_purpose::STANDARD.encode(png), "mimeType": "image/png" }))
}

fn encode_png(w: u32, h: u32, rgba: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w, h);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header()?.write_image_data(rgba)?;
    }
    Ok(out)
}

// ---------------------------------------------------------------- args

fn get<'a>(args: &'a Value, key: &str) -> Option<&'a Value> {
    args.get(key).filter(|v| !v.is_null())
}

fn ivec3(args: &Value, key: &str) -> Result<IVec3> {
    opt_ivec3(args, key)?.ok_or_else(|| anyhow!("missing `{key}` ([x, y, z])"))
}

fn opt_ivec3(args: &Value, key: &str) -> Result<Option<IVec3>> {
    let Some(v) = get(args, key) else { return Ok(None) };
    let a = v.as_array().filter(|a| a.len() == 3).ok_or_else(|| anyhow!("`{key}` must be [x, y, z]"))?;
    let n = |i: usize| a[i].as_f64().map(|f| f.round() as i32).ok_or_else(|| anyhow!("`{key}` must hold numbers"));
    Ok(Some(IVec3::new(n(0)?, n(1)?, n(2)?)))
}

fn num(args: &Value, key: &str) -> Option<f32> {
    get(args, key).and_then(Value::as_f64).map(|v| v as f32)
}

fn int(args: &Value, key: &str) -> Option<i64> {
    get(args, key).and_then(Value::as_i64)
}

fn boolean(args: &Value, key: &str) -> Option<bool> {
    get(args, key).and_then(Value::as_bool)
}

fn string<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    get(args, key).and_then(Value::as_str)
}

fn axis(args: &Value) -> Result<usize> {
    axis_named(args, "axis")
}

fn axis_named(args: &Value, key: &str) -> Result<usize> {
    match string(args, key).unwrap_or("z") {
        "x" | "X" => Ok(0),
        "y" | "Y" => Ok(1),
        "z" | "Z" => Ok(2),
        a => bail!("unknown axis {a:?}"),
    }
}

fn parse_hex(s: &str) -> Result<[u8; 4]> {
    crate::palette_ops::parse_color_code(s).ok_or_else(|| anyhow!("bad color {s:?}; use \"#rrggbb\""))
}

fn hex_linear(s: &str) -> Result<Vec3> {
    let c = parse_hex(s)?;
    Ok(srgb_to_linear(Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32) / 255.0))
}

fn hex(c: [u8; 4]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

/// A color argument: an index, or "#rrggbb" matched to the nearest palette
/// entry (1..=255).
fn color(scene: &Scene, v: &Value) -> Result<u8> {
    if let Some(i) = v.as_i64() {
        return u8::try_from(i).map_err(|_| anyhow!("color index {i} is out of 0..=255"));
    }
    let s = v.as_str().ok_or_else(|| anyhow!("color must be an index or \"#rrggbb\""))?;
    let c = parse_hex(s)?;
    let dist = |p: &[u8; 4]| (0..3).map(|k| (p[k] as i32 - c[k] as i32).pow(2)).sum::<i32>();
    Ok((1..256).min_by_key(|&i| dist(&scene.palette[i])).unwrap() as u8)
}

fn color_arg(scene: &Scene, args: &Value, key: &str) -> Result<u8> {
    color(scene, get(args, key).ok_or_else(|| anyhow!("missing `{key}`"))?)
}

// ---------------------------------------------------------------- helpers

/// Commit an edit to the volume and push it to the GPU.
pub(crate) fn apply(ctx: &mut Ctx, new: Volume) -> bool {
    let new = build::before_commit(ctx, new);
    let edit = LastEdit::diff(ctx.scene.editing(), &ctx.scene.volume, &new);
    let changed = ctx.ed.editor.commit(&mut ctx.scene.volume, new);
    if changed {
        app::sync_volume(ctx.r, ctx.scene, &mut ctx.ed.editor);
        ctx.ed.tools.last_edit = Some(edit);
        inspect::after_commit(ctx);
    }
    changed
}

fn refresh_selection(ctx: &mut Ctx) {
    if ctx.ed.editor.sel_dirty {
        ctx.r.upload_selection(&ctx.scene.volume, ctx.ed.editor.selection());
        ctx.ed.editor.sel_dirty = false;
    }
}

fn materials_changed(ctx: &mut Ctx) {
    ctx.r.upload_materials(&ctx.scene.materials);
    ctx.r.set_palette(&ctx.scene.palette);
    ctx.scene.rebuild_lights();
    ctx.r.set_lights(&ctx.scene.lights, ctx.scene.light_power);
}

fn count(v: &Volume) -> usize {
    v.cells.iter().filter(|&&c| c != 0).count()
}

fn bounds(v: &Volume) -> Option<(IVec3, IVec3)> {
    let mut b: Option<(IVec3, IVec3)> = None;
    for (i, &c) in v.cells.iter().enumerate() {
        if c != 0 {
            let p = v.position(i);
            b = Some(b.map_or((p, p), |(lo, hi)| (lo.min(p), hi.max(p))));
        }
    }
    b
}

fn arr(p: IVec3) -> [i32; 3] {
    p.to_array()
}

/// A summary line after an edit.
pub(crate) fn edited(ctx: &Ctx, changed: bool, what: &str) -> Vec<Value> {
    let n = count(&ctx.scene.volume);
    let d = ctx.scene.volume.dims;
    if changed {
        vec![text(format!("{what}. Canvas {}x{}x{}, {n} voxels.", d.x, d.y, d.z))]
    } else {
        vec![text(format!("{what}: nothing changed. Canvas {}x{}x{}, {n} voxels.", d.x, d.y, d.z))]
    }
}

const NEIGHBORS: [IVec3; 6] = [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::Z, IVec3::NEG_Z];

/// Fill `shape` between two corners with `c`.
fn shape_volume(v: &Volume, shape: &str, axis: usize, a: IVec3, b: IVec3, c: u8, hollow: bool) -> Volume {
    build::fill_solid(v, &build::Solid::new(shape, axis, a, b), c, hollow)
}

fn shader_path(ctx: &Ctx, name: &str) -> Result<PathBuf> {
    ctx.ed
        .shaders
        .files
        .iter()
        .find(|p| p.file_stem().is_some_and(|s| s.eq_ignore_ascii_case(name)))
        .cloned()
        .ok_or_else(|| anyhow!("no shader named {name:?}; see list_shaders"))
}

/// Pixels of a final render from the current camera.
fn render_pixels(ctx: &Ctx, w: u32, h: u32, spp: u32) -> Result<Vec<u8>> {
    let mut out = Renderer::new(ctx.r.device.clone(), ctx.r.queue.clone(), ctx.scene, None, w, h);
    out.settings = ctx.r.settings.clone();
    out.set_camera(ctx.r.camera);
    out.render_image(spp)
}

// ---------------------------------------------------------------- tools

fn part_key(v: &Value) -> Option<String> {
    v.as_str().map(str::to_string).or_else(|| v.as_u64().map(|n| n.to_string()))
}

fn find_part(scene: &Scene, v: Option<&Value>) -> Result<usize> {
    let key = v.and_then(part_key).ok_or_else(|| anyhow!("missing `part`"))?;
    scene.world.find(&key).filter(|&id| id != crate::world::ROOT).ok_or_else(|| anyhow!("no part named `{key}`; see list_parts"))
}

/// A node and its subtree as JSON.
fn part_json(scene: &Scene, id: usize) -> Value {
    let w = &scene.world;
    let Some(n) = w.node(id) else { return Value::Null };
    let (lo, size) = w.bounds(id).unwrap_or((IVec3::ZERO, IVec3::ZERO));
    let mut v = json!({
        "id": id,
        "name": w.label(id),
        "kind": if n.is_group() { "group" } else { "part" },
        "position": arr(lo),
        "size": arr(size),
    });
    if n.hidden {
        v["hidden"] = json!(true);
    }
    if scene.editing() == Some(id) {
        v["editing"] = json!(true);
    }
    if let Some(m) = n.volume() {
        v["voxels"] = json!(if scene.editing() == Some(id) { count(&scene.volume) } else { count(m) });
    }
    if n.pivot.is_some() {
        v["pivot"] = json!(w.to_part(id, w.pivot_local(id)).to_array());
    }
    if n.tilted() {
        v["free_rotation"] = json!(n.tilt.to_array());
    }
    if !n.sockets.is_empty() {
        v["sockets"] = n
            .sockets
            .iter()
            .map(|s| json!({ "name": s.name, "position": w.to_part(id, s.pos).to_array(), "world": w.affine(id).transform_point3(s.pos).to_array() }))
            .collect();
    }
    if !n.children().is_empty() {
        let key = if n.is_group() { "children" } else { "attached" };
        v[key] = n.children().iter().map(|&c| part_json(scene, c)).collect();
    }
    v
}

fn vec3_value(v: &Value) -> Option<Vec3> {
    let a: Vec<f32> = v.as_array()?.iter().map(|x| x.as_f64().map(|f| f as f32)).collect::<Option<_>>()?;
    (a.len() == 3).then(|| Vec3::new(a[0], a[1], a[2]))
}

/// After a tree change: reset voxel history and upload.
fn tree_changed(ctx: &mut Ctx) {
    ctx.ed.editor.reset();
    app::sync_volume(ctx.r, ctx.scene, &mut ctx.ed.editor);
}

/// Switch to editing `part` (a name or id) for a voxel tool, or check that
/// some part is being edited if none is given.
pub(crate) fn focus_part(ctx: &mut Ctx, part: Option<&Value>) -> Result<()> {
    if let Some(p) = part.filter(|p| !p.is_null()) {
        let id = find_part(ctx.scene, Some(p))?;
        if ctx.scene.editing() != Some(id) {
            if ctx.scene.world.node(id).is_some_and(|n| n.is_group()) {
                bail!("`{}` is a group; pass one of its parts", ctx.scene.world.label(id));
            }
            app::set_mode(ctx.scene, ctx.ed, Some(id));
            app::sync_volume(ctx.r, ctx.scene, &mut ctx.ed.editor);
        }
    } else if ctx.scene.editing().is_none() {
        bail!("the window is in World mode; pass `part` or call edit_part first (list_parts shows the parts)");
    }
    Ok(())
}

fn dispatch(ctx: &mut Ctx, name: &str, args: &Value) -> Result<Vec<Value>> {
    let a = args;
    if is_part_tool(name) {
        focus_part(ctx, get(a, "part"))?;
    }
    for family in [build::dispatch, guide::dispatch, inspect::dispatch, surface::dispatch, view::dispatch] {
        if let Some(result) = family(ctx, name, a) {
            return result;
        }
    }
    match name {
        "list_parts" => {
            let top = ctx.scene.world.node(crate::world::ROOT).map(|n| n.children().to_vec()).unwrap_or_default();
            let tree: Vec<Value> = top.iter().map(|&c| part_json(ctx.scene, c)).collect();
            Ok(vec![text(serde_json::to_string_pretty(&json!({ "mode": if ctx.scene.editing().is_some() { "edit" } else { "world" }, "parts": tree }))?)])
        }
        "edit_part" => {
            if get(a, "part").and_then(Value::as_str).is_some_and(|s| s.eq_ignore_ascii_case("world")) {
                app::set_mode(ctx.scene, ctx.ed, None);
                app::sync_volume(ctx.r, ctx.scene, &mut ctx.ed.editor);
                return Ok(vec![text(format!("World mode: {} parts", ctx.scene.world.models().len()))]);
            }
            let id = find_part(ctx.scene, get(a, "part"))?;
            if !app::set_mode(ctx.scene, ctx.ed, Some(id)) {
                bail!("`{}` has no parts to edit", ctx.scene.world.label(id));
            }
            app::sync_volume(ctx.r, ctx.scene, &mut ctx.ed.editor);
            let id = ctx.scene.editing().unwrap_or(id);
            Ok(edited(ctx, true, &format!("Editing `{}` at {:?}", ctx.scene.world.label(id), arr(ctx.scene.origin))))
        }
        "create_part" => {
            let name = string(a, "name").context("missing `name`")?.trim().to_string();
            let size = ivec3(a, "size")?;
            if size.cmplt(IVec3::ONE).any() || size.cmpgt(IVec3::splat(ops::MAX_DIM)).any() {
                bail!("size must be 1..={} per axis", ops::MAX_DIM);
            }
            if !name.is_empty() && ctx.scene.world.find(&name).is_some() {
                bail!("a part named `{name}` already exists");
            }
            let parent = match get(a, "parent") {
                Some(p) => find_part(ctx.scene, Some(p))?,
                None => crate::world::ROOT,
            };
            let w = &ctx.scene.world;
            let position = match opt_ivec3(a, "position")? {
                Some(p) => p,
                None => w.bounds(crate::world::ROOT).map_or(IVec3::ZERO, |(lo, s)| IVec3::new(lo.x + s.x + 2, lo.y, lo.z.max(0))),
            };
            let mut v = Volume::new(size);
            if let Some(c) = get(a, "fill") {
                v = ops::full(&v, color(ctx.scene, c)?);
            }
            let id = ctx.scene.change(|w| w.add_model(parent, &name, v, position));
            app::set_mode(ctx.scene, ctx.ed, Some(id));
            app::sync_volume(ctx.r, ctx.scene, &mut ctx.ed.editor);
            Ok(edited(ctx, true, &format!("Created part `{}` (id {id}) at {:?}; now editing it", ctx.scene.world.label(id), arr(position))))
        }
        "transform_part" => {
            let id = find_part(ctx.scene, get(a, "part"))?;
            let mv = opt_ivec3(a, "move")?;
            let pos = opt_ivec3(a, "position")?;
            let turn = match (get(a, "rotate_axis"), int(a, "quarter_turns")) {
                (Some(_), q) => Some((axis_named(a, "rotate_axis")?, q.unwrap_or(1) as i32)),
                (None, Some(_)) => bail!("quarter_turns needs rotate_axis"),
                _ => None,
            };
            let flip = match get(a, "flip_axis") {
                Some(_) => Some(axis_named(a, "flip_axis")?),
                None => None,
            };
            let tilt = match get(a, "free_rotation") {
                Some(v) => Some(vec3_value(v).context("free_rotation is [x, y, z] degrees")?),
                None => None,
            };
            let attach = match string(a, "attach_to") {
                Some(spec) => {
                    let (p, sock) = match spec.split_once(':') {
                        Some((p, s)) => (p, Some(s.to_string())),
                        None => (spec, None),
                    };
                    let target = find_part(ctx.scene, Some(&json!(p)))?;
                    if let Some(sock) = &sock {
                        ctx.scene.world.socket_world(target, sock).with_context(|| format!("`{p}` has no socket `{sock}`"))?;
                    }
                    let own = string(a, "attach_socket").map(str::to_string);
                    Some((target, sock, own))
                }
                None => None,
            };
            let link = boolean(a, "link").unwrap_or(true);
            if mv.is_none() && pos.is_none() && turn.is_none() && flip.is_none() && tilt.is_none() && attach.is_none() {
                bail!("give move, position, rotate_axis, flip_axis, free_rotation or attach_to");
            }
            if let Some((_, _, Some(own))) = &attach {
                if ctx.scene.world.socket_world(id, own).is_none() {
                    bail!("`{}` has no socket `{own}`", ctx.scene.world.label(id));
                }
            }
            let linked = ctx.scene.change(|w| {
                if let Some((axis, q)) = turn {
                    w.reorient(id, crate::world::Rot::turn(axis, q));
                }
                if let Some(axis) = flip {
                    w.reorient(id, crate::world::Rot::flip(axis));
                }
                if let Some(p) = pos {
                    if let Some((lo, _)) = w.bounds(id) {
                        w.translate(id, p - lo);
                    }
                }
                if let Some(d) = mv {
                    w.translate(id, d);
                }
                if let Some(t) = tilt {
                    w.set_tilt(id, t);
                }
                match &attach {
                    Some((target, sock, own)) if link => Some(w.attach_to(id, *target, sock.as_deref(), own.as_deref())),
                    Some((target, Some(sock), own)) => {
                        // Snap only: line up, don't link.
                        let at = w.socket_world(*target, sock).unwrap_or_default();
                        let own_pos = own.as_deref().and_then(|o| w.node(id)?.sockets.iter().find(|s| s.name.eq_ignore_ascii_case(o)).map(|s| s.pos));
                        if let (Some(n), Some(pos)) = (w.node_mut(id), own_pos) {
                            n.pivot = n.pivot.or(Some(pos));
                        }
                        let from = own_pos.map_or_else(|| w.pivot_world(id), |p| w.affine(id).transform_point3(p));
                        w.translate(id, (at - from).round().as_ivec3());
                        None
                    }
                    _ => None,
                }
            });
            tree_changed(ctx);
            if linked == Some(false) {
                bail!("can't attach `{}` to something attached to it", ctx.scene.world.label(id));
            }
            let (lo, size) = ctx.scene.world.bounds(id).unwrap_or_default();
            let w = &ctx.scene.world;
            let to = w.node(id).and_then(|n| n.parent).filter(|&p| p != crate::world::ROOT).map(|p| format!(", attached to `{}`", w.label(p))).unwrap_or_default();
            Ok(vec![text(format!("`{}` now at {:?}, size {:?}{to}", w.label(id), arr(lo), arr(size)))])
        }
        "set_pivot" => {
            let id = find_part(ctx.scene, get(a, "part"))?;
            let w = &ctx.scene.world;
            let pivot = match get(a, "pivot") {
                Some(Value::String(s)) if s == "center" => None,
                Some(Value::String(s)) if s == "bottom" => {
                    let (_, size) = w.bounds(id).context("empty")?;
                    Some(w.from_part(id, Vec3::new(size.x as f32 * 0.5, size.y as f32 * 0.5, 0.0)))
                }
                Some(v) => Some(w.from_part(id, vec3_value(v).context("pivot is [x, y, z], \"center\" or \"bottom\"")?)),
                None => bail!("missing `pivot`"),
            };
            ctx.scene.change(|w| w.node_mut(id).map(|n| n.pivot = pivot));
            tree_changed(ctx);
            let w = &ctx.scene.world;
            Ok(vec![text(format!("`{}` pivot at {:?} (part coordinates), {:?} world", w.label(id), w.to_part(id, w.pivot_local(id)).to_array(), w.pivot_world(id).to_array()))])
        }
        "set_socket" => {
            let id = find_part(ctx.scene, get(a, "part"))?;
            let name = string(a, "name").context("missing `name`")?.trim().to_string();
            if name.is_empty() || name.contains(['|', ':']) {
                bail!("socket names can't be empty or contain | or :");
            }
            let pos = match get(a, "position") {
                Some(v) => Some(ctx.scene.world.from_part(id, vec3_value(v).context("position is [x, y, z]")?)),
                None => None,
            };
            let existed = ctx.scene.world.node(id).is_some_and(|n| n.sockets.iter().any(|s| s.name == name));
            if pos.is_none() && !existed {
                bail!("`{}` has no socket `{name}`", ctx.scene.world.label(id));
            }
            ctx.scene.change(|w| {
                let Some(n) = w.node_mut(id) else { return };
                n.sockets.retain(|s| s.name != name);
                if let Some(pos) = pos {
                    n.sockets.push(crate::world::Socket { name: name.clone(), pos });
                }
            });
            tree_changed(ctx);
            let w = &ctx.scene.world;
            Ok(vec![text(match pos {
                Some(_) => format!("Socket `{name}` on `{}` at {:?} world", w.label(id), w.socket_world(id, &name).unwrap_or_default().to_array()),
                None => format!("Removed socket `{name}`"),
            })])
        }
        "modify_part" => {
            let id = find_part(ctx.scene, get(a, "part"))?;
            let label = ctx.scene.world.label(id);
            let msg = match string(a, "action").context("missing `action`")? {
                "rename" => {
                    let name = string(a, "name").context("rename needs `name`")?.trim().to_string();
                    ctx.scene.change(|w| w.node_mut(id).map(|n| n.name = name.clone()));
                    format!("Renamed `{label}` to `{name}`")
                }
                "duplicate" => {
                    let width = ctx.scene.world.bounds(id).map_or(0, |b| b.1.x);
                    let offset = opt_ivec3(a, "offset")?.unwrap_or(IVec3::new(width + 1, 0, 0));
                    let name = string(a, "name").map(|n| n.trim().to_string());
                    let copy = ctx.scene.change(|w| {
                        let c = w.duplicate(id)?;
                        w.translate(c, offset);
                        if let (Some(n), Some(name)) = (w.node_mut(c), name) {
                            n.name = name;
                        }
                        Some(c)
                    });
                    let copy = copy.context("can't duplicate the root")?;
                    format!("Copied `{label}` to `{}` (id {copy})", ctx.scene.world.label(copy))
                }
                "delete" => {
                    let mut left = ctx.scene.world.clone();
                    left.delete(id);
                    if left.models().is_empty() {
                        bail!("can't delete the last part");
                    }
                    ctx.scene.change(|w| w.delete(id));
                    format!("Deleted `{label}`")
                }
                act @ ("hide" | "show") => {
                    ctx.scene.change(|w| w.node_mut(id).map(|n| n.hidden = act == "hide"));
                    format!("{} `{label}`", if act == "hide" { "Hid" } else { "Showed" })
                }
                "detach" => {
                    if !ctx.scene.change(|w| w.detach(id)) {
                        bail!("`{label}` isn't attached to anything");
                    }
                    format!("Detached `{label}`")
                }
                "ungroup" => {
                    if !ctx.scene.world.node(id).is_some_and(|n| n.is_group()) {
                        bail!("`{label}` isn't a group");
                    }
                    ctx.scene.change(|w| w.ungroup(id));
                    format!("Ungrouped `{label}`")
                }
                other => bail!("unknown action `{other}`"),
            };
            tree_changed(ctx);
            Ok(vec![text(msg)])
        }
        "group_parts" => {
            let parts = get(a, "parts").and_then(Value::as_array).context("missing `parts`")?;
            let ids = parts.iter().map(|p| find_part(ctx.scene, Some(p))).collect::<Result<Vec<_>>>()?;
            let name = string(a, "name").unwrap_or_default().trim().to_string();
            let g = ctx.scene.change(|w| {
                let g = w.group(&ids)?;
                if let Some(n) = w.node_mut(g) {
                    n.name = name;
                }
                Some(g)
            });
            let g = g.context("nothing to group")?;
            tree_changed(ctx);
            Ok(vec![text(format!("Grouped {} into `{}` (id {g})", ids.len(), ctx.scene.world.label(g)))])
        }
        "get_scene_info" => {
            let v = &ctx.scene.volume;
            let mut counts = [0usize; 256];
            for &c in &v.cells {
                counts[c as usize] += 1;
            }
            let colors: Vec<Value> = (1..256)
                .filter(|&i| counts[i] > 0)
                .map(|i| json!({ "index": i, "color": hex(ctx.scene.palette[i]), "count": counts[i] }))
                .collect();
            let cam = &ctx.r.camera;
            let info = json!({
                "mode": if ctx.live { "live window" } else { "headless" },
                "view": if ctx.r.modeling { "model" } else { "render" },
                "size": arr(v.dims),
                "voxels": count(v),
                "bounds": bounds(v).map(|(lo, hi)| json!({ "min": arr(lo), "max": arr(hi) })),
                "colors_in_use": colors,
                "current_color": ctx.ed.editor.color,
                "selected": ctx.ed.editor.selected_count(),
                "file": ctx.ed.source.as_ref().map(|p| p.display().to_string()),
                "save_path": ctx.ed.save_path.display().to_string(),
                "can_undo": ctx.ed.editor.can_undo() || ctx.scene.can_undo_tree(),
                "editing_part": ctx.scene.editing().map(|id| ctx.scene.world.label(id)),
                "part_position": arr(ctx.scene.origin),
                "parts": ctx.scene.world.models().len(),
                "camera": { "yaw": cam.yaw.to_degrees(), "pitch": cam.pitch.to_degrees(), "distance": cam.dist, "target": cam.target.to_array() },
            });
            Ok(vec![text(serde_json::to_string_pretty(&info)?)])
        }
        "new_scene" => {
            let size = opt_ivec3(a, "size")?.unwrap_or(IVec3::splat(40));
            if size.cmplt(IVec3::ONE).any() || size.cmpgt(IVec3::splat(ops::MAX_DIM)).any() {
                bail!("size must be 1..={} per axis", ops::MAX_DIM);
            }
            let ibl = ctx.r.settings.ibl_path.clone();
            *ctx.scene = Scene::empty(size);
            ctx.scene.settings.ibl_path = ibl;
            if let Some(c) = get(a, "fill") {
                let c = color(ctx.scene, c)?;
                ctx.scene.volume = ops::full(&ctx.scene.volume, c);
            }
            ctx.ed.source = None;
            ctx.ed.save_path = app::default_save_path(None);
            app::loaded(ctx.r, ctx.scene, ctx.ed);
            Ok(edited(ctx, true, "New scene"))
        }
        "open_file" => {
            let path = PathBuf::from(string(a, "path").context("missing `path`")?);
            app::open_project(path.clone(), ctx.r, ctx.scene, ctx.ed)?;
            Ok(edited(ctx, true, &format!("Opened {}", path.display())))
        }
        "save_file" => {
            let path = string(a, "path").map_or_else(|| ctx.ed.save_path.clone(), |p| PathBuf::from(p).with_extension("vox"));
            let path = app::save_project(ctx.scene, ctx.ed, path)?;
            let full = std::fs::canonicalize(&path).unwrap_or(path);
            Ok(vec![text(format!("Saved {}", full.display()))])
        }
        "new_project" => {
            let name = string(a, "name").context("missing `name`")?.trim();
            if name.is_empty() || name.len() > 80 || !name.chars().all(|c| c.is_alphanumeric() || " -_".contains(c)) {
                bail!("project names use letters, digits, spaces, - and _");
            }
            let path = app::project_path(name);
            if path.exists() && !boolean(a, "overwrite").unwrap_or(false) {
                bail!("a project named {name:?} exists; open_project it, or pass overwrite: true");
            }
            let size = opt_ivec3(a, "size")?.unwrap_or(IVec3::splat(40));
            if size.cmplt(IVec3::ONE).any() || size.cmpgt(IVec3::splat(ops::MAX_DIM)).any() {
                bail!("size must be 1..={} per axis", ops::MAX_DIM);
            }
            let ibl = ctx.r.settings.ibl_path.clone();
            *ctx.scene = Scene::empty(size);
            ctx.scene.settings.ibl_path = ibl;
            if let Some(c) = get(a, "fill") {
                let c = color(ctx.scene, c)?;
                ctx.scene.volume = ops::full(&ctx.scene.volume, c);
            }
            app::loaded(ctx.r, ctx.scene, ctx.ed);
            let path = app::save_project(ctx.scene, ctx.ed, path)?;
            Ok(edited(ctx, true, &format!("Created project {name:?} at {}", path.display())))
        }
        "list_projects" => {
            ctx.ed.rescan();
            let modified = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
            let mut projects: Vec<&PathBuf> = ctx.ed.projects.iter().collect();
            projects.sort_by_key(|p| std::cmp::Reverse(modified(p)));
            let stem = |p: &PathBuf| p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            let current = ctx.ed.source.as_ref();
            let list: Vec<Value> = projects.iter().map(|p| json!({ "name": stem(p), "current": Some(*p) == current })).collect();
            let samples: Vec<String> = ctx.ed.samples.iter().map(stem).collect();
            let out = json!({ "workspace": app::workspace_dir().display().to_string(), "projects": list, "samples": samples });
            Ok(vec![text(serde_json::to_string_pretty(&out)?)])
        }
        "open_project" => {
            let name = string(a, "name").context("missing `name`")?;
            ctx.ed.rescan();
            let matches = |p: &&PathBuf| p.file_stem().is_some_and(|s| s.to_string_lossy().eq_ignore_ascii_case(name));
            let path = ctx.ed.projects.iter().find(matches).or_else(|| ctx.ed.samples.iter().find(matches)).cloned().ok_or_else(|| anyhow!("no project or sample named {name:?}; see list_projects"))?;
            app::open_project(path.clone(), ctx.r, ctx.scene, ctx.ed)?;
            Ok(edited(ctx, true, &format!("Opened {}", path.display())))
        }
        "set_voxels" => {
            let list = get(a, "voxels").and_then(Value::as_array).context("`voxels` must be [[x, y, z, color], ...]")?;
            let mut new = ctx.scene.volume.clone();
            let mut skipped = 0;
            for item in list {
                let v = item.as_array().filter(|v| v.len() == 4).context("each voxel is [x, y, z, color]")?;
                let n = |i: usize| v[i].as_f64().map(|f| f.round() as i32).context("voxel coordinates must be numbers");
                let p = IVec3::new(n(0)?, n(1)?, n(2)?);
                let c = color(ctx.scene, &v[3])?;
                if new.contains(p) {
                    new.set_index(new.index(p), c);
                } else {
                    skipped += 1;
                }
            }
            let changed = apply(ctx, new);
            let mut out = edited(ctx, changed, &format!("Set {} voxels", list.len() - skipped));
            if skipped > 0 {
                out.push(text(format!("{skipped} voxels were outside the canvas and skipped; use resize_canvas to grow it.")));
            }
            Ok(out)
        }
        "fill_box" => {
            let (from, to) = (ivec3(a, "from")?, ivec3(a, "to")?);
            let c = color_arg(ctx.scene, a, "color")?;
            let new = shape_volume(&ctx.scene.volume, "box", 2, from, to, c, boolean(a, "hollow").unwrap_or(false));
            let changed = apply(ctx, new);
            Ok(edited(ctx, changed, "Filled box"))
        }
        "draw_shape" => {
            let shape = string(a, "shape").context("missing `shape`")?;
            let (from, to) = (ivec3(a, "from")?, ivec3(a, "to")?);
            let c = color_arg(ctx.scene, a, "color")?;
            let solid = build::Solid::build(shape, axis(a)?, from, to, a)?;
            let new = build::fill_solid(&ctx.scene.volume, &solid, c, boolean(a, "hollow").unwrap_or(false));
            let changed = apply(ctx, new);
            Ok(edited(ctx, changed, &format!("Drew {shape}")))
        }
        "draw_line" => {
            let (from, to) = (ivec3(a, "from")?, ivec3(a, "to")?);
            let c = color_arg(ctx.scene, a, "color")?;
            let thick = int(a, "thickness").unwrap_or(1).clamp(1, 64) as i32;
            let mut new = ctx.scene.volume.clone();
            let d = to - from;
            let steps = d.abs().max_element().max(1);
            let r = thick as f32 * 0.5;
            let k = (thick - 1) / 2 + 1;
            for i in 0..=steps {
                let p = (from.as_vec3() + d.as_vec3() * (i as f32 / steps as f32)).round().as_ivec3();
                for z in -k..=k {
                    for y in -k..=k {
                        for x in -k..=k {
                            let o = IVec3::new(x, y, z);
                            if thick == 1 && o != IVec3::ZERO || o.as_vec3().length() > r {
                                continue;
                            }
                            new.set(p + o, c);
                        }
                    }
                }
            }
            let changed = apply(ctx, new);
            Ok(edited(ctx, changed, "Drew line"))
        }
        "get_voxels" => {
            let v = &ctx.scene.volume;
            let lo = opt_ivec3(a, "from")?.unwrap_or(IVec3::ZERO);
            let hi = opt_ivec3(a, "to")?.unwrap_or(v.dims - IVec3::ONE);
            let (lo, hi) = (lo.min(hi), lo.max(hi));
            let limit = int(a, "limit").unwrap_or(5000).max(1) as usize;
            let mut list = Vec::new();
            let mut total = 0;
            for (i, &c) in v.cells.iter().enumerate() {
                let p = v.position(i);
                if c != 0 && p.cmpge(lo).all() && p.cmple(hi).all() {
                    total += 1;
                    if list.len() < limit {
                        list.push(json!([p.x, p.y, p.z, c]));
                    }
                }
            }
            let out = json!({ "total": total, "returned": list.len(), "voxels": list });
            Ok(vec![text(serde_json::to_string(&out)?)])
        }
        "get_slice" => {
            let v = &ctx.scene.volume;
            let ax = axis(a)?;
            let index = int(a, "index").context("missing `index`")? as i32;
            if index < 0 || index >= v.dims[ax] {
                bail!("index {index} is outside 0..{}", v.dims[ax]);
            }
            // Rows along the second in-plane axis (top = high), columns
            // along the first.
            let (u, w) = match ax {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            let names = ["x", "y", "z"];
            let mut out = format!("Slice {}={index}: columns {} 0..{}, rows {} from {} (top) down to 0\n", names[ax], names[u], v.dims[u] - 1, names[w], v.dims[w] - 1);
            let mut used = std::collections::BTreeSet::new();
            for row in (0..v.dims[w]).rev() {
                let mut line = format!("{row:>4} ");
                for col in 0..v.dims[u] {
                    let mut p = IVec3::ZERO;
                    p[ax] = index;
                    p[u] = col;
                    p[w] = row;
                    let c = v.get(p);
                    if c == 0 {
                        line.push_str("  .");
                    } else {
                        used.insert(c);
                        line.push_str(&format!("{c:>3}"));
                    }
                }
                out.push_str(line.trim_end());
                out.push('\n');
            }
            let legend: Vec<String> = used.iter().map(|&c| format!("{c}={}", hex(ctx.scene.palette[c as usize]))).collect();
            out.push_str(&format!("colors: {}", legend.join(" ")));
            Ok(vec![text(out)])
        }
        "edit_model" => {
            let op_name = string(a, "op").context("missing `op`")?;
            if let Some(c) = get(a, "color") {
                ctx.ed.editor.color = color(ctx.scene, c)?;
            }
            let ax = axis(a)?;
            let op = match op_name {
                "full" => Op::Full,
                "fill" => Op::Fill,
                "clear" => Op::Clear,
                "reverse" => Op::Reverse,
                "double" => Op::Double,
                "half" => Op::Half,
                "flip" => Op::Flip(ax),
                "rotate" => Op::Rotate(ax),
                "fit" => Op::Fit,
                "loop" => Op::Loop(ax, if int(a, "direction").unwrap_or(1) < 0 { -1 } else { 1 }),
                "mirror" => Op::Mirror(ax),
                "diagonal" => Op::Diagonal(ax),
                "hull" => Op::Hull,
                "flood" => Op::Flood,
                "dilate" => Op::Dilate,
                "erode" => Op::Erode,
                "maze" => Op::Maze,
                "perlin" => Op::Perlin,
                other => bail!("unknown op {other:?}"),
            };
            let new = app::apply_op(op, ctx.scene, &ctx.ed.editor).ok_or_else(|| anyhow!("{op_name} isn't possible here (empty model, or the canvas would exceed {})", ops::MAX_DIM))?;
            let changed = apply(ctx, new);
            Ok(edited(ctx, changed, &format!("Applied {op_name}")))
        }
        "transform_model" => {
            let kind = match string(a, "kind").context("missing `kind`")? {
                "scale" => TransformKind::Scale,
                "shear" => TransformKind::Shear,
                "rotate" => TransformKind::Rotate,
                "repeat" => TransformKind::Repeat,
                k => bail!("unknown kind {k:?}"),
            };
            let text_args = string(a, "args").context("missing `args`, e.g. \"xyz 2\"")?;
            let new = app::transform(kind, text_args, &ctx.scene.volume).ok_or_else(|| anyhow!("couldn't apply {text_args:?}: expected axes then values, within size limits"))?;
            let changed = apply(ctx, new);
            Ok(edited(ctx, changed, &format!("Applied {text_args}")))
        }
        "resize_canvas" => {
            let size = ivec3(a, "size")?;
            let new = ops::resize(&ctx.scene.volume, size);
            let changed = apply(ctx, new);
            Ok(edited(ctx, changed, "Resized canvas"))
        }
        "list_shaders" => {
            let mut out = Vec::new();
            for p in &ctx.ed.shaders.files {
                if let Ok(sh) = crate::xs::XsShader::load(p) {
                    let args: Vec<Value> = sh.args.iter().map(|x| json!({ "var": x.var, "name": x.name, "default": x.value, "min": x.min, "max": x.max })).collect();
                    out.push(json!({ "name": sh.name, "args": args }));
                }
            }
            if out.is_empty() {
                return Ok(vec![text("No shader folder found. Open a model from MagicaVoxel's vox folder first, or pass inline GLSL `source` to run_shader.")]);
            }
            Ok(vec![text(serde_json::to_string_pretty(&out)?)])
        }
        "run_shader" => {
            let shader = match (string(a, "name"), string(a, "source")) {
                (_, Some(src)) => crate::xs::XsShader::parse("inline", src)?,
                (Some(n), None) => crate::xs::XsShader::load(&shader_path(ctx, n)?)?,
                (None, None) => bail!("give a shader `name` or GLSL `source`"),
            };
            let mut values: Vec<f32> = shader.args.iter().map(|x| x.value).collect();
            match get(a, "args") {
                Some(Value::Array(list)) => {
                    for (slot, v) in values.iter_mut().zip(list) {
                        *slot = v.as_f64().context("shader args must be numbers")? as f32;
                    }
                }
                Some(Value::Object(map)) => {
                    for (k, v) in map {
                        let i = shader.args.iter().position(|x| x.var == *k || x.name.eq_ignore_ascii_case(k)).ok_or_else(|| anyhow!("shader has no arg {k:?}"))?;
                        values[i] = v.as_f64().context("shader args must be numbers")? as f32;
                    }
                }
                _ => {}
            }
            if let Some(c) = get(a, "color") {
                ctx.ed.editor.color = color(ctx.scene, c)?;
            }
            let sels: Vec<u8> = (1..256).filter(|&i| ctx.ed.palette.selected[i]).map(|i| i as u8).collect();
            let out = crate::xs::run(&ctx.r.device, &ctx.r.queue, &shader, &values, &ctx.scene.volume, &ctx.scene.palette, ctx.ed.editor.color, &sels)?;
            let new = match ctx.ed.editor.selection() {
                Some(sel) => Volume::from_cells(
                    out.dims,
                    ctx.scene.volume.cells.iter().zip(&out.cells).enumerate().map(|(i, (&old, &n))| if sel[i] { n } else { old }).collect(),
                ),
                None => out,
            };
            let changed = apply(ctx, new);
            Ok(edited(ctx, changed, &format!("Ran shader {}", shader.name)))
        }
        "get_palette" => {
            let all = boolean(a, "all").unwrap_or(false);
            let mut used = [false; 256];
            for &c in &ctx.scene.volume.cells {
                used[c as usize] = true;
            }
            let kinds = ["diffuse", "metal", "glass", "emit", "blend", "cloud"];
            let list: Vec<Value> = (1..256)
                .filter(|&i| all || used[i])
                .map(|i| {
                    let m = &ctx.scene.materials[i];
                    json!({ "index": i, "color": hex(ctx.scene.palette[i]), "material": kinds.get(m.kind as usize).copied().unwrap_or("?"), "in_use": used[i] })
                })
                .collect();
            Ok(vec![text(serde_json::to_string(&list)?)])
        }
        "set_palette_colors" => {
            let list = get(a, "colors").and_then(Value::as_array).context("`colors` must be [{index, color}, ...]")?;
            for item in list {
                let i = item.get("index").and_then(Value::as_u64).filter(|&i| (1..=255).contains(&i)).context("index must be 1..=255")? as usize;
                let c = parse_hex(item.get("color").and_then(Value::as_str).context("color must be \"#rrggbb\"")?)?;
                ctx.scene.palette[i] = [c[0], c[1], c[2], 255];
                ctx.scene.sync_color(i);
            }
            materials_changed(ctx);
            Ok(vec![text(format!("Set {} palette colors", list.len()))])
        }
        "load_palette" => {
            let path = match (string(a, "path"), int(a, "page")) {
                (Some(p), _) => PathBuf::from(p),
                (None, Some(k)) => ctx.ed.palettes.get(k as usize).cloned().ok_or_else(|| anyhow!("no palette page {k}; {} available", ctx.ed.palettes.len()))?,
                _ => bail!("give `path` or `page`"),
            };
            let colors = app::load_palette(&path)?;
            for (k, c) in colors.into_iter().enumerate() {
                ctx.scene.palette[k + 1] = c;
                ctx.scene.sync_color(k + 1);
            }
            materials_changed(ctx);
            Ok(vec![text(format!("Loaded palette {}", path.display()))])
        }
        "set_material" => {
            let i = color_arg(ctx.scene, a, "color")? as usize;
            if i == 0 {
                bail!("color 0 is empty space");
            }
            let m = &mut ctx.scene.materials[i];
            if let Some(t) = string(a, "type") {
                m.kind = match t {
                    "diffuse" => MaterialKind::Diffuse,
                    "metal" => MaterialKind::Metal,
                    "glass" => MaterialKind::Glass,
                    "emit" => MaterialKind::Emit,
                    "blend" => MaterialKind::Blend,
                    "cloud" => MaterialKind::Media,
                    t => bail!("unknown material type {t:?}"),
                } as u32;
                if m.kind == MaterialKind::Emit as u32 && m.emit <= 0.0 {
                    m.emit = 1.0;
                }
                if m.kind == MaterialKind::Glass as u32 && m.trans <= 0.0 {
                    m.trans = 1.0;
                }
            }
            if let Some(v) = num(a, "roughness") {
                m.rough = v.clamp(0.0, 1.0);
            }
            if let Some(v) = num(a, "metallic") {
                m.metal = v.clamp(0.0, 1.0);
            }
            if let Some(v) = num(a, "ior") {
                m.ior = v.max(1.0);
            }
            if let Some(v) = num(a, "transparency") {
                m.trans = v.clamp(0.0, 1.0);
            }
            if let Some(v) = num(a, "emission") {
                m.emit = v.max(0.0);
            }
            if let Some(v) = num(a, "density") {
                m.density = v.max(0.0);
            }
            materials_changed(ctx);
            Ok(vec![text(format!("Set material of color {i}"))])
        }
        "select" => {
            let mode = match string(a, "combine").unwrap_or("replace") {
                "add" => SelMode::Add,
                "remove" => SelMode::Remove,
                _ => SelMode::Replace,
            };
            let v = &ctx.scene.volume;
            match string(a, "mode").context("missing `mode`")? {
                "all" => ctx.ed.editor.select_all(v),
                "none" => ctx.ed.editor.select_none(),
                "invert" => ctx.ed.editor.select_invert(v),
                "box" => {
                    let (from, to) = (ivec3(a, "from")?, ivec3(a, "to")?);
                    let (lo, hi) = (from.min(to), from.max(to));
                    let cells: Vec<usize> = (0..v.cells.len()).filter(|&i| {
                        let p = v.position(i);
                        p.cmpge(lo).all() && p.cmple(hi).all()
                    }).collect();
                    ctx.ed.editor.select(v, cells, mode);
                }
                "color" => {
                    let c = color_arg(ctx.scene, a, "color")?;
                    let cells: Vec<usize> = (0..v.cells.len()).filter(|&i| v.cells[i] == c).collect();
                    ctx.ed.editor.select(v, cells, mode);
                }
                "region" => {
                    let at = ivec3(a, "at")?;
                    let o = ops::RegionOptions { same_color: false, ..Default::default() };
                    let cells = ops::region(v, at, o);
                    ctx.ed.editor.select(v, cells, mode);
                }
                m => bail!("unknown mode {m:?}"),
            }
            refresh_selection(ctx);
            Ok(vec![text(format!("{} voxels selected", ctx.ed.editor.selected_count()))])
        }
        "move_selection" => {
            let offset = ivec3(a, "offset")?;
            let v = &ctx.scene.volume;
            let sel = ctx.ed.editor.selection().map(<[bool]>::to_vec);
            let moving: Vec<(IVec3, u8)> =
                v.cells.iter().enumerate().filter(|&(i, &c)| c != 0 && sel.as_ref().is_none_or(|s| s[i])).map(|(i, &c)| (v.position(i), c)).collect();
            let mut new = v.clone();
            for &(p, _) in &moving {
                new.set(p, 0);
            }
            let mut moved = Vec::new();
            for &(p, c) in &moving {
                let q = p + offset;
                if new.contains(q) {
                    let i = new.index(q);
                    new.set_index(i, c);
                    moved.push(i);
                }
            }
            new.refresh_bricks();
            let lost = moving.len() - moved.len();
            let changed = apply(ctx, new);
            if sel.is_some() {
                ctx.ed.editor.select(&ctx.scene.volume, moved, SelMode::Replace);
                refresh_selection(ctx);
            }
            let mut out = edited(ctx, changed, &format!("Moved {} voxels", moving.len()));
            if lost > 0 {
                out.push(text(format!("{lost} voxels moved off the canvas and were removed.")));
            }
            Ok(out)
        }
        "undo" | "redo" => {
            let steps = int(a, "steps").unwrap_or(1).max(1);
            let mut done = 0;
            for _ in 0..steps {
                let ok = app::history_step(ctx.scene, ctx.ed, name == "redo");
                if !ok {
                    break;
                }
                done += 1;
            }
            if done > 0 {
                app::sync_volume(ctx.r, ctx.scene, &mut ctx.ed.editor);
            }
            Ok(edited(ctx, done > 0, &format!("{name} x{done}")))
        }
        "set_camera" => {
            let r = &mut *ctx.r;
            if boolean(a, "frame").unwrap_or(false) {
                r.frame(ctx.scene.volume.dims.as_vec3());
            }
            let mut cam: Camera = r.camera;
            if let Some(v) = num(a, "yaw") {
                cam.yaw = v.to_radians();
            }
            if let Some(v) = num(a, "pitch") {
                cam.pitch = v.clamp(-89.0, 89.0).to_radians();
            }
            if let Some(v) = num(a, "distance") {
                cam.dist = v.max(0.5);
            }
            if let Some(t) = get(a, "target").and_then(Value::as_array).filter(|t| t.len() == 3) {
                let f = |i: usize| t[i].as_f64().unwrap_or(0.0) as f32;
                cam.target = Vec3::new(f(0), f(1), f(2));
            }
            r.set_camera(cam);
            if let Some(v) = num(a, "fov") {
                r.settings.fov = v.clamp(1.0, 170.0);
            }
            match string(a, "projection") {
                Some("perspective") => r.settings.projection = Projection::Perspective,
                Some("orthographic") => r.settings.projection = Projection::Orthographic,
                Some("isometric") => app::iso_camera(r),
                Some("stereographic") => r.settings.projection = Projection::Stereographic,
                Some("panorama") => r.settings.projection = Projection::Panorama,
                Some(p) => bail!("unknown projection {p:?}"),
                None => {}
            }
            r.reset();
            let cam = &r.camera;
            Ok(vec![text(format!(
                "Camera yaw {:.1}, pitch {:.1}, distance {:.1}, target [{:.1}, {:.1}, {:.1}]",
                cam.yaw.to_degrees(),
                cam.pitch.to_degrees(),
                cam.dist,
                cam.target.x,
                cam.target.y,
                cam.target.z
            ))])
        }
        "set_render_settings" => {
            let s = &mut ctx.r.settings;
            let mut set = Vec::new();
            macro_rules! f32_field {
                ($key:literal, $field:ident) => {
                    if let Some(v) = num(a, $key) {
                        s.$field = v;
                        set.push($key);
                    }
                };
            }
            macro_rules! bool_field {
                ($key:literal, $field:ident) => {
                    if let Some(v) = boolean(a, $key) {
                        s.$field = v;
                        set.push($key);
                    }
                };
            }
            macro_rules! color_field {
                ($key:literal, $field:ident) => {
                    if let Some(v) = string(a, $key) {
                        s.$field = hex_linear(v)?;
                        set.push($key);
                    }
                };
            }
            f32_field!("sun_pitch", sun_pitch);
            f32_field!("sun_yaw", sun_yaw);
            f32_field!("sun_intensity", sun_intensity);
            f32_field!("sun_area", sun_area);
            color_field!("sun_color", sun_color);
            f32_field!("sky_intensity", sky_intensity);
            color_field!("sky_color", sky_color);
            f32_field!("exposure", exposure);
            f32_field!("gamma", gamma);
            bool_field!("aces", aces);
            bool_field!("bloom", bloom);
            bool_field!("denoise", denoise);
            f32_field!("fog_density", fog_density);
            color_field!("fog_color", fog_color);
            bool_field!("ground", ground);
            color_field!("ground_color", ground_color);
            color_field!("background_color", bg_color);
            bool_field!("edges", show_edges);
            bool_field!("grid", show_grid);
            f32_field!("aperture", aperture);
            if let Some(v) = string(a, "sky") {
                s.env_mode = match v {
                    "uniform" => crate::env::EnvMode::Uniform,
                    "atmosphere" => crate::env::EnvMode::Atmosphere,
                    "image" => crate::env::EnvMode::Image,
                    v => bail!("unknown sky {v:?}"),
                };
                set.push("sky");
            }
            if let Some(v) = string(a, "background") {
                s.bg_mode = match v {
                    "sky" => BgMode::Sky,
                    "color" => BgMode::Color,
                    "transparent" => BgMode::Transparent,
                    v => bail!("unknown background {v:?}"),
                };
                set.push("background");
            }
            ctx.r.reset();
            Ok(vec![text(if set.is_empty() { "No settings given".to_string() } else { format!("Set {}", set.join(", ")) })])
        }
        "screenshot" => {
            let (w, h) = (int(a, "width").unwrap_or(800).clamp(16, 4096) as u32, int(a, "height").unwrap_or(600).clamp(16, 4096) as u32);
            let px = ctx.r.modeler_image(w, h)?;
            Ok(vec![image(w, h, &px)?])
        }
        "render" => {
            let (w, h) = (int(a, "width").unwrap_or(960).clamp(16, 8192) as u32, int(a, "height").unwrap_or(640).clamp(16, 8192) as u32);
            let spp = int(a, "samples").unwrap_or(128).clamp(1, 65536) as u32;
            let t = std::time::Instant::now();
            let px = render_pixels(ctx, w, h, spp)?;
            let mut out = vec![image(w, h, &px)?];
            let mut note = format!("Rendered {w}x{h} at {spp} spp in {:.1?}", t.elapsed());
            if let Some(p) = string(a, "save_path") {
                let path = Path::new(p);
                crate::save_png(path, w, h, &px)?;
                note.push_str(&format!("; saved {}", std::fs::canonicalize(path).unwrap_or(path.into()).display()));
            }
            out.push(text(note));
            Ok(out)
        }
        "export" => {
            let name = string(a, "format").context("missing `format`")?;
            let format = crate::export::Format::from_name(name).ok_or_else(|| anyhow!("unknown format {name:?}"))?;
            let path = match string(a, "path") {
                Some(p) => PathBuf::from(p),
                None => app::workspace_dir().join("export").join(app::export_file_name(&ctx.ed.name(), format)),
            };
            if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
                std::fs::create_dir_all(dir)?;
            }
            let files = crate::export::export(format, ctx.scene, &path)?;
            ctx.ed.export_status = Some(app::export_summary(&files));
            let mut out = vec![text(app::export_summary(&files))];
            // Image formats come back as images too, so the agent can look.
            if let Some(png) = files.iter().find(|f| f.extension().is_some_and(|e| e == "png") && matches!(format, crate::export::Format::Iso | crate::export::Format::TwoD | crate::export::Format::Slice)) {
                let data = std::fs::read(png)?;
                out.push(json!({ "type": "image", "data": base64::engine::general_purpose::STANDARD.encode(data), "mimeType": "image/png" }));
            }
            Ok(out)
        }
        "run_script" => {
            let (source, file) = match (string(a, "source"), string(a, "path")) {
                (Some(src), _) => (src.to_string(), "script.js".to_string()),
                (None, Some(p)) => (std::fs::read_to_string(p).with_context(|| format!("reading {p}"))?, p.to_string()),
                (None, None) => bail!("give JavaScript `source` or a `path`"),
            };
            let timeout = num(a, "timeout").map_or(crate::script::DEFAULT_TIMEOUT, |s| std::time::Duration::from_secs_f32(s.clamp(0.1, 3600.0)));
            let outcome = crate::script::run(ctx, source, file, get(a, "args").cloned().unwrap_or(Value::Null), timeout);
            Ok(vec![text(crate::script::report(&outcome)?)])
        }
        "set_view_mode" => {
            let modeling = match string(a, "mode").context("missing `mode`")? {
                "model" => true,
                "render" => false,
                m => bail!("unknown mode {m:?}"),
            };
            if ctx.r.modeler.is_some() {
                ctx.r.modeling = modeling;
                ctx.ed.enabled = modeling;
                ctx.r.reset();
            }
            Ok(vec![text(format!("View: {}", if modeling { "model" } else { "render" }))])
        }
        other => bail!("unknown tool {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_fill_expected_cells() {
        let v = Volume::new(IVec3::splat(10));
        let b = shape_volume(&v, "box", 2, IVec3::ZERO, IVec3::new(2, 2, 2), 1, false);
        assert_eq!(count(&b), 27);
        let h = shape_volume(&v, "box", 2, IVec3::ZERO, IVec3::new(2, 2, 2), 1, true);
        assert_eq!(count(&h), 26);
        let s = shape_volume(&v, "ellipsoid", 2, IVec3::ZERO, IVec3::splat(9), 1, false);
        assert!(count(&s) > 450 && count(&s) < 560, "{}", count(&s));
        let c = shape_volume(&v, "cone", 2, IVec3::ZERO, IVec3::splat(9), 1, false);
        assert!(c.get(IVec3::new(4, 4, 0)) == 1 && c.get(IVec3::new(0, 0, 9)) == 0);
    }

    #[test]
    fn tool_list_is_valid() {
        let t = tools();
        let names: Vec<&str> = t.as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"render") && names.contains(&"screenshot"));
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len());
    }
}
