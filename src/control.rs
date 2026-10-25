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
// isn't one of its tools) and, if it needs one, a `State` kept in
// `EditState::tools`.
mod build;
pub mod guide;
mod import;
pub(crate) mod inspect;
mod reference;
pub(crate) mod surface;
mod validate;
mod view;

/// Per-session state for the tool families, plus a record of the last edit.
#[derive(Default)]
pub struct ToolState {
    pub last_edit: Option<LastEdit>,
    pub build: build::State,
    pub inspect: inspect::State,
    pub reference: reference::State,
    pub surface: surface::State,
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
        let mut changes = old.cells.diff(&new.cells);
        changes.sort_unstable_by_key(|c| c.0);
        for (i, a, b) in changes {
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
Typical flow: new_project -> set_palette_colors -> draw with fill_box / draw_shape / draw_line / \
set_voxels -> look with screenshot (views \"quad\" shows front/side/top/3-4 with rulers) -> \
set_material and set_render_settings -> render (path traced) -> save_file. \
Every edit is undoable. If the voxely window is open, edits appear in it live.

House style: clean, designed voxel art. Chunky boxy forms, ONE flat color per material plus a few \
deliberate accents, and small details placed by hand (eyes with pupils and highlights, gems, buckles, \
trim, emblems). The renderer does the lighting, so never paint shading, noise or speckle. Designing \
from your head, a character is 32-48 voxels tall. With a picture, it is the design reference: read its \
parts, proportions, flat colors and recognizable details, then build them cleanly; if the picture is \
voxel art, match its block size (one picture block = one voxel; analyze_image measures \
it). get_guide (craft, character, image, ...) has more.

Scenes are a tree of parts (models) and groups. Voxel tools work on one part at a time, in that part's own coordinates (0..size-1): the part being edited (see get_scene_info), or the one named by a tool's `part` argument. Build separate pieces (body, head, arms, legs, gear, props) as separate parts with create_part, place them with transform_part and group_parts, inspect with list_parts. Pivots (set_pivot), sockets (set_socket) and transform_part attach_to make a skeleton for animation and glTF export. Part positions are world voxel coordinates of the part's minimum corner. run_script runs one JavaScript program that calls every tool (voxely.<camelCaseName>) for repetitive or exactly computed layouts.";

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
    json!({ "type": ["integer", "string"], "minimum": 0, "maximum": 255, "description": COLOR })
}

/// A canvas or part size: [x, y, z], 1 to `ops::MAX_DIM` each.
fn size3(desc: &str) -> Value {
    json!({ "type": "array", "items": { "type": "integer", "minimum": 1, "maximum": ops::MAX_DIM }, "minItems": 3, "maxItems": 3, "description": desc })
}

/// Every tool's definition (see `tools`).
fn base_tools() -> Value {
    let axis = json!({ "type": "string", "enum": ["x", "y", "z"] });
    json!([
        tool("get_scene_info", "Canvas size, voxel count and bounds, colors in use, file, selection, camera and mode.", json!({}), &[]),
        tool("new_project", "Create a named project in the workspace (it appears in the window's Project list) and make it current. Empty unless `fill` is given.", json!({
            "name": { "type": "string", "description": "Letters, digits, spaces, - and _" },
            "size": size3("Canvas size, default [40, 40, 40]; up to 1024 per axis"),
            "fill": color_schema(),
            "overwrite": { "type": "boolean", "description": "Replace an existing project with this name" },
        }), &["name"]),
        tool("list_projects", "The workspace's projects (newest first), MagicaVoxel's sample models, and (as `opened_folder`) the .vox files beside a file opened from another folder; open_project opens any of them by name.", json!({}), &[]),
        tool("open_project", "Open a project or sample by name.", json!({ "name": { "type": "string" } }), &["name"]),
        tool("new_scene", "Start a new unnamed project (saved as project, project-2, ... in the workspace). Empty unless `fill` is given.", json!({
            "size": size3("Canvas size, default [40, 40, 40]; up to 1024 per axis"),
            "fill": color_schema(),
        }), &[]),
        tool("open_file", "Open a MagicaVoxel .vox file, keeping its models and groups as parts.", json!({ "path": { "type": "string" } }), &["path"]),
        tool("save_file", "Save the project as .vox. Without `path`, saves to the current save path (get_scene_info `save_path`): the project's workspace file, or the last path saved to. With a `path` it's Save As: that file becomes the current one (later saves and autosave go there) and the project's references are copied to its new name.", json!({ "path": { "type": "string" } }), &[]),
        tool("set_voxels", "Set individual voxels. Color 0 erases.", json!({
            "voxels": { "type": "array", "description": "[[x, y, z, color], ...]; color is an index or \"#rrggbb\"", "items": { "type": "array", "items": { "type": ["integer", "string"] }, "minItems": 4, "maxItems": 4 } },
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
            "rotation": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3, "description": "[x, y, z] degrees about the box center, applied x then y then z, e.g. a tilted cylinder or a leaning cone; the shape may extend past from/to" },
            "hollow": { "type": "boolean", "description": "Only the outer shell" },
        }), &["shape", "from", "to", "color"]),
        tool("draw_line", "Draw a straight line of voxels between two points.", json!({
            "from": xyz(XYZ), "to": xyz(XYZ), "color": color_schema(),
            "thickness": { "type": "integer", "description": "Width in voxels, 1-64 (default 1): a round brush this many voxels across swept along the line" },
        }), &["from", "to", "color"]),
        tool("get_voxels", "List solid voxels, optionally within a box.", json!({
            "from": xyz(XYZ), "to": xyz(XYZ), "limit": { "type": "integer", "description": "Max voxels returned (default 5000)" },
        }), &[]),
        tool("get_slice", "A text map of one layer, or a range of layers: color indices, '.' for empty. For axis z, rows run from high y (top) to low y. Rows are labeled with their coordinate and a ruler labels every 5th column. Ranges print each layer under a header and collapse empty layers. Big canvases are downsampled (see max_cells); crop with from/to to zoom in. For an overview, get_projection shows the top surface in one map.", json!({
            "axis": axis.clone(), "index": { "type": "integer", "description": "The layer" },
            "from_index": { "type": "integer", "description": "First layer of a range (instead of index)" },
            "to_index": { "type": "integer", "description": "Last layer of a range, inclusive" },
            "from": xyz("Crop corner, inclusive (the axis coordinate is ignored)"), "to": xyz("Other crop corner"),
            "max_cells": { "type": "integer", "description": "Max map width/height before downsampling (default 128)" },
        }), &[]),
        tool("edit_model", "MagicaVoxel's whole-model tools. fill/clear apply to the selection if there is one.", json!({
            "op": { "type": "string", "enum": ["full", "fill", "clear", "reverse", "double", "half", "flip", "rotate", "fit", "loop", "mirror", "diagonal", "hull", "flood", "dilate", "erode", "maze", "perlin"],
                "description": "full: fill empty space; fill: recolor voxels; reverse: swap empty/solid; double/half: scale 2x/0.5x; flip/rotate(90 deg)/mirror(make symmetric)/diagonal/loop(shift 1 with wrap) need `axis`; fit: shrink canvas to model; hull: remove hidden voxels; flood: fill enclosed space; dilate/erode: grow/shrink 1; maze/perlin: generate" },
            "axis": axis.clone(),
            "direction": { "type": "integer", "description": "loop: -1 or 1" },
            "color": { "type": ["integer", "string"], "description": "full/fill/reverse/flood/maze/perlin: the color for this op only (1-255 or \"#rrggbb\"); default: the current color" },
        }), &["op"]),
        tool("transform_model", "Scale, shear, rotate by any angle, or repeat the model. The canvas grows to fit.", json!({
            "kind": { "type": "string", "enum": ["scale", "shear", "rotate", "repeat"] },
            "args": { "type": "string", "description": "Axes then values, like MagicaVoxel: scale \"xyz 2\", shear \"z 0.2 0.2\", rotate \"z 45\", repeat \"xy 2\"" },
        }), &["kind", "args"]),
        tool("resize_canvas", "Change the canvas size (1-1024 per axis). Voxels stay where they are in the world (the part's position shifts to make up for it). The canvas stays centered in x/y on the same floor, or centered on the model where it wouldn't fit; voxels that still don't fit are cut off and the reply says how many.", json!({ "size": size3("New size") }), &["size"]),
        tool("list_shaders", "MagicaVoxel voxel shaders available to run_shader, with their arguments.", json!({}), &[]),
        tool("run_shader", "Run a MagicaVoxel voxel shader (GLSL `float map(vec3 v)` returning a color index) over the model, or the selection.", json!({
            "name": { "type": "string", "description": "A shader from list_shaders" },
            "source": { "type": "string", "description": "Or inline GLSL defining float map(vec3 v); built-ins: i_volume_size, i_color_index, voxel(v), palette(i), color_sel(k)" },
            "args": { "type": ["array", "object"], "items": { "type": "number" }, "description": "Argument values: an array in order, or an object by arg var name" },
            "color": { "type": ["integer", "string"], "description": "The shader's i_color_index for this run (1-255 or \"#rrggbb\"); default: the current color" },
        }), &[]),
        tool("get_palette", "Palette colors as hex with their materials. Default: only colors in use.", json!({ "all": { "type": "boolean" } }), &[]),
        tool("set_palette_colors", "Set exact palette colors.", json!({
            "colors": { "type": "array", "items": { "type": "object", "properties": { "index": { "type": "integer", "minimum": 1, "maximum": 255 }, "color": { "type": "string" } }, "required": ["index", "color"] }, "description": "[{\"index\": 1, \"color\": \"#ff8800\"}, ...]" },
        }), &["colors"]),
        tool("load_palette", "Load a whole palette into colors 1.. : a palette file by path (MagicaVoxel 256x1 PNG, Lospec PNG, .act, .gpl or .hex), a MagicaVoxel palette page 0-3, or a preset name. With match: true the model's colors are recolored to the closest new colors (each keeps its own slot, so loading the old palette again restores them exactly); otherwise voxels keep their indices and take the new colors.", json!({
            "path": { "type": "string" }, "page": { "type": "integer" },
            "preset": { "type": "string", "description": "A name from list_palette_presets" },
            "match": { "type": "boolean", "description": "Recolor the model to the nearest new colors, reversibly (default false)" },
        }), &[]),
        tool("edit_palette", "Rearrange or tidy the palette. swap / move (from, to): colors change places and voxels follow, so the model looks the same; copy (from, to): copy a color and its material onto another slot; gradient (from, to): blend the slots between two colors evenly; sort (by: hue | saturation | value): used colors first; pack: used colors into 1.. with duplicates merged; save (name): save the palette to the workspace's palettes folder; name_ramp (row, name): name a row of 8 colors (row k holds 8k+1..8k+8; saved in the .vox); undo: undo the last change if it is a palette edit. Palette edits are steps of the normal undo too.", json!({
            "action": { "type": "string", "enum": ["swap", "move", "copy", "gradient", "sort", "pack", "save", "name_ramp", "undo"] },
            "from": { "type": "integer", "minimum": 1, "maximum": 255 }, "to": { "type": "integer", "minimum": 1, "maximum": 255 },
            "by": { "type": "string", "enum": ["hue", "saturation", "value"] },
            "name": { "type": "string" }, "row": { "type": "integer", "minimum": 0, "maximum": 31 },
        }), &["action"]),
        tool("set_material", "Set the render material of a palette color. Each type takes its own parameters: metal: roughness, metallic, ior; glass: roughness, ior, transparency; blend: roughness, metallic, ior, transparency; emit: emission; cloud: density; diffuse: none. A parameter the type doesn't use is an error. get_palette reads them back.", json!({
            "color": color_schema(),
            "type": { "type": "string", "enum": ["diffuse", "metal", "glass", "emit", "blend", "cloud"] },
            "roughness": { "type": "number", "minimum": 0, "maximum": 1, "description": "0 mirror-smooth, 1 matte; metal looks good at 0.2-0.5" },
            "metallic": { "type": "number", "minimum": 0, "maximum": 1 },
            "ior": { "type": "number", "minimum": 1, "maximum": 3, "description": "Index of refraction: water 1.33, glass 1.5, diamond 2.4" },
            "transparency": { "type": "number", "minimum": 0, "maximum": 1, "description": "Glass: 0.7-0.95" },
            "emission": { "type": "number", "minimum": 0, "description": "Emit strength: 1-3 soft glow, 5-20 lamps" },
            "density": { "type": "number", "minimum": 0, "description": "Cloud density" },
        }), &["color"]),
        tool("select", "Change the selection, which limits fill/clear/run_shader/move_selection.", json!({
            "mode": { "type": "string", "enum": ["all", "none", "invert", "box", "color", "region"],
                "description": "box: solid voxels between from/to; color: every voxel of `color`; region: voxels of the same color connected to `at`" },
            "from": xyz(XYZ), "to": xyz(XYZ), "at": xyz(XYZ), "color": color_schema(),
            "combine": { "type": "string", "enum": ["replace", "add", "remove"] },
        }), &["mode"]),
        tool("move_selection", "Move the selected voxels (or the whole model if none) by an offset.", json!({ "offset": xyz("[dx, dy, dz]") }), &["offset"]),
        tool("undo", "Undo edits.", json!({ "steps": { "type": "integer", "minimum": 1 } }), &[]),
        tool("redo", "Redo edits.", json!({ "steps": { "type": "integer", "minimum": 1 } }), &[]),
        tool("set_camera", "Aim the camera (used by screenshot, render and the window). Angles in degrees.", json!({
            "frame": { "type": "boolean", "description": "Reset to a three-quarter view of the whole canvas first" },
            "yaw": { "type": "number", "description": "Orbit angle around z in degrees; 35-45 gives a three-quarter view" },
            "pitch": { "type": "number", "minimum": -90, "maximum": 90, "description": "Elevation in degrees; 25-35 shows the top and two sides, 90 looks straight down" },
            "distance": { "type": "number", "exclusiveMinimum": 0, "description": "From the target, in voxels" },
            "target": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3, "description": "[x, y, z] world point to orbit around (the same coordinates render, screenshot and get_scene_info report)" },
            "projection": { "type": "string", "enum": ["perspective", "orthographic", "isometric", "stereographic", "panorama"] },
            "fov": { "type": "number", "minimum": 1, "maximum": 170, "description": "Field of view in degrees" },
        }), &[]),
        tool("set_render_settings", "Lighting, sky, film and display settings for render. Colors are \"#rrggbb\".", json!({
            "sun_pitch": { "type": "number", "minimum": -90, "maximum": 90, "description": "Sun elevation in degrees; 30-50 gives readable shadows" },
            "sun_yaw": { "type": "number", "description": "Sun direction around z in degrees" },
            "sun_intensity": { "type": "number", "minimum": 0 },
            "sun_area": { "type": "number", "minimum": 0, "description": "Larger is softer shadows" },
            "sun_color": { "type": "string" },
            "sky": { "type": "string", "enum": ["uniform", "atmosphere", "image"] },
            "sky_intensity": { "type": "number", "minimum": 0 }, "sky_color": { "type": "string" },
            "exposure": { "type": "number", "minimum": 0, "description": "Brightness multiplier, 1 = neutral" },
            "gamma": { "type": "number", "exclusiveMinimum": 0, "description": "Usually 2.2" },
            "aces": { "type": "boolean", "description": "Filmic tone mapping" },
            "bloom": { "type": "boolean" }, "denoise": { "type": "boolean" },
            "fog_density": { "type": "number", "minimum": 0 }, "fog_color": { "type": "string" },
            "ground": { "type": "boolean" }, "ground_color": { "type": "string" },
            "background": { "type": "string", "enum": ["sky", "color", "transparent"] }, "background_color": { "type": "string" },
            "edges": { "type": "boolean" }, "grid": { "type": "boolean" },
            "aperture": { "type": "number", "minimum": 0, "description": "Depth of field, 0 = off" },
        }), &[]),
        tool("render", "Path-traced image with the current render settings. By default from the window's camera exactly as it is. \
            frame: true keeps the camera's yaw and pitch but fits distance and target to all the voxels with a margin (a good default for a first look). \
            views path traces several auto-framed views tiled side by side with labels, same lighting: \"turnaround\" (front, right, back, left), \
            \"quad\", \"all\", or a list of front, back, left, right, top, bottom, iso, 3/4. Named views are orthographic except 3/4 unless projection says otherwise. \
            Neither changes the window's camera; the result lists the cameras used.", json!({
            "width": { "type": "integer", "minimum": 16, "maximum": 8192, "description": "Whole image width (default 960; with views 480 per tile, at least 16 per tile)" },
            "height": { "type": "integer", "minimum": 16, "maximum": 8192, "description": "Whole image height (default 640; with views 600 per tile, at least 16 per tile)" },
            "samples": { "type": "integer", "minimum": 1, "maximum": 65536, "description": "Samples per pixel (default 128; more is smoother and slower)" },
            "frame": { "type": "boolean", "description": "Fit the camera to the whole scene from its current angle" },
            "views": { "type": ["string", "array"], "items": { "type": "string" }, "description": "\"turnaround\", \"quad\", \"all\", a view name, or a list of view names" },
            "projection": { "type": "string", "enum": ["orthographic", "perspective"], "description": "For frame/views (default: each view's own; frame keeps the current projection)" },
            "save_path": { "type": "string", "description": "Also save the PNG here (the whole tiled image with views)" },
        }), &[]),
        tool("export", "Export the model: gltf (.glb, best for game engines), or MagicaVoxel's formats: obj/ply/mc/cube meshes, xraw/slab/qb/vox volumes, iso/2d/slice sprite images, point clouds.", json!({
            "format": { "type": "string", "enum": ["gltf", "obj", "ply", "mc", "xraw", "slab", "qb", "vox", "iso", "2d", "slice", "point", "cube"],
                "description": "gltf: a .glb for game engines (Godot, Unity, Unreal, Three.js, Bevy) with metal/roughness, emission and glass materials; use a .gltf path for JSON" },
            "path": { "type": "string", "description": "Output file (default: <workspace>/export/<project>.<ext>)" },
        }), &["format"]),
        tool("run_script", "Run a JavaScript program (QuickJS) inside voxely: one call instead of many. Every tool is voxely.<camelCase name>(args) with the same arguments, e.g. voxely.fillBox({from: [0, 0, 0], to: [9, 9, 0], color: \"#88aa44\"}); it returns parsed JSON or text and throws on errors. For bulk edits, v = voxely.getVolume({part}) gives a part's grid in its own coordinates (default: the part being edited): v.size and v.cells (a Uint8Array of palette indices, x fastest, then y, then z) with v.get(x, y, z), v.set(x, y, z, c) and v.forEach(fn); voxely.setVolume(v, {part}) writes it back (new Volume([x, y, z]) starts empty and resizes the part). voxely.color(\"#rrggbb\") gives the nearest palette index. Returns console.log output and the last expression's value. The whole script is one undo step. Scripts reach files only through the tools (saveFile, openFile, importFile, export, set_reference ...), with the same access those tools have; there is no other file, network or process access.", json!({
            "source": { "type": "string", "description": "JavaScript source" },
            "path": { "type": "string", "description": "Or a .js file to run" },
            "args": { "description": "Any JSON value, available to the script as `args`" },
            "timeout": { "type": "number", "minimum": 0.1, "maximum": 3600, "description": "Seconds before the script is stopped, 0.1-3600 (default 120)" },
        }), &[]),
        tool("set_view_mode", "Show the modeler or the live path tracer in the window.", json!({
            "mode": { "type": "string", "enum": ["model", "render"] },
        }), &["mode"]),
        tool("list_parts", "The scene tree: every part (model) and group with its id, name, position (world min corner), size and children.", json!({}), &[]),
        tool("edit_part", "Edit a part's voxels (the voxel tools then work in its coordinates), or pass \"world\" for World mode, where the window shows every part and you arrange them.", json!({
            "part": { "type": ["string", "integer"], "description": "Part name or id, or \"world\"" },
        }), &["part"]),
        tool("create_part", "Add a new empty part (a separate model) and start editing it.", json!({
            "name": { "type": "string", "description": "A taken name gets a number (\"arm 2\")" },
            "size": size3("Size in voxels, up to 256 per axis for MagicaVoxel compatibility"),
            "position": xyz("World position of its minimum corner (default: beside the existing parts, on the ground)"),
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
            "free_rotation": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3, "description": "[x, y, z] degrees about the pivot and world axes (like rotate_axis), applied x then y then z; replaces any previous free rotation. [0, 0, 0] straightens." },
            "attach_to": { "type": "string", "description": "\"part:socket\" (e.g. \"body:shoulder_r\") to snap onto a socket and attach, or \"part\" to attach where it is. Attached parts follow their parent's moves and turns (and nest in glTF exports)." },
            "link": { "type": "boolean", "description": "With attach_to: false only lines it up without attaching (default true)" },
            "frame": { "type": "integer", "description": "Go to this animation frame first (the change is keyed there when animating)" },
            "attach_socket": { "type": "string", "description": "With attach_to: this part's own socket to line up (default: its pivot). It becomes the pivot if none was set, so later turns keep it attached." },
        }), &["part"]),
        tool("set_animation", "Set up the animation timeline. Frames are numbered from 0. With more than one frame, moving or turning a part (transform_part, or set_keyframe) records a keyframe on the current frame; the first key off frame 0 also keys frame 0 with where the part was. Attached parts follow their parent, so animate joints from the root down.", json!({
            "frames": { "type": "integer", "minimum": 1, "maximum": 10000, "description": "Number of frames, e.g. 24" },
            "fps": { "type": "number", "minimum": 0.5, "maximum": 120, "description": "Playback speed, e.g. 12" },
            "loop": { "type": "boolean", "description": "Loop playback; with smooth keys the last key blends back into the first" },
            "smooth": { "type": "boolean", "description": "Blend between keys (default) or hold each key until the next, like MagicaVoxel" },
        }), &[]),
        tool("go_to_frame", "Show a frame: the window, screenshot, render and the voxel tools all use it. A part with flipbook drawings shows (and edits) the drawing for this frame.", json!({
            "frame": { "type": "integer" },
        }), &["frame"]),
        tool("set_keyframe", "Key a part's pose on a frame: goes to the frame, applies any of transform_part's changes (move, position, rotate_axis + quarter_turns, flip_axis, free_rotation, attach_to), then records the key. Free rotations are absolute, about the part's pivot and world axes.", json!({
            "part": { "type": ["string", "integer"] },
            "frame": { "type": "integer" },
            "move": xyz("[dx, dy, dz] offset"),
            "position": xyz("World position of its minimum corner"),
            "rotate_axis": axis,
            "quarter_turns": { "type": "integer" },
            "flip_axis": axis,
            "free_rotation": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3, "description": "[x, y, z] degrees" },
            "attach_to": { "type": "string", "description": "As in transform_part: \"part:socket\" or \"part\"" },
            "attach_socket": { "type": "string", "description": "With attach_to: this part's own socket to line up" },
            "link": { "type": "boolean", "description": "With attach_to: false only lines it up (default true)" },
        }), &["part", "frame"]),
        tool("delete_keyframe", "Remove a part's keyframe on a frame (also one past the timeline's end, which doesn't play).", json!({
            "part": { "type": ["string", "integer"] }, "frame": { "type": "integer" },
        }), &["part", "frame"]),
        tool("new_drawing", "Flipbook animation: from `frame` on (until its next drawing) the part shows its own copy of its voxels, which the voxel tools then edit while on that frame. Copy, then change only what moves (a blink, a lifted foot).", json!({
            "part": { "type": ["string", "integer"] }, "frame": { "type": "integer" },
        }), &["part", "frame"]),
        tool("remove_drawing", "Remove a part's flipbook drawing that starts on `frame` (the previous drawing shows instead).", json!({
            "part": { "type": ["string", "integer"] }, "frame": { "type": "integer" },
        }), &["part", "frame"]),
        tool("play", "Play or stop the animation in the voxely window, so the person can watch it (it plays at the timeline's fps, looping if set). Use frame_strip to check motion yourself.", json!({
            "playing": { "type": "boolean", "description": "true to play (default), false to stop" },
            "from_frame": { "type": "integer", "description": "Start from this frame (default: where it is)" },
        }), &[]),
        tool("frame_strip", "See the motion: modeler images of several frames side by side in one picture, left to right then top to bottom, from the current camera.", json!({
            "frames": { "type": "array", "items": { "type": "integer", "minimum": 0 }, "minItems": 1, "maxItems": 64, "description": "Which frames (default: up to 12 spread over the animation)" },
            "width": { "type": "integer", "minimum": 32, "maximum": 1024, "description": "Each frame's width (default 240)" },
            "height": { "type": "integer", "minimum": 32, "maximum": 1024, "description": "Each frame's height (default 240)" },
            "columns": { "type": "integer", "minimum": 1, "description": "Frames per row (default 6; at most the number of frames)" },
        }), &[]),
        tool("create_rig", "Add a ready-to-animate character made of blocks: parts with joints (pivots) attached in a chain, named so apply_motion and animation imports find them. humanoid: hips (root), torso, head, upper_arm_l/r, lower_arm_l/r (hand_l/r sockets), upper_leg_l/r, lower_leg_l/r. quadruped: body (root), head, tail, leg_front_l/r, leg_back_l/r. Facing -y, left is +x. Then redraw each part (voxel tools with `part`) to make it your character; keep each part's joint end where it is.", json!({
            "template": { "type": "string", "enum": ["humanoid", "quadruped"] },
            "prefix": { "type": "string", "description": "Prepended to part names, e.g. \"knight_\" (to have several rigs)" },
            "scale": { "type": "integer", "description": "Voxels per template unit, default 1 (a humanoid is 26 tall at 1)" },
            "position": xyz("World position of its feet (default: beside the existing parts)"),
        }), &["template"]),
        tool("apply_motion", "Key a ready-made motion onto a rig (parts found by name: create_rig's names, or Mixamo / Unreal / Blender style like LeftArm, upperarm_l, thigh.L). Replaces those parts' keys, starting from their pose before any motion (applying again replaces it, never adds to it). Sets the timeline to the motion's length; keeps its loop and smooth settings. Motions: idle, walk, run, jump, attack, wave, death, quad_walk, quad_idle. Check with frame_strip, then adjust with set_keyframe.", json!({
            "motion": { "type": "string", "enum": ["idle", "walk", "run", "jump", "attack", "wave", "death", "quad_walk", "quad_idle"] },
            "frames": { "type": "integer", "description": "Length of the motion in frames, 2-10000; the timeline becomes this long (default: the timeline's length, or 24)" },
            "amount": { "type": "number", "description": "Scale the motion, e.g. 0.5 for subtle, 1.5 for exaggerated (default 1)" },
            "rig": { "type": ["string", "integer"], "description": "Any part of the rig to animate when there are several (default: the first rig with the right limbs)" },
            "fps": { "type": "number", "description": "Playback speed, 0.5-120" },
        }), &["motion"]),
        tool("import_animation", "Retarget an animation clip from a glTF file (.glb/.gltf) onto a rig: free CC0 libraries such as Quaternius's Universal Animation Library, Blender exports, or voxely's own exports. Joints are matched to parts by name (hips, spine, head, upper arms, forearms, thighs, shins, in voxely / Mixamo / Unreal / Rigify naming); each part turns rigidly like its joint, with rest poses lined up (T-pose clips work on arms-down rigs). Without `clip`, lists the file's clips.", json!({
            "path": { "type": "string" },
            "clip": { "type": ["string", "integer"], "description": "Clip name or index" },
            "rig": { "type": ["string", "integer"], "description": "Any part of the rig to animate (default: the first rig with legs)" },
            "root_motion": { "type": "boolean", "description": "Let the hips travel and bob with the clip (default true; false keeps them in place)" },
        }), &["path"]),
        tool("set_pivot", "Set the point a part or group turns about (and its origin in glTF exports), in the part's own coordinates (the ones the voxel tools use while editing it; world coordinates for groups). E.g. a shoulder for an arm, the hinge for a door.", json!({
            "part": { "type": ["string", "integer"] },
            "pivot": { "type": ["array", "string"], "items": { "type": "number" }, "minItems": 3, "maxItems": 3, "description": "[x, y, z], or \"center\" (default), or \"bottom\" (middle of the bottom face)" },
        }), &["part", "pivot"]),
        tool("set_socket", "Add, move or remove a named attachment point (socket) on a part, in its own coordinates. Other parts attach to sockets with transform_part attach_to; glTF exports them as empty nodes engines can parent things to (hand, muzzle, seat ...).", json!({
            "part": { "type": ["string", "integer"] },
            "name": { "type": "string" },
            "position": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3, "description": "[x, y, z] in the part's own coordinates; omit to remove the socket" },
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
    [PART_TOOLS, build::PART_TOOLS, guide::PART_TOOLS, import::PART_TOOLS, inspect::PART_TOOLS, reference::PART_TOOLS, surface::PART_TOOLS, view::PART_TOOLS].iter().any(|l| l.contains(&name))
}

/// Tools that pull agents away from the house style (noise, painted
/// shading, reference colors copied with their lighting): not offered, but
/// still callable by name for anyone who wants that look.
const UNLISTED: &[&str] = &["vary_colors", "shade_edges", "paint_from_reference"];

/// The tool list for MCP `tools/list`, with the shared `part` argument.
pub fn tools() -> Value {
    let mut list = all_tools();
    list.as_array_mut().unwrap().retain(|t| !UNLISTED.contains(&t["name"].as_str().unwrap_or_default()));
    list
}

/// `tools`, unlisted tools included.
fn all_tools() -> Value {
    let mut list = base_tools();
    let extra = [build::tools(), guide::tools(), import::tools(), inspect::tools(), reference::tools(), surface::tools(), view::tools()];
    list.as_array_mut().unwrap().extend(extra.into_iter().flatten());
    for t in list.as_array_mut().into_iter().flatten() {
        if is_part_tool(t["name"].as_str().unwrap_or_default()) {
            t["inputSchema"]["properties"]["part"] = json!({ "type": ["string", "integer"], "description": "Part to work on (name or id); default: the part being edited" });
        }
    }
    list
}

/// Run one tool and build the MCP `tools/call` result. Arguments that don't
/// fit the tool's schema are rejected before anything runs.
pub fn call(ctx: &mut Ctx, name: &str, args: &Value) -> Value {
    let args = match validate::check_call(name, args) {
        Ok(a) => a,
        Err(e) => return json!({ "content": [text(format!("{e:#}"))], "isError": true }),
    };
    match dispatch(ctx, name, &args) {
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

/// Integer `key` (or `default`), which must be `lo..=hi`.
pub(crate) fn int_in(args: &Value, key: &str, default: i64, lo: i64, hi: i64) -> Result<i64> {
    let v = int(args, key).unwrap_or(default);
    if !(lo..=hi).contains(&v) {
        bail!("`{key}` must be {lo} to {hi}, not {v}");
    }
    Ok(v)
}

/// The largest image side the GPU renders.
pub(crate) fn max_image_side(ctx: &Ctx) -> i64 {
    ctx.r.device.limits().max_texture_dimension_2d.min(8192) as i64
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

/// Replace the edited part's canvas with `new` (cell 0 at old cell `at`)
/// without moving anything in the world; one undo step.
fn recanvas(ctx: &mut Ctx, new: Volume, at: IVec3) {
    let edit = LastEdit { part: ctx.scene.editing(), resized: true, ..Default::default() };
    ctx.scene.recanvas(&new, at);
    tree_changed(ctx);
    ctx.ed.tools.last_edit = Some(edit);
    inspect::after_commit(ctx);
}

/// edit_model fit: crop the canvas to the voxels, keeping them in place.
fn fit_part(ctx: &mut Ctx) -> Result<Vec<Value>> {
    let (new, lo) = ops::crop(&ctx.scene.volume).ok_or_else(|| anyhow!("fit isn't possible here: the part is empty"))?;
    if new.dims == ctx.scene.volume.dims {
        return Ok(edited(ctx, false, "Applied fit"));
    }
    let new = build::before_commit(ctx, new);
    let before = ctx.scene.origin;
    recanvas(ctx, new, lo);
    let d = ctx.scene.volume.dims;
    let note = format!(
        "Applied fit: canvas is now {}x{}x{}; its position moved {:?} -> {:?} so the voxels stay where they were (part coordinates shifted by {:?}; pivot and sockets follow)",
        d.x,
        d.y,
        d.z,
        before.to_array(),
        ctx.scene.origin.to_array(),
        (-lo).to_array()
    );
    Ok(edited(ctx, true, &note))
}

fn refresh_selection(ctx: &mut Ctx) {
    if ctx.ed.editor.sel_dirty {
        ctx.r.upload_selection(&ctx.scene.volume, ctx.ed.editor.selection());
        ctx.ed.editor.sel_dirty = false;
    }
}

/// Run a palette-wide edit and redraw what it changed.
fn palette_edit(ctx: &mut Ctx, cmd: crate::palette_lib::Cmd) -> Result<Vec<Value>> {
    let (msg, voxels) = crate::palette_lib::apply(cmd, ctx.r, ctx.scene, ctx.ed)?;
    if voxels {
        app::sync_volume(ctx.r, ctx.scene, &mut ctx.ed.editor);
    }
    Ok(vec![text(msg)])
}

fn materials_changed(ctx: &mut Ctx) {
    ctx.r.upload_materials(&ctx.scene.materials);
    ctx.r.set_palette(&ctx.scene.palette);
    ctx.scene.rebuild_lights();
    ctx.r.set_lights(&ctx.scene.lights, ctx.scene.light_power);
}

fn count(v: &Volume) -> usize {
    v.count()
}

fn bounds(v: &Volume) -> Option<(IVec3, IVec3)> {
    let mut b: Option<(IVec3, IVec3)> = None;
    for (p, _) in v.filled() {
        b = Some(b.map_or((p, p), |(lo, hi)| (lo.min(p), hi.max(p))));
    }
    b
}

fn arr(p: IVec3) -> [i32; 3] {
    p.to_array()
}

/// The report after an edit: voxels added/removed/recolored and where (when
/// this call committed one), the part's size and count, and the unsaved
/// count or an autosave once due (see `inspect::report`).
pub(crate) fn edited(ctx: &Ctx, changed: bool, what: &str) -> Vec<Value> {
    let why = if changed {
        None
    } else if what.starts_with("undo") || what.starts_with("redo") {
        Some(format!("nothing to {}", &what[..4]))
    } else if ctx.ed.editor.selection().is_some_and(|s| !s.contains(&true)) {
        Some("the selection is empty; select mode none clears it".to_string())
    } else {
        None
    };
    inspect::report(ctx, changed, what, why, Vec::new())
}

/// `edited` for fill_box / draw_shape / draw_line (`shape` "line"), which
/// also warns when the shape spilled past the canvas.
fn edited_shape(ctx: &Ctx, changed: bool, what: &str, shape: &str, args: &Value) -> Vec<Value> {
    inspect::shape_report(ctx, changed, what, shape, args)
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
        v["free_rotation"] = json!(w.tilt_world(id).to_array());
    }
    if !n.sockets.is_empty() {
        v["sockets"] = n
            .sockets
            .iter()
            .map(|s| json!({ "name": s.name, "position": w.to_part(id, s.pos).to_array(), "world": w.affine(id).transform_point3(s.pos).to_array() }))
            .collect();
    }
    if !n.keys.is_empty() {
        v["keyframes"] = json!(n.keys.iter().map(|k| k.frame).collect::<Vec<_>>());
    }
    if !n.flip.is_empty() {
        v["drawings"] = json!(n.flip.iter().map(|s| s.frame).collect::<Vec<_>>());
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

/// Run `f` with World mode's pick highlight off, so agents see the model's
/// real colors.
fn unhighlighted<T>(ctx: &mut Ctx, f: impl FnOnce(&mut Ctx) -> T) -> T {
    if ctx.scene.mode != crate::scene::Mode::World || ctx.scene.picked.is_empty() {
        return f(ctx);
    }
    let picked = std::mem::take(&mut ctx.scene.picked);
    app::sync_selection(ctx.r, ctx.scene, &mut ctx.ed.editor);
    let out = f(ctx);
    ctx.scene.picked = picked;
    app::sync_selection(ctx.r, ctx.scene, &mut ctx.ed.editor);
    out
}

fn frame_arg(ctx: &Ctx, a: &Value) -> Result<u32> {
    let f = int(a, "frame").context("missing `frame`")?;
    let n = ctx.scene.world.anim.frames;
    if f < 0 || f as u32 >= n {
        bail!("frame {f} is outside 0-{} (set_animation changes the frame count)", n - 1);
    }
    Ok(f as u32)
}

fn go_to(ctx: &mut Ctx, f: u32) {
    if f != ctx.scene.world.frame {
        let e = &mut ctx.ed.editor;
        if !ctx.scene.set_frame(f, |v| e.undo(v)) {
            ctx.ed.editor.reset();
        }
        app::sync_volume(ctx.r, ctx.scene, &mut ctx.ed.editor);
    }
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

/// Tools that only edit the part's voxels through the voxel editor (or
/// don't edit at all). Any other tool may change the tree, so the voxel
/// edits so far move into the tree's history first, one step each.
const KEEPS_VOXEL_HISTORY: [&str; 34] = [
    "set_voxels", "fill_box", "draw_shape", "draw_line", "draw_tube", "extrude_polygon", "place_stamp", "paint_faces", "vary_colors",
    "shade_edges", "select", "move_selection", "undo", "redo", "get_scene_info", "get_voxels", "get_slice", "get_projection", "get_palette",
    "list_parts", "list_projects", "list_shaders", "list_stamps", "list_palette_presets", "list_references", "render", "screenshot",
    "set_camera", "set_render_settings", "set_view_mode", "set_symmetry", "save_stamp", "save_file", "checkpoint",
];

fn dispatch(ctx: &mut Ctx, name: &str, args: &Value) -> Result<Vec<Value>> {
    let a = args;
    if !KEEPS_VOXEL_HISTORY.contains(&name) {
        app::keep_history(ctx.scene, &mut ctx.ed.editor);
    }
    if is_part_tool(name) {
        focus_part(ctx, get(a, "part"))?;
    }
    for family in [build::dispatch, guide::dispatch, import::dispatch, inspect::dispatch, reference::dispatch, surface::dispatch, view::dispatch] {
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
            if name.is_empty() {
                bail!("`name` can't be empty");
            }
            let size = ivec3(a, "size")?;
            if size.cmplt(IVec3::ONE).any() || size.cmpgt(IVec3::splat(ops::MAX_DIM)).any() {
                bail!("size must be 1..={} per axis", ops::MAX_DIM);
            }
            let parent = match get(a, "parent") {
                Some(p) => find_part(ctx.scene, Some(p))?,
                None => crate::world::ROOT,
            };
            let w = &ctx.scene.world;
            let position = match opt_ivec3(a, "position")? {
                Some(p) => p,
                None => w.reach().map_or(IVec3::ZERO, |(lo, s)| IVec3::new(lo.x + s.x + 2, lo.y, 0)),
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
            if get(a, "frame").is_some() {
                let f = frame_arg(ctx, a)?;
                go_to(ctx, f);
            }
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
                    w.set_tilt_world(id, t);
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
        "set_animation" => {
            let mut an = ctx.scene.world.anim.clone();
            if get(a, "frames").is_some() {
                an.frames = int_in(a, "frames", 1, 1, 10_000)? as u32;
            }
            if let Some(f) = get(a, "fps").and_then(Value::as_f64) {
                if !(0.5..=120.0).contains(&f) {
                    bail!("`fps` must be 0.5 to 120, not {f}");
                }
                an.fps = f as f32;
            }
            an.looping = boolean(a, "loop").unwrap_or(an.looping);
            an.smooth = boolean(a, "smooth").unwrap_or(an.smooth);
            app::tree_op(crate::outline::TreeOp::SetAnim(an), ctx.scene, ctx.ed);
            app::sync_volume(ctx.r, ctx.scene, &mut ctx.ed.editor);
            ctx.ed.timeline.open = true;
            Ok(vec![text(format!("Animation: {}; on frame {}", crate::timeline::describe(&ctx.scene.world.anim), ctx.scene.world.frame))])
        }
        "go_to_frame" => {
            let f = frame_arg(ctx, a)?;
            go_to(ctx, f);
            Ok(vec![text(format!("On frame {f} of 0-{}", ctx.scene.world.anim.frames - 1))])
        }
        "set_keyframe" => {
            let id = find_part(ctx.scene, get(a, "part"))?;
            let f = frame_arg(ctx, a)?;
            go_to(ctx, f);
            let changes = ["move", "position", "rotate_axis", "flip_axis", "free_rotation", "attach_to"].iter().any(|k| get(a, k).is_some());
            if changes {
                dispatch(ctx, "transform_part", a)?;
            }
            app::tree_op(crate::outline::TreeOp::Key(id), ctx.scene, ctx.ed);
            app::sync_volume(ctx.r, ctx.scene, &mut ctx.ed.editor);
            let keys: Vec<u32> = ctx.scene.world.node(id).map(|n| n.keys.iter().map(|k| k.frame).collect()).unwrap_or_default();
            Ok(vec![text(format!("Keyed `{}` on frame {f}; its keys: {keys:?}", ctx.scene.world.label(id)))])
        }
        "delete_keyframe" => {
            let id = find_part(ctx.scene, get(a, "part"))?;
            let f = int(a, "frame").context("missing `frame`")?;
            let keys: Vec<u32> = ctx.scene.world.node(id).map(|n| n.keys.iter().map(|k| k.frame).collect()).unwrap_or_default();
            if f < 0 || !keys.contains(&(f as u32)) {
                bail!("`{}` has no key on frame {f}; its keys: {keys:?}", ctx.scene.world.label(id));
            }
            let f = f as u32;
            if f < ctx.scene.world.anim.frames {
                go_to(ctx, f);
                app::tree_op(crate::outline::TreeOp::DeleteKey(id), ctx.scene, ctx.ed);
            } else {
                // Past the end of the timeline: nothing to go to.
                ctx.scene.change_quiet(|w| w.remove_key(id, f));
                ctx.ed.editor.reset();
            }
            app::sync_volume(ctx.r, ctx.scene, &mut ctx.ed.editor);
            Ok(vec![text(format!("Removed `{}`'s key on frame {f}", ctx.scene.world.label(id)))])
        }
        "new_drawing" | "remove_drawing" => {
            let id = find_part(ctx.scene, get(a, "part"))?;
            if ctx.scene.world.node(id).is_none_or(|n| n.is_group()) {
                bail!("drawings belong to parts, not groups");
            }
            let f = frame_arg(ctx, a)?;
            go_to(ctx, f);
            let has = ctx.scene.world.node(id).is_some_and(|n| n.flip.iter().any(|s| s.frame == f));
            let op = if name == "new_drawing" {
                if f == 0 {
                    bail!("frame 0 always shows `{}`'s first drawing; start a new one on a later frame", ctx.scene.world.label(id));
                }
                if has {
                    bail!("`{}` already has a drawing starting on frame {f}", ctx.scene.world.label(id));
                }
                crate::outline::TreeOp::SplitFrame(id)
            } else {
                if !has || f == 0 {
                    bail!("`{}` has no drawing starting on frame {f} (frame 0's can't be removed)", ctx.scene.world.label(id));
                }
                crate::outline::TreeOp::MergeFrame(id)
            };
            app::tree_op(op, ctx.scene, ctx.ed);
            app::sync_volume(ctx.r, ctx.scene, &mut ctx.ed.editor);
            let drawings: Vec<u32> = ctx.scene.world.node(id).map(|n| n.flip.iter().map(|s| s.frame).collect()).unwrap_or_default();
            let label = ctx.scene.world.label(id);
            Ok(vec![text(if name == "new_drawing" {
                format!("`{label}` has a new drawing from frame {f} (a copy of the one before); edit it with the voxel tools (part = `{label}`) while on frame {f}. Drawings start on frames {drawings:?}")
            } else if drawings.is_empty() {
                format!("Removed `{label}`'s drawing from frame {f}; it shows its one drawing on every frame again")
            } else {
                let from = drawings.iter().rev().find(|&&d| d < f).copied().unwrap_or(0);
                format!("Removed `{label}`'s drawing from frame {f}; frame {f} on shows the drawing from frame {from}. Drawings start on frames {drawings:?}")
            })])
        }
        "play" => {
            let on = boolean(a, "playing").unwrap_or(true);
            if !ctx.live {
                bail!("no voxely window is open to play in (running headless); use frame_strip to see the motion");
            }
            if on && ctx.scene.world.anim.frames < 2 {
                bail!("there's only one frame; set_animation sets the frame count");
            }
            if get(a, "from_frame").is_some() {
                let f = frame_arg(ctx, &json!({ "frame": a["from_frame"] }))?;
                go_to(ctx, f);
            }
            ctx.ed.timeline.open = true;
            ctx.ed.timeline.play(on);
            let an = &ctx.scene.world.anim;
            Ok(vec![text(if on {
                format!("Playing from frame {}: {}", ctx.scene.world.frame, crate::timeline::describe(an))
            } else {
                format!("Stopped on frame {}", ctx.scene.world.frame)
            })])
        }
        "frame_strip" => {
            let total = ctx.scene.world.anim.frames;
            let frames: Vec<u32> = match get(a, "frames").and_then(Value::as_array) {
                Some(list) => {
                    let mut frames = Vec::new();
                    for f in list {
                        match f.as_u64() {
                            Some(f) if f < total as u64 => frames.push(f as u32),
                            _ => bail!("frame {f} isn't in the animation: frames are 0 to {}", total - 1),
                        }
                    }
                    frames
                }
                None => {
                    let n = total.min(12);
                    (0..n).map(|i| i * total / n).collect()
                }
            };
            if frames.is_empty() {
                bail!("no frames");
            }
            let (w, h) = (int_in(a, "width", 240, 32, 1024)? as u32, int_in(a, "height", 240, 32, 1024)? as u32);
            let cols = (int_in(a, "columns", 6, 1, i64::MAX)? as usize).min(frames.len());
            let rows = frames.len().div_ceil(cols);
            let (sw, sh) = (w as usize * cols, h as usize * rows);
            let mut sheet = vec![0u8; sw * sh * 4];
            let back = ctx.scene.world.frame;
            for (k, &f) in frames.iter().enumerate() {
                go_to(ctx, f);
                let px = unhighlighted(ctx, |ctx| ctx.r.modeler_image(w, h))?;
                let (ox, oy) = ((k % cols) * w as usize, (k / cols) * h as usize);
                for y in 0..h as usize {
                    let src = &px[y * w as usize * 4..(y + 1) * w as usize * 4];
                    let at = ((oy + y) * sw + ox) * 4;
                    sheet[at..at + src.len()].copy_from_slice(src);
                }
                // A thin divider between frames.
                for y in 0..h as usize {
                    let at = ((oy + y) * sw + ox) * 4;
                    sheet[at..at + 4].copy_from_slice(&[90, 90, 90, 255]);
                }
            }
            go_to(ctx, back);
            Ok(vec![text(format!("Frames {frames:?}, left to right then top to bottom")), image(sw as u32, sh as u32, &sheet)?])
        }
        "create_rig" => {
            let quad = match string(a, "template") {
                Some("humanoid") => false,
                Some("quadruped") => true,
                _ => bail!("template is humanoid or quadruped"),
            };
            let prefix = string(a, "prefix").unwrap_or_default().trim().to_string();
            let scale = int(a, "scale").unwrap_or(1).clamp(1, 16) as i32;
            let at = match opt_ivec3(a, "position")? {
                Some(p) => p,
                None => ctx.scene.world.reach().map_or(IVec3::ZERO, |(lo, s)| IVec3::new(lo.x + s.x + 3, lo.y, 0)),
            };
            let colors = app::rig_colors(&ctx.scene.palette);
            let root = ctx.scene.change_quiet(|w| if quad { crate::rig::quadruped(w, at, scale, &prefix, colors) } else { crate::rig::humanoid(w, at, scale, &prefix, colors) });
            app::set_mode(ctx.scene, ctx.ed, None);
            ctx.scene.picked = vec![root];
            app::sync_volume(ctx.r, ctx.scene, &mut ctx.ed.editor);
            let mut names = Vec::new();
            ctx.scene.world.visit(root, &mut |id, _| names.push(ctx.scene.world.label(id)));
            Ok(vec![text(format!("Added a {} rig at {:?}: {}. Redraw parts with the voxel tools (pass `part`), then apply_motion.", if quad { "four-legged" } else { "humanoid" }, arr(at), names.join(", ")))])
        }
        "apply_motion" => {
            let name = string(a, "motion").context("missing `motion`")?.to_string();
            let fps = match get(a, "fps") {
                Some(v) => match v.as_f64() {
                    Some(f) if (0.5..=120.0).contains(&f) => Some(f as f32),
                    _ => bail!("fps must be 0.5-120"),
                },
                None => None,
            };
            let frames = match int(a, "frames") {
                Some(f) if (2..=10_000).contains(&f) => f as u32,
                Some(_) => bail!("frames must be 2-10000"),
                None if ctx.scene.world.anim.frames > 1 => ctx.scene.world.anim.frames,
                None => 24,
            };
            let amount = get(a, "amount").and_then(Value::as_f64).unwrap_or(1.0) as f32;
            let rig = match get(a, "rig") {
                Some(v) => Some(find_part(ctx.scene, Some(v))?),
                None => None,
            };
            // Try it on a copy first: a failure changes nothing.
            crate::rig::apply_motion(&mut ctx.scene.world.clone(), rig, &name, frames, amount).map_err(|e| anyhow!(e))?;
            let res = ctx.scene.change_quiet(|w| {
                if let Some(f) = fps {
                    w.anim.fps = f;
                }
                crate::rig::apply_motion(w, rig, &name, frames, amount)
            });
            ctx.ed.editor.reset();
            app::sync_volume(ctx.r, ctx.scene, &mut ctx.ed.editor);
            let applied = res.map_err(|e| anyhow!(e))?;
            ctx.ed.timeline.open = true;
            let mut msg = format!(
                "Keyed `{name}` over {frames} frames on {} ({}). Timeline: {}.",
                applied.bones.iter().map(|b| b.1.as_str()).collect::<Vec<_>>().join(", "),
                applied.keys,
                crate::timeline::describe(&ctx.scene.world.anim)
            );
            if !applied.missing.is_empty() {
                msg += &format!(" No part found for: {}.", applied.missing.iter().map(|b| crate::rig::bone_name(*b)).collect::<Vec<_>>().join(", "));
            }
            Ok(vec![text(msg)])
        }
        "import_animation" => {
            let path = PathBuf::from(string(a, "path").context("missing `path`")?);
            let g = crate::gltf_import::Gltf::load(&path)?;
            let clips = g.clips();
            let Some(clip) = get(a, "clip").and_then(part_key) else {
                let list: Vec<String> = clips.iter().enumerate().map(|(k, c)| format!("{k}: {} ({:.1}s)", c.name, c.duration)).collect();
                return Ok(vec![text(format!("{} clips in {}:\n{}\nPass `clip` to import one.", clips.len(), path.display(), list.join("\n")))]);
            };
            let rig = match get(a, "rig") {
                Some(v) => Some(find_part(ctx.scene, Some(v))?),
                None => None,
            };
            let root_motion = boolean(a, "root_motion").unwrap_or(true);
            let res = ctx.scene.change_quiet(|w| crate::gltf_import::import(w, &g, Some(&clip), rig, root_motion));
            ctx.ed.editor.reset();
            app::sync_volume(ctx.r, ctx.scene, &mut ctx.ed.editor);
            let done = res?;
            ctx.ed.timeline.open = true;
            let mut msg = format!(
                "Imported `{}`: {} frames at {} fps onto {}.",
                done.clip,
                done.frames,
                ctx.scene.world.anim.fps,
                done.bones.iter().map(|(_, src, dst)| format!("{dst} <- {src}")).collect::<Vec<_>>().join(", ")
            );
            if !done.missing.is_empty() {
                msg += &format!(" Not in the clip: {}.", done.missing.iter().map(|b| crate::rig::bone_name(*b)).collect::<Vec<_>>().join(", "));
            }
            Ok(vec![text(msg)])
        }
        "set_pivot" => {
            let id = find_part(ctx.scene, get(a, "part"))?;
            let w = &ctx.scene.world;
            let pivot = match get(a, "pivot") {
                // A part's box in part coordinates starts at 0; a group's is
                // its world box.
                Some(Value::String(s)) if s == "center" || s == "bottom" => {
                    let n = w.node(id).context("no such part")?;
                    let (lo, size) = match n.volume() {
                        Some(v) => (Vec3::ZERO, w.place(id).rot.extent(v.dims).as_vec3()),
                        None => w.bounds(id).map(|(lo, size)| (lo.as_vec3(), size.as_vec3())).context("the group is empty")?,
                    };
                    let z = if s == "bottom" { 0.0 } else { size.z * 0.5 };
                    // The default center follows a group's contents; a freely
                    // rotated one keeps a fixed point so it doesn't jump.
                    (s == "bottom" || (n.is_group() && w.tilted(id))).then(|| w.from_part(id, lo + Vec3::new(size.x * 0.5, size.y * 0.5, z)))
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
                    if name.is_empty() {
                        bail!("`name` can't be empty");
                    }
                    let unique = ctx.scene.world.unique_name(&name, Some(id));
                    ctx.scene.change(|w| w.node_mut(id).map(|n| n.name = unique.clone()));
                    if unique == name {
                        format!("Renamed `{label}` to `{name}`")
                    } else {
                        format!("Renamed `{label}` to `{unique}` (`{name}` is taken)")
                    }
                }
                "duplicate" => {
                    let width = ctx.scene.world.bounds_all(id).map_or(0, |b| b.1.x);
                    let offset = opt_ivec3(a, "offset")?.unwrap_or(IVec3::new(width + 1, 0, 0));
                    let name = string(a, "name").map(|n| n.trim().to_string()).filter(|n| !n.is_empty());
                    let copy = ctx.scene.change(|w| {
                        let c = w.duplicate(id)?;
                        w.shift(c, offset);
                        if let Some(name) = name {
                            let name = w.unique_name(&name, Some(c));
                            w.node_mut(c)?.name = name;
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
                    let mut attached = Vec::new();
                    ctx.scene.world.visit(id, &mut |i, _| attached.push(i));
                    let attached: Vec<String> = attached[1..].iter().map(|&i| format!("`{}`", ctx.scene.world.label(i))).collect();
                    ctx.scene.change(|w| w.delete(id));
                    if attached.is_empty() {
                        format!("Deleted `{label}`")
                    } else {
                        format!("Deleted `{label}` and what was in or attached to it: {}", attached.join(", "))
                    }
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
                let name = w.unique_name(&name, Some(g));
                w.node_mut(g)?.name = name;
                Some(g)
            });
            let g = g.context("nothing to group")?;
            tree_changed(ctx);
            let w = &ctx.scene.world;
            let inside: Vec<String> = w.node(g).map(|n| n.children().iter().map(|&c| format!("`{}`", w.label(c))).collect()).unwrap_or_default();
            Ok(vec![text(format!("Grouped {} into `{}` (id {g}); attached parts come with what they hang from", inside.join(", "), w.label(g)))])
        }
        "get_scene_info" => {
            let v = &ctx.scene.volume;
            let mut counts = [0usize; 256];
            for (_, c) in v.cells.filled() {
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
                // Like list_parts: the world box on the current frame.
                "part_position": arr(ctx.scene.editing().and_then(|id| ctx.scene.world.bounds(id)).map_or(ctx.scene.origin, |b| b.0)),
                "part_position_frame": ctx.scene.world.frame,
                "parts": ctx.scene.world.models().len(),
                "animation": { "frames": ctx.scene.world.anim.frames, "frame": ctx.scene.world.frame, "fps": ctx.scene.world.anim.fps, "loop": ctx.scene.world.anim.looping, "smooth": ctx.scene.world.anim.smooth },
                "camera": { "yaw": cam.yaw.to_degrees(), "pitch": cam.pitch.to_degrees(), "distance": cam.dist, "target": (cam.target + ctx.scene.origin.as_vec3()).to_array() },
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
            let path = match string(a, "path") {
                Some(p) if p.trim().is_empty() => bail!("`path` is empty; give a file path, or leave `path` out to save to {}", ctx.ed.save_path.display()),
                Some(p) => PathBuf::from(p).with_extension("vox"),
                None => ctx.ed.save_path.clone(),
            };
            let (old_path, old_name) = (ctx.ed.save_path.clone(), ctx.ed.name());
            let path = app::save_project(ctx.r, ctx.scene, ctx.ed, path)?;
            let full = std::fs::canonicalize(&path).unwrap_or(path);
            let mut msg = format!("Saved {}", full.display());
            if std::fs::canonicalize(&old_path).ok().as_ref() != Some(&full) {
                msg += &format!(". This file is now the current one (was {}); later saves and autosave go here.", old_path.display());
                msg += &reference::carry_over(&old_name, &ctx.ed.name());
            }
            Ok(vec![text(msg)])
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
            let path = app::save_project(ctx.r, ctx.scene, ctx.ed, path)?;
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
            let mut out = json!({ "workspace": app::workspace_dir().display().to_string(), "projects": list, "samples": samples });
            if let Some((name, files)) = &ctx.ed.folder {
                out["opened_folder"] = json!({ "name": name, "files": files.iter().map(stem).collect::<Vec<_>>() });
            }
            Ok(vec![text(serde_json::to_string_pretty(&out)?)])
        }
        "open_project" => {
            let name = string(a, "name").context("missing `name`")?;
            ctx.ed.rescan();
            let key = name.to_lowercase();
            let matches = |p: &&PathBuf| p.file_stem().is_some_and(|s| s.to_string_lossy().to_lowercase() == key);
            let folder = ctx.ed.folder.as_ref().map_or(&[][..], |(_, f)| f.as_slice());
            let path = ctx.ed.projects.iter().find(matches).or_else(|| ctx.ed.samples.iter().find(matches)).or_else(|| folder.iter().find(matches)).cloned().ok_or_else(|| anyhow!("no project or sample named {name:?}; see list_projects"))?;
            app::open_project(path.clone(), ctx.r, ctx.scene, ctx.ed)?;
            Ok(edited(ctx, true, &format!("Opened {}", path.display())))
        }
        "set_voxels" => {
            let list = get(a, "voxels").and_then(Value::as_array).context("`voxels` must be [[x, y, z, color], ...]")?;
            let mut new = ctx.scene.volume.clone();
            let mut skipped = 0;
            let mut sides = (IVec3::ZERO, new.dims - IVec3::ONE);
            for item in list {
                let v = item.as_array().filter(|v| v.len() == 4).context("each voxel is [x, y, z, color]")?;
                let n = |i: usize| -> Result<i32> {
                    let f = v[i].as_f64().context("voxel coordinates must be numbers")?;
                    if f.fract() != 0.0 || f.abs() > i32::MAX as f64 {
                        bail!("voxel coordinates must be whole numbers, not {f} (in {item})");
                    }
                    Ok(f as i32)
                };
                let p = IVec3::new(n(0)?, n(1)?, n(2)?);
                let c = color(ctx.scene, &v[3])?;
                if new.contains(p) {
                    new.set_index(new.index(p), c);
                } else {
                    skipped += 1;
                    sides = (sides.0.min(p), sides.1.max(p));
                }
            }
            let changed = apply(ctx, new);
            let mut out = edited(ctx, changed, &format!("Set {} voxels", list.len() - skipped));
            if skipped > 0 {
                let d = ctx.scene.volume.dims;
                let names = ["x", "y", "z"];
                let mut past = Vec::new();
                for k in 0..3 {
                    if sides.0[k] < 0 {
                        past.push(format!("{} down to {}", names[k], sides.0[k]));
                    }
                    if sides.1[k] >= d[k] {
                        past.push(format!("{} up to {} (max {})", names[k], sides.1[k], d[k] - 1));
                    }
                }
                let all = if skipped == list.len() { "All " } else { "" };
                out.push(text(format!("{all}{skipped} voxels were outside the canvas {}x{}x{} ({}) and skipped; use resize_canvas to grow it, or another part.", d.x, d.y, d.z, past.join(", "))));
            }
            Ok(out)
        }
        "fill_box" => {
            let (from, to) = (ivec3(a, "from")?, ivec3(a, "to")?);
            let c = color_arg(ctx.scene, a, "color")?;
            let new = shape_volume(&ctx.scene.volume, "box", 2, from, to, c, boolean(a, "hollow").unwrap_or(false));
            let changed = apply(ctx, new);
            Ok(edited_shape(ctx, changed, "Filled box", "box", a))
        }
        "draw_shape" => {
            let shape = string(a, "shape").context("missing `shape`")?;
            let (from, to) = (ivec3(a, "from")?, ivec3(a, "to")?);
            let c = color_arg(ctx.scene, a, "color")?;
            let solid = build::Solid::build(shape, axis(a)?, from, to, a)?;
            let new = build::fill_solid(&ctx.scene.volume, &solid, c, boolean(a, "hollow").unwrap_or(false));
            let changed = apply(ctx, new);
            Ok(edited_shape(ctx, changed, &format!("Drew {shape}"), shape, a))
        }
        "draw_line" => {
            let (from, to) = (ivec3(a, "from")?, ivec3(a, "to")?);
            let c = color_arg(ctx.scene, a, "color")?;
            let thick = int(a, "thickness").unwrap_or(1);
            if !(1..=64).contains(&thick) {
                bail!("`thickness` must be 1-64 voxels, not {thick}");
            }
            let mut new = ctx.scene.volume.clone();
            let brush = build::line_brush(thick as i32);
            build::line_cells(IVec3::ZERO, new.dims - IVec3::ONE, from, to, &brush, |p| new.set(p, c));
            let changed = apply(ctx, new);
            Ok(edited_shape(ctx, changed, "Drew line", "line", a))
        }
        "get_voxels" => {
            let v = &ctx.scene.volume;
            let max = v.dims - IVec3::ONE;
            let lo = opt_ivec3(a, "from")?.unwrap_or(IVec3::ZERO);
            let hi = opt_ivec3(a, "to")?.unwrap_or(max);
            let (lo, hi) = (lo.min(hi), lo.max(hi));
            if hi.cmplt(IVec3::ZERO).any() || lo.cmpgt(max).any() {
                bail!("the box {:?}..{:?} lies outside the canvas; coordinates run [0, 0, 0]..{:?}", arr(lo), arr(hi), arr(max));
            }
            let limit = int(a, "limit").unwrap_or(5000);
            if limit < 1 {
                bail!("`limit` must be 1 or more, not {limit}");
            }
            let limit = limit as usize;
            let mut list = Vec::new();
            let mut total = 0;
            for (p, c) in v.filled_sorted() {
                if p.cmpge(lo).all() && p.cmple(hi).all() {
                    total += 1;
                    if list.len() < limit {
                        list.push(json!([p.x, p.y, p.z, c]));
                    }
                }
            }
            let mut out = json!({ "total": total, "returned": list.len(), "voxels": list });
            if total > limit {
                out["note"] = json!(format!("Only the first {limit} of {total}; narrow from/to, raise limit, or see get_slice / get_projection for an overview."));
            }
            Ok(vec![text(serde_json::to_string(&out)?)])
        }
        "get_slice" => Ok(vec![text(inspect::slice(&ctx.scene.volume, &ctx.scene.palette, a)?)]),
        "edit_model" => {
            let op_name = string(a, "op").context("missing `op`")?;
            let paint = get(a, "color").map(|c| color(ctx.scene, c)).transpose()?;
            if paint.is_some() && !matches!(op_name, "full" | "fill" | "reverse" | "flood" | "maze" | "perlin") {
                bail!("op {op_name} doesn't use a color (full, fill, reverse, flood, maze and perlin do)");
            }
            if paint == Some(0) {
                bail!("color 0 is empty space; use op clear to erase");
            }
            if matches!(op_name, "flip" | "rotate" | "mirror" | "diagonal" | "loop") && get(a, "axis").is_none() {
                bail!("op {op_name} needs `axis` (x, y or z)");
            }
            let ax = axis(a)?;
            let dir = match int(a, "direction") {
                None | Some(1) => 1,
                Some(-1) => -1,
                Some(d) => bail!("`direction` must be 1 or -1, not {d}"),
            };
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
                "loop" => Op::Loop(ax, dir),
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
            if let Op::Fit = op {
                return fit_part(ctx);
            }
            // `color` is for this op only; the current color stays.
            let current = ctx.ed.editor.color;
            ctx.ed.editor.color = paint.unwrap_or(current);
            let new = app::apply_op(op, ctx.scene, &ctx.ed.editor);
            ctx.ed.editor.color = current;
            let new = new.ok_or_else(|| anyhow!("{op_name} isn't possible here (empty model, or the canvas would exceed {})", ops::MAX_DIM))?;
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
            let old = ctx.scene.volume.dims;
            if matches!(kind, TransformKind::Rotate) && new.dims != old {
                // The turn is about the canvas center; keep that center in place.
                let new = build::before_commit(ctx, new);
                let at = -(new.dims - old).div_euclid(IVec3::splat(2));
                recanvas(ctx, new, at);
                return Ok(edited(ctx, true, &format!("Applied {text_args}; the canvas grew to {:?} around the same center", ctx.scene.volume.dims.to_array())));
            }
            let changed = apply(ctx, new);
            Ok(edited(ctx, changed, &format!("Applied {text_args}")))
        }
        "resize_canvas" => {
            let size = ivec3(a, "size")?;
            if size.cmplt(IVec3::ONE).any() || size.cmpgt(IVec3::splat(ops::MAX_DIM)).any() {
                bail!("size must be 1..={} per axis", ops::MAX_DIM);
            }
            let old = ctx.scene.volume.dims;
            let before = count(&ctx.scene.volume);
            let shift = ops::resize_shift(&ctx.scene.volume, size);
            let new = ops::resize(&ctx.scene.volume, size);
            let cut = before.saturating_sub(count(&new));
            let cut = if cut > 0 { format!("; {cut} voxels outside the new canvas were cut off (undo brings them back)") } else { String::new() };
            if new.dims == old {
                let changed = apply(ctx, new);
                return Ok(edited(ctx, changed, &format!("Resized canvas{cut}")));
            }
            let new = build::before_commit(ctx, new);
            recanvas(ctx, new, -shift);
            Ok(edited(ctx, true, &format!("Resized canvas to {:?}; the part's world position moved by {:?} so its voxels stay where they are in the world{cut}", ctx.scene.volume.dims.to_array(), (-shift).to_array())))
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
            // `color` (the shader's i_color_index) is for this run only.
            let paint = match get(a, "color") {
                Some(c) => color(ctx.scene, c)?,
                None => ctx.ed.editor.color,
            };
            if paint == 0 {
                bail!("color 0 is empty space; pass a palette color 1-255");
            }
            let sels: Vec<u8> = (1..256).filter(|&i| ctx.ed.palette.selected[i]).map(|i| i as u8).collect();
            let out = crate::xs::run(&ctx.r.device, &ctx.r.queue, &shader, &values, &ctx.scene.volume, &ctx.scene.palette, paint, &sels)?;
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
            // Every part and drawing, not just the one being edited.
            let used = ctx.scene.used_colors();
            let kinds = ["diffuse", "metal", "glass", "emit", "blend", "cloud"];
            let list: Vec<Value> = (1..256)
                .filter(|&i| all || used[i])
                .map(|i| {
                    let m = &ctx.scene.materials[i];
                    let mut e = json!({ "index": i, "color": hex(ctx.scene.palette[i]), "material": kinds.get(m.kind as usize).copied().unwrap_or("?"), "in_use": used[i] });
                    // The parameters set_material takes for this type.
                    let round = |v: f32| (v as f64 * 1000.0).round() / 1000.0;
                    let k = m.kind;
                    let has = |kinds: &[MaterialKind]| kinds.iter().any(|x| *x as u32 == k);
                    if has(&[MaterialKind::Metal, MaterialKind::Glass, MaterialKind::Blend]) {
                        e["roughness"] = json!(round(m.rough));
                        e["ior"] = json!(round(m.ior));
                    }
                    if has(&[MaterialKind::Metal, MaterialKind::Blend]) {
                        e["metallic"] = json!(round(m.metal));
                    }
                    if has(&[MaterialKind::Glass, MaterialKind::Blend]) {
                        e["transparency"] = json!(round(m.trans));
                    }
                    if has(&[MaterialKind::Emit]) {
                        e["emission"] = json!(round(m.emit));
                    }
                    if has(&[MaterialKind::Media]) {
                        e["density"] = json!(round(m.density));
                    }
                    if let Some(n) = ctx.scene.notes.get((i - 1) / 8).filter(|n| !n.is_empty()) {
                        e["ramp"] = json!(n);
                    }
                    e
                })
                .collect();
            Ok(vec![text(serde_json::to_string(&list)?)])
        }
        "set_palette_colors" => {
            let list = get(a, "colors").and_then(Value::as_array).context("`colors` must be [{index, color}, ...]")?;
            // Check every entry before changing anything.
            let mut set = Vec::new();
            for item in list {
                let i = item.get("index").and_then(Value::as_u64).filter(|&i| (1..=255).contains(&i)).context("index must be 1..=255")? as usize;
                set.push((i, parse_hex(item.get("color").and_then(Value::as_str).context("color must be \"#rrggbb\"")?)?));
            }
            crate::palette_lib::begin_step(ctx.scene, &mut ctx.ed.editor);
            for (i, c) in set {
                ctx.scene.palette[i] = [c[0], c[1], c[2], 255];
                ctx.scene.look[i] = None;
                ctx.scene.sync_color(i);
            }
            ctx.scene.palette_step_update(None);
            materials_changed(ctx);
            Ok(vec![text(format!("Set {} palette colors", list.len()))])
        }
        "load_palette" => {
            use crate::palette_lib::Source;
            let src = match (string(a, "path"), int(a, "page"), string(a, "preset")) {
                (Some(p), ..) => Source::File(PathBuf::from(p)),
                (None, Some(k), _) => Source::File(ctx.ed.palettes.get(k as usize).cloned().ok_or_else(|| anyhow!("no palette page {k}; {} available", ctx.ed.palettes.len()))?),
                (None, None, Some(n)) => Source::from_key(&format!("preset:{n}")).ok_or_else(|| anyhow!("no preset {n:?}; see list_palette_presets"))?,
                _ => bail!("give `path`, `page` or `preset`"),
            };
            palette_edit(ctx, crate::palette_lib::Cmd::Load(src, boolean(a, "match").unwrap_or(false)))
        }
        "edit_palette" => {
            use crate::palette_lib::Cmd;
            let action = string(a, "action").context("give an `action`")?;
            let idx = |k: &str| -> Result<usize> {
                let v = int(a, k).with_context(|| format!("{action} needs `{k}`"))?;
                if !(1..=255).contains(&v) {
                    bail!("`{k}` must be a palette index 1-255, not {v}");
                }
                Ok(v as usize)
            };
            let cmd = match action {
                "swap" => Cmd::Swap(idx("from")?, idx("to")?),
                "move" => Cmd::Move(idx("from")?, idx("to")?),
                "copy" => Cmd::Copy(idx("from")?, idx("to")?),
                "gradient" => Cmd::Gradient(idx("from")?, idx("to")?),
                "sort" => Cmd::Sort(string(a, "by").map_or(Some(crate::palette_ops::SortKey::Hue), crate::palette_ops::SortKey::parse).context("`by` is hue, saturation or value")?),
                "pack" => Cmd::Pack,
                "undo" => Cmd::Undo,
                "save" => Cmd::Save(string(a, "name").context("save needs `name`")?.to_string()),
                "name_ramp" => {
                    let row = int(a, "row").filter(|r| (0..32).contains(r)).context("name_ramp needs `row` 0-31 (colors 8*row+1 ..= 8*row+8)")? as usize;
                    let name = string(a, "name").unwrap_or_default().to_string();
                    crate::palette_lib::begin_step(ctx.scene, &mut ctx.ed.editor);
                    if ctx.scene.notes.len() < 32 {
                        ctx.scene.notes.resize(32, String::new());
                    }
                    ctx.scene.notes[row] = name.clone();
                    ctx.scene.palette_step_update(None);
                    return Ok(vec![text(format!("Ramp {row} (colors {}-{}) is now {name:?}", row * 8 + 1, (row * 8 + 8).min(255)))]);
                }
                other => bail!("unknown action {other:?}"),
            };
            palette_edit(ctx, cmd)
        }
        "set_material" => {
            let i = color_arg(ctx.scene, a, "color")? as usize;
            if i == 0 {
                bail!("color 0 is empty space");
            }
            let kinds = ["diffuse", "metal", "glass", "emit", "blend", "cloud"];
            let kind = match string(a, "type") {
                Some(t) => kinds.iter().position(|k| *k == t).with_context(|| format!("unknown material type {t:?}; one of {}", kinds.join(", ")))? as u32,
                None => ctx.scene.materials[i].kind,
            };
            let kind_name = kinds.get(kind as usize).copied().unwrap_or("?");
            // Check every value before changing anything.
            let params: [(&str, f32, f32, &[MaterialKind]); 6] = [
                ("roughness", 0.0, 1.0, &[MaterialKind::Metal, MaterialKind::Glass, MaterialKind::Blend]),
                ("metallic", 0.0, 1.0, &[MaterialKind::Metal, MaterialKind::Blend]),
                ("ior", 1.0, 3.0, &[MaterialKind::Metal, MaterialKind::Glass, MaterialKind::Blend]),
                ("transparency", 0.0, 1.0, &[MaterialKind::Glass, MaterialKind::Blend]),
                ("emission", 0.0, f32::INFINITY, &[MaterialKind::Emit]),
                ("density", 0.0, f32::INFINITY, &[MaterialKind::Media]),
            ];
            let mut values = [None; 6];
            for (k, (key, lo, hi, applies)) in params.iter().enumerate() {
                let Some(v) = get(a, key) else { continue };
                let v = v.as_f64().with_context(|| format!("`{key}` must be a number"))? as f32;
                if !(v >= *lo && v <= *hi) {
                    bail!("`{key}` must be {}, not {v}", if hi.is_finite() { format!("{lo}-{hi}") } else { format!("{lo} or more") });
                }
                if !applies.iter().any(|m| *m as u32 == kind) {
                    let names: Vec<&str> = applies.iter().map(|m| kinds[*m as usize]).collect();
                    bail!("`{key}` only affects {} materials; color {i} is {kind_name} (pass `type` to change it)", names.join(" and "));
                }
                values[k] = Some(v);
            }
            crate::palette_lib::begin_step(ctx.scene, &mut ctx.ed.editor);
            let m = &mut ctx.scene.materials[i];
            m.kind = kind;
            if m.kind == MaterialKind::Emit as u32 && m.emit <= 0.0 {
                m.emit = 1.0;
            }
            if m.kind == MaterialKind::Glass as u32 && m.trans <= 0.0 {
                m.trans = 1.0;
            }
            let [rough, metal, ior, trans, emit, density] = values;
            m.rough = rough.unwrap_or(m.rough);
            m.metal = metal.unwrap_or(m.metal);
            m.ior = ior.unwrap_or(m.ior);
            m.trans = trans.unwrap_or(m.trans);
            m.emit = emit.unwrap_or(m.emit);
            m.density = density.unwrap_or(m.density);
            ctx.scene.palette_step_update(None);
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
            let sel = ctx.ed.editor.selection().cloned();
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
            let steps = int(a, "steps").unwrap_or(1);
            if steps < 1 {
                bail!("`steps` must be 1 or more, not {steps}");
            }
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
            // Check everything before changing anything.
            let range = |key: &str, lo: f32, hi: f32| -> Result<Option<f32>> {
                let Some(v) = get(a, key) else { return Ok(None) };
                let v = v.as_f64().with_context(|| format!("`{key}` must be a number"))? as f32;
                if !(v >= lo && v <= hi) {
                    bail!("`{key}` must be {lo} to {hi}, not {v}");
                }
                Ok(Some(v))
            };
            let yaw = range("yaw", f32::MIN, f32::MAX)?;
            let pitch = range("pitch", -90.0, 90.0)?;
            let fov = range("fov", 1.0, 170.0)?;
            let dist = range("distance", 0.0, f32::MAX)?;
            if dist == Some(0.0) {
                bail!("`distance` must be more than 0");
            }
            let target = match get(a, "target") {
                Some(t) => Some(vec3_value(t).context("`target` must be [x, y, z] numbers")?),
                None => None,
            };
            let projection = match string(a, "projection") {
                None => None,
                Some(p @ ("perspective" | "orthographic" | "isometric" | "stereographic" | "panorama")) => Some(p),
                Some(p) => bail!("unknown projection {p:?}; one of perspective, orthographic, isometric, stereographic, panorama"),
            };
            let origin = ctx.scene.origin.as_vec3();
            let r = &mut *ctx.r;
            if boolean(a, "frame").unwrap_or(false) {
                r.frame(ctx.scene.volume.dims.as_vec3());
            }
            let mut cam: Camera = r.camera;
            if let Some(v) = yaw {
                cam.yaw = v.to_radians();
            }
            if let Some(v) = pitch {
                // Exactly ±90 would make the camera basis degenerate.
                cam.pitch = v.clamp(-89.99, 89.99).to_radians();
            }
            if let Some(v) = dist {
                cam.dist = v;
            }
            if let Some(t) = target {
                // The renderer's camera works in the edited volume's cells.
                cam.target = t - origin;
            }
            r.set_camera(cam);
            if let Some(v) = fov {
                r.settings.fov = v;
            }
            match projection {
                Some("perspective") => r.settings.projection = Projection::Perspective,
                Some("orthographic") => r.settings.projection = Projection::Orthographic,
                Some("isometric") => app::iso_camera(r),
                Some("stereographic") => r.settings.projection = Projection::Stereographic,
                Some("panorama") => r.settings.projection = Projection::Panorama,
                _ => {}
            }
            r.reset();
            let cam = &r.camera;
            let t = cam.target + origin;
            Ok(vec![text(format!(
                "Camera yaw {:.1}, pitch {:.1}, distance {:.1}, target [{:.1}, {:.1}, {:.1}] world",
                cam.yaw.to_degrees(),
                cam.pitch.to_degrees(),
                cam.dist,
                t.x,
                t.y,
                t.z
            ))])
        }
        "set_render_settings" => {
            // Work on a copy so a bad value leaves every setting as it was.
            let mut s = ctx.r.settings.clone();
            let mut set = Vec::new();
            macro_rules! f32_field {
                ($key:literal, $field:ident, $ok:expr, $range:literal) => {
                    if let Some(v) = get(a, $key) {
                        let v = v.as_f64().context(concat!("`", $key, "` must be a number"))? as f32;
                        if !($ok)(v) {
                            bail!("`{}` must be {}, not {v}", $key, $range);
                        }
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
                        s.$field = hex_linear(v).with_context(|| format!("`{}`", $key))?;
                        set.push($key);
                    }
                };
            }
            let any = |v: f32| v.is_finite();
            let not_negative = |v: f32| v >= 0.0 && v.is_finite();
            f32_field!("sun_pitch", sun_pitch, |v: f32| (-90.0..=90.0).contains(&v), "-90 to 90");
            f32_field!("sun_yaw", sun_yaw, any, "a number");
            f32_field!("sun_intensity", sun_intensity, not_negative, "0 or more");
            f32_field!("sun_area", sun_area, not_negative, "0 or more");
            color_field!("sun_color", sun_color);
            f32_field!("sky_intensity", sky_intensity, not_negative, "0 or more");
            color_field!("sky_color", sky_color);
            f32_field!("exposure", exposure, not_negative, "0 or more");
            f32_field!("gamma", gamma, |v: f32| v > 0.0 && v.is_finite(), "more than 0 (usually 2.2)");
            bool_field!("aces", aces);
            bool_field!("bloom", bloom);
            bool_field!("denoise", denoise);
            f32_field!("fog_density", fog_density, not_negative, "0 or more");
            color_field!("fog_color", fog_color);
            bool_field!("ground", ground);
            color_field!("ground_color", ground_color);
            color_field!("background_color", bg_color);
            bool_field!("edges", show_edges);
            bool_field!("grid", show_grid);
            f32_field!("aperture", aperture, not_negative, "0 or more");
            if let Some(v) = string(a, "sky") {
                s.env_mode = match v {
                    "uniform" => crate::env::EnvMode::Uniform,
                    "atmosphere" => crate::env::EnvMode::Atmosphere,
                    "image" => crate::env::EnvMode::Image,
                    v => bail!("unknown sky {v:?}; uniform, atmosphere or image"),
                };
                set.push("sky");
            }
            if let Some(v) = string(a, "background") {
                s.bg_mode = match v {
                    "sky" => BgMode::Sky,
                    "color" => BgMode::Color,
                    "transparent" => BgMode::Transparent,
                    v => bail!("unknown background {v:?}; sky, color or transparent"),
                };
                set.push("background");
            }
            ctx.r.settings = s;
            ctx.r.reset();
            Ok(vec![text(if set.is_empty() { "No settings given".to_string() } else { format!("Set {}", set.join(", ")) })])
        }
        // "screenshot" lives in view.rs.
        "render" => {
            if get(a, "views").is_some() || boolean(a, "frame").unwrap_or(false) {
                return view::render_framed(ctx, a);
            }
            let max = max_image_side(ctx);
            let (w, h) = (int_in(a, "width", 960, 16, max)? as u32, int_in(a, "height", 640, 16, max)? as u32);
            let spp = int_in(a, "samples", 128, 1, 65536)? as u32;
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
            ctx.scene.settings = ctx.r.settings.clone();
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
            let timeout = match get(a, "timeout") {
                None => crate::script::DEFAULT_TIMEOUT,
                Some(t) => match t.as_f64() {
                    Some(s) if (0.1..=3600.0).contains(&s) => std::time::Duration::from_secs_f64(s),
                    _ => bail!("`timeout` must be a number of seconds from 0.1 to 3600, not {t}"),
                },
            };
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
        other => {
            let names = tools();
            let close: Vec<&str> = names
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|t| t["name"].as_str())
                .filter(|n| {
                    let bare = |s: &str| s.replace(['_', '-', ' '], "").to_lowercase();
                    bare(n).contains(&bare(other)) || bare(other).contains(&bare(n)) || other.split('_').any(|w| w.len() > 3 && n.contains(w))
                })
                .collect();
            if close.is_empty() {
                bail!("unknown tool {other:?}; see tools/list")
            }
            bail!("unknown tool {other:?}; did you mean {}?", close.join(", "))
        }
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
