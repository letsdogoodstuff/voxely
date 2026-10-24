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
Typical flow: new_scene -> draw with fill_box / draw_shape / draw_line / set_voxels -> look with \
screenshot (fast) -> set_material and set_render_settings -> render (path traced) -> save_file. \
Every edit is undoable. If the voxely window is open, edits appear in it live.";

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

/// The tool list for MCP `tools/list`.
pub fn tools() -> Value {
    let axis = json!({ "type": "string", "enum": ["x", "y", "z"] });
    json!([
        tool("get_scene_info", "Canvas size, voxel count and bounds, colors in use, file, selection, camera and mode.", json!({}), &[]),
        tool("new_scene", "Start a new project. Empty unless `fill` is given.", json!({
            "size": xyz("Canvas size, default [40, 40, 40]; up to 1024 per axis"),
            "fill": color_schema(),
        }), &[]),
        tool("open_file", "Open a MagicaVoxel .vox file (all models are merged into one canvas).", json!({ "path": { "type": "string" } }), &["path"]),
        tool("save_file", "Save the project as .vox. Without `path`, saves to the current save path.", json!({ "path": { "type": "string" } }), &[]),
        tool("set_voxels", "Set individual voxels. Color 0 erases.", json!({
            "voxels": { "type": "array", "description": "[[x, y, z, color], ...]; color is an index or \"#rrggbb\"", "items": { "type": "array", "items": { "type": ["integer", "string"] } } },
        }), &["voxels"]),
        tool("fill_box", "Fill (or with color 0, clear) the box between two corners, inclusive.", json!({
            "from": xyz(XYZ), "to": xyz(XYZ), "color": color_schema(),
            "hollow": { "type": "boolean", "description": "Only the outer shell" },
        }), &["from", "to", "color"]),
        tool("draw_shape", "Draw a solid shape inscribed in the box between two corners (inclusive).", json!({
            "shape": { "type": "string", "enum": ["box", "ellipsoid", "cylinder", "cone", "pyramid"] },
            "from": xyz(XYZ), "to": xyz(XYZ), "color": color_schema(),
            "axis": { "type": "string", "enum": ["x", "y", "z"], "description": "Cylinder/cone/pyramid axis (default z; cones and pyramids point toward +axis)" },
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
        tool("set_view_mode", "Show the modeler or the live path tracer in the window.", json!({
            "mode": { "type": "string", "enum": ["model", "render"] },
        }), &["mode"]),
    ])
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
    match string(args, "axis").unwrap_or("z") {
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
fn apply(ctx: &mut Ctx, new: Volume) -> bool {
    let changed = ctx.ed.editor.commit(&mut ctx.scene.volume, new);
    if changed {
        app::sync_volume(ctx.r, ctx.scene, &mut ctx.ed.editor);
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
fn edited(ctx: &Ctx, changed: bool, what: &str) -> Vec<Value> {
    let n = count(&ctx.scene.volume);
    let d = ctx.scene.volume.dims;
    if changed {
        vec![text(format!("{what}. Canvas {}x{}x{}, {n} voxels.", d.x, d.y, d.z))]
    } else {
        vec![text(format!("{what}: nothing changed. Canvas {}x{}x{}, {n} voxels.", d.x, d.y, d.z))]
    }
}

/// Whether cell `p` lies inside `shape` inscribed in [lo, hi].
fn in_shape(shape: &str, axis: usize, lo: IVec3, hi: IVec3, p: IVec3) -> bool {
    if p.cmplt(lo).any() || p.cmpgt(hi).any() {
        return false;
    }
    let center = (lo.as_vec3() + hi.as_vec3() + 1.0) * 0.5;
    let half = (hi - lo + IVec3::ONE).as_vec3() * 0.5;
    let q = (p.as_vec3() + 0.5 - center) / half;
    let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
    // 0 at the base (low end of the axis), 1 at the tip.
    let t = (q[axis] + 1.0) * 0.5;
    match shape {
        "box" => true,
        "ellipsoid" | "sphere" => q.length_squared() <= 1.0,
        "cylinder" => q[u] * q[u] + q[w] * q[w] <= 1.0,
        "cone" => (q[u] * q[u] + q[w] * q[w]).sqrt() <= 1.0 - t,
        "pyramid" => q[u].abs().max(q[w].abs()) <= 1.0 - t,
        _ => false,
    }
}

const NEIGHBORS: [IVec3; 6] = [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::Z, IVec3::NEG_Z];

/// Fill `shape` between two corners with `c`.
fn shape_volume(v: &Volume, shape: &str, axis: usize, a: IVec3, b: IVec3, c: u8, hollow: bool) -> Volume {
    let (lo, hi) = (a.min(b), a.max(b));
    let inside = |p: IVec3| in_shape(shape, axis, lo, hi, p);
    let mut out = v.clone();
    for z in lo.z.max(0)..=hi.z.min(v.dims.z - 1) {
        for y in lo.y.max(0)..=hi.y.min(v.dims.y - 1) {
            for x in lo.x.max(0)..=hi.x.min(v.dims.x - 1) {
                let p = IVec3::new(x, y, z);
                if inside(p) && (!hollow || NEIGHBORS.iter().any(|&n| !inside(p + n))) {
                    out.set(p, c);
                }
            }
        }
    }
    out
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

fn dispatch(ctx: &mut Ctx, name: &str, args: &Value) -> Result<Vec<Value>> {
    let a = args;
    match name {
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
                "can_undo": ctx.ed.editor.can_undo(),
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
            if let Some(p) = string(a, "path") {
                ctx.ed.save_path = PathBuf::from(p).with_extension("vox");
            }
            let path = ctx.ed.save_path.clone();
            if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
                std::fs::create_dir_all(dir)?;
            }
            ctx.scene.save(&path)?;
            let full = std::fs::canonicalize(&path).unwrap_or(path);
            Ok(vec![text(format!("Saved {}", full.display()))])
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
            if !["box", "ellipsoid", "sphere", "cylinder", "cone", "pyramid"].contains(&shape) {
                bail!("unknown shape {shape:?}");
            }
            let (from, to) = (ivec3(a, "from")?, ivec3(a, "to")?);
            let c = color_arg(ctx.scene, a, "color")?;
            let new = shape_volume(&ctx.scene.volume, shape, axis(a)?, from, to, c, boolean(a, "hollow").unwrap_or(false));
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
                let ok = if name == "undo" { ctx.ed.editor.undo(&mut ctx.scene.volume) } else { ctx.ed.editor.redo(&mut ctx.scene.volume) };
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
