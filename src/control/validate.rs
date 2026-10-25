//! Checks a tool call's arguments against the tool's `inputSchema` before it
//! runs, so a misspelled or out-of-range argument is an error instead of
//! being ignored or clamped. Handles the small part of JSON Schema the tool
//! definitions use: type, enum, minimum / maximum (and exclusive), items,
//! minItems / maxItems, properties, required and anyOf.

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{Result, bail};
use serde_json::{Map, Value};

/// Every tool's input schema by name, unlisted tools included.
fn schemas() -> &'static HashMap<String, Value> {
    static SCHEMAS: OnceLock<HashMap<String, Value>> = OnceLock::new();
    SCHEMAS.get_or_init(|| {
        super::all_tools().as_array().into_iter().flatten().filter_map(|t| Some((t["name"].as_str()?.to_string(), t["inputSchema"].clone()))).collect()
    })
}

/// Check `args` for tool `name`. Returns the arguments to run with: the same,
/// except whole-number floats given for integers become integers (2.0 -> 2).
/// Unknown tools pass through (dispatch reports them).
pub fn check_call(name: &str, args: &Value) -> Result<Value> {
    let Some(schema) = schemas().get(name) else { return Ok(args.clone()) };
    let mut out = match args {
        Value::Null => Value::Object(Map::new()),
        Value::Object(_) => args.clone(),
        other => bail!("arguments must be an object, got {}", kind(other)),
    };
    check_object(name, schema, out.as_object_mut().unwrap(), "")?;
    Ok(out)
}

/// The top-level arguments, or a nested object with `properties`.
fn check_object(tool: &str, schema: &Value, obj: &mut Map<String, Value>, path: &str) -> Result<()> {
    let empty = Map::new();
    let props = schema.get("properties").and_then(Value::as_object).unwrap_or(&empty);
    let open = schema.get("additionalProperties").is_some_and(|v| v != &Value::Bool(false)) || schema.get("properties").is_none();
    if !open {
        for key in obj.keys() {
            if !props.contains_key(key) {
                let at = if path.is_empty() { format!("for {tool}") } else { format!("in `{path}` for {tool}") };
                let valid: Vec<&str> = props.keys().map(String::as_str).collect();
                let hint = closest(key, &valid).map(|c| format!(" (did you mean \"{c}\"?)")).unwrap_or_default();
                if valid.is_empty() {
                    bail!("unknown parameter \"{key}\" {at}; it takes no parameters");
                }
                bail!("unknown parameter \"{key}\" {at}{hint}; valid: {}", valid.join(", "));
            }
        }
    }
    for req in schema.get("required").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
        // Null counts as not given, as it does for the handlers.
        if obj.get(req).is_none_or(Value::is_null) {
            bail!("missing required parameter `{}`", join(path, req));
        }
    }
    for (key, v) in obj.iter_mut() {
        if v.is_null() && path.is_empty() {
            continue;
        }
        if let Some(s) = props.get(key) {
            check_value(tool, s, v, &join(path, key))?;
        }
    }
    Ok(())
}

fn check_value(tool: &str, schema: &Value, v: &mut Value, path: &str) -> Result<()> {
    if let Some(branches) = schema.get("anyOf").and_then(Value::as_array) {
        let mut first_err = None;
        for b in branches {
            let mut try_v = v.clone();
            match check_value(tool, b, &mut try_v, path) {
                Ok(()) => {
                    *v = try_v;
                    return Ok(());
                }
                // The branch of the value's own type explains best.
                Err(e) if types(b).is_some_and(|ts| ts.iter().any(|t| is_type(v, t))) => return Err(e),
                Err(e) => first_err = first_err.or(Some(e)),
            }
        }
        let names: Vec<String> = branches.iter().filter_map(types).flatten().map(str::to_string).collect();
        if names.is_empty() {
            return Err(first_err.unwrap_or_else(|| anyhow::anyhow!("`{path}` doesn't match its schema")));
        }
        bail!("`{path}` must be {}, got {}", or_list(&names), kind(v));
    }
    if let Some(ts) = types(schema) {
        let Some(t) = ts.iter().find(|t| is_type(v, t)) else {
            bail!("`{path}` must be {}, got {}", or_list(&ts.iter().map(|t| t.to_string()).collect::<Vec<_>>()), shown(v));
        };
        if *t == "integer" && !v.is_i64() && !v.is_u64() {
            *v = Value::from(v.as_f64().unwrap() as i64);
        }
    }
    if let Some(options) = schema.get("enum").and_then(Value::as_array) {
        if !options.contains(v) {
            let names: Vec<String> = options.iter().map(|o| o.as_str().map_or_else(|| o.to_string(), str::to_string)).collect();
            let hint = v.as_str().and_then(|s| closest(s, &names.iter().map(String::as_str).collect::<Vec<_>>())).map(|c| format!(" (did you mean \"{c}\"?)")).unwrap_or_default();
            bail!("`{path}` is {}{hint}; must be one of: {}", shown(v), names.join(", "));
        }
    }
    if let Some(x) = v.as_f64() {
        check_range(schema, x, v, path)?;
    }
    match v {
        Value::Array(items) => {
            let n = items.len() as u64;
            let (lo, hi) = (schema.get("minItems").and_then(Value::as_u64), schema.get("maxItems").and_then(Value::as_u64));
            match (lo, hi) {
                (Some(lo), Some(hi)) if lo == hi && n != lo => bail!("`{path}` must have exactly {lo} items, got {n}"),
                (Some(lo), _) if n < lo => bail!("`{path}` must have at least {lo} items, got {n}"),
                (_, Some(hi)) if n > hi => bail!("`{path}` must have at most {hi} items, got {n}"),
                _ => {}
            }
            if let Some(item) = schema.get("items") {
                for (i, x) in items.iter_mut().enumerate() {
                    check_value(tool, item, x, &format!("{path}[{i}]"))?;
                }
            }
        }
        Value::Object(obj) if schema.get("properties").is_some() => check_object(tool, schema, obj, path)?,
        _ => {}
    }
    Ok(())
}

fn check_range(s: &Value, x: f64, v: &Value, path: &str) -> Result<()> {
    let bound = |k: &str| s.get(k).and_then(Value::as_f64);
    let (min, max, xmin, xmax) = (bound("minimum"), bound("maximum"), bound("exclusiveMinimum"), bound("exclusiveMaximum"));
    let ok = min.is_none_or(|m| x >= m) && max.is_none_or(|m| x <= m) && xmin.is_none_or(|m| x > m) && xmax.is_none_or(|m| x < m);
    if ok {
        return Ok(());
    }
    let mut want = Vec::new();
    match (min, max) {
        (Some(lo), Some(hi)) => want.push(format!("{} to {}", num(lo), num(hi))),
        (Some(lo), None) => want.push(format!("at least {}", num(lo))),
        (None, Some(hi)) => want.push(format!("at most {}", num(hi))),
        _ => {}
    }
    if let Some(m) = xmin {
        want.push(format!("more than {}", num(m)));
    }
    if let Some(m) = xmax {
        want.push(format!("less than {}", num(m)));
    }
    bail!("`{path}` is {v}; must be {}", want.join(" and "))
}

fn types(schema: &Value) -> Option<Vec<&str>> {
    match schema.get("type")? {
        Value::String(t) => Some(vec![t.as_str()]),
        Value::Array(ts) => Some(ts.iter().filter_map(Value::as_str).collect()),
        _ => None,
    }
}

fn is_type(v: &Value, t: &str) -> bool {
    match t {
        "integer" => v.is_i64() || v.is_u64() || v.as_f64().is_some_and(|f| f.fract() == 0.0 && f.abs() < 9.0e15),
        "number" => v.is_number(),
        "string" => v.is_string(),
        "boolean" => v.is_boolean(),
        "array" => v.is_array(),
        "object" => v.is_object(),
        "null" => v.is_null(),
        _ => true,
    }
}

/// How a value reads in an error: short values in full, others by kind.
fn shown(v: &Value) -> String {
    match v {
        Value::Array(_) | Value::Object(_) => kind(v).to_string(),
        Value::String(s) if s.len() > 40 => kind(v).to_string(),
        _ => v.to_string(),
    }
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(n) if n.is_f64() => "a number",
        Value::Number(_) => "an integer",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

fn or_list(ts: &[String]) -> String {
    let named: Vec<String> = ts.iter().map(|t| match t.as_str() {
        "integer" => "an integer".into(),
        "array" => "an array".into(),
        "object" => "an object".into(),
        "null" => "null".into(),
        t => format!("a {t}"),
    }).collect();
    match named.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} or {last}", rest.join(", ")),
        _ => named.concat(),
    }
}

fn num(x: f64) -> String {
    if x.fract() == 0.0 { format!("{}", x as i64) } else { format!("{x}") }
}

fn join(path: &str, key: &str) -> String {
    if path.is_empty() { key.to_string() } else { format!("{path}.{key}") }
}

/// The option `s` was most likely meant to be: a small edit away, or the
/// same word with a different ending.
fn closest<'a>(s: &str, options: &[&'a str]) -> Option<&'a str> {
    let s = s.to_lowercase();
    let best = options.iter().map(|o| (edit_distance(&s, &o.to_lowercase()), *o)).min_by_key(|(d, _)| *d)?;
    if best.0 <= (s.len().max(best.1.len()) / 3).max(1) {
        return Some(best.1);
    }
    let stem = |w: &str| w.chars().take(4).collect::<String>();
    let starts: Vec<&str> = options.iter().copied().filter(|o| s.len() >= 4 && stem(o) == stem(&s)).collect();
    (starts.len() == 1).then(|| starts[0])
}

fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut prev = row[0];
        row[0] = i;
        for j in 1..=b.len() {
            let cur = row[j];
            row[j] = (prev + usize::from(a[i - 1] != b[j - 1])).min(row[j] + 1).min(row[j - 1] + 1);
            prev = cur;
        }
    }
    row[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// One ordinary call per tool, using most of its arguments.
    fn examples() -> Vec<(&'static str, Value)> {
        vec![
            ("get_scene_info", json!({})),
            ("new_project", json!({ "name": "hut", "size": [32, 32, 24], "fill": 0, "overwrite": true })),
            ("list_projects", json!({})),
            ("open_project", json!({ "name": "hut" })),
            ("new_scene", json!({ "size": [40, 40, 40], "fill": "#ff0000" })),
            ("open_file", json!({ "path": "a.vox" })),
            ("save_file", json!({ "path": "a.vox" })),
            ("set_voxels", json!({ "voxels": [[1, 2, 3, 4], [0, 0, 0, "#00ff00"]], "part": "body" })),
            ("fill_box", json!({ "from": [0, 0, 0], "to": [3, 3, 3], "color": 5, "hollow": true })),
            ("draw_shape", json!({ "shape": "cylinder", "from": [0, 0, 0], "to": [9, 9, 9], "color": 2, "axis": "x", "top_scale": 0.5, "radius": 2, "up": "z", "rotation": [0, 30.5, 0], "hollow": false })),
            ("draw_line", json!({ "from": [0, 0, 0], "to": [5, 5, 5], "color": 1, "thickness": 2 })),
            ("get_voxels", json!({ "from": [0, 0, 0], "to": [9, 9, 9], "limit": 100 })),
            ("get_slice", json!({ "axis": "z", "index": 3, "from": [0, 0, 0], "to": [9, 9, 0], "max_cells": 64 })),
            ("edit_model", json!({ "op": "loop", "axis": "y", "direction": -1 })),
            ("transform_model", json!({ "kind": "scale", "args": "xyz 2" })),
            ("resize_canvas", json!({ "size": [64, 64, 64] })),
            ("list_shaders", json!({})),
            ("run_shader", json!({ "name": "poly", "args": [3, 1.5], "color": 4 })),
            ("get_palette", json!({ "all": true })),
            ("set_palette_colors", json!({ "colors": [{ "index": 1, "color": "#ff8800" }, { "index": 255, "color": "#000000" }] })),
            ("load_palette", json!({ "page": 1, "match": true })),
            ("edit_palette", json!({ "action": "name_ramp", "row": 3, "name": "skin" })),
            ("set_material", json!({ "color": 7, "type": "glass", "roughness": 0.1, "metallic": 0, "ior": 1.5, "transparency": 0.9, "emission": 0, "density": 0.5 })),
            ("select", json!({ "mode": "box", "from": [0, 0, 0], "to": [3, 3, 3], "combine": "add" })),
            ("move_selection", json!({ "offset": [1, 0, -1] })),
            ("undo", json!({ "steps": 2 })),
            ("redo", json!({})),
            ("set_camera", json!({ "frame": true, "yaw": 40, "pitch": 30, "distance": 80, "target": [10.5, 10, 0], "projection": "orthographic", "fov": 45 })),
            ("set_render_settings", json!({ "sun_pitch": 40, "sun_intensity": 1.5, "sky": "atmosphere", "gamma": 2.2, "denoise": true, "background": "color", "background_color": "#202020" })),
            ("render", json!({ "width": 640, "height": 480, "samples": 32, "views": ["front", "3/4"], "projection": "orthographic", "save_path": "r.png" })),
            ("export", json!({ "format": "gltf", "path": "a.glb" })),
            ("run_script", json!({ "source": "1 + 1", "args": { "n": 3 }, "timeout": 10 })),
            ("set_view_mode", json!({ "mode": "render" })),
            ("list_parts", json!({})),
            ("edit_part", json!({ "part": "world" })),
            ("create_part", json!({ "name": "arm", "size": [4, 4, 12], "position": [10, 0, 0], "parent": 1, "fill": 3 })),
            ("transform_part", json!({ "part": "arm", "move": [1, 0, 0], "rotate_axis": "z", "quarter_turns": -1, "free_rotation": [0, 0, 15], "attach_to": "body:shoulder_r", "attach_socket": "grip", "link": true, "frame": 2 })),
            ("set_animation", json!({ "frames": 24, "fps": 12, "loop": true, "smooth": true })),
            ("go_to_frame", json!({ "frame": 3 })),
            ("set_keyframe", json!({ "part": "arm", "frame": 6, "free_rotation": [20, 0, 0], "attach_to": "body" })),
            ("delete_keyframe", json!({ "part": "arm", "frame": 6 })),
            ("new_drawing", json!({ "part": "head", "frame": 4 })),
            ("remove_drawing", json!({ "part": "head", "frame": 4 })),
            ("play", json!({ "playing": false, "from_frame": 0 })),
            ("frame_strip", json!({ "frames": [0, 6, 12], "width": 200, "height": 200, "columns": 3 })),
            ("create_rig", json!({ "template": "humanoid", "prefix": "hero", "scale": 2, "position": [0, 0, 0] })),
            ("apply_motion", json!({ "motion": "walk", "rig": "hero", "amount": 0.8, "frames": 24, "fps": 12 })),
            ("import_animation", json!({ "path": "walk.glb", "clip": 0, "rig": "hero", "root_motion": false })),
            ("set_pivot", json!({ "part": "arm", "pivot": "bottom" })),
            ("set_socket", json!({ "part": "arm", "name": "grip", "position": [1.5, 1.5, 0] })),
            ("modify_part", json!({ "part": "arm", "action": "duplicate", "name": "arm 2", "offset": [6, 0, 0] })),
            ("group_parts", json!({ "parts": ["arm", 3], "name": "limbs" })),
            ("set_symmetry", json!({ "axes": ["x"], "center": [10.5, null, null], "y": false })),
            ("draw_tube", json!({ "points": [[0, 0, 0], [5, 5, 5.5], [9, 0, 2]], "color": 3, "radius": 1.5, "end_radius": 0.5, "curve": "bezier", "closed": false })),
            ("extrude_polygon", json!({ "points": [[0, 0], [8, 0], [4, 6]], "depth": [0, 3], "color": 2, "axis": "y", "holes": [[[3, 1], [5, 1], [4, 3]]], "hollow": false })),
            ("save_stamp", json!({ "name": "tree", "from": [0, 0, 0], "to": [5, 5, 9] })),
            ("list_stamps", json!({})),
            ("delete_stamp", json!({ "name": "tree" })),
            ("place_stamp", json!({ "name": "tree", "at": [10, 10, 0], "anchor": "bottom", "rotate": 1, "flip": ["x"], "color": 4, "replace": { "3": "#00ff00" } })),
            ("get_guide", json!({ "topic": "craft" })),
            ("import_file", json!({ "path": "map.png", "read_as": "heightmap", "size": 64, "height": 16, "water": 2, "colors": "terrain", "invert": false })),
            ("get_projection", json!({ "axis": "z", "from_side": "min", "show": "color", "max_cells": 64 })),
            ("checkpoint", json!({ "name": "before" })),
            ("restore", json!({ "name": "before" })),
            ("set_autosave", json!({ "enabled": true, "every": 25 })),
            ("analyze_image", json!({ "path": "a.png", "crop": [0, 0, 99, 99], "threshold": 15 })),
            ("set_reference", json!({ "view": "front", "path": "a.png", "blob": 1, "height": 40, "origin": [20, 0], "flip": false, "name": "knight front" })),
            ("list_references", json!({})),
            ("reference_profile", json!({ "view": "left", "height": 40, "origin": [0, 0] })),
            ("compare_reference", json!({ "view": "front", "align": "pinned", "part": "body" })),
            ("paint_from_reference", json!({ "view": "front" })),
            ("vary_colors", json!({ "with": [3, 4, "#00ff00"], "weights": [1, 2, 1], "pattern": "noise", "scale": 4, "seed": 7 })),
            ("shade_edges", json!({ "mode": "cavity", "radius": 2, "levels": 2 })),
            ("paint_faces", json!({ "direction": ["top", "+x"], "color": 5, "depth": 2, "coverage": 0.5, "pattern": "dither", "seed": 1 })),
            ("list_palette_presets", json!({})),
            ("apply_palette_preset", json!({ "name": "pico8", "start_index": 9, "overwrite": false })),
            ("voxelize_image", json!({ "path": "sprite.png", "mode": "extrude", "depth": 2, "plane": "xz", "alpha_threshold": 128, "palette_range": [1, 32], "offset": [0, 0, 0] })),
            ("screenshot", json!({ "views": "quad", "width": 800, "height": 600, "scope": "part", "focus": { "from": [0, 0, 0], "to": [5, 5, 5] }, "highlight_last_edit": true })),
        ]
    }

    fn err(name: &str, args: Value) -> String {
        format!("{:#}", check_call(name, &args).unwrap_err())
    }

    const KEYWORDS: &[&str] = &["type", "description", "enum", "items", "minItems", "maxItems", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "properties", "required", "anyOf", "additionalProperties"];
    const TYPES: &[&str] = &["integer", "number", "string", "boolean", "array", "object", "null"];

    /// A property schema uses only what the validator understands.
    fn well_formed(tool: &str, path: &str, s: &Value) {
        let obj = s.as_object().unwrap_or_else(|| panic!("{tool}.{path}: schema is not an object"));
        for k in obj.keys() {
            assert!(KEYWORDS.contains(&k.as_str()), "{tool}.{path}: unsupported keyword {k}");
        }
        if let Some(ts) = types(s) {
            assert!(!ts.is_empty() && ts.iter().all(|t| TYPES.contains(t)), "{tool}.{path}: bad type {:?}", s["type"]);
            if ts.contains(&"array") {
                assert!(s.get("items").is_some(), "{tool}.{path}: array without items");
            }
        } else {
            assert!(s.get("anyOf").is_some() || obj.keys().all(|k| k == "description"), "{tool}.{path}: no type");
        }
        if let Some(e) = s.get("enum") {
            assert!(e.as_array().is_some_and(|e| !e.is_empty()), "{tool}.{path}: empty enum");
        }
        if let Some(i) = s.get("items") {
            well_formed(tool, &format!("{path}[]"), i);
        }
        for b in s.get("anyOf").and_then(Value::as_array).into_iter().flatten() {
            well_formed(tool, path, b);
        }
        if let Some(p) = s.get("properties").and_then(Value::as_object) {
            for (k, v) in p {
                well_formed(tool, &format!("{path}.{k}"), v);
            }
            for r in s.get("required").and_then(Value::as_array).into_iter().flatten() {
                assert!(p.contains_key(r.as_str().unwrap()), "{tool}.{path}: required {r} isn't a property");
            }
        }
    }

    #[test]
    fn every_schema_is_well_formed() {
        for (name, schema) in schemas() {
            assert_eq!(schema["type"], "object", "{name}");
            let props = schema["properties"].as_object().unwrap_or_else(|| panic!("{name}: no properties"));
            for (k, v) in props {
                well_formed(name, k, v);
            }
            for r in schema["required"].as_array().unwrap() {
                assert!(props.contains_key(r.as_str().unwrap()), "{name}: required {r} isn't a property");
            }
        }
    }

    #[test]
    fn every_tool_accepts_an_ordinary_call() {
        let ex = examples();
        for name in schemas().keys() {
            assert!(ex.iter().any(|(n, _)| n == name), "add an example call for {name}");
        }
        for (name, args) in ex {
            assert!(schemas().contains_key(name), "{name} isn't a tool");
            if let Err(e) = check_call(name, &args) {
                panic!("{name} {args}: {e:#}");
            }
        }
    }

    #[test]
    fn unknown_parameters_are_rejected() {
        assert_eq!(err("play", json!({ "stop": true })), "unknown parameter \"stop\" for play; valid: from_frame, playing");
        assert!(err("play", json!({ "playng": false })).contains("(did you mean \"playing\"?)"));
        assert!(err("fill_box", json!({ "from": [0, 0, 0], "to": [1, 1, 1], "colour": 3, "color": 3 })).contains("did you mean \"color\""));
        assert!(err("list_parts", json!({ "part": "x" })).contains("takes no parameters"));
        // Nested objects with properties are checked too; free-form ones aren't.
        assert!(err("set_palette_colors", json!({ "colors": [{ "index": 1, "colour": "#fff000" }] })).contains("in `colors[0]`"));
        assert!(check_call("place_stamp", &json!({ "name": "a", "at": [0, 0, 0], "replace": { "anything": 1 } })).is_ok());
        assert!(check_call("run_script", &json!({ "source": "1", "args": [1, "x", { "y": null }] })).is_ok());
        // Unknown tools are left to dispatch, which suggests names.
        assert!(check_call("fill_boxes", &json!({ "x": 1 })).is_ok());
    }

    #[test]
    fn types_ranges_and_options_are_enforced() {
        let at = |from: Value| json!({ "from": from, "to": [1, 1, 1], "color": 1 });
        assert_eq!(err("fill_box", at(json!([1.5, 0, 0]))), "`from[0]` must be an integer, got 1.5");
        assert_eq!(err("fill_box", at(json!([0, 0]))), "`from` must have exactly 3 items, got 2");
        assert!(err("fill_box", at(json!("0,0,0"))).contains("must be an array"));
        assert_eq!(err("fill_box", json!({ "from": [0, 0, 0], "to": [1, 1, 1] })), "missing required parameter `color`");
        assert!(err("fill_box", json!({ "from": [0, 0, 0], "to": [1, 1, 1], "color": null })).contains("missing required parameter `color`"));
        assert!(err("fill_box", json!({ "from": [0, 0, 0], "to": [1, 1, 1], "color": 300 })).contains("must be 0 to 255"));
        assert!(err("fill_box", json!({ "from": [0, 0, 0], "to": [1, 1, 1], "color": 1, "hollow": "yes" })).contains("must be a boolean"));
        assert!(err("set_voxels", json!({ "voxels": [[1, 2, 3, 4], [1, 2, 3.5, 4]] })).contains("`voxels[1][2]` must be an integer or a string"));
        assert!(err("set_voxels", json!({ "voxels": [[1, 2, 3]] })).contains("`voxels[0]` must have exactly 4 items"));
        assert_eq!(err("set_camera", json!({ "fov": 200 })), "`fov` is 200; must be 1 to 179");
        assert_eq!(err("set_camera", json!({ "distance": 0 })), "`distance` is 0; must be more than 0");
        assert_eq!(err("set_material", json!({ "color": 1, "emission": -1 })), "`emission` is -1; must be at least 0");
        assert!(err("draw_shape", json!({ "shape": "spher", "from": [0, 0, 0], "to": [1, 1, 1], "color": 1 })).contains("`shape` is \"spher\" (did you mean \"sphere\"?); must be one of: box, ellipsoid, sphere"));
        assert!(err("set_symmetry", json!({ "axes": ["w"] })).contains("`axes[0]` is \"w\""));
        assert!(err("new_scene", json!({ "size": [0, 10, 10] })).contains("`size[0]` is 0; must be 1 to 1024"));
        assert!(err("screenshot", json!({ "focus": { "from": [0, 0, 0] } })).contains("missing required parameter `focus.to`"));
        assert!(err("screenshot", json!({ "focus": 3 })).contains("`focus` must be a string or an object"));
        assert!(err("set_symmetry", json!({ "center": [1, "a", null] })).contains("`center[1]` must be a number or null"));
        assert!(err("get_scene_info", json!([1])).contains("arguments must be an object"));
    }

    #[test]
    fn whole_floats_count_as_integers() {
        let a = check_call("fill_box", &json!({ "from": [2.0, 0, 0], "to": [1, 1, 1], "color": 3.0 })).unwrap();
        assert!(a["from"][0].is_i64() && a["color"].is_i64());
        assert_eq!(a["from"][0], 2);
        // Numbers stay as given; null is the same as leaving it out.
        let b = check_call("set_camera", &json!({ "yaw": 2.0, "pitch": null })).unwrap();
        assert!(b["yaw"].is_f64());
        assert_eq!(check_call("get_scene_info", &Value::Null).unwrap(), json!({}));
    }
}
