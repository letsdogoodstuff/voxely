//! JavaScript scripting (QuickJS). A script runs on its own thread and
//! drives voxely through the same tools agents use over MCP (`control.rs`),
//! plus bulk access to the voxel grid as a `Uint8Array`. Every call is sent
//! back to the thread that owns the scene, so scripts work the same in the
//! window (`run_script`, `voxely script` with a window open) and headless.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc::{self, Sender};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use glam::IVec3;
use rquickjs::context::EvalOptions;
use rquickjs::{CatchResultExt, Context, Ctx, Function, Object, Runtime, TypedArray};
use serde_json::{Value, json};

use crate::control::{self, Ctx as ToolCtx};
use crate::scene::Volume;

/// Default time limit for a script.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// Tools that replace the document (or rewind history), after which a
/// script's edits can't be folded into one undo step.
const RESETS_HISTORY: [&str; 6] = ["new_scene", "new_project", "open_project", "open_file", "undo", "redo"];

/// What the script thread asks the scene's owner for. Volumes belong to a
/// part: the one named (if any), else the part being edited.
enum Request {
    Call(String, Value, Sender<Value>),
    GetVolume(Option<Value>, Sender<Result<(IVec3, Vec<u8>), String>>),
    SetVolume(IVec3, Vec<u8>, Option<Value>, Sender<Value>),
}

fn tool_error(msg: impl std::fmt::Display) -> Value {
    json!({ "content": [{ "type": "text", "text": msg.to_string() }], "isError": true })
}

/// The result of a script run.
pub struct Outcome {
    /// `console.log` lines.
    pub log: Vec<String>,
    /// The value of the script's last expression as JSON (None if
    /// undefined), or the error it threw.
    pub result: Result<Option<String>, String>,
}

/// Run `source` against the scene. Edits made by the script form a single
/// undo step.
pub fn run(ctx: &mut ToolCtx, source: String, file: String, args: Value, timeout: Duration) -> Outcome {
    // Where history stands, to fold the script's edits into one step.
    let (mode, steps) = (ctx.scene.mode, ctx.scene.tree_steps());
    let (volume, mark) = (ctx.scene.volume.clone(), ctx.ed.editor.history_len());
    let world = ctx.scene.synced_world().into_owned();
    let mut squash = true;
    let (tx, rx) = mpsc::channel();
    let js = std::thread::spawn(move || run_js(&source, &file, &args, timeout, tx));
    // Serve the script until it finishes and drops its sender.
    for req in rx {
        match req {
            Request::Call(name, args, reply) => {
                let result = if name == "run_script" {
                    tool_error("run_script can't be called from a script")
                } else {
                    squash &= !RESETS_HISTORY.contains(&name.as_str());
                    control::call(ctx, &name, &args)
                };
                let _ = reply.send(result);
            }
            Request::GetVolume(part, reply) => {
                let result = control::focus_part(ctx, part.as_ref()).map(|()| (ctx.scene.volume.dims, ctx.scene.volume.cells.clone()));
                let _ = reply.send(result.map_err(|e| format!("{e:#}")));
            }
            Request::SetVolume(dims, cells, part, reply) => {
                let result = if dims.cmplt(IVec3::ONE).any() || dims.max_element() > crate::ops::MAX_DIM {
                    tool_error(format!("volume size must be 1-{} per axis", crate::ops::MAX_DIM))
                } else if let Err(e) = control::focus_part(ctx, part.as_ref()) {
                    tool_error(format!("{e:#}"))
                } else {
                    let changed = control::apply(ctx, Volume::from_cells(dims, cells));
                    json!({ "content": control::edited(ctx, changed, "Set volume"), "isError": false })
                };
                let _ = reply.send(result);
            }
        }
    }
    let outcome = js.join().unwrap_or_else(|_| Outcome { log: Vec::new(), result: Err("the script thread panicked".into()) });
    if squash {
        if ctx.scene.mode == mode && ctx.scene.tree_steps() == steps {
            // Voxel edits to the one part being edited.
            ctx.ed.editor.squash(&mut ctx.scene.volume, mark, volume);
        } else {
            // Other parts or the tree changed: one step back to the whole
            // tree as it was.
            ctx.scene.squash_tree(steps, world);
            ctx.ed.editor.reset();
        }
    }
    outcome
}

fn run_js(source: &str, file: &str, args: &Value, timeout: Duration, tx: Sender<Request>) -> Outcome {
    let log = Rc::new(RefCell::new(Vec::new()));
    let result = eval(source, file, args, timeout, tx, log.clone());
    Outcome { log: log.take(), result }
}

fn eval(source: &str, file: &str, args: &Value, timeout: Duration, tx: Sender<Request>, log: Rc<RefCell<Vec<String>>>) -> Result<Option<String>, String> {
    let fail = |e: rquickjs::Error| e.to_string();
    let rt = Runtime::new().map_err(fail)?;
    // A safety net, not a budget: big canvases are big Uint8Arrays.
    rt.set_memory_limit(4 << 30);
    rt.set_max_stack_size(4 << 20);
    let deadline = Instant::now() + timeout;
    rt.set_interrupt_handler(Some(Box::new(move || Instant::now() > deadline)));
    let cx = Context::full(&rt).map_err(fail)?;
    cx.with(|ctx| {
        install(&ctx, tx, log).map_err(fail)?;
        let tools: Vec<Value> = control::tools().as_array().into_iter().flatten().filter_map(|t| t.get("name").cloned()).collect();
        let globals = ctx.globals();
        let set_json = |key: &str, v: &Value| -> rquickjs::Result<()> { globals.set(key, ctx.json_parse(v.to_string())?) };
        set_json("__tools", &Value::Array(tools)).map_err(fail)?;
        set_json("args", args).map_err(fail)?;
        let mut opts = EvalOptions::default();
        opts.filename = Some("voxely.js".into());
        ctx.eval_with_options::<(), _>(PRELUDE, opts).catch(&ctx).map_err(|e| format!("prelude: {e}"))?;
        let mut opts = EvalOptions::default();
        opts.strict = false;
        opts.filename = Some(file.into());
        let value = ctx.eval_with_options::<rquickjs::Value, _>(source, opts).catch(&ctx).map_err(|e| {
            if Instant::now() > deadline {
                return format!("the script ran past its {}s time limit", timeout.as_secs_f32());
            }
            // Keep the stack to the script's own frames.
            e.to_string().lines().filter(|l| !l.contains("(voxely.js:")).collect::<Vec<_>>().join("\n")
        })?;
        if value.is_undefined() {
            return Ok(None);
        }
        Ok(ctx.json_stringify(value).map_err(fail)?.map(|s| s.to_string()).transpose().map_err(fail)?)
    })
}

/// The native functions the prelude builds `voxely` from.
fn install<'js>(ctx: &Ctx<'js>, tx: Sender<Request>, log: Rc<RefCell<Vec<String>>>) -> rquickjs::Result<()> {
    let g = ctx.globals();
    g.set("__log", Function::new(ctx.clone(), move |line: String| log.borrow_mut().push(line))?)?;
    let t = tx.clone();
    g.set(
        "__call",
        Function::new(ctx.clone(), move |name: String, args: String| -> String {
            let args = serde_json::from_str(&args).unwrap_or(json!({}));
            let (reply, rx) = mpsc::channel();
            let _ = t.send(Request::Call(name, args, reply));
            rx.recv().unwrap_or_else(|_| json!({ "content": [], "isError": true })).to_string()
        })?,
    )?;
    let t = tx.clone();
    g.set(
        "__getVolume",
        Function::new(ctx.clone(), move |ctx: Ctx<'js>, part: String| -> rquickjs::Result<Object<'js>> {
            let (reply, rx) = mpsc::channel();
            let _ = t.send(Request::GetVolume(serde_json::from_str(&part).ok(), reply));
            let out = Object::new(ctx.clone())?;
            match rx.recv().unwrap_or_else(|_| Err("voxely stopped".into())) {
                Ok((dims, cells)) => {
                    out.set("size", dims.to_array().to_vec())?;
                    out.set("cells", TypedArray::<u8>::new(ctx, cells)?)?;
                }
                Err(e) => out.set("error", e)?,
            }
            Ok(out)
        })?,
    )?;
    g.set(
        "__setVolume",
        Function::new(ctx.clone(), move |x: i32, y: i32, z: i32, cells: TypedArray<'js, u8>, part: String| -> String {
            // SAFETY: copied out before any more JavaScript runs.
            let cells = unsafe { cells.as_slice() }.to_vec();
            let dims = IVec3::new(x, y, z);
            if cells.len() as i64 != dims.x as i64 * dims.y as i64 * dims.z as i64 {
                return tool_error(format!("cells has {} entries but the size {x}x{y}x{z} needs {}", cells.len(), x as i64 * y as i64 * z as i64)).to_string();
            }
            let (reply, rx) = mpsc::channel();
            let _ = tx.send(Request::SetVolume(dims, cells, serde_json::from_str(&part).ok(), reply));
            rx.recv().unwrap_or_else(|_| json!({ "content": [], "isError": true })).to_string()
        })?,
    )?;
    Ok(())
}

/// Builds `voxely`, `Volume` and `console` on top of the natives.
const PRELUDE: &str = r##"
const fmt = (v) => typeof v === "string" ? v : (() => { try { return JSON.stringify(v); } catch { return String(v); } })();
globalThis.console = Object.fromEntries(["log", "info", "warn", "error", "debug"].map((k) => [k, (...a) => __log(a.map(fmt).join(" "))]));

function unwrap(name, raw) {
  const r = JSON.parse(raw);
  const texts = (r.content || []).filter((c) => c.type === "text").map((c) => c.text);
  if (r.isError) throw new Error(`${name}: ${texts.join("\n") || "failed"}`);
  const images = (r.content || []).filter((c) => c.type === "image").map((c) => ({ data: c.data, mimeType: c.mimeType }));
  const text = texts.join("\n");
  if (images.length) return { text, images };
  try { return JSON.parse(text); } catch { return text; }
}

class Volume {
  constructor(size, cells) {
    this.size = [size[0] | 0, size[1] | 0, size[2] | 0];
    this.cells = cells || new Uint8Array(this.size[0] * this.size[1] * this.size[2]);
  }
  inside(x, y, z) {
    return x >= 0 && y >= 0 && z >= 0 && x < this.size[0] && y < this.size[1] && z < this.size[2];
  }
  index(x, y, z) {
    return x + this.size[0] * (y + this.size[1] * z);
  }
  get(x, y, z) {
    return this.inside(x, y, z) ? this.cells[this.index(x, y, z)] : 0;
  }
  set(x, y, z, color) {
    if (this.inside(x, y, z)) this.cells[this.index(x, y, z)] = color;
  }
  forEach(fn) {
    const [sx, sy, sz] = this.size;
    let i = 0;
    for (let z = 0; z < sz; z++) for (let y = 0; y < sy; y++) for (let x = 0; x < sx; x++, i++) {
      if (this.cells[i]) fn(x, y, z, this.cells[i]);
    }
  }
  clone() {
    return new Volume(this.size, this.cells.slice());
  }
}
globalThis.Volume = Volume;

const camel = (s) => s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
const voxely = {
  call(name, args = {}) {
    return unwrap(name, __call(name, JSON.stringify(args)));
  },
  getVolume({ part = null } = {}) {
    const v = __getVolume(JSON.stringify(part));
    if (v.error) throw new Error(`getVolume: ${v.error}`);
    return new Volume(v.size, v.cells);
  },
  setVolume(volume, { part = null } = {}) {
    return unwrap("setVolume", __setVolume(volume.size[0], volume.size[1], volume.size[2], volume.cells, JSON.stringify(part)));
  },
  /** The nearest palette index (1-255) to "#rrggbb". */
  color(hex) {
    const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
    if (!m) throw new Error(`bad color ${JSON.stringify(hex)}; use "#rrggbb"`);
    const want = [1, 2, 3].map((k) => parseInt(m[k], 16));
    let best = 1, bestD = Infinity;
    for (const p of voxely.call("get_palette", { all: true })) {
      const c = [1, 3, 5].map((k) => parseInt(p.color.slice(k, k + 2), 16));
      const d = c.reduce((s, v, k) => s + (v - want[k]) ** 2, 0);
      if (d < bestD) { best = p.index; bestD = d; }
    }
    return best;
  },
};
for (const name of __tools) {
  if (name !== "run_script") voxely[camel(name)] = (args = {}) => voxely.call(name, args);
}
globalThis.voxely = voxely;
"##;

// ---------------------------------------------------------------- typings

/// TypeScript declarations for the script API (`voxely script --types`).
pub fn typings() -> String {
    let mut out = String::from(
        "// voxely script API. Generated by `voxely script --types`.\n\
         // Coordinates are [x, y, z] with z up; colors are palette indices 1-255 (0 = empty)\n\
         // or \"#rrggbb\" (nearest palette color).\n\n\
         type Vec3 = [number, number, number];\n\
         type Color = number | string;\n\n\
         /** A dense voxel grid: one palette index per cell, x fastest, then y, then z. */\n\
         declare class Volume {\n\
         \x20 constructor(size: Vec3, cells?: Uint8Array);\n\
         \x20 size: Vec3;\n\
         \x20 cells: Uint8Array;\n\
         \x20 inside(x: number, y: number, z: number): boolean;\n\
         \x20 index(x: number, y: number, z: number): number;\n\
         \x20 /** The color at a cell; 0 if empty or outside. */\n\
         \x20 get(x: number, y: number, z: number): number;\n\
         \x20 /** Set a cell (color index 0-255); ignored outside the grid. */\n\
         \x20 set(x: number, y: number, z: number, color: number): void;\n\
         \x20 /** Call fn for every solid voxel. */\n\
         \x20 forEach(fn: (x: number, y: number, z: number, color: number) => void): void;\n\
         \x20 clone(): Volume;\n\
         }\n\n\
         /** Arguments passed after `--` on the command line, or `args` to run_script. */\n\
         declare const args: any;\n\n\
         /** Lines are returned with the script's result. */\n\
         declare const console: { [level in \"log\" | \"info\" | \"warn\" | \"error\" | \"debug\"]: (...values: any[]) => void };\n\n\
         declare const voxely: {\n\
         \x20 /** Call any tool by its MCP name. Throws if the tool fails. */\n\
         \x20 call(name: string, args?: object): any;\n\
         \x20 /** A copy of a part's voxels, in the part's own coordinates. Default: the part being edited. */\n\
         \x20 getVolume(options?: { part?: string | number }): Volume;\n\
         \x20 /** Replace a part's voxels (resizing it to the volume's size). Default: the part being edited. */\n\
         \x20 setVolume(volume: Volume, options?: { part?: string | number }): string;\n\
         \x20 /** The nearest palette index to \"#rrggbb\". */\n\
         \x20 color(hex: string): number;\n",
    );
    for t in control::tools().as_array().into_iter().flatten() {
        let name = t["name"].as_str().unwrap_or_default();
        if name == "run_script" {
            continue;
        }
        let props = t["inputSchema"]["properties"].as_object();
        let required: Vec<&str> = t["inputSchema"]["required"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        let fields: Vec<String> = props
            .into_iter()
            .flatten()
            .map(|(k, s)| {
                let doc = s.get("description").and_then(Value::as_str).map(|d| format!("    /** {} */\n", d.replace("*/", "*\\/"))).unwrap_or_default();
                format!("{doc}    {k}{}: {};\n", if required.contains(&k.as_str()) { "" } else { "?" }, ts_type(s))
            })
            .collect();
        let arg = if fields.is_empty() { String::new() } else { format!("args{}: {{\n{}  }}", if required.is_empty() { "?" } else { "" }, fields.concat()) };
        let camel = name.split('_').enumerate().map(|(i, w)| if i == 0 { w.to_string() } else { w[..1].to_uppercase() + &w[1..] }).collect::<String>();
        let doc = t["description"].as_str().unwrap_or_default().replace("*/", "*\\/");
        out.push_str(&format!("  /** {doc} */\n  {camel}({arg}): any;\n"));
    }
    out.push_str("};\n");
    out
}

fn ts_type(s: &Value) -> String {
    if let Some(e) = s.get("enum").and_then(Value::as_array) {
        return e.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" | ");
    }
    let one = |t: &str| match t {
        "integer" | "number" => "number".to_string(),
        "string" | "boolean" => t.to_string(),
        "array" => match s.get("items") {
            Some(items) if s.get("minItems") == Some(&json!(3)) && s.get("maxItems") == Some(&json!(3)) && ts_type(items) == "number" => "Vec3".into(),
            Some(items) => format!("Array<{}>", ts_type(items)),
            None => "any[]".into(),
        },
        "object" => match s.get("properties").and_then(Value::as_object) {
            Some(p) => format!("{{ {} }}", p.iter().map(|(k, v)| format!("{k}?: {}", ts_type(v))).collect::<Vec<_>>().join("; ")),
            None => "Record<string, any>".into(),
        },
        _ => "any".into(),
    };
    match s.get("type") {
        Some(Value::String(t)) => one(t),
        Some(Value::Array(ts)) => {
            let ts: Vec<String> = ts.iter().filter_map(Value::as_str).map(one).collect();
            if ts == ["number", "string"] { "Color".into() } else { ts.join(" | ") }
        }
        _ => "any".into(),
    }
}

// ---------------------------------------------------------------- cli

const USAGE: &str = "\
usage: voxely script <file.js> [--headless] [--assets <dir>] [--timeout <s>] [-- <args>...]
       voxely script --types > voxely.d.ts

Runs a script in the open voxely window (edits show up live), or headless if
none is open (call voxely.saveFile() to keep the result). Words after `--`
reach the script as the `args` array.";

/// `voxely script ...`
pub fn cli(argv: &[String]) -> Result<()> {
    let (mut file, mut assets, mut headless, mut timeout, mut script_args) = (None, None, false, None, Vec::new());
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--types" => {
                print!("{}", typings());
                return Ok(());
            }
            "--headless" => headless = true,
            "--assets" => assets = it.next().map(std::path::PathBuf::from),
            "--timeout" => timeout = Some(it.next().and_then(|s| s.parse::<f64>().ok()).ok_or_else(|| anyhow!("--timeout needs seconds"))?),
            "--" => script_args.extend(it.by_ref().cloned()),
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            f if file.is_none() && !f.starts_with('-') => file = Some(f.to_string()),
            other => anyhow::bail!("voxely script: unexpected {other:?}\n\n{USAGE}"),
        }
    }
    let file = file.ok_or_else(|| anyhow!("{USAGE}"))?;
    let path = std::fs::canonicalize(&file).map_err(|e| anyhow!("{file}: {e}"))?;
    let args = json!({ "path": path, "args": script_args, "timeout": timeout });
    let live = if headless { None } else { crate::mcp::call_live("run_script", &args) };
    let result = match live {
        Some(r) => r?,
        None => crate::mcp::Headless::new(crate::mcp::assets_dir(assets).as_ref())?.call("run_script", &args),
    };
    let text: Vec<&str> = result["content"].as_array().into_iter().flatten().filter_map(|c| c["text"].as_str()).collect();
    if result["isError"].as_bool().unwrap_or(false) {
        eprintln!("{}", text.join("\n"));
        std::process::exit(1);
    }
    println!("{}", text.join("\n"));
    Ok(())
}

/// Text for a run's MCP result: the log, then the value.
pub fn report(o: &Outcome) -> Result<String> {
    let mut lines = o.log.clone();
    match &o.result {
        Ok(Some(v)) => lines.push(format!("=> {v}")),
        Ok(None) => {}
        Err(e) => return Err(anyhow!("{}{}", lines.iter().map(|l| format!("{l}\n")).collect::<String>(), e)),
    }
    Ok(if lines.is_empty() { "Script finished.".to_string() } else { lines.join("\n") })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run `source` with a stand-in scene owner: a 4x4x4 grid and tools
    /// that echo their name and arguments.
    fn run_fake(source: &str) -> (Outcome, Vec<String>) {
        let (tx, rx) = mpsc::channel();
        let src = source.to_string();
        let js = std::thread::spawn(move || run_js(&src, "test.js", &json!(["a"]), Duration::from_secs(5), tx));
        let mut seen = Vec::new();
        let mut cells = vec![0u8; 64];
        for req in rx {
            match req {
                Request::Call(name, args, reply) => {
                    seen.push(name.clone());
                    let text = if name == "get_palette" {
                        json!([{ "index": 1, "color": "#000000" }, { "index": 7, "color": "#ff0000" }]).to_string()
                    } else {
                        json!({ "tool": name, "args": args }).to_string()
                    };
                    let _ = reply.send(json!({ "content": [{ "type": "text", "text": text }], "isError": name == "fail" }));
                }
                Request::GetVolume(part, reply) => {
                    seen.push(format!("get {}", part.unwrap_or_default()));
                    let _ = reply.send(if cells[0] == 99 { Err("no such part".into()) } else { Ok((IVec3::splat(4), cells.clone())) });
                }
                Request::SetVolume(_, c, part, reply) => {
                    seen.push(format!("set {}", part.unwrap_or_default()));
                    cells = c;
                    let _ = reply.send(json!({ "content": [{ "type": "text", "text": "Set volume" }], "isError": false }));
                }
            }
        }
        seen.push(format!("solid {}", cells.iter().filter(|&&c| c != 0).count()));
        (js.join().unwrap(), seen)
    }

    #[test]
    fn tools_are_camel_case_and_return_json() {
        let (o, seen) = run_fake("console.log('args', args); const r = voxely.fillBox({ from: [0, 0, 0] }); [r.tool, r.args.from]");
        assert_eq!(o.log, vec!["args [\"a\"]"]);
        assert_eq!(o.result, Ok(Some(r#"["fill_box",[0,0,0]]"#.to_string())));
        assert_eq!(seen[0], "fill_box");
    }

    #[test]
    fn volumes_round_trip_and_colors_match() {
        let (o, seen) = run_fake("const v = voxely.getVolume(); v.set(1, 2, 3, voxely.color('#ee1100')); v.set(9, 9, 9, 1); voxely.setVolume(v, { part: 'arm' }); v.get(1, 2, 3)");
        assert_eq!(o.result, Ok(Some("7".into())));
        assert_eq!(seen, ["get null", "get_palette", "set \"arm\"", "solid 1"]);
        let (o, _) = run_fake("const v = new Volume([4, 4, 4]); v.cells[0] = 99; voxely.setVolume(v); voxely.getVolume()");
        assert!(o.result.unwrap_err().contains("getVolume: no such part"));
    }

    #[test]
    fn errors_and_time_limit() {
        let (o, _) = run_fake("voxely.call('fail')");
        assert!(o.result.as_ref().unwrap_err().contains("fail:"), "{:?}", o.result);
        let (tx, _rx) = mpsc::channel();
        let o = run_js("for (;;) {}", "spin.js", &Value::Null, Duration::from_millis(200), tx);
        assert!(o.result.unwrap_err().contains("time limit"));
    }

    #[test]
    fn typings_cover_the_tools() {
        let d = typings();
        assert!(d.contains("  fillBox(args: {") && d.contains("from: Vec3;") && d.contains("color: Color;"));
        assert!(!d.contains("runScript"));
    }
}
