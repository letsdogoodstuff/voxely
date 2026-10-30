//! `voxely mcp`: a Model Context Protocol server on stdio, so agents
//! (Claude Code, Codex, ...) can drive voxely. Each tool call goes to the
//! open voxely window if there is one (so edits show up live), otherwise to
//! a headless session in this process.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::{Value, json};

use crate::app::{self, EditState};
use crate::control::{self, Ctx};
use crate::gpu::{self, Renderer};
use crate::scene::Scene;

/// Where a running window listens for control requests. `VOXELY_SOCKET`
/// overrides it, to keep a second window (and its clients) apart.
pub fn socket_path() -> PathBuf {
    std::env::var_os("VOXELY_SOCKET").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("voxely-control.sock"))
}

/// The control socket's request/reply format. A window answers
/// `{"hello": true}` with this, so `voxely mcp` can tell when the window is
/// from another build.
pub const CONTROL_PROTOCOL: u32 = 1;

/// This program's file and when it was last written: a rebuild changes it.
pub(crate) fn exe_stamp() -> Option<(PathBuf, std::time::SystemTime)> {
    let exe = std::env::current_exe().ok()?;
    let modified = std::fs::metadata(&exe).ok()?.modified().ok()?;
    Some((exe, modified))
}

/// The window's answer to `{"hello": true}`. `started` is `exe_stamp` from
/// when it opened.
pub(crate) fn hello_reply(started: &Option<(PathBuf, std::time::SystemTime)>) -> Value {
    let now = exe_stamp();
    json!({
        "protocol": CONTROL_PROTOCOL,
        "exe": started.as_ref().map(|s| s.0.display().to_string()),
        "rebuilt": matches!((started, &now), (Some(a), Some(b)) if a.1 != b.1),
        "pid": std::process::id(),
    })
}

/// Whether the window and this server run the same, current build. Checked
/// once per window (socket file); the note goes on the first reply from a
/// window and on its errors after that.
#[derive(Default)]
struct BuildCheck {
    started: Option<(PathBuf, std::time::SystemTime)>,
    socket: Option<u64>,
    note: Option<String>,
}

impl BuildCheck {
    fn new() -> Self {
        BuildCheck { started: exe_stamp(), ..Default::default() }
    }

    /// This server's own file was rebuilt after it started.
    fn self_stale(&self) -> Option<String> {
        let (exe, then) = self.started.as_ref()?;
        let (_, now) = exe_stamp()?;
        (now != *then).then(|| {
            format!("This voxely MCP server is older than {} on disk (it was rebuilt since the server started): reconnect the server (/mcp in Claude Code) to get the new tools.", exe.display())
        })
    }

    /// A note for this call's reply from the window, if any: Some on the
    /// first call to a new window, and on errors while something is stale.
    fn window_note(&mut self, is_error: bool) -> Option<String> {
        use std::os::unix::fs::MetadataExt;
        let inode = std::fs::metadata(socket_path()).ok().map(|m| m.ino());
        let fresh = inode != self.socket;
        if fresh {
            self.socket = inode;
            self.note = self.hello();
        }
        let note = [self.note.clone(), self.self_stale()].into_iter().flatten().collect::<Vec<_>>().join(" ");
        (!note.is_empty() && (fresh || is_error)).then_some(note)
    }

    fn hello(&self) -> Option<String> {
        let stream = UnixStream::connect(socket_path()).ok()?;
        stream.set_read_timeout(Some(std::time::Duration::from_secs(2))).ok()?;
        let mut writer = stream.try_clone().ok()?;
        writeln!(writer, "{}", json!({ "hello": true })).ok()?;
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line).ok()?;
        let reply: Value = serde_json::from_str(&line).unwrap_or(Value::Null);
        let restart = "Save, then close and reopen the voxely window to run the current build.";
        let Some(protocol) = reply.get("protocol").and_then(Value::as_u64) else {
            return Some(format!("The open voxely window is from an older build than this MCP server, so tools or arguments added since may fail there. {restart}"));
        };
        if protocol != CONTROL_PROTOCOL as u64 {
            return Some(format!("The open voxely window speaks control protocol {protocol}, this server {CONTROL_PROTOCOL}. {restart}"));
        }
        let mut notes = Vec::new();
        if reply["rebuilt"].as_bool() == Some(true) {
            notes.push(format!("voxely was rebuilt after the open window started, so the window still runs the old code. {restart}"));
        }
        let mine = self.started.as_ref().map(|s| s.0.display().to_string());
        if let (Some(theirs), Some(mine)) = (reply["exe"].as_str(), mine) {
            if theirs != mine {
                notes.push(format!("The open window runs {theirs}; this MCP server is {mine}."));
            }
        }
        (!notes.is_empty()).then(|| notes.join(" "))
    }
}

/// Opt-in local log of how much context tool definitions and results take:
/// set `VOXELY_CONTEXT_LOG=1` (or a file path) in the MCP client's env for
/// this server. One JSON line per tools/list and tools/call, with names and
/// sizes only: never arguments or result content. Nothing is sent anywhere.
struct ContextLog(Option<PathBuf>);

impl ContextLog {
    fn from_env() -> Self {
        ContextLog(std::env::var_os("VOXELY_CONTEXT_LOG").and_then(|v| match v.to_str() {
            Some("" | "0" | "false" | "off") => None,
            Some("1" | "true" | "on" | "yes") => context_log_path(),
            _ => Some(PathBuf::from(v)),
        }))
    }

    fn write(&self, entry: Value) {
        let Some(path) = &self.0 else { return };
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let mut entry = entry;
        entry["t"] = json!(t);
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(f, "{entry}");
        }
    }

    fn call(&self, tool: &str, result: &Value, ms: u128) {
        if self.0.is_none() {
            return;
        }
        let (mut text, mut images, mut image_bytes) = (0, 0, 0);
        for c in result["content"].as_array().into_iter().flatten() {
            match c["type"].as_str() {
                Some("text") => text += c["text"].as_str().map_or(0, str::len),
                Some("image") => {
                    images += 1;
                    image_bytes += c["data"].as_str().map_or(0, str::len);
                }
                _ => {}
            }
        }
        let ok = result["isError"].as_bool() != Some(true);
        self.write(json!({ "kind": "call", "tool": tool, "ok": ok, "text_chars": text, "images": images, "image_b64_chars": image_bytes, "ms": ms as u64 }));
    }
}

/// `context-log.jsonl` beside the preferences file.
fn context_log_path() -> Option<PathBuf> {
    Some(crate::prefs::Prefs::path()?.with_file_name("context-log.jsonl"))
}

/// `voxely mcp --context-report [log]`: the tool definitions' sizes, and
/// per-tool result sizes from a context log.
fn context_report(log: Option<PathBuf>) -> Result<()> {
    let tools = control::tools();
    let list = tools.as_array().map_or(&[][..], Vec::as_slice);
    let total = serde_json::to_string(&tools)?.len();
    let mut sizes: Vec<(usize, &str)> = list.iter().map(|t| (t.to_string().len(), t["name"].as_str().unwrap_or("?"))).collect();
    sizes.sort_by(|a, b| b.cmp(a));
    println!("Instructions: {} chars (~{} tokens)", control::INSTRUCTIONS.len(), control::INSTRUCTIONS.len() / 4);
    println!("Tool definitions: {} tools, {total} chars (~{} tokens). Largest:", list.len(), total / 4);
    for (n, name) in sizes.iter().take(15) {
        println!("  {n:6}  {name}");
    }
    let Some(path) = log.or_else(context_log_path) else { return Ok(()) };
    let Ok(text) = std::fs::read_to_string(&path) else {
        println!("\nNo context log at {} (set VOXELY_CONTEXT_LOG=1 in the MCP client's env for voxely to record one).", path.display());
        return Ok(());
    };
    // tool -> (calls, errors, text chars, image chars)
    let mut per: std::collections::BTreeMap<String, (usize, usize, usize, usize)> = Default::default();
    for e in text.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()).filter(|e| e["kind"] == "call") {
        let r = per.entry(e["tool"].as_str().unwrap_or("?").to_string()).or_default();
        r.0 += 1;
        r.1 += usize::from(e["ok"] == false);
        r.2 += e["text_chars"].as_u64().unwrap_or(0) as usize;
        r.3 += e["image_b64_chars"].as_u64().unwrap_or(0) as usize;
    }
    let mut rows: Vec<_> = per.into_iter().collect();
    rows.sort_by_key(|(_, r)| std::cmp::Reverse(r.2 + r.3));
    println!("\nResults in {} (by total size; images as base64 chars):", path.display());
    println!("  {:<22} {:>6} {:>6} {:>12} {:>12}", "tool", "calls", "errors", "text", "images");
    for (name, (calls, errors, text, img)) in rows {
        println!("  {name:<22} {calls:>6} {errors:>6} {text:>12} {img:>12}");
    }
    Ok(())
}

/// MagicaVoxel's install folder (for shaders, palettes and models).
pub(crate) fn assets_dir(arg: Option<PathBuf>) -> Option<PathBuf> {
    arg.or_else(app::magicavoxel_dir)
}

/// The in-process session used when no window is open.
pub(crate) struct Headless {
    r: Renderer,
    scene: Scene,
    ed: EditState,
}

impl Headless {
    pub(crate) fn new(assets: Option<&PathBuf>) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let (_adapter, device, queue) = pollster::block_on(gpu::request_device(&instance, None))?;
        let scene = Scene::empty(glam::IVec3::splat(40));
        // A modeler (for screenshots) drawing into RGBA textures.
        let mut r = Renderer::new(device, queue, &scene, Some(wgpu::TextureFormat::Rgba8Unorm), 960, 640);
        r.modeling = true;
        let mut ed = EditState::new(1, None, app::default_save_path(None));
        if let Some(dir) = assets {
            // The lists are found relative to a model in MagicaVoxel's vox folder.
            let probe = dir.join("vox").join("_.vox");
            app::set_asset_lists(&mut ed, &probe);
            r.settings.ibl_path = crate::find_ibl(&probe);
        }
        Ok(Headless { r, scene, ed })
    }

    pub(crate) fn call(&mut self, name: &str, args: &Value) -> Value {
        let mut ctx = Ctx { r: &mut self.r, scene: &mut self.scene, ed: &mut self.ed, live: false };
        call_guarded(&mut ctx, name, args)
    }
}

/// `control::call`, with a panic in the tool turned into an error reply
/// so one bad call can't take the server (or the window) down.
pub(crate) fn call_guarded(ctx: &mut Ctx, name: &str, args: &Value) -> Value {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| control::call(ctx, name, args))) {
        Ok(v) => v,
        Err(p) => {
            let why = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_default();
            let msg = format!("internal error in {name}: {why}. It may have been partly applied; check get_scene_info, and undo if needed.");
            json!({ "content": [{ "type": "text", "text": msg }], "isError": true })
        }
    }
}

/// Send a tool call to the open window. None if no window is listening.
pub(crate) fn call_live(name: &str, args: &Value) -> Option<Result<Value>> {
    let stream = UnixStream::connect(socket_path()).ok()?;
    Some((|| {
        let mut writer = stream.try_clone()?;
        writeln!(writer, "{}", json!({ "tool": name, "args": args }))?;
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line)?;
        serde_json::from_str(&line).context("bad reply from the voxely window")
    })())
}

pub fn run(args: &[String]) -> Result<()> {
    let mut assets = None;
    let mut headless_only = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--headless" => headless_only = true,
            "--assets" => assets = it.next().map(PathBuf::from),
            "--context-report" => return context_report(it.next().map(PathBuf::from)),
            other => anyhow::bail!("voxely mcp: unknown option {other}"),
        }
    }
    let assets = assets_dir(assets);
    let mut headless: Option<Headless> = None;
    let mut build = BuildCheck::new();
    let log = ContextLog::from_env();
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_str(&line) {
            Ok(m) => m,
            Err(e) => {
                reply(&mut stdout, json!({ "jsonrpc": "2.0", "id": null, "error": { "code": -32700, "message": e.to_string() } }))?;
                continue;
            }
        };
        let Some(id) = msg.get("id").cloned() else {
            // Notifications (e.g. notifications/initialized) need no reply.
            continue;
        };
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let params = msg.get("params").cloned().unwrap_or(Value::Null);
        let result = match method {
            "initialize" => Ok(json!({
                "protocolVersion": params.get("protocolVersion").and_then(Value::as_str).unwrap_or("2025-06-18"),
                "capabilities": { "tools": { "listChanged": false }, "prompts": { "listChanged": false } },
                "serverInfo": { "name": "voxely", "version": env!("CARGO_PKG_VERSION") },
                "instructions": control::INSTRUCTIONS,
            })),
            "ping" => Ok(json!({})),
            "tools/list" => {
                let tools = control::tools();
                log.write(json!({ "kind": "tools/list", "tools": tools.as_array().map_or(0, Vec::len), "chars": tools.to_string().len(), "instructions_chars": control::INSTRUCTIONS.len() }));
                Ok(json!({ "tools": tools }))
            }
            "prompts/list" => Ok(control::guide::prompts()),
            "prompts/get" => control::guide::prompt(&params).map_err(|e| json!({ "code": -32602, "message": e.to_string() })),
            "tools/call" => {
                let name = params.get("name").and_then(Value::as_str).unwrap_or("");
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                let started = std::time::Instant::now();
                let live = if headless_only { None } else { call_live(name, &args) };
                let r = match live {
                    Some(Ok(mut v)) => {
                        let is_error = v["isError"].as_bool() == Some(true);
                        if let (Some(note), Some(content)) = (build.window_note(is_error), v["content"].as_array_mut()) {
                            content.push(json!({ "type": "text", "text": note }));
                        }
                        v
                    }
                    Some(Err(e)) => json!({ "content": [{ "type": "text", "text": format!("voxely window error: {e:#}") }], "isError": true }),
                    None => {
                        if headless.is_none() {
                            match Headless::new(assets.as_ref()) {
                                Ok(h) => headless = Some(h),
                                Err(e) => {
                                    let msg = format!("couldn't start headless voxely: {e:#}");
                                    reply(&mut stdout, json!({ "jsonrpc": "2.0", "id": id, "result": { "content": [{ "type": "text", "text": msg }], "isError": true } }))?;
                                    continue;
                                }
                            }
                        }
                        headless.as_mut().unwrap().call(name, &args)
                    }
                };
                log.call(name, &r, started.elapsed().as_millis());
                Ok(r)
            }
            _ => Err(json!({ "code": -32601, "message": format!("method not found: {method}") })),
        };
        let response = match result {
            Ok(r) => json!({ "jsonrpc": "2.0", "id": id, "result": r }),
            Err(e) => json!({ "jsonrpc": "2.0", "id": id, "error": e }),
        };
        reply(&mut stdout, response)?;
    }
    Ok(())
}

fn reply(out: &mut std::io::Stdout, v: Value) -> Result<()> {
    let mut lock = out.lock();
    writeln!(lock, "{v}")?;
    lock.flush()?;
    Ok(())
}
