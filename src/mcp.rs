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

/// MagicaVoxel's install folder (for shaders, palettes and models).
pub(crate) fn assets_dir(arg: Option<PathBuf>) -> Option<PathBuf> {
    arg.or_else(|| std::env::var_os("VOXELY_ASSETS").map(PathBuf::from)).or_else(|| {
        let home = std::env::var_os("HOME")?;
        let guess = PathBuf::from(home).join("Downloads/MagicaVoxel-0.99.6.2-macos-10.15");
        guess.join("vox").is_dir().then_some(guess)
    })
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
        control::call(&mut ctx, name, args)
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
            other => anyhow::bail!("voxely mcp: unknown option {other}"),
        }
    }
    let assets = assets_dir(assets);
    let mut headless: Option<Headless> = None;
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
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": { "name": "voxely", "version": env!("CARGO_PKG_VERSION") },
                "instructions": control::INSTRUCTIONS,
            })),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": control::tools() })),
            "tools/call" => {
                let name = params.get("name").and_then(Value::as_str).unwrap_or("");
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                let live = if headless_only { None } else { call_live(name, &args) };
                Ok(match live {
                    Some(Ok(v)) => v,
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
                })
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
