//! The interactive window: modeler and path-traced views, editor and
//! settings panels,
//! brush overlay, and MagicaVoxel-style mouse and keyboard controls.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use glam::{IVec3, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{Window, WindowId};

use crate::edit::{self, Brush, Editor, Pick, Tool};
use crate::gpu::{self, Renderer};
use crate::scene::{MaterialKind, Scene};
use crate::ui;

/// Editing state shared by the window and the offscreen UI snapshot.
pub struct EditState {
    pub editor: Editor,
    pub hover: Option<Pick>,
    /// Editing happens in the modeler; in the renderer the left mouse
    /// button orbits instead (F1 switches).
    pub enabled: bool,
    pub save_path: PathBuf,
}

/// Push voxel edits to the GPU, refreshing emissive faces if any exist.
pub fn sync_volume(r: &mut Renderer, scene: &mut Scene) {
    r.upload_volume(&scene.volume);
    let any_emit = scene.materials.iter().any(|m| m.kind == MaterialKind::Emit as u32 && m.emit > 0.0);
    if any_emit || !scene.lights.is_empty() {
        scene.rebuild_lights();
        r.set_lights(&scene.lights, scene.light_power);
    }
}

pub fn save(scene: &Scene, path: &Path) {
    let result = path.parent().map_or(Ok(()), std::fs::create_dir_all).map_err(anyhow::Error::from).and_then(|_| scene.save(path));
    match result {
        Ok(()) => println!("saved {}", path.display()),
        Err(e) => eprintln!("save failed: {e:#}"),
    }
}

pub fn snapshot(r: &Renderer) {
    let (w, h) = r.size();
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let name = format!("export/voxely-{secs}.png");
    match r.read_pixels().and_then(|px| crate::save_png(Path::new(&name), w, h, &px)) {
        Ok(()) => println!("saved {name}"),
        Err(e) => eprintln!("snapshot failed: {e:#}"),
    }
}

fn outline(p: &egui::Painter, r: &Renderer, ppp: f32, lo: Vec3, hi: Vec3, stroke: egui::Stroke) {
    let corner = |i: u32| Vec3::new(if i & 1 != 0 { hi.x } else { lo.x }, if i & 2 != 0 { hi.y } else { lo.y }, if i & 4 != 0 { hi.z } else { lo.z });
    let pts: Vec<Option<egui::Pos2>> = (0..8).map(|i| r.project(corner(i)).map(|(x, y)| egui::pos2(x / ppp, y / ppp))).collect();
    for a in 0..8u32 {
        for bit in [1, 2, 4] {
            let b = a | bit;
            if b != a {
                if let (Some(pa), Some(pb)) = (pts[a as usize], pts[b as usize]) {
                    p.line_segment([pa, pb], stroke);
                }
            }
        }
    }
}

/// Model bounds and the cells the current brush would touch.
fn draw_overlay(p: &egui::Painter, r: &Renderer, scene: &Scene, ed: &EditState, ppp: f32) {
    // Only draw over the 3D view, not the panels.
    let (x, y, w, h) = r.viewport();
    let p = &p.with_clip_rect(egui::Rect::from_min_size(egui::pos2(x as f32, y as f32) / ppp, egui::vec2(w as f32, h as f32) / ppp));
    // Like MagicaVoxel, the bounds show only in the modeler, and can be hidden.
    let Some(m) = r.modeler.as_ref().filter(|_| r.modeling) else { return };
    if m.options.bounds {
        let dims = scene.volume.dims.as_vec3();
        outline(p, r, ppp, Vec3::ZERO, dims, egui::Stroke::new(1.0, egui::Color32::from_white_alpha(60)));
    }
    if !ed.enabled {
        return;
    }
    let cells = ed.editor.preview(&scene.volume, ed.hover);
    if cells.is_empty() {
        return;
    }
    let color = match ed.editor.tool {
        Tool::Attach => egui::Color32::WHITE,
        Tool::Erase => egui::Color32::from_rgb(255, 80, 80),
        Tool::Paint => {
            let c = scene.palette[ed.editor.color as usize];
            egui::Color32::from_rgb(c[0], c[1], c[2])
        }
    };
    let stroke = egui::Stroke::new(1.5, color);
    if cells.len() <= 64 {
        for c in &cells {
            outline(p, r, ppp, c.as_vec3(), c.as_vec3() + Vec3::ONE, stroke);
        }
    } else {
        let lo = cells.iter().fold(IVec3::MAX, |a, c| a.min(*c));
        let hi = cells.iter().fold(IVec3::MIN, |a, c| a.max(*c));
        outline(p, r, ppp, lo.as_vec3(), (hi + IVec3::ONE).as_vec3(), stroke);
    }
}

/// Lay out both panels and the overlay, then apply what they asked for.
/// Fit the renderer to the area between the panels (in points).
pub fn fit_viewport(r: &mut Renderer, rect: egui::Rect, ppp: f32, window: (u32, u32)) {
    let x = (rect.min.x * ppp).round().clamp(0.0, window.0 as f32 - 1.0) as u32;
    let y = (rect.min.y * ppp).round().clamp(0.0, window.1 as f32 - 1.0) as u32;
    let w = ((rect.width() * ppp).round() as u32).clamp(1, window.0 - x);
    let h = ((rect.height() * ppp).round() as u32).clamp(1, window.1 - y);
    if r.viewport() != (x, y, w, h) {
        r.set_viewport(x, y, w, h);
    }
}

pub fn build_ui(
    ui: &mut egui::Ui,
    r: &mut Renderer,
    scene: &mut Scene,
    ed: &mut EditState,
    panel: &ui::PanelState,
    show: bool,
) -> ui::Actions {
    let ppp = ui.ctx().pixels_per_point();
    let (spp, size) = (r.frame, r.size());
    let mut actions = ui::Actions::default();
    let mut edit = ui::EditActions::default();
    let before = r.settings.clone();
    if show {
        egui::Panel::left("editor").default_size(190.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                if r.modeler.is_some() {
                    let was = r.modeling;
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut r.modeling, true, "Model").on_hover_text("F1");
                        ui.selectable_value(&mut r.modeling, false, "Render").on_hover_text("F1");
                    });
                    if r.modeling != was {
                        ed.enabled = r.modeling;
                        r.reset();
                    }
                    ui.separator();
                }
                edit = ui::editor_panel(ui, &mut ed.editor, scene, r.modeling);
            });
        });
        egui::Panel::right("settings").default_size(280.0).show(ui, |ui| {
            match (r.modeling, r.modeler.as_mut()) {
                (true, Some(m)) => actions = ui::display_panel(ui, &mut m.options),
                _ => actions = ui::settings_panel(ui, &mut r.settings, panel, spp, size),
            }
        });
    }
    actions.viewport = Some(ui.available_rect_before_wrap());
    draw_overlay(&ui.ctx().layer_painter(egui::LayerId::background()), r, scene, ed, ppp);

    if r.settings != before {
        r.reset();
    }
    if edit.materials_changed {
        r.upload_materials(&scene.materials);
        r.set_palette(&scene.palette);
        scene.rebuild_lights();
        r.set_lights(&scene.lights, scene.light_power);
    }
    if (edit.undo && ed.editor.undo(&mut scene.volume)) || (edit.redo && ed.editor.redo(&mut scene.volume)) {
        sync_volume(r, scene);
    }
    if edit.save {
        save(scene, &ed.save_path);
    }
    actions
}

/// Paint one egui frame on top of `view`.
pub fn paint_egui(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    egui_renderer: &mut egui_wgpu::Renderer,
    ctx: &egui::Context,
    view: &wgpu::TextureView,
    size_in_pixels: [u32; 2],
    out: egui::FullOutput,
) {
    let jobs = ctx.tessellate(out.shapes, out.pixels_per_point);
    let screen = egui_wgpu::ScreenDescriptor { size_in_pixels, pixels_per_point: out.pixels_per_point };
    for (id, deltas) in &out.textures_delta.set {
        for delta in deltas {
            egui_renderer.update_texture(device, queue, *id, delta);
        }
    }
    let mut enc = device.create_command_encoder(&Default::default());
    let extra = egui_renderer.update_buffers(device, queue, &mut enc, &jobs, &screen);
    {
        let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        egui_renderer.render(&mut pass.forget_lifetime(), &jobs, &screen);
    }
    queue.submit(extra.into_iter().chain([enc.finish()]));
    for id in &out.textures_delta.free {
        egui_renderer.free_texture(id);
    }
}

struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

#[derive(Default)]
struct Mouse {
    left: bool,
    right: bool,
    middle: bool,
    space: bool,
    pos: Option<(f64, f64)>,
}

pub struct App {
    scene: Scene,
    title: String,
    gfx: Option<Gfx>,
    mouse: Mouse,
    modifiers: ModifiersState,
    panel: ui::PanelState,
    show_panel: bool,
    edit: EditState,
}

impl App {
    pub fn new(scene: Scene, title: String, panel: ui::PanelState, save_path: PathBuf) -> Self {
        // Start on the most used color in the model, like picking it up.
        let mut counts = [0usize; 256];
        for &c in &scene.volume.cells {
            counts[c as usize] += 1;
        }
        let color = (1..256).max_by_key(|&i| counts[i]).filter(|&i| counts[i] > 0).unwrap_or(1) as u8;
        App {
            scene,
            title,
            gfx: None,
            mouse: Mouse::default(),
            modifiers: ModifiersState::empty(),
            panel,
            show_panel: true,
            edit: EditState { editor: Editor::new(color), hover: None, enabled: true, save_path },
        }
    }

    fn init(&mut self, el: &ActiveEventLoop) -> Result<()> {
        let window = Arc::new(
            el.create_window(
                Window::default_attributes()
                    .with_title(&self.title)
                    .with_inner_size(winit::dpi::LogicalSize::new(1280, 800)),
            )?,
        );
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(
            el.owned_display_handle(),
        )));
        let surface = instance.create_surface(window.clone())?;
        let (adapter, device, queue) = pollster::block_on(gpu::request_device(&instance, Some(&surface)))?;
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .context("surface not supported by adapter")?;
        // We apply gamma ourselves, so prefer a non-sRGB format.
        let caps = surface.get_capabilities(&adapter);
        if let Some(f) = caps.formats.iter().find(|f| !f.is_srgb()) {
            config.format = *f;
        }
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);
        let egui_renderer = egui_wgpu::Renderer::new(&device, config.format, egui_wgpu::RendererOptions::default());
        let renderer = Renderer::new(device, queue, &self.scene, Some(config.format), config.width, config.height);
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        self.gfx = Some(Gfx { window, surface, config, renderer, egui_ctx, egui_state, egui_renderer });
        Ok(())
    }

    fn ray(&self) -> Option<(Vec3, Vec3)> {
        let (g, (x, y)) = (self.gfx.as_ref()?, self.mouse.pos?);
        Some(g.renderer.camera_ray(x as f32, y as f32))
    }

    /// Upload the volume after an edit.
    fn sync(&mut self) {
        if let Some(g) = self.gfx.as_mut() {
            sync_volume(&mut g.renderer, &mut self.scene);
        }
    }

    fn update_hover(&mut self) {
        self.edit.hover = self.ray().and_then(|(ro, rd)| edit::pick(&self.scene.volume, ro, rd));
    }

    fn key(&mut self, key: Key) {
        let Some(g) = self.gfx.as_mut() else { return };
        let cmd = self.modifiers.control_key() || self.modifiers.super_key();
        let shift = self.modifiers.shift_key();
        let ed = &mut self.edit.editor;
        let r = &mut g.renderer;
        let mut volume_changed = false;
        match key.as_ref() {
            Key::Character(c) if cmd => match c.to_ascii_lowercase().as_str() {
                "z" if shift => volume_changed = ed.redo(&mut self.scene.volume),
                "z" => volume_changed = ed.undo(&mut self.scene.volume),
                "y" => volume_changed = ed.redo(&mut self.scene.volume),
                "s" => save(&self.scene, &self.edit.save_path),
                "g" | "f" | "b" => {
                    if let Some(m) = r.modeler.as_mut() {
                        let o = &mut m.options;
                        match c.to_ascii_lowercase().as_str() {
                            "g" => o.grid ^= true,
                            "f" => o.frame ^= true,
                            _ => o.bounds ^= true,
                        }
                    }
                }
                _ => {}
            },
            // MagicaVoxel's model-editor hotkeys.
            Key::Character("v") => ed.brush = Brush::Voxel,
            Key::Character("f") => ed.brush = Brush::Face,
            Key::Character("b") => ed.brush = Brush::Box,
            Key::Character("l") => ed.brush = Brush::Line,
            Key::Character("t") => ed.tool = Tool::Attach,
            Key::Character("r") => ed.tool = Tool::Erase,
            Key::Character("g") => ed.tool = Tool::Paint,
            Key::Character("1") => ed.color = ed.color.saturating_sub(1).max(1),
            Key::Character("2") => ed.color = ed.color.saturating_add(1),
            Key::Character("6") => snapshot(r),
            Key::Named(NamedKey::F5) => ed.mirror[0] ^= true,
            Key::Named(NamedKey::F6) => ed.mirror[1] ^= true,
            Key::Named(NamedKey::F7) => ed.mirror[2] ^= true,
            Key::Named(NamedKey::F1) => {
                r.modeling ^= true;
                self.edit.enabled = r.modeling;
                r.reset();
            }
            Key::Named(NamedKey::F12) => {
                r.camera = gpu::Camera::framing(self.scene.volume.dims.as_vec3());
                r.reset();
            }
            Key::Named(NamedKey::Tab) => self.show_panel ^= true,
            Key::Named(NamedKey::Space) => self.mouse.space = true,
            Key::Named(NamedKey::ArrowLeft) => r.settings.sun_yaw -= 10.0,
            Key::Named(NamedKey::ArrowRight) => r.settings.sun_yaw += 10.0,
            Key::Named(NamedKey::ArrowUp) => r.settings.sun_pitch = (r.settings.sun_pitch + 5.0).min(90.0),
            Key::Named(NamedKey::ArrowDown) => r.settings.sun_pitch = (r.settings.sun_pitch - 5.0).max(-90.0),
            _ => return,
        }
        // Sun changes need a fresh accumulation.
        if matches!(key.as_ref(), Key::Named(NamedKey::ArrowLeft | NamedKey::ArrowRight | NamedKey::ArrowUp | NamedKey::ArrowDown)) {
            g.renderer.reset();
        }
        let window = g.window.clone();
        if volume_changed {
            self.sync();
        }
        self.update_hover();
        window.request_redraw();
    }

    fn mouse_moved(&mut self, x: f64, y: f64) {
        let last = self.mouse.pos.replace((x, y));
        let Some(g) = self.gfx.as_mut() else { return };
        let (dx, dy) = last.map(|(lx, ly)| ((x - lx) as f32, (y - ly) as f32)).unwrap_or((0.0, 0.0));
        let cam = &mut g.renderer.camera;
        let editing = self.edit.enabled && !self.mouse.space;
        if self.mouse.right || (self.mouse.left && !editing && !self.edit.editor.active()) && !self.mouse.space {
            cam.yaw -= dx * 0.005;
            cam.pitch = (cam.pitch + dy * 0.005).clamp(-1.55, 1.55);
            g.renderer.reset();
        } else if self.mouse.middle || (self.mouse.left && self.mouse.space) {
            let (_, right, up) = cam.basis();
            let k = cam.dist * 0.0015;
            cam.target += (-right * dx + up * dy) * k;
            g.renderer.reset();
        } else if self.mouse.left && self.edit.editor.active() {
            let (ro, rd) = g.renderer.camera_ray(x as f32, y as f32);
            let hit = edit::pick(&self.scene.volume, ro, rd);
            if self.edit.editor.drag(&mut self.scene.volume, ro, rd, hit) {
                sync_volume(&mut g.renderer, &mut self.scene);
            }
        }
        self.update_hover();
        if let Some(g) = self.gfx.as_ref() {
            g.window.request_redraw();
        }
    }

    fn mouse_button(&mut self, button: MouseButton, down: bool) {
        match button {
            MouseButton::Left => {
                self.mouse.left = down;
                if down && self.edit.enabled && !self.mouse.space {
                    self.update_hover();
                    if let Some(hit) = self.edit.hover {
                        if self.edit.editor.begin(&mut self.scene.volume, hit) {
                            self.sync();
                        }
                    }
                } else if !down && self.edit.editor.active() && self.edit.editor.end(&mut self.scene.volume) {
                    self.sync();
                }
                self.update_hover();
                if let Some(g) = self.gfx.as_ref() {
                    g.window.request_redraw();
                }
            }
            MouseButton::Right => self.mouse.right = down,
            MouseButton::Middle => self.mouse.middle = down,
            _ => {}
        }
    }

    fn redraw(&mut self) {
        let Some(g) = self.gfx.as_mut() else { return };
        let frame = match g.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                g.surface.configure(&g.renderer.device, &g.config);
                g.window.request_redraw();
                return;
            }
            _ => {
                g.window.request_redraw();
                return;
            }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        // Keep refining until the image converges, then idle.
        let samples = if g.renderer.converging() { 1 } else { 0 };
        g.renderer.render_to_view(&view, (g.config.width, g.config.height), samples);

        let raw = g.egui_state.take_egui_input(&g.window);
        let (scene, ed, panel, show) = (&mut self.scene, &mut self.edit, &self.panel, self.show_panel);
        let renderer = &mut g.renderer;
        let mut actions = ui::Actions::default();
        let out = g.egui_ctx.run_ui(raw, |ui| actions = build_ui(ui, renderer, scene, ed, panel, show));
        g.egui_state.handle_platform_output(&g.window, out.platform_output.clone());
        paint_egui(&g.renderer.device, &g.renderer.queue, &mut g.egui_renderer, &g.egui_ctx, &view, [g.config.width, g.config.height], out);
        g.renderer.queue.present(frame);
        if let Some(rect) = actions.viewport {
            let ppp = g.egui_ctx.pixels_per_point();
            fit_viewport(&mut g.renderer, rect, ppp, (g.config.width, g.config.height));
        }

        let ed = &self.edit.editor;
        let title = if g.renderer.modeling {
            format!("{} — Model — {:?} {:?}", self.title, ed.tool, ed.brush)
        } else {
            format!("{} — Render — {} spp", self.title, g.renderer.frame)
        };
        g.window.set_title(&title);
        if actions.reset_camera {
            g.renderer.camera = gpu::Camera::framing(self.scene.volume.dims.as_vec3());
            g.renderer.reset();
        }
        if actions.snapshot {
            snapshot(&g.renderer);
        }
        if g.renderer.converging() {
            g.window.request_redraw();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_none() {
            if let Err(e) = self.init(el) {
                eprintln!("failed to start renderer: {e:#}");
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let Some(g) = self.gfx.as_mut() {
            let resp = g.egui_state.on_window_event(&g.window, &event);
            if resp.repaint {
                g.window.request_redraw();
            }
            let pointer_on_ui = g.egui_ctx.egui_wants_pointer_input();
            let press = matches!(event, WindowEvent::MouseInput { state: ElementState::Pressed, .. } | WindowEvent::MouseWheel { .. });
            let typing = g.egui_ctx.egui_wants_keyboard_input() && matches!(event, WindowEvent::KeyboardInput { .. });
            if (resp.consumed && !matches!(event, WindowEvent::CursorMoved { .. } | WindowEvent::MouseInput { .. })) || (pointer_on_ui && press) || typing {
                return;
            }
        }
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(size) => {
                if let Some(g) = self.gfx.as_mut() {
                    g.config.width = size.width.max(1);
                    g.config.height = size.height.max(1);
                    g.surface.configure(&g.renderer.device, &g.config);
                    g.renderer.set_viewport(0, 0, g.config.width, g.config.height);
                    g.window.request_redraw();
                }
            }
            WindowEvent::ModifiersChanged(m) => self.modifiers = m.state(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    self.key(event.logical_key);
                } else if event.logical_key == Key::Named(NamedKey::Space) {
                    self.mouse.space = false;
                }
            }
            WindowEvent::MouseInput { state, button, .. } => self.mouse_button(button, state == ElementState::Pressed),
            WindowEvent::CursorMoved { position, .. } => self.mouse_moved(position.x, position.y),
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(g) = self.gfx.as_mut() {
                    let d = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y * 0.1,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.005,
                    };
                    g.renderer.camera.dist = (g.renderer.camera.dist * (1.0 - d)).max(1.0);
                    g.renderer.reset();
                    g.window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }
}
