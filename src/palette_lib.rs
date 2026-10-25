//! Palette-wide edits and the palette library: swapping, moving, copying
//! and blending swatches, sorting and packing, loading palettes (with an
//! option to match colors so the model keeps its look), saving them,
//! previewing one before it is applied. Each edit is a step of the normal
//! undo history.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

use crate::app::{EditState, workspace_dir};
use crate::edit::{Editor, SelMode};
use crate::gpu::Renderer;
use crate::palette_ops::{self as ops, SortKey};
use crate::scene::{Mode, Scene, Volume};

/// Where a palette comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    File(PathBuf),
    /// A built-in preset (see `apply_palette_preset`).
    Preset(&'static str),
}

impl Source {
    /// Stable key for favorites: the path, or "preset:<name>".
    pub fn key(&self) -> String {
        match self {
            Source::File(p) => p.display().to_string(),
            Source::Preset(n) => format!("preset:{n}"),
        }
    }

    pub fn from_key(k: &str) -> Option<Source> {
        match k.strip_prefix("preset:") {
            Some(n) => presets::find(n).map(|p| Source::Preset(p.name)),
            None => Some(Source::File(PathBuf::from(k))).filter(|s| matches!(s, Source::File(p) if p.is_file())),
        }
    }

    pub fn name(&self) -> String {
        match self {
            Source::File(p) => p.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            Source::Preset(n) => n.to_string(),
        }
    }

    /// The colors in order; the first goes to color index 1.
    pub fn colors(&self) -> Result<Vec<[u8; 4]>> {
        match self {
            Source::File(p) => ops::read_palette(p),
            Source::Preset(n) => {
                let p = presets::find(n).with_context(|| format!("no preset {n:?}"))?;
                Ok(p.colors.iter().filter_map(|c| ops::parse_color_code(c)).collect())
            }
        }
    }
}

use crate::control::surface::presets;

/// Where saved palettes go: `palettes` in the workspace.
pub fn saved_dir() -> PathBuf {
    workspace_dir().join("palettes")
}

/// Palette files in `dir`, sorted by name.
pub fn list_files(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| ops::is_palette_file(p)).collect();
    out.sort();
    out
}

/// Stop remembering the original colors of `slots` (their colors were
/// edited by hand).
pub fn forget(look: &mut [Option<[u8; 4]>], slots: impl IntoIterator<Item = usize>) {
    for i in slots {
        if let Some(l) = look.get_mut(i) {
            *l = None;
        }
    }
}

/// Start a palette edit as its own undo step, after the voxel edits so
/// far (which stay steps of their own); `Scene::palette_step_update`
/// finishes it.
pub fn begin_step(scene: &mut Scene, editor: &mut Editor) {
    crate::app::keep_history(scene, editor);
    let before = scene.palette_state();
    scene.begin_change();
    scene.palette_step(before);
}

/// For a GUI drag that edits colors over many frames: begin its undo step,
/// unless the drag goes on (`fresh` is false) and the step begun at `mark`
/// is still the latest change.
pub fn drag_step(scene: &mut Scene, editor: &mut Editor, mark: &mut Option<(u64, usize)>, fresh: bool) {
    let open = !fresh && *mark == Some(scene.history_mark()) && !editor.can_undo() && !editor.can_redo();
    if !open {
        begin_step(scene, editor);
        *mark = Some(scene.history_mark());
    }
}

/// Whether the next undo step is a palette edit (for `Cmd::Undo`).
pub fn can_undo(scene: &Scene, editor: &Editor) -> bool {
    !(scene.editing().is_some() && editor.can_undo()) && scene.undoes_palette()
}

/// A palette-wide edit.
#[derive(Clone, Debug, PartialEq)]
pub enum Cmd {
    /// Trade two colors' places; voxels follow, so the model looks the
    /// same [drag, Alt+drag].
    Swap(usize, usize),
    /// Take a color out and put it back in elsewhere, the colors between
    /// sliding over; voxels follow [Ctrl+drag].
    Move(usize, usize),
    /// Copy a color and its material onto another slot [Ctrl+Shift+drag].
    Copy(usize, usize),
    /// Blend the slots between two colors [Shift+drag].
    Gradient(usize, usize),
    Sort(SortKey),
    /// Used colors first, exact duplicates merged.
    Pack,
    /// Load a palette into slots 1... `matched`: recolor the model's own
    /// slots to the nearest new colors instead (moving them out of the
    /// way first when there is room), so the model keeps its look and
    /// loading the old palette again brings its exact colors back;
    /// otherwise voxels keep their indices and take the new colors.
    Load(Source, bool),
    /// Save the palette as `<name>.png` in the saved palettes folder.
    Save(String),
    /// Select (or add, or remove from the selection) the voxels of a color.
    SelectVoxels(usize, SelMode),
    /// Delete the voxels of a color from the part being edited.
    RemoveVoxels(usize),
    /// Undo the last change if it was a palette edit (palette edits are
    /// steps of the normal undo history).
    Undo,
}

/// Run a palette edit. Returns a message and whether voxels changed.
pub fn apply(cmd: Cmd, r: &mut Renderer, scene: &mut Scene, ed: &mut EditState) -> Result<(String, bool)> {
    let slot = |i: usize| -> Result<usize> {
        ensure!((1..=255).contains(&i), "palette slots are 1-255, got {i}");
        Ok(i)
    };
    let out = match cmd {
        Cmd::Swap(a, b) => {
            let (a, b) = (slot(a)?, slot(b)?);
            reorder(r, scene, ed, ops::swap_order(a, b), None);
            (format!("Swapped colors {a} and {b}"), true)
        }
        Cmd::Move(a, b) => {
            let (a, b) = (slot(a)?, slot(b)?);
            reorder(r, scene, ed, ops::move_order(a, b), None);
            (format!("Moved color {a} to {b}"), true)
        }
        Cmd::Copy(a, b) => {
            let (a, b) = (slot(a)?, slot(b)?);
            begin_step(scene, &mut ed.editor);
            scene.look[b] = scene.look[a];
            scene.palette[b] = scene.palette[a];
            scene.materials[b] = scene.materials[a];
            upload(r, scene);
            scene.palette_step_update(None);
            (format!("Copied color {a} onto {b}"), false)
        }
        Cmd::Gradient(a, b) => {
            let (a, b) = (slot(a)?, slot(b)?);
            begin_step(scene, &mut ed.editor);
            forget(&mut scene.look, a.min(b) + 1..a.max(b));
            ops::gradient(&mut scene.palette, a, b);
            for i in a.min(b) + 1..a.max(b) {
                scene.sync_color(i);
            }
            upload(r, scene);
            scene.palette_step_update(None);
            (format!("Blended colors {}-{}", a.min(b), a.max(b)), false)
        }
        Cmd::Sort(key) => {
            let order = ops::sort_order(&scene.palette, &scene.used_colors(), key);
            reorder(r, scene, ed, order, None);
            (format!("Sorted by {}", key.name()), true)
        }
        Cmd::Pack => {
            let used = scene.used_colors();
            // Same color with a different material (glass, metal) stays.
            let keys: Vec<([u8; 4], &[u8])> = scene.palette.iter().zip(&scene.materials).map(|(c, m)| (*c, bytemuck::bytes_of(m))).collect();
            let (order, remap) = ops::compact(&keys, &used);
            let n = used.iter().filter(|&&u| u).count();
            let kept = order.iter().skip(1).take_while(|&&i| used[i]).count();
            let merged = n - kept;
            reorder(r, scene, ed, order, Some(remap));
            (format!("Packed {kept} used colors into 1-{kept}{}", if merged > 0 { format!(", merged {merged} duplicates") } else { String::new() }), true)
        }
        Cmd::Load(src, matched) => {
            let colors = src.colors()?;
            let n = colors.len();
            begin_step(scene, &mut ed.editor);
            if matched {
                // Move the model's colors out of 1..=n (when there's room)
                // so the new palette shows there whole; nothing merges.
                let used = scene.used_colors();
                let order = clear_order(&used, n);
                let moved = order.is_some();
                if let Some(order) = &order {
                    scene.palette = ops::permute(&scene.palette, order);
                    scene.materials = ops::permute(&scene.materials, order);
                    scene.look = ops::permute(&scene.look, order);
                    remap_voxels(scene, ed, &ops::remap_from_order(order));
                }
                let used = scene.used_colors();
                let new = matched_palette(&scene.palette, &scene.look, &used, &colors);
                for c in 1..256 {
                    let look = &mut scene.look[c];
                    *look = if used[c] { Some(look.unwrap_or(scene.palette[c])) } else { None };
                }
                scene.palette = new;
                for c in 1..256 {
                    scene.sync_color(c);
                }
                upload(r, scene);
                scene.palette_step_update(order.as_deref().map(remaps));
                (format!("Loaded {} ({n} colors); the model's colors matched to it", src.name()), moved)
            } else {
                forget(&mut scene.look, 1..=n);
                scene.palette = loaded(&scene.palette, &colors);
                for i in 1..=n {
                    scene.sync_color(i);
                }
                upload(r, scene);
                scene.palette_step_update(None);
                (format!("Loaded {} into colors 1-{n}", src.name()), false)
            }
        }
        Cmd::Save(name) => {
            let name = name.trim();
            ensure!(!name.is_empty() && !name.contains(['/', '\\']), "give the palette a name");
            let path = saved_dir().join(format!("{name}.png"));
            ops::write_palette(&path, &scene.palette)?;
            ed.palette.cache.remove(&Source::File(path.clone()).key());
            ed.palette.saved_scanned = None;
            (format!("Saved {}", path.display()), false)
        }
        Cmd::SelectVoxels(i, mode) => {
            let i = slot(i)?;
            ensure!(matches!(scene.mode, Mode::Edit(_)), "selecting voxels needs edit mode");
            let cells: Vec<usize> = scene.volume.cells.iter().enumerate().filter(|(_, c)| **c as usize == i).map(|(k, _)| k).collect();
            let n = cells.len();
            ed.editor.select(&scene.volume, cells, mode);
            (format!("{n} voxels of color {i}"), false)
        }
        Cmd::RemoveVoxels(i) => {
            let i = slot(i)?;
            ensure!(matches!(scene.mode, Mode::Edit(_)), "removing voxels needs edit mode");
            let v = &scene.volume;
            let n = v.cells.iter().filter(|&&c| c as usize == i).count();
            let new = Volume::from_cells(v.dims, v.cells.iter().map(|&c| if c as usize == i { 0 } else { c }).collect());
            (format!("Removed {n} voxels of color {i}"), ed.editor.commit(&mut scene.volume, new))
        }
        Cmd::Undo => {
            ensure!(can_undo(scene, &ed.editor), "the last change isn't a palette edit; palette edits are steps of the normal undo, so use undo");
            crate::app::history_step(scene, ed, false);
            ("Undid palette change".to_string(), true)
        }
    };
    Ok(out)
}

/// `palette` with entries 1..=n replaced by `colors`.
fn loaded(palette: &[[u8; 4]], colors: &[[u8; 4]]) -> Vec<[u8; 4]> {
    let mut new = palette.to_vec();
    new[1..=colors.len()].copy_from_slice(colors);
    new
}

/// The order that swaps the used colors in 1..=n with unused slots past n,
/// if any are in the way and there are enough free slots.
fn clear_order(used: &[bool], n: usize) -> Option<Vec<usize>> {
    let blocking: Vec<usize> = (1..=n).filter(|&i| used[i]).collect();
    let free: Vec<usize> = (n + 1..256).filter(|&i| !used[i]).collect();
    if blocking.is_empty() || free.len() < blocking.len() {
        return None;
    }
    let mut order: Vec<usize> = (0..256).collect();
    for (&a, &b) in blocking.iter().zip(&free) {
        order.swap(a, b);
    }
    Some(order)
}

/// The palette after loading `colors` while keeping the model's look:
/// unused slots 1..=n take the new colors, and each used slot takes the
/// new color closest to its original (`look`, else its current color).
/// No voxel changes, so different colors stay different slots.
pub fn matched_palette(palette: &[[u8; 4]], look: &[Option<[u8; 4]>], used: &[bool], colors: &[[u8; 4]]) -> Vec<[u8; 4]> {
    let n = colors.len();
    let new = loaded(palette, colors);
    let original: Vec<[u8; 4]> = (0..palette.len()).map(|c| look.get(c).copied().flatten().unwrap_or(palette[c])).collect();
    let nearest = ops::nearest_map(&original, &new, &(1..=n).collect::<Vec<_>>());
    (0..palette.len())
        .map(|c| match (c, used.get(c).copied().unwrap_or(false)) {
            (0, _) => palette[0],
            (_, true) => new[nearest[c] as usize],
            (c, false) if c <= n => new[c],
            _ => palette[c],
        })
        .collect()
}

/// Put colors at new indices (`order`: new index -> old index) and move
/// the voxels, history, selection and recent colors with them. `remap`
/// overrides the voxel remap (for packing, which merges duplicates).
fn reorder(r: &mut Renderer, scene: &mut Scene, ed: &mut EditState, order: Vec<usize>, remap: Option<[u8; 256]>) {
    reorder_scene(scene, ed, order, remap);
    upload(r, scene);
}

/// `reorder` without the GPU upload.
fn reorder_scene(scene: &mut Scene, ed: &mut EditState, order: Vec<usize>, remap: Option<[u8; 256]>) {
    let [fwd, back] = remaps(&order);
    let remap = remap.unwrap_or(fwd);
    begin_step(scene, &mut ed.editor);
    scene.palette = ops::permute(&scene.palette, &order);
    scene.materials = ops::permute(&scene.materials, &order);
    scene.look = ops::permute(&scene.look, &order);
    remap_voxels(scene, ed, &remap);
    scene.palette_step_update(Some([remap, back]));
}

/// The color remaps of `order` (new index -> old index): old to new, and
/// new back to old.
fn remaps(order: &[usize]) -> [[u8; 256]; 2] {
    [ops::remap_from_order(order), std::array::from_fn(|k| order.get(k).copied().unwrap_or(k) as u8)]
}

/// Recolor every voxel (and the brush color, swatch selection and recent
/// colors) through `remap`.
fn remap_voxels(scene: &mut Scene, ed: &mut EditState, remap: &[u8; 256]) {
    scene.remap(remap);
    remap_tools(ed, remap);
}

/// Recolor the voxel editor's brush color, clipboard and history, the
/// swatch selection and recent colors through `remap`.
pub fn remap_tools(ed: &mut EditState, remap: &[u8; 256]) {
    ed.editor.remap_colors(remap);
    ed.tools.surface.remap_colors(remap);
    let p = &mut ed.palette;
    let old = std::mem::replace(&mut p.selected, vec![false; 256]);
    for (i, s) in old.into_iter().enumerate() {
        if s {
            p.selected[remap[i] as usize] = true;
        }
    }
    let mut recent = Vec::new();
    for c in p.recent.iter().map(|&c| remap[c as usize]) {
        if !recent.contains(&c) {
            recent.push(c);
        }
    }
    p.recent = recent;
}

fn upload(r: &mut Renderer, scene: &mut Scene) {
    r.upload_materials(&scene.materials);
    r.set_palette(&scene.palette);
    scene.rebuild_lights();
    r.set_lights(&scene.lights, scene.light_power);
}

/// What the model would look like with `colors` loaded: the palette to
/// draw with (moving colors out of the way doesn't change the look, so
/// the preview skips it).
pub fn preview_palette(scene: &Scene, look: &[Option<[u8; 4]>], colors: &[[u8; 4]], matched: bool) -> Vec<[u8; 4]> {
    if matched { matched_palette(&scene.palette, look, &scene.used_colors(), colors) } else { loaded(&scene.palette, colors) }
}

/// Show the hovered palette on the model, or put the real one back.
/// `force` redraws even if the hovered palette hasn't changed (after the
/// real palette was edited).
pub fn sync_preview(r: &mut Renderer, scene: &mut Scene, ed: &mut EditState, force: bool) {
    let p = &mut ed.palette;
    let want = p.preview.take().map(|s| (s.key(), p.match_colors, s));
    let key = want.as_ref().map(|(k, m, _)| (k.clone(), *m));
    if key == p.previewing && !force {
        return;
    }
    p.previewing = key;
    match want.and_then(|(k, m, s)| p.cache_colors(&k, &s).map(|c| (c, m))) {
        Some((colors, matched)) => {
            let pal = preview_palette(scene, &scene.look, &colors, matched);
            let mut mats = scene.materials.clone();
            for (i, m) in mats.iter_mut().enumerate().skip(1) {
                let c = crate::scene::material(i as u32, pal[i], None).color;
                m.color = c;
            }
            r.upload_materials(&mats);
            r.set_palette(&pal);
        }
        None => {
            r.upload_materials(&scene.materials);
            r.set_palette(&scene.palette);
        }
    }
}

/// Every palette the library offers, by section.
pub fn sections(ed: &EditState) -> Vec<(&'static str, Vec<Source>)> {
    let favorites: Vec<Source> = ed.prefs.palette_favorites.iter().filter_map(|k| Source::from_key(k)).collect();
    vec![
        ("Favorites", favorites),
        ("Saved", ed.palette.saved.iter().cloned().map(Source::File).collect()),
        ("MagicaVoxel", ed.palettes.iter().cloned().map(Source::File).collect()),
        ("Presets", presets::PRESETS.iter().map(|p| Source::Preset(p.name)).collect()),
    ]
}

/// The quick-load buttons: the first four favorites, else MagicaVoxel's
/// palette pages, else the first presets.
pub fn quick(ed: &EditState) -> Vec<Source> {
    let mut out: Vec<Source> = ed.prefs.palette_favorites.iter().filter_map(|k| Source::from_key(k)).take(4).collect();
    if out.is_empty() {
        out = ed.palettes.iter().take(4).cloned().map(Source::File).collect();
    }
    if out.is_empty() {
        out = presets::PRESETS.iter().take(4).map(|p| Source::Preset(p.name)).collect();
    }
    out
}

/// A strip of a palette's colors.
pub fn strip(ui: &mut egui::Ui, colors: &[[u8; 4]], size: egui::Vec2) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    let shown: Vec<&[u8; 4]> = colors.iter().take(64).collect();
    let w = rect.width() / shown.len().max(1) as f32;
    for (k, c) in shown.iter().enumerate() {
        let x = rect.min.x + k as f32 * w;
        let r = egui::Rect::from_min_max(egui::pos2(x, rect.min.y), egui::pos2(x + w + 0.5, rect.max.y));
        ui.painter().rect_filled(r, 0.0, egui::Color32::from_rgb(c[0], c[1], c[2]));
    }
    resp
}

/// The Palettes window: favorites, saved palettes, MagicaVoxel's and the
/// presets. Hover one to see it on the model; click to load it.
pub fn window(ctx: &egui::Context, scene: &Scene, ed: &mut EditState, cmds: &mut Vec<crate::ui::Command>) {
    if !ed.palette.library_open {
        return;
    }
    let undo_ok = can_undo(scene, &ed.editor);
    if ed.palette.saved_scanned.is_none_or(|t| t.elapsed().as_secs() > 2) {
        ed.palette.saved = list_files(&saved_dir());
        ed.palette.saved_scanned = Some(std::time::Instant::now());
    }
    let sections = sections(ed);
    let mut open = true;
    let mut toggle_fav: Option<String> = None;
    egui::Window::new("Palettes").open(&mut open).default_width(250.0).default_pos([170.0, 60.0]).show(ctx, |ui| {
        let p = &mut ed.palette;
        ui.checkbox(&mut p.match_colors, "Keep the model's look").on_hover_text(
            "On: the model's colors change to the closest colors of the new palette, and loading your old palette again brings the exact colors back.\nOff: voxels keep their slot numbers and simply take whatever color the new palette puts there.",
        );
        ui.horizontal(|ui| {
            if ui.button("Load file…").on_hover_text("PNG (MagicaVoxel or Lospec), .act, .gpl or .hex; or drop one on the window").clicked() {
                cmds.push(crate::ui::Command::PalettePick(p.match_colors));
            }
            if ui.add_enabled(undo_ok, egui::Button::new("Undo")).on_hover_text("Undo the last palette change").clicked() {
                cmds.push(crate::ui::Command::Palette(Cmd::Undo));
            }
        });
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut p.save_name).hint_text("name").desired_width(120.0));
            let ok = !p.save_name.trim().is_empty();
            if ui.add_enabled(ok, egui::Button::new("Save current")).on_hover_text(format!("Save this palette to {}", saved_dir().display())).clicked() {
                cmds.push(crate::ui::Command::Palette(Cmd::Save(std::mem::take(&mut p.save_name))));
                p.saved_scanned = None;
            }
        });
        ui.label(egui::RichText::new("Hover to preview on the model, click to load, Pin to put it on the quick buttons 0-3.").weak().small());
        ui.separator();
        egui::ScrollArea::vertical().max_height(420.0).show(ui, |ui| {
            for (title, list) in &sections {
                if list.is_empty() && *title != "Favorites" {
                    continue;
                }
                egui::CollapsingHeader::new(*title).default_open(*title != "Presets" || list.len() < 20).show(ui, |ui| {
                    if list.is_empty() {
                        ui.label(egui::RichText::new("Pin a palette to keep it here").weak());
                    }
                    for src in list {
                        let key = src.key();
                        let fav = ed.prefs.palette_favorites.contains(&key);
                        ui.horizontal(|ui| {
                            if ui.selectable_label(fav, "Pin").on_hover_text(if fav { "Unpin" } else { "Pin to the quick buttons 0-3" }).clicked() {
                                toggle_fav = Some(key.clone());
                            }
                            let colors = ed.palette.cache_colors(&key, src).unwrap_or_default();
                            let name = ui.add(egui::Label::new(src.name()).truncate().sense(egui::Sense::click()));
                            let w = (ui.available_width() - 4.0).clamp(40.0, 110.0);
                            let s = strip(ui, &colors, egui::vec2(w, 12.0));
                            let hovered = name.hovered() || s.hovered();
                            if hovered {
                                ed.palette.preview = Some(src.clone());
                            }
                            if name.clicked() || s.clicked() {
                                cmds.push(crate::ui::Command::Palette(Cmd::Load(src.clone(), ed.palette.match_colors)));
                            }
                            let tip = format!("{} · {} colors", src.name(), colors.len());
                            name.on_hover_text(&tip);
                            s.on_hover_text(tip);
                        });
                    }
                });
            }
        });
    });
    if let Some(k) = toggle_fav {
        let favs = &mut ed.prefs.palette_favorites;
        match favs.iter().position(|f| *f == k) {
            Some(i) => {
                favs.remove(i);
            }
            None => favs.push(k),
        }
        ed.prefs.save();
    }
    ed.palette.library_open = open;
}

/// A picture small enough to be a palette strip rather than a model,
/// a heightmap or a reference.
fn palette_strip(p: &Path) -> bool {
    !crate::import::is_importable(p) || crate::control::surface::load_image(p).is_ok_and(|i| i.h <= 32 || i.w * i.h <= 4096)
}

/// Load palette files dropped on the window, open dropped models, import
/// other formats and pictures (Shift: show a picture as the reference).
pub fn handle_drops(ctx: &egui::Context, ed: &EditState, cmds: &mut Vec<crate::ui::Command>) {
    let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).filter(|p| !p.as_os_str().is_empty()).collect());
    let shift = ctx.input(|i| i.modifiers.shift);
    for p in dropped {
        let picture = crate::import::IMAGES.iter().any(|e| p.extension().is_some_and(|x| x.eq_ignore_ascii_case(e)));
        if shift && picture {
            cmds.push(crate::ui::Command::Reference(crate::backdrop::RefCmd::Load(p)));
        } else if ops::is_palette_file(&p) && palette_strip(&p) {
            cmds.push(crate::ui::Command::Palette(Cmd::Load(Source::File(p), ed.palette.match_colors)));
        } else if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("vox")) {
            cmds.push(crate::ui::Command::OpenPath(p));
        } else if crate::import::is_importable(&p) || p.is_dir() {
            cmds.push(crate::ui::Command::ImportPath(p));
        }
    }
    let hovering = ctx.input(|i| i.raw.hovered_files.iter().any(|f| f.path.as_deref().is_some_and(ops::is_palette_file)));
    if hovering {
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::unique("palette_drop")));
        let rect = ctx.content_rect();
        painter.rect_filled(rect, 0.0, egui::Color32::from_black_alpha(140));
        painter.text(rect.center(), egui::Align2::CENTER_CENTER, "Drop to load this palette", egui::FontId::proportional(22.0), egui::Color32::WHITE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(scene: &mut Scene, ed: &mut EditState, p: glam::IVec3, c: u8) {
        let mut new = scene.volume.clone();
        new.set(p, c);
        assert!(ed.editor.commit(&mut scene.volume, new));
    }

    #[test]
    fn a_pack_is_an_undo_step_in_order_with_voxel_edits() {
        let mut scene = Scene::empty(glam::IVec3::splat(4));
        let mut ed = EditState::new(1, None, PathBuf::from("x.vox"));
        // Colors 5 and 9 are the same; pack merges 9 into 5.
        scene.palette[9] = scene.palette[5];
        scene.sync_color(9);
        set(&mut scene, &mut ed, glam::IVec3::ZERO, 5);
        set(&mut scene, &mut ed, glam::IVec3::X, 200);
        set(&mut scene, &mut ed, glam::IVec3::Y, 9);
        let before = (scene.volume.filled_sorted(), scene.palette.clone());
        ed.editor.color = 200;
        let keys: Vec<([u8; 4], &[u8])> = scene.palette.iter().zip(&scene.materials).map(|(c, m)| (*c, bytemuck::bytes_of(m))).collect();
        let (order, remap) = ops::compact(&keys, &scene.used_colors());
        reorder_scene(&mut scene, &mut ed, order, Some(remap));
        let packed = (scene.volume.filled_sorted(), scene.palette.clone());
        assert_eq!(packed.0.iter().map(|v| v.1).collect::<Vec<_>>(), vec![1, 2, 1]);
        assert_eq!(ed.editor.color, 2, "the brush follows its color");
        assert!(can_undo(&scene, &ed.editor));
        // Undo takes back the pack (merged duplicates split again), then
        // the voxel edits one at a time.
        assert!(crate::app::history_step(&mut scene, &mut ed, false));
        assert_eq!((scene.volume.filled_sorted(), scene.palette.clone()), before);
        assert_eq!(ed.editor.color, 200);
        assert!(!can_undo(&scene, &ed.editor), "the next undo is a voxel edit");
        assert!(crate::app::history_step(&mut scene, &mut ed, false));
        assert_eq!(scene.volume.count(), 2);
        // Redo brings both back, in order.
        assert!(crate::app::history_step(&mut scene, &mut ed, true));
        assert_eq!(scene.volume.filled_sorted(), before.0);
        assert!(crate::app::history_step(&mut scene, &mut ed, true));
        assert_eq!((scene.volume.filled_sorted(), scene.palette.clone()), packed);
        assert!(!crate::app::history_step(&mut scene, &mut ed, true));
    }

    /// Load `colors` the way `Cmd::Load(_, true)` does, without moving
    /// colors (they are all past `colors.len()` here).
    fn load(palette: &mut Vec<[u8; 4]>, look: &mut [Option<[u8; 4]>], used: &[bool], colors: &[[u8; 4]]) {
        let new = matched_palette(palette, look, used, colors);
        for c in 1..256 {
            look[c] = if used[c] { Some(look[c].unwrap_or(palette[c])) } else { None };
        }
        *palette = new;
    }

    #[test]
    fn matched_loads_are_reversible() {
        let mut palette = crate::vox::default_palette();
        let mut look = vec![None; 256];
        let mut used = vec![false; 256];
        // A red, a green and a blue part, all past the loaded palettes.
        for i in [40, 120, 200] {
            used[i] = true;
        }
        palette[40] = [200, 30, 30, 255];
        palette[120] = [30, 200, 30, 255];
        palette[200] = [30, 30, 200, 255];
        let colored = palette.clone();
        // A palette of two grays: every color goes gray...
        let grays = [[60, 60, 60, 255], [180, 180, 180, 255]];
        load(&mut palette, &mut look, &used, &grays);
        assert!([40, 120, 200].iter().all(|&c| grays.contains(&palette[c])));
        assert_eq!(&palette[1..3], &grays, "the new palette fills the front");
        // ...but the slots stay apart, and loading the colors back restores
        // them exactly.
        let back: Vec<[u8; 4]> = colored[1..=255].to_vec();
        load(&mut palette, &mut look, &used, &back);
        for c in [40, 120, 200] {
            assert_eq!(palette[c], colored[c], "slot {c}");
        }
    }

    #[test]
    fn clear_order_moves_used_colors_past_the_new_palette() {
        let mut used = vec![false; 256];
        used[2] = true;
        used[20] = true;
        let order = clear_order(&used, 16).unwrap();
        let remap = ops::remap_from_order(&order);
        assert!(remap[2] > 16 && !used[remap[2] as usize]);
        assert_eq!(remap[20], 20);
        assert!(clear_order(&used, 1).is_none(), "nothing in the way");
        assert!(clear_order(&vec![true; 256], 16).is_none(), "no room");
    }
}
