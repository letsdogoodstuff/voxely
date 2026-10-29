//! The command palette: Ctrl+K (Cmd+K on macOS) opens a search box over
//! the 3D view that lists every command the panels and hotkeys offer,
//! filtered as you type. Up / Down choose, Enter runs, Esc closes.
//!
//! The list is built each frame from what the scene can do right now
//! (voxel edits while editing a model, part operations in World mode),
//! so nothing shows that would only fail.

use crate::app::EditState;
use crate::edit::{Brush, Tool};
use crate::gpu::Renderer;
use crate::ops::Shape;
use crate::outline::TreeOp;
use crate::scene::{Mode, Scene};
use crate::ui::{Command, Display, Op, Panel, View};
use egui::text::LayoutJob;
use egui::{Align2, Color32, Key, Modifiers, RichText, Sense, TextFormat};

/// One row of the palette.
pub struct Entry {
    pub name: String,
    pub group: &'static str,
    /// The hotkey, written the way the tooltips do ("Ctrl+Z"), or "".
    pub keys: &'static str,
    pub cmd: Command,
}

impl Entry {
    fn new(group: &'static str, name: impl Into<String>, keys: &'static str, cmd: Command) -> Entry {
        Entry { name: name.into(), group, keys, cmd }
    }
}

/// How many commands run from the palette it remembers, to offer first.
const RECENT: usize = 8;

#[derive(Default)]
pub struct CommandPalette {
    pub open: bool,
    query: String,
    selected: usize,
    /// Names of commands run from here, newest first.
    recent: Vec<String>,
    /// Focus the search box on the next frame.
    focus: bool,
}

impl CommandPalette {
    pub fn toggle(&mut self) {
        if self.open { self.close() } else { self.show() }
    }

    pub fn show(&mut self) {
        self.open = true;
        self.focus = true;
        self.query.clear();
        self.selected = 0;
    }

    pub fn close(&mut self) {
        self.open = false;
        self.query.clear();
        self.selected = 0;
    }

    fn ran(&mut self, name: &str) {
        self.recent.retain(|n| n != name);
        self.recent.insert(0, name.to_string());
        self.recent.truncate(RECENT);
    }
}

/// Every command that applies right now, in a stable order: file, then
/// what you do to the model, then view and windows.
pub fn entries(r: &Renderer, scene: &Scene, ed: &EditState) -> Vec<Entry> {
    use Entry as E;
    let mut v = Vec::with_capacity(160);
    let modeling = r.modeling;
    let editing = matches!(scene.mode, Mode::Edit(_));

    // File.
    v.push(E::new("File", "New project", "Ctrl+P", Command::New));
    v.push(E::new("File", "Open…", "Ctrl+O", Command::Open));
    v.push(E::new("File", "Import a file… (voxel formats, meshes, heightmaps, slices, pictures)", "Ctrl+Shift+O", Command::ImportPick));
    v.push(E::new("File", "Save", "Ctrl+S", Command::Save));
    v.push(E::new("File", "Save as…", "Ctrl+Shift+S", Command::SaveAs));
    v.push(E::new("File", "Undo", "Ctrl+Z", Command::Undo));
    v.push(E::new("File", "Redo", "Ctrl+Y", Command::Redo));
    v.push(E::new("File", "Preferences", "Ctrl+,", Command::Preferences));
    v.push(E::new("File", "Quit voxely", "", Command::Quit));

    // Mode switches.
    v.push(E::new("Mode", if modeling { "Render (path trace the scene)" } else { "Model (back to editing)" }, "F1", Command::ToggleRender));
    if modeling {
        let (name, hint) = if editing { ("World: arrange whole models and groups", "Tab") } else { ("Edit the picked model's voxels", "Tab") };
        v.push(E::new("Mode", name, hint, Command::ToggleWorld));
    }

    // Selection and clipboard work in both modes (World remaps them to parts).
    let (what, ctx) = if editing { ("voxels", "Select") } else { ("objects", "World") };
    v.push(E::new(ctx, format!("Select all {what}"), "Ctrl+A", Command::SelectAll));
    v.push(E::new(ctx, format!("Select no {what}"), "Ctrl+D", Command::SelectNone));
    v.push(E::new(ctx, "Invert selection", "Ctrl+I", Command::SelectInvert));
    v.push(E::new(ctx, format!("Copy {what}"), "Ctrl+C", Command::Copy));
    if editing {
        v.push(E::new(ctx, "Cut voxels", "Ctrl+X", Command::Cut));
    }
    v.push(E::new(ctx, if editing { "Paste voxels" } else { "Paste (duplicate the copied objects)" }, "Ctrl+V", Command::Paste));

    if modeling && editing {
        // Brushes, modes and tools.
        for (b, name, k) in [
            (Brush::Voxel, "Voxel brush", "V"),
            (Brush::Face, "Face brush", "F"),
            (Brush::Box, "Box brush", "B"),
            (Brush::Geo, "Line / square / circle brush", "L"),
            (Brush::Pattern, "Pattern brush", "P"),
            (Brush::Code, "Code brush (voxel shader in a box)", "C"),
        ] {
            v.push(E::new("Brush", name, k, Command::SetBrush(b)));
        }
        for (t, name, k) in [
            (Tool::Attach, "Attach: add voxels", "T"),
            (Tool::Erase, "Erase: remove voxels", "R"),
            (Tool::Paint, "Paint: recolor voxels", "G"),
            (Tool::Marquee, "Marquee select", "M"),
            (Tool::Region, "Region select", "N"),
            (Tool::RegionRemove, "Region remove", ""),
            (Tool::RegionReplace, "Region recolor", ""),
            (Tool::Move, "Move / transform", ""),
            (Tool::Pick, "Pick color from a voxel", ""),
        ] {
            v.push(E::new("Tool", name, k, Command::SetTool(t)));
        }
        for (a, name, k) in [(0, "Mirror X", "F5"), (1, "Mirror Y", "F6"), (2, "Mirror Z", "F7")] {
            let on = ed.editor.mirror[a];
            v.push(E::new("Brush", format!("{name} {}", if on { "off" } else { "on" }), k, Command::ToggleMirror(a)));
        }
        v.push(E::new("Brush", "Pattern from selection", "", Command::PatternFromSelection));
        if ed.shaders.current.is_some() {
            v.push(E::new("Brush", "Run the chosen voxel shader", "", Command::RunShader));
        }

        // The Edit panel.
        v.push(E::new("Model", "Full: fill the whole canvas", "U", Command::Op(Op::Full)));
        v.push(E::new("Model", "Fill: fill enclosed space", "I", Command::Op(Op::Fill)));
        v.push(E::new("Model", "Clear voxels", "Delete", Command::Op(Op::Clear)));
        v.push(E::new("Model", "Reverse empty space", "", Command::Op(Op::Reverse)));
        v.push(E::new("Model", "Double size (2X)", "", Command::Op(Op::Double)));
        v.push(E::new("Model", "Half size (1/2)", "", Command::Op(Op::Half)));
        v.push(E::new("Model", "Fit the canvas to the model", "", Command::Op(Op::Fit)));
        for (a, n) in ["X", "Y", "Z"].iter().enumerate() {
            v.push(E::new("Model", format!("Rotate 90° about {n}"), "", Command::Op(Op::Rotate(a))));
        }
        for (a, n) in ["X", "Y", "Z"].iter().enumerate() {
            v.push(E::new("Model", format!("Flip {n}"), "", Command::Op(Op::Flip(a))));
        }
        for (a, n) in ["X", "Y", "Z"].iter().enumerate() {
            v.push(E::new("Model", format!("Loop +{n} (shift one voxel, wrapping)"), "", Command::Op(Op::Loop(a, 1))));
            v.push(E::new("Model", format!("Loop -{n}"), "", Command::Op(Op::Loop(a, -1))));
        }
        for (a, n) in ["X", "Y", "Z"].iter().enumerate() {
            v.push(E::new("Model", format!("Mirror the model across {n}"), "", Command::Op(Op::Mirror(a))));
        }
        for (a, n) in ["X", "Y", "Z"].iter().enumerate() {
            v.push(E::new("Model", format!("Diagonal across {n}"), "", Command::Op(Op::Diagonal(a))));
        }
        for (s, n) in [(Shape::Ellipsoid, "Ellipsoid"), (Shape::Cylinder, "Cylinder"), (Shape::Pyramid, "Pyramid"), (Shape::Cone, "Cone")] {
            v.push(E::new("Shape", format!("{n}: fill the canvas with one"), "", Command::Op(Op::Shape(s))));
        }
        v.push(E::new("Shape", "Maze", "", Command::Op(Op::Maze)));
        v.push(E::new("Shape", "Perlin noise", "", Command::Op(Op::Perlin)));
        v.push(E::new("Shape", "Fractal: each voxel becomes the pattern", "", Command::Op(Op::Fractal)));
        v.push(E::new("Modify", "Hull: remove hidden voxels", "", Command::Op(Op::Hull)));
        v.push(E::new("Modify", "Flood: fill hidden space", "", Command::Op(Op::Flood)));
        v.push(E::new("Modify", "Mask with the pattern", "", Command::Op(Op::Mask)));
        v.push(E::new("Modify", "Texture with the pattern", "", Command::Op(Op::Tex)));
        v.push(E::new("Modify", "Dilate (grow by one voxel)", "", Command::Op(Op::Dilate)));
        v.push(E::new("Modify", "Erode (shrink by one voxel)", "", Command::Op(Op::Erode)));
        v.push(E::new("Modify", "Random voxel colors from the selected palette colors", "", Command::PaletteRandom));
    }

    if modeling && !editing {
        let picked = &scene.picked;
        let any = !picked.is_empty();
        let w = &scene.world;
        v.push(E::new("World", "New empty model", "Ctrl+N", Command::Tree(TreeOp::New)));
        if any {
            let names = picked.iter().map(|&p| w.label(p)).collect::<Vec<_>>().join(", ");
            let it = if picked.len() == 1 { names.clone() } else { format!("{} objects", picked.len()) };
            v.push(E::new("World", format!("Duplicate {it}"), "Ctrl+C, Ctrl+V", Command::Tree(TreeOp::Duplicate)));
            v.push(E::new("World", format!("Group {it}"), "Ctrl+R", Command::Tree(TreeOp::Group)));
            if picked.iter().any(|&p| w.node(p).is_some_and(|n| n.is_group())) {
                v.push(E::new("World", format!("Ungroup {it}"), "Ctrl+Shift+R", Command::Tree(TreeOp::Ungroup)));
            }
            v.push(E::new("World", format!("Hide {it}"), "Ctrl+H", Command::Tree(TreeOp::Hide(true))));
            v.push(E::new("World", format!("Delete {it}"), "Delete", Command::Tree(TreeOp::Delete)));
            if let [id] = picked.as_slice() {
                v.push(E::new("World", format!("Edit {names}"), "Tab", Command::EditPart(*id)));
                if w.node(*id).is_some_and(|n| !n.children().is_empty()) {
                    v.push(E::new("World", format!("Go inside {names}"), "", Command::Tree(TreeOp::Enter(*id))));
                }
                if w.anim.frames > 1 {
                    let keyed = w.node(*id).is_some_and(|n| n.keys.iter().any(|k| k.frame == w.frame));
                    v.push(E::new("Animation", format!("Key {names} on this frame"), "", Command::Tree(TreeOp::Key(*id))));
                    if keyed {
                        v.push(E::new("Animation", format!("Remove {names}'s key on this frame"), "", Command::Tree(TreeOp::DeleteKey(*id))));
                    }
                }
            }
        }
        v.push(E::new("World", "Show all objects", "Ctrl+Shift+H", Command::Tree(TreeOp::Hide(false))));
        if scene.inside != crate::world::ROOT {
            let up = w.node(scene.inside).and_then(|n| n.parent).unwrap_or(crate::world::ROOT);
            v.push(E::new("World", "Back out (up one level)", "Esc", Command::Tree(TreeOp::Enter(up))));
        }
        v.push(E::new("World", "Add a humanoid rig", "", Command::Tree(TreeOp::AddRig(false))));
        v.push(E::new("World", "Add a four-legged rig", "", Command::Tree(TreeOp::AddRig(true))));
    }

    // Animation.
    let a = &scene.world.anim;
    v.push(E::new("Animation", if ed.timeline.open { "Hide the timeline" } else { "Show the timeline" }, "", Command::TogglePanel(Panel::Timeline)));
    if a.frames > 1 {
        let f = scene.world.frame;
        v.push(E::new("Animation", if ed.timeline.playing { "Pause" } else { "Play" }, "/", Command::TogglePlay));
        v.push(E::new("Animation", "Next frame", ".", Command::Frame((f + 1) % a.frames)));
        v.push(E::new("Animation", "Previous frame", ",", Command::Frame(f.checked_sub(1).unwrap_or(a.frames - 1))));
        v.push(E::new("Animation", "First frame", "", Command::Frame(0)));
    }
    v.push(E::new("Animation", "Import animation clips from a glTF file…", "", Command::ImportAnimation(None)));

    // Camera and the view.
    v.push(E::new("View", "Reset camera", "F12", Command::ResetCamera));
    v.push(E::new("View", "Focus the view on the selection or model", "5", Command::Focus));
    v.push(E::new("View", "Save a PNG snapshot of the view", "6", Command::Snapshot));
    v.push(E::new("View", "Perspective camera", "", Command::SetView(View::Perspective)));
    v.push(E::new("View", "Free camera (look around from the eye, WASD to move)", "", Command::SetView(View::Free)));
    v.push(E::new("View", "Orthographic camera", "", Command::SetView(View::Orthographic)));
    v.push(E::new("View", "Isometric view", "", Command::SetView(View::Isometric)));
    if modeling {
        if let Some(o) = r.modeler.as_ref().map(|m| &m.options) {
            let state = |on: bool| if on { "off" } else { "on" };
            for (d, name, k, on) in [
                (Display::Others, "other models while editing", "Ctrl+B", o.others),
                (Display::Edge, "voxel edges", "Ctrl+E", o.edge),
                (Display::Grid, "grid", "Ctrl+G", o.grid),
                (Display::Frame, "frame lines every 10 voxels", "Ctrl+F", o.frame),
                (Display::Ao, "ambient occlusion", "", o.ao),
                (Display::Shadow, "shadows", "Ctrl+W", o.shadow),
                (Display::Wire, "model boxes in World mode", "Ctrl+M", o.wire),
                (Display::CanvasBox, "canvas box while editing", "", o.canvas_box),
            ] {
                v.push(E::new("Display", format!("Turn {} {name}", state(on)), k, Command::ToggleDisplay(d)));
            }
        }
        v.push(E::new("View", if ed.backdrop.open { "Hide the Reference picture window" } else { "Reference picture…" }, "", Command::ToggleReference));
    } else {
        v.push(E::new("Render", "Render the final image at the Image size", "", Command::RenderImage));
        v.push(E::new("Render", "Reset the current color's material to diffuse", "Ctrl+R", Command::ResetMaterial));
        if ed.material_clipboard.is_some() {
            v.push(E::new("Render", "Paste the copied material onto the current color", "Ctrl+V", Command::PasteMaterial));
        }
    }
    v.push(E::new("View", "Bigger interface", "Ctrl+=", Command::UiScale(1)));
    v.push(E::new("View", "Smaller interface", "Ctrl+-", Command::UiScale(-1)));
    v.push(E::new("View", "Interface at 100%", "Ctrl+0", Command::UiScale(0)));
    v.push(E::new("View", if ed.agent.open { "Hide the Agents panel" } else { "Agents panel" }, "", Command::TogglePanel(Panel::Agents)));

    // Palette.
    use crate::palette_lib::Cmd as P;
    v.push(E::new("Palette", if ed.palette.library_open { "Close the palette library" } else { "Palette library (favorites, saved, presets)" }, "", Command::TogglePanel(Panel::PaletteLibrary)));
    v.push(E::new("Palette", "Load a palette file…", "", Command::PalettePick(ed.palette.match_colors)));
    for k in crate::palette_ops::SortKey::ALL {
        v.push(E::new("Palette", format!("Sort the palette by {}", k.name()), "", Command::Palette(P::Sort(k))));
    }
    v.push(E::new("Palette", "Pack the palette: used colors first, duplicates merged", "", Command::Palette(P::Pack)));
    v.push(E::new("Palette", "Undo the last palette change", "", Command::Palette(P::Undo)));

    // Export.
    for f in std::iter::once(crate::export::Format::Gltf).chain(crate::export::Format::ALL) {
        let about = f.hint().split_once(" : ").map(|(_, h)| h).unwrap_or(f.hint());
        v.push(E::new("Export", format!("Export {}: {about}", f.name()), "", Command::Export(f)));
    }

    // Projects and samples, so the palette is also a quick switcher.
    let stem = |p: &std::path::Path| p.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    for p in &ed.projects {
        v.push(E::new("Project", format!("Open project: {}", stem(p)), "", Command::OpenPath(p.clone())));
    }
    for p in &ed.samples {
        v.push(E::new("Project", format!("Open sample: {}", stem(p)), "", Command::OpenPath(p.clone())));
    }
    v
}

/// How well `query` fits `text`: every word of the query must appear in
/// order (contiguous, or as a subsequence that prefers word starts). The
/// score is higher for closer matches; the indices are the matched
/// characters of `text`, for highlighting.
fn score(query: &str, text: &str) -> Option<(i32, Vec<usize>)> {
    let t: Vec<char> = text.chars().map(|c| c.to_ascii_lowercase()).collect();
    let mut total = 0;
    let mut hits = Vec::new();
    for word in query.split_whitespace() {
        let w: Vec<char> = word.chars().map(|c| c.to_ascii_lowercase()).collect();
        let (s, h) = match_word(&w, &t)?;
        total += s;
        hits.extend(h);
    }
    // Shorter names win ties.
    Some((total * 16 - t.len() as i32, hits))
}

fn word_start(t: &[char], k: usize) -> bool {
    k == 0 || !t[k - 1].is_alphanumeric()
}

fn match_word(w: &[char], t: &[char]) -> Option<(i32, Vec<usize>)> {
    if w.is_empty() {
        return Some((0, Vec::new()));
    }
    // A contiguous match first, best at the start of a word.
    if w.len() <= t.len() {
        if let Some(start) = (0..=t.len() - w.len()).find(|&k| t[k..k + w.len()] == *w) {
            let bonus = if start == 0 { 40 } else if word_start(t, start) { 30 } else { 10 };
            return Some((60 + bonus + w.len() as i32 * 4, (start..start + w.len()).collect()));
        }
    }
    // Otherwise a subsequence: consecutive letters, then word starts, then anything.
    let mut pos = 0;
    let mut hits = Vec::with_capacity(w.len());
    let mut s = 0;
    for &c in w {
        let next = if pos < t.len() && t[pos] == c && !hits.is_empty() { Some((pos, 6)) } else { None };
        let start = (pos..t.len()).find(|&k| t[k] == c && word_start(t, k)).map(|k| (k, 4));
        let any = (pos..t.len()).find(|&k| t[k] == c).map(|k| (k, 1));
        let (k, pts) = next.or(start).or(any)?;
        s += pts - ((k - pos) as i32 / 8).min(3);
        hits.push(k);
        pos = k + 1;
    }
    Some((s, hits))
}

/// The hotkey as this platform writes it (egui's font has no key symbols).
fn keys_label(keys: &str) -> String {
    if cfg!(target_os = "macos") {
        keys.replace("Ctrl+", "Cmd+").replace("Alt+", "Option+")
    } else {
        keys.to_string()
    }
}

/// Draw the palette (if open) and handle its keys. Commands chosen go to
/// `cmds`, run after the frame like the panels' own.
pub fn show(ctx: &egui::Context, pal: &mut CommandPalette, entries: Vec<Entry>, cmds: &mut Vec<Command>) {
    if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::K)) {
        pal.toggle();
    }
    if !pal.open {
        return;
    }
    // Take the navigation keys before the text box sees them.
    let (down, up, page_down, page_up, enter, esc) = ctx.input_mut(|i| {
        (
            i.consume_key(Modifiers::NONE, Key::ArrowDown),
            i.consume_key(Modifiers::NONE, Key::ArrowUp),
            i.consume_key(Modifiers::NONE, Key::PageDown),
            i.consume_key(Modifiers::NONE, Key::PageUp),
            i.consume_key(Modifiers::NONE, Key::Enter),
            i.consume_key(Modifiers::NONE, Key::Escape),
        )
    });
    if esc {
        pal.close();
        return;
    }

    // Filter: with no query, recently run commands first, then all.
    let query = pal.query.trim().to_string();
    let mut rows: Vec<(usize, Vec<usize>)> = if query.is_empty() {
        let mut recent: Vec<usize> = pal.recent.iter().filter_map(|n| entries.iter().position(|e| e.name == *n)).collect();
        let rest: Vec<usize> = (0..entries.len()).filter(|k| !recent.contains(k)).collect();
        recent.extend(rest);
        recent.into_iter().map(|k| (k, Vec::new())).collect()
    } else {
        let mut scored: Vec<(i32, usize, Vec<usize>)> = entries
            .iter()
            .enumerate()
            .filter_map(|(k, e)| {
                let name_len = e.name.chars().count();
                // The group counts too ("export obj"), but only the name is highlighted.
                let (s, hits) = score(&query, &format!("{} {}", e.name, e.group))?;
                Some((s, k, hits.into_iter().filter(|&h| h < name_len).collect()))
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.into_iter().map(|(_, k, hits)| (k, hits)).collect()
    };
    let n = rows.len();
    if n == 0 {
        pal.selected = 0;
    } else {
        pal.selected = pal.selected.min(n - 1);
        if down {
            pal.selected = (pal.selected + 1).min(n - 1);
        }
        if up {
            pal.selected = pal.selected.saturating_sub(1);
        }
        if page_down {
            pal.selected = (pal.selected + 10).min(n - 1);
        }
        if page_up {
            pal.selected = pal.selected.saturating_sub(10);
        }
    }
    let moved = down || up || page_down || page_up;
    let mut run: Option<usize> = None;
    if enter && n > 0 {
        run = Some(rows[pal.selected].0);
    }

    let screen = ctx.content_rect();
    let width = (screen.width() - 40.0).min(560.0);
    let max_list = (screen.height() * 0.5).clamp(120.0, 420.0);
    let pointer_moved = ctx.input(|i| i.pointer.delta() != egui::Vec2::ZERO);
    let area = egui::Area::new(egui::Id::unique("command_palette"))
        .order(egui::Order::Foreground)
        .anchor(Align2::CENTER_TOP, [0.0, (screen.height() * 0.12).max(48.0)])
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).inner_margin(8.0).show(ui, |ui| {
                ui.set_width(width);
                let edit = ui.add(
                    egui::TextEdit::singleline(&mut pal.query)
                        .hint_text("Type a command, a project, an export format…")
                        .font(egui::TextStyle::Heading)
                        .desired_width(f32::INFINITY)
                        .lock_focus(true),
                );
                if pal.focus || !edit.has_focus() {
                    edit.request_focus();
                    pal.focus = false;
                }
                if edit.changed() {
                    pal.selected = 0;
                }
                ui.add_space(4.0);
                ui.separator();
                if n == 0 {
                    ui.add_space(6.0);
                    ui.weak("No command matches");
                    ui.add_space(6.0);
                }
                let row_h = 26.0;
                egui::ScrollArea::vertical().max_height(max_list).auto_shrink([false, true]).show(ui, |ui| {
                    let weak = ui.visuals().weak_text_color();
                    let strong = ui.visuals().strong_text_color();
                    let text_color = ui.visuals().text_color();
                    let font = egui::TextStyle::Body.resolve(ui.style());
                    let small = egui::TextStyle::Small.resolve(ui.style());
                    for (k, (ei, hits)) in rows.iter_mut().enumerate() {
                        let e = &entries[*ei];
                        let sel = k == pal.selected;
                        let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), row_h), Sense::click());
                        if sel {
                            ui.painter().rect_filled(rect, 4.0, ui.visuals().selection.bg_fill);
                        } else if resp.hovered() {
                            ui.painter().rect_filled(rect, 4.0, ui.visuals().widgets.hovered.weak_bg_fill);
                        }
                        // Name with the matched letters emphasized, then the group.
                        let mut job = LayoutJob::default();
                        let name_color = if sel { ui.visuals().selection.stroke.color } else { text_color };
                        let hit_color = if sel { Color32::WHITE } else { strong };
                        let mut run_text = String::new();
                        let mut run_hit = false;
                        hits.sort_unstable();
                        for (ci, ch) in e.name.chars().enumerate() {
                            let hit = hits.binary_search(&ci).is_ok();
                            if hit != run_hit && !run_text.is_empty() {
                                job.append(&run_text, 0.0, TextFormat { font_id: font.clone(), color: if run_hit { hit_color } else { name_color }, underline: if run_hit { egui::Stroke::new(1.0, hit_color) } else { egui::Stroke::NONE }, ..Default::default() });
                                run_text.clear();
                            }
                            run_hit = hit;
                            run_text.push(ch);
                        }
                        if !run_text.is_empty() {
                            job.append(&run_text, 0.0, TextFormat { font_id: font.clone(), color: if run_hit { hit_color } else { name_color }, underline: if run_hit { egui::Stroke::new(1.0, hit_color) } else { egui::Stroke::NONE }, ..Default::default() });
                        }
                        job.append(&format!("   {}", e.group), 0.0, TextFormat { font_id: small.clone(), color: weak, ..Default::default() });
                        job.wrap.max_width = rect.width() - 90.0;
                        job.wrap.max_rows = 1;
                        job.wrap.break_anywhere = true;
                        let galley = ui.fonts_mut(|f| f.layout_job(job));
                        let text_pos = egui::pos2(rect.left() + 8.0, rect.center().y - galley.size().y * 0.5);
                        ui.painter().galley(text_pos, galley, name_color);
                        if !e.keys.is_empty() {
                            ui.painter().text(egui::pos2(rect.right() - 8.0, rect.center().y), Align2::RIGHT_CENTER, keys_label(e.keys), small.clone(), weak);
                        }
                        if sel && (moved || edit.changed() || pal.focus) {
                            ui.scroll_to_rect(rect, None);
                        }
                        if resp.hovered() && pointer_moved {
                            pal.selected = k;
                        }
                        if resp.clicked() {
                            run = Some(*ei);
                        }
                    }
                });
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Up / Down choose   Enter run   Esc close").weak().small());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(format!("{n} command{}", if n == 1 { "" } else { "s" })).weak().small());
                    });
                });
            });
        });

    if let Some(k) = run {
        let e = &entries[k];
        pal.ran(&e.name);
        cmds.push(e.cmd.clone());
        pal.close();
        return;
    }
    // A click anywhere else closes it.
    let pressed_outside = ctx.input(|i| i.pointer.any_pressed() && i.pointer.interact_pos().is_some_and(|p| !area.response.rect.contains(p)));
    if pressed_outside {
        pal.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(q: &str, texts: &[&str]) -> Vec<&'static str> {
        let mut v: Vec<(i32, &str)> = texts.iter().filter_map(|t| score(q, t).map(|(s, _)| (s, *t))).collect();
        v.sort_by(|a, b| b.0.cmp(&a.0));
        v.into_iter().map(|(_, t)| unsafe { std::mem::transmute::<&str, &'static str>(t) }).collect()
    }

    #[test]
    fn contiguous_beats_scattered_and_word_starts_beat_insides() {
        let texts = ["Save", "Save as…", "Reverse empty space", "Pack the palette: used colors first"];
        assert_eq!(names("save", &texts), ["Save", "Save as…"]);
        assert_eq!(names("pal", &texts)[0], "Pack the palette: used colors first");
        assert!(score("xyz", "Save").is_none());
    }

    #[test]
    fn every_word_must_match_and_initials_work() {
        assert!(score("export obj", "Export obj: Export obj mesh Export").is_some());
        assert!(score("export nope", "Export obj: Export obj mesh").is_none());
        let (s, hits) = score("rc", "Reset camera").unwrap();
        assert_eq!(hits, vec![0, 6]);
        assert!(s > 0);
    }

    #[test]
    fn recent_and_query_state_reset_on_close() {
        let mut p = CommandPalette::default();
        p.show();
        p.query = "sa".into();
        p.ran("Save");
        p.ran("Undo");
        p.ran("Save");
        assert_eq!(p.recent, ["Save", "Undo"]);
        p.close();
        assert!(p.query.is_empty() && !p.open);
    }
}
