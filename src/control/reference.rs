//! Reference images: analyze a photo or concept sheet, keep views of it
//! with the project, trace them at voxel resolution and measure the model
//! against them.

mod compare;
mod pitch;
mod segment;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use glam::{IVec3, Vec3};
use serde_json::{Value, json};

use super::surface::{Image, load_image};
use super::view::draw::{Canvas, Rgb, text_size};
use super::{Ctx, boolean, get, int, num, string, text, tool};
use compare::{AXES, Grid, Placement, RefPic, View};
use segment::Segmentation;

/// A reference view attached to the project.
struct Ref {
    name: String,
    source: String,
    /// The silhouette (alpha) cropped from the source.
    pic: RefPic,
    /// How it was cut: blob number or crop box, and whether it was mirrored.
    blob: Option<usize>,
    crop: Option<[usize; 4]>,
    flip: bool,
    /// Pinned by set_reference: voxels tall and the world [h, v] of its
    /// bottom-center (compare_reference then uses them by default).
    height: Option<f32>,
    origin: Option<[f32; 2]>,
    /// The last reference_profile's scale and placement (world).
    traced: Option<(f32, [f32; 2])>,
}

#[derive(Default)]
pub struct State {
    /// The project the views below belong to (loaded lazily).
    project: Option<String>,
    refs: BTreeMap<View, Ref>,
    /// The last analyzed image, so set_reference's blob numbers match.
    analysis: Option<(PathBuf, Option<f32>, Image, Segmentation)>,
}

pub const PART_TOOLS: &[&str] = &[];

const VIEWS: [&str; 5] = ["front", "back", "left", "right", "top"];

pub fn tools() -> Vec<Value> {
    let view = json!({ "type": "string", "enum": VIEWS,
        "description": "front: the model seen from -y (screen right = +x), as screenshot's FRONT; back: from +y (right = -x); right: from +x (right = +y); left: from -x (right = -y); top: from +z (right = +x, up = +y). A model faces -y, so a sheet's SIDE view is \"right\" when the subject faces image-left, \"left\" when it faces image-right." });
    let part = json!({ "type": ["string", "integer"], "description": "Part (name or id); default: the part being edited, or the whole scene in World mode" });
    vec![
        tool(
            "analyze_image",
            "First look at a reference picture (photo, concept art, turnaround sheet) as numbers. Separates subjects from the background (a smooth fit to the border colors, plus texture so pale subjects on pale backgrounds still count), numbers the blobs left to right and flags caption-like blobs (short rows of glyphs such as FRONT / SIDE / BACK) as text, not views. \
             Per subject: box in pixels, aspect ratio, the voxel block size if the picture is voxel art (so the model can match it: one picture block = one voxel), a width profile (filled extent at 20 heights, as fractions of the box width, so proportions read off directly) and dominant colors (k-means in OKLab) with where they sit. Returns an annotated image with numbered boxes; then attach views with set_reference.",
            json!({
                "path": { "type": "string", "description": "PNG or JPEG" },
                "threshold": { "type": "number", "minimum": 0, "maximum": 255, "description": "Color difference from the background (0-255, largest channel) that counts as subject; default automatic (about 15). Lower it if parts of the subject are missed, raise it if shadows join in" },
                "crop": { "type": "array", "items": { "type": "integer" }, "minItems": 4, "maxItems": 4, "description": "[x0, y0, x1, y1] pixel box around one subject (e.g. the character in a busy scene): also reports its voxel block size and how many blocks tall the box is" },
            }),
            &["path"],
        ),
        tool(
            "set_reference",
            "Attach a reference view to the current project: one subject from a picture, cut out by its silhouette. Kept in memory and saved under <workspace>/refs/<project>/ so it comes back when the project is reopened (set references after new_project / open_project). \
             Pick the subject with `blob` (a number from analyze_image), or `crop` a box; with neither, the picture's only subject. Then reference_profile traces it in voxels and compare_reference measures the model against it. `clear` removes a view.",
            json!({
                "view": view,
                "path": { "type": "string", "description": "PNG or JPEG (PNGs with a transparent background use alpha as the silhouette)" },
                "blob": { "type": "integer", "description": "Subject number from analyze_image" },
                "crop": { "type": "array", "items": { "type": "integer" }, "minItems": 4, "maxItems": 4, "description": "[x0, y0, x1, y1] pixel box (inclusive) holding the subject; the silhouette inside it is used" },
                "name": { "type": "string", "description": "A label, e.g. \"knight front\"" },
                "flip": { "type": "boolean", "description": "Mirror left-right, e.g. to use a side view drawn facing the other way" },
                "height": { "type": "number", "description": "Pin the scale: how many voxels tall (top view: deep along y) the subject is; compare_reference then measures at this scale and position instead of fitting to the model" },
                "origin": { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2, "description": "With height: world [h, v] where the reference's bottom-center sits (h along the view's horizontal axis, v up the vertical one: [x, z] for front/back, [y, z] for left/right, [x, y] for top). Default: the middle of the model's canvas at v 0" },
                "threshold": { "type": "number", "minimum": 0, "maximum": 255, "description": "As in analyze_image" },
                "part": { "type": ["string", "integer"], "description": "With height and no origin: center the default origin on this part (default: every visible part)" },
                "clear": { "type": "boolean", "description": "Remove this view's reference" },
            }),
            &["view"],
        ),
        tool("list_references", "The project's reference views: name, source, size, pinned scale, and the saved files.", json!({}), &[]),
        tool(
            "reference_profile",
            "Trace a reference view at voxel resolution, for building: give the size you want (height in voxels, or width, or scale = voxels per pixel) and get, per row (z for front/back/left/right, y for top) from the top down, the filled spans [from..to] along the view's horizontal axis (several spans where there are gaps, e.g. between legs or under a car), \
             then a labeled letter map of the silhouette (rows and every 5th column labeled with coordinates, as seen in the view) where each letter is the reference's dominant color there (legend gives hex). \
             Coordinates are the part's own (what fill_box uses), or world coordinates in World mode. Fill the spans row by row (or extrude_polygon / fill_box per run) and you have the silhouette; compare_reference then measures it.",
            json!({
                "view": view,
                "height": { "type": "number", "description": "Voxels tall (top view: deep along y)" },
                "width": { "type": "number", "description": "Or voxels wide" },
                "scale": { "type": "number", "description": "Or voxels per reference pixel" },
                "origin": { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2, "description": "[h, v] where the reference's bottom-center goes (h along the horizontal axis, v up). Default: the canvas's middle at v 0 (top view: centered)" },
                "part": part.clone(),
            }),
            &["view"],
        ),
        tool(
            "compare_reference",
            "Measure the model against a reference view. Projects the model exactly (no rendering: the first voxel along the view direction, oriented as screenshot's views), lays the reference silhouette over it at voxel resolution and reports: silhouette IoU with precision (share of model cells inside the reference) and recall (share of the reference covered); \
             per-row outline findings, largest first (\"z=2-14: model narrower on both sides ...\", \"model extends 3 rows above the reference's top\"); color agreement by region (OKLab dE with lightness counted half, since references are shaded) and the worst regions with model vs reference colors; \
             and a score 0-100 = 75% IoU + 25% color agreement (agreement = mean over shared cells of max(0, 1 - dE/25)). The image shows reference | model | overlay (white both, magenta model only, green reference only) with coordinates. \
             By default the reference is scaled to the model's projected height, bottom-aligned and centered (nudged for best overlap), so the numbers are about shape; align \"pinned\" uses the scale and position from set_reference's height/origin or the last reference_profile, so misplacement shows too.",
            json!({
                "view": view,
                "part": { "type": ["string", "integer"], "description": "Compare one part (name or id); default: every visible part together" },
                "align": { "type": "string", "enum": ["fit", "pinned"], "description": "fit: scale to the model (default unless set_reference pinned a height); pinned: the pinned or traced scale and position" },
            }),
            &["view"],
        ),
        tool(
            "paint_from_reference",
            "Copies the reference's colors INCLUDING its lighting, which shows up as speckle and muddy shading; not the house style (pick flat colors from the sheet instead). Use it only when the user asks for a painted copy. Color the model from a reference view: every voxel visible from that view (across all parts, so whatever is in front gets painted, not what it hides) takes the reference's color at that spot, matched to the nearest allowed palette color (OKLab, lightness counted half since references are shaded). \
             Gets straps, trim, faces, markings, liveries and facades right in one call; then shade and fix by hand. Aligned like compare_reference (pinned scale if set, else fit). Run it per view (front, then back and sides) on a finished shape; later views only repaint what they see. Each part changed is one undo step.",
            json!({
                "view": view,
                "colors": { "type": "array", "items": { "type": ["integer", "string"] }, "description": "Palette colors to choose from (indices or \"#rrggbb\"); default: the colors the scene already uses" },
                "palette_range": { "type": "array", "items": { "type": "integer" }, "minItems": 2, "maxItems": 2, "description": "Or [first, last] palette indices to choose from" },
                "keep": { "type": "array", "items": { "type": ["integer", "string"] }, "description": "Colors never painted over (e.g. emissive eyes, a logo you placed by hand)" },
                "only_color": { "type": "array", "items": { "type": ["integer", "string"] }, "description": "Paint only voxels that currently have one of these colors" },
                "parts": { "type": "array", "items": { "type": ["string", "integer"] }, "description": "Only paint these parts (names or ids); default: every visible part" },
                "depth": { "type": "integer", "minimum": 1, "maximum": 8, "description": "Voxels deep to paint from the visible surface, within the same part (default 1)" },
                "clean": { "type": "integer", "minimum": 0, "maximum": 4, "description": "Majority-filter passes over the chosen colors before painting, removing the photo's speckle while keeping features 2+ voxels wide (default 1; 0 = raw)" },
                "align": { "type": "string", "enum": ["fit", "pinned"] },
            }),
            &["view"],
        ),
    ]
}

pub fn dispatch(ctx: &mut Ctx, name: &str, args: &Value) -> Option<Result<Vec<Value>>> {
    Some(match name {
        "analyze_image" => analyze_image(ctx, args),
        "set_reference" => set_reference(ctx, args),
        "list_references" => list_references(ctx),
        "reference_profile" => reference_profile(ctx, args),
        "compare_reference" => compare_reference(ctx, args),
        "paint_from_reference" => paint_from_reference(ctx, args),
        _ => return None,
    })
}

// ---------------------------------------------------------------- storage

fn refs_dir(project: &str) -> PathBuf {
    crate::app::workspace_dir().join("refs").join(project)
}

/// Make sure the current project's saved references are loaded.
fn ensure_loaded(ctx: &mut Ctx) {
    let project = ctx.ed.name();
    let st = &mut ctx.ed.tools.reference;
    if st.project.as_deref() == Some(project.as_str()) {
        return;
    }
    st.refs.clear();
    let dir = refs_dir(&project);
    for view in View::ALL {
        if let Some(r) = load_ref(&dir, view) {
            st.refs.insert(view, r);
        }
    }
    st.project = Some(project);
}

fn load_ref(dir: &Path, view: View) -> Option<Ref> {
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{}.json", view.key()))).ok()?).ok()?;
    let img = load_image(&dir.join(format!("{}.png", view.key()))).ok()?;
    let pair = |v: &Value| -> Option<[f32; 2]> {
        let a = v.as_array()?;
        Some([a.first()?.as_f64()? as f32, a.get(1)?.as_f64()? as f32])
    };
    let crop = meta["crop"].as_array().and_then(|a| {
        let v: Vec<usize> = a.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect();
        (v.len() == 4).then(|| [v[0], v[1], v[2], v[3]])
    });
    Some(Ref {
        name: meta["name"].as_str().unwrap_or(view.key()).to_string(),
        source: meta["source"].as_str().unwrap_or_default().to_string(),
        pic: RefPic::new(img.w, img.h, img.px),
        blob: meta["blob"].as_u64().map(|b| b as usize),
        crop,
        flip: meta["flip"].as_bool().unwrap_or(false),
        height: meta["height"].as_f64().map(|h| h as f32),
        origin: pair(&meta["origin"]),
        traced: meta["traced_height"].as_f64().zip(pair(&meta["traced_origin"])).map(|(h, o)| (h as f32, o)),
    })
}

/// Write a view's PNG and JSON next to the project.
fn save_ref(project: &str, view: View, r: &Ref, png: bool) -> Result<PathBuf> {
    let dir = refs_dir(project);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join(format!("{}.png", view.key()));
    if png {
        let rgba: Vec<u8> = r.pic.px.iter().flatten().copied().collect();
        std::fs::write(&path, super::encode_png(r.pic.w as u32, r.pic.h as u32, &rgba)?)?;
    }
    let meta = json!({
        "view": view.key(),
        "name": r.name,
        "source": r.source,
        "size": [r.pic.w, r.pic.h],
        "blob": r.blob,
        "crop": r.crop,
        "flip": r.flip,
        "height": r.height,
        "origin": r.origin,
        "traced_height": r.traced.map(|t| t.0),
        "traced_origin": r.traced.map(|t| t.1),
    });
    std::fs::write(dir.join(format!("{}.json", view.key())), serde_json::to_string_pretty(&meta)?)?;
    Ok(path)
}

fn view_arg(a: &Value) -> Result<View> {
    let s = string(a, "view").context("missing `view` (front, back, left, right or top)")?;
    View::parse(s).ok_or_else(|| anyhow!("unknown view {s:?}; use front, back, left, right or top (a sheet's SIDE is usually right when the subject faces image-left)"))
}

fn get_ref<'a>(ctx: &'a mut Ctx, view: View) -> Result<&'a Ref> {
    ensure_loaded(ctx);
    let st = &ctx.ed.tools.reference;
    if !st.refs.contains_key(&view) {
        let have: Vec<&str> = st.refs.keys().map(|v| v.key()).collect();
        let have = if have.is_empty() { "none yet".to_string() } else { have.join(", ") };
        bail!("no {} reference for project `{}` (have: {have}); attach one with set_reference", view.key(), ctx.ed.name());
    }
    Ok(&ctx.ed.tools.reference.refs[&view])
}

/// The stored reference for a screenshot view name ("front", "FRONT -Y",
/// "right" ...), fitted inside `w` x `h` on a dark tile with its title, or
/// None if the project has no reference for that view. For screenshot's
/// optional reference tiles.
#[allow(dead_code)]
pub(super) fn reference_tile(ctx: &mut Ctx, view: &str, w: u32, h: u32) -> Option<Canvas> {
    let view = View::parse(view)?;
    ensure_loaded(ctx);
    let r = ctx.ed.tools.reference.refs.get(&view)?;
    let mut cv = Canvas::new(w, h, [24, 24, 24, 255].repeat((w * h) as usize));
    let sc = if w.min(h) >= 280 { 2 } else { 1 };
    let top = 12 + 7 * sc;
    let (aw, ah) = (w as f32 - 12.0, h as f32 - top as f32 - 6.0);
    let s = (aw / r.pic.w as f32).min(ah / r.pic.h as f32).max(1e-3);
    let (dw, dh) = ((r.pic.w as f32 * s) as i32, (r.pic.h as f32 * s) as i32);
    let (x0, y0) = ((w as i32 - dw) / 2, top + (ah as i32 - dh) / 2);
    for y in 0..dh {
        for x in 0..dw {
            let (sx, sy) = (((x as f32 + 0.5) / s) as usize, ((y as f32 + 0.5) / s) as usize);
            let p = r.pic.px[sy.min(r.pic.h - 1) * r.pic.w + sx.min(r.pic.w - 1)];
            if p[3] >= 128 {
                cv.rect(x0 + x, y0 + y, 1, 1, [p[0], p[1], p[2]], 1.0);
            }
        }
    }
    cv.text(6, 6, &format!("REFERENCE {}", view.title()), sc, [235, 235, 235]);
    Some(cv)
}

// ---------------------------------------------------------------- analyze

fn analysis<'a>(ctx: &'a mut Ctx, path: &str, threshold: Option<f32>) -> Result<&'a (PathBuf, Option<f32>, Image, Segmentation)> {
    let p = PathBuf::from(path);
    let st = &mut ctx.ed.tools.reference;
    let fresh = st.analysis.as_ref().is_some_and(|a| a.0 == p && a.1 == threshold);
    if !fresh {
        let img = load_image(&p)?;
        if img.w < 8 || img.h < 8 {
            bail!("{path} is only {}x{} pixels", img.w, img.h);
        }
        let seg = segment::segment(&img, threshold);
        st.analysis = Some((p, threshold, img, seg));
    }
    Ok(st.analysis.as_ref().unwrap())
}

const BOX_COLORS: [Rgb; 6] = [[255, 80, 80], [80, 200, 255], [120, 230, 90], [255, 170, 40], [200, 120, 255], [255, 110, 200]];
const TEXT_COLOR: Rgb = [250, 230, 60];

fn hex3(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

fn pct(f: f32) -> String {
    format!("{:.0}%", f * 100.0)
}

fn analyze_image(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let path = string(a, "path").context("missing `path`")?.to_string();
    let threshold = num(a, "threshold");
    let (_, _, img, seg) = analysis(ctx, &path, threshold)?;
    let mut out = Vec::new();
    let subjects: Vec<&segment::Blob> = seg.blobs.iter().filter(|b| !b.text).collect();
    let texts: Vec<&segment::Blob> = seg.blobs.iter().filter(|b| b.text).collect();
    let file = Path::new(&path).file_name().map_or(path.clone(), |f| f.to_string_lossy().into_owned());
    let bg = &seg.bg;
    out.push(format!("{file}: {}x{} px.", img.w, img.h));
    if bg.alpha {
        out.push("Transparent background: alpha is the silhouette.".to_string());
    } else {
        out.push(format!(
            "Background ~{} (smooth fit matches {} of the border, noise {:.0}); subject = differs by more than {:.0} or is textured.",
            hex3(bg.color),
            pct(bg.inliers),
            bg.noise,
            seg.threshold
        ));
        if bg.inliers < 0.6 {
            out.push("The border is busy (not a plain backdrop): blobs may be unreliable; use set_reference crop to pick the subject by hand.".to_string());
        }
    }
    // Block size over the whole picture and in the box asked about: on busy
    // scenes the blobs are fragments, but the agent can see the subject.
    let all = |_: usize, _: usize| true;
    let mut boxes = vec![("The whole picture".to_string(), [0, 0, img.w - 1, img.h - 1])];
    if let Some(c) = get(a, "crop").and_then(Value::as_array) {
        let v: Vec<usize> = c.iter().filter_map(Value::as_u64).map(|n| n as usize).collect();
        if v.len() != 4 || v[0] >= v[2] || v[1] >= v[3] {
            bail!("`crop` is [x0, y0, x1, y1] in pixels");
        }
        boxes.push((format!("Box [{}, {}]-[{}, {}]", v[0], v[1], v[2], v[3]), [v[0], v[1], v[2].min(img.w - 1), v[3].min(img.h - 1)]));
    }
    for (what, bb) in boxes {
        out.push(match pitch::voxel_pitch(&img, &all, bb) {
            Some((p, conf)) => format!(
                "{what}: if it is voxel art, its blocks are {p:.1} px (confidence {:.0}%), so {} px tall = {:.0} blocks: build at that size, one picture block = one voxel. (Photos and textures can show a spacing too; you judge whether it is voxel art.)",
                conf * 100.0,
                bb[3] - bb[1] + 1,
                (bb[3] - bb[1] + 1) as f32 / p
            ),
            None => format!("{what}: no regular block spacing (not voxel art, or too blurry to tell)."),
        });
    }
    out.push(format!("{} subject blob(s), {} text label(s) (captions, not views), {} tiny specks ignored.", subjects.len(), texts.len(), seg.specks));
    for b in &subjects {
        let id = b.id as u32;
        let on = |x: usize, y: usize| seg.labels[y * seg.w + x] == id;
        let bx = (b.x0, b.y0, b.x1, b.y1);
        out.push(String::new());
        out.push(format!(
            "Blob {} (subject): box [{}, {}]-[{}, {}] px, {}x{} (w/h {:.2}); fills {} of its box; center x {}.",
            b.id,
            b.x0,
            b.y0,
            b.x1,
            b.y1,
            b.w(),
            b.h(),
            b.w() as f32 / b.h() as f32,
            pct(b.area as f32 / (b.w() * b.h()) as f32),
            (b.x0 + b.x1) / 2
        ));
        out.push(match pitch::voxel_pitch(&img, &on, [b.x0, b.y0, b.x1, b.y1]) {
            Some((p, conf)) => format!(
                "  If voxel art: {p:.1}-pixel blocks (confidence {:.0}%), so about {:.0} blocks tall and {:.0} wide: build it at that size, one picture block = one voxel.",
                conf * 100.0,
                b.h() as f32 / p,
                b.w() as f32 / p
            ),
            None => "  No regular block spacing (not voxel art, or too blurry to tell).".to_string(),
        });
        out.push("  Width profile, top to bottom (row position in the box (pixel y): span as fractions of the box width, 0 = left edge -> width; filled share; spans if >1):".to_string());
        let max_w = b.w() as f32;
        for r in segment::width_profile(&on, bx, 20) {
            let line = match r.extent {
                None => format!("    {:>3.0}% (y {}): empty", r.at * 100.0, r.y),
                Some((l, rr)) => {
                    let (fl, fr) = ((l - b.x0) as f32 / max_w, (rr - b.x0 + 1) as f32 / max_w);
                    let spans = if r.spans > 1 { format!(", {} spans", r.spans) } else { String::new() };
                    format!("    {:>3.0}% (y {}): {:.2}-{:.2} -> {:.2} wide, filled {:.2}{spans}", r.at * 100.0, r.y, fl, fr, fr - fl, r.filled as f32 / max_w)
                }
            };
            out.push(line);
        }
        let colors = segment::dominant(img, &on, bx, 8);
        out.push("  Colors (share; where in the box, rows/cols 10th-90th percentile from top/left):".to_string());
        for s in colors {
            out.push(format!(
                "    {} {:>3}  {} (rows {:.0}-{:.0}%, cols {:.0}-{:.0}%)",
                hex3(s.rgb),
                pct(s.share),
                segment::region_words(&s),
                s.rows.0 * 100.0,
                s.rows.1 * 100.0,
                s.cols.0 * 100.0,
                s.cols.1 * 100.0
            ));
        }
    }
    for t in &texts {
        // Which subject it labels: the one above it it overlaps most.
        let under = subjects
            .iter()
            .filter(|s| s.y1 <= t.y0 + t.h())
            .max_by_key(|s| (s.x1.min(t.x1) as i64 - s.x0.max(t.x0) as i64).max(-1_000_000))
            .filter(|s| s.x1 >= t.x0 && s.x0 <= t.x1)
            .map_or(String::new(), |s| format!(", under blob {}", s.id));
        out.push(format!("Blob {} (text label): box [{}, {}]-[{}, {}] px, {}x{}{under}.", t.id, t.x0, t.y0, t.x1, t.y1, t.w(), t.h()));
    }
    // Overall colors over every subject.
    if !subjects.is_empty() {
        let on = |x: usize, y: usize| {
            let l = seg.labels[y * seg.w + x];
            l != 0 && !seg.blobs[l as usize - 1].text
        };
        let colors = segment::dominant(img, &on, (0, 0, img.w - 1, img.h - 1), 12);
        let list: Vec<String> = colors.iter().map(|s| format!("{} {}", hex3(s.rgb), pct(s.share))).collect();
        out.push(String::new());
        out.push(format!("All subjects' colors: {}.", list.join(", ")));
    }
    out.push(String::new());
    out.push(
        "Next: set_reference {path, view, blob} for each view to build from (a SIDE view is \"right\" if the subject faces image-left, \"left\" if it faces image-right), then reference_profile for voxel spans."
            .to_string(),
    );
    let annotated = annotate(img, seg);
    Ok(vec![super::image(annotated.w, annotated.h, &annotated.px)?, text(out.join("\n"))])
}

/// The picture (shrunk to at most 1024 px wide) with numbered blob boxes.
fn annotate(img: &Image, seg: &Segmentation) -> Canvas {
    let s = (1024.0 / img.w as f32).min(1024.0 / img.h as f32).min(1.0);
    let (w, h) = (((img.w as f32 * s) as u32).max(1), ((img.h as f32 * s) as u32).max(1));
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = (((x as f32 + 0.5) / s) as usize, ((y as f32 + 0.5) / s) as usize);
            let p = img.px[sy.min(img.h - 1) * img.w + sx.min(img.w - 1)];
            // Transparent pixels over a checker-free dark gray.
            let a = p[3] as f32 / 255.0;
            px.extend((0..3).map(|k| (p[k] as f32 * a + 40.0 * (1.0 - a)) as u8));
            px.push(255);
        }
    }
    let mut cv = Canvas::new(w, h, px);
    let sc = if w >= 600 { 3 } else { 2 };
    let mut subject = 0;
    for b in &seg.blobs {
        let color = if b.text {
            TEXT_COLOR
        } else {
            subject += 1;
            BOX_COLORS[(subject - 1) % BOX_COLORS.len()]
        };
        let (x0, y0, x1, y1) = (b.x0 as f32 * s - 2.0, b.y0 as f32 * s - 2.0, (b.x1 + 1) as f32 * s + 1.0, (b.y1 + 1) as f32 * s + 1.0);
        for (p, q) in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))] {
            cv.line(p, q, 2.0, color, 1.0);
        }
        let label = if b.text { format!("{} TEXT", b.id) } else { b.id.to_string() };
        let (lw, lh) = text_size(&label, sc);
        let (lx, ly) = ((x0 as i32 + 2).max(0), if b.text { (y1 as i32 + 3).min(h as i32 - lh - 2) } else { (y0 as i32 + 3).max(0) });
        cv.rect(lx - 2, ly - 2, lw + 4, lh + 4, [0, 0, 0], 0.6);
        cv.text(lx, ly, &label, sc, color);
    }
    cv
}

// ---------------------------------------------------------------- set / list

fn set_reference(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let view = view_arg(a)?;
    ensure_loaded(ctx);
    let project = ctx.ed.name();
    if boolean(a, "clear").unwrap_or(false) {
        let had = ctx.ed.tools.reference.refs.remove(&view).is_some();
        let dir = refs_dir(&project);
        for ext in ["png", "json"] {
            let _ = std::fs::remove_file(dir.join(format!("{}.{ext}", view.key())));
        }
        return Ok(vec![text(if had { format!("Removed the {} reference.", view.key()) } else { format!("There was no {} reference.", view.key()) })]);
    }
    let path = string(a, "path").context("missing `path` (or clear: true)")?.to_string();
    let threshold = num(a, "threshold");
    let blob_arg = int(a, "blob");
    let crop_arg = match get(a, "crop") {
        Some(v) => {
            let c: Vec<i64> = v.as_array().map(|x| x.iter().filter_map(Value::as_i64).collect()).unwrap_or_default();
            if c.len() != 4 {
                bail!("`crop` is [x0, y0, x1, y1] in pixels");
            }
            Some(c)
        }
        None => None,
    };
    let (_, _, img, seg) = analysis(ctx, &path, threshold)?;
    // Which pixels are the subject, in which box.
    let (keep, (x0, y0, x1, y1), blob, crop): (Box<dyn Fn(usize) -> bool>, _, _, _) = if let Some(c) = crop_arg {
        let cl = |v: i64, n: usize| v.clamp(0, n as i64 - 1) as usize;
        let (x0, x1) = (cl(c[0].min(c[2]), img.w), cl(c[0].max(c[2]), img.w));
        let (y0, y1) = (cl(c[1].min(c[3]), img.h), cl(c[1].max(c[3]), img.h));
        let labels = &seg.labels;
        let blobs = &seg.blobs;
        let f = move |i: usize| labels[i] != 0 && !blobs[labels[i] as usize - 1].text;
        (Box::new(f), (x0, y0, x1, y1), None, Some([x0, y0, x1, y1]))
    } else {
        let subjects: Vec<&segment::Blob> = seg.blobs.iter().filter(|b| !b.text).collect();
        let b = match blob_arg {
            Some(n) => seg.blob(n.max(0) as usize).ok_or_else(|| anyhow!("no blob {n}; analyze_image numbers them 1..{}", seg.blobs.len()))?,
            None if subjects.len() == 1 => subjects[0],
            None => bail!("the picture has {} subjects; pass `blob` (see analyze_image) or `crop`", subjects.len()),
        };
        let id = b.id as u32;
        let labels = &seg.labels;
        (Box::new(move |i: usize| labels[i] == id), (b.x0, b.y0, b.x1, b.y1), Some(b.id), None)
    };
    // Tighten to the silhouette and cut it out.
    let (mut tx0, mut ty0, mut tx1, mut ty1) = (usize::MAX, usize::MAX, 0, 0);
    for y in y0..=y1 {
        for x in x0..=x1 {
            if keep(y * img.w + x) {
                (tx0, ty0, tx1, ty1) = (tx0.min(x), ty0.min(y), tx1.max(x), ty1.max(y));
            }
        }
    }
    if tx0 == usize::MAX {
        bail!("no subject pixels in that box; check analyze_image's blobs, or lower `threshold`");
    }
    // Edge pixels that still look like the backdrop (anti-aliasing) aren't
    // subject.
    let edge_bg = |x: usize, y: usize| {
        let i = y * img.w + x;
        let outside = |xx: usize, yy: usize| xx < tx0 || yy < ty0 || xx > tx1 || yy > ty1 || !keep(yy * img.w + xx);
        !seg.bg.alpha
            && seg.bg.dist(x, y, img.px[i]) <= seg.threshold
            && (outside(x.wrapping_sub(1), y) || outside(x + 1, y) || outside(x, y.wrapping_sub(1)) || outside(x, y + 1))
    };
    let flip = boolean(a, "flip").unwrap_or(false);
    let (w, h) = (tx1 - tx0 + 1, ty1 - ty0 + 1);
    let mut px = Vec::with_capacity(w * h);
    for y in ty0..=ty1 {
        for k in 0..w {
            let x = if flip { tx1 - k } else { tx0 + k };
            let i = y * img.w + x;
            let p = img.px[i];
            px.push(if keep(i) && !edge_bg(x, y) { [p[0], p[1], p[2], 255] } else { [p[0], p[1], p[2], 0] });
        }
    }
    drop(keep);
    let height = num(a, "height");
    let origin = match get(a, "origin") {
        Some(v) => Some(compare_pair(v).context("`origin` is [h, v]")?),
        None => None,
    };
    if origin.is_some() && height.is_none() {
        bail!("`origin` needs `height` (it pins scale and position together)");
    }
    // A pinned scale also pins the position, in world coordinates, so later
    // cropping or adding parts doesn't slide the reference around.
    let origin = origin.or_else(|| {
        let hh = height.filter(|h| *h > 0.0)?;
        let t = target(ctx, a, true).ok()?;
        let (hx, vx) = (view.h_axis(), view.v_axis());
        let o = default_origin(&t, view, hh);
        Some([o[0] + t.zero[hx] as f32, o[1] + t.zero[vx] as f32])
    });
    let r = Ref {
        name: string(a, "name").map_or_else(|| format!("{} {}", Path::new(&path).file_stem().map_or("reference".into(), |s| s.to_string_lossy()), view.key()), str::to_string),
        source: path.clone(),
        pic: RefPic::new(w, h, px),
        blob,
        crop,
        flip,
        height: height.filter(|h| *h > 0.0),
        origin,
        traced: None,
    };
    let saved = save_ref(&project, view, &r, true)?;
    let what = blob.map_or_else(|| format!("crop [{x0}, {y0}]-[{x1}, {y1}]"), |b| format!("blob {b}"));
    let mut msg = format!(
        "Set the {} reference `{}` for project `{project}`: {what} of {path}, silhouette {w}x{h} px (w/h {:.2}){}. Saved to {}.",
        view.key(),
        r.name,
        w as f32 / h as f32,
        if flip { ", mirrored" } else { "" },
        saved.display()
    );
    if let Some(hh) = r.height {
        msg.push_str(&format!(" Pinned at {hh} voxels tall; compare_reference measures at that scale."));
    }
    msg.push_str(&format!(" Next: reference_profile {{view: \"{}\", height: N}} for voxel spans.", view.key()));
    let pic = Canvas::new(w as u32, h as u32, r.pic.px.iter().flat_map(|p| if p[3] >= 128 { *p } else { [40, 40, 40, 255] }).collect());
    ctx.ed.tools.reference.refs.insert(view, r);
    let shown = shrink(&pic, 480);
    Ok(vec![super::image(shown.w, shown.h, &shown.px)?, text(msg)])
}

fn compare_pair(v: &Value) -> Option<[f32; 2]> {
    let a = v.as_array().filter(|a| a.len() == 2)?;
    Some([a[0].as_f64()? as f32, a[1].as_f64()? as f32])
}

/// Scale a canvas down to fit `max` pixels (nearest neighbor).
fn shrink(c: &Canvas, max: u32) -> Canvas {
    let s = (max as f32 / c.w.max(c.h) as f32).min(1.0);
    let (w, h) = (((c.w as f32 * s) as u32).max(1), ((c.h as f32 * s) as u32).max(1));
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = (((x as f32 + 0.5) / s) as u32, ((y as f32 + 0.5) / s) as u32);
            let i = ((sy.min(c.h - 1) * c.w + sx.min(c.w - 1)) * 4) as usize;
            px.extend_from_slice(&c.px[i..i + 4]);
        }
    }
    Canvas::new(w, h, px)
}

fn list_references(ctx: &mut Ctx) -> Result<Vec<Value>> {
    ensure_loaded(ctx);
    let project = ctx.ed.name();
    let st = &ctx.ed.tools.reference;
    if st.refs.is_empty() {
        return Ok(vec![text(format!("Project `{project}` has no references; analyze_image then set_reference adds them."))]);
    }
    let dir = refs_dir(&project);
    let list: Vec<Value> = st
        .refs
        .iter()
        .map(|(v, r)| {
            json!({
                "view": v.key(),
                "name": r.name,
                "source": r.source,
                "blob": r.blob,
                "crop": r.crop,
                "flip": r.flip,
                "size_px": [r.pic.w, r.pic.h],
                "pinned_height": r.height,
                "pinned_origin": r.origin,
                "traced": r.traced.map(|t| json!({ "height": t.0, "origin": t.1 })),
                "file": dir.join(format!("{}.png", v.key())).display().to_string(),
            })
        })
        .collect();
    Ok(vec![text(serde_json::to_string_pretty(&json!({ "project": project, "references": list }))?)])
}

// ---------------------------------------------------------------- targets

/// What a reference is traced onto or compared with: a volume, the world
/// position of its cell 0, and the offset reported coordinates are taken
/// from (a part's own coordinates, or world).
struct Target {
    label: String,
    vol: crate::scene::Volume,
    origin: IVec3,
    /// World position of the reported coordinates' 0.
    zero: IVec3,
}

fn part_target(ctx: &Ctx, id: usize) -> Result<Target> {
    let label = format!("part `{}`", ctx.scene.world.label(id));
    if ctx.scene.editing() == Some(id) {
        return Ok(Target { label, vol: ctx.scene.volume.clone(), origin: ctx.scene.origin, zero: ctx.scene.origin });
    }
    let w = ctx.scene.synced_world();
    let (vol, origin) = w.checkout(id).ok_or_else(|| anyhow!("`{}` isn't a part with voxels", w.label(id)))?;
    Ok(Target { label, vol, origin, zero: origin })
}

fn scene_target(ctx: &Ctx) -> Result<Target> {
    let w = ctx.scene.synced_world();
    let c = w.composite(None).ok_or_else(|| anyhow!("the scene has no visible parts"))?;
    Ok(Target { label: "the whole scene (world coordinates)".to_string(), vol: c.volume, origin: c.origin, zero: IVec3::ZERO })
}

/// `part` if given, else the part being edited, else the whole scene.
fn target(ctx: &Ctx, a: &Value, whole_by_default: bool) -> Result<Target> {
    match get(a, "part") {
        Some(p) => part_target(ctx, super::find_part(ctx.scene, Some(p))?),
        None => match ctx.scene.editing() {
            Some(id) if !whole_by_default => part_target(ctx, id),
            _ => scene_target(ctx),
        },
    }
}

/// Default bottom-center for a reference `cells_v` tall: the middle of the
/// target's box along h, v at its bottom (top view: centered).
fn default_origin(t: &Target, view: View, cells_v: f32) -> [f32; 2] {
    let (h, v) = (view.h_axis(), view.v_axis());
    let lo = t.origin - t.zero;
    let mid = lo[h] as f32 + t.vol.dims[h] as f32 * 0.5;
    let bottom = if view == View::Top { lo[v] as f32 + (t.vol.dims[v] as f32 - cells_v) * 0.5 } else { lo[v] as f32 };
    [mid, bottom.round()]
}

/// Bottom-center [h, v] in model coordinates to a placement in screen
/// cells.
fn placement(view: View, pic: &RefPic, scale: f32, origin: [f32; 2]) -> Placement {
    let su = if view.mirrored() { -origin[0] } else { origin[0] };
    Placement::anchored(pic, scale, su, origin[1])
}

// ---------------------------------------------------------------- profile

const LETTERS: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ";

fn reference_profile(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let view = view_arg(a)?;
    let t = target(ctx, a, false)?;
    let r = get_ref(ctx, view)?;
    let pic = &r.pic;
    let scale = match (num(a, "height"), num(a, "width"), num(a, "scale")) {
        (Some(h), _, _) => h / pic.h as f32,
        (None, Some(w), _) => w / pic.w as f32,
        (None, None, Some(s)) => s,
        _ => match r.height {
            Some(h) => h / pic.h as f32,
            None => bail!("give the size to trace at: `height` in voxels (or `width`, or `scale` = voxels per pixel)"),
        },
    };
    if !(scale > 0.0) || pic.h as f32 * scale > 1024.0 || pic.w as f32 * scale > 1024.0 {
        bail!("that's {:.0}x{:.0} voxels; pick a size between 1 and 1024", pic.w as f32 * scale, pic.h as f32 * scale);
    }
    let cells_v = pic.h as f32 * scale;
    let origin = match get(a, "origin") {
        Some(v) => compare_pair(v).context("`origin` is [h, v]")?,
        None => default_origin(&t, view, cells_v),
    };
    let place = placement(view, pic, scale, origin);
    let su0 = place.left.floor() as i32;
    let cols = ((place.left + place.width(pic)).ceil() as i32 - su0).max(1) as usize;
    let v0 = place.bottom.floor() as i32;
    let rows = ((place.bottom + cells_v).ceil() as i32 - v0).max(1) as usize;
    let grid = compare::raster(pic, &place, su0, v0, cols, rows);
    // Dominant reference colors, and each cell's most common one.
    let all: Vec<Vec3> = (0..pic.px.len()).filter(|&i| pic.on(i)).step_by(((pic.w * pic.h) / 40000).max(1)).map(|i| pic.labs[i]).collect();
    let clusters = segment::kmeans(&all, 8, 12);
    let centers: Vec<Vec3> = clusters.iter().map(|c| c.0).collect();
    let total: usize = clusters.iter().map(|c| c.1).sum();
    let mut letters = Grid::new(su0, v0, cols, rows, None::<usize>);
    for rr in 0..rows {
        for c in 0..cols {
            if grid.cells[rr * cols + c].is_none() {
                continue;
            }
            let (on, _) = compare::cell_pixels(pic, &place, su0 + c as i32, v0 + rr as i32);
            let mut counts = vec![0usize; centers.len()];
            for i in on {
                counts[segment::nearest(&centers, pic.labs[i])] += 1;
            }
            letters.cells[rr * cols + c] = (0..centers.len()).max_by_key(|&k| counts[k]);
        }
    }
    let (hname, vname) = (AXES[view.h_axis()], AXES[view.v_axis()]);
    let mut out = vec![format!(
        "{} reference `{}` traced at {:.1} x {:.1} voxels ({:.4} voxels per pixel) onto {}. Bottom-center at [{hname} {:.1}, {vname} {:.1}].",
        view.title(),
        r.name,
        pic.w as f32 * scale,
        cells_v,
        scale,
        t.label,
        origin[0],
        origin[1]
    )];
    let lo = t.origin - t.zero;
    let hi = lo + t.vol.dims - IVec3::ONE;
    let (h, v) = (view.h_axis(), view.v_axis());
    let mut outside = Vec::new();
    let (fs0, fs1) = (view.coord(su0), view.coord(su0 + cols as i32 - 1));
    if fs0.min(fs1) < lo[h] || fs0.max(fs1) > hi[h] {
        outside.push(format!("{hname} {}..{} (canvas {}..{})", fs0.min(fs1), fs0.max(fs1), lo[h], hi[h]));
    }
    if v0 < lo[v] || v0 + rows as i32 - 1 > hi[v] {
        outside.push(format!("{vname} {}..{} (canvas {}..{})", v0, v0 + rows as i32 - 1, lo[v], hi[v]));
    }
    if !outside.is_empty() {
        out.push(format!("Warning: the trace spans {}; resize_canvas or pick a smaller size so it fits.", outside.join(" and ")));
    }
    // Spans per row, top down, identical rows merged.
    out.push(format!("Filled spans per {vname} row (top down), {hname} from..to inclusive:"));
    let row_spans = |rr: usize| -> Vec<(i32, i32)> {
        let mut spans: Vec<(i32, i32)> = Vec::new();
        let mut c = 0;
        while c < cols {
            if grid.cells[rr * cols + c].is_some() {
                let start = c;
                while c + 1 < cols && grid.cells[rr * cols + c + 1].is_some() {
                    c += 1;
                }
                let (p, q) = (view.coord(su0 + start as i32), view.coord(su0 + c as i32));
                spans.push((p.min(q), p.max(q)));
            }
            c += 1;
        }
        spans.sort();
        spans
    };
    let mut rr = rows;
    while rr > 0 {
        rr -= 1;
        let spans = row_spans(rr);
        let mut end = rr;
        while end > 0 && row_spans(end - 1) == spans {
            end -= 1;
        }
        let (top, bot) = (v0 + rr as i32, v0 + end as i32);
        let label = if top == bot { format!("{vname}={top}") } else { format!("{vname}={bot}-{top}") };
        let body = if spans.is_empty() { "empty".to_string() } else { spans.iter().map(|(p, q)| if p == q { format!("{p}") } else { format!("{p}..{q}") }).collect::<Vec<_>>().join(", ") };
        out.push(format!("  {label}: {body}"));
        rr = end;
    }
    // The letter map, as the view shows it.
    if rows * (cols + 6) > 40_000 {
        out.push("(Map skipped: too big; trace at a smaller size to see it.)".to_string());
    } else {
        out.push(format!("Map as seen from the {} (screen right = {}), letters = colors below, '.' empty:", view.key(), view.side(true)));
        out.push(ruler(view, su0, cols));
        for rr in (0..rows).rev() {
            let mut line = format!("{:>4} ", v0 + rr as i32);
            for c in 0..cols {
                line.push(letters.cells[rr * cols + c].map_or('.', |k| LETTERS[k.min(LETTERS.len() - 1)] as char));
            }
            out.push(line);
        }
        let legend: Vec<String> = clusters.iter().enumerate().map(|(k, c)| format!("{}={} {}", LETTERS[k.min(LETTERS.len() - 1)] as char, hex3(segment::srgb(c.0)), pct(c.1 as f32 / total.max(1) as f32))).collect();
        out.push(format!("colors: {}", legend.join(" ")));
    }
    // Remember how it was traced, in world coordinates, for compare_reference's pinned mode.
    let world = [origin[0] + t.zero[h] as f32, origin[1] + t.zero[v] as f32];
    let project = ctx.ed.name();
    if let Some(r) = ctx.ed.tools.reference.refs.get_mut(&view) {
        r.traced = Some((cells_v, world));
        let _ = save_ref(&project, view, r, false);
    }
    out.push("compare_reference with align \"pinned\" measures against this exact trace (scale and position).".to_string());
    Ok(vec![text(out.join("\n"))])
}

/// Column labels at coordinates divisible by 5, right-aligned on their
/// column, like get_slice's ruler.
fn ruler(view: View, su0: i32, cols: usize) -> String {
    let mut line: Vec<char> = format!("{:>4} ", AXES[view.h_axis()]).chars().collect();
    let prefix = line.len();
    line.resize(prefix + cols, ' ');
    let mut free_from = prefix;
    for c in 0..cols {
        let h = view.coord(su0 + c as i32);
        if h.rem_euclid(5) != 0 {
            continue;
        }
        let label: Vec<char> = h.to_string().chars().collect();
        let end = prefix + c + 1;
        let Some(start) = end.checked_sub(label.len()) else { continue };
        if start < free_from {
            continue;
        }
        line[start..end].copy_from_slice(&label);
        free_from = end + 1;
    }
    line.into_iter().collect::<String>().trim_end().to_string()
}

// ---------------------------------------------------------------- compare

fn compare_reference(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let view = view_arg(a)?;
    let t = target(ctx, a, true)?;
    let palette = ctx.scene.palette.clone();
    let r = get_ref(ctx, view)?;
    let pic = &r.pic;
    let model = compare::project(&t.vol, t.origin - t.zero, view);
    if compare::bounds(&model, |&c| c != 0).is_none() {
        bail!("{} has no voxels to compare", t.label);
    }
    let (place, how, align, pinned) = alignment(view, &t, r, &model, a)?;
    let c = compare::compare(view, &model, &palette, pic, place);
    compare_report(view, r, &t, &c, &palette, &how, align, pinned)
}

/// Where the reference sits over `model` (a projection of `t`): its
/// placement, a description, the align mode used and whether a pinned scale
/// exists.
fn alignment<'a>(view: View, t: &Target, r: &Ref, model: &compare::Grid<u8>, a: &'a Value) -> Result<(Placement, String, &'a str, bool)> {
    let pic = &r.pic;
    let (h, v) = (view.h_axis(), view.v_axis());
    let pinned = r.height.map(|hh| (hh, r.origin)).or(r.traced.map(|(hh, o)| (hh, Some(o))));
    let align = string(a, "align").unwrap_or(if r.height.is_some() { "pinned" } else { "fit" });
    let (place, how) = match align {
        "fit" => {
            let (p, shift) = compare::fit(view, &model, pic).context("nothing to fit")?;
            let nudge = if shift != 0 { format!(", nudged {shift:+} cells toward screen right for best overlap") } else { String::new() };
            (p, format!("fit: reference scaled to the model's projected height ({:.4} voxels per pixel), bottom-aligned and centered{nudge}", p.scale))
        }
        "pinned" => {
            let (hh, o) = pinned.ok_or_else(|| anyhow!("nothing pinned for {}: give set_reference a height, or trace it with reference_profile first; or use align \"fit\"", view.key()))?;
            let scale = hh / pic.h as f32;
            let o = match o {
                Some(w) => [w[0] - t.zero[h] as f32, w[1] - t.zero[v] as f32],
                None => default_origin(&t, view, hh),
            };
            (placement(view, pic, scale, o), format!("pinned: {hh:.1} voxels tall, bottom-center at [{} {:.1}, {} {:.1}]", AXES[h], o[0], AXES[v], o[1]))
        }
        other => bail!("unknown align {other:?}; use \"fit\" or \"pinned\""),
    };
    Ok((place, how, align, pinned.is_some()))
}

#[allow(clippy::too_many_arguments)]
fn compare_report(view: View, r: &Ref, t: &Target, c: &compare::Compare, palette: &[[u8; 4]], how: &str, align: &str, pinned: bool) -> Result<Vec<Value>> {
    let pic = &r.pic;
    let mut out = vec![
        format!("compare_reference {} `{}` vs {}. Alignment {how}.", view.title(), r.name, t.label),
        format!(
            "Silhouette IoU {} (precision {}: model cells inside the reference; recall {}: reference covered by the model). Both {} cells, model-only {} (magenta), reference-only {} (green).",
            pct(c.iou),
            pct(c.precision),
            pct(c.recall),
            c.both,
            c.model_only,
            c.ref_only
        ),
    ];
    let findings = compare::width_findings(c, 8);
    if findings.is_empty() {
        out.push("Outline: matches row by row (within a voxel).".to_string());
    } else {
        out.push("Outline findings, largest first (left/right as the view shows; cells = mismatched area):".to_string());
        out.extend(findings.iter().enumerate().map(|(i, f)| format!("  {}. {f}", i + 1)));
    }
    let (table, worst) = compare::color_findings(c, palette, 5);
    out.push(format!("Color: mean dE {:.1} over {} shared cells; agreement {} (dE 0 = same, ~10 = clearly different shade, 25+ = different color).", c.mean_de, c.both, pct(c.agreement)));
    if !table.is_empty() {
        out.push("  Mean dE by region, top down:".to_string());
        out.extend(table.iter().map(|l| format!("    {l}")));
    }
    if !worst.is_empty() {
        out.push("  Worst color regions:".to_string());
        out.extend(worst.iter().map(|l| format!("    - {l}")));
    }
    out.push(format!("Score {:.0}/100 = 75% x IoU {:.2} + 25% x color agreement {:.2}.", c.score, c.iou, c.agreement));
    if align == "fit" && pinned {
        out.push("A trace or pinned scale exists: align \"pinned\" also checks size and position.".to_string());
    }
    let img = compare::diff_image(c, pic, palette);
    Ok(vec![super::image(img.w, img.h, &img.px)?, text(out.join("\n"))])
}

// ---------------------------------------------------------------- painting

fn color_set(ctx: &Ctx, a: &Value, key: &str) -> Result<Option<Vec<u8>>> {
    let Some(list) = get(a, key) else { return Ok(None) };
    let list = list.as_array().ok_or_else(|| anyhow!("`{key}` is a list of colors"))?;
    list.iter().map(|c| super::color(ctx.scene, c)).collect::<Result<Vec<u8>>>().map(Some)
}

fn paint_from_reference(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let view = view_arg(a)?;
    let prev = ctx.scene.editing();
    let w = ctx.scene.synced_world();
    let comp = w.composite(None).ok_or_else(|| anyhow!("the scene has no visible parts"))?;
    let t = Target { label: "the whole scene".into(), vol: comp.volume.clone(), origin: comp.origin, zero: IVec3::ZERO };
    let model = compare::project(&t.vol, t.origin, view);
    let r = get_ref(ctx, view)?;
    let (place, how, _, _) = alignment(view, &t, r, &model, a)?;
    let refc = compare::raster(&r.pic, &place, model.su0, model.v0, model.cols, model.rows);
    let name = r.name.clone();

    // Allowed colors, protected colors and the parts to touch.
    let palette = ctx.scene.palette.clone();
    let mut allowed: Vec<u8> = match (color_set(ctx, a, "colors")?, get(a, "palette_range").and_then(Value::as_array)) {
        (Some(c), _) => c,
        (None, Some(r)) if r.len() == 2 => {
            let (lo, hi) = (r[0].as_u64().unwrap_or(1).clamp(1, 255) as u8, r[1].as_u64().unwrap_or(255).clamp(1, 255) as u8);
            (lo.min(hi)..=lo.max(hi)).collect()
        }
        _ => {
            let mut used = [false; 256];
            comp.volume.cells.iter().for_each(|&c| used[c as usize] = true);
            (1..=255u8).filter(|&c| used[c as usize]).collect()
        }
    };
    let keep = color_set(ctx, a, "keep")?.unwrap_or_default();
    allowed.retain(|c| !keep.contains(c));
    if allowed.is_empty() {
        bail!("no colors to paint with; pass `colors` or `palette_range`");
    }
    let only = color_set(ctx, a, "only_color")?;
    let parts: Option<Vec<u32>> = match get(a, "parts").and_then(Value::as_array) {
        Some(list) => Some(list.iter().map(|p| super::find_part(ctx.scene, Some(p)).map(|id| id as u32)).collect::<Result<_>>()?),
        None => None,
    };
    let labs: Vec<(u8, Vec3)> = allowed.iter().map(|&c| (c, segment::lab(palette[c as usize]))).collect();
    let depth = int(a, "depth").unwrap_or(1).clamp(1, 8) as i32;

    // Walk each screen cell into the scene: paint the first voxel (and
    // `depth` behind it in the same part) with the nearest allowed color.
    let (h, v, d, mirrored, from_max) = view.axes();
    let dims = comp.volume.dims;
    let (cols, rows) = (dims[h], dims[v]);
    let nearest = |want: &Vec3| labs.iter().min_by(|x, y| compare::delta_e(x.1, *want).total_cmp(&compare::delta_e(y.1, *want))).map(|b| b.0).unwrap();
    let mut chosen: Vec<u8> = (0..rows * cols)
        .map(|i| match (refc.get(model.su0 + i % cols, model.v0 + i / cols), model.cells[i as usize]) {
            (Some(Some(want)), m) if m != 0 => nearest(want),
            _ => 0,
        })
        .collect();
    // Majority filter: a cell takes the color most of its 3x3 neighbours
    // share (5 of 9) so single-voxel speckle from the photo goes away.
    for _ in 0..int(a, "clean").unwrap_or(1).clamp(0, 4) {
        let prev = chosen.clone();
        for r in 0..rows {
            for c in 0..cols {
                let i = (r * cols + c) as usize;
                if prev[i] == 0 {
                    continue;
                }
                let mut count = [0u8; 256];
                for (dr, dc) in (-1..=1).flat_map(|dr| (-1..=1).map(move |dc| (dr, dc))) {
                    let (rr, cc) = (r + dr, c + dc);
                    if rr >= 0 && cc >= 0 && rr < rows && cc < cols {
                        count[prev[(rr * cols + cc) as usize] as usize] += 1;
                    }
                }
                let (best, n) = (1..256).map(|k| (k as u8, count[k])).max_by_key(|&(_, n)| n).unwrap();
                if n >= 5 {
                    chosen[i] = best;
                }
            }
        }
    }
    let mut edits: std::collections::BTreeMap<u32, Vec<(IVec3, u8)>> = Default::default();
    for row in 0..dims[v] {
        for col in 0..dims[h] {
            let best = chosen[(row * cols + col) as usize];
            if best == 0 {
                continue;
            }
            let lh = if mirrored { dims[h] - 1 - col } else { col };
            let mut owner = 0u32;
            let mut painted = 0;
            for step in 0..dims[d] {
                let mut p = IVec3::ZERO;
                p[h] = lh;
                p[v] = row;
                p[d] = if from_max { dims[d] - 1 - step } else { step };
                let i = comp.volume.index(p);
                let c = comp.volume.cells[i];
                if c == 0 {
                    if owner != 0 { break } else { continue }
                }
                let o = comp.owners[i];
                if owner == 0 {
                    owner = o;
                } else if o != owner {
                    break;
                }
                let skip = keep.contains(&c) || only.as_ref().is_some_and(|l| !l.contains(&c)) || parts.as_ref().is_some_and(|l| !l.contains(&o));
                if !skip && c != best {
                    edits.entry(o).or_default().push((comp.origin + p, best));
                }
                painted += 1;
                if painted >= depth {
                    break;
                }
            }
        }
    }

    let mut lines = Vec::new();
    let mut total = 0;
    for (id, cells) in &edits {
        super::focus_part(ctx, Some(&json!(*id)))?;
        let origin = ctx.scene.origin;
        let mut new = ctx.scene.volume.clone();
        let mut n = 0;
        for &(p, c) in cells {
            let l = p - origin;
            if new.contains(l) {
                new.set_index(new.index(l), c);
                n += 1;
            }
        }
        super::apply(ctx, new);
        total += n;
        lines.push(format!("`{}` {n}", ctx.scene.world.label(*id as usize)));
    }
    // Back to what was being edited (or World mode).
    if ctx.scene.editing() != prev {
        crate::app::set_mode(ctx.scene, ctx.ed, prev);
        crate::app::sync_volume(ctx.r, ctx.scene, &mut ctx.ed.editor);
    }
    let mut out = format!("Painted {total} voxels from `{name}` ({}), choosing among {} colors. Alignment {how}.", view.title(), allowed.len());
    if !lines.is_empty() {
        out.push_str(&format!("\nPer part: {}.", lines.join(", ")));
    }
    out.push_str("\nNext: compare_reference to check color agreement; shade_edges / vary_colors for depth.");
    Ok(vec![text(out)])
}
