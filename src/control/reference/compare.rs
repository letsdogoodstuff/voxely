//! Seeing the model from a view without the GPU, laying a reference
//! silhouette over it at voxel resolution, and measuring the difference.
//!
//! Everything is in screen cells: column `su` grows to the right as the
//! view shows it, row `v` grows up. `su` maps to the model coordinate on
//! the view's horizontal axis with `coord` (views seen from +y or -x are
//! mirrored: screen right is the axis's negative direction).

use glam::{IVec3, Vec3};

use super::segment::{lab, srgb};
use crate::control::view::draw::{Canvas, Rgb, text_size};
use crate::scene::Volume;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum View {
    Front,
    Back,
    Left,
    Right,
    Top,
}

pub const AXES: [&str; 3] = ["x", "y", "z"];

impl View {
    pub const ALL: [View; 5] = [View::Front, View::Back, View::Left, View::Right, View::Top];

    /// "front", or a screenshot title like "FRONT -Y".
    pub fn parse(s: &str) -> Option<View> {
        let first = s.split_whitespace().next().unwrap_or("").to_ascii_lowercase();
        View::ALL.into_iter().find(|v| v.key() == first)
    }

    pub fn key(self) -> &'static str {
        match self {
            View::Front => "front",
            View::Back => "back",
            View::Left => "left",
            View::Right => "right",
            View::Top => "top",
        }
    }

    /// The screenshot tile title for this view.
    pub fn title(self) -> &'static str {
        match self {
            View::Front => "FRONT -Y",
            View::Back => "BACK +Y",
            View::Left => "LEFT -X",
            View::Right => "RIGHT +X",
            View::Top => "TOP +Z",
        }
    }

    /// Horizontal axis, vertical axis, depth axis, whether screen right is
    /// the horizontal axis's negative direction, and whether the eye is on
    /// the depth axis's + side. Matches screenshot's views: front looks
    /// from -y (right = +x), back from +y (right = -x), right from +x
    /// (right = +y), left from -x (right = -y), top from +z (right = +x,
    /// up = +y).
    pub fn axes(self) -> (usize, usize, usize, bool, bool) {
        match self {
            View::Front => (0, 2, 1, false, false),
            View::Back => (0, 2, 1, true, true),
            View::Right => (1, 2, 0, false, true),
            View::Left => (1, 2, 0, true, false),
            View::Top => (0, 1, 2, false, true),
        }
    }

    pub fn h_axis(self) -> usize {
        self.axes().0
    }
    pub fn v_axis(self) -> usize {
        self.axes().1
    }
    pub fn mirrored(self) -> bool {
        self.axes().3
    }

    /// Model coordinate of screen column `su`, and back (it's its own
    /// inverse).
    pub fn coord(self, su: i32) -> i32 {
        if self.mirrored() { -su - 1 } else { su }
    }

    /// "-x" / "+y": the model direction toward screen left or right.
    pub fn side(self, right: bool) -> String {
        let sign = if right != self.mirrored() { "+" } else { "-" };
        format!("{sign}{}", AXES[self.h_axis()])
    }
}

/// Cells over screen columns `su0..su0+cols` and rows `v0..v0+rows`,
/// row-major from the bottom row.
#[derive(Clone)]
pub struct Grid<T> {
    pub su0: i32,
    pub v0: i32,
    pub cols: usize,
    pub rows: usize,
    pub cells: Vec<T>,
}

impl<T: Clone> Grid<T> {
    pub fn new(su0: i32, v0: i32, cols: usize, rows: usize, fill: T) -> Self {
        Grid { su0, v0, cols, rows, cells: vec![fill; cols * rows] }
    }

    pub fn get(&self, su: i32, v: i32) -> Option<&T> {
        let (c, r) = (su - self.su0, v - self.v0);
        (c >= 0 && r >= 0 && (c as usize) < self.cols && (r as usize) < self.rows).then(|| &self.cells[r as usize * self.cols + c as usize])
    }
}

/// The model seen from `view`: per screen cell, the color of the first
/// voxel along the view direction (0 = empty). `origin` is the world
/// position of the volume's cell 0.
pub fn project(vol: &Volume, origin: IVec3, view: View) -> Grid<u8> {
    let (h, v, d, mirrored, from_max) = view.axes();
    let dims = vol.dims;
    let su0 = if mirrored { -(origin[h] + dims[h]) } else { origin[h] };
    let mut g = Grid::new(su0, origin[v], dims[h] as usize, dims[v] as usize, 0u8);
    for r in 0..dims[v] {
        for c in 0..dims[h] {
            let lh = if mirrored { dims[h] - 1 - c } else { c };
            for t in 0..dims[d] {
                let mut p = IVec3::ZERO;
                p[h] = lh;
                p[v] = r;
                p[d] = if from_max { dims[d] - 1 - t } else { t };
                let color = vol.get(p);
                if color != 0 {
                    g.cells[(r * dims[h] + c) as usize] = color;
                    break;
                }
            }
        }
    }
    g
}

/// Filled cells' bounds: (su_min, su_max, v_min, v_max), inclusive.
pub fn bounds<T>(g: &Grid<T>, filled: impl Fn(&T) -> bool) -> Option<(i32, i32, i32, i32)> {
    let mut b: Option<(i32, i32, i32, i32)> = None;
    for (i, c) in g.cells.iter().enumerate() {
        if filled(c) {
            let (su, v) = (g.su0 + (i % g.cols) as i32, g.v0 + (i / g.cols) as i32);
            b = Some(b.map_or((su, su, v, v), |(a, z, lo, hi)| (a.min(su), z.max(su), lo.min(v), hi.max(v))));
        }
    }
    b
}

// ---------------------------------------------------------------- reference

/// A reference view: its pixels, with alpha marking the silhouette.
pub struct RefPic {
    pub w: usize,
    pub h: usize,
    pub px: Vec<[u8; 4]>,
    pub labs: Vec<Vec3>,
}

impl RefPic {
    pub fn new(w: usize, h: usize, px: Vec<[u8; 4]>) -> Self {
        let labs = px.iter().map(|&p| lab(p)).collect();
        RefPic { w, h, px, labs }
    }

    pub fn on(&self, i: usize) -> bool {
        self.px[i][3] >= 128
    }
}

/// Where the reference sits in screen cells: its left and bottom edges and
/// cells per pixel.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    pub left: f32,
    pub bottom: f32,
    pub scale: f32,
}

impl Placement {
    /// Its bottom-center at screen point (su, v).
    pub fn anchored(pic: &RefPic, scale: f32, su: f32, v: f32) -> Self {
        Placement { left: su - pic.w as f32 * scale * 0.5, bottom: v, scale }
    }

    pub fn width(&self, pic: &RefPic) -> f32 {
        pic.w as f32 * self.scale
    }

    pub fn height(&self, pic: &RefPic) -> f32 {
        pic.h as f32 * self.scale
    }
}

/// The reference pixels in cell (su, v): silhouette pixel indices and the
/// number of pixels the cell spans (including empty ones).
pub fn cell_pixels(pic: &RefPic, p: &Placement, su: i32, v: i32) -> (Vec<usize>, usize) {
    let s = p.scale;
    let (xa, xb) = ((su as f32 - p.left) / s, (su as f32 + 1.0 - p.left) / s);
    // Pixel rows count from the top.
    let (ya, yb) = (pic.h as f32 - (v as f32 + 1.0 - p.bottom) / s, pic.h as f32 - (v as f32 - p.bottom) / s);
    let range = |a: f32, b: f32| -> (i64, i64) {
        let (lo, hi) = ((a - 0.5).ceil() as i64, (b - 0.5).ceil() as i64);
        if hi > lo { (lo, hi) } else { (((a + b) * 0.5).floor() as i64, ((a + b) * 0.5).floor() as i64 + 1) }
    };
    let ((x0, x1), (y0, y1)) = (range(xa, xb), range(ya, yb));
    let mut on = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            if x >= 0 && y >= 0 && (x as usize) < pic.w && (y as usize) < pic.h {
                let i = y as usize * pic.w + x as usize;
                if pic.on(i) {
                    on.push(i);
                }
            }
        }
    }
    (on, ((x1 - x0) * (y1 - y0)) as usize)
}

/// The reference at voxel resolution: cells at least half covered by the
/// silhouette, with their mean OKLab color.
pub fn raster(pic: &RefPic, p: &Placement, su0: i32, v0: i32, cols: usize, rows: usize) -> Grid<Option<Vec3>> {
    let mut g = Grid::new(su0, v0, cols, rows, None);
    for r in 0..rows {
        for c in 0..cols {
            let (on, total) = cell_pixels(pic, p, su0 + c as i32, v0 + r as i32);
            if !on.is_empty() && on.len() * 2 >= total {
                let mean = on.iter().map(|&i| pic.labs[i]).sum::<Vec3>() / on.len() as f32;
                g.cells[r * cols + c] = Some(mean);
            }
        }
    }
    g
}

// ---------------------------------------------------------------- comparing

/// Color difference ×100 in OKLab with lightness counted half: the
/// reference is lit and shaded, palette colors are flat.
pub fn delta_e(a: Vec3, b: Vec3) -> f32 {
    let d = a - b;
    ((0.5 * d.x).powi(2) + d.y * d.y + d.z * d.z).sqrt() * 100.0
}

pub struct Compare {
    pub view: View,
    pub place: Placement,
    /// Model colors (0 = empty) and reference cells over one grid.
    pub model: Grid<u8>,
    pub refc: Grid<Option<Vec3>>,
    pub model_lab: Vec<Option<Vec3>>,
    pub both: usize,
    pub model_only: usize,
    pub ref_only: usize,
    pub iou: f32,
    pub precision: f32,
    pub recall: f32,
    /// Mean color difference and agreement (0..1) over cells both fill.
    pub mean_de: f32,
    pub agreement: f32,
    pub score: f32,
}

fn overlap(model: &Grid<u8>, pic: &RefPic, place: &Placement) -> (Grid<u8>, Grid<Option<Vec3>>) {
    let (mut lo_s, mut hi_s, mut lo_v, mut hi_v) = bounds(model, |&c| c != 0).unwrap_or((0, 0, 0, 0));
    lo_s = lo_s.min(place.left.floor() as i32);
    hi_s = hi_s.max((place.left + place.width(pic)).ceil() as i32 - 1);
    lo_v = lo_v.min(place.bottom.floor() as i32);
    hi_v = hi_v.max((place.bottom + place.height(pic)).ceil() as i32 - 1);
    let (su0, v0) = (lo_s - 1, lo_v - 1);
    let (cols, rows) = ((hi_s - lo_s + 3) as usize, (hi_v - lo_v + 3) as usize);
    let mut m = Grid::new(su0, v0, cols, rows, 0u8);
    for r in 0..rows {
        for c in 0..cols {
            m.cells[r * cols + c] = model.get(su0 + c as i32, v0 + r as i32).copied().unwrap_or(0);
        }
    }
    (m, raster(pic, place, su0, v0, cols, rows))
}

pub fn compare(view: View, model: &Grid<u8>, palette: &[[u8; 4]], pic: &RefPic, place: Placement) -> Compare {
    let (m, r) = overlap(model, pic, &place);
    let model_lab: Vec<Option<Vec3>> = m.cells.iter().map(|&c| (c != 0).then(|| lab(palette[c as usize]))).collect();
    let (mut both, mut mo, mut ro) = (0, 0, 0);
    let (mut de_sum, mut agree) = (0.0, 0.0);
    for (i, &c) in m.cells.iter().enumerate() {
        match (c != 0, r.cells[i]) {
            (true, Some(rc)) => {
                both += 1;
                let de = delta_e(model_lab[i].unwrap(), rc);
                de_sum += de;
                agree += (1.0 - de / 25.0).clamp(0.0, 1.0);
            }
            (true, None) => mo += 1,
            (false, Some(_)) => ro += 1,
            _ => {}
        }
    }
    let ratio = |a: usize, b: usize| if b == 0 { 0.0 } else { a as f32 / b as f32 };
    let iou = ratio(both, both + mo + ro);
    let agreement = if both == 0 { 0.0 } else { agree / both as f32 };
    Compare {
        view,
        place,
        model: m,
        refc: r,
        model_lab,
        both,
        model_only: mo,
        ref_only: ro,
        iou,
        precision: ratio(both, both + mo),
        recall: ratio(both, both + ro),
        mean_de: if both == 0 { 0.0 } else { de_sum / both as f32 },
        agreement,
        score: 100.0 * (0.75 * iou + 0.25 * agreement),
    }
}

/// Scale the reference to the model's projected height, bottom-aligned,
/// centered on the model's box, then nudged sideways (up to an eighth of
/// the model's width) for the best overlap. Returns the placement and the
/// nudge in cells.
pub fn fit(view: View, model: &Grid<u8>, pic: &RefPic) -> Option<(Placement, i32)> {
    let (s0, s1, v0, v1) = bounds(model, |&c| c != 0)?;
    let scale = (v1 - v0 + 1) as f32 / pic.h as f32;
    let center = (s0 + s1 + 1) as f32 * 0.5;
    let reach = ((s1 - s0 + 1) / 8).max(1);
    let mut best = (f32::MIN, 0i32);
    for shift in -reach..=reach {
        let p = Placement::anchored(pic, scale, center + shift as f32, v0 as f32);
        let iou = compare(view, model, &[[0; 4]; 256], pic, p).iou;
        if iou > best.0 + 1e-6 || (iou > best.0 - 1e-6 && shift.abs() < best.1.abs()) {
            best = (iou, shift);
        }
    }
    Some((Placement::anchored(pic, scale, center + best.1 as f32, v0 as f32), best.1))
}

// ---------------------------------------------------------------- findings

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side {
    Ok,
    Short,
    Over,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Missing,
    Extra,
    Inner,
    Shifted(bool),
    Sides(Side, Side),
}

struct RowInfo {
    v: i32,
    kind: Option<Kind>,
    /// Left and right edge errors (positive: model short of the reference).
    dl: i32,
    dr: i32,
    model: Option<(i32, i32)>,
    refe: Option<(i32, i32)>,
    mo: usize,
    ro: usize,
}

fn extent(row: impl Iterator<Item = (i32, bool)>) -> Option<(i32, i32)> {
    row.filter(|r| r.1).fold(None, |e, (su, _)| Some(e.map_or((su, su), |(a, _): (i32, i32)| (a, su))))
}

fn rows_info(c: &Compare) -> Vec<RowInfo> {
    let (m, r) = (&c.model, &c.refc);
    let tol = 1;
    (0..m.rows)
        .map(|row| {
            let v = m.v0 + row as i32;
            let cells = |k: usize| (m.su0 + k as i32, m.cells[row * m.cols + k] != 0, r.cells[row * m.cols + k].is_some());
            let model = extent((0..m.cols).map(|k| (cells(k).0, cells(k).1)));
            let refe = extent((0..m.cols).map(|k| (cells(k).0, cells(k).2)));
            let mo = (0..m.cols).filter(|&k| cells(k).1 && !cells(k).2).count();
            let ro = (0..m.cols).filter(|&k| !cells(k).1 && cells(k).2).count();
            let (mut dl, mut dr) = (0, 0);
            let kind = match (model, refe) {
                (None, None) => None,
                (None, Some(_)) => Some(Kind::Missing),
                (Some(_), None) => Some(Kind::Extra),
                (Some((ml, mr)), Some((rl, rr))) => {
                    (dl, dr) = (ml - rl, rr - mr);
                    let side = |d: i32| if d > tol { Side::Short } else if d < -tol { Side::Over } else { Side::Ok };
                    match (side(dl), side(dr)) {
                        (Side::Ok, Side::Ok) => (mo + ro >= 3).then_some(Kind::Inner),
                        (Side::Short, Side::Over) if (dl + dr).abs() <= tol => Some(Kind::Shifted(true)),
                        (Side::Over, Side::Short) if (dl + dr).abs() <= tol => Some(Kind::Shifted(false)),
                        (a, b) => Some(Kind::Sides(a, b)),
                    }
                }
            };
            RowInfo { v, kind, dl, dr, model, refe, mo, ro }
        })
        .collect()
}

fn span(a: i32, b: i32) -> String {
    if a == b { a.to_string() } else { format!("{a}-{b}") }
}

fn range_of(view: View, e: (i32, i32)) -> String {
    let (a, b) = (view.coord(e.0), view.coord(e.1));
    format!("{}..{}", a.min(b), a.max(b))
}

/// Rows where the outline differs, grouped into findings, largest first.
pub fn width_findings(c: &Compare, max: usize) -> Vec<String> {
    let info = rows_info(c);
    let view = c.view;
    let vname = AXES[view.v_axis()];
    let hname = AXES[view.h_axis()];
    let (lname, rname) = (format!("left ({})", view.side(false)), format!("right ({})", view.side(true)));
    let model_rows = info.iter().filter(|r| r.model.is_some()).map(|r| r.v).fold(None, |e: Option<(i32, i32)>, v| Some(e.map_or((v, v), |(a, b)| (a.min(v), b.max(v)))));
    let ref_rows = info.iter().filter(|r| r.refe.is_some()).map(|r| r.v).fold(None, |e: Option<(i32, i32)>, v| Some(e.map_or((v, v), |(a, b)| (a.min(v), b.max(v)))));
    // Group runs of rows with the same kind.
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < info.len() {
        let Some(k) = info[i].kind else {
            i += 1;
            continue;
        };
        let mut j = i;
        while j + 1 < info.len() && info[j + 1].kind == Some(k) {
            j += 1;
        }
        groups.push((i, j));
        i = j + 1;
    }
    let mut out: Vec<(usize, String)> = groups
        .into_iter()
        .filter_map(|(a, b)| {
            let rows = &info[a..=b];
            let area: usize = rows.iter().map(|r| r.mo + r.ro).sum();
            if area < 2 {
                return None;
            }
            let k = rows[0].kind?;
            let (v0, v1) = (rows[0].v, rows[b - a].v);
            let n = b - a + 1;
            let lo_hi = |f: &dyn Fn(&RowInfo) -> i32| {
                let vals: Vec<i32> = rows.iter().map(f).collect();
                span(*vals.iter().min().unwrap(), *vals.iter().max().unwrap())
            };
            let worst = rows.iter().max_by_key(|r| r.mo + r.ro).unwrap();
            let at = match (worst.model, worst.refe) {
                (Some(m), Some(r)) => format!(" (at {vname}={}: model {hname} {}, reference {hname} {})", worst.v, range_of(view, m), range_of(view, r)),
                _ => String::new(),
            };
            let text = match k {
                Kind::Missing => match model_rows {
                    Some((_, top)) if v0 > top => format!("reference extends {n} rows above the model's top ({vname}={top}); reference {hname} {}", worst.refe.map_or(String::new(), |e| range_of(view, e))),
                    Some((bot, _)) if v1 < bot => format!("reference extends {n} rows below the model's bottom ({vname}={bot})"),
                    _ => format!("model has nothing in these rows; reference spans {hname} {}", worst.refe.map_or(String::new(), |e| range_of(view, e))),
                },
                Kind::Extra => match ref_rows {
                    Some((_, top)) if v0 > top => format!("model extends {n} rows above the reference's top ({vname}={top})"),
                    Some((bot, _)) if v1 < bot => format!("model extends {n} rows below the reference's bottom ({vname}={bot})"),
                    _ => format!("model has voxels where the reference has none ({hname} {})", worst.model.map_or(String::new(), |e| range_of(view, e))),
                },
                Kind::Inner => {
                    let (mo, ro): (usize, usize) = (rows.iter().map(|r| r.mo).sum(), rows.iter().map(|r| r.ro).sum());
                    format!("outline matches but the inside differs: {mo} model-only cells (filled gaps), {ro} reference-only (holes in the model){at}")
                }
                Kind::Shifted(right) => {
                    let dir = if right { &rname } else { &lname };
                    format!("model shifted {} toward the {dir}{at}", lo_hi(&|r| r.dl.abs().max(r.dr.abs())))
                }
                Kind::Sides(l, r) => {
                    let phrase = |s: Side, name: &str, d: &dyn Fn(&RowInfo) -> i32| match s {
                        Side::Short => Some(format!("{} short on the {name}", lo_hi(&|x| d(x)))),
                        Side::Over => Some(format!("{} too far out on the {name}", lo_hi(&|x| -d(x)))),
                        Side::Ok => None,
                    };
                    let parts: Vec<String> = [phrase(l, &lname, &|x| x.dl), phrase(r, &rname, &|x| x.dr)].into_iter().flatten().collect();
                    let lead = match (l, r) {
                        (Side::Short, Side::Short) => "model narrower on both sides: ",
                        (Side::Over, Side::Over) => "model wider on both sides: ",
                        _ => "model ",
                    };
                    format!("{lead}{}{at}", parts.join(", "))
                }
            };
            Some((area, format!("{vname}={}: {text} [{area} cells]", span(v0, v1))))
        })
        .collect();
    out.sort_by_key(|o| std::cmp::Reverse(o.0));
    out.into_iter().take(max).map(|o| o.1).collect()
}

/// Color difference by region (row bands x left/center/right thirds):
/// the band table and the worst regions.
pub fn color_findings(c: &Compare, palette: &[[u8; 4]], max: usize) -> (Vec<String>, Vec<String>) {
    let (m, r) = (&c.model, &c.refc);
    let filled = |i: usize| m.cells[i] != 0 || r.cells[i].is_some();
    let Some((s0, s1, v0, v1)) = bounds(&Grid { su0: m.su0, v0: m.v0, cols: m.cols, rows: m.rows, cells: (0..m.cells.len()).map(filled).collect() }, |&b| b) else {
        return (Vec::new(), Vec::new());
    };
    let view = c.view;
    let (vname, hname) = (AXES[view.v_axis()], AXES[view.h_axis()]);
    let height = v1 - v0 + 1;
    let bands = (height / 6).clamp(2, 8);
    let width = s1 - s0 + 1;
    let thirds = ["left", "center", "right"];
    let mut table = Vec::new();
    let mut regions: Vec<(f32, String)> = Vec::new();
    for b in (0..bands).rev() {
        let (b0, b1) = (v0 + height * b / bands, v0 + height * (b + 1) / bands - 1);
        let mut cells_txt = Vec::new();
        for (t, name) in thirds.iter().enumerate() {
            let (t0, t1) = (s0 + width * t as i32 / 3, s0 + width * (t as i32 + 1) / 3 - 1);
            let (mut n, mut sum, mut ref_sum) = (0usize, 0.0f32, Vec3::ZERO);
            let mut counts = [0usize; 256];
            for v in b0..=b1 {
                for su in t0..=t1 {
                    let i = ((v - m.v0) as usize) * m.cols + (su - m.su0) as usize;
                    if let (Some(ml), Some(rl)) = (c.model_lab[i], r.cells[i]) {
                        n += 1;
                        sum += delta_e(ml, rl);
                        ref_sum += rl;
                        counts[m.cells[i] as usize] += 1;
                    }
                }
            }
            if n == 0 {
                cells_txt.push(format!("{name} -"));
                continue;
            }
            let de = sum / n as f32;
            cells_txt.push(format!("{name} {de:.0}"));
            let mode = (1..256).max_by_key(|&k| counts[k]).unwrap_or(1);
            let rc = srgb(ref_sum / n as f32);
            let (a, z) = (view.coord(t0), view.coord(t1));
            regions.push((
                de * (n as f32).sqrt(),
                format!(
                    "{vname}={} {name} ({hname} {}..{}): model {} vs reference ~#{:02x}{:02x}{:02x}, dE {de:.0} over {n} cells",
                    span(b0, b1),
                    a.min(z),
                    a.max(z),
                    crate::control::hex(palette[mode]),
                    rc[0],
                    rc[1],
                    rc[2]
                ),
            ));
        }
        table.push(format!("{vname}={}: {}", span(b0, b1), cells_txt.join(" | ")));
    }
    regions.sort_by(|a, b| b.0.total_cmp(&a.0));
    let worst = regions.into_iter().filter(|r| r.0 > 0.0).take(max).map(|r| r.1).collect();
    (table, worst)
}

// ---------------------------------------------------------------- images

const PANEL_BG: Rgb = [30, 30, 34];
pub const MAGENTA: Rgb = [235, 50, 210];
pub const LIME: Rgb = [60, 210, 90];
pub const BOTH: Rgb = [215, 215, 215];
const INK: Rgb = [235, 235, 235];

/// Reference | model | overlay, side by side at a legible scale, with
/// coordinate labels in model units.
pub fn diff_image(c: &Compare, pic: &RefPic, palette: &[[u8; 4]]) -> Canvas {
    let (m, r) = (&c.model, &c.refc);
    let k = (440 / m.rows.max(1)).min(720 / m.cols.max(1)).clamp(2, 14) as i32;
    let (pw, ph) = (m.cols as i32 * k, m.rows as i32 * k);
    let sc = if ph >= 160 { 2 } else { 1 };
    let left = text_size("-000", sc).0 + 10;
    let (top, bottom, legend) = (10 + 7 * sc + 8, 7 * sc + 16, 7 * sc + 14);
    let stats = format!("IOU {:.2}  SCORE {:.0}", c.iou, c.score);
    let keys = [("WHITE BOTH", BOTH), ("MAGENTA MODEL ONLY", MAGENTA), ("GREEN REFERENCE ONLY", LIME)];
    let legend_w = keys.iter().map(|k| text_size(k.0, sc).0 + 18 * sc).sum::<i32>() + text_size(&stats, sc).0;
    let w = (3 * (left + pw) + 12).max(left + legend_w + 12);
    let h = top + ph + bottom + legend;
    let mut cv = Canvas::new(w as u32, h as u32, [18, 18, 20, 255].repeat((w * h) as usize));
    let view = c.view;
    let titles = [String::from("REFERENCE"), format!("MODEL {}", view.title()), String::from("OVERLAY")];
    for (p, title) in titles.iter().enumerate() {
        let x0 = left + p as i32 * (left + pw);
        cv.rect(x0, top, pw, ph, PANEL_BG, 1.0);
        cv.text(x0, 8, title, sc, INK);
        for py in 0..ph {
            for px in 0..pw {
                let (col, row) = ((px / k) as usize, (m.rows as i32 - 1 - py / k) as usize);
                let i = row * m.cols + col;
                let color = match p {
                    0 => {
                        // The photo itself, resampled to the panel.
                        let su = m.su0 as f32 + (px as f32 + 0.5) / k as f32;
                        let v = m.v0 as f32 + m.rows as f32 - (py as f32 + 0.5) / k as f32;
                        let x = ((su - c.place.left) / c.place.scale).floor();
                        let y = (pic.h as f32 - (v - c.place.bottom) / c.place.scale).floor();
                        if x >= 0.0 && y >= 0.0 && (x as usize) < pic.w && (y as usize) < pic.h && pic.on(y as usize * pic.w + x as usize) {
                            let q = pic.px[y as usize * pic.w + x as usize];
                            Some([q[0], q[1], q[2]])
                        } else {
                            None
                        }
                    }
                    1 => (m.cells[i] != 0).then(|| {
                        let q = palette[m.cells[i] as usize];
                        // Darken cell edges a little so voxels read.
                        let edge = k >= 6 && (px % k == 0 || py % k == 0);
                        let f = if edge { 0.8 } else { 1.0 };
                        [(q[0] as f32 * f) as u8, (q[1] as f32 * f) as u8, (q[2] as f32 * f) as u8]
                    }),
                    _ => match (m.cells[i] != 0, r.cells[i].is_some()) {
                        (true, true) => Some(BOTH),
                        (true, false) => Some(MAGENTA),
                        (false, true) => Some(LIME),
                        _ => None,
                    },
                };
                if let Some(color) = color {
                    cv.rect(x0 + px, top + py, 1, 1, color, 1.0);
                }
            }
        }
        rulers(&mut cv, c, x0, top, k, sc);
    }
    // Legend.
    let y = top + ph + bottom;
    let mut x = left;
    for (label, color) in keys {
        cv.text(x, y, label, sc, color);
        x += text_size(label, sc).0 + 18 * sc;
    }
    cv.text(x, y, &stats, sc, INK);
    cv
}

/// Row labels left of a panel and column labels under it, every 5 or 10
/// model units, at cell centers.
fn rulers(cv: &mut Canvas, c: &Compare, x0: i32, y0: i32, k: i32, sc: i32) {
    let m = &c.model;
    let ph = m.rows as i32 * k;
    let step = if k * 5 >= 7 * sc + 6 && k * 5 >= text_size("-00", sc).0 + 6 { 5 } else if k * 10 >= text_size("-000", sc).0 + 4 { 10 } else { 20 };
    let grid: Rgb = [70, 70, 78];
    for row in 0..m.rows as i32 {
        let v = m.v0 + row;
        if v.rem_euclid(step) != 0 {
            continue;
        }
        let y = y0 + ph - row * k - k / 2;
        cv.line(((x0 - 5) as f32, y as f32), ((x0 - 1) as f32, y as f32), 1.0, grid, 1.0);
        let s = v.to_string();
        let (lw, lh) = text_size(&s, sc);
        cv.text(x0 - 7 - lw, y - lh / 2, &s, sc, INK);
    }
    for col in 0..m.cols as i32 {
        let h = c.view.coord(m.su0 + col);
        if h.rem_euclid(step) != 0 {
            continue;
        }
        let x = x0 + col * k + k / 2;
        cv.line((x as f32, (y0 + ph + 1) as f32), (x as f32, (y0 + ph + 5) as f32), 1.0, grid, 1.0);
        let s = h.to_string();
        let (lw, _) = text_size(&s, sc);
        cv.text(x - lw / 2, y0 + ph + 7, &s, sc, INK);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An L-shaped model: a tall post at low x/y and a foot toward +x, plus
    /// a nub toward +y at the top, so every view is asymmetric.
    fn ell() -> Volume {
        let mut v = Volume::new(IVec3::new(10, 6, 12));
        for z in 0..12 {
            v.set(IVec3::new(1, 1, z), 1);
        }
        for x in 1..9 {
            v.set(IVec3::new(x, 1, 0), 2);
        }
        for y in 2..5 {
            v.set(IVec3::new(1, y, 11), 3);
        }
        v
    }

    #[test]
    fn projections_face_the_right_way() {
        let v = ell();
        // Front (from -y): screen right = +x, so the foot runs right.
        let g = project(&v, IVec3::ZERO, View::Front);
        let (s0, s1, _, _) = bounds(&g, |&c| c != 0).unwrap();
        assert_eq!((View::Front.coord(s0), View::Front.coord(s1)), (1, 8));
        assert_eq!(*g.get(View::Front.coord(8), 0).unwrap(), 2);
        assert_eq!(*g.get(1, 11).unwrap(), 1, "front sees the post (y=1) before the nub");
        // Back (from +y): screen right = -x, the foot runs left of the post.
        let g = project(&v, IVec3::ZERO, View::Back);
        let post = g.get(View::Back.coord(1), 5).copied();
        let foot = g.get(View::Back.coord(8), 0).copied();
        assert_eq!((post, foot), (Some(1), Some(2)));
        assert!(View::Back.coord(8) < View::Back.coord(1), "foot is screen-left of the post");
        // Seen from the back, the nub (y=4) is nearest at the top.
        assert_eq!(g.get(View::Back.coord(1), 11).copied(), Some(3));
        // Right (from +x): screen right = +y, nub toward the right at the top.
        let g = project(&v, IVec3::ZERO, View::Right);
        assert_eq!(g.get(4, 11).copied(), Some(3));
        assert_eq!(g.get(1, 0).copied(), Some(2), "right sees the foot's end first");
        // Left (from -x): screen right = -y, the nub points left.
        let g = project(&v, IVec3::ZERO, View::Left);
        assert!(View::Left.coord(4) < View::Left.coord(1));
        assert_eq!(g.get(View::Left.coord(4), 11).copied(), Some(3));
        assert_eq!(g.get(View::Left.coord(1), 5).copied(), Some(1));
        // Top (from +z): right = +x, up = +y; the nub is the top layer.
        let g = project(&v, IVec3::ZERO, View::Top);
        assert_eq!(g.get(1, 4).copied(), Some(3));
        assert_eq!(g.get(8, 1).copied(), Some(2));
        // World origin offsets carry through.
        let g = project(&v, IVec3::new(5, 0, 2), View::Back);
        assert_eq!(g.get(View::Back.coord(13), 2).copied(), Some(2));
    }

    fn pic_from(mask: &[&str]) -> RefPic {
        let h = mask.len();
        let w = mask[0].len();
        let px = mask.iter().flat_map(|row| row.bytes().map(|b| if b == b'#' { [200, 40, 40, 255] } else { [0, 0, 0, 0] })).collect();
        RefPic::new(w, h, px)
    }

    #[test]
    fn iou_of_known_shapes() {
        // A 4x4 reference square vs a 4-wide, 4-tall model box shifted by 2.
        let pic = pic_from(&["####", "####", "####", "####"]);
        let mut v = Volume::new(IVec3::new(12, 1, 4));
        for z in 0..4 {
            for x in 2..6 {
                v.set(IVec3::new(x, 0, z), 1);
            }
        }
        let g = project(&v, IVec3::ZERO, View::Front);
        let mut pal = [[0u8; 4]; 256];
        pal[1] = [200, 40, 40, 255];
        // Pinned at the same place: perfect.
        let c = compare(View::Front, &g, &pal, &pic, Placement { left: 2.0, bottom: 0.0, scale: 1.0 });
        assert_eq!((c.both, c.model_only, c.ref_only), (16, 0, 0));
        assert!((c.iou - 1.0).abs() < 1e-6 && c.mean_de < 1.0 && c.score > 99.0);
        // Two cells right: overlap 2 columns of 4.
        let c = compare(View::Front, &g, &pal, &pic, Placement { left: 4.0, bottom: 0.0, scale: 1.0 });
        assert_eq!((c.both, c.model_only, c.ref_only), (8, 8, 8));
        assert!((c.iou - 8.0 / 24.0).abs() < 1e-6);
        let f = width_findings(&c, 8);
        assert!(f[0].contains("shifted 2 toward the left (-x)"), "{f:?}");
        // Fit finds the scale and position.
        let (p, _) = fit(View::Front, &g, &pic).unwrap();
        let c = compare(View::Front, &g, &pal, &pic, p);
        assert!((c.iou - 1.0).abs() < 1e-6, "{p:?}");
    }

    #[test]
    fn car_and_house_findings() {
        // Reference: a car (wide body, cabin, gap between wheels), 2 px per cell.
        let car = [
            "......########......",
            "......########......",
            "....############....",
            "....############....",
            "####################",
            "####################",
            "####################",
            "####################",
            "..####........####..",
            "..####........####..",
        ];
        let pic = pic_from(&car);
        // Model: same body but the cabin is too wide and there's no gap.
        let mut v = Volume::new(IVec3::new(12, 3, 5));
        let rows = [".########.", "..######..", "##########", "##########", "##########"];
        for (k, row) in rows.iter().enumerate() {
            for (x, ch) in row.bytes().enumerate() {
                if ch == b'#' {
                    v.set(IVec3::new(x as i32 + 1, 1, 4 - k as i32), 1);
                }
            }
        }
        let g = project(&v, IVec3::ZERO, View::Front);
        let pal = [[200u8, 40, 40, 255]; 256];
        let (p, _) = fit(View::Front, &g, &pic).unwrap();
        assert!((p.scale - 0.5).abs() < 1e-6);
        let c = compare(View::Front, &g, &pal, &pic, p);
        assert!(c.iou > 0.7 && c.iou < 1.0, "{}", c.iou);
        let f = width_findings(&c, 8);
        let all = f.join("\n");
        assert!(all.contains("z=4: model wider on both sides"), "{all}");
        assert!(all.contains("z=0:") && all.contains("filled gaps"), "{all}");
        // The diff image has the three panels' colors.
        let img = diff_image(&c, &pic, &pal);
        assert!(img.px.chunks(4).any(|p| p[..3] == MAGENTA));
        // A house reference against a box model: the roof shows up as
        // model-only cells above the eaves.
        let house = ["...##...", "..####..", ".######.", "########", "########", "########", "########", "########"];
        let pic = pic_from(&house);
        let mut v = Volume::new(IVec3::new(8, 1, 8));
        for z in 0..8 {
            for x in 0..8 {
                v.set(IVec3::new(x, 0, z), 1);
            }
        }
        let g = project(&v, IVec3::ZERO, View::Front);
        let (p, shift) = fit(View::Front, &g, &pic).unwrap();
        assert_eq!(shift, 0);
        let c = compare(View::Front, &g, &pal, &pic, p);
        assert_eq!(c.model_only, 12);
        let f = width_findings(&c, 8).join("\n");
        assert!(f.contains("z=6-7: model wider on both sides"), "{f}");
    }

    #[test]
    fn colors_compare_by_region() {
        let pic = pic_from(&["####", "####"]);
        let mut v = Volume::new(IVec3::new(4, 1, 2));
        for x in 0..4 {
            v.set(IVec3::new(x, 0, 0), 1);
            v.set(IVec3::new(x, 0, 1), 2);
        }
        let mut pal = [[0u8; 4]; 256];
        pal[1] = [200, 40, 40, 255];
        pal[2] = [40, 80, 200, 255];
        let g = project(&v, IVec3::ZERO, View::Front);
        let c = compare(View::Front, &g, &pal, &pic, Placement { left: 0.0, bottom: 0.0, scale: 1.0 });
        assert!((c.iou - 1.0).abs() < 1e-6);
        assert!(c.agreement > 0.4 && c.agreement < 0.6, "{}", c.agreement);
        let (_, worst) = color_findings(&c, &pal, 4);
        assert!(worst[0].starts_with("z=1") && worst[0].contains("#2850c8"), "{worst:?}");
    }
}
