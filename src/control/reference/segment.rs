//! Finding the subjects in a reference image: a smooth background model
//! fitted to the border, a foreground mask (color difference or texture),
//! blobs, text labels, width profiles and dominant colors.

use glam::Vec3;

use super::super::surface::{Image, oklab};

/// A smooth background: a quadratic surface per channel fitted to the
/// border pixels that agree with each other.
pub struct Background {
    coef: [[f32; 6]; 3],
    w: usize,
    h: usize,
    /// Median border color.
    pub color: [u8; 3],
    /// 95th percentile of the fitted border's residual and pixel-to-pixel
    /// gradient: how noisy the background is.
    pub noise: f32,
    pub grad_noise: f32,
    /// Share of border pixels that fit the model (low: busy background).
    pub inliers: f32,
    /// The image has a transparent background; alpha is the mask.
    pub alpha: bool,
}

fn terms(x: usize, y: usize, w: usize, h: usize) -> [f32; 6] {
    let u = x as f32 / (w.max(2) - 1) as f32 * 2.0 - 1.0;
    let v = y as f32 / (h.max(2) - 1) as f32 * 2.0 - 1.0;
    [1.0, u, v, u * u, u * v, v * v]
}

/// Solve the 6x6 normal equations; None if singular.
fn solve(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> Option<[f64; 6]> {
    for c in 0..6 {
        let p = (c..6).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-9 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in 0..6 {
            if r != c {
                let f = a[r][c] / a[c][c];
                for k in c..6 {
                    a[r][k] -= f * a[c][k];
                }
                b[r] -= f * b[c];
            }
        }
    }
    Some(std::array::from_fn(|i| b[i] / a[i][i]))
}

fn percentile(v: &mut [f32], p: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let k = ((v.len() - 1) as f32 * p).round() as usize;
    *v.select_nth_unstable_by(k, f32::total_cmp).1
}

impl Background {
    pub fn fit(img: &Image) -> Background {
        let (w, h) = (img.w, img.h);
        let band = (w.min(h) / 100).max(2);
        let mut samples: Vec<(usize, usize)> = Vec::new();
        for y in 0..h {
            for x in 0..w {
                if x < band || y < band || x >= w.saturating_sub(band) || y >= h.saturating_sub(band) {
                    samples.push((x, y));
                }
            }
        }
        let step = (samples.len() / 20000).max(1);
        let samples: Vec<(usize, usize)> = samples.into_iter().step_by(step).collect();
        let px = |(x, y): (usize, usize)| img.px[y * w + x];
        let transparent = samples.iter().filter(|&&s| px(s)[3] < 128).count();
        let median = |pts: &[(usize, usize)], k: usize| {
            let mut v: Vec<f32> = pts.iter().map(|&s| px(s)[k] as f32).collect();
            percentile(&mut v, 0.5)
        };
        let color = std::array::from_fn(|k| median(&samples, k) as u8);
        let mut bg = Background { coef: [[0.0; 6]; 3], w, h, color, noise: 0.0, grad_noise: 0.0, inliers: 1.0, alpha: false };
        for k in 0..3 {
            bg.coef[k][0] = color[k] as f32;
        }
        if transparent * 2 > samples.len() {
            bg.alpha = true;
            return bg;
        }
        // Fit to pixels near the median, then refit to those near the fit.
        let mut tol = 24.0;
        let mut inl: Vec<(usize, usize)> = Vec::new();
        for _ in 0..2 {
            inl = samples.iter().copied().filter(|&s| bg.dist(s.0, s.1, px(s)) <= tol).collect();
            if inl.len() < 12 {
                break;
            }
            let mut fitted = bg.coef;
            for (k, row) in fitted.iter_mut().enumerate() {
                let (mut a, mut b) = ([[0f64; 6]; 6], [0f64; 6]);
                for &(x, y) in &inl {
                    let t = terms(x, y, w, h);
                    for i in 0..6 {
                        for j in 0..6 {
                            a[i][j] += (t[i] * t[j]) as f64;
                        }
                        b[i] += (t[i] * px((x, y))[k] as f32) as f64;
                    }
                }
                for (i, r) in a.iter_mut().enumerate() {
                    r[i] += 1e-3;
                }
                if let Some(c) = solve(a, b) {
                    *row = c.map(|v| v as f32);
                }
            }
            bg.coef = fitted;
            let mut res: Vec<f32> = inl.iter().map(|&s| bg.dist(s.0, s.1, px(s))).collect();
            tol = (percentile(&mut res, 0.95) * 3.0).max(10.0);
        }
        let mut res: Vec<f32> = inl.iter().map(|&s| bg.dist(s.0, s.1, px(s))).collect();
        bg.noise = percentile(&mut res, 0.95);
        let mut grads: Vec<f32> = inl.iter().map(|&(x, y)| gradient(img, x, y)).collect();
        bg.grad_noise = percentile(&mut grads, 0.95);
        bg.inliers = inl.len() as f32 / samples.len().max(1) as f32;
        bg
    }

    pub fn at(&self, x: usize, y: usize) -> [f32; 3] {
        let t = terms(x, y, self.w, self.h);
        std::array::from_fn(|k| (0..6).map(|i| self.coef[k][i] * t[i]).sum())
    }

    /// Largest channel difference from the background at (x, y).
    pub fn dist(&self, x: usize, y: usize, p: [u8; 4]) -> f32 {
        let b = self.at(x, y);
        (0..3).map(|k| (p[k] as f32 - b[k]).abs()).fold(0.0, f32::max)
    }
}

/// Darker than the background with the same hue: a shadow on it.
fn is_shade(p: [u8; 4], bg: [f32; 3]) -> bool {
    let (pm, bm) = ((p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0, (bg[0] + bg[1] + bg[2]) / 3.0);
    let hue = (0..3).map(|k| ((p[k] as f32 - pm) - (bg[k] - bm)).abs()).fold(0.0, f32::max);
    pm < bm && hue <= 6.0 + 0.04 * (bm - pm)
}

/// Largest channel step to the right or downward neighbor.
fn gradient(img: &Image, x: usize, y: usize) -> f32 {
    let p = img.px[y * img.w + x];
    let r = img.px[y * img.w + (x + 1).min(img.w - 1)];
    let d = img.px[(y + 1).min(img.h - 1) * img.w + x];
    (0..3).map(|k| (p[k] as i32 - r[k] as i32).abs().max((p[k] as i32 - d[k] as i32).abs())).max().unwrap_or(0) as f32
}

// ---------------------------------------------------------------- masks

/// Grow a mask by a (2r+1)-square; pixels outside count as off.
pub fn dilate(m: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    if r == 0 {
        return m.to_vec();
    }
    let pass = |src: &[bool], len: usize, lines: usize, at: &dyn Fn(usize, usize) -> usize| {
        let mut out = vec![false; src.len()];
        for l in 0..lines {
            let mut count = 0usize;
            // Window [i - r, i + r].
            for i in 0..len.min(r) {
                count += src[at(l, i)] as usize;
            }
            for i in 0..len {
                if i + r < len {
                    count += src[at(l, i + r)] as usize;
                }
                if i > r {
                    count -= src[at(l, i - r - 1)] as usize;
                }
                out[at(l, i)] = count > 0;
            }
        }
        out
    };
    let rows = pass(m, w, h, &|l, i| l * w + i);
    pass(&rows, h, w, &|l, i| i * w + l)
}

/// Shrink a mask by a (2r+1)-square; pixels outside count as on.
pub fn erode(m: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    let inv: Vec<bool> = m.iter().map(|&b| !b).collect();
    dilate(&inv, w, h, r).into_iter().map(|b| !b).collect()
}

/// Connected components of the on pixels (8-connected, or 4); labels from
/// 1, 0 = off. Returns the labels and the count.
pub fn components(m: &[bool], w: usize, h: usize, eight: bool) -> (Vec<u32>, usize) {
    let mut labels = vec![0u32; m.len()];
    let mut n = 0u32;
    let mut stack = Vec::new();
    for start in 0..m.len() {
        if !m[start] || labels[start] != 0 {
            continue;
        }
        n += 1;
        labels[start] = n;
        stack.push(start);
        while let Some(i) = stack.pop() {
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    if (dx == 0 && dy == 0) || (!eight && dx != 0 && dy != 0) {
                        continue;
                    }
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if m[j] && labels[j] == 0 {
                        labels[j] = n;
                        stack.push(j);
                    }
                }
            }
        }
    }
    (labels, n as usize)
}

/// Turn on enclosed off regions smaller than `max_area`, or mostly
/// `strong`.
fn fill_holes(m: &mut [bool], w: usize, h: usize, max_area: usize, strong: &[bool]) {
    let inv: Vec<bool> = m.iter().map(|&b| !b).collect();
    let (labels, n) = components(&inv, w, h, false);
    let mut area = vec![0usize; n + 1];
    let mut solid = vec![0usize; n + 1];
    let mut border = vec![false; n + 1];
    for (i, &l) in labels.iter().enumerate() {
        area[l as usize] += 1;
        solid[l as usize] += strong[i] as usize;
        let (x, y) = (i % w, i / w);
        if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
            border[l as usize] = true;
        }
    }
    for (i, &l) in labels.iter().enumerate() {
        let l = l as usize;
        if l != 0 && !border[l] && (area[l] <= max_area || solid[l] * 2 > area[l]) {
            m[i] = true;
        }
    }
}

/// Split subjects joined by thin bridges (a shared shadow, a sword tip
/// touching a cape): erode by `r` to find cores, grow them back over the
/// mask, and keep apart cores that only touch along a short boundary.
/// Pixels no core reaches keep their `joined` label. Returns labels (0 off)
/// and the largest label.
fn split_necks(fg: &[bool], joined: &[u32], nj: usize, w: usize, h: usize, r: usize, min_core: usize) -> (Vec<u32>, usize) {
    let (cores, nc) = components(&erode(fg, w, h, r), w, h, true);
    let mut core_area = vec![0usize; nc + 1];
    for &c in &cores {
        core_area[c as usize] += 1;
    }
    // Grow the cores over the mask, breadth first.
    let mut region = vec![0u32; fg.len()];
    let mut queue = std::collections::VecDeque::new();
    for (i, &c) in cores.iter().enumerate() {
        if c != 0 && core_area[c as usize] >= min_core {
            region[i] = c;
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        let (x, y) = ((i % w) as i64, (i / w) as i64);
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)] {
            let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64 {
                let j = ny as usize * w + nx as usize;
                if fg[j] && region[j] == 0 {
                    region[j] = region[i];
                    queue.push_back(j);
                }
            }
        }
    }
    // Rejoin regions that share a long boundary (two legs, a head on a neck).
    let mut area = vec![0usize; nc + 1];
    for &g in &region {
        area[g as usize] += 1;
    }
    let mut contact: std::collections::HashMap<(u32, u32), usize> = std::collections::HashMap::new();
    for i in 0..fg.len() {
        let (x, y) = (i % w, i / w);
        for j in [(x + 1 < w).then(|| i + 1), (y + 1 < h).then(|| i + w)].into_iter().flatten() {
            let (a, b) = (region[i], region[j]);
            if a != 0 && b != 0 && a != b {
                *contact.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
    }
    let mut parent: Vec<u32> = (0..=nc as u32).collect();
    fn root(p: &mut [u32], mut a: u32) -> u32 {
        while p[a as usize] != a {
            p[a as usize] = p[p[a as usize] as usize];
            a = p[a as usize];
        }
        a
    }
    let mut pairs: Vec<((u32, u32), usize)> = contact.into_iter().collect();
    pairs.sort();
    for ((a, b), len) in pairs {
        // Boundary pixels are counted along both axes; halve.
        if len as f32 * 0.5 >= 0.15 * (area[a as usize].min(area[b as usize]) as f32).sqrt() {
            let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
            parent[ra.max(rb) as usize] = ra.min(rb);
        }
    }
    let labels = (0..fg.len())
        .map(|i| match (fg[i], region[i]) {
            (false, _) => 0,
            (true, 0) => joined[i],
            (true, g) => nj as u32 + root(&mut parent, g),
        })
        .collect();
    (labels, nj + nc)
}

// ---------------------------------------------------------------- blobs

#[derive(Clone, Debug)]
pub struct Blob {
    /// 1-based, as shown on the annotated image.
    pub id: usize,
    /// Bounding box in pixels, inclusive.
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
    /// Foreground pixels.
    pub area: usize,
    /// Looks like a caption: a short row of glyphs.
    pub text: bool,
}

impl Blob {
    pub fn w(&self) -> usize {
        self.x1 - self.x0 + 1
    }
    pub fn h(&self) -> usize {
        self.y1 - self.y0 + 1
    }
}

pub struct Segmentation {
    pub w: usize,
    pub bg: Background,
    /// Foreground color-difference threshold used.
    pub threshold: f32,
    /// Blob id per pixel (0 = background or a dropped speck).
    pub labels: Vec<u32>,
    pub blobs: Vec<Blob>,
    /// Tiny components ignored.
    pub specks: usize,
}

impl Segmentation {
    pub fn blob(&self, id: usize) -> Option<&Blob> {
        self.blobs.iter().find(|b| b.id == id)
    }
}

/// Runs of clearly-foreground pixels labeled `l` crossing row `y`
/// between x0..=x1.
fn runs(labels: &[u32], strong: &[bool], w: usize, y: usize, x0: usize, x1: usize, l: u32) -> usize {
    let mut n = 0;
    let mut inside = false;
    for x in x0..=x1 {
        let on = labels[y * w + x] == l && strong[y * w + x];
        if on && !inside {
            n += 1;
        }
        inside = on;
    }
    n
}

/// Separate the subjects from the background. `threshold` overrides the
/// color difference (0-255, largest channel) that counts as foreground.
pub fn segment(img: &Image, threshold: Option<f32>) -> Segmentation {
    let (w, h) = (img.w, img.h);
    let bg = Background::fit(img);
    let big = w.max(h);
    let t = threshold.unwrap_or((bg.noise * 3.0 + 6.0).max(14.0));
    // Clearly different from the background (or opaque, for transparent
    // backgrounds) ...
    let strong: Vec<bool> = if bg.alpha {
        img.px.iter().map(|p| p[3] >= 128).collect()
    } else {
        (0..w * h).map(|i| img.px[i][3] >= 128 && bg.dist(i % w, i / w, img.px[i]) > t).collect()
    };
    // ... or textured where the background is smooth (catches subjects
    // close to its color). Smooth, darker pixels of the background's own
    // hue are cast shadows on the backdrop, not subject.
    let mut fg = strong.clone();
    if !bg.alpha {
        let tg = (bg.grad_noise * 2.5 + 6.0).max(10.0);
        let tex: Vec<bool> = (0..w * h).map(|i| img.px[i][3] >= 128 && gradient(img, i % w, i / w) > tg).collect();
        let near_tex = dilate(&tex, w, h, 2);
        for (i, f) in fg.iter_mut().enumerate() {
            if *f && !near_tex[i] && is_shade(img.px[i], bg.at(i % w, i / w)) {
                *f = false;
            }
        }
        let rt = ((big as f32 * 0.003).round() as usize).max(2);
        let tex = erode(&dilate(&tex, w, h, rt), w, h, rt);
        fg.iter_mut().zip(&tex).for_each(|(a, &b)| *a |= b);
    }
    // Enclosed gaps: small ones, or any that aren't background-colored
    // (e.g. a flat dark panel taken for shadow).
    fill_holes(&mut fg, w, h, (w * h / 500).max(64), &strong);
    let fg = dilate(&erode(&fg, w, h, 1), w, h, 1);

    // Components, joining pieces a few pixels apart (letters of a word).
    let rm = ((big as f32 * 0.004).round() as usize).max(2);
    let (joined, nj) = components(&dilate(&fg, w, h, rm), w, h, true);
    let (mut labels, n) = split_necks(&fg, &joined, nj, w, h, ((big as f32 * 0.008).round() as usize).max(2), (w * h / 10000).max(8));
    let mut blobs: Vec<Blob> = (0..=n).map(|id| Blob { id, x0: usize::MAX, y0: usize::MAX, x1: 0, y1: 0, area: 0, text: false }).collect();
    for (i, &l) in labels.iter().enumerate() {
        if l != 0 {
            let b = &mut blobs[l as usize];
            let (x, y) = (i % w, i / w);
            b.x0 = b.x0.min(x);
            b.y0 = b.y0.min(y);
            b.x1 = b.x1.max(x);
            b.y1 = b.y1.max(y);
            b.area += 1;
        }
    }
    let min_area = (w * h / 2500).max(16);
    let mut keep = vec![false; n + 1];
    let mut specks = 0;
    for b in blobs.iter_mut().skip(1) {
        if b.area == 0 {
            continue;
        }
        if b.area < min_area {
            specks += 1;
            continue;
        }
        keep[b.id] = true;
        // A caption: short, wide, crossed by many strokes.
        let aspect = b.w() as f32 / b.h() as f32;
        if b.h() as f32 <= h as f32 * 0.15 && aspect >= 1.5 {
            let rows = [0.2, 0.35, 0.5, 0.65, 0.8].map(|f| b.y0 + ((b.h() - 1) as f32 * f) as usize);
            let r = rows.iter().map(|&y| runs(&labels, &strong, w, y, b.x0, b.x1, b.id as u32)).sum::<usize>() as f32 / 5.0;
            b.text = r >= (aspect * 1.2).max(3.0);
        }
    }
    // Rows of separate glyphs: similar heights side by side.
    let small: Vec<usize> = (1..=n).filter(|&i| keep[i] && !blobs[i].text && blobs[i].h() as f32 <= h as f32 * 0.12 && blobs[i].area * 100 < w * h).collect();
    for &a in &small {
        let near = small
            .iter()
            .filter(|&&b| {
                let (p, q) = (&blobs[a], &blobs[b]);
                let overlap = p.y1.min(q.y1) as f32 - p.y0.max(q.y0) as f32;
                let hh = p.h().max(q.h()) as f32;
                let gap = (p.x0.max(q.x0) as f32 - p.x1.min(q.x1) as f32).max(0.0);
                b != a && overlap > 0.6 * p.h().min(q.h()) as f32 && p.h().min(q.h()) as f32 > 0.6 * hh && gap < 1.2 * hh
            })
            .count();
        if near >= 2 {
            blobs[a].text = true;
        }
    }
    // Fold small fragments into a nearby subject.
    let largest = (1..=n).filter(|&i| keep[i] && !blobs[i].text).map(|i| blobs[i].area).max().unwrap_or(0);
    let reach = big as f32 * 0.02;
    let mut parent: Vec<u32> = (0..=n as u32).collect();
    for i in 1..=n {
        if !keep[i] || blobs[i].text || blobs[i].area * 50 >= largest {
            continue;
        }
        let gap = |q: &Blob| {
            let p = &blobs[i];
            let dx = (p.x0.max(q.x0) as f32 - p.x1.min(q.x1) as f32).max(0.0);
            let dy = (p.y0.max(q.y0) as f32 - p.y1.min(q.y1) as f32).max(0.0);
            dx.hypot(dy)
        };
        let host = (1..=n).filter(|&j| j != i && keep[j] && !blobs[j].text && blobs[j].area > blobs[i].area * 4).min_by(|&a, &b| gap(&blobs[a]).total_cmp(&gap(&blobs[b])));
        if let Some(j) = host.filter(|&j| gap(&blobs[j]) <= reach) {
            parent[i] = j as u32;
            keep[i] = false;
            let (x0, y0, x1, y1, area) = (blobs[i].x0, blobs[i].y0, blobs[i].x1, blobs[i].y1, blobs[i].area);
            let b = &mut blobs[j];
            (b.x0, b.y0, b.x1, b.y1, b.area) = (b.x0.min(x0), b.y0.min(y0), b.x1.max(x1), b.y1.max(y1), b.area + area);
        }
    }
    // Subjects left to right, then captions; renumber from 1.
    let mut order: Vec<usize> = (1..=n).filter(|&i| keep[i]).collect();
    order.sort_by_key(|&i| (blobs[i].text, blobs[i].x0, blobs[i].y0));
    let mut remap = vec![0u32; n + 1];
    for (k, &i) in order.iter().enumerate() {
        remap[i] = k as u32 + 1;
    }
    for l in labels.iter_mut() {
        let mut r = *l;
        while r != 0 && parent[r as usize] != r {
            r = parent[r as usize];
        }
        *l = remap[r as usize];
    }
    let blobs = order.iter().enumerate().map(|(k, &i)| Blob { id: k + 1, ..blobs[i].clone() }).collect();
    Segmentation { w, bg, threshold: t, labels, blobs, specks }
}

// ---------------------------------------------------------------- profiles

/// One height through a silhouette.
pub struct ProfileRow {
    /// 0 = top of the blob, 1 = bottom (the row's center).
    pub at: f32,
    pub y: usize,
    /// Leftmost/rightmost filled pixel, if any.
    pub extent: Option<(usize, usize)>,
    pub filled: usize,
    pub spans: usize,
}

/// `n` evenly spaced rows through the pixels where `on(x, y)`, inside a
/// box (inclusive).
pub fn width_profile(on: &dyn Fn(usize, usize) -> bool, (x0, y0, x1, y1): (usize, usize, usize, usize), n: usize) -> Vec<ProfileRow> {
    let hgt = y1 - y0 + 1;
    (0..n)
        .map(|k| {
            let at = (k as f32 + 0.5) / n as f32;
            let y = y0 + ((hgt as f32 * at) as usize).min(hgt - 1);
            let (mut extent, mut filled, mut spans, mut inside) = (None::<(usize, usize)>, 0, 0, false);
            for x in x0..=x1 {
                let o = on(x, y);
                if o {
                    filled += 1;
                    extent = Some(extent.map_or((x, x), |(a, _)| (a, x)));
                    if !inside {
                        spans += 1;
                    }
                }
                inside = o;
            }
            ProfileRow { at, y, extent, filled, spans }
        })
        .collect()
}

// ---------------------------------------------------------------- colors

/// k-means in OKLab with a seeded k-means++ start. Returns centers and
/// how many samples each took, largest first; near-duplicates merged.
pub fn kmeans(samples: &[Vec3], k: usize, iters: usize) -> Vec<(Vec3, usize)> {
    if samples.is_empty() || k == 0 {
        return Vec::new();
    }
    let mut seed = 0x9e37_79b9u32;
    let mut rand = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed as f32 / u32::MAX as f32
    };
    let mut centers = vec![samples[samples.len() / 2]];
    let mut d2: Vec<f32> = samples.iter().map(|s| s.distance_squared(centers[0])).collect();
    while centers.len() < k {
        let total: f32 = d2.iter().sum();
        if total <= 1e-9 {
            break;
        }
        let mut pick = rand() * total;
        let mut i = 0;
        while i + 1 < samples.len() && pick > d2[i] {
            pick -= d2[i];
            i += 1;
        }
        centers.push(samples[i]);
        for (d, s) in d2.iter_mut().zip(samples) {
            *d = d.min(s.distance_squared(samples[i]));
        }
    }
    let mut counts = vec![0usize; centers.len()];
    for _ in 0..iters {
        let mut sum = vec![Vec3::ZERO; centers.len()];
        counts.iter_mut().for_each(|c| *c = 0);
        for s in samples {
            let j = nearest(&centers, *s);
            sum[j] += *s;
            counts[j] += 1;
        }
        for j in 0..centers.len() {
            if counts[j] > 0 {
                centers[j] = sum[j] / counts[j] as f32;
            }
        }
    }
    let mut out: Vec<(Vec3, usize)> = Vec::new();
    let mut order: Vec<usize> = (0..centers.len()).filter(|&j| counts[j] > 0).collect();
    order.sort_by_key(|&j| std::cmp::Reverse(counts[j]));
    for j in order {
        match out.iter_mut().find(|(c, _)| c.distance(centers[j]) < 0.05) {
            Some(o) => o.1 += counts[j],
            None => out.push((centers[j], counts[j])),
        }
    }
    out
}

pub fn nearest(centers: &[Vec3], c: Vec3) -> usize {
    (0..centers.len()).min_by(|&a, &b| centers[a].distance_squared(c).total_cmp(&centers[b].distance_squared(c))).unwrap_or(0)
}

/// OKLab back to sRGB.
pub fn srgb(lab: Vec3) -> [u8; 3] {
    let l = lab.x + 0.396_337_78 * lab.y + 0.215_803_76 * lab.z;
    let m = lab.x - 0.105_561_346 * lab.y - 0.063_854_17 * lab.z;
    let s = lab.x - 0.089_484_18 * lab.y - 1.291_485_5 * lab.z;
    let (l, m, s) = (l * l * l, m * m * m, s * s * s);
    let lin = [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ];
    lin.map(|v| {
        let v = v.clamp(0.0, 1.0);
        let e = if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
        (e * 255.0).round() as u8
    })
}

pub fn lab(p: [u8; 4]) -> Vec3 {
    oklab(p)
}

/// A dominant color and where it sits in its box.
pub struct Swatch {
    pub rgb: [u8; 3],
    pub share: f32,
    /// 10th-90th percentile of its pixels' positions, 0..1 across the box
    /// (rows from the top).
    pub rows: (f32, f32),
    pub cols: (f32, f32),
}

/// Up to `k` dominant colors of the pixels where `on(x, y)` in a box.
pub fn dominant(img: &Image, on: &dyn Fn(usize, usize) -> bool, (x0, y0, x1, y1): (usize, usize, usize, usize), k: usize) -> Vec<Swatch> {
    let mut pts: Vec<(usize, usize)> = Vec::new();
    for y in y0..=y1 {
        for x in x0..=x1 {
            if on(x, y) {
                pts.push((x, y));
            }
        }
    }
    let step = (pts.len() / 40000).max(1);
    let pts: Vec<(usize, usize)> = pts.into_iter().step_by(step).collect();
    let labs: Vec<Vec3> = pts.iter().map(|&(x, y)| lab(img.px[y * img.w + x])).collect();
    let clusters = kmeans(&labs, k, 12);
    let centers: Vec<Vec3> = clusters.iter().map(|c| c.0).collect();
    let mut members: Vec<Vec<(usize, usize)>> = vec![Vec::new(); centers.len()];
    for (p, l) in pts.iter().zip(&labs) {
        members[nearest(&centers, *l)].push(*p);
    }
    let (bw, bh) = ((x1 - x0).max(1) as f32, (y1 - y0).max(1) as f32);
    let mut out: Vec<Swatch> = centers
        .iter()
        .zip(&members)
        .filter(|(_, m)| !m.is_empty())
        .map(|(c, m)| {
            let mut ys: Vec<f32> = m.iter().map(|p| (p.1 - y0) as f32 / bh).collect();
            let mut xs: Vec<f32> = m.iter().map(|p| (p.0 - x0) as f32 / bw).collect();
            Swatch {
                rgb: srgb(*c),
                share: m.len() as f32 / pts.len() as f32,
                rows: (percentile(&mut ys, 0.1), percentile(&mut ys, 0.9)),
                cols: (percentile(&mut xs, 0.1), percentile(&mut xs, 0.9)),
            }
        })
        .collect();
    out.sort_by(|a, b| b.share.total_cmp(&a.share));
    out.retain(|s| s.share >= 0.01);
    out
}

/// "upper left", "bottom", "throughout" ... from a swatch's spread.
pub fn region_words(s: &Swatch) -> String {
    let (r0, r1) = s.rows;
    let (c0, c1) = s.cols;
    let my = (r0 + r1) * 0.5;
    let mx = (c0 + c1) * 0.5;
    let v = if r1 - r0 > 0.7 {
        "top to bottom"
    } else if my < 0.2 {
        "top"
    } else if my < 0.4 {
        "upper"
    } else if my < 0.6 {
        "middle"
    } else if my < 0.8 {
        "lower"
    } else {
        "bottom"
    };
    let h = if c1 - c0 > 0.7 {
        ""
    } else if mx < 0.35 {
        " left"
    } else if mx > 0.65 {
        " right"
    } else {
        " center"
    };
    format!("{v}{h}")
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn canvas(w: usize, h: usize, bg: [u8; 4]) -> Image {
        Image { w, h, px: vec![bg; w * h] }
    }

    pub fn rect(img: &mut Image, x0: usize, y0: usize, x1: usize, y1: usize, c: [u8; 4]) {
        for y in y0..=y1 {
            for x in x0..=x1 {
                img.px[y * img.w + x] = c;
            }
        }
    }

    /// Blocky capital letters, each 5x7 cells of `s` pixels.
    fn word(img: &mut Image, x: usize, y: usize, s: usize, glyphs: &[[u8; 7]]) {
        for (g, rows) in glyphs.iter().enumerate() {
            for (r, bits) in rows.iter().enumerate() {
                for c in 0..5 {
                    if bits >> (4 - c) & 1 == 1 {
                        let (px, py) = (x + (g * 6 + c) * s, y + r * s);
                        rect(img, px, py, px + s - 1, py + s - 1, [10, 10, 10, 255]);
                    }
                }
            }
        }
    }

    const C: [u8; 7] = [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110];
    const A: [u8; 7] = [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001];
    const R: [u8; 7] = [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001];

    #[test]
    fn finds_a_car_a_house_and_a_caption() {
        let gray = [205, 205, 205, 255];
        let mut img = canvas(400, 260, gray);
        // A red car: body, cabin, two dark wheels.
        rect(&mut img, 20, 120, 170, 160, [200, 30, 30, 255]);
        rect(&mut img, 60, 90, 130, 120, [200, 30, 30, 255]);
        rect(&mut img, 35, 160, 65, 185, [30, 30, 30, 255]);
        rect(&mut img, 125, 160, 155, 185, [30, 30, 30, 255]);
        // A house: wall and a stepped roof.
        rect(&mut img, 240, 110, 360, 185, [220, 200, 150, 255]);
        for k in 0..30 {
            rect(&mut img, 230 + k * 2, 109 - k, 370 - k * 2, 109 - k, [120, 50, 40, 255]);
        }
        // A caption below.
        word(&mut img, 150, 215, 4, &[C, A, R]);
        let seg = segment(&img, None);
        let subjects: Vec<&Blob> = seg.blobs.iter().filter(|b| !b.text).collect();
        let texts: Vec<&Blob> = seg.blobs.iter().filter(|b| b.text).collect();
        assert_eq!(subjects.len(), 2, "{:?}", seg.blobs);
        assert_eq!(texts.len(), 1, "{:?}", seg.blobs);
        // Car first (left to right), its box includes the wheels.
        let car = subjects[0];
        assert!(car.x0 <= 21 && car.x1 >= 169 && car.y0 <= 91 && car.y1 >= 184, "{car:?}");
        assert!(subjects[1].x0 >= 225);
        assert!(texts[0].y0 >= 210);
        // Width profile of the car: narrow cabin on top, two wheel spans at the bottom.
        let id = car.id as u32;
        let on = |x: usize, y: usize| seg.labels[y * seg.w + x] == id;
        let prof = width_profile(&on, (car.x0, car.y0, car.x1, car.y1), 20);
        assert!(prof[0].filled < prof[10].filled / 2 + 5);
        assert_eq!(prof[19].spans, 2);
    }

    #[test]
    fn pale_textured_subject_on_a_pale_background() {
        // A subject barely brighter than the background but with voxel-like
        // shading steps is still found by its texture.
        let mut img = canvas(200, 200, [208, 208, 208, 255]);
        for y in 50..150 {
            for x in 60..140 {
                let v = if ((x / 6) + (y / 6)) % 2 == 0 { 214 } else { 198 };
                img.px[y * 200 + x] = [v, v, v, 255];
            }
        }
        let seg = segment(&img, None);
        let b = seg.blobs.iter().find(|b| !b.text).expect("a blob");
        // Found to within one shading step of its edge.
        assert!(b.x0 <= 66 && b.x1 >= 133 && b.y0 <= 54 && b.y1 >= 142, "{b:?}");
        assert!(b.area > 65 * 85);
    }

    #[test]
    fn background_gradient_is_not_foreground() {
        let mut img = canvas(300, 200, [0, 0, 0, 255]);
        for y in 0..200 {
            for x in 0..300 {
                let v = (215.0 - y as f32 * 0.12 - x as f32 * 0.03) as u8;
                img.px[y * 300 + x] = [v, v, v, 255];
            }
        }
        rect(&mut img, 100, 60, 180, 150, [40, 90, 200, 255]);
        let seg = segment(&img, None);
        assert_eq!(seg.blobs.len(), 1, "{:?}", seg.blobs);
        assert!(seg.bg.inliers > 0.9);
    }

    #[test]
    fn colors_round_trip_and_cluster() {
        for c in [[200u8, 30, 30], [20, 200, 90], [240, 240, 240], [5, 5, 5]] {
            let back = srgb(lab([c[0], c[1], c[2], 255]));
            assert!((0..3).all(|k| (back[k] as i32 - c[k] as i32).abs() <= 1), "{c:?} {back:?}");
        }
        let mut img = canvas(100, 100, [255, 255, 255, 255]);
        rect(&mut img, 0, 0, 99, 24, [200, 30, 30, 255]);
        rect(&mut img, 0, 25, 99, 99, [30, 60, 200, 255]);
        let sw = dominant(&img, &|_, _| true, (0, 0, 99, 99), 6);
        assert_eq!(sw.len(), 2);
        assert!((sw[0].share - 0.75).abs() < 0.02 && sw[0].rgb[2] > 150);
        assert_eq!(region_words(&sw[1]), "top");
    }
}
