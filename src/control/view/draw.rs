//! A tiny CPU canvas for screenshot overlays: lines, boxes and 5x7 bitmap
//! text, drawn straight onto RGBA8 pixels.

pub type Rgb = [u8; 3];

pub const WHITE: Rgb = [235, 235, 235];
pub const RED: Rgb = [255, 90, 80];
pub const GREEN: Rgb = [90, 220, 90];
pub const BLUE: Rgb = [100, 150, 255];
const SHADOW: Rgb = [10, 10, 10];

pub struct Canvas {
    pub w: u32,
    pub h: u32,
    pub px: Vec<u8>,
}

impl Canvas {
    pub fn new(w: u32, h: u32, px: Vec<u8>) -> Self {
        debug_assert_eq!(px.len(), (w * h * 4) as usize);
        Canvas { w, h, px }
    }

    fn at(&self, x: i32, y: i32) -> Option<usize> {
        (x >= 0 && y >= 0 && (x as u32) < self.w && (y as u32) < self.h).then(|| (y as usize * self.w as usize + x as usize) * 4)
    }

    pub fn get(&self, x: i32, y: i32) -> Option<Rgb> {
        self.at(x, y).map(|i| [self.px[i], self.px[i + 1], self.px[i + 2]])
    }

    /// Blend `c` over pixel (x, y) with opacity `a` (0..1).
    pub fn blend(&mut self, x: i32, y: i32, c: Rgb, a: f32) {
        if let Some(i) = self.at(x, y) {
            for k in 0..3 {
                let d = self.px[i + k] as f32;
                self.px[i + k] = (d + (c[k] as f32 - d) * a).round() as u8;
            }
        }
    }

    pub fn rect(&mut self, x: i32, y: i32, w: i32, h: i32, c: Rgb, a: f32) {
        for yy in y..y + h {
            for xx in x..x + w {
                self.blend(xx, yy, c, a);
            }
        }
    }

    /// A line `thick` pixels wide.
    pub fn line(&mut self, (x0, y0): (f32, f32), (x1, y1): (f32, f32), thick: f32, c: Rgb, a: f32) {
        let steps = (x1 - x0).abs().max((y1 - y0).abs()).ceil().clamp(1.0, 20000.0) as i32;
        let r = (thick - 1.0) * 0.5;
        let mut last = (i32::MIN, i32::MIN);
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            let (x, y) = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
            let (cx, cy) = ((x - r).round() as i32, (y - r).round() as i32);
            if (cx, cy) == last {
                continue;
            }
            last = (cx, cy);
            let n = thick.round().max(1.0) as i32;
            // Overlapping squares would blend twice; draw opaque or accept it.
            self.rect(cx, cy, n, n, c, a);
        }
    }

    /// Text with its top-left at (x, y), `scale` pixels per font pixel,
    /// with a dark outline so it reads on any background.
    pub fn text(&mut self, x: i32, y: i32, s: &str, scale: i32, c: Rgb) {
        for pass in 0..2 {
            let mut cx = x;
            for ch in s.chars() {
                let g = glyph(ch);
                for (row, bits) in g.iter().enumerate() {
                    for col in 0..5 {
                        if bits >> (4 - col) & 1 == 0 {
                            continue;
                        }
                        let (px, py) = (cx + col * scale, y + row as i32 * scale);
                        if pass == 0 {
                            self.rect(px - 1, py - 1, scale + 2, scale + 2, SHADOW, 0.85);
                        } else {
                            self.rect(px, py, scale, scale, c, 1.0);
                        }
                    }
                }
                cx += 6 * scale;
            }
        }
    }

    /// Paste `src` with its top-left at (x, y).
    pub fn blit(&mut self, src: &Canvas, x: i32, y: i32) {
        for yy in 0..src.h as i32 {
            for xx in 0..src.w as i32 {
                if let (Some(d), Some(s)) = (self.at(x + xx, y + yy), src.at(xx, yy)) {
                    self.px[d..d + 4].copy_from_slice(&src.px[s..s + 4]);
                }
            }
        }
    }
}

/// Size in pixels of `s` drawn at `scale`.
pub fn text_size(s: &str, scale: i32) -> (i32, i32) {
    let n = s.chars().count() as i32;
    ((n * 6 - 1).max(0) * scale, 7 * scale)
}

/// 5x7 glyph rows, bit 4 leftmost. Lowercase draws as uppercase.
fn glyph(c: char) -> [u8; 7] {
    match c.to_ascii_uppercase() {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '-' => [0, 0, 0, 0b11111, 0, 0, 0],
        '+' => [0, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0],
        '/' => [0, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0],
        '.' => [0, 0, 0, 0, 0, 0b01100, 0b01100],
        ':' => [0, 0b01100, 0b01100, 0, 0b01100, 0b01100, 0],
        '(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        ')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        '=' => [0, 0, 0b11111, 0, 0b11111, 0, 0],
        ',' => [0, 0, 0, 0, 0b01100, 0b00100, 0b01000],
        '_' => [0, 0, 0, 0, 0, 0, 0b11111],
        ' ' => [0; 7],
        _ => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0, 0b00100],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_draws_inside_its_box() {
        let mut c = Canvas::new(40, 20, vec![0; 40 * 20 * 4]);
        c.text(2, 2, "8", 1, WHITE);
        let (w, h) = text_size("8", 1);
        assert_eq!((w, h), (5, 7));
        // The top bar of an 8 is lit, its middle column on row 1 isn't.
        assert_eq!(c.get(3, 2), Some(WHITE));
        assert_ne!(c.get(4, 3), Some(WHITE));
        // Nothing lit beyond the outline.
        assert_eq!(c.get(20, 15), Some([0, 0, 0]));
    }
}
