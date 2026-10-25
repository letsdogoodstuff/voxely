//! User preferences kept between sessions (not per project): the
//! interface scale, palette favorites and autosave. Stored as JSON in the platform's settings folder.

use std::path::PathBuf;

use serde_json::{json, Value};

/// The scales Preferences offers and Cmd+= / Cmd+- step through.
pub const SCALES: [f32; 8] = [0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0];

/// Width in points the widest layout (render mode: palette, light,
/// matter and project columns) needs, with room for the 3D view.
pub const LAYOUT_WIDTH: f32 = 160.0 + 250.0 + 250.0 + 165.0 + 320.0;

/// Smallest window (logical pixels): the layout at the smallest scale.
pub const MIN_WINDOW: (f64, f64) = (LAYOUT_WIDTH as f64 * 0.75, 560.0);

#[derive(Clone, Debug, PartialEq)]
pub struct Prefs {
    /// Interface scale the user chose, 1 = 100%.
    pub ui_scale: f32,
    /// Starred palettes (file paths, or "preset:<name>"), in the order
    /// starred; the first four fill the palette's quick buttons.
    pub palette_favorites: Vec<String>,
    /// Save the project after every `autosave_every` edits.
    pub autosave: bool,
    pub autosave_every: usize,
}

impl Default for Prefs {
    fn default() -> Self {
        Prefs { ui_scale: 1.0, palette_favorites: Vec::new(), autosave: true, autosave_every: 25 }
    }
}

impl Prefs {
    /// `$VOXELY_PREFS`, or `~/Library/Application Support/voxely/prefs.json`
    /// on macOS and `$XDG_CONFIG_HOME/voxely/prefs.json` elsewhere.
    pub fn path() -> Option<PathBuf> {
        if let Some(p) = std::env::var_os("VOXELY_PREFS") {
            return Some(PathBuf::from(p));
        }
        let home = PathBuf::from(std::env::var_os("HOME")?);
        let dir = if cfg!(target_os = "macos") {
            home.join("Library/Application Support")
        } else {
            std::env::var_os("XDG_CONFIG_HOME").map_or_else(|| home.join(".config"), PathBuf::from)
        };
        Some(dir.join("voxely").join("prefs.json"))
    }

    /// The saved preferences, or the defaults if there are none.
    pub fn load() -> Self {
        let Some(text) = Self::path().and_then(|p| std::fs::read_to_string(p).ok()) else {
            return Self::default();
        };
        Self::from_json(&text)
    }

    fn from_json(text: &str) -> Self {
        let mut p = Self::default();
        if let Ok(v) = serde_json::from_str::<Value>(text) {
            if let Some(s) = v["ui_scale"].as_f64() {
                p.ui_scale = (s as f32).clamp(SCALES[0], SCALES[SCALES.len() - 1]);
            }
            if let Some(a) = v["palette_favorites"].as_array() {
                p.palette_favorites = a.iter().filter_map(|s| s.as_str().map(String::from)).collect();
            }
            if let Some(on) = v["autosave"].as_bool() {
                p.autosave = on;
            }
            if let Some(n) = v["autosave_every"].as_u64() {
                p.autosave_every = n.clamp(1, 100_000) as usize;
            }
        }
        p
    }

    pub fn save(&self) {
        let Some(path) = Self::path() else { return };
        let text = serde_json::to_string_pretty(&json!({ "ui_scale": self.ui_scale, "palette_favorites": self.palette_favorites, "autosave": self.autosave, "autosave_every": self.autosave_every })).unwrap_or_default();
        let res = path.parent().map_or(Ok(()), std::fs::create_dir_all).and_then(|_| std::fs::write(&path, text));
        if let Err(e) = res {
            eprintln!("saving preferences to {}: {e}", path.display());
        }
    }

    /// Step to the next larger (`dir > 0`) or smaller scale; 0 resets to 100%.
    pub fn step_scale(&mut self, dir: i32) {
        self.ui_scale = match dir {
            0 => 1.0,
            d if d > 0 => SCALES.iter().copied().find(|&s| s > self.ui_scale + 1e-3).unwrap_or(self.ui_scale),
            _ => SCALES.iter().rev().copied().find(|&s| s < self.ui_scale - 1e-3).unwrap_or(self.ui_scale),
        };
    }
}

/// The scale to draw at: the chosen one, unless the panels would then
/// cover the 3D view in a window `width` logical pixels wide.
pub fn effective_scale(chosen: f32, width: f32) -> f32 {
    chosen.min(width / LAYOUT_WIDTH).max(SCALES[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_through_scales() {
        let mut p = Prefs::default();
        p.step_scale(1);
        assert_eq!(p.ui_scale, 1.1);
        p.step_scale(-1);
        p.step_scale(-1);
        assert_eq!(p.ui_scale, 0.9);
        for _ in 0..10 {
            p.step_scale(-1);
        }
        assert_eq!(p.ui_scale, 0.75);
        for _ in 0..10 {
            p.step_scale(1);
        }
        assert_eq!(p.ui_scale, 2.0);
        p.step_scale(0);
        assert_eq!(p.ui_scale, 1.0);
    }

    #[test]
    fn reads_and_clamps_json() {
        assert_eq!(Prefs::from_json(r#"{"ui_scale": 1.5}"#).ui_scale, 1.5);
        assert_eq!(Prefs::from_json(r#"{"ui_scale": 9}"#).ui_scale, 2.0);
        assert_eq!(Prefs::from_json("not json").ui_scale, 1.0);
        assert_eq!(Prefs::from_json(r#"{"palette_favorites": ["preset:pico-8", 3]}"#).palette_favorites, vec!["preset:pico-8"]);
        let p = Prefs::from_json(r#"{"autosave": false, "autosave_every": 0}"#);
        assert_eq!((p.autosave, p.autosave_every), (false, 1));
        let d = Prefs::from_json("{}");
        assert_eq!((d.autosave, d.autosave_every), (true, 25));
    }

    #[test]
    fn scale_is_capped_to_fit_the_window() {
        // A 1280-wide laptop window fits 100% but not 150%.
        assert_eq!(effective_scale(1.0, 1280.0), 1.0);
        let s = effective_scale(1.5, 1280.0);
        assert!(s < 1.5 && s * LAYOUT_WIDTH <= 1280.0 + 0.01);
        // A wide monitor gets what was chosen.
        assert_eq!(effective_scale(2.0, 2560.0), 2.0);
        // Never below the smallest scale.
        assert_eq!(effective_scale(1.0, 500.0), 0.75);
    }
}
