//! Temporary stand-in for the export formats (replaced by the real module).

use std::path::{Path, PathBuf};

use crate::scene::Scene;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format { Obj, Ply, Mc, Xraw, Slab, Qb, Vox, Iso, TwoD, Slice, Point, Cube }

impl Format {
    pub const ALL: [Format; 12] = [Format::Obj, Format::Ply, Format::Mc, Format::Xraw, Format::Slab, Format::Qb, Format::Vox, Format::Iso, Format::TwoD, Format::Slice, Format::Point, Format::Cube];
    pub fn name(self) -> &'static str {
        ["obj", "ply", "mc", "xraw", "slab", "qb", "vox", "iso", "2d", "slice", "point", "cube"][self as usize]
    }
    pub fn extension(self) -> &'static str {
        ["obj", "ply", "ply", "xraw", "vox", "qb", "vox", "png", "png", "png", "ply", "ply"][self as usize]
    }
    pub fn hint(self) -> &'static str {
        self.name()
    }
    pub fn from_name(s: &str) -> Option<Format> {
        Format::ALL.into_iter().find(|f| f.name().eq_ignore_ascii_case(s))
    }
}

pub fn export(format: Format, _scene: &Scene, _path: &Path) -> anyhow::Result<Vec<PathBuf>> {
    anyhow::bail!("{} export not built yet", format.name())
}
