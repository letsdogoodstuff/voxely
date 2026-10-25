//! Importing other formats: voxel files, meshes, heightmaps, slice sheets
//! and pictures, as new parts or a new project.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use super::{Ctx, boolean, int, string, text, tool};
use crate::app;
use crate::import::{How, mesh, picture};

pub const PART_TOOLS: &[&str] = &[];

pub fn tools() -> Vec<Value> {
    vec![tool(
        "import_file",
        "Bring in a model made elsewhere, as new parts right of the scene (one undo step; grouped when the file holds several models) or, with as_new_project, as a new project named after the file. Reads: other voxel formats (Qubicle .qb, .xraw, Ace of Spades .vxl, .kvx, .binvox, Minecraft .schematic / .schem); meshes (.obj with .mtl colors and textures, .glb / .gltf), voxelized at `size` voxels along the longest side; and PNG / JPEG pictures, read as `image` (a flat or extruded picture), `heightmap` (terrain: brighter is higher) or `slices` (a slice sheet: every z layer side by side seen from above, bottom first, as MagicaVoxel's slice export; or a folder of one image per layer). \
         Colors fill unused palette slots, then match the nearest palette colors. For building your own design from a picture, prefer looking at it and placing voxels; import is for bringing in finished models and terrain.",
        json!({
            "path": { "type": "string", "description": "The file (or, for slices, a folder of slice images)" },
            "read_as": { "type": "string", "enum": ["voxels", "mesh", "image", "heightmap", "slices"], "description": "Default: by extension (pictures default to image)" },
            "as_new_project": { "type": "boolean", "description": "Replace the scene with a new project holding the import, its palette made from the file's colors (default false: add parts)" },
            "add_colors": { "type": "boolean", "description": "Put the file's colors into unused palette slots (default true); false only matches existing colors" },
            "size": { "type": "integer", "description": "mesh: voxels along the longest side (default 64); image / heightmap: along the picture's longer side (default one voxel per pixel)" },
            "solid": { "type": "boolean", "description": "mesh: fill the inside (default true); heightmap: fill columns to the ground (default true)" },
            "z_up": { "type": "boolean", "description": "mesh: the file is z-up (default false: y-up, as glTF and most OBJ)" },
            "height": { "type": "integer", "description": "heightmap: voxels from lowest to highest point (default 32)" },
            "invert": { "type": "boolean", "description": "heightmap: dark is high" },
            "colors": { "type": "string", "enum": ["terrain", "picture", "gray"], "description": "heightmap: terrain bands by height (default), the picture's own colors, or grays" },
            "water": { "type": "integer", "description": "heightmap: fill water up to this height (default 0, none)" },
            "mode": { "type": "string", "enum": ["flat", "extrude", "relief"], "description": "image: one voxel thick (default), `depth` thick, or brighter pixels standing out up to `depth`" },
            "depth": { "type": "integer", "description": "image: thickness for extrude / relief (default 4)" },
            "lying": { "type": "boolean", "description": "image: lie on the ground (default: stand up facing -y)" },
            "columns": { "type": "integer", "description": "slices: slices across the sheet (default: found from the layout)" },
            "rows": { "type": "integer", "description": "slices: slices down the sheet" },
            "gap": { "type": "integer", "description": "slices: pixels between slices in a grid (default 0)" },
            "top_first": { "type": "boolean", "description": "slices: the first slice is the top layer" },
        }),
        &["path"],
    )]
}

pub fn dispatch(ctx: &mut Ctx, name: &str, a: &Value) -> Option<Result<Vec<Value>>> {
    match name {
        "import_file" => Some(import_file(ctx, a)),
        _ => None,
    }
}

/// The reading options from tool arguments.
pub fn how(path: &Path, a: &Value) -> Result<How> {
    let kind = match string(a, "read_as") {
        Some(k) => k.to_string(),
        None => crate::import::default_how(path).map(|h| h.name().to_string()).unwrap_or_else(|| if path.is_dir() { "slices".into() } else { String::new() }),
    };
    let u = |k: &str| int(a, k).map(|v| v.max(0) as u32);
    Ok(match kind.as_str() {
        "voxels" => How::Voxels,
        "mesh" => {
            let d = mesh::MeshOptions::default();
            How::Mesh(mesh::MeshOptions { size: u("size").unwrap_or(d.size), solid: boolean(a, "solid").unwrap_or(d.solid), z_up: boolean(a, "z_up").unwrap_or(d.z_up) })
        }
        "image" => {
            let depth = u("depth").unwrap_or(4);
            let mode = match string(a, "mode").unwrap_or("flat") {
                "flat" => picture::ImageMode::Flat,
                "extrude" => picture::ImageMode::Extrude(depth),
                "relief" => picture::ImageMode::Relief(depth),
                m => bail!("unknown mode {m:?}; use flat, extrude or relief"),
            };
            How::Image(picture::ImageOptions { size: u("size"), mode, standing: !boolean(a, "lying").unwrap_or(false), ..Default::default() })
        }
        "heightmap" => {
            let d = picture::HeightOptions::default();
            let colors = match string(a, "colors").unwrap_or("terrain") {
                "terrain" => picture::HeightColors::Terrain,
                "picture" => picture::HeightColors::Picture,
                "gray" => picture::HeightColors::Gray,
                c => bail!("unknown colors {c:?}; use terrain, picture or gray"),
            };
            How::Heightmap(picture::HeightOptions {
                size: u("size"),
                height: u("height").unwrap_or(d.height),
                invert: boolean(a, "invert").unwrap_or(false),
                colors,
                water: u("water").unwrap_or(0),
                solid: boolean(a, "solid").unwrap_or(d.solid),
            })
        }
        "slices" => How::Slices(picture::SliceOptions {
            columns: u("columns").unwrap_or(0),
            rows: u("rows").unwrap_or(0),
            gap: u("gap").unwrap_or(0),
            top_first: boolean(a, "top_first").unwrap_or(false),
        }),
        "" => bail!("{} isn't a format import_file reads; it takes {}", path.display(), crate::import::extensions().join(", ")),
        k => bail!("unknown read_as {k:?}"),
    })
}

fn import_file(ctx: &mut Ctx, a: &Value) -> Result<Vec<Value>> {
    let path = Path::new(string(a, "path").context("missing `path`")?);
    if !path.exists() {
        bail!("no file at {}", path.display());
    }
    let how = how(path, a)?;
    let new = boolean(a, "as_new_project").unwrap_or(false);
    let msg = app::import_file(path, &how, new, boolean(a, "add_colors").unwrap_or(true), ctx.r, ctx.scene, ctx.ed)?;
    super::materials_changed(ctx);
    Ok(vec![text(msg)])
}
