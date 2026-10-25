# voxely

A MagicaVoxel-style voxel editor and path tracer in Rust and wgpu.

```sh
cargo build --release
./target/release/voxely path/to/model.vox     # open a model
./target/release/voxely --new                 # start from a 40x40x40 block
./target/release/voxely --help                # all options and controls
```

## Driving voxely from an agent (MCP)

`voxely mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server on stdio.
Agents such as Claude Code or Codex can use it to build, edit, light and render models.

- If a voxely window is open, every call goes to that window, so you can watch the
  agent work (and keep editing by hand).
- Otherwise it runs headless in the MCP process. The scene lives until the client
  disconnects, so ask the agent to `save_file`.

### Claude Code

This repo's `.mcp.json` registers the server for Claude Code sessions started here
(build first with `cargo build --release`). To use it from anywhere:

```sh
claude mcp add voxely -- /path/to/voxely/target/release/voxely mcp
```

### Codex

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.voxely]
command = "/path/to/voxely/target/release/voxely"
args = ["mcp"]
```

### Other clients

Any MCP client that launches stdio servers works: run `voxely mcp`.

Options:
- `--headless`: never use an open window.
- `--assets <dir>`: MagicaVoxel's install folder, for its shaders, palettes and models.
  You can also set `VOXELY_ASSETS`.

### Projects

Projects live in a workspace folder, `~/Documents/voxely` (set `VOXELY_WORKSPACE` to change it).
The window's Project column lists them under **My projects**, above MagicaVoxel's **Samples**, and
re-reads the folder every second, so projects an agent creates show up while you watch. Ctrl+S saves
the current project there; opening a sample and saving makes a copy, leaving the original alone.
Exports (the Export box at the bottom of the Project column) go to `<workspace>/export` by default.

### Tools

| Area | Tools |
|---|---|
| Projects | `new_project`, `list_projects`, `open_project`, `save_file` |
| Scene | `get_scene_info`, `new_scene`, `open_file`, `resize_canvas` |
| Drawing | `set_voxels`, `fill_box`, `draw_shape` (box, ellipsoid, cylinder, cone, pyramid), `draw_line` |
| Reading | `get_voxels`, `get_slice` (a text map of one layer) |
| Model tools | `edit_model` (MagicaVoxel's Edit panel: fill, hull, mirror, rotate, …), `transform_model` (scale/shear/rotate/repeat) |
| Shaders | `list_shaders`, `run_shader` (MagicaVoxel voxel shaders, or inline GLSL) |
| Color | `get_palette`, `set_palette_colors`, `load_palette`, `set_material` (metal, glass, emit, …) |
| Selection | `select`, `move_selection` |
| History | `undo`, `redo` |
| Viewing | `set_camera`, `screenshot` (fast modeler image), `set_render_settings`, `render` (path traced), `set_view_mode` |
| Export | `export`: **gltf** (`.glb` for game engines), plus MagicaVoxel's formats: obj, ply, mc, cube (meshes), xraw, slab, qb, vox (volumes), iso, 2d, slice (sprite images), point (point cloud) |

Conventions:
- Coordinates are voxel cells `[x, y, z]`, with z up.
- Colors are palette indices 1–255, or `"#rrggbb"`, which picks the nearest palette color.
  Use `set_palette_colors` for exact colors.
- `screenshot` and `render` return PNG images that the agent can see.

## glTF export

The Export box's **glTF (.glb)** button (and the `export` tool with `format: "gltf"`) writes one
self-contained `.glb` that Godot, Unity (glTFast), Unreal, Three.js, Babylon and Bevy import directly.
Name the file `.gltf` instead to get readable JSON with the data embedded.

- Mesh: same-color faces merged, 1 voxel = 0.1 m, Y up, centered, front facing +Z.
- Colors: the palette as a 256x1 texture with nearest filtering, so colors stay crisp.
- Materials: one per distinct voxely material, on glTF's metal/roughness model:
  metal maps to metallic/roughness, emit to emissive (`KHR_materials_emissive_strength`),
  glass to `KHR_materials_transmission` + `KHR_materials_ior`.
- Clouds: glTF can't describe volumetric scattering, so cloud materials degrade in layers:
  - Everywhere: alpha blending, with opacity from the cloud's density over its average
    thickness (`1 - e^(-density x 4V/S)`), so clouds read as soft see-through puffs.
  - Where supported: `KHR_materials_diffuse_transmission` and `KHR_materials_volume`
    for light diffusing through and fading with the same density.
  - For engine fog: each cloud color is its own node (`<project>_cloud`) whose `extras` hold
    `"voxely_material": "cloud"`, `palette_index`, `color`, `density` (extinction per voxel,
    1 voxel = 0.1 m), `phase` and `opacity`. Find that node on import and swap in Godot's
    FogVolume, Unity/Unreal volumetric fog, or a custom shader.
- Structure: a root node named after the project, with the model and each cloud as child nodes.

Exports pass the Khronos glTF Validator with no errors or warnings (it can't yet check
`KHR_materials_diffuse_transmission`, which is newer than the validator).
