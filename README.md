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

### Tools

| Area | Tools |
|---|---|
| Scene | `get_scene_info`, `new_scene`, `open_file`, `save_file`, `resize_canvas` |
| Drawing | `set_voxels`, `fill_box`, `draw_shape` (box, ellipsoid, cylinder, cone, pyramid), `draw_line` |
| Reading | `get_voxels`, `get_slice` (a text map of one layer) |
| Model tools | `edit_model` (MagicaVoxel's Edit panel: fill, hull, mirror, rotate, …), `transform_model` (scale/shear/rotate/repeat) |
| Shaders | `list_shaders`, `run_shader` (MagicaVoxel voxel shaders, or inline GLSL) |
| Color | `get_palette`, `set_palette_colors`, `load_palette`, `set_material` (metal, glass, emit, …) |
| Selection | `select`, `move_selection` |
| History | `undo`, `redo` |
| Viewing | `set_camera`, `screenshot` (fast modeler image), `set_render_settings`, `render` (path traced), `set_view_mode` |

Conventions:
- Coordinates are voxel cells `[x, y, z]`, with z up.
- Colors are palette indices 1–255, or `"#rrggbb"`, which picks the nearest palette color.
  Use `set_palette_colors` for exact colors.
- `screenshot` and `render` return PNG images that the agent can see.
