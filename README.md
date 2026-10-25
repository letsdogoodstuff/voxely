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

### The Agents panel

Click **Agents** in the bottom bar to open the Agents panel under the 3D view:

- **Allow agents to control this window**: turn off to pause agents; their requests get a
  reply saying control is paused (instead of quietly working headless).
- Status (active, idle or paused) and the agent's recent actions, with failures in red.
- **Add to Claude Code** registers voxely for all your Claude Code sessions;
  **Copy command** and **Copy Codex config** copy the setup for a terminal or `~/.codex/config.toml`.

There is no server to start or stop: agents launch `voxely mcp` themselves for each session.
Agent edits go through normal undo, so Cmd+Z steps back through them.

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
| Animation | `set_animation`, `go_to_frame`, `play` (plays it in the window), `set_keyframe`, `delete_keyframe`, `new_drawing`, `remove_drawing`, `frame_strip` (several frames in one image) |
| Rigs | `create_rig` (humanoid, quadruped), `apply_motion` (idle, walk, run, jump, attack, wave, death, quad_walk, quad_idle), `import_animation` (glTF clips) |
| Parts | `list_parts`, `create_part`, `edit_part` (or `"world"`), `transform_part` (move, 90° rotate, flip, free rotation, attach), `set_pivot`, `set_socket`, `modify_part` (rename, duplicate, delete, hide, show, ungroup, detach), `group_parts` |
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
- Coordinates are voxel cells `[x, y, z]`, with z up. Voxel tools work in the edited part's own
  coordinates; pass `part` to any of them to switch parts first. Part positions are world
  coordinates of the part's minimum corner.
- Colors are palette indices 1–255, or `"#rrggbb"`, which picks the nearest palette color.
  Use `set_palette_colors` for exact colors.
- `screenshot` and `render` return PNG images that the agent can see.

## Parts and the World view

Like MagicaVoxel's World editor, a scene is a tree of models ("parts") and groups, each with a name,
a hidden flag and a 90° rotation / flip plus a position. `.vox` files keep this tree when opened and
saved, so files from MagicaVoxel round-trip with their objects, groups, names and layers.

- **Tab** switches between editing one part and the **World** view (also the Edit / World switch above
  the 3D view). While editing, the other parts show dimmed for reference, and every brush and Edit
  operation works on the edited part only. Render mode always shows every part.
- In World view, click a part to pick it (Shift adds, Alt picks a part inside a group), drag to move,
  double-click to edit. The World panel on the right lists the tree and has New, Copy, Group,
  Ungroup, Hide, Delete, and rotate / flip / move buttons. MagicaVoxel's World-editor keys work too
  (arrows and PageUp/PageDown move, -/= and 9/0 turn, Ctrl+R group, Ctrl+H hide, Delete).
- Undo steps through voxel edits in the current edit session, then whole-part changes.
- A part's box stays put when you resize or reshape it, so cutting a piece out and fitting it no
  longer shifts it.
- Exports other than `.vox` and glTF flatten all visible parts.

### Pivots, free rotation and sockets

These go beyond MagicaVoxel (they are stored in `.vox` attributes MagicaVoxel ignores, so the file
still opens there, just without them):

- **Pivot**: the point a part or group turns about, e.g. an arm's shoulder or a door's hinge. Default:
  the center. Set it in the World panel (Center / Bottom buttons or exact values) or with `set_pivot`.
- **Free rotation**: any angle about the pivot, on top of the exact 90° turns. Shown resampled to the
  voxel grid; the part is edited upright and on its own. MagicaVoxel shows such parts unrotated.
- **Sockets**: named attachment points (hand, muzzle, seat). Shown as green dots on picked parts.
- **Attaching**: a part can be attached to another part, like bones in a skeleton: it then follows
  that part's moves and turns (raise the arm and the sword in its hand comes along). Set it with
  "Attached to" in the World panel (with buttons to snap onto the parent's sockets), or
  `transform_part` with `attach_to: "body:shoulder_r"`, which snaps the part's pivot (or its own
  socket, `attach_socket`) onto that socket and attaches it. The joint becomes the part's pivot.
  In `.vox` files an attached chain is saved as a MagicaVoxel group, so MagicaVoxel shows and moves
  it as one object. Positions are whole voxels, so a socket on a voxel corner and one on a voxel
  center line up to within half a voxel.

## Animation

Click **Anim** in the bottom bar for the timeline: play / pause (`/`), step frames (`,` `.`), frame
count, speed, loop, and smooth (blended) or stepped keys. Frames are numbered from 0.

- **Keyframes**: with more than one frame, moving or turning a part keys it on the current frame
  (the first key off frame 0 also keys frame 0 with where it was). The timeline's **Key** /
  **Remove key** buttons work on the picked part, and its keys show as diamonds. Keys blend about
  the part's pivot, and attached parts follow, so animate from the root down.
- **Flipbook drawings**: **New drawing here** gives the picked part its own copy of its voxels from
  this frame on, to redraw (a blink, a lifted foot). Drawings show as squares.
- Both are saved the way MagicaVoxel stores animation (`nTRN` frames, `nSHP` models with `_f`).
- **Rigs and motions**: World panel ▸ Rig adds a humanoid or four-legged rig of jointed blocks
  (redraw each part as your character). The timeline's motion menu keys a ready-made motion onto
  it: idle, walk, run, jump, attack, wave, death, quad_walk, quad_idle.
- **Importing clips**: **Import...** reads the clips in a `.glb` / `.gltf` file and retargets one onto
  the rig (picked part's rig, or the first one): joints are matched to parts by name (voxely, Mixamo,
  Unreal and Rigify naming), rest poses are lined up so T-pose clips drive arms-down rigs, each part
  turns rigidly like its joint, and the hips carry the clip's movement, scaled to the rig. A good
  free source is Quaternius's [Universal Animation Library](https://quaternius.com/packs/universalanimationlibrary.html)
  (CC0, 45 clips in the free Standard pack; use the Godot `.glb`).
- **glTF export** includes the animation as one clip (sampled every frame; drawings switch by scale).

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
- Structure: the scene tree. A root node named after the project, then a node per group and part
  with its transform; attached parts are child nodes of their parent part, ready to animate as a
  rig. Each node's origin is the part's pivot, so rotating it in an engine turns it the same way it
  turns in voxely. Mirrored parts are baked into their meshes, so every node is a plain rotation
  (some engines mishandle mirrored nodes). Sockets are empty child nodes (`extras.voxely_socket: true`) to
  parent weapons, riders or effects to; clouds are child nodes of their part.

Exports pass the Khronos glTF Validator with no errors or warnings (it can't yet check
`KHR_materials_diffuse_transmission`, which is newer than the validator).
