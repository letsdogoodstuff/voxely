# Agent build quality: what went wrong and what to keep

This records why AI agents' models got worse as the MCP server grew, and what
fixed it, so we don't repeat it. Read this before changing the MCP
instructions, guides, prompts or tool descriptions (`src/control.rs`,
`src/control/guide.rs`, tool `description` strings).

## What happened (Oct 24-25, 2026)

- **The best builds came first.** The Skeleton Warrior, anime_war_princess,
  goblin_dnd and grazer_bear (Oct 24, 17:43-18:39) were made with the first
  MCP: a few basic tools and five lines of instructions. Agents designed the
  models themselves: chunky shapes, flat colors, details placed by hand.
- **Quality dropped as guidance was added.** From 20:30 on we added a craft
  guide, workflow prompts, surface tools and reference tools. The
  white_hair_death_knight, built with all of it, came out muddy and
  speckled, and looked worse than the early builds.
- **A picture-to-voxel pipeline made it worse.** A branch tried to turn any
  picture into a voxel scene with a depth model (Depth Anything V2 via ONNX),
  projecting the picture's colors onto depth-built voxels. The result lost
  most of the character's detail, was built at about half the picture's
  block size, and was judged "trash". It was removed.
- **The fix was to go back.** Instructions, guides and prompts were rewritten
  around the early builds' style, and the speckle tools were unlisted
  (commit 53aad09).

## What didn't work

1. **Telling agents to shade.** "3-5 shades per material", a shading stage in
   every prompt, and `shade_edges` (darken edges, lighten tops) produced
   painted gradients that read as mud at voxel scale.
2. **Noise and color variation.** `vary_colors` and noise-based coloring
   added speckle. Designed voxel art uses flat color areas.
3. **Copying colors from a reference picture.** `paint_from_reference` (and
   the pipeline's projection) painted each voxel from the picture's pixels.
   Lighting, shadows and averaging came along with it: speckled, relit,
   blurry surfaces, never the clean colors of the original.
4. **Building from estimated depth.** A depth model gives a smooth, noisy
   surface, not the designed blocks of the original. Detail the picture
   shows (faces, trim, ornaments) was lost.
5. **Choosing the size by guess.** The ruins knight was built about 88
   voxels tall when the picture's knight was about 120-150 blocks tall, which
   halved the detail. **For voxel-art pictures, always build at the
   picture's own block size: one picture block = one voxel.**
6. **Chasing scores.** Optimizing silhouette overlap (IoU) and similar
   numbers didn't make models look better. Judge by eye, side by side
   with the reference.
7. **Smooth SDF blobs** (smoothUnion, organic blending) as the default way to
   shape things: soft lumps instead of chunky, readable forms.
8. **More instructions.** Every rule we added competed with the agent's own
   sense of design. Long, prescriptive workflows made agents follow
   steps instead of looking at what they were building.
9. **Large changes in one go, on main.** Many MCP changes landed together,
   so it took a long time to see which ones hurt. Big guidance changes now
   go in a worktree and get a test build first.

## What to keep (the house style)

- Chunky boxy forms; one flat color per material plus a few deliberate
  accents; small details placed by hand (eyes with pupils and highlights,
  buckles, gems, trim). No painted shading, noise or speckle.
- A character designed from imagination is 32-48 voxels tall.
- A picture is a **design reference**: the agent looks at it, writes down
  what it sees, and builds each object by hand. It doesn't project colors or
  build from depth.
- If the picture is voxel art, match its block size (`analyze_image`
  measures it; the agent judges whether the picture really is voxel art).
- `vary_colors`, `shade_edges` and `paint_from_reference` stay unlisted
  (`UNLISTED` in `src/control.rs`); they still work when called by name or
  when a user asks for them.
- Keep the instructions short. Prefer giving the agent good tools and
  good feedback (multi-view screenshots) over telling it how to design.

## Before changing agent guidance

- Make the change in a worktree, not on main.
- Build something with it (a character, a prop, a scene) and compare it
  side by side with the early builds in `~/Documents/voxely/*.png`
  (Skeleton Warrior, anime_war_princess, grazer_bear).
- If it isn't at least as good, don't merge it.
