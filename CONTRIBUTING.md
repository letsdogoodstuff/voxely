# Contributing to voxely

Thanks for helping. Bug reports, ideas, docs and code are all welcome.

## Reporting bugs and asking for features

Open an [issue](https://github.com/letsdogoodstuff/voxely/issues/new/choose) using one of the
templates. For a bug, the most useful things are:

- Your OS and voxely version (`voxely --version`, or Preferences).
- What you did, what you expected, and what happened.
- A `.vox` file that shows it, if a model is involved, and a screenshot for anything visual.

Security problems go privately through [SECURITY.md](SECURITY.md), not in a public issue.

For a larger change (a new tool, panel or file format), open an issue first so we can agree on the
approach before you spend time on it.

## Building and running

You need [Rust](https://rustup.rs) (stable) and a C compiler (QuickJS is built from C): Xcode's
command line tools on macOS, `build-essential` on Linux, the Visual Studio C++ build tools on
Windows.

```sh
cargo build --release
./target/release/voxely             # opens your newest project in ~/Documents/voxely
./target/release/voxely --help      # options and hotkeys
```

To try changes without touching your own projects, point voxely at scratch folders:

```sh
VOXELY_WORKSPACE=/tmp/vx-work VOXELY_PREFS=/tmp/vx-prefs.json VOXELY_SOCKET=/tmp/vx.sock ./target/release/voxely
```

`VOXELY_SOCKET` matters if you already have voxely open: otherwise the newest window takes over
the control socket that agents and `voxely script` use.

## Tests and checks

CI runs these on Linux, macOS and Windows for every pull request. The tests must pass:

```sh
cargo test --locked
cargo clippy --locked --all-targets
```

Clippy doesn't fail the build yet, because older code has about 100 warnings. Don't add new ones in
code you write; fixing old ones in a separate pull request is welcome.

- Some tests render on the GPU. On Linux without a GPU, install Mesa's software Vulkan
  (`mesa-vulkan-drivers`) and they run on the CPU. CI skips them on Windows.
- `tests/scenes/make_materials.py` writes a scene with one sphere per material, for checking
  rendering by eye.
- For changes to the window, run it and try the change by hand. Screenshots in the pull request
  help reviewers.

## Pull requests

1. Fork the repo and branch from `main`.
2. Keep each pull request to one change, with tests where the behavior can be tested.
3. Describe what changed and why, and how you checked it (the template asks).
4. CI must be green. A maintainer reviews, and merges once it's ready.

Commit messages: a short summary line saying what changed for the user, then details if they
help. Look at `git log` for the style.

## Code style

- Match the code around you. There's no `rustfmt` config yet, so don't reformat files you aren't
  otherwise changing; it buries the real change in the diff.
- Doc comments say what a function is for and why, in plain sentences. Comments explain why, not
  what.
- Prefer small, plain functions over new abstractions, and keep dependencies few: each one is code
  every user downloads.
- Platform-specific code goes behind `cfg` in as few places as possible (see `src/ipc.rs` and
  `src/update.rs`).

## Agent tools and guidance

voxely's MCP server (`src/control.rs`, `src/control/`, `src/mcp.rs`) is how AI agents build with
it. Before changing tool descriptions, the server instructions, guides or prompts, read
[docs/agent-build-quality.md](docs/agent-build-quality.md). More guidance has made agents' models
worse before, so changes there need a test build compared side by side with earlier ones. New
tools need an example call in `src/control/validate.rs` (a test fails without one), and a line in
the README's tools table.

## Releasing (maintainers)

1. Bump `version` in `Cargo.toml`, run `cargo build` so `Cargo.lock` follows, and commit.
2. Tag and push: `git tag v0.2.0 && git push origin main v0.2.0`.
3. The [Release workflow](.github/workflows/release.yml) builds macOS, Linux and Windows and
   publishes one GitHub release with release notes from the merged pull requests. Installed copies
   pick it up within a day, or at their next launch.

Build a package locally with `packaging/build-macos.sh` (`--native` for this Mac only),
`packaging/build-linux.sh` or `packaging/build-windows.ps1`; each writes to `dist/`. Only these
builds update themselves (they set `VOXELY_RELEASE`); a plain `cargo build` never does.
`packaging/make-icons.sh` regenerates every icon from `packaging/icon-source.png`.

The macOS app is ad-hoc signed until these repository secrets are set; then it's signed with a
Developer ID and notarized: `MACOS_CERT_P12`, `MACOS_CERT_PASSWORD`, `VOXELY_SIGN_IDENTITY`,
`VOXELY_NOTARY_APPLE_ID`, `VOXELY_NOTARY_TEAM_ID`, `VOXELY_NOTARY_PASSWORD` (details in the
workflow file).

## License

voxely is licensed under the [GPL-3.0](LICENSE). By contributing, you agree that your
contributions are licensed under it too.
