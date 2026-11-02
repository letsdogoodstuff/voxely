# Security

Please report security problems privately, not in a public issue: use **Report a vulnerability**
on this repository's [Security tab](https://github.com/letsdogoodstuff/voxely/security/advisories/new).
Say what's affected, how to reproduce it, and what an attacker could do. We'll reply as soon as
we can and credit you in the fix unless you'd rather we didn't.

Areas where a problem would matter most:

- **Updates** (`src/update.rs`): voxely downloads and installs releases from GitHub by itself.
- **The control socket** (`src/ipc.rs`, `src/app.rs`): local programs drive the open window
  through it.
- **Scripts** (`src/script.rs`): JavaScript run by `voxely script` and agents is meant to reach
  files only through voxely's tools.
- **File import**: `.vox`, meshes, images and other formats come from anywhere.

Only the latest release gets security fixes.
