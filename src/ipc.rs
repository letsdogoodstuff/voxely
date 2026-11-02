//! The local socket a voxely window listens on for `voxely mcp` and
//! `voxely script`: a Unix domain socket everywhere (Windows 10 and later
//! have them too, through `uds_windows`).

use std::fs::Metadata;
use std::path::Path;

#[cfg(unix)]
pub use std::os::unix::net::{UnixListener as Listener, UnixStream as Stream};
#[cfg(windows)]
pub use uds_windows::{UnixListener as Listener, UnixStream as Stream};

/// Identifies the socket file at a path, so a window can tell its own from
/// a newer window's (the inode; on Windows, when it was created).
pub fn file_id(path: &Path) -> Option<u64> {
    std::fs::symlink_metadata(path).ok().map(|m| id_of(&m))
}

#[cfg(unix)]
fn id_of(m: &Metadata) -> u64 {
    std::os::unix::fs::MetadataExt::ino(m)
}

#[cfg(windows)]
fn id_of(m: &Metadata) -> u64 {
    std::os::windows::fs::MetadataExt::creation_time(m)
}

/// Only this user may connect. (On Windows the socket lives in the user's
/// own temp folder, which other users can't open.)
pub fn owner_only(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    #[cfg(windows)]
    let _ = path;
}
