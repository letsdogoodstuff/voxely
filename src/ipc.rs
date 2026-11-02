//! The local socket a voxely window listens on for `voxely mcp` and
//! `voxely script`: a Unix domain socket everywhere (Windows 10 and later
//! have them too, through `uds_windows`).

use std::path::Path;

#[cfg(unix)]
pub use std::os::unix::net::{UnixListener as Listener, UnixStream as Stream};
#[cfg(windows)]
pub use uds_windows::{UnixListener as Listener, UnixStream as Stream};

/// Identifies the socket file at a path, so a window can tell its own from
/// a newer window's: its inode, or on Windows its NTFS file index. (Not the
/// creation time: NTFS gives a file made soon after another of the same name
/// was deleted that file's creation time.)
#[cfg(unix)]
pub fn file_id(path: &Path) -> Option<u64> {
    std::fs::symlink_metadata(path).ok().map(|m| std::os::unix::fs::MetadataExt::ino(&m))
}

#[cfg(windows)]
pub fn file_id(path: &Path) -> Option<u64> {
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        BY_HANDLE_FILE_INFORMATION, FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
        GetFileInformationByHandle,
    };
    // A socket is a reparse point: open the point itself, asking for no
    // access, just its attributes.
    let file = std::fs::OpenOptions::new()
        .access_mode(0)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)
        .ok()?;
    // SAFETY: a valid open handle and a zeroed struct for the call to fill.
    let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
    if unsafe { GetFileInformationByHandle(file.as_raw_handle() as _, &mut info) } == 0 {
        return None;
    }
    Some((u64::from(info.nFileIndexHigh) << 32) | u64::from(info.nFileIndexLow))
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
