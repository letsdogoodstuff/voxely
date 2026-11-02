//! Updates from GitHub releases.
//!
//! A background thread asks GitHub for the latest release at startup and
//! once a day. When it is newer, it downloads this platform's file from the
//! release, checks it against GitHub's SHA-256, and puts the new version in
//! place of the running one:
//!
//! - macOS: `voxely-<v>-macos.dmg`; the app inside replaces `voxely.app`
//!   (its version and code signature are checked first).
//! - Linux: `voxely-<v>-linux-<arch>.tar.gz`; its `voxely` replaces this
//!   binary.
//! - Windows: `voxely-<v>-windows-<arch>.zip`; its `voxely.exe` replaces this
//!   one (the running one is renamed aside, since Windows won't overwrite
//!   it, and removed on the next start).
//!
//! The running copy keeps working; the status bar offers a restart into the
//! new version, and the next launch gets it anyway. Only release builds
//! update themselves (packaging sets `VOXELY_RELEASE` at build time; a plain
//! `cargo build` never does), and only where they can replace themselves:
//! not from a read-only place like the disk image, a translocated download
//! or a system folder. Then the status bar links to the release instead.
//! Downloads use the system's `curl` and archives its `tar` (both ship with
//! macOS and Windows 10 and later).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde_json::Value;

pub const REPO: &str = "letsdogoodstuff/voxely";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Built by `packaging/` for a release (and so allowed to update itself).
const RELEASE_BUILD: bool = option_env!("VOXELY_RELEASE").is_some();
/// How often a running app checks again.
const EVERY: Duration = Duration::from_secs(24 * 60 * 60);

#[derive(Clone, Debug, PartialEq)]
pub enum State {
    /// Not checked (or checking is off, or this isn't a release build).
    Idle,
    Checking,
    UpToDate,
    Downloading(String),
    /// Installed; a restart runs it.
    Ready(String),
    /// Newer, but this copy can't replace itself: `(version, why)`.
    Available(String, String),
    Failed(String),
}

static STATE: Mutex<State> = Mutex::new(State::Idle);
static ENABLED: AtomicBool = AtomicBool::new(false);
static STARTED: AtomicBool = AtomicBool::new(false);
static BUSY: AtomicBool = AtomicBool::new(false);
static RELAUNCH: AtomicBool = AtomicBool::new(false);

pub fn state() -> State {
    STATE.lock().map(|s| s.clone()).unwrap_or(State::Idle)
}

fn set(s: State) {
    log::info!("update: {s:?}");
    if let Ok(mut g) = STATE.lock() {
        *g = s;
    }
}

/// Turn automatic checks on or off; the first time they are on, start the
/// checking thread.
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
    if !on || !RELEASE_BUILD || std::env::var_os("VOXELY_NO_UPDATE").is_some() || STARTED.swap(true, Ordering::Relaxed) {
        return;
    }
    remove_leftovers();
    std::thread::spawn(|| {
        loop {
            if ENABLED.load(Ordering::Relaxed) {
                check_once();
            }
            std::thread::sleep(EVERY);
        }
    });
}

/// Check now (Preferences' button), on another thread.
pub fn check_now() {
    if !RELEASE_BUILD {
        set(State::Failed("this is a development build; only downloaded releases update themselves".into()));
        return;
    }
    std::thread::spawn(check_once);
}

/// `voxely update`: check and install now, in this process.
pub fn cli() -> Result<()> {
    ensure!(RELEASE_BUILD, "this is a development build; only downloaded releases update themselves");
    remove_leftovers();
    println!("voxely {VERSION}: checking {}", release_page());
    run()?;
    println!("{}", describe());
    Ok(())
}

/// Ask for the new version once the app has quit (see `relaunch_if_asked`).
pub fn request_relaunch() {
    RELAUNCH.store(true, Ordering::Relaxed);
}

/// After the event loop ends: start the updated app if the user asked.
pub fn relaunch_if_asked() {
    if !RELAUNCH.load(Ordering::Relaxed) {
        return;
    }
    let Some(target) = installed() else { return };
    if cfg!(target_os = "macos") {
        // -n: a new instance even while this one is still exiting.
        let _ = Command::new("/usr/bin/open").arg("-n").arg(&target).spawn();
    } else {
        let _ = Command::new(&target).spawn();
    }
}

/// The release's page, for a copy that can't update itself.
pub fn release_page() -> String {
    format!("https://github.com/{REPO}/releases/latest")
}

/// Open a web page in the default browser.
pub fn open_url(url: &str) {
    let _ = if cfg!(target_os = "macos") {
        Command::new("/usr/bin/open").arg(url).spawn()
    } else if cfg!(windows) {
        Command::new("explorer").arg(url).spawn()
    } else {
        Command::new("xdg-open").arg(url).spawn()
    };
}

fn check_once() {
    if BUSY.swap(true, Ordering::Relaxed) {
        return;
    }
    if matches!(state(), State::Ready(_)) {
        // Already installed; don't download it again.
        BUSY.store(false, Ordering::Relaxed);
        return;
    }
    set(State::Checking);
    if let Err(e) = run() {
        set(State::Failed(format!("{e:#}")));
    }
    BUSY.store(false, Ordering::Relaxed);
}

fn run() -> Result<()> {
    let rel = latest()?;
    if !newer(&rel.version, VERSION) {
        set(State::UpToDate);
        return Ok(());
    }
    let Some(url) = rel.download.clone() else {
        set(State::Available(rel.version, format!("the release has no download for this computer ({})", asset_suffix())));
        return Ok(());
    };
    let target = installed().context("can't tell where voxely is installed")?;
    if let Err(why) = can_replace(&target) {
        set(State::Available(rel.version, why));
        return Ok(());
    }
    set(State::Downloading(rel.version.clone()));
    install(&rel.version, &url, rel.sha256.as_deref(), &target)?;
    set(State::Ready(rel.version));
    Ok(())
}

// ---------------------------------------------------------------- releases

#[derive(Debug, PartialEq)]
struct Release {
    version: String,
    /// This platform's file, if the release has one.
    download: Option<String>,
    /// GitHub's SHA-256 of that file, hex.
    sha256: Option<String>,
}

/// How this platform's release file name ends.
fn asset_suffix() -> String {
    let arch = std::env::consts::ARCH;
    if cfg!(target_os = "macos") {
        "-macos.dmg".into()
    } else if cfg!(windows) {
        format!("-windows-{arch}.zip")
    } else {
        format!("-linux-{arch}.tar.gz")
    }
}

fn curl() -> Command {
    Command::new(if cfg!(target_os = "macos") { "/usr/bin/curl" } else { "curl" })
}

fn latest() -> Result<Release> {
    // VOXELY_UPDATE_URL: another release JSON (any URL curl takes), for testing.
    let url = std::env::var("VOXELY_UPDATE_URL").unwrap_or_else(|_| format!("https://api.github.com/repos/{REPO}/releases/latest"));
    let out = curl()
        .args(["-fsSL", "--max-time", "30", "-H", "Accept: application/vnd.github+json"])
        .arg("-H")
        .arg(format!("User-Agent: voxely/{VERSION}"))
        .arg(&url)
        .output()
        .context("running curl")?;
    if !out.status.success() {
        bail!("couldn't reach GitHub: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    parse_release(&serde_json::from_slice(&out.stdout).context("GitHub's reply isn't JSON")?, &asset_suffix())
}

/// The version and the file ending in `suffix` of a GitHub release (API
/// JSON).
fn parse_release(v: &Value, suffix: &str) -> Result<Release> {
    let tag = v["tag_name"].as_str().context("release has no tag")?;
    let version = tag.trim_start_matches('v').to_string();
    let asset = v["assets"].as_array().into_iter().flatten().find(|a| a["name"].as_str().is_some_and(|n| n.ends_with(suffix)));
    let download = asset.and_then(|a| a["browser_download_url"].as_str()).map(String::from);
    let sha256 = asset.and_then(|a| a["digest"].as_str()).and_then(|d| d.strip_prefix("sha256:")).map(str::to_lowercase);
    Ok(Release { version, download, sha256 })
}

/// "1.2.10" > "1.2.9"; missing parts count as 0.
fn newer(a: &str, b: &str) -> bool {
    let parts = |s: &str| -> Vec<u64> { s.split(['.', '-', '+']).take(3).map(|p| p.parse().unwrap_or(0)).collect() };
    let (mut x, mut y) = (parts(a), parts(b));
    x.resize(3, 0);
    y.resize(3, 0);
    x > y
}

// ---------------------------------------------------------------- install

/// What an update replaces: the `.app` on macOS, else this executable.
fn installed() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?.canonicalize().ok()?;
    if cfg!(target_os = "macos") { bundle_of(&exe) } else { Some(strip_verbatim(exe)) }
}

/// Windows' canonical paths start with `\\?\`, which some programs reject.
fn strip_verbatim(p: PathBuf) -> PathBuf {
    match p.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
        Some(s) if !s.starts_with("UNC") => PathBuf::from(s),
        _ => p,
    }
}

fn bundle_of(exe: &Path) -> Option<PathBuf> {
    let macos = exe.parent()?;
    let contents = macos.parent()?;
    let app = contents.parent()?;
    (macos.file_name()? == "MacOS" && contents.file_name()? == "Contents" && app.extension()? == "app").then(|| app.to_path_buf())
}

/// Why `target` can't be replaced, if it can't.
fn can_replace(target: &Path) -> std::result::Result<(), String> {
    let s = target.to_string_lossy();
    if s.contains("/AppTranslocation/") {
        return Err("move voxely to Applications (drag it out of Downloads) so it can update itself".into());
    }
    if s.starts_with("/Volumes/") {
        return Err("you're running voxely from its disk image; drag it to Applications so it can update itself".into());
    }
    let dir = target.parent().ok_or("voxely's folder is unknown")?;
    let probe = dir.join(format!(".voxely-write-test-{}", std::process::id()));
    match std::fs::write(&probe, b"") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            Ok(())
        }
        Err(_) => Err(format!("{} isn't writable by you, so voxely can't update itself there", dir.display())),
    }
}

fn cache_dir() -> PathBuf {
    std::env::temp_dir().join(format!("voxely-update-{}", std::process::id()))
}

fn sh(cmd: &mut Command) -> Result<String> {
    let out = cmd.output().with_context(|| format!("running {:?}", cmd.get_program()))?;
    if !out.status.success() {
        bail!("{:?} failed: {}", cmd.get_program(), String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn sha256_file(path: &Path) -> Result<String> {
    use sha2::Digest;
    let mut hash = sha2::Sha256::new();
    std::io::copy(&mut std::fs::File::open(path)?, &mut hash)?;
    Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn install(version: &str, url: &str, sha256: Option<&str>, target: &Path) -> Result<()> {
    let dir = cache_dir();
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    let file = dir.join(url.rsplit('/').next().filter(|n| !n.is_empty()).unwrap_or("download"));
    let r = (|| {
        sh(curl().args(["-fsSL", "--max-time", "900", "-o"]).arg(&file).arg(url)).context("downloading the update")?;
        if let Some(want) = sha256 {
            ensure!(sha256_file(&file)? == want, "the download doesn't match GitHub's checksum");
        }
        if cfg!(target_os = "macos") { install_app(version, &file, &dir, target) } else { install_exe(version, &file, &dir, target) }
    })();
    let _ = std::fs::remove_dir_all(&dir);
    r
}

/// macOS: copy the disk image's app next to the installed one, check it,
/// and swap them.
fn install_app(version: &str, dmg: &Path, dir: &Path, app: &Path) -> Result<()> {
    let mnt = dir.join("mnt");
    std::fs::create_dir_all(&mnt)?;
    sh(Command::new("/usr/bin/hdiutil").args(["attach", "-nobrowse", "-readonly", "-noautoopen", "-mountpoint"]).arg(&mnt).arg(dmg)).context("opening the disk image")?;
    let staged = app.with_file_name(format!(".voxely-{version}.app"));
    let copied = (|| {
        let new = find(&mnt, |p| p.extension().is_some_and(|e| e == "app")).context("the disk image has no app")?;
        let _ = std::fs::remove_dir_all(&staged);
        sh(Command::new("/usr/bin/ditto").arg(&new).arg(&staged)).context("copying the new app")?;
        let plist = staged.join("Contents/Info.plist");
        let v = sh(Command::new("/usr/bin/plutil").args(["-extract", "CFBundleShortVersionString", "raw"]).arg(&plist))?;
        ensure!(v == version, "the release says {version} but its app is {v}");
        sh(Command::new("/usr/bin/codesign").args(["--verify", "--deep", "--strict"]).arg(&staged)).context("the new app's signature doesn't check out")?;
        Ok(())
    })();
    let _ = sh(Command::new("/usr/bin/hdiutil").args(["detach", "-force"]).arg(&mnt));
    if let Err(e) = copied {
        let _ = std::fs::remove_dir_all(&staged);
        return Err(e);
    }
    let old = app.with_file_name(".voxely-old.app");
    let _ = std::fs::remove_dir_all(&old);
    std::fs::rename(app, &old).context("moving the old app aside")?;
    if let Err(e) = std::fs::rename(&staged, app) {
        let _ = std::fs::rename(&old, app);
        return Err(anyhow!(e).context("putting the new app in place"));
    }
    let _ = std::fs::remove_dir_all(&old);
    Ok(())
}

/// Linux and Windows: unpack the archive, check the new program's version,
/// and swap it in for this one.
fn install_exe(version: &str, archive: &Path, dir: &Path, exe: &Path) -> Result<()> {
    let out = dir.join("x");
    std::fs::create_dir_all(&out)?;
    // GNU tar reads .tar.gz; Windows' bsdtar also reads .zip.
    sh(Command::new("tar").arg("-xf").arg(archive).arg("-C").arg(&out)).context("unpacking the update")?;
    let name = if cfg!(windows) { "voxely.exe" } else { "voxely" };
    let new = find(&out, |p| p.file_name().is_some_and(|n| n == name) && p.is_file()).with_context(|| format!("the download has no {name}"))?;
    let staged = exe.with_file_name(if cfg!(windows) { "voxely.new.exe" } else { ".voxely-new" });
    std::fs::copy(&new, &staged).context("copying the new program")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o755))?;
    }
    let v = sh(Command::new(&staged).arg("--version")).map_err(|e| {
        let _ = std::fs::remove_file(&staged);
        e.context("the new program doesn't run")
    })?;
    if v.trim_start_matches("voxely ").trim() != version {
        let _ = std::fs::remove_file(&staged);
        bail!("the release says {version} but its program says {v:?}");
    }
    swap_exe(exe, &staged)
}

#[cfg(unix)]
fn swap_exe(exe: &Path, staged: &Path) -> Result<()> {
    // A rename over a running program is fine: it keeps the old file open.
    std::fs::rename(staged, exe).context("putting the new program in place")
}

#[cfg(windows)]
fn swap_exe(exe: &Path, staged: &Path) -> Result<()> {
    // A running .exe can be renamed but not replaced.
    let old = old_exe(exe);
    let _ = std::fs::remove_file(&old);
    std::fs::rename(exe, &old).context("moving the running program aside")?;
    if let Err(e) = std::fs::rename(staged, exe) {
        let _ = std::fs::rename(&old, exe);
        return Err(anyhow!(e).context("putting the new program in place"));
    }
    Ok(())
}

fn old_exe(exe: &Path) -> PathBuf {
    exe.with_file_name("voxely.old.exe")
}

/// The program an earlier update moved aside (Windows), now free to go.
fn remove_leftovers() {
    if cfg!(windows)
        && let Some(exe) = installed()
    {
        let _ = std::fs::remove_file(old_exe(&exe));
    }
}

/// The first path under `dir` (two levels deep) that `ok` accepts.
fn find(dir: &Path, ok: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    let entries = |d: &Path| std::fs::read_dir(d).into_iter().flatten().flatten().map(|e| e.path()).collect::<Vec<_>>();
    let top = entries(dir);
    top.iter().find(|p| ok(p)).cloned().or_else(|| top.iter().filter(|p| p.is_dir()).flat_map(|d| entries(d)).find(|p| ok(p)))
}

// ---------------------------------------------------------------- ui

/// Status bar: a restart button once an update is in, or a link when this
/// copy can't update itself. Returns true when the user asked to restart.
pub fn badge(ui: &mut egui::Ui) -> bool {
    match state() {
        State::Ready(v) => {
            let text = egui::RichText::new(format!("Restart for voxely {v}")).color(egui::Color32::from_rgb(120, 200, 120));
            return ui.button(text).on_hover_text("The update is installed; restart to use it (you're asked to save first)").clicked();
        }
        State::Available(v, why) => {
            if ui.link(format!("voxely {v} is out")).on_hover_text(&why).clicked() {
                open_url(&release_page());
            }
        }
        State::Checking | State::Downloading(_) => {
            // Notice when the thread finishes.
            ui.ctx().request_repaint_after(Duration::from_secs(2));
        }
        _ => {}
    }
    false
}

/// One line for Preferences.
pub fn describe() -> String {
    match state() {
        State::Idle if !RELEASE_BUILD => format!("voxely {VERSION} (development build: doesn't update itself)"),
        State::Idle => format!("voxely {VERSION}"),
        State::Checking => format!("voxely {VERSION}: checking for updates..."),
        State::UpToDate => format!("voxely {VERSION} is up to date"),
        State::Downloading(v) => format!("voxely {VERSION}: downloading {v}..."),
        State::Ready(v) => format!("voxely {v} is installed; restart to use it"),
        State::Available(v, why) => format!("voxely {v} is out: {why}"),
        State::Failed(e) => format!("voxely {VERSION}: couldn't update ({e})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn versions_compare_by_number() {
        assert!(newer("0.2.0", "0.1.9"));
        assert!(newer("1.2.10", "1.2.9"));
        assert!(newer("1.0", "0.9.9"));
        assert!(!newer("0.1.0", "0.1.0"));
        assert!(!newer("0.1.0", "0.2.0"));
    }

    #[test]
    fn reads_a_release_for_each_platform() {
        let v = json!({ "tag_name": "v0.2.0", "assets": [
            { "name": "voxely-0.2.0-macos.dmg", "browser_download_url": "https://x/mac.dmg", "digest": "sha256:ABC123" },
            { "name": "voxely-0.2.0-linux-x86_64.tar.gz", "browser_download_url": "https://x/linux.tar.gz" },
            { "name": "voxely-0.2.0-windows-x86_64.zip", "browser_download_url": "https://x/win.zip" },
        ]});
        let mac = parse_release(&v, "-macos.dmg").unwrap();
        assert_eq!(mac, Release { version: "0.2.0".into(), download: Some("https://x/mac.dmg".into()), sha256: Some("abc123".into()) });
        assert_eq!(parse_release(&v, "-windows-x86_64.zip").unwrap().download.as_deref(), Some("https://x/win.zip"));
        assert_eq!(parse_release(&v, "-linux-aarch64.tar.gz").unwrap().download, None);
        assert!(parse_release(&json!({ "assets": [] }), "-macos.dmg").is_err());
    }

    #[test]
    fn finds_the_bundle() {
        assert_eq!(bundle_of(Path::new("/Applications/voxely.app/Contents/MacOS/voxely")), Some(PathBuf::from("/Applications/voxely.app")));
        assert_eq!(bundle_of(Path::new("/Users/a/voxely/target/release/voxely")), None);
    }

    #[test]
    fn checksums_files() {
        let p = std::env::temp_dir().join(format!("voxely-sha-{}", std::process::id()));
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(sha256_file(&p).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        std::fs::remove_file(p).unwrap();
    }

    #[test]
    fn read_only_places_dont_update() {
        assert!(can_replace(Path::new("/Volumes/voxely/voxely.app")).is_err());
        assert!(can_replace(Path::new("/private/var/folders/x/AppTranslocation/y/voxely.app")).is_err());
        assert!(can_replace(&std::env::temp_dir().join("voxely")).is_ok());
    }
}
