//! Updates from GitHub releases, for the macOS app (`voxely.app`).
//!
//! A background thread asks GitHub for the latest release at startup and
//! once a day. When it is newer, it downloads the release's `.dmg`, checks
//! it (GitHub's SHA-256 for the file, the app's code signature and version),
//! and swaps the new app in for the running one. The running copy keeps
//! working; the status bar offers a restart into the new version, and the
//! next launch gets it anyway.
//!
//! Only an app bundle updates itself (not `cargo run` or a bare binary), and
//! only where it can replace itself: not when run from the disk image or from
//! macOS's read-only "translocated" copy of a downloaded app. Then the status
//! bar links to the release instead. Network and disk-image work go through
//! macOS's own `curl`, `hdiutil`, `ditto` and `codesign`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde_json::Value;

pub const REPO: &str = "letsdogoodstuff/voxely";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// How often a running app checks again.
const EVERY: Duration = Duration::from_secs(24 * 60 * 60);

#[derive(Clone, Debug, PartialEq)]
pub enum State {
    /// Not checked (or checking is off, or this isn't an app bundle).
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
    if !on || std::env::var_os("VOXELY_NO_UPDATE").is_some() || bundle().is_none() || STARTED.swap(true, Ordering::Relaxed) {
        return;
    }
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
    if bundle().is_none() {
        set(State::Failed("only the voxely app updates itself; this is a development build".into()));
        return;
    }
    std::thread::spawn(check_once);
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
    if let Some(app) = bundle() {
        // -n: a new instance even while this one is still exiting.
        let _ = Command::new("/usr/bin/open").arg("-n").arg(&app).spawn();
    }
}

/// The release's page, for a copy that can't update itself.
pub fn release_page() -> String {
    format!("https://github.com/{REPO}/releases/latest")
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
    let r = run();
    if let Err(e) = r {
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
    let app = bundle().context("not running from voxely.app")?;
    if let Err(why) = can_replace(&app) {
        set(State::Available(rel.version, why));
        return Ok(());
    }
    set(State::Downloading(rel.version.clone()));
    install(&rel, &app)?;
    set(State::Ready(rel.version));
    Ok(())
}

// ---------------------------------------------------------------- releases

#[derive(Debug, PartialEq)]
struct Release {
    version: String,
    dmg: String,
    /// GitHub's SHA-256 of the file, hex.
    sha256: Option<String>,
}

fn latest() -> Result<Release> {
    // VOXELY_UPDATE_URL: another release JSON (any URL curl takes), for testing.
    let url = std::env::var("VOXELY_UPDATE_URL").unwrap_or_else(|_| format!("https://api.github.com/repos/{REPO}/releases/latest"));
    let out = Command::new("/usr/bin/curl")
        .args(["-fsSL", "--max-time", "30", "-H", "Accept: application/vnd.github+json"])
        .arg("-H")
        .arg(format!("User-Agent: voxely/{VERSION}"))
        .arg(&url)
        .output()
        .context("running curl")?;
    if !out.status.success() {
        bail!("couldn't reach GitHub: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    parse_release(&serde_json::from_slice(&out.stdout).context("GitHub's reply isn't JSON")?)
}

/// The version and `.dmg` asset of a GitHub release (API JSON).
fn parse_release(v: &Value) -> Result<Release> {
    let tag = v["tag_name"].as_str().context("release has no tag")?;
    let version = tag.trim_start_matches('v').to_string();
    let asset = v["assets"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|a| a["name"].as_str().is_some_and(|n| n.ends_with(".dmg")))
        .with_context(|| format!("release {tag} has no .dmg"))?;
    let dmg = asset["browser_download_url"].as_str().context("asset has no download URL")?.to_string();
    let sha256 = asset["digest"].as_str().and_then(|d| d.strip_prefix("sha256:")).map(str::to_lowercase);
    Ok(Release { version, dmg, sha256 })
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

/// The `.app` this binary runs from.
pub fn bundle() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?.canonicalize().ok()?;
    bundle_of(&exe)
}

fn bundle_of(exe: &Path) -> Option<PathBuf> {
    let macos = exe.parent()?;
    let contents = macos.parent()?;
    let app = contents.parent()?;
    (macos.file_name()? == "MacOS" && contents.file_name()? == "Contents" && app.extension()? == "app").then(|| app.to_path_buf())
}

/// Why `app` can't replace itself, if it can't.
fn can_replace(app: &Path) -> std::result::Result<(), String> {
    let s = app.to_string_lossy();
    if s.contains("/AppTranslocation/") {
        return Err("move voxely to Applications (drag it out of Downloads) so it can update itself".into());
    }
    if s.starts_with("/Volumes/") {
        return Err("you're running voxely from its disk image; drag it to Applications so it can update itself".into());
    }
    let dir = app.parent().ok_or("the app has no folder")?;
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
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("/tmp"), PathBuf::from);
    home.join("Library/Caches/voxely/update")
}

fn sh(cmd: &mut Command) -> Result<String> {
    let out = cmd.output().with_context(|| format!("running {:?}", cmd.get_program()))?;
    if !out.status.success() {
        bail!("{:?} failed: {}", cmd.get_program(), String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn install(rel: &Release, app: &Path) -> Result<()> {
    let dir = cache_dir();
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    let dmg = dir.join("voxely.dmg");
    sh(Command::new("/usr/bin/curl").args(["-fsSL", "--max-time", "900", "-o"]).arg(&dmg).arg(&rel.dmg)).context("downloading the update")?;
    if let Some(want) = &rel.sha256 {
        let got = sh(Command::new("/usr/bin/shasum").args(["-a", "256"]).arg(&dmg))?;
        ensure!(got.split_whitespace().next() == Some(want.as_str()), "the download doesn't match GitHub's checksum");
    }
    let mnt = dir.join("mnt");
    std::fs::create_dir_all(&mnt)?;
    sh(Command::new("/usr/bin/hdiutil").args(["attach", "-nobrowse", "-readonly", "-noautoopen", "-mountpoint"]).arg(&mnt).arg(&dmg)).context("opening the disk image")?;
    let staged = app.with_file_name(format!(".voxely-{}.app", rel.version));
    let copied = stage(&mnt, &staged, &rel.version);
    let _ = sh(Command::new("/usr/bin/hdiutil").args(["detach", "-force"]).arg(&mnt));
    let _ = std::fs::remove_dir_all(&dir);
    if let Err(e) = copied {
        let _ = std::fs::remove_dir_all(&staged);
        return Err(e);
    }
    swap(app, &staged)
}

/// Copy the image's app next to the installed one and check it.
fn stage(mnt: &Path, staged: &Path, version: &str) -> Result<()> {
    let new = std::fs::read_dir(mnt)?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e == "app"))
        .context("the disk image has no app")?;
    let _ = std::fs::remove_dir_all(staged);
    sh(Command::new("/usr/bin/ditto").arg(&new).arg(staged)).context("copying the new app")?;
    let plist = staged.join("Contents/Info.plist");
    let v = sh(Command::new("/usr/bin/plutil").args(["-extract", "CFBundleShortVersionString", "raw"]).arg(&plist))?;
    ensure!(v == version, "the release says {version} but its app is {v}");
    sh(Command::new("/usr/bin/codesign").args(["--verify", "--deep", "--strict"]).arg(staged)).context("the new app's signature doesn't check out")?;
    Ok(())
}

/// Put `staged` where `app` is, keeping the old one until the new is in.
fn swap(app: &Path, staged: &Path) -> Result<()> {
    let old = app.with_file_name(".voxely-old.app");
    let _ = std::fs::remove_dir_all(&old);
    std::fs::rename(app, &old).context("moving the old app aside")?;
    if let Err(e) = std::fs::rename(staged, app) {
        let _ = std::fs::rename(&old, app);
        return Err(anyhow!(e).context("putting the new app in place"));
    }
    let _ = std::fs::remove_dir_all(&old);
    Ok(())
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
                let _ = Command::new("/usr/bin/open").arg(release_page()).spawn();
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
    fn reads_a_release() {
        let v = json!({ "tag_name": "v0.2.0", "assets": [
            { "name": "notes.txt", "browser_download_url": "x" },
            { "name": "voxely-0.2.0-macos.dmg", "browser_download_url": "https://github.com/a/b/voxely.dmg", "digest": "sha256:ABC123" },
        ]});
        assert_eq!(parse_release(&v).unwrap(), Release { version: "0.2.0".into(), dmg: "https://github.com/a/b/voxely.dmg".into(), sha256: Some("abc123".into()) });
        assert!(parse_release(&json!({ "tag_name": "v1", "assets": [] })).is_err());
    }

    #[test]
    fn finds_the_bundle() {
        assert_eq!(bundle_of(Path::new("/Applications/voxely.app/Contents/MacOS/voxely")), Some(PathBuf::from("/Applications/voxely.app")));
        assert_eq!(bundle_of(Path::new("/Users/a/voxely/target/release/voxely")), None);
    }

    #[test]
    fn swap_replaces_and_cleans_up() {
        let dir = std::env::temp_dir().join(format!("voxely-swap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (app, staged) = (dir.join("voxely.app"), dir.join(".voxely-9.app"));
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(app.join("v"), "old").unwrap();
        std::fs::create_dir_all(&staged).unwrap();
        std::fs::write(staged.join("v"), "new").unwrap();
        swap(&app, &staged).unwrap();
        assert_eq!(std::fs::read_to_string(app.join("v")).unwrap(), "new");
        assert!(!staged.exists() && !dir.join(".voxely-old.app").exists());
        assert!(can_replace(&app).is_ok());
        assert!(can_replace(Path::new("/Volumes/voxely/voxely.app")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
