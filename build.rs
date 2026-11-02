// Windows: give voxely.exe its icon (packaging/voxely.ico) for Explorer and
// the taskbar.
fn main() {
    println!("cargo:rerun-if-changed=packaging/voxely.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("packaging/voxely.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=voxely.exe gets no icon: {e}");
        }
    }
}
