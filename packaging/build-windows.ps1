# Build dist\voxely-<version>-windows-<arch>.zip: a voxely folder with
# voxely.exe and the docs. Unzip it anywhere you can write to (not Program
# Files) and voxely updates itself there.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$version = (Select-String -Path Cargo.toml -Pattern '^version = "(.*)"' | Select-Object -First 1).Matches[0].Groups[1].Value
$arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "aarch64" } else { "x86_64" }
$name = "voxely-$version-windows-$arch"
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
$stage = "dist\voxely"
New-Item -ItemType Directory -Force $stage | Out-Null

# Marks a release build: only these update themselves (src/update.rs).
$env:VOXELY_RELEASE = "1"
cargo build --release --locked
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { "target" }
Copy-Item (Join-Path $target "release\voxely.exe"), README.md, LICENSE $stage
Compress-Archive -Path $stage -DestinationPath "dist\$name.zip"
Write-Output "dist\$name.zip"
