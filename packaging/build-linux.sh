#!/bin/sh
# Build dist/voxely-<version>-linux-<arch>.tar.gz: the voxely program, an
# install.sh that adds it to the app menu, the icon and the docs.
# Needs a C compiler (QuickJS is built from C). Build on the oldest Linux you
# want to support: the program needs that glibc or newer (releases are built
# on Ubuntu 22.04).
set -eu
cd "$(dirname "$0")/.."

version=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
arch=$(uname -m)
name="voxely-$version-linux-$arch"
out=dist
rm -rf "$out"
mkdir -p "$out/$name"

# Marks a release build: only these update themselves (src/update.rs).
VOXELY_RELEASE=1 cargo build --release --locked
cp "${CARGO_TARGET_DIR:-target}/release/voxely" "$out/$name/voxely"
strip "$out/$name/voxely" 2>/dev/null || true
cp assets/icon.png "$out/$name/voxely.png"
cp packaging/linux/voxely.desktop packaging/linux/install.sh README.md LICENSE "$out/$name/"
chmod +x "$out/$name/voxely" "$out/$name/install.sh"
tar -C "$out" -czf "$out/$name.tar.gz" "$name"
echo "$out/$name.tar.gz"
