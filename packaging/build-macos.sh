#!/bin/sh
# Build voxely.app and dist/voxely-<version>-macos.dmg.
#
#   packaging/build-macos.sh            universal app (Apple Silicon + Intel)
#   packaging/build-macos.sh --native   this Mac's architecture only (quicker)
#
# Signing: ad-hoc by default. With a Developer ID set, it signs for
# distribution (hardened runtime) and, with notary credentials, notarizes
# and staples the disk image so it opens without Gatekeeper warnings:
#   VOXELY_SIGN_IDENTITY  "Developer ID Application: Name (TEAMID)"
#   VOXELY_NOTARY_APPLE_ID, VOXELY_NOTARY_TEAM_ID, VOXELY_NOTARY_PASSWORD
#     (an app-specific password for that Apple ID)
set -eu
cd "$(dirname "$0")/.."

native=false
[ "${1:-}" = "--native" ] && native=true

version=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
identity=${VOXELY_SIGN_IDENTITY:--}
out=dist
app="$out/voxely.app"
dmg="$out/voxely-$version-macos.dmg"
rm -rf "$out"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"

# Old enough for every Mac that runs Metal 2 through wgpu.
export MACOSX_DEPLOYMENT_TARGET=11.0
# Marks a release build: only these update themselves (src/update.rs).
export VOXELY_RELEASE=1
if $native; then
  cargo build --release --locked
  cp "${CARGO_TARGET_DIR:-target}/release/voxely" "$app/Contents/MacOS/voxely"
else
  for t in aarch64-apple-darwin x86_64-apple-darwin; do
    rustup target add "$t" >/dev/null 2>&1 || true
    cargo build --release --locked --target "$t"
  done
  lipo -create -output "$app/Contents/MacOS/voxely" \
    "${CARGO_TARGET_DIR:-target}/aarch64-apple-darwin/release/voxely" "${CARGO_TARGET_DIR:-target}/x86_64-apple-darwin/release/voxely"
fi

cp packaging/voxely.icns "$app/Contents/Resources/voxely.icns"
cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>voxely</string>
  <key>CFBundleDisplayName</key><string>voxely</string>
  <key>CFBundleIdentifier</key><string>com.letsdogoodstuff.voxely</string>
  <key>CFBundleExecutable</key><string>voxely</string>
  <key>CFBundleIconFile</key><string>voxely</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$version</string>
  <key>CFBundleVersion</key><string>$version</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSSupportsAutomaticGraphicsSwitching</key><true/>
</dict>
</plist>
PLIST
plutil -lint "$app/Contents/Info.plist" >/dev/null

if [ "$identity" = "-" ]; then
  codesign --force --sign - "$app"
else
  codesign --force --options runtime --timestamp --sign "$identity" "$app"
fi
codesign --verify --deep --strict "$app"

# The disk image: the app and a link to Applications to drag it onto.
stage=$(mktemp -d)
ditto "$app" "$stage/voxely.app"
ln -s /Applications "$stage/Applications"
hdiutil create -volname "voxely $version" -srcfolder "$stage" -ov -format UDZO "$dmg" >/dev/null
rm -rf "$stage"

if [ "$identity" != "-" ]; then
  codesign --force --timestamp --sign "$identity" "$dmg"
  if [ -n "${VOXELY_NOTARY_APPLE_ID:-}" ]; then
    xcrun notarytool submit "$dmg" --wait \
      --apple-id "$VOXELY_NOTARY_APPLE_ID" --team-id "$VOXELY_NOTARY_TEAM_ID" --password "$VOXELY_NOTARY_PASSWORD"
    xcrun stapler staple "$dmg"
  fi
fi

echo "$dmg"
