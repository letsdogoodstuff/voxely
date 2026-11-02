#!/bin/sh
# Rebuild the app icons from packaging/icon-source.png (needs Pillow):
# assets/icon.png (built into the binary), packaging/voxely.icns (the macOS app's)
# and packaging/voxely.ico (the Windows .exe's).
set -eu
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
python3 -I packaging/mac_icon.py packaging/icon-source.png "$tmp/1024.png" 1024
sips -Z 512 "$tmp/1024.png" --out assets/icon.png >/dev/null
set_dir="$tmp/voxely.iconset"
mkdir "$set_dir"
for s in 16 32 128 256 512; do
  sips -Z $s "$tmp/1024.png" --out "$set_dir/icon_${s}x${s}.png" >/dev/null
  sips -Z $((s * 2)) "$tmp/1024.png" --out "$set_dir/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$set_dir" -o packaging/voxely.icns
python3 -I -c 'import sys; from PIL import Image; Image.open(sys.argv[1]).save(sys.argv[2], sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])' "$tmp/1024.png" packaging/voxely.ico
rm -rf "$tmp"
echo "wrote assets/icon.png, packaging/voxely.icns and packaging/voxely.ico"
