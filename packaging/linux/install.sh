#!/bin/sh
# Install voxely for this user: the program in ~/.local/bin, plus an app menu
# entry and icon. voxely updates itself there. Run ./install.sh --uninstall to
# remove it (your projects in ~/Documents/voxely stay).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
bin="$HOME/.local/bin"
apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
icons="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/512x512/apps"

if [ "${1:-}" = "--uninstall" ]; then
  rm -f "$bin/voxely" "$apps/voxely.desktop" "$icons/voxely.png"
  echo "voxely removed"
  exit 0
fi

mkdir -p "$bin" "$apps" "$icons"
cp "$here/voxely" "$bin/voxely"
chmod +x "$bin/voxely"
cp "$here/voxely.png" "$icons/voxely.png"
sed "s|^Exec=voxely|Exec=$bin/voxely|" "$here/voxely.desktop" > "$apps/voxely.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database "$apps" 2>/dev/null || true
echo "voxely installed to $bin/voxely and added to your app menu"
case ":$PATH:" in
  *":$bin:"*) ;;
  *) echo "add $bin to your PATH to run voxely from a terminal" ;;
esac
