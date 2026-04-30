#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
DEB_ROOT="$ROOT_DIR/packaging/deb_root"
DEB_OUTPUT="$ROOT_DIR/dist"
DEB_NAME="lilim-linux-component.deb"

echo "Building Debian package into $DEB_OUTPUT/$DEB_NAME"
mkdir -p "$DEB_ROOT" "$DEB_OUTPUT"

set -x
# Ensure a clean Debian layout exists
rm -rf "$DEB_ROOT"/ || true
mkdir -p "$DEB_ROOT/usr/local/bin" "$DEB_ROOT/DEBIAN" "$DEB_ROOT/etc/lilith" "$DEB_ROOT/usr/lib/lilim" || true

## Build real runtime binary into the package (require it to be present)
ZEROCLAW_BIN="${ROOT_DIR:-$(pwd)}/zeroclaw/target/release/zeroclaw"
if [ -x "$ZEROCLAW_BIN" ]; then
  mkdir -p "$DEB_ROOT/usr/bin"
  cp "$ZEROCLAW_BIN" "$DEB_ROOT/usr/bin/zeroclaw"
  chmod +x "$DEB_ROOT/usr/bin/zeroclaw"
else
  echo "ERROR: zeroclaw binary not found at $ZEROCLAW_BIN; please build zeroclaw before packaging" >&2
  exit 1
fi

# Desktop UI assets (if present)
if [ -d "$ROOT_DIR/lilim_desktop/dist" ]; then
  mkdir -p "$DEB_ROOT/usr/share/lilim-desktop"
  cp -r "$ROOT_DIR/lilim_desktop/dist/." "$DEB_ROOT/usr/share/lilim-desktop/"
fi
if [ -d "$ROOT_DIR/lilim_desktop/build" ]; then
  mkdir -p "$DEB_ROOT/usr/share/lilim-desktop"
  cp -r "$ROOT_DIR/lilim_desktop/build/." "$DEB_ROOT/usr/share/lilim-desktop/"
fi

## Desktop launcher and desktop file
if [ -d "$DEB_ROOT/usr/share/lilim-desktop" ]; then
  mkdir -p "$DEB_ROOT/usr/share/applications"
  cat > "$DEB_ROOT/usr/share/applications/lilim-desktop.desktop" <<'DES'
[Desktop Entry]
Name=Lilim Desktop Chat
Comment=Desktop chat UI for Lilim
Exec=/usr/bin/lilim-desktop
Icon=lilim
Type=Application
Categories=Utility;
DES
  cat > "$DEB_ROOT/usr/bin/lilim-desktop" <<'DESBIN'
#!/usr/bin/env bash
if [ -d "/usr/share/lilim-desktop" ]; then
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open http://127.0.0.1:8000/ || true
  else
    echo "Lilim Desktop UI available at /usr/share/lilim-desktop"
  fi
fi
DESBIN
  chmod +x "$DEB_ROOT/usr/bin/lilim-desktop"
fi

# Debian control file
cat > "$DEB_ROOT/DEBIAN/control" <<'CTRL'
Package: lilim
Version: 0.1.0
Section: base
Priority: optional
Architecture: amd64
Maintainer: Lilim Maintainers <maintainer@example.com>
Description: Lilim component for production-ready Open Interpreter-based AI assistant
 This package includes the production-ready Lilim runtime components and their launcher.
CTRL

dpkg-deb --root-owner-group --build "$DEB_ROOT" "$DEB_OUTPUT/$DEB_NAME"
echo "DEB built at $DEB_OUTPUT/$DEB_NAME"
