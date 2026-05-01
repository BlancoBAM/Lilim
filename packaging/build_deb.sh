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
RUNTIME_BIN="${ROOT_DIR:-$(pwd)}/crates/lilim-runtime/target/release/lilim-runtime"
if [ -x "$RUNTIME_BIN" ]; then
  mkdir -p "$DEB_ROOT/usr/bin"
  cp "$RUNTIME_BIN" "$DEB_ROOT/usr/bin/lilim-runtime"
  chmod +x "$DEB_ROOT/usr/bin/lilim-runtime"
else
  echo "ERROR: lilim-runtime not found at $RUNTIME_BIN; please build the Rust runtime before packaging" >&2
  exit 1
fi

# Desktop UI (Tauri binary)
TAURI_BIN="$ROOT_DIR/lilim_desktop/src-tauri/target/release/tauri-applilim-desktop"
if [ -x "$TAURI_BIN" ]; then
  cp "$TAURI_BIN" "$DEB_ROOT/usr/bin/lilim"
else
  echo "WARNING: Tauri binary not found at $TAURI_BIN. Skipping UI." >&2
fi

# Python Brain & Configuration
cp -r "$ROOT_DIR/lilim_core" "$DEB_ROOT/usr/lib/lilim/"
mkdir -p "$DEB_ROOT/etc/lilith"
cp -r "$ROOT_DIR/config/"* "$DEB_ROOT/etc/lilith/"

# Systemd Service
mkdir -p "$DEB_ROOT/lib/systemd/system"
cp "$ROOT_DIR/systemd/system/lilith-ai.service" "$DEB_ROOT/lib/systemd/system/"

## Desktop file
mkdir -p "$DEB_ROOT/usr/share/applications"
cat > "$DEB_ROOT/usr/share/applications/lilim.desktop" <<'DES'
[Desktop Entry]
Name=Lilim Assistant
Comment=AI Assistant for Lilith Linux
Exec=/usr/bin/lilim
Icon=lilim
Type=Application
Categories=Utility;
DES

# Debian control file
cat > "$DEB_ROOT/DEBIAN/control" <<'CTRL'
Package: lilim
Version: 0.1.0
Section: base
Priority: optional
Architecture: amd64
Maintainer: Lilim Maintainers <maintainer@example.com>
Depends: python3, python3-venv, systemd, libwebkit2gtk-4.0-37 | libwebkit2gtk-4.1-0
Description: Lilim component for production-ready AI assistant
 This package includes the production-ready Lilim runtime components, the Python brain, and the Tauri desktop UI.
CTRL

cat > "$DEB_ROOT/DEBIAN/postinst" <<'POSTINST'
#!/usr/bin/env bash
set -e
echo "Creating python venv for Lilim..."
python3 -m venv /usr/lib/lilim/venv
/usr/lib/lilim/venv/bin/pip install fastapi uvicorn litellm apscheduler
mkdir -p /var/log/lilim
chown -R aegon:aegon /var/log/lilim
mkdir -p /home/aegon/.local/share/lilim
chown -R aegon:aegon /home/aegon/.local/share/lilim
systemctl daemon-reload
systemctl enable lilith-ai.service
systemctl restart lilith-ai.service
POSTINST
chmod +x "$DEB_ROOT/DEBIAN/postinst"

dpkg-deb --root-owner-group --build "$DEB_ROOT" "$DEB_OUTPUT/$DEB_NAME"
echo "DEB built at $DEB_OUTPUT/$DEB_NAME"
