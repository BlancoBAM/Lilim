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

RUNTIME_BIN="${ROOT_DIR:-$(pwd)}/target/release/lilim-runtime"
TAURI_BIN="${ROOT_DIR:-$(pwd)}/lilim_desktop/src-tauri/target/release/bundle/appimage/lilim_0.1.0_amd64.AppImage" # fallback

while [[ $# -gt 0 ]]; do
  case $1 in
    --runtime)
      RUNTIME_BIN="$2"
      shift 2
      ;;
    --tauri-bundle)
      # In the CI we get the bundle dir, the actual binary inside should be copied, or we just grab the executable.
      # Wait, the workflow uploads `lilim_desktop/src-tauri/target/release/bundle/`
      # but we actually just need the raw binary `tauri-app` or whatever it's called.
      # Let's just point to the directory, and we'll extract the binary.
      TAURI_BUNDLE_DIR="$2"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      shift
      ;;
  esac
done

  # The CI pipeline uploads the binary as 'lilim-ui-executable'
  TAURI_BIN=$(find "$TAURI_BUNDLE_DIR" -type f -name "lilim-ui-executable" | head -n 1)
  if [ -z "$TAURI_BIN" ]; then
    # Fallback to case-insensitive find
    TAURI_BIN=$(find "$TAURI_BUNDLE_DIR" -type f -iname "lilim" | head -n 1)
  fi
  if [ -z "$TAURI_BIN" ]; then
    # Fallback to local dev path (case insensitive)
    TAURI_BIN=$(find "${ROOT_DIR:-$(pwd)}/lilim_desktop/src-tauri/target/release" -maxdepth 1 -type f -iname "lilim" | head -n 1)
  fi
  # Default local dev path
  TAURI_BIN="${ROOT_DIR:-$(pwd)}/lilim_desktop/src-tauri/target/release/lilim"
fi

## Build real runtime binary into the package (require it to be present)
if [ -x "$RUNTIME_BIN" ]; then
  mkdir -p "$DEB_ROOT/usr/bin"
  cp "$RUNTIME_BIN" "$DEB_ROOT/usr/bin/lilim-runtime"
  chmod +x "$DEB_ROOT/usr/bin/lilim-runtime"
else
  echo "ERROR: lilim-runtime not found at $RUNTIME_BIN; please build the Rust runtime before packaging" >&2
  exit 1
fi

# Desktop UI (Tauri binary)
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

# Model (if provided via --model-dir)
if [ -n "${MODEL_DIR:-}" ] && [ -d "$MODEL_DIR" ]; then
  echo "Bundling Phi-2 model from $MODEL_DIR..."
  mkdir -p "$DEB_ROOT/usr/lib/lilim/models/phi-2-q4"
  cp -r "$MODEL_DIR/"* "$DEB_ROOT/usr/lib/lilim/models/phi-2-q4/"
fi

## Desktop file & Icon
mkdir -p "$DEB_ROOT/usr/share/applications"
mkdir -p "$DEB_ROOT/usr/share/pixmaps"

if [ -f "/home/aegon/Downloads/lilim-icon.png" ]; then
    cp "/home/aegon/Downloads/lilim-icon.png" "$DEB_ROOT/usr/share/pixmaps/lilim.png"
fi

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
