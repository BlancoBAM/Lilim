#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
DEB_ROOT="$ROOT_DIR/packaging/deb_root"
DEB_OUTPUT="$ROOT_DIR/dist"
DEB_NAME="lilim_0.1.0_amd64.deb"

echo "Building Debian package into $DEB_OUTPUT/$DEB_NAME"
mkdir -p "$DEB_ROOT" "$DEB_OUTPUT"

set -x
# Ensure a clean Debian layout exists
rm -rf "$DEB_ROOT"/ || true
mkdir -p \
  "$DEB_ROOT/usr/local/bin" \
  "$DEB_ROOT/DEBIAN" \
  "$DEB_ROOT/etc/lilith" \
  "$DEB_ROOT/usr/lib/lilim" \
  "$DEB_ROOT/usr/bin" \
  "$DEB_ROOT/usr/share/applications" \
  "$DEB_ROOT/usr/share/pixmaps" \
  "$DEB_ROOT/lib/systemd/system" || true

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
  TAURI_BIN_FOUND=$(find "${TAURI_BUNDLE_DIR:-.}" -type f -name "lilim-ui-executable" | head -n 1)
  if [ -n "$TAURI_BIN_FOUND" ]; then
    TAURI_BIN="$TAURI_BIN_FOUND"
  elif [ -z "${TAURI_BIN:-}" ] || [ ! -x "$TAURI_BIN" ]; then
    # Fallback to case-insensitive find in local release dir
    TAURI_BIN=$(find "${ROOT_DIR:-$(pwd)}/lilim_desktop/src-tauri/target/release" -maxdepth 1 -type f -iname "lilim" | head -n 1)
    if [ -z "$TAURI_BIN" ]; then
      # Absolute default
      TAURI_BIN="${ROOT_DIR:-$(pwd)}/lilim_desktop/src-tauri/target/release/lilim"
    fi
  fi

## Build real runtime binary into the package (require it to be present)
if [ -f "$RUNTIME_BIN" ]; then
  mkdir -p "$DEB_ROOT/usr/bin"
  cp "$RUNTIME_BIN" "$DEB_ROOT/usr/bin/lilim-runtime"
  chmod +x "$DEB_ROOT/usr/bin/lilim-runtime"
else
  echo "ERROR: lilim-runtime not found at $RUNTIME_BIN; please build the Rust runtime before packaging" >&2
  exit 1
fi

# Desktop UI (Tauri binary)
if [ -f "$TAURI_BIN" ]; then
  cp "$TAURI_BIN" "$DEB_ROOT/usr/bin/lilim"
  chmod +x "$DEB_ROOT/usr/bin/lilim"
else
  echo "WARNING: Tauri binary not found at $TAURI_BIN. Skipping UI." >&2
fi

# Python Brain & Configuration
cp -r "$ROOT_DIR/lilim_core" "$DEB_ROOT/usr/lib/lilim/"
mkdir -p "$DEB_ROOT/etc/lilith"
cp -r "$ROOT_DIR/config/"* "$DEB_ROOT/etc/lilith/"

# Systemd Service
cp "$ROOT_DIR/systemd/system/lilith-ai.service" "$DEB_ROOT/lib/systemd/system/"

# User-agnostic runtime wrapper (resolves desktop user at service start)
cat > "$DEB_ROOT/usr/lib/lilim/run-lilim.sh" << 'WRAPPER'
#!/usr/bin/env bash
set -e
TARGET_USER=""
for candidate in $(ls /home/ 2>/dev/null); do
    if id "$candidate" &>/dev/null && [ "$candidate" != "root" ]; then
        TARGET_USER="$candidate"
        break
    fi
done
TARGET_USER="${TARGET_USER:-lilith}"
LILIM_HOME="/home/${TARGET_USER}/.local/share/lilim"
mkdir -p "$LILIM_HOME"
chown "${TARGET_USER}:${TARGET_USER}" "$LILIM_HOME" 2>/dev/null || true
export HOME="/home/${TARGET_USER}"
exec sudo -u "${TARGET_USER}" \
    --preserve-env=LILIM_INSTALL,LILIM_VENV,PYTHONUNBUFFERED,RUST_LOG,HOME \
    /usr/bin/lilim-runtime
WRAPPER
chmod +x "$DEB_ROOT/usr/lib/lilim/run-lilim.sh"

# Model (if provided via --model-dir)
if [ -n "${MODEL_DIR:-}" ] && [ -d "$MODEL_DIR" ]; then
  echo "Bundling Phi-2 model from $MODEL_DIR..."
  mkdir -p "$DEB_ROOT/usr/lib/lilim/models/phi-2-q4"
  cp -r "$MODEL_DIR/"* "$DEB_ROOT/usr/lib/lilim/models/phi-2-q4/"
fi

## Desktop file & Icon
# Search repo assets rather than a hardcoded user path
ICON_SRC=""
for icon_path in \
    "$ROOT_DIR/assets/lilim-icon.png" \
    "$ROOT_DIR/assets/icon.png" \
    "$(find "$ROOT_DIR" -maxdepth 3 -name "lilim-icon.png" 2>/dev/null | head -1)"; do
    if [ -n "$icon_path" ] && [ -f "$icon_path" ]; then
        ICON_SRC="$icon_path"
        break
    fi
done
if [ -n "$ICON_SRC" ]; then
    cp "$ICON_SRC" "$DEB_ROOT/usr/share/pixmaps/lilim.png"
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

# Debian control file — libwebkit2gtk-4.0-37 does not exist on Ubuntu 26.04;
# use 4.1-0 or libwebkitgtk-6.0-4 instead.
cat > "$DEB_ROOT/DEBIAN/control" << 'CTRL'
Package: lilim
Version: 0.1.0
Section: base
Priority: optional
Architecture: amd64
Maintainer: BlancoBAM <blancobam@protonmail.com>
Depends: python3, python3-venv, systemd, libwebkit2gtk-4.1-0 | libwebkitgtk-6.0-4
Description: Lilim AI Assistant for Lilith Linux
 Production-ready runtime: Rust backend proxy, Python AI brain,
 and Tauri desktop UI. Includes embedded Phi-2 local inference model.
CTRL

# postinst — dynamically detects the primary desktop user; no hardcoded name
cat > "$DEB_ROOT/DEBIAN/postinst" << 'POSTINST'
#!/usr/bin/env bash
set -e
echo "[lilim] Setting up Lilim AI Assistant..."

# Determine primary user dynamically
TARGET_USER=""
for candidate in $(ls /home/ 2>/dev/null); do
    if id "$candidate" &>/dev/null && [ "$candidate" != "root" ]; then
        TARGET_USER="$candidate"
        break
    fi
done
TARGET_USER="${TARGET_USER:-lilith}"
echo "[lilim] Installing for user: $TARGET_USER"

echo "[lilim] Creating Python virtual environment..."
python3 -m venv /usr/lib/lilim/venv
/usr/lib/lilim/venv/bin/pip install --quiet fastapi uvicorn litellm apscheduler

mkdir -p /var/log/lilim
chown -R "${TARGET_USER}:${TARGET_USER}" /var/log/lilim

mkdir -p "/home/${TARGET_USER}/.local/share/lilim"
chown -R "${TARGET_USER}:${TARGET_USER}" "/home/${TARGET_USER}/.local/share/lilim"

systemctl daemon-reload
systemctl enable lilith-ai.service || true
systemctl restart lilith-ai.service || true
echo "[lilim] Installation complete."
POSTINST
chmod +x "$DEB_ROOT/DEBIAN/postinst"

dpkg-deb --root-owner-group --build "$DEB_ROOT" "$DEB_OUTPUT/$DEB_NAME"
echo "DEB built at $DEB_OUTPUT/$DEB_NAME"
