#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
DEB_ROOT="$ROOT_DIR/packaging/deb_root"
DEB_OUTPUT="$ROOT_DIR/dist"
PACKAGE_VERSION="${LILIM_VERSION:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$ROOT_DIR/lilim_desktop/src-tauri/tauri.conf.json")}"
DEB_NAME="lilim_${PACKAGE_VERSION}_amd64.deb"

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
  "$DEB_ROOT/usr/share/icons/hicolor/128x128/apps" \
  "$DEB_ROOT/usr/share/icons/hicolor/scalable/apps" \
  "$DEB_ROOT/lib/systemd/system" || true

RUNTIME_BIN="${ROOT_DIR:-$(pwd)}/target/release/lilim-runtime"
TAURI_BIN=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --runtime)
      RUNTIME_BIN="$2"
      shift 2
      ;;
    --ui-binary)
      # Direct path to the Tauri binary (preferred over --tauri-bundle)
      TAURI_BIN="$2"
      shift 2
      ;;
    --tauri-bundle)
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

# Resolve Tauri binary if not set directly via --ui-binary
if [ -z "${TAURI_BIN:-}" ] || [ ! -f "${TAURI_BIN:-}" ]; then
  RELEASE_DIR="${ROOT_DIR}/lilim_desktop/src-tauri/target/release"
  for bin_name in "tauri-lilim-desktop" "tauri-applilim-desktop" "Lilim" "lilim" "tauri-app"; do
    candidate="$RELEASE_DIR/$bin_name"
    if [ -f "$candidate" ] && [ -x "$candidate" ]; then
      TAURI_BIN="$candidate"
      break
    fi
  done
  # ELF fallback: any executable binary in the release dir
  if [ -z "${TAURI_BIN:-}" ] || [ ! -f "${TAURI_BIN:-}" ]; then
    TAURI_BIN=$(find "${RELEASE_DIR}" -maxdepth 1 -type f -executable \
      ! -name "*.so" ! -name "*.d" ! -name ".cargo-lock" \
      -exec sh -c 'file "$1" | grep -q ELF && echo "$1"' _ {} \; 2>/dev/null | head -n 1)
  fi
  # CI artifact fallback: lilim-ui-executable in TAURI_BUNDLE_DIR
  if [ -z "${TAURI_BIN:-}" ] || [ ! -f "${TAURI_BIN:-}" ]; then
    if [ -n "${TAURI_BUNDLE_DIR:-}" ] && [ -f "${TAURI_BUNDLE_DIR}/lilim-ui-executable" ]; then
      TAURI_BIN="${TAURI_BUNDLE_DIR}/lilim-ui-executable"
    fi
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
if [ -n "$TAURI_BIN" ] && [ -f "$TAURI_BIN" ]; then
  cp "$TAURI_BIN" "$DEB_ROOT/usr/bin/lilim"
  chmod +x "$DEB_ROOT/usr/bin/lilim"
else
  echo "ERROR: Tauri UI binary not found. Searched in:"
  echo "  $ROOT_DIR/lilim_desktop/src-tauri/target/release/"
  echo "Run: cd lilim_desktop && npm run tauri build" >&2
  exit 1
fi

# Python Brain & Configuration
cp -r "$ROOT_DIR/lilim_core" "$DEB_ROOT/usr/lib/lilim/"
install -m 0755 "$ROOT_DIR/bin/lilim-cli" "$DEB_ROOT/usr/bin/lilim-cli"
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
  echo "Bundling Phi-3.5-mini model from $MODEL_DIR..."
  MODEL_DEST="$DEB_ROOT/usr/lib/lilim/models/phi-3.5-mini-q4"
  mkdir -p "$MODEL_DEST"
  install -m 0644 "$MODEL_DIR/Phi-3.5-mini-instruct-Q4_K_M.gguf" "$MODEL_DEST/"
  install -m 0644 "$MODEL_DIR/tokenizer.json" "$MODEL_DEST/"
  if [ -f "$MODEL_DIR/tokenizer_config.json" ]; then
    install -m 0644 "$MODEL_DIR/tokenizer_config.json" "$MODEL_DEST/"
  fi
fi

## Desktop file & Icon
ICON_SRC="$ROOT_DIR/lilim_desktop/src-tauri/icons/128x128.png"
if [ ! -s "$ICON_SRC" ]; then
  echo "ERROR: Required Lilim launcher icon is missing: $ICON_SRC" >&2
  exit 1
fi
install -m 0644 "$ICON_SRC" "$DEB_ROOT/usr/share/pixmaps/lilim.png"
install -m 0644 "$ICON_SRC" "$DEB_ROOT/usr/share/icons/hicolor/128x128/apps/lilim.png"
if [ -s "$ROOT_DIR/assets/lilim-col.svg" ]; then
  install -m 0644 "$ROOT_DIR/assets/lilim-col.svg" \
    "$DEB_ROOT/usr/share/icons/hicolor/scalable/apps/lilim.svg"
fi
echo "Installed Lilim launcher icon from $ICON_SRC"

cat > "$DEB_ROOT/usr/share/applications/lilim.desktop" <<'DES'
[Desktop Entry]
Version=1.0
Type=Application
Name=Lilim
Comment=AI Assistant for Lilith Linux
Exec=/usr/bin/lilim
Icon=lilim
Categories=Utility;
Terminal=false
StartupWMClass=Lilim
StartupNotify=true
DES

# Debian control file — libwebkit2gtk-4.0-37 does not exist on Ubuntu 26.04;
# use 4.1-0 or libwebkitgtk-6.0-4 instead.
cat > "$DEB_ROOT/DEBIAN/control" <<CTRL
Package: lilim
Version: $PACKAGE_VERSION
Section: base
Priority: optional
Architecture: amd64
Maintainer: BlancoBAM <blancobam@protonmail.com>
Depends: python3, python3-venv, systemd, libwebkit2gtk-4.1-0 | libwebkitgtk-6.0-4
Description: Lilim AI Assistant for Lilith Linux
 Rust runtime, Python agent service, Tauri desktop UI, and Phi-3.5-mini local inference.
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
TARGET_USER="${TARGET_USER:-$(logname 2>/dev/null || echo lilith)}"
echo "[lilim] Installing for user: $TARGET_USER"

# Patch the service file with the real username
SERVICE_FILE="/lib/systemd/system/lilith-ai.service"
sed -i "s|LILIM_USER_PLACEHOLDER|${TARGET_USER}|g" "$SERVICE_FILE"
sed -i "s|^WorkingDirectory=.*|WorkingDirectory=/home/${TARGET_USER}|g" "$SERVICE_FILE"
sed -i "s|^Environment=HOME=.*|Environment=HOME=/home/${TARGET_USER}|g" "$SERVICE_FILE"
sed -i "s|ReadWritePaths=.*|ReadWritePaths=/home/${TARGET_USER} /var/log/lilim /tmp /usr/lib/lilim/venv|g" "$SERVICE_FILE"

echo "[lilim] Creating Python virtual environment..."
python3 -m venv /usr/lib/lilim/venv
/usr/lib/lilim/venv/bin/pip install --quiet fastapi uvicorn litellm apscheduler pyyaml httpx "beautifulsoup4>=4.12" "mcp>=1.19,<2"

mkdir -p /var/log/lilim
chown -R "${TARGET_USER}:${TARGET_USER}" /var/log/lilim

mkdir -p "/home/${TARGET_USER}/.local/share/lilim"
chown -R "${TARGET_USER}:${TARGET_USER}" "/home/${TARGET_USER}/.local/share/lilim"

# Ensure the desktop launcher and icon are discoverable immediately after install.
test -s /usr/share/applications/lilim.desktop
test -s /usr/share/icons/hicolor/128x128/apps/lilim.png

# Allow user to write to the venv (for pip updates)
chown -R "${TARGET_USER}:${TARGET_USER}" /usr/lib/lilim/venv 2>/dev/null || true

systemctl daemon-reload
systemctl enable lilith-ai.service || true
systemctl restart lilith-ai.service || true

# Refresh icon cache so the launcher shows the Lilim icon immediately
gtk-update-icon-cache -f /usr/share/icons/hicolor/ 2>/dev/null || true
update-desktop-database /usr/share/applications/ 2>/dev/null || true

echo "[lilim] Installation complete. Service running as: $TARGET_USER"
POSTINST
chmod +x "$DEB_ROOT/DEBIAN/postinst"

dpkg-deb --root-owner-group --build "$DEB_ROOT" "$DEB_OUTPUT/$DEB_NAME"
cp "$DEB_OUTPUT/$DEB_NAME" "$DEB_OUTPUT/lilim_amd64.deb"
echo "DEB built at $DEB_OUTPUT/$DEB_NAME"
