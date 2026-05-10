#!/usr/bin/env bash
# packaging/build_appimage.sh
# Creates a standalone AppImage for Lilim from a built .deb package.

set -euo pipefail

DEB_FILE="${1:-}"
if [[ -z "$DEB_FILE" || ! -f "$DEB_FILE" ]]; then
    echo "Usage: $0 <path-to-lilim.deb>"
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APPDIR="$ROOT_DIR/dist/lilim.AppDir"
OUTPUT_DIR="$ROOT_DIR/dist"

echo "Building AppImage from $DEB_FILE..."

# 1. Clean and prepare AppDir
rm -rf "$APPDIR"
mkdir -p "$APPDIR"
dpkg-deb -x "$DEB_FILE" "$APPDIR"

# 2. Create the AppRun script
# This script starts the background runtime and then the UI.
cat > "$APPDIR/AppRun" <<'EOF'
#!/usr/bin/env bash
HERE="$(dirname "$(readlink -f "${0}")")"
LOG_FILE="/tmp/lilim-appimage.log"

# Setup environment
export LILIM_INSTALL="$HERE/usr/lib/lilim"
export PATH="$HERE/usr/bin:$PATH"
export LILIM_BRAIN_PORT=8081
export RUST_LOG=info

echo "$(date) - Starting Lilim AppImage..." > "$LOG_FILE"

# 1. Start the Rust runtime in the background
# It will automatically spawn the Python brain.
"$HERE/usr/bin/lilim-runtime" --port 8080 >> "$LOG_FILE" 2>&1 &
RUNTIME_PID=$!

# Wait for backend to be ready (health check)
echo "Waiting for backend..." >> "$LOG_FILE"
for i in {1..30}; do
    if curl -s http://127.0.0.1:8080/health > /dev/null; then
        echo "Backend ready." >> "$LOG_FILE"
        break
    fi
    sleep 0.5
done

# Ensure cleanup on exit
cleanup() {
    echo "Shutting down Lilim components..." >> "$LOG_FILE"
    kill $RUNTIME_PID 2>/dev/null || true
    exit
}
trap cleanup SIGINT SIGTERM EXIT

# 2. Start the UI
# The UI will connect to the runtime at localhost:8080
"$HERE/usr/bin/lilim" "$@" >> "$LOG_FILE" 2>&1
EOF

chmod +x "$APPDIR/AppRun"

# 3. Add Desktop File & Icon to AppDir root (required by AppImage)
# We copy from the installed paths and fix them
DESKTOP_FILE="$APPDIR/usr/share/applications/lilim.desktop"
if [ -f "$DESKTOP_FILE" ]; then
    cp "$DESKTOP_FILE" "$APPDIR/lilim.desktop"
    # Fix Exec and Icon for AppImage
    sed -i 's|^Exec=.*|Exec=lilim|' "$APPDIR/lilim.desktop"
    sed -i 's|^Icon=.*|Icon=lilim|' "$APPDIR/lilim.desktop"
fi

# Create a symlink for the Exec entry
ln -sf AppRun "$APPDIR/lilim"

# Handle Icon
ICON_FILE="$APPDIR/usr/share/pixmaps/lilim.png"
if [ -f "$ICON_FILE" ]; then
    cp "$ICON_FILE" "$APPDIR/lilim.png"
else
    # Fallback: check if tauri built one in src-tauri
    TAURI_ICON="$ROOT_DIR/lilim_desktop/src-tauri/icons/128x128.png"
    if [ -f "$TAURI_ICON" ]; then
        cp "$TAURI_ICON" "$APPDIR/lilim.png"
    else
        touch "$APPDIR/lilim.png"
    fi
fi

# 4. Handle Python dependencies inside the AppImage
# We create a portable venv inside the AppDir
echo "Bundling Python virtual environment into AppDir..."
python3 -m venv "$APPDIR/usr/lib/lilim/venv"
"$APPDIR/usr/lib/lilim/venv/bin/pip" install --quiet fastapi uvicorn litellm apscheduler

# 5. Download appimagetool if not present
APPIMAGE_TOOL="$ROOT_DIR/packaging/appimagetool"
if [ ! -f "$APPIMAGE_TOOL" ]; then
    echo "Downloading appimagetool..."
    wget -q "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage" -O "$APPIMAGE_TOOL"
    chmod +x "$APPIMAGE_TOOL"
fi

# 6. Build the AppImage
echo "Running appimagetool..."
export ARCH=x86_64
"$APPIMAGE_TOOL" --appimage-extract-and-run "$APPDIR" "$OUTPUT_DIR/lilim_0.1.0_amd64.AppImage"

echo "=========================================="
echo "✅ AppImage created: $OUTPUT_DIR/lilim_0.1.0_amd64.AppImage"
echo "=========================================="
