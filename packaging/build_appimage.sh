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

# Setup environment
export LILIM_INSTALL="$HERE/usr/lib/lilim"
export PATH="$HERE/usr/bin:$PATH"
export LILIM_BRAIN_PORT=5005

echo "Starting Lilim AppImage environment..."

# 1. Start the Rust runtime in the background
# It will automatically spawn the Python brain.
# We tell it to use the system python if no venv is found, 
# or we can point it to a bundled venv.
"$HERE/usr/bin/lilim-runtime" --port 5005 &
RUNTIME_PID=$!

# Ensure cleanup on exit
cleanup() {
    echo "Shutting down Lilim components..."
    kill $RUNTIME_PID 2>/dev/null || true
    exit
}
trap cleanup SIGINT SIGTERM EXIT

# 2. Start the UI
# The UI will connect to the runtime at localhost:5005
"$HERE/usr/bin/lilim" "$@"
EOF

chmod +x "$APPDIR/AppRun"

# 3. Add Desktop File & Icon to AppDir root (required by AppImage)
cp "$APPDIR/usr/share/applications/lilim.desktop" "$APPDIR/"
if [ -f "$APPDIR/usr/share/pixmaps/lilim.png" ]; then
    cp "$APPDIR/usr/share/pixmaps/lilim.png" "$APPDIR/"
else
    # Fallback to a generic icon if not found
    touch "$APPDIR/lilim.png"
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
