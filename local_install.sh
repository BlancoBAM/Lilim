#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "Lilim Local Development Installation"
echo "=========================================="
echo

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKIP_UI=false

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --skip-ui) SKIP_UI=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# 1. Build the Rust runtime (from workspace root)
echo "[1/5] Building Rust runtime with Candle inference..."
cargo build --release --manifest-path="$ROOT_DIR/Cargo.toml"
echo "      ✓ Runtime built"

# 2. Build the Tauri Desktop UI (unless --skip-ui)
if [ "$SKIP_UI" = true ]; then
    echo "[2/5] Skipping Tauri Desktop UI build (--skip-ui)"
else
    echo "[2/5] Building Tauri Desktop UI..."
    cd "$ROOT_DIR/lilim_desktop"
    npm install --silent 2>/dev/null
    npm run tauri build
    cd "$ROOT_DIR"
    echo "      ✓ UI built"
fi

# 3. Create the Debian package
echo "[3/5] Building Debian package..."
cd "$ROOT_DIR"
./packaging/build_deb.sh
echo "      ✓ Package built"

# 4. Stop existing service (avoid port conflict)
echo "[4/5] Stopping existing service..."
sudo systemctl stop lilith-ai.service 2>/dev/null || true
sleep 1

# 5. Install the package
echo "[5/5] Installing Lilim locally..."
DEB_FILE="$ROOT_DIR/dist/lilim-linux-component.deb"
if [ -f "$DEB_FILE" ]; then
    sudo dpkg -i "$DEB_FILE"

    # Ensure updated Python brain is deployed
    sudo cp -r "$ROOT_DIR/lilim_core/"*.py /usr/lib/lilim/lilim_core/ 2>/dev/null || true

    # Sync service file from workspace (dpkg may have an older version)
    # Key changes: CPUQuota removed (throttle kills tok/s), RUST_LOG added
    sudo cp "$ROOT_DIR/systemd/system/lilith-ai.service" /usr/lib/systemd/system/lilith-ai.service

    echo "Restarting background service..."
    sudo systemctl daemon-reload
    sudo systemctl restart lilith-ai.service

    echo "=========================================="
    echo "✅ Success! Lilim is updated."
    echo "Please completely close your current Lilim Desktop App and relaunch it."
    echo
    echo "Monitor service: journalctl -u lilith-ai.service -f"
else
    echo "❌ Error: Debian package was not built successfully."
    exit 1
fi
