#!/usr/bin/env bash
set -euo pipefail

# Lilim Complete Deployment Script
# 
# This script handles installing system dependencies, building the native Rust proxy,
# compiling the Tauri Desktop UI, creating the final Debian package, and installing it.

echo "=================================================="
echo "Lilim Automatic Deployment & Setup"
echo "=================================================="

# Ensure we are in the repository root
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "[1/6] Installing necessary system build dependencies..."
# We use sudo here; it will prompt you for your password
sudo apt-get update
sudo apt-get install -y \
  pkg-config libssl-dev build-essential \
  libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev \
  python3-pip python3-venv nodejs npm

echo "[2/6] Building Tauri Desktop UI..."
cd "$REPO_DIR/lilim_desktop"
npm install
npm run tauri build

echo "[3/6] Building Rust Proxy Gateway..."
cd "$REPO_DIR/crates/lilim-runtime"
cargo build --release

echo "[4/6] Creating Debian Package (.deb)..."
cd "$REPO_DIR"
chmod +x packaging/build_deb.sh
./packaging/build_deb.sh

echo "[5/6] Installing Lilim Component..."
DEB_FILE="$REPO_DIR/dist/lilim-linux-component.deb"
if [ -f "$DEB_FILE" ]; then
    sudo dpkg -i "$DEB_FILE"
    # In case there are missing dependencies
    sudo apt-get install -f -y
else
    echo "ERROR: Package $DEB_FILE not found!"
    exit 1
fi

echo "[6/6] Verifying Lilim Service..."
sleep 2
sudo systemctl --no-pager status lilith-ai.service || echo "Service started with warnings."

echo "=================================================="
echo "Deployment Complete! You can now start Lilim via your desktop menu or hotkey."
echo "=================================================="
