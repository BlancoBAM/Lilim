#!/usr/bin/env bash
# local_install.sh — Build and install Lilim from source on this machine.
# Run from the repo root: ./local_install.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKIP_UI=false

echo "=========================================="
echo "  Lilim — Local Build & Install"
echo "=========================================="
echo

for arg in "$@"; do
    case "$arg" in
        --skip-ui) SKIP_UI=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# ── 1. Rust runtime ───────────────────────────────────────────
echo "[1/5] Building Rust runtime..."
cargo build --release --manifest-path="$ROOT_DIR/Cargo.toml" -p lilim-runtime
echo "      ✓ Runtime built: $ROOT_DIR/target/release/lilim-runtime"

# ── 2. Tauri Desktop UI ───────────────────────────────────────
if [ "$SKIP_UI" = true ]; then
    echo "[2/5] Skipping Tauri UI (--skip-ui)"
else
    echo "[2/5] Building Tauri Desktop UI..."
    cd "$ROOT_DIR/lilim_desktop"
    npm install --silent 2>/dev/null
    npm run tauri build
    cd "$ROOT_DIR"

    # Find what Tauri actually produced — check all known binary names
    TAURI_BIN=""
    RELEASE_DIR="$ROOT_DIR/lilim_desktop/src-tauri/target/release"
    for bin_name in "tauri-lilim-desktop" "tauri-applilim-desktop" "Lilim" "lilim" "tauri-app"; do
        candidate="$RELEASE_DIR/$bin_name"
        if [ -f "$candidate" ] && [ -x "$candidate" ]; then
            TAURI_BIN="$candidate"
            break
        fi
    done
    # Final fallback: any ELF binary in the release dir (not a .so or debug file)
    if [ -z "$TAURI_BIN" ]; then
        TAURI_BIN=$(find "$RELEASE_DIR" -maxdepth 1 -type f -executable \
            ! -name "*.so" ! -name "*.d" ! -name ".cargo-lock" \
            -exec sh -c 'file "$1" | grep -q ELF && echo "$1"' _ {} \; 2>/dev/null | head -n 1)
    fi
    if [ -z "$TAURI_BIN" ]; then
        echo "ERROR: Tauri build succeeded but could not find the output binary." >&2
        echo "       Searched: $RELEASE_DIR/" >&2
        ls -la "$RELEASE_DIR/" >&2
        exit 1
    fi
    echo "      ✓ UI built: $TAURI_BIN"
fi

# ── 3. Debian package ─────────────────────────────────────────
echo "[3/5] Building Debian package..."
export ROOT_DIR
bash "$ROOT_DIR/packaging/build_deb.sh"

DEB_FILE="$ROOT_DIR/dist/lilim_0.1.0_amd64.deb"
if [ ! -f "$DEB_FILE" ]; then
    echo "❌ Error: Debian package was not created at $DEB_FILE" >&2
    exit 1
fi
echo "      ✓ Package built: $DEB_FILE"

# ── 4. Stop any existing service ──────────────────────────────
echo "[4/5] Stopping existing service (if any)..."
sudo systemctl stop lilith-ai.service 2>/dev/null || true
sleep 1

# ── 5. Install ────────────────────────────────────────────────
echo "[5/5] Installing..."
sudo dpkg -i "$DEB_FILE"

# ── Post-install verification: ensure every component is on disk ──────────
echo "      Verifying installation..."

# Runtime binary (most critical — service won't start without it)
if [ ! -f /usr/bin/lilim-runtime ]; then
    echo "      ⚠ Runtime binary missing from deb — copying directly..."
    sudo cp "$ROOT_DIR/target/release/lilim-runtime" /usr/bin/lilim-runtime
    sudo chmod +x /usr/bin/lilim-runtime
fi

# Desktop UI binary
if [ ! -f /usr/bin/lilim ]; then
    echo "      ⚠ UI binary missing from deb — copying directly..."
    RELEASE_DIR="$ROOT_DIR/lilim_desktop/src-tauri/target/release"
    for bin_name in "tauri-lilim-desktop" "tauri-applilim-desktop" "Lilim" "lilim"; do
        if [ -f "$RELEASE_DIR/$bin_name" ]; then
            sudo cp "$RELEASE_DIR/$bin_name" /usr/bin/lilim
            sudo chmod +x /usr/bin/lilim
            break
        fi
    done
fi

# Python brain
sudo mkdir -p /usr/lib/lilim/lilim_core /etc/lilith
sudo cp -r "$ROOT_DIR/lilim_core/"*.py /usr/lib/lilim/lilim_core/ 2>/dev/null || true
sudo cp -r "$ROOT_DIR/config/"* /etc/lilith/ 2>/dev/null || true

# Python venv
if [ ! -f /usr/lib/lilim/venv/bin/python3 ]; then
    echo "      Setting up Python virtual environment..."
    sudo python3 -m venv /usr/lib/lilim/venv
fi
sudo /usr/lib/lilim/venv/bin/pip install --quiet fastapi uvicorn litellm apscheduler pyyaml httpx "beautifulsoup4>=4.12"

# Service file (always sync from repo to pick up latest)
sudo cp "$ROOT_DIR/systemd/system/lilith-ai.service" /lib/systemd/system/lilith-ai.service
# Patch the User= line to this machine's actual user
CURRENT_USER="$(logname 2>/dev/null || echo "$SUDO_USER" || echo "$USER")"
sudo sed -i "s|^User=.*|User=${CURRENT_USER}|g" /lib/systemd/system/lilith-ai.service
sudo sed -i "s|^Group=.*|Group=${CURRENT_USER}|g" /lib/systemd/system/lilith-ai.service
sudo sed -i "s|^WorkingDirectory=.*|WorkingDirectory=/home/${CURRENT_USER}|g" /lib/systemd/system/lilith-ai.service
sudo sed -i "s|Environment=HOME=.*|Environment=HOME=/home/${CURRENT_USER}|g" /lib/systemd/system/lilith-ai.service

sudo systemctl daemon-reload
sudo systemctl enable lilith-ai.service
sudo systemctl restart lilith-ai.service

echo
echo "=========================================="
echo "✅ Lilim installed successfully!"
echo
echo "  Launch:          lilim"
echo "  Or find it in:   Applications menu"
echo "  Service logs:    journalctl -u lilith-ai -f"
echo "=========================================="
