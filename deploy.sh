#!/usr/bin/env bash
# Lilim Deploy Script — syncs ~/Lilim dev code to the installed system copy
# Run as: bash ~/Lilim/deploy.sh

set -e

SRC="/home/kt/Lilim/lilim_core"
DST="/usr/lib/lilim/lilim_core"
SERVICE="lilith-ai.service"

echo "=== Lilim Deploy ==="
echo "Source: $SRC"
echo "Dest:   $DST"
echo ""

# 1. Backup installed code
BACKUP="${DST}.bak.$(date +%Y%m%d_%H%M%S)"
echo "[1/4] Backing up installed code to $BACKUP ..."
sudo cp -r "$DST" "$BACKUP"
echo "      OK"

# 2. Sync dev code to installed location (preserves venv and __pycache__)
echo "[2/4] Syncing updated source files ..."
sudo rsync -av --exclude='__pycache__' --exclude='*.pyc' "$SRC/" "$DST/"
echo "      OK"

# 3. Compile Python files so the new code is bytecode-cached immediately
echo "[3/4] Compiling Python files ..."
sudo /usr/lib/lilim/venv/bin/python3 -m compileall -q "$DST/" || true
echo "      OK"

# 4. Restart the service
echo "[4/4] Restarting service ..."
# Try system service first, then user service
if sudo systemctl is-active --quiet "$SERVICE" 2>/dev/null; then
    sudo systemctl restart "$SERVICE"
elif sudo systemctl is-active --quiet "lilith-ai@kt.service" 2>/dev/null; then
    sudo systemctl restart "lilith-ai@kt.service"
elif systemctl --user is-active --quiet "$SERVICE" 2>/dev/null; then
    systemctl --user restart "$SERVICE"
else
    # Find whatever lilim service is running and restart it
    FOUND=$(sudo systemctl list-units --type=service --state=running | grep -i 'lilim\|lilith' | awk '{print $1}' | head -1)
    if [ -n "$FOUND" ]; then
        echo "      Found service: $FOUND"
        sudo systemctl restart "$FOUND"
    else
        echo "      WARNING: No running Lilim service found."
        echo "      Start it manually: sudo systemctl start lilith-ai@kt.service"
    fi
fi

echo ""
echo "=== Deploy complete ==="
echo ""
echo "Verify with: sudo systemctl status lilith-ai@kt.service"
echo "Live logs:   sudo journalctl -fu lilith-ai@kt.service"
