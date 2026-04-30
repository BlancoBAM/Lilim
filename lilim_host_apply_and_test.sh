#!/usr/bin/env bash
set -euo pipefail

############################
# Configurable variables
############################
REPO_DIR="${REPO_DIR:-$HOME/workspace/Lilim}"   # Path to the Lilim repository
BRANCH="${BRANCH:-main}"                       # Default branch
REMOTE="${REMOTE:-origin}"
RUN_PUSH="${RUN_PUSH:-0}"                      # 0 or 1 to push
API_PORT="${API_PORT:-8000}"
API_KEY="${LILIM_API_KEY:-change-me}"          # override before run if desired
LOG_DIR="${LOG_DIR:-/var/log/lilim}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/host_apply.log}"

echo "=================================================="
echo "Lilim host apply + build + service + test workflow"
echo "=================================================="
echo "Repo:   $REPO_DIR"
echo "Branch: $BRANCH"
echo "Remote: $REMOTE"
echo "Logs:   $LOG_FILE"
echo
mkdir -p "$LOG_DIR" || true
exec >> "$LOG_FILE" 2>&1
date
echo "Starting run..."

############################
# 1) Repo sync (idempotent)
############################
cd "$REPO_DIR"
git rev-parse --verify "$BRANCH" >/dev/null 2>&1 || echo "Branch $BRANCH may not exist yet"
echo "[1/12] Git pre-check"; git status --short || true
echo "[2/12] Fetch/checkout/pull"; git fetch --all --prune
git checkout "$BRANCH" || true
git pull --rebase "$REMOTE" "$BRANCH" || true

############################
# 2) Host dependencies
############################
echo "[3/12] Installing required packages"
sudo apt-get update
sudo apt-get install -y \
  python3 python3-pip python3-venv \
  curl jq git rsync \
  build-essential rpm dpkg-dev || true

############################
# 3) Python environment
############################
echo "[4/12] Creating/using virtualenv + Python deps"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --upgrade fastapi uvicorn pydantic litellm

############################
# 4) Validate code/tests
############################
echo "[5/12] Running compile checks"
set +e
python -m py_compile lilim_core/*.py
RC=$?
set -e
if [ "$RC" -ne 0 ]; then
  echo "Warning: Python compilation reported issues; continuing anyway."
fi

echo "[6/12] Running tests"
if [ -d "open-interpreter/tests" ]; then
  python -m unittest open-interpreter/tests/test_interpreter.py || true
else
  echo "Tests directory not found, skipping."
fi

############################
# 5) Build artifact (Optional/Custom)
############################
echo "[7/12] Build step"
if [ -d "packaging" ]; then
  mkdir -p dist
  ./packaging/build_deb.sh || true
else
  echo "Skipping .deb build: packaging directory not found"
fi

############################
# 6) Install service wrappers
############################
echo "[8/12] Installing runtime files and systemd units"
sudo mkdir -p /etc/lilith
for f in routing.toml zeroclaw.toml lilim-identity.json; do
  src="config/$f"; dst="/etc/lilith/$f"
  if [ -f "$dst" ]; then
    echo "$dst already exists, skipping overwrite"
  else
    sudo install -m 0644 "$src" "$dst" || true
  fi
done
sudo install -m 0755 scripts/lilim-serve /usr/bin/lilim-serve
sudo install -m 0644 systemd/system/lilith-ai.service /etc/systemd/system/lilith-ai.service
sudo mkdir -p /var/lib/lilim/cortex-index
sudo mkdir -p /var/backups/lilim

############################
# 7) Service environment override
############################
echo "[9/12] Creating systemd override for AI env"
sudo mkdir -p /etc/systemd/system/lilith-ai.service.d
cat <<EOT | sudo tee /etc/systemd/system/lilith-ai.service.d/override.conf >/dev/null
[Service]
Environment=LILIM_API_KEY=${API_KEY}
Environment=LILIM_ENABLE_PROVIDER_CALLS=0
WorkingDirectory=$REPO_DIR
EOT

############################
# 8) Enable/start services
############################
echo "[10/12] Reloading and starting services"
sudo systemctl daemon-reload
sudo systemctl enable --now lilith-ai.service || true

############################
# 9) Functional verification
############################
echo "[11/12] Health and status checks"
if command -v curl >/dev/null 2>&1; then
  curl -s "http://127.0.0.1:${API_PORT}/health" | jq . || echo "Health check failed (port ${API_PORT})"
else
  echo "curl not available; skipping health check."
fi
sudo systemctl --no-pager --full status lilith-ai.service | head -n 20
echo "--- LOG TAIL ---"
sudo journalctl -u lilith-ai.service -n 60 --no-pager

############################
# 10) Optional push
############################
if [[ "$RUN_PUSH" == "1" ]]; then
  echo "[12/12] Pushing branch to remote..."
  git push "$REMOTE" "$BRANCH"
else
  echo "[12/12] Skipping push. Set RUN_PUSH=1 to auto-push."
fi

echo
echo "Done."
