#!/usr/bin/env bash
# Deploy, setup and start PiVis on the Pi.
# Usage: bash scripts/deploy.sh
set -euo pipefail

HOST="repka@192.168.8.118"
REMOTE_DIR="/home/repka/pivis"
REPO="https://github.com/manonthegithub/PiVis.git"

echo "==> Pulling latest code on Pi"
ssh -o StrictHostKeyChecking=no "$HOST" bash <<REMOTE
  set -euo pipefail
  if [ -d "$REMOTE_DIR/.git" ]; then
    cd "$REMOTE_DIR" && git pull
  else
    git clone "$REPO" "$REMOTE_DIR"
  fi
REMOTE

echo "==> Installing Python deps"
ssh "$HOST" bash <<REMOTE
  set -euo pipefail
  cd "$REMOTE_DIR"
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
  mkdir -p tmp/audio models
REMOTE

echo "==> Downloading models (skips if already present)"
ssh "$HOST" bash <<REMOTE
  set -euo pipefail
  cd "$REMOTE_DIR"
  bash scripts/download_models.sh
REMOTE

echo "==> Checking .env"
if ! ssh "$HOST" test -f "$REMOTE_DIR/.env"; then
  if [ -f .env ]; then
    echo "    Copying local .env to Pi"
    scp .env "$HOST:$REMOTE_DIR/.env"
  else
    echo "    WARNING: no .env on Pi and none found locally."
    echo "    Create $REMOTE_DIR/.env on the Pi with at least ANTHROPIC_API_KEY=..."
    exit 1
  fi
fi

echo "==> Stopping any existing pivis process"
ssh "$HOST" "pkill -f 'python.*pivis' || true"
sleep 1

echo "==> Starting pivis (nohup, logs at $REMOTE_DIR/pivis.log)"
ssh "$HOST" bash <<REMOTE
  cd "$REMOTE_DIR"
  nohup .venv/bin/python -m pivis > pivis.log 2>&1 &
  echo "PID \$!"
REMOTE

echo ""
echo "Done. Open http://192.168.8.118:8000 in your browser."
echo "Tail logs: ssh $HOST 'tail -f $REMOTE_DIR/pivis.log'"
