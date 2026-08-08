#!/usr/bin/env bash
# Deploy, setup and start PiVis on the Pi.
# Usage: bash scripts/deploy.sh
set -euo pipefail

HOST="repka@192.168.8.118"
HOST_IP="192.168.8.118"
KEY="$(dirname "$0")/../.ssh_key"
REMOTE_DIR="/home/repka/pivis"
REPO="https://github.com/manonthegithub/PiVis.git"
SSH="ssh -i $KEY -o StrictHostKeyChecking=no $HOST"
SCP="scp -i $KEY -o StrictHostKeyChecking=no"

echo "==> Pulling latest code on Pi"
$SSH bash <<REMOTE
  set -euo pipefail
  if [ -d "$REMOTE_DIR/.git" ]; then
    cd "$REMOTE_DIR" && git pull
  else
    git clone "$REPO" "$REMOTE_DIR"
  fi
REMOTE

echo "==> Installing Python deps"
$SSH bash <<REMOTE
  set -euo pipefail
  cd "$REMOTE_DIR"
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
  mkdir -p tmp/audio models
REMOTE

echo "==> Downloading models (skips if already present)"
$SSH bash <<REMOTE
  set -euo pipefail
  cd "$REMOTE_DIR"
  bash scripts/download_models.sh
REMOTE

echo "==> Copying .env to Pi"
$SCP "$(dirname "$0")/../.env" "$HOST:$REMOTE_DIR/.env"

echo "==> Stopping any existing pivis process"
$SSH "pkill -f 'python.*pivis' || true"
sleep 1

echo "==> Starting pivis (logs at $REMOTE_DIR/pivis.log)"
$SSH bash <<REMOTE
  cd "$REMOTE_DIR"
  nohup .venv/bin/python -m pivis > pivis.log 2>&1 &
  echo "Started PID \$!"
REMOTE

echo ""
echo "Done. Open http://$HOST_IP:8000"
echo "Tail logs: ssh -i .ssh_key $HOST 'tail -f $REMOTE_DIR/pivis.log'"
