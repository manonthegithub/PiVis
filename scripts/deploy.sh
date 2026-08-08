#!/usr/bin/env bash
# Usage: bash scripts/deploy.sh [pi-host]
set -euo pipefail

HOST="${1:-repka@192.168.8.118}"
REMOTE_DIR="/home/repka/pivis"

echo "==> Syncing code to $HOST:$REMOTE_DIR"
rsync -avz --exclude='.venv' --exclude='models' --exclude='tmp' --exclude='.git' \
  ./ "$HOST:$REMOTE_DIR/"

echo "==> Installing dependencies on Pi"
ssh "$HOST" bash <<'REMOTE'
  set -euo pipefail
  cd ~/pivis
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
  mkdir -p tmp/audio models
REMOTE

echo "==> Downloading models on Pi (if missing)"
ssh "$HOST" bash <<'REMOTE'
  set -euo pipefail
  cd ~/pivis
  bash scripts/download_models.sh
REMOTE

echo ""
echo "Done. To start:"
echo "  ssh $HOST 'cd ~/pivis && ANTHROPIC_API_KEY=<key> .venv/bin/python -m pivis'"
echo "Or copy .env first:"
echo "  scp .env $HOST:~/pivis/.env && ssh $HOST 'cd ~/pivis && .venv/bin/python -m pivis'"
