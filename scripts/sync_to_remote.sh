#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:?Set REMOTE_HOST in your shell or .env first}"
REMOTE_APP_DIR="${REMOTE_APP_DIR:?Set REMOTE_APP_DIR in your shell or .env first}"

rsync -av \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  ./ "${REMOTE_HOST}:${REMOTE_APP_DIR}/"
