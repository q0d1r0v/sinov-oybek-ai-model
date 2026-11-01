#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] && [ -f weights/chat_best.pt ] || { echo "Run ./install.sh first"; exit 1; }
exec .venv/bin/python server.py
