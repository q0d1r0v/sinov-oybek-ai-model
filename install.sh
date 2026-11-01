#!/usr/bin/env bash
# Kutubxonalar va modelni o'rnatadi. Model fayli GitHub Releases'dan yuklanadi.
set -euo pipefail
cd "$(dirname "$0")"
REPO="q0d1r0v/sinov-oybek-ai-model"
TAG="v1"
FILE="chat_best.pt"

command -v python3 >/dev/null || { echo "python3 topilmadi"; exit 1; }
python3 -m venv .venv || { echo "venv yaratilmadi. O'rnating: sudo apt install python3-venv"; exit 1; }
.venv/bin/pip install -q --upgrade pip
echo "Kutubxonalar o'rnatilmoqda..."
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q tokenizers numpy

mkdir -p weights
if [ -n "${MODEL_FILE:-}" ]; then
  cp "$MODEL_FILE" "weights/$FILE"
elif [ ! -f "weights/$FILE" ]; then
  echo "Model yuklanmoqda (40 MB)..."
  if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    gh release download "$TAG" -R "$REPO" -p "$FILE" -D weights --clobber
  else
    curl -fL --retry 3 -o "weights/$FILE" "https://github.com/$REPO/releases/download/$TAG/$FILE"
  fi
fi
sha256sum -c weights.sha256 || { echo "Model fayli buzilgan"; rm -f "weights/$FILE"; exit 1; }
echo "Tayyor. Ishga tushirish: ./run.sh"
