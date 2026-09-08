#!/usr/bin/env bash
# Download the HF dataset into this folder (or refresh it).
set -euo pipefail
cd "$(dirname "$0")"
OUT=mimic_apple_pick_and_place
REPO=pashuparthis/mimic_apple_pick_and_place

if [[ -L "$OUT" ]]; then
  echo "refusing to overwrite symlink $OUT -> $(readlink "$OUT")"
  echo "remove it first if you want a fresh download"
  exit 1
fi

if command -v hf >/dev/null; then
  hf download "$REPO" --repo-type dataset --local-dir "$OUT"
elif command -v huggingface-cli >/dev/null; then
  huggingface-cli download "$REPO" --repo-type dataset --local-dir "$OUT"
else
  echo "install huggingface_hub:  pip install -U huggingface_hub" >&2
  exit 1
fi

echo
echo "ready: $(pwd)/$OUT"
du -sh "$OUT" | cut -f1
