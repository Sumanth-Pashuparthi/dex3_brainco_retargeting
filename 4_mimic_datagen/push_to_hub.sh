#!/usr/bin/env bash
# Push the Mimic generation shards to the Hugging Face Hub.
#
#   HF_TOKEN=hf_... ./push_to_hub.sh <username>/<dataset-name>
#
# Uploads $DATA_DIR/gen_w*.hdf5 exactly as generate_dataset.py wrote them (run_merge.sh and
# run_convert.sh reproduce the merged HDF5 and the LeRobot tree from these), plus the dataset card
# next to this script if present. Nothing in this repo contains the data itself; this is the only
# step that moves it anywhere.
#
# The shards are ~2.3 GB each, so this goes through `hf upload-large-folder`, which uploads the LFS
# blobs in parallel and commits at the end. It is resumable: re-run the same command after an
# interruption and it continues from the per-file state in <staging>/.cache/huggingface. The repo
# looks empty on the Hub until the final commit lands.

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

REPO=${1:?usage: HF_TOKEN=hf_... ./push_to_hub.sh <username>/<dataset-name>}
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN is not set." >&2; exit 1; }

shopt -s nullglob
SHARDS=("$DATA_DIR"/gen_w*.hdf5)
[[ ${#SHARDS[@]} -gt 0 ]] || { echo "no gen_w*.hdf5 in $DATA_DIR. Run ./run_generate.sh." >&2; exit 1; }

HF=$(command -v hf) || HF="$GR00T_DIR/.venv/bin/hf"
[[ -x "$HF" ]] || { echo "hf CLI not found; pip install -U huggingface_hub" >&2; exit 1; }

# Stage symlinks so upload-large-folder sees one folder with only the files we want to publish.
STAGE="${STAGE:-/tmp/$(basename "$REPO")_hf}"
mkdir -p "$STAGE"
for f in "${SHARDS[@]}"; do ln -sfn "$f" "$STAGE/$(basename "$f")"; done
[[ -f dataset_card.md ]] && cp dataset_card.md "$STAGE/README.md"

echo "uploading ${#SHARDS[@]} shards ($(du -cshL "${SHARDS[@]}" | tail -1 | cut -f1)) -> https://huggingface.co/datasets/$REPO"
HF_TOKEN="$HF_TOKEN" "$HF" upload-large-folder "$REPO" "$STAGE" --repo-type dataset --num-workers 4

echo
echo "done: https://huggingface.co/datasets/$REPO"
