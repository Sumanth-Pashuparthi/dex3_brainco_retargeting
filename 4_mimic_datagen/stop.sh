#!/usr/bin/env bash
# Stop every container this pipeline started. Shards written so far stay on disk and remain valid.

set -uo pipefail
IDS=$(docker ps --filter label=g1_apple_mimic -q)
[[ -z "$IDS" ]] && { echo "nothing running"; exit 0; }
docker stop $IDS
echo "stopped $(wc -w <<< "$IDS") container(s)"
