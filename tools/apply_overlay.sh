#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: tools/apply_overlay.sh /path/to/DiffSynth-Studio"
  exit 1
fi

TARGET_ROOT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "${TARGET_ROOT}/diffsynth" ] || [ ! -d "${TARGET_ROOT}/examples" ]; then
  echo "Target does not look like a DiffSynth-Studio checkout: ${TARGET_ROOT}"
  exit 1
fi

rsync -av "${REPO_ROOT}/framework/diffsynth/" "${TARGET_ROOT}/diffsynth/"
rsync -av "${REPO_ROOT}/framework/examples/" "${TARGET_ROOT}/examples/"

echo "Applied WildIcon overlay to ${TARGET_ROOT}"

