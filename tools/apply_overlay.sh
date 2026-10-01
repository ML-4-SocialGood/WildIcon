#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 WildIcon authors.
#
# This file is part of WildIcon and is licensed under the GNU Lesser
# General Public License, version 3 or (at your option) any later version.
# Distributed without any warranty; see WildIcon LICENSE and COPYING.
# Installed overlays include these terms under LICENSES/WildIcon/.

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

LICENSE_DIR="${TARGET_ROOT}/LICENSES/WildIcon"
mkdir -p "${LICENSE_DIR}"
cp "${REPO_ROOT}/LICENSE" "${LICENSE_DIR}/LGPL-3.0.txt"
cp "${REPO_ROOT}/COPYING" "${LICENSE_DIR}/GPL-3.0.txt"
cp "${REPO_ROOT}/LICENSES/Apache-2.0.txt" "${LICENSE_DIR}/Apache-2.0.txt"
cp "${REPO_ROOT}/NOTICE" "${LICENSE_DIR}/NOTICE"

echo "Applied WildIcon overlay to ${TARGET_ROOT}"

