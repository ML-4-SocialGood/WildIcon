#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 WildIcon authors.
#
# This file is part of WildIcon and is licensed under the GNU Lesser
# General Public License, version 3 or (at your option) any later version.
# Distributed without any warranty; see WildIcon LICENSE and COPYING.
# Installed overlays include these terms under LICENSES/WildIcon/.

import csv
import os
import re
from pathlib import Path


ROOT = Path(os.environ.get("WILDLIFEVID_ROOT", "data/WildlifeVid"))
CSV_FILES = [
    ROOT / "AiM_videos_wan_i2v_metadata_v3.0.csv",
    ROOT / "AnimalKingdom_wan_i2v_metadata.csv",
    ROOT / "LoTE_Animal_wan_i2v_metadata.csv",
    ROOT / "MammalNet_wan_i2v_metadata.csv",
]


EXPLICIT_MAP = {
    "red_panda": "RedPanda",
    "red panda": "RedPanda",
    "redpanda": "RedPanda",
    "rhino": "Rhinoceros",
    "rhinoceros": "Rhinoceros",
    "hippo": "Hippopotamus",
    "hippopotamus": "Hippopotamus",
    "echina": "Echidna",
    "echidna": "Echidna",
    "panther": "BlackPanther",
    "black panther": "BlackPanther",
    "black_panther": "BlackPanther",
    "sea lion": "SeaLion",
    "sea_lion": "SeaLion",
    "sealion": "SeaLion",
    "polar_bear": "PolarBear",
    "polar bear": "PolarBear",
    "racoon": "Raccoon",
    "raccoon": "Raccoon",
    "analoga": "Crab",
    "cuvier": "Whale",
}


def normalize_species(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return raw

    key = raw.strip().lower()
    key = key.replace("-", " ")
    key = re.sub(r"\s+", " ", key)

    if key in EXPLICIT_MAP:
        return EXPLICIT_MAP[key]

    parts = re.split(r"[_\s]+", raw.strip())
    parts = [p for p in parts if p]
    return "".join(part[:1].upper() + part[1:] for part in parts)


def process_csv(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = f.readline()

    if not rows:
        return {"path": str(path), "changed_rows": 0, "changed_values": {}}

    fields = list(rows[0].keys())
    if "species" not in fields:
        return {"path": str(path), "changed_rows": 0, "changed_values": {}}

    changed_rows = 0
    changed_values = {}
    for row in rows:
        original = row.get("species", "")
        normalized = normalize_species(original)
        if normalized != original:
            row["species"] = normalized
            changed_rows += 1
            changed_values.setdefault(original, normalized)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "path": str(path),
        "changed_rows": changed_rows,
        "changed_values": changed_values,
    }


def main():
    for path in CSV_FILES:
        result = process_csv(path)
        print(path.name, "changed_rows=", result["changed_rows"])
        for src, dst in sorted(result["changed_values"].items()):
            print(f"  {src} -> {dst}")


if __name__ == "__main__":
    main()
