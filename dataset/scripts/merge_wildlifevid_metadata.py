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
from pathlib import Path


ROOT = Path(os.environ.get("WILDLIFEVID_ROOT", str(Path(__file__).resolve().parents[2] / "annotations")))
DATASETS = [
    {
        "name": "AiM",
        "csv": ROOT / "source/AiM_metadata_identity_merged.csv",
        "path_prefix": "Animal-in-Motion/viscam/downloads/AiM",
    },
    {
        "name": "AnimalKingdom",
        "csv": ROOT / "source/AnimalKingdom_metadata.csv",
        "path_prefix": "AnimalKingdom",
    },
    {
        "name": "LoTE",
        "csv": ROOT / "source/LoTE-Animal_metadata.csv",
        "path_prefix": "LoTE-Animal",
    },
    {
        "name": "MammalNet",
        "csv": ROOT / "source/MammalNet_metadata.csv",
        "path_prefix": "MammalNet",
    },
]

OUTPUT_CSV = Path(os.environ.get("WILDLIFEVID_OUTPUT_CSV", str(ROOT / "WildlifeVid.rebuilt.csv")))


def prefix_path(prefix: str, value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    return f"{prefix}/{value}"


def load_rows():
    merged_rows = []
    all_species = set()

    for spec in DATASETS:
        with open(spec["csv"], newline="") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            row = dict(row)
            row["dataset"] = spec["name"]
            row["source_species_id"] = row.get("species_id", "")
            row["source_identity"] = row.get("identity", "")
            row["source_identity_str"] = row.get("identity_str", "")
            row["source_global_identity"] = row.get("global_identity", "")

            row["video"] = prefix_path(spec["path_prefix"], row.get("video", ""))
            row["reference_image"] = prefix_path(spec["path_prefix"], row.get("reference_image", ""))
            row["segmented_image"] = prefix_path(spec["path_prefix"], row.get("segmented_image", ""))

            merged_rows.append(row)
            all_species.add(row["species"])

    return merged_rows, sorted(all_species)


def assign_species_ids(rows, all_species):
    species_to_id = {species: idx for idx, species in enumerate(all_species)}
    for row in rows:
        row["species_id"] = str(species_to_id[row["species"]])


def assign_identity_ids(rows):
    next_identity_by_species = {}
    seen_identity_group = {}
    next_global_identity = 0

    for row in rows:
        species = row["species"]
        source_identity_str = row.get("source_identity_str", "").strip()
        source_identity = row.get("source_identity", "").strip()
        source_track_id = row.get("track_id", "").strip()

        identity_group_key = (
            row["dataset"],
            species,
            source_identity_str or source_identity or source_track_id,
        )

        if species not in next_identity_by_species:
            next_identity_by_species[species] = 0

        if identity_group_key not in seen_identity_group:
            new_identity = next_identity_by_species[species]
            next_identity_by_species[species] += 1

            seen_identity_group[identity_group_key] = {
                "identity": new_identity,
                "global_identity": next_global_identity,
            }
            next_global_identity += 1

        identity_info = seen_identity_group[identity_group_key]
        row["identity"] = str(identity_info["identity"])
        row["global_identity"] = str(identity_info["global_identity"])
        row["identity_str"] = f"{species}_ID_{identity_info['identity']:04d}"


def write_rows(rows):
    fieldnames = [
        "video",
        "prompt",
        "dataset",
        "species",
        "species_id",
        "track_id",
        "identity",
        "identity_str",
        "global_identity",
        "reference_image",
        "segmented_image",
        "segment_prompt",
        "segment_status",
        "segment_score",
        "source_species_id",
        "source_identity",
        "source_identity_str",
        "source_global_identity",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows, all_species = load_rows()
    assign_species_ids(rows, all_species)
    assign_identity_ids(rows)
    write_rows(rows)
    print(f"Wrote {len(rows)} rows across {len(all_species)} species to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
