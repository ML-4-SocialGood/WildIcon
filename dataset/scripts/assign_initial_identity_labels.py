# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 WildIcon authors.
#
# This file is part of WildIcon and is licensed under the GNU Lesser
# General Public License, version 3 or (at your option) any later version.
# Distributed without any warranty; see WildIcon LICENSE and COPYING.
# Installed overlays include these terms under LICENSES/WildIcon/.

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    return parser.parse_args()


def extract_species(video_path):
    parts = Path(video_path).parts
    if len(parts) < 3:
        raise ValueError(f"Unexpected video path format: {video_path}")
    return parts[0]


def extract_track_id(video_path):
    path = Path(video_path)
    if len(path.parts) < 3:
        raise ValueError(f"Unexpected video path format: {video_path}")
    return path.parts[1]


def main():
    args = parse_args()

    with open(args.input_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Input CSV is empty")

    species_names = sorted({extract_species(row["video"]) for row in rows})
    species_to_id = {species: idx for idx, species in enumerate(species_names)}

    track_ids_by_species = {species: set() for species in species_names}
    for row in rows:
        species = extract_species(row["video"])
        track_ids_by_species[species].add(extract_track_id(row["video"]))

    identity_by_track = {}
    global_identity_by_track = {}
    global_identity = 0
    for species in species_names:
        for identity, track_id in enumerate(sorted(track_ids_by_species[species])):
            identity_by_track[(species, track_id)] = identity
            global_identity_by_track[(species, track_id)] = global_identity
            global_identity += 1

    output_rows = []
    for row in rows:
        species = extract_species(row["video"])
        track_id = extract_track_id(row["video"])
        identity = identity_by_track[(species, track_id)]
        updated_row = dict(row)
        updated_row.update(
            {
                "species": species,
                "species_id": species_to_id[species],
                "track_id": track_id,
                "identity": identity,
                "identity_str": f"{species}_{identity:04d}",
                "global_identity": global_identity_by_track[(species, track_id)],
            }
        )
        output_rows.append(updated_row)

    added_fieldnames = [
        "video",
        "prompt",
        "species",
        "species_id",
        "track_id",
        "identity",
        "identity_str",
        "global_identity",
    ]
    fieldnames = list(rows[0].keys())
    for fieldname in added_fieldnames:
        if fieldname not in fieldnames:
            fieldnames.append(fieldname)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved labeled metadata to {output_path}")
    print(f"Rows: {len(output_rows)}")
    print(f"Species: {len(species_names)}")
    print(f"Global identities: {global_identity}")


if __name__ == "__main__":
    main()
