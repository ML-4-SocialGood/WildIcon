# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 WildIcon authors.
#
# This file is part of WildIcon and is licensed under the GNU Lesser
# General Public License, version 3 or (at your option) any later version.
# Distributed without any warranty; see WildIcon LICENSE and COPYING.
# Installed overlays include these terms under LICENSES/WildIcon/.

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild AiM metadata using prefix-constrained identity clustering results."
        )
    )
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--cluster_json", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    return parser.parse_args()


def normalize_species_name(species):
    return species.lower().replace(" ", "_")


def load_cluster_map(cluster_json_path):
    with open(cluster_json_path) as f:
        entries = json.load(f)

    cluster_by_track = {}
    for entry in entries:
        key = (entry["species_class"], entry["track_id"])
        cluster_by_track[key] = entry["identity_cluster_id"]
    return cluster_by_track


def rebuild_rows(rows, cluster_by_track):
    species_names = sorted({row["species"] for row in rows})
    species_to_id = {species: idx for idx, species in enumerate(species_names)}

    clusters_by_species = defaultdict(set)
    resolved_cluster_keys = []

    for row in rows:
        species = row["species"]
        track_id = row["track_id"]
        cluster_key = cluster_by_track.get((species, track_id))
        if cluster_key is None:
            cluster_key = f"{species}/__singleton__/{track_id}"
        resolved_cluster_keys.append(cluster_key)
        clusters_by_species[species].add(cluster_key)

    identity_by_cluster = {}
    global_identity_by_cluster = {}
    next_global_identity = 0
    for species in species_names:
        for local_identity, cluster_key in enumerate(sorted(clusters_by_species[species])):
            identity_by_cluster[(species, cluster_key)] = local_identity
            global_identity_by_cluster[(species, cluster_key)] = next_global_identity
            next_global_identity += 1

    output_rows = []
    for row, cluster_key in zip(rows, resolved_cluster_keys):
        species = row["species"]
        identity = identity_by_cluster[(species, cluster_key)]
        global_identity = global_identity_by_cluster[(species, cluster_key)]

        updated_row = dict(row)
        updated_row["species_id"] = species_to_id[species]
        updated_row["identity"] = identity
        updated_row["identity_str"] = f"{normalize_species_name(species)}_{identity:04d}"
        updated_row["global_identity"] = global_identity
        output_rows.append(updated_row)

    return output_rows


def main():
    args = parse_args()

    with open(args.input_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Input CSV is empty")

    cluster_by_track = load_cluster_map(args.cluster_json)
    output_rows = rebuild_rows(rows, cluster_by_track)
    fieldnames = list(rows[0].keys())

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved rebuilt metadata to {output_path}")
    print(f"Rows: {len(output_rows)}")


if __name__ == "__main__":
    main()
