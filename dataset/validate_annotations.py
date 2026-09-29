#!/usr/bin/env python3
"""Validate annotation structure and report counts without loading source media."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath


def validate(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"video", "prompt", "dataset", "species", "species_id", "track_id",
                    "identity", "identity_str", "global_identity", "reference_image", "segmented_image"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing annotation columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty annotation table")
    videos = set()
    identity_owners = {}
    species_ids = {}
    by_source = defaultdict(list)
    for line, row in enumerate(rows, 2):
        for key in required - {"segmented_image"}:
            if not row.get(key, "").strip():
                raise ValueError(f"Line {line}: empty {key}")
        for key in ("video", "reference_image"):
            value = row[key]
            relative = PurePosixPath(value)
            if relative.is_absolute() or ".." in relative.parts or "\\" in value or ":" in value:
                raise ValueError(f"Line {line}: {key} must be a portable relative path")
        for key in ("identity", "global_identity", "species_id"):
            if not row[key].isdigit():
                raise ValueError(f"Line {line}: invalid {key}")
        if row["video"] in videos:
            raise ValueError(f"Duplicate video: {row['video']}")
        videos.add(row["video"])
        owner = (row["dataset"], row["species"], row["identity"], row["identity_str"])
        if identity_owners.setdefault(row["global_identity"], owner) != owner:
            raise ValueError(f"Identity collision: {row['global_identity']}")
        if species_ids.setdefault(row["species_id"], row["species"]) != row["species"]:
            raise ValueError(f"Species id collision: {row['species_id']}")
        by_source[row["dataset"]].append(row)
    return {
        "videos": len(rows), "identities": len(identity_owners),
        "species_labels": len({r["species"] for r in rows}),
        "segmented_image_paths": sum(bool(r["segmented_image"].strip()) for r in rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sources": {
            source: {"videos": len(group), "identities": len({r["global_identity"] for r in group}),
                     "species_labels": len({r["species"] for r in group})}
            for source, group in sorted(by_source.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?",
                        default=Path(__file__).resolve().parents[1] / "annotations/WildlifeVid.csv")
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.csv), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
