#!/usr/bin/env python3
"""Attach locally prepared foreground images without altering public annotations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def prepare(input_csv: Path, output_csv: Path, data_root: Path,
            foreground_root: Path | None = None, segmentation_csv: Path | None = None) -> int:
    if input_csv.resolve() == output_csv.resolve():
        raise ValueError("Use a separate output CSV to preserve the public annotations.")
    with input_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"video", "reference_image", "prompt"}
    if not rows or not required.issubset(fields):
        raise ValueError("Input must contain rows with video, reference_image, and prompt columns.")
    if "segmented_image" not in fields:
        fields.insert(fields.index("reference_image") + 1, "segmented_image")
    mapping = {}
    if segmentation_csv is not None:
        with segmentation_csv.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not {"video", "segmented_image"}.issubset(reader.fieldnames or []):
                raise ValueError("Segmentation CSV requires video and segmented_image columns.")
            for row in reader:
                if row["video"] in mapping:
                    raise ValueError(f"Duplicate video in segmentation CSV: {row['video']}")
                mapping[row["video"]] = row["segmented_image"].strip()
    errors = []
    for row in rows:
        for key in ("video", "reference_image"):
            value = row[key].strip()
            path = data_root / value
            if not value or not path.is_file():
                errors.append(f"{key}: {path}")
        if segmentation_csv is not None:
            value = mapping.get(row["video"], "")
            path = data_root / value
        elif foreground_root is not None:
            reference = Path(row["reference_image"])
            if reference.is_absolute() or ".." in reference.parts:
                raise ValueError("Foreground-root mode requires relative reference_image paths.")
            path = foreground_root / reference
            value = str(path)
        else:
            value = row.get("segmented_image", "").strip()
            path = data_root / value
        if not value or not path.is_file():
            errors.append(f"segmented_image for {row['video']}: {value or '(empty)'}")
        else:
            row["segmented_image"] = str(path.resolve())
    if errors:
        raise ValueError(f"{len(errors)} missing local assets; no CSV written.\n" + "\n".join(errors[:10]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--foreground_root", type=Path,
                       help="Local foreground directory mirroring the relative reference_image paths.")
    group.add_argument("--segmentation_csv", type=Path,
                       help="Local video,segmented_image mapping; image paths resolve under data_root.")
    args = parser.parse_args()
    try:
        count = prepare(**vars(args))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Wrote {count} complete local training rows to {args.output_csv}")


if __name__ == "__main__":
    main()
