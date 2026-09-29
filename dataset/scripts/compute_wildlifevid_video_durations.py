#!/usr/bin/env python3
import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2


def parse_args():
    wildlifevid_root = Path(os.environ.get("WILDLIFEVID_ROOT", "data/WildlifeVid"))
    dataset_root = Path(os.environ.get("WILDLIFEVID_DATASET_ROOT", str(wildlifevid_root)))
    parser = argparse.ArgumentParser(
        description="Compute per-video duration metadata for WildlifeVid.csv."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=wildlifevid_root / "WildlifeVid.csv",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=dataset_root,
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=wildlifevid_root / "WildlifeVid_video_durations.csv",
    )
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=wildlifevid_root / "WildlifeVid_video_durations_summary.json",
    )
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def read_rows(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def probe_video(dataset_root: Path, row: dict):
    rel_video = row["video"]
    abs_video = dataset_root / rel_video

    result = {
        "dataset": row["dataset"],
        "species": row["species"],
        "track_id": row["track_id"],
        "video": rel_video,
        "video_exists": str(abs_video.exists()).lower(),
        "readable": "false",
        "frame_count": "",
        "fps": "",
        "duration_sec": "",
        "width": "",
        "height": "",
        "resolution": "",
    }

    if not abs_video.exists():
        return result

    cap = cv2.VideoCapture(str(abs_video))
    if not cap.isOpened():
        cap.release()
        return result

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    duration = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0

    result.update(
        {
            "readable": "true",
            "frame_count": frame_count,
            "fps": round(fps, 6) if fps > 0 else "",
            "duration_sec": round(duration, 6) if duration > 0 else "",
            "width": width or "",
            "height": height or "",
            "resolution": f"{width}x{height}" if width and height else "",
        }
    )
    return result


def write_csv(rows, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "species",
        "track_id",
        "video",
        "video_exists",
        "readable",
        "frame_count",
        "fps",
        "duration_sec",
        "width",
        "height",
        "resolution",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows, summary_json: Path):
    summary = {
        "total_rows": len(rows),
        "existing_videos": sum(1 for row in rows if row["video_exists"] == "true"),
        "readable_videos": sum(1 for row in rows if row["readable"] == "true"),
        "per_dataset": {},
    }

    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        durations = [float(row["duration_sec"]) for row in subset if row["duration_sec"] not in ("", None)]
        summary["per_dataset"][dataset] = {
            "rows": len(subset),
            "readable_videos": sum(1 for row in subset if row["readable"] == "true"),
            "mean_duration_sec": round(sum(durations) / len(durations), 6) if durations else None,
            "min_duration_sec": round(min(durations), 6) if durations else None,
            "max_duration_sec": round(max(durations), 6) if durations else None,
        }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    args = parse_args()
    rows = read_rows(args.input_csv)

    results = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_index = {
            executor.submit(probe_video, args.dataset_root, row): idx
            for idx, row in enumerate(rows)
        }
        for idx, future in enumerate(as_completed(future_to_index), start=1):
            row_index = future_to_index[future]
            results[row_index] = future.result()
            if idx % 500 == 0 or idx == len(rows):
                print(f"Processed {idx}/{len(rows)} videos", flush=True)

    write_csv(results, args.output_csv)
    write_summary(results, args.summary_json)
    print(f"Wrote duration table to {args.output_csv}", flush=True)
    print(f"Wrote summary to {args.summary_json}", flush=True)


if __name__ == "__main__":
    main()
