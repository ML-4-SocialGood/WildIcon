# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 WildIcon authors.
#
# This file is part of WildIcon and is licensed under the GNU Lesser
# General Public License, version 3 or (at your option) any later version.
# Distributed without any warranty; see WildIcon LICENSE and COPYING.
# Installed overlays include these terms under LICENSES/WildIcon/.

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cluster AiM track videos into pseudo-identities within the same "
            "species and YouTube prefix group."
        )
    )
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--viz_output_dir", type=str, required=True)
    parser.add_argument(
        "--dino_repo",
        type=str,
        default=os.environ.get("DINO_REPO"),
        help="Path to a local DINOv3 repository, or set DINO_REPO.",
    )
    parser.add_argument(
        "--dino_weights",
        type=str,
        default=os.environ.get("DINO_WEIGHTS", ""),
        help="Optional DINOv3 checkpoint path, or set DINO_WEIGHTS.",
    )
    parser.add_argument("--distance_threshold", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--limit_groups",
        type=int,
        default=None,
        help="Optional debug limit on number of (species, prefix) groups.",
    )
    args = parser.parse_args()
    if not args.dino_repo:
        parser.error("--dino_repo is required unless DINO_REPO is set")
    return args


def load_runtime_dependencies():
    global AgglomerativeClustering, Image, T, cv2, np, torch, tqdm

    import cv2
    import numpy as np
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from sklearn.cluster import AgglomerativeClustering
    from tqdm import tqdm


def get_device(device_arg):
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_transform():
    return T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_dino_model(repo_path, weights_path, device):
    if str(Path(repo_path).parent) not in sys.path:
        sys.path.append(str(Path(repo_path).parent))
    model = torch.hub.load(repo_path, "dinov3_vith16plus", source="local").to(device)
    if weights_path and Path(weights_path).exists():
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_prefix(track_id):
    parts = track_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected track id format: {track_id}")
    return "_".join(parts[:-2])


def get_rgb_video_path(track_dir):
    exact = track_dir / f"{track_dir.name}_rgb.mp4"
    if exact.exists():
        return exact
    matches = sorted(track_dir.glob("*_rgb.mp4"))
    if matches:
        return matches[0]
    return None


def get_middle_frame(video_path):
    if video_path is None or not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return None
    middle_idx = max(0, frame_count // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def fallback_reference_frame(track_dir):
    rgb_frames = sorted(track_dir.glob("*_rgb.png"))
    if not rgb_frames:
        return None
    frame = cv2.imread(str(rgb_frames[0]))
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def encode_frames(model, transform, frames, device, batch_size):
    features = []
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start : start + batch_size]
        batch = torch.stack([transform(Image.fromarray(frame)) for frame in batch_frames]).to(device)
        with torch.no_grad():
            batch_features = model(batch).detach().cpu().numpy()
        features.append(batch_features)
    features = np.concatenate(features, axis=0)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.where(norms == 0, 1e-10, norms)


def make_grid(images, max_images=16, size=224):
    images = images[:max_images]
    if not images:
        return np.zeros((size, size, 3), dtype=np.uint8)
    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    grid = np.zeros((rows * size, cols * size, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        image_resized = cv2.resize(image, (size, size))
        row = index // cols
        col = index % cols
        grid[row * size : (row + 1) * size, col * size : (col + 1) * size, :] = image_resized
    return grid


def stable_relabel(labels, track_ids):
    cluster_to_tracks = defaultdict(list)
    for track_id, label in zip(track_ids, labels):
        cluster_to_tracks[int(label)].append(track_id)
    old_to_new = {}
    for new_label, old_label in enumerate(
        sorted(cluster_to_tracks, key=lambda label: min(cluster_to_tracks[label]))
    ):
        old_to_new[old_label] = new_label
    return [old_to_new[int(label)] for label in labels]


def scan_groups(base_dir):
    groups = defaultdict(list)
    for species_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        for track_dir in sorted(p for p in species_dir.iterdir() if p.is_dir()):
            track_id = track_dir.name
            prefix = extract_prefix(track_id)
            groups[(species_dir.name, prefix)].append(track_dir)
    return groups


def write_csv(rows, output_csv):
    import csv

    fieldnames = [
        "species_class",
        "youtube_prefix",
        "track_id",
        "relative_path",
        "video_relative_path",
        "identity_cluster_id",
        "identity_within_prefix",
        "cluster_size",
        "group_track_count",
        "frame_source",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    load_runtime_dependencies()

    base_dir = Path(args.base_dir)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)
    viz_output_dir = Path(args.viz_output_dir)
    viz_output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    transform = build_transform()
    model = load_dino_model(args.dino_repo, args.dino_weights, device)

    groups = scan_groups(base_dir)
    group_items = sorted(groups.items(), key=lambda item: (item[0][0], item[0][1]))
    if args.limit_groups is not None:
        group_items = group_items[: args.limit_groups]

    output_rows = []
    summary = {
        "base_dir": str(base_dir),
        "device": device,
        "distance_threshold": args.distance_threshold,
        "species_count": 0,
        "group_count": 0,
        "multi_track_group_count": 0,
        "total_tracks": 0,
        "cluster_count": 0,
        "failed_tracks": 0,
        "frame_source_counts": defaultdict(int),
    }

    species_seen = set()
    for (species, prefix), track_dirs in tqdm(group_items, desc="Clustering AiM prefix groups"):
        species_seen.add(species)
        summary["group_count"] += 1
        summary["total_tracks"] += len(track_dirs)
        if len(track_dirs) > 1:
            summary["multi_track_group_count"] += 1

        frames = []
        valid_track_dirs = []
        frame_sources = {}
        frame_cache = {}

        for track_dir in track_dirs:
            video_path = get_rgb_video_path(track_dir)
            frame = get_middle_frame(video_path)
            frame_source = "video_middle"
            if frame is None:
                frame = fallback_reference_frame(track_dir)
                frame_source = "rgb_png_fallback"
            if frame is None:
                summary["failed_tracks"] += 1
                continue
            frames.append(frame)
            valid_track_dirs.append(track_dir)
            frame_sources[track_dir.name] = frame_source
            frame_cache[track_dir.name] = frame
            summary["frame_source_counts"][frame_source] += 1

        if not valid_track_dirs:
            continue

        track_ids = [track_dir.name for track_dir in valid_track_dirs]
        features = encode_frames(model, transform, frames, device, args.batch_size)

        if len(track_ids) > 1:
            labels = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=args.distance_threshold,
            ).fit_predict(features)
        else:
            labels = np.array([0], dtype=np.int32)

        labels = stable_relabel(labels, track_ids)
        cluster_map = defaultdict(list)
        for track_dir, label in zip(valid_track_dirs, labels):
            cluster_map[int(label)].append(track_dir)

        summary["cluster_count"] += len(cluster_map)

        prefix_viz_dir = viz_output_dir / species / prefix
        prefix_viz_dir.mkdir(parents=True, exist_ok=True)

        for label, grouped_track_dirs in sorted(cluster_map.items(), key=lambda item: item[0]):
            identity_id = f"{species}/{prefix}/cluster_{label:03d}"
            grouped_track_dirs = sorted(grouped_track_dirs, key=lambda path: path.name)
            grid_img = make_grid([frame_cache[track_dir.name] for track_dir in grouped_track_dirs])
            viz_path = prefix_viz_dir / f"cluster_{label:03d}.jpg"
            cv2.imwrite(str(viz_path), cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))

            cluster_size = len(grouped_track_dirs)
            for track_dir in grouped_track_dirs:
                video_path = get_rgb_video_path(track_dir)
                output_rows.append(
                    {
                        "species_class": species,
                        "youtube_prefix": prefix,
                        "track_id": track_dir.name,
                        "relative_path": f"{species}/{track_dir.name}",
                        "video_relative_path": (
                            f"{species}/{track_dir.name}/{video_path.name}" if video_path else ""
                        ),
                        "identity_cluster_id": identity_id,
                        "identity_within_prefix": label,
                        "cluster_size": cluster_size,
                        "group_track_count": len(valid_track_dirs),
                        "frame_source": frame_sources[track_dir.name],
                    }
                )

    summary["species_count"] = len(species_seen)
    summary["frame_source_counts"] = dict(summary["frame_source_counts"])

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(output_rows, f, indent=2)
    write_csv(output_rows, output_csv)
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved clustering JSON to {output_json}")
    print(f"Saved clustering CSV to {output_csv}")
    print(f"Saved summary JSON to {summary_json}")
    print(f"Rows: {len(output_rows)}")
    print(f"Clusters: {summary['cluster_count']}")
    print(f"Failed tracks: {summary['failed_tracks']}")


if __name__ == "__main__":
    main()
