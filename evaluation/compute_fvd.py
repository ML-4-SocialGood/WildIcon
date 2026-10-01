#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 WildIcon authors.
#
# This file is part of WildIcon and is licensed under the GNU Lesser
# General Public License, version 3 or (at your option) any later version.
# Distributed without any warranty; see WildIcon LICENSE and COPYING.
# Installed overlays include these terms under LICENSES/WildIcon/.

"""
Compute FVD (Fréchet Video Distance) between a real-video directory
and one or more generated-video directories.

Feature extractor: torchvision R3D-18 (Kinetics-pretrained), penultimate layer.
Clip sampling: uniform T frames resized to H×W, normalised with Kinetics stats.
FVD = Fréchet distance between Gaussian fitted on real features vs generated features.

Usage:
    python compute_fvd.py \
        --real_dir  /path/to/real_videos \
        --gen_dirs  /path/to/generated_videos_a /path/to/generated_videos_b \
        --gen_names MethodA MethodB \
        --num_frames 16 \
        --resize 224 224 \
        --output_json /path/to/fvd_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torchvision.models.video import r3d_18, R3D_18_Weights


# ──────────────────────────────────────────────
# Video loading utilities
# ──────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".avi", ".mkv", ".webm", ".MOV"}

# Kinetics normalisation constants (same as torchvision video models)
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD  = [0.22803, 0.22145, 0.216989]


def list_videos(directory: Path) -> List[Path]:
    """Recursively collect video files from directory."""
    videos = []
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix in VIDEO_EXTS:
            videos.append(p)
    return videos


def read_video_aligned(
    path: Path,
    num_frames: int,
    resize: tuple[int, int],
    target_fps: float = 16.0,
    exclude_first: bool = False,
) -> Optional[torch.Tensor]:
    """
    Read a video file, sample `num_frames` frames aligned to `target_fps`,
    resize to (H, W), and return a float32 tensor of shape (T, C, H, W)
    normalised to [0,1].  Returns None on failure.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total < 1 or fps <= 0:
        cap.release()
        return None

    start_idx = 1 if exclude_first and total > 1 else 0
    stride = fps / target_fps
    
    indices = [start_idx + int(round(i * stride)) for i in range(num_frames)]
    indices = [min(idx, total - 1) for idx in indices]
    
    frames = []
    frame_idx = 0
    idx_set = set(indices)
    max_idx = max(indices)
    
    while frame_idx <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in idx_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resize[1], resize[0]), interpolation=cv2.INTER_LINEAR)
            frames.append((frame_idx, frame))
        frame_idx += 1
    cap.release()

    if not frames:
        return None

    frame_dict = {fi: fr for fi, fr in frames}
    final_frames = []
    for target_idx in indices:
        if target_idx not in frame_dict:
            target_idx = frames[-1][0]
        final_frames.append(frame_dict[target_idx])

    arr = np.stack(final_frames, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)
    return tensor


def normalise_clip(clip: torch.Tensor) -> torch.Tensor:
    """Apply Kinetics mean/std normalisation. clip: (T,3,H,W) in [0,1]."""
    mean = torch.tensor(KINETICS_MEAN, dtype=clip.dtype).view(1, 3, 1, 1)
    std  = torch.tensor(KINETICS_STD,  dtype=clip.dtype).view(1, 3, 1, 1)
    return (clip - mean) / std


# ──────────────────────────────────────────────
# Feature extractor: R3D-18 penultimate layer
# ──────────────────────────────────────────────

class R3D18Extractor(nn.Module):
    """
    Extract 512-d pooled feature vectors from R3D-18 (Kinetics-pretrained).
    Input:  (B, C, T, H, W)  normalised with Kinetics stats
    Output: (B, 512)
    """

    def __init__(self, device: torch.device):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1
        backbone = r3d_18(weights=weights)
        # Drop the final FC; keep up to avgpool
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # ends at AdaptiveAvgPool3d
        self.to(device)
        self.eval()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        feat = self.features(x)          # (B, 512, 1, 1, 1)
        feat = feat.flatten(1)           # (B, 512)
        return feat


# ──────────────────────────────────────────────
# Feature extraction loop
# ──────────────────────────────────────────────

def extract_features(
    video_paths: List[Path],
    extractor: R3D18Extractor,
    device: torch.device,
    num_frames: int,
    resize: tuple[int, int],
    batch_size: int,
    target_fps: float = 16.0,
    exclude_first: bool = False,
    label: str = "",
) -> np.ndarray:
    """
    Extract R3D-18 features for all videos.
    Returns array of shape (N, 512).
    """
    clips = []
    skipped = 0
    for i, vp in enumerate(video_paths):
        clip = read_video_aligned(vp, num_frames=num_frames,
                                   resize=resize, target_fps=target_fps, exclude_first=exclude_first)
        if clip is None:
            print(f"  [WARN] skipping unreadable: {vp.name}", flush=True)
            skipped += 1
            continue
        clip = normalise_clip(clip)     # (T,3,H,W)
        clips.append(clip)
        if (i + 1) % 10 == 0 or (i + 1) == len(video_paths):
            print(f"  [{label}] loaded {i+1}/{len(video_paths)} "
                  f"(skipped={skipped})", flush=True)

    if not clips:
        raise RuntimeError(f"No readable videos found for {label}")

    all_feats = []
    for start in range(0, len(clips), batch_size):
        batch_clips = clips[start:start + batch_size]
        # Stack: (B, T, C, H, W) → permute to (B, C, T, H, W)
        batch = torch.stack(batch_clips, dim=0).permute(0, 2, 1, 3, 4).to(device)
        feats = extractor(batch)        # (B, 512)
        all_feats.append(feats.cpu().numpy())

    features = np.concatenate(all_feats, axis=0)
    print(f"  [{label}] extracted features: {features.shape}", flush=True)
    return features


# ──────────────────────────────────────────────
# Fréchet Video Distance
# ──────────────────────────────────────────────

def compute_frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray,
    mu2: np.ndarray, sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Fréchet distance between two Gaussians N(mu1,sigma1) and N(mu2,sigma2).
    FD = ||mu1-mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1@sigma2))
    """
    diff = mu1 - mu2
    # Symmetric matrix square root
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    # Drop imaginary part (numerical artefact)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            im = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {im}")
        covmean = covmean.real
    fd = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(np.real(fd))


def features_to_stats(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(feats, axis=0)
    # rowvar=False: each column is a variable
    sigma = np.atleast_2d(np.cov(feats, rowvar=False))
    return mu, sigma


def compute_fvd(real_feats: np.ndarray, gen_feats: np.ndarray) -> float:
    mu_r, sigma_r = features_to_stats(real_feats)
    mu_g, sigma_g = features_to_stats(gen_feats)
    return compute_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)


def compute_kvd(real_feats: np.ndarray, gen_feats: np.ndarray, degree: int = 3) -> float:
    """Kernel Video Distance using polynomial MMD. Unbiased estimator. 
    Does not require N > D."""
    m = real_feats.shape[0]
    n = gen_feats.shape[0]
    if m < 2 or n < 2:
        return 0.0
    
    d = real_feats.shape[1]
    gamma = 1.0 / d
    coef0 = 1.0

    K_XX = (gamma * (real_feats @ real_feats.T) + coef0) ** degree
    K_YY = (gamma * (gen_feats @ gen_feats.T) + coef0) ** degree
    K_XY = (gamma * (real_feats @ gen_feats.T) + coef0) ** degree

    term_XX = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1))
    term_YY = (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1))
    term_XY = np.sum(K_XY) / (m * n)

    # Scale by 100 or 1000 for readability
    return float((term_XX + term_YY - 2 * term_XY) * 100.0)


def compute_pca_fvd(real_feats: np.ndarray, gen_feats: np.ndarray, k: int = 50) -> float:
    """FVD projected onto the top-k principal components to handle N < D."""
    n_components = min(k, real_feats.shape[0]-1, gen_feats.shape[0]-1)
    if n_components <= 0:
        return 0.0
    all_feats = np.concatenate([real_feats, gen_feats], axis=0)
    mu = np.mean(all_feats, axis=0)
    centered = all_feats - mu
    U, S, Vt = linalg.svd(centered, full_matrices=False)
    pca_comps = Vt[:n_components]
    
    real_proj = real_feats @ pca_comps.T
    gen_proj = gen_feats @ pca_comps.T
    
    return compute_fvd(real_proj, gen_proj)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute FVD between real and generated video directories.")
    p.add_argument("--real_dir", type=str, required=True,
                   help="Directory containing real wildlife videos.")
    p.add_argument("--gen_dirs", type=str, nargs="+", required=True,
                   help="One or more generated-video directories.")
    p.add_argument("--gen_names", type=str, nargs="+", default=None,
                   help="Display names for each gen_dir (default: directory names).")
    p.add_argument("--num_frames", type=int, default=16,
                   help="Frames per video clip (default: 16).")
    p.add_argument("--target_fps", type=float, default=16.0,
                   help="Target FPS to align temporal duration (default: 16.0).")
    p.add_argument("--resize", type=int, nargs=2, default=[224, 224],
                   metavar=("H", "W"),
                   help="Resize each frame to H×W (default: 224 224).")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Videos per GPU batch (default: 4).")
    p.add_argument("--device", type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--exclude_first_frame", action="store_true",
                   help="Skip the first frame (Wan2.x conditioning frame).")
    p.add_argument("--gen_glob", type=str, default=None,
                   help="Optional glob pattern to filter generated videos, e.g. '*480p.mp4'.")
    p.add_argument("--output_json", type=str, default=None,
                   help="Save results to this JSON path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    resize = tuple(args.resize)

    real_dir = Path(args.real_dir)
    gen_dirs = [Path(d) for d in args.gen_dirs]
    gen_names = args.gen_names
    if gen_names is None:
        gen_names = [d.name for d in gen_dirs]
    if len(gen_names) != len(gen_dirs):
        print("[ERROR] --gen_names must have same length as --gen_dirs", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Device      : {device}")
    print(f"[INFO] Num frames  : {args.num_frames}")
    print(f"[INFO] Target FPS  : {args.target_fps}")
    print(f"[INFO] Resize      : {resize[0]}×{resize[1]}")
    print(f"[INFO] Batch size  : {args.batch_size}")
    print(f"[INFO] Real dir    : {real_dir}")
    for name, gd in zip(gen_names, gen_dirs):
        print(f"[INFO] Gen '{name}' : {gd}")

    # Load extractor
    print("\n[INFO] Loading R3D-18 (Kinetics-pretrained)...", flush=True)
    extractor = R3D18Extractor(device)
    print("[INFO] R3D-18 loaded.\n", flush=True)

    # Extract real features
    real_videos = list_videos(real_dir)
    print(f"[INFO] Real videos found: {len(real_videos)}")
    real_feats = extract_features(
        real_videos, extractor, device,
        num_frames=args.num_frames,
        resize=resize,
        batch_size=args.batch_size,
        target_fps=args.target_fps,
        label="REAL",
    )

    # Extract generated features and compute FVD for each model
    results = {}
    for name, gd in zip(gen_names, gen_dirs):
        print(f"\n[INFO] ── Processing '{name}' ──", flush=True)
        gen_videos = list_videos(gd)
        if args.gen_glob:
            import fnmatch
            gen_videos = [v for v in gen_videos if fnmatch.fnmatch(v.name, args.gen_glob)]
        print(f"[INFO] '{name}' videos found: {len(gen_videos)}")
        gen_feats = extract_features(
            gen_videos, extractor, device,
            num_frames=args.num_frames,
            resize=resize,
            batch_size=args.batch_size,
            target_fps=args.target_fps,
            exclude_first=args.exclude_first_frame,
            label=name,
        )
        fvd_score = compute_fvd(real_feats, gen_feats)
        pfvd_score = compute_pca_fvd(real_feats, gen_feats, k=50)
        kvd_score = compute_kvd(real_feats, gen_feats)
        
        results[name] = {
            "fvd": fvd_score,
            "pca_fvd_50": pfvd_score,
            "kvd_x100": kvd_score,
            "num_real_videos": len(real_videos),
            "num_gen_videos": len(gen_videos),
            "real_feats_shape": list(real_feats.shape),
            "gen_feats_shape": list(gen_feats.shape),
        }
        print(f"\n  ✓ FVD ({name}): {fvd_score:.2f} | PCA-FVD-50: {pfvd_score:.2f} | KVD: {kvd_score:.3f}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Model':<20} {'FVD':>10}  {'PCA-FVD-50':>12}  {'KVD (x100)':>13}  {'# Real':>8}  {'# Gen':>7}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<20} {r['fvd']:>10.2f}  {r['pca_fvd_50']:>12.2f}  {r['kvd_x100']:>13.3f}  {r['num_real_videos']:>8}  {r['num_gen_videos']:>7}")
    print("=" * 80)
    print("(Lower score = more realistic / closer to real distribution)")

    # Save JSON
    out = {
        "config": {
            "num_frames": args.num_frames,
            "resize": list(resize),
            "feature_extractor": "R3D-18 (Kinetics400-V1), penultimate layer (512-d)",
            "fvd_metric": "Fréchet distance on R3D-18 features",
            "real_dir": str(real_dir),
            "exclude_first_frame": args.exclude_first_frame,
            "gen_glob": args.gen_glob,
        },
        "results": results,
    }
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[INFO] Results saved → {out_path}")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
