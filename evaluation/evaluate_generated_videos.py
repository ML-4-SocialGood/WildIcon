#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 WildIcon authors.
#
# This file is part of WildIcon and is licensed under the GNU Lesser
# General Public License, version 3 or (at your option) any later version.
# Distributed without any warranty; see WildIcon LICENSE and COPYING.
# Installed overlays include these terms under LICENSES/WildIcon/.

"""
Evaluate generated videos against reference images and prompts.

Paper-style metrics:
  - I2V Subject: DINO cosine similarity between the reference image embedding
    and sampled generated-frame embeddings.
  - I2V Background: DreamSim cosine similarity between the reference image
    embedding and sampled generated-frame embeddings.
  - Subject Consistency: mean cosine similarity between adjacent sampled-frame
    DINO embeddings.
  - Background Consistency: mean cosine similarity between adjacent sampled-frame
    CLIP image embeddings.
  - Text Relevance: CLIPScore-style average over sampled frames,
    mean_t max(100 * cos(CLIP_text(prompt), CLIP_image(frame_t)), 0).
  - Motion Smoothness: AMT interpolation-based score following the VBench /
    AMT-style evaluator used in the paper.

Notes:
  - Standard FVD is not computed here. Standard FVD requires a real-video
    distribution plus an I3D-based video feature extractor.
"""

from __future__ import annotations

import argparse
import cv2
import csv
import importlib
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

if __package__:
    from .records import load_manifest, load_directory
else:
    from records import load_manifest, load_directory


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated wildlife videos.")
    parser.add_argument("--manifest", type=Path, help="Generation manifest with exact video/reference/prompt pairs.")
    parser.add_argument("--check_inputs", action="store_true", help="Validate all input pairs without loading metric models.")
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Root directory that contains generated videos.",
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default=None,
        help="Directory containing reference images.",
    )
    parser.add_argument(
        "--ref_dir",
        dest="reference_dir_legacy",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--prompts_json",
        type=str,
        default=None,
        help="JSON file used during generation.",
    )
    parser.add_argument(
        "--prompt_json",
        dest="prompts_json_legacy",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--video_glob",
        type=str,
        default="**/*.mp4",
        help=(
            "Glob relative to --video_dir; ignored when --manifest is supplied."
        ),
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for CSV/JSON outputs. Default: <video_dir>/_eval",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device. Default uses CUDA if available.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Computation dtype on the selected device.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Uniformly sample this many frames per video.",
    )
    parser.add_argument(
        "--sample_frames",
        dest="num_frames_legacy",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--exclude_first_frame",
        action="store_true",
        help="Exclude the first frame before uniform sampling.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Image batch size for image-encoder inference.",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Optional cap for quick testing.",
    )
    parser.add_argument(
        "--num_videos",
        dest="max_videos_legacy",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP model for CLIP-I, Background Consistency, and Text Relevance.",
    )
    parser.add_argument(
        "--dino_model_name",
        type=str,
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="DINO image encoder for DINO-I.",
    )
    parser.add_argument(
        "--skip_non_ascii_clip_t",
        action="store_true",
        default=False,
        help=(
            "Skip CLIP-T for prompts containing CJK characters. Useful when the "
            "chosen CLIP backbone is English-only."
        ),
    )
    parser.add_argument(
        "--compute_i2v_background",
        action="store_true",
        help="Compute I2V Background with DreamSim similarity.",
    )
    parser.add_argument(
        "--dreamsim_type",
        type=str,
        default="ensemble",
        help="DreamSim model type. Default: ensemble.",
    )
    parser.add_argument(
        "--dreamsim_cache_dir",
        type=str,
        default=os.environ.get("DREAMSIM_CACHE_DIR", str(Path.home() / ".cache" / "dreamsim")),
        help="Cache directory for DreamSim weights.",
    )
    parser.add_argument(
        "--compute_motion_smoothness",
        action="store_true",
        help="Compute Motion Smoothness with the AMT interpolation-based evaluator.",
    )
    parser.add_argument(
        "--amt_repo_dir",
        type=str,
        default=os.environ.get("AMT_REPO_DIR", "third_party/AMT"),
        help="Local AMT repository path.",
    )
    parser.add_argument(
        "--amt_config",
        type=str,
        default=os.environ.get("AMT_CONFIG", "third_party/AMT/cfgs/AMT-S.yaml"),
        help="AMT config file used for Motion Smoothness.",
    )
    parser.add_argument(
        "--amt_ckpt",
        type=str,
        default=os.environ.get("AMT_CKPT", "checkpoints/amt/amt-s.pth"),
        help="AMT checkpoint used for Motion Smoothness.",
    )
    args = parser.parse_args()
    if args.reference_dir is None:
        args.reference_dir = args.reference_dir_legacy
    if args.prompts_json is None:
        args.prompts_json = args.prompts_json_legacy
    if args.manifest is None and (not args.video_dir or not args.reference_dir or not args.prompts_json):
        parser.error("Use --manifest, or supply --video_dir, --reference_dir and --prompts_json.")
    if args.num_frames_legacy is not None:
        args.num_frames = args.num_frames_legacy
    if args.max_videos_legacy is not None:
        args.max_videos = args.max_videos_legacy
    if args.suffix:
        suffix = args.suffix
        if suffix.endswith(".mp4"):
            suffix = suffix[:-4]
        args.video_glob = f"**/*__p??__*{suffix}.mp4"
    if args.num_frames < 2 or (args.max_videos is not None and args.max_videos < 1):
        parser.error("--num_frames must be at least 2 and --max_videos must be positive.")
    return args


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def contains_cjk_text(text: str) -> bool:
    return re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text) is not None


def is_missing_prompt_text(text: str | None) -> bool:
    return text is None or not text.strip()


def read_video_frames(
    video_path: Path,
    num_frames: int,
    exclude_first_frame: bool = False,
) -> list[Image.Image]:
    reader = imageio.get_reader(str(video_path), format="ffmpeg")
    frames = []
    try:
        for frame in reader:
            frames.append(Image.fromarray(frame).convert("RGB"))
    finally:
        reader.close()

    if not frames:
        raise RuntimeError(f"No frames could be decoded from {video_path}")

    start_idx = 1 if exclude_first_frame and len(frames) > 1 else 0
    usable = frames[start_idx:]
    if not usable:
        usable = [frames[-1]]

    count = min(num_frames, len(usable))
    indices = np.linspace(0, len(usable) - 1, num=count, dtype=int)
    return [usable[i] for i in indices.tolist()]


class ClipBackbone:
    def __init__(self, model_name: str, device: torch.device, dtype: torch.dtype):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.inference_mode():
            if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    vision_outputs = self.model.vision_model(pixel_values=pixel_values)
                    feats = self.model.visual_projection(vision_outputs.pooler_output)
            else:
                vision_outputs = self.model.vision_model(pixel_values=pixel_values)
                feats = self.model.visual_projection(vision_outputs.pooler_output)
        return F.normalize(feats.float(), dim=-1)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.inference_mode():
            if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    text_outputs = self.model.text_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    feats = self.model.text_projection(text_outputs.pooler_output)
            else:
                text_outputs = self.model.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                feats = self.model.text_projection(text_outputs.pooler_output)
        return F.normalize(feats.float(), dim=-1)


class DinoBackbone:
    def __init__(self, model_name: str, device: torch.device, dtype: torch.dtype):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.inference_mode():
            if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    outputs = self.model(pixel_values=pixel_values)
            else:
                outputs = self.model(pixel_values=pixel_values)
        feats = outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None else outputs.last_hidden_state[:, 0]
        return F.normalize(feats.float(), dim=-1)


class DreamSimBackbone:
    def __init__(
        self,
        device: torch.device,
        cache_dir: Path,
        dreamsim_type: str,
    ):
        from dreamsim import dreamsim

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.model, self.preprocess = dreamsim(
            pretrained=True,
            device=str(device),
            cache_dir=str(cache_dir),
            dreamsim_type=dreamsim_type,
        )

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        batch = torch.cat([self.preprocess(img) for img in images], dim=0).to(self.device)
        with torch.inference_mode():
            embeds = self.model.embed(batch)
        if embeds.ndim == 1:
            embeds = embeds.unsqueeze(0)
        return F.normalize(embeds.float(), dim=-1)


class AMTMotionSmoothness:
    @staticmethod
    def _load_module(module_name: str, file_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module {module_name} from {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def __init__(
        self,
        device: torch.device,
        repo_dir: Path,
        config_path: Path,
        ckpt_path: Path,
    ):
        self.device = device
        self.repo_dir = repo_dir
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        if not self.repo_dir.is_dir():
            raise FileNotFoundError(f"AMT repo not found: {self.repo_dir}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"AMT config not found: {self.config_path}")
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"AMT checkpoint not found: {self.ckpt_path}")
        sys.path.insert(0, str(self.repo_dir))
        from omegaconf import OmegaConf

        self.OmegaConf = OmegaConf
        for module_name in list(sys.modules):
            if module_name == "utils" or module_name.startswith("utils."):
                del sys.modules[module_name]
        importlib.invalidate_caches()
        utils_module = importlib.import_module("utils.utils", package=None)
        self.InputPadder = utils_module.InputPadder
        self.check_dim_and_resize = utils_module.check_dim_and_resize
        self.img2tensor = utils_module.img2tensor
        self.tensor2img = utils_module.tensor2img
        self.embt = torch.tensor(0.5, dtype=torch.float32).view(1, 1, 1, 1).to(device)

        network_cfg = self.OmegaConf.load(self.config_path).network
        module_name, class_name = network_cfg["name"].rsplit(".", 1)
        model_cls = getattr(importlib.import_module(module_name, package=None), class_name)
        self.model = model_cls(**network_cfg.get("params", {}))
        checkpoint = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(device)
        self.model.eval()

        if device.type == "cuda":
            self.anchor_resolution = 1024 * 512
            self.anchor_memory = 1500 * 1024**2
            self.anchor_memory_bias = 2500 * 1024**2
            self.vram_avail = torch.cuda.get_device_properties(device).total_memory
        else:
            self.anchor_resolution = 8192 * 8192
            self.anchor_memory = 1
            self.anchor_memory_bias = 0
            self.vram_avail = 1

    def _read_frames(self, video_path: Path) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        video = cv2.VideoCapture(str(video_path))
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video.release()
        return frames

    @staticmethod
    def _extract_every_other(frames: list[Any], start_from: int) -> list[Any]:
        return [frame for i, frame in enumerate(frames) if i >= start_from and (i - start_from) % 2 == 0]

    @staticmethod
    def _mean_absdiff(img1: np.ndarray, img2: np.ndarray) -> float:
        return float(np.mean(cv2.absdiff(img1, img2)))

    def score_video(self, video_path: Path) -> float | None:
        frames = self._read_frames(video_path)
        if len(frames) < 3:
            return None
        inputs = [self.img2tensor(frame).to(self.device) for frame in self._extract_every_other(frames, start_from=0)]
        if len(inputs) < 2:
            return None
        inputs = self.check_dim_and_resize(inputs)
        height, width = inputs[0].shape[-2:]
        scale = self.anchor_resolution / (height * width)
        scale *= np.sqrt((self.vram_avail - self.anchor_memory_bias) / self.anchor_memory)
        scale = 1 if scale > 1 else scale
        scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
        padder = self.InputPadder(inputs[0].shape, int(16 / scale))
        inputs = padder.pad(*inputs)

        outputs = [inputs[0]]
        for in_0, in_1 in zip(inputs[:-1], inputs[1:]):
            with torch.inference_mode():
                predicted = self.model(in_0, in_1, self.embt, scale_factor=scale, eval=True)["imgt_pred"]
            outputs.extend([predicted.cpu(), in_1.cpu()])

        outputs = padder.unpad(*outputs)
        outputs = [self.tensor2img(out) for out in outputs]
        original_mid = self._extract_every_other(frames, start_from=1)
        predicted_mid = self._extract_every_other(outputs, start_from=1)
        if not original_mid or len(original_mid) != len(predicted_mid):
            return None
        vfi_score = float(np.mean([self._mean_absdiff(a, b) for a, b in zip(original_mid, predicted_mid)]))
        return float((255.0 - vfi_score) / 255.0)


def batched_encode_images(
    backbone: Any,
    images: list[Image.Image],
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        outputs.append(backbone.encode_images(batch))
    return torch.cat(outputs, dim=0)


def mean_or_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def mean_adjacent_cosine(embeds: torch.Tensor) -> float | None:
    if embeds.ndim != 2 or embeds.shape[0] < 2:
        return None
    pairwise = (embeds[:-1] * embeds[1:]).sum(dim=-1)
    return float(pairwise.mean().item())


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    if args.manifest is not None:
        cases = load_manifest(args.manifest)
        video_dir = args.manifest.resolve().parent
        reference_dir = prompts_json = None
    else:
        video_dir = Path(args.video_dir)
        reference_dir = Path(args.reference_dir)
        prompts_json = Path(args.prompts_json)
        cases = load_directory(video_dir, reference_dir, prompts_json, args.video_glob)
    if args.max_videos is not None:
        cases = cases[:args.max_videos]
    if args.check_inputs:
        print(json.dumps({"validated_cases": len(cases), "cases": [
            {"video_path": str(case.video_path), "reference_image": str(case.reference_image),
             "prompt_text": case.prompt_text, "prompt_index": case.prompt_index}
            for case in cases
        ]}, indent=2))
        return
    output_dir = Path(args.output_dir) if args.output_dir else video_dir / "_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] validated videos: {len(cases)}")
    print(f"[INFO] clip/dino: {args.clip_model_name} / {args.dino_model_name}")

    clip_backbone = ClipBackbone(args.clip_model_name, device=device, dtype=dtype)
    dino_backbone = DinoBackbone(args.dino_model_name, device=device, dtype=dtype)
    dreamsim_backbone = None
    motion_scorer = None
    if args.compute_i2v_background:
        dreamsim_backbone = DreamSimBackbone(
            device=device,
            cache_dir=Path(args.dreamsim_cache_dir),
            dreamsim_type=args.dreamsim_type,
        )
    if args.compute_motion_smoothness:
        motion_scorer = AMTMotionSmoothness(
            device=device,
            repo_dir=Path(args.amt_repo_dir),
            config_path=Path(args.amt_config),
            ckpt_path=Path(args.amt_ckpt),
        )

    clip_image_cache: dict[str, torch.Tensor] = {}
    dino_image_cache: dict[str, torch.Tensor] = {}
    dreamsim_image_cache: dict[str, torch.Tensor] = {}
    clip_text_cache: dict[str, torch.Tensor] = {}

    rows: list[dict[str, Any]] = []
    missing_refs = 0
    missing_prompts = 0
    skipped_clip_t_non_ascii = 0

    for idx, case in enumerate(cases, start=1):
        video_path = case.video_path
        ref_path = case.reference_image
        prompt_filename, prompt_text = case.prompt_source_filename, case.prompt_text

        ref_key = str(ref_path)
        if ref_key not in clip_image_cache:
            ref_image = Image.open(ref_path).convert("RGB")
            clip_image_cache[ref_key] = batched_encode_images(
                clip_backbone,
                [ref_image],
                batch_size=1,
            )[0]
            dino_image_cache[ref_key] = batched_encode_images(
                dino_backbone,
                [ref_image],
                batch_size=1,
            )[0]
            if dreamsim_backbone is not None:
                dreamsim_image_cache[ref_key] = batched_encode_images(
                    dreamsim_backbone,
                    [ref_image],
                    batch_size=1,
                )[0]

        sampled_frames = read_video_frames(
            video_path,
            num_frames=args.num_frames,
            exclude_first_frame=args.exclude_first_frame,
        )
        clip_frame_embeds = batched_encode_images(
            clip_backbone,
            sampled_frames,
            batch_size=args.batch_size,
        )
        dino_frame_embeds = batched_encode_images(
            dino_backbone,
            sampled_frames,
            batch_size=args.batch_size,
        )
        dreamsim_i2v_background: float | None = None
        if dreamsim_backbone is not None:
            dreamsim_frame_embeds = batched_encode_images(
                dreamsim_backbone,
                sampled_frames,
                batch_size=args.batch_size,
            )

        ref_clip = clip_image_cache[ref_key].unsqueeze(0)
        ref_dino = dino_image_cache[ref_key].unsqueeze(0)

        clip_i = float((clip_frame_embeds @ ref_clip.T).mean().item())
        dino_i = float((dino_frame_embeds @ ref_dino.T).mean().item())
        subject_consistency = mean_adjacent_cosine(dino_frame_embeds)
        background_consistency = mean_adjacent_cosine(clip_frame_embeds)
        if dreamsim_backbone is not None:
            ref_dreamsim = dreamsim_image_cache[ref_key].unsqueeze(0)
            dreamsim_i2v_background = float((dreamsim_frame_embeds @ ref_dreamsim.T).mean().item())

        clip_t: float | None = None
        clip_t_cosine: float | None = None
        clip_t_status = "ok"
        if is_missing_prompt_text(prompt_text):
            clip_t_status = "missing_prompt"
            prompt_text = None
        elif args.skip_non_ascii_clip_t and contains_cjk_text(prompt_text):
            clip_t_status = "skipped_cjk_prompt"
            skipped_clip_t_non_ascii += 1
        else:
            if prompt_text not in clip_text_cache:
                clip_text_cache[prompt_text] = clip_backbone.encode_texts([prompt_text])[0]
            text_embed = clip_text_cache[prompt_text].unsqueeze(0)
            clip_t_cosine = float((clip_frame_embeds @ text_embed.T).mean().item())
            clip_t = float(torch.clamp((clip_frame_embeds @ text_embed.T) * 100.0, min=0.0).mean().item())

        motion_smoothness: float | None = None
        if motion_scorer is not None:
            motion_smoothness = motion_scorer.score_video(video_path)

        row = {
            "video_path": str(video_path),
            "image_stem": case.image_stem,
            "reference_image": str(ref_path),
            "prompt_source_filename": prompt_filename,
            "prompt_idx": case.prompt_index,
            "prompt_text": prompt_text,
            "resolution_tag": None,
            "num_sampled_frames": len(sampled_frames),
            "clip_i": clip_i,
            "dino_i": dino_i,
            "i2v_subject": dino_i,
            "i2v_background": dreamsim_i2v_background,
            "subject_consistency": subject_consistency,
            "background_consistency": background_consistency,
            "clip_t": clip_t,
            "clip_t_cosine": clip_t_cosine,
            "text_relevance": clip_t,
            "motion_smoothness": motion_smoothness,
            "clip_t_status": clip_t_status,
        }
        rows.append(row)

        print(
            f"[INFO] [{idx}/{len(cases)}] {video_path.name} | "
            f"CLIP-I={clip_i:.4f} DINO-I={dino_i:.4f} "
            f"I2V-BG={'NA' if dreamsim_i2v_background is None else f'{dreamsim_i2v_background:.4f}'} "
            f"MOTION={'NA' if motion_smoothness is None else f'{motion_smoothness:.4f}'} "
            f"CLIPScore={'NA' if clip_t is None else f'{clip_t:.4f}'}"
        )

    summary = {
        "video_dir": str(video_dir),
        "manifest": str(args.manifest) if args.manifest else None,
        "reference_dir": str(reference_dir) if reference_dir is not None else None,
        "prompts_json": str(prompts_json) if prompts_json is not None else None,
        "video_glob": args.video_glob,
        "clip_model_name": args.clip_model_name,
        "dino_model_name": args.dino_model_name,
        "device": str(device),
        "dtype": args.dtype,
        "num_frames": args.num_frames,
        "exclude_first_frame": args.exclude_first_frame,
        "num_requested_videos": len(cases),
        "num_scored_videos": len(rows),
        "num_missing_references": missing_refs,
        "num_missing_prompts": missing_prompts,
        "num_skipped_clip_t_cjk": skipped_clip_t_non_ascii,
        "clip_i_mean": mean_or_none([row["clip_i"] for row in rows]),
        "dino_i_mean": mean_or_none([row["dino_i"] for row in rows]),
        "i2v_subject_mean": mean_or_none([row["i2v_subject"] for row in rows]),
        "i2v_background_mean": mean_or_none([row["i2v_background"] for row in rows]),
        "subject_consistency_mean": mean_or_none([row["subject_consistency"] for row in rows]),
        "background_consistency_mean": mean_or_none([row["background_consistency"] for row in rows]),
        "clip_t_mean": mean_or_none([row["clip_t"] for row in rows]),
        "clip_t_cosine_mean": mean_or_none([row["clip_t_cosine"] for row in rows]),
        "text_relevance_mean": mean_or_none([row["text_relevance"] for row in rows]),
        "motion_smoothness_mean": mean_or_none([row["motion_smoothness"] for row in rows]),
        "i2v_background_metric": (
            f"dreamsim cosine similarity ({args.dreamsim_type})" if args.compute_i2v_background else None
        ),
        "motion_smoothness_metric": (
            "amt interpolation error score" if args.compute_motion_smoothness else None
        ),
        "fvd": None,
        "fvd_note": (
            "Standard FVD was not computed. You need a real-video directory "
            "plus an I3D-based evaluator to compute standard FVD."
        ),
    }
    summary["paper_metrics"] = {
        "I2V Subject": summary["i2v_subject_mean"],
        "I2V Background": summary["i2v_background_mean"],
        "Subject Consistency": summary["subject_consistency_mean"],
        "Background Consistency": summary["background_consistency_mean"],
        "Text Relevance": summary["text_relevance_mean"],
        "Motion Smoothness": summary["motion_smoothness_mean"],
        "FVD": summary["fvd"],
    }

    csv_path = output_dir / "per_video_metrics.csv"
    json_path = output_dir / "summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_path",
                "image_stem",
                "reference_image",
                "prompt_source_filename",
                "prompt_idx",
                "prompt_text",
                "resolution_tag",
                "num_sampled_frames",
                "clip_i",
                "dino_i",
                "i2v_subject",
                "i2v_background",
                "subject_consistency",
                "background_consistency",
                "clip_t",
                "clip_t_cosine",
                "text_relevance",
                "motion_smoothness",
                "clip_t_status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[RESULT] Summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[RESULT] Per-video CSV : {csv_path}")
    print(f"[RESULT] Summary JSON  : {json_path}")
    print(
        "[NOTE] For standard FVD, prepare a directory of real videos "
        "with the same evaluation split, then run an I3D-based FVD evaluator "
        "on real vs generated videos."
    )


if __name__ == "__main__":
    main()
