#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 WildIcon authors.
#
# This file is part of WildIcon and is licensed under the GNU Lesser
# General Public License, version 3 or (at your option) any later version.
# Distributed without any warranty; see WildIcon LICENSE and COPYING.
# Installed overlays include these terms under LICENSES/WildIcon/.

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from diffsynth.core import load_state_dict
from diffsynth.models.wildicon_identity import WildIconIdentityAdapter
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


DEFAULT_REFERENCE_DIR = os.environ.get("WILDICON_REFERENCE_DIR", "data/wildlife_eval/reference_images")
DEFAULT_PROMPT_JSON = os.environ.get("WILDICON_PROMPT_JSON", str(Path(DEFAULT_REFERENCE_DIR) / "prompts.json"))
DEFAULT_SEGMENTED_DIR = os.environ.get("WILDICON_SEGMENTED_DIR", "data/wildlife_eval/segmented_images")
DEFAULT_LOCAL_MODEL_ROOT = os.environ.get("WAN_I2V_A14B_ROOT", "checkpoints/Wan2.2-I2V-A14B")
DEFAULT_LOCAL_MODEL_BASE_PATH = "/tmp/diffsynth_models"
DEFAULT_HIGH_CHECKPOINT_DIR = os.environ.get("WILDICON_HIGH_CHECKPOINT_DIR", "models/train/Wan2.2-I2V-A14B_WildIcon_WildlifeVid_high_noise")
DEFAULT_LOW_CHECKPOINT_DIR = os.environ.get("WILDICON_LOW_CHECKPOINT_DIR", "models/train/Wan2.2-I2V-A14B_WildIcon_WildlifeVid_low_noise")
DEFAULT_OUTPUT_ROOT = os.environ.get("WILDICON_OUTPUT_ROOT", "models/validation/Wan2.2-I2V-A14B_WildIcon_WildlifeVid")
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate wildlife benchmark videos for each A14B WildIcon checkpoint.")
    parser.add_argument("--high_checkpoint_dir", type=str, default=DEFAULT_HIGH_CHECKPOINT_DIR, help="Directory containing high-noise WildIcon adapter checkpoints.")
    parser.add_argument("--low_checkpoint_dir", type=str, default=DEFAULT_LOW_CHECKPOINT_DIR, help="Directory containing low-noise WildIcon adapter checkpoints.")
    parser.add_argument("--high_checkpoint_glob", type=str, default="step-*.safetensors", help="Checkpoint glob under --high_checkpoint_dir.")
    parser.add_argument("--low_checkpoint_glob", type=str, default="step-*.safetensors", help="Checkpoint glob under --low_checkpoint_dir.")
    parser.add_argument("--stage_mode", type=str, default="low_only", choices=("high_only", "low_only", "both"), help="Which checkpoint family to render.")
    parser.add_argument("--reference_dir", type=str, default=DEFAULT_REFERENCE_DIR, help="Directory containing wildlife benchmark reference images.")
    parser.add_argument("--prompt_json", type=str, default=DEFAULT_PROMPT_JSON, help="Prompt json mapping image file names to prompt lists.")
    parser.add_argument("--segmented_dir", type=str, default=DEFAULT_SEGMENTED_DIR, help="Directory containing segmented reference images with matching file names.")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Root directory for generated videos.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip videos that are already present on disk.")
    parser.add_argument("--force", action="store_true", help="Regenerate even if a checkpoint manifest already exists.")

    parser.add_argument("--local_model_root", type=str, default=DEFAULT_LOCAL_MODEL_ROOT, help="Local Wan2.2-I2V-A14B root.")
    parser.add_argument("--local_model_base_path", type=str, default=DEFAULT_LOCAL_MODEL_BASE_PATH, help="Local model base path used to create Wan-AI symlink.")
    parser.add_argument("--generation_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Generation device.")
    parser.add_argument("--distributed_timeout_minutes", type=int, default=120, help="Process-group timeout used by torch.distributed when running under torchrun.")

    parser.add_argument("--height", type=int, default=480, help="Generation height.")
    parser.add_argument("--width", type=int, default=832, help="Generation width.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per generated video.")
    parser.add_argument("--fps", type=int, default=16, help="Output video fps.")
    parser.add_argument("--quality", type=int, default=5, help="Output video quality.")
    parser.add_argument("--seed", type=int, default=1, help="Base generation seed.")
    parser.add_argument("--tiled", action="store_true", default=True, help="Enable tiled generation.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of Wan sampling steps.")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="CFG scale.")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt used during generation.")

    parser.add_argument("--max_images", type=int, default=None, help="Optional cap on number of reference images.")
    parser.add_argument("--max_prompts_per_image", type=int, default=None, help="Optional cap on prompts sampled per image.")

    parser.add_argument("--wildicon_selected_block_ids", type=str, default="26,27,28,29,30,31,32,33,34,35,36,37,38,39", help="Comma-separated DiT block ids that receive identity tokens.")
    parser.add_argument("--wildicon_segmentor_type", type=str, default="external", choices=("external", "center_prior", "deeplabv3_resnet50"), help="WildIcon segmentor source used when segmented images are unavailable.")
    parser.add_argument("--wildicon_highpass_sigma", type=float, default=3.0, help="Gaussian sigma used by the high-pass branch.")
    parser.add_argument("--wildicon_encoder_hidden_dim", type=int, default=128, help="Hidden width of the lightweight high-frequency branch.")
    parser.add_argument("--wildicon_encoder_token_dim", type=int, default=256, help="Identity token width before projection.")
    parser.add_argument("--wildicon_token_grid_size", type=str, default="4,4", help="Identity token grid size, formatted as 'H,W'.")
    parser.add_argument("--wildicon_encoder_type", type=str, default="dinov3", choices=("light_cnn", "dinov3"), help="Identity encoder backbone used by WildIcon.")
    parser.add_argument("--wildicon_backbone_model_name_or_path", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m", help="Backbone model name or local path for the WildIcon visual backbone.")
    parser.add_argument("--wildicon_backbone_train_mode", type=str, default="frozen", choices=("frozen", "last_n", "full"), help="Backbone fine-tuning policy used to instantiate the adapter.")
    parser.add_argument("--wildicon_backbone_trainable_layers", type=int, default=1, help="Number of final backbone blocks to unfreeze when using `last_n` mode.")
    parser.add_argument("--wildicon_backbone_local_files_only", action="store_true", default=True, help="Load WildIcon backbone from local cache only.")
    return parser.parse_args()


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_grid_size(value: str) -> tuple[int, int]:
    parts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid token grid size: {value}")
    return parts[0], parts[1]


def safe_name(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")
    return text or "item"


def checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    step_match = re.search(r"step-(\d+)", path.name)
    if step_match:
        return 0, int(step_match.group(1)), path.name
    epoch_match = re.search(r"epoch-(\d+)", path.name)
    if epoch_match:
        return 1, int(epoch_match.group(1)), path.name
    return 2, 0, path.name


def distributed_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def init_distributed_if_needed(timeout_minutes: int) -> tuple[int, int, int]:
    rank, local_rank, world_size = distributed_env()
    if world_size > 1 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=timeout_minutes),
        )
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier_if_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_main_process(rank: int) -> bool:
    return rank == 0


def log_info(message: str, rank: int = 0, main_process_only: bool = True) -> None:
    if (not main_process_only) or is_main_process(rank):
        print(message)


def resolve_generation_device(device_arg: str, local_rank: int, world_size: int) -> str:
    if world_size <= 1:
        return device_arg
    if device_arg == "cuda":
        return f"cuda:{local_rank}"
    if device_arg.startswith("cuda:"):
        return device_arg
    return device_arg


def shard_prompt_records(prompt_records: list[dict[str, str | int]], rank: int, world_size: int) -> list[dict[str, str | int]]:
    if world_size <= 1:
        return prompt_records
    return prompt_records[rank::world_size]


def setup_local_model_root(args: argparse.Namespace) -> None:
    rank, _, world_size = distributed_env()
    local_model_base_path = Path(args.local_model_base_path)
    local_model_root = Path(args.local_model_root)
    target = local_model_base_path / "Wan-AI" / "Wan2.2-I2V-A14B"
    if world_size > 1:
        if is_main_process(rank):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or target.exists():
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
            target.symlink_to(local_model_root)
        barrier_if_distributed()
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            try:
                target.unlink()
            except FileNotFoundError:
                pass
        target.symlink_to(local_model_root)
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(local_model_base_path)
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"


def build_pipeline(args: argparse.Namespace) -> WanVideoPipeline:
    setup_local_model_root(args)
    local_model_root = Path(args.local_model_root)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.generation_device,
        tokenizer_config=ModelConfig(path=str(local_model_root / "google" / "umt5-xxl")),
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        redirect_common_files=False,
    )
    pipe.wildicon_adapter = WildIconIdentityAdapter(
        context_dim=pipe.dit.dim,
        selected_block_ids=parse_int_list(args.wildicon_selected_block_ids),
        segmentor_type=args.wildicon_segmentor_type,
        highpass_sigma=args.wildicon_highpass_sigma,
        encoder_hidden_dim=args.wildicon_encoder_hidden_dim,
        encoder_token_dim=args.wildicon_encoder_token_dim,
        token_grid_size=parse_grid_size(args.wildicon_token_grid_size),
        encoder_type=args.wildicon_encoder_type,
        backbone_model_name_or_path=args.wildicon_backbone_model_name_or_path,
        backbone_train_mode=args.wildicon_backbone_train_mode,
        backbone_trainable_layers=args.wildicon_backbone_trainable_layers,
        backbone_local_files_only=args.wildicon_backbone_local_files_only,
    ).to(device=args.generation_device)
    pipe.wildicon_adapter.eval()
    pipe.dit.wildicon_selected_block_ids = pipe.wildicon_adapter.selected_block_ids
    if pipe.dit2 is not None:
        pipe.dit2.wildicon_selected_block_ids = pipe.wildicon_adapter.selected_block_ids
    return pipe


def collect_checkpoint_paths(checkpoint_dir: Path, pattern: str) -> list[Path]:
    paths = [Path(path) for path in glob.glob(str(checkpoint_dir / pattern))]
    paths = [path for path in paths if path.is_file()]
    return sorted(paths, key=checkpoint_sort_key)


def load_prompt_records(args: argparse.Namespace) -> list[dict[str, str | int]]:
    reference_dir = Path(args.reference_dir).resolve()
    segmented_dir = Path(args.segmented_dir).resolve() if args.segmented_dir is not None else None
    prompts = json.load(open(args.prompt_json, "r", encoding="utf-8"))
    records: list[dict[str, str | int]] = []
    skipped_missing_references: list[str] = []
    image_names = sorted(prompts.keys())
    if args.max_images is not None:
        image_names = image_names[: args.max_images]

    for image_name in image_names:
        reference_path = reference_dir / image_name
        if not reference_path.is_file():
            alt_matches = sorted(reference_dir.glob(Path(image_name).stem + ".*"))
            if not alt_matches:
                skipped_missing_references.append(image_name)
                continue
            reference_path = alt_matches[0]
        segmented_path = None
        if segmented_dir is not None:
            candidate = segmented_dir / reference_path.name
            if candidate.is_file():
                segmented_path = candidate
            elif args.wildicon_segmentor_type == "external":
                raise FileNotFoundError(f"Segmented image not found for {reference_path.name} under {segmented_dir}")
        prompt_list = prompts[image_name]
        if isinstance(prompt_list, dict):
            prompt_list = prompt_list.get("prompts")
        if isinstance(prompt_list, str):
            prompt_list = [prompt_list]
        if not isinstance(prompt_list, list) or not prompt_list or not all(isinstance(p, str) and p.strip() for p in prompt_list):
            raise ValueError(f"Expected nonempty prompts for {image_name}")
        if args.max_prompts_per_image is not None:
            prompt_list = prompt_list[: args.max_prompts_per_image]
        for prompt_index, prompt_text in enumerate(prompt_list, start=1):
            records.append(
                {
                    "reference_image": str(reference_path),
                    "segmented_image": "" if segmented_path is None else str(segmented_path),
                    "prompt_text": str(prompt_text),
                    "image_name": reference_path.name,
                    "prompt_index": prompt_index,
                }
            )
    if skipped_missing_references:
        print(
            f"[WARN] skipped {len(skipped_missing_references)} prompt entries because their reference images were missing under {reference_dir}"
        )
        for image_name in skipped_missing_references[:20]:
            print(f"[WARN] missing reference image: {image_name}")
        if len(skipped_missing_references) > 20:
            print(f"[WARN] ... and {len(skipped_missing_references) - 20} more missing reference images")
    return records


def write_manifest(manifest_rows: list[dict[str, str | int]], checkpoint_output_dir: Path) -> None:
    manifest_path = checkpoint_output_dir / "generation_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "checkpoint_name",
                "checkpoint_path",
                "checkpoint_stage",
                "image_name",
                "prompt_index",
                "prompt_text",
                "reference_image",
                "segmented_image",
                "video_path",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


def write_rank_manifest(manifest_rows: list[dict[str, str | int]], checkpoint_output_dir: Path, rank: int) -> Path:
    manifest_path = checkpoint_output_dir / f"generation_manifest.rank{rank}.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "checkpoint_name",
                "checkpoint_path",
                "checkpoint_stage",
                "image_name",
                "prompt_index",
                "prompt_text",
                "reference_image",
                "segmented_image",
                "video_path",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path


def merge_rank_manifests(checkpoint_output_dir: Path, world_size: int) -> None:
    merged_rows: list[dict[str, str]] = []
    for rank in range(world_size):
        rank_manifest = checkpoint_output_dir / f"generation_manifest.rank{rank}.csv"
        if not rank_manifest.is_file():
            continue
        with rank_manifest.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            merged_rows.extend(list(reader))
    merged_rows.sort(key=lambda row: (row["checkpoint_stage"], row["image_name"], int(row["prompt_index"]), int(row["seed"])))
    write_manifest(merged_rows, checkpoint_output_dir)


def generate_for_checkpoint(
    pipe: WanVideoPipeline,
    checkpoint_path: Path,
    checkpoint_stage: str,
    prompt_records: list[dict[str, str | int]],
    args: argparse.Namespace,
    output_root: Path,
    rank: int,
    world_size: int,
) -> None:
    checkpoint_output_dir = output_root.resolve() / checkpoint_stage / checkpoint_path.stem
    manifest_path = checkpoint_output_dir / "generation_manifest.csv"
    if manifest_path.is_file() and not args.force:
        log_info(f"[INFO] skipping {checkpoint_stage}/{checkpoint_path.name} because manifest already exists", rank=rank)
        return

    checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    generated_video_dir = checkpoint_output_dir / "generated_videos"
    generated_video_dir.mkdir(parents=True, exist_ok=True)

    log_info(f"[INFO][rank {rank}] loading {checkpoint_stage} checkpoint: {checkpoint_path}", rank=rank, main_process_only=False)
    state_dict = load_state_dict(str(checkpoint_path))
    missing, unexpected = pipe.wildicon_adapter.load_state_dict(state_dict, strict=False)
    trainable = {name for name, _ in pipe.wildicon_adapter.named_parameters() if not name.startswith("identity_encoder.model.")}
    missing_trainable = trainable.intersection(missing)
    if missing_trainable or unexpected:
        raise ValueError(f"Incompatible adapter checkpoint: missing={sorted(missing_trainable)}, unexpected={unexpected}")
    pipe.wildicon_adapter.eval()

    manifest_rows: list[dict[str, str | int]] = []
    local_prompt_records = shard_prompt_records(prompt_records, rank, world_size)
    for local_index, record in enumerate(local_prompt_records):
        reference_path = Path(str(record["reference_image"]))
        segmented_value = str(record["segmented_image"])
        segmented_path = None if segmented_value == "" else Path(segmented_value)
        prompt_text = str(record["prompt_text"])
        prompt_index = int(record["prompt_index"])
        global_index = rank + local_index * world_size
        seed = args.seed + global_index

        stem = safe_name(reference_path.stem)
        video_name = f"{stem}__p{prompt_index:02d}__seed{seed}.mp4"
        video_path = generated_video_dir / video_name

        if not (args.skip_existing and video_path.is_file()):
            input_image = Image.open(reference_path).convert("RGB")
            reference_image = Image.open(reference_path).convert("RGB")
            segmented_image = None if segmented_path is None else Image.open(segmented_path).convert("RGB")
            log_info(
                f"[INFO][rank {rank}] generating {checkpoint_stage}/{checkpoint_path.stem} :: {reference_path.name} :: prompt {prompt_index}",
                rank=rank,
                main_process_only=False,
            )
            video = pipe(
                prompt=prompt_text,
                negative_prompt=args.negative_prompt,
                input_image=input_image,
                identity_reference_image=reference_image,
                segmented_image=segmented_image,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.guidance_scale,
                seed=seed,
                tiled=args.tiled,
            )
            save_video(video, str(video_path), fps=args.fps, quality=args.quality)

        manifest_rows.append(
            {
                "checkpoint_name": checkpoint_path.stem,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_stage": checkpoint_stage,
                "image_name": str(record["image_name"]),
                "prompt_index": prompt_index,
                "prompt_text": prompt_text,
                "reference_image": str(reference_path),
                "segmented_image": "" if segmented_path is None else str(segmented_path),
                "video_path": str(video_path),
                "seed": seed,
            }
        )

    write_rank_manifest(manifest_rows, checkpoint_output_dir, rank)
    barrier_if_distributed()
    if is_main_process(rank):
        merge_rank_manifests(checkpoint_output_dir, world_size)
    barrier_if_distributed()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = init_distributed_if_needed(args.distributed_timeout_minutes)
    args.generation_device = resolve_generation_device(args.generation_device, local_rank, world_size)

    try:
        output_root = Path(args.output_root)
        if is_main_process(rank):
            output_root.mkdir(parents=True, exist_ok=True)
        barrier_if_distributed()

        prompt_records = load_prompt_records(args)
        if not prompt_records:
            raise RuntimeError("No prompt records were loaded.")

        high_checkpoint_paths = collect_checkpoint_paths(Path(args.high_checkpoint_dir), args.high_checkpoint_glob)
        low_checkpoint_paths = collect_checkpoint_paths(Path(args.low_checkpoint_dir), args.low_checkpoint_glob)

        if args.stage_mode == "high_only":
            checkpoint_jobs = [("high_noise", path) for path in high_checkpoint_paths]
        elif args.stage_mode == "low_only":
            checkpoint_jobs = [("low_noise", path) for path in low_checkpoint_paths]
        else:
            checkpoint_jobs = [("high_noise", path) for path in high_checkpoint_paths] + [("low_noise", path) for path in low_checkpoint_paths]

        if not checkpoint_jobs:
            raise RuntimeError("No checkpoints matched the requested stage / glob configuration.")

        log_info(f"[INFO] stage_mode     : {args.stage_mode}", rank=rank)
        log_info(f"[INFO] high_ckpts     : {len(high_checkpoint_paths)}", rank=rank)
        log_info(f"[INFO] low_ckpts      : {len(low_checkpoint_paths)}", rank=rank)
        log_info(f"[INFO] output_root    : {output_root}", rank=rank)
        log_info(f"[INFO] prompt_records : {len(prompt_records)}", rank=rank)
        log_info(f"[INFO] generation_dev : {args.generation_device}", rank=rank)
        log_info(f"[INFO] world_size     : {world_size}", rank=rank)

        pipe = build_pipeline(args)
        for checkpoint_stage, checkpoint_path in checkpoint_jobs:
            generate_for_checkpoint(
                pipe=pipe,
                checkpoint_path=checkpoint_path,
                checkpoint_stage=checkpoint_stage,
                prompt_records=prompt_records,
                args=args,
                output_root=output_root,
                rank=rank,
                world_size=world_size,
            )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
