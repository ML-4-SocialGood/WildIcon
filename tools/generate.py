#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 WildIcon authors.
# Distributed without any warranty; see LICENSE and COPYING.

"""Generate videos with one explicitly selected WildIcon checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


ENTRY_POINT = "examples/wanvideo/model_training/validate_full/Wan2.2-I2V-A14B-WildIcon-WildlifeEval.py"
GENERATION_KEYS = {
    "height", "width", "num_frames", "fps", "quality", "seed",
    "num_inference_steps", "guidance_scale",
}
ADAPTER_KEYS = {
    "selected_block_ids", "highpass_sigma", "encoder_hidden_dim",
    "encoder_token_dim", "token_grid_size",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Animal generation JSON config.")
    parser.add_argument("--diffsynth-root", type=Path, required=True, help="Pinned DiffSynth-Studio checkout with the WildIcon overlay installed.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained WildIcon adapter .safetensors file.")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--segmented-dir", type=Path, required=True, help="Locally prepared foregrounds with the same names as the references.")
    parser.add_argument("--prompt-json", type=Path, required=True, help="Reference file names mapped to motion prompts.")
    parser.add_argument("--wan-model-root", type=Path, required=True)
    parser.add_argument("--dino-model", type=Path, required=True, help="Local DINOv3 ViT-L/16 encoder directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=1, help="Use torchrun when greater than 1; shard references across GPUs.")
    parser.add_argument("--device", default="cuda", help="Generation device for a single process.")
    rerun = parser.add_mutually_exclusive_group()
    rerun.add_argument("--force", action="store_true", help="Regenerate an existing checkpoint run.")
    rerun.add_argument("--resume", action="store_true", help="Finish a partial run, retaining valid videos already on disk.")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and print the command without loading models or creating outputs.")
    return parser.parse_args()


def require_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"File not found or empty: {path}")
    return path


def require_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Directory not found: {path}")
    return path


def load_config(path: Path) -> dict:
    config = json.loads(require_file(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("dataset"), str) or not config["dataset"].strip():
        raise ValueError("Config must name a dataset.")
    if config.get("model") != "Wan2.2-I2V-A14B":
        raise ValueError("This entry point requires model=Wan2.2-I2V-A14B.")
    generation = config.get("generation", {})
    adapter = config.get("identity_adapter", {})
    if not isinstance(generation, dict) or not isinstance(adapter, dict) or set(generation) != GENERATION_KEYS or set(adapter) != ADAPTER_KEYS:
        raise ValueError("Config must specify all generation and identity_adapter fields; see configs/generation/.")
    for key in ("height", "width", "num_frames", "fps", "num_inference_steps", "quality"):
        if type(generation[key]) is not int or generation[key] <= 0:
            raise ValueError(f"generation.{key} must be a positive integer.")
    if generation["height"] % 16 or generation["width"] % 16 or generation["num_frames"] % 4 != 1:
        raise ValueError("Wan requires dimensions divisible by 16 and num_frames=4n+1.")
    if type(generation["seed"]) is not int or generation["seed"] < 0:
        raise ValueError("generation.seed must be a nonnegative integer.")
    if type(generation["guidance_scale"]) not in (int, float) or not math.isfinite(generation["guidance_scale"]) or generation["guidance_scale"] <= 0:
        raise ValueError("generation.guidance_scale must be positive.")
    blocks = adapter["selected_block_ids"]
    grid = adapter["token_grid_size"]
    if not isinstance(blocks, list) or not blocks or any(type(n) is not int or not 0 <= n < 40 for n in blocks) or len(set(blocks)) != len(blocks):
        raise ValueError("identity_adapter.selected_block_ids must contain distinct block indices in [0, 39].")
    if not isinstance(grid, list) or len(grid) != 2 or any(type(n) is not int or n <= 0 for n in grid):
        raise ValueError("identity_adapter.token_grid_size must contain two positive integers.")
    for key in ("encoder_hidden_dim", "encoder_token_dim"):
        if type(adapter[key]) is not int or adapter[key] <= 0:
            raise ValueError(f"identity_adapter.{key} must be a positive integer.")
    if type(adapter["highpass_sigma"]) not in (int, float) or not math.isfinite(adapter["highpass_sigma"]) or adapter["highpass_sigma"] <= 0:
        raise ValueError("identity_adapter.highpass_sigma must be positive.")
    if type(config.get("videos_per_reference")) is not int or config["videos_per_reference"] <= 0:
        raise ValueError("videos_per_reference must be a positive integer.")
    return config


def check_prompts(path: Path, reference_dir: Path, segmented_dir: Path, count: int) -> int:
    prompts = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError("Prompt JSON must be a nonempty mapping of reference file names to prompts.")
    stems: set[str] = set()
    for name, values in prompts.items():
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Use a file name directly inside --reference-dir: {name!r}")
        require_file(reference_dir / name)
        require_file(segmented_dir / name)
        if isinstance(values, dict):
            values = values.get("prompts")
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list) or len(values) != count or any(not isinstance(p, str) or not p.strip() for p in values):
            raise ValueError(f"Expected exactly {count} nonempty prompts for {name}.")
        stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(name).stem).strip("_") or "item"
        if stem in stems:
            raise ValueError(f"Reference names produce the same output stem: {name}")
        stems.add(stem)
    return len(prompts)


def prepare_command(args: argparse.Namespace) -> tuple[list[str], Path, dict]:
    config = load_config(args.config)
    root = require_dir(args.diffsynth_root)
    entry = require_file(root / ENTRY_POINT)
    checkpoint = require_file(args.checkpoint)
    if checkpoint.suffix != ".safetensors":
        raise ValueError("--checkpoint must be an adapter .safetensors file.")
    if '"--checkpoint"' not in entry.read_text(encoding="utf-8"):
        raise ValueError("Reapply tools/apply_overlay.sh: the installed inference entry point lacks --checkpoint.")
    references = require_dir(args.reference_dir)
    foregrounds = require_dir(args.segmented_dir)
    prompt_json = require_file(args.prompt_json)
    num_references = check_prompts(prompt_json, references, foregrounds, config["videos_per_reference"])
    wan_root = require_dir(args.wan_model_root)
    dino_model = require_dir(args.dino_model)
    for name in ("high_noise_model", "low_noise_model"):
        directory = require_dir(wan_root / name)
        shards = sorted(directory.glob("diffusion_pytorch_model*.safetensors"))
        if not shards:
            raise ValueError(f"No Wan model shards found in {directory}")
        for shard in shards:
            require_file(shard)
    for name in ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"):
        require_file(wan_root / name)
    require_dir(wan_root / "google" / "umt5-xxl")
    require_file(dino_model / "config.json")
    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be at least 1.")
    if args.num_gpus > 1 and args.device != "cuda":
        raise ValueError("Use --device cuda with --num-gpus greater than 1.")
    output_dir = args.output_dir.expanduser().resolve()
    command = [sys.executable]
    if args.num_gpus > 1:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.num_gpus}"]
    command += [
        str(entry), "--checkpoint", str(checkpoint),
        "--reference_dir", str(references), "--segmented_dir", str(foregrounds),
        "--prompt_json", str(prompt_json), "--local_model_root", str(wan_root),
        "--local_model_base_path", str(output_dir / ".model-links"),
        "--wildicon_backbone_model_name_or_path", str(dino_model),
        "--wildicon_segmentor_type", "external", "--wildicon_encoder_type", "dinov3",
        "--wildicon_backbone_train_mode", "frozen", "--wildicon_backbone_local_files_only",
        "--generation_device", args.device, "--output_root", str(output_dir),
    ]
    for key, value in config["generation"].items():
        command += [f"--{key}", str(value)]
    for key, value in config["identity_adapter"].items():
        command += [f"--wildicon_{key}", ",".join(map(str, value)) if isinstance(value, list) else str(value)]
    if args.force or args.resume:
        command += ["--force"]
    if args.resume:
        command += ["--skip_existing"]
    print(f"{config['dataset']}: {num_references} references, {num_references * config['videos_per_reference']} videos", flush=True)
    print(shlex.join(command), flush=True)
    return command, root, config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_run(args: argparse.Namespace, config: dict) -> Path:
    checkpoint_output = args.output_dir.expanduser().resolve() / "selected" / args.checkpoint.expanduser().resolve().stem
    request = {
        "config": config,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint.expanduser().resolve()),
        "prompt_sha256": file_sha256(args.prompt_json.expanduser().resolve()),
        "inputs": {
            key: str(getattr(args, key).expanduser().resolve())
            for key in ("reference_dir", "segmented_dir", "wan_model_root", "dino_model")
        },
    }
    prompts = json.loads(args.prompt_json.expanduser().read_text(encoding="utf-8"))
    request["reference_files"] = {
        name: {
            "reference_sha256": file_sha256(args.reference_dir.expanduser() / name),
            "foreground_sha256": file_sha256(args.segmented_dir.expanduser() / name),
        }
        for name in sorted(prompts)
    }
    request_path = checkpoint_output / "run_request.json"
    if checkpoint_output.exists() and any(checkpoint_output.iterdir()) and not args.force:
        if not request_path.is_file() or json.loads(request_path.read_text(encoding="utf-8")) != request:
            raise ValueError("Existing output uses different inputs or settings. Use a new --output-dir or --force.")
    checkpoint_output.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    return checkpoint_output


def main() -> None:
    args = parse_args()
    try:
        command, root, config = prepare_command(args)
        if args.dry_run:
            return
        checkpoint_output = prepare_run(args, config)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(command, cwd=root, env=env, check=True)
        (checkpoint_output / "animal_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
