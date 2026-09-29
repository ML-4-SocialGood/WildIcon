"""Resolve generated videos to their exact reference images and prompts."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationCase:
    video_path: Path
    reference_image: Path
    prompt_text: str
    image_stem: str
    prompt_index: int | None = None
    prompt_source_filename: str | None = None


def _existing_path(value: str, base: Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.resolve()


def load_manifest(path: Path) -> list[EvaluationCase]:
    """Paths may be absolute or relative to the manifest's directory.

    The recorded prompt text is authoritative, including for historical
    manifests whose prompt indices started at zero.
    """
    path = path.resolve()
    cases = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"video_path", "reference_image", "prompt_text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            if any(not row.get(key, "").strip() for key in required):
                raise ValueError(f"Manifest line {line} has an empty video, reference, or prompt.")
            video = _existing_path(row["video_path"], path.parent, "Video")
            reference = _existing_path(row["reference_image"], path.parent, "Reference image")
            if video in seen:
                raise ValueError(f"Duplicate video in manifest: {video}")
            seen.add(video)
            index = row.get("prompt_index", "").strip()
            cases.append(EvaluationCase(
                video, reference, row["prompt_text"], reference.stem,
                int(index) if index else None, row.get("image_name") or reference.name,
            ))
    if not cases:
        raise ValueError("The generation manifest contains no cases.")
    return cases


def load_directory(video_dir: Path, reference_dir: Path, prompts_json: Path,
                   pattern: str = "**/*.mp4") -> list[EvaluationCase]:
    """Support released WildIcon flat filenames and legacy per-image folders.

    Filename prompt numbers are one-based; prefer manifests for old runs.
    """
    references = {}
    for path in reference_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            key = path.stem.lower()
            if key in references:
                raise ValueError(f"Ambiguous reference stem: {path.stem}")
            references[key] = path.resolve()
    with prompts_json.open(encoding="utf-8") as stream:
        prompts = json.load(stream)
    prompt_index = {}
    for name, value in prompts.items():
        key = Path(name).stem.lower()
        if key in prompt_index:
            raise ValueError(f"Ambiguous prompt key: {name}")
        values = value.get("prompts") if isinstance(value, dict) else value
        values = [values] if isinstance(values, str) else values
        if not isinstance(values, list) or not all(isinstance(x, str) and x.strip() for x in values):
            raise ValueError(f"Invalid prompt list for {name}")
        prompt_index[key] = (name, values)

    cases = []
    for video in sorted(video_dir.glob(pattern)):
        if not video.is_file() or video.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            continue
        match = re.search(r"__p(\d+)__", video.name)
        if not match:
            raise ValueError(f"No prompt index in {video.name}; use --manifest.")
        prefix = video.name[:match.start()].lower()
        stem = prefix if prefix in references else video.parent.name.lower()
        if stem not in references or stem not in prompt_index:
            raise ValueError(f"Cannot match reference and prompt for {video}; use --manifest.")
        index = int(match.group(1))
        name, values = prompt_index[stem]
        if not 1 <= index <= len(values):
            raise ValueError(f"Prompt index {index} is invalid for {video}; use --manifest for zero-based runs.")
        cases.append(EvaluationCase(video.resolve(), references[stem], values[index - 1],
                                    references[stem].stem, index, name))
    if not cases:
        raise ValueError(f"No videos matched {pattern} under {video_dir}")
    return cases
