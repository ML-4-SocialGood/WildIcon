"""Copy the curated Wan 2.2 media into the static page, retaining source metadata.

    python tools/add_preview.py

Open index.html to display the media. Existing WildIcon outputs take precedence.
Files reuse the verified web encodes without re-encoding.
"""
from pathlib import Path
import hashlib
import json
import shutil

from build_data import load, save

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT.parent / "outputs/website"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    assets = {asset["id"]: asset for asset in manifest["assets"]}
    data_path = ROOT / "static/data.js"
    data = load(data_path)
    destination = ROOT / "static/preview"
    destination.mkdir(exist_ok=True)
    mapping = []
    for item in data["items"]:
        asset = assets[item["id"]]
        assert item["referenceId"] == asset["reference_id"]
        assert sha256(ROOT / item["reference"]) == asset["reference_source_sha256"]
        video = "static/preview/%s.mp4" % item["id"]
        poster = "static/preview/%s.jpg" % item["id"]
        for source, target in [(asset["video"], video), (asset["poster"], poster)]:
            shutil.copy2(BUNDLE / source, ROOT / target)
        assert sha256(ROOT / video) == asset["web_sha256"]
        assert sha256(ROOT / poster) == asset["poster_sha256"]
        metadata = asset["web_metadata"]
        item["preview"] = {
            "model": "Wan 2.2", "video": video, "poster": poster,
            "width": metadata["width"], "height": metadata["height"],
            "fps": metadata["fps"], "frames": metadata["frames"],
            "source": asset["source_video"], "sha256": asset["web_sha256"],
        }
        mapping.append({
            "id": item["id"], "reference_id": asset["reference_id"],
            "reference": item["reference"], "source_reference": asset["source_reference"],
            "reference_sha256": asset["reference_source_sha256"],
            "video": video, "source_video": asset["source_video"],
            "video_sha256": asset["web_sha256"], "poster": poster,
            "prompt_source": asset["prompt_source"], "prompt_key": asset["prompt_key"],
            "prompt_index": asset["prompt_index"], "model": "Wan 2.2",
        })
    save(str(data_path), data)
    (destination / "manifest.json").write_text(json.dumps({
        "purpose": "Curated Wan 2.2 media displayed by the static page; exact video/reference provenance retained.",
        "source_manifest": str(BUNDLE / "manifest.json"), "items": mapping,
    }, ensure_ascii=False, indent=2) + "\n")
    print("Added %s exact video/reference pairs; open index.html" % len(mapping))


if __name__ == "__main__":
    main()
