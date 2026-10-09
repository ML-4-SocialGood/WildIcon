"""Small copies of the reference photographs for the hero wall and the gallery.

    python tools/make_thumbs.py

Writes static/thumb/<id>.jpg (long side 640 px) for every item in static/data.js.
The full-size originals in static/ref/ are still used in the carousel and the viewer.
"""
import os, sys
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import load  # noqa: E402

out = os.path.join(ROOT, "static", "thumb")
os.makedirs(out, exist_ok=True)
for it in load(os.path.join(ROOT, "static", "data.js"))["items"]:
    im = ImageOps.exif_transpose(Image.open(os.path.join(ROOT, it["reference"]))).convert("RGB")
    im.thumbnail((640, 640), Image.LANCZOS)
    im.save(os.path.join(out, it["id"] + ".jpg"), quality=84, optimize=True, progressive=True)
print("wrote", len(os.listdir(out)), "thumbnails to", out)
