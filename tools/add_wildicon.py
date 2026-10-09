"""Put a WildIcon output into its slot on the page.

    python tools/add_wildicon.py 01_tiger /path/to/wildicon_tiger.mp4 [more id/path pairs ...]

Re-encodes the clip for the web (H.264, yuv420p, CRF 18, +faststart, native size and frame rate),
saves a poster from the middle frame, and records the clip in static/data.js.
Only WildIcon outputs belong here.
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import load, save  # noqa: E402


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,duration", "-of", "json", path], capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    return int(s["width"]), int(s["height"]), float(s.get("duration", 0) or 0)


def main(pairs):
    data_path = os.path.join(ROOT, "static", "data.js")
    data = load(data_path)
    items = {it["id"]: it for it in data["items"]}
    os.makedirs(os.path.join(ROOT, "static", "wildicon"), exist_ok=True)
    for i, src in pairs:
        if i not in items:
            sys.exit("unknown id %s; known: %s" % (i, ", ".join(items)))
        video = "static/wildicon/%s.mp4" % i
        poster = "static/wildicon/%s.jpg" % i
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "18", "-preset", "slow", "-movflags", "+faststart", os.path.join(ROOT, video)], check=True)
        w, h, dur = probe(os.path.join(ROOT, video))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "%.3f" % (dur / 2), "-i", os.path.join(ROOT, video),
                        "-frames:v", "1", "-q:v", "3", os.path.join(ROOT, poster)], check=True)
        items[i]["wildicon"] = {"video": video, "poster": poster, "width": w, "height": h, "source": os.path.abspath(src)}
        print("%s <- %s (%dx%d, %.2fs)" % (i, src, w, h, dur))
    save(data_path, data)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or len(args) % 2:
        sys.exit(__doc__)
    main(list(zip(args[0::2], args[1::2])))
