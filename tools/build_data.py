"""Build static/data.js (reference photos + prompts) from the website selection.

    python tools/build_data.py [--ids 01_tiger 02_panda ...]

Reads outputs/website/selection_spec.json, copies each reference photo byte for byte
to static/ref/, and writes static/data.js. --all includes the complete curated selection.
WildIcon slots start empty; fill them with tools/add_wildicon.py.
Existing WildIcon and temporary preview entries in data.js are kept.
"""
import argparse, hashlib, json, os, shutil
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = "/data/yil708/GenerativeModel/outputs/website/selection_spec.json"
DEFAULT_IDS = ["01_tiger", "02_panda", "03_stoat", "04_cat", "05_nyala", "06_zebra", "07_tiger", "08_panda",
               "09_hyena", "10_cat", "11_stoat", "12_zebra", "13_bear", "14_deer", "15_elephant", "16_sheep"]
LENS = [("06_zebra", "Zebra"), ("01_tiger", "Tiger"), ("05_nyala", "Nyala"), ("09_hyena", "Hyena"), ("11_stoat", "Stoat")]


def load(path):
    with open(path) as f:
        text = f.read()
    return json.loads(text[text.index("=") + 1:].rstrip().rstrip(";"))


def save(path, data):
    with open(path, "w") as f:
        f.write("window.PAGE = " + json.dumps(data, indent=2, ensure_ascii=False) + ";\n")


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--ids", nargs="+")
    group.add_argument("--all", action="store_true", help="Include every curated video/reference pair")
    args = ap.parse_args()

    spec = {a["id"]: a for a in json.load(open(SPEC))["assets"]}
    out = os.path.join(ROOT, "static", "data.js")
    old = {it["id"]: it for it in load(out)["items"]} if os.path.exists(out) else {}
    os.makedirs(os.path.join(ROOT, "static", "ref"), exist_ok=True)

    items = []
    selected = list(spec) if args.all else (args.ids or DEFAULT_IDS)
    for i in selected:
        a = spec[i]
        extension = os.path.splitext(a["source_reference"])[1].lower()
        reference = "static/ref/%s%s" % (i, extension)
        shutil.copy2(a["source_reference"], os.path.join(ROOT, reference))
        with Image.open(a["source_reference"]) as im:
            width, height = im.size
        with open(a["source_reference"], "rb") as handle:
            reference_sha256 = hashlib.sha256(handle.read()).hexdigest()
        items.append({
            "id": i,
            "species": a["species_label"],
            "reference": reference,
            "referenceId": a["reference_id"],
            "referenceSource": a["source_reference"],
            "referenceSha256": reference_sha256,
            "refWidth": width,
            "refHeight": height,
            "prompt": a["prompt_display"],
            "wildicon": old.get(i, {}).get("wildicon"),
            "preview": old.get(i, {}).get("preview"),
        })

    lens = []
    for i, name in LENS:
        w, h = Image.open(os.path.join(ROOT, "static", "lens", i + "_rgb.jpg")).size
        lens.append({"id": i, "species": name, "width": w, "height": h})

    ids = [it["id"] for it in items]
    pick = lambda xs: [x for x in xs if x in ids]
    save(out, {
        "items": items,
        "heroWall": pick([
            "01_tiger", "32_zebra", "15_elephant", "36_hyena", "21_panda", "05_nyala", "24_stoat",
            "14_deer", "13_bear", "17_tiger", "10_cat", "06_zebra", "16_sheep", "29_nyala",
            "09_hyena", "44_elephant", "02_panda", "33_zebra", "11_stoat", "42_deer", "18_tiger",
            "38_bear", "30_nyala", "47_sheep", "37_hyena", "20_panda", "34_zebra", "45_elephant"
        ]),
        "featured": pick(["01_tiger", "05_nyala", "06_zebra", "09_hyena", "11_stoat", "04_cat", "02_panda"]),
        "gallery": ids,
        "lens": lens,
    })
    print("wrote", out, len(items), "items")


if __name__ == "__main__":
    main()
