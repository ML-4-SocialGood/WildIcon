"""Foreground masks (SAM 3, prompt "Animal", as in release/WildIcon/wildicon/segmentation.py)
and Gaussian high-pass maps B(x) = x - K*x (as in wildicon/identity.py) for the page's frequency lens."""
import sys, json, numpy as np, torch
from PIL import Image, ImageFilter
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

src, out = sys.argv[1], sys.argv[2]
ids = ["01_tiger", "05_nyala", "06_zebra", "09_hyena", "11_stoat"]
model = build_sam3_image_model(device="cuda").to("cuda")
proc = Sam3Processor(model, device=torch.device("cuda"))
W = 1200
meta = []
for i in ids:
    im = Image.open(f"{src}/{i}.jpg").convert("RGB")
    st = proc.set_image(im)
    o = proc.set_text_prompt(state=st, prompt="Animal")
    sc = o["scores"].float().cpu().numpy().reshape(-1)
    m = o["masks"].cpu(); m = m.reshape(-1, *m.shape[-2:]).numpy()
    b = int(np.argmax(sc)); mask = m[b].astype(np.uint8)
    if mask.shape != (im.height, im.width):
        mask = np.asarray(Image.fromarray(mask * 255).resize(im.size, Image.NEAREST)) // 255
    # display resolution
    w = min(W, im.width); h = round(im.height * w / im.width)
    im = im.resize((w, h), Image.LANCZOS)
    mask = np.asarray(Image.fromarray(mask * 255).resize((w, h), Image.NEAREST), dtype=np.float32) / 255
    rgb = np.asarray(im, dtype=np.float32)
    seg = rgb * mask[..., None]
    # same operator as the model; sigma scaled from 832-px model width to this width
    sigma = 3.0 * w / 832
    t = torch.from_numpy(seg).permute(2, 0, 1)[None]
    r = max(1, int(np.ceil(3 * sigma))); x = torch.arange(-r, r + 1, dtype=torch.float32)
    k1 = torch.exp(-x**2 / (2 * sigma**2)); k1 /= k1.sum(); k = torch.outer(k1, k1)[None, None].expand(3, 1, -1, -1)
    blur = torch.nn.functional.conv2d(torch.nn.functional.pad(t, (r, r, r, r), mode="replicate"), k, groups=3)
    hf = (t - blur)[0].permute(1, 2, 0).numpy()
    mag = np.abs(hf).mean(-1) * mask
    mag = np.clip(mag / np.percentile(mag[mask > 0], 99.0), 0, 1) ** 0.75
    im.save(f"{out}/{i}_rgb.jpg", quality=88)
    Image.fromarray(seg.round().astype(np.uint8)).save(f"{out}/{i}_seg.jpg", quality=88)
    Image.fromarray((mag * 255).round().astype(np.uint8)).save(f"{out}/{i}_hf.jpg", quality=90)
    meta.append(dict(id=i, width=w, height=h, score=float(sc[b]), fg_fraction=float(mask.mean()), sigma=sigma))
    print(meta[-1], flush=True)
json.dump(meta, open(f"{out}/lens.json", "w"), indent=1)
