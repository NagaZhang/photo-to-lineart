"""Model selection evaluation (Task 2).

Renders every input image with two mature open-source line extractors at
multiple processing resolutions, on CPU, and records wall-clock timings:

  * LineartAnimeDetector  - Anime2Sketch / informative-drawings generator,
                            weights `netG.pth` from lllyasviel/Annotators
  * AnyLine (TEED/TED)    - weights MTEED.pth from TheMistoAI/MistoLine,
                            architecture reused from controlnet_aux

Outputs (black lines on white background) are written to samples/out/ and a
comparison grid per image to samples/grid_<stem>.png. Timings are appended
to samples/out/timings.txt.

Usage (China network):
    set HF_ENDPOINT=https://hf-mirror.com
    .venv\\Scripts\\python.exe backend\\eval_models.py
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download

from controlnet_aux import LineartAnimeDetector
from controlnet_aux.teed.ted import TED
from controlnet_aux.util import HWC3, resize_image, safe_step

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "samples" / "input"
OUT_DIR = ROOT / "samples" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANIME_RES = [512, 768, 1024]
ANYLINE_RES = [512, 1024, 1280]


def load_anime():
    return LineartAnimeDetector.from_pretrained("lllyasviel/Annotators").to("cpu")


def load_anyline():
    model_path = hf_hub_download(
        "TheMistoAI/MistoLine", "MTEED.pth", subfolder="Anyline"
    )
    model = TED()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def anyline_forward(model, input_image, detect_resolution, safe_steps=2):
    """Same pipeline as controlnet_aux TEEDdetector.__call__."""
    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image, dtype=np.uint8)
    oh, ow, _ = input_image.shape
    x = resize_image(HWC3(input_image), detect_resolution)
    h, w, _ = x.shape
    with torch.no_grad():
        t = torch.from_numpy(x.copy()).float()
        t = t.permute(2, 0, 1).unsqueeze(0)
        edges = model(t)
        edges = [e.detach().numpy().astype(np.float32)[0, 0] for e in edges]
        edges = [cv2.resize(e, (w, h), interpolation=cv2.INTER_LINEAR) for e in edges]
        edge = 1 / (1 + np.exp(-np.mean(np.stack(edges, 2), 2).astype(np.float64)))
        if safe_steps:
            edge = safe_step(edge, safe_steps)
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
    edge = cv2.resize(edge, (ow, oh), interpolation=cv2.INTER_LINEAR)
    return 255 - edge  # product convention: black lines on white


def anime_forward(model, pil_img, detect_resolution):
    # Wrapper returns white-on-black (ControlNet convention); invert back.
    out = model(
        pil_img,
        detect_resolution=detect_resolution,
        image_resolution=detect_resolution,
        output_type="np",
    )
    return 255 - np.array(out)[..., 0]


def timed(fn, repeats=1):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = fn()
        ts.append(time.perf_counter() - t0)
    return r, ts


def make_grid(stem, input_img, results):
    """results: list of (label, np.uint8 black-on-white)."""
    th = 480
    imgs = [("input", input_img.convert("RGB"))]
    for label, arr in results:
        imgs.append((label, Image.fromarray(arr).convert("RGB")))
    tiles = []
    for label, im in imgs:
        scale = th / im.height
        tile = im.resize((int(im.width * scale), th), Image.LANCZOS)
        canvas = Image.new("RGB", (tile.width, th + 28), "white")
        canvas.paste(tile, (0, 28))
        d = ImageDraw.Draw(canvas)
        d.text((6, 6), label, fill="black")
        tiles.append(canvas)
    gap = 8
    w = sum(t.width for t in tiles) + gap * (len(tiles) - 1)
    grid = Image.new("RGB", (w, th + 28), (235, 235, 235))
    x = 0
    for t in tiles:
        grid.paste(t, (x, 0))
        x += t.width + gap
    grid.save(OUT_DIR / f"grid_{stem}.png")


def main():
    inputs = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not inputs:
        print("no input images in", INPUT_DIR)
        sys.exit(1)

    print("loading lineart_anime weights ...", flush=True)
    anime = load_anime()
    print("loading anyline weights ...", flush=True)
    anyline = load_anyline()

    log_lines = []
    for path in inputs:
        stem = path.stem
        pil = Image.open(path).convert("RGB")
        print(f"== {path.name} {pil.size}", flush=True)
        results = []

        for res in ANIME_RES:
            arr, ts = timed(lambda r=res: anime_forward(anime, pil, r))
            label = f"anime{res}"
            Image.fromarray(arr).save(OUT_DIR / f"{stem}__{label}.png")
            results.append((label, arr))
            line = f"{path.name} {label} {ts[0]:.2f}s"
            print(line, flush=True)
            log_lines.append(line)

        for res in ANYLINE_RES:
            arr, ts = timed(lambda r=res: anyline_forward(anyline, pil, r))
            label = f"anyline{res}"
            Image.fromarray(arr).save(OUT_DIR / f"{stem}__{label}.png")
            results.append((label, arr))
            line = f"{path.name} {label} {ts[0]:.2f}s"
            print(line, flush=True)
            log_lines.append(line)

        make_grid(stem, pil, results)

    (OUT_DIR / "timings.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print("done ->", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
