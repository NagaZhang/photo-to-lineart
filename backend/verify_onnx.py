"""Task 3 verification: full-pipeline equivalence.

Compares the PyTorch AnyLine (TEED) reference pipeline against the ONNX
Runtime pipeline on real images end to end.

Usage:
    set HF_ENDPOINT=https://hf-mirror.com
    python backend/verify_onnx.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

import cv2
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.pipeline import LineartPipeline, resize_image  # noqa: E402
from controlnet_aux.teed.ted import TED  # noqa: E402
from controlnet_aux.util import HWC3  # noqa: E402

TEST_IMAGES = ["madoka.jpg", "anime_scene.webp", "real_photo_astronaut.png"]
THRESHOLD = 2.0 / 255.0  # mean abs difference, in 0..1


def torch_reference(pil: Image.Image, model, resolution: int) -> np.ndarray:
    x = np.asarray(pil.convert("RGB"), dtype=np.uint8)
    oh, ow = x.shape[:2]
    x = resize_image(HWC3(x), resolution)
    h, w, _ = x.shape
    with torch.no_grad():
        t = torch.from_numpy(x.copy()).float().permute(2, 0, 1).unsqueeze(0)
        edges = model(t)
        edges = [e.numpy().astype(np.float32)[0, 0] for e in edges]
        edges = [cv2.resize(e, (w, h), interpolation=cv2.INTER_LINEAR) for e in edges]
        edge = 1 / (1 + np.exp(-np.mean(np.stack(edges, 2), 2).astype(np.float64)))
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.resize(edge, (ow, oh), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    weights = hf_hub_download("TheMistoAI/MistoLine", "MTEED.pth", subfolder="Anyline")
    model = TED()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    pipe = LineartPipeline()
    out_dir = ROOT / "samples" / "out"
    worst = 0.0
    for name in TEST_IMAGES:
        pil = Image.open(ROOT / "samples" / "input" / name).convert("RGB")
        ref = torch_reference(pil, model, pipe.detect_resolution)
        onnx_edge = pipe.edge_map(pil)
        mae = float(np.abs(ref.astype(np.float32) - onnx_edge.astype(np.float32)).mean() / 255.0)
        worst = max(worst, mae)

        pair = Image.new("L", (ref.shape[1] * 2 + 12, ref.shape[0]), 255)
        pair.paste(Image.fromarray(ref), (0, 0))
        pair.paste(Image.fromarray(onnx_edge), (ref.shape[1] + 12, 0))
        pair.save(out_dir / f"verify_{Path(name).stem}.png")
        print(f"{name}: torch-vs-onnx mean abs diff = {mae:.6f} (limit {THRESHOLD:.5f})")

    assert worst <= THRESHOLD, f"exceeds threshold: {worst}"
    print(f"PASS, worst mean abs diff {worst:.6f}")


if __name__ == "__main__":
    main()
