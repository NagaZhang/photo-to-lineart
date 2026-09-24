"""Runtime image -> lineart pipeline.

ONNX Runtime (CPU) only. Must never import torch. The preprocessing and
postprocessing are deliberately pixel-identical to the reference open-source
TEED / controlnet_aux AnyLine path:

  * resize with min-side = resolution, dimensions rounded to multiples of 64,
    INTER_LANCZOS4 when upscaling else INTER_AREA
  * raw RGB 0..255 fed to the network (no normalization)
  * mean of the multi-scale edge logits, sigmoid, continuous grayscale
    (AnyLine selected with safe_steps = 0 during model evaluation)
  * resize edge map back to the original dimensions (INTER_LINEAR)
  * invert => black lines on white background
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def resize_image(input_image: np.ndarray, resolution: int) -> np.ndarray:
    h, w = float(input_image.shape[0]), float(input_image.shape[1])
    k = float(resolution) / min(h, w)
    h2 = int(np.round(h * k / 64.0)) * 64
    w2 = int(np.round(w * k / 64.0)) * 64
    interp = cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    return cv2.resize(input_image, (w2, h2), interpolation=interp)


class LineartPipeline:
    def __init__(
        self,
        assets_dir: Path | str = ASSETS_DIR,
        detect_resolution: int | None = None,
    ) -> None:
        assets_dir = Path(assets_dir)
        self.manifest = json.loads((assets_dir / "manifest.json").read_text("utf-8"))
        if self.manifest.get("model") != "anyline":
            raise RuntimeError(
                f"unsupported model in manifest: {self.manifest.get('model')}"
            )
        self.input_multiple = int(self.manifest["input_multiple"])
        self.detect_resolution = int(
            detect_resolution or os.environ.get("LINEART_RESOLUTION", "704")
        )
        # Final white-point "levels" on the white-on-black edge map: weak
        # texture below the black point becomes pure white after inversion,
        # strong lines above the white point become pure black. black=60 was
        # chosen with the input denoise below: blurry/noisy photos otherwise
        # produce blotchy disconnected patches (see samples/out/blur_sweep_*).
        self.levels_black = int(os.environ.get("LINEART_LEVEL_BLACK", "60"))
        self.levels_white = int(os.environ.get("LINEART_LEVEL_WHITE", "200"))
        # Mild gaussian denoise before edge detection: suppresses the sensor
        # noise / JPEG grain that blurry photos turn into speckled edges.
        # 0 disables (restores the exact reference AnyLine behaviour).
        self.denoise_sigma = float(os.environ.get("LINEART_DENOISE_SIGMA", "1.0"))

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        threads = os.environ.get("OMP_NUM_THREADS")
        if threads:
            so.intra_op_num_threads = int(threads)
        # "0": arena + memory pattern off; "p": only the arena off (pattern
        # reuse stays enabled). The arena retains peak buffers for the whole
        # process lifetime, which dominates RSS on small instances.
        arena_mode = os.environ.get("LINEART_CPU_ARENA", "1")
        if arena_mode in {"0", "p"}:
            so.enable_cpu_mem_arena = False
        if arena_mode == "0":
            so.enable_mem_pattern = False
        self.session = ort.InferenceSession(
            str(assets_dir / "model.onnx"),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

    def edge_map(self, pil_image: Image.Image) -> np.ndarray:
        """Return the white-on-black continuous edge map at original size."""
        x = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
        orig_h, orig_w = x.shape[:2]

        if self.denoise_sigma > 0:
            x = cv2.GaussianBlur(x, (0, 0), self.denoise_sigma)
        x = resize_image(x, self.detect_resolution)
        h, w = x.shape[:2]
        if h % self.input_multiple or w % self.input_multiple:
            raise RuntimeError("internal resolution is not a network multiple")

        feed = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)
        outputs = self.session.run(None, {"image": feed})
        # Accumulate resized logits one at a time and stay in float32: holding
        # all full-resolution maps plus a float64 stack is needlessly large.
        acc = None
        for out in outputs:
            resized = cv2.resize(
                out[0, 0], (w, h), interpolation=cv2.INTER_LINEAR
            )
            acc = resized.copy() if acc is None else acc + resized
            del resized, out
        edge = 1.0 / (1.0 + np.exp(-(acc / len(outputs)), dtype=np.float32))
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.resize(
            edge, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )

    def process(self, pil_image: Image.Image) -> Image.Image:
        """Return a mode-L PIL image: black lines on a white background."""
        edge = self.edge_map(pil_image).astype(np.float32)
        scale = 255.0 / (self.levels_white - self.levels_black)
        edge = np.clip(
            (edge - self.levels_black) * scale, 0, 255
        ).astype(np.uint8)
        # 2-D uint8 -> Pillow infers mode "L" (no deprecated mode= argument).
        return Image.fromarray(255 - edge)
