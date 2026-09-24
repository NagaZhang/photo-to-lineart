"""Build-stage ONNX exporter (Task 3).

Downloads the selected pretrained weights (mature open-source models, no
self-trained weights are ever created) and exports the inference network to
ONNX, then numerically verifies ONNX Runtime vs PyTorch on random tensors.

Supported models:
  anime   UnetGenerator (Anime2Sketch / informative-drawings), weights
          netG.pth from lllyasviel/Annotators.
          Input  : NCHW RGB float in [0,255], H/W multiples of 256.
          Output : 1-channel raw network output (tanh space); the service
                   applies *127.5+127.5 (=> black lines on white).
  anyline TED with MTEED.pth from TheMistoAI/MistoLine (AnyLine).
          Input  : NCHW RGB float in [0,255], H/W multiples of 8.
          Output : 4 raw edge logits, postprocessed by the service.

Usage:
    python backend/export_onnx.py --model anyline --out backend/assets/model.onnx
"""
import argparse
import functools
import json
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from controlnet_aux.lineart_anime import UnetGenerator
from controlnet_aux.teed.ted import TED

ASSETS = Path(__file__).resolve().parent / "assets"

MODEL_SPECS = {
    "anime": {"multiple": 256, "outputs": 1},
    "anyline": {"multiple": 8, "outputs": 4},
}


def build_anime():
    path = hf_hub_download("lllyasviel/Annotators", "netG.pth")
    norm = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    net = UnetGenerator(3, 1, 8, 64, norm_layer=norm, use_dropout=False)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    for key in list(ckpt.keys()):
        if "module." in key:
            ckpt[key.replace("module.", "")] = ckpt[key]
            del ckpt[key]
    net.load_state_dict(ckpt)
    net.eval()
    return net


def build_anyline():
    path = hf_hub_download("TheMistoAI/MistoLine", "MTEED.pth", subfolder="Anyline")
    net = TED()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net


BUILDERS = {"anime": build_anime, "anyline": build_anyline}


def export(model_name: str, out_path: Path, opset: int):
    spec = MODEL_SPECS[model_name]
    net = BUILDERS[model_name]()
    dummy = torch.randn(1, 3, 512, 512, dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            net,
            (dummy,),
            str(out_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["image"],
            output_names=(["line"] if spec["outputs"] == 1
                          else [f"edge{i}" for i in range(spec["outputs"])]),
            dynamic_axes={"image": {2: "height", 3: "width"}},
            dynamo=False,
        )

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    manifest = {
        "model": model_name,
        "opset": opset,
        "input_multiple": spec["multiple"],
        "input_range": "0_255",
        "outputs": spec["outputs"],
        "source": (
            "lllyasviel/Annotators netG.pth (Anime2Sketch/informative-drawings)"
            if model_name == "anime"
            else "TheMistoAI/MistoLine Anyline/MTEED.pth (TEED)"
        ),
    }
    (out_path.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"exported {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    return net


def verify_torch_vs_onnx(model_name, net, onnx_path):
    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    sizes = [(512, 512), (640, 768), (1024, 768)]
    spec = MODEL_SPECS[model_name]
    m = spec["multiple"]
    worst = 0.0
    with torch.no_grad():
        for h0, w0 in sizes:
            h, w = (h0 // m) * m, (w0 // m) * m
            x = torch.rand(1, 3, h, w, dtype=torch.float32) * 255
            yt = net(x)
            if isinstance(yt, (list, tuple)):
                yt = [t.numpy() for t in yt]
            else:
                yt = [yt.numpy()]
            yo = sess.run(None, {"image": x.numpy()})
            for a, b in zip(yt, yo):
                mae = float(np.abs(a - b).mean())
                worst = max(worst, mae)
            print(f"verify {h}x{w}: outputs={len(yt)} mean_abs_diff={mae:.6f}")
    assert worst < 1e-3, f"ONNX mismatch, worst mean abs diff = {worst}"
    print(f"verification passed (worst mean abs diff {worst:.6f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(BUILDERS), default="anyline")
    ap.add_argument("--out", default=str(ASSETS / "model.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    out = Path(args.out)
    net = export(args.model, out, args.opset)
    verify_torch_vs_onnx(args.model, net, out)


if __name__ == "__main__":
    main()
