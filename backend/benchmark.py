"""Task 7 resource benchmark.

Runs the runtime pipeline (ONNX Runtime, no torch import) on 10 generated
1080p inputs, reporting latency percentiles and peak process RSS.

Usage (from repo root):
    python backend/benchmark.py
"""
import sys
import threading
import time
from pathlib import Path

import psutil
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.pipeline import LineartPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples" / "input"
N_IMAGES = 10
TARGET = (1920, 1080)
P95_LIMIT_S = 10.0
RSS_LIMIT_MB = 512.0


def build_inputs():
    """10 deterministic 1080p RGB images derived from the evaluation samples."""
    bases = sorted(p for p in SAMPLES.iterdir() if p.suffix in {".jpg", ".png", ".webp"})
    inputs = []
    for i in range(N_IMAGES):
        src = Image.open(bases[i % len(bases)]).convert("RGB")
        # alternate between stretch and center-crop to vary aspect handling
        if i % 2 == 0:
            im = src.resize(TARGET, Image.LANCZOS)
        else:
            sw, sh = src.size
            scale = max(TARGET[0] / sw, TARGET[1] / sh)
            im2 = src.resize((round(sw * scale), round(sh * scale)), Image.LANCZOS)
            left = (im2.width - TARGET[0]) // 2
            top = (im2.height - TARGET[1]) // 2
            im = im2.crop((left, top, left + TARGET[0], top + TARGET[1]))
        inputs.append(im)
    return inputs


class RssSampler(threading.Thread):
    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.proc = psutil.Process()
        self.peak = self.proc.memory_info().rss
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self.join()


def percentile(values, q):
    values = sorted(values)
    idx = round((len(values) - 1) * q)
    return values[idx]


def main():
    assert "torch" not in sys.modules
    pipe = LineartPipeline()
    inputs = build_inputs()

    pipe.process(inputs[0])  # warmup

    sampler = RssSampler()
    sampler.start()
    timings = []
    for i, im in enumerate(inputs):
        t0 = time.perf_counter()
        out = pipe.process(im)
        timings.append(time.perf_counter() - t0)
        assert out.size == TARGET
        print(f"  image {i + 1:02d}: {timings[-1]:.2f}s")
    sampler.stop()

    p50 = percentile(timings, 0.50)
    p95 = percentile(timings, 0.95)
    peak_mb = sampler.peak / 1024 / 1024
    print(f"\nlatency p50={p50:.2f}s  p95={p95:.2f}s  (limit {P95_LIMIT_S}s)")
    print(f"peak RSS={peak_mb:.0f} MB  (limit {RSS_LIMIT_MB:.0f} MB)")

    ok = p95 <= P95_LIMIT_S and peak_mb <= RSS_LIMIT_MB
    print("BENCHMARK PASS" if ok else "BENCHMARK FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
