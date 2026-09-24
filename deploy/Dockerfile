# syntax=docker/dockerfile:1
# Multi-stage build:
#   builder  - heavy torch/controlnet_aux toolchain, downloads open-source
#              weights and exports the tiny ONNX graph (not shipped in repo).
#   runtime - slim image with onnxruntime + numpy + Pillow + OpenCV only.

FROM python:3.13-slim AS builder

WORKDIR /build

COPY backend/requirements.txt backend/requirements-build.txt /build/
RUN pip install --no-cache-dir -r /build/requirements-build.txt

COPY backend /build/backend
# Downloads MTEED.pth (TheMistoAI/MistoLine) and verifies torch vs ONNX.
RUN python /build/backend/export_onnx.py --model anyline \
    --out /build/backend/assets/model.onnx

# ------------------------------- runtime ----------------------------------

FROM python:3.13-slim AS runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend \
    OMP_NUM_THREADS=2 \
    LINEART_CPU_ARENA=0 \
    LINEART_RESOLUTION=640

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend /app/backend
COPY frontend /app/frontend
# Freshly exported artifact (roughly 0.2 MB); no torch in this stage.
COPY --from=builder /build/backend/assets/model.onnx \
                    /build/backend/assets/manifest.json \
                    /app/backend/assets/

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--app-dir", "/app/backend", \
     "--host", "0.0.0.0", "--port", "8000"]
