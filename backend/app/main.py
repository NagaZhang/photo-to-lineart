"""FastAPI service: upload an image, get a black-on-white lineart PNG back.

User images are processed transiently and are never persisted by this app.
Inference runs in a worker thread behind a semaphore so a small CPU instance
handles one heavy request at a time (others queue).
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .pipeline import LineartPipeline

logger = logging.getLogger("lineart")
logging.basicConfig(level=logging.INFO)

MAX_FILE_BYTES = 10 * 1024 * 1024
# Multipart framing overhead allowance for the Content-Length pre-check.
BODY_SLACK_BYTES = 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024
MAX_SIDE = 2048
# Reject decompression-bomb style payloads before allocating decoded buffers.
MAX_DECODE_PIXELS = 25_000_000
Image.MAX_IMAGE_PIXELS = MAX_DECODE_PIXELS
ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}

BACKEND_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"

_pipeline: LineartPipeline | None = None
_inference_semaphore = asyncio.Semaphore(1)


def get_pipeline() -> LineartPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LineartPipeline()
    return _pipeline


def current_rss_mb() -> float | None:
    """RSS in MB from /proc (Linux containers); None where unavailable."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/self/statm", "r", encoding="ascii") as fh:
            rss_pages = int(fh.read().split()[1])
        return rss_pages * page_size / 1024 / 1024
    except (OSError, ValueError, AttributeError):
        return None


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the ONNX model once at boot (also makes the first request fast).
    get_pipeline()
    logger.info("lineart pipeline ready")
    yield


app = FastAPI(
    title="Photo to Lineart",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


def too_large_response():
    return JSONResponse(
        status_code=413,
        content={
            "detail": f"图片过大，请上传 {MAX_FILE_BYTES // 1024 // 1024}MB 以内的文件"
        },
    )


def looks_like_image(ext: str, data: bytes) -> bool:
    if ext not in ALLOWED_EXT:
        return False
    if data.startswith(b"\xff\xd8\xff") and ext in {"jpg", "jpeg"}:
        return True
    if data.startswith(b"\x89PNG\r\n\x1a\n") and ext == "png":
        return True
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP" and ext == "webp":
        return True
    return False


def decode_and_normalize(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        # Dimensions come from the header; reject before allocating pixels.
        if img.width * img.height > MAX_DECODE_PIXELS:
            raise ValueError("图片像素过大，请上传边长更小的图片")
        img.load()
    except Image.DecompressionBombError as exc:
        raise ValueError("图片像素过大，请上传边长更小的图片") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and "像素过大" in str(exc):
            raise
        raise ValueError("图片已损坏或无法解码") from exc

    img = img.convert("RGB")
    w, h = img.size
    scale = MAX_SIDE / max(w, h)
    if scale < 1:
        img = img.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS
        )
    return img


@app.exception_handler(Exception)
async def unhandled_handler(_, exc: Exception):
    logger.exception("unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "服务器内部错误，请稍后重试"})


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/sketch")
async def sketch(request: Request, file: UploadFile = File(...)):
    filename = (file.filename or "").strip()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXT or file.content_type not in ALLOWED_MIME:
        return JSONResponse(
            status_code=400,
            content={"detail": "仅支持 JPG / PNG / WebP 格式的图片"},
        )

    # Hard pre-check: never buffer an obviously oversized request body.
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_FILE_BYTES + BODY_SLACK_BYTES:
                return too_large_response()
        except ValueError:
            pass

    # Bounded streaming read: bail out as soon as the payload exceeds the
    # limit instead of accumulating a huge upload in memory.
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_FILE_BYTES:
            return too_large_response()
        chunks.append(chunk)
    data = b"".join(chunks)

    if not data:
        return JSONResponse(status_code=400, content={"detail": "上传内容为空"})
    if not looks_like_image(ext, data[:16]):
        return JSONResponse(
            status_code=400,
            content={"detail": "文件内容与图片格式不符，请上传真实的图片文件"},
        )

    try:
        img = decode_and_normalize(data)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    pipe = get_pipeline()
    started = time.perf_counter()
    async with _inference_semaphore:
        lineart = await asyncio.to_thread(pipe.process, img)
    rss = current_rss_mb()
    safe_name = re.sub(r"[\r\n\t]", "_", filename)[:120]
    logger.info(
        "sketch %s -> %s in %.2fs rss=%s",
        safe_name,
        lineart.size,
        time.perf_counter() - started,
        f"{rss:.0f}MB" if rss is not None else "n/a",
    )

    buf = io.BytesIO()
    lineart.save(buf, format="PNG", optimize=True)
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="web")
