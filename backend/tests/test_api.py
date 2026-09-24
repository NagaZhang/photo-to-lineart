"""Task 4 interface tests. Run from repo root:

    pytest backend/tests -q
"""
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from fastapi.testclient import TestClient

from app.main import MAX_FILE_BYTES, MAX_SIDE, app

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "samples" / "input"
client = TestClient(app)


def setup_module():
    # Trigger startup (model load) once for the module.
    client.__enter__()


def teardown_module():
    client.__exit__(None, None, None)


def _post(filename, content, mime):
    return client.post(
        "/api/sketch",
        files={"file": (filename, io.BytesIO(content), mime)},
    )


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_valid_formats_match_original_size():
    cases = [
        ("madoka.jpg", "image/jpeg"),
        ("saber.png", "image/png"),
        ("anime_scene.webp", "image/webp"),
    ]
    for name, mime in cases:
        data = (SAMPLES / name).read_bytes()
        with Image.open(io.BytesIO(data)) as src:
            w, h = src.size
            scale = MAX_SIDE / max(w, h)
            if scale < 1:
                expected = (max(1, round(w * scale)), max(1, round(h * scale)))
            else:
                expected = src.size
        r = _post(name, data, mime)
        assert r.status_code == 200, (name, r.text)
        assert r.headers["content-type"] == "image/png"
        out = Image.open(io.BytesIO(r.content))
        assert out.mode == "L"
        assert out.size == expected, (name, out.size, expected)


def test_non_image_rejected():
    r = _post("notes.txt", b"hello, not an image", "text/plain")
    assert r.status_code == 400
    assert client.get("/api/health").status_code == 200


def test_forged_extension_rejected():
    r = _post("fake.png", b"this is definitely not a png", "image/png")
    assert r.status_code == 400
    assert client.get("/api/health").status_code == 200


def test_oversized_rejected():
    payload = b"\xff\xd8\xff" + b"0" * (MAX_FILE_BYTES + 1024)
    r = _post("huge.jpg", payload, "image/jpeg")
    assert r.status_code == 413
    assert client.get("/api/health").status_code == 200


def test_corrupted_image_rejected():
    good = bytearray((SAMPLES / "madoka.jpg").read_bytes()[:200])
    good.extend(b"\x00" * 2000)
    r = _post("broken.jpg", bytes(good), "image/jpeg")
    assert r.status_code == 400
    assert client.get("/api/health").status_code == 200


def test_decompression_bomb_rejected():
    # 6000x5000 = 30MP solid-color PNG: tiny on disk, huge if decoded.
    buf = io.BytesIO()
    Image.new("RGB", (6000, 5000), (128, 128, 128)).save(buf, format="PNG")
    r = _post("bomb.png", buf.getvalue(), "image/png")
    assert r.status_code == 400, r.text
    assert "像素" in r.json()["detail"]
    assert client.get("/api/health").status_code == 200


def test_concurrent_requests_queue_safely():
    # One shared in-process server (its own portal thread/event loop); 4
    # blocking HTTP calls from worker threads. The app semaphore must queue
    # inference without deadlocking and all requests must succeed.
    data = (SAMPLES / "madoka.jpg").read_bytes()

    def one(_):
        r = client.post(
            "/api/sketch",
            files={"file": ("madoka.jpg", io.BytesIO(data), "image/jpeg")},
        )
        return r.status_code, r.headers.get("content-type")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, range(4)))
    assert all(code == 200 and ctype == "image/png" for code, ctype in results)
