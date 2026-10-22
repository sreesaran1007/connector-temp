"""
Local HTTP server around the analysis pipeline (analyse_image.analyse).

The app asks it for the connectors to choose from, then sends a photo and the
chosen connector to POST /match. The server runs the YOLO model, builds the
connector structure, draws the diagram, and compares what it found with the
chosen reference connector. Everything runs on this computer; nothing is sent
to any cloud service.

Endpoints:
    GET  /health                           is the server up
    GET  /references                       the connectors to choose from
    GET  /references/{id}/thumbnail        small JPEG of a connector
    GET  /references/{id}/image            larger JPEG of a connector
    POST /match     image, reference_id    analyse a photo and compare it
    POST /analyse   image                  analyse a photo (no comparison)

Run, from this folder, with the virtual environment's Python:

    python server.py                      (Linux/macOS: .venv/bin/python server.py)

It listens on 127.0.0.1:4445 by default (HOST / PORT below change that). Binding
to 127.0.0.1 is right behind a Cloudflare tunnel: cloudflared runs on the same
machine and connects to http://localhost:4445, so nothing else can reach the
service directly. Use HOST=0.0.0.0 to also accept connections from other devices
on the network. run_server.cmd / run_server.ps1 / run_server.sh do the same thing.
See DEPLOY.md for the full server setup.

Optional environment variables:
    HOST                    address to listen on (default: 127.0.0.1)
    PORT                    port to listen on (default: 4445)
    ANALYSE_API_KEY         if set, requests must send it as the X-API-Key header.
                            SET THIS when the service is reachable from the internet.
    ALLOWED_ORIGIN_REGEX    browser origins allowed to call the API (default:
                            http(s)://localhost or 127.0.0.1 on any port)
    REFERENCES_DIR          folder with the reference connector photos
                            (default: references/ next to this file). Photos can be
                            added or replaced while the server runs.
    MATCH_THRESHOLD         similarity (0..1) needed to count as a match
                            (default: 0.70)
"""

import base64
import hmac
import io
import logging
import os
import re
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

from analyse_image import analyse, get_model
from build_structure import DEFAULT_CONF, build_structure
from common import ROOT
from matching import DEFAULT_THRESHOLD, compare, positions_of
from references import ReferenceStore

log = logging.getLogger("analyse-server")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024   # reject bigger uploads
MAX_SIDE = 2048                       # longest side after downscaling (speed/memory)
ALLOWED_FORMATS = {"JPEG", "PNG"}
Image.MAX_IMAGE_PIXELS = 80_000_000   # refuse decompression bombs

API_KEY = os.environ.get("ANALYSE_API_KEY")
ORIGIN_REGEX = os.environ.get(
    "ALLOWED_ORIGIN_REGEX", r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
)


def _threshold_from_env():
    raw = os.environ.get("MATCH_THRESHOLD")
    if raw is None:
        return DEFAULT_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not 0.0 <= value <= 1.0:
        raise SystemExit(f"MATCH_THRESHOLD must be a number between 0 and 1, got {raw!r}")
    return value


MATCH_THRESHOLD = _threshold_from_env()
store = ReferenceStore(os.environ.get("REFERENCES_DIR") or ROOT / "references")

# Matches "position 3" inside the pipeline's warning messages.
_POSITION = re.compile(r"\bposition (\d+)")

# The model object is not safe to call from several threads at once, so photos
# are processed one at a time. Requests queue up behind this lock.
_inference_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app):
    # Load the weights and run one throw-away prediction at start-up. The first
    # prediction is much slower than later ones (about 12 s vs 1 s), and this
    # keeps that cost away from the first real request.
    model = get_model()
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    log.info("model loaded and warmed up")
    if not API_KEY:
        log.warning("ANALYSE_API_KEY is not set: anyone who can reach this service can use it")
    store.load(_derive_positions)
    yield


app = FastAPI(title="Connector analysis", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=ORIGIN_REGEX,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


def require_api_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and not hmac.compare_digest((x_api_key or "").encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def normalise_photo(data: bytes) -> Image.Image:
    """Decode an upload into an upright RGB image, or raise HTTP 400/415.

    Phones store rotation in EXIF instead of rotating the pixels. The model
    loader honours it but the diagram step (Pillow) does not, so the two would
    disagree about the image size. Rotating the pixels here removes the issue.
    """
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # cheap integrity check, does not decode everything
        image = Image.open(io.BytesIO(data))
        if image.format not in ALLOWED_FORMATS:
            raise HTTPException(status_code=415, detail="Only JPEG and PNG photos are supported.")
        image = ImageOps.exif_transpose(image).convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError):
        raise HTTPException(status_code=400, detail="The file is not a valid image.") from None
    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return image


def _read_upload(image: UploadFile) -> Image.Image:
    data = image.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The upload is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The photo is larger than 15 MB.")
    return normalise_photo(data)


def _derive_positions(path: Path):
    """The parts the model sees in a reference photo (used once, to fill in the
    reference's JSON). Goes through the same normalising as an upload."""
    photo = normalise_photo(path.read_bytes())
    with tempfile.TemporaryDirectory(prefix="reference-") as tmp:
        copy = Path(tmp) / "reference.jpg"
        photo.save(copy, "JPEG", quality=95)
        with _inference_lock:
            structure = build_structure(get_model(), copy, DEFAULT_CONF)
    return positions_of(structure)


def _analyse(photo: Image.Image, conf: float):
    """Run the whole pipeline on a photo. Returns (diagram image or None, structure)."""
    # A fresh temp folder per request; only server-chosen names are used, never
    # anything from the client, so the upload cannot choose where it is written.
    with tempfile.TemporaryDirectory(prefix="analyse-") as tmp:
        path = Path(tmp) / "upload.jpg"
        photo.save(path, "JPEG", quality=95)
        try:
            with _inference_lock:
                return analyse(path, conf=conf)
        except Exception:
            log.exception("analysis failed")
            raise HTTPException(status_code=500, detail="The analysis failed on the server.") from None


def _payload(diagram, structure, started):
    diagram_png = None
    if diagram is not None:
        buffer = io.BytesIO()
        diagram.save(buffer, "PNG")
        diagram_png = base64.b64encode(buffer.getvalue()).decode("ascii")

    # The pipeline counts positions from 0; people count from 1. The raw
    # `structure` below is left untouched for any later comparison logic.
    positions = [
        {
            "number": p["position"] + 1,
            "components": [
                {"class": c["class"], "kind": c["kind"], "confidence": c["confidence"]}
                for c in p["components"]
            ],
        }
        for p in structure["sequence"]
    ]
    warnings = [_POSITION.sub(lambda m: f"position {int(m.group(1)) + 1}", w) for w in structure["warnings"]]
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    log.info("analysed a photo in %d ms: %d positions", elapsed_ms, len(positions))
    return {
        "detected": diagram is not None,
        "diagram_png_base64": diagram_png,
        "axis": structure["axis"],
        "positions": positions,
        "warnings": warnings,
        "elapsed_ms": elapsed_ms,
        "structure": structure,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/references", dependencies=[Depends(require_api_key)])
def list_references():
    # Look at the folder first, so a photo that was added or replaced since the
    # last request shows up (and is read by the model if it is new).
    store.refresh()
    return [
        {
            "id": r.id,
            "name": r.name,
            "positions": len(r.positions),
            "aspect_ratio": r.aspect_ratio,
            "version": r.version,
            # The version in the address is what makes a browser fetch a replaced
            # picture instead of showing the one it kept (see _jpeg).
            "thumbnail_url": f"/references/{r.id}/thumbnail?v={r.version}",
            "image_url": f"/references/{r.id}/image?v={r.version}",
        }
        for r in store.all()
    ]


def _jpeg(data):
    if data is None:
        raise HTTPException(status_code=404, detail="That connector is not known to the server.")
    # Safe to cache for an hour: the address contains the photo's version, so a
    # replaced photo is requested under a new address.
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/references/{reference_id}/thumbnail", dependencies=[Depends(require_api_key)])
def reference_thumbnail(reference_id: str):
    store.refresh()
    return _jpeg(store.thumbnail(reference_id))


@app.get("/references/{reference_id}/image", dependencies=[Depends(require_api_key)])
def reference_image(reference_id: str):
    store.refresh()
    return _jpeg(store.full_image(reference_id))


@app.post("/analyse", dependencies=[Depends(require_api_key)])
def analyse_photo(
    image: UploadFile = File(...),
    conf: float = Form(DEFAULT_CONF),
):
    conf = min(max(conf, 0.05), 0.95)
    photo = _read_upload(image)
    started = time.perf_counter()
    diagram, structure = _analyse(photo, conf)
    return _payload(diagram, structure, started)


@app.post("/match", dependencies=[Depends(require_api_key)])
def match_photo(
    image: UploadFile = File(...),
    reference_id: str = Form(...),
    conf: float = Form(DEFAULT_CONF),
):
    # Cheap checks first: no point analysing a photo for an unknown connector.
    store.refresh()
    reference = store.get(reference_id)
    if reference is None:
        raise HTTPException(status_code=404, detail="That connector is not known to the server.")
    conf = min(max(conf, 0.05), 0.95)
    photo = _read_upload(image)

    started = time.perf_counter()
    diagram, structure = _analyse(photo, conf)
    outcome = compare(reference.positions, positions_of(structure), MATCH_THRESHOLD)
    payload = _payload(diagram, structure, started)
    payload["match"] = {
        "reference_id": reference.id,
        "reference_name": reference.name,
        **outcome.as_dict(),
    }
    log.info("matched against %s: %.0f%% (%s)", reference.id, outcome.similarity * 100,
             "match" if outcome.matches else "no match")
    return payload


def main():
    """Run the service: `python server.py` (HOST and PORT come from the environment)."""
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("PORT", "4445"))
    except ValueError:
        raise SystemExit("PORT must be a number") from None
    # One process only: the model is loaded once and photos are processed one at
    # a time (see _inference_lock), so extra workers would just multiply memory.
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
