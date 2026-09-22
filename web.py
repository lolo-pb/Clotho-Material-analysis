"""Small local web interface for fiberglass segmentation."""

import argparse
import base64
import csv
import io
import logging
import os
import sys
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Annotated
from urllib.parse import unquote


PROJECT_ROOT = Path(__file__).resolve().parent
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))
if __name__ == "__main__" and sys.prefix == sys.base_prefix:
    executable = "python.exe" if os.name == "nt" else "python"
    venv_python = PROJECT_ROOT / ".ven" / ("Scripts" if os.name == "nt" else "bin") / executable
    if venv_python.is_file():
        os.execv(
            str(venv_python),
            [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        )

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response

from common.labels import CLASS_NAMES, decode_mask
from common.model import FiberglassUNet
from predicting.predict import predict_image


CHECKPOINT_PATH = RESOURCE_ROOT / "checkpoints/final.pt"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_BATCH_IMAGES = 25
INFERENCE_LOCK = Lock()
BATCH_LOCK = Lock()
LOGGER = logging.getLogger(__name__)


@dataclass
class BatchResult:
    filename: str
    width: int
    height: int
    statistics: dict[str, object]
    mask_png: bytes
    overlay_png: bytes
    thumbnail_data_url: str


@dataclass
class BatchSession:
    submitted_count: int = 0
    cancelled: bool = False
    results: list[BatchResult] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)


BATCH_SESSIONS: dict[str, BatchSession] = {}


class BatchInputError(ValueError):
    def __init__(self, detail: str, status_code: int):
        super().__init__(detail)
        self.status_code = status_code


def load_model() -> tuple[FiberglassUNet, torch.device]:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model = FiberglassUNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, device


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.device = load_model()
    yield


app = FastAPI(
    title="Clotho Material Analysis",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


def require_session(request: Request) -> None:
    token = getattr(request.app.state, "auth_token", None)
    if token and request.headers.get("X-Clotho-Session") != token:
        raise HTTPException(status_code=403, detail="Invalid local application session.")


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    require_session(request)
    return {"status": "ready"}


def decode_uploaded_image(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("Choose a non-empty image file.")

    encoded = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("The uploaded file is not a readable image.")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def png_data_url(image_rgb: np.ndarray) -> str:
    payload = base64.b64encode(png_bytes(image_rgb)).decode("ascii")
    return f"data:image/png;base64,{payload}"


def png_bytes(image_rgb: np.ndarray) -> bytes:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        raise RuntimeError("Could not encode prediction image.")
    return encoded.tobytes()


def thumbnail_data_url(image_rgb: np.ndarray) -> str:
    height, width = image_rgb.shape[:2]
    scale = min(1, 280 / max(height, width))
    if scale < 1:
        image_rgb = cv2.resize(
            image_rgb,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return png_data_url(image_rgb)


def calculate_statistics(class_ids: np.ndarray) -> dict[str, object]:
    counts = np.bincount(class_ids.ravel(), minlength=len(CLASS_NAMES))
    total_pixels = int(class_ids.size)
    classes = {
        name: {
            "pixels": int(counts[index]),
            "percent": float(counts[index] / total_pixels * 100),
        }
        for index, name in enumerate(CLASS_NAMES)
    }
    return {"total_pixels": total_pixels, "statistics": classes}


def statistics_csv_data_url(result: dict[str, object]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    statistics = result["statistics"]
    header = ["total_pixels"]
    row = [result["total_pixels"]]
    for name in CLASS_NAMES:
        values = statistics[name]
        header.extend((f"{name}_pixels", f"{name}_percent"))
        row.extend((values["pixels"], f"{values['percent']:.6f}"))
    writer.writerow(header)
    writer.writerow(row)
    payload = base64.b64encode(output.getvalue().encode("utf-8")).decode("ascii")
    return f"data:text/csv;base64,{payload}"


def analyze_image(
    model: torch.nn.Module,
    device: torch.device,
    image_rgb: np.ndarray,
) -> dict[str, object]:
    class_ids = predict_image(model, image_rgb, device)
    mask = decode_mask(class_ids)
    overlay = cv2.addWeighted(image_rgb, 0.55, mask, 0.45, 0)
    result = calculate_statistics(class_ids)
    return {
        "original_data_url": png_data_url(image_rgb),
        "mask_data_url": png_data_url(mask),
        "overlay_data_url": png_data_url(overlay),
        "statistics_csv_data_url": statistics_csv_data_url(result),
        **result,
    }


def analyze_batch_image(
    model: torch.nn.Module,
    device: torch.device,
    image_rgb: np.ndarray,
) -> tuple[dict[str, object], bytes, bytes, str]:
    class_ids = predict_image(model, image_rgb, device)
    mask = decode_mask(class_ids)
    overlay = cv2.addWeighted(image_rgb, 0.55, mask, 0.45, 0)
    return (
        calculate_statistics(class_ids),
        png_bytes(mask),
        png_bytes(overlay),
        thumbnail_data_url(overlay),
    )


def batch_session_or_404(batch_id: str) -> BatchSession:
    with BATCH_LOCK:
        session = BATCH_SESSIONS.get(batch_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Batch not found or already discarded.")
    return session


def batch_filename(filename: str | None, image_number: int) -> str:
    candidate = Path(unquote(filename or "").replace("\\", "/")).name.strip()
    return candidate or f"image-{image_number}.png"


def record_batch_error(session: BatchSession, filename: str, detail: str) -> None:
    with BATCH_LOCK:
        session.errors.append({"filename": filename, "error": detail})


async def read_batch_upload(request: Request) -> bytes:
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_UPLOAD_BYTES:
            raise BatchInputError("The image exceeds the 20 MiB limit.", 413)
    return bytes(data)


def batch_summary(session: BatchSession) -> dict[str, object]:
    if not session.results:
        return {"successful_images": 0, "failed_images": len(session.errors), "statistics": {}}

    total_pixels = sum(int(result.statistics["total_pixels"]) for result in session.results)
    totals = {
        name: sum(
            int(result.statistics["statistics"][name]["pixels"])
            for result in session.results
        )
        for name in CLASS_NAMES
    }
    variation = {}
    for name in CLASS_NAMES:
        percentages = np.asarray(
            [result.statistics["statistics"][name]["percent"] for result in session.results],
            dtype=float,
        )
        variation[name] = {
            "mean_percent": float(np.mean(percentages)),
            "std_percent": float(np.std(percentages, ddof=1)) if len(percentages) > 1 else 0.0,
            "median_percent": float(np.median(percentages)),
            "min_percent": float(np.min(percentages)),
            "max_percent": float(np.max(percentages)),
        }
    return {
        "successful_images": len(session.results),
        "failed_images": len(session.errors),
        "total_pixels": total_pixels,
        "statistics": {
            name: {
                "pixels": totals[name],
                "percent": float(totals[name] / total_pixels * 100),
            }
            for name in CLASS_NAMES
        },
        "variation": variation,
    }


def batch_csv_bytes(session: BatchSession) -> dict[str, bytes]:
    image_output = io.StringIO(newline="")
    image_fields = ["filename", "width", "height", "total_pixels"]
    for name in CLASS_NAMES:
        image_fields.extend((f"{name}_pixels", f"{name}_percent"))
    image_writer = csv.DictWriter(image_output, fieldnames=image_fields)
    image_writer.writeheader()
    for result in session.results:
        row = {
            "filename": result.filename,
            "width": result.width,
            "height": result.height,
            "total_pixels": result.statistics["total_pixels"],
        }
        for name in CLASS_NAMES:
            values = result.statistics["statistics"][name]
            row[f"{name}_pixels"] = values["pixels"]
            row[f"{name}_percent"] = f"{values['percent']:.6f}"
        image_writer.writerow(row)

    summary = batch_summary(session)
    summary_output = io.StringIO(newline="")
    summary_fields = ["successful_images", "failed_images", "total_pixels"]
    for name in CLASS_NAMES:
        summary_fields.extend((f"{name}_pixels", f"{name}_percent"))
    summary_writer = csv.DictWriter(summary_output, fieldnames=summary_fields)
    summary_writer.writeheader()
    summary_row = {
        "successful_images": summary["successful_images"],
        "failed_images": summary["failed_images"],
        "total_pixels": summary.get("total_pixels", 0),
    }
    for name in CLASS_NAMES:
        values = summary["statistics"].get(name, {"pixels": 0, "percent": 0.0})
        summary_row[f"{name}_pixels"] = values["pixels"]
        summary_row[f"{name}_percent"] = f"{values['percent']:.6f}"
    summary_writer.writerow(summary_row)

    variation_output = io.StringIO(newline="")
    variation_writer = csv.DictWriter(
        variation_output,
        fieldnames=(
            "class",
            "image_count",
            "mean_percent",
            "std_percent",
            "median_percent",
            "min_percent",
            "max_percent",
        ),
    )
    variation_writer.writeheader()
    for name in CLASS_NAMES:
        values = summary.get("variation", {}).get(name)
        if values:
            variation_writer.writerow({"class": name, "image_count": summary["successful_images"], **values})

    files = {
        "statistics.csv": image_output.getvalue().encode("utf-8"),
        "batch-summary.csv": summary_output.getvalue().encode("utf-8"),
        "image-variation.csv": variation_output.getvalue().encode("utf-8"),
    }
    if session.errors:
        error_output = io.StringIO(newline="")
        error_writer = csv.DictWriter(error_output, fieldnames=("filename", "error"))
        error_writer.writeheader()
        error_writer.writerows(session.errors)
        files["errors.csv"] = error_output.getvalue().encode("utf-8")
    return files


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE_HTML


@app.post("/batches")
def create_batch(request: Request) -> dict[str, object]:
    require_session(request)
    batch_id = uuid.uuid4().hex
    with BATCH_LOCK:
        BATCH_SESSIONS[batch_id] = BatchSession()
    return {"batch_id": batch_id, "maximum_images": MAX_BATCH_IMAGES}


@app.post("/batches/{batch_id}/cancel")
def cancel_batch(batch_id: str, request: Request) -> dict[str, bool]:
    require_session(request)
    session = batch_session_or_404(batch_id)
    with BATCH_LOCK:
        session.cancelled = True
    return {"cancelled": True}


@app.delete("/batches/{batch_id}")
def discard_batch(batch_id: str, request: Request) -> dict[str, bool]:
    require_session(request)
    with BATCH_LOCK:
        if BATCH_SESSIONS.pop(batch_id, None) is None:
            raise HTTPException(status_code=404, detail="Batch not found or already discarded.")
    return {"discarded": True}


@app.post("/batches/{batch_id}/images")
async def add_batch_image(batch_id: str, request: Request) -> dict[str, object]:
    require_session(request)
    session = batch_session_or_404(batch_id)
    with BATCH_LOCK:
        if session.cancelled:
            raise HTTPException(status_code=409, detail="The batch was cancelled.")
        if session.submitted_count >= MAX_BATCH_IMAGES:
            raise HTTPException(status_code=413, detail="A batch can contain at most 25 images.")
        session.submitted_count += 1
        image_number = session.submitted_count

    filename = batch_filename(request.headers.get("X-Clotho-Filename"), image_number)
    try:
        data = await read_batch_upload(request)
        image_rgb = decode_uploaded_image(data)
        height, width = image_rgb.shape[:2]
        if height * width > MAX_IMAGE_PIXELS:
            raise BatchInputError("The image is larger than the 25-megapixel limit.", 413)
        with INFERENCE_LOCK:
            statistics, mask_png, overlay_png, thumbnail = analyze_batch_image(
                request.app.state.model,
                request.app.state.device,
                image_rgb,
            )
    except BatchInputError as error:
        record_batch_error(session, filename, str(error))
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    except ValueError as error:
        record_batch_error(session, filename, str(error))
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        LOGGER.exception("Batch prediction failed")
        detail = "Prediction failed. Check the server terminal for details."
        record_batch_error(session, filename, detail)
        raise HTTPException(status_code=500, detail=detail) from error

    result = BatchResult(
        filename=filename,
        width=width,
        height=height,
        statistics=statistics,
        mask_png=mask_png,
        overlay_png=overlay_png,
        thumbnail_data_url=thumbnail,
    )
    with BATCH_LOCK:
        session.results.append(result)
        result_index = len(session.results) - 1
        summary = batch_summary(session)
    return {
        "result_index": result_index,
        "filename": filename,
        "width": width,
        "height": height,
        "statistics": statistics,
        "thumbnail_data_url": thumbnail,
        "summary": summary,
    }


@app.get("/batches/{batch_id}/download")
def download_batch(batch_id: str, request: Request) -> Response:
    require_session(request)
    session = batch_session_or_404(batch_id)
    with BATCH_LOCK:
        if not session.results:
            raise HTTPException(status_code=400, detail="The batch has no successful images to download.")
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for index, result in enumerate(session.results, start=1):
                stem = Path(result.filename).stem
                prefix = f"{index:03d}-{stem}"
                output.writestr(f"masks/{prefix}-mask.png", result.mask_png)
                output.writestr(f"overlays/{prefix}-overlay.png", result.overlay_png)
            for filename, content in batch_csv_bytes(session).items():
                output.writestr(filename, content)
    return Response(
        archive.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=clotho-batch-results.zip"},
    )


@app.get("/batches/{batch_id}/images/{image_index}/overlay")
def batch_overlay(batch_id: str, image_index: int, request: Request) -> Response:
    require_session(request)
    session = batch_session_or_404(batch_id)
    with BATCH_LOCK:
        try:
            overlay_png = session.results[image_index].overlay_png
        except IndexError as error:
            raise HTTPException(status_code=404, detail="Batch image not found.") from error
    return Response(overlay_png, media_type="image/png")


@app.post("/predict")
def predict(
    request: Request,
    image: Annotated[UploadFile, File(description="Micrograph to segment")],
) -> dict[str, object]:
    require_session(request)
    data = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The image exceeds the 20 MiB limit.")

    try:
        image_rgb = decode_uploaded_image(data)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    height, width = image_rgb.shape[:2]
    if height * width > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail="The image is larger than the 25-megapixel limit.",
        )

    try:
        with INFERENCE_LOCK:
            return analyze_image(
                request.app.state.model,
                request.app.state.device,
                image_rgb,
            )
    except Exception as error:
        LOGGER.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server terminal for details.",
        ) from error


PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Clotho Material Analysis</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0c1110;
      --panel: #151c1a;
      --panel-light: #1c2522;
      --text: #ecf4f0;
      --muted: #9fb0a8;
      --accent: #79d9a7;
      --border: #2b3833;
      --error: #ff9b91;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 12% 0%, #183027 0, transparent 34rem),
        var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    }
    main { width: min(1120px, calc(100% - 32px)); margin: 0 auto; padding: 64px 0; }
    header { margin-bottom: 36px; }
    .eyebrow { color: var(--accent); font-size: .78rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
    h1 { max-width: 760px; margin: 10px 0 12px; font-size: clamp(2.2rem, 6vw, 4.8rem); line-height: .98; letter-spacing: -.055em; }
    header p { max-width: 650px; margin: 0; color: var(--muted); font-size: 1.05rem; line-height: 1.65; }
    .panel { background: color-mix(in srgb, var(--panel) 94%, transparent); border: 1px solid var(--border); border-radius: 22px; box-shadow: 0 24px 70px #0005; }
    .upload-panel { padding: 20px; }
    .dropzone {
      min-height: 280px;
      display: grid;
      place-items: center;
      padding: 32px;
      border: 1.5px dashed #51665d;
      border-radius: 16px;
      background: #111816;
      cursor: pointer;
      text-align: center;
      transition: border-color .2s, background .2s, transform .2s;
    }
    .dropzone:hover, .dropzone.dragging { border-color: var(--accent); background: #14231d; transform: translateY(-1px); }
    .dropzone input { display: none; }
    .upload-icon { display: block; margin-bottom: 16px; color: var(--accent); font-size: 2.6rem; }
    .dropzone strong { display: block; margin-bottom: 8px; font-size: 1.15rem; }
    .dropzone small, #file-name { color: var(--muted); }
    #file-name { display: block; margin-top: 12px; word-break: break-word; }
    .preview { display: none; max-width: 100%; max-height: 380px; margin: 20px auto 0; border-radius: 12px; object-fit: contain; }
    .actions { display: flex; align-items: center; gap: 16px; margin-top: 18px; }
    button, .download {
      border: 0;
      border-radius: 999px;
      padding: 12px 20px;
      background: var(--accent);
      color: #092016;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
      text-decoration: none;
    }
    button:disabled { cursor: not-allowed; opacity: .45; }
    #status { color: var(--muted); }
    #status.error { color: var(--error); }
    #results { margin-top: 28px; }
    #results[hidden] { display: none; }
    h2 { margin: 0 0 18px; font-size: 1.55rem; letter-spacing: -.025em; }
    .stats, .images { display: grid; gap: 14px; }
    .stats { grid-template-columns: repeat(4, 1fr); margin-bottom: 28px; }
    .stat { padding: 18px; border: 1px solid var(--border); border-radius: 16px; background: var(--panel-light); }
    .stat::before { content: ""; display: block; width: 28px; height: 5px; margin-bottom: 22px; border-radius: 9px; background: var(--class-color); }
    .stat span { display: block; color: var(--muted); font-size: .86rem; }
    .stat strong { display: block; margin-top: 5px; font-size: 1.65rem; }
    .images { grid-template-columns: repeat(3, 1fr); }
    .image-card { overflow: hidden; }
    .image-card img { display: block; width: 100%; aspect-ratio: 16 / 10; background: #080b0a; object-fit: contain; }
    .image-meta { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px; }
    .image-meta h3 { margin: 0; font-size: .95rem; }
    .download { padding: 8px 13px; font-size: .8rem; }
    .result-actions { display: flex; justify-content: flex-end; margin-top: 18px; }
    .selection { margin-top: 18px; }
    .selection-header { display: flex; justify-content: space-between; gap: 16px; color: var(--muted); font-size: .9rem; }
    .selection-list { max-height: 210px; margin: 10px 0 0; padding: 0; overflow-y: auto; border-top: 1px solid var(--border); list-style: none; }
    .selection-list li { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: 12px; padding: 10px 2px; border-bottom: 1px solid var(--border); font-size: .9rem; }
    .selection-list .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .selection-list .size, .selection-list .state { color: var(--muted); }
    .selection-list .state.complete { color: var(--accent); }
    .selection-list .state.failed { color: var(--error); }
    .batch-table { width: 100%; border-collapse: collapse; font-size: .9rem; }
    .batch-table th, .batch-table td { padding: 10px; border-bottom: 1px solid var(--border); text-align: left; }
    .batch-table th { color: var(--muted); font-weight: 700; }
    .batch-table td:last-child { text-align: right; }
    .batch-errors { color: var(--error); margin: 18px 0; }
    .batch-thumbnails { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 14px; margin-top: 22px; }
    .batch-thumbnails img { display: block; width: 100%; aspect-ratio: 1; object-fit: contain; background: #080b0a; border-radius: 12px; }
    .thumbnail-button { display: block; width: 100%; padding: 0; border: 0; border-radius: 12px; background: transparent; cursor: pointer; }
    .thumbnail-button:focus-visible { outline: 3px solid var(--accent); outline-offset: 3px; }
    .batch-actions { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 22px; }
    .progress { margin: 0 0 24px; }
    .progress-track { height: 10px; overflow: hidden; border-radius: 99px; background: var(--panel-light); }
    .progress-bar { width: 0; height: 100%; border-radius: inherit; background: var(--accent); transition: width .2s ease; }
    .progress-label { margin: 9px 0 0; color: var(--muted); font-size: .9rem; }
    dialog { width: min(1180px, calc(100% - 32px)); padding: 0; border: 1px solid var(--border); border-radius: 22px; background: var(--panel); color: var(--text); box-shadow: 0 30px 100px #000a; }
    dialog::backdrop { background: #000b; }
    .detail-header { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 20px 22px; border-bottom: 1px solid var(--border); }
    .detail-header h2 { margin: 0; font-size: 1.15rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .detail-close { padding: 8px 13px; }
    .detail-body { display: grid; grid-template-columns: minmax(0, 2fr) minmax(240px, 1fr); gap: 20px; padding: 22px; }
    .detail-images { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .detail-images figure { margin: 0; }
    .detail-images img { display: block; width: 100%; aspect-ratio: 1; background: #080b0a; object-fit: contain; border-radius: 12px; }
    .detail-images figcaption { margin-top: 8px; color: var(--muted); font-size: .9rem; }
    .detail-stats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; align-content: start; }
    .detail-stat { padding: 14px; border: 1px solid var(--border); border-radius: 12px; background: var(--panel-light); }
    .detail-stat span { display: block; color: var(--muted); font-size: .82rem; }
    .detail-stat strong { display: block; margin-top: 5px; font-size: 1.3rem; }
    .secondary { background: var(--panel-light); color: var(--text); border: 1px solid var(--border); }
    @media (max-width: 760px) {
      main { padding: 36px 0; }
      .stats { grid-template-columns: repeat(2, 1fr); }
      .images { grid-template-columns: 1fr; }
      .actions { align-items: flex-start; flex-direction: column; }
      .detail-body, .detail-images { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div class="eyebrow">Clotho · Material analysis</div>
      <h1>See what your material is made of.</h1>
      <p>Upload one or a batch of fiberglass micrographs to identify fiber, resin, pore, and unidentified regions. Images are processed locally and are not retained.</p>
    </header>

    <section class="panel upload-panel">
      <label class="dropzone" id="dropzone">
        <input id="file-input" type="file" multiple accept="image/jpeg,image/png,image/bmp,image/tiff,image/webp">
        <span>
          <span class="upload-icon">↥</span>
          <strong>Drop micrographs here or choose files</strong>
          <small>JPEG, PNG, BMP, TIFF, or WebP · maximum 20 MiB each, up to 25 images</small>
          <span id="file-name">No image selected</span>
        </span>
      </label>
      <img class="preview" id="preview" alt="Selected micrograph preview">
      <div id="selection" class="selection" hidden>
        <div class="selection-header"><strong>Selected images</strong><span id="selection-total"></span></div>
        <ul id="selection-list" class="selection-list"></ul>
      </div>
      <div class="actions">
        <button id="analyze" type="button" disabled>Analyze image</button>
        <span id="status" role="status" aria-live="polite"></span>
      </div>
    </section>

    <section id="results" hidden>
      <h2>Composition</h2>
      <div class="stats">
        <div class="stat" style="--class-color:#ff5d5d"><span>Fiber</span><strong id="fiber">—</strong></div>
        <div class="stat" style="--class-color:#55d889"><span>Resin</span><strong id="resin">—</strong></div>
        <div class="stat" style="--class-color:#668cff"><span>Pore</span><strong id="pore">—</strong></div>
        <div class="stat" style="--class-color:#69736f"><span>Unidentified</span><strong id="unidentified">—</strong></div>
      </div>

      <h2>Segmentation</h2>
      <div class="images">
        <article class="panel image-card">
          <img id="result-original" alt="Original micrograph">
          <div class="image-meta"><h3>Original</h3></div>
        </article>
        <article class="panel image-card">
          <img id="result-mask" alt="Exact-color segmentation mask">
          <div class="image-meta"><h3>Mask</h3><a class="download" id="mask-download" download="segmentation-mask.png">Download</a></div>
        </article>
        <article class="panel image-card">
          <img id="result-overlay" alt="Segmentation overlay">
          <div class="image-meta"><h3>Overlay</h3><a class="download" id="overlay-download" download="segmentation-overlay.png">Download</a></div>
        </article>
      </div>
      <div class="result-actions">
        <a class="download" id="csv-download" download="segmentation-statistics.csv">Download statistics CSV</a>
      </div>
    </section>

    <section id="batch-results" hidden>
      <h2>Batch composition</h2>
      <div class="progress">
        <div id="batch-progress-track" class="progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><div id="batch-progress-bar" class="progress-bar"></div></div>
        <p id="batch-progress-label" class="progress-label">Waiting to start.</p>
      </div>
      <div class="stats">
        <div class="stat" style="--class-color:#ff5d5d"><span>Fiber</span><strong id="batch-fiber">—</strong></div>
        <div class="stat" style="--class-color:#55d889"><span>Resin</span><strong id="batch-resin">—</strong></div>
        <div class="stat" style="--class-color:#668cff"><span>Pore</span><strong id="batch-pore">—</strong></div>
        <div class="stat" style="--class-color:#69736f"><span>Unidentified</span><strong id="batch-unidentified">—</strong></div>
      </div>
      <p id="batch-summary" style="color:var(--muted)"></p>
      <h2>Images</h2>
      <div class="panel" style="overflow-x:auto">
        <table class="batch-table">
          <thead><tr><th>Image</th><th>Fiber</th><th>Resin</th><th>Pore</th><th>Unidentified</th><th>Status</th></tr></thead>
          <tbody id="batch-rows"></tbody>
        </table>
      </div>
      <p id="batch-errors" class="batch-errors" hidden></p>
      <div id="batch-thumbnails" class="batch-thumbnails"></div>
      <div class="batch-actions">
        <button id="batch-cancel" class="secondary" type="button" hidden>Stop</button>
        <button id="batch-download" type="button" hidden>Download batch ZIP</button>
        <button id="batch-discard" class="secondary" type="button" hidden>Discard batch</button>
      </div>
    </section>
  </main>

  <dialog id="batch-detail">
    <div class="detail-header">
      <h2 id="detail-title">Batch image</h2>
      <button id="detail-close" class="secondary detail-close" type="button">Close</button>
    </div>
    <div class="detail-body">
      <div class="detail-images">
        <figure><img id="detail-original" alt="Original micrograph"><figcaption>Original</figcaption></figure>
        <figure><img id="detail-overlay" alt="Segmentation overlay"><figcaption>Overlay</figcaption></figure>
      </div>
      <div class="detail-stats">
        <div class="detail-stat"><span>Fiber</span><strong id="detail-fiber">—</strong></div>
        <div class="detail-stat"><span>Resin</span><strong id="detail-resin">—</strong></div>
        <div class="detail-stat"><span>Pore</span><strong id="detail-pore">—</strong></div>
        <div class="detail-stat"><span>Unidentified</span><strong id="detail-unidentified">—</strong></div>
      </div>
    </div>
  </dialog>

  <script>
    const input = document.querySelector("#file-input");
    const dropzone = document.querySelector("#dropzone");
    const preview = document.querySelector("#preview");
    const fileName = document.querySelector("#file-name");
    const selection = document.querySelector("#selection");
    const selectionTotal = document.querySelector("#selection-total");
    const selectionList = document.querySelector("#selection-list");
    const analyze = document.querySelector("#analyze");
    const status = document.querySelector("#status");
    const results = document.querySelector("#results");
    const batchResults = document.querySelector("#batch-results");
    const batchRows = document.querySelector("#batch-rows");
    const batchThumbnails = document.querySelector("#batch-thumbnails");
    const batchSummary = document.querySelector("#batch-summary");
    const batchErrors = document.querySelector("#batch-errors");
    const batchCancel = document.querySelector("#batch-cancel");
    const batchDownload = document.querySelector("#batch-download");
    const batchDiscard = document.querySelector("#batch-discard");
    const batchProgressTrack = document.querySelector("#batch-progress-track");
    const batchProgressBar = document.querySelector("#batch-progress-bar");
    const batchProgressLabel = document.querySelector("#batch-progress-label");
    const batchDetail = document.querySelector("#batch-detail");
    const detailTitle = document.querySelector("#detail-title");
    const detailOriginal = document.querySelector("#detail-original");
    const detailOverlay = document.querySelector("#detail-overlay");
    const detailClose = document.querySelector("#detail-close");
    let selectedFiles = [];
    let previewUrl = null;
    let batchId = null;
    let cancelRequested = false;
    let batchSuccessfulCount = 0;
    let detailUrls = [];
    let detailRequestId = 0;

    function sessionHeaders(headers = {}) {
      const token = new URLSearchParams(window.location.search).get("token");
      return token ? { ...headers, "X-Clotho-Session": token } : headers;
    }

    function formatSize(bytes) {
      return `${(bytes / 1024 / 1024).toFixed(2)} MiB`;
    }

    function renderSelection(files) {
      selectionList.replaceChildren();
      selection.hidden = false;
      selectionTotal.textContent = `${files.length} image${files.length === 1 ? "" : "s"}`;
      files.forEach((file, index) => {
        const item = document.createElement("li");
        item.dataset.index = index;
        const name = document.createElement("span");
        name.className = "name";
        name.textContent = file.name;
        const size = document.createElement("span");
        size.className = "size";
        size.textContent = formatSize(file.size);
        const state = document.createElement("span");
        state.className = "state";
        state.textContent = "Selected";
        item.append(name, size, state);
        selectionList.append(item);
      });
    }

    function setFileState(index, text, state = "") {
      const item = selectionList.querySelector(`[data-index="${index}"] .state`);
      if (!item) return;
      item.textContent = text;
      item.className = `state ${state}`;
    }

    function updateBatchProgress(completed, total, label) {
      const percent = total ? Math.round(completed / total * 100) : 0;
      batchProgressBar.style.width = `${percent}%`;
      batchProgressTrack.setAttribute("aria-valuenow", String(percent));
      batchProgressLabel.textContent = label;
    }

    function chooseFiles(files) {
      const chosen = Array.from(files || []);
      if (!chosen.length) return;
      if (batchId) {
        status.textContent = "Download or discard the current batch before choosing another one.";
        status.className = "error";
        return;
      }
      if (chosen.length > 25) {
        status.textContent = "Choose at most 25 images in one batch.";
        status.className = "error";
        return;
      }
      selectedFiles = chosen;
      renderSelection(chosen);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (chosen.length === 1) {
        previewUrl = URL.createObjectURL(chosen[0]);
        preview.src = previewUrl;
        preview.style.display = "block";
        fileName.textContent = `${chosen[0].name} · ${(chosen[0].size / 1024 / 1024).toFixed(2)} MiB`;
      } else {
        preview.src = "";
        preview.style.display = "none";
        fileName.textContent = `${chosen.length} images selected`;
      }
      analyze.disabled = false;
      results.hidden = true;
      batchResults.hidden = true;
      status.textContent = chosen.length === 1 ? "Ready to analyze." : "Ready to analyze this batch.";
      status.className = "";
    }

    input.addEventListener("change", () => chooseFiles(input.files));
    for (const eventName of ["dragenter", "dragover"]) {
      dropzone.addEventListener(eventName, event => {
        event.preventDefault();
        dropzone.classList.add("dragging");
      });
    }
    for (const eventName of ["dragleave", "drop"]) {
      dropzone.addEventListener(eventName, event => {
        event.preventDefault();
        dropzone.classList.remove("dragging");
      });
    }
    dropzone.addEventListener("drop", event => chooseFiles(event.dataTransfer.files));

    async function analyzeSingle(file) {
      analyze.disabled = true;
      analyze.textContent = "Analyzing…";
      status.textContent = "Running the segmentation model. This can take a few seconds.";
      status.className = "";
      results.hidden = true;

      const form = new FormData();
      form.append("image", file);
      try {
        const response = await fetch("/predict", { method: "POST", body: form, headers: sessionHeaders() });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Prediction failed.");

        document.querySelector("#result-original").src = payload.original_data_url;
        document.querySelector("#result-mask").src = payload.mask_data_url;
        document.querySelector("#result-overlay").src = payload.overlay_data_url;
        document.querySelector("#mask-download").href = payload.mask_data_url;
        document.querySelector("#overlay-download").href = payload.overlay_data_url;
        document.querySelector("#csv-download").href = payload.statistics_csv_data_url;
        for (const name of ["fiber", "resin", "pore", "unidentified"]) {
          document.querySelector(`#${name}`).textContent = `${payload.statistics[name].percent.toFixed(2)}%`;
        }
        results.hidden = false;
        status.textContent = `Analyzed ${payload.total_pixels.toLocaleString()} pixels.`;
        results.scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (error) {
        status.textContent = error.message;
        status.className = "error";
      } finally {
        analyze.disabled = false;
        analyze.textContent = "Analyze image";
      }
    }

    function addBatchRow(filename, statistics, state) {
      const row = document.createElement("tr");
      const cells = [filename, ...["fiber", "resin", "pore", "unidentified"].map(name => statistics ? `${statistics[name].percent.toFixed(2)}%` : "—"), state];
      for (const value of cells) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      batchRows.append(row);
    }

    function showBatchSummary(summary) {
      if (!summary.statistics || !summary.successful_images) return;
      for (const name of ["fiber", "resin", "pore", "unidentified"]) {
        document.querySelector(`#batch-${name}`).textContent = `${summary.statistics[name].percent.toFixed(2)}%`;
      }
      batchSummary.textContent = `${summary.successful_images} successful image${summary.successful_images === 1 ? "" : "s"} · ${summary.total_pixels.toLocaleString()} analyzed pixels · pixel-weighted composition`;
    }

    function closeBatchDetail() {
      detailRequestId += 1;
      for (const url of detailUrls) URL.revokeObjectURL(url);
      detailUrls = [];
      detailOriginal.removeAttribute("src");
      detailOverlay.removeAttribute("src");
      if (batchDetail.open) batchDetail.close();
    }

    async function openBatchDetail(selectedIndex, resultIndex, filename, statistics) {
      const original = selectedFiles[selectedIndex];
      const activeBatchId = batchId;
      if (!original || !activeBatchId) return;
      closeBatchDetail();
      const requestId = ++detailRequestId;
      detailTitle.textContent = filename;
      for (const name of ["fiber", "resin", "pore", "unidentified"]) {
        document.querySelector(`#detail-${name}`).textContent = `${statistics[name].percent.toFixed(2)}%`;
      }
      const originalUrl = URL.createObjectURL(original);
      detailUrls.push(originalUrl);
      detailOriginal.src = originalUrl;
      detailOverlay.removeAttribute("src");
      batchDetail.showModal();
      try {
        const response = await fetch(`/batches/${activeBatchId}/images/${resultIndex}/overlay`, { headers: sessionHeaders() });
        if (!response.ok) {
          const payload = await response.json();
          throw new Error(payload.detail || "Could not load the overlay.");
        }
        const overlayUrl = URL.createObjectURL(await response.blob());
        if (requestId !== detailRequestId || !batchDetail.open) {
          URL.revokeObjectURL(overlayUrl);
          return;
        }
        detailUrls.push(overlayUrl);
        detailOverlay.src = overlayUrl;
      } catch (error) {
        closeBatchDetail();
        status.textContent = error.message;
        status.className = "error";
      }
    }

    function addThumbnail(selectedIndex, resultIndex, filename, dataUrl, statistics) {
      const card = document.createElement("article");
      card.className = "panel image-card";
      const button = document.createElement("button");
      button.className = "thumbnail-button";
      button.type = "button";
      button.setAttribute("aria-label", `Open details for ${filename}`);
      const image = document.createElement("img");
      image.src = dataUrl;
      image.alt = `Overlay for ${filename}`;
      button.append(image);
      button.addEventListener("click", () => openBatchDetail(selectedIndex, resultIndex, filename, statistics));
      const label = document.createElement("div");
      label.className = "image-meta";
      label.textContent = filename;
      card.append(button, label);
      batchThumbnails.append(card);
    }

    async function analyzeBatch() {
      analyze.disabled = true;
      analyze.textContent = "Analyzing batch…";
      results.hidden = true;
      batchResults.hidden = false;
      batchRows.replaceChildren();
      batchThumbnails.replaceChildren();
      batchErrors.hidden = true;
      batchErrors.textContent = "";
      batchCancel.hidden = false;
      batchCancel.disabled = false;
      batchCancel.textContent = "Stop";
      batchDownload.hidden = true;
      batchDiscard.hidden = true;
      cancelRequested = false;
      batchSuccessfulCount = 0;
      updateBatchProgress(0, selectedFiles.length, `0 of ${selectedFiles.length} complete`);

      try {
        const created = await fetch("/batches", { method: "POST", headers: sessionHeaders() });
        const payload = await created.json();
        if (!created.ok) throw new Error(payload.detail || "Could not create batch.");
        batchId = payload.batch_id;

        for (let index = 0; index < selectedFiles.length; index += 1) {
          if (cancelRequested) break;
          const file = selectedFiles[index];
          status.textContent = `Analyzing ${index + 1} of ${selectedFiles.length}: ${file.name}`;
          setFileState(index, "Processing");
          updateBatchProgress(index, selectedFiles.length, `Processing ${index + 1} of ${selectedFiles.length}: ${file.name}`);
          const response = await fetch(`/batches/${batchId}/images`, {
            method: "POST",
            headers: sessionHeaders({ "Content-Type": "application/octet-stream", "X-Clotho-Filename": encodeURIComponent(file.name) }),
            body: file,
          });
          const payload = await response.json();
          if (!response.ok) {
            addBatchRow(file.name, null, "Failed");
            setFileState(index, "Failed", "failed");
            batchErrors.hidden = false;
            batchErrors.textContent += `${file.name}: ${payload.detail || "Processing failed."} `;
            updateBatchProgress(index + 1, selectedFiles.length, `${index + 1} of ${selectedFiles.length} complete`);
            continue;
          }
          addBatchRow(payload.filename, payload.statistics.statistics, "Complete");
          setFileState(index, "Complete", "complete");
          addThumbnail(index, payload.result_index, payload.filename, payload.thumbnail_data_url, payload.statistics.statistics);
          showBatchSummary(payload.summary);
          batchSuccessfulCount += 1;
          updateBatchProgress(index + 1, selectedFiles.length, `${index + 1} of ${selectedFiles.length} complete`);
        }

        const completed = batchRows.querySelectorAll("tr").length;
        status.textContent = cancelRequested ? `Batch cancelled after ${completed} image${completed === 1 ? "" : "s"}.` : `Batch complete: ${completed} image${completed === 1 ? "" : "s"} processed.`;
        if (cancelRequested) updateBatchProgress(completed, selectedFiles.length, `Cancelled after ${completed} of ${selectedFiles.length} images`);
        batchDownload.hidden = batchSuccessfulCount === 0;
        batchDiscard.hidden = false;
      } catch (error) {
        status.textContent = error.message;
        status.className = "error";
        if (batchId) batchDiscard.hidden = false;
      } finally {
        batchCancel.hidden = true;
        analyze.disabled = Boolean(batchId) || selectedFiles.length === 0;
        analyze.textContent = "Analyze image";
      }
    }

    batchCancel.addEventListener("click", async () => {
      cancelRequested = true;
      batchCancel.disabled = true;
      batchCancel.textContent = "Cancelling…";
      batchProgressLabel.textContent = "Cancelling after the current image finishes…";
      if (batchId) await fetch(`/batches/${batchId}/cancel`, { method: "POST", headers: sessionHeaders() });
    });

    batchDownload.addEventListener("click", async () => {
      try {
        const response = await fetch(`/batches/${batchId}/download`, { headers: sessionHeaders() });
        if (!response.ok) {
          const payload = await response.json();
          throw new Error(payload.detail || "Could not create the batch download.");
        }
        const url = URL.createObjectURL(await response.blob());
        const link = document.createElement("a");
        link.href = url;
        link.download = "clotho-batch-results.zip";
        link.click();
        URL.revokeObjectURL(url);
      } catch (error) {
        status.textContent = error.message;
        status.className = "error";
      }
    });

    batchDiscard.addEventListener("click", async () => {
      if (!batchId) return;
      closeBatchDetail();
      await fetch(`/batches/${batchId}`, { method: "DELETE", headers: sessionHeaders() });
      batchId = null;
      selectedFiles = [];
      input.value = "";
      selection.hidden = true;
      selectionList.replaceChildren();
      batchResults.hidden = true;
      fileName.textContent = "No image selected";
      status.textContent = "Batch discarded.";
      status.className = "";
      analyze.disabled = true;
    });

    analyze.addEventListener("click", () => {
      if (batchId) {
        status.textContent = "Download or discard the current batch before starting another one.";
        status.className = "error";
        return;
      }
      if (selectedFiles.length === 1) analyzeSingle(selectedFiles[0]);
      if (selectedFiles.length > 1) analyzeBatch();
    });

    detailClose.addEventListener("click", closeBatchDetail);
    batchDetail.addEventListener("click", event => {
      if (event.target === batchDetail) closeBatchDetail();
    });
    batchDetail.addEventListener("close", () => {
      for (const url of detailUrls) URL.revokeObjectURL(url);
      detailUrls = [];
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Clotho local analysis server.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--auth-token")
    args = parser.parse_args()
    app.state.auth_token = args.auth_token
    uvicorn.run(app, host="127.0.0.1", port=args.port)
