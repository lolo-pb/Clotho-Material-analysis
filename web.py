"""Small local web interface for fiberglass segmentation."""

import argparse
import base64
import csv
import io
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated


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
from fastapi.responses import HTMLResponse

from common.labels import CLASS_NAMES, decode_mask
from common.model import FiberglassUNet
from predicting.predict import predict_image


CHECKPOINT_PATH = RESOURCE_ROOT / "checkpoints/final.pt"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
INFERENCE_LOCK = Lock()
LOGGER = logging.getLogger(__name__)


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
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        raise RuntimeError("Could not encode prediction image.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


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


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE_HTML


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
    @media (max-width: 760px) {
      main { padding: 36px 0; }
      .stats { grid-template-columns: repeat(2, 1fr); }
      .images { grid-template-columns: 1fr; }
      .actions { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div class="eyebrow">Clotho · Material analysis</div>
      <h1>See what your material is made of.</h1>
      <p>Upload a fiberglass micrograph to identify fiber, resin, pore, and unidentified regions. Your image is processed locally and is not retained.</p>
    </header>

    <section class="panel upload-panel">
      <label class="dropzone" id="dropzone">
        <input id="file-input" type="file" accept="image/jpeg,image/png,image/bmp,image/tiff,image/webp">
        <span>
          <span class="upload-icon">↥</span>
          <strong>Drop a micrograph here or choose a file</strong>
          <small>JPEG, PNG, BMP, TIFF, or WebP · maximum 20 MiB</small>
          <span id="file-name">No image selected</span>
        </span>
      </label>
      <img class="preview" id="preview" alt="Selected micrograph preview">
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
  </main>

  <script>
    const input = document.querySelector("#file-input");
    const dropzone = document.querySelector("#dropzone");
    const preview = document.querySelector("#preview");
    const fileName = document.querySelector("#file-name");
    const analyze = document.querySelector("#analyze");
    const status = document.querySelector("#status");
    const results = document.querySelector("#results");
    let selectedFile = null;
    let previewUrl = null;

    function chooseFile(file) {
      if (!file) return;
      selectedFile = file;
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = URL.createObjectURL(file);
      preview.src = previewUrl;
      preview.style.display = "block";
      fileName.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MiB`;
      analyze.disabled = false;
      results.hidden = true;
      status.textContent = "Ready to analyze.";
      status.className = "";
    }

    input.addEventListener("change", () => chooseFile(input.files[0]));
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
    dropzone.addEventListener("drop", event => chooseFile(event.dataTransfer.files[0]));

    analyze.addEventListener("click", async () => {
      if (!selectedFile) return;
      analyze.disabled = true;
      analyze.textContent = "Analyzing…";
      status.textContent = "Running the segmentation model. This can take a few seconds.";
      status.className = "";
      results.hidden = true;

      const form = new FormData();
      form.append("image", selectedFile);
      try {
        const sessionToken = new URLSearchParams(window.location.search).get("token");
        const headers = sessionToken ? { "X-Clotho-Session": sessionToken } : {};
        const response = await fetch("/predict", { method: "POST", body: form, headers });
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
