#!/usr/bin/env python3
"""Live detection demo — repaired and re-wired onto the ``painter`` package.

What was wrong before
---------------------
This file used to be a 1052-line standalone WSGI script that could not even be
imported:

* ``IndentationError`` in ``_scan_regions_fallback`` (the ``if bands`` block was
  indented outside its ``for`` loop),
* a missing parenthesis in ``scan_document``'s sort key,
* a corrupted JPEG-quality literal (``cv2.IMWRITE_JPEG_QUALITY, <mojibake>90``),
* a stray ``"\\r\\n"`` glued to the ``if path in ("/stream", "/stream/"):`` line,
* undefined helpers and duplicated model loaders,
* and a **live SerpApi API key committed in source**:
  ``SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "428809a6...")`` — removed.

The demo now delegates detection, OCR and image search to the ``painter``
package, so it cannot drift from the engine, and every credential comes from the
environment (see ``.env.example``). The route surface is preserved:

``GET  /``                 the original static results page (``index.html``)
``GET  /static/<name>``    any file in this directory
``GET  /health``           component availability
``GET  /models``           detector load report (loader, classes, timing)
``GET  /stream``           MJPEG stream of ``sample_video.mp4`` with boxes
``POST /boxes``            layout boxes for an uploaded frame (JSON)
``POST /detect``           COCO object detection for an uploaded frame (JSON)
``POST /textregions``      CV text regions (no model needed) (JSON)
``POST /ocr``              OCR text for an uploaded frame (JSON)
``POST /scan``             document scan: layout + text per region (+ Lens)
``POST /annotate``         annotated JPEG for an uploaded frame
``GET  /scan.html``        upload page for /scan
``GET  /webcam``           webcam page that posts frames to /scan

Run::

    python yolo_demo/live_feed_server.py --port 12001
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import cv2
import numpy as np

# Running this file directly puts ``yolo_demo/`` on sys.path, not the repository
# root, so make the ``painter`` package importable either way.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from painter.config import Settings, load_settings  # noqa: E402
from painter.detection import (
    YoloDetector,
    find_text_regions,
    postprocess_detections,
)
from painter.logging_utils import get_logger, redact
from painter.preprocess import encode_jpeg
from painter.serpapi_client import SerpApiClient, SerpApiError

logger = get_logger("demo")

HERE = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

TEXT_ELEMENT_TYPES = {
    "heading",
    "subheading",
    "paragraph",
    "text_block",
    "list",
    "figure_caption",
    "table_caption",
    "footer",
    "header",
    "page_number",
    "formula",
    "form_field",
}

CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "heading": (60, 76, 231),
    "subheading": (120, 160, 255),
    "paragraph": (80, 200, 120),
    "list": (120, 220, 180),
    "table": (255, 170, 60),
    "image": (255, 120, 90),
    "logo": (200, 100, 255),
    "signature": (0, 200, 255),
    "stamp": (0, 140, 255),
    "footer": (170, 170, 170),
    "header": (170, 170, 170),
    "page_number": (170, 170, 170),
}


def _colour(element_type: str) -> tuple[int, int, int]:
    return CLASS_COLOURS.get(element_type, (200, 200, 200))


def annotate(image: np.ndarray, detections: list[dict[str, Any]], label_text: bool = True) -> np.ndarray:
    """Draw boxes, class names, confidences and recognised text."""
    canvas = image.copy()
    thickness = max(2, round(max(canvas.shape[:2]) / 500))
    for detection in detections:
        box = detection.get("bbox") or []
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (int(round(float(value))) for value in box)
        element_type = str(detection.get("type") or detection.get("class") or "?")
        colour = _colour(element_type)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)
        confidence = detection.get("confidence", detection.get("conf"))
        caption = f"{element_type} {float(confidence):.2f}" if isinstance(confidence, (int, float)) else element_type
        text = detection.get("text")
        if label_text and text:
            snippet = " ".join(str(text).split())[:48]
            caption = f"{caption} | {snippet}"
        (text_width, text_height), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(0, y1 - text_height - baseline - 6)
        cv2.rectangle(canvas, (x1, top), (x1 + text_width + 8, y1), colour, -1)
        cv2.putText(
            canvas,
            caption,
            (x1 + 4, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (16, 16, 16),
            2,
            cv2.LINE_AA,
        )
    return canvas


class DemoState:
    """Settings, lazily loaded detectors and the OCR chain."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.layout = YoloDetector("layout", settings.yolo_layout_model, settings)
        self.coco = YoloDetector(
            "coco", settings.yolo_coco_model, settings, imgsz=640, conf=0.25
        )
        self._ocr = None
        self._ocr_lock = threading.Lock()
        self.lens: SerpApiClient | None = None
        if settings.serpapi.usable:
            self.lens = SerpApiClient(settings.serpapi, settings.cache_dir)
        else:
            logger.warning(
                "Google Lens disabled (enabled=%s, key_configured=%s); "
                "set PAINTER_ENABLE_SERPAPI=true and SERPAPI_KEY to enable it",
                settings.serpapi.enabled,
                settings.serpapi.configured,
            )

    def ocr(self):
        """Build the OCR chain on first use so startup stays fast."""
        if self._ocr is None:
            with self._ocr_lock:
                if self._ocr is None:
                    from painter.ocr import build_ocr_chain

                    self._ocr = build_ocr_chain(self.settings, self.lens)
        return self._ocr

    # -- detection -------------------------------------------------------
    def layout_detections(self, image: np.ndarray) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
        height, width = image.shape[:2]
        report = self.layout.load()
        if not report.ok:
            return [], [f"layout model unavailable: {report.error}"], report.to_dict()
        raw = self.layout.predict(image)
        detections, warnings = postprocess_detections(raw, width, height, self.settings)
        return [detection.to_dict() for detection in detections], warnings, report.to_dict()

    def scan(self, image: np.ndarray, with_lens: bool = False) -> dict[str, Any]:
        """Layout detection plus OCR text for every text region."""
        height, width = image.shape[:2]
        detections, warnings, report = self.layout_detections(image)
        regions: list[dict[str, Any]] = []
        for detection in detections:
            element_type = detection["type"]
            region = dict(detection)
            region["class"] = element_type
            box = detection["bbox"]
            x1, y1 = int(max(0, box[0])), int(max(0, box[1]))
            x2, y2 = int(min(width, box[2])), int(min(height, box[3]))
            if element_type in TEXT_ELEMENT_TYPES and x2 - x1 > 10 and y2 - y1 > 10:
                try:
                    crop = image[y1:y2, x1:x2]
                    outcome = self.ocr().recognize(crop)
                    region["text"] = outcome.text
                    region["text_source"] = outcome.provider
                    region["text_confidence"] = outcome.confidence
                    region["text_uncertain"] = bool(outcome.uncertain)
                except Exception as exc:  # noqa: BLE001 - one bad crop must not kill the scan
                    region["text_error"] = f"{type(exc).__name__}: {exc}"
            if with_lens and self.lens is not None and element_type in {"image", "logo", "stamp", "signature"}:
                try:
                    crop = image[y1:y2, x1:x2]
                    response = self.lens.search_image(crop, lens_type=self.settings.serpapi.lens_type)
                    region["lens"] = response.to_summary()
                except SerpApiError as exc:
                    region["lens_error"] = str(exc)
            regions.append(region)
        return {
            "regions": regions,
            "count": len(regions),
            "detector": "doclayout-yolo" if report.get("ok") else "unavailable",
            "detector_report": report,
            "page": {"w": width, "h": height},
            "warnings": warnings,
        }

    def text_regions(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Model-free CV text regions, in pixel and normalised coordinates."""
        height, width = image.shape[:2]
        boxes = find_text_regions(image)
        return [
            {
                "bbox": [box.x1, box.y1, box.x2, box.y2],
                "norm": [
                    round(box.x1 / width, 4),
                    round(box.y1 / height, 4),
                    round(box.x2 / width, 4),
                    round(box.y2 / height, 4),
                ],
                "class": "text",
            }
            for box in boxes
        ]


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "PainterDemo/1.0"
    state: DemoState  # injected by serve()

    # -- plumbing --------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib signature
        logger.info("%s - %s", self.address_string(), redact(fmt % args))

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(payload, indent=2, default=str).encode(), "application/json")

    def _send_html(self, markup: str) -> None:
        self._send(HTTPStatus.OK, markup.encode(), "text/html; charset=utf-8")

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        if length > MAX_UPLOAD_BYTES:
            raise ValueError(f"upload of {length} bytes exceeds the {MAX_UPLOAD_BYTES} byte limit")
        return self.rfile.read(length)

    def _image(self, body: bytes) -> np.ndarray | None:
        """Decode an uploaded image, accepting multipart or a raw body."""
        payload = body
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" in content_type and "boundary=" in content_type:
            boundary = content_type.split("boundary=", 1)[1].strip().strip('"').encode()
            for part in body.split(b"--" + boundary):
                headers, _, data = part.partition(b"\r\n\r\n")
                if b"filename=" in headers or b'name="file"' in headers:
                    payload = data.rstrip(b"\r\n-")
                    break
        if not payload:
            return None
        return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        route = self.path.split("?", 1)[0]
        if route in {"/", "/index.html"}:
            page = HERE / "index.html"
            if not page.is_file():
                self._send(HTTPStatus.NOT_FOUND, b"index.html missing", "text/plain")
                return
            self._send(HTTPStatus.OK, page.read_bytes(), "text/html; charset=utf-8")
            return
        if route == "/health":
            # load() is cached after the first call, so a health check tells the
            # truth about the detector instead of echoing an empty report.
            report = self.state.layout.load()
            self._send_json(
                {
                    "status": "ok",
                    "layout_model": report.to_dict(),
                    "lens_available": self.state.lens is not None,
                    "lens_hint": None
                    if self.state.lens is not None
                    else "set PAINTER_ENABLE_SERPAPI=true and SERPAPI_KEY",
                    "ocr_provider": self.state.settings.ocr_provider,
                }
            )
            return
        if route == "/models":
            self._send_json(
                {
                    "layout": self.state.layout.load().to_dict(),
                    "coco": self.state.coco.load().to_dict(),
                }
            )
            return
        if route == "/stream":
            self._stream()
            return
        if route == "/scan.html":
            self._send_html(_SCAN_PAGE)
            return
        if route == "/webcam":
            self._send_html(_WEBCAM_PAGE)
            return
        if route == "/highlight-demo":
            self._send_html(_HIGHLIGHT_PAGE)
            return
        if route.startswith("/static/"):
            self._static(unquote(route[len("/static/") :]))
            return
        self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def _static(self, relative: str) -> None:
        candidate = (HERE / relative).resolve()
        if candidate != HERE and HERE not in candidate.parents:
            self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
            return
        if not candidate.is_file():
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self._send(HTTPStatus.OK, candidate.read_bytes(), content_type)

    def _stream(self) -> None:
        video = HERE / "sample_video.mp4"
        if not video.is_file():
            self._send(HTTPStatus.NOT_FOUND, b"sample_video.mp4 missing", "text/plain")
            return
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, b"cannot open video", "text/plain")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        interval = 1.0 / 12.0
        try:
            while True:
                started = time.perf_counter()
                ok, frame = capture.read()
                if not ok or frame is None:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                detections, _warnings, _report = self.state.layout_detections(frame)
                jpeg = encode_jpeg(annotate(frame, detections, label_text=False), quality=70)
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                remaining = interval - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            capture.release()

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        route = self.path.split("?", 1)[0]
        try:
            body = self._body()
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        image = self._image(body)
        if image is None:
            self._send_json(
                {"ok": False, "error": "no decodable image in the request"},
                HTTPStatus.BAD_REQUEST,
            )
            return

        if route == "/boxes":
            detections, warnings, report = self.state.layout_detections(image)
            self._send_json(
                {
                    "ok": True,
                    "detector": "doclayout-yolo" if report.get("ok") else "unavailable",
                    "boxes": detections,
                    "warnings": warnings,
                }
            )
            return

        if route == "/detect":
            height, width = image.shape[:2]
            report = self.state.coco.load()
            if not report.ok:
                self._send_json(
                    {"ok": False, "error": f"COCO model unavailable: {report.error}"},
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            boxes = []
            for detection in self.state.coco.predict(image):
                boxes.append(
                    {
                        "xyxy": detection.bbox.as_list(),
                        "name": detection.class_name,
                        "conf": round(detection.confidence, 3),
                        "norm": [
                            round(detection.bbox.x1 / width, 4),
                            round(detection.bbox.y1 / height, 4),
                            round(detection.bbox.x2 / width, 4),
                            round(detection.bbox.y2 / height, 4),
                        ],
                    }
                )
            self._send_json({"ok": True, "boxes": boxes, "count": len(boxes)})
            return

        if route == "/textregions":
            regions = self.state.text_regions(image)
            self._send_json({"ok": True, "regions": regions, "count": len(regions)})
            return

        if route == "/ocr":
            try:
                outcome = self.state.ocr().recognize(image)
            except Exception as exc:  # noqa: BLE001 - reported to the caller
                self._send_json(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                    HTTPStatus.BAD_GATEWAY,
                )
                return
            self._send_json(
                {
                    "ok": True,
                    "text": outcome.text,
                    "provider": outcome.provider,
                    "confidence": outcome.confidence,
                    "uncertain": bool(outcome.uncertain),
                    "lines": [{"text": line.text, "bbox": line.bbox.as_list()} for line in outcome.lines],
                }
            )
            return

        if route == "/scan":
            with_lens = self.path.split("?", 1)[-1].lower() in {"lens=1", "lens=true"}
            self._send_json({"ok": True, **self.state.scan(image, with_lens=with_lens)})
            return

        if route == "/annotate":
            detections, _warnings, _report = self.state.layout_detections(image)
            self._send(HTTPStatus.OK, encode_jpeg(annotate(image, detections), quality=88), "image/jpeg")
            return

        self._send_json({"ok": False, "error": "unknown route"}, HTTPStatus.NOT_FOUND)


# --------------------------------------------------------------------------
# small inline pages (the legacy server shipped equivalents)
# --------------------------------------------------------------------------
_STYLE = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#111;color:#eee;
     margin:0;padding:24px;text-align:center}
h1{font-weight:600}h1 span{color:#4fc3f7}
.card{background:#1c1c1e;border-radius:12px;padding:16px;margin:20px auto;max-width:1100px;
      box-shadow:0 4px 24px rgba(0,0,0,.5)}
img,video{max-width:100%;border-radius:8px}
pre{text-align:left;background:#0d0d0f;border-radius:8px;padding:12px;overflow:auto;max-height:340px}
button{background:#4fc3f7;border:0;border-radius:8px;padding:10px 18px;font-size:15px;cursor:pointer}
input[type=file]{color:#eee}
"""

_SCAN_PAGE = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Painter demo — document scan</title><style>{_STYLE}</style></head><body>
<h1>Document <span>scan</span></h1>
<div class="card">
  <input type="file" id="file" accept="image/*">
  <button id="go">Scan</button>
  <p><img id="preview" alt=""></p>
  <pre id="out">Pick a page image and press Scan.</pre>
</div>
<script>
const out = document.getElementById('out'), preview = document.getElementById('preview');
document.getElementById('go').onclick = async () => {{
  const file = document.getElementById('file').files[0];
  if (!file) {{ out.textContent = 'Choose a file first.'; return; }}
  preview.src = URL.createObjectURL(file);
  out.textContent = 'Scanning…';
  const body = new FormData(); body.append('file', file);
  const response = await fetch('/scan', {{method: 'POST', body}});
  out.textContent = JSON.stringify(await response.json(), null, 2);
}};
</script></body></html>"""

_WEBCAM_PAGE = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Painter demo — webcam</title><style>{_STYLE}</style></head><body>
<h1>Webcam <span>scan</span></h1>
<div class="card">
  <video id="cam" autoplay playsinline muted></video>
  <p><button id="snap">Scan current frame</button>
     <button id="toggle">Pause</button></p>
  <pre id="out">Camera starting…</pre>
</div>
<script>
const video = document.getElementById('cam'), out = document.getElementById('out');
let running = true;
navigator.mediaDevices.getUserMedia({{video: {{facingMode: 'environment'}}}})
  .then(stream => {{ video.srcObject = stream; }})
  .catch(error => {{ out.textContent = 'Camera unavailable: ' + error; }});
document.getElementById('snap').onclick = async () => {{
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.92));
  const body = new FormData(); body.append('file', blob, 'frame.jpg');
  out.textContent = 'Scanning…';
  const response = await fetch('/scan', {{method: 'POST', body}});
  out.textContent = JSON.stringify(await response.json(), null, 2);
}};
document.getElementById('toggle').onclick = event => {{
  running = !running; event.target.textContent = running ? 'Pause' : 'Resume';
}};
</script></body></html>"""

_HIGHLIGHT_PAGE = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Painter demo — live overlay</title><style>{_STYLE}</style></head><body>
<h1>Live <span>overlay</span></h1>
<div class="card"><img src="/stream" alt="live detection stream"></div>
<p><a style="color:#4fc3f7" href="/scan.html">scan a page</a> ·
   <a style="color:#4fc3f7" href="/webcam">use the webcam</a></p>
</body></html>"""


def serve(port: int = 12001, host: str = "0.0.0.0", settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    DemoHandler.state = DemoState(settings)
    server = ThreadingHTTPServer((host, port), DemoHandler)
    logger.info("demo listening on http://%s:%d/ (layout=%s)", host, port, settings.yolo_layout_model)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("demo stopped")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Painter live detection demo")
    parser.add_argument("--port", type=int, default=12001)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--warm", action="store_true", help="load the layout model before serving")
    arguments = parser.parse_args()
    settings = load_settings()
    if arguments.warm:
        DemoState(settings).layout.load()
    serve(arguments.port, arguments.host, settings)


if __name__ == "__main__":
    main()
