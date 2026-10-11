#!/usr/bin/env python3
"""YOLOv8 live detection server: MJPEG video feed + browser webcam detection."""
import json
import os
import queue
import threading
import time
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIServer, WSGIRequestHandler, make_server

import cv2
import numpy as np
import requests
from ultralytics import YOLO

VIDEO = os.environ.get("VIDEO", "sample_video.mp4")
PORT = int(os.environ.get("PORT", "12001"))
HOST = os.environ.get("HOST", "0.0.0.0")
CONF = float(os.environ.get("CONF", "0.25"))
MODEL_PATH = os.environ.get("MODEL", "yolov8n.pt")
IMGSZ = int(os.environ.get("IMGSZ", "640"))
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "428809a6dff141ab96f94ae9584cb272824ab19e48649bce6096f5196b9d4621")
SERPAPI_TIMEOUT = 40

DOCLAYOUT_PATH = os.environ.get("DOCLAYOUT_MODEL", "doclayout.pt")
LAYOUT_IMGSZ = int(os.environ.get("LAYOUT_IMGSZ", "1120"))
TEXT_CLASSES = {"Text", "Title", "List-item", "Section-header", "Caption", "Footnote", "Page-header", "Page-footer"}

_layout_model = None
_layout_lock = threading.Lock()


def get_layout_model():
    """Lazily load the DocLayout-YOLO layout model once."""
    global _layout_model
    if _layout_model is None:
        with _layout_lock:
            if _layout_model is None:
                _layout_model = YOLO(DOCLAYOUT_PATH)
                log(f"loaded {DOCLAYOUT_PATH}")
    return _layout_model
_state = {"annotated": None, "frame_id": -1}
_model = None
_model_lock = threading.Lock()


def log(msg):
    print(f"[yolo-server] {msg}", flush=True)


def get_model():
    """Lazily load the shared YOLO model once."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = YOLO(MODEL_PATH)
                log(f"loaded {MODEL_PATH}")
    return _model


def detect_frame(frame: np.ndarray) -> np.ndarray:
    """Run detection on a single frame, return annotated BGR image."""
    model = get_model()
    result = model.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    return result.plot()


def detect_boxes(frame: np.ndarray) -> list:
    """Run detection and return a JSON-able list of boxes."""
    model = get_model()
    result = model.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    boxes = []
    names = result.names
    for box in result.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        cls = int(box.cls[0])
        boxes.append({
            "xyxy": [x1, y1, x2, y2],
            "cls": cls,
            "name": names[cls],
            "conf": round(float(box.conf[0]), 3),
        })
    return boxes


def _scan_regions_fallback(frame: np.ndarray) -> list:
    """CV fallback: text-block grouping + COCO image/logo-ish objects."""
    h, w = frame.shape[:2]
    regions = []
    # 1) text blocks via adaptive-threshold word/line grouping
    words = find_text_regions(frame)
    if words:
        # group overlapping/nearby word rects into paragraph bands
        rects = [(w[0] * w, w[1] * h, w[2] * w, w[3] * h) for w in words]
        rects.sort(key=lambda r: (r[1], r[0]))
        bands = []
        for r in rects:
    if bands and abs(r[1] - bands[-1][3]) < 0.35 * (r[3] - r[1]) and r[0] < bands[-1][2] + 40:
                bw = bands[-1]
                bands[-1] = [min(bw[0], r[0]), min(bw[1], r[1]), max(bw[2], r[2]), max(bw[3], r[3])]
            else:
                bands.append(list(r))
        for x1, y1, x2, y2 in bands:
            if (x2 - x1) * (y2 - y1) < 0.02 * w * h:
                continue
            regions.append({
                "class": "text",
                "conf": 0.7,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "norm": [round(x1 / w, 3), round(y1 / h, 3), round(x2 / w,  3), round(y2 / h,  3)],
                "text": [],
            })
    # 2) image/logo-ish objects via COCO
    img_ids = {76: "scissors", 73: "book", 77: "teddy_bear"}
    try:
        for b in detect_boxes(frame):
            if b["name"] in ("book", "scissors", "laptop", "cell phone", "remote", "tv"):
                x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
                regions.append({
                    "class": "image",
                    "conf": b["conf"],
                    "bbox": [x1, y1, x2, y2],
                    "norm": [round(x1 / w, 3), round(y1 / h, 3), round(x2 / w,  3), round(y2 / h,  3)],
                    "text": [],
                })
    except Exception:
        pass
    return regions


def scan_document(frame: np.ndarray) -> dict:
    """Document layout scan: YOLO(layout) regions -> Lens text per text region.
    Uses DocLayout-YOLO when loadable (may be unavailable in this env) with
    a CV+COCO fallback so /scan works regardless."""
    h, w = frame.shape[:2]
    boxes = []
    detector = "fallback"
    try:
        model = get_layout_model()
        result = model.predict(frame, imgsz=LAYOUT_IMGSZ, conf=0.25, verbose=False)[0]
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            boxes.append({"xyxy": [x1, y1, x2, y2], "name": names[int(box.cls[0])],"conf": round(float(box.conf[0]), 3)})
        if boxes:
            detector = "doclayout-yolo"
    except Exception as e:
        log(f"layout model unavailable: {e}")
    regions = []
    if boxes:
        for b in sorted(boxes, key=lambda b: (b["xyxy"][1], b["xyxy"][0]):
            x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
            name = b["name"]
            region = {
                "class": name.lower().replace("-", "_"),
                "conf": b["conf"],
                "bbox": [x1, y1, x2, y2],
                "norm": [round(x1 / w, 3), round(y1 / h,  3), round(x2 / w,  3), round(y2 / h,  3)],
                "text": [],
            }
            if name in TEXT_CLASSES and x2 - x1 > 10 and y2 - y1 > 10:
                try:
                    crop = frame[y1:y2, x1:x2]
                    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if ok:
                        res = serpapi_lens_ocr(buf.tobytes())
                        region["text"] = res["text"]
                        region["visual"] = res["visual"]
                        region["sources"] = res["sources"]
                        region["text_results_flag"] = res["text_results"]
                except Exception as e:
                    region["text_error"] = str(e)[:120]
            regions.append(region)
    else:
        regions = _scan_regions_fallback(frame)
        for r in regions:
            if r["class"] == "text":
                x1, y1, x2, y2 = r["bbox"]
                try:
                    crop = frame[y1:y2, x1:x2]
                    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY,  ގެންވެތި90])
                    if ok:
                        res = serpapi_lens_ocr(buf.tobytes())
                        r["text"] = res["text"]
                        r["sources"] = res["sources"]
                        r["visual"] = res["visual"]
                        r["text_results_flag"] = res["text_results"]
                except Exception as e:
                    r["text_error"] = str(e)[:120]
    return {
        "regions": regions,
        "count": len(regions),
        "detector": detector,
        "page": {"w": w, "h": h},
    }


def find_text_regions(frame: np.ndarray) -> list:
    """
    Find text-like regions, robust to text polarity (dark-on-light or
    light-on-dark signs/docs). Uses adaptive thresholding on both polarities,
    filters border-touching slabs, and dedupes.
    Returns normalized [x1, y1, x2, y2] relative to frame.
    """
    h, w = frame.shape[:2]
    small = frame
    if max(h, w) > 1400:
        scale = 1400 / max(h, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    area = gray.shape[0] * gray.shape[1]
    H, W = gray.shape

    ker_h = max(5, int(W * 0.015))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ker_h, 5))
    dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def blob_regions(mask):
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        closed = cv2.dilate(closed, dilate, iterations=1)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw < 8 or ch < 6:
                continue
            if cw * ch < area * 0.0004:
                continue
            if cw * ch > area * 0.32:  # no text zone covers a third of frame
                continue
            if ch > cw * 4:  # tall thin strip
                continue
            # skip blobs glued to the frame border (sign borders, bg edges)
            if (x < W * 0.015 or y < H * 0.015 or
                    x + cw > W * 0.985 or y + ch > H * 0.985):
                continue
            out.append([x, y, cw, ch])
        return out

    thr_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 11)          # dark text
    thr_bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 15)              # bright text

    raw = blob_regions(thr_inv) + blob_regions(thr_bin)

    # Dedupe overlapping boxes, sorted top->down
    picked = []
    for b in sorted(raw, key=lambda r: (r[1], r[0])):
        keep = True
        for p in picked:
            ix = max(0, min(b[0] + b[2], p[0] + p[2]) - max(b[0], p[0]))
            iy = max(0, min(b[1] + b[3], p[1] + p[3]) - max(b[1], p[1]))
            inter = ix * iy
            b_area = b[2] * b[3]
            if b_area > 0 and inter / b_area > 0.6:
                keep = False
                break
        if keep:
            picked.append(b)

    out = [[
        round(b[0] / W, 4),
        round(b[1] / H, 4),
        round((b[0] + b[2]) / W, 4),
        round((b[1] + b[3]) / H, 4),
    ] for b in picked]
    return out[:60]


def detect_worker(frame_q: queue.Queue):
    """Pull frames off the queue, annotate, publish to _state."""
    get_model()
    while True:
        item = frame_q.get()
        if item is None:
            break
        frame, frame_id = item
        _state["annotated"] = detect_frame(frame)
        _state["frame_id"] = frame_id


def stream_source():
    """Read frames from the source video, feed the detector, loop forever."""
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        log(f"cannot open {VIDEO}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    frame_q = queue.Queue(maxsize=2)
    threading.Thread(target=detect_worker, args=(frame_q,), daemon=True).start()
    log(f"streaming {VIDEO} at ~{fps:.1f} fps")
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        try:
            frame_q.put((frame.copy(), frame_id), timeout=0.05)
        except queue.Full:
            pass
        frame_id += 1
        time.sleep(1.0 / fps)


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


class QuietHandler(WSGIRequestHandler):
    def log_message(self, *args):
        pass


def serpapi_upload_image(jpeg_bytes: bytes) -> str:
    """Upload a JPEG to SerpApi Image API, return image_id."""
    r = requests.post(
        "https://serpapi.com/image",
        files={"image": ("crop.jpg", jpeg_bytes, "image/jpeg")},
        data={"api_key": SERPAPI_KEY},
        timeout=SERPAPI_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["image_id"]


def serpapi_lens_ocr(jpeg_bytes: bytes, max_size: int = 900) -> dict:
    """
    Run Google Lens via SerpApi on a JPEG and extract whatever text Google
    "sees" in it (organic result titles/snippets + ai_overview + top matches).

    Returns {"text": [...], "visual": [...], "sources": [...]}
    """
    # Downscale/compress so the upload stays under SerpApi's 500KB cap.
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("bad image bytes")
    h, w = img.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("crop encode failed")
    if len(buf) > 500_000:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if len(buf) > 500_000:
        raise ValueError("image too large for SerpApi (500KB cap)")

    image_id = serpapi_upload_image(buf.tobytes())
    params = {
        "engine": "google_lens",
        "api_key": SERPAPI_KEY,
        "image_id": image_id,
        "hl": "en",
    }
    r = requests.get("https://serpapi.com/search.json", params=params, timeout=SERPAPI_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"SerpApi error: {data['error']}")

    text = []
    sources = []
    # SerpApi added a dedicated Lens text_results block (Aug 2026); use it
    # when the plan exposes it, otherwise fall back to organic_results + ai_overview.
    for item in (data.get("text_results") or []) + (data.get("organic_results") or []):
        t = (item.get("title") or "").strip()
        s = (item.get("snippet") or "").strip()
        link = item.get("link") or ""
        if t:
            text.append(t)
        if s:
            text.append(s)
        if t and link:
            sources.append({"title": t, "link": link})
    ao = data.get("ai_overview") or {}
    if isinstance(ao, dict):
        answer = (ao.get("answer") or ao.get("content") or "").strip()
        if answer:
            text.append(answer)
    visual = []
    for item in (data.get("visual_matches") or [])[:5]:
        t = (item.get("title") or "").strip()
        if t:
            visual.append(t)
    seen = set()
    uniq = []
    for t in text:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return {
        "text": uniq[:12],
        "visual": visual,
        "sources": sources[:6],
        "text_results": bool(data.get("text_results")),
        "raw_keys": list(data.keys()),
    }

def app(environ, start_response):
    path = environ.get("PATH_INFO", "/") or "/"
    method = environ.get("REQUEST_METHOD", "GET").upper()

    if path in ("/stream", "/stream/"):
        start_response("200 OK", [
            ("Content-Type", "multipart/x-mixed-replace; boundary=--frame"),
            ("Cache-Control", "no-cache"),
            ("Access-Control-Allow-Origin", "*"),
        ])
        return mjpeg_stream()

    if path == "/detect" and method == "POST":
        return handle_detect(environ, start_response)

    if path == "/ocr" and method == "POST":
        return handle_ocr(environ, start_response)

    if path == "/boxes" and method == "POST":
        return handle_boxes(environ, start_response)

    if path == "/textregions" and method == "POST":
        return handle_textregions(environ, start_response)

    if path == "/scan" and method == "POST":
        return handle_scan(environ, start_response)

    if path == "/scan.html":
        body = scan_html().encode()
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [body]

    if path == "/highlight-demo":
        body = highlight_demo_html().encode()
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [body]

    if path == "/webcam":
        body = webcam_html().encode()
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [body]

    if path in ("/", ""):
        body = index_html().encode()
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [body]

    start_response("404 Not Found", [("Content-Type", "text/plain")])
    return [b"not found"]


def handle_detect(environ, start_response):
    """Read a JPEG from the POST body, annotate it, return annotated JPEG."""
    try:
        length = int(environ.get("CONTENT_LENGTH", "0") or "0")
        raw = environ["wsgi.input"].read(length)
        if len(raw) == 0:
            raise ValueError("empty body")
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("bad JPEG")
        annotated = detect_frame(frame)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("encode failed")
        start_response("200 OK", [
            ("Content-Type", "image/jpeg"),
            ("Cache-Control", "no-cache"),
            ("Access-Control-Allow-Origin", "*"),
        ])
        return [buf.tobytes()]
    except Exception as e:
        log(f"/detect error: {e}")
        start_response("400 Bad Request", [("Content-Type", "text/plain")])
        return [f"error: {e}".encode()]


def handle_ocr(environ, start_response):
    """POST a JPEG crop -> run SerpApi Google Lens -> return extracted text JSON."""
    try:
        length = int(environ.get("CONTENT_LENGTH", "0") or "0")
        raw = environ["wsgi.input"].read(length)
        if len(raw) == 0:
            raise ValueError("empty body")
        result = serpapi_lens_ocr(raw)
        body = json.dumps(result).encode()
        start_response("200 OK", [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Cache-Control", "no-cache"),
            ("Access-Control-Allow-Origin", "*"),
        ])
        return [body]
    except Exception as e:
        log(f"/ocr error: {e}")
        start_response("502 Bad Gateway", [("Content-Type", "text/plain")])
        return [f"ocr error: {e}".encode()]


def handle_boxes(environ, start_response):
    """POST a JPEG frame -> YOLO detection -> JSON list of box coordinates."""
    try:
        length = int(environ.get("CONTENT_LENGTH", "0") or "0")
        raw = environ["wsgi.input"].read(length)
        if len(raw) == 0:
            raise ValueError("empty body")
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("bad JPEG")
        boxes = detect_boxes(frame)
        body = json.dumps({"boxes": boxes, "count": len(boxes)}).encode()
        start_response("200 OK", [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Cache-Control", "no-cache"),
            ("Access-Control-Allow-Origin", "*"),
        ])
        return [body]
    except Exception as e:
        log(f"/boxes error: {e}")
        start_response("502 Bad Gateway", [("Content-Type", "text/plain")])
        return [f"boxes error: {e}".encode()]


def handle_scan(environ, start_response):
    """POST a page JPEG -> DocLayout region scan + per-region Lens text -> JSON."""
    try:
        length = int(environ.get("CONTENT_LENGTH", "0") or "0")
        raw = environ["wsgi.input"].read(length)
        if len(raw) == 0:
            raise ValueError("empty body")
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("bad JPEG")
        result = scan_document(frame)
        body = json.dumps(result).encode()
        start_response("200 OK", [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Cache-Control", "no-cache"),
            ("Access-Control-Allow-Origin", "*"),
        ])
        return [body]
    except Exception as e:
        log(f"/scan error: {e}")
        start_response("502 Bad Gateway", [("Content-Type", "text/plain")])
        return [f"scan error: {e}".encode()]


def handle_textregions(environ, start_response):
    """POST a JPEG crop -> find text-like regions -> return normalized rects JSON."""
    try:
        length = int(environ.get("CONTENT_LENGTH", "0") or "0")
        raw = environ["wsgi.input"].read(length)
        if len(raw) == 0:
            raise ValueError("empty body")
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("bad JPEG")
        regions = find_text_regions(frame)
        body = json.dumps({"regions": regions, "count": len(regions)}).encode()
        start_response("200 OK", [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Cache-Control", "no-cache"),
            ("Access-Control-Allow-Origin", "*"),
        ])
        return [body]
    except Exception as e:
        log(f"/textregions error: {e}")
        start_response("502 Bad Gateway", [("Content-Type", "text/plain")])
        return [f"textregions error: {e}".encode()]


_last_frame_id = {"value": -1}


def mjpeg_stream():
    while True:
        frame = _state["annotated"]
        frame_id = _state.get("frame_id", -1)
        if frame is None:
            time.sleep(0.05)
            continue
        if frame_id == _last_frame_id["value"]:
            time.sleep(0.02)
            continue
        _last_frame_id["value"] = frame_id
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n"
            + f"Content-Length: {len(buf)}\r\n\r\n".encode()
            + buf.tobytes()
            + b"\r\n"
        )


def scan_html():
    return """<!doctype html>
<html><head><meta charset=utf-8>
<title>Document Scan</title>
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;background:#fafafa;color:#1c1c1c}
h1{font-size:22px}
label{border:2px dashed #9aa5b1;border-radius:12px;padding:28px;display:block;text-align:center;cursor:pointer;background:#fff}
label:hover{border-color:#4285f4}
input[type=file]{display:none}
button{background:#4285f4;color:#fff;border:0;border-radius:8px;padding:10px 18px;font-size:15px;cursor:pointer}
button:disabled{opacity:.5}
img{max-width:100%;border:1px solid #ddd;border-radius:8px}
.row{display:flex;gap:16px;align-items:flex-start}
.col{flex:1}
.region{border-left:3px solid #4285f4;padding:6px 10px;margin:8px 0;background:#fff;border-radius:6px}
.region b{color:#4285f4}
.src{color:#666;font-size:12px;word-break:break-all}
#status{color:#777;margin-left:8px}
</style></head><body>
<h1>YOLO + Google Lens Document Scan</h1>
<p>Detect→Crop→Lens: DocLayout-YOLO finds Title/Text/Table/Picture regions; each text region is uploaded to Google Lens via SerpApi for text extraction.</p>
<label>📄 Choose a scanned page… <input type=file id=file accept=image/*></label>
<p><button id=go disabled>Scan page</button><span id=status></span></p>
<div class=row><div class=col><img id=img alt=preview hidden></div>
<div class=col id=out></div></div>
<script>
const fileInput=document.getElementById('file'),img=document.getElementById('img'),go=document.getElementById('go'),out=document.getElementById('out'),statusEl=document.getElementById('status');
let jpegBlob=null;
fileInput.onchange=()=>{const f=fileInput.files[0];if(!f)return;const reader=new FileReader();reader.onload=e=>{const J=document.createElement('canvas');J.width=1600;const i=new Image();i.onload=()=>{const s=Math.min(1,J.width/i.width);J.width=Math.round(i.width*s);J.height=Math.round(i.height*s);J.getContext('2d').drawImage(i,0,0,J.width,J.height);J.toBlob(b=>{jpegBlob=b;img.src=URL.createObjectURL(b;img.hidden=false;go.disabled=false;statusEl.textContent='';},'image/jpeg',0.9);};i.src=URL.createObjectURL(f;};reader.readAsDataURL(f;)};
go.onclick=async()=>{if(!jpegBlob)return;go.disabled=true;statusEl.textContent='Scanning… (lens per region takes ~30s)';out.innerHTML='';try{const r=await fetch('/scan',{method:'POST',body:jpegBlob});if(!r.ok){const t=await r.text();throw new Error(t);}const d=await r.json();statusEl.textContent='Done: '+d.count+' regions ('+d.detector+'). ';out.innerHTML='<h2>Detected regions</h2>';for(const R of d.regions){const t=(R.text||[]).join(' ');const src=(R.sources||[]).map(s=>'<a class=src href="'+(s.link||'')+'" target=_blank rel=noopener>'+s.title+'</a>').join(' ');out.innerHTML+='<div class=region><b>'+R.class.toUpperCase()+'</b> conf='+R.conf.toFixed(2)+' <code>'+JSON.stringify(R.bbox)+'</code><br>'+(t?t:'<i>(no text / visual match)</i>')+' '+src+'</div>';}console.log(d);}catch(e){statusEl.textContent='Error: '+e.message;}go.disabled=false;};
</script>
</body></html>"""


def index_html():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>YOLOv8 Live Detection</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background: #111; color: #eee;
         margin: 0; padding: 24px; text-align: center; }}
  h1 span {{ color: #4fc3f7; }}
  img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 24px rgba(0,0,0,.6); }}
  .meta {{ color: #888; font-size: 14px; margin-top: 14px; }}
  a {{ color: #4fc3f7; }}
</style>
</head>
<body>
  <h1>YOLOv8 <span>Live</span> Detection</h1>
  <div style="margin: 16px 0;">
    <a href="/webcam"><b>&#128248; Use my webcam</b></a> &middot; or watch the
    <a href="/stream">demo video stream</a>
  </div>
  <img src="/stream" alt="YOLOv8 live video stream">
  <div class="meta">
    Model: YOLOv8n (COCO) &middot; confidence &ge; {CONF} &middot; source: {os.path.basename(VIDEO)} (looped)
  </div>
  <div class="meta">Objects: person &middot; car &middot; bicycle &middot; bus &middot; truck &middot; traffic light &middot; stop sign &middot; etc.</div>
</body>
</html>"""


def _render_highlights(img_bgr: np.ndarray, hex_fill: str = "#80c4ff",
                        alpha: float = 0.30) -> np.ndarray:
    """Draw Google-Lens-style highlights over detected text regions."""
    out = img_bgr.copy()
    regions = find_text_regions(img_bgr)
    r = int(hex_fill[1:3], 16)
    g = int(hex_fill[3:5], 16)
    b = int(hex_fill[5:7], 16)
    for x1n, y1n, x2n, y2n in regions:
        x1 = int(x1n * out.shape[1]); y1 = int(y1n * out.shape[0])
        x2 = int(x2n * out.shape[1]); y2 = int(y2n * out.shape[0])
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (b, g, r), -1)
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
        cv2.rectangle(out, (x1, y1), (x2, y2), (b, g, r), 2)
    return out


def highlight_demo_html():
    """Demo page showing the Google-Lens-style text highlight effect."""
    import base64
    from io import BytesIO

    try:
        from PIL import Image, ImageDraw, ImageFont
        has_pil = True
    except Exception:
        has_pil = False

    panels = []

    if has_pil:
        # --- stop sign (white text on red) ---
        sign = Image.new("RGB", (420, 420), (200, 30, 30))
        d = ImageDraw.Draw(sign)
        try:
            f1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 88)
            f2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
        except Exception:
            f1 = f2 = ImageFont.load_default()
        d.text((62, 55), "STOP", fill="white", font=f1)
        d.text((58, 215), "LIMIT  25", fill="white", font=f2)
        sign_arr = cv2.cvtColor(np.array(sign), cv2.COLOR_RGB2BGR)
        hl_sign = _render_highlights(sign_arr)
        buf = cv2.imencode(".jpg", hl_sign, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
        panels.append(("Red sign (white text on red)", base64.b64encode(buf).decode(), len(find_text_regions(sign_arr))))

        # --- document (dark text on light) ---
        doc = Image.new("RGB", (720, 360), "#f6f4ef")
        d2 = ImageDraw.Draw(doc)
        try:
            f3 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
            f4 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 38)
        except Exception:
            f3 = f4 = ImageFont.load_default()
        d2.text((36, 28), "WELCOME TO SERPVIZ", fill="#1a1a1a", font=f3)
        d2.text((36, 118), "Camera  a  sign  or  page  and  it", fill="#222", font=f4)
        d2.text((36, 178), "will  highlight  text  &  read  it", fill="#222", font=f4)
        d2.text((36, 248), "License plate: 7ABC234", fill="#111", font=f4)
        doc_arr = cv2.cvtColor(np.array(doc), cv2.COLOR_RGB2BGR)
        hl_doc = _render_highlights(doc_arr)
        buf2 = cv2.imencode(".jpg", hl_doc, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
        panels.append(("Document (dark text on paper)", base64.b64encode(buf2).decode(), len(find_text_regions(doc_arr))))
    else:
        panels.append(("PIL unavailable", "", 0))

    cards = "".join(
        f'<div class="card"><h2>{title}</h2><img src="data:image/jpeg;base64,{b64}" alt="{title}">'
        f'<p>{n} text regions highlighted</p></div>'
        for title, b64, n in panels
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Text Highlight Demo</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background: #111; color: #eee;
         margin: 0; padding: 24px; text-align: center; }}
  h1 span {{ color: #4fc3f7; }}
  .cards {{ display: flex; gap: 24px; justify-content: center; flex-wrap: wrap; margin-top: 18px; }}
  .card {{ background: #1c1c20; border-radius: 12px; padding: 14px; border: 1px solid #333; }}
  .card h2 {{ margin: 6px 0; font-size: 15px; }}
  .card p {{ color: #888; font-size: 13px; }}
  img {{ border-radius: 8px; max-height: 420px; }}
  a {{ color: #4fc3f7; }}
</style></head>
<body>
  <h1>Google-Lens-style <span>text highlight</span></h1>
  <p style="color:#888">Light translucent highlight over text regions detected on the webcam feed.</p>
  <div class="cards">{cards}</div>
  <p style="margin-top:20px"><a href="/webcam">Try it live on your camera &rarr;</a></p>
</body></html>"""


def webcam_html():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>YOLOv8 Webcam OCR</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background: #111; color: #eee;
         margin: 0; padding: 24px; text-align: center; }}
  h1 span {{ color: #4fc3f7; }}
  .layout {{ display: flex; gap: 20px; justify-content: center; align-items: flex-start; flex-wrap: wrap; }}
  .stage {{ position: relative; display: inline-block; }}
  video, canvas {{ max-width: 100%; border-radius: 10px; box-shadow: 0 4px 24px rgba(0,0,0,.6); }}
  canvas {{ display: block; max-width: 100%; }}
  .panel {{ background: #1c1c20; border: 1px solid #333; border-radius: 10px; padding: 16px;
            width: 340px; text-align: left; font-size: 14px; }}
  .panel h3 {{ margin: 0 0 10px; color: #4fc3f7; font-size: 15px; }}
  .ocr-block {{ background: #121216; border-left: 3px solid #4fc3f7; padding: 8px 10px;
               margin: 6px 0; border-radius: 6px; }}
  .ocr-block .cls {{ color: #ffb86b; font-weight: bold; font-size: 12px; }}
  .ocr-block .txt {{ margin-top: 4px; line-height: 1.4; color: #ddd; }}
  .ocr-block a {{ color: #4fc3f7; font-size: 12px; }}
  .meta {{ color: #888; font-size: 13px; margin-top: 12px; line-height: 1.6; }}
  .warn {{ color: #ffb86b; font-size: 13px; margin-top: 8px; }}
  button {{ background: #2a2a2e; color: #eee; border: 1px solid #444; border-radius: 8px;
            padding: 8px 18px; font-size: 15px; cursor: pointer; margin: 4px; }}
  button:hover {{ border-color: #4fc3f7; }}
  .loading {{ color: #ffb86b; font-size: 12px; }}
</style>
</head>
<body>
  <h1>YOLOv8 <span>Webcam</span> + Google Lens Text</h1>
  <p class="warn" id="permMsg"></p>
  <div class="layout">
    <div>
      <div class="stage">
        <canvas id="out" width="960" height="540"></canvas>
      </div>
      <div class="meta">
        <button id="startBtn">&#128264; Start camera</button>
        <button id="toggleBtn" style="display:none;">Pause</button>
        <button id="ocrOnceBtn" style="display:none;">&#128269; OCR all detected</button><br>
        <span id="stat">not started</span>
      </div>
    </div>
    <div class="panel" id="ocrPanel">
      <h3>&#128269; Extracted Text (Google Lens)</h3>
      <div id="ocrBody"><span class="meta">Detected signs/documents/books will appear here.</span></div>
    </div>
  </div>
  <script>
    const canvas = document.getElementById('out');
    const stat = document.getElementById('stat');
    const startBtn = document.getElementById('startBtn');
    const toggleBtn = document.getElementById('toggleBtn');
    const ocrOnceBtn = document.getElementById('ocrOnceBtn');
    const ocrBody = document.getElementById('ocrBody');
    const ctx = canvas.getContext('2d', {{ willReadFrequently: true }});
    let stream = null, running = false, lastBoxes = [];
    // Classes considered text-bearing in COCO: stop sign=11, book=73, tv=72
    const TEXT_CLASSES = new Set([11, 73, 72]);
    const TEXT_CLASS_NAMES = {{ 11: 'stop sign', 73: 'book', 72: 'tv screen' }};
    const video = document.createElement('video');
    video.setAttribute('playsinline', '');
    let ocrBusy = false, ocrPending = new Set();
    // boxKey -> array of text region rects in FULL-frame canvas coords
    const textHighlights = new Map();
    const highlightFetchTimes = new Map(); // boxKey -> last fetch ms
    const HL_FRESH_MS = 700;

    async function start() {{
      try {{
        stream = await navigator.mediaDevices.getUserMedia({{
          video: {{ width: {{ ideal: 1280 }}, height: {{ ideal: 720 }}, facingMode: 'environment' }}
        }});
        video.srcObject = stream;
        await video.play();
        canvas.width = video.videoWidth || 960;
        canvas.height = video.videoHeight || 540;
        running = true;
        startBtn.style.display = 'none';
        toggleBtn.style.display = '';
        ocrOnceBtn.style.display = '';
        document.getElementById('permMsg').textContent = '';
        frameLoop();
      }} catch (e) {{
        document.getElementById('permMsg').textContent =
          'Camera access denied or unavailable: ' + e.message;
      }}
    }}

    async function frameLoop() {{
      if (!running) return;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.6));
      try {{
        const resp = await fetch('/boxes', {{ method: 'POST', body: blob }});
        if (resp.ok) {{
          const data = await resp.json();
          lastBoxes = data.boxes || [];
          drawBoxes(lastBoxes);
          stat.textContent = 'running &mdash; ' + lastBoxes.length + ' object(s)';
          // Auto-OCR new text-bearing objects
          lastBoxes.forEach(b => triggerAutoOcr(b));
          // Keep Google-Lens-style text highlights fresh
          updateTextHighlights(lastBoxes);
        }} else {{
          stat.textContent = 'server error';
        }}
      }} catch (e) {{
        stat.textContent = 'network error: ' + e.message;
      }}
      setTimeout(frameLoop, 120);
    }}

    function boxKey(box) {{
      return Math.round(box.xyxy[0]) + ',' + Math.round(box.xyxy[1]) + ',' +
             Math.round(box.xyxy[2]) + ',' + Math.round(box.xyxy[3]);
    }}

    function makeCrop(box, pad) {{
      const [x1, y1, x2, y2] = box.xyxy;
      const sx = Math.max(0, x1 - pad), sy = Math.max(0, y1 - pad);
      const sw = Math.min(canvas.width - sx, (x2 - x1) + pad * 2);
      const sh = Math.min(canvas.height - sy, (y2 - y1) + pad * 2);
      const crop = document.createElement('canvas');
      crop.width = Math.round(sw); crop.height = Math.round(sh);
      crop.getContext('2d').drawImage(video, sx, sy, sw, sh, 0, 0, crop.width, crop.height);
      return {{ canvas: crop, sx: sx, sy: sy, sw: sw, sh: sh }};
    }}

    async function updateTextHighlights(boxes) {{
      const now = Date.now();
      for (const b of boxes) {{
        if (!TEXT_CLASSES.has(b.cls)) continue;
        const key = boxKey(b);
        if (now - (highlightFetchTimes.get(key) || 0) < HL_FRESH_MS) continue;
        highlightFetchTimes.set(key, now);
        const {{ canvas: crop, sx, sy }} = makeCrop(b, 6);
        if (crop.width < 8 || crop.height < 8) continue;
        await new Promise(res => crop.toBlob(async (cropBlob) => {{
          try {{
            const resp = await fetch('/textregions', {{ method: 'POST', body: cropBlob }});
            if (resp.ok) {{
              const data = await resp.json();
              const highlights = (data.regions || []).map(r => {{
                return {{
                  x1: sx + r[0] * crop.width,
                  y1: sy + r[1] * crop.height,
                  x2: sx + r[2] * crop.width,
                  y2: sy + r[3] * crop.height
                }};
              }});
              textHighlights.set(key, highlights);
            }}
          }} catch (e) {{}}
          res();
        }}, 'image/jpeg', 0.8));
      }}
      // drop stale keys that are no longer on screen
      const live = new Set(boxes.map(boxKey));
      for (const k of textHighlights.keys()) {{
        if (!live.has(k)) textHighlights.delete(k);
      }}
    }}

    function drawBoxes(boxes) {{
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      boxes.forEach(b => {{
        const [x1, y1, x2, y2] = b.xyxy;
        // Google-Lens-style light translucent highlight over detected text
        const key = boxKey(b);
        const hls = textHighlights.get(key);
        if (hls && hls.length) {{
          ctx.save();
          hls.forEach(h => {{
            const rw = h.x2 - h.x1, rh = h.y2 - h.y1;
            // soft translucent glow fill
            ctx.fillStyle = 'rgba(128, 196, 255, 0.28)';
            ctx.fillRect(h.x1, h.y1, rw, rh);
            ctx.strokeStyle = 'rgba(80, 160, 255, 0.85)';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(h.x1, h.y1, rw, rh);
          }});
          ctx.restore();
        }}
        const isText = TEXT_CLASSES.has(b.cls);
        ctx.strokeStyle = isText ? '#4fc3f7' : '#4caf50';
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = isText ? '#4fc3f7' : '#4caf50';
        const label = b.name + ' ' + b.conf.toFixed(2);
        ctx.font = 'bold 16px sans-serif';
        const tw = ctx.measureText(label).width;
        ctx.fillRect(x1, Math.max(y1 - 22, 0), tw + 10, 22);
        ctx.fillStyle = '#111';
        ctx.fillText(label, x1 + 5, Math.max(y1 - 6, 16));
      }});
    }}

    function triggerAutoOcr(box) {{
      if (!TEXT_CLASSES.has(box.cls)) return;
      const key = boxKey(box);
      if (ocrPending.has(key) || ocrBusy) return;
      ocrPending.add(key);
      const {{ canvas: crop }} = makeCrop(box, 6);
      if (crop.width < 8 || crop.height < 8) return;
      crop.toBlob(async (cropBlob) => {{
        ocrPending.delete(key);
        ocrBusy = true;
        setOcrStatus('OCR-ing ' + TEXT_CLASS_NAMES[box.cls] + '...');
        try {{
          const resp = await fetch('/ocr', {{ method: 'POST', body: cropBlob }});
          if (resp.ok) {{
            const data = await resp.json();
            if (data.text.length) appendOcr(box, data);
            else setOcrStatus('No text recognized in ' + box.name + '.');
          }}
        }} catch (e) {{ setOcrStatus('OCR error: ' + e.message); }}
        ocrBusy = false;
      }}, 'image/jpeg', 0.85);
    }}

    function appendOcr(box, data) {{
      const div = document.createElement('div');
      div.className = 'ocr-block';
      const head = document.createElement('div');
      head.className = 'cls';
      head.textContent = '\u2b50 ' + box.name;
      div.appendChild(head);
      data.text.slice(0, 4).forEach(t => {{
        const p = document.createElement('div');
        p.className = 'txt';
        p.textContent = t;
        div.appendChild(p);
      }});
      if (data.sources.length) {{
        const a = document.createElement('a');
        a.href = data.sources[0].link;
        a.target = '_blank';
        a.textContent = 'view source';
        div.appendChild(document.createElement('br'));
        div.appendChild(a);
      }}
      ocrBody.prepend(div);
      setOcrStatus('Text extracted from ' + box.name + '.');
      while (ocrBody.children.length > 8) ocrBody.removeChild(ocrBody.lastChild);
    }}

    function setOcrStatus(msg) {{
      stat.textContent = msg;
    }}

    ocrOnceBtn.onclick = async () => {{
      if (ocrBusy) return;
      ocrBusy = true;
      setOcrStatus('OCR-ing all detected objects...');
      for (const b of lastBoxes) {{
        const {{ canvas: crop }} = makeCrop(b, 6);
        if (crop.width < 8 || crop.height < 8) continue;
        const cropBlob = await new Promise(res => crop.toBlob(res, 'image/jpeg', 0.85));
        try {{
          const resp = await fetch('/ocr', {{ method: 'POST', body: cropBlob }});
          if (resp.ok) {{
            const data = await resp.json();
            if (data.text.length) appendOcr(b, data);
          }}
        }} catch (e) {{}}
      }}
      ocrBusy = false;
      setOcrStatus('OCR pass finished.');
    }};

    startBtn.onclick = start;
    toggleBtn.onclick = () => {{
      running = !running;
      toggleBtn.textContent = running ? 'Pause' : 'Resume';
      if (running) frameLoop();
    }};
    window.addEventListener('beforeunload', () => {{
      if (stream) stream.getTracks().forEach(t => t.stop());
    }});
  </script>
</body>
</html>"""


if __name__ == "__main__":
    threading.Thread(target=stream_source, daemon=True).start()
    log(f"listening on http://{HOST}:{PORT}/")
    httpd = make_server(HOST, PORT, app, ThreadingWSGIServer, QuietHandler)
    httpd.serve_forever()