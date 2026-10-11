# Painter — repository audit

Audit performed before any change was made, on branch `master`, commit
`faa06d9` ("first commit"), the repository's only commit.

## 1. What the repository actually contained

```
Painter/
└── yolo_demo/
    ├── live_feed_server.py        1053 lines, single-file demo
    ├── index.html                 static "detection results" page
    ├── doclayout.pt               40.7 MB  DocLayout-YOLO checkpoint
    ├── yolov8n.pt                 6.5 MB   COCO detector
    ├── yolo12n.pt                 5.6 MB   COCO detector
    ├── yolo26n.pt                 5.5 MB   COCO detector
    ├── sample_video.mp4           6.0 MB   demo footage
    ├── comparison*.jpg            before/after images
    ├── input.jpg, live_frame_sample.jpg
    ├── feed.log, http.log         runtime logs from a previous session
    ├── runs/detect/predict*/      YOLO output images
    └── __pycache__/               stale bytecode (CPython 3.13)
```

20 files, 126 MB on disk (60 MB of that is `.git`).

**Absent:** README, `requirements.txt`, `.env.example`, any test, any CI
configuration, any backend service, any frontend application, any database, any
Document IR, any renderer, any geometry engine, any verification logic.

## 2. What is implemented vs. what is claimed

| Area | Status | Evidence |
|---|---|---|
| MJPEG video stream + browser webcam demo | Implemented | `/stream`, `/webcam`, `index.html` |
| COCO object detection (`yolov8n`) | Implemented, works | `feed.log`: "loaded yolov8n.pt" |
| DocLayout-YOLO document layout detection | **Broken as committed** | `feed.log`: `layout model failed: 'Conv' object has no attribute 'bn'; using fallback` |
| CV text-region fallback | **Never executed** | the function cannot even be imported (see §3) |
| SerpApi Google Lens OCR | Partially implemented, wrong model | `serpapi_lens_ocr()` treats search-result titles/snippets as document text |
| SerpApi image identification | Not implemented | no visual/exact match parsing, no candidate ranking, no download |
| Coordinate mapping / transforms | Not implemented | detections are reported in whatever space the client uploaded |
| Document IR, geometry, typography, tables, renderers, verification | Not implemented | — |
| Tests | None | — |

## 3. Defects found in `yolo_demo/live_feed_server.py`

The module **cannot be imported or run**; `python3 -m py_compile` fails:

1. **`IndentationError`** — `_scan_regions_fallback()` (around line 98): the body
   of `for r in rects:` is missing and the following `if bands and ...` is
   dedented to function level, where `bands` is undefined.
2. **Corrupted source bytes** — line 182 contains Thaana script characters
   inside the JPEG quality literal:
   `cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, ގެންވެތި90])`.
3. **Unbalanced parentheses** — `scan_document()`:
   `for b in sorted(boxes, key=lambda b: (b["xyxy"][1], b["xyxy"][0]):` is missing
   its closing `)`.
4. **Broken JavaScript** — `webcam_html()` contains a stray `92` token
   (`}}92`) in the middle of `updateTextHighlights`.
5. **Type error** — `rects = [(w[0] * w, ...) for w in words]` multiplies a
   rectangle by itself because the loop variable `w` shadows the frame width.
6. **Hardcoded credential** — line 22 embeds a live SerpApi API key in source
   control. This is a public credential exposure and must be rotated.
7. **Silent degradation** — when the layout model failed, the server logged a
   warning and continued, so `/scan` returned CV-fallback results that the UI
   presented as "DocLayout-YOLO" output.

## 4. Verified behaviour of the shipped model (reproduced)

`doclayout.pt` is a **pickled DocLayout-YOLO (YOLOv10m) checkpoint**, not a plain
Ultralytics checkpoint:

* `ultralytics.YOLO("doclayout.pt")` →
  `ModuleNotFoundError: No module named 'doclayout_yolo'`
  (the pickle references `doclayout_yolo.models.yolov10.model.YOLOv10`).
* After installing the `doclayout_yolo` runtime the model loads, but inference
  then requires `py-cpuinfo`.
* `doclayout_yolo.YOLOv10("doclayout.pt")` loads **and predicts correctly**.
* `doclayout_yolo.YOLO("doclayout.pt")` fails with
  `AttributeError: 'dict' object has no attribute 'shape'`.

Real class set reported by the checkpoint (11 classes, DocStructBench):

```
Caption, Footnote, Formula, List-item, Page-footer, Page-header,
Picture, Section-header, Table, Text, Title
```

Two consequences drove the architecture:

* **The model has no classes for signature, stamp, logo, checkbox, form field,
  line, rectangle, circle or page number.** YOLO can never be asked for them.
  Painter recovers them with deterministic OpenCV heuristics
  (`painter/heuristics.py`) and supports a purpose-trained model through
  `PAINTER_CUSTOM_MODEL` (see `docs/DATASET.md`).
* **`Text` boxes are line level, not paragraphs.** A measured inference on a
  synthetic A4 page returned 6 separate `Text` boxes plus `Title` and `Table`.
  Grouping lines into paragraphs is therefore a required pipeline stage, not an
  optional refinement.

Measured inference cost on this sandbox (6 vCPU, no GPU): about **3.0 s** per
page at `imgsz=1024`.

## 5. Environment verified during the audit

| Component | Version / result |
|---|---|
| Python | 3.12.3 |
| OpenCV (`opencv-python-headless`) | installed during the audit |
| ultralytics | installed; used for COCO models only |
| torch | CPU build |
| tesseract | 5.3.4 (apt), `pytesseract` installed |
| WeasyPrint | 70.0 (HTML to PDF) |
| Chromium | `/usr/bin/chromium` (rasterisation and PDF fallback) |
| poppler-utils | `pdftoppm`, `pdftotext`, `pdfinfo` |
| python-docx, scikit-image, FastAPI, uvicorn, playwright | installed |

## 6. Reuse decisions

* The demo's **detection idea** (DocLayout-YOLO + COCO + CV fallback) is reused
  in `painter/detection.py`, with the loader chain fixed and every attempt
  reported instead of silently degrading.
* The demo's **Lens call shape** (`POST /image` → `image_id` → `search.json`)
  is reused in `painter/serpapi_client.py`, with retries, timeouts, caching, a
  request budget and correct error classification added.
* The demo's **text-region routine** is reused after repair as
  `painter/detection.py::find_text_regions` (the multiply-by-itself bug, the
  undefined `bands` reference and the indentation error are fixed).
* `yolo_demo/live_feed_server.py` now imports those modules instead of
  duplicating them, keeps every original endpoint and page, and reads the API
  key from the environment only.

## 7. Follow-up required from the repository owner

1. **Rotate the SerpApi key** that was committed in `yolo_demo/live_feed_server.py`
   — it must be considered compromised. Set the new value as `SERPAPI_KEY` in the
   environment; the code no longer contains a default.
2. Decide whether the 60 MB of binary demo assets (video, comparison images,
   `runs/`, logs, `__pycache__`) should stay in version control; they are not
   needed by the application. They were left untouched.

## 8. Status after the repair

Everything listed as missing in section 2 and every defect in section 3 was
addressed in the same change set that ships this document:

| Item | Status |
|---|---|
| Importable `painter` package (config, schemas, logging, errors) | done |
| Geometry, typography, tables, structure, heuristics, assets | done |
| OCR provider chain (Tesseract local, Google Lens via SerpApi, explicit `none`) | done |
| HTML / PDF / DOCX renderers with editable text and native tables | done |
| Visual verification with a bounded correction loop | done |
| FastAPI service, web UI, job manager, SQLite repository, CLI | done |
| `yolo_demo/live_feed_server.py` import error and `_find_text_regions` bug | fixed |
| Hardcoded SerpApi key | removed (environment only) |
| `pytest` suite | 121 tests, all passing |
| Selftest on the generated page | 4/4 expected elements, table cells 12/12, OCR 0.846 |

Deviations from the original claim, kept explicit rather than papered over:

* no fine-tuned `painter_doclayout.pt` is trained or shipped; the code path
  exists and is documented in `docs/DATASET.md`;
* the document title is promoted from `Section-header` by the structure layer
  because the shipped model scores that label above `Title` (see README §7);
* the API returns typed errors with their own status codes, but it is an
  internal service: it has one shared token, no user accounts and no TLS.
