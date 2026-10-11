# Painter architecture

This document describes how the engine is put together, why each decision was
made, and where the seams are for extending it. It reflects the code as
implemented; anything not implemented is listed as such in
[README §7](../README.md#7-known-limitations).

---

## 1. Design principles

1. **The Document IR is the single source of truth.** Detection, OCR, geometry,
   typography, tables and editing all mutate the IR; every renderer only reads
   it. Nothing bypasses it, which is why editing an element never requires
   re-running analysis.
2. **Deterministic where measurable, probabilistic where ambiguous.** Geometry,
   borders, columns and units are computed; region class, reading order and
   semantic grouping may come from a model and are always marked with a
   confidence and a provenance record.
3. **Never fabricate.** Missing OCR text is flagged (`TEXT_MISSING`), an inferred
   paper size is recorded as an assumption, a guessed font is marked
   `estimated=true`, and an unavailable model is reported instead of silently
   replaced by a stub.
4. **Everything is resumable and auditable.** Each stage writes a JSON artefact
   plus a database row; a failed run resumes from the failing stage, and every
   element carries provenance entries naming the stage, method and provider.
5. **Modularity over cleverness.** Each concern has one module with an explicit
   input/output contract, so a component can be replaced (a different OCR engine,
   a purpose-trained detector, a different rasteriser) without touching the rest.

---

## 2. Module map

| Module | Responsibility | Key entry points |
|---|---|---|
| `config.py` | environment-driven settings, `.env` loading, model resolution | `load_settings`, `get_settings`, `Settings` |
| `errors.py` | typed error taxonomy with stable codes and HTTP status mapping | `PainterError` and subclasses |
| `logging_utils.py` | structured logs, secret redaction, per-stage timers | `get_logger`, `redact`, `stage_timer` |
| `schemas.py` | Document IR models and validation helpers | `DocumentIR`, `PageIR`, `Element`, `TableContent`, `Typography` |
| `geometry.py` | units, page-format inference, coordinate mapping, columns, margins, overlap | `infer_page_format`, `detect_page_format`, `map_bbox_original_to_page`, `analyse_page_geometry` |
| `preprocess.py` | validation, loading (image/PDF), orientation, perspective, deskew, illumination, resolution, line-pitch measurement | `validate_upload`, `load_document_pages`, `preprocess_page`, `estimate_text_line_pitch` |
| `detection.py` | YOLO loader chain, class mapping, post-processing, CV fallbacks | `YoloDetector`, `postprocess_detections`, `find_text_regions`, `find_photo_regions` |
| `heuristics.py` | shapes, lines, checkboxes, signatures, stamps, logos | `detect_shapes`, `detect_checkboxes`, `looks_like_signature`, `looks_like_stamp` |
| `ocr.py` | OCR provider interface, Tesseract provider, Lens fallback, chain | `build_ocr_chain`, `TesseractProvider`, `LensTextProvider`, `OcrChain` |
| `serpapi_client.py` | Google Lens via SerpApi: retries, caching, budgets, SSRF-safe downloads | `SerpApiClient`, `SerpApiResponse`, `SerpApiError` |
| `tables.py` | table geometry, borders, merged cells, cell OCR | `detect_table_regions`, `analyse_table_geometry`, `reconstruct_table` |
| `structure.py` | element assembly, reading order, sections, IR validation | `build_page_ir`, `order_reading`, `build_sections`, `validate_document` |
| `typography.py` | calibrated font size/weight/slant/colour/family estimation | `analyse_ink`, `estimate_typography`, `estimate_page_typography` |
| `assets.py` | asset extraction, candidate ranking, safe download | `extract_asset`, `rank_candidates`, `download_candidate` |
| `render/` | HTML, PDF and DOCX renderers | `render_document`, `render_document_for_print`, `render_pdf`, `render_docx` |
| `verify.py` | rasterisation, comparison metrics, diff maps, bounded correction loop | `Rasterizer`, `compare_images`, `region_report`, `correction_loop` |
| `storage.py` | workspaces on disk, SQLite repository, job records | `DocumentWorkspace`, `Repository` |
| `pipeline.py` | stage orchestration, resumability, progress, job results | `Pipeline`, `JobOptions`, `JobResult` |
| `editing.py` | IR edits and re-render without re-analysis | `apply_element_edit`, `re_render`, `export_artifacts` |
| `api.py` | FastAPI application, upload, jobs, elements, exports, static files | `create_app` |
| `cli.py` | command line interface | `main` |
| `testsupport.py` | synthetic document generator, selftest, benchmark harness | `build_synthetic_document`, `run_selftest`, `run_benchmark` |

---

## 3. Pipeline

```
validate → preprocess → detect → recognize → reconstruct → render → verify → export
```

`Pipeline.run()` walks the stage list and, for each stage, calls `_run_stage`,
which writes `stages/<stage>.json` inside the document workspace and a
`document_jobs` progress entry. On start, stages whose artefact already exists are
loaded instead of recomputed (unless `JobOptions.force` is set), which is what
makes a failed run cheap to resume.

### Stage contracts

| Stage | Reads | Writes (artefact) |
|---|---|---|
| `validate` | uploaded bytes | format, size, SHA-256, page count |
| `preprocess` | original file | `pages/page_NN.png`, `pages/page_NN_preview.jpg`, transforms, metrics, assumptions |
| `detect` | processed pages | per-page detections with class, confidence, bbox, source |
| `recognize` | processed pages + detections | OCR text per detection (provider, confidence, uncertainty) |
| `reconstruct` | detections + OCR + geometry | `rendered/document_ir.json` (validated) |
| `render` | Document IR | `rendered/reconstruction.html`, `.pdf`, `.docx` |
| `verify` | IR + scan + rendered HTML | `verify/` metrics, diff maps, iterations, corrected IR |
| `export` | IR + rendered files | `exports/` files and their metadata |

Job states follow the brief: `QUEUED → PREPROCESSING → DETECTING → RECOGNIZING
→ RECONSTRUCTING → RENDERING → VERIFYING → COMPLETED`, plus `FAILED` with a typed
error and a stage pointer.

---

## 4. Coordinate spaces

Three spaces are used and never mixed:

| Space | Definition | Where |
|---|---|---|
| `original` | pixels of the uploaded file | `Element.original_bbox`, `PageIR.transforms` |
| `page` | pixels of the processed page (`PageIR.width_px × height_px`) | detection, OCR, `Element.bbox` |
| `normalized` | `0..1` fractions of the processed page | `Element.norm_bbox`, storage |

Millimetres are derived from `PageFormat` and are what the renderers consume
(`PageIR.mm_per_px_x/y`). `geometry.map_bbox_original_to_page` replays the
recorded `PageTransform` list (perspective, rotate, deskew, resize, crop) so a box
can be mapped in either direction. Every transform is a data record, not a
side effect, which is why boxes stay mappable after cropping, rotation or
perspective correction.

### Page-format inference

A bitmap cannot prove paper size, so `infer_page_format` reasons explicitly and
returns its assumptions:

1. the aspect ratio selects a **family** (ISO 216 √2 → A6/A5/A4/A3; 1.294 →
   Letter/Legal/Tabloid) — this is scale invariant and therefore reliable;
2. the most common member is assumed (A4, or Letter);
3. the measured text line pitch cross-checks the assumption: `pitch / dpi × 72 /
   1.2` must be a plausible body size (5–20 pt). If it is not, the neighbouring
   formats in the same family are tried and the closest is used;
4. the effective DPI is derived from the chosen width and the pixel count, and
   the whole reasoning is stored in `PageFormat.notes` and surfaced in the job
   result as an assumption.

The line pitch comes from the **autocorrelation** of the row-ink profile
(`estimate_text_line_pitch`), which is robust to a text line splitting into
several ink bands; band counting is not.

---

## 5. Detection

`YoloDetector` wraps a checkpoint with lazy loading and a loader chain
(`doclayout_yolo.YOLOv10` → `doclayout_yolo.YOLO` → `ultralytics.YOLO`), each
attempt recorded in a `ModelLoadReport` with its error. A model that loads but
cannot infer is rejected by a smoke prediction, so "loaded" always means usable.

The bundled `doclayout.pt` is a DocLayout-YOLO (YOLOv10) checkpoint whose real
classes were verified by running it: `Caption, Footnote, Formula, List-item,
Page-footer, Page-header, Picture, Section-header, Table, Text, Title`. Those 11
map onto the IR through `CLASS_TO_ELEMENT_TYPE`. Classes the model was never
trained on (signature, stamp, logo, checkbox, form field, shapes, page numbers)
are **never invented**: they are recovered by `heuristics.py` and flagged, and a
purpose-trained checkpoint can be supplied through `PAINTER_CUSTOM_MODEL`
(see [DATASET.md](DATASET.md)).

`postprocess_detections` then:

1. clips boxes, drops noise below an area threshold;
2. merges duplicates **of the same class** (IoU > 0.55 or containment > 0.85);
3. merges duplicates **across classes** when one region is claimed twice (a title
   reported as both `Title` and `Section-header`): the most confident wins and the
   alternative label is kept in `extra["also_detected_as"]`;
4. drops text boxes inside picture containers, keeps those inside tables (they are
   used to read cell text) and marks them `inside_table`;
5. collapses repeated header/footer regions into one.

---

## 6. OCR and Google Lens

`ocr.py` defines an `OcrProvider` interface with `recognize(image) →
OcrResult(text, lines, confidence, provider, uncertain)`. Providers:

* `TesseractProvider` — the local default. Page segmentation is chosen per crop
  size, small crops are upscaled before recognition, and the output is parsed
  into `TextLine` records with confidences.
* `LensTextProvider` — Google Lens text results through SerpApi, used only when
  the chain is configured for it, because it costs credits and returns
  search-snippet text rather than a calibrated OCR confidence.
* `OcrChain` — tries providers in order, records which one produced the text, and
  marks the result uncertain when confidence is low.

**Google Lens is not treated as a universal OCR engine.** Its primary role here
is *image and logo recovery*: crop → Lens → visual/exact matches → ranked
candidates → optional download → comparison against the crop. An image extracted
from the scan is always preferred over an online replacement, and a replacement
is only accepted above `PAINTER_ASSET_MATCH_THRESHOLD`.

`serpapi_client.py` implements the reliability requirements: key from the
environment only, connect/read timeouts, retries with exponential backoff and
jitter honouring `Retry-After`, quota and rate-limit error translation, response
shape validation that tolerates missing blocks, a persistent request/credit
ledger with per-document and global budgets, a SHA-256 response cache, and log
redaction so the key can never reach a log file. Downloads go through
`assets.py`, which rejects non-http(s) schemes, resolves DNS and refuses
private/loopback/link-local addresses, caps the size, validates the content type
and decodes the image before accepting it.

---

## 7. Typography

For every text element `analyse_ink` measures the ink profile, stroke width
(from a distance transform), slant (second moments) and ink colour. Font size
comes from the **dense ink core** of a text line: the row band where ink density
is at least 40 % of that line's peak. That core measures ≈0.55 em for mixed-case
text and ≈0.73 em for upper-case text — constants calibrated against rendered
DejaVu samples at several sizes (18/26/36/46 px), which reproduced the true size
within ~2 %.

The line pitch gives line height, and the pitch-to-size ratio gives line spacing.
Alignment is inferred from the line boxes, indentation from the left edges, and
family candidates are ranked by comparing the rendered width of the recognised
text in each installed family. Everything is `estimated=true` with a confidence
and a note explaining the measurement — exact font identification is never
claimed.

---

## 8. Tables, shapes and special elements

`analyse_table_geometry` derives the grid from morphological rule masks. A real
rule spans most of the region while text strokes span a few pixels, so a
boundary must cover **55 % of the perpendicular extent** to count; that is what
stops text columns being mistaken for table columns. If no plausible grid exists,
the module falls back to whitespace projection and lowers its confidence.

`reconstruct_table` then reads each cell with the OCR chain, detects merged cells
by testing whether the internal border actually exists along that span
(`_border_present` clamps the span to the correct axis), decides the header row
from text density, and records column widths, row heights and border thickness in
millimetres.

Shapes, lines, checkboxes, form fields, signatures and stamps come from
`heuristics.py`; signatures and stamps are extracted as image assets with their
position and size preserved and are never synthesised.

---

## 9. Renderers

| Renderer | Strategy | Fallbacks and honesty |
|---|---|---|
| `render/html_renderer.py` | one absolutely positioned `div` per element on a page-sized canvas, millimetre CSS units, explicit font stacks, no reliance on browser defaults for critical dimensions; `render_document_for_print` adds `@page` rules and removes viewer padding for pixel comparison | — |
| `render/pdf_renderer.py` | WeasyPrint from the print HTML (selectable text, embedded fonts) | Chromium headless → reportlab text-only PDF; the engine used is recorded in the artefact and job result |
| `render/docx_renderer.py` | native Word paragraphs, runs, tables and inline/anchored images; page setup from `PageFormat`; spacing and indentation computed from the IR | elements Word cannot express are drawn as images and flagged `shape_image_fallback` |

The DOCX never contains the scan as a page background: it always carries editable
text and native tables.

---

## 10. Verification

`Rasterizer` renders the print HTML at `PAINTER_RASTERIZE_DPI` (default 150) via
Chromium (`--window-size` in CSS pixels, `--force-device-scale-factor` for DPI)
and falls back to `pdftoppm` on the PDF. Both images are normalised (resized,
grayscale, contrast-normalised) before comparison so a photo, a scan and a
digital render can be compared fairly.

`compare_images` returns SSIM, pixel-difference ratio, edge IoU (dilated edges,
so slight antialiasing differences do not dominate) and projection correlation,
and combines them into a single similarity with fixed weights. `region_report`
localises the differences to a grid and names the elements that live there, and
`build_issue_list` turns them into human-readable statements ("region row 2,
column 3 differs most (mean diff 130); elements there: image:el_ab12").

`correction_loop` is deliberately bounded and safe: it may only apply **global
translation and a global font scale** (plus re-render), it stops at the target
similarity, when improvement falls below `PAINTER_VERIFY_MIN_IMPROVEMENT`, or at
`PAINTER_VERIFY_MAX_ITERATIONS`, and it can never alter recognised text to raise a
pixel score. Every iteration is saved with its metrics. Unresolved issues are
reported instead of being presented as success.

---

## 11. Storage

```
data/
├── painter.db                     SQLite (WAL): documents, pages, jobs, elements,
│                                  assets, typography, verifications, edits, usage
├── cache/serpapi/                 response cache keyed by crop hash
└── documents/<document_id>/
    ├── original/                  untouched upload
    ├── pages/                     processed pages + previews
    ├── stages/                    one JSON artefact per stage
    ├── rendered/                  document_ir.json, reconstruction.html/.pdf/.docx
    ├── verify/                    diff maps, iterations, verification.json
    └── exports/                   downloadable artefacts
```

Large objects live on the file system; the database stores metadata, IR versions
and pointers. `Repository` is the only module that touches SQL, so moving to
PostgreSQL means reimplementing that class, not the pipeline.

---

## 12. Security and reliability

* credentials only from the environment; `.env.example` contains placeholders
  only, and `logging_utils.redact` scrubs keys and tokens from every log record;
* upload validation by magic bytes (not extension), size and format allow-lists,
  page-count and pixel-count limits;
* path traversal protection in the API file route and in the demo's static route
  (`resolve()` must stay inside the workspace/demo directory);
* bearer-token authentication for the whole API when `PAINTER_API_TOKEN` is set;
* SSRF-hardened external downloads (see §6);
* typed errors mapped to HTTP status codes, with internal details logged but not
  returned to clients;
* corrupt or malformed input is caught per stage and reported, never fatal to the
  process.

---

## 13. Performance and cost

* models are loaded once per process and reused (`YoloDetector` caches the
  loaded model behind a lock);
* deterministic stage artefacts are cached, so re-running a document repeats
  nothing;
* SerpApi responses are cached by crop hash and de-duplicated, and the whole
  integration is off unless `PAINTER_ENABLE_SERPAPI=true`;
* budgets cap requests per document and globally, and `/api/usage` exposes the
  ledger;
* images are downscaled for model input (`PAINTER_YOLO_IMGSZ`, default 1024) and
  OCR upscales only small crops;
* long jobs run on a bounded worker pool (`PAINTER_JOB_WORKERS`), with states and
  progress visible through `/status`;
* `painter cleanup` removes workspaces older than a threshold (dry-run by
  default).

---

## 14. Environment used for the measurements

| Component | Version |
|---|---|
| OS / Python | Ubuntu 24.04, Python 3.12 |
| OpenCV / NumPy / Pillow | 4.x headless, 2.5.3, 12.3.0 |
| PyTorch / Ultralytics / doclayout-yolo | 2.14.1, 8.4.175, bundled runtime |
| Pydantic / FastAPI / Uvicorn | 2.13.5, 0.141.1, 0.54.0 |
| WeasyPrint / python-docx / scikit-image | 70.0, 1.2.0, 0.26.0 |
| Tesseract / poppler / Chromium | apt packages, English model |

Layout inference on an A4 page at `imgsz=1024` on this CPU-only machine takes
about 3 seconds; the rest of the pipeline (preprocess, OCR, reconstruct, render,
verify) is dominated by Tesseract and the browser rasteriser.

---

## 15. Extension points

| Goal | Where to change |
|---|---|
| Add an OCR engine | implement `OcrProvider` in `ocr.py` and register it in `build_ocr_chain` |
| Add a detector class set | supply `PAINTER_CUSTOM_MODEL`; extend `CLASS_TO_ELEMENT_TYPE` |
| Support another page format | add it to `geometry.PAGE_FORMATS` and to the family lists |
| Add an export format | add a renderer in `render/` and register it in `export_artifacts` |
| Add a verification metric | extend `verify.compare_images` and its weights |
| Move to PostgreSQL/S3 | reimplement `storage.Repository` / `DocumentWorkspace` |
| Add authentication/ownership | `api.py` dependency + a `users`/`ownership` table |
