# Painter — AI document reconstruction and replication engine

Painter takes a scan, photo or PDF of a document and rebuilds it as an **editable
document**: a Document IR that can be inspected and corrected, rendered to HTML,
PDF and DOCX, and compared against the original to prove how faithful the
reconstruction is.

This repository previously contained a single-file YOLO video demo
(`yolo_demo/live_feed_server.py`) that could not even be imported. The demo is
kept (repaired) as a visual playground; the application itself is the `painter`
package. The full audit of what was there, what was broken and what was reused is
in [docs/AUDIT.md](docs/AUDIT.md).

---

## 1. What the system does

```
upload ─► validate ─► preprocess ─► detect ─► recognize ─► reconstruct
                                                              │
        export ◄─ verify ◄─ render ◄──────────────────────────┘
```

| Stage | Output |
|---|---|
| `validate` | format sniffing, size/type limits, SHA-256, stored original |
| `preprocess` | orientation, boundary/perspective, deskew, shadow/noise/contrast, resolution normalisation, **every transform recorded** |
| `detect` | document regions from DocLayout-YOLO + COCO + deterministic OpenCV heuristics |
| `recognize` | OCR per region (Tesseract locally, Google Lens via SerpApi as an opt-in fallback) |
| `reconstruct` | Document IR: elements, reading order, tables, typography, geometry, image assets |
| `render` | HTML (positioned, page-exact), PDF (WeasyPrint or Chromium), DOCX (editable) |
| `verify` | rasterise the reconstruction, compare with the scan, bounded geometry-only correction loop |
| `export` | final files plus the IR JSON |

Every stage writes a JSON artefact and a database row, so a failed run resumes
from the failing stage instead of repeating OCR and paid API calls.

### Supported element classes

`heading`, `subheading`, `paragraph`, `text_block`, `list`, `image`, `logo`,
`table`, `table_caption`, `figure_caption`, `shape`, `line`, `rectangle`,
`circle`, `checkbox`, `form_field`, `signature`, `stamp`, `header`, `footer`,
`page_number`, `formula`.

The bundled `doclayout.pt` can only ever produce 11 of these (its real classes
are `Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture,
Section-header, Table, Text, Title`). Signatures, stamps, logos, checkboxes,
form fields and vector shapes are therefore recovered by deterministic OpenCV
heuristics, and a purpose-trained model can be dropped in through
`PAINTER_CUSTOM_MODEL` (see [docs/DATASET.md](docs/DATASET.md)).

---

## 2. Quick start

```bash
# 1. system packages
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils \
                        chromium fonts-dejavu fonts-liberation

# 2. python packages
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 3. configuration (all settings have safe defaults)
cp .env.example .env        # add SERPAPI_KEY only if you want Google Lens

# 4. check what the environment can actually do
python -m painter.cli capabilities

# 5. run the web application
python -m painter.cli serve --port 8000
#    → open http://localhost:8000
```

### CLI

```bash
python -m painter.cli process scan.png --out ./rebuilt      # full pipeline
python -m painter.cli elements doc_ab12cd34ef56 --review    # what needs a human
python -m painter.cli verify doc_ab12cd34ef56               # quality metrics
python -m painter.cli export doc_ab12cd34ef56 --formats pdf,docx,html
python -m painter.cli selftest                              # generated page, end to end
python -m painter.cli benchmark --cases 6                   # measured quality report
python -m painter.cli cleanup --keep-days 30                # dry run by default
```

### Web UI

The UI at `/` is a three-column workspace: upload and processing controls on the
left, a viewer in the middle (original / detection overlay / reconstruction /
side-by-side comparison, with page navigation and zoom), and on the right the
reconstructed elements, an element editor and the export buttons. The editor
patches the Document IR and re-renders — it never re-runs detection or OCR.

---

## 3. HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/documents/upload` | multipart upload, returns `document_id` |
| `GET` | `/api/documents` | list documents |
| `GET` | `/api/documents/{id}` | document, pages, jobs, stats, quality |
| `DELETE` | `/api/documents/{id}` | delete document and workspace |
| `POST` | `/api/documents/{id}/process` | start a pipeline job (returns `job_id`) |
| `GET` | `/api/documents/{id}/status?job_id=` | job status, stage list, result |
| `GET` | `/api/documents/{id}/ir` | the Document IR JSON |
| `GET` | `/api/documents/{id}/elements` | flattened elements with flags and provenance |
| `GET` | `/api/documents/{id}/elements/{element_id}` | one element |
| `PATCH` | `/api/documents/{id}/elements/{element_id}` | edit text, typography, geometry, image, table cells |
| `GET` | `/api/documents/{id}/preview` | rendered HTML (iframe-able) |
| `GET` | `/api/documents/{id}/preview.png?page=&dpi=` | rasterised reconstruction page |
| `POST` | `/api/documents/{id}/reconstruct` | re-render formats from the current IR |
| `POST` | `/api/documents/{id}/verify` | run verification / correction loop |
| `POST` | `/api/documents/{id}/export` | generate downloadable artefacts |
| `GET` | `/api/documents/{id}/exports/{kind}` | download `pdf`, `docx` or `html` |
| `GET` | `/api/documents/{id}/files/{path}` | workspace files (page images, overlays, diff maps) |
| `GET` | `/api/capabilities` | models, OCR, SerpApi and verification availability |
| `GET` | `/api/usage` | SerpApi request/credit ledger |
| `GET` | `/api/health` | liveness |

Set `PAINTER_API_TOKEN` to require `Authorization: Bearer <token>` (or
`X-API-Token`) on every `/api` route.

Example:

```bash
DOC=$(curl -s -F file=@scan.png localhost:8000/api/documents/upload | jq -r .document_id)
JOB=$(curl -s -X POST localhost:8000/api/documents/$DOC/process \
        -H 'content-type: application/json' -d '{}' | jq -r .job_id)
curl -s "localhost:8000/api/documents/$DOC/status?job_id=$JOB" | jq .job.status
curl -s -X POST localhost:8000/api/documents/$DOC/export \
        -H 'content-type: application/json' -d '{"formats":["pdf","docx","html"]}' | jq .exports
```

---

## 4. Configuration

Everything is environment driven (see [.env.example](.env.example) for the full
list). The most important switches:

| Variable | Default | Meaning |
|---|---|---|
| `PAINTER_DATA_DIR` | `./data` | workspaces, database, cache |
| `PAINTER_ENABLE_YOLO` | `true` | use the YOLO detectors (falls back to OpenCV when off) |
| `PAINTER_LAYOUT_MODEL` | `doclayout.pt` | layout checkpoint |
| `PAINTER_CUSTOM_MODEL` | `painter_doclayout.pt` | purpose-trained model, used when present |
| `PAINTER_OCR_PROVIDER` | `auto` | `auto`, `tesseract`, `lens`, `none` |
| `SERPAPI_KEY` | – | **required only for Google Lens features** |
| `PAINTER_ENABLE_SERPAPI` | `false` | master switch for every SerpApi call |
| `PAINTER_ENABLE_ASSET_DOWNLOAD` | `false` | allow downloading matched images |
| `PAINTER_VERIFY` | `true` | run the visual verification stage |
| `PAINTER_API_TOKEN` | – | require a bearer token on the API |

**No credential is ever hardcoded.** The legacy demo embedded a live SerpApi key
in source control; that key must be rotated (see [docs/AUDIT.md](docs/AUDIT.md) §7).

---

## 5. Reliability and cost control

* **Retries with exponential backoff and jitter**, plus `Retry-After` support, for
  every SerpApi call; timeouts on connect and read.
* **Request budgets**: `PAINTER_SERPAPI_MAX_REQUESTS_PER_DOC` (default 30) and
  `PAINTER_SERPAPI_MAX_REQUESTS_TOTAL` (default 400), tracked in a persistent
  ledger and reported at `/api/usage`.
* **Response caching** keyed by the SHA-256 of the uploaded crop, so re-running a
  document costs nothing.
* **SSRF protection** on asset downloads: http/https only, DNS resolved and
  private/loopback/link-local addresses refused, size cap, content-type check and
  image decoding validation.
* **Bounded correction loop**: at most `PAINTER_VERIFY_MAX_ITERATIONS` geometry
  attempts, stopping when the target similarity is reached or improvement falls
  below `PAINTER_VERIFY_MIN_IMPROVEMENT`. It can only move elements and scale
  fonts — **it cannot rewrite recognised text to improve a pixel score**.
* **Graceful degradation with disclosure**: a missing model, a disabled OCR
  provider or an unavailable rasteriser is recorded as a warning/assumption and
  surfaced in the job result, never hidden.

---

## 6. Tests and measured results

```bash
pytest                       # unit + integration suite
python -m painter.cli selftest    # generated A4 page, full pipeline
python -m painter.cli benchmark   # six generated cases with ground truth
```

Because no legally usable corpus could be downloaded, `painter/testsupport.py`
**generates** documents whose exact text is known, including photographed
variants (rotation, shading, noise, JPEG). That makes OCR accuracy measurable
rather than asserted.

Measured on this sandbox (CPU-only, `doclayout.pt` at `imgsz=1024`, on the
generated A4 `mixed` page whose exact text is known):

| Measurement | Value |
|---|---|
| Layout inference | ≈ 3 s per A4 page |
| OCR accuracy vs. ground-truth text | **0.846** |
| Table cell accuracy | **1.00** (12/12: header, labels, amounts, statuses) |
| Visual similarity to the original | **0.83** |
| Rendered-text similarity | **0.81** |
| Expected elements found (heading, image, table, footer) | **4/4** |
| Exports per document | HTML + PDF (text-extractable) + DOCX (native paragraphs/tables) |

These numbers are produced by the selftest itself and are reproducible with
`python -m painter.cli selftest`, because the synthetic page is generated from a
fixed seed. The residual error sits mostly in the estimated font family and in
the picture region, which is re-rendered from the extracted bitmap rather than
redrawn.

The benchmark report is written to `data/benchmark/benchmark_report.json` with
per-case metrics: OCR accuracy, table-cell accuracy, visual similarity (SSIM,
pixel difference, edge IoU, projection correlation) and export success.

---

## 7. Known limitations

* **Fidelity is not a pixel-perfect rebuild.** Fonts are estimated from measured
  ink metrics and picked from the families installed on the machine, so a
  reconstruction matches the original closely but not exactly.
* **A document title can be mislabelled by the detector.** On the generated page
  DocLayout-YOLO scores `Section-header` (0.79) above `Title` (0.66) for the same
  region, so the structure layer promotes the topmost heading-sized block to a
  heading and records the decision in the element metadata and provenance.
* **Page size is inferred.** A bitmap cannot prove paper size; the assumption
  (aspect ratio → A4/Letter, cross-checked against the measured text line pitch)
  is recorded in the IR rather than presented as fact.
* **DOCX is flow-based.** Text and tables are native and editable; exact
  positioning is preserved with anchored images, while paragraphs are placed by
  computed spacing and indentation. Shapes Word cannot express are drawn as
  images and flagged with `shape_image_fallback`.
* **Google Lens is not an OCR engine.** SerpApi is used for *image* recovery and
  as a last-resort text fallback; the module never pretends search snippets are
  OCR results, and every external match is marked low-confidence until verified.
* **Single-document scale.** Storage is SQLite plus the local file system with a
  bounded worker pool. The repository interface is the seam for moving to
  PostgreSQL and object storage, but that work has not been done.
* **Training data is not included.** A purpose-trained detector needs thousands
  of labelled pages; [docs/DATASET.md](docs/DATASET.md) specifies the dataset,
  classes and evaluation protocol.

---

## 8. Repository layout

```
painter/
├── config.py          environment-driven settings (no secrets in code)
├── errors.py          typed error taxonomy
├── logging_utils.py   structured logs with secret redaction
├── schemas.py         Document IR (elements, tables, typography, quality)
├── geometry.py        units, page-format inference, coordinate mapping, columns
├── preprocess.py      validation, load, deskew, illumination, resolution
├── detection.py       YOLO chain (DocLayout → generic → Ultralytics) + CV fallback
├── heuristics.py      signatures, stamps, logos, checkboxes, shapes, lines
├── ocr.py             Tesseract provider, Lens fallback, OCR chain
├── serpapi_client.py  Lens client: retries, caching, budgets, redaction
├── tables.py          table geometry, merged cells, cell OCR
├── structure.py       reading order, sections, Document IR assembly
├── typography.py      calibrated font size/weight/slant/colour estimation
├── assets.py          asset extraction, candidate ranking, safe download
├── render/            html_renderer, pdf_renderer, docx_renderer
├── verify.py          rasteriser, metrics, diff maps, correction loop
├── storage.py         workspaces + SQLite repository
├── pipeline.py        resumable stage orchestration
├── editing.py         IR edits and re-render without re-analysis
├── api.py             FastAPI application
├── cli.py             command line interface
├── testsupport.py     synthetic documents, selftest, benchmark
└── web/               single-page front end
tests/                 unit and integration tests
docs/                  audit, architecture, dataset specification
yolo_demo/             repaired video demo (uses the painter package)
```

Documentation: [docs/AUDIT.md](docs/AUDIT.md) ·
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/DATASET.md](docs/DATASET.md)

---

## 9. Licensing and ownership

This repository is private to its owner. Note that **external assets require
their own licensing**: the bundled checkpoints (`doclayout.pt`, `yolov8n.pt`,
`yolo12n.pt`, `yolo26n.pt`) carry their upstream licences, Tesseract and the
DejaVu/Liberation fonts have their own, and any image recovered through Google
Lens is third-party content whose licence must be checked before reuse. The
generated benchmark material in `painter/testsupport.py` is produced locally and
carries no third-party rights.
