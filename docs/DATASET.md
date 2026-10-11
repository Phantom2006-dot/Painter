# Dataset and model-training specification

The brief asks for a detector that recognises document elements. The checkpoint
that ships with this repository cannot do that on its own, so this document
specifies exactly what a purpose-trained model needs, how it is evaluated, and
how it plugs into the engine. **No training data is included** — no legally
usable corpus was available in this environment, and inventing one would violate
the "do not fabricate" rule. Everything else (the loader, the class mapping, the
registration point, the evaluation harness) is implemented and works today.

---

## 1. Why a purpose-trained model is needed

`doclayout.pt` is a DocLayout-YOLO checkpoint. Its class set was verified by
loading it and running inference (see `docs/AUDIT.md`):

```
Caption, Footnote, Formula, List-item, Page-footer, Page-header,
Picture, Section-header, Table, Text, Title
```

Measured behaviour on a generated A4 page: 11 classes, `Text` emitted at
line level (a paragraph arrives as several boxes that must be grouped), and
**no** class for signature, stamp, logo, checkbox, form field, line, rectangle,
circle or page number. Those are currently recovered by deterministic heuristics
in `painter/heuristics.py` and flagged with `DETECTION_FALLBACK` or
`SHAPE_IMAGE_FALLBACK`. A purpose-trained model replaces those heuristics with
learned detections — it does not change the rest of the pipeline.

---

## 2. Target class list

Train **semantic regions**, never individual words (the brief is explicit about
this; word-level boxes would make the model unusable for layout).

| id | class | notes |
|---|---|---|
| 0 | `heading` | top-level document title |
| 1 | `subheading` | section heading inside the body |
| 2 | `paragraph` | a full text block, not a single line |
| 3 | `text_block` | free text that is not a clean paragraph (notes, footnotes) |
| 4 | `list` | ordered or bulleted list block |
| 5 | `image` | photograph or illustration |
| 6 | `logo` | company mark, usually in the header |
| 7 | `table` | the whole table, including borders |
| 8 | `table_caption` | caption attached to a table |
| 9 | `figure_caption` | caption attached to a figure |
| 10 | `shape` | filled/unfilled shape that is not a rule |
| 11 | `line` | horizontal or vertical rule |
| 12 | `rectangle` | box, frame or border |
| 13 | `circle` | circle or ellipse |
| 14 | `checkbox` | empty or ticked checkbox |
| 15 | `form_field` | fill-in field with a label |
| 16 | `signature` | handwritten signature |
| 17 | `stamp` | rubber stamp or seal |
| 18 | `header` | running header region |
| 19 | `footer` | running footer region |
| 20 | `page_number` | page number |
| 21 | `formula` | mathematical formula |
| 22 | `unknown` | reserved; do not train, it is never a ground-truth label |

`painter/detection.py::CLASS_TO_ELEMENT_TYPE` already maps these names (and the
DocLayout names) onto `ElementType`, so a trained model needs **no code change**
to be used: name the classes exactly as above and point
`PAINTER_CUSTOM_MODEL` at the checkpoint.

---

## 3. Annotation workflow

1. **Collect pages** you have the right to use: your own scans, documents you
   own, or openly licensed corpora (e.g. academic/technical report collections
   with permissive licences, or public-domain scans). Record the licence and
   source of every page — the recovered-asset policy in `assets.py` assumes the
   same discipline applies to training data.
2. **Annotate** with any YOLO-compatible tool: LabelImg, CVAT, Label Studio,
   Roboflow, or `ultralytics`' own labelling flow.
   * one box per semantic region, tight to the ink (include borders in a table,
     include the caption in its caption class, not in the figure);
   * do **not** label individual lines inside a paragraph;
   * label partially visible elements at the page edge only if more than half the
     element is visible, otherwise skip it;
   * for signature/stamp/logo, include the whole visible mark;
   * keep one consistent convention for `line` (thin rules) versus `rectangle`
     (frames) — inconsistent labels here are the most common source of a model
     that "works" but confuses borders.
3. **Export** to YOLO format:

```
datasets/painter_doclayout/
├── images/train/page_0001.jpg
├── images/val/page_0100.jpg
├── labels/train/page_0001.txt
└── labels/val/page_0100.txt
```

Each label line is `class_id cx cy w h`, all four values normalised to `0..1`
relative to the image. Class ids must match the table in §2.

4. **Dataset YAML** (`datasets/painter_doclayout.yaml`):

```yaml
path: ./datasets/painter_doclayout
train: images/train
val: images/val
names:
  0: heading
  1: subheading
  2: paragraph
  3: text_block
  4: list
  5: image
  6: logo
  7: table
  8: table_caption
  9: figure_caption
  10: shape
  11: line
  12: rectangle
  13: circle
  14: checkbox
  15: form_field
  16: signature
  17: stamp
  18: header
  19: footer
  20: page_number
  21: formula
```

---

## 4. Volume, splits and augmentation

* **Minimum viable**: ~2 000 labelled pages for the 12 common classes; the rare
  classes (signature, stamp, checkbox, form field, circle) need at least 300
  instances each or the model will ignore them.
* **Splits**: 80 / 10 / 10 train/val/test, **split by source document**, not by
  page — pages from the same document must never be in both train and test, or
  the evaluation is meaningless.
* **Augmentation**: photographed pages are the hard case, so augment the way the
  preprocessor sees the world: small rotations (±5°), perspective jitter,
  brightness/contrast shifts, shadows, JPEG compression, and downscaling to
  simulate phone cameras. Do **not** flip text horizontally or vertically.
* **Balance**: keep tables and images well represented (they drive most of the
  reconstruction value) and do not let a single class exceed ~40 % of instances.

---

## 5. Training

The repository's loader accepts either runtime, so train with whichever is
installed. DocLayout-YOLO (matches the shipped architecture):

```bash
git clone https://github.com/opendatalab/DocLayout-YOLO
cd DocLayout-YOLO
python tools/train.py --config configs/doclayout_yolo.yaml \
  --data /path/to/painter_doclayout.yaml \
  --epochs 100 --imgsz 1024 --batch 8 --device 0
```

Plain Ultralytics YOLO (smaller and easier to serve):

```bash
yolo detect train model=yolo11n.pt data=datasets/painter_doclayout.yaml \
     imgsz=1024 epochs=150 batch=8 patience=25 project=runs name=painter
```

Then register the result:

```bash
cp runs/detect/painter/weights/best.pt yolo_demo/painter_doclayout.pt
# .env
PAINTER_CUSTOM_MODEL=painter_doclayout.pt
python -m painter.cli capabilities      # confirm the loader and class list
```

`painter.cli capabilities` prints the loader that succeeded and the exact class
names the checkpoint reports, which is the fastest way to catch a
class-name/class-id mismatch.

### Versioning

Keep the checkpoint under version control or object storage, never overwriting a
previous one, and record alongside it: dataset hash, YAML, training command,
`mAP50`/`mAP50-95` per class, and the date. A model is only promoted when it beats
the previous one on the held-out **test** split *and* improves at least one
pipeline-level metric in §6.

---

## 6. Evaluation protocol

Two levels, both required.

### 6.1 Detection level (per class)

Report `mAP50`, `mAP50-95`, precision and recall per class on the test split,
plus the confusion matrix between `line`/`rectangle`, `header`/`heading`,
`figure_caption`/`table_caption` and `text_block`/`paragraph` — those are the
pairs that actually matter downstream.

```bash
yolo detect val model=yolo_demo/painter_doclayout.pt \
     data=datasets/painter_doclayout.yaml split=test imgsz=1024
```

### 6.2 Pipeline level (what the product actually promises)

Detection quality alone does not prove reconstruction quality. The engine ships a
harness that measures the product metrics on documents whose content is known:

```bash
python -m painter.cli benchmark --cases 6
# → data/benchmark/benchmark_report.json
```

Metrics produced per case:

| Metric | Meaning | Where it comes from |
|---|---|---|
| `ocr_accuracy` | normalised similarity between recognised text and ground truth | `testsupport` knows the exact text it drew |
| `table_cell_accuracy` | cells whose text matches ground truth | same |
| `visual_similarity` | combined SSIM / pixel-diff / edge-IoU / projection score | `verify.compare_images` |
| `rendered_text_similarity` | OCR of the rasterised reconstruction vs the IR text | `verify` |
| `elements_by_type` | what the detector found, per class | pipeline result |
| `expectations_met` | heading / image / table / footer present | ground-truth expectations |

**Do not quote a model as "better" without a measured improvement on an
identified test set.** The numbers in the README were produced by these commands
on generated pages, and are labelled as such.

---

## 7. Active learning from user corrections

Every manual edit is stored in `Element.edits` with the field, old value, new
value and timestamp. That table is the cheapest source of new training signal:

1. export edited elements as crops (`GET /api/documents/{id}/elements` plus the
   page image) — the corrected text is a free label for the element class;
2. pages where users changed the *type* of an element are the most valuable:
   they mark exactly where the model is wrong;
3. pages where OCR confidence was low and the user rewrote the text identify the
   regions worth re-annotating for a text-specific model.

A scheduled job can collect these into a review queue; nothing in the current
implementation does that automatically, and it is listed as future work rather
than claimed as a feature.

---

## 8. Licensing of training data

Training data carries the same constraints as the recovered assets: use only
pages you own or that are explicitly licensed for machine-learning use, record
the licence per source, and keep the provenance with the dataset. Do not train on
scans of copyrighted documents whose licence you have not verified, even if the
resulting model is only used internally.
