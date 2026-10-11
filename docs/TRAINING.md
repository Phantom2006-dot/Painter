# Training the layout detector

This document records what was trained, on what data, and — most importantly —
what the measurements actually showed.

## Summary

| Model | Training data | mAP50 | mAP50-95 | CPU inference |
|---|---|---:|---:|---:|
| Shipped `yolo_demo/doclayout.pt` | full DocLayNet (≈80k pages) | **0.889** | **0.707** | 765 ms/page |
| Fine-tuned `yolov8n` (this work) | DocLayNet-small, 691 pages | 0.533 | 0.331 | 24 ms/page |

Evaluated on the same 64-page DocLayNet-small validation split (1,034 boxes, 11 classes).

**Conclusion: the fine-tune is worse than the model we already ship, so it is not
wired in as the default.** A 3.1M-parameter nano model trained for 15 epochs on
691 pages cannot beat a model trained on the full dataset. The measurements say
so, and pretending otherwise would have shipped a regression.

That is not the same as saying training is useless. It says the *detector was
never the problem*: see below.

## Where the bad results actually came from

The reconstruction looked wrong, and the instinct was that the model needed
training. Measuring the model first showed the opposite: on a held-out layout
benchmark the shipped detector scores **mAP50 0.889**. A detector at that level
does not produce text that loses its opening words, a table rendered at 750mm on
a 141mm page, or a sentence that disappears.

Those were pipeline defects, each reproduced on a real document and each now
fixed with a regression test:

| Symptom on the real document | Cause | Fix |
|---|---|---|
| Body text ~4.8pt, page proportions wrong | A 2.19 aspect ratio was forced onto A6 | Non-paper documents get a custom page scaled so body text stays ≈11pt (`geometry.infer_page_format`) |
| Table overflowed, cells looked enormous | Column widths measured in pixels, stored as millimetres | Page scale passed into `tables.reconstruct_table`; renderer rescales legacy pixel values |
| Table covered the paragraph below it | Table cells had no typography and fell back to a fixed 11pt | Cells inherit the page body size and are capped to their row height |
| "Having attained the University-approved cut-off m" | The detector returned a *fragment* of the line, so OCR read a fragment | `detection.complete_text_regions` widens a region to the ink extent of its own line |
| Two sentences missing entirely | The detector never fired on those lines | `detection.recover_missing_text_lines` finds ink rows no detection covers |
| Lines drifted out of position | A multi-line paragraph was one block with a single line-height | Every OCR line is placed at its measured position |

Measured on the same real document, after the fixes: 14 truncated lines recovered,
2 missed lines recovered, 0 elements flagged for review, body text at 9.6pt on a
282×135mm page.

## The dataset

[DocLayNet-small](https://huggingface.co/datasets/pierreguillou/DocLayNet-small)
(CDLA-Permissive-1.0), a 380MB subset of DocLayNet:

| Split | Pages | Boxes |
|---|---:|---:|
| train | 691 | 9,017 |
| val | 64 | 1,034 |
| test | 49 | 544 |

Its 11 classes are exactly the classes the shipped model uses — `Caption`,
`Footnote`, `Formula`, `List-item`, `Page-footer`, `Page-header`, `Picture`,
`Section-header`, `Table`, `Text`, `Title` — so the two models are directly
comparable and a fine-tune needs no class remapping.

DocLayNet-small repeats the same box once per text line inside it; the converter
drops those duplicates (66,734 raw boxes → 9,017 unique).

## How to reproduce

```bash
# 1. data
mkdir -p /home/ubuntu/datasets && cd /home/ubuntu/datasets
curl -L -o doclaynet_small.zip \
  'https://huggingface.co/datasets/pierreguillou/DocLayNet-small/resolve/main/data/dataset_small.zip'
unzip -q doclaynet_small.zip

# 2. convert to YOLO format (writes data/layout_dataset/data.yaml)
cd /home/ubuntu/Painter
python3 tools/prepare_layout_dataset.py \
    --source /home/ubuntu/datasets/small_dataset \
    --output data/layout_dataset

# 3. train, evaluate, and compare against the shipped model
python3 tools/train_layout.py \
    --data data/layout_dataset/data.yaml \
    --epochs 15 --imgsz 512 --batch 8 \
    --baseline yolo_demo/doclayout.pt \
    --report data/training/report.json
```

The run above took 1,684 seconds (28 minutes) for 15 epochs on CPU. The machine
report is written next to the metrics in `data/training/report.json`, because a
CPU number is not a GPU number and the report should say so.

### Per-class result of the fine-tune (val)

| Class | mAP50 | | Class | mAP50 |
|---|---:|---|---|---:|
| Table | 0.848 | | Page-footer | 0.614 |
| Page-header | 0.749 | | List-item | 0.525 |
| Text | 0.726 | | Picture | 0.500 |
| Section-header | 0.648 | | Formula | 0.452 |
| Caption | 0.639 | | Footnote | 0.115 |
| | | | Title | 0.048 |

The rare classes (`Title`, `Footnote`, 4 and 9 validation instances) are
essentially unlearned — with 193 and 156 training boxes there is not enough
signal. The bulk classes (`Text`, `Table`, `Page-header`) are healthy, which is
consistent with "the model can see the structure; the rare classes need data".

## Closing the loop on our own documents

The training data that matters most is the documents this tool gets wrong. Every
correction made in the web UI is recorded on the element it changed, and

```bash
python3 painter.cli training-data --out data/training_export
python3 painter.cli training-data --out data/training_export --only-corrected
```

exports every reconstructed document as YOLO labels, with a `manifest.csv`
marking which documents a human touched. Those labels can be mixed into the
dataset above, so the next fine-tune learns from real corrections rather than
only from public corpora.

## When training is worth it

Not yet, on this evidence. The honest conditions:

1. **Data**: DocLayNet-small is 691 pages; the full DocLayNet is ≈80,000. A
   fine-tune on the full set is the first thing that could plausibly beat 0.889.
2. **Hardware**: 28 minutes for 15 epochs at 512px on CPU. Full-scale training
   wants a GPU.
3. **Purpose**: the detector already handles ordinary documents well. Training
   pays off on the *specific* documents it fails on — which is why the
   correction export above exists, and why it should be populated before the
   next training run rather than after.