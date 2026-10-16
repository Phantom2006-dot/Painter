# Paired document datasets: what was used and where it came from

This records the sources behind `tools/build_page_dataset.py`, including the ones
that were evaluated and rejected, so the training data can be rebuilt or audited
without repeating the search.

## Used

### SmartDoc 2015 — Challenge 1 (real captures with corner ground truth)

- **Source:** https://github.com/jchazalon/smartdoc15-ch1-dataset
- **Download:** release `v2.0.0`, assets `frames.tar.gz` (1.02 GB) and
  `models.tar.gz` (0.41 GB) from
  `https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/download/v2.0.0/`
- **Checksums (verified after download):**
  - `models.tar.gz` `6f9068624073f76b20f88352b2bac60b9e5de5a59819fc9db37fba1ee07cce8a`
  - `frames.tar.gz` `3acb8be143fc86c507d90d298097cba762e91a3abf7e2d35ccd5303e13a79eae`
- **Licence:** Creative Commons Attribution 4.0 International
- **Contents:** 24,889 frames at 1920×1080 from 30 printed documents across 5
  backgrounds, each frame with annotated page corners, plus 30 clean model images
  in five variants.
- **Metadata columns:** `bg_name, bg_id, model_name, model_id, modeltype_name,
  modeltype_id, model_subid, image_path, frame_index, model_width, model_height,
  tl_x, tl_y, bl_x, bl_y, br_x, br_y, tr_x, tr_y`.
- **Corner order in the file is** `tl, bl, br, tr`; this project stores
  `tl, tr, br, bl`, so the loader reorders. Getting this wrong mirrors the labels.
- **Citation:** Burie, Chazalon, Coustaty, Eskenazi, Luqman, Mehri, Nayef, Ogier,
  Prum, Rusinol. "ICDAR2015 Competition on Smartphone Document Capture and OCR
  (SmartDoc)", ICDAR 2015.

### DocUNet benchmark (real photographs paired with flatbed scans)

- **Source:** https://www3.cs.stonybrook.edu/~cvl/docunet.html
- **Download:** `original.zip` (328 MB) and `scan.zip` (416 MB) from
  `http://vision.cs.stonybrook.edu/~kema/docwarp/`
- **Contents:** 130 photographs and 65 flatbed scans. Pairing is by the leading
  identifier: `10_1.jpg` and `10_2.jpg` are two captures of the document scanned
  as `10.png`. **Scans are PNG, photographs are JPEG** — a `*.jpg` glob finds no
  scans and the pairing silently comes back empty.
- **Note:** this release does not annotate page corners, so these pairs supervise
  the reconstruction side, not the corner detector.
- **Citation:** Ma, Shu, Bai, Wang, Samaras. "DocUNet: Document Image Unwarping
  via A Stacked U-Net", CVPR 2018.

### DocLayNet (clean digital pages, used as the synthetic reference side)

- **Source:** https://github.com/ibm-aur-nlp/PubLayNet (DocLayNet distribution)
- **Local copy:** `/home/ubuntu/datasets/small_dataset` (419 MB, `train/val/test`)
- **Role:** provides clean pages to composite onto surfaces. Its layout
  annotations are also the source of the layout-model labels.

### The operator's own photograph

- `/tmp/qa4/original_small.png` — the certificate photographed on a concrete
  floor that the boundary heuristic failed on. The floor texture is extracted from
  it (with the page inpainted away) and used as a synthetic background, so the
  failure domain is represented in training.

## Evaluated, not used

| Dataset | Why not |
| --- | --- |
| **SmartDoc-QA** (https://zenodo.org/records/5293201) | ~13.7 GB; its ground truth is OCR quality assessment rather than page geometry, so it duplicates what SmartDoc already gives for this task. |
| **Brno Mobile OCR** (https://pero.fit.vut.cz/brno_mobile_ocr_dataset) | Non-commercial research licence and registration-gated; 19,725 photographs with line transcriptions would suit a text-line model, which is not the model being trained here. |
| **Doc3D** (https://github.com/cvlab-stonybrook/doc3D-dataset) | Tens of GB of synthetic warps with depth and UV maps. The capture simulator already produces the distortion this task needs, with labels that match the rest of the pipeline. |
| **PubLayNet / DocBank** | Clean digital pages only, no photograph of the same printed page. DocLayNet already covers the digital side. |

## Rules this dataset follows

1. **A pair means the same document instance.** A digital page and a photograph
   are only paired when one is a capture of the other. Similar-looking pages are
   not a pair; treating them as one teaches document-type similarity instead of
   the paper-to-pixels transformation.
2. **Split by document, not by page or frame.** Frames of one document are nearly
   identical, so a frame-level split reports accuracy the model does not have.
   SmartDoc's `patent001`–`patent005` are held out entirely, and synthetic
   validation holds out whole source pages.
3. **Labels come from a transformation, not from an impression.** Synthetic
   corners, boxes and capture conditions are recorded as they are applied. Real
   frames use SmartDoc's annotations.
## Known limitation: text comparison needs better OCR than Tesseract

`tools/label_document_pairs.py` was validated on a synthetic pair where the page
and the photograph are the same document by construction, so every line should
come back as `same_text`. It does not, and the reason is the OCR rather than the
comparison:

| stage | what Tesseract returned |
| --- | --- |
| clean page, native size | `270 Notes to the consolidated financaal statements of Aegon NV. Note 50` |
| rectified photo, native size | `aa. a te ' , Mere $9 96 Semenary of total frnencial asects` |
| rectified photo, contrast-normalised and upscaled | `$0 Swrsmary ef total faancial assets and fiasnclal Labaties` |

The line is recognisably the right line, but `follows` reads as `folaws` and
`summarizes` as `aanenar`, which is past the point where word matching can confirm
identity. Rectifying to the page and normalising contrast both help and are kept,
but they do not close the gap on a blurred capture.

Two consequences:

1. **Content-difference labels need the Google Lens OCR path**, which is exactly
   what the SerpApi key enables. Tesseract remains the right engine for clean
   scans and for the reconstruction pipeline's own text, not for deciding whether
   a photographed page says the same thing as its digital original.
2. **Capture-condition labels are unaffected** — blur, glare, shadow and crop are
   measured from pixels, not from recognised text, and those are correct on the
   same sample (`blur` detected).

Until the Lens path is available, treat the `content` block as diagnostic output
rather than as training supervision.
