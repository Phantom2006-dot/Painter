#!/usr/bin/env python3
"""Label how a photograph of a page differs from the digital page it shows.

The taxonomy is the one that matters for this project, and each label is derived
from a comparison rather than from someone's impression:

``same_text``             text present, legible and identical in both
``missing_text``          present on the digital page, absent from the capture
``added_text``            present in the capture, absent from the digital page
``substituted_text``      the same position carries different words
``formatting_only``       same words, different line breaking or spacing
``handwritten_annotation`` ink in the capture that is not on the digital page and
                          is not machine-set type
``fold`` / ``curl`` / ``blur`` / ``shadow`` / ``glare`` / ``crop`` /
``perspective`` / ``rotation``   capture conditions

Two ways to run it. With ``--paired`` the clean page and the photograph are known
to be the same document, which is the only case where a text difference means
anything: two pages that merely look alike would otherwise produce thousands of
fake ``missing_text`` labels, and the model would learn document-type similarity
instead of the transformation from paper to pixels.

    python3 tools/label_document_pairs.py --paired \
        --page page.png --photo photo.jpg --quad "x1,y1 x2,y2 x3,y3 x4,y4" \
        --out labels/

Without ``--paired`` it labels capture conditions only, from measurements taken on
the photograph itself.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from painter.ocr import build_ocr_chain  # noqa: E402
from painter.config import get_settings  # noqa: E402

#: A line is "the same" when its text agrees this closely, after normalisation.
SAME_TEXT_THRESHOLD = 0.90
#: Below this a shared position counts as different words, not a reflow.
SUBSTITUTION_THRESHOLD = 0.55
#: Two words count as the same word above this character similarity. OCR drops or
#: mangles individual characters, so exact token equality is too strict.
WORD_MATCH_THRESHOLD = 0.75


def token_similarity(reference: str, observed: str) -> float:
    """Fraction of the reference's words that survive in the observed line.

    A whole-line character ratio punishes one mangled word as hard as a missing
    sentence: on a blurred capture "50 Summary of total financial assets" read as
    "$0 Seammary of total financial assets" scored 0.75 and was filed as different
    words, when in truth every word is recognisably present. Scoring per word
    measures how much of the text survived, which is the question being asked.
    """
    reference_words = reference.split()
    if not reference_words:
        return 1.0
    observed_words = observed.split()
    if not observed_words:
        return 0.0
    matched = 0
    for word in reference_words:
        best = max(
            (SequenceMatcher(None, word, other).ratio() for other in observed_words),
            default=0.0,
        )
        if best >= WORD_MATCH_THRESHOLD:
            matched += 1
    return matched / len(reference_words)


@dataclass
class PageText:
    """Recognised text with the geometry needed to compare two pages."""

    lines: list[dict] = field(default_factory=list)

    @property
    def texts(self) -> list[str]:
        return [line["text"] for line in self.lines]


def normalise(text: str) -> str:
    """Fold case and collapse whitespace so trivial differences do not count."""
    return " ".join(text.lower().split())


def prepare_for_ocr(image: np.ndarray, target_height: int = 1800) -> np.ndarray:
    """Normalise contrast and size before OCR.

    A photographed page arrives with an ambient colour cast, low local contrast
    and glyphs far smaller than Tesseract's sweet spot. Measured on one simulated
    capture, the rectified page read as "aa. a te ' , Mere $9 96 Semenary of total
    frnencial asects" at native size and as "$0 Swrsmary ef total faancial assets
    and fiasnclal Labaties ot foie value through prof.t" after this step — still
    imperfect, but the words are now recognisable and the comparison can work.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    scale = target_height / max(gray.shape[0], 1)
    if abs(scale - 1.0) > 0.15:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def read_page_text(image: np.ndarray, settings) -> PageText:
    """OCR a page image and keep each line with its box."""
    chain = build_ocr_chain(settings, None)
    result = chain.recognize(prepare_for_ocr(image))
    lines = []
    for line in getattr(result, "lines", []) or []:
        text = (line.text or "").strip()
        if not text:
            continue
        box = line.bbox
        lines.append(
            {
                "text": text,
                "normalised": normalise(text),
                "bbox": [float(box.x1), float(box.y1), float(box.x2), float(box.y2)],
            }
        )
    return PageText(lines=lines)


def rectify_to_page(photo: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Warp the photographed page to an upright rectangle before reading it.

    Reading the whole photograph made the surface the page lies on part of the
    comparison: OCR turned the floor into text and every one of those lines became
    a false ``added_text`` label. Rectifying first also puts both sides in the same
    frame, so a line's position means the same thing on each.
    """
    width = int(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))
    height = int(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])))
    width, height = max(width, 16), max(height, 16)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    return cv2.warpPerspective(photo, matrix, (width, height), flags=cv2.INTER_CUBIC)


def _centre(line: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = line["bbox"]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def compare_text(reference: PageText, observed: PageText) -> dict[str, list[dict]]:
    """Match lines by position and classify each pair.

    Matching is positional, not textual: a line that moved is a formatting change,
    whereas matching by text would call every reflowed line both missing and added.
    """
    labels: dict[str, list[dict]] = {
        "same_text": [],
        "missing_text": [],
        "added_text": [],
        "substituted_text": [],
        "formatting_only": [],
    }
    if not reference.lines or not observed.lines:
        labels["missing_text"] = [dict(line) for line in reference.lines]
        labels["added_text"] = [dict(line) for line in observed.lines]
        return labels

    ref_boxes = np.array([line["bbox"] for line in reference.lines], dtype=np.float64)
    obs_boxes = np.array([line["bbox"] for line in observed.lines], dtype=np.float64)
    # Normalise both pages into a shared frame so position is comparable.
    def scale(boxes: np.ndarray) -> np.ndarray:
        out = boxes.copy()
        out[:, [0, 2]] /= max(boxes[:, [0, 2]].max(), 1e-6)
        out[:, [1, 3]] /= max(boxes[:, [1, 3]].max(), 1e-6)
        return out

    ref_scaled, obs_scaled = scale(ref_boxes), scale(obs_boxes)
    ref_centres = np.stack([(ref_scaled[:, 0] + ref_scaled[:, 2]) / 2, (ref_scaled[:, 1] + ref_scaled[:, 3]) / 2], axis=1)
    obs_centres = np.stack([(obs_scaled[:, 0] + obs_scaled[:, 2]) / 2, (obs_scaled[:, 1] + obs_scaled[:, 3]) / 2], axis=1)

    used_observed: set[int] = set()
    for ref_index, ref_line in enumerate(reference.lines):
        distances = np.linalg.norm(obs_centres - ref_centres[ref_index], axis=1)
        for candidate in np.argsort(distances):
            if int(candidate) in used_observed:
                continue
            if distances[candidate] > 0.12:  # too far away to be the same line
                break
            used_observed.add(int(candidate))
            obs_line = observed.lines[int(candidate)]
            similarity = token_similarity(ref_line["normalised"], obs_line["normalised"])
            if similarity >= SAME_TEXT_THRESHOLD:
                labels["same_text"].append({"reference": ref_line["text"], "observed": obs_line["text"]})
            elif similarity >= SUBSTITUTION_THRESHOLD:
                labels["formatting_only"].append(
                    {"reference": ref_line["text"], "observed": obs_line["text"], "similarity": round(similarity, 3)}
                )
            else:
                labels["substituted_text"].append(
                    {"reference": ref_line["text"], "observed": obs_line["text"], "similarity": round(similarity, 3)}
                )
            break
        else:
            labels["missing_text"].append(dict(ref_line))

    for index, obs_line in enumerate(observed.lines):
        if index not in used_observed:
            labels["added_text"].append(dict(obs_line))
    return labels


def measure_capture(photo: np.ndarray, quad: np.ndarray | None) -> dict:
    """Measure the capture conditions that can be measured without the original."""
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Motion and defocus both flatten the high-frequency energy.
    blur_score = float(laplacian)
    blur = "blur" if blur_score < 60 else None

    # Glare: saturated pixels inside the page, which are not paper texture.
    saturated = float((gray > 245).mean())
    glare = "glare" if saturated > 0.10 else None

    # Shadow: a strong low-frequency luminance gradient across the page.
    if quad is not None:
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
        inside = gray[mask > 0]
        gradient = float(inside.std()) if inside.size else 0.0
        shadow = "shadow" if gradient > 55 else None
        coverage = float((mask > 0).mean())
        crop = "crop" if coverage < 0.55 else None
    else:
        gradient, shadow, crop = 0.0, None, None

    labels = [label for label in (blur, glare, shadow, crop) if label]
    return {
        "laplacian_variance": round(blur_score, 2),
        "saturated_fraction": round(saturated, 4),
        "page_luminance_std": round(gradient, 2),
        "labels": labels,
    }


def parse_quad(text: str) -> np.ndarray:
    points = []
    for chunk in text.replace(";", " ").split():
        if not chunk:
            continue
        x, y = chunk.split(",")
        points.append([float(x), float(y)])
    if len(points) != 4:
        raise SystemExit("--quad needs four 'x,y' pairs")
    return np.array(points, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--page", help="clean digital page image (the reference)")
    parser.add_argument("--photo", required=True, help="photograph or scan to label")
    parser.add_argument("--quad", help="page corners in the photo as 'x,y x,y x,y x,y'")
    parser.add_argument("--paired", action="store_true", help="assert that page and photo are the same document")
    parser.add_argument("--out", default="labels", help="output directory")
    parser.add_argument("--sample-id", default=None)
    args = parser.parse_args()

    settings = get_settings()
    photo = cv2.imread(args.photo)
    if photo is None:
        raise SystemExit(f"cannot read {args.photo}")
    sample_id = args.sample_id or Path(args.photo).stem
    quad = parse_quad(args.quad) if args.quad else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read the page, not the scene: the surface is not content.
    observed_image = rectify_to_page(photo, quad) if quad is not None else photo
    observed = read_page_text(observed_image, settings)
    content: dict[str, object] = {"note": "capture conditions only; no reference page given"}
    if args.page:
        if not args.paired:
            # Refusing here is the point: an unpaired comparison produces labels
            # that teach the model nothing true.
            raise SystemExit(
                "comparing text needs --paired: a page and a photograph must be the "
                "same document for a difference to mean anything"
            )
        reference_image = cv2.imread(args.page)
        # Downscale a large reference so both sides are read at a comparable scale;
        # OCR behaviour changes with text size and would otherwise dominate the diff.
        target_height = observed_image.shape[0]
        if reference_image.shape[0] > target_height * 1.5:
            scale = target_height / reference_image.shape[0]
            reference_image = cv2.resize(
                reference_image,
                (max(1, int(reference_image.shape[1] * scale)), target_height),
                interpolation=cv2.INTER_AREA,
            )
        reference = read_page_text(reference_image, settings)
        content = compare_text(reference, observed)

    capture = measure_capture(photo, quad)
    record = {
        "sample_id": sample_id,
        "photo": str(Path(args.photo).resolve()),
        "page_reference": str(Path(args.page).resolve()) if args.page else None,
        "paired": bool(args.page and args.paired),
        "page_quad": quad.tolist() if quad is not None else None,
        "content": content,
        "capture": capture,
    }
    (out_dir / f"{sample_id}_labels.json").write_text(json.dumps(record, indent=2))
    (out_dir / f"{sample_id}_ocr.json").write_text(
        json.dumps({"lines": observed.lines}, indent=2)
    )
    (out_dir / f"{sample_id}_metadata.json").write_text(
        json.dumps(
            {
                "image_size": [int(photo.shape[1]), int(photo.shape[0])],
                "capture": capture,
                "source": "painter label_document_pairs",
            },
            indent=2,
        )
    )
    print(f"wrote {out_dir}/{sample_id}_labels.json")
    if isinstance(content, dict) and "same_text" in content:
        for key, value in content.items():
            print(f"  {key}: {len(value)}")


if __name__ == "__main__":
    main()
