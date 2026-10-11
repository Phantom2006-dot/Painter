"""Typography estimation.

For every text element the module estimates size, weight, slant, colour,
alignment, line height, spacing and indentation, and picks the closest available
font family from a candidate list. Everything it produces is marked
``estimated=True`` and carries a confidence: the brief is explicit that exact
font identification must never be claimed when the evidence only supports an
approximation.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import Settings
from .geometry import infer_alignment, mm_to_pt, pt_to_mm
from .logging_utils import get_logger
from .schemas import Alignment, BBox, Element, ElementType, PageIR, Typography

logger = get_logger("typography")

#: Plausible range for body text. Estimates outside this range are measurement
#: artefacts rather than real typography.
MIN_FONT_PT = 4.0
MAX_FONT_PT = 40.0

#: Candidate families, in rough order of likelihood for office documents.
CANDIDATE_FAMILIES = [
    "DejaVu Sans",
    "DejaVu Serif",
    "Liberation Sans",
    "Liberation Serif",
    "Liberation Mono",
    "Noto Sans",
    "Noto Serif",
    "FreeSans",
    "FreeSerif",
]

#: Approximate x-height / em ratio per family (used to convert measured glyph
#: heights into point sizes).
X_HEIGHT_RATIO = {
    "DejaVu Sans": 0.545,
    "DejaVu Serif": 0.520,
    "Liberation Sans": 0.530,
    "Liberation Serif": 0.510,
    "Liberation Mono": 0.530,
    "Noto Sans": 0.536,
    "Noto Serif": 0.520,
    "FreeSans": 0.520,
    "FreeSerif": 0.500,
}


@dataclass
class InkProfile:
    cap_height_px: float
    x_height_px: float
    stroke_width_px: float
    slant_deg: float
    ink_color: str
    ink_density: float


def _cv2():
    import cv2

    return cv2


def analyse_ink(crop: np.ndarray) -> InkProfile | None:
    """Measure glyph metrics from an element crop."""
    cv2 = _cv2()
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) < 120:
        gray = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = binary > 0
    if ink.sum() < 20:
        return None

    rows = ink.sum(axis=1).astype(np.float64)
    if rows.max() <= 0:
        return None
    dense_heights: list[float] = []
    band_heights: list[float] = []
    for start, end in _ink_bands(rows):
        band = rows[start:end]
        peak = float(band.max())
        if peak <= 0:
            continue
        dense = np.nonzero(band >= peak * 0.4)[0]
        if dense.size:
            dense_heights.append(float(dense[-1] - dense[0] + 1))
        band_heights.append(float(end - start))
    if not dense_heights:
        return None
    # The dense core of a text line is its x-height for mixed-case text and its
    # cap height for upper-case text. Both ratios were calibrated against
    # rendered DejaVu samples (see docs/ARCHITECTURE.md).
    x_height = float(np.median(dense_heights))
    cap_height = float(np.median(band_heights))

    # stroke width from the distance transform of the ink mask
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    stroke_samples = distance[ink & (distance > 0)]
    stroke_width = float(np.median(stroke_samples) * 2) if stroke_samples.size else 1.0

    # slant: second moments of the ink distribution
    ys, xs = np.nonzero(ink)
    slant = 0.0
    if xs.size > 30:
        cov = np.cov(np.vstack([xs, ys]))
        if cov[0, 0] > 0 and np.isfinite(cov[0, 1]):
            slant = float(np.degrees(np.arctan(cov[0, 1] / cov[0, 0])))
            slant = max(-20.0, min(20.0, slant))

    dark_pixels = crop[ink]
    if dark_pixels.size:
        bgr = np.median(dark_pixels.reshape(-1, 3), axis=0)
        color = "#{:02x}{:02x}{:02x}".format(int(bgr[2]), int(bgr[1]), int(bgr[0]))
    else:
        color = "#000000"

    return InkProfile(
        cap_height_px=cap_height,
        x_height_px=x_height,
        stroke_width_px=stroke_width,
        slant_deg=slant,
        ink_color=color,
        ink_density=float(ink.mean()),
    )


def _ink_bands(row_profile: np.ndarray) -> list[tuple[int, int]]:
    """Split a row-ink profile into contiguous text bands (one per line)."""
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(row_profile):
        if value > 0 and start is None:
            start = index
        elif value <= 0 and start is not None:
            bands.append((start, index))
            start = None
    if start is not None:
        bands.append((start, len(row_profile)))
    return bands


@lru_cache(maxsize=64)
def _font_path(family: str) -> str | None:
    """Resolve a family name to a font file via fontconfig."""
    import subprocess

    try:
        completed = subprocess.run(
            ["fc-match", "-f", "%{file}", family],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    path = completed.stdout.strip()
    return path if path and Path(path).is_file() else None


@lru_cache(maxsize=256)
def _measure_text(family: str, size_px: int, text: str) -> tuple[float, float] | None:
    """Measure rendered text width/height for a candidate family."""
    path = _font_path(family)
    if path is None:
        return None
    try:
        from PIL import ImageFont

        font = ImageFont.truetype(path, size_px)
        bbox = font.getbbox(text or "M")
        return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
    except Exception:
        return None


def estimate_font_family(text: str, element_width_px: float, size_px: float) -> tuple[str | None, list[str], float]:
    """Pick the family whose rendered width best matches the observed width.

    Returns ``(best_family, ranked_candidates, confidence)``. Confidence is the
    relative margin between the best and the runner-up, so a near-tie yields a
    low confidence instead of a false claim of certainty.
    """
    if not text.strip() or size_px <= 1 or element_width_px <= 1:
        return None, [], 0.0
    sample = text.strip().splitlines()[0][:80]
    scores: list[tuple[str, float]] = []
    for family in CANDIDATE_FAMILIES:
        measured = _measure_text(family, int(round(size_px)), sample)
        if not measured:
            continue
        width, _ = measured
        if width <= 0:
            continue
        ratio = width / element_width_px
        # ideal ratio is ~1.0; penalise both over- and under-estimation
        score = 1.0 / (1.0 + abs(ratio - 1.0))
        scores.append((family, score))
    if not scores:
        return None, [], 0.0
    scores.sort(key=lambda item: -item[1])
    best_family, best_score = scores[0]
    runner_up = scores[1][1] if len(scores) > 1 else 0.0
    confidence = max(0.0, min(1.0, (best_score - runner_up) / best_score if best_score else 0.0))
    confidence = round(0.35 + 0.6 * confidence, 3)
    return best_family, [family for family, _ in scores[:3]], confidence


def estimate_typography(
    element: Element,
    crop: np.ndarray,
    page: PageIR,
    settings: Settings,
) -> Typography:
    """Estimate typography for one text element."""
    dpi = page.dpi or settings.target_dpi
    notes: list[str] = []
    profile = analyse_ink(crop)

    size_pt: float | None = None
    weight = 400
    italic = False
    color = "#000000"
    confidence = 0.3

    if profile is not None:
        x_height_px = max(profile.x_height_px, 1.0)
        # Calibrated on rendered DejaVu text: the dense ink core measures about
        # 0.55 em for mixed-case text and 0.73 em for upper-case text.
        has_lowercase = any(character.islower() for character in (element.text or ""))
        core_ratio = 0.55 if has_lowercase else 0.73
        em_px = x_height_px / core_ratio
        family_guess, candidates, family_confidence = estimate_font_family(
            element.text or "", element.bbox.width, em_px
        )
        size_pt = round(em_px * 72.0 / dpi, 2)
        # guard rails: nothing below 4pt or above 96pt is plausible
        size_pt = max(4.0, min(96.0, size_pt))
        stroke_ratio = profile.stroke_width_px / x_height_px if x_height_px else 0.0
        if stroke_ratio > 0.19:
            weight = 700
        elif stroke_ratio > 0.145:
            weight = 600
        elif stroke_ratio < 0.075:
            weight = 300
        italic = abs(profile.slant_deg) > 6.0
        color = profile.ink_color
        confidence = round(min(0.9, 0.35 + family_confidence * 0.5), 3)
        notes.append(
            f"size from the dense ink core ({x_height_px:.1f}px / {core_ratio:.2f} em "
            f"at {dpi:.0f} dpi)"
        )
        if italic:
            notes.append(f"slant measured at {profile.slant_deg:.1f} degrees")
    else:
        family_guess, candidates, family_confidence = None, [], 0.0
        notes.append("no ink measured; typography is a default estimate")

    line_boxes = [line.bbox for line in element.lines] or [element.bbox]
    alignment = element.alignment or infer_alignment(line_boxes, element.bbox)

    line_height_pt: float | None = None
    spacing_ratio: float | None = None
    if len(line_boxes) > 1:
        ordered = sorted(line_boxes, key=lambda b: b.y1)
        gaps = [
            ordered[i + 1].y1 - ordered[i].y1
            for i in range(len(ordered) - 1)
            if ordered[i + 1].y1 > ordered[i].y1
        ]
        if gaps:
            median_gap = statistics.median(gaps)
            line_height_pt = round(median_gap * 72.0 / dpi, 2)
            if size_pt:
                spacing_ratio = round(line_height_pt / size_pt, 3)
    if line_height_pt is None and size_pt:
        line_height_pt = round(size_pt * 1.2, 2)
        spacing_ratio = 1.2

    indent_first_line_mm: float | None = None
    indent_left_mm: float | None = None
    if len(line_boxes) > 1:
        lefts = [b.x1 for b in sorted(line_boxes, key=lambda b: b.y1)]
        indent_left_mm = round(min(lefts) * 25.4 / dpi, 2)
        indent_first_line_mm = round(max(0.0, (lefts[0] - min(lefts[1:] or lefts)) * 25.4 / dpi), 2)

    paragraph_spacing_pt = None
    if element.metadata.get("gap_below"):
        paragraph_spacing_pt = round(float(element.metadata["gap_below"]) * 72.0 / dpi, 2)

    # guard against implausible estimates: a stray measurement must never turn
    # body text into a headline or a smudge into unreadable 2pt type
    size_pt = round(min(MAX_FONT_PT, max(MIN_FONT_PT, float(size_pt))), 2)
    line_height_pt = round(min(MAX_FONT_PT * 2.0, max(MIN_FONT_PT, float(line_height_pt))), 2)

    typography = Typography(
        font_family=family_guess,
        font_family_candidates=candidates,
        font_size_pt=size_pt,
        font_weight=weight,
        italic=italic,
        color=color,
        line_height_pt=line_height_pt,
        line_spacing_ratio=spacing_ratio,
        paragraph_spacing_pt=paragraph_spacing_pt,
        indent_first_line_mm=indent_first_line_mm,
        indent_left_mm=indent_left_mm,
        estimated=True,
        confidence=confidence,
        notes=notes,
    )
    return typography


def estimate_page_typography(page: PageIR, image: np.ndarray, settings: Settings) -> list[str]:
    """Estimate typography for every text element of a page."""
    from .preprocess import crop_box

    warnings: list[str] = []
    for element in page.elements:
        if element.type not in {
            ElementType.HEADING,
            ElementType.SUBHEADING,
            ElementType.PARAGRAPH,
            ElementType.TEXT_BLOCK,
            ElementType.LIST,
            ElementType.FIGURE_CAPTION,
            ElementType.TABLE_CAPTION,
            ElementType.FOOTER,
            ElementType.HEADER,
            ElementType.PAGE_NUMBER,
            ElementType.FORMULA,
            ElementType.FORM_FIELD,
        }:
            continue
        try:
            crop = crop_box(image, element.bbox, padding=1)
            element.typography = estimate_typography(element, crop, page, settings)
            if element.typography.confidence is not None and element.typography.confidence < 0.5:
                from .schemas import ReviewFlag

                element.add_flag(ReviewFlag.TYPOGRAPHY_ESTIMATED)
        except Exception as exc:  # noqa: BLE001 - never fail the page for typography
            warnings.append(f"typography estimation failed for {element.id}: {exc}")
    return warnings


def typography_to_css(typography: Typography | None, default_size_pt: float = 11.0) -> dict[str, str]:
    """Translate a :class:`Typography` into CSS declarations."""
    css: dict[str, str] = {}
    if typography is None:
        css["font-size"] = f"{default_size_pt}pt"
        return css
    size = typography.font_size_pt or default_size_pt
    css["font-size"] = f"{round(size, 2)}pt"
    if typography.font_family:
        css["font-family"] = f"'{typography.font_family}', 'DejaVu Sans', 'Liberation Sans', sans-serif"
    if typography.font_weight:
        css["font-weight"] = str(typography.font_weight)
    if typography.italic:
        css["font-style"] = "italic"
    if typography.color:
        css["color"] = typography.color
    if typography.line_height_pt:
        css["line-height"] = f"{round(typography.line_height_pt, 2)}pt"
    if typography.letter_spacing_em:
        css["letter-spacing"] = f"{typography.letter_spacing_em}em"
    return css


def typography_to_docx(typography: Typography | None) -> dict[str, Any]:
    """Translate a :class:`Typography` into python-docx run properties."""
    result: dict[str, Any] = {"size_pt": 11.0, "bold": False, "italic": False, "color": None, "family": None}
    if typography is None:
        return result
    if typography.font_size_pt:
        result["size_pt"] = typography.font_size_pt
    if typography.font_weight:
        result["bold"] = typography.font_weight >= 600
    if typography.italic:
        result["italic"] = True
    if typography.color:
        result["color"] = typography.color.lstrip("#").upper()
    if typography.font_family:
        result["family"] = typography.font_family
    return result


def pt_from_mm(mm: float) -> float:
    return mm_to_pt(mm)


def mm_from_pt(pt: float) -> float:
    return pt_to_mm(pt)
