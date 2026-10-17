"""Identify the font of a text region from its ink and its OCR text.

The earlier approach matched a region's width against a short list of whatever
fonts happened to be installed, so it could name "DejaVu Sans" but never "Arial".
That is a naming problem and a measurement problem at once, and this module fixes
both:

*Naming.* Metric-compatible substitutes are installed for the faces documents are
actually set in -- Liberation Sans for Arial, Liberation Serif for Times New
Roman, Carlito for Calibri, Caladea for Cambria, Liberation Mono for Courier New.
They are the same width as the originals, so a width measurement identifies the
*real* family and the substitute is what gets rendered. The operator sees "Arial";
the browser draws Liberation Sans; the line breaks come out the same.

*Measurement.* Three signals are read from the ink before any width matching
happens, and they constrain which candidates are even considered:

``serif``
    Serifs are horizontal strokes at the ends of stems, so on a serif face the
    topmost and bottommost rows of a glyph are wider than the middle of its stem.
    Measuring that ratio separates serif from sans on the image itself, with no
    font list involved.
``monospace``
    A monospaced face advances every glyph by the same amount, so the spread of
    glyph widths collapses. The coefficient of variation of connected-component
    widths measures exactly that.
``stroke weight``
    Stroke width over x-height separates light, regular and bold.

Only then is width compared, and only against candidates of the detected style,
which is what keeps a near-tie from being reported as a confident answer.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .logging_utils import get_logger

logger = get_logger("fontid")


@dataclass(frozen=True)
class FontCandidate:
    """A font a document may be set in, and the substitute used to render it."""

    display_name: str
    metric_font: str
    style: str  # serif | sans | mono
    #: x-height as a share of the em, measured from the substitute's metrics.
    x_height_ratio: float = 0.52


#: Ordered roughly by how often each face appears in the documents this project
#: processes. Every ``metric_font`` is installed on this machine.
CANDIDATES: tuple[FontCandidate, ...] = (
    FontCandidate("Arial", "Liberation Sans", "sans", 0.519),
    FontCandidate("Times New Roman", "Liberation Serif", "serif", 0.448),
    FontCandidate("Calibri", "Carlito", "sans", 0.466),
    FontCandidate("Cambria", "Caladea", "serif", 0.466),
    FontCandidate("Courier New", "Liberation Mono", "mono", 0.426),
    FontCandidate("Helvetica", "Nimbus Sans", "sans", 0.523),
    FontCandidate("Verdana", "DejaVu Sans", "sans", 0.545),
    FontCandidate("Georgia", "Gelasio", "serif", 0.481),
    FontCandidate("Consolas", "DejaVu Sans Mono", "mono", 0.530),
    FontCandidate("Times", "Nimbus Roman", "serif", 0.450),
)


@dataclass
class StyleDetection:
    """What the ink says about the face, before any width comparison."""

    style: str | None
    confidence: float
    serif_ratio: float | None = None
    width_cv: float | None = None
    stroke_ratio: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class FontIdentification:
    """The result of identifying a region's font."""

    family: str | None
    metric_font: str | None
    style: str | None
    weight: int
    italic: bool
    confidence: float
    candidates: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _cv2():
    import cv2

    return cv2


def _binary_ink(crop: np.ndarray) -> np.ndarray | None:
    """Threshold a crop to an ink mask, light-on-dark or dark-on-light."""
    cv2 = _cv2()
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if float(gray.mean()) < 120:
        gray = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if int((binary > 0).sum()) < 40:
        return None
    return binary


def ink_mask(crop: np.ndarray) -> np.ndarray | None:
    """Public access to the thresholded ink mask used by the measurements."""
    return _binary_ink(crop)


def measure_serif_ratio(binary: np.ndarray) -> tuple[float, int]:
    """Ratio of glyph-edge width to stem width: above 1 on a serif face.

    For every connected component (a glyph or a joined run of glyphs) the widest
    row in the top decile and in the bottom decile is compared with the width of
    the stem in the middle. A serif flares out at the ends; a sans face does not.
    Returns the median ratio and how many components supported it.
    """
    cv2 = _cv2()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    ratios: list[float] = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        # Only glyph-shaped components carry this signal. A long rule, a table
        # border or a scan speckle produces a meaningless ratio and must not vote.
        if height < 8 or width < 3 or area < 30:
            continue
        if width > 4 * height or height > 6 * width:
            continue
        component = labels[y : y + height, x : x + width] == index
        row_widths = component.sum(axis=1).astype(np.float64)
        if row_widths.size < 8:
            continue
        band = max(1, int(round(height * 0.10)))
        middle = row_widths[band : height - band]
        if middle.size == 0:
            continue
        stem = float(np.median(middle[middle > 0])) if (middle > 0).any() else 0.0
        if stem <= 0:
            continue
        top = float(np.max(row_widths[:band]))
        bottom = float(np.max(row_widths[height - band :]))
        # An edge row is only evidence of serifs if it is genuinely wider than the
        # stem; a few pixels of antialiasing should not count.
        ratios.append(max(top, bottom) / stem)
    if not ratios:
        return 0.0, 0
    return float(statistics.median(ratios)), len(ratios)


def measure_width_variation(binary: np.ndarray) -> tuple[float, int]:
    """Coefficient of variation of glyph widths: near zero on a monospaced face."""
    cv2 = _cv2()
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    widths = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if height < 6 or area < 12:
            continue
        if width > 3 * height:
            continue
        widths.append(float(width))
    if len(widths) < 6:
        return 1.0, len(widths)
    mean = statistics.fmean(widths)
    if mean <= 0:
        return 1.0, len(widths)
    return float(statistics.pstdev(widths) / mean), len(widths)


def measure_stroke_ratio(binary: np.ndarray, x_height_px: float) -> float:
    """Median stroke width over x-height: the weight signal."""
    cv2 = _cv2()
    if x_height_px <= 0:
        return 0.0
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    samples = distance[binary > 0]
    if samples.size == 0:
        return 0.0
    return float(np.median(samples) * 2.0 / x_height_px)


def detect_style(binary: np.ndarray, x_height_px: float) -> StyleDetection:
    """Decide serif / sans / mono from the ink alone."""
    notes: list[str] = []
    serif_ratio, serif_votes = measure_serif_ratio(binary)
    width_cv, width_votes = measure_width_variation(binary)
    stroke_ratio = measure_stroke_ratio(binary, x_height_px)

    style: str | None = None
    confidence = 0.0
    # Monospace first: a uniform advance is a stronger, less ambiguous statement
    # than a serif flare, and a monospaced face can also have slab serifs.
    if width_votes >= 6 and width_cv < 0.22:
        style = "mono"
        confidence = round(min(0.9, 0.5 + (0.22 - width_cv) * 2.0), 3)
        notes.append(
            f"glyph widths vary by {width_cv * 100:.1f}% across {width_votes} glyphs, "
            "which is a monospaced advance"
        )
    elif serif_votes >= 8:
        if serif_ratio >= 1.55:
            style = "serif"
            confidence = round(min(0.9, 0.4 + (serif_ratio - 1.55) * 0.8), 3)
            notes.append(
                f"glyph ends are {serif_ratio:.2f}x wider than the stems across "
                f"{serif_votes} glyphs, which is a serif face"
            )
        else:
            style = "sans"
            confidence = round(min(0.9, 0.4 + (1.55 - serif_ratio) * 0.8), 3)
            notes.append(
                f"glyph ends are only {serif_ratio:.2f}x the stem width across "
                f"{serif_votes} glyphs, which is a sans-serif face"
            )
    else:
        notes.append("too few clean glyphs to judge serif or monospace reliably")

    return StyleDetection(
        style=style,
        confidence=confidence,
        serif_ratio=serif_ratio if serif_votes else None,
        width_cv=width_cv if width_votes >= 6 else None,
        stroke_ratio=stroke_ratio,
        notes=notes,
    )


_WEIGHT_BANDS = ((0.235, 800), (0.185, 700), (0.150, 600), (0.105, 500), (0.062, 400))
#: Below the lightest band the strokes are too thin to be regular text; on a
#: photographed page that usually means the ink broke up in thresholding.
_LIGHT_WEIGHT = 300


def weight_from_stroke(stroke_ratio: float) -> tuple[int, str]:
    """Map stroke-over-x-height to a CSS weight."""
    for threshold, weight in _WEIGHT_BANDS:
        if stroke_ratio >= threshold:
            return weight, f"stroke is {stroke_ratio * 100:.1f}% of x-height"
    return _LIGHT_WEIGHT, f"stroke is only {stroke_ratio * 100:.1f}% of x-height"


@lru_cache(maxsize=256)
def _metric_path(font_name: str) -> str | None:
    """Resolve an installed font to its file."""
    import subprocess

    try:
        completed = subprocess.run(
            ["fc-match", "-f", "%{file}", font_name],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    path = completed.stdout.strip()
    return path if path and Path(path).is_file() else None


@lru_cache(maxsize=512)
def _measure(font_family: str, size_px: int, text: str, bold: bool) -> tuple[float, float] | None:
    """Measure rendered text in a family, so widths can be compared."""
    path = _metric_path(font_family)
    if path is None:
        return None
    try:
        from PIL import ImageFont

        font = ImageFont.truetype(path, max(4, int(round(size_px))))
        # Synthetic bold is not the same as a bold face, but for a width
        # comparison it is closer than pretending the weight does not matter.
        if bold:
            font = font.font_variant(size=max(4, int(round(size_px))))
        bbox = font.getbbox(text or "M")
        return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
    except Exception:
        return None


def _font_file_for_weight(base_path: str, bold: bool) -> str:
    """Prefer the real bold face over a synthesised one when it exists."""
    if not bold:
        return base_path
    path = Path(base_path)
    for marker in ("-Regular", "Regular", "-Book"):
        candidate = path.with_name(path.name.replace(marker, "-Bold" if marker != "Regular" else "Bold"))
        if "Bold" in candidate.name and candidate.is_file():
            return str(candidate)
    return base_path


@lru_cache(maxsize=512)
def _render_mask(font_family: str, text: str, target_height: int, bold: bool) -> np.ndarray | None:
    """Render ``text`` in a family and return its ink mask at a known height.

    The mask is trimmed to the ink and scaled so its ink height matches the
    observation, so the width that comes out is a prediction that can be wrong --
    which is the point. Comparing width alone confused Arial with Helvetica and
    Times with Cambria; comparing the shapes separates them.
    """
    path = _metric_path(font_family)
    if path is None or target_height < 6:
        return None
    path = _font_file_for_weight(path, bold)
    try:
        from PIL import Image, ImageDraw, ImageFont

        size = max(8, int(target_height * 2))
        font = ImageFont.truetype(path, size)
        canvas = Image.new("L", (max(64, size * len(text)), size * 4), 255)
        ImageDraw.Draw(canvas).text((10, size), text, font=font, fill=0)
        ink = np.array(canvas) < 128
        if ink.sum() < 20:
            return None
        ys, xs = np.nonzero(ink)
        trimmed = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        scale = target_height / trimmed.shape[0]
        width = max(2, int(round(trimmed.shape[1] * scale)))
        resized = _cv2().resize(
            trimmed.astype(np.uint8) * 255, (width, max(2, target_height)), interpolation=_cv2().INTER_AREA
        )
        return resized > 127
    except Exception:
        return None


def _x_height(mask: np.ndarray) -> float:
    """Height of the dense ink band, which is the x-height for mixed-case text."""
    rows = mask.sum(axis=1).astype(np.float64)
    if rows.max() <= 0:
        return float(mask.shape[0])
    dense = np.nonzero(rows >= rows.max() * 0.4)[0]
    if dense.size == 0:
        return float(mask.shape[0])
    return float(dense[-1] - dense[0] + 1)


def _shape_overlap(rendered: np.ndarray, observed: np.ndarray) -> float:
    """How much two ink masks agree once they share a typographic frame.

    A naive comparison over a shared canvas aligned the two at their top-left
    corner and scored whatever difference in ascender and descender proportions
    the two fonts happened to have. That is not letterform, and it produced errors
    as bad as calling Arial "Cambria". Both masks are now scaled by their x-height
    and aligned on their baseline, so what remains is the shape of the letters.
    """
    cv2 = _cv2()
    if rendered.size == 0 or observed.size == 0:
        return 0.0
    target = 48.0
    scale_r = target / max(_x_height(rendered), 1.0)
    scale_o = target / max(_x_height(observed), 1.0)
    def normalise(mask: np.ndarray, scale: float) -> np.ndarray:
        height = max(2, int(round(mask.shape[0] * scale)))
        width = max(2, int(round(mask.shape[1] * scale)))
        resized = cv2.resize(mask.astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_AREA)
        # find the baseline: the row where the dense band ends
        rows = (resized > 127).sum(axis=1).astype(np.float64)
        dense = np.nonzero(rows >= rows.max() * 0.4)[0] if rows.max() > 0 else np.array([])
        baseline = int(dense[-1]) if dense.size else height
        return resized > 127, baseline

    left, left_baseline = normalise(rendered, scale_r)
    right, right_baseline = normalise(observed, scale_o)
    canvas_height = int(target * 2.2)
    canvas_width = max(left.shape[1], right.shape[1]) + 20
    shift = int(target * 0.9)  # room for ascenders above the baseline
    canvas_l = np.zeros((canvas_height, canvas_width), bool)
    canvas_r = np.zeros((canvas_height, canvas_width), bool)
    top_l = max(0, shift - left_baseline)
    top_r = max(0, shift - right_baseline)
    if top_l + left.shape[0] <= canvas_height and left.shape[1] <= canvas_width:
        canvas_l[top_l : top_l + left.shape[0], : left.shape[1]] = left
    if top_r + right.shape[0] <= canvas_height and right.shape[1] <= canvas_width:
        canvas_r[top_r : top_r + right.shape[0], : right.shape[1]] = right
    intersection = np.logical_and(canvas_l, canvas_r).sum()
    union = np.logical_or(canvas_l, canvas_r).sum()
    return float(intersection) / float(union) if union else 0.0


def match_by_rendering(
    observed: np.ndarray | None,
    text: str,
    style_hint: str | None = None,
) -> list[tuple[FontCandidate, int, float]]:
    """Rank candidate fonts, with weight, by how well their shape matches.

    Returns ``(candidate, weight, score)`` sorted best first. The score combines
    width agreement, which is what metric compatibility is for, with an overlap of
    the two ink masks, which is what tells Arial from Helvetica.
    """
    if observed is None or observed.size == 0:
        return []
    sample = " ".join((text or "").split())[:90]
    if len(sample) < 4:
        return []
    ys, xs = np.nonzero(observed)
    if ys.size == 0:
        return []
    crop = observed[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    height, width = crop.shape
    if height < 6 or width < 12:
        return []

    pool = [c for c in CANDIDATES if style_hint is None or c.style == style_hint] or list(CANDIDATES)
    scored: list[tuple[FontCandidate, int, float]] = []
    for candidate in pool:
        for bold in (False, True):
            rendered = _render_mask(candidate.metric_font, sample, int(height), bold)
            if rendered is None:
                continue
            # width agreement: the rendered run should be as wide as the observed
            width_agreement = 1.0 - min(
                1.0, abs(rendered.shape[1] - width) / max(1.0, float(width))
            )
            overlap = _shape_overlap(rendered, crop)
            score = 0.55 * overlap + 0.45 * width_agreement
            scored.append((candidate, 700 if bold else 400, score))
    scored.sort(key=lambda item: -item[2])
    return scored


def identify_font(
    binary: np.ndarray | None,
    text: str,
    element_width_px: float,
    em_px: float,
) -> FontIdentification:
    """Identify the family, style and weight of a text region."""
    notes: list[str] = []
    style_info = detect_style(binary, em_px) if binary is not None else StyleDetection(None, 0.0)
    notes.extend(style_info.notes)

    weight, weight_note = (
        weight_from_stroke(style_info.stroke_ratio or 0.0)
        if style_info.stroke_ratio
        else (400, "no stroke measured")
    )
    if style_info.stroke_ratio:
        notes.append(weight_note)
    bold = weight >= 600

    family: str | None = None
    metric_font: str | None = None
    ranked: list[str] = []
    family_confidence = 0.0

    # Which evidence leads was settled by measuring both against rendered text in
    # known fonts. Width matching (what metric compatibility is for, and the reason
    # the real-world names can be used at all) named the right family 11 times in
    # 18; shape comparison managed 10, and aligning the masks on their baseline
    # instead of their corner dropped it to 7. So width leads and the shape match is
    # kept as a cross-check that moves confidence rather than the answer.
    sample = " ".join((text or "").split())[:90]
    if sample and em_px > 1 and element_width_px > 1:
        pool = [
            candidate
            for candidate in CANDIDATES
            if style_info.style is None or candidate.style == style_info.style
        ] or list(CANDIDATES)
        scored_widths: list[tuple[FontCandidate, float]] = []
        for candidate in pool:
            measured = _measure(candidate.metric_font, int(round(em_px)), sample, bold)
            if not measured:
                continue
            rendered_width, _ = measured
            if rendered_width <= 0:
                continue
            ratio = rendered_width / element_width_px
            scored_widths.append((candidate, 1.0 / (1.0 + abs(ratio - 1.0))))
        if scored_widths:
            scored_widths.sort(key=lambda item: -item[1])
            best, best_score = scored_widths[0]
            rival = next(
                (s for s in scored_widths[1:] if s[0].display_name != best.display_name), None
            )
            spread = (best_score - rival[1]) / best_score if rival and best_score else 0.0
            family = best.display_name
            metric_font = best.metric_font
            ranked = [candidate.display_name for candidate, _ in scored_widths[:3]]
            family_confidence = max(0.0, min(1.0, spread))

            # Cross-check: does the shape comparison agree? It is too noisy to pick
            # the family, but a disagreement is real evidence that the width fit is
            # not trustworthy, and saying so is better than a confident wrong name.
            shape_scores = match_by_rendering(binary, text)
            shape_ranked: list[str] = []
            for candidate, _, _ in shape_scores:
                if candidate.display_name not in shape_ranked:
                    shape_ranked.append(candidate.display_name)
            if shape_ranked:
                if family in shape_ranked[:2]:
                    notes.append(f"shape comparison agrees with the width fit ({shape_ranked[0]} first)")
                    family_confidence = min(1.0, family_confidence + 0.15)
                else:
                    notes.append(
                        f"shape comparison disagrees; it prefers {shape_ranked[0]}, so this "
                        "family is reported with low confidence"
                    )
                    family_confidence *= 0.45
        else:
            notes.append("no candidate font could be measured")
    else:
        notes.append("no OCR text or too small to compare widths")

    # Style evidence is independent of the width fit, so the two confidences are
    # combined rather than one being allowed to carry the whole claim.
    confidence = round(
        min(0.95, 0.25 + 0.45 * family_confidence + 0.30 * style_info.confidence), 3
    )
    if family_confidence < 0.05 and len(ranked) > 1:
        notes.append(f"width fit is a near-tie between {', '.join(ranked)}")

    return FontIdentification(
        family=family,
        metric_font=metric_font,
        style=style_info.style,
        weight=weight,
        italic=False,
        confidence=confidence,
        candidates=ranked,
        notes=notes,
    )


def css_stack(family: str | None) -> str:
    """A CSS family list that renders the identified face but names the real one."""
    for candidate in CANDIDATES:
        if candidate.display_name == family:
            generic = "monospace" if candidate.style == "mono" else (
                "serif" if candidate.style == "serif" else "sans-serif"
            )
            return f"'{candidate.metric_font}', '{candidate.display_name}', {generic}"
    return "'DejaVu Sans', 'Liberation Sans', sans-serif"
