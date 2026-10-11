"""Geometry engine: page formats, coordinate mapping and layout measurements.

This module is deliberately free of OpenCV/YOLO imports so it can be unit
tested in isolation. It answers the questions the renderer needs:

* how large is the page, in millimetres and points?
* where is each element, relative to the page and to its neighbours?
* how are elements aligned, how much space surrounds them?
* how do coordinates in the uploaded image relate to the analysed page?
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .schemas import Alignment, BBox, Element, ElementType, MarginMM, PageFormat, PageTransform

MM_PER_INCH = 25.4
PT_PER_INCH = 72.0

#: Named page formats in millimetres (width x height, portrait).
PAGE_FORMATS: dict[str, tuple[float, float]] = {
    "A3": (297.0, 420.0),
    "A4": (210.0, 297.0),
    "A5": (148.0, 210.0),
    "A6": (105.0, 148.0),
    "B5": (176.0, 250.0),
    "Letter": (215.9, 279.4),
    "Legal": (215.9, 355.6),
    "Tabloid": (279.4, 431.8),
}

DEFAULT_PAGE_FORMAT = "A4"


# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------
def mm_to_pt(mm: float) -> float:
    return mm * PT_PER_INCH / MM_PER_INCH


def pt_to_mm(pt: float) -> float:
    return pt * MM_PER_INCH / PT_PER_INCH


def px_to_mm(px: float, dpi: float) -> float:
    return px * MM_PER_INCH / dpi if dpi else 0.0


def mm_to_px(mm: float, dpi: float) -> float:
    return mm * dpi / MM_PER_INCH if dpi else 0.0


def mm_to_css(mm: float) -> str:
    return f"{round(mm, 3)}mm"


# --------------------------------------------------------------------------
# page format detection
# --------------------------------------------------------------------------
def infer_page_format(
    width_px: int,
    height_px: int,
    line_height_px: float | None = None,
) -> tuple[PageFormat, float, list[str]]:
    """Infer the physical page size and effective DPI of a scan.

    A bitmap alone cannot prove paper size, so the reasoning is explicit and
    returned with the result:

    1. the aspect ratio selects a format *family* (ISO A-series or US Letter
       series) — this is reliable, because it is scale invariant,
    2. the most common member of that family is assumed (A4 or Letter),
    3. when a text line height is measurable, the assumption is cross-checked
       against a plausible body-text size and neighbouring formats in the same
       family are tried instead.

    Returns ``(page_format, dpi, notes)``.
    """
    if width_px <= 0 or height_px <= 0:
        return (
            PageFormat(name=DEFAULT_PAGE_FORMAT, width_mm=210.0, height_mm=297.0, estimated=True),
            300.0,
            ["degenerate page dimensions; assuming A4 at 300 dpi"],
        )

    landscape = width_px > height_px
    short_px, long_px = (height_px, width_px) if landscape else (width_px, height_px)
    ratio = long_px / short_px
    notes: list[str] = []

    families = {"A4": A_SERIES, "Letter": LETTER_SERIES}

    def family_delta(candidate: str) -> float:
        w_mm, h_mm = PAGE_FORMATS[candidate]
        target = max(w_mm, h_mm) / min(w_mm, h_mm)
        return abs(ratio - target) / target

    family = min(families, key=family_delta)
    if family_delta(family) > 0.08:
        notes.append(
            f"aspect ratio {ratio:.3f} matches no standard format closely; assuming the {family} family"
        )

    def dpi_for(name: str) -> float:
        w_mm, h_mm = PAGE_FORMATS[name]
        short_mm = min(w_mm, h_mm)
        return short_px / (short_mm / MM_PER_INCH) if short_mm else 300.0

    chosen = "A4" if family == "A4" else "Letter"
    notes.append(f"assumed {chosen} from the aspect ratio ({ratio:.3f})")

    if line_height_px and line_height_px > 2:
        best_name, best_error = chosen, None
        for name in families[family]:
            body_pt = line_height_px / dpi_for(name) * PT_PER_INCH / ASSUMED_LINE_SPACING
            error = abs(body_pt - TYPICAL_BODY_PT)
            if best_error is None or error < best_error:
                best_name, best_error = name, error
        chosen_body_pt = line_height_px / dpi_for(chosen) * PT_PER_INCH / ASSUMED_LINE_SPACING
        if best_name != chosen and not (PLAUSIBLE_BODY_PT[0] <= chosen_body_pt <= PLAUSIBLE_BODY_PT[1]):
            notes.append(
                f"line pitch {line_height_px:.0f}px implies {chosen_body_pt:.1f}pt body text on "
                f"{chosen}; using {best_name} instead"
            )
            chosen = best_name
        else:
            notes.append(f"body text estimated at {chosen_body_pt:.1f}pt on {chosen}")

    dpi = dpi_for(chosen)
    w_mm, h_mm = PAGE_FORMATS[chosen]
    if landscape:
        w_mm, h_mm = h_mm, w_mm
    notes.append(f"effective resolution {dpi:.0f} dpi (estimated, not read from metadata)")
    return (
        PageFormat(name=chosen, width_mm=w_mm, height_mm=h_mm, estimated=True, notes=notes),
        round(dpi, 1),
        notes,
    )


def detect_page_format(
    width_px: int,
    height_px: int,
    dpi: float | None = None,
    tolerance: float = 0.06,
    line_height_px: float | None = None,
) -> PageFormat:
    """Infer the physical page format.

    With an explicit ``dpi`` the physical size follows from the pixel count;
    otherwise :func:`infer_page_format` is used and its assumptions are recorded
    on the result. The format is always marked ``estimated=True`` because a scan
    alone cannot prove paper size.
    """
    if width_px <= 0 or height_px <= 0:
        return PageFormat(name=DEFAULT_PAGE_FORMAT, width_mm=210.0, height_mm=297.0, estimated=True)

    if dpi:
        w_mm = px_to_mm(width_px, dpi)
        h_mm = px_to_mm(height_px, dpi)
        if w_mm > 20 and h_mm > 20:
            ratio = max(w_mm, h_mm) / min(w_mm, h_mm)
            for name, (candidate_w, candidate_h) in PAGE_FORMATS.items():
                target = max(candidate_w, candidate_h) / min(candidate_w, candidate_h)
                if abs(ratio - target) / target <= tolerance:
                    return PageFormat(
                        name=name,
                        width_mm=round(w_mm, 2),
                        height_mm=round(h_mm, 2),
                        estimated=True,
                        notes=[f"physical size from a supplied {dpi:.0f} dpi"],
                    )
            return PageFormat(
                name="custom",
                width_mm=round(w_mm, 2),
                height_mm=round(h_mm, 2),
                estimated=True,
                notes=[f"physical size from a supplied {dpi:.0f} dpi"],
            )

    fmt, _dpi, _notes = infer_page_format(width_px, height_px, line_height_px)
    return fmt


def orientation_of(width_px: int, height_px: int) -> str:
    return "landscape" if width_px > height_px else "portrait"


# --------------------------------------------------------------------------
# coordinate mapping through preprocessing transforms
# --------------------------------------------------------------------------
def _map_point(kind: str, params: dict[str, Any], x: float, y: float) -> tuple[float, float]:
    if kind == "crop":
        return x - float(params.get("x", 0)), y - float(params.get("y", 0))
    if kind == "resize":
        return x * float(params.get("scale_x", 1.0)), y * float(params.get("scale_y", 1.0))
    if kind == "rotate":
        quarter_turns = int(params.get("quarter_turns", 0)) % 4
        out_w = float(params.get("output_w", 0) or 0)
        out_h = float(params.get("output_h", 0) or 0)
        if quarter_turns == 1:
            return out_w - y, x
        if quarter_turns == 2:
            return out_w - x, out_h - y
        if quarter_turns == 3:
            return y, out_h - x
        return x, y
    if kind == "perspective":
        matrix = params.get("matrix") or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        m = [[float(v) for v in row] for row in matrix]
        denom = m[2][0] * x + m[2][1] * y + m[2][2]
        if abs(denom) < 1e-9:
            denom = 1e-9
        return (m[0][0] * x + m[0][1] * y + m[0][2]) / denom, (
            m[1][0] * x + m[1][1] * y + m[1][2]
        ) / denom
    if kind == "deskew":
        angle = math.radians(float(params.get("angle", 0.0)))
        cx = float(params.get("center_x", 0.0))
        cy = float(params.get("center_y", 0.0))
        dx, dy = x - cx, y - cy
        cos_a, sin_a = math.cos(-angle), math.sin(-angle)
        nx = dx * cos_a - dy * sin_a
        ny = dx * sin_a + dy * cos_a
        return nx + float(params.get("new_center_x", cx)), ny + float(params.get("new_center_y", cy))
    return x, y


def map_point_through_transforms(
    point: tuple[float, float], transforms: Sequence[PageTransform]
) -> tuple[float, float]:
    """Map a point from the original image into the processed page space."""
    x, y = point
    for transform in transforms:
        x, y = _map_point(transform.kind, transform.params, x, y)
    return x, y


def map_bbox_through_transforms(box: BBox, transforms: Sequence[PageTransform]) -> BBox:
    """Map a box, taking the bounding box of the mapped corners."""
    corners = [
        map_point_through_transforms((box.x1, box.y1), transforms),
        map_point_through_transforms((box.x2, box.y1), transforms),
        map_point_through_transforms((box.x2, box.y2), transforms),
        map_point_through_transforms((box.x1, box.y2), transforms),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))


def map_bbox_original_to_page(
    box: BBox, page_transforms: Sequence[PageTransform]
) -> BBox:
    """Convenience wrapper used by the pipeline to keep provenance coordinates."""
    return map_bbox_through_transforms(box, page_transforms)


# --------------------------------------------------------------------------
# layout measurements
# --------------------------------------------------------------------------
def infer_margins(
    elements: Sequence[Element], width_px: int, height_px: int, page_format: PageFormat
) -> MarginMM:
    """Estimate page margins from the extreme extents of content elements."""
    candidates = [
        e
        for e in elements
        if e.type
        not in {
            ElementType.HEADER,
            ElementType.FOOTER,
            ElementType.PAGE_NUMBER,
        }
        and e.bbox.width > 0
        and e.bbox.height > 0
    ]
    if not candidates:
        return MarginMM(top=15.0, right=15.0, bottom=15.0, left=15.0)

    mm_x = page_format.width_mm / width_px if width_px else 0.0
    mm_y = page_format.height_mm / height_px if height_px else 0.0
    left = min(e.bbox.x1 for e in candidates) * mm_x
    right = (width_px - max(e.bbox.x2 for e in candidates)) * mm_x
    top = min(e.bbox.y1 for e in candidates) * mm_y
    bottom = (height_px - max(e.bbox.y2 for e in candidates)) * mm_y

    def clamp(value: float) -> float:
        return round(max(0.0, min(value, 60.0)), 2)

    return MarginMM(top=clamp(top), right=clamp(right), bottom=clamp(bottom), left=clamp(left))


def infer_alignment(line_boxes: Sequence[BBox], container: BBox) -> Alignment:
    """Classify horizontal alignment of a multi-line text block.

    Uses the variance of the left edge, right edge and centre of the lines.
    Justified text has aligned left and right edges while its last line is
    short, which is why the last line is excluded from the right-edge test.
    """
    if not line_boxes:
        return Alignment.LEFT
    if len(line_boxes) == 1:
        box = line_boxes[0]
        left_gap = box.x1 - container.x1
        right_gap = container.x2 - box.x2
        centre_off = abs(box.center[0] - container.center[0])
        if centre_off < max(2.0, container.width * 0.04):
            return Alignment.CENTER
        if right_gap < left_gap and right_gap < max(2.0, container.width * 0.05):
            return Alignment.RIGHT
        return Alignment.LEFT

    def spread(values: Iterable[float]) -> float:
        values = list(values)
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return max(abs(v - mean) for v in values)

    left_spread = spread(b.x1 for b in line_boxes)
    centre_spread = spread(b.center[0] for b in line_boxes)
    body = list(line_boxes[:-1])
    right_spread = spread(b.x2 for b in body)
    widths = [b.width for b in line_boxes]
    width_tolerance = max(3.0, container.width * 0.03)

    # Centred text keeps its centres together while both edges move; the
    # earlier rule also demanded ragged left edges, which misread a centred
    # block whose lines happen to start at nearly the same x.
    if centre_spread <= max(2.0, container.width * 0.02) and centre_spread <= min(
        left_spread, right_spread
    ) + 0.5:
        return Alignment.CENTER
    if right_spread < width_tolerance and left_spread > width_tolerance:
        return Alignment.RIGHT
    if (
        len(body) >= 2
        and right_spread < width_tolerance
        and left_spread < width_tolerance
        and (max(widths) - widths[-1]) > width_tolerance * 2
    ):
        return Alignment.JUSTIFY
    return Alignment.LEFT


def estimate_line_spacing(line_boxes: Sequence[BBox]) -> tuple[float | None, float | None]:
    """Return ``(median_line_gap_px, median_line_height_px)`` for a text block."""
    if len(line_boxes) < 2:
        height = line_boxes[0].height if line_boxes else None
        return None, height
    ordered = sorted(line_boxes, key=lambda b: b.y1)
    gaps = [
        ordered[i + 1].y1 - ordered[i].y1
        for i in range(len(ordered) - 1)
        if ordered[i + 1].y1 > ordered[i].y1
    ]
    heights = [b.height for b in ordered if b.height > 0]
    gaps.sort()
    heights.sort()
    gap = gaps[len(gaps) // 2] if gaps else None
    height = heights[len(heights) // 2] if heights else None
    return gap, height


def detect_columns(elements: Sequence[Element], width_px: int, min_gap_ratio: float = 0.04) -> list[tuple[float, float]]:
    """Detect vertical column bands by projecting element x-extents.

    Returns a list of ``(x_start, x_end)`` bands in pixel space, ordered left to
    right. A single band means single-column layout.
    """
    boxes = [
        (e.bbox.x1, e.bbox.x2)
        for e in elements
        if e.type in {ElementType.PARAGRAPH, ElementType.TEXT_BLOCK, ElementType.HEADING, ElementType.LIST}
        and e.bbox.width > width_px * 0.05
    ]
    if not boxes:
        return [(0.0, float(width_px))]
    boxes.sort()
    min_gap = max(8.0, width_px * min_gap_ratio)
    bands: list[list[float]] = [[boxes[0][0], boxes[0][1]]]
    for x1, x2 in boxes[1:]:
        if x1 - bands[-1][1] <= min_gap:
            bands[-1][1] = max(bands[-1][1], x2)
        else:
            bands.append([x1, x2])
    return [(round(b[0], 2), round(b[1], 2)) for b in bands]


def find_overlaps(elements: Sequence[Element], threshold: float = 0.35) -> list[tuple[str, str, float]]:
    """Find element pairs whose boxes overlap more than ``threshold``."""
    result: list[tuple[str, str, float]] = []
    items = list(elements)
    for i, first in enumerate(items):
        for second in items[i + 1 :]:
            iou = first.bbox.iou(second.bbox)
            if iou >= threshold:
                result.append((first.id, second.id, round(iou, 4)))
    return result


def neighbour_gaps(elements: Sequence[Element]) -> dict[str, dict[str, float]]:
    """Vertical gap above/below each element, in pixels."""
    ordered = sorted(elements, key=lambda e: e.bbox.y1)
    gaps: dict[str, dict[str, float]] = {}
    for index, element in enumerate(ordered):
        above = ordered[index - 1].bbox.y2 if index > 0 else None
        below = ordered[index + 1].bbox.y1 if index < len(ordered) - 1 else None
        gaps[element.id] = {
            "gap_above": round(element.bbox.y1 - above, 2) if above is not None else 0.0,
            "gap_below": round(below - element.bbox.y2, 2) if below is not None else 0.0,
        }
    return gaps


@dataclass
class GeometrySummary:
    """Aggregated geometry facts for one page."""

    columns: list[tuple[float, float]]
    margins_mm: MarginMM
    overlaps: list[tuple[str, str, float]]
    gaps: dict[str, dict[str, float]]
    line_spacing_ratio: float | None = None
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []


def analyse_page_geometry(
    elements: Sequence[Element], width_px: int, height_px: int, page_format: PageFormat
) -> GeometrySummary:
    """Run the deterministic geometry analysis for one page."""
    notes: list[str] = []
    columns = detect_columns(elements, width_px)
    if len(columns) > 1:
        notes.append(f"{len(columns)} column layout detected")
    margins = infer_margins(elements, width_px, height_px, page_format)
    overlaps = find_overlaps(elements)
    if overlaps:
        notes.append(f"{len(overlaps)} overlapping element pair(s)")
    gaps = neighbour_gaps(elements)

    ratio: float | None = None
    paragraphs = [e for e in elements if e.type in {ElementType.PARAGRAPH, ElementType.TEXT_BLOCK} and len(e.lines) >= 2]
    if paragraphs:
        ratios = []
        for paragraph in paragraphs:
            gap, height = estimate_line_spacing([line.bbox for line in paragraph.lines])
            if gap and height:
                ratios.append(gap / height)
        if ratios:
            ratios.sort()
            ratio = round(ratios[len(ratios) // 2], 3)
    return GeometrySummary(columns=columns, margins_mm=margins, overlaps=overlaps, gaps=gaps, line_spacing_ratio=ratio, notes=notes)


def scale_font_size_to_fit(
    text: str, box_mm: dict[str, float], base_size_pt: float, char_width_em: float = 0.5
) -> float:
    """Shrink a font size until ``text`` plausibly fits inside ``box_mm``.

    Used by the correction loop and the renderer as a safety net so that long
    strings never overflow their reconstructed box.
    """
    width_pt = mm_to_pt(box_mm.get("width", 0.0))
    height_pt = mm_to_pt(box_mm.get("height", 0.0))
    if width_pt <= 0 or not text:
        return base_size_pt
    estimated_width = len(text) * base_size_pt * char_width_em
    if estimated_width <= width_pt:
        return base_size_pt
    scaled = base_size_pt * (width_pt / estimated_width)
    return max(4.0, min(base_size_pt, round(scaled, 2)) if height_pt else round(scaled, 2))

#: Formats that share the ISO 216 (square root of two) aspect ratio.
A_SERIES: tuple[str, ...] = ("A6", "A5", "A4", "A3")
#: Formats that share the US Letter aspect ratio.
LETTER_SERIES: tuple[str, ...] = ("Letter", "Legal", "Tabloid")
#: Body text a scan is expected to contain, in points.
#: Body text a scan may plausibly contain. The assumed format is only
#: overridden when the default would be implausible for the measured pitch.
PLAUSIBLE_BODY_PT: tuple[float, float] = (6.0, 17.0)
TYPICAL_BODY_PT = 11.0
ASSUMED_LINE_SPACING = 1.2
