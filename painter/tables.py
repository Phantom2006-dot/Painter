"""Table detection and reconstruction.

Two strategies are used, in order of reliability:

1. **Ruled tables** — horizontal and vertical rules are extracted with
   morphology, their projections give row and column boundaries, and cells are
   read individually. Missing internal segments between two neighbouring cells
   are treated as merged cells.
2. **Whitespace tables** — when no rules exist, columns and rows are inferred
   from gaps in the ink projection of the region.

Cell text is read with the configured OCR chain, so tables reconstruct into
editable DOCX tables rather than images.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .config import Settings
from .errors import TableReconstructionError
from .geometry import infer_alignment
from .logging_utils import get_logger
from .preprocess import crop_box
from .schemas import (
    Alignment,
    BBox,
    ElementType,
    ReviewFlag,
    TableCell,
    TableContent,
    TextLine,
)

logger = get_logger("tables")


def _cv2():
    import cv2

    return cv2


@dataclass
class TableGeometry:
    row_bounds: list[int] = field(default_factory=list)
    col_bounds: list[int] = field(default_factory=list)
    border_width: float = 1.0
    method: str = "rules"
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return max(0, len(self.row_bounds) - 1)

    @property
    def cols(self) -> int:
        return max(0, len(self.col_bounds) - 1)


def _line_masks(region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cv2 = _cv2()
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )
    height, width = gray.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, width // 25), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, height // 25)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    return horizontal, vertical


def _cluster_positions(
    projection: np.ndarray, threshold_ratio: float = 0.35, absolute_min: float = 0.0
) -> list[int]:
    """Cluster a 1-D projection into band centres."""
    if projection.size == 0:
        return []
    peak = float(projection.max())
    if peak <= 0:
        return []
    threshold = max(peak * threshold_ratio, absolute_min)
    active = projection >= threshold
    clusters: list[list[int]] = []
    current: list[int] = []
    for index, flag in enumerate(active):
        if flag:
            current.append(index)
        elif current:
            clusters.append(current)
            current = []
    if current:
        clusters.append(current)
    return [int(round(sum(cluster) / len(cluster))) for cluster in clusters]


def detect_table_regions(image: np.ndarray, min_area_ratio: float = 0.01) -> list[BBox]:
    """Find ruled table regions by intersecting horizontal and vertical rules."""
    cv2 = _cv2()
    height, width = image.shape[:2]
    horizontal, vertical = _line_masks(image)
    grid = cv2.bitwise_or(horizontal, vertical)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 60), max(9, height // 60)))
    closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(height * width)
    regions: list[BBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if float(w * h) < frame_area * min_area_ratio:
            continue
        if w < 40 or h < 30:
            continue
        regions.append(BBox(x1=x, y1=y, x2=x + w, y2=y + h))
    regions.sort(key=lambda b: (b.y1, b.x1))
    return regions


def analyse_table_geometry(region: np.ndarray) -> TableGeometry:
    """Derive row/column boundaries for a table region."""
    horizontal, vertical = _line_masks(region)
    height, width = region.shape[:2]
    geometry = TableGeometry()

    row_projection = (horizontal > 0).sum(axis=1).astype(np.float64)
    col_projection = (vertical > 0).sum(axis=0).astype(np.float64)
    # A real table rule spans most of the region, while text strokes only span a
    # few pixels; requiring 55% coverage is what separates the two.
    row_centres = _cluster_positions(
        row_projection, 0.5, absolute_min=max(3.0, 0.55 * width)
    )
    col_centres = _cluster_positions(
        col_projection, 0.5, absolute_min=max(3.0, 0.55 * height)
    )

    if len(row_centres) > 41 or len(col_centres) > 25:
        # implausible grid: the masks picked up text, fall back to whitespace
        row_centres, col_centres = [], []

    if len(row_centres) >= 2 and len(col_centres) >= 2:
        geometry.row_bounds = row_centres
        geometry.col_bounds = col_centres
        # Border thickness: the vertical extent of the horizontal rule clusters.
        active_rows = np.nonzero(row_projection > 0)[0]
        thickness = 1.0
        if active_rows.size:
            runs: list[int] = []
            run = 1
            for previous, current in zip(active_rows, active_rows[1:]):
                if current == previous + 1:
                    run += 1
                else:
                    runs.append(run)
                    run = 1
            runs.append(run)
            thickness = float(max(1.0, min(6.0, statistics.median(runs))))
        geometry.border_width = thickness
        geometry.method = "rules"
        geometry.confidence = 0.85
        geometry.notes.append(f"{len(row_centres) - 1} rows x {len(col_centres) - 1} columns from rules")
        return geometry

    # whitespace fallback: project ink and split on empty bands
    gray = _cv2().cvtColor(region, _cv2().COLOR_BGR2GRAY)
    binary = _cv2().adaptiveThreshold(
        gray, 255, _cv2().ADAPTIVE_THRESH_GAUSSIAN_C, _cv2().THRESH_BINARY_INV, 25, 12
    )
    rows = binary.sum(axis=1).astype(np.float64)
    row_bounds = _bounds_from_gaps(rows, min_run=max(4, height // 120))
    col_bounds = _gutter_bounds(binary, min_run=max(6, width // 120))
    if len(col_bounds) < 2:
        # no consistent gutter: fall back to the plain projection
        cols = binary.sum(axis=0).astype(np.float64)
        col_bounds = _bounds_from_gaps(cols, min_run=max(6, width // 120))
    if len(row_bounds) >= 2 and len(col_bounds) >= 2:
        row_bounds = _merge_short_bands(row_bounds)
        geometry.row_bounds = row_bounds
        geometry.col_bounds = col_bounds
        geometry.method = "whitespace"
        geometry.confidence = 0.5
        geometry.notes.append("grid inferred from whitespace; borders may be missing")
        return geometry

    raise TableReconstructionError("could not derive a table grid from this region")


def _merge_short_bands(bounds: list[int], min_fraction: float = 0.55) -> list[int]:
    """Absorb row bands that are too short to be a row of their own.

    Ink from a QR code or a stamp inside the table creates extra gaps, which
    splits one row into several slivers. Anything much shorter than the typical
    row is folded into its neighbour.
    """
    if len(bounds) < 4:
        return bounds
    heights = [bounds[index + 1] - bounds[index] for index in range(len(bounds) - 1)]
    heights = [h for h in heights if h > 0]
    if not heights:
        return bounds
    typical = statistics.median(heights)
    if typical <= 0:
        return bounds
    merged = [bounds[0]]
    for index in range(1, len(bounds) - 1):
        if bounds[index] - merged[-1] < typical * min_fraction:
            continue
        merged.append(bounds[index])
    merged.append(bounds[-1])
    return merged


def _gutter_bounds(binary: np.ndarray, min_run: int = 6) -> list[int]:
    """Split columns on gutters that are empty in every row.

    Summing ink down a table and looking for empty columns works for a ruled
    grid, but not for a label/value form: the label is on the left in one row and
    the value on the right, so almost every column holds ink somewhere and the
    projection never drops to zero. What identifies the gutter between the two
    columns is that it is empty in *all* rows, which this measures directly.
    """
    height, width = binary.shape[:2]
    if height == 0 or width == 0:
        return []
    empty_fraction = (binary == 0).mean(axis=0)
    gutters: list[tuple[int, int]] = []
    start: int | None = None
    for x in range(width):
        if empty_fraction[x] >= 0.97:
            if start is None:
                start = x
        elif start is not None:
            gutters.append((start, x))
            start = None
    if start is not None:
        gutters.append((start, width))

    # keep only interior gutters wide enough to be a real column separator
    centres = [(a + b) // 2 for a, b in gutters if b - a >= min_run and a > 0 and b < width]
    bounds = [0]
    for centre in centres:
        if centre - bounds[-1] >= max(4, width // 200):
            bounds.append(centre)
    bounds.append(width)

    # drop leading/trailing bands that hold no ink at all
    ink = (binary > 0)
    while len(bounds) > 2 and not ink[:, bounds[0] : bounds[1]].any():
        bounds.pop(0)
    while len(bounds) > 2 and not ink[:, bounds[-2] : bounds[-1]].any():
        bounds.pop()
    if len(bounds) > 8:
        return []
    return bounds


def _bounds_from_gaps(projection: np.ndarray, min_run: int = 4) -> list[int]:
    """Split a projection into content bands separated by empty gaps."""
    if projection.size == 0:
        return []
    threshold = max(1.0, float(projection.max()) * 0.02)
    bounds: list[int] = []
    in_content = False
    start = 0
    gap = 0
    for index, value in enumerate(projection):
        if value > threshold:
            if not in_content:
                start = index
                in_content = True
            gap = 0
        elif in_content:
            gap += 1
            if gap >= min_run:
                bounds.append(start)
                bounds.append(index - gap + 1)
                in_content = False
    if in_content:
        bounds.append(start)
        bounds.append(len(projection) - 1)
    return bounds


def graphic_mask(crop: np.ndarray) -> np.ndarray:
    """Mask the QR codes, stamps and logos that sit inside a table region.

    Measured on a real certificate: dilating the ink turns a text line into a
    wide, short block (aspect ratio 2.7–10) while a QR code becomes a nearly
    square, dense one (aspect 0.9, fill 0.80). That difference is what separates
    them — fill alone does not, because a dilated text line is dense too. The
    mask is used to clean the projections and to blank the graphics before OCR,
    which is otherwise read as text and emitted as nonsense.
    """
    cv2 = _cv2()
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 8:
        return np.zeros((0, 0), np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )
    height, width = binary.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    dilated = cv2.dilate(binary, np.ones((7, 7), np.uint8), iterations=2)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(dilated)
    crop_area = float(height * width)
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if w < 40 or h < 40:
            continue
        if float(area) < crop_area * 0.002:
            continue
        fill = area / float(w * h)
        aspect = w / float(h)
        if aspect <= 1.6 and fill >= 0.6:
            mask[y : y + h, x : x + w] = 255
        elif aspect <= 3.0 and fill >= 0.82:
            mask[y : y + h, x : x + w] = 255
    return mask


def looks_like_text(crop: np.ndarray) -> bool:
    """Reject QR codes, logos and stamps before OCR reads them as text.

    A certificate photo puts a QR code and a stamp *inside* the table region. OCR
    on those crops returns confident nonsense ("Po PT et Bory em eae AT"), which
    then appears in the reconstructed table as if it were content. Text has a low
    ink ratio and no single solid block; both graphics fail one of those tests.
    """
    cv2 = _cv2()
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 4:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )
    ink = float((binary > 0).mean())
    if ink < 0.003 or ink > 0.32:
        return False
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    crop_area = float(crop.shape[0] * crop.shape[1])
    for contour in contours:
        _, _, width, height = cv2.boundingRect(contour)
        box_area = float(width * height)
        if box_area <= crop_area * 0.5:
            continue
        fill = float(cv2.contourArea(contour)) / box_area if box_area else 0.0
        if fill > 0.45:
            return False
    return True


def _border_present(mask: np.ndarray, orientation: str, position: int, span: tuple[int, int], tolerance: int = 3) -> bool:
    """Check whether a border segment exists along ``position``."""
    lo, hi = span
    # A horizontal rule spans columns (axis 1) and a vertical rule spans rows
    # (axis 0). Clamping to the wrong axis silently rejected every wide span,
    # which made real borders look missing and produced phantom merged cells.
    limit = mask.shape[1] if orientation == "horizontal" else mask.shape[0]
    lo = max(0, lo)
    hi = min(limit, hi)
    if hi <= lo:
        return False
    if orientation == "horizontal":
        window = mask[max(0, position - tolerance) : position + tolerance + 1, lo:hi]
    else:
        window = mask[lo:hi, max(0, position - tolerance) : position + tolerance + 1]
    if window.size == 0:
        return False
    # Coverage asks whether the rule crosses the span, not how thick it is:
    # measuring raw ink density made a 3px rule in a 7px window score 0.43 and
    # every real border looked missing.
    if orientation == "horizontal":
        coverage = float((window > 0).any(axis=0).mean())
    else:
        coverage = float((window > 0).any(axis=1).mean())
    return bool(coverage > 0.45)


def reconstruct_table(
    image: np.ndarray,
    region: BBox,
    ocr_chain: Any,
    settings: Settings,
    mm_per_px: float | None = None,
) -> tuple[TableContent, list[TextLine]]:
    """Reconstruct a table: grid, merged cells, borders and cell text."""
    crop = crop_box(image, region)
    if crop.size == 0:
        raise TableReconstructionError("empty table region")
    # QR codes and stamps inside the table are graphics, not cells: blanking them
    # keeps them out of the grid and out of the OCR text
    graphics = graphic_mask(crop)
    if graphics.size and int(graphics.sum()) > 0:
        cleaned = crop.copy()
        cleaned[graphics > 0] = 255
    else:
        cleaned = crop
    geometry = analyse_table_geometry(cleaned)
    horizontal, vertical = _line_masks(cleaned)

    row_bounds = geometry.row_bounds
    col_bounds = geometry.col_bounds
    rows = len(row_bounds) - 1
    cols = len(col_bounds) - 1
    if rows < 1 or cols < 1:
        raise TableReconstructionError("table grid is degenerate")

    cells: list[TableCell] = []
    all_lines: list[TextLine] = []
    grid: list[list[TableCell | None]] = [[None] * cols for _ in range(rows)]

    for row in range(rows):
        for col in range(cols):
            y1, y2 = row_bounds[row], row_bounds[row + 1]
            x1, x2 = col_bounds[col], col_bounds[col + 1]
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            # The grid bounds sit on the rule centres, so inset the crop: a cell
            # framed by border pixels makes Tesseract return nothing at all for
            # short cell values.
            inset = 3
            ix1, iy1, ix2, iy2 = x1 + inset, y1 + inset, x2 - inset, y2 - inset
            if ix2 - ix1 < 4 or iy2 - iy1 < 4:
                ix1, iy1, ix2, iy2 = x1, y1, x2, y2
            cell_box = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
            absolute = BBox(
                x1=region.x1 + x1, y1=region.y1 + y1, x2=region.x1 + x2, y2=region.y1 + y2
            )
            cell_crop = cleaned[iy1:iy2, ix1:ix2]
            if not looks_like_text(cell_crop):
                # a QR code, a logo or a stamp: record the cell but do not let OCR
                # invent text for it
                cells.append(
                    TableCell(
                        row=row,
                        col=col,
                        text="",
                        bbox=absolute,
                        confidence=None,
                        # text_missing is the honest flag: the cell exists and
                        # holds a graphic, not words
                        flags=["text_missing"],
                    )
                )
                grid[row][col] = cells[-1]
                continue
            result = ocr_chain.recognize(cell_crop, offset=(absolute.x1, absolute.y1))
            alignment = infer_alignment([line.bbox for line in result.lines], absolute) if result.lines else None
            cell = TableCell(
                row=row,
                col=col,
                text=result.text.replace("\n", " ").strip(),
                bbox=absolute,
                alignment=alignment,
                confidence=result.confidence,
                flags=list(result.flags),
            )
            for line in result.lines:
                all_lines.append(
                    TextLine(text=line.text, bbox=line.bbox, confidence=line.confidence, source=result.provider)
                )
            cells.append(cell)
            grid[row][col] = cell

    # merged cells: a missing internal border between two cells means one cell
    merged_notes: list[str] = []
    if geometry.method == "rules":
        for row in range(rows):
            for col in range(cols - 1):
                border_x = col_bounds[col + 1]
                if not _border_present(vertical, "vertical", border_x, (row_bounds[row], row_bounds[row + 1])):
                    left, right = grid[row][col], grid[row][col + 1]
                    if left and right:
                        left.col_span += right.col_span
                        left.text = (left.text + " " + right.text).strip()
                        cells = [c for c in cells if c is not right]
                        grid[row][col + 1] = None
                        merged_notes.append(f"merged horizontal cells at row {row}, col {col}")
        for col in range(cols):
            for row in range(rows - 1):
                border_y = row_bounds[row + 1]
                if not _border_present(horizontal, "horizontal", border_y, (col_bounds[col], col_bounds[col + 1])):
                    top, bottom = grid[row][col], grid[row + 1][col]
                    if top and bottom:
                        top.row_span += bottom.row_span
                        top.text = (top.text + " " + bottom.text).strip()
                        cells = [c for c in cells if c is not bottom]
                        grid[row + 1][col] = None
                        merged_notes.append(f"merged vertical cells at row {row}, col {col}")

    # header row: first row whose text is heavier than the body average
    header_rows = 0
    if rows > 1:
        first_row = [c for c in cells if c.row == 0]
        body = [c for c in cells if c.row > 0]
        if first_row and body:
            first_mean = sum(len(c.text) for c in first_row) / len(first_row)
            body_mean = sum(len(c.text) for c in body) / len(body)
            if first_mean >= body_mean * 0.8:
                header_rows = 1
        for cell in first_row:
            cell.is_header = bool(header_rows)

    # TableContent stores millimetres, so the measured pixel bounds must be
    # converted with the page scale. Without a scale the widths stay empty and
    # the renderer derives them from the element geometry.
    scale_mm = float(mm_per_px) if mm_per_px and mm_per_px > 0 else 0.0
    column_widths = (
        [float(col_bounds[i + 1] - col_bounds[i]) * scale_mm for i in range(cols)] if scale_mm else []
    )
    row_heights = (
        [float(row_bounds[i + 1] - row_bounds[i]) * scale_mm for i in range(rows)] if scale_mm else []
    )

    content = TableContent(
        rows=rows,
        cols=cols,
        cells=cells,
        column_widths_mm=column_widths,
        row_heights_mm=row_heights,
        border_width_mm=geometry.border_width,
        has_outer_border=geometry.method == "rules",
        header_rows=header_rows,
        detection_method=geometry.method,
        structure_confidence=geometry.confidence,
        notes=geometry.notes + merged_notes,
    )
    return content, all_lines


def table_element_flags(content: TableContent) -> list[ReviewFlag]:
    flags: list[ReviewFlag] = []
    if content.detection_method == "whitespace":
        flags.append(ReviewFlag.TABLE_UNCERTAIN)
    if content.structure_confidence is not None and content.structure_confidence < 0.6:
        flags.append(ReviewFlag.TABLE_UNCERTAIN)
    if any(cell.flags for cell in content.cells):
        flags.append(ReviewFlag.OCR_UNCERTAIN)
    if content.rows * content.cols > 0 and not content.cells:
        flags.append(ReviewFlag.TABLE_UNCERTAIN)
    return list(dict.fromkeys(flags))


def table_to_markdown(content: TableContent) -> str:
    """Render a table as Markdown (used by the API preview and the CLI)."""
    grid: list[list[str]] = [["" for _ in range(content.cols)] for _ in range(content.rows)]
    for cell in content.cells:
        if 0 <= cell.row < content.rows and 0 <= cell.col < content.cols:
            grid[cell.row][cell.col] = cell.text.replace("|", "\\|")
    if not grid:
        return ""
    lines = ["| " + " | ".join(row) + " |" for row in grid]
    separator = "|" + "---|" * content.cols
    return "\n".join([lines[0], separator] + lines[1:])
