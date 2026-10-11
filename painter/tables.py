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
    cols = binary.sum(axis=0).astype(np.float64)
    row_bounds = _bounds_from_gaps(rows, min_run=max(4, height // 120))
    col_bounds = _bounds_from_gaps(cols, min_run=max(6, width // 120))
    if len(row_bounds) >= 2 and len(col_bounds) >= 2:
        geometry.row_bounds = row_bounds
        geometry.col_bounds = col_bounds
        geometry.method = "whitespace"
        geometry.confidence = 0.5
        geometry.notes.append("grid inferred from whitespace; borders may be missing")
        return geometry

    raise TableReconstructionError("could not derive a table grid from this region")


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
) -> tuple[TableContent, list[TextLine]]:
    """Reconstruct a table: grid, merged cells, borders and cell text."""
    crop = crop_box(image, region)
    if crop.size == 0:
        raise TableReconstructionError("empty table region")
    geometry = analyse_table_geometry(crop)
    horizontal, vertical = _line_masks(crop)

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
            cell_crop = crop[iy1:iy2, ix1:ix2]
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

    mm_x = image.shape[1] and 0.0
    column_widths = [float(col_bounds[i + 1] - col_bounds[i]) for i in range(cols)]
    row_heights = [float(row_bounds[i + 1] - row_bounds[i]) for i in range(rows)]

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
