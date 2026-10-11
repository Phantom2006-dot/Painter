"""DOCX renderer.

The generated Word file contains **editable** content:

* text as native Word paragraphs with real font, size, weight, colour and
  alignment,
* tables as native Word tables (rows, columns, merged cells, header row),
* images/logos/signatures/stamps as pictures, anchored to page coordinates when
  they sit next to text so the layout survives,
* shapes as drawn assets only when Word cannot express them natively — and every
  such element is flagged in the document metadata.

The result never consists of a full-page image of the scan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..errors import RenderingError
from ..geometry import mm_to_pt
from ..logging_utils import get_logger
from ..schemas import (
    Alignment,
    AssetOrigin,
    DocumentIR,
    Element,
    ElementType,
    PageIR,
    ReviewFlag,
)

logger = get_logger("render.docx")

EMU_PER_MM = 36000


@dataclass
class DocxResult:
    path: Path
    paragraphs: int = 0
    tables: int = 0
    pictures: int = 0
    anchored_pictures: int = 0
    shape_fallbacks: int = 0
    editable_characters: int = 0
    warnings: list[str] = field(default_factory=list)


def _alignment_enum(alignment: Alignment | None, docx_module: Any) -> Any:
    WD_ALIGN_PARAGRAPH = docx_module.enum.text.WD_ALIGN_PARAGRAPH
    if alignment is Alignment.CENTER:
        return WD_ALIGN_PARAGRAPH.CENTER
    if alignment is Alignment.RIGHT:
        return WD_ALIGN_PARAGRAPH.RIGHT
    if alignment is Alignment.JUSTIFY:
        return WD_ALIGN_PARAGRAPH.JUSTIFY
    return WD_ALIGN_PARAGRAPH.LEFT


def _rgb(value: str | None, docx_module: Any) -> Any:
    if not value:
        return None
    cleaned = value.lstrip("#")
    if len(cleaned) != 6:
        return None
    try:
        return docx_module.shared.RGBColor.from_string(cleaned.upper())
    except Exception:
        return None


def _anchor_picture(run: Any, x_mm: float, y_mm: float, width_mm: float, height_mm: float, behind_text: bool = True) -> bool:
    """Convert an inline picture inside ``run`` into a page-anchored one."""
    from docx.oxml.ns import qn
    from lxml import etree

    drawing = run._r.find(qn("w:drawing"))
    if drawing is None:
        return False
    inline = drawing.find(qn("wp:inline"))
    if inline is None:
        return False

    anchor = etree.SubElement(drawing, qn("wp:anchor"))
    for key, value in {
        "distT": "0",
        "distB": "0",
        "distL": "0",
        "distR": "0",
        "simplePos": "0",
        "relativeHeight": "2",
        "behindDoc": "1" if behind_text else "0",
        "locked": "0",
        "layoutInCell": "1",
        "allowOverlap": "1",
    }.items():
        anchor.set(key, value)

    simple_pos = etree.SubElement(anchor, qn("wp:simplePos"))
    simple_pos.set("x", "0")
    simple_pos.set("y", "0")

    position_h = etree.SubElement(anchor, qn("wp:positionH"))
    position_h.set("relativeFrom", "page")
    offset_h = etree.SubElement(position_h, qn("wp:posOffset"))
    offset_h.text = str(int(x_mm * EMU_PER_MM))

    position_v = etree.SubElement(anchor, qn("wp:positionV"))
    position_v.set("relativeFrom", "page")
    offset_v = etree.SubElement(position_v, qn("wp:posOffset"))
    offset_v.text = str(int(y_mm * EMU_PER_MM))

    extent = inline.find(qn("wp:extent"))
    if extent is not None:
        extent.set("cx", str(int(width_mm * EMU_PER_MM)))
        extent.set("cy", str(int(height_mm * EMU_PER_MM)))
        anchor.append(extent)
    effect = inline.find(qn("wp:effectExtent"))
    if effect is not None:
        anchor.append(effect)

    etree.SubElement(anchor, qn("wp:wrapNone"))

    doc_pr = inline.find(qn("wp:docPr"))
    if doc_pr is not None:
        anchor.append(doc_pr)
    graphic = inline.find(qn("a:graphic"))
    if graphic is not None:
        anchor.append(graphic)

    drawing.remove(inline)
    return True


def _shape_png(element: Element, page: PageIR, output_dir: Path) -> Path | None:
    """Draw a shape into a PNG so Word can place it (documented fallback)."""
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    width_px = max(4, int(element.bbox.width))
    height_px = max(4, int(element.bbox.height))
    canvas = np.full((height_px, width_px, 4), 0, dtype=np.uint8)
    shape = element.shape
    color = (40, 40, 40, 255)
    thickness = 2
    if element.type is ElementType.CHECKBOX:
        cv2.rectangle(canvas, (1, 1), (width_px - 2, height_px - 2), color, thickness)
    elif element.type is ElementType.CIRCLE:
        cv2.ellipse(
            canvas,
            (width_px // 2, height_px // 2),
            (max(1, width_px // 2 - 2), max(1, height_px // 2 - 2)),
            0,
            0,
            360,
            color,
            thickness,
        )
    else:
        cv2.rectangle(canvas, (1, 1), (width_px - 2, height_px - 2), color, thickness)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{element.id}_shape.png"
    if not cv2.imwrite(str(path), canvas):
        return None
    return path


def render_docx(
    document: DocumentIR,
    output_path: Path,
    asset_base: Path | None = None,
    include_shapes: bool = True,
    page_offset_mm: tuple[float, float] = (0.0, 0.0),
) -> DocxResult:
    """Render the Document IR to an editable DOCX."""
    try:
        import docx as docx_module
        from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: F401
        from docx.shared import Mm, Pt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RenderingError(f"python-docx is required for DOCX export: {exc}") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = DocxResult(path=output_path)
    doc = docx_module.Document()

    style = doc.styles["Normal"]
    style.font.name = "DejaVu Sans"
    style.font.size = Pt(11)

    # remove the default empty paragraph later, after content exists
    for page_index, page in enumerate(document.pages):
        if page_index > 0:
            doc.add_page_break()
        _render_page(page, doc, docx_module, result, asset_base, page_offset_mm)

    doc.core_properties.title = document.metadata.title or "Painter reconstruction"
    doc.core_properties.comments = (
        "Reconstructed by Painter from a scan. Text and tables are editable. "
        "Positions are approximated for flow content and exact for anchored images."
    )
    doc.save(str(output_path))
    result.path = output_path
    return result


def _render_page(
    page: PageIR,
    doc: Any,
    docx_module: Any,
    result: DocxResult,
    asset_base: Path | None,
    page_offset_mm: tuple[float, float] = (0.0, 0.0),
) -> None:
    from docx.enum.section import WD_SECTION
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.shared import Mm, Pt

    section = doc.sections[-1]
    section.page_width = Mm(page.page_format.width_mm)
    section.page_height = Mm(page.page_format.height_mm)
    section.top_margin = Mm(max(page.margins_mm.top + page_offset_mm[1], 4.0))
    section.bottom_margin = Mm(max(page.margins_mm.bottom, 8.0))
    section.left_margin = Mm(max(page.margins_mm.left + page_offset_mm[0], 4.0))
    section.right_margin = Mm(max(page.margins_mm.right, 10.0))

    ordered = _ordered_flow(page)
    previous_bottom_px: float | None = None
    mm_y = page.mm_per_px_y
    mm_x = page.mm_per_px_x

    for element in ordered:
        if element.type is ElementType.TABLE:
            _add_table(doc, element, page, docx_module, result)
            previous_bottom_px = element.bbox.y2
            continue
        if element.type in {
            ElementType.IMAGE,
            ElementType.LOGO,
            ElementType.SIGNATURE,
            ElementType.STAMP,
        }:
            _add_picture(doc, element, page, result, asset_base, docx_module)
            previous_bottom_px = element.bbox.y2
            continue
        if element.type in {
            ElementType.SHAPE,
            ElementType.LINE,
            ElementType.RECTANGLE,
            ElementType.CIRCLE,
            ElementType.CHECKBOX,
        }:
            if _add_shape(doc, element, page, result, asset_base, docx_module):
                result.shape_fallbacks += 1
            continue
        if not (element.text or "").strip() and element.type not in {
            ElementType.PAGE_NUMBER,
            ElementType.HEADER,
            ElementType.FOOTER,
        }:
            continue

        paragraph = doc.add_paragraph()
        paragraph.alignment = _alignment_enum(element.alignment, docx_module)
        typography = element.typography
        gap_pt = 0.0
        if previous_bottom_px is not None:
            gap_mm = max(0.0, (element.bbox.y1 - previous_bottom_px) * mm_y)
            gap_pt = mm_to_pt(gap_mm)
        paragraph.paragraph_format.space_before = Pt(round(max(0.0, min(gap_pt, 120.0)), 2))
        paragraph.paragraph_format.space_after = Pt(0)
        if typography and typography.line_spacing_ratio:
            paragraph.paragraph_format.line_spacing = max(0.85, min(3.0, typography.line_spacing_ratio))
        indent_mm = element.bbox.x1 * mm_x - page.margins_mm.left
        if indent_mm > 1.0:
            paragraph.paragraph_format.left_indent = Mm(round(indent_mm, 2))

        run = paragraph.add_run((element.text or "").strip())
        size_pt = (typography.font_size_pt if typography and typography.font_size_pt else 11.0)
        run.font.size = Pt(round(size_pt, 2))
        run.font.bold = bool(typography and typography.font_weight and typography.font_weight >= 600)
        run.font.italic = bool(typography and typography.italic)
        if typography and typography.font_family:
            run.font.name = typography.font_family
        color = _rgb(typography.color if typography else None, docx_module)
        if color is not None:
            run.font.color.rgb = color
        if typography and typography.underline:
            run.font.underline = True
        result.paragraphs += 1
        result.editable_characters += len(element.text or "")
        previous_bottom_px = element.bbox.y2


def _ordered_flow(page: PageIR) -> list[Element]:
    return sorted(
        page.elements,
        key=lambda e: (e.order_index if e.order_index is not None else 0, e.bbox.y1, e.bbox.x1),
    )


def _add_table(doc: Any, element: Element, page: PageIR, docx_module: Any, result: DocxResult) -> None:
    from docx.shared import Mm, Pt

    table_content = element.table
    if table_content is None or table_content.rows < 1 or table_content.cols < 1:
        return
    table = doc.add_table(rows=table_content.rows, cols=table_content.cols)
    table.style = "Table Grid"
    table.autofit = False
    grid: dict[tuple[int, int], Any] = {(c.row, c.col): c for c in table_content.cells}
    mm_x = page.mm_per_px_x
    widths = table_content.column_widths_mm or []
    for row in range(table_content.rows):
        for col in range(table_content.cols):
            cell_model = grid.get((row, col))
            if cell_model is None:
                continue
            cell = table.cell(row, col)
            cell.text = cell_model.text
            if col < len(widths) and widths[col] > 0:
                cell.width = Mm(round(float(widths[col]) * mm_x, 2))
            for paragraph in cell.paragraphs:
                paragraph.alignment = _alignment_enum(cell_model.alignment, docx_module)
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                for run in paragraph.runs:
                    size = 10.0
                    if cell_model.typography and cell_model.typography.font_size_pt:
                        size = cell_model.typography.font_size_pt
                    run.font.size = Pt(round(size, 2))
                    run.font.bold = bool(cell_model.is_header)
            if cell_model.col_span > 1 or cell_model.row_span > 1:
                try:
                    target = table.cell(
                        min(row + cell_model.row_span - 1, table_content.rows - 1),
                        min(col + cell_model.col_span - 1, table_content.cols - 1),
                    )
                    cell.merge(target)
                except Exception:  # pragma: no cover - invalid merge geometry
                    pass
    result.tables += 1


def _add_picture(
    doc: Any,
    element: Element,
    page: PageIR,
    result: DocxResult,
    asset_base: Path | None,
    docx_module: Any,
) -> None:
    from docx.shared import Mm

    content = element.image
    if content is None or not content.asset_path:
        return
    path = Path(content.asset_path)
    if not path.is_absolute() and asset_base is not None:
        path = asset_base / content.asset_path
    if not path.is_file():
        result.warnings.append(f"missing asset for {element.id}: {content.asset_path}")
        return

    width_mm = element.bbox.width * page.mm_per_px_x
    height_mm = element.bbox.height * page.mm_per_px_y
    x_mm = element.bbox.x1 * page.mm_per_px_x
    y_mm = element.bbox.y1 * page.mm_per_px_y
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = 0
    paragraph.paragraph_format.space_after = 0
    run = paragraph.add_run()
    try:
        run.add_picture(str(path), width=Mm(round(width_mm, 2)), height=Mm(round(height_mm, 2)))
    except Exception as exc:  # pragma: no cover - unsupported image
        result.warnings.append(f"could not embed {path.name}: {exc}")
        return
    anchored = _anchor_picture(run, x_mm, y_mm, width_mm, height_mm, behind_text=True)
    if anchored:
        result.anchored_pictures += 1
    else:
        result.warnings.append(f"{element.id} placed inline (anchoring unavailable)")
    result.pictures += 1


def _add_shape(
    doc: Any,
    element: Element,
    page: PageIR,
    result: DocxResult,
    asset_base: Path | None,
    docx_module: Any,
) -> bool:
    from docx.shared import Mm

    output_dir = (asset_base or Path(".")) / "shape_assets"
    path = _shape_png(element, page, output_dir)
    if path is None:
        return False
    width_mm = element.bbox.width * page.mm_per_px_x
    height_mm = element.bbox.height * page.mm_per_px_y
    x_mm = element.bbox.x1 * page.mm_per_px_x
    y_mm = element.bbox.y1 * page.mm_per_px_y
    paragraph = doc.add_paragraph()
    run = paragraph.add_run()
    try:
        run.add_picture(str(path), width=Mm(round(max(width_mm, 1.0), 2)), height=Mm(round(max(height_mm, 1.0), 2)))
    except Exception:
        return False
    _anchor_picture(run, x_mm, y_mm, width_mm, height_mm, behind_text=True)
    element.add_flag(ReviewFlag.SHAPE_IMAGE_FALLBACK)
    result.warnings.append(
        f"{element.id} ({element.type.value}) rendered as an image because Word has no "
        "native equivalent for this shape"
    )
    return True
