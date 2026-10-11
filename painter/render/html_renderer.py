"""HTML/CSS renderer.

The renderer is page based: every page becomes a fixed-size ``.page`` box whose
dimensions come from the Document IR (millimetres), and every element is
positioned with explicit millimetre offsets. Nothing relies on browser defaults
for critical dimensions, so the HTML is also the input for the PDF renderer and
for the rasterised comparison in the verification stage.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..geometry import mm_to_pt
from ..schemas import (
    Alignment,
    DocumentIR,
    Element,
    ElementType,
    PageIR,
    TableContent,
    Typography,
)

DEFAULT_FONT_STACK = "'DejaVu Sans', 'Liberation Sans', Arial, sans-serif"


@dataclass
class HtmlRenderOptions:
    """Options controlling the produced markup."""

    asset_url_prefix: str | None = None
    embed_assets: bool = False
    include_element_ids: bool = True
    include_debug_boxes: bool = False
    page_break_between_pages: bool = True
    title: str = "Painter reconstruction"
    extra_css: str = ""
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Global millimetre offset applied to every element. Used by the bounded
    #: correction loop in :mod:`painter.verify`.
    global_offset_mm: tuple[float, float] = (0.0, 0.0)
    #: Global multiplier applied to every font size (correction loop).
    font_scale: float = 1.0


def _escape(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _asset_href(path: str | None, options: HtmlRenderOptions) -> str:
    if not path:
        return ""
    if path.startswith(("http://", "https://", "data:")):
        return path
    if options.asset_url_prefix:
        return f"{options.asset_url_prefix.rstrip('/')}/{path.lstrip('/')}"
    return path


def _page_body_size(page: PageIR) -> float:
    """The page's dominant body text size, used when an element has none."""
    sizes = sorted(
        float(element.typography.font_size_pt)
        for element in page.elements
        if element.typography and element.typography.font_size_pt
    )
    if not sizes:
        return 11.0
    return sizes[len(sizes) // 2]


def _typography_css(
    typography: Typography | None,
    overrides: dict[str, Any] | None = None,
    font_scale: float = 1.0,
    default_size_pt: float = 11.0,
) -> str:
    overrides = overrides or {}
    declarations: dict[str, str] = {}
    size_pt = (
        overrides.get("font_size_pt")
        or (typography.font_size_pt if typography else None)
        or default_size_pt
    )
    declarations["font-size"] = f"{round(float(size_pt) * max(0.5, min(2.0, font_scale)), 2)}pt"
    family = overrides.get("font_family") or (typography.font_family if typography else None)
    declarations["font-family"] = f"'{family}', {DEFAULT_FONT_STACK}" if family else DEFAULT_FONT_STACK
    weight = overrides.get("font_weight") or (typography.font_weight if typography else None)
    declarations["font-weight"] = str(int(weight)) if weight else "400"
    italic = overrides.get("italic")
    if italic is None:
        italic = typography.italic if typography else False
    declarations["font-style"] = "italic" if italic else "normal"
    color = overrides.get("color") or (typography.color if typography else None) or "#000000"
    declarations["color"] = str(color)
    line_height = overrides.get("line_height_pt") or (typography.line_height_pt if typography else None)
    declarations["line-height"] = f"{round(float(line_height), 2)}pt" if line_height else "1.2"
    if typography and typography.letter_spacing_em:
        declarations["letter-spacing"] = f"{typography.letter_spacing_em}em"
    if typography and typography.underline:
        declarations["text-decoration"] = "underline"
    return "; ".join(f"{key}: {value}" for key, value in declarations.items())


def _alignment_css(alignment: Alignment | None) -> str:
    if alignment is Alignment.CENTER:
        return "center"
    if alignment is Alignment.RIGHT:
        return "right"
    if alignment is Alignment.JUSTIFY:
        return "justify"
    return "left"


def _position_css(
    page: PageIR,
    element: Element,
    overrides: dict[str, Any] | None = None,
    offset_mm: tuple[float, float] = (0.0, 0.0),
) -> str:
    overrides = overrides or {}
    box = element.bbox
    mm_x = page.mm_per_px_x
    mm_y = page.mm_per_px_y
    left = overrides.get("x_mm")
    top = overrides.get("y_mm")
    width = overrides.get("width_mm")
    height = overrides.get("height_mm")
    left = (box.x1 * mm_x if left is None else float(left)) + offset_mm[0]
    top = (box.y1 * mm_y if top is None else float(top)) + offset_mm[1]
    width = box.width * mm_x if width is None else float(width)
    height = box.height * mm_y if height is None else float(height)
    return (
        f"position:absolute;left:{round(left, 3)}mm;top:{round(top, 3)}mm;"
        f"width:{round(max(width, 0.4), 3)}mm;height:{round(max(height, 0.4), 3)}mm;"
    )


def _render_text_lines(
    element: Element,
    page: PageIR,
    options: HtmlRenderOptions,
    overrides: dict[str, Any],
    data_id: str,
) -> str | None:
    """Place every measured text line at its own position.

    A single block with one line-height drifts on dense pages: the paragraph's
    lines end up spread over a taller box than the original and collide with the
    element below. OCR already gives us each line's box, so use it.
    """
    lines = [line for line in (element.lines or []) if (line.text or "").strip()]
    if not lines:
        return None
    mm_x = page.mm_per_px_x
    mm_y = page.mm_per_px_y
    x1 = min(line.bbox.x1 for line in lines)
    y1 = min(line.bbox.y1 for line in lines)
    x2 = max(line.bbox.x2 for line in lines)
    y2 = max(line.bbox.y2 for line in lines)
    offset_x, offset_y = options.global_offset_mm
    typography_css = _typography_css(element.typography, overrides, options.font_scale)
    parts: list[str] = []
    for line in lines:
        left = (line.bbox.x1 - x1) * mm_x
        top = (line.bbox.y1 - y1) * mm_y
        width = max(line.bbox.width * mm_x, 0.4)
        parts.append(
            f'<div style="position:absolute;left:{round(left, 3)}mm;top:{round(top, 3)}mm;'
            f'width:{round(width, 3)}mm;{typography_css};line-height:1.05;'
            f'white-space:nowrap;">{_escape(line.text)}</div>'
        )
    return (
        f'<div class="el el-text"{data_id} style="position:absolute;'
        f'left:{round(x1 * mm_x + offset_x, 3)}mm;top:{round(y1 * mm_y + offset_y, 3)}mm;'
        f'width:{round(max((x2 - x1) * mm_x, 0.4), 3)}mm;'
        f'height:{round(max((y2 - y1) * mm_y, 0.4), 3)}mm;">{"".join(parts)}</div>'
    )


def _render_text(element: Element, page: PageIR, options: HtmlRenderOptions) -> str:
    overrides = options.overrides.get(element.id, {})
    text = overrides.get("text", element.text) or ""
    data_id = f' data-element-id="{_escape(element.id)}"' if options.include_element_ids else ""
    if "text" not in overrides:
        placed = _render_text_lines(element, page, options, overrides, data_id)
        if placed is not None:
            return placed
    css = _position_css(page, element, overrides, options.global_offset_mm) + _typography_css(
        element.typography, overrides, options.font_scale
    )
    align = overrides.get("alignment") or (element.alignment.value if element.alignment else "left")
    padding = overrides.get("padding_mm", 0)
    return (
        f'<div class="el el-text"{data_id} style="{css};text-align:{align};'
        f'padding:{padding}mm;overflow:hidden;white-space:pre-wrap;">{_escape(text)}</div>'
    )


def _render_image(element: Element, page: PageIR, options: HtmlRenderOptions) -> str:
    overrides = options.overrides.get(element.id, {})
    content = element.image
    source = overrides.get("asset_path") or (content.asset_path if content else None)
    href = _asset_href(source, options)
    css = _position_css(page, element, overrides, options.global_offset_mm)
    data_id = f' data-element-id="{_escape(element.id)}"' if options.include_element_ids else ""
    if not href:
        return (
            f'<div class="el el-image-missing"{data_id} style="{css}border:1px dashed #b00;'
            f'display:flex;align-items:center;justify-content:center;font-size:6pt;color:#b00;">'
            f"no asset</div>"
        )
    return (
        f'<img class="el el-image"{data_id} style="{css}object-fit:contain;" '
        f'src="{_escape(href)}" alt="{_escape(element.type.value)}">'
    )


def _render_table(element: Element, page: PageIR, options: HtmlRenderOptions) -> str:  # noqa: C901
    overrides = options.overrides.get(element.id, {})
    table: TableContent | None = element.table
    css = _position_css(page, element, overrides, options.global_offset_mm)
    data_id = f' data-element-id="{_escape(element.id)}"' if options.include_element_ids else ""
    if table is None or not table.cells:
        return f'<div class="el el-table"{data_id} style="{css}border:1px dashed #999;"></div>'

    mm_x = page.mm_per_px_x
    element_width_mm = max((element.bbox.width or 1.0) * mm_x, 0.1)
    column_widths = list(table.column_widths_mm or [])
    if len(column_widths) != table.cols or any(w <= 0 for w in column_widths):
        column_widths = [element_width_mm / max(table.cols, 1) for _ in range(table.cols)]
    else:
        total = sum(column_widths)
        if total > element_width_mm * 1.25:
            # widths were stored in pixels by an older version: rescale to fit
            factor = element_width_mm / total
            column_widths = [w * factor for w in column_widths]
    colgroup = "".join(f'<col style="width:{round(float(w), 3)}mm">' for w in column_widths)

    grid: dict[tuple[int, int], Any] = {}
    for cell in table.cells:
        grid[(cell.row, cell.col)] = cell
    border_style = "1px solid #333" if table.has_outer_border else "none"
    row_heights = list(table.row_heights_mm or [])
    body_size = _page_body_size(page)
    rows_html: list[str] = []
    for row in range(table.rows):
        # keep a row inside the height it had in the original: an oversized cell
        # font would push the table over the content below it
        row_height_mm = row_heights[row] if row < len(row_heights) else None
        max_pt = (row_height_mm * (72.0 / 25.4) / 1.25) if row_height_mm else None
        row_style = f' style="height:{round(row_height_mm, 3)}mm"' if row_height_mm else ""
        cells_html: list[str] = []
        for col in range(table.cols):
            cell = grid.get((row, col))
            if cell is None:
                continue
            cell_css = [f"border:{border_style}", "padding:0.2mm", "vertical-align:top", "overflow:hidden", "line-height:1.15"]
            if cell.is_header:
                cell_css.append("font-weight:600")
            if cell.background_color:
                cell_css.append(f"background-color:{cell.background_color}")
            if cell.alignment:
                cell_css.append(f"text-align:{cell.alignment.value}")
            cell_size = (cell.typography.font_size_pt if cell.typography else None) or body_size
            if max_pt:
                cell_size = min(float(cell_size), max_pt)
            cell_css.append(
                _typography_css(
                    cell.typography,
                    {"font_size_pt": round(cell_size, 2)},
                    options.font_scale,
                    body_size,
                )
            )
            span = ""
            if cell.col_span > 1:
                span += f' colspan="{cell.col_span}"'
            if cell.row_span > 1:
                span += f' rowspan="{cell.row_span}"'
            cells_html.append(f'<td style="{";".join(cell_css)}">{_escape(cell.text)}</td>')
        rows_html.append(f"<tr{row_style}>{''.join(cells_html)}</tr>")
    return (
        f'<div class="el el-table"{data_id} style="{css}">'
        f'<table style="border-collapse:collapse;width:100%;table-layout:fixed;">'
        f"<colgroup>{colgroup}</colgroup>{''.join(rows_html)}</table></div>"
    )


def _render_shape(element: Element, page: PageIR, options: HtmlRenderOptions) -> str:
    overrides = options.overrides.get(element.id, {})
    shape = element.shape
    css = _position_css(page, element, overrides, options.global_offset_mm)
    data_id = f' data-element-id="{_escape(element.id)}"' if options.include_element_ids else ""
    if element.type is ElementType.LINE or (shape and shape.kind == "line"):
        orientation = (shape.orientation if shape else None) or (
            "horizontal" if element.bbox.width >= element.bbox.height else "vertical"
        )
        stroke = (shape.stroke_color if shape and shape.stroke_color else "#333333")
        thickness = mm_to_pt(shape.stroke_width_mm) if shape and shape.stroke_width_mm else 0.75
        if orientation == "horizontal":
            return (
                f'<div class="el el-line"{data_id} style="{css}border-top:{round(thickness, 2)}pt solid '
                f'{stroke};height:0;"></div>'
            )
        return (
            f'<div class="el el-line"{data_id} style="{css}border-left:{round(thickness, 2)}pt solid '
            f'{stroke};width:0;"></div>'
        )
    if element.type is ElementType.CHECKBOX:
        checked = bool(shape.checked) if shape else False
        return (
            f'<div class="el el-checkbox"{data_id} style="{css}border:0.75pt solid #333;'
            f'display:flex;align-items:center;justify-content:center;font-size:8pt;">'
            f"{'&#10003;' if checked else ''}</div>"
        )
    radius = "50%" if element.type is ElementType.CIRCLE else (
        f"{shape.corner_radius_mm}mm" if shape and shape.corner_radius_mm else "0"
    )
    stroke = (shape.stroke_color if shape and shape.stroke_color else "#333333")
    thickness = mm_to_pt(shape.stroke_width_mm) if shape and shape.stroke_width_mm else 0.75
    fill = (shape.fill_color if shape and shape.fill_color else "transparent")
    return (
        f'<div class="el el-shape"{data_id} style="{css}border:{round(thickness, 2)}pt solid {stroke};'
        f'border-radius:{radius};background:{fill};"></div>'
    )


def render_element(element: Element, page: PageIR, options: HtmlRenderOptions) -> str:
    """Render one element to HTML."""
    if element.type in {
        ElementType.HEADING,
        ElementType.SUBHEADING,
        ElementType.PARAGRAPH,
        ElementType.TEXT_BLOCK,
        ElementType.LIST,
        ElementType.FIGURE_CAPTION,
        ElementType.TABLE_CAPTION,
        ElementType.HEADER,
        ElementType.FOOTER,
        ElementType.PAGE_NUMBER,
        ElementType.FORMULA,
        ElementType.FORM_FIELD,
    }:
        return _render_text(element, page, options)
    if element.type in {ElementType.IMAGE, ElementType.LOGO, ElementType.SIGNATURE, ElementType.STAMP}:
        return _render_image(element, page, options)
    if element.type is ElementType.TABLE:
        return _render_table(element, page, options)
    if element.type in {
        ElementType.SHAPE,
        ElementType.LINE,
        ElementType.RECTANGLE,
        ElementType.CIRCLE,
        ElementType.CHECKBOX,
    }:
        return _render_shape(element, page, options)
    return ""


def page_css(page: PageIR) -> str:
    return (
        f"@page {{ size: {page.page_format.width_mm:.2f}mm "
        f"{page.page_format.height_mm:.2f}mm; margin: 0; }}\n"
        f".page {{ width: {page.page_format.width_mm:.2f}mm; "
        f"height: {page.page_format.height_mm:.2f}mm; }}\n"
    )


def render_page(page: PageIR, options: HtmlRenderOptions | None = None) -> str:
    """Render one page to a positioned HTML fragment."""
    options = options or HtmlRenderOptions()
    ordered = sorted(page.elements, key=lambda e: (e.z, e.order_index if e.order_index is not None else 0))
    body = "\n".join(filter(None, (render_element(element, page, options) for element in ordered)))
    debug = ""
    if options.include_debug_boxes:
        debug = "".join(
            f'<div style="{_position_css(page, element, None, options.global_offset_mm)}'
            f'border:0.5pt dashed #f0f;'
            f'pointer-events:none;"></div>'
            for element in ordered
        )
    return (
        f'<div class="page" data-page="{page.index}" '
        f'style="position:relative;overflow:hidden;background:#ffffff;">\n{body}\n{debug}\n</div>'
    )


def render_document(document: DocumentIR, options: HtmlRenderOptions | None = None) -> str:
    """Render a complete, standalone HTML document."""
    options = options or HtmlRenderOptions()
    pages_css = "\n".join(page_css(page) for page in document.pages) if document.pages else (
        "@page { size: 210mm 297mm; margin: 0; }\n.page { width: 210mm; height: 297mm; }\n"
    )
    page_break = (
        ".page + .page { page-break-before: always; break-before: page; }\n"
        if options.page_break_between_pages
        else ""
    )
    body = "\n".join(render_page(page, options) for page in document.pages)
    return f"""<!DOCTYPE html>
<html lang="{_escape(document.metadata.language or 'en')}">
<head>
<meta charset="utf-8">
<title>{_escape(options.title)}</title>
<style>
html, body {{ margin: 0; padding: 0; background: #f2f2f2; }}
.pages {{ display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 12px; }}
.page {{ box-shadow: 0 1px 6px rgba(0,0,0,.25); }}
{pages_css}{page_break}
.el {{ box-sizing: border-box; }}
.el-text {{ overflow-wrap: break-word; }}
.el-image {{ display: block; }}
{options.extra_css}
@media print {{
  html, body {{ background: #ffffff; }}
  .pages {{ padding: 0; gap: 0; }}
  .page {{ box-shadow: none; }}
}}
</style>
</head>
<body>
<div class="pages">
{body}
</div>
</body>
</html>"""


def render_document_for_print(document: DocumentIR, options: HtmlRenderOptions | None = None) -> str:
    """Same markup, tuned for the PDF pass (white background, no shadows)."""
    options = options or HtmlRenderOptions()
    css = "html, body { background: #ffffff; } .pages { padding: 0; gap: 0; } .page { box-shadow: none; }"
    merged = HtmlRenderOptions(**{**options.__dict__, "extra_css": options.extra_css + "\n" + css})
    return render_document(document, merged)
