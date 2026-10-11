"""Regression tests for the reconstruction-quality fixes.

Each test here exists because of a specific defect seen on a real document:

* a text line was detected as a fragment, so OCR only read its middle;
* two whole sentences were never detected, so they vanished;
* a screenshot was forced onto A6 and came out as 4.8pt body text;
* table column widths were stored in pixels but used as millimetres, which blew
  the table up to 750mm on a 148mm page and made the cells look enormous;
* paragraph lines were spread over a tall box instead of keeping the pitch they
  had in the original.
"""

from __future__ import annotations

import re

import cv2
import numpy as np
import pytest

from painter.detection import (
    Detection,
    complete_text_regions,
    recover_missing_text_lines,
)
from painter.geometry import infer_page_format
from painter.render import HtmlRenderOptions, render_document
from painter.schemas import (
    BBox,
    DocumentIR,
    DocumentSource,
    Element,
    ElementType,
    PageFormat,
    PageIR,
    TableCell,
    TableContent,
    TextLine,
    Typography,
)


def text_page(lines: list[str], width: int = 1200, height: int = 400) -> np.ndarray:
    """A white page with evenly spaced black text lines."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    for index, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (60, 80 + index * 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return image


# --------------------------------------------------------------------- fragments
def test_fragment_detection_is_widened_to_the_whole_line():
    image = text_page(["Having attained the University-approved cut-off mark of 65%"])
    # a detection covering only the middle of the line, as the model returned
    fragment = Detection(
        class_name="Text",
        element_type=ElementType.PARAGRAPH,
        confidence=0.32,
        bbox=BBox(x1=420.0, y1=55.0, x2=700.0, y2=85.0),
    )
    completed, warnings = complete_text_regions(image, [fragment])
    box = completed[0].bbox
    assert box.x1 < 200, f"left edge not recovered: {box}"
    assert box.x2 > 900, f"right edge not recovered: {box}"
    assert completed[0].extra["widened_from"] == [420.0, 700.0]
    assert any("part of their line" in warning for warning in warnings)


def test_completion_leaves_unrelated_detections_alone():
    image = text_page(["Short heading"])
    table = Detection(
        class_name="Table",
        element_type=ElementType.TABLE,
        confidence=0.9,
        bbox=BBox(x1=60.0, y1=50.0, x2=500.0, y2=200.0),
    )
    completed, _ = complete_text_regions(image, [table])
    assert completed[0].bbox == table.bbox


# ------------------------------------------------------------------ missed lines
def test_missed_text_line_is_recovered_from_the_ink():
    image = text_page(["Recovered line one", "Recovered line two"])
    # the detector only saw the first line
    detected = Detection(
        class_name="Text",
        element_type=ElementType.PARAGRAPH,
        confidence=0.9,
        bbox=BBox(x1=55.0, y1=50.0, x2=800.0, y2=90.0),
    )
    recovered, warnings = recover_missing_text_lines(image, [detected])
    assert len(recovered) == 2, "the second line should have been recovered"
    added = recovered[-1]
    assert added.source == "cv-recovery"
    assert added.extra.get("recovered") is True
    assert added.bbox.y1 > 90, "the recovered region should sit below the known line"
    assert any("missed by the layout model" in warning for warning in warnings)


def test_recovery_ignores_dense_blocks():
    """A logo or a solid rule is not a text line."""
    image = np.full((400, 1200, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (60, 60), (1100, 120), (0, 0, 0), -1)
    recovered, warnings = recover_missing_text_lines(image, [])
    assert recovered == []
    assert warnings == []


# -------------------------------------------------------------------- page scale
def test_wide_screenshot_gets_a_readable_custom_page():
    """A 2.2 aspect ratio is not paper: the old code forced A6 and 4.8pt text."""
    page_format, dpi, notes = infer_page_format(1417, 647, line_height_px=22.0)
    assert page_format.name == "custom"
    assert page_format.width_mm > 200, "the page must be wide enough to be readable"
    # the implied body text size must be plausible, not 5pt
    body_pt = 22.0 / dpi * 72.0 / 1.2
    assert 8.0 <= body_pt <= 14.0, f"implausible body size: {body_pt:.1f}pt"
    assert any("custom" in note for note in notes)


def test_paper_aspect_still_maps_to_a_real_format():
    """A4 proportions must keep selecting A4."""
    page_format, dpi, _ = infer_page_format(2480, 3508, line_height_px=40.0)
    assert page_format.name.startswith("A")
    assert 200 <= dpi <= 400


# ------------------------------------------------------------------------ tables
def build_table_page(column_widths_mm: list[float], row_heights_mm: list[float]) -> DocumentIR:
    page = PageIR(
        index=0,
        width_px=1000,
        height_px=600,
        page_format=PageFormat(name="A4", width_mm=210.0, height_mm=297.0, estimated=True),
        elements=[
            Element(
                id="el_body",
                type=ElementType.PARAGRAPH,
                bbox=BBox(x1=10, y1=10, x2=400, y2=30),
                text="body text",
                typography=Typography(font_size_pt=10.0, estimated=True),
            ),
            Element(
                id="el_table",
                type=ElementType.TABLE,
                bbox=BBox(x1=50, y1=100, x2=450, y2=200),
                table=TableContent(
                    rows=2,
                    cols=2,
                    cells=[
                        TableCell(row=0, col=0, text="Assessment Information"),
                        TableCell(row=0, col=1, text="Result"),
                        TableCell(row=1, col=0, text="Your CBT Score"),
                        TableCell(row=1, col=1, text="80.6"),
                    ],
                    column_widths_mm=column_widths_mm,
                    row_heights_mm=row_heights_mm,
                    has_outer_border=True,
                ),
            ),
        ],
    )
    return DocumentIR(
        document_id="doc_test",
        job_id="job_test",
        source=DocumentSource(filename="test.png", stored_path="x", sha256="0", size_bytes=0),
        pages=[page],
    )


def test_pixel_column_widths_are_rescaled_to_the_page():
    """750mm of columns on a 210mm page is the bug that produced giant text."""
    document = build_table_page(column_widths_mm=[412.0, 338.0], row_heights_mm=[8.0, 8.0])
    html = render_document(document, HtmlRenderOptions(include_element_ids=True))
    assert "412" not in html.split("<colgroup>")[1].split("</colgroup>")[0]
    widths = [float(value) for value in re.findall(r'<col style="width:([\d.]+)mm"', html)]
    assert len(widths) == 2, f"expected two columns, got {widths}"
    element_mm = (450 - 50) * document.pages[0].mm_per_px_x
    total = sum(widths)
    assert total <= element_mm * 1.25, f"columns total {total}mm for a {element_mm}mm element"
    assert abs(total - element_mm) < 0.5, "columns should exactly fill the element"


def test_cell_font_is_capped_to_the_row_height():
    """A 9pt cell in a 3mm row pushed the table over the text below it."""
    document = build_table_page(column_widths_mm=[], row_heights_mm=[3.0, 3.0])
    html = render_document(document, HtmlRenderOptions())
    table_html = html.split('<div class="el el-table"')[1]
    sizes = [float(value) for value in re.findall(r"font-size:\s*([\d.]+)pt", table_html)]
    assert sizes, "no font sizes in the table"
    cap = 3.0 * (72.0 / 25.4) / 1.25
    assert max(sizes) <= cap + 0.01, f"cell font {max(sizes)}pt exceeds the row cap {cap:.2f}pt"
    assert max(sizes) > 3.0, "the cap should not collapse the text either"


def test_table_cells_inherit_the_page_body_size_when_unset():
    document = build_table_page(column_widths_mm=[], row_heights_mm=[])
    html = render_document(document, HtmlRenderOptions())
    table_html = html.split('<div class="el el-table"')[1]
    sizes = {float(value) for value in re.findall(r"font-size:\s*([\d.]+)pt", table_html)}
    # the page's only typography is the 10pt body element
    assert sizes == {10.0}, f"cells should inherit the 10pt body size, got {sizes}"


# --------------------------------------------------------------- line placement
def test_each_ocr_line_is_placed_at_its_measured_position():
    """A block with one line-height drifted into the element below it."""
    page = PageIR(
        index=0,
        width_px=1000,
        height_px=600,
        page_format=PageFormat(name="A4", width_mm=210.0, height_mm=297.0, estimated=True),
        elements=[
            Element(
                id="el_multi",
                type=ElementType.PARAGRAPH,
                bbox=BBox(x1=100, y1=100, x2=600, y2=160),
                text="first line of the paragraph\nsecond line of the paragraph",
                typography=Typography(font_size_pt=11.0, estimated=True),
                lines=[
                    TextLine(text="first line of the paragraph", bbox=BBox(x1=100, y1=100, x2=500, y2=116)),
                    TextLine(text="second line of the paragraph", bbox=BBox(x1=100, y1=126, x2=520, y2=142)),
                ],
            )
        ],
    )
    document = DocumentIR(
        document_id="doc_lines",
        job_id="job_lines",
        source=DocumentSource(filename="x.png", stored_path="x", sha256="0", size_bytes=0),
        pages=[page],
    )
    html = render_document(document, HtmlRenderOptions(include_element_ids=True))
    assert "first line of the paragraph" in html
    assert "second line of the paragraph" in html
    mm_x = page.mm_per_px_x
    mm_y = page.mm_per_px_y
    # the second line must sit 26px below the first, not one line-height below
    expected_top = round((126 - 100) * mm_y, 3)
    assert f"top:{expected_top}mm" in html, "per-line placement missing"
    assert f"left:0.0mm" in html or "left:0mm" in html
    assert mm_x > 0


def test_edited_text_still_renders_as_a_block():
    """A manual text edit must replace the OCR lines entirely."""
    page = PageIR(
        index=0,
        width_px=1000,
        height_px=600,
        page_format=PageFormat(name="A4", width_mm=210.0, height_mm=297.0, estimated=True),
        elements=[
            Element(
                id="el_edit",
                type=ElementType.PARAGRAPH,
                bbox=BBox(x1=100, y1=100, x2=600, y2=160),
                text="ocr text",
                typography=Typography(font_size_pt=11.0, estimated=True),
                lines=[TextLine(text="ocr text", bbox=BBox(x1=100, y1=100, x2=300, y2=116))],
            )
        ],
    )
    document = DocumentIR(
        document_id="doc_edit",
        job_id="job_edit",
        source=DocumentSource(filename="x.png", stored_path="x", sha256="0", size_bytes=0),
        pages=[page],
    )
    options = HtmlRenderOptions(overrides={"el_edit": {"text": "corrected by hand"}})
    html = render_document(document, options)
    assert "corrected by hand" in html
    assert "ocr text" not in html