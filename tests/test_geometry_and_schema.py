"""Geometry engine and Document IR validation tests."""

from __future__ import annotations

import pytest

from painter.geometry import (
    detect_columns,
    detect_page_format,
    infer_alignment,
    infer_page_format,
    infer_margins,
    map_bbox_original_to_page,
    map_point_through_transforms,
    mm_to_pt,
    pt_to_mm,
)
from painter.schemas import (
    Alignment,
    BBox,
    DocumentIR,
    Element,
    ElementType,
    NormBBox,
    PageFormat,
    PageIR,
    PageTransform,
    ReviewFlag,
    validate_document,
)


def test_bbox_helpers():
    box = BBox(x1=10, y1=20, x2=60, y2=80)
    assert box.width == 50
    assert box.height == 60
    assert box.area == 3000
    assert box.center == (35.0, 50.0)
    assert box.iou(box) == pytest.approx(1.0)
    # overlap_fraction is the share of *this* box covered by the other one
    assert box.overlap_fraction(BBox(x1=0, y1=0, x2=20, y2=40)) == pytest.approx(200 / 3000)
    assert box.overlap_fraction(BBox(x1=0, y1=0, x2=5, y2=5)) == 0.0
    clipped = BBox(x1=-10, y1=-10, x2=100, y2=100).clip(50, 50)
    assert clipped.as_list() == [0.0, 0.0, 50.0, 50.0]


def test_norm_bbox_round_trip():
    box = BBox(x1=100, y1=200, x2=300, y2=400)
    norm = NormBBox.from_bbox(box, 1000, 2000)
    assert (norm.x1, norm.y1, norm.x2, norm.y2) == (0.1, 0.1, 0.3, 0.2)
    restored = norm.to_bbox(1000, 2000)
    assert restored.as_list() == box.as_list()


@pytest.mark.parametrize(
    "width,height,expected",
    [
        (1000, 1414, "A4"),
        (1414, 1000, "A4"),  # landscape A4
        (850, 1100, "Letter"),
        (1240, 1754, "A4"),
        (700, 1000, "A4"),  # ISO 216 aspect: the A4 default is assumed
    ],
)
def test_detect_page_format(width, height, expected):
    fmt = detect_page_format(width, height)
    assert fmt.name == expected
    assert fmt.estimated is True
    assert fmt.notes, "an inferred format must state its assumption"


def test_detect_page_format_custom():
    fmt = detect_page_format(900, 901, dpi=100)
    assert fmt.name == "custom"
    assert fmt.width_mm == pytest.approx(228.6, abs=1.0)


def test_infer_page_format_assumes_a4_for_iso_aspect():
    fmt, dpi, notes = infer_page_format(1654, 2339)
    assert fmt.name == "A4"
    assert dpi == pytest.approx(200, abs=2)
    assert any("A4" in note for note in notes)


def test_infer_page_format_plausible_line_pitch_keeps_a4():
    # a 38px line pitch at 200 dpi implies ~11.4pt body text: plausible for A4
    fmt, dpi, notes = infer_page_format(1654, 2339, line_height_px=38)
    assert fmt.name == "A4"
    assert any("body text estimated" in note for note in notes)


def test_infer_page_format_implausible_pitch_switches_format():
    # 19px pitch would imply 5.7pt on A4, so the A3 member is tried instead
    fmt, _dpi, notes = infer_page_format(1654, 2339, line_height_px=19)
    assert fmt.name == "A3"
    assert any("A3" in note for note in notes)

    # 60px pitch would imply 18pt on A4, which is too large: A5 fits better
    fmt_small, _dpi_small, _ = infer_page_format(1654, 2339, line_height_px=60)
    assert fmt_small.name == "A5"


def test_infer_page_format_letter_family():
    fmt, dpi, _notes = infer_page_format(2550, 3300)
    assert fmt.name == "Letter"
    assert dpi == pytest.approx(300, abs=3)
    assert fmt.width_mm == pytest.approx(215.9)


def test_infer_page_format_handles_landscape():
    fmt, _dpi, _ = infer_page_format(2339, 1654)
    assert fmt.name == "A4"
    assert fmt.width_mm == pytest.approx(297.0)
    assert fmt.height_mm == pytest.approx(210.0)


def test_infer_page_format_rejects_degenerate_size():
    fmt, dpi, notes = infer_page_format(0, 0)
    assert fmt.name == "A4"
    assert dpi == 300.0
    assert notes


def test_units():
    assert mm_to_pt(25.4) == pytest.approx(72.0)
    assert pt_to_mm(72.0) == pytest.approx(25.4)


def test_coordinate_mapping_through_transforms():
    transforms = [
        PageTransform(kind="crop", params={"x": 10, "y": 20}),
        PageTransform(kind="resize", params={"scale_x": 2.0, "scale_y": 2.0}),
    ]
    assert map_point_through_transforms((30, 40), transforms) == (40.0, 40.0)
    box = BBox(x1=10, y1=20, x2=20, y2=30)
    mapped = map_bbox_original_to_page(box, transforms)
    assert mapped.as_list() == [0.0, 0.0, 20.0, 20.0]


def test_coordinate_mapping_perspective_and_rotation():
    matrix = [[1, 0, 5], [0, 1, 7], [0, 0, 1]]
    mapped = map_point_through_transforms((1, 2), [PageTransform(kind="perspective", params={"matrix": matrix})])
    assert mapped == pytest.approx((6.0, 9.0))
    rotate = PageTransform(kind="rotate", params={"quarter_turns": 1, "output_w": 100, "output_h": 200})
    assert map_point_through_transforms((10, 20), [rotate]) == (80.0, 10.0)


def test_infer_alignment_center_and_left():
    container = BBox(x1=0, y1=0, x2=100, y2=100)
    centered = [BBox(x1=40, y1=0, x2=60, y2=10), BBox(x1=42, y1=20, x2=58, y2=30)]
    assert infer_alignment(centered, container) is Alignment.CENTER
    left = [BBox(x1=0, y1=0, x2=60, y2=10), BBox(x1=0, y1=20, x2=95, y2=30)]
    assert infer_alignment(left, container) is Alignment.LEFT


def test_infer_margins():
    page_format = PageFormat(name="A4", width_mm=210, height_mm=297)
    elements = [Element(type=ElementType.PARAGRAPH, bbox=BBox(x1=100, y1=100, x2=900, y2=900), confidence=0.9)]
    margins = infer_margins(elements, 1000, 1000, page_format)
    assert margins.left == pytest.approx(21.0, abs=0.1)
    assert margins.top == pytest.approx(29.7, abs=0.1)


def test_detect_columns_single_and_multi():
    single = [Element(type=ElementType.PARAGRAPH, bbox=BBox(x1=50, y1=0, x2=550, y2=50), confidence=0.5)]
    assert len(detect_columns(single, 600)) == 1
    two = [
        Element(type=ElementType.PARAGRAPH, bbox=BBox(x1=20, y1=0, x2=280, y2=50), confidence=0.5),
        Element(type=ElementType.PARAGRAPH, bbox=BBox(x1=320, y1=0, x2=580, y2=50), confidence=0.5),
    ]
    assert len(detect_columns(two, 600)) == 2


def _document_with_element(element: Element) -> DocumentIR:
    page = PageIR(index=0, width_px=1000, height_px=1400, elements=[element], reading_order=[element.id])
    return DocumentIR(document_id="doc_test", pages=[page])


def test_validation_accepts_clean_document():
    element = Element(type=ElementType.PARAGRAPH, bbox=BBox(x1=10, y1=10, x2=100, y2=50), confidence=0.9, text="hello")
    report = validate_document(_document_with_element(element))
    assert report.ok
    assert not report.errors


def test_validation_rejects_out_of_page_bbox():
    element = Element(type=ElementType.PARAGRAPH, bbox=BBox(x1=10, y1=10, x2=2000, y2=50), confidence=0.9, text="x")
    report = validate_document(_document_with_element(element))
    assert not report.ok
    assert any(issue.code == "bbox_out_of_page" for issue in report.errors)


def test_validation_flags_missing_text():
    element = Element(type=ElementType.PARAGRAPH, bbox=BBox(x1=10, y1=10, x2=100, y2=50), confidence=0.9)
    report = validate_document(_document_with_element(element))
    assert any(issue.code == "text_missing" for issue in report.warnings)


def test_validation_accepts_flagged_missing_text():
    element = Element(type=ElementType.PARAGRAPH, bbox=BBox(x1=10, y1=10, x2=100, y2=50), confidence=0.9)
    element.add_flag(ReviewFlag.OCR_FAILED)
    report = validate_document(_document_with_element(element))
    assert not any(issue.code == "text_missing" for issue in report.warnings)


def test_validation_detects_reading_order_problem():
    element = Element(type=ElementType.PARAGRAPH, bbox=BBox(x1=10, y1=10, x2=100, y2=50), confidence=0.9, text="x")
    page = PageIR(index=0, width_px=1000, height_px=1400, elements=[element], reading_order=["ghost"])
    document = DocumentIR(document_id="doc", pages=[page])
    report = validate_document(document)
    assert not report.ok
    assert any(issue.code == "reading_order_unknown_id" for issue in report.errors)


def test_document_stats():
    element = Element(type=ElementType.PARAGRAPH, bbox=BBox(x1=0, y1=0, x2=10, y2=10), confidence=0.5, text="abc")
    document = _document_with_element(element)
    document.recompute_stats()
    assert document.stats["elements"] == 1
    assert document.stats["text_characters"] == 3
