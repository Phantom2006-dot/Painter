"""Detection, heuristics, preprocessing and OCR tests (no network access)."""

from __future__ import annotations

import numpy as np
import pytest

from painter.config import Settings, load_settings
from painter.detection import (
    CLASS_PRIORITY,
    CLASS_TO_ELEMENT_TYPE,
    CvFallbackDetector,
    Detection,
    DetectionBundle,
    LayoutDetectorService,
    ModelLoadReport,
    YoloDetector,
    find_photo_regions,
    find_text_regions,
    map_class_to_element_type,
    postprocess_detections,
)
from painter.errors import ModelUnavailableError
from painter.heuristics import (
    HeuristicConfig,
    collect_heuristic_detections,
    detect_lines,
    detect_page_numbers,
    detect_rectangles_and_checkboxes,
    detect_signatures,
    detect_stamps,
)
from painter.ocr import (
    NullProvider,
    OcrChain,
    OcrResult,
    TesseractProvider,
    build_ocr_chain,
    recognize_regions,
)
from painter.preprocess import (
    estimate_skew_angle,
    estimate_text_line_pitch,
    preprocess_page,
    reduce_shadow,
    validate_upload,
)
from painter.schemas import BBox, ElementType, ReviewFlag


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return load_settings(data_dir=tmp_path / "data", enable_yolo=False, ocr_provider="tesseract")


def _blank(width: int = 900, height: int = 1200) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _has_tesseract() -> bool:
    import shutil

    return shutil.which("tesseract") is not None


# --------------------------------------------------------------------------
# class mapping
# --------------------------------------------------------------------------
def test_class_mapping_covers_the_shipped_model():
    # the classes doclayout.pt really emits (verified by loading the checkpoint)
    assert map_class_to_element_type("Title") is ElementType.HEADING
    assert map_class_to_element_type("Section-header") is ElementType.SUBHEADING
    assert map_class_to_element_type("Text") is ElementType.PARAGRAPH
    assert map_class_to_element_type("List-item") is ElementType.LIST
    assert map_class_to_element_type("Table") is ElementType.TABLE
    assert map_class_to_element_type("Picture") is ElementType.IMAGE
    assert map_class_to_element_type("Page-footer") is ElementType.FOOTER
    assert map_class_to_element_type("not-a-class") is ElementType.UNKNOWN
    # target classes of a purpose-trained model are mapped as well
    assert map_class_to_element_type("signature") is ElementType.SIGNATURE
    assert map_class_to_element_type("checkbox") is ElementType.CHECKBOX
    assert "heading" in CLASS_TO_ELEMENT_TYPE


def test_class_priority_ranks_headings_above_body_text():
    assert CLASS_PRIORITY[ElementType.HEADING] > CLASS_PRIORITY[ElementType.SUBHEADING]
    assert CLASS_PRIORITY[ElementType.SUBHEADING] > CLASS_PRIORITY[ElementType.PARAGRAPH]


# --------------------------------------------------------------------------
# post-processing
# --------------------------------------------------------------------------
def test_postprocess_merges_same_class_duplicates(settings):
    first = Detection("Text", ElementType.PARAGRAPH, 0.8, BBox(x1=100, y1=100, x2=400, y2=200))
    second = Detection("Text", ElementType.PARAGRAPH, 0.6, BBox(x1=110, y1=105, x2=410, y2=205))
    merged, _warnings = postprocess_detections([first, second], 900, 1200, settings)
    assert len(merged) == 1
    assert merged[0].confidence == pytest.approx(0.8)
    assert merged[0].bbox.x2 >= 410  # boxes are unioned


def test_postprocess_keeps_more_significant_class_on_a_tie(settings):
    title = Detection("Title", ElementType.HEADING, 0.66, BBox(x1=153, y1=210, x2=1196, y2=290))
    section = Detection(
        "Section-header", ElementType.SUBHEADING, 0.78, BBox(x1=153, y1=210, x2=1196, y2=290)
    )
    merged, _warnings = postprocess_detections([title, section], 1400, 2000, settings)
    assert len(merged) == 1
    # a document title stays a title even though the other class scored higher
    assert merged[0].element_type is ElementType.HEADING
    assert merged[0].extra["also_detected_as"] == ["Section-header"]


def test_postprocess_keeps_clear_winner_regardless_of_priority(settings):
    title = Detection("Title", ElementType.HEADING, 0.30, BBox(x1=153, y1=210, x2=1196, y2=290))
    section = Detection(
        "Section-header", ElementType.SUBHEADING, 0.95, BBox(x1=153, y1=210, x2=1196, y2=290)
    )
    merged, _warnings = postprocess_detections([title, section], 1400, 2000, settings)
    assert len(merged) == 1
    assert merged[0].element_type is ElementType.SUBHEADING


def test_postprocess_does_not_merge_across_kinds(settings):
    paragraph = Detection("Text", ElementType.PARAGRAPH, 0.9, BBox(x1=100, y1=100, x2=600, y2=300))
    table = Detection("Table", ElementType.TABLE, 0.9, BBox(x1=100, y1=100, x2=600, y2=300))
    merged, _warnings = postprocess_detections([paragraph, table], 900, 1200, settings)
    assert {d.element_type for d in merged} == {ElementType.PARAGRAPH, ElementType.TABLE}


def test_postprocess_drops_text_inside_a_picture_but_keeps_table_text(settings):
    picture = Detection("Picture", ElementType.IMAGE, 0.9, BBox(x1=50, y1=50, x2=500, y2=500))
    inside_picture = Detection("Text", ElementType.PARAGRAPH, 0.7, BBox(x1=60, y1=60, x2=480, y2=480))
    merged, _warnings = postprocess_detections([picture, inside_picture], 900, 1200, settings)
    assert [d.element_type for d in merged] == [ElementType.IMAGE]

    table = Detection("Table", ElementType.TABLE, 0.9, BBox(x1=50, y1=50, x2=500, y2=500))
    inside_table = Detection("Text", ElementType.PARAGRAPH, 0.7, BBox(x1=60, y1=60, x2=480, y2=480))
    merged_table, _warnings = postprocess_detections([table, inside_table], 900, 1200, settings)
    texts = [d for d in merged_table if d.element_type is ElementType.PARAGRAPH]
    assert texts and texts[0].extra.get("inside_table") is True


def test_postprocess_collapses_repeated_footers(settings):
    first = Detection("Page-footer", ElementType.FOOTER, 0.9, BBox(x1=50, y1=1150, x2=200, y2=1180))
    second = Detection("Page-footer", ElementType.FOOTER, 0.7, BBox(x1=600, y1=1150, x2=800, y2=1180))
    merged, warnings = postprocess_detections([first, second], 900, 1200, settings)
    footers = [d for d in merged if d.element_type is ElementType.FOOTER]
    assert len(footers) == 1
    assert footers[0].bbox.x1 == 50 and footers[0].bbox.x2 == 800
    assert any("footer" in warning for warning in warnings)


def test_postprocess_drops_noise(settings):
    tiny = Detection("Text", ElementType.PARAGRAPH, 0.9, BBox(x1=10, y1=10, x2=13, y2=13))
    merged, _warnings = postprocess_detections([tiny], 900, 1200, settings)
    assert merged == []


# --------------------------------------------------------------------------
# detector behaviour
# --------------------------------------------------------------------------
def test_yolo_detector_reports_a_missing_model(tmp_path, settings):
    detector = YoloDetector("layout", tmp_path / "nope.pt", settings)
    report = detector.load()
    assert report.ok is False
    assert "not found" in (report.error or "")
    assert report.loader is None
    with pytest.raises(ModelUnavailableError):
        detector.predict(_blank())


def test_yolo_detector_parses_boxes_from_a_model(settings):
    class _FakeBox:
        def __init__(self, xyxy, cls, conf):
            self.xyxy = [np.array(xyxy, dtype=float)]
            self.cls = [cls]
            self.conf = [conf]

    class _FakeResult:
        names = {0: "Table", 1: "Picture"}
        boxes = [_FakeBox([10, 20, 110, 220], 0, 0.91), _FakeBox([200, 300, 400, 500], 1, 0.42)]

    class _FakeModel:
        names = _FakeResult.names

        def predict(self, image, **kwargs):
            return [_FakeResult()]

    detector = YoloDetector("layout", None, settings)
    detector._model = _FakeModel()
    detector.report.ok = True
    detections = detector.predict(_blank())
    assert [d.element_type for d in detections] == [ElementType.TABLE, ElementType.IMAGE]
    assert detections[0].confidence == pytest.approx(0.91)
    assert detections[0].bbox.as_list() == [10.0, 20.0, 110.0, 220.0]
    assert detections[0].source.startswith("yolo:")


def test_model_report_serialises(tmp_path):
    report = ModelLoadReport(spec="layout", path="/tmp/model.pt", loader="ultralytics.YOLO", ok=True)
    payload = report.to_dict()
    assert payload["ok"] is True
    assert payload["loader"] == "ultralytics.YOLO"


def test_layout_service_falls_back_when_no_model_is_available(settings):
    bundle = LayoutDetectorService(settings).detect(_blank())
    assert isinstance(bundle, DetectionBundle)
    assert bundle.detector in {"none", "cv-fallback"}
    assert isinstance(bundle.warnings, list)


def test_cv_fallback_detector_finds_text_and_photos():
    import cv2

    image = _blank()
    for index in range(6):
        cv2.putText(
            image,
            "Reconstructed line of text",
            (60, 200 + index * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
    image[600:900, 100:600] = 40  # a dark block that reads as a picture
    detections = CvFallbackDetector(load_settings(enable_yolo=False)).predict(image)
    assert detections, "the CV fallback must find something on a page with ink"
    assert all(detection.source == "cv-fallback" for detection in detections)


def test_find_text_regions_returns_boxes_on_ink():
    import cv2

    image = _blank()
    for index in range(4):
        cv2.putText(
            image, "Paragraph text", (80, 300 + index * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2
        )
    boxes = find_text_regions(image)
    assert boxes
    assert all(box.width > 0 and box.height > 0 for box in boxes)
    assert all(0 <= box.x1 < box.x2 <= image.shape[1] for box in boxes)


def test_find_text_regions_on_a_blank_page():
    assert find_text_regions(_blank()) == []


def test_find_photo_regions_finds_a_dense_block():
    image = _blank()
    rng = np.random.default_rng(3)
    image[300:700, 150:650] = rng.integers(0, 255, (400, 500, 3), dtype=np.uint8)
    regions = find_photo_regions(image)
    assert any(region.width > 200 and region.height > 200 for region in regions)


# --------------------------------------------------------------------------
# heuristics
# --------------------------------------------------------------------------
def test_detect_lines_finds_rules():
    import cv2

    image = _blank()
    cv2.line(image, (60, 400), (840, 400), (0, 0, 0), 3)
    cv2.line(image, (60, 500), (60, 1000), (0, 0, 0), 3)
    lines = detect_lines(image)
    assert lines, "a page with rules must yield line detections"
    assert all(
        detection.element_type in {ElementType.LINE, ElementType.RECTANGLE} for detection in lines
    )


def test_detect_rectangles_and_checkboxes_accepts_context():
    import cv2

    image = _blank()
    cv2.rectangle(image, (100, 100), (600, 300), (0, 0, 0), 3)
    cv2.rectangle(image, (100, 400), (126, 426), (0, 0, 0), 2)  # checkbox-sized
    extras = detect_rectangles_and_checkboxes(image, [])
    assert isinstance(extras, list)
    assert all(isinstance(detection, Detection) for detection in extras)


def test_detect_stamps_finds_coloured_seal():
    import cv2

    image = _blank()
    cv2.circle(image, (450, 800), 90, (30, 30, 200), 12)  # red ink
    stamps = detect_stamps(image)
    assert stamps, "a saturated circular mark should be detected as a stamp"
    assert all(detection.element_type is ElementType.STAMP for detection in stamps)


def test_detect_signatures_returns_detections():
    import cv2

    image = _blank()
    points = np.array([[120 + i * 12, 1000 + int(30 * np.sin(i / 3))] for i in range(40)], np.int32)
    cv2.polylines(image, [points], False, (40, 40, 40), 3)
    signatures = detect_signatures(image, [])
    assert isinstance(signatures, list)
    assert all(detection.element_type is ElementType.SIGNATURE for detection in signatures)


def test_detect_page_numbers_flags_numeric_footer_text():
    flagged = detect_page_numbers(
        _blank(), [("el_1", BBox(x1=400, y1=1150, x2=500, y2=1180), "3")]
    )
    assert flagged and flagged[0][0] == "el_1"


def test_collect_heuristic_detections_never_raises():
    assert isinstance(collect_heuristic_detections(_blank(), []), list)


def test_heuristic_config_is_tunable():
    config = HeuristicConfig(min_line_length_ratio=0.5)
    assert config.min_line_length_ratio == 0.5
    assert HeuristicConfig().checkbox_max_px > HeuristicConfig().checkbox_min_px


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------
def test_validate_upload_rejects_unknown_and_empty(tmp_path, settings):
    from painter.errors import UnsupportedFormatError

    unknown = tmp_path / "file.txt"
    unknown.write_text("hello")
    with pytest.raises(UnsupportedFormatError):
        validate_upload(unknown, settings)

    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(UnsupportedFormatError):
        validate_upload(empty, settings)


def test_validate_upload_accepts_png_by_magic_bytes(tmp_path, settings):
    import cv2

    path = tmp_path / "page.png"
    cv2.imwrite(str(path), _blank(400, 500))
    assert validate_upload(path, settings) == "png"


def test_validate_upload_enforces_size_limit(tmp_path, settings):
    from dataclasses import replace

    from painter.errors import UnsupportedFormatError

    path = tmp_path / "big.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 4096)
    with pytest.raises(UnsupportedFormatError):
        validate_upload(path, replace(settings, max_upload_mb=0))


def test_estimate_skew_angle_detects_a_rotation():
    import cv2

    image = _blank(1200, 1600)
    for index in range(14):
        cv2.putText(
            image, "Skewed body text line", (120, 200 + index * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2
        )
    matrix = cv2.getRotationMatrix2D((800, 600), 3.0, 1.0)
    rotated = cv2.warpAffine(image, matrix, (1200, 1600), borderValue=(255, 255, 255))
    angle = estimate_skew_angle(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY))
    assert abs(abs(angle) - 3.0) < 1.0


def test_preprocess_records_transforms_and_assumptions(settings):
    result = preprocess_page(_blank(1654, 2339), settings)
    assert result.image is not None
    assert result.metrics["output_width_px"] > 0
    assert result.page_format is not None
    # even a blank page yields a stated assumption, never a silent guess
    assert result.page_format.notes
    assert result.assumptions
    assert result.metrics.get("output_dpi")


def test_preprocess_keeps_the_original_ink(settings):
    import cv2

    image = _blank(1654, 2339)
    for index in range(10):
        cv2.putText(
            image, "Line of body text", (150, 300 + index * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )
    result = preprocess_page(image, settings)
    before = result.metrics["ink_ratio_before"]
    after = result.metrics["ink_ratio_after"]
    assert after > before * 0.45, "preprocessing must not destroy the text"


def test_estimate_text_line_pitch_matches_a_known_pitch():
    import cv2

    image = _blank(1654, 2339)
    for index in range(14):
        cv2.putText(
            image, "Body line of the report", (110, 260 + index * 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )
    pitch = estimate_text_line_pitch(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    assert pitch is not None
    assert abs(pitch - 38) <= 2


def test_estimate_text_line_pitch_is_none_on_a_blank_page():
    assert estimate_text_line_pitch(np.full((600, 400), 255, dtype=np.uint8)) is None


def test_reduce_shadow_flattens_illumination():
    import cv2

    image = _blank(600, 800)
    gradient = np.linspace(120, 255, 600, dtype=np.float32)[None, :, None]
    image = np.clip(image * (gradient / 255.0), 0, 255).astype(np.uint8)
    cv2.putText(image, "Text under a shadow", (40, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    flattened = reduce_shadow(image)
    # the shadow is a horizontal gradient, so the per-column means must flatten
    before = float(image.mean(axis=0).std())
    after = float(flattened.mean(axis=0).std())
    assert after < before, f"shadow not flattened ({before:.1f} -> {after:.1f})"


# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------
def test_build_ocr_chain_respects_the_configured_mode(settings):
    chain = build_ocr_chain(settings)
    assert isinstance(chain, OcrChain)
    assert isinstance(chain.name, str)

    assert chain.available() is True

    disabled = build_ocr_chain(load_settings(ocr_provider="none"))
    assert disabled.name == "none"
    # a chain of nothing but the no-op provider must not look usable
    assert disabled.available() is False


def test_null_provider_is_explicit_about_failure():
    provider = NullProvider()
    # it is "usable" only so the chain can return an explanatory result
    assert provider.available() is True
    result = provider.recognize(_blank(100, 100))
    assert result.empty is True
    assert result.provider == "none"
    assert ReviewFlag.OCR_FAILED in result.flags


def test_recognize_regions_maps_results_to_element_ids():
    chain = OcrChain([NullProvider()])
    results = recognize_regions(chain, _blank(400, 400), [("el_1", BBox(x1=10, y1=10, x2=200, y2=60))])
    assert set(results) == {"el_1"}
    assert isinstance(results["el_1"], OcrResult)


def test_recognize_regions_flags_regions_that_are_too_small():
    chain = OcrChain([NullProvider()])
    results = recognize_regions(chain, _blank(100, 100), [("tiny", BBox(x1=1, y1=1, x2=2, y2=2))])
    assert results["tiny"].empty is True
    assert results["tiny"].warnings


@pytest.mark.skipif(not _has_tesseract(), reason="tesseract is not installed")
def test_tesseract_provider_reads_rendered_text(settings):
    import cv2

    image = _blank(500, 160)
    cv2.putText(image, "Invoice 1250", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    provider = TesseractProvider(settings)
    assert provider.available() is True
    result = provider.recognize(image)
    assert "1250" in result.text.replace(" ", "")
    assert result.provider == "tesseract"
    assert result.lines
