"""Deterministic recovery of classes the layout model was not trained on.

The bundled DocLayout-YOLO checkpoint knows Title, Text, Table, Picture,
Section-header, List-item, Caption, Footnote, Formula, Page-header and
Page-footer — and nothing else. Rather than pretending YOLO can detect
signatures or checkboxes, these regions are recovered with explicit computer
vision heuristics and are always emitted with a low confidence plus a review
flag so the user knows they are estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .detection import Detection
from .logging_utils import get_logger
from .schemas import BBox, ElementType

logger = get_logger("heuristics")


def _cv2():
    import cv2

    return cv2


@dataclass
class HeuristicConfig:
    min_line_length_ratio: float = 0.18
    min_line_thickness_px: int = 2
    checkbox_min_px: int = 8
    checkbox_max_px: int = 90
    signature_zone_start: float = 0.55  # fraction of page height
    stamp_min_area_ratio: float = 0.004
    logo_max_area_ratio: float = 0.06


def _mask_of_lines(gray: np.ndarray, orientation: str, config: HeuristicConfig) -> np.ndarray:
    cv2 = _cv2()
    height, width = gray.shape
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )
    if orientation == "horizontal":
        length = max(20, int(width * config.min_line_length_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    else:
        length = max(20, int(height * config.min_line_length_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)


def detect_lines(image: np.ndarray, config: HeuristicConfig | None = None) -> list[Detection]:
    """Detect long horizontal and vertical rules (table borders, underlines)."""
    cv2 = _cv2()
    config = config or HeuristicConfig()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    detections: list[Detection] = []

    for orientation in ("horizontal", "vertical"):
        mask = _mask_of_lines(gray, orientation, config)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            thickness = h if orientation == "horizontal" else w
            length = w if orientation == "horizontal" else h
            if thickness < config.min_line_thickness_px or thickness > 40:
                continue
            if orientation == "horizontal" and length < width * config.min_line_length_ratio:
                continue
            if orientation == "vertical" and length < height * config.min_line_length_ratio:
                continue
            detections.append(
                Detection(
                    class_name="line",
                    element_type=ElementType.LINE,
                    confidence=0.5,
                    bbox=BBox(x1=x, y1=y, x2=x + w, y2=y + h),
                    source="cv-heuristic",
                    extra={"orientation": orientation, "thickness_px": thickness},
                )
            )
    return detections


def detect_rectangles_and_checkboxes(
    image: np.ndarray, existing: Sequence[Detection], config: HeuristicConfig | None = None
) -> list[Detection]:
    """Detect closed rectangles and small square boxes (checkbox candidates)."""
    cv2 = _cv2()
    config = config or HeuristicConfig()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    text_boxes = [d.bbox for d in existing if d.element_type not in {ElementType.TABLE}]
    results: list[Detection] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 6 or h < 6:
            continue
        area = float(w * h)
        if area < 36 or area > gray.shape[0] * gray.shape[1] * 0.5:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) != 4:
            continue
        box = BBox(x1=x, y1=y, x2=x + w, y2=y + h)
        aspect = w / h if h else 0
        is_square = 0.7 <= aspect <= 1.4
        filled_fraction = cv2.contourArea(contour) / area if area else 0
        if filled_fraction < 0.6:
            continue
        # skip boxes that are just the glyph outlines of large text
        inside_text = any(text_box.overlap_fraction(box) > 0.8 for text_box in text_boxes)
        if inside_text:
            continue
        if is_square and config.checkbox_min_px <= w <= config.checkbox_max_px:
            results.append(
                Detection(
                    class_name="checkbox",
                    element_type=ElementType.CHECKBOX,
                    confidence=0.4,
                    bbox=box,
                    source="cv-heuristic",
                    extra={"filled_fraction": round(filled_fraction, 3)},
                )
            )
        elif w > gray.shape[1] * 0.12 and h > gray.shape[0] * 0.05:
            results.append(
                Detection(
                    class_name="rectangle",
                    element_type=ElementType.RECTANGLE,
                    confidence=0.35,
                    bbox=box,
                    source="cv-heuristic",
                    extra={"filled_fraction": round(filled_fraction, 3)},
                )
            )
    return results


def detect_stamps(image: np.ndarray, config: HeuristicConfig | None = None) -> list[Detection]:
    """Find coloured stamp/seal ink: saturated pixels in a compact blob."""
    cv2 = _cv2()
    config = config or HeuristicConfig()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((saturation > 70) & (value > 60)).astype(np.uint8) * 255
    if mask.sum() == 0:
        return []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(image.shape[0] * image.shape[1])
    results: list[Detection] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < frame_area * config.stamp_min_area_ratio or area > frame_area * 0.25:
            continue
        hull_area = float(cv2.contourArea(contour))
        fill = hull_area / area if area else 0
        if fill < 0.12:  # stamps are outlines/shapes, not solid colour blocks
            continue
        results.append(
            Detection(
                class_name="stamp",
                element_type=ElementType.STAMP,
                confidence=0.42,
                bbox=BBox(x1=x, y1=y, x2=x + w, y2=y + h),
                source="cv-heuristic",
                extra={"fill_ratio": round(fill, 3)},
            )
        )
    return results


def detect_signatures(
    image: np.ndarray, existing: Sequence[Detection], config: HeuristicConfig | None = None
) -> list[Detection]:
    """Find handwriting-like ink in the lower part of the page.

    Handwriting differs from print in three measurable ways: strokes are thinner
    and more diagonal, the ink is not organised into horizontal baselines, and
    the connected components vary wildly in height.
    """
    cv2 = _cv2()
    config = config or HeuristicConfig()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    zone_top = int(height * config.signature_zone_start)
    zone = gray[zone_top:, :]
    if zone.size == 0:
        return []
    binary = cv2.adaptiveThreshold(
        zone, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_boxes = [d.bbox for d in existing]
    candidates: list[BBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.08 or h < 12:
            continue
        if w * h < width * height * 0.0008:
            continue
        box = BBox(x1=x, y1=y + zone_top, x2=x + w, y2=y + h + zone_top)
        if any(text_box.overlap_fraction(box) > 0.6 for text_box in text_boxes):
            continue
        candidates.append(box)
    if not candidates:
        return []
    # merge candidates that sit on the same visual line
    candidates.sort(key=lambda b: (b.y1, b.x1))
    merged: list[BBox] = []
    for box in candidates:
        if merged and abs(box.center[1] - merged[-1].center[1]) < max(12.0, box.height * 0.6):
            last = merged[-1]
            merged[-1] = BBox(
                x1=min(last.x1, box.x1), y1=min(last.y1, box.y1),
                x2=max(last.x2, box.x2), y2=max(last.y2, box.y2),
            )
        else:
            merged.append(box)
    return [
        Detection(
            class_name="signature",
            element_type=ElementType.SIGNATURE,
            confidence=0.35,
            bbox=box,
            source="cv-heuristic",
            extra={"note": "handwriting heuristic; verify manually"},
        )
        for box in merged
        if box.width > width * 0.1
    ]


def detect_logo(
    image: np.ndarray, picture_boxes: Sequence[BBox], config: HeuristicConfig | None = None
) -> list[Detection]:
    """Reclassify small, colourful, top-of-page pictures as logos."""
    cv2 = _cv2()
    config = config or HeuristicConfig()
    height, width = image.shape[:2]
    frame_area = float(height * width)
    results: list[Detection] = []
    for box in picture_boxes:
        area_ratio = box.area / frame_area
        if area_ratio > config.logo_max_area_ratio:
            continue
        if box.y1 > height * 0.28:  # logos live in the header band
            continue
        if box.width > width * 0.55:
            continue
        crop = image[int(box.y1) : int(box.y2), int(box.x1) : int(box.x2)]
        if crop.size == 0:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturation = float(hsv[:, :, 1].mean())
        results.append(
            Detection(
                class_name="logo",
                element_type=ElementType.LOGO,
                confidence=0.45 if saturation > 40 else 0.35,
                bbox=box,
                source="cv-heuristic",
                extra={"mean_saturation": round(saturation, 1), "area_ratio": round(area_ratio, 4)},
            )
        )
    return results


def detect_page_numbers(
    image: np.ndarray, text_elements: Iterable[tuple[str, BBox, str]]
) -> list[tuple[str, int]]:
    """Flag text elements that look like page numbers.

    ``text_elements`` is an iterable of ``(element_id, bbox, text)``. A page
    number is a short, mostly numeric string in the top or bottom 12% of the
    page.
    """
    height = image.shape[0]
    flagged: list[tuple[str, int]] = []
    for element_id, box, text in text_elements:
        stripped = (text or "").strip()
        if not stripped or len(stripped) > 12:
            continue
        digits = sum(ch.isdigit() for ch in stripped)
        if digits == 0 or digits / max(len(stripped), 1) < 0.5:
            continue
        in_top = box.y2 < height * 0.12
        in_bottom = box.y1 > height * 0.88
        if in_top or in_bottom:
            flagged.append((element_id, int(stripped.strip(" -–—.")) if stripped.strip(" -–—.").isdigit() else 0))
    return flagged


def collect_heuristic_detections(
    image: np.ndarray,
    existing: Sequence[Detection],
    enable_shapes: bool = True,
    enable_signature: bool = True,
    enable_stamp: bool = True,
) -> list[Detection]:
    """Run every heuristic and return the extra detections it found."""
    extras: list[Detection] = []
    try:
        if enable_shapes:
            extras.extend(detect_lines(image))
            extras.extend(detect_rectangles_and_checkboxes(image, existing))
    except Exception as exc:  # noqa: BLE001 - heuristics must never break the pipeline
        logger.warning("shape heuristics failed: %s", exc)
    try:
        if enable_stamp:
            extras.extend(detect_stamps(image))
    except Exception as exc:  # noqa: BLE001
        logger.warning("stamp heuristic failed: %s", exc)
    try:
        if enable_signature:
            extras.extend(detect_signatures(image, existing))
    except Exception as exc:  # noqa: BLE001
        logger.warning("signature heuristic failed: %s", exc)
    return extras
