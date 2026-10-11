"""Region detection: YOLO models plus a deterministic OpenCV fallback.

The repository ships ``doclayout.pt`` — a DocLayout-YOLO checkpoint whose real
class set (verified by loading it, see ``docs/AUDIT.md``) is::

    Caption, Footnote, Formula, List-item, Page-footer, Page-header,
    Picture, Section-header, Table, Text, Title

Those classes are mapped onto :class:`~painter.schemas.ElementType`. Classes the
model was *not* trained on (signature, stamp, logo, checkbox, form field, lines,
page numbers) are never invented by YOLO: they are recovered by the
deterministic heuristics in :mod:`painter.heuristics`, and a custom model can be
dropped in through ``PAINTER_CUSTOM_MODEL``.

Loading is defensive. A checkpoint may require a specific runtime (this one is a
pickled ``doclayout_yolo`` model), so several loaders are attempted in order and
each attempt is reported instead of silently falling back.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .config import Settings
from .errors import DetectionError, ModelUnavailableError
from .logging_utils import get_logger
from .schemas import BBox, ElementType

logger = get_logger("detection")


# --------------------------------------------------------------------------
# class mapping
# --------------------------------------------------------------------------
#: DocLayout-YOLO DocStructBench class names (verified against doclayout.pt).
CLASS_TO_ELEMENT_TYPE: dict[str, ElementType] = {
    "title": ElementType.HEADING,
    "section-header": ElementType.SUBHEADING,
    "section_header": ElementType.SUBHEADING,
    "text": ElementType.PARAGRAPH,
    "list-item": ElementType.LIST,
    "list_item": ElementType.LIST,
    "table": ElementType.TABLE,
    "picture": ElementType.IMAGE,
    "figure": ElementType.IMAGE,
    "caption": ElementType.FIGURE_CAPTION,
    "footnote": ElementType.TEXT_BLOCK,
    "formula": ElementType.FORMULA,
    "page-header": ElementType.HEADER,
    "page_header": ElementType.HEADER,
    "page-footer": ElementType.FOOTER,
    "page_footer": ElementType.FOOTER,
    "abandon": ElementType.UNKNOWN,
    "plain text": ElementType.PARAGRAPH,
    # classes a purpose-trained Painter model would emit
    "heading": ElementType.HEADING,
    "subheading": ElementType.SUBHEADING,
    "paragraph": ElementType.PARAGRAPH,
    "text_block": ElementType.TEXT_BLOCK,
    "list": ElementType.LIST,
    "image": ElementType.IMAGE,
    "logo": ElementType.LOGO,
    "signature": ElementType.SIGNATURE,
    "stamp": ElementType.STAMP,
    "header": ElementType.HEADER,
    "footer": ElementType.FOOTER,
    "page_number": ElementType.PAGE_NUMBER,
    "line": ElementType.LINE,
    "rectangle": ElementType.RECTANGLE,
    "circle": ElementType.CIRCLE,
    "checkbox": ElementType.CHECKBOX,
    "form_field": ElementType.FORM_FIELD,
}

#: Generic COCO classes that weakly indicate a photo/logo region. They are only
#: used as a secondary signal, never as the primary document layout detector.
COCO_IMAGE_HINTS = {
    "book",
    "laptop",
    "cell phone",
    "tv",
    "scissors",
    "remote",
    "keyboard",
    "mouse",
    "clock",
    "bottle",
    "cup",
    "person",
}


#: When two classes claim the same region, the more structurally significant
#: class wins if the confidences are close: a document title reported as both
#: "Title" and "Section-header" must stay a title.
CLASS_PRIORITY: dict[ElementType, int] = {
    ElementType.HEADING: 5,
    ElementType.SUBHEADING: 4,
    ElementType.TABLE: 4,
    ElementType.IMAGE: 3,
    ElementType.LIST: 3,
    ElementType.TEXT_BLOCK: 2,
    ElementType.PARAGRAPH: 2,
    ElementType.FIGURE_CAPTION: 2,
    ElementType.TABLE_CAPTION: 2,
    ElementType.HEADER: 2,
    ElementType.FOOTER: 2,
}

#: Confidence difference within which CLASS_PRIORITY decides instead.
CONFIDENCE_TIE = 0.15


def map_class_to_element_type(class_name: str) -> ElementType:
    return CLASS_TO_ELEMENT_TYPE.get(class_name.strip().lower(), ElementType.UNKNOWN)


@dataclass
class Detection:
    """One detected region, in page pixel coordinates."""

    class_name: str
    element_type: ElementType
    confidence: float
    bbox: BBox
    source: str = "yolo"
    raw_class_id: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "type": self.element_type.value,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.as_list(),
            "source": self.source,
        }


@dataclass
class ModelLoadReport:
    spec: str
    path: str | None
    loader: str | None = None
    ok: bool = False
    error: str | None = None
    names: dict[int, str] = field(default_factory=dict)
    elapsed_s: float | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "path": self.path,
            "loader": self.loader,
            "ok": self.ok,
            "error": self.error,
            "classes": list(self.names.values()),
            "elapsed_s": self.elapsed_s,
            "attempts": self.attempts,
        }


def _doclayout_loader(path: str) -> Any:
    import doclayout_yolo  # type: ignore

    return doclayout_yolo.YOLOv10(path)


def _doclayout_generic_loader(path: str) -> Any:
    import doclayout_yolo  # type: ignore

    return doclayout_yolo.YOLO(path)


def _ultralytics_loader(path: str) -> Any:
    from ultralytics import YOLO

    return YOLO(path)


#: Loaders attempted in order. The first one that loads *and* runs a smoke
#: prediction wins.
LOADER_CHAIN: list[tuple[str, Callable[[str], Any]]] = [
    ("doclayout_yolo.YOLOv10", _doclayout_loader),
    ("doclayout_yolo.YOLO", _doclayout_generic_loader),
    ("ultralytics.YOLO", _ultralytics_loader),
]


class YoloDetector:
    """Thread-safe wrapper around a YOLO checkpoint with lazy loading."""

    def __init__(
        self,
        spec_name: str,
        model_path: Path | None,
        settings: Settings,
        imgsz: int | None = None,
        conf: float | None = None,
        iou: float | None = None,
    ) -> None:
        self.spec_name = spec_name
        self.model_path = model_path
        self.settings = settings
        self.imgsz = imgsz or settings.yolo_imgsz
        self.conf = settings.yolo_conf if conf is None else conf
        self.iou = settings.yolo_iou if iou is None else iou
        self._model: Any = None
        self._names: dict[int, str] = {}
        self._lock = threading.Lock()
        self.report = ModelLoadReport(spec=spec_name, path=str(model_path) if model_path else None)

    # -- loading ---------------------------------------------------------
    @property
    def available(self) -> bool:
        return self.report.ok

    @property
    def names(self) -> dict[int, str]:
        return dict(self._names)

    def load(self, force: bool = False) -> ModelLoadReport:
        with self._lock:
            if self._model is not None and self.report.ok and not force:
                return self.report
            if self.model_path is None or not Path(self.model_path).is_file():
                self.report.ok = False
                self.report.error = f"model file not found: {self.model_path}"
                logger.warning("model %s unavailable: %s", self.spec_name, self.report.error)
                return self.report

            started = time.perf_counter()
            for loader_name, loader in LOADER_CHAIN:
                attempt: dict[str, Any] = {"loader": loader_name}
                try:
                    model = loader(str(self.model_path))
                    names = {int(k): str(v) for k, v in (getattr(model, "names", {}) or {}).items()}
                    # Smoke test: a model that loads but cannot infer is useless.
                    smoke = np.full((self.imgsz, self.imgsz, 3), 255, dtype=np.uint8)
                    model.predict(smoke, imgsz=self.imgsz, conf=self.conf, verbose=False)
                    self._model = model
                    self._names = names
                    self.report.loader = loader_name
                    self.report.ok = True
                    self.report.error = None
                    self.report.names = names
                    attempt["status"] = "ok"
                    self.report.attempts.append(attempt)
                    logger.info(
                        "loaded %s with %s (%d classes)", self.spec_name, loader_name, len(names)
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - we report, never crash
                    attempt["status"] = "failed"
                    attempt["error"] = f"{type(exc).__name__}: {exc}"
                    self.report.attempts.append(attempt)
                    self.report.error = attempt["error"]
                    logger.warning(
                        "loader %s failed for %s: %s", loader_name, self.spec_name, attempt["error"]
                    )
            self.report.elapsed_s = round(time.perf_counter() - started, 2)
            if not self.report.ok:
                self._model = None
            return self.report

    # -- inference -------------------------------------------------------
    def predict(self, image: np.ndarray, conf: float | None = None) -> list[Detection]:
        if self._model is None:
            self.load()
        if self._model is None:
            raise ModelUnavailableError(
                f"{self.spec_name} could not be loaded: {self.report.error or 'unknown error'}",
                details={"spec": self.spec_name, "path": str(self.model_path)},
            )
        try:
            kwargs: dict[str, Any] = {
                "imgsz": self.imgsz,
                "conf": self.conf if conf is None else conf,
                "iou": self.iou,
                "verbose": False,
            }
            if self.settings.yolo_device:
                kwargs["device"] = self.settings.yolo_device
            results = self._model.predict(image, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise DetectionError(
                f"inference failed for {self.spec_name}: {type(exc).__name__}: {exc}"
            ) from exc

        detections: list[Detection] = []
        for result in results:
            names = getattr(result, "names", None) or self._names
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                try:
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                except Exception:  # pragma: no cover - malformed box
                    continue
                class_name = str(names.get(class_id, f"class_{class_id}"))
                detections.append(
                    Detection(
                        class_name=class_name,
                        element_type=map_class_to_element_type(class_name),
                        confidence=confidence,
                        bbox=BBox.from_list([x1, y1, x2, y2]),
                        source=f"yolo:{self.spec_name}",
                        raw_class_id=class_id,
                    )
                )
        return detections


# --------------------------------------------------------------------------
# deterministic OpenCV fallback
# --------------------------------------------------------------------------
def find_text_regions(image: np.ndarray, max_regions: int = 80) -> list[BBox]:
    """Group text-like ink into blocks, robust to text polarity.

    This is the repaired version of the routine in the legacy demo: the original
    multiplied a rectangle by itself, referenced ``bands`` outside its loop and
    raised ``IndentationError`` at import time, so it never actually ran.
    """
    height, width = image.shape[:2]
    scale = min(1.0, 1400.0 / max(height, width))
    small = cv2_resize(image, scale)
    gray = to_gray(small)
    area = gray.shape[0] * gray.shape[1]
    H, W = gray.shape
    if area == 0:
        return []

    kernel = cv2_get_structuring(W, H)
    dilate = cv2_dilate_kernel()
    thr_inv = cv2_adaptive(gray, invert=True)
    thr_bin = cv2_adaptive(gray, invert=False)

    rects: list[list[float]] = []
    for mask in (thr_inv, thr_bin):
        closed = cv2_close(mask, kernel)
        closed = cv2_dilate(closed, dilate)
        for x, y, cw, ch in cv2_contours(closed):
            if cw < 8 or ch < 6:
                continue
            if cw * ch < area * 0.0004 or cw * ch > area * 0.32:
                continue
            if ch > cw * 4:
                continue
            if x < W * 0.012 or y < H * 0.012 or x + cw > W * 0.988 or y + ch > H * 0.988:
                continue
            rects.append([x, y, cw, ch])

    # merge nearby rectangles into paragraph bands
    rects.sort(key=lambda r: (r[1], r[0]))
    bands: list[list[float]] = []
    for rect in rects:
        merged = False
        for band in bands:
            vertical_overlap = min(band[1] + band[3], rect[1] + rect[3]) - max(band[1], rect[1])
            if vertical_overlap > 0.35 * min(band[3], rect[3]) and rect[0] < band[0] + band[2] + W * 0.05:
                band[0] = min(band[0], rect[0])
                band[1] = min(band[1], rect[1])
                band[2] = max(band[0] + band[2], rect[0] + rect[2]) - band[0]
                band[3] = max(band[1] + band[3], rect[1] + rect[3]) - band[1]
                merged = True
                break
        if not merged:
            bands.append(list(rect))

    boxes = [BBox(x1=b[0], y1=b[1], x2=b[0] + b[2], y2=b[1] + b[3]) for b in bands]
    boxes.sort(key=lambda b: (b.y1, b.x1))
    if scale < 1.0:
        boxes = [
            BBox(x1=b.x1 / scale, y1=b.y1 / scale, x2=b.x2 / scale, y2=b.y2 / scale) for b in boxes
        ]
    return boxes[:max_regions]


def find_photo_regions(image: np.ndarray, min_area_ratio: float = 0.01) -> list[BBox]:
    """Locate photo/figure-like blocks: non-paper areas with dense structure."""
    gray = to_gray(image)
    height, width = gray.shape
    blurred = _cv2().GaussianBlur(gray, (7, 7), 0)
    _, bright = _cv2().threshold(blurred, 0, 255, _cv2().THRESH_OTSU)
    dark = _cv2().bitwise_not(bright)
    kernel = _cv2().getStructuringElement(
        _cv2().MORPH_RECT, (max(9, width // 60), max(9, height // 60))
    )
    closed = _cv2().morphologyEx(dark, _cv2().MORPH_CLOSE, kernel, iterations=2)
    regions: list[BBox] = []
    contours, _ = _cv2().findContours(closed, _cv2().RETR_EXTERNAL, _cv2().CHAIN_APPROX_SIMPLE)
    frame_area = float(height * width)
    for contour in contours:
        x, y, w, h = _cv2().boundingRect(contour)
        box_area = float(w * h)
        if box_area < frame_area * min_area_ratio:
            continue
        fill = float(_cv2().contourArea(contour)) / box_area if box_area else 0
        aspect = w / h if h else 0
        if fill > 0.82 and 0.4 < aspect < 2.5 and w > width * 0.12 and h > height * 0.06:
            regions.append(BBox(x1=x, y1=y, x2=x + w, y2=y + h))
    regions.sort(key=lambda b: (b.y1, b.x1))
    return regions


class CvFallbackDetector:
    """Detector used when no YOLO checkpoint is usable.

    Produces paragraph/heading/image candidates from ink morphology only. It is
    explicitly weaker than the layout model, which is why every element it
    produces is flagged for review by the pipeline.
    """

    name = "cv-fallback"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def predict(self, image: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        text_boxes = find_text_regions(image)
        if text_boxes:
            heights = sorted(b.height for b in text_boxes)
            median_height = heights[len(heights) // 2] if heights else 0.0
            for box in text_boxes:
                is_heading = median_height > 0 and box.height > median_height * 1.7 and box.width > image.shape[1] * 0.1
                detections.append(
                    Detection(
                        class_name="heading" if is_heading else "text",
                        element_type=ElementType.HEADING if is_heading else ElementType.PARAGRAPH,
                        confidence=0.45 if is_heading else 0.4,
                        bbox=box,
                        source="cv-fallback",
                    )
                )
        for box in find_photo_regions(image):
            detections.append(
                Detection(
                    class_name="picture",
                    element_type=ElementType.IMAGE,
                    confidence=0.35,
                    bbox=box,
                    source="cv-fallback",
                )
            )
        return detections


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
@dataclass
class DetectionBundle:
    detections: list[Detection]
    detector: str
    model_reports: list[ModelLoadReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    used_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "used_fallback": self.used_fallback,
            "count": len(self.detections),
            "elapsed_s": self.elapsed_s,
            "warnings": self.warnings,
            "models": [r.to_dict() for r in self.model_reports],
            "detections": [d.to_dict() for d in self.detections],
        }


def _detection_kind(detection: "Detection") -> str:
    """Coarse grouping used when merging overlapping detections."""
    if detection.element_type in {
        ElementType.PARAGRAPH,
        ElementType.TEXT_BLOCK,
        ElementType.HEADING,
        ElementType.SUBHEADING,
        ElementType.LIST,
        ElementType.FIGURE_CAPTION,
        ElementType.TABLE_CAPTION,
        ElementType.FORMULA,
        ElementType.PAGE_NUMBER,
        ElementType.HEADER,
        ElementType.FOOTER,
        ElementType.FORM_FIELD,
    }:
        return "text"
    if detection.element_type in {ElementType.IMAGE, ElementType.LOGO, ElementType.SIGNATURE, ElementType.STAMP}:
        return "image"
    if detection.element_type is ElementType.TABLE:
        return "table"
    if detection.element_type in {
        ElementType.SHAPE,
        ElementType.LINE,
        ElementType.RECTANGLE,
        ElementType.CIRCLE,
        ElementType.CHECKBOX,
    }:
        return "shape"
    return "other"


def postprocess_detections(
    detections: Sequence[Detection], page_width: int, page_height: int, settings: Settings
) -> tuple[list[Detection], list[str]]:
    """Clean raw detections: clip, drop noise, de-duplicate, resolve nesting."""
    warnings: list[str] = []
    cleaned: list[Detection] = []
    min_area = page_width * page_height * 0.0004

    for detection in detections:
        box = detection.bbox.clip(page_width, page_height)
        if box.width <= 1 or box.height <= 1:
            continue
        if box.area < min_area and detection.element_type is not ElementType.TABLE:
            continue
        detection.bbox = box
        cleaned.append(detection)

    # merge duplicates of the same class
    merged: list[Detection] = []
    for detection in sorted(cleaned, key=lambda d: -d.confidence):
        duplicate = False
        for existing in merged:
            if existing.element_type is not detection.element_type:
                continue
            iou = existing.bbox.iou(detection.bbox)
            contained = existing.bbox.overlap_fraction(detection.bbox) > 0.85
            if iou > 0.55 or contained:
                existing.bbox = BBox(
                    x1=min(existing.bbox.x1, detection.bbox.x1),
                    y1=min(existing.bbox.y1, detection.bbox.y1),
                    x2=max(existing.bbox.x2, detection.bbox.x2),
                    y2=max(existing.bbox.y2, detection.bbox.y2),
                )
                existing.confidence = max(existing.confidence, detection.confidence)
                duplicate = True
                break
        if not duplicate:
            merged.append(detection)

    # Different classes often claim the same region (a heading reported as both
    # "Title" and "Section-header"). Keep the most confident detection and record
    # the alternative label so nothing is silently discarded.
    cross_merged: list[Detection] = []
    for detection in sorted(merged, key=lambda d: -d.confidence):
        replaced = False
        for index, existing in enumerate(cross_merged):
            if _detection_kind(existing) != _detection_kind(detection):
                continue
            if existing.bbox.iou(detection.bbox) > 0.8:
                alternatives = list(existing.extra.get("also_detected_as") or [])
                prefer_detection = detection.confidence > existing.confidence + CONFIDENCE_TIE or (
                    abs(detection.confidence - existing.confidence) <= CONFIDENCE_TIE
                    and CLASS_PRIORITY.get(detection.element_type, 0)
                    > CLASS_PRIORITY.get(existing.element_type, 0)
                )
                if prefer_detection:
                    alternatives.append(existing.class_name)
                    detection.extra["also_detected_as"] = alternatives
                    cross_merged[index] = detection
                else:
                    alternatives.append(detection.class_name)
                    existing.extra["also_detected_as"] = alternatives
                replaced = True
                break
        if not replaced:
            cross_merged.append(detection)
    merged = cross_merged

    # drop text boxes that live inside a table/picture container
    containers = [d for d in merged if d.element_type in {ElementType.TABLE, ElementType.IMAGE}]
    result: list[Detection] = []
    for detection in merged:
        if detection.element_type in {ElementType.PARAGRAPH, ElementType.TEXT_BLOCK}:
            inside = next(
                (c for c in containers if c.bbox.overlap_fraction(detection.bbox) > 0.85), None
            )
            if inside is not None:
                if inside.element_type is ElementType.TABLE:
                    detection.extra["inside_table"] = True
                    result.append(detection)  # kept: used to read cell text
                    continue
                continue
        result.append(detection)

    # collapse repeated header/footer boxes into one each
    for target in (ElementType.HEADER, ElementType.FOOTER):
        group = [d for d in result if d.element_type is target]
        if len(group) > 1:
            kept = max(group, key=lambda d: d.confidence)
            kept.bbox = BBox(
                x1=min(d.bbox.x1 for d in group),
                y1=min(d.bbox.y1 for d in group),
                x2=max(d.bbox.x2 for d in group),
                y2=max(d.bbox.y2 for d in group),
            )
            result = [d for d in result if d is not kept and d.element_type is not target] + [kept]
            warnings.append(f"merged {len(group)} {target.value} regions into one")

    result.sort(key=lambda d: (round(d.bbox.y1, 1), round(d.bbox.x1, 1)))
    return result, warnings


class LayoutDetectorService:
    """Loads models once per process and detects page regions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._detectors: dict[str, YoloDetector] = {}
        self._lock = threading.Lock()

    def detector(self, spec: str, path: Path | None, **kwargs: Any) -> YoloDetector:
        with self._lock:
            if spec not in self._detectors:
                self._detectors[spec] = YoloDetector(spec, path, self.settings, **kwargs)
            return self._detectors[spec]

    def detect(self, image: np.ndarray) -> DetectionBundle:
        started = time.perf_counter()
        reports: list[ModelLoadReport] = []
        warnings: list[str] = []
        detections: list[Detection] = []
        detector_name = "none"

        candidates: list[tuple[str, Path | None]] = []
        if self.settings.yolo_custom_model is not None:
            candidates.append(("painter-custom", self.settings.yolo_custom_model))
        if self.settings.yolo_layout_model is not None:
            candidates.append(("doclayout", self.settings.yolo_layout_model))

        if self.settings.enable_yolo:
            for spec, path in candidates:
                detector = self.detector(spec, path)
                report = detector.load()
                reports.append(report)
                if not report.ok:
                    continue
                try:
                    detections = detector.predict(image)
                except DetectionError as exc:
                    warnings.append(str(exc))
                    continue
                if detections:
                    detector_name = f"{spec}:{report.loader}"
                    break
                warnings.append(f"{spec} returned no detections")

        used_fallback = False
        if not detections:
            fallback = CvFallbackDetector(self.settings)
            detections = fallback.predict(image)
            used_fallback = True
            detector_name = fallback.name
            warnings.append(
                "no YOLO layout model produced detections; using the OpenCV fallback "
                "(results are weaker and flagged for review)"
            )

        cleaned, extra_warnings = postprocess_detections(
            detections, image.shape[1], image.shape[0], self.settings
        )
        warnings.extend(extra_warnings)
        return DetectionBundle(
            detections=cleaned,
            detector=detector_name,
            model_reports=reports,
            warnings=warnings,
            elapsed_s=round(time.perf_counter() - started, 2),
            used_fallback=used_fallback,
        )


# --------------------------------------------------------------------------
# thin cv2 indirection so the module imports even if OpenCV is absent
# --------------------------------------------------------------------------
def _cv2() -> Any:
    import cv2

    return cv2


def cv2_resize(image: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return image
    cv2 = _cv2()
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))


def to_gray(image: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def cv2_get_structuring(width: int, height: int) -> np.ndarray:
    cv2 = _cv2()
    kernel_width = max(5, int(width * 0.015))
    return cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 5))


def cv2_dilate_kernel() -> np.ndarray:
    cv2 = _cv2()
    return cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def cv2_adaptive(gray: np.ndarray, invert: bool) -> np.ndarray:
    cv2 = _cv2()
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, 21, 11 if invert else 15)


def cv2_close(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return _cv2().morphologyEx(mask, _cv2().MORPH_CLOSE, kernel)


def cv2_dilate(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return _cv2().dilate(mask, kernel, iterations=1)


def cv2_contours(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    cv2 = _cv2()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours]


def iter_detections_by_type(
    detections: Iterable[Detection], *types: ElementType
) -> list[Detection]:
    wanted = set(types)
    return [d for d in detections if d.element_type in wanted]