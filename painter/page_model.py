"""The trained page-corner detector, with the heuristic as a safety net.

The model (``data/models/page_corners.pt``, YOLO11n-pose, four page corners) was
trained on SmartDoc's real frames and on synthetic captures composited onto real
surfaces. It reaches 0.995 box mAP50 and 0.919 pose mAP50 on a validation split
that shares no document with training.

It is deliberately not a drop-in replacement, because of what was measured on the
operator's own photograph of a certificate on a concrete floor:

===========================  ===========================================
trained model                no detection at any sane confidence; 0.067 at
                             the loosest threshold, with a wrong quad shape
brightness heuristic         a correct, axis-aligned page quad
===========================  ===========================================

On synthetic captures from the same domain the model is accurate to 1-3% of the
image diagonal, so it has learned "page on a floor" but not yet the artefacts of a
real camera: true optics, sensor noise and the particular lighting of that photo.
Until real captures are in the training set, the honest arrangement is to let the
model lead only when it is confident and otherwise defer to the heuristic that is
known to work. The confidence gate is what makes that safe: a wrong detection
would rectify the page using the wrong quad and damage every later stage, whereas
a fallback costs only the accuracy the heuristic already had.

Retrain with real captures via ``painter.cli training-data`` and this gate can be
relaxed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .logging_utils import get_logger

logger = get_logger("page_model")

#: Below this the model's quad is not trusted and the heuristic is used instead.
#: Chosen from measurement: synthetic captures score above 0.9, the real
#: photograph scored 0.067, so anything in between is genuinely uncertain.
MIN_CONFIDENCE = 0.50

#: Where the trained weights live inside a workspace.
DEFAULT_WEIGHTS = Path("data/models/page_corners.pt")

#: The heuristic's own test of whether a quad can be a page, applied to the
#: model's answer too. A geometry-only guard was not enough: on a clean generated
#: page the model returned a confident, skewed quad covering 88% of the frame, and
#: warping by it moved text that needed no correction. What that test adds is the
#: contrast between the quad's inside and outside, which is what a page on a
#: surface has and a page that *is* the frame does not.
MIN_PAGE_SCORE = 0.01


class PageCornerModel:
    """Loads the trained corner detector once and keeps it for reuse."""

    def __init__(self, weights: Path | str | None = None, device: str = "cpu") -> None:
        self.weights = Path(weights) if weights else DEFAULT_WEIGHTS
        self.device = device
        self._model = None
        self._failed = False

    @property
    def available(self) -> bool:
        return self.weights.is_file() and not self._failed

    def _load(self):
        if self._model is None and not self._failed:
            try:
                from ultralytics import YOLO

                self._model = YOLO(str(self.weights))
            except Exception as exc:  # a missing model must not break preprocessing
                logger.warning("page corner model unavailable: %s", exc)
                self._failed = True
        return self._model

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, float] | None:
        """Return ``(quad, confidence)`` in top-left, top-right, bottom-right, bottom-left order."""
        model = self._load()
        if model is None:
            return None
        try:
            results = model.predict(image, imgsz=384, device=self.device, conf=0.05, verbose=False)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("page corner inference failed: %s", exc)
            return None
        result = results[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return None
        confidence = float(result.boxes.conf[0])
        quad = result.keypoints.xy[0].cpu().numpy().astype(np.float64)
        if quad.shape != (4, 2):
            return None
        return quad, confidence


def detect_page_quad(image: np.ndarray, model: PageCornerModel | None = None) -> np.ndarray | None:
    """Find the page: the trained model when confident, the heuristic otherwise."""
    if model is None:
        model = PageCornerModel()
    if model.available:
        detected = model.detect(image)
        if detected is not None:
            quad, confidence = detected
            if confidence >= MIN_CONFIDENCE:
                if _looks_like_a_page(quad, image):
                    logger.info("page quad from trained model (confidence %.3f)", confidence)
                    return quad
                logger.info(
                    "page model quad (confidence %.3f) failed the page test; "
                    "using the heuristic",
                    confidence,
                )
            else:
                logger.info(
                    "page model confidence %.3f below %.2f; using the heuristic",
                    confidence,
                    MIN_CONFIDENCE,
                )
    from .preprocess import find_document_quad

    return find_document_quad(image)


def _looks_like_a_page(quad: np.ndarray, image: np.ndarray) -> bool:
    """Ask the heuristic's plausibility test whether this quad is a page."""
    import cv2

    from .preprocess import _quad_is_page

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        return _quad_is_page(quad, gray, 0.2) >= MIN_PAGE_SCORE
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("page plausibility test failed: %s", exc)
        return True
