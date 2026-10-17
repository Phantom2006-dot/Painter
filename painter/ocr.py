"""Text recognition providers.

The pipeline talks to OCR through :class:`OcrProvider` so engines can be swapped
without touching the rest of the system. Two providers ship with Painter:

``TesseractProvider``
    Local, deterministic, line-accurate OCR. This is the default because it
    produces real line boxes and costs nothing per call.
``SerpApiLensProvider``
    Google Lens via SerpApi. Excellent for *identifying images* and useful as a
    secondary text hint, but its text output is derived from search results, so
    the provider marks every result as low-confidence and never claims to be
    line-accurate OCR.

The rule from the brief — "do not fabricate text when OCR fails" — is enforced
by returning an empty result plus a flag instead of inventing content.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence

import numpy as np

from .config import Settings
from .errors import OcrError, OcrUnavailableError
from .logging_utils import get_logger
from .schemas import BBox, ReviewFlag

logger = get_logger("ocr")


@dataclass
class OcrLine:
    text: str
    bbox: BBox
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox.as_list(),
            "confidence": None if self.confidence is None else round(self.confidence, 4),
        }


@dataclass
class OcrResult:
    text: str = ""
    lines: list[OcrLine] = field(default_factory=list)
    confidence: float | None = None
    provider: str = "none"
    method: str = "none"
    flags: list[ReviewFlag] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "lines": [line.to_dict() for line in self.lines],
            "confidence": None if self.confidence is None else round(self.confidence, 4),
            "provider": self.provider,
            "method": self.method,
            "flags": [f.value for f in self.flags],
            "warnings": self.warnings,
        }


class OcrProvider(Protocol):
    name: str

    def available(self) -> bool: ...

    def recognize(self, image: np.ndarray, *, offset: tuple[float, float] = (0.0, 0.0)) -> OcrResult: ...


# --------------------------------------------------------------------------
# tesseract
# --------------------------------------------------------------------------
class TesseractProvider:
    """Local OCR using the ``tesseract`` binary through pytesseract."""

    name = "tesseract"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pytesseract: Any = None
        self._import_error: str | None = None
        try:
            import pytesseract

            self._pytesseract = pytesseract
        except Exception as exc:  # pragma: no cover - optional dependency
            self._import_error = str(exc)

    def available(self) -> bool:
        if self._pytesseract is None:
            return False
        if shutil.which("tesseract") is None:
            return False
        try:
            version = self._pytesseract.get_tesseract_version()
            return bool(version)
        except Exception:
            return False

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        import cv2

        crop = image
        height, width = crop.shape[:2]
        longest = max(height, width)
        if self.settings.ocr_upscale_small_crops and longest < 900:
            factor = min(3.0, 900.0 / max(longest, 1))
            crop = cv2.resize(crop, (int(width * factor), int(height * factor)), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if gray.size and float(gray.mean()) < 120:  # light text on dark background
            gray = cv2.bitwise_not(gray)
        if max(gray.shape) > 2200:
            scale = 2200.0 / max(gray.shape)
            gray = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)))
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )

    def recognize(
        self,
        image: np.ndarray,
        *,
        offset: tuple[float, float] = (0.0, 0.0),
        psm: int | None = None,
    ) -> OcrResult:
        if not self.available():
            raise OcrUnavailableError(
                "tesseract is not available"
                + (f" ({self._import_error})" if self._import_error else "")
            )
        if image is None or image.size == 0:
            return OcrResult(provider=self.name, method="tesseract", warnings=["empty crop"])
        import cv2

        prepared = self._prepare(image)
        config = f"--oem 3 --psm {psm if psm is not None else 6}"
        try:
            data = self._pytesseract.image_to_data(
                prepared,
                lang=self.settings.tesseract_lang,
                config=config,
                output_type=self._pytesseract.Output.DICT,
            )
        except Exception as exc:  # pragma: no cover - binary failure
            raise OcrError(f"tesseract failed: {exc}") from exc

        scale_y = image.shape[0] / prepared.shape[0] if prepared.shape[0] else 1.0
        scale_x = image.shape[1] / prepared.shape[1] if prepared.shape[1] else 1.0

        lines: dict[tuple[int, int, int], dict[str, Any]] = {}
        confidences: list[float] = []
        total = len(data.get("text", []))
        for index in range(total):
            word = str(data["text"][index] or "").strip()
            if not word:
                continue
            try:
                confidence = float(data["conf"][index])
            except (TypeError, ValueError):
                confidence = -1.0
            if confidence < 0:
                continue
            key = (int(data["block_num"][index]), int(data["par_num"][index]), int(data["line_num"][index]))
            x = float(data["left"][index]) * scale_x + offset[0]
            y = float(data["top"][index]) * scale_y + offset[1]
            w = float(data["width"][index]) * scale_x
            h = float(data["height"][index]) * scale_y
            entry = lines.setdefault(key, {"words": [], "boxes": [], "conf": []})
            entry["words"].append(word)
            entry["boxes"].append(BBox(x1=x, y1=y, x2=x + w, y2=y + h))
            entry["conf"].append(confidence)
            confidences.append(confidence)

        ocr_lines: list[OcrLine] = []
        for key in sorted(lines):
            entry = lines[key]
            boxes = entry["boxes"]
            box = BBox(
                x1=min(b.x1 for b in boxes),
                y1=min(b.y1 for b in boxes),
                x2=max(b.x2 for b in boxes),
                y2=max(b.y2 for b in boxes),
            )
            ocr_lines.append(
                OcrLine(
                    text=" ".join(entry["words"]),
                    bbox=box,
                    confidence=round(sum(entry["conf"]) / len(entry["conf"]) / 100.0, 4),
                )
            )
        ocr_lines.sort(key=lambda line: (line.bbox.y1, line.bbox.x1))
        text = "\n".join(line.text for line in ocr_lines).strip()
        mean_confidence = round(sum(confidences) / len(confidences) / 100.0, 4) if confidences else None

        flags: list[ReviewFlag] = []
        warnings: list[str] = []
        if not text:
            flags.append(ReviewFlag.OCR_FAILED)
            warnings.append("tesseract recognised no text in this region")
        elif mean_confidence is not None and mean_confidence < 0.6:
            flags.append(ReviewFlag.OCR_UNCERTAIN)
            warnings.append(f"low mean OCR confidence ({mean_confidence:.2f})")

        return OcrResult(
            text=text,
            lines=ocr_lines,
            confidence=mean_confidence,
            provider=self.name,
            method=f"tesseract-psm{psm if psm is not None else 6}",
            flags=flags,
            warnings=warnings,
        )


# --------------------------------------------------------------------------
# google lens via serpapi
# --------------------------------------------------------------------------
class SerpApiLensProvider:
    """Google Lens text as a *hint*, never as authoritative OCR."""

    name = "serpapi_lens"

    def __init__(self, client: Any, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def available(self) -> bool:
        return bool(self.client is not None and getattr(self.client, "usable", False))

    def recognize(self, image: np.ndarray, *, offset: tuple[float, float] = (0.0, 0.0)) -> OcrResult:
        from .preprocess import encode_jpeg

        if not self.available():
            raise OcrUnavailableError("SerpApi Google Lens is not enabled or not configured")
        jpeg = encode_jpeg(image, quality=90, max_bytes=self.settings.serpapi.upload_max_bytes)
        payload = self.client.lens_search(jpeg)
        entries = self.client.parse_text_results(payload)
        text = "\n".join(entry["text"] for entry in entries).strip()
        flags = [ReviewFlag.OCR_UNCERTAIN, ReviewFlag.LOW_CONFIDENCE]
        warnings = [
            "text derived from Google Lens search results; treat as a hint, not verified OCR"
        ]
        if not text:
            flags.append(ReviewFlag.OCR_FAILED)
            warnings.append("Google Lens returned no text for this crop")
        return OcrResult(
            text=text,
            lines=[],
            confidence=0.35 if text else None,
            provider=self.name,
            method="google_lens_search_text",
            flags=flags,
            warnings=warnings,
            raw={
                "has_text_results": self.client.has_text_results(payload),
                "origins": sorted({entry["origin"] for entry in entries}),
            },
        )


class NullProvider:
    """Explicit no-op provider so callers never special-case ``None``."""

    name = "none"

    def available(self) -> bool:
        return True

    def recognize(self, image: np.ndarray, *, offset: tuple[float, float] = (0.0, 0.0)) -> OcrResult:
        return OcrResult(
            provider=self.name,
            method="disabled",
            flags=[ReviewFlag.OCR_FAILED],
            warnings=["no OCR provider is enabled"],
        )


# --------------------------------------------------------------------------
# registry / chain
# --------------------------------------------------------------------------
class OcrChain:
    """Try providers in order and keep the best non-empty result."""

    def __init__(self, providers: Sequence[OcrProvider]) -> None:
        self.providers = list(providers)
        self.usage: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "+".join(p.name for p in self.providers) or "none"

    def available(self) -> bool:
        """True only when a provider that can actually read text is configured.

        ``NullProvider`` reports itself as usable so the chain can return its
        explanatory "OCR disabled" result, but it must not make a disabled chain
        look available in ``/health``.
        """
        return any(
            provider.available() and not isinstance(provider, NullProvider)
            for provider in self.providers
        )

    def recognize(self, image: np.ndarray, *, offset: tuple[float, float] = (0.0, 0.0)) -> OcrResult:
        warnings: list[str] = []
        fallback_result: OcrResult | None = None
        for provider in self.providers:
            if not provider.available():
                continue
            try:
                result = provider.recognize(image, offset=offset)
            except OcrUnavailableError as exc:
                warnings.append(f"{provider.name} unavailable: {exc}")
                continue
            except OcrError as exc:
                warnings.append(f"{provider.name} failed: {exc}")
                continue
            self.usage[provider.name] = self.usage.get(provider.name, 0) + 1
            if not result.empty:
                result.warnings.extend(warnings)
                return result
            if fallback_result is None:
                fallback_result = result
        if fallback_result is not None:
            fallback_result.warnings.extend(warnings)
            return fallback_result
        return OcrResult(
            provider="none",
            method="no-provider",
            flags=[ReviewFlag.OCR_FAILED],
            warnings=warnings + ["no OCR provider could process this region"],
        )


def build_ocr_chain(settings: Settings, serpapi_client: Any | None = None) -> OcrChain:
    """Create the provider chain for the configured mode.

    ``auto`` prefers local Tesseract and only adds Google Lens when the operator
    explicitly enabled SerpApi, keeping API cost predictable.
    """
    mode = (settings.ocr_provider or "auto").lower()
    providers: list[OcrProvider] = []
    tesseract = TesseractProvider(settings)
    lens = SerpApiLensProvider(serpapi_client, settings) if serpapi_client is not None else None

    if mode == "none":
        return OcrChain([NullProvider()])
    if mode == "tesseract":
        providers.append(tesseract)
    elif mode == "lens":
        if lens is not None:
            providers.append(lens)
        providers.append(tesseract)
    else:  # auto
        if tesseract.available():
            providers.append(tesseract)
        if lens is not None and lens.available():
            providers.append(lens)
        if not providers:
            providers.append(NullProvider())
    return OcrChain(providers)


def recognize_regions(
    chain: OcrChain,
    image: np.ndarray,
    regions: Iterable[tuple[str, BBox]],
    *,
    padding: int = 2,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, OcrResult]:
    """Run OCR for ``(element_id, bbox)`` pairs and return results by id."""
    from .preprocess import crop_box

    region_list = list(regions)
    total = len(region_list)
    results: dict[str, OcrResult] = {}
    for position, (element_id, box) in enumerate(region_list, start=1):
        # Reported first, so every region is counted exactly once no matter which
        # path it takes afterwards. This loop is the slowest part of a dense page.
        if on_progress:
            on_progress(position, total, element_id)
        crop = crop_box(image, box, padding=padding)
        if crop.shape[0] < 3 or crop.shape[1] < 3:
            results[element_id] = OcrResult(
                provider="none",
                method="skipped",
                flags=[ReviewFlag.OCR_FAILED],
                warnings=["region too small to OCR"],
            )
            continue
        try:
            results[element_id] = chain.recognize(crop, offset=(box.x1 - padding, box.y1 - padding))
        except OcrError as exc:
            results[element_id] = OcrResult(
                provider="none",
                method="error",
                flags=[ReviewFlag.OCR_FAILED],
                warnings=[str(exc)],
            )
    return results
