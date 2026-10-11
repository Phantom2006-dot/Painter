"""Image preprocessing (OpenCV).

Responsibilities:

* validate and load uploads (images and multi-page PDFs),
* keep the original bytes untouched and store them separately,
* detect the document boundary, correct perspective, deskew and fix
  orientation,
* reduce noise/shadows and normalise contrast without destroying fine text,
  thin borders or signatures,
* normalise the resolution so downstream geometry is comparable,
* record every geometric operation as a :class:`PageTransform` so that every
  box can be mapped back to the original upload.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import Settings
from .errors import CorruptImageError, PageLimitExceededError, PreprocessingError, UnsupportedFormatError
from .logging_utils import get_logger
from .schemas import BBox, PageTransform

logger = get_logger("preprocess")

MAGIC_SIGNATURES: dict[bytes, str] = {
    b"\xff\xd8\xff": "jpg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"BM": "bmp",
    b"II*\x00": "tif",
    b"MM\x00*": "tif",
    b"RIFF": "webp",
}


@dataclass
class LoadedPage:
    """One page decoded from an upload (image or PDF page)."""

    index: int
    path: Path
    image: np.ndarray
    source_kind: str = "image"


@dataclass
class PreprocessResult:
    image: np.ndarray
    transforms: list[PageTransform] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    #: Stated assumptions (for example the inferred paper format). These are not
    #: errors, so they are kept separate from ``warnings``.
    assumptions: list[str] = field(default_factory=list)
    #: Physical page size used for the geometry, with the assumptions behind it.
    page_format: "PageFormat | None" = None


# --------------------------------------------------------------------------
# loading and validation
# --------------------------------------------------------------------------
def sniff_format(path: Path) -> str:
    """Determine the real container format from magic bytes."""
    header = path.open("rb").read(16)
    for signature, name in MAGIC_SIGNATURES.items():
        if header.startswith(signature):
            if name == "webp" and header[8:12] != b"WEBP":
                continue
            return name
    if header.startswith(b"%PDF"):
        return "pdf"
    return "unknown"


def validate_upload(path: Path, settings: Settings) -> str:
    """Validate an uploaded file, returning its detected format."""
    if not path.is_file():
        raise UnsupportedFormatError("uploaded file is missing")
    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise UnsupportedFormatError("uploaded file is empty")
    if size_bytes > settings.max_upload_mb * 1024 * 1024:
        raise UnsupportedFormatError(
            f"file is {size_bytes / 1e6:.1f} MB which exceeds the "
            f"{settings.max_upload_mb} MB limit"
        )
    detected = sniff_format(path)
    allowed = set(settings.allowed_image_formats) | set(settings.allowed_document_formats)
    if detected == "unknown" or detected not in allowed:
        raise UnsupportedFormatError(
            f"unsupported format {detected!r}; allowed: {sorted(allowed)}"
        )
    return detected


def _render_pdf(path: Path, workdir: Path, settings: Settings) -> list[LoadedPage]:
    """Rasterise PDF pages with poppler's ``pdftoppm`` (preinstalled)."""
    if shutil.which("pdftoppm") is None:
        raise PreprocessingError("pdftoppm (poppler-utils) is required for PDF input")
    workdir.mkdir(parents=True, exist_ok=True)
    prefix = workdir / "page"
    dpi = str(int(settings.target_dpi))
    cmd = ["pdftoppm", "-r", dpi, "-png", "-f", "1", "-l", str(settings.max_pages_per_document), str(path), str(prefix)]
    try:
        completed = subprocess.run(cmd, capture_output=True, timeout=300, check=False)
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - defensive
        raise PreprocessingError("PDF rasterisation timed out") from exc
    if completed.returncode != 0:
        raise PreprocessingError(
            f"pdftoppm failed: {completed.stderr.decode('utf-8', 'replace')[:300]}"
        )
    pages: list[LoadedPage] = []
    for index, page_path in enumerate(sorted(workdir.glob("page-*.png"))):
        image = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        pages.append(LoadedPage(index=index, path=page_path, image=image, source_kind="pdf"))
    if not pages:
        raise CorruptImageError("PDF produced no readable pages")
    if len(pages) >= settings.max_pages_per_document:
        raise PageLimitExceededError(
            f"PDF has at least {len(pages)} pages, limit is {settings.max_pages_per_document}"
        )
    return pages


def load_document_pages(path: Path, workdir: Path, settings: Settings) -> list[LoadedPage]:
    """Load every page of an upload as a BGR numpy array.

    The uploaded file is never modified; page images are written into
    ``workdir`` so the original bytes stay available for comparison.
    """
    detected = validate_upload(path, settings)
    if detected == "pdf":
        return _render_pdf(path, workdir, settings)

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise CorruptImageError("image could not be decoded (corrupt or unsupported encoding)")
    image = apply_exif_orientation(path, image)
    height, width = image.shape[:2]
    if width < 32 or height < 32:
        raise CorruptImageError(f"image is too small ({width}x{height}) to process")
    if width * height > settings.max_page_pixels * 4:
        raise PreprocessingError(
            f"image has {width * height / 1e6:.1f} megapixels which exceeds the configured limit"
        )
    workdir.mkdir(parents=True, exist_ok=True)
    stored = workdir / f"page-{0:02d}.png"
    cv2.imwrite(str(stored), image)
    return [LoadedPage(index=0, path=stored, image=image, source_kind="image")]


def apply_exif_orientation(path: Path, image: np.ndarray) -> np.ndarray:
    """Apply the EXIF orientation tag so phone photos are upright."""
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as handle:
            orientation = handle.getexif().get(274)
        if orientation in (None, 1):
            return image
        with Image.open(path) as handle:
            fixed = ImageOps.exif_transpose(handle)
            return cv2.cvtColor(np.array(fixed.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:  # pragma: no cover - PIL missing or unreadable EXIF
        return image


# --------------------------------------------------------------------------
# preprocessing steps
# --------------------------------------------------------------------------
def _order_quad(points: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(points, dtype="float32").reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def find_document_quad(image: np.ndarray, min_area_ratio: float = 0.25) -> np.ndarray | None:
    """Find a plausible page boundary for a photographed document."""
    height, width = image.shape[:2]
    scale = min(1.0, 1000.0 / max(height, width))
    small = cv2.resize(image, (int(width * scale), int(height * scale))) if scale < 1.0 else image
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = small.shape[0] * small.shape[1]
    best: tuple[float, np.ndarray] | None = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(contour)
        if area < frame_area * min_area_ratio:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        if best is None or area > best[0]:
            best = (area, approx)
    if best is None:
        return None
    return _order_quad(best[1] / scale)


def warp_perspective(image: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, PageTransform]:
    """Rectify a page quad and return the transform that produced it."""
    top_left, top_right, bottom_right, bottom_left = quad
    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    out_w = int(max(width_a, width_b))
    out_h = int(max(height_a, height_b))
    if out_w < 64 or out_h < 64:
        raise PreprocessingError("detected page quad is degenerate")
    destination = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32"
    )
    matrix = cv2.getPerspectiveTransform(quad.astype("float32"), destination)
    warped = cv2.warpPerspective(image, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC)
    transform = PageTransform(
        kind="perspective",
        params={"matrix": matrix.tolist(), "output_w": out_w, "output_h": out_h},
        note="document boundary detected and rectified",
    )
    return warped, transform


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate page skew in degrees using a text-line mask."""
    small = gray
    if max(gray.shape) > 1600:
        scale = 1600.0 / max(gray.shape)
        small = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)))
    binary = cv2.adaptiveThreshold(
        small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, small.shape[1] // 40), 3))
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    coords = cv2.findNonZero(joined)
    if coords is None or len(coords) < 200:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    return float(angle)


def deskew(image: np.ndarray, angle: float) -> tuple[np.ndarray, PageTransform]:
    """Rotate the page by ``-angle`` degrees about its centre."""
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a = abs(matrix[0, 0])
    sin_a = abs(matrix[0, 1])
    new_w = int(height * sin_a + width * cos_a)
    new_h = int(height * cos_a + width * sin_a)
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    rotated = cv2.warpAffine(
        image, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    transform = PageTransform(
        kind="deskew",
        params={
            "angle": angle,
            "center_x": center[0],
            "center_y": center[1],
            "new_center_x": new_w / 2.0,
            "new_center_y": new_h / 2.0,
        },
        note=f"deskewed by {angle:.2f} degrees",
    )
    return rotated, transform


def _row_projection_score(gray: np.ndarray) -> float:
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )
    projection = (binary > 0).sum(axis=1).astype(np.float64)
    if projection.size == 0 or projection.sum() == 0:
        return 0.0
    return float(projection.var() / (projection.mean() ** 2 + 1e-6))


def best_quarter_turn(gray: np.ndarray) -> int:
    """Return the quarter-turn rotation (0..3) that best aligns text rows."""
    scores = []
    for turns in range(4):
        rotated = np.rot90(gray, k=turns)
        scores.append(_row_projection_score(rotated))
    best = int(np.argmax(scores))
    ordered = sorted(scores, reverse=True)
    if len(ordered) > 1 and ordered[0] < ordered[1] * 1.35:
        return 0
    return best


def apply_quarter_turn(image: np.ndarray, turns: int) -> tuple[np.ndarray, PageTransform]:
    rotated = np.rot90(image, k=turns).copy()
    transform = PageTransform(
        kind="rotate",
        params={
            "quarter_turns": turns,
            "output_w": rotated.shape[1],
            "output_h": rotated.shape[0],
        },
        note=f"rotated {turns * 90} degrees to align text rows",
    )
    return rotated, transform


def reduce_shadow(image: np.ndarray) -> np.ndarray:
    """Flatten uneven illumination by dividing out an estimated background."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    kernel_size = max(15, (min(image.shape[:2]) // 15) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(lightness, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), sigmaX=max(3, kernel_size / 4))
    normalised = cv2.divide(lightness, background, scale=255)
    lab[:, :, 0] = normalised
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise(image: np.ndarray) -> np.ndarray:
    if max(image.shape[:2]) > 3000:
        return cv2.bilateralFilter(image, 5, 40, 40)
    return cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def ink_ratio(gray: np.ndarray) -> float:
    """Fraction of dark pixels — used to detect destructive transformations."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float((binary > 0).mean())


def normalize_resolution(
    image: np.ndarray, page_format_width_mm: float, page_format_height_mm: float, settings: Settings
) -> tuple[np.ndarray, PageTransform | None, float]:
    """Scale the page towards the target DPI, within safe bounds."""
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise PreprocessingError("empty image")
    mm_per_inch = 25.4
    current_dpi_x = width / max(page_format_width_mm / mm_per_inch, 1e-6)
    current_dpi_y = height / max(page_format_height_mm / mm_per_inch, 1e-6)
    current_dpi = (current_dpi_x + current_dpi_y) / 2.0
    target = settings.target_dpi
    scale = target / current_dpi if current_dpi > 1 else 1.0
    # Never destroy detail: cap upscaling and avoid needless downscaling.
    scale = max(0.5, min(scale, 2.0))
    if abs(scale - 1.0) < 0.08:
        return image, None, current_dpi
    max_pixels = settings.max_page_pixels
    if width * height * scale * scale > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
    if scale <= 0:
        return image, None, current_dpi
    new_size = (max(32, int(width * scale)), max(32, int(height * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    transform = PageTransform(
        kind="resize",
        params={"scale_x": new_size[0] / width, "scale_y": new_size[1] / height, "output_w": new_size[0], "output_h": new_size[1]},
        note=f"resolution normalised to ~{target} dpi",
    )
    return resized, transform, current_dpi


def preprocess_page(
    image: np.ndarray,
    settings: Settings,
    page_format_width_mm: float | None = None,
    page_format_height_mm: float | None = None,
    is_photograph: bool | None = None,
) -> PreprocessResult:
    """Run the full preprocessing chain on one page image.

    Every step is guarded: if an operation fails or looks destructive it is
    skipped and a warning is recorded instead, because losing thin borders or a
    signature is worse than a slightly noisy page.
    """
    if image is None or image.size == 0:
        raise CorruptImageError("empty page image")
    transforms: list[PageTransform] = []
    warnings: list[str] = []
    assumptions: list[str] = []
    metrics: dict[str, float] = {}

    height, width = image.shape[:2]
    metrics["input_width_px"] = float(width)
    metrics["input_height_px"] = float(height)

    gray_before = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    metrics["ink_ratio_before"] = round(ink_ratio(gray_before), 4)
    measured_line_pitch = estimate_text_line_pitch(gray_before)
    if measured_line_pitch:
        metrics["line_pitch_px"] = round(measured_line_pitch, 1)

    # 1. orientation
    try:
        turns = best_quarter_turn(gray_before)
        if turns:
            image, transform = apply_quarter_turn(image, turns)
            transforms.append(transform)
            warnings.append(f"page rotated by {turns * 90} degrees based on text-row alignment")
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"orientation detection failed: {exc}")

    # 2. document boundary + perspective (only for photographs / when a quad is found)
    try:
        quad = find_document_quad(image)
        if quad is not None:
            area = cv2.contourArea(quad.astype("float32"))
            frame_area = float(image.shape[0] * image.shape[1])
            ratio = area / frame_area if frame_area else 0.0
            if 0.25 <= ratio <= 0.995:
                warped, transform = warp_perspective(image, quad)
                if warped.shape[0] > 200 and warped.shape[1] > 200:
                    image = warped
                    transforms.append(transform)
                    metrics["boundary_area_ratio"] = round(ratio, 4)
            else:
                metrics["boundary_area_ratio"] = round(ratio, 4)
    except Exception as exc:
        warnings.append(f"document boundary detection skipped: {exc}")

    # 3. deskew
    try:
        angle = estimate_skew_angle(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        metrics["skew_angle_deg"] = round(angle, 3)
        if 0.2 <= abs(angle) <= 15.0:
            image, transform = deskew(image, angle)
            transforms.append(transform)
    except Exception as exc:
        warnings.append(f"deskew skipped: {exc}")

    # 4. illumination, noise and contrast (photographs benefit most)
    if is_photograph is None:
        is_photograph = bool(transforms) or _looks_like_photograph(gray_before)
    if is_photograph:
        try:
            image = reduce_shadow(image)
        except Exception as exc:
            warnings.append(f"shadow reduction skipped: {exc}")
    try:
        image = denoise(image)
    except Exception as exc:
        warnings.append(f"denoising skipped: {exc}")
    try:
        image = enhance_contrast(image)
    except Exception as exc:
        warnings.append(f"contrast enhancement skipped: {exc}")

    # 5. resolution normalisation
    try:
        if page_format_width_mm and page_format_height_mm:
            page_w, page_h = page_format_width_mm, page_format_height_mm
            page_format_value = PageFormat(
                name="custom",
                width_mm=round(page_w, 2),
                height_mm=round(page_h, 2),
                estimated=True,
                notes=["page size supplied by the caller"],
            )
        else:
            from .geometry import infer_page_format

            inferred, inferred_dpi, format_notes = infer_page_format(
                width, height, line_height_px=measured_line_pitch
            )
            page_w, page_h = inferred.width_mm, inferred.height_mm
            page_format_value = inferred
            metrics["assumed_input_dpi"] = inferred_dpi
            assumptions.extend(format_notes)
        image, transform, detected_dpi = normalize_resolution(image, page_w, page_h, settings)
        metrics["estimated_input_dpi"] = round(detected_dpi, 1)
        if transform is not None:
            transforms.append(transform)
        # DPI of the image that downstream stages actually analyse.
        output_width = max(1, image.shape[1])
        metrics["output_dpi"] = round(output_width / max(page_w / 25.4, 1e-6), 1)
        metrics["page_width_mm"] = round(page_w, 2)
        metrics["page_height_mm"] = round(page_h, 2)
    except Exception as exc:
        warnings.append(f"resolution normalisation skipped: {exc}")

    gray_after = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    metrics["ink_ratio_after"] = round(ink_ratio(gray_after), 4)
    before = metrics.get("ink_ratio_before", 0.0)
    after = metrics.get("ink_ratio_after", 0.0)
    if before > 0.005 and after < before * 0.45:
        warnings.append(
            "preprocessing reduced ink coverage substantially; verify that thin lines and text survived"
        )
    metrics["output_width_px"] = float(image.shape[1])
    metrics["output_height_px"] = float(image.shape[0])

    return PreprocessResult(
        image=image,
        transforms=transforms,
        warnings=warnings,
        metrics=metrics,
        assumptions=assumptions,
        page_format=page_format_value,
    )


def _looks_like_photograph(gray: np.ndarray) -> bool:
    """Heuristic: photos have noisy backgrounds and wide intensity spread."""
    small = gray if max(gray.shape) <= 800 else cv2.resize(gray, (800, int(800 * gray.shape[0] / gray.shape[1])))
    laplacian = cv2.Laplacian(small, cv2.CV_64F).var()
    background = np.percentile(small, 90)
    return bool(laplacian > 400 and background < 250)


def _guess_format(width: int, height: int) -> tuple[float, float]:
    from .geometry import PAGE_FORMATS, detect_page_format

    fmt = detect_page_format(width, height)
    if fmt.name in PAGE_FORMATS:
        return PAGE_FORMATS[fmt.name]
    return (fmt.width_mm, fmt.height_mm)


def crop_box(image: np.ndarray, box: BBox, padding: int = 0) -> np.ndarray:
    """Crop ``box`` from ``image`` with optional padding, clipped to bounds."""
    height, width = image.shape[:2]
    padded = box.pad(padding, width, height) if padding else box.clip(width, height)
    x1, y1 = int(max(0, np.floor(padded.x1))), int(max(0, np.floor(padded.y1)))
    x2, y2 = int(min(width, np.ceil(padded.x2))), int(min(height, np.ceil(padded.y2)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=image.dtype)
    return image[y1:y2, x1:x2].copy()


def encode_jpeg(image: np.ndarray, quality: int = 90, max_bytes: int | None = None) -> bytes:
    """Encode to JPEG, reducing quality until ``max_bytes`` is respected."""
    quality = int(quality)
    while quality >= 30:
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise PreprocessingError("JPEG encoding failed")
        data = buffer.tobytes()
        if max_bytes is None or len(data) <= max_bytes:
            return data
        quality -= 15
    raise PreprocessingError("image could not be encoded under the size limit")


def estimate_text_line_pitch(gray: np.ndarray, max_pitch: int = 240) -> float | None:
    """Estimate the body-text line pitch in pixels.

    The pitch divided by the assumed DPI gives a body-text size, which is used to
    sanity check the assumed paper format. The estimate comes from the
    autocorrelation of the row-ink profile: unlike counting text bands, it is not
    confused by a line splitting into several ink bands (descenders, table
    rules), because a periodic signal peaks at its true period.
    """
    try:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
        )
    except Exception:
        return None
    rows = (binary > 0).sum(axis=1).astype(np.float64)
    if rows.size < 64 or rows.max() < 5:
        return None
    profile = rows - rows.mean()
    if float(np.abs(profile).sum()) <= 0:
        return None
    size = int(2 ** np.ceil(np.log2(rows.size * 2)))
    spectrum = np.fft.rfft(profile, size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum))[: rows.size]
    if correlation[0] <= 0:
        return None
    correlation = correlation / correlation[0]

    low = 6
    high = int(min(rows.size - 2, max_pitch))
    if high <= low:
        return None
    # the first local maximum above the noise floor is the fundamental period,
    # not one of its harmonics
    for lag in range(low + 1, high):
        if (
            correlation[lag] > correlation[lag - 1]
            and correlation[lag] >= correlation[lag + 1]
            and correlation[lag] > 0.18
        ):
            return float(lag)
    best = int(np.argmax(correlation[low:high])) + low
    return float(best) if correlation[best] > 0.12 else None
