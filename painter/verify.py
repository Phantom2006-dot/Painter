"""Visual verification: compare the reconstruction with the original scan.

The engine rasterises the rendered HTML, aligns it with the preprocessed page,
computes several independent metrics (never a single one), produces a difference
map and reports the regions and elements that differ most. A bounded correction
loop may then adjust *geometry only* — page offsets and a font scale factor —
and never touches recognised text.
"""

from __future__ import annotations

import difflib
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .config import Settings
from .errors import RasterizationError, VerificationError
from .geometry import mm_to_px
from .logging_utils import get_logger
from .schemas import DocumentIR, PageIR

logger = get_logger("verify")

CSS_PX_PER_MM = 96.0 / 25.4


def _cv2():
    import cv2

    return cv2


# --------------------------------------------------------------------------
# rasterisation
# --------------------------------------------------------------------------
def _chromium_path() -> str | None:
    for candidate in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


class Rasterizer:
    """Turns rendered HTML into a page image."""

    def __init__(self, dpi: int = 150, timeout_s: int = 240) -> None:
        self.dpi = dpi
        self.timeout_s = timeout_s
        self.last_engine: str | None = None
        self.warnings: list[str] = []

    def rasterize(
        self, html_path: Path, output_png: Path, page_width_mm: float, page_height_mm: float
    ) -> Path:
        html_path = Path(html_path)
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        scale = max(1.0, self.dpi / 96.0)
        css_w = page_width_mm * CSS_PX_PER_MM
        css_h = page_height_mm * CSS_PX_PER_MM
        # Chromium multiplies --window-size by --force-device-scale-factor, so the
        # window must be expressed in CSS pixels (1mm = 96/25.4 CSS px).
        window_w = int(math.ceil(css_w))
        window_h = int(math.ceil(css_h))

        errors: list[str] = []
        chromium = _chromium_path()
        if chromium:
            try:
                self._screenshot_chromium(chromium, html_path, output_png, window_w, window_h, scale)
                self.last_engine = "chromium"
                return output_png
            except Exception as exc:  # noqa: BLE001
                errors.append(f"chromium screenshot failed: {exc}")
        try:
            self._screenshot_playwright(html_path, output_png, css_w, css_h, scale)
            self.last_engine = "playwright"
            return output_png
        except Exception as exc:  # noqa: BLE001
            errors.append(f"playwright screenshot failed: {exc}")
        try:
            self._rasterize_via_pdf(html_path, output_png, page_width_mm, page_height_mm)
            self.last_engine = "weasyprint+pdftoppm"
            return output_png
        except Exception as exc:  # noqa: BLE001
            errors.append(f"pdf rasterisation failed: {exc}")

        self.warnings.extend(errors)
        raise RasterizationError("could not rasterise the rendered page", details={"errors": errors})

    def _screenshot_chromium(
        self,
        chromium: str,
        html_path: Path,
        output_png: Path,
        window_w: int,
        window_h: int,
        scale: float,
    ) -> None:
        cmd = [
            chromium,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--hide-scrollbars",
            "--disable-lcd-text",
            "--default-background-color=FFFFFFFF",
            f"--force-device-scale-factor={round(scale, 4)}",
            f"--window-size={window_w},{window_h}",
            "--virtual-time-budget=6000",
            f"--screenshot={output_png}",
            html_path.as_uri(),
        ]
        completed = subprocess.run(cmd, capture_output=True, timeout=self.timeout_s, check=False)
        if not output_png.is_file() or output_png.stat().st_size == 0:
            raise RasterizationError(
                completed.stderr.decode("utf-8", "replace")[:300] or "chromium produced no screenshot"
            )

    def _screenshot_playwright(
        self, html_path: Path, output_png: Path, css_w: float, css_h: float, scale: float
    ) -> None:
        from playwright.sync_api import sync_playwright

        executable = _chromium_path()
        with sync_playwright() as playwright:
            launch_kwargs: dict[str, Any] = {"args": ["--no-sandbox", "--disable-gpu"]}
            if executable:
                launch_kwargs["executable_path"] = executable
            browser = playwright.chromium.launch(**launch_kwargs)
            try:
                page = browser.new_page(
                    viewport={"width": int(math.ceil(css_w)), "height": int(math.ceil(css_h))},
                    device_scale_factor=scale,
                )
                page.goto(html_path.as_uri(), wait_until="load")
                page.wait_for_timeout(400)
                element = page.query_selector(".page")
                if element is not None:
                    element.screenshot(path=str(output_png))
                else:
                    page.screenshot(path=str(output_png), full_page=True)
            finally:
                browser.close()
        if not output_png.is_file() or output_png.stat().st_size == 0:
            raise RasterizationError("playwright produced no screenshot")

    def _rasterize_via_pdf(
        self, html_path: Path, output_png: Path, page_width_mm: float, page_height_mm: float
    ) -> None:
        if shutil.which("pdftoppm") is None:
            raise RasterizationError("pdftoppm is not available")
        markup = html_path.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "page.pdf"
            from weasyprint import HTML

            HTML(string=markup, base_url=str(html_path.parent)).write_pdf(str(pdf_path))
            completed = subprocess.run(
                [
                    "pdftoppm",
                    "-r",
                    str(self.dpi),
                    "-png",
                    "-singlefile",
                    "-f",
                    "1",
                    "-l",
                    "1",
                    str(pdf_path),
                    str(output_png.with_suffix("")),
                ],
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
            )
            if completed.returncode != 0:
                raise RasterizationError(
                    "pdftoppm failed: " + completed.stderr.decode("utf-8", "replace")[:200]
                )
        if not output_png.is_file():
            raise RasterizationError("pdftoppm produced no image")


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
@dataclass
class ComparisonMetrics:
    ssim: float = 0.0
    pixel_diff_ratio: float = 1.0
    edge_iou: float = 0.0
    projection_correlation: float = 0.0
    similarity: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ssim": round(self.ssim, 4),
            "pixel_diff_ratio": round(self.pixel_diff_ratio, 4),
            "edge_iou": round(self.edge_iou, 4),
            "projection_correlation": round(self.projection_correlation, 4),
            "similarity": round(self.similarity, 4),
            "weights": self.weights,
            "notes": self.notes,
        }


def _to_gray(image: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _normalise_pair(original: np.ndarray, rendered: np.ndarray, width: int = 800) -> tuple[np.ndarray, np.ndarray]:
    cv2 = _cv2()
    gray_a = _to_gray(original)
    gray_b = _to_gray(rendered)
    height_a = max(1, int(gray_a.shape[0] * width / gray_a.shape[1]))
    height_b = max(1, int(gray_b.shape[0] * width / gray_b.shape[1]))
    resized_a = cv2.resize(gray_a, (width, height_a), interpolation=cv2.INTER_AREA)
    resized_b = cv2.resize(gray_b, (width, height_b), interpolation=cv2.INTER_AREA)
    height = min(resized_a.shape[0], resized_b.shape[0])
    resized_a = resized_a[:height, :]
    resized_b = resized_b[:height, :]
    # normalise illumination and blur away scan texture so the comparison is fair
    resized_a = cv2.GaussianBlur(resized_a, (3, 3), 0)
    resized_b = cv2.GaussianBlur(resized_b, (3, 3), 0)
    return resized_a, resized_b


def _projection_correlation(first: np.ndarray, second: np.ndarray) -> float:
    def profile(image: np.ndarray, axis: int) -> np.ndarray:
        binary = (image < 160).astype(np.float64)
        return binary.sum(axis=axis)

    correlations: list[float] = []
    for axis in (1, 0):
        a = profile(first, axis)
        b = profile(second, axis)
        if a.std() < 1e-6 or b.std() < 1e-6:
            continue
        correlations.append(float(np.corrcoef(a, b)[0, 1]))
    if not correlations:
        return 0.0
    value = float(np.mean(correlations))
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


def compare_images(original: np.ndarray, rendered: np.ndarray) -> ComparisonMetrics:
    """Compute several independent similarity metrics."""
    cv2 = _cv2()
    metrics = ComparisonMetrics()
    try:
        from skimage.metrics import structural_similarity
    except Exception:  # pragma: no cover - optional dependency
        structural_similarity = None  # type: ignore[assignment]

    gray_a, gray_b = _normalise_pair(original, rendered)
    if gray_a.shape != gray_b.shape:
        metrics.notes.append("images differed in aspect ratio; comparison used the common area")

    if structural_similarity is not None:
        try:
            score, _ = structural_similarity(gray_a, gray_b, full=True, data_range=255)
            metrics.ssim = float(score)
        except Exception as exc:  # noqa: BLE001
            metrics.notes.append(f"ssim failed: {exc}")

    difference = cv2.absdiff(gray_a, gray_b)
    metrics.pixel_diff_ratio = float((difference > 45).mean())

    edges_a = cv2.dilate(cv2.Canny(gray_a, 60, 160), np.ones((3, 3), np.uint8), iterations=1) > 0
    edges_b = cv2.dilate(cv2.Canny(gray_b, 60, 160), np.ones((3, 3), np.uint8), iterations=1) > 0
    union = np.logical_or(edges_a, edges_b).sum()
    metrics.edge_iou = float(np.logical_and(edges_a, edges_b).sum() / union) if union else 0.0

    metrics.projection_correlation = _projection_correlation(gray_a, gray_b)

    weights = {
        "ssim": 0.35,
        "pixel_diff": 0.2,
        "edge_iou": 0.2,
        "projection": 0.25,
    }
    metrics.weights = weights
    metrics.similarity = round(
        weights["ssim"] * max(0.0, metrics.ssim)
        + weights["pixel_diff"] * (1.0 - metrics.pixel_diff_ratio)
        + weights["edge_iou"] * metrics.edge_iou
        + weights["projection"] * metrics.projection_correlation,
        4,
    )
    return metrics


def estimate_shift(original: np.ndarray, rendered: np.ndarray, max_shift_px: int = 40) -> tuple[int, int]:
    """Estimate the (dx, dy) pixel shift that best aligns the two pages."""
    cv2 = _cv2()
    gray_a, gray_b = _normalise_pair(original, rendered)
    a = (gray_a < 160).astype(np.float32)
    b = (gray_b < 160).astype(np.float32)
    row_a = a.sum(axis=1)
    row_b = b.sum(axis=1)
    col_a = a.sum(axis=0)
    col_b = b.sum(axis=0)

    def best_offset(reference: np.ndarray, target: np.ndarray) -> int:
        best, best_score = 0, -np.inf
        for shift in range(-max_shift_px, max_shift_px + 1):
            if shift >= 0:
                left, right = reference[shift:], target[: len(target) - shift]
            else:
                left, right = reference[: len(reference) + shift], target[-shift:]
            length = min(len(left), len(right))
            if length < 20:
                continue
            left, right = left[:length], right[:length]
            if left.std() < 1e-6 or right.std() < 1e-6:
                continue
            score = float(np.corrcoef(left, right)[0, 1])
            if score > best_score:
                best, best_score = shift, score
        return best

    dy = best_offset(row_a, row_b)
    dx = best_offset(col_a, col_b)
    return dx, dy


def build_diff_artifacts(
    original: np.ndarray, rendered: np.ndarray, out_dir: Path
) -> dict[str, str]:
    """Write side-by-side, difference-map and overlay images."""
    cv2 = _cv2()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gray_a, gray_b = _normalise_pair(original, rendered)
    height, width = gray_a.shape

    difference = cv2.absdiff(gray_a, gray_b)
    heat = cv2.applyColorMap(cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(gray_a, cv2.COLOR_GRAY2BGR), 0.55, heat, 0.45, 0)
    side_by_side = np.hstack(
        [cv2.cvtColor(gray_a, cv2.COLOR_GRAY2BGR), np.full((height, 6, 3), 255, np.uint8), cv2.cvtColor(gray_b, cv2.COLOR_GRAY2BGR)]
    )

    paths = {
        "difference_map": str(out_dir / "difference_map.png"),
        "overlay": str(out_dir / "overlay.png"),
        "side_by_side": str(out_dir / "side_by_side.png"),
    }
    cv2.imwrite(paths["difference_map"], heat)
    cv2.imwrite(paths["overlay"], overlay)
    cv2.imwrite(paths["side_by_side"], side_by_side)
    return paths


def region_issues(
    original: np.ndarray, rendered: np.ndarray, page: PageIR, grid: int = 6, top_n: int = 5
) -> list[str]:
    """Report the page regions (and elements) that differ most."""
    cv2 = _cv2()
    gray_a, gray_b = _normalise_pair(original, rendered)
    difference = cv2.absdiff(gray_a, gray_b).astype(np.float32)
    height, width = difference.shape
    cell_h = max(1, height // grid)
    cell_w = max(1, width // grid)
    scores: list[tuple[float, int, int]] = []
    for row in range(grid):
        for col in range(grid):
            cell = difference[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w]
            if cell.size == 0:
                continue
            scores.append((float(cell.mean()), row, col))
    scores.sort(reverse=True)

    issues: list[str] = []
    scale_x = page.width_px / width if width else 1.0
    scale_y = page.height_px / height if height else 1.0
    for score, row, col in scores[:top_n]:
        if score < 12.0:
            continue
        region = (
            col * cell_w * scale_x,
            row * cell_h * scale_y,
            (col + 1) * cell_w * scale_x,
            (row + 1) * cell_h * scale_y,
        )
        overlapping = [
            element
            for element in page.elements
            if element.bbox.x1 < region[2]
            and element.bbox.x2 > region[0]
            and element.bbox.y1 < region[3]
            and element.bbox.y2 > region[1]
        ]
        labels = ", ".join(f"{e.type.value}:{e.id}" for e in overlapping[:3]) or "empty area"
        issues.append(
            f"region row {row + 1}, column {col + 1} differs most (mean diff {score:.1f}); "
            f"elements there: {labels}"
        )
    return issues


def text_similarity(expected_text: str, rendered_image: np.ndarray) -> float | None:
    """OCR the rendered page and compare it with the reconstructed text."""
    try:
        import pytesseract
        import shutil as _shutil

        if _shutil.which("tesseract") is None:
            return None
        actual = pytesseract.image_to_string(rendered_image, lang="eng")
    except Exception:
        return None

    def normalise(value: str) -> str:
        return " ".join(value.split()).lower()

    a, b = normalise(expected_text), normalise(actual)
    if not a or not b:
        return None
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 4)


# --------------------------------------------------------------------------
# verification + bounded correction loop
# --------------------------------------------------------------------------
@dataclass
class VerificationResult:
    metrics: ComparisonMetrics
    artifacts: dict[str, str] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    iterations: list[dict[str, Any]] = field(default_factory=list)
    text_similarity: float | None = None
    applied_adjustments: dict[str, Any] = field(default_factory=dict)
    rasterizer_engine: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "artifacts": self.artifacts,
            "issues": self.issues,
            "iterations": self.iterations,
            "text_similarity": self.text_similarity,
            "applied_adjustments": self.applied_adjustments,
            "rasterizer": self.rasterizer_engine,
            "warnings": self.warnings,
        }


def verify_page(
    page: PageIR,
    document: DocumentIR,
    original_image: np.ndarray,
    html_path: Path,
    work_dir: Path,
    settings: Settings,
) -> VerificationResult:
    """Rasterise one rendered page and compare it with the original."""
    from .render.html_renderer import HtmlRenderOptions, render_document_for_print

    rasterizer = Rasterizer(dpi=settings.rasterize_dpi)
    page_png = Path(work_dir) / f"rendered_page_{page.index}.png"
    rasterizer.rasterize(html_path, page_png, page.page_format.width_mm, page.page_format.height_mm)
    rendered = _cv2().imread(str(page_png), _cv2().IMREAD_COLOR)
    if rendered is None:
        raise VerificationError("rasterised page could not be read back")

    metrics = compare_images(original_image, rendered)
    artifacts = build_diff_artifacts(original_image, rendered, Path(work_dir) / "artifacts")
    issues = region_issues(original_image, rendered, page)
    expected_text = "\n".join(element.text or "" for element in page.elements)
    similarity = text_similarity(expected_text, rendered)

    return VerificationResult(
        metrics=metrics,
        artifacts=artifacts,
        issues=issues,
        text_similarity=similarity,
        rasterizer_engine=rasterizer.last_engine,
        warnings=list(rasterizer.warnings),
    )


def correction_loop(
    document: DocumentIR,
    page: PageIR,
    original_image: np.ndarray,
    workspace_html: Path,
    work_dir: Path,
    settings: Settings,
    render_page_html: Callable[[dict[str, Any]], Path],
) -> VerificationResult:
    """Bounded geometry-only correction loop.

    Only three parameters are searched: a global millimetre offset (dx, dy) and a
    font scale factor. Text content is never modified — the loop cannot rewrite
    recognised text to improve a pixel score.
    """
    from .render.html_renderer import HtmlRenderOptions, render_document_for_print

    rasterizer = Rasterizer(dpi=settings.rasterize_dpi)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    current = {"dx_mm": 0.0, "dy_mm": 0.0, "font_scale": 1.0}
    iterations: list[dict[str, Any]] = []
    best_score = -1.0
    best_params = dict(current)
    warnings = list(rasterizer.warnings)

    def evaluate(params: dict[str, Any], label: str) -> tuple[float, Path, np.ndarray]:
        html_path = render_page_html(params)
        png_path = work_dir / f"iter_{len(iterations)}_{label}.png"
        rasterizer.rasterize(html_path, png_path, page.page_format.width_mm, page.page_format.height_mm)
        rendered = _cv2().imread(str(png_path), _cv2().IMREAD_COLOR)
        if rendered is None:
            raise VerificationError("rasterised iteration could not be read back")
        metrics = compare_images(original_image, rendered)
        iterations.append({"iteration": len(iterations), "label": label, **params, **metrics.to_dict()})
        return metrics.similarity, png_path, rendered

    # iteration 0: baseline
    score, baseline_png, baseline_rendered = evaluate(dict(current), "baseline")
    best_score = score
    best_png, best_rendered = baseline_png, baseline_rendered

    if score >= settings.verify_target_similarity:
        metrics = compare_images(original_image, best_rendered)
        return VerificationResult(
            metrics=metrics,
            artifacts=build_diff_artifacts(original_image, best_rendered, work_dir / "artifacts"),
            issues=region_issues(original_image, best_rendered, page),
            iterations=iterations,
            text_similarity=text_similarity(
                "\n".join(e.text or "" for e in page.elements), best_rendered
            ),
            applied_adjustments=dict(best_params),
            rasterizer_engine=rasterizer.last_engine,
            warnings=warnings,
        )

    # propose a shift from the profile cross-correlation of the baseline render
    dx_px, dy_px = estimate_shift(original_image, best_rendered)
    mm_x = page.mm_per_px_x
    mm_y = page.mm_per_px_y
    candidates: list[dict[str, Any]] = [
        {"dx_mm": round(-dx_px * mm_x * 0.75, 2), "dy_mm": round(-dy_px * mm_y * 0.75, 2), "font_scale": 1.0},
        {"dx_mm": 0.0, "dy_mm": round(-dy_px * mm_y, 2), "font_scale": 1.0},
        {"dx_mm": round(-dx_px * mm_x, 2), "dy_mm": 0.0, "font_scale": 1.0},
        {"dx_mm": 0.0, "dy_mm": 0.0, "font_scale": 0.97},
        {"dx_mm": 0.0, "dy_mm": 0.0, "font_scale": 1.03},
    ]

    iteration_budget = max(0, settings.verify_max_iterations)
    for index, candidate in enumerate(candidates[:iteration_budget]):
        merged = {
            "dx_mm": round(current["dx_mm"] + candidate["dx_mm"], 2),
            "dy_mm": round(current["dy_mm"] + candidate["dy_mm"], 2),
            "font_scale": round(candidate.get("font_scale", current["font_scale"]), 3),
        }
        try:
            candidate_score, png_path, rendered = evaluate(merged, f"cand{index}")
        except Exception as exc:  # noqa: BLE001 - a failed iteration must not abort the run
            warnings.append(f"correction iteration {index} failed: {exc}")
            continue
        if candidate_score > best_score + settings.verify_min_improvement:
            best_score = candidate_score
            best_params = merged
            current = merged
            best_png, best_rendered = png_path, rendered
        elif candidate_score > best_score:
            best_score = candidate_score
            best_params = merged
            best_png, best_rendered = png_path, rendered

    metrics = compare_images(original_image, best_rendered)
    issues = region_issues(original_image, best_rendered, page)
    if best_score < settings.verify_target_similarity:
        issues.append(
            f"similarity {best_score:.3f} is below the target "
            f"{settings.verify_target_similarity:.2f}; remaining differences are listed above"
        )
    return VerificationResult(
        metrics=metrics,
        artifacts=build_diff_artifacts(original_image, best_rendered, work_dir / "artifacts"),
        issues=issues,
        iterations=iterations,
        text_similarity=text_similarity("\n".join(e.text or "" for e in page.elements), best_rendered),
        applied_adjustments=dict(best_params),
        rasterizer_engine=rasterizer.last_engine,
        warnings=warnings,
    )
