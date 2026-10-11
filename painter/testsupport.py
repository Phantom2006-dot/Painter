"""Synthetic test documents, the self-test and the benchmark harness.

The project needs legally usable test material with known ground truth, so this
module *generates* documents instead of downloading them: the exact text is
known, which makes it possible to measure OCR accuracy honestly. Photographed
variants add rotation, shading and noise to exercise the preprocessing chain.
"""

from __future__ import annotations

import difflib
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import Settings, get_settings
from .logging_utils import get_logger

logger = get_logger("testsupport")

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

TITLE_TEXT = "ANNUAL FINANCIAL REPORT"
PARAGRAPH_TEXT = (
    "This document summarises the financial performance of the company for the "
    "financial year. Revenue grew across all operating segments while operating "
    "costs remained broadly stable. The board recommends the payment of a final "
    "dividend to shareholders of record."
)
TABLE_HEADERS = ["Item", "Amount", "Status"]
TABLE_ROWS = [
    ["Consulting", "1250000", "Paid"],
    ["Licences", "480000", "Pending"],
    ["Support", "96500", "Paid"],
]
FOOTER_TEXT = "Page 1 of 1"
LOGO_TEXT = "ACME"


@dataclass
class SyntheticCase:
    name: str
    image_path: Path
    ground_truth_text: str
    expectations: dict[str, Any] = field(default_factory=dict)
    is_photograph: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "image_path": str(self.image_path),
            "is_photograph": self.is_photograph,
            "expectations": self.expectations,
        }


def _font(size: int, bold: bool = False) -> Any:
    from PIL import ImageFont

    ordered = FONT_CANDIDATES[:: 2 if bold else 1] if bold else FONT_CANDIDATES
    for path in ordered:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _canvas(width: int = 1654, height: int = 2339) -> Any:
    """A4 at 200 dpi, white background."""
    from PIL import Image

    return Image.new("RGB", (width, height), (255, 255, 255))


def _draw_paragraph(draw: Any, text: str, x: int, y: int, width: int, size: int = 26, leading: int = 38) -> int:
    font = _font(size)
    words = text.split()
    line = ""
    lines: list[str] = []
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textlength(candidate, font=font) > width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    for index, content in enumerate(lines):
        draw.text((x, y + index * leading), content, fill=(20, 20, 20), font=font)
    return y + len(lines) * leading


def _draw_table(draw: Any, x: int, y: int, width: int, headers: list[str], rows: list[list[str]]) -> int:
    font = _font(24)
    bold = _font(24, bold=True)
    row_height = 46
    col_width = width // len(headers)
    all_rows = [headers] + rows
    for row_index, row in enumerate(all_rows):
        top = y + row_index * row_height
        for col_index, cell in enumerate(row):
            left = x + col_index * col_width
            draw.rectangle([left, top, left + col_width, top + row_height], outline=(40, 40, 40), width=2)
            draw.text((left + 10, top + 11), cell, fill=(15, 15, 15), font=bold if row_index == 0 else font)
    return y + len(all_rows) * row_height


def _draw_picture(draw: Any, x: int, y: int, width: int, height: int) -> None:
    for offset in range(0, height, 4):
        shade = int(60 + 150 * offset / max(height, 1))
        draw.rectangle([x, y + offset, x + width, y + offset + 4], fill=(shade, int(shade * 0.7), 90))
    draw.rectangle([x, y, x + width, y + height], outline=(30, 30, 30), width=2)


def build_synthetic_document(
    out_dir: Path,
    case: str = "mixed",
    seed: int = 7,
    photograph: bool = False,
    dpi_scale: float = 1.0,
) -> SyntheticCase:
    """Render one synthetic page with known ground truth."""
    from PIL import Image, ImageDraw, ImageFilter

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = _canvas()
    draw = ImageDraw.Draw(image)
    width, _ = image.size

    texts: list[str] = []
    expectations: dict[str, Any] = {}

    if case in {"text_only", "text_images", "tables", "logo_header", "mixed"}:
        draw.text((110, 150), TITLE_TEXT, fill=(10, 10, 10), font=_font(46, bold=True))
        texts.append(TITLE_TEXT)
        y = _draw_paragraph(draw, PARAGRAPH_TEXT, 110, 260, width - 220)
        texts.append(PARAGRAPH_TEXT)
        expectations["heading"] = 1

    if case in {"text_images", "mixed"}:
        _draw_picture(draw, 110, y + 60, 520, 380)
        expectations["image"] = 1

    if case in {"tables", "mixed"}:
        table_top = y + (520 if case == "mixed" else 80)
        _draw_table(draw, 110, table_top, width - 220, TABLE_HEADERS, TABLE_ROWS)
        texts.extend(TABLE_HEADERS)
        for row in TABLE_ROWS:
            texts.extend(row)
        expectations["table"] = 1

    if case == "logo_header":
        draw.rectangle([110, 90, 240, 190], fill=(30, 90, 200))
        draw.text((135, 115), LOGO_TEXT, fill=(255, 255, 255), font=_font(36, bold=True))
        texts.append(LOGO_TEXT)
        expectations["logo_or_image"] = 1

    draw.text((110, 2230), FOOTER_TEXT, fill=(40, 40, 40), font=_font(22))
    texts.append(FOOTER_TEXT)
    expectations["footer_or_page_number"] = 1

    ground_truth = "\n".join(texts)

    if photograph:
        angle = random.uniform(-4.5, 4.5)
        image = image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(235, 235, 235))
        gradient = Image.linear_gradient("L").resize(image.size)
        shading = Image.new("RGB", image.size, (255, 255, 255))
        shading = Image.composite(Image.new("RGB", image.size, (150, 150, 150)), shading, gradient.point(lambda v: int(v * 0.55)))
        image = Image.blend(image, shading, 0.22)
        array = np.array(image).astype(np.int16)
        noise = np.random.normal(0, 6.0, array.shape)
        array = np.clip(array + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(array).filter(ImageFilter.GaussianBlur(0.6))

    if dpi_scale != 1.0:
        image = image.resize((int(image.width * dpi_scale), int(image.height * dpi_scale)))

    suffix = ".jpg" if photograph else ".png"
    path = out_dir / f"synthetic_{case}{'_photo' if photograph else ''}{suffix}"
    image.save(path, quality=88 if photograph else None)
    return SyntheticCase(
        name=f"{case}{'_photo' if photograph else ''}",
        image_path=path,
        ground_truth_text=ground_truth,
        expectations=expectations,
        is_photograph=photograph,
    )


BENCHMARK_CASES: list[tuple[str, bool]] = [
    ("text_only", False),
    ("text_images", False),
    ("tables", False),
    ("logo_header", False),
    ("mixed", False),
    ("mixed", True),
]


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def text_accuracy(expected: str, actual: str) -> float:
    return round(difflib.SequenceMatcher(None, _normalise(expected), _normalise(actual)).ratio(), 4)


def evaluate_document(document: Any, case: SyntheticCase) -> dict[str, Any]:
    """Measure the reconstruction of one case against its ground truth."""
    elements = [element for page in document.pages for element in page.elements]
    counts: dict[str, int] = {}
    for element in elements:
        counts[element.type.value] = counts.get(element.type.value, 0) + 1

    reconstructed_text = "\n".join(element.text or "" for element in elements)
    accuracy = text_accuracy(case.ground_truth_text, reconstructed_text)

    table_cells_ok = 0
    table_cells_total = 0
    for element in elements:
        if element.table is None:
            continue
        for cell in element.table.cells:
            table_cells_total += 1
            expected_cells = TABLE_HEADERS + [value for row in TABLE_ROWS for value in row]
            if _normalise(cell.text) and any(
                _normalise(cell.text) == _normalise(value) for value in expected_cells
            ):
                table_cells_ok += 1

    similarities = [
        page.quality.similarity for page in document.pages if page.quality.similarity is not None
    ]
    text_similarities = [
        page.quality.text_similarity for page in document.pages if page.quality.text_similarity is not None
    ]
    return {
        "elements": len(elements),
        "elements_by_type": counts,
        "ocr_accuracy": accuracy,
        "table_cells_total": table_cells_total,
        "table_cells_correct": table_cells_ok,
        "table_cell_accuracy": round(table_cells_ok / table_cells_total, 4) if table_cells_total else None,
        "visual_similarity": round(sum(similarities) / len(similarities), 4) if similarities else None,
        "rendered_text_similarity": (
            round(sum(text_similarities) / len(text_similarities), 4) if text_similarities else None
        ),
        "expectations_met": {
            key: counts.get(key.split("_or_")[0], 0) + counts.get(key.split("_or_")[-1], 0) > 0
            if "_or_" in key
            else counts.get(key, 0) >= value
            for key, value in case.expectations.items()
        },
    }


def run_case(settings: Settings, case: SyntheticCase, out_dir: Path, verify: bool = True) -> dict[str, Any]:
    """Run the pipeline on one synthetic case and measure the result."""
    from .pipeline import DocumentPipeline, PipelineOptions

    pipeline = DocumentPipeline(settings)
    document_id = pipeline.create_document(case.image_path, case.image_path.name)
    options = PipelineOptions(enable_verify=verify, run_correction_loop=True, enable_lens=False)
    result = pipeline.run(document_id, options=options)
    document = result.document
    payload: dict[str, Any] = {
        "case": case.to_dict(),
        "document_id": document_id,
        "status": result.status,
        "elapsed_s": result.elapsed_s,
        "exports": result.exports,
        "warnings": result.warnings[:10],
        "errors": result.errors,
    }
    if document is not None:
        payload["metrics"] = evaluate_document(document, case)
        payload["rendered"] = result.stages.get("render", {}).get("details", {})
    return payload


def run_benchmark(settings: Settings | None = None, cases: int = 6, verbose: bool = False) -> dict[str, Any]:
    """Run the synthetic benchmark set and aggregate the measurements."""
    settings = settings or get_settings()
    work_dir = settings.data_dir / "benchmark"
    work_dir.mkdir(parents=True, exist_ok=True)
    selected = BENCHMARK_CASES[: max(1, min(cases, len(BENCHMARK_CASES)))]
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (case_name, photograph) in enumerate(selected):
        synthetic = build_synthetic_document(work_dir, case=case_name, seed=index, photograph=photograph)
        if verbose:
            print(f"[benchmark] {synthetic.name} …", flush=True)
        try:
            payload = run_case(settings, synthetic, work_dir)
        except Exception as exc:  # noqa: BLE001 - a failing case is a data point
            payload = {"case": synthetic.to_dict(), "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        results.append(payload)
        if verbose:
            metrics = payload.get("metrics") or {}
            print(
                f"    status={payload.get('status')} ocr_accuracy={metrics.get('ocr_accuracy')} "
                f"similarity={metrics.get('visual_similarity')}",
                flush=True,
            )

    measured = [r for r in results if r.get("metrics")]
    aggregate = {
        "cases": len(results),
        "completed": sum(1 for r in results if r.get("status") == "completed"),
        "mean_ocr_accuracy": _mean(r["metrics"].get("ocr_accuracy") for r in measured),
        "mean_visual_similarity": _mean(r["metrics"].get("visual_similarity") for r in measured),
        "mean_table_cell_accuracy": _mean(r["metrics"].get("table_cell_accuracy") for r in measured),
        "exports_produced": sum(1 for r in results if (r.get("exports") or {}).get("pdf")),
        "wall_clock_s": round(time.perf_counter() - started, 1),
    }
    report = {"aggregate": aggregate, "results": results}
    (work_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 4)


def run_selftest(settings: Settings | None = None, verbose: bool = True) -> dict[str, Any]:
    """End-to-end smoke test used by ``painter selftest`` and the test suite."""
    settings = settings or get_settings()
    work_dir = settings.data_dir / "selftest"
    synthetic = build_synthetic_document(work_dir, case="mixed", seed=3, photograph=False)
    payload = run_case(settings, synthetic, work_dir)
    checks = {}
    metrics = payload.get("metrics") or {}
    checks["status_completed"] = payload.get("status") == "completed"
    checks["elements_detected"] = (metrics.get("elements") or 0) >= 4
    checks["text_recovered"] = (metrics.get("ocr_accuracy") or 0) >= 0.6
    checks["table_reconstructed"] = (metrics.get("elements_by_type", {}).get("table", 0) >= 1)
    checks["pdf_exported"] = bool((payload.get("exports") or {}).get("pdf"))
    checks["docx_exported"] = bool((payload.get("exports") or {}).get("docx"))
    payload["checks"] = checks
    payload["status"] = "completed" if all(checks.values()) else "failed"
    if verbose:
        print(json.dumps({"checks": checks, "metrics": metrics}, indent=2, default=str))
    return payload
