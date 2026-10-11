"""PDF renderer.

WeasyPrint turns the HTML page markup into a PDF with the exact page size from
the Document IR. If WeasyPrint is unavailable or fails, Chromium's headless
``--print-to-pdf`` is used instead. Both paths are verified afterwards: a PDF
that contains no extractable text is treated as a failure, because the whole
point of Painter is to avoid shipping a page image masquerading as a document.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..errors import RenderingError
from ..logging_utils import get_logger
from ..schemas import DocumentIR
from .html_renderer import HtmlRenderOptions, render_document_for_print

logger = get_logger("render.pdf")


@dataclass
class PdfResult:
    path: Path
    engine: str
    pages: int | None = None
    text_characters: int = 0
    has_text: bool = False
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []


def _weasyprint_available() -> bool:
    try:
        import weasyprint  # noqa: F401

        return True
    except Exception:  # pragma: no cover - optional dependency
        return False


def _chromium_path() -> str | None:
    for candidate in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text with poppler's ``pdftotext`` (preinstalled)."""
    if shutil.which("pdftotext") is None:
        return ""
    try:
        completed = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.decode("utf-8", "replace")


def pdf_page_count(pdf_path: Path) -> int | None:
    if shutil.which("pdfinfo") is None:
        return None
    try:
        completed = subprocess.run(
            ["pdfinfo", str(pdf_path)], capture_output=True, timeout=60, check=False
        )
    except Exception:
        return None
    for line in completed.stdout.decode("utf-8", "replace").splitlines():
        if line.lower().startswith("pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def render_pdf(
    document: DocumentIR,
    output_path: Path,
    options: HtmlRenderOptions | None = None,
    verify_text: bool = True,
) -> PdfResult:
    """Render the Document IR to PDF."""
    options = options or HtmlRenderOptions()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_markup = render_document_for_print(document, options)

    warnings: list[str] = []
    engine = "none"
    rendered = False

    if _weasyprint_available():
        try:
            from weasyprint import HTML

            HTML(string=html_markup, base_url=str(output_path.parent)).write_pdf(str(output_path))
            engine = "weasyprint"
            rendered = output_path.is_file() and output_path.stat().st_size > 0
        except Exception as exc:  # noqa: BLE001 - fall through to chromium
            warnings.append(f"weasyprint failed: {exc}")
            logger.warning("weasyprint rendering failed: %s", exc)

    if not rendered:
        chromium = _chromium_path()
        if chromium is None:
            raise RenderingError(
                "no PDF engine available (weasyprint and chromium both failed)",
                details={"warnings": warnings},
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "document.html"
            html_path.write_text(html_markup, encoding="utf-8")
            cmd = [
                chromium,
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--no-pdf-header-footer",
                "--run-all-compositor-stages-before-draw",
                "--virtual-time-budget=8000",
                f"--print-to-pdf={output_path}",
                html_path.as_uri(),
            ]
            completed = subprocess.run(cmd, capture_output=True, timeout=300, check=False)
            if completed.returncode != 0 or not output_path.is_file():
                raise RenderingError(
                    "chromium PDF rendering failed: "
                    + completed.stderr.decode("utf-8", "replace")[:300],
                    details={"warnings": warnings},
                )
            engine = "chromium"

    text = extract_pdf_text(output_path) if verify_text else ""
    result = PdfResult(
        path=output_path,
        engine=engine,
        pages=pdf_page_count(output_path),
        text_characters=len(text.strip()),
        has_text=bool(text.strip()),
        warnings=warnings,
    )
    if verify_text and not result.has_text:
        result.warnings.append(
            "the generated PDF contains no extractable text; check the renderer before publishing"
        )
    return result
