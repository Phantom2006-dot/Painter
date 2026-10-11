"""Manual editing of the Document IR.

Edits mutate the IR directly and only trigger a re-render — the AI pipeline is
never re-run for a font change or a moved element, as required by the brief.
Every change is recorded on the element and in the repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .config import Settings
from .errors import NotFoundError, ValidationError
from .logging_utils import get_logger
from .schemas import (
    Alignment,
    BBox,
    DocumentIR,
    EditRecord,
    Element,
    ElementType,
    ImageContent,
    NormBBox,
    ReviewFlag,
    TableContent,
)

logger = get_logger("editing")

TYPOGRAPHY_FIELDS = {
    "font_family",
    "font_size_pt",
    "font_weight",
    "italic",
    "underline",
    "color",
    "line_height_pt",
    "line_spacing_ratio",
    "paragraph_spacing_pt",
    "letter_spacing_em",
    "word_spacing_em",
    "indent_first_line_mm",
    "indent_left_mm",
}

GEOMETRY_FIELDS = {"x_mm", "y_mm", "width_mm", "height_mm"}
GENERAL_FIELDS = {"alignment", "z", "rotation", "type", "asset_path", "cells", "text"}


def _resolve(document: DocumentIR, element_id: str) -> tuple[Any, Element]:
    found = document.find_element(element_id)
    if found is None:
        raise NotFoundError(f"element {element_id} not found")
    return found


def _apply_geometry(page: Any, element: Element, patch: dict[str, Any]) -> dict[str, Any]:
    mm_x = page.mm_per_px_x
    mm_y = page.mm_per_px_y
    if not mm_x or not mm_y:
        raise ValidationError("page has no physical dimensions; cannot position elements")
    current = {
        "x_mm": element.bbox.x1 * mm_x,
        "y_mm": element.bbox.y1 * mm_y,
        "width_mm": element.bbox.width * mm_x,
        "height_mm": element.bbox.height * mm_y,
    }
    for field_name in GEOMETRY_FIELDS:
        if field_name in patch and patch[field_name] is not None:
            current[field_name] = float(patch[field_name])
    x1 = max(0.0, current["x_mm"]) / mm_x
    y1 = max(0.0, current["y_mm"]) / mm_y
    x2 = x1 + max(0.4, current["width_mm"]) / mm_x
    y2 = y1 + max(0.4, current["height_mm"]) / mm_y
    element.bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2).clip(page.width_px, page.height_px)
    element.norm_bbox = NormBBox.from_bbox(element.bbox, page.width_px, page.height_px)
    return {key: round(value, 3) for key, value in current.items()}


def _apply_typography(element: Element, patch: dict[str, Any]) -> dict[str, Any]:
    from .schemas import Typography

    if element.typography is None:
        element.typography = Typography(estimated=True)
    applied: dict[str, Any] = {}
    for field_name in TYPOGRAPHY_FIELDS:
        if field_name in patch and patch[field_name] is not None:
            value = patch[field_name]
            if field_name in {"font_size_pt", "line_height_pt", "paragraph_spacing_pt"}:
                value = float(value)
            elif field_name in {"font_weight"}:
                value = int(value)
            setattr(element.typography, field_name, value)
            applied[field_name] = value
    if applied:
        element.typography.estimated = False
        element.typography.notes.append("manually adjusted")
        element.flags = [f for f in element.flags if f is not ReviewFlag.TYPOGRAPHY_ESTIMATED]
    return applied


def _apply_table(element: Element, cells: Iterable[dict[str, Any]]) -> dict[str, Any]:
    if element.table is None:
        element.table = TableContent()
    updated = 0
    for update in cells:
        row = int(update.get("row", -1))
        col = int(update.get("col", -1))
        for cell in element.table.cells:
            if cell.row == row and cell.col == col:
                if "text" in update and update["text"] is not None:
                    cell.text = str(update["text"])
                    updated += 1
                if "alignment" in update and update["alignment"]:
                    cell.alignment = Alignment(update["alignment"])
                if "background_color" in update:
                    cell.background_color = update["background_color"]
                break
    element.text = "\n".join(
        " | ".join(cell.text for cell in element.table.cells if cell.row == row)
        for row in range(element.table.rows)
    )
    return {"cells_updated": updated}


def apply_element_edit(
    document: DocumentIR,
    element_id: str,
    patch: dict[str, Any],
    *,
    editor: str = "user",
    repository: Any | None = None,
) -> dict[str, Any]:
    """Apply a partial update to one element and return what changed."""
    page, element = _resolve(document, element_id)
    unknown = set(patch) - (TYPOGRAPHY_FIELDS | GEOMETRY_FIELDS | GENERAL_FIELDS)
    if unknown:
        raise ValidationError(f"unsupported fields: {sorted(unknown)}")

    changes: dict[str, Any] = {}

    if "text" in patch and patch["text"] is not None:
        old = element.text
        element.text = str(patch["text"])
        element.flags = [
            f for f in element.flags if f not in {ReviewFlag.TEXT_MISSING, ReviewFlag.OCR_FAILED}
        ]
        if element.lines:
            # keep a single synthetic line so the renderer has geometry
            from .schemas import TextLine

            if len(element.lines) == 1:
                element.lines[0].text = element.text
            else:
                element.lines = [
                    TextLine(text=line, bbox=element.lines[min(index, len(element.lines) - 1)].bbox)
                    for index, line in enumerate(element.text.splitlines() or [element.text])
                ]
        changes["text"] = {"old": old, "new": element.text}
        element.edits.append(EditRecord(field="text", old_value=old, new_value=element.text, editor=editor))
        if repository is not None:
            repository.record_edit(document.document_id, element_id, "text", old, element.text, editor)

    if "type" in patch and patch["type"]:
        old_type = element.type
        element.type = ElementType(patch["type"])
        changes["type"] = {"old": old_type.value, "new": element.type.value}

    if "alignment" in patch:
        value = patch["alignment"]
        element.alignment = Alignment(value) if value else None
        changes["alignment"] = value

    if "z" in patch and patch["z"] is not None:
        element.z = int(patch["z"])
        changes["z"] = element.z

    if "rotation" in patch and patch["rotation"] is not None:
        element.rotation = float(patch["rotation"])
        changes["rotation"] = element.rotation

    geometry = _apply_geometry(page, element, patch)
    if any(field in patch for field in GEOMETRY_FIELDS):
        changes["geometry"] = geometry
        element.flags = [f for f in element.flags if f is not ReviewFlag.GEOMETRY_ESTIMATED]

    typography = _apply_typography(element, patch)
    if typography:
        changes["typography"] = typography

    if "asset_path" in patch and patch["asset_path"]:
        if element.image is None:
            element.image = ImageContent()
        element.image.asset_path = str(patch["asset_path"])
        element.image.notes.append(f"asset replaced manually by {editor}")
        element.flags = [f for f in element.flags if f is not ReviewFlag.IMAGE_UNMATCHED]
        changes["asset_path"] = element.image.asset_path

    if "cells" in patch and patch["cells"]:
        changes["table"] = _apply_table(element, patch["cells"])

    element.edits.append(EditRecord(field=",".join(sorted(changes)) or "noop", editor=editor))
    document.revision += 1
    document.touch()
    document.recompute_stats()
    return changes


def re_render(
    document: DocumentIR,
    settings: Settings,
    workspace: Any,
    asset_url_prefix: str | None = None,
    formats: Iterable[str] = ("html", "pdf", "docx"),
) -> dict[str, Any]:
    """Re-render the requested formats from the current IR (no OCR, no API calls)."""
    from .pipeline import _correction_options
    from .render import HtmlRenderOptions, render_docx, render_document, render_pdf

    formats = set(formats)
    corrections = _correction_options(document)
    options = HtmlRenderOptions(
        asset_url_prefix=asset_url_prefix,
        title=document.source.filename if document.source else "Painter reconstruction",
        global_offset_mm=corrections["offset_mm"],
        font_scale=corrections["font_scale"],
    )
    result: dict[str, Any] = {}
    if "html" in formats:
        html = render_document(document, options)
        path = workspace.path("rendered", "reconstruction.html")
        path.write_text(html, encoding="utf-8")
        result["html"] = {"path": workspace.relative(path), "size_bytes": path.stat().st_size}
    if "pdf" in formats:
        pdf = render_pdf(document, workspace.path("rendered", "reconstruction.pdf"), options)
        result["pdf"] = {
            "path": workspace.relative(pdf.path),
            "size_bytes": pdf.path.stat().st_size,
            "engine": pdf.engine,
            "has_text": pdf.has_text,
        }
    if "docx" in formats:
        docx_result = render_docx(
            document,
            workspace.path("rendered", "reconstruction.docx"),
            asset_base=workspace.root,
            page_offset_mm=corrections["offset_mm"],
        )
        result["docx"] = {
            "path": workspace.relative(docx_result.path),
            "size_bytes": docx_result.path.stat().st_size,
            "paragraphs": docx_result.paragraphs,
            "tables": docx_result.tables,
            "pictures": docx_result.pictures,
        }
    return result


def export_artifacts(document: DocumentIR, workspace: Any, kinds: Iterable[str]) -> dict[str, Any]:
    """Copy rendered artefacts into the exports folder and return descriptors."""
    import shutil

    exports_dir = workspace.dir("exports")
    mapping = {
        "html": ("reconstruction.html", "text/html"),
        "pdf": ("reconstruction.pdf", "application/pdf"),
        "docx": (
            "reconstruction.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
    }
    exports: dict[str, Any] = {}
    for kind in kinds:
        if kind not in mapping:
            continue
        filename, content_type = mapping[kind]
        source = workspace.path("rendered", filename)
        if not source.is_file():
            continue
        target = exports_dir / f"{document.document_id}_{filename}"
        shutil.copy2(source, target)
        exports[kind] = {
            "path": workspace.relative(target),
            "size_bytes": target.stat().st_size,
            "content_type": content_type,
            "filename": filename,
        }
    return exports
