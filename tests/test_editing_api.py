"""Element editing tests plus an end-to-end API test through TestClient."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from painter.editing import apply_element_edit, export_artifacts, re_render
from painter.errors import NotFoundError, ValidationError
from painter.schemas import (
    BBox,
    DocumentIR,
    Element,
    ElementType,
    PageFormat,
    PageIR,
    ReviewFlag,
    Typography,
)


def build_document() -> DocumentIR:
    element = Element(
        id="el_body",
        type=ElementType.PARAGRAPH,
        bbox=BBox(x1=100, y1=200, x2=700, y2=300),
        confidence=0.8,
        text="original text",
        typography=Typography(font_size_pt=11.0, estimated=True),
        flags=[ReviewFlag.OCR_UNCERTAIN, ReviewFlag.TEXT_MISSING],
    )
    page = PageIR(
        index=0,
        width_px=1000,
        height_px=1400,
        page_format=PageFormat(name="A4", width_mm=210, height_mm=297),
        elements=[element],
        reading_order=["el_body"],
    )
    document = DocumentIR(document_id="doc_edit", pages=[page])
    document.recompute_stats()
    return document


def test_edit_text_clears_review_flags():
    document = build_document()
    changes = apply_element_edit(document, "el_body", {"text": "corrected text"})
    element = document.pages[0].elements[0]
    assert element.text == "corrected text"
    assert ReviewFlag.TEXT_MISSING not in element.flags
    assert changes["text"]["old"] == "original text"
    assert element.edits
    assert document.revision == 2  # the initial revision plus this edit


def test_edit_typography_marks_not_estimated():
    document = build_document()
    apply_element_edit(document, "el_body", {"font_size_pt": 14.5, "font_family": "DejaVu Serif", "font_weight": 700})
    typography = document.pages[0].elements[0].typography
    assert typography.font_size_pt == 14.5
    assert typography.font_family == "DejaVu Serif"
    assert typography.font_weight == 700
    assert typography.estimated is False


def test_edit_geometry_converts_millimetres():
    document = build_document()
    changes = apply_element_edit(document, "el_body", {"x_mm": 20.0, "y_mm": 30.0, "width_mm": 100.0, "height_mm": 10.0})
    element = document.pages[0].elements[0]
    mm_x = document.pages[0].mm_per_px_x
    assert changes["geometry"]["x_mm"] == pytest.approx(20.0)
    assert element.bbox.x1 == pytest.approx(20.0 / mm_x, rel=1e-3)
    assert element.norm_bbox is not None
    assert ReviewFlag.GEOMETRY_ESTIMATED not in element.flags


def test_edit_alignment_and_type():
    document = build_document()
    apply_element_edit(document, "el_body", {"alignment": "center", "type": "heading"})
    element = document.pages[0].elements[0]
    assert element.alignment.value == "center"
    assert element.type is ElementType.HEADING


def test_edit_table_cells():
    document = build_document()
    element = document.pages[0].elements[0]
    element.type = ElementType.TABLE
    from painter.schemas import TableCell, TableContent

    element.table = TableContent(
        rows=2,
        cols=2,
        cells=[
            TableCell(row=0, col=0, text="A", is_header=True),
            TableCell(row=0, col=1, text="B", is_header=True),
            TableCell(row=1, col=0, text="1"),
            TableCell(row=1, col=1, text="2"),
        ],
    )
    changes = apply_element_edit(document, "el_body", {"cells": [{"row": 1, "col": 1, "text": "200"}]})
    assert changes["table"]["cells_updated"] == 1
    assert element.table.cells[3].text == "200"
    assert "200" in (element.text or "")


def test_edit_rejects_unknown_field():
    document = build_document()
    with pytest.raises(ValidationError):
        apply_element_edit(document, "el_body", {"nonexistent": 1})


def test_edit_unknown_element():
    document = build_document()
    with pytest.raises(NotFoundError):
        apply_element_edit(document, "el_ghost", {"text": "x"})


def test_re_render_writes_files(tmp_path, settings):
    from painter.storage import DocumentWorkspace

    document = build_document()
    workspace = DocumentWorkspace(tmp_path / "doc", "doc_edit")
    result = re_render(document, settings, workspace, formats=("html", "docx"))
    assert (workspace.root / result["html"]["path"]).is_file()
    assert (workspace.root / result["docx"]["path"]).is_file()
    assert "pdf" not in result


def test_export_artifacts_copies_files(tmp_path, settings):
    from painter.storage import DocumentWorkspace

    document = build_document()
    workspace = DocumentWorkspace(tmp_path / "doc2", "doc_edit")
    re_render(document, settings, workspace, formats=("html",))
    exports = export_artifacts(document, workspace, ["html", "pdf"])
    assert "html" in exports
    assert "pdf" not in exports  # never rendered
    assert (workspace.root / exports["html"]["path"]).is_file()


# --------------------------------------------------------------------------
# API integration (no network: SerpApi disabled in the settings fixture)
# --------------------------------------------------------------------------
@pytest.fixture()
def client(settings):
    from fastapi.testclient import TestClient

    from painter.api import create_app

    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
    app.state.jobs.shutdown()


def _make_document(settings, name: str):
    """Register a minimal document the way the upload route does."""
    import hashlib

    from painter.storage import DocumentWorkspace, Repository, utcnow

    repository = Repository(settings.db_path)
    document_id = f"doc_test_{name.replace('.', '_')}"
    workspace = DocumentWorkspace(root=settings.documents_dir / document_id, document_id=document_id)
    workspace.ensure()
    payload = b"not really an image"
    source = workspace.path("source", "original.png")
    source.write_bytes(payload)
    repository.create_document(
        document_id=document_id,
        filename=name,
        stored_path=str(source),
        sha256=hashlib.sha256(payload).hexdigest(),
        content_type="image/png",
        size_bytes=len(payload),
    )
    return repository, workspace


def test_api_clear_documents_empties_the_list(client, settings):
    """Clearing must remove every document, and a second clear must be a no-op."""
    repository, _ = _make_document(settings, "one.png")
    _make_document(settings, "two.png")
    assert len(repository.list_documents(limit=100)) >= 2

    from painter.storage import DocumentWorkspace

    workspace = DocumentWorkspace(
        root=settings.documents_dir / "doc_test_one_png", document_id="doc_test_one_png"
    )
    assert workspace.root.is_dir(), "the workspace should exist before clearing"

    response = client.delete("/api/documents")
    assert response.status_code == 200
    assert response.json()["removed"] >= 2
    assert repository.list_documents(limit=100) == []
    assert not workspace.root.exists(), "clearing must delete the files, not just the rows"

    again = client.delete("/api/documents")
    assert again.status_code == 200
    assert again.json()["removed"] == 0


def test_api_delete_single_document(client, settings):
    """Deleting one document must leave the others alone."""
    repository, keep = _make_document(settings, "keep.png")
    _, drop = _make_document(settings, "drop.png")

    assert client.delete(f"/api/documents/{drop.document_id}").status_code == 200
    remaining = [d["document_id"] for d in repository.list_documents(limit=100)]
    assert drop.document_id not in remaining
    assert keep.document_id in remaining
    assert not drop.root.exists(), "the deleted document's files should be gone too"


def test_api_health_and_capabilities(client):
    assert client.get("/api/health").json()["status"] == "ok"
    capabilities = client.get("/api/capabilities").json()
    assert capabilities["serpapi"]["key_configured"] is False
    assert "supported_formats" in capabilities


def test_api_rejects_unknown_document(client):
    assert client.get("/api/documents/does_not_exist").status_code == 404


def test_api_upload_rejects_bad_format(client, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with bad.open("rb") as handle:
        response = client.post("/api/documents/upload", files={"file": ("notes.txt", handle, "text/plain")})
    assert response.status_code == 415


def test_api_full_flow(client, tmp_path, settings):
    import cv2

    from painter.testsupport import build_synthetic_document

    case = build_synthetic_document(tmp_path, case="text_only", seed=5)
    with case.image_path.open("rb") as handle:
        upload = client.post(
            "/api/documents/upload",
            files={"file": (case.image_path.name, handle, "image/png")},
        )
    assert upload.status_code == 200, upload.text
    document_id = upload.json()["document_id"]

    started = client.post(
        f"/api/documents/{document_id}/process",
        json={"enable_verify": False, "run_correction_loop": False},
    )
    assert started.status_code == 200
    job_id = started.json()["job_id"]

    deadline = time.time() + 240
    status = "queued"
    while time.time() < deadline:
        payload = client.get(f"/api/documents/{document_id}/status?job_id={job_id}").json()
        status = payload["job"]["status"]
        if status in {"completed", "failed"}:
            break
        time.sleep(1.5)
    assert status == "completed", payload

    elements = client.get(f"/api/documents/{document_id}/elements").json()
    assert elements["elements"], "the API must expose reconstructed elements"
    element_id = elements["elements"][0]["id"]

    patched = client.patch(
        f"/api/documents/{document_id}/elements/{element_id}",
        json={"text": "API corrected text", "font_size_pt": 13.0},
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["changes"]["text"]["new"] == "API corrected text"

    preview = client.get(f"/api/documents/{document_id}/preview")
    assert preview.status_code == 200
    assert "API corrected text" in preview.text

    exported = client.post(f"/api/documents/{document_id}/export", json={"formats": ["pdf", "html"]})
    assert exported.status_code == 200, exported.text
    urls = exported.json()["exports"]
    assert "pdf" in urls and "html" in urls

    download = client.get(urls["pdf"]["url"])
    assert download.status_code == 200
    assert download.content.startswith(b"%PDF")

    files = client.get(f"/api/documents/{document_id}/files/pages/page_00_preview.jpg")
    assert files.status_code == 200

    assert client.get(f"/api/documents/{document_id}/files/../../etc/passwd").status_code in {400, 404}


def test_api_requires_token_when_configured(settings):
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from painter.api import create_app

    guarded = replace(settings, api_token="secret-token")
    app = create_app(guarded)
    with TestClient(app) as test_client:
        assert test_client.get("/api/documents").status_code == 401
        assert test_client.get("/api/documents", headers={"X-API-Token": "secret-token"}).status_code == 200
    app.state.jobs.shutdown()
