"""Pipeline tests: stages, resumability, failure handling and quality gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from painter.pipeline import STAGES, DocumentPipeline, PipelineOptions
from painter.storage import DocumentWorkspace, Repository
from painter.testsupport import build_synthetic_document, text_accuracy


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    directory = tmp_path_factory.mktemp("synthetic")
    return build_synthetic_document(directory, case="mixed", seed=1)


@pytest.fixture()
def pipeline(settings):
    return DocumentPipeline(settings)


def test_document_creation_validates_and_stores(pipeline, synthetic):
    document_id = pipeline.create_document(synthetic.image_path, "page.png")
    record = pipeline.repository.get_document(document_id)
    assert record["filename"] == "page.png"
    assert record["sha256"]
    workspace = pipeline.workspace(document_id)
    assert (workspace.root / record["stored_path"]).is_file()


def test_pipeline_runs_all_stages(pipeline, synthetic, settings):
    document_id = pipeline.create_document(synthetic.image_path)
    result = pipeline.run(
        document_id,
        options=PipelineOptions(enable_verify=False, run_correction_loop=False, enable_lens=False),
    )
    assert result.status == "completed"
    for stage in STAGES:
        assert stage in result.stages, f"stage {stage} missing"
        assert result.stages[stage]["status"] == "completed", result.stages[stage]

    document = result.document
    assert document is not None
    assert len(document.pages) == 1
    types = {element.type.value for element in document.pages[0].elements}
    assert "paragraph" in types or "text_block" in types
    assert any(element.text for element in document.pages[0].elements)

    # exports exist on disk
    workspace = pipeline.workspace(document_id)
    for kind in ("html", "pdf", "docx", "ir"):
        assert kind in result.exports, f"{kind} export missing"
        assert workspace.absolute(result.exports[kind]["path"]).is_file()

    # the reconstructed text is recognisable
    reconstructed = "\n".join(element.text or "" for element in document.pages[0].elements)
    assert text_accuracy(synthetic.ground_truth_text, reconstructed) > 0.5


def test_pipeline_records_stage_rows(pipeline, synthetic):
    document_id = pipeline.create_document(synthetic.image_path)
    result = pipeline.run(document_id, options=PipelineOptions(enable_verify=False))
    stages = pipeline.repository.get_job_stages(result.job_id)
    assert {row["stage"] for row in stages} == set(STAGES)
    assert all(row["status"] == "completed" for row in stages)


def test_pipeline_is_resumable(pipeline, synthetic):
    document_id = pipeline.create_document(synthetic.image_path)
    first = pipeline.run(document_id, options=PipelineOptions(enable_verify=False))
    assert first.status == "completed"
    second = pipeline.run(document_id, options=PipelineOptions(enable_verify=False))
    reused = [stage for stage, info in second.stages.items() if info["status"] == "reused"]
    assert reused, "a second run must reuse the cached stage artefacts"
    assert "reconstruct" in reused or "detect" in reused


def test_force_stage_reruns_it(pipeline, synthetic):
    document_id = pipeline.create_document(synthetic.image_path)
    pipeline.run(document_id, options=PipelineOptions(enable_verify=False))
    forced = pipeline.run(
        document_id, options=PipelineOptions(enable_verify=False, force_stages={"detect"})
    )
    assert forced.stages["detect"]["status"] == "completed"
    assert forced.stages["preprocess"]["status"] == "reused"


def test_pipeline_fails_cleanly_when_source_is_missing(pipeline, synthetic):
    document_id = pipeline.create_document(synthetic.image_path)
    record = pipeline.repository.get_document(document_id)
    workspace = pipeline.workspace(document_id)
    workspace.absolute(record["stored_path"]).unlink()
    result = pipeline.run(document_id, options=PipelineOptions(enable_verify=False))
    assert result.status == "failed"
    assert result.errors
    assert result.stages["validate"]["status"] == "failed"


def test_ocr_disabled_marks_elements(pipeline, synthetic):
    document_id = pipeline.create_document(synthetic.image_path)
    result = pipeline.run(
        document_id,
        options=PipelineOptions(enable_verify=False, ocr_provider="none", force_stages={"recognize", "reconstruct"}),
    )
    assert result.status == "completed"
    document = result.document
    text_elements = [
        element
        for page in document.pages
        for element in page.elements
        if element.type.value in {"paragraph", "text_block", "heading", "list", "subheading"}
    ]
    assert text_elements
    assert any(
        "ocr_failed" in [flag.value for flag in element.flags] for element in text_elements
    )


def test_verification_produces_metrics(pipeline, synthetic):
    document_id = pipeline.create_document(synthetic.image_path)
    result = pipeline.run(
        document_id,
        options=PipelineOptions(enable_verify=True, run_correction_loop=False, force_stages={"verify"}),
    )
    assert result.status == "completed"
    page = result.document.pages[0]
    assert page.quality.similarity is not None
    assert 0.0 <= page.quality.similarity <= 1.0
    assert page.quality.metrics_used
    verify_dir = pipeline.workspace(document_id).dir("verify") / "page_00" / "artifacts"
    assert (verify_dir / "side_by_side.png").is_file()
    assert (verify_dir / "difference_map.png").is_file()


def test_verification_can_be_disabled(pipeline, synthetic):
    document_id = pipeline.create_document(synthetic.image_path)
    result = pipeline.run(document_id, options=PipelineOptions(enable_verify=False, force_stages={"verify"}))
    assert result.stages["verify"]["details"].get("skipped") is True


def test_stage_artefacts_are_written(pipeline, synthetic):
    document_id = pipeline.create_document(synthetic.image_path)
    pipeline.run(document_id, options=PipelineOptions(enable_verify=False))
    stages_dir = pipeline.workspace(document_id).dir("stages")
    for stage in STAGES:
        path = stages_dir / f"{stage}.json"
        assert path.is_file(), f"missing artefact for {stage}"
        json.loads(path.read_text())


def test_repository_persists_ir_and_quality(pipeline, synthetic):
    document_id = pipeline.create_document(synthetic.image_path)
    result = pipeline.run(document_id, options=PipelineOptions(enable_verify=False))
    payload = pipeline.repository.latest_ir(document_id)
    assert payload is not None
    assert payload["document_id"] == document_id
    assert pipeline.repository.stats()["documents"] >= 1
    exports = pipeline.repository.list_exports(document_id)
    assert {row["format"] for row in exports} >= {"pdf", "docx", "html"}


def test_workspace_rejects_path_traversal(tmp_path):
    from painter.errors import StorageError

    workspace = DocumentWorkspace(tmp_path / "doc", "doc_x")
    with pytest.raises(StorageError):
        workspace.absolute("../../etc/passwd")


def test_cleanup_dry_run_keeps_workspaces(pipeline, synthetic, settings):
    from painter.storage import cleanup_workspaces

    document_id = pipeline.create_document(synthetic.image_path)
    removed = cleanup_workspaces(settings.documents_dir, pipeline.repository, keep_days=0, dry_run=True)
    assert document_id in removed
    assert pipeline.workspace(document_id).root.is_dir()
    cleanup_workspaces(settings.documents_dir, pipeline.repository, keep_days=0, dry_run=False)
    assert not pipeline.workspace(document_id).root.exists()


def test_multi_page_pdf_document(pipeline, tmp_path):
    """A two-page PDF upload must produce a two-page reconstruction."""
    import cv2

    from painter.testsupport import build_synthetic_document as build

    page_one = build(tmp_path, case="text_only", seed=1)
    page_two = build(tmp_path, case="tables", seed=2)
    images = [cv2.imread(str(page_one.image_path)), cv2.imread(str(page_two.image_path))]
    pdf_path = tmp_path / "two_pages.pdf"
    import PIL.Image

    frames = [PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]
    frames[0].save(pdf_path, save_all=True, append_images=frames[1:], resolution=200)

    document_id = pipeline.create_document(pdf_path, "two_pages.pdf")
    result = pipeline.run(document_id, options=PipelineOptions(enable_verify=False))
    assert result.status == "completed"
    assert len(result.document.pages) == 2
    assert result.exports["pdf"]["size_bytes"] > 0
