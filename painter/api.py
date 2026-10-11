"""FastAPI application: upload, process, inspect, edit, verify and export.

The routes follow the logical endpoints from the brief. Long-running work is
executed in a bounded thread pool so uploads return immediately and the client
polls ``/status``.
"""

from __future__ import annotations

import json
import secrets
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import Cookie, Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings, get_settings, load_settings
from .editing import apply_element_edit, export_artifacts, re_render
from .errors import (
    NotFoundError,
    PainterError,
    UnsupportedFormatError,
    UploadTooLargeError,
    ValidationError,
)
from .logging_utils import configure_logging, get_logger, log_event
from .ocr import TesseractProvider
from .pipeline import DocumentPipeline, PipelineOptions
from .render import HtmlRenderOptions, render_document
from .schemas import DocumentIR
from .serpapi_client import SerpApiClient, UsageLedger, describe_configuration
from .settings_store import (
    describe as describe_settings,
    save_overrides,
    validate_api_key,
    verify_api_key,
)
from .storage import Repository, cleanup_workspaces, remove_document_files

logger = get_logger("api")

WEB_DIR = Path(__file__).resolve().parent / "web"


# --------------------------------------------------------------------------
# request models
# --------------------------------------------------------------------------
class ProcessRequest(BaseModel):
    force_stages: list[str] = Field(default_factory=list)
    enable_verify: bool | None = None
    run_correction_loop: bool = True
    enable_lens: bool | None = None
    enable_assets_download: bool | None = None
    ocr_provider: str | None = None
    max_pages: int | None = None


class ElementPatch(BaseModel):
    text: str | None = None
    type: str | None = None
    alignment: str | None = None
    z: int | None = None
    rotation: float | None = None
    x_mm: float | None = None
    y_mm: float | None = None
    width_mm: float | None = None
    height_mm: float | None = None
    font_family: str | None = None
    font_size_pt: float | None = None
    font_weight: int | None = None
    italic: bool | None = None
    underline: bool | None = None
    color: str | None = None
    line_height_pt: float | None = None
    paragraph_spacing_pt: float | None = None
    letter_spacing_em: float | None = None
    asset_path: str | None = None
    cells: list[dict[str, Any]] | None = None

    model_config = {"extra": "forbid"}


class ExportRequest(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["html", "pdf", "docx"])


class ReconstructRequest(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["html", "pdf", "docx"])


class VerifyRequest(BaseModel):
    run_correction_loop: bool = True
    page: int | None = None


class SerpApiSettingsRequest(BaseModel):
    """Paste a SerpApi key at runtime instead of editing the environment."""

    api_key: str | None = None
    enabled: bool | None = None
    verify: bool = False


# --------------------------------------------------------------------------
# job manager
# --------------------------------------------------------------------------
class JobManager:
    """Runs pipeline jobs in a bounded pool and keeps their last result."""

    def __init__(self, pipeline: DocumentPipeline, max_workers: int = 2) -> None:
        self.pipeline = pipeline
        self.pool = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="painter-job")
        self._lock = threading.Lock()
        self._results: dict[str, Any] = {}
        self._futures: dict[str, Any] = {}

    def submit(self, document_id: str, options: PipelineOptions) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        self.pipeline.repository.create_job(job_id, document_id, options.to_dict())
        future = self.pool.submit(self._run, document_id, job_id, options)
        with self._lock:
            self._futures[job_id] = future
        return job_id

    def _run(self, document_id: str, job_id: str, options: PipelineOptions) -> None:
        try:
            result = self.pipeline.run(document_id, job_id=job_id, options=options)
            with self._lock:
                self._results[job_id] = result.to_dict()
        except Exception as exc:  # noqa: BLE001 - the job must always terminate
            logger.exception("job %s crashed", job_id)
            self.pipeline.repository.update_job(job_id, status="failed", error=str(exc))
            self.pipeline.repository.record_error(document_id, job_id, None, "job_crash", str(exc))
            with self._lock:
                self._results[job_id] = {"status": "failed", "error": str(exc)}

    def result(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._results.get(job_id)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


# --------------------------------------------------------------------------
# application factory
# --------------------------------------------------------------------------
def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_dirs()
    configure_logging(settings.log_level)

    repository = Repository(settings.db_path)
    pipeline = DocumentPipeline(settings, repository)
    jobs = JobManager(pipeline, settings.job_workers)

    app = FastAPI(
        title="Painter",
        version="1.0.0",
        description="AI document reconstruction and replication engine",
    )

    @app.exception_handler(PainterError)
    async def painter_error_handler(_request: Any, exc: PainterError) -> JSONResponse:
        """Return the typed error with its own status code and stable code."""
        logger.warning("api error: %s", exc.to_dict())
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())
    app.state.settings = settings
    app.state.repository = repository
    app.state.pipeline = pipeline
    app.state.jobs = jobs

    def settings_state() -> dict[str, Any]:
        """Effective settings, with the SerpApi key masked to a hint."""
        return describe_settings(settings, settings.data_dir)

    # -- auth ------------------------------------------------------------
    def require_token(
        authorization: str | None = Header(default=None),
        x_api_token: str | None = Header(default=None),
        token: str | None = Query(default=None),
        painter_token: str | None = Cookie(default=None),
    ) -> None:
        """Accept the token as a header, a query parameter or a cookie.

        ``<img>`` and ``<iframe>`` requests cannot carry an Authorization header,
        so the browser may also send the token in the URL or as a same-origin
        cookie set by the web UI.
        """
        expected = settings.api_token
        if not expected:
            return
        supplied = None
        if authorization and authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
        elif x_api_token:
            supplied = x_api_token.strip()
        elif token:
            supplied = token.strip()
        elif painter_token:
            supplied = painter_token.strip()
        if not supplied or not secrets.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="invalid or missing API token")

    guarded = [Depends(require_token)]

    def workspace_for(document_id: str):
        try:
            repository.get_document(document_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return pipeline.workspace(document_id)

    def load_ir(document_id: str) -> DocumentIR:
        payload = repository.latest_ir(document_id)
        if payload is None:
            raise HTTPException(status_code=409, detail="document has not been processed yet")
        return DocumentIR.model_validate(payload)

    # -- meta ------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": "1.0.0",
            "documents": repository.stats().get("documents", 0),
        }

    @app.get("/api/capabilities", dependencies=guarded)
    def capabilities() -> dict[str, Any]:
        tesseract = TesseractProvider(settings)
        from .detection import LayoutDetectorService

        service = LayoutDetectorService(settings)
        models = []
        for name, path in (
            ("layout", settings.yolo_layout_model),
            ("coco", settings.yolo_coco_model),
            ("custom", settings.yolo_custom_model),
        ):
            if path is None:
                models.append({"name": name, "path": None, "present": False})
                continue
            models.append(
                {
                    "name": name,
                    "path": str(path),
                    "present": Path(path).is_file(),
                }
            )
        return {
            "yolo_enabled": settings.enable_yolo,
            "models": models,
            "ocr_provider_mode": settings.ocr_provider,
            "tesseract_available": tesseract.available(),
            "serpapi": describe_configuration(settings.serpapi),
            "verification_enabled": settings.verify_enabled,
            "llm_enabled": settings.enable_llm,
            "asset_download_enabled": settings.enable_assets_download,
            "max_upload_mb": settings.max_upload_mb,
            "supported_formats": sorted(
                set(settings.allowed_image_formats) | set(settings.allowed_document_formats)
            ),
        }

    @app.get("/api/usage", dependencies=guarded)
    def usage() -> dict[str, Any]:
        ledger = UsageLedger(settings.cache_dir / "serpapi_usage.json")
        return {"serpapi": ledger.snapshot(), "configuration": describe_configuration(settings.serpapi)}

    # -- documents -------------------------------------------------------
    @app.post("/api/documents/upload", dependencies=guarded)
    async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
        limit = settings.max_upload_mb * 1024 * 1024
        suffix = Path(file.filename or "upload").suffix.lower() or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            temp_path = Path(handle.name)
            total = 0
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > limit:
                    temp_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"file exceeds the {settings.max_upload_mb} MB limit",
                    )
                handle.write(chunk)
        try:
            document_id = pipeline.create_document(temp_path, file.filename)
        except UploadTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except UnsupportedFormatError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        except PainterError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            temp_path.unlink(missing_ok=True)
        record = repository.get_document(document_id)
        return {
            "document_id": document_id,
            "filename": record["filename"],
            "size_bytes": record["size_bytes"],
            "content_type": record["content_type"],
            "status": record["status"],
        }

    # -- runtime settings -------------------------------------------------
    @app.get("/api/settings", dependencies=guarded)
    def get_settings_state() -> dict[str, Any]:
        """Report the effective SerpApi state without ever returning the key."""
        return settings_state()

    @app.put("/api/settings/serpapi", dependencies=guarded)
    def put_serpapi_settings(request: SerpApiSettingsRequest) -> dict[str, Any]:
        """Store a SerpApi key (and enable/disable Lens) for this instance."""
        nonlocal settings
        updates: dict[str, Any] = {}
        if request.api_key is not None:
            updates["serpapi_api_key"] = validate_api_key(request.api_key)
        if request.enabled is not None:
            updates["serpapi_enabled"] = bool(request.enabled)
        if not updates:
            raise ValidationError("nothing to update: pass api_key and/or enabled")
        verified: bool | None = None
        message: str | None = None
        if request.verify and updates.get("serpapi_api_key"):
            verified, message = verify_api_key(
                updates["serpapi_api_key"],
                settings.serpapi.base_url,
                timeout_s=settings.serpapi.timeout_s,
            )
        save_overrides(settings.data_dir, updates)
        # rebuild the live settings so the running pipeline sees the change now
        settings = load_settings(refresh=True)
        app.state.settings = settings
        pipeline.settings = settings
        logger.info("serpapi settings updated: key_set=%s enabled=%s", bool(updates.get("serpapi_api_key")), settings.serpapi.enabled)
        state = settings_state()
        state["verified"] = verified
        state["message"] = message
        return state

    @app.delete("/api/settings/serpapi", dependencies=guarded)
    def delete_serpapi_settings() -> dict[str, Any]:
        """Forget a stored key and switch Lens off."""
        nonlocal settings
        save_overrides(settings.data_dir, {"serpapi_api_key": None, "serpapi_enabled": False})
        settings = load_settings(refresh=True)
        app.state.settings = settings
        pipeline.settings = settings
        return settings_state()

    @app.post("/api/settings/serpapi/verify", dependencies=guarded)
    def verify_serpapi_settings() -> dict[str, Any]:
        """Check the key currently in effect, without changing anything."""
        key = settings.serpapi.api_key
        if not key:
            raise ValidationError("no SerpApi key is configured")
        ok, message = verify_api_key(key, settings.serpapi.base_url, timeout_s=settings.serpapi.timeout_s)
        return {"verified": ok, "message": message}

    @app.get("/api/documents", dependencies=guarded)
    def list_documents(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
        return {"documents": repository.list_documents(limit)}

    @app.get("/api/documents/{document_id}", dependencies=guarded)
    def get_document(document_id: str) -> dict[str, Any]:
        record = repository.get_document(document_id)
        payload = repository.latest_ir(document_id)
        document = DocumentIR.model_validate(payload) if payload else None
        return {
            "document": record,
            "pages": repository.get_pages(document_id),
            "jobs": repository.get_jobs_for_document(document_id, limit=5),
            "stats": document.stats if document else None,
            "quality": [
                {"page": page.index, "quality": page.quality.model_dump()} for page in (document.pages if document else [])
            ],
            "warnings": document.warnings if document else [],
        }

    @app.delete("/api/documents", dependencies=guarded)
    def clear_documents() -> dict[str, Any]:
        """Remove every document, its files and its exports.

        Uploading a new file is easier when the list can be emptied first, and
        the only other way was to delete records one at a time.
        """
        documents = repository.list_documents(limit=1000)
        removed: list[str] = []
        failures: list[str] = []
        for document in documents:
            document_id = document.get("document_id")
            if not document_id:
                continue
            try:
                repository.delete_document(document_id)
                remove_document_files(settings.documents_dir, document_id)
                removed.append(document_id)
            except Exception as exc:  # keep going: one bad record must not block the rest
                failures.append(f"{document_id}: {exc}")
        logger.info("cleared documents", extra={"removed": len(removed), "failed": len(failures)})
        return {"removed": len(removed), "document_ids": removed, "failures": failures}

    @app.delete("/api/documents/{document_id}", dependencies=guarded)
    def delete_document(document_id: str) -> dict[str, Any]:  # noqa: F811 - route body follows
        workspace = workspace_for(document_id)
        repository.delete_document(document_id)
        shutil.rmtree(workspace.root, ignore_errors=True)
        return {"deleted": document_id}

    # -- processing ------------------------------------------------------
    @app.post("/api/documents/{document_id}/process", dependencies=guarded)
    def process_document(document_id: str, request: ProcessRequest) -> dict[str, Any]:
        workspace_for(document_id)
        options = PipelineOptions(
            force_stages=set(request.force_stages),
            enable_verify=request.enable_verify,
            run_correction_loop=request.run_correction_loop,
            enable_lens=request.enable_lens,
            enable_assets_download=request.enable_assets_download,
            ocr_provider=request.ocr_provider,
            max_pages=request.max_pages,
        )
        job_id = jobs.submit(document_id, options)
        log_event(logger, "job submitted", document_id=document_id, job_id=job_id)
        return {"job_id": job_id, "document_id": document_id, "status": "queued"}

    @app.get("/api/documents/{document_id}/status", dependencies=guarded)
    def status(document_id: str, job_id: str | None = None) -> dict[str, Any]:
        if job_id:
            job = repository.get_job(job_id)
        else:
            recent = repository.get_jobs_for_document(document_id, limit=1)
            if not recent:
                raise HTTPException(status_code=404, detail="no job has been started for this document")
            job = repository.get_job(recent[0]["job_id"])
        return {"job": job, "result": jobs.result(job["job_id"])}

    # -- inspection ------------------------------------------------------
    @app.get("/api/documents/{document_id}/ir", dependencies=guarded)
    def get_ir(document_id: str) -> Any:
        # distinguish "no such document" (404) from "not processed yet" (409)
        repository.get_document(document_id)
        payload = repository.latest_ir(document_id)
        if payload is None:
            raise HTTPException(status_code=409, detail="document has not been processed yet")
        return JSONResponse(payload)

    @app.get("/api/documents/{document_id}/elements", dependencies=guarded)
    def list_elements(
        document_id: str,
        page: int | None = None,
        needs_review: bool = False,
        type_filter: str | None = Query(default=None, alias="type"),
    ) -> dict[str, Any]:
        document = load_ir(document_id)
        elements: list[dict[str, Any]] = []
        for page_ir in document.pages:
            if page is not None and page_ir.index != page:
                continue
            for element in page_ir.elements:
                if needs_review and not element.needs_review:
                    continue
                if type_filter and element.type.value != type_filter:
                    continue
                elements.append(
                    {
                        "id": element.id,
                        "page": page_ir.index,
                        "type": element.type.value,
                        "confidence": element.confidence,
                        "bbox": element.bbox.as_list(),
                        "norm_bbox": element.norm_bbox.model_dump() if element.norm_bbox else None,
                        "bbox_mm": page_ir.px_to_mm(element.bbox),
                        "text": element.text,
                        "alignment": element.alignment.value if element.alignment else None,
                        "flags": [flag.value for flag in element.flags],
                        "order_index": element.order_index,
                        "typography": element.typography.model_dump() if element.typography else None,
                        "table": element.table.model_dump() if element.table else None,
                        "image": element.image.model_dump() if element.image else None,
                        "section": element.metadata.get("section_id"),
                        "provenance": [p.model_dump(mode="json") for p in element.provenance],
                        "edits": [e.model_dump(mode="json") for e in element.edits],
                    }
                )
        return {
            "document_id": document_id,
            "revision": document.revision,
            "elements": elements,
            "stats": document.stats,
            "pages": [
                {
                    "index": page_ir.index,
                    "width_px": page_ir.width_px,
                    "height_px": page_ir.height_px,
                    "page_format": page_ir.page_format.model_dump(),
                    "margins_mm": page_ir.margins_mm.model_dump(),
                    "quality": page_ir.quality.model_dump(),
                    "sections": page_ir.metadata.get("sections", []),
                }
                for page_ir in document.pages
            ],
        }

    @app.get("/api/documents/{document_id}/elements/{element_id}", dependencies=guarded)
    def get_element(document_id: str, element_id: str) -> dict[str, Any]:
        document = load_ir(document_id)
        found = document.find_element(element_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"element {element_id} not found")
        page, element = found
        return {
            "element": json.loads(element.model_dump_json()),
            "page": {"index": page.index, "bbox_mm": page.px_to_mm(element.bbox)},
        }

    @app.patch("/api/documents/{document_id}/elements/{element_id}", dependencies=guarded)
    def patch_element(document_id: str, element_id: str, patch: ElementPatch) -> dict[str, Any]:
        document = load_ir(document_id)
        payload = patch.model_dump(exclude_none=True)
        if not payload:
            raise HTTPException(status_code=400, detail="empty patch")
        try:
            changes = apply_element_edit(document, element_id, payload, repository=repository)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PainterError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        repository.save_ir(document_id, document.revision, document.model_dump_json())
        rendered = re_render(
            document,
            settings,
            pipeline.workspace(document_id),
            asset_url_prefix=f"/api/documents/{document_id}/files",
            formats=("html",),
        )
        return {
            "changes": changes,
            "revision": document.revision,
            "rendered": rendered,
            "element": json.loads(document.find_element(element_id)[1].model_dump_json()),  # type: ignore[index]
        }

    # -- rendering / verification / export -------------------------------
    @app.get("/api/documents/{document_id}/preview", dependencies=guarded)
    def preview(document_id: str) -> HTMLResponse:
        document = load_ir(document_id)
        markup = render_document(
            document,
            HtmlRenderOptions(
                asset_url_prefix=f"/api/documents/{document_id}/files",
                include_debug_boxes=False,
            ),
        )
        return HTMLResponse(markup)

    @app.get("/api/documents/{document_id}/preview.png", dependencies=guarded)
    def preview_png(document_id: str, page: int = 0, dpi: int = 120) -> FileResponse:
        workspace = workspace_for(document_id)
        rendered = workspace.path("rendered", "reconstruction.html")
        if not rendered.is_file():
            raise HTTPException(status_code=409, detail="document has not been rendered yet")
        document = load_ir(document_id)
        page_ir = document.page(page)
        if page_ir is None:
            raise HTTPException(status_code=404, detail=f"page {page} not found")
        from .verify import Rasterizer

        target = workspace.path("rendered", f"preview_page_{page}.png")
        rasterizer = Rasterizer(dpi=max(60, min(dpi, 220)))
        try:
            rasterizer.rasterize(rendered, target, page_ir.page_format.width_mm, page_ir.page_format.height_mm)
        except PainterError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return FileResponse(target, media_type="image/png")

    @app.post("/api/documents/{document_id}/reconstruct", dependencies=guarded)
    def reconstruct(document_id: str, request: ReconstructRequest) -> dict[str, Any]:
        document = load_ir(document_id)
        rendered = re_render(
            document,
            settings,
            pipeline.workspace(document_id),
            asset_url_prefix=f"/api/documents/{document_id}/files",
            formats=request.formats,
        )
        return {"revision": document.revision, "rendered": rendered}

    @app.post("/api/documents/{document_id}/verify", dependencies=guarded)
    def verify(document_id: str, request: VerifyRequest) -> dict[str, Any]:
        import cv2

        from .verify import correction_loop, verify_page

        document = load_ir(document_id)
        workspace = pipeline.workspace(document_id)
        from .render import HtmlRenderOptions, render_document_for_print

        def write_verification_html(params: dict[str, Any], page_index: int) -> Path:
            """Print-styled markup with absolute asset URLs, for fair comparison."""
            options = HtmlRenderOptions(
                asset_url_prefix=f"file://{workspace.root}",
                global_offset_mm=(
                    float(params.get("dx_mm", 0.0)),
                    float(params.get("dy_mm", 0.0)),
                ),
                font_scale=float(params.get("font_scale", 1.0)),
            )
            markup = render_document_for_print(document, options)
            path = workspace.dir("verify") / (
                f"api_iter_{page_index}_{abs(hash(json.dumps(params, sort_keys=True))) % 10**8}.html"
            )
            path.write_text(markup, encoding="utf-8")
            return path

        html_path = write_verification_html({}, document.pages[0].index if document.pages else 0)

        results: dict[str, Any] = {}
        for page_ir in document.pages:
            if request.page is not None and page_ir.index != request.page:
                continue
            original = cv2.imread(str(workspace.absolute(page_ir.image_path)), cv2.IMREAD_COLOR)
            if original is None:
                continue
            work_dir = workspace.dir("verify") / f"page_{page_ir.index:02d}"
            work_dir.mkdir(parents=True, exist_ok=True)

            def render_with(params: dict[str, Any], page_index: int = page_ir.index) -> Path:
                return write_verification_html(params, page_index)

            if request.run_correction_loop:
                result = correction_loop(document, page_ir, original, html_path, work_dir, settings, render_with)
            else:
                result = verify_page(page_ir, document, original, html_path, work_dir, settings)

            page_ir.quality.similarity = result.metrics.similarity
            page_ir.quality.ssim = result.metrics.ssim
            page_ir.quality.edge_iou = result.metrics.edge_iou
            page_ir.quality.pixel_diff_ratio = result.metrics.pixel_diff_ratio
            page_ir.quality.projection_correlation = result.metrics.projection_correlation
            page_ir.quality.text_similarity = result.text_similarity
            page_ir.quality.unresolved_issues = result.issues
            page_ir.quality.iterations = len(result.iterations)
            page_ir.metadata["correction"] = result.applied_adjustments
            repository.save_quality(document_id, page_ir.index, result.metrics.to_dict())
            repository.save_iterations(document_id, page_ir.index, result.iterations)
            results[str(page_ir.index)] = result.to_dict()

        document.revision += 1
        repository.save_ir(document_id, document.revision, document.model_dump_json())
        return {"pages": results, "revision": document.revision}

    @app.post("/api/documents/{document_id}/export", dependencies=guarded)
    def export(document_id: str, request: ExportRequest) -> dict[str, Any]:
        document = load_ir(document_id)
        workspace = pipeline.workspace(document_id)
        formats = [fmt for fmt in request.formats if fmt in {"html", "pdf", "docx"}]
        if not formats:
            raise HTTPException(status_code=400, detail="no supported formats requested")
        re_render(
            document,
            settings,
            workspace,
            asset_url_prefix=f"/api/documents/{document_id}/files",
            formats=formats,
        )
        exports = export_artifacts(document, workspace, formats)
        for kind, descriptor in exports.items():
            repository.record_export(document_id, kind, descriptor["path"], descriptor["size_bytes"])
        return {
            "exports": {
                kind: {
                    **descriptor,
                    "url": f"/api/documents/{document_id}/exports/{kind}",
                }
                for kind, descriptor in exports.items()
            }
        }

    @app.get("/api/documents/{document_id}/exports/{kind}", dependencies=guarded)
    def download_export(document_id: str, kind: str) -> FileResponse:
        workspace = workspace_for(document_id)
        names = {
            "html": ("reconstruction.html", "text/html"),
            "pdf": ("reconstruction.pdf", "application/pdf"),
            "docx": (
                "reconstruction.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
        }
        if kind not in names:
            raise HTTPException(status_code=400, detail=f"unknown export format {kind}")
        filename, content_type = names[kind]
        path = workspace.path("exports", f"{document_id}_{filename}")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="export not generated yet; POST /export first")
        return FileResponse(path, media_type=content_type, filename=filename)

    # -- workspace files --------------------------------------------------
    @app.get("/api/documents/{document_id}/files/{relative_path:path}", dependencies=guarded)
    def serve_file(document_id: str, relative_path: str) -> FileResponse:
        workspace = workspace_for(document_id)
        try:
            target = workspace.absolute(relative_path)
        except PainterError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not target.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(target)

    # -- maintenance ------------------------------------------------------
    @app.post("/api/maintenance/cleanup", dependencies=guarded)
    def cleanup(keep_days: int = 30, dry_run: bool = True) -> dict[str, Any]:
        removed = cleanup_workspaces(settings.documents_dir, repository, keep_days, dry_run)
        return {"removed": removed, "dry_run": dry_run}

    # -- frontend ---------------------------------------------------------
    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

        @app.get("/", response_class=HTMLResponse)
        def index() -> HTMLResponse:
            return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))

    return app


app = create_app()
