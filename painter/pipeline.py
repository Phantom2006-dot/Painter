"""Pipeline orchestration.

Stages (each with defined inputs, outputs and error handling)::

    validate -> preprocess -> detect -> recognize -> reconstruct
             -> render -> verify -> export

Every stage writes a JSON artefact into the document workspace and a row into
``job_stages``, so a failed run can be retried from the failing stage instead of
repeating OCR and API calls. Statuses follow the brief:
QUEUED -> PREPROCESSING -> DETECTING -> RECOGNIZING -> RECONSTRUCTING ->
VERIFYING -> COMPLETED (or FAILED).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .assets import AssetStore, recover_image_asset, render_overlay
from .config import Settings, get_settings
from .detection import Detection, DetectionBundle, LayoutDetectorService
from .errors import PainterError, PipelineError, StageFailed, UploadValidationError
from .geometry import analyse_page_geometry, detect_page_format, infer_margins, orientation_of
from .logging_utils import get_logger, log_event
from .ocr import OcrChain, OcrResult, build_ocr_chain, recognize_regions
from .preprocess import LoadedPage, load_document_pages, preprocess_page, validate_upload
from .schemas import (
    DocumentIR,
    DocumentSource,
    Element,
    ElementType,
    PageFormat,
    PageIR,
    Provenance,
    ReviewFlag,
    new_id,
    utcnow,
    validate_document,
)
from .serpapi_client import SerpApiClient, UsageLedger
from .storage import DocumentWorkspace, Repository
from .structure import TEXT_DETECTION_TYPES, build_page_ir
from .tables import reconstruct_table
from .typography import estimate_page_typography
from .verify import correction_loop, verify_page

logger = get_logger("pipeline")

STAGES: tuple[str, ...] = (
    "validate",
    "preprocess",
    "detect",
    "recognize",
    "reconstruct",
    "render",
    "verify",
    "export",
)

STAGE_STATUS = {
    "validate": "preprocessing",
    "preprocess": "preprocessing",
    "detect": "detecting",
    "recognize": "recognizing",
    "reconstruct": "reconstructing",
    "render": "reconstructing",
    "verify": "verifying",
    "export": "completed",
}


@dataclass
class PipelineOptions:
    """Per-run options."""

    force_stages: set[str] = field(default_factory=set)
    enable_verify: bool | None = None
    run_correction_loop: bool = True
    enable_lens: bool | None = None
    enable_assets_download: bool | None = None
    ocr_provider: str | None = None
    max_pages: int | None = None
    asset_url_prefix: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "force_stages": sorted(self.force_stages),
            "enable_verify": self.enable_verify,
            "run_correction_loop": self.run_correction_loop,
            "enable_lens": self.enable_lens,
            "enable_assets_download": self.enable_assets_download,
            "ocr_provider": self.ocr_provider,
            "max_pages": self.max_pages,
        }


@dataclass
class JobContext:
    """State shared by the stages of one run."""

    document_id: str
    job_id: str
    workspace: DocumentWorkspace
    settings: Settings
    repository: Repository
    options: PipelineOptions
    data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    #: Assumptions the pipeline had to make (inferred paper size, missing model,
    #: disabled OCR...). Reported to the user instead of being hidden.
    assumptions: list[str] = field(default_factory=list)
    serpapi: SerpApiClient | None = None
    ocr_chain: OcrChain | None = None
    detectors: LayoutDetectorService | None = None

    def log(self, message: str, level: str = "info", stage: str | None = None) -> None:
        """Write one line to this job's live log."""
        try:
            self.repository.append_log(self.job_id, message, level=level, stage=stage)
        except Exception:  # logging must never break a run
            logger.debug("could not append job log: %s", message)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
            log_event(logger, "warning", document_id=self.document_id, message=message)

    def fail(self, stage: str, exc: Exception) -> None:
        payload = (
            exc.to_dict()
            if isinstance(exc, PainterError)
            else {"error": type(exc).__name__, "code": "unexpected", "message": str(exc)}
        )
        payload["stage"] = stage
        self.errors.append(payload)
        self.repository.record_error(
            self.document_id, self.job_id, stage, payload.get("code", "error"), payload.get("message", "")
        )


@dataclass
class JobResult:
    document_id: str
    job_id: str
    status: str
    stages: dict[str, Any] = field(default_factory=dict)
    exports: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    document: DocumentIR | None = None
    elapsed_s: float = 0.0

    def to_dict(self, include_document: bool = False) -> dict[str, Any]:
        payload = {
            "document_id": self.document_id,
            "job_id": self.job_id,
            "status": self.status,
            "stages": self.stages,
            "exports": self.exports,
            "warnings": self.warnings,
            "errors": self.errors,
            "assumptions": self.assumptions,
            "elapsed_s": self.elapsed_s,
        }
        if include_document and self.document is not None:
            payload["document"] = json.loads(self.document.model_dump_json())
        return payload


class DocumentPipeline:
    """Runs the reconstruction pipeline for one document."""

    def __init__(self, settings: Settings | None = None, repository: Repository | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.repository = repository or Repository(self.settings.db_path)

    # -- document creation ----------------------------------------------
    def workspace(self, document_id: str) -> DocumentWorkspace:
        return DocumentWorkspace(self.settings.documents_dir / document_id, document_id)

    def create_document(self, source_path: Path, filename: str | None = None) -> str:
        """Validate an upload, copy it into a workspace and register it."""
        source_path = Path(source_path)
        detected = validate_upload(source_path, self.settings)
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        document_id = new_id("doc")
        workspace = self.workspace(document_id).ensure()
        target = workspace.path("source", f"original.{detected}")
        shutil.copy2(source_path, target)
        self.repository.create_document(
            document_id=document_id,
            filename=filename or source_path.name,
            stored_path=workspace.relative(target),
            sha256=digest,
            content_type=detected,
            size_bytes=source_path.stat().st_size,
            metadata={"original_filename": filename or source_path.name},
        )
        log_event(logger, "document created", document_id=document_id, format=detected)
        return document_id

    # -- stage artefact helpers -----------------------------------------
    def _stage_path(self, ctx: JobContext, stage: str) -> Path:
        return ctx.workspace.path("stages", f"{stage}.json")

    def _write_stage(self, ctx: JobContext, stage: str, payload: dict[str, Any]) -> None:
        path = self._stage_path(ctx, stage)
        path.write_text(json.dumps(payload, default=str, ensure_ascii=False), encoding="utf-8")

    def _read_stage(self, ctx: JobContext, stage: str) -> dict[str, Any] | None:
        path = self._stage_path(ctx, stage)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    # -- main entry point ------------------------------------------------
    def run(
        self,
        document_id: str,
        job_id: str | None = None,
        options: PipelineOptions | None = None,
        progress: Callable[[str, float], None] | None = None,
    ) -> JobResult:
        options = options or PipelineOptions()
        job_id = job_id or new_id("job")
        workspace = self.workspace(document_id)
        # A per-run OCR provider override must reach the chain that is built for
        # this run, otherwise the option is silently ignored.
        run_settings = self.settings
        if options.ocr_provider and options.ocr_provider != self.settings.ocr_provider:
            run_settings = replace(self.settings, ocr_provider=options.ocr_provider)
        ctx = JobContext(
            document_id=document_id,
            job_id=job_id,
            workspace=workspace,
            settings=run_settings,
            repository=self.repository,
            options=options,
        )
        started = time.perf_counter()
        self.repository.create_job(job_id, document_id, options.to_dict())
        self.repository.update_job(job_id, status="running", started_at=utcnow())

        ctx.serpapi = self._build_serpapi(document_id, options)
        ctx.ocr_chain = build_ocr_chain(run_settings, ctx.serpapi)
        ctx.detectors = LayoutDetectorService(run_settings)

        result = JobResult(document_id=document_id, job_id=job_id, status="running")
        completed = self.repository.completed_stages(job_id)
        total = len(STAGES)

        for position, stage in enumerate(STAGES):
            self.repository.update_job(
                job_id, stage=stage, status=STAGE_STATUS.get(stage, "running"), progress=round(position / total, 3)
            )
            if progress:
                progress(stage, position / total)
            cached = self._read_stage(ctx, stage)
            if cached is not None and stage not in options.force_stages and stage not in completed:
                # artefacts exist from an earlier run: reuse them
                log_event(logger, f"stage {stage} reused from cache", document_id=document_id, stage=stage)
                result.stages[stage] = {"status": "reused", "details": cached.get("summary", {})}
                self._hydrate(ctx, stage, cached)
                continue
            if stage in completed and stage not in options.force_stages and cached is None:
                pass  # fall through and re-run; artefacts were removed

            stage_started = time.perf_counter()
            ctx.log(f"stage {stage} starting", stage=stage)
            try:
                details = self._run_stage(ctx, stage)
            except Exception as exc:  # noqa: BLE001 - recorded, then reported
                duration_ms = round((time.perf_counter() - stage_started) * 1000, 2)
                ctx.fail(stage, exc)
                self.repository.record_stage(
                    job_id, stage, "failed", duration_ms, error=f"{type(exc).__name__}: {exc}"
                )
                if stage in {"validate", "preprocess"}:
                    self.repository.update_job(job_id, status="failed", error=str(exc), finished_at=utcnow())
                    result.status = "failed"
                    result.stages[stage] = {"status": "failed", "error": str(exc)}
                    result.warnings = ctx.warnings
                    result.errors = ctx.errors
                    result.elapsed_s = round(time.perf_counter() - started, 2)
                    return result
                result.stages[stage] = {"status": "failed", "error": str(exc)}
                ctx.log(f"stage {stage} failed: {type(exc).__name__}: {exc}", level="error", stage=stage)
                log_event(logger, f"stage {stage} failed", document_id=document_id, stage=stage, error=str(exc))
                continue
            duration_ms = round((time.perf_counter() - stage_started) * 1000, 2)
            self.repository.record_stage(job_id, stage, "completed", duration_ms, details=details)
            result.stages[stage] = {"status": "completed", "duration_ms": duration_ms, "details": details}
            _summary = _summarise_details(details)
            ctx.log(
                f"stage {stage} completed in {duration_ms / 1000:.1f}s"
                + (f" \u2014 {_summary}" if _summary else ""),
                stage=stage,
            )
            log_event(
                logger,
                f"stage {stage} completed",
                document_id=document_id,
                stage=stage,
                duration_ms=duration_ms,
            )

        result.document = ctx.data.get("document")
        if result.document is not None:
            result.document.recompute_stats()
            result.document.touch()
            self.repository.save_ir(
                document_id, result.document.revision, result.document.model_dump_json()
            )
            self.repository.update_document(
                document_id,
                status="completed",
                page_count=len(result.document.pages),
                metadata_json={"stats": result.document.stats},
            )
        result.exports = ctx.data.get("exports", {})
        result.warnings = ctx.warnings
        result.errors = ctx.errors
        result.assumptions = ctx.assumptions
        result.status = "failed" if any(s.get("status") == "failed" for s in result.stages.values()) else "completed"
        result.elapsed_s = round(time.perf_counter() - started, 2)
        self.repository.update_job(
            job_id,
            status=result.status,
            stage="export",
            progress=1.0,
            finished_at=utcnow(),
            error=None if result.status == "completed" else "one or more stages failed",
        )
        if progress:
            progress("export", 1.0)
        return result

    def _build_serpapi(self, document_id: str, options: PipelineOptions) -> SerpApiClient | None:
        settings = self.settings
        serpapi = settings.serpapi
        enabled = serpapi.enabled if options.enable_lens is None else options.enable_lens
        if not enabled:
            return None
        if not serpapi.configured:
            logger.warning("SerpApi was requested but SERPAPI_KEY is not set; continuing without it")
            return None
        ledger = UsageLedger(settings.cache_dir / "serpapi_usage.json")
        return SerpApiClient(serpapi, settings.cache_dir, ledger, document_id)

    def _hydrate(self, ctx: JobContext, stage: str, payload: dict[str, Any]) -> None:
        """Reload cached stage artefacts into the context."""
        if stage == "validate":
            ctx.data["source"] = payload.get("source", {})
        elif stage == "preprocess":
            ctx.data["pages"] = payload.get("pages", [])
        elif stage == "detect":
            ctx.data["detections"] = payload.get("pages", {})
        elif stage == "recognize":
            ctx.data["ocr"] = payload.get("pages", {})
        elif stage == "reconstruct":
            document = payload.get("document")
            if document:
                ctx.data["document"] = DocumentIR.model_validate(document)
        elif stage == "render":
            ctx.data["render"] = payload.get("render", {})
        elif stage == "verify":
            ctx.data["verify"] = payload.get("pages", {})
        elif stage == "export":
            ctx.data["exports"] = payload.get("exports", {})

    # -- stages ----------------------------------------------------------
    def _run_stage(self, ctx: JobContext, stage: str) -> dict[str, Any]:
        handler = getattr(self, f"_stage_{stage}")
        details = handler(ctx) or {}
        self._write_stage(ctx, stage, details)
        return details.get("summary", details)

    def _verification_html(
        self, ctx: JobContext, document: DocumentIR, params: dict[str, Any]
    ) -> Path:
        """Render the print-styled markup that the rasteriser compares.

        Asset URLs are absolute (``file://``) because this markup lives in the
        verification folder, and the print stylesheet removes the on-screen
        padding so the rasterised page coincides with the scanned page.
        """
        from .render import HtmlRenderOptions, render_document_for_print

        options = HtmlRenderOptions(
            title=f"verification {ctx.document_id}",
            asset_url_prefix=f"file://{ctx.workspace.root}",
            global_offset_mm=(
                float(params.get("dx_mm", 0.0)),
                float(params.get("dy_mm", 0.0)),
            ),
            font_scale=float(params.get("font_scale", 1.0)),
        )
        markup = render_document_for_print(document, options)
        digest = abs(hash(json.dumps(params, sort_keys=True))) % 10**8
        path = ctx.workspace.path("verify", f"iteration_{digest}.html")
        path.write_text(markup, encoding="utf-8")
        return path

    # 1. validate --------------------------------------------------------
    def _stage_validate(self, ctx: JobContext) -> dict[str, Any]:
        record = ctx.repository.get_document(ctx.document_id)
        source_path = ctx.workspace.absolute(record["stored_path"])
        detected = validate_upload(source_path, ctx.settings)
        ctx.data["source"] = {
            "path": str(source_path),
            "relative_path": record["stored_path"],
            "filename": record["filename"],
            "format": detected,
            "sha256": record.get("sha256"),
            "size_bytes": record.get("size_bytes"),
        }
        ctx.repository.update_document(ctx.document_id, status="validated")
        return {
            "source": ctx.data["source"],
            "summary": {"format": detected, "size_bytes": record.get("size_bytes")},
        }

    # 2. preprocess ------------------------------------------------------
    def _stage_preprocess(self, ctx: JobContext) -> dict[str, Any]:
        source = ctx.data.get("source") or {}
        source_path = Path(source.get("path"))
        pages = load_document_pages(source_path, ctx.workspace.dir("pages") / "_tmp", ctx.settings)
        max_pages = ctx.options.max_pages or ctx.settings.max_pages_per_document
        records: list[dict[str, Any]] = []
        for page in pages[:max_pages]:
            # Let preprocessing infer the paper format and effective resolution:
            # it cross-checks the assumed format against the measured text line
            # pitch, which is what keeps geometry and typography consistent.
            result = preprocess_page(page.image, ctx.settings)
            page_format = result.page_format or detect_page_format(
                page.image.shape[1],
                page.image.shape[0],
                dpi=result.metrics.get("output_dpi"),
            )
            processed_path = ctx.workspace.path("pages", f"page_{page.index:02d}.png")
            import cv2

            cv2.imwrite(str(processed_path), result.image)
            preview_path = ctx.workspace.path("pages", f"page_{page.index:02d}_preview.jpg")
            cv2.imwrite(str(preview_path), result.image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            record = {
                "index": page.index,
                "image_path": ctx.workspace.relative(processed_path),
                "preview_path": ctx.workspace.relative(preview_path),
                "original_path": ctx.workspace.relative(page.path),
                "width_px": int(result.image.shape[1]),
                "height_px": int(result.image.shape[0]),
                "transforms": [t.model_dump() for t in result.transforms],
                "metrics": result.metrics,
                "warnings": result.warnings,
                "page_format": page_format.model_dump(),
                "assumptions": result.assumptions,
                "dpi": float(
                    result.metrics.get("output_dpi")
                    or result.metrics.get("estimated_input_dpi")
                    or ctx.settings.target_dpi
                ),
            }
            records.append(record)
            for warning in result.warnings:
                ctx.warn(f"page {page.index}: {warning}")
            for assumption in result.assumptions:
                ctx.assumptions.append(f"page {page.index}: {assumption}")
            ctx.repository.upsert_page(
                ctx.document_id,
                {
                    "page_index": page.index,
                    "width_px": record["width_px"],
                    "height_px": record["height_px"],
                    "dpi": record["dpi"],
                    "page_format": page_format.name,
                    "orientation": orientation_of(record["width_px"], record["height_px"]),
                    "image_path": record["image_path"],
                    "original_path": record["original_path"],
                    "preview_path": record["preview_path"],
                    "metadata": {
                        "metrics": result.metrics,
                        "assumptions": result.assumptions,
                    },
                },
            )
        ctx.data["pages"] = records
        ctx.repository.update_document(ctx.document_id, status="preprocessed", page_count=len(records))
        return {
            "pages": records,
            "summary": {"pages": len(records), "warnings": len(ctx.warnings)},
        }

    # 3. detect ----------------------------------------------------------
    def _stage_detect(self, ctx: JobContext) -> dict[str, Any]:
        import cv2

        from .heuristics import collect_heuristic_detections

        pages = ctx.data.get("pages") or []
        detector_service = ctx.detectors or LayoutDetectorService(ctx.settings)
        results: dict[str, Any] = {}
        for record in pages:
            image = cv2.imread(str(ctx.workspace.absolute(record["image_path"])), cv2.IMREAD_COLOR)
            if image is None:
                raise StageFailed(f"page image missing for page {record['index']}")
            bundle: DetectionBundle = detector_service.detect(image)
            extra = collect_heuristic_detections(image, bundle.detections)
            all_detections = list(bundle.detections) + extra
            payload = {
                "detector": bundle.detector,
                "used_fallback": bundle.used_fallback,
                "elapsed_s": bundle.elapsed_s,
                "models": [report.to_dict() for report in bundle.model_reports],
                "detections": [d.to_dict() for d in all_detections],
                "warnings": bundle.warnings,
            }
            results[str(record["index"])] = payload
            for warning in bundle.warnings:
                ctx.warn(f"page {record['index']}: {warning}")
            overlay = render_overlay(image, _to_elements(all_detections))
            overlay_path = ctx.workspace.path("pages", f"page_{record['index']:02d}_overlay.jpg")
            cv2.imwrite(str(overlay_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            results[str(record["index"])]["overlay_path"] = ctx.workspace.relative(overlay_path)
            ctx.repository.save_detections(ctx.document_id, record["index"], payload["detections"])
        ctx.data["detections"] = results
        return {
            "pages": results,
            "summary": {
                "detector": next(iter(results.values()))["detector"] if results else "none",
                "detections": sum(len(r["detections"]) for r in results.values()),
            },
        }

    # 4. recognize -------------------------------------------------------
    def _stage_recognize(self, ctx: JobContext) -> dict[str, Any]:
        import cv2

        pages = ctx.data.get("pages") or []
        detections = ctx.data.get("detections") or {}
        chain = ctx.ocr_chain or build_ocr_chain(ctx.settings, ctx.serpapi)
        results: dict[str, Any] = {}
        for record in pages:
            key = str(record["index"])
            page_detections = [
                _from_dict(d) for d in (detections.get(key, {}).get("detections") or [])
            ]
            image = cv2.imread(str(ctx.workspace.absolute(record["image_path"])), cv2.IMREAD_COLOR)
            text_regions = [
                (f"det_{index}", detection.bbox)
                for index, detection in enumerate(
                    [d for d in page_detections if d.element_type in TEXT_DETECTION_TYPES]
                )
            ]
            if text_regions:
                ctx.log(
                    f"page {record['index']}: recognising {len(text_regions)} text region(s) "
                    f"with {chain.name if chain else 'no provider'}",
                    stage="recognize",
                )
            progress_state = {"last": time.perf_counter()}

            def _ocr_progress(done: int, total_regions: int, _element_id: str) -> None:
                # Report every 5% and at least every 4 seconds. One dense page is a
                # single long loop, and without this the UI has nothing to show.
                now = time.perf_counter()
                step = max(1, total_regions // 20)
                if done >= total_regions or done % step == 0 or now - progress_state["last"] > 4.0:
                    progress_state["last"] = now
                    ctx.log(
                        f"page {record['index']}: OCR {done}/{total_regions} regions "
                        f"({done * 100 // max(1, total_regions)}%)",
                        stage="recognize",
                    )

            ocr_results = (
                recognize_regions(chain, image, text_regions, on_progress=_ocr_progress)
                if text_regions
                else {}
            )
            serialised = {name: result.to_dict() for name, result in ocr_results.items()}
            results[key] = serialised
            for name, result in ocr_results.items():
                ctx.repository.save_ocr(ctx.document_id, record["index"], name, result.to_dict())
            empty = sum(1 for result in ocr_results.values() if not result.text.strip())
            if empty:
                ctx.warn(f"page {record['index']}: {empty} text region(s) produced no text")
        ctx.data["ocr"] = results
        provider = chain.name if chain else "none"
        return {
            "pages": results,
            "summary": {
                "provider": provider,
                "regions": sum(len(r) for r in results.values()),
                "serpapi": ctx.serpapi.usage_snapshot() if ctx.serpapi else None,
            },
        }

    # 5. reconstruct -----------------------------------------------------
    def _stage_reconstruct(self, ctx: JobContext) -> dict[str, Any]:
        import cv2

        pages = ctx.data.get("pages") or []
        detections = ctx.data.get("detections") or {}
        ocr_pages = ctx.data.get("ocr") or {}
        chain = ctx.ocr_chain or build_ocr_chain(ctx.settings, ctx.serpapi)

        document = DocumentIR(
            document_id=ctx.document_id,
            job_id=ctx.job_id,
            source=DocumentSource(
                filename=(ctx.data.get("source") or {}).get("filename", "upload"),
                stored_path=(ctx.data.get("source") or {}).get("relative_path", ""),
                sha256=(ctx.data.get("source") or {}).get("sha256") or "",
                content_type=(ctx.data.get("source") or {}).get("format"),
                size_bytes=int((ctx.data.get("source") or {}).get("size_bytes") or 0),
                page_count=len(pages),
            ),
            metadata={"title": None, "language": "en"},
        )
        document.metadata.extra["preprocessing"] = [
            {"page": record["index"], "transforms": record["transforms"], "metrics": record["metrics"]}
            for record in pages
        ]

        def make_table_builder(mm_per_px: float) -> Any:
            def table_builder(image: np.ndarray, box: Any) -> tuple[Any, list[Any]]:
                return reconstruct_table(image, box, chain, ctx.settings, mm_per_px=mm_per_px)

            return table_builder

        for record in pages:
            key = str(record["index"])
            image = cv2.imread(str(ctx.workspace.absolute(record["image_path"])), cv2.IMREAD_COLOR)
            if image is None:
                raise StageFailed(f"page image missing for page {record['index']}")
            page_detections = [_from_dict(d) for d in (detections.get(key, {}).get("detections") or [])]
            ocr_results = {
                name: _ocr_from_dict(payload)
                for name, payload in (ocr_pages.get(key) or {}).items()
            }
            page_format = PageFormat.model_validate(record["page_format"])
            # px -> mm for this page, so tables can be stored in real units
            mm_per_px = page_format.width_mm / max(int(image.shape[1]), 1)
            table_builder = make_table_builder(mm_per_px)
            from .schemas import PageTransform

            transforms = [PageTransform.model_validate(t) for t in record["transforms"]]
            build = build_page_ir(
                index=record["index"],
                image=image,
                detections=page_detections,
                ocr_results=ocr_results,
                page_format=page_format,
                transforms=transforms,
                settings=ctx.settings,
                detector_name=(detections.get(key, {}).get("detector") or "unknown"),
                table_builder=table_builder,
                image_path=record["image_path"],
                original_path=record.get("original_path"),
                dpi=record.get("dpi"),
            )
            page = build.page
            page.preview_path = record.get("preview_path")
            for warning in build.warnings:
                ctx.warn(f"page {record['index']}: {warning}")

            # image assets: extract from the page, then optionally search online
            store = AssetStore(ctx.workspace.dir("assets"))
            for element in page.elements:
                if element.type in {
                    ElementType.IMAGE,
                    ElementType.LOGO,
                    ElementType.SIGNATURE,
                    ElementType.STAMP,
                }:
                    try:
                        content = recover_image_asset(
                            element, image, store, ctx.settings, ctx.serpapi
                        )
                        element.image = content
                        ctx.repository.save_asset(ctx.document_id, element.id, content.model_dump())
                        if content.asset_origin.value == "extracted":
                            element.add_flag(ReviewFlag.IMAGE_UNMATCHED)
                    except Exception as exc:  # noqa: BLE001
                        ctx.warn(f"asset recovery failed for {element.id}: {exc}")

            # typography
            for warning in estimate_page_typography(page, image, ctx.settings):
                ctx.warn(f"page {record['index']}: {warning}")
            _promote_page_title(page, ctx)
            for element in page.elements:
                if element.typography is not None:
                    ctx.repository.save_typography(ctx.document_id, element.id, element.typography.model_dump())

            # geometry
            geometry = analyse_page_geometry(page.elements, page.width_px, page.height_px, page_format)
            page.margins_mm = geometry.margins_mm
            page.metadata["geometry"] = {
                "columns": geometry.columns,
                "overlaps": geometry.overlaps,
                "line_spacing_ratio": geometry.line_spacing_ratio,
                "notes": geometry.notes,
            }
            for note in geometry.notes:
                ctx.warn(f"page {record['index']}: {note}")
            page.metadata["margins_source"] = "estimated from content extents"
            document.pages.append(page)

        report = validate_document(document)
        for issue in report.issues:
            message = f"{issue.code}: {issue.message}"
            if issue.level == "error":
                ctx.warn(message)
                if issue.element_id:
                    found = document.find_element(issue.element_id)
                    if found:
                        found[1].add_flag(ReviewFlag.NEEDS_REVIEW)
            else:
                ctx.warn(message)
                # Missing text and low confidence always require a human look.
                if issue.code in {"text_missing", "low_confidence"} and issue.element_id:
                    found = document.find_element(issue.element_id)
                    if found:
                        found[1].add_flag(ReviewFlag.NEEDS_REVIEW)
        document.warnings = list(ctx.warnings)
        document.recompute_stats()
        document.metadata.extra["validation"] = report.model_dump()
        document.metadata.extra["assumptions"] = list(ctx.assumptions)
        document.metadata.extra["page_formats"] = [
            {"page": page.index, "format": page.page_format.model_dump()} for page in document.pages
        ]
        ctx.data["document"] = document

        payload = json.loads(document.model_dump_json())
        return {
            "document": payload,
            "summary": {
                "pages": len(document.pages),
                "elements": document.stats.get("elements", 0),
                "validation_ok": report.ok,
                "validation_issues": len(report.issues),
            },
        }

    # 6. render ----------------------------------------------------------
    def _stage_render(self, ctx: JobContext) -> dict[str, Any]:
        document = ctx.data.get("document")
        if document is None:
            raise StageFailed("no Document IR available to render")
        from .render import HtmlRenderOptions, render_docx, render_document, render_pdf

        corrections = _correction_options(document)
        options = HtmlRenderOptions(
            asset_url_prefix=ctx.options.asset_url_prefix,
            title=document.source.filename if document.source else "Painter reconstruction",
            global_offset_mm=corrections.get("offset_mm", (0.0, 0.0)),
            font_scale=corrections.get("font_scale", 1.0),
        )
        html = render_document(document, options)
        html_path = ctx.workspace.path("rendered", "reconstruction.html")
        html_path.write_text(html, encoding="utf-8")

        pdf_result = render_pdf(document, ctx.workspace.path("rendered", "reconstruction.pdf"), options)
        docx_result = render_docx(
            document,
            ctx.workspace.path("rendered", "reconstruction.docx"),
            asset_base=ctx.workspace.root,
            page_offset_mm=options.global_offset_mm,
        )
        render_info = {
            "html": ctx.workspace.relative(html_path),
            "pdf": ctx.workspace.relative(pdf_result.path),
            "docx": ctx.workspace.relative(docx_result.path),
            "pdf_engine": pdf_result.engine,
            "pdf_has_text": pdf_result.has_text,
            "pdf_text_characters": pdf_result.text_characters,
            "docx_paragraphs": docx_result.paragraphs,
            "docx_tables": docx_result.tables,
            "docx_pictures": docx_result.pictures,
            "docx_anchored_pictures": docx_result.anchored_pictures,
            "warnings": pdf_result.warnings + docx_result.warnings,
        }
        for warning in render_info["warnings"]:
            ctx.warn(warning)
        if not pdf_result.has_text:
            ctx.warn("the generated PDF has no extractable text; verify the renderer output")
        ctx.data["render"] = render_info
        return {
            "render": render_info,
            "summary": {
                "pdf_engine": pdf_result.engine,
                "pdf_has_text": pdf_result.has_text,
                "docx_paragraphs": docx_result.paragraphs,
                "docx_tables": docx_result.tables,
            },
        }

    # 7. verify ----------------------------------------------------------
    def _stage_verify(self, ctx: JobContext) -> dict[str, Any]:
        import cv2

        document: DocumentIR | None = ctx.data.get("document")
        render_info = ctx.data.get("render") or {}
        enabled = ctx.settings.verify_enabled if ctx.options.enable_verify is None else ctx.options.enable_verify
        if document is None or not render_info.get("html"):
            raise StageFailed("nothing to verify")
        if not enabled:
            return {"pages": {}, "summary": {"skipped": True, "reason": "verification disabled"}}

        from .render import HtmlRenderOptions, render_document_for_print

        # The comparison uses the print stylesheet (no viewer padding) and
        # absolute asset URLs, so the rasterised page lines up with the scan and
        # images are actually visible.
        html_path = self._verification_html(ctx, document, {"dx_mm": 0.0, "dy_mm": 0.0, "font_scale": 1.0})
        results: dict[str, Any] = {}
        for page in document.pages:
            original = cv2.imread(str(ctx.workspace.absolute(page.image_path)), cv2.IMREAD_COLOR)
            if original is None:
                ctx.warn(f"page {page.index}: original page image missing, verification skipped")
                continue
            work_dir = ctx.workspace.dir("verify") / f"page_{page.index:02d}"
            work_dir.mkdir(parents=True, exist_ok=True)

            if ctx.options.run_correction_loop:
                def render_with(params: dict[str, Any]) -> Path:
                    return self._verification_html(ctx, document, params)

                verification = correction_loop(
                    document, page, original, html_path, work_dir, ctx.settings, render_with
                )
            else:
                verification = verify_page(page, document, original, html_path, work_dir, ctx.settings)

            page.quality.similarity = verification.metrics.similarity
            page.quality.ssim = verification.metrics.ssim
            page.quality.pixel_diff_ratio = verification.metrics.pixel_diff_ratio
            page.quality.edge_iou = verification.metrics.edge_iou
            page.quality.projection_correlation = verification.metrics.projection_correlation
            page.quality.text_similarity = verification.text_similarity
            page.quality.metrics_used = sorted(verification.metrics.weights)
            page.quality.iterations = len(verification.iterations)
            page.quality.unresolved_issues = verification.issues
            page.metadata["correction"] = verification.applied_adjustments
            for warning in verification.warnings:
                ctx.warn(f"page {page.index}: {warning}")
            ctx.repository.save_quality(ctx.document_id, page.index, verification.metrics.to_dict())
            ctx.repository.save_iterations(ctx.document_id, page.index, verification.iterations)
            results[str(page.index)] = verification.to_dict()

        # if the loop improved geometry, re-render the final artefacts once
        corrections = _correction_options(document)
        if corrections.get("applied"):
            from .render import HtmlRenderOptions, render_docx, render_document, render_pdf

            options = HtmlRenderOptions(
                asset_url_prefix=ctx.options.asset_url_prefix,
                global_offset_mm=corrections["offset_mm"],
                font_scale=corrections["font_scale"],
            )
            html_path.write_text(render_document(document, options), encoding="utf-8")
            render_pdf(document, ctx.workspace.path("rendered", "reconstruction.pdf"), options)
            render_docx(
                document,
                ctx.workspace.path("rendered", "reconstruction.docx"),
                asset_base=ctx.workspace.root,
                page_offset_mm=corrections["offset_mm"],
            )
            render_info["corrected"] = True
            render_info["correction"] = corrections
            ctx.data["render"] = render_info
            log_event(logger, "re-rendered with correction offsets", document_id=ctx.document_id)

        ctx.data["verify"] = results
        similarities = [
            page.quality.similarity for page in document.pages if page.quality.similarity is not None
        ]
        return {
            "pages": results,
            "summary": {
                "pages_verified": len(results),
                "mean_similarity": round(sum(similarities) / len(similarities), 4) if similarities else None,
                "corrected": bool(corrections.get("applied")),
            },
        }

    # 8. export ----------------------------------------------------------
    def _stage_export(self, ctx: JobContext) -> dict[str, Any]:
        document = ctx.data.get("document")
        render_info = ctx.data.get("render") or {}
        exports_dir = ctx.workspace.dir("exports")
        exports: dict[str, Any] = {}
        mapping = {
            "html": ("reconstruction.html", "text/html"),
            "pdf": ("reconstruction.pdf", "application/pdf"),
            "docx": ("reconstruction.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        }
        for kind, (filename, content_type) in mapping.items():
            source = render_info.get(kind)
            if not source:
                continue
            source_path = ctx.workspace.absolute(source)
            if not source_path.is_file():
                ctx.warn(f"export artefact missing: {source}")
                continue
            target = exports_dir / f"{ctx.document_id}_{filename}"
            shutil.copy2(source_path, target)
            size = target.stat().st_size
            exports[kind] = {
                "path": ctx.workspace.relative(target),
                "size_bytes": size,
                "content_type": content_type,
                "filename": f"{filename}",
            }
            ctx.repository.record_export(ctx.document_id, kind, ctx.workspace.relative(target), size)

        if document is not None:
            ir_path = ctx.workspace.path("rendered", "document_ir.json")
            ir_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
            exports["ir"] = {
                "path": ctx.workspace.relative(ir_path),
                "size_bytes": ir_path.stat().st_size,
                "content_type": "application/json",
                "filename": "document_ir.json",
            }
            ctx.repository.save_ir(ctx.document_id, document.revision, document.model_dump_json())

        ctx.data["exports"] = exports
        ctx.repository.update_document(ctx.document_id, status="completed")
        return {
            "exports": exports,
            "summary": {"formats": sorted(exports), "files": len(exports)},
        }

    # -- convenience -----------------------------------------------------
    def process_file(
        self,
        path: Path,
        filename: str | None = None,
        options: PipelineOptions | None = None,
    ) -> JobResult:
        """Create a document from ``path`` and run the full pipeline."""
        document_id = self.create_document(path, filename)
        return self.run(document_id, options=options)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _to_elements(detections: Sequence[Detection]) -> list[Element]:
    """Lightweight elements used only for the detection overlay image."""
    elements: list[Element] = []
    for detection in detections:
        try:
            elements.append(
                Element(
                    type=detection.element_type,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    detection_class=detection.class_name,
                )
            )
        except Exception:
            continue
    return elements


def _summarise_details(details: Any) -> str:
    """One readable clause describing what a stage produced."""
    if not isinstance(details, dict):
        return ""
    summary = details.get("summary") if isinstance(details.get("summary"), dict) else {}
    parts: list[str] = []
    for key in ("pages", "detections", "elements", "regions", "provider"):
        value = summary.get(key, details.get(key))
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, dict)):
            value = len(value)
        parts.append(f"{key}={value}")
    return ", ".join(parts[:4])


def _from_dict(payload: dict[str, Any]) -> Detection:
    from .schemas import BBox

    return Detection(
        class_name=payload.get("class_name", "unknown"),
        element_type=ElementType(payload.get("type", "unknown")),
        confidence=float(payload.get("confidence") or 0.0),
        bbox=BBox.from_list(payload.get("bbox") or [0, 0, 0, 0]),
        source=payload.get("source", "unknown"),
    )


def _promote_page_title(page: PageIR, ctx: JobContext) -> None:
    """Treat a page's first heading-sized text block as the document title.

    Measured on the generated page, DocLayout-YOLO labels the title
    ``Section-header`` (0.79) rather than ``Title`` (0.66) — both are defensible
    labels for one region, so the structure layer decides: the topmost text block
    that is clearly larger than the body text is a heading. The decision is
    recorded in the element metadata and provenance, never applied silently.
    """
    text_types = {
        ElementType.HEADING,
        ElementType.SUBHEADING,
        ElementType.PARAGRAPH,
        ElementType.TEXT_BLOCK,
    }
    text_elements = [
        element
        for element in page.elements
        if element.type in text_types and (element.text or "").strip()
    ]
    if not text_elements:
        return
    top = min(text_elements, key=lambda element: element.bbox.y1)
    if top.type is not ElementType.SUBHEADING or top.typography is None:
        return
    size = top.typography.font_size_pt
    if not size:
        return
    body_sizes = sorted(
        element.typography.font_size_pt
        for element in text_elements
        if element.type in {ElementType.PARAGRAPH, ElementType.TEXT_BLOCK}
        and element.typography is not None
        and element.typography.font_size_pt
    )
    body = body_sizes[len(body_sizes) // 2] if body_sizes else None
    if body and size < body * 1.4:
        return
    if top.bbox.y1 > page.height_px * 0.35:
        return
    top.type = ElementType.HEADING
    top.metadata["promoted_from"] = "subheading"
    top.metadata["promotion_reason"] = (
        f"topmost text block, {size:.1f}pt against {body:.1f}pt body text" if body
        else f"topmost text block, {size:.1f}pt"
    )
    top.provenance.append(
        Provenance(
            stage="structure",
            method="promoted to heading: topmost heading-sized block",
            provider="painter.structure",
            notes=top.metadata["promotion_reason"],
        )
    )
    ctx.warn(f"page {page.index}: promoted {top.id} from subheading to heading (topmost heading-sized block)")


def _ocr_from_dict(payload: dict[str, Any]) -> OcrResult:
    from .schemas import BBox, ReviewFlag

    lines = []
    from .ocr import OcrLine

    for line in payload.get("lines") or []:
        lines.append(
            OcrLine(
                text=line.get("text", ""),
                bbox=BBox.from_list(line.get("bbox") or [0, 0, 0, 0]),
                confidence=line.get("confidence"),
            )
        )
    flags = []
    for flag in payload.get("flags") or []:
        try:
            flags.append(ReviewFlag(flag))
        except ValueError:
            continue
    return OcrResult(
        text=payload.get("text", ""),
        lines=lines,
        confidence=payload.get("confidence"),
        provider=payload.get("provider", "unknown"),
        method=payload.get("method", "unknown"),
        flags=flags,
        warnings=payload.get("warnings") or [],
    )


def _correction_options(document: DocumentIR) -> dict[str, Any]:
    """Aggregate the correction parameters chosen per page."""
    offsets: list[tuple[float, float]] = []
    scales: list[float] = []
    applied = False
    for page in document.pages:
        correction = page.metadata.get("correction") or {}
        dx = float(correction.get("dx_mm", 0.0) or 0.0)
        dy = float(correction.get("dy_mm", 0.0) or 0.0)
        scale = float(correction.get("font_scale", 1.0) or 1.0)
        offsets.append((dx, dy))
        scales.append(scale)
        if abs(dx) > 0.01 or abs(dy) > 0.01 or abs(scale - 1.0) > 0.001:
            applied = True
    if not offsets:
        return {"offset_mm": (0.0, 0.0), "font_scale": 1.0, "applied": False}
    mean_dx = sum(o[0] for o in offsets) / len(offsets)
    mean_dy = sum(o[1] for o in offsets) / len(offsets)
    mean_scale = sum(scales) / len(scales)
    return {
        "offset_mm": (round(mean_dx, 2), round(mean_dy, 2)),
        "font_scale": round(mean_scale, 3),
        "applied": applied,
        "per_page": [
            {"page": page.index, **(page.metadata.get("correction") or {})} for page in document.pages
        ],
    }
