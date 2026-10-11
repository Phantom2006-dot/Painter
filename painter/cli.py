"""Command line interface: ``python -m painter.cli <command>``.

Useful for headless runs, CI and quick inspection without the web UI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import get_settings
from .errors import PainterError
from .logging_utils import configure_logging
from .pipeline import DocumentPipeline, PipelineOptions
from .storage import Repository, cleanup_workspaces


def _print(payload: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
    elif isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, indent=2, default=str))


def cmd_capabilities(args: argparse.Namespace) -> int:
    from .detection import LayoutDetectorService
    from .ocr import TesseractProvider
    from .serpapi_client import describe_configuration

    settings = get_settings()
    report = {
        "data_dir": str(settings.data_dir),
        "model_dir": str(settings.model_dir),
        "yolo_enabled": settings.enable_yolo,
        "models": {
            "layout": str(settings.yolo_layout_model) if settings.yolo_layout_model else None,
            "layout_present": bool(settings.yolo_layout_model and Path(settings.yolo_layout_model).is_file()),
            "coco": str(settings.yolo_coco_model) if settings.yolo_coco_model else None,
            "custom": str(settings.yolo_custom_model) if settings.yolo_custom_model else None,
        },
        "tesseract_available": TesseractProvider(settings).available(),
        "ocr_provider_mode": settings.ocr_provider,
        "serpapi": describe_configuration(settings.serpapi),
        "verification": {"enabled": settings.verify_enabled, "target": settings.verify_target_similarity},
        "asset_download_enabled": settings.enable_assets_download,
        "llm_enabled": settings.enable_llm,
    }
    if settings.yolo_layout_model and Path(settings.yolo_layout_model).is_file():
        detector = LayoutDetectorService(settings).detector("doclayout", settings.yolo_layout_model)
        load_report = detector.load()
        report["layout_model_load"] = load_report.to_dict()
    _print(report, args.json)
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    settings = get_settings()
    pipeline = DocumentPipeline(settings)
    options = PipelineOptions(
        force_stages=set(filter(None, (args.force or "").split(","))),
        enable_verify=not args.no_verify,
        run_correction_loop=not args.no_correction,
        enable_lens=args.lens,
        enable_assets_download=args.assets,
        ocr_provider=args.ocr,
        max_pages=args.max_pages,
    )
    source = Path(args.file).expanduser()
    if not source.is_file():
        print(f"file not found: {source}", file=sys.stderr)
        return 2

    def progress(stage: str, fraction: float) -> None:
        if not args.json:
            print(f"  [{fraction * 100:5.1f}%] {stage}", file=sys.stderr, flush=True)

    document_id = pipeline.create_document(source)
    result = pipeline.run(document_id, options=options, progress=progress)
    payload = result.to_dict(include_document=args.json)
    if args.out:
        out_dir = Path(args.out).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        workspace = pipeline.workspace(document_id)
        import shutil

        for kind, descriptor in (result.exports or {}).items():
            path = workspace.absolute(descriptor["path"])
            if path.is_file():
                shutil.copy2(path, out_dir / descriptor["filename"])
        payload["out_dir"] = str(out_dir)
    if not args.json:
        print(f"document: {document_id}")
        print(f"status:   {result.status} in {result.elapsed_s}s")
        for stage, info in result.stages.items():
            print(f"  {stage:12s} {info.get('status')}")
        if result.document is not None:
            stats = result.document.stats
            print(f"  elements: {stats.get('elements')} (review: {stats.get('elements_needing_review')})")
        for warning in result.warnings[:10]:
            print(f"  warning: {warning}")
        for error in result.errors:
            print(f"  error:   {error.get('message')}")
    else:
        _print(payload, True)
    return 0 if result.status == "completed" else 1


def cmd_status(args: argparse.Namespace) -> int:
    settings = get_settings()
    repository = Repository(settings.db_path)
    payload: dict[str, Any] = {"document": repository.get_document(args.document_id)}
    jobs = repository.get_jobs_for_document(args.document_id, limit=1)
    payload["job"] = repository.get_job(jobs[0]["job_id"]) if jobs else None
    _print(payload, args.json)
    return 0


def cmd_elements(args: argparse.Namespace) -> int:
    settings = get_settings()
    repository = Repository(settings.db_path)
    payload = repository.latest_ir(args.document_id)
    if payload is None:
        print("document has not been processed yet", file=sys.stderr)
        return 1
    from .schemas import DocumentIR

    document = DocumentIR.model_validate(payload)
    rows = []
    for page in document.pages:
        if args.page is not None and page.index != args.page:
            continue
        for element in page.elements:
            if args.review and not element.needs_review:
                continue
            rows.append(
                {
                    "id": element.id,
                    "page": page.index,
                    "type": element.type.value,
                    "confidence": round(element.confidence, 3),
                    "bbox": [round(v, 1) for v in element.bbox.as_list()],
                    "flags": [flag.value for flag in element.flags],
                    "text": (element.text or "")[:160],
                }
            )
    if args.json:
        _print({"elements": rows, "stats": document.stats}, True)
    else:
        for row in rows:
            print(
                f"{row['id']:16s} p{row['page']} {row['type']:14s} conf={row['confidence']:.2f} "
                f"flags={','.join(row['flags']) or '-':20s} {row['text'][:60]!r}"
            )
        print(f"\n{len(rows)} element(s); stats: {document.stats}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    settings = get_settings()
    repository = Repository(settings.db_path)
    pipeline = DocumentPipeline(settings, repository)
    from .editing import export_artifacts, re_render
    from .schemas import DocumentIR

    payload = repository.latest_ir(args.document_id)
    if payload is None:
        print("document has not been processed yet", file=sys.stderr)
        return 1
    document = DocumentIR.model_validate(payload)
    workspace = pipeline.workspace(args.document_id)
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    re_render(document, settings, workspace, formats=formats)
    exports = export_artifacts(document, workspace, formats)
    for kind, descriptor in exports.items():
        repository.record_export(args.document_id, kind, descriptor["path"], descriptor["size_bytes"])
        if not args.json:
            print(f"{kind}: {workspace.absolute(descriptor['path'])} ({descriptor['size_bytes']} bytes)")
    if args.json:
        _print(exports, True)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    import cv2

    settings = get_settings()
    repository = Repository(settings.db_path)
    pipeline = DocumentPipeline(settings, repository)
    from .render import HtmlRenderOptions, render_document
    from .schemas import DocumentIR
    from .verify import correction_loop, verify_page

    payload = repository.latest_ir(args.document_id)
    if payload is None:
        print("document has not been processed yet", file=sys.stderr)
        return 1
    document = DocumentIR.model_validate(payload)
    workspace = pipeline.workspace(args.document_id)

    def write_verification_html(params: dict[str, Any], page_index: int) -> Path:
        options = HtmlRenderOptions(
            asset_url_prefix=f"file://{workspace.root}",
            global_offset_mm=(float(params.get("dx_mm", 0.0)), float(params.get("dy_mm", 0.0))),
            font_scale=float(params.get("font_scale", 1.0)),
        )
        markup = render_document_for_print(document, options)
        path = workspace.dir("verify") / (
            f"cli_iter_{page_index}_{abs(hash(json.dumps(params, sort_keys=True))) % 10**8}.html"
        )
        path.write_text(markup, encoding="utf-8")
        return path

    html_path = write_verification_html({}, document.pages[0].index if document.pages else 0)

    results: dict[str, Any] = {}
    for page in document.pages:
        if args.page is not None and page.index != args.page:
            continue
        original = cv2.imread(str(workspace.absolute(page.image_path)), cv2.IMREAD_COLOR)
        if original is None:
            continue
        work_dir = workspace.dir("verify") / f"page_{page.index:02d}"
        work_dir.mkdir(parents=True, exist_ok=True)

        def render_with(params: dict[str, Any]) -> Path:
            return write_verification_html(params, page.index)

        result = (
            verify_page(page, document, original, html_path, work_dir, settings)
            if args.no_correction
            else correction_loop(document, page, original, html_path, work_dir, settings, render_with)
        )
        results[str(page.index)] = result.to_dict()
        repository.save_quality(args.document_id, page.index, result.metrics.to_dict())
    document.revision += 1
    repository.save_ir(args.document_id, document.revision, document.model_dump_json())
    _print(results if args.json else {k: v["metrics"] for k, v in results.items()}, True)
    return 0


CLASS_ORDER = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]

#: IR element types mapped onto the layout classes the detector is trained on.
TYPE_TO_CLASS = {
    "figure_caption": "Caption",
    "table_caption": "Caption",
    "footnote": "Footnote",
    "formula": "Formula",
    "list": "List-item",
    "footer": "Page-footer",
    "header": "Page-header",
    "image": "Picture",
    "logo": "Picture",
    "subheading": "Section-header",
    "heading": "Section-header",
    "table": "Table",
    "paragraph": "Text",
    "text_block": "Text",
    "title": "Title",
}


def cmd_training_data(args: argparse.Namespace) -> int:
    """Export reconstructed documents as layout-detection training data.

    The point is the feedback loop: correct an element in the web UI, and the
    corrected document becomes a labelled example for the next training run.
    ``--only-corrected`` exports just the documents a human touched.
    """
    import csv
    import shutil

    from .config import load_settings
    from .schemas import DocumentIR
    from .storage import Repository

    settings = load_settings()
    repository = Repository(settings.db_path)
    output = Path(args.out)
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, object]] = []
    for record in repository.list_documents(limit=args.limit):
        document_id = record["document_id"]
        payload = repository.latest_ir(document_id)
        if payload is None:
            continue
        try:
            document = DocumentIR.model_validate(payload)
        except Exception:  # noqa: BLE001 - an unreadable IR is simply skipped
            continue
        # a hand correction is recorded per element, so the document counts as
        # corrected when any element carries an edit record
        corrected = any(
            element.edits for page in document.pages for element in page.elements
        )
        if args.only_corrected and not corrected:
            continue
        for page in document.pages:
            for element in page.elements:
                class_name = TYPE_TO_CLASS.get(element.type.value)
                if class_name is None:
                    continue
                stem = f"{document_id}_p{page.index:02d}_{class_name.lower()}"
                image_source = settings.data_dir / "documents" / document_id / "pages" / f"page_{page.index:02d}.png"
                if not image_source.is_file():
                    continue
                image_target = output / "images" / f"{document_id}_page{page.index:02d}.png"
                if not image_target.is_file():
                    shutil.copyfile(image_source, image_target)
                label_path = output / "labels" / f"{document_id}_page{page.index:02d}.txt"
                box = element.bbox
                width = float(page.width_px) or 1.0
                height = float(page.height_px) or 1.0
                centre_x = (box.x1 + box.x2) / 2 / width
                centre_y = (box.y1 + box.y2) / 2 / height
                line = (
                    f"{CLASS_ORDER.index(class_name)} {centre_x:.6f} {centre_y:.6f} "
                    f"{box.width / width:.6f} {box.height / height:.6f}"
                )
                with label_path.open("a") as handle:
                    handle.write(line + "\n")
                exported.append(
                    {
                        "document_id": document_id,
                        "page": page.index,
                        "element_id": element.id,
                        "class": class_name,
                        "corrected": corrected,
                        "stem": stem,
                    }
                )

    manifest = output / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["document_id", "page", "element_id", "class", "corrected", "stem"])
        writer.writeheader()
        writer.writerows(exported)

    corrected_count = sum(1 for row in exported if row["corrected"])
    print(f"exported {len(exported)} regions from {len({row['document_id'] for row in exported})} document(s)")
    print(f"  {corrected_count} from hand-corrected documents")
    print(f"  images -> {output / 'images'}")
    print(f"  labels -> {output / 'labels'}")
    print(f"  manifest -> {manifest}")
    print("Train with: python3 tools/train_layout.py --data <dataset>/data.yaml")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("painter.api:app", host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    settings = get_settings()
    repository = Repository(settings.db_path)
    removed = cleanup_workspaces(settings.documents_dir, repository, args.keep_days, dry_run=not args.apply)
    _print({"removed": removed, "applied": args.apply}, args.json)
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Generate a synthetic document and run the full pipeline on it."""
    from .testsupport import build_synthetic_document, run_selftest

    settings = get_settings()
    if args.data_dir:
        settings = get_settings(refresh=True)
    report = run_selftest(settings, verbose=not args.json)
    _print(report, args.json)
    return 0 if report.get("status") == "completed" else 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .testsupport import run_benchmark

    settings = get_settings()
    report = run_benchmark(settings, cases=args.cases, verbose=not args.json)
    _print(report, args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="painter", description="Painter document reconstruction engine")
    parser.add_argument("--json", action="store_true", help="emit machine readable JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("capabilities", help="report models, OCR and API availability")
    p.set_defaults(func=cmd_capabilities)

    p = sub.add_parser("process", help="run the full pipeline on a file")
    p.add_argument("file")
    p.add_argument("--out", help="copy exports into this directory")
    p.add_argument("--ocr", choices=["auto", "tesseract", "lens", "none"], default=None)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--no-correction", action="store_true")
    p.add_argument("--lens", action="store_true", help="allow SerpApi Google Lens calls")
    p.add_argument("--assets", action="store_true", help="download matched candidate images")
    p.add_argument("--force", help="comma separated stages to re-run")
    p.add_argument("--max-pages", type=int, default=None)
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("status", help="show document and job status")
    p.add_argument("document_id")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("elements", help="list reconstructed elements")
    p.add_argument("document_id")
    p.add_argument("--page", type=int, default=None)
    p.add_argument("--review", action="store_true", help="only elements flagged for review")
    p.set_defaults(func=cmd_elements)

    p = sub.add_parser("export", help="render and export the document")
    p.add_argument("document_id")
    p.add_argument("--formats", default="html,pdf,docx")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("verify", help="compare the reconstruction with the original")
    p.add_argument("document_id")
    p.add_argument("--page", type=int, default=None)
    p.add_argument("--no-correction", action="store_true")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("serve", help="run the web application")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log-level", default="info")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("cleanup", help="remove old document workspaces")
    p.add_argument("--keep-days", type=int, default=30)
    p.add_argument("--apply", action="store_true", help="actually delete (default is dry run)")
    p.set_defaults(func=cmd_cleanup)

    p = sub.add_parser("selftest", help="generate a synthetic page and run the pipeline")
    p.add_argument("--data-dir", action="store_true", help="reserved for compatibility")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("training-data", help="export corrected documents as layout training data")
    p.add_argument("--out", default="data/training_export", help="destination directory")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--only-corrected", action="store_true", help="export only hand-corrected documents")
    p.set_defaults(func=cmd_training_data)

    p = sub.add_parser("benchmark", help="run the synthetic benchmark set")
    p.add_argument("--cases", type=int, default=6)
    p.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(get_settings().log_level, json_output=False)
    try:
        return args.func(args)
    except PainterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
    from .render import render_document_for_print
