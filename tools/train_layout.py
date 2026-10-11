#!/usr/bin/env python3
"""Fine-tune the document-layout detector and report honest metrics.

The pipeline ships a pretrained DocLayNet model. This script trains a layout
detector on a prepared YOLO dataset (see ``tools/prepare_layout_dataset.py``),
evaluates it on the validation split and, optionally, evaluates the shipped
model on the same split so the two numbers can be compared directly.

Usage:
    python3 tools/train_layout.py \
        --data data/layout_dataset/data.yaml \
        --epochs 15 --imgsz 512 --batch 8 \
        --baseline yolo_demo/doclayout.pt \
        --project data/training

Everything runs on CPU by default because that is what the environment has; the
script says so in the report rather than pretending the numbers are comparable
to a GPU run.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


def evaluate(weights: str, data: str, imgsz: int, device: str, loader: str) -> dict:
    """Run validation and return the headline metrics."""
    if loader == "doclayout":
        from doclayout_yolo import YOLOv10

        model = YOLOv10(weights)
    else:
        from ultralytics import YOLO

        model = YOLO(weights)
    metrics = model.val(data=data, imgsz=imgsz, device=device, verbose=False, plots=False)
    results = getattr(metrics, "results_dict", {}) or {}
    return {
        "weights": str(weights),
        "loader": loader,
        "mAP50": float(results.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(results.get("metrics/mAP50-95(B)", 0.0)),
        "precision": float(results.get("metrics/precision(B)", 0.0)),
        "recall": float(results.get("metrics/recall(B)", 0.0)),
        "per_class": {
            str(key).replace("metrics/mAP50(B)-", ""): float(value)
            for key, value in results.items()
            if str(key).startswith("metrics/mAP50(B)-")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="dataset yaml")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base", default="yolov8n.pt", help="pretrained weights to start from")
    parser.add_argument("--project", default="data/training", help="where runs are written")
    parser.add_argument("--name", default=None)
    parser.add_argument("--baseline", default=None, help="weights to evaluate for comparison")
    parser.add_argument("--baseline-loader", default="doclayout", choices=["doclayout", "ultralytics"])
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--report", default="data/training/report.json")
    args = parser.parse_args()

    report: dict = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.platform(),
        "processor": platform.processor(),
        "device": args.device,
        "dataset": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "base_weights": args.base,
    }

    if args.baseline and not args.skip_baseline:
        print(f"evaluating baseline {args.baseline} on the validation split…")
        try:
            report["baseline"] = evaluate(
                args.baseline, args.data, args.imgsz, args.device, args.baseline_loader
            )
            print(f"  baseline mAP50 = {report['baseline']['mAP50']:.4f}")
        except Exception as exc:  # noqa: BLE001 - a failed baseline must not stop training
            report["baseline_error"] = str(exc)
            print(f"  baseline evaluation failed: {exc}")

    from ultralytics import YOLO

    print(f"fine-tuning {args.base} for {args.epochs} epochs at {args.imgsz}px on {args.device}…")
    started = time.time()
    model = YOLO(args.base)
    run = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name or f"layout-{int(started)}",
        exist_ok=True,
        pretrained=True,
        plots=False,
        val=True,
        seed=0,
        deterministic=True,
        workers=2,
    )
    report["train_seconds"] = round(time.time() - started, 1)
    best = Path(getattr(run, "save_dir", args.project)) / "weights" / "best.pt"
    report["best_weights"] = str(best)

    if best.is_file():
        print(f"evaluating the fine-tuned model {best}…")
        report["tuned"] = evaluate(str(best), args.data, args.imgsz, args.device, "ultralytics")
        print(f"  fine-tuned mAP50 = {report['tuned']['mAP50']:.4f}")
        if "baseline" in report:
            delta = report["tuned"]["mAP50"] - report["baseline"]["mAP50"]
            report["mAP50_delta"] = round(delta, 4)
            print(f"  change vs baseline: {delta:+.4f}")

    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())