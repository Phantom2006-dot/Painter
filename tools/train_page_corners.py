#!/usr/bin/env python3
"""Train the page-corner detector that replaces the boundary heuristic.

The heuristic this replaces segments on brightness and validates a quad. It
works on a clean desk and returned ``None`` on a certificate photographed on a
concrete floor, because "the page is the bright region" stops being true when the
surroundings are a lit surface. A model trained on 24,889 real frames of pages on
cluttered backgrounds plus synthetic captures on the user's own floor learns the
page's shape and edges rather than a brightness threshold.

The task is YOLO pose with four keypoints: the page corners in
top-left, top-right, bottom-right, bottom-left order. Those four points are
exactly what the rectification step needs, so the model output feeds the existing
pipeline directly — no second stage to convert a mask into a quad.

Usage:
    python3 tools/train_page_corners.py --data data/page_dataset/data.yaml \
        --epochs 20 --imgsz 384 --batch 8 --device cpu
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

#: The Ultralytics release this project was verified against. A missing pin here
#: is how a training run silently changes behaviour between machines.
VERIFIED_ULTRALYTICS = "8.4.175"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/page_dataset/data.yaml")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="starting weights")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="data/training")
    parser.add_argument("--name", default="page_corners")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--patience", type=int, default=8, help="early-stopping patience in epochs")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import ultralytics
    from ultralytics import YOLO

    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"dataset config not found: {data_path}")

    print(f"ultralytics {ultralytics.__version__} (verified against {VERIFIED_ULTRALYTICS})")
    print(f"data={data_path} model={args.model} imgsz={args.imgsz} batch={args.batch} device={args.device}")

    model = YOLO(args.model)
    started = time.perf_counter()
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        resume=args.resume,
        # Page corners survive horizontal flips but the corner *order* must follow
        # them; the dataset declares flip_idx so Ultralytics swaps the keypoints.
        fliplr=0.5,
        # A page is never upside down in these captures, so avoid teaching the
        # model to expect it.
        flipud=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        mosaic=0.3,
        exist_ok=True,
        verbose=True,
    )
    elapsed = time.perf_counter() - started

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    report = {
        "model": args.model,
        "data": str(data_path),
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "seconds": round(elapsed, 1),
        "save_dir": str(save_dir),
        "best_weights": str(save_dir / "weights" / "best.pt"),
    }
    try:
        metrics = model.val(data=str(data_path), imgsz=args.imgsz, device=args.device, verbose=False)
        report["val"] = {
            "box_map50": round(float(metrics.box.map50), 4),
            "box_map": round(float(metrics.box.map), 4),
            "pose_map50": round(float(metrics.pose.map50), 4),
            "pose_map": round(float(metrics.pose.map), 4),
        }
        print("validation:", report["val"])
    except Exception as exc:  # a missing metric must not lose the training report
        report["val_error"] = str(exc)

    report_path = Path(args.project) / f"{args.name}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print("report written to", report_path)


if __name__ == "__main__":
    main()