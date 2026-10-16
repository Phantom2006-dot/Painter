#!/usr/bin/env python3
"""Build a labelled, paired document-photograph dataset.

Three sources, one label format:

``smartdoc``
    Real photographs of printed pages with hand-annotated page corners. 24,889
    frames across five backgrounds, which is exactly the supervision the
    heuristic boundary detector never had.

``synthetic``
    A clean digital page composited onto a real surface texture with a known
    transformation, so corners, layout boxes, text and degradation parameters are
    all exact. The surface textures include the concrete floor the user's own
    certificate was photographed on.

``docunet``
    Real photographs paired with flatbed scans of the same physical documents.

Everything lands in one directory as YOLO-pose samples (a box plus four page
corner keypoints), with a JSON manifest carrying the rest of the taxonomy:
content differences, capture conditions, layout boxes and text. Splits are made
by *document*, never by frame, because frames of one document are nearly
identical and a page-level split would report an accuracy the model does not have.

Usage:
    python3 tools/build_page_dataset.py \
        --smartdoc /home/ubuntu/datasets/smartdoc \
        --pages /home/ubuntu/datasets/small_dataset/train/images \
        --background /tmp/qa4/original_small.png \
        --output data/page_dataset \
        --smartdoc-train 3000 --smartdoc-val 500 --synthetic-train 2000 --synthetic-val 300
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from painter.synthesis import background_from_photo, simulate_capture  # noqa: E402

#: Single class: this model finds the page, not its contents.
CLASS_NAMES = ["document_page"]

#: SmartDoc's corner columns, in the order this project stores corners.
SMARTDOC_CORNER_COLUMNS = ("tl", "tr", "br", "bl")

#: Models held out entirely so validation shares no document with training.
SMARTDOC_VAL_MODELS = ("patent001", "patent002", "patent003", "patent004", "patent005")


@dataclass
class Sample:
    """One labelled photograph."""

    sample_id: str
    source: str
    document_id: str
    image_path: Path
    quad: np.ndarray
    metadata: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    page_path: Path | None = None


def _clip_quad(quad: np.ndarray, width: int, height: int) -> np.ndarray:
    quad = np.asarray(quad, dtype=np.float64)
    quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
    return quad


def _quad_is_usable(quad: np.ndarray, width: int, height: int, min_area_fraction: float = 0.03) -> bool:
    """Reject degenerate or tiny pages before they become noisy supervision."""
    area = float(cv2.contourArea(quad.astype(np.float32)))
    if area < min_area_fraction * width * height:
        return False
    # A self-intersecting quad cannot be a page outline.
    if area > 0.98 * width * height:
        return False
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        if np.linalg.norm(b - a) < 12:
            return False
    return True


def load_smartdoc(smartdoc_dir: Path, limit: int, rng: random.Random, stride: int = 1) -> list[Sample]:
    """Load real frames with their corner ground truth."""
    frames_dir = smartdoc_dir / "frames"
    metadata_path = frames_dir / "metadata.csv.gz"
    if not metadata_path.is_file():
        raise SystemExit(f"SmartDoc metadata not found at {metadata_path}")

    samples: list[Sample] = []
    with gzip.open(metadata_path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    rng.shuffle(rows)
    for index, row in enumerate(rows):
        if index % stride:
            continue
        if len(samples) >= limit:
            break
        image_path = frames_dir / row["image_path"]
        if not image_path.is_file():
            continue
        width = float(row["model_width"])
        height = float(row["model_height"])
        quad = np.array(
            [[float(row[f"{corner}_x"]), float(row[f"{corner}_y"])] for corner in SMARTDOC_CORNER_COLUMNS]
        )
        frame_width = 1920
        frame_height = 1080
        quad = _clip_quad(quad, frame_width, frame_height)
        if not _quad_is_usable(quad, frame_width, frame_height):
            continue
        samples.append(
            Sample(
                sample_id=f"smartdoc_{Path(row['image_path']).stem}_{row['bg_id']}",
                source="smartdoc",
                document_id=row["model_name"],
                image_path=image_path,
                quad=quad,
                metadata={
                    "background": row["bg_name"],
                    "frame_index": int(row["frame_index"]),
                    "model_type": row["modeltype_name"],
                    "frame_size": [frame_width, frame_height],
                    "corner_source": "smartdoc ground truth",
                },
                labels={"capture": [], "content": {"same_text": "unannotated"}},
                page_path=smartdoc_dir / "models" / "05-corrected-nexus-scaled33" / f"{row['model_name']}.png",
            )
        )
    return samples


def load_synthetic(
    page_paths: list[Path],
    backgrounds: list[np.ndarray],
    count: int,
    rng: np.random.Generator,
    frame_size: tuple[int, int] = (1240, 1650),
) -> list[Sample]:
    """Composite clean pages onto real surfaces with exact labels."""
    samples: list[Sample] = []
    width, height = frame_size
    for index in range(count):
        page_path = page_paths[int(rng.integers(0, len(page_paths)))]
        page = cv2.imread(str(page_path))
        if page is None:
            continue
        target_h = int(rng.integers(900, 1250))
        scale = target_h / page.shape[0]
        page = cv2.resize(page, (max(1, int(page.shape[1] * scale)), target_h), interpolation=cv2.INTER_AREA)

        background = backgrounds[int(rng.integers(0, len(backgrounds)))]
        if background.shape[1] != width or background.shape[0] != height:
            background = cv2.resize(background, (width, height), interpolation=cv2.INTER_AREA)

        capture = simulate_capture(page, background, rng)
        photo = capture.photo
        samples.append(
            Sample(
                sample_id=f"synthetic_{index:06d}",
                source="synthetic",
                # The clean page is the document identity, so a page never spans splits.
                document_id=f"synthetic_page_{page_path.stem[:16]}",
                image_path=None,  # written by the caller
                quad=capture.page_quad.astype(np.float64),
                metadata={
                    "frame_size": [photo.shape[1], photo.shape[0]],
                    "corner_source": "simulated transformation",
                    "page_size": list(capture.page_size),
                    "capture": capture.params.as_dict(),
                },
                labels={
                    "capture": capture.params.distortion_labels(),
                    "content": {"same_text": "identical to the paired clean page"},
                },
                page_path=page_path,
            )
        )
        samples[-1]._photo = photo  # type: ignore[attr-defined]
    return samples


def load_docunet(docunet_dir: Path, limit: int | None = None) -> list[Sample]:
    """Pair DocUNet's photographs with the flatbed scans of the same documents.

    The pairing is by the leading identifier in the filename: ``10_1.jpg`` and
    ``10_2.jpg`` are two captures of the document whose scan is ``10.jpg``. Two
    photographs of one document are not two documents, so they share a
    ``document_id`` and can never be split across train and validation.
    """
    photos_dir = docunet_dir / "original" / "original"
    scans_dir = docunet_dir / "scan" / "scan"
    if not photos_dir.is_dir() or not scans_dir.is_dir():
        return []
    # Scans ship as PNG while the photographs are JPEG, so a *.jpg glob silently
    # matched nothing and the pairing came back empty.
    scans = {path.stem: path for path in sorted(scans_dir.glob("*.png"))}
    scans.update({path.stem: path for path in sorted(scans_dir.glob("*.jpg"))})
    samples: list[Sample] = []
    for photo_path in sorted(photos_dir.glob("*.jpg")):
        document_id = photo_path.stem.split("_")[0]
        scan_path = scans.get(document_id)
        if scan_path is None:
            continue
        samples.append(
            Sample(
                sample_id=f"docunet_{photo_path.stem}",
                source="docunet",
                document_id=f"docunet_{document_id}",
                image_path=photo_path,
                quad=np.zeros((4, 2), dtype=np.float64),  # not annotated in this release
                metadata={"corner_source": "not annotated", "scan": str(scan_path)},
                labels={"capture": [], "content": {"same_text": "paired with the flatbed scan"}},
                page_path=scan_path,
            )
        )
        if limit and len(samples) >= limit:
            break
    return samples


def write_pose_label(sample: Sample, label_path: Path, width: int, height: int) -> None:
    """Write one YOLO-pose line: box plus four ordered corner keypoints."""
    quad = sample.quad
    x1, y1 = quad[:, 0].min(), quad[:, 1].min()
    x2, y2 = quad[:, 0].max(), quad[:, 1].max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    box_w, box_h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    parts = [
        "0",
        f"{cx / width:.6f}",
        f"{cy / height:.6f}",
        f"{box_w / width:.6f}",
        f"{box_h / height:.6f}",
    ]
    for x, y in quad:
        parts.extend([f"{x / width:.6f}", f"{y / height:.6f}", "2"])  # 2 = visible
    label_path.write_text(" ".join(parts) + "\n")


def build(args: argparse.Namespace) -> None:
    rng_py = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)
    output = Path(args.output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    pages = sorted(Path(args.pages).glob("*.png")) if args.pages else []
    backgrounds: list[np.ndarray] = []
    for path in args.background:
        photo = cv2.imread(str(path))
        if photo is None:
            continue
        # Recover the surface: the document that was photographed there is
        # inpainted away so it cannot leak into the training data.
        backgrounds.append(background_from_photo(photo))
    if not backgrounds:
        raise SystemExit("no usable background textures; pass --background")

    train: list[Sample] = []
    val: list[Sample] = []

    if args.smartdoc:
        smartdoc_dir = Path(args.smartdoc)
        pool = load_smartdoc(smartdoc_dir, args.smartdoc_train + args.smartdoc_val, rng_py)
        for sample in pool:
            (val if sample.document_id in SMARTDOC_VAL_MODELS else train).append(sample)
        print(f"smartdoc: {len(train)} train, {len(val)} val (held-out models: {', '.join(SMARTDOC_VAL_MODELS)})")

    if args.synthetic_train or args.synthetic_val:
        if not pages:
            raise SystemExit("--pages is required for synthetic samples")
        synthetic_val_docs = set()
        synthetic = load_synthetic(
            pages, backgrounds, args.synthetic_train + args.synthetic_val, rng_np
        )
        # Hold out whole pages, not individual captures of one page.
        for sample in sorted({s.document_id for s in synthetic})[: max(1, len(synthetic) // 12)]:
            synthetic_val_docs.add(sample)
        for sample in synthetic:
            (val if sample.document_id in synthetic_val_docs else train).append(sample)
        print(f"synthetic: {sum(1 for s in train if s.source == 'synthetic')} train, "
              f"{sum(1 for s in val if s.source == 'synthetic')} val")

    if args.docunet:
        # Paired photo/scan records are for the reconstruction side, not for
        # corner supervision, so they are recorded in the manifest only.
        docunet = load_docunet(Path(args.docunet))
        manifest_extra = output / "docunet_pairs.json"
        manifest_extra.write_text(
            json.dumps(
                [
                    {
                        "sample_id": s.sample_id,
                        "document_id": s.document_id,
                        "photo": str(s.image_path),
                        "scan": str(s.page_path),
                    }
                    for s in docunet
                ],
                indent=2,
            )
        )
        print(f"docunet: {len(docunet)} photo/scan pairs recorded in {manifest_extra.name}")

    manifest_path = output / "manifest.jsonl"
    counts: Counter[str] = Counter()
    with manifest_path.open("w") as manifest:
        for split, samples in (("train", train), ("val", val)):
            for index, sample in enumerate(samples):
                photo = getattr(sample, "_photo", None)
                if photo is None:
                    photo = cv2.imread(str(sample.image_path))
                if photo is None:
                    continue
                height, width = photo.shape[:2]
                image_name = f"{sample.sample_id}.jpg"
                image_path = output / "images" / split / image_name
                cv2.imwrite(str(image_path), photo, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                write_pose_label(sample, output / "labels" / split / f"{sample.sample_id}.txt", width, height)
                manifest.write(
                    json.dumps(
                        {
                            "sample_id": sample.sample_id,
                            "split": split,
                            "source": sample.source,
                            "document_id": sample.document_id,
                            "image": str(image_path.relative_to(output)),
                            "page_quad": [[round(float(x), 2), round(float(y), 2)] for x, y in sample.quad],
                            "page_reference": str(sample.page_path) if sample.page_path else None,
                            "metadata": sample.metadata,
                            "labels": sample.labels,
                        }
                    )
                    + "\n"
                )
                counts[f"{split}/{sample.source}"] += 1

    (output / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output.resolve()}",
                "train: images/train",
                "val: images/val",
                "kpt_shape: [4, 3]",
                "flip_idx: [1, 0, 3, 2]",
                "names:",
                "  0: document_page",
                "",
            ]
        )
    )
    print("counts:", dict(counts))
    print("dataset written to", output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smartdoc", help="SmartDoc dataset root")
    parser.add_argument("--docunet", help="DocUNet dataset root")
    parser.add_argument("--pages", help="directory of clean page images for synthetic capture")
    parser.add_argument("--background", action="append", default=[], help="surface texture (repeatable)")
    parser.add_argument("--output", default="data/page_dataset")
    parser.add_argument("--smartdoc-train", type=int, default=3000)
    parser.add_argument("--smartdoc-val", type=int, default=500)
    parser.add_argument("--synthetic-train", type=int, default=2000)
    parser.add_argument("--synthetic-val", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261016)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
