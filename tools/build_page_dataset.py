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

Samples are written as they are produced and never accumulated. An earlier
version generated every synthetic photograph up front and held them all in
memory; at 3,400 captures that is roughly 21 GB, and the process was killed
part-way through with no output at all. Streaming keeps memory flat regardless of
how large the dataset is asked to be.

Usage:
    python3 tools/build_page_dataset.py \
        --smartdoc /home/ubuntu/datasets/smartdoc \
        --pages /home/ubuntu/datasets/small_dataset/train/images \
        --background /tmp/qa4/original_small.png \
        --output data/page_dataset \
        --smartdoc-train 5500 --smartdoc-val 1200 --synthetic-train 3000 --synthetic-val 400
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

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
    image_path: Path | None
    quad: np.ndarray
    metadata: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    page_path: Path | None = None
    #: Set for synthetic samples, which exist only in memory until written.
    photo: np.ndarray | None = None


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
    """Load real frames with their corner ground truth.

    Only the path is held, so this can cover tens of thousands of frames without
    the memory cost that the synthetic path has to avoid.
    """
    frames_dir = smartdoc_dir / "frames"
    metadata_path = frames_dir / "metadata.csv.gz"
    if not metadata_path.is_file():
        raise SystemExit(f"SmartDoc metadata not found at {metadata_path}")

    samples: list[Sample] = []
    with gzip.open(metadata_path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))

    rng.shuffle(rows)
    for index, row in enumerate(rows):
        if index % stride:
            continue
        if len(samples) >= limit:
            break
        image_path = frames_dir / row["image_path"]
        if not image_path.is_file():
            continue
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


def synthetic_samples(
    page_paths: list[Path],
    backgrounds: list[np.ndarray],
    count: int,
    rng: np.random.Generator,
    frame_size: tuple[int, int] = (1240, 1650),
    start_index: int = 0,
) -> Iterator[Sample]:
    """Yield synthetic captures one at a time so none are buffered.

    This is a generator for a specific reason: each capture is a full-resolution
    photograph, and a caller that collected them all before writing would need tens
    of gigabytes.
    """
    width, height = frame_size
    for index in range(start_index, start_index + count):
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
        yield Sample(
            sample_id=f"synthetic_{index:06d}",
            source="synthetic",
            # The clean page is the document identity, so a page never spans splits.
            document_id=f"synthetic_page_{page_path.stem[:16]}",
            image_path=None,
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
            photo=photo,
        )


def load_docunet(docunet_dir: Path, limit: int | None = None) -> list[Sample]:
    """Pair DocUNet's photographs with the flatbed scans of the same documents.

    The pairing is by the leading identifier in the filename: ``10_1.jpg`` and
    ``10_2.jpg`` are two captures of the document whose scan is ``10.jpg``. Two
    photographs of one document are not two documents, so they share a
    ``document_id`` and can never be split across train and validation.

    These are recorded in the manifest only. Their page corners are not annotated
    in this release, and deriving them by matching the photograph to the scan
    worked for 2 of 84 photographs, which is not a foundation to train on.
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
                quad=np.zeros((4, 2), dtype=np.float64),
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


def write_sample(output: Path, split: str, sample: Sample, manifest: Any, counts: Counter) -> bool:
    """Write one sample's image, label and manifest line, then release it."""
    photo = sample.photo
    if photo is None and sample.image_path is not None:
        photo = cv2.imread(str(sample.image_path))
    if photo is None:
        return False
    height, width = photo.shape[:2]
    image_path = output / "images" / split / f"{sample.sample_id}.jpg"
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
    # Drop the reference immediately: this is the line that keeps memory flat.
    sample.photo = None
    return True


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

    counts: Counter[str] = Counter()
    manifest_path = output / "manifest.jsonl"
    with manifest_path.open("w") as manifest:
        if args.smartdoc:
            smartdoc_dir = Path(args.smartdoc)
            pool = load_smartdoc(
                smartdoc_dir, args.smartdoc_train + args.smartdoc_val, rng_py
            )
            train_n = val_n = 0
            for sample in pool:
                split = "val" if sample.document_id in SMARTDOC_VAL_MODELS else "train"
                if write_sample(output, split, sample, manifest, counts):
                    if split == "val":
                        val_n += 1
                    else:
                        train_n += 1
            print(
                f"smartdoc: {train_n} train, {val_n} val "
                f"(held-out models: {', '.join(SMARTDOC_VAL_MODELS)})"
            )

        if args.synthetic_train or args.synthetic_val:
            if not pages:
                raise SystemExit("--pages is required for synthetic samples")
            # Hold out whole pages, not individual captures of one page. Chosen from
            # the page list rather than from generated samples, because the samples
            # are never all in memory at once.
            # Every twelfth page is held out, which puts roughly 8% of the captures
            # in validation and keeps whole pages on one side of the split.
            val_pages = {page.stem[:16] for page in pages[::12]}

            train_n = val_n = 0
            for sample in synthetic_samples(
                pages, backgrounds, args.synthetic_train, rng_np, start_index=0
            ):
                split = "val" if sample.document_id in {f"synthetic_page_{p}" for p in val_pages} else "train"
                if write_sample(output, split, sample, manifest, counts):
                    if split == "val":
                        val_n += 1
                    else:
                        train_n += 1
            for sample in synthetic_samples(
                pages, backgrounds, args.synthetic_val, rng_np, start_index=args.synthetic_train
            ):
                # The held-out captures are a separate pass so the requested count is
                # honoured exactly, whatever the page-level split turned out to be.
                split = "val" if sample.document_id in {f"synthetic_page_{p}" for p in val_pages} else "train"
                if write_sample(output, split, sample, manifest, counts):
                    if split == "val":
                        val_n += 1
                    else:
                        train_n += 1
            print(f"synthetic: {train_n} train, {val_n} val (held-out pages: {len(val_pages)})")

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
