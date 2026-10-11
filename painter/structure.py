"""Document structure: detections -> Document IR elements.

This module turns raw detections and OCR results into a typed, hierarchical
representation. Two principles from the brief drive the design:

* deterministic geometric rules decide everything that is measurable
  (grouping lines into paragraphs, reading order, header/footer zones);
* nothing is invented — an element with no recognised text is flagged, not
  filled with plausible-looking content.

The DocLayout-YOLO checkpoint emits **line level** ``Text`` boxes, so grouping
those lines into paragraphs is a core job of this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .config import Settings
from .detection import Detection
from .geometry import detect_columns, infer_alignment, map_bbox_through_transforms
from .logging_utils import get_logger
from .ocr import OcrResult
from .schemas import (
    Alignment,
    BBox,
    Element,
    ElementType,
    ListInfo,
    NormBBox,
    PageFormat,
    PageIR,
    PageTransform,
    Provenance,
    ReviewFlag,
    TextLine,
)

logger = get_logger("structure")

#: Vertical gap (as a fraction of line height) below which two text lines belong
#: to the same paragraph.
PARAGRAPH_GAP_RATIO = 0.65
#: Minimum horizontal overlap (fraction of the narrower line) to stay grouped.
PARAGRAPH_OVERLAP_RATIO = 0.45

BULLET_MARKERS = ("-", "•", "*", "·", "–", "—", "o ")
HEADER_ZONE_RATIO = 0.12
FOOTER_ZONE_RATIO = 0.88

#: Detection classes that carry text and therefore need OCR. Shared with the
#: pipeline so that the ``det_<n>`` keys stay identical between stages.
TEXT_DETECTION_TYPES: frozenset[ElementType] = frozenset(
    {
        ElementType.PARAGRAPH,
        ElementType.TEXT_BLOCK,
        ElementType.HEADING,
        ElementType.SUBHEADING,
        ElementType.LIST,
        ElementType.FIGURE_CAPTION,
        ElementType.TABLE_CAPTION,
        ElementType.FORMULA,
        ElementType.PAGE_NUMBER,
        # headers and footers carry real text (document titles, page numbers) and
        # were previously detected but never read
        ElementType.HEADER,
        ElementType.FOOTER,
        ElementType.FORM_FIELD,
    }
)


@dataclass
class StructureBuildResult:
    page: PageIR
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def _horizontal_overlap(first: BBox, second: BBox) -> float:
    overlap = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    narrower = min(first.width, second.width)
    return overlap / narrower if narrower > 0 else 0.0


def group_lines_into_paragraphs(
    detections: Sequence[Detection], gap_ratio: float = PARAGRAPH_GAP_RATIO
) -> list[tuple[list[Detection], BBox]]:
    """Group line-level text detections into paragraph blocks.

    Lines join when they are vertically close (relative to their own height) and
    horizontally overlapping, which keeps two side-by-side columns apart.
    """
    lines = [d for d in detections if d.element_type in {ElementType.PARAGRAPH, ElementType.TEXT_BLOCK}]
    if not lines:
        return []
    lines = sorted(lines, key=lambda d: (d.bbox.y1, d.bbox.x1))
    groups: list[list[Detection]] = [[lines[0]]]
    for detection in lines[1:]:
        placed = False
        for group in groups:
            last = group[-1]
            gap = detection.bbox.y1 - last.bbox.y2
            reference = max(min(last.bbox.height, detection.bbox.height), 1.0)
            if gap < -0.35 * reference:  # vertical overlap, same visual line
                continue
            if gap <= reference * gap_ratio and _horizontal_overlap(last.bbox, detection.bbox) >= PARAGRAPH_OVERLAP_RATIO:
                group.append(detection)
                placed = True
                break
        if not placed:
            groups.append([detection])

    blocks: list[tuple[list[Detection], BBox]] = []
    for group in groups:
        group.sort(key=lambda d: d.bbox.y1)
        box = BBox(
            x1=min(d.bbox.x1 for d in group),
            y1=min(d.bbox.y1 for d in group),
            x2=max(d.bbox.x2 for d in group),
            y2=max(d.bbox.y2 for d in group),
        )
        blocks.append((group, box))
    blocks.sort(key=lambda item: (item[1].y1, item[1].x1))
    return blocks


def detect_list_info(lines: Sequence[TextLine], text: str) -> ListInfo | None:
    """Detect list markers and split items."""
    items: list[str] = []
    ordered = False
    marker: str | None = None
    for line in lines:
        stripped = line.text.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith(BULLET_MARKERS):
            marker = marker or stripped[0]
            items.append(stripped[1:].strip())
        else:
            head = stripped.split(" ", 1)[0]
            head_core = head.rstrip(".)")
            if head_core.isdigit():
                ordered = True
                marker = marker or "."
                items.append(stripped[len(head) :].strip())
    if not items:
        return None
    return ListInfo(ordered=ordered, marker=marker, items=items, item_count=len(items))


def assign_reading_order(elements: Sequence[Element], page_width: int) -> list[str]:
    """Reading order: column band first, then top-to-bottom within the band."""
    columns = detect_columns(list(elements), page_width)
    if len(columns) <= 1:
        ordered = sorted(elements, key=lambda e: (round(e.bbox.y1, 1), round(e.bbox.x1, 1)))
        return [e.id for e in ordered]

    def band_index(element: Element) -> int:
        centre = element.bbox.center[0]
        for index, (start, end) in enumerate(columns):
            if start <= centre <= end:
                return index
        return len(columns) - 1

    ordered = sorted(
        elements,
        key=lambda e: (
            band_index(e),
            0 if e.type in {ElementType.HEADER, ElementType.HEADING, ElementType.SUBHEADING} else 1,
            round(e.bbox.y1, 1),
            round(e.bbox.x1, 1),
        ),
    )
    return [e.id for e in ordered]


def build_sections(elements: Sequence[Element]) -> list[dict[str, Any]]:
    """Group elements under their nearest preceding heading.

    Hierarchy is recorded in ``metadata`` rather than as synthetic element types
    so that renderers and editors keep working on real content elements.
    """
    headings = [
        e
        for e in elements
        if e.type in {ElementType.HEADING, ElementType.SUBHEADING}
    ]
    headings.sort(key=lambda e: e.bbox.y1)
    if not headings:
        return []
    sizes = [
        (h.typography.font_size_pt if h.typography and h.typography.font_size_pt else 0.0) for h in headings
    ]
    distinct_sizes = sorted({round(s, 1) for s in sizes if s}, reverse=True)

    sections: list[dict[str, Any]] = []
    for index, heading in enumerate(headings):
        next_heading_y = headings[index + 1].bbox.y1 if index + 1 < len(headings) else float("inf")
        members = [
            e.id
            for e in elements
            if e.id != heading.id
            and heading.bbox.y1 <= e.bbox.y1 < next_heading_y
            and e.type
            not in {ElementType.HEADER, ElementType.FOOTER, ElementType.PAGE_NUMBER}
        ]
        size = heading.typography.font_size_pt if heading.typography else None
        level = 1
        if size and distinct_sizes:
            level = distinct_sizes.index(round(size, 1)) + 1 if round(size, 1) in distinct_sizes else 1
        section = {
            "id": f"sec_{index + 1}",
            "heading_element_id": heading.id,
            "level": min(level, 3),
            "element_ids": members,
            "title": (heading.text or "").strip() or None,
        }
        sections.append(section)
        heading.metadata["section_id"] = section["id"]
        heading.metadata["section_level"] = section["level"]
        for member_id in members:
            element = next((e for e in elements if e.id == member_id), None)
            if element is not None:
                element.metadata["section_id"] = section["id"]
                element.metadata["section_level"] = section["level"]
    return sections


def build_page_ir(
    *,
    index: int,
    image: np.ndarray,
    detections: Sequence[Detection],
    ocr_results: dict[str, OcrResult],
    page_format: PageFormat,
    transforms: Sequence[PageTransform],
    settings: Settings,
    detector_name: str,
    table_builder: Any | None = None,
    image_path: str | None = None,
    original_path: str | None = None,
    dpi: float | None = None,
    extra_flags: Iterable[ReviewFlag] = (),
) -> StructureBuildResult:
    """Build one :class:`PageIR` from detections and OCR results.

    ``ocr_results`` is keyed by a synthetic detection key (``det_<n>``) produced
    by the pipeline before OCR runs.
    """
    warnings: list[str] = []
    height, width = image.shape[:2]
    elements: list[Element] = []
    used_flags = set(extra_flags)

    def make_element(
        element_type: ElementType,
        box: BBox,
        confidence: float,
        detection: Detection | None,
        key: str | None = None,
    ) -> Element:
        element = Element(
            type=element_type,
            bbox=box,
            norm_bbox=NormBBox.from_bbox(box, width, height),
            original_bbox=map_bbox_through_transforms(box, transforms) if transforms else None,
            confidence=confidence,
            detection_class=detection.class_name if detection else None,
            provenance=[
                Provenance(
                    stage="detection",
                    method=detection.source if detection else detector_name,
                    provider=detection.source.split(":")[0] if detection else None,
                    confidence=confidence,
                )
            ],
        )
        if detection is not None and detection.extra:
            element.metadata.update(detection.extra)
        if key and key in ocr_results:
            result = ocr_results[key]
            element.text = result.text or None
            element.lines = list(result.lines)
            element.flags.extend(result.flags)
            element.metadata["ocr"] = {
                "provider": result.provider,
                "method": result.method,
                "confidence": result.confidence,
                "warnings": result.warnings,
            }
            element.provenance.append(
                Provenance(
                    stage="ocr",
                    method=result.method,
                    provider=result.provider,
                    confidence=result.confidence,
                )
            )
            if result.confidence is not None:
                element.confidence = round(min(0.99, (confidence + result.confidence) / 2), 4)
        return element

    # -- 1. containers first (tables, pictures) so text can be nested ---------
    table_detections = [d for d in detections if d.element_type is ElementType.TABLE]
    picture_detections = [
        d for d in detections if d.element_type in {ElementType.IMAGE, ElementType.LOGO}
    ]
    header_footer = [
        d for d in detections if d.element_type in {ElementType.HEADER, ElementType.FOOTER}
    ]
    text_detections = [
        d
        for d in detections
        if d.element_type in TEXT_DETECTION_TYPES
    ]

    # The recognise stage OCRs every text type in detection order, so the det_N
    # keys must be built from that same list.
    keyed: dict[int, str] = {}
    for order, detection in enumerate(text_detections):
        keyed[id(detection)] = f"det_{order}"
    # Headers, footers and form fields keep their own type and are built in step
    # 4; they must not also be grouped into paragraphs here.
    groupable = [
        d
        for d in text_detections
        if d.element_type not in {ElementType.HEADER, ElementType.FOOTER, ElementType.FORM_FIELD}
    ]

    # -- 2. tables ------------------------------------------------------------
    for detection in table_detections:
        element = make_element(ElementType.TABLE, detection.bbox, detection.confidence, detection)
        element.add_flag(ReviewFlag.TABLE_UNCERTAIN)
        if table_builder is not None:
            try:
                content, lines = table_builder(image, detection.bbox)
                element.table = content
                element.lines = lines
                element.text = "\n".join(
                    " | ".join(cell.text for cell in content.cells if cell.row == row)
                    for row in range(content.rows)
                )
                element.confidence = round(
                    min(0.98, (detection.confidence + (content.structure_confidence or 0.5)) / 2), 4
                )
                element.flags = [f for f in element.flags if f is not ReviewFlag.TABLE_UNCERTAIN]
                element.flags.extend(
                    [ReviewFlag.TABLE_UNCERTAIN]
                    if content.detection_method == "whitespace"
                    else []
                )
            except Exception as exc:  # noqa: BLE001 - table failure must not kill the page
                warnings.append(f"table reconstruction failed: {exc}")
                element.metadata["table_error"] = str(exc)
        elements.append(element)

    # -- 3. pictures / logos --------------------------------------------------
    for detection in picture_detections:
        element = make_element(detection.element_type, detection.bbox, detection.confidence, detection)
        if detection.element_type is ElementType.LOGO:
            element.add_flag(ReviewFlag.GEOMETRY_ESTIMATED)
        elements.append(element)

    # -- 4. headers / footers -------------------------------------------------
    for detection in header_footer:
        element = make_element(
            detection.element_type,
            detection.bbox,
            detection.confidence,
            detection,
            keyed.get(id(detection)),
        )
        elements.append(element)

    # -- 5. text: group lines into paragraphs, then OCR each block -----------
    blocks = group_lines_into_paragraphs(groupable)
    heading_detections = [
        d for d in groupable if d.element_type in {ElementType.HEADING, ElementType.SUBHEADING}
    ]
    heading_ids = {id(d) for d in heading_detections}
    handled: set[int] = set()

    for detection in heading_detections:
        element = make_element(
            detection.element_type, detection.bbox, detection.confidence, detection, keyed.get(id(detection))
        )
        if element.text is None and ReviewFlag.OCR_FAILED not in element.flags:
            element.add_flag(ReviewFlag.TEXT_MISSING)
        if element.alignment is None:
            element.alignment = Alignment.LEFT
        elements.append(element)
        handled.add(id(detection))

    for group, box in blocks:
        remaining = [d for d in group if id(d) not in handled]
        if not remaining:
            continue
        element_type = ElementType.PARAGRAPH
        if any(d.element_type is ElementType.LIST for d in remaining):
            element_type = ElementType.LIST
        elif any(d.element_type is ElementType.TEXT_BLOCK for d in remaining):
            element_type = ElementType.TEXT_BLOCK
        merged = Detection(
            class_name=remaining[0].class_name,
            element_type=element_type,
            confidence=sum(d.confidence for d in remaining) / len(remaining),
            bbox=box,
            source=remaining[0].source,
        )
        # merge the OCR text of every line in the group
        texts: list[str] = []
        lines: list[TextLine] = []
        flags: list[ReviewFlag] = []
        confidences: list[float] = []
        providers: list[str] = []
        for detection in remaining:
            result = ocr_results.get(keyed.get(id(detection), ""))
            if result is None:
                continue
            if result.text:
                texts.append(result.text)
            lines.extend(result.lines)
            flags.extend(result.flags)
            providers.append(result.provider)
            if result.confidence is not None:
                confidences.append(result.confidence)
        element = make_element(element_type, box, merged.confidence, merged)
        element.text = "\n".join(texts).strip() or None
        element.lines = sorted(lines, key=lambda line: (line.bbox.y1, line.bbox.x1))
        element.flags = list(dict.fromkeys(flags))
        element.provenance.append(
            Provenance(
                stage="ocr",
                method="line-group-merge",
                provider="+".join(sorted(set(providers))) or None,
                confidence=round(sum(confidences) / len(confidences), 4) if confidences else None,
            )
        )
        if confidences:
            element.confidence = round(min(0.99, (element.confidence + sum(confidences) / len(confidences)) / 2), 4)
        element.metadata["line_count"] = len(remaining)
        if not element.text:
            element.add_flag(ReviewFlag.TEXT_MISSING if ReviewFlag.OCR_FAILED not in element.flags else ReviewFlag.OCR_FAILED)
        element.alignment = infer_alignment([line.bbox for line in element.lines], box) if element.lines else Alignment.LEFT
        if element_type is ElementType.LIST:
            info = detect_list_info(element.lines, element.text or "")
            if info:
                element.list_info = info
        elements.append(element)
        handled.update(id(d) for d in remaining)

    # captions and formulas that were not part of a paragraph group
    # (headers, footers and form fields were already built in step 4)
    for detection in groupable:
        if id(detection) in handled:
            continue
        element = make_element(
            detection.element_type, detection.bbox, detection.confidence, detection, keyed.get(id(detection))
        )
        if element.text is None and ReviewFlag.OCR_FAILED not in element.flags:
            element.add_flag(ReviewFlag.TEXT_MISSING)
        elements.append(element)

    # -- 6. page number heuristic on header/footer text ----------------------
    from .heuristics import detect_page_numbers

    page_number_candidates = detect_page_numbers(
        image, [(e.id, e.bbox, e.text or "") for e in elements if e.text]
    )
    for element_id, number in page_number_candidates:
        element = next((e for e in elements if e.id == element_id), None)
        if element is None:
            continue
        element.type = ElementType.PAGE_NUMBER
        element.metadata["page_number_value"] = number

    # -- 7. reading order, sections, quality bookkeeping --------------------
    elements.sort(key=lambda e: (round(e.bbox.y1, 1), round(e.bbox.x1, 1)))
    reading_order = assign_reading_order(elements, width)
    order_index = {element_id: position for position, element_id in enumerate(reading_order)}
    for element in elements:
        element.order_index = order_index.get(element.id)

    # header/footer containment (a logo inside the header band belongs to it)
    for element in elements:
        if element.type in {ElementType.IMAGE, ElementType.LOGO}:
            for container in elements:
                if container.type in {ElementType.HEADER, ElementType.FOOTER} and container.bbox.overlap_fraction(
                    element.bbox
                ) > 0.7:
                    element.parent_id = container.id
                    container.children.append(element.id)
                    break

    sections = build_sections(elements)

    if used_flags or (detector_name == "cv-fallback"):
        for element in elements:
            if detector_name == "cv-fallback" and element.type in {
                ElementType.PARAGRAPH,
                ElementType.HEADING,
                ElementType.TEXT_BLOCK,
            }:
                element.add_flag(ReviewFlag.DETECTION_FALLBACK)

    page = PageIR(
        index=index,
        image_path=image_path,
        original_path=original_path,
        width_px=width,
        height_px=height,
        dpi=dpi or settings.target_dpi,
        page_format=page_format,
        orientation="landscape" if width > height else "portrait",
        elements=elements,
        reading_order=reading_order,
        transforms=list(transforms),
        warnings=list(warnings),
        metadata={"sections": sections, "detector": detector_name},
    )
    return StructureBuildResult(
        page=page,
        warnings=warnings,
        stats={
            "elements": len(elements),
            "paragraphs": sum(1 for e in elements if e.type is ElementType.PARAGRAPH),
            "tables": sum(1 for e in elements if e.type is ElementType.TABLE),
            "images": sum(1 for e in elements if e.type is ElementType.IMAGE),
            "sections": len(sections),
        },
    )
