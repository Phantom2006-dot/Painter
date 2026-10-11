"""Document Intermediate Representation (Document IR).

The Document IR is the single source of truth for a reconstructed document: it
is produced by the analysis stages, validated before rendering, mutated by the
element editor, and consumed by every renderer. Nothing in the pipeline is
allowed to bypass it.

Coordinate conventions
----------------------
Three coordinate spaces are used consistently across the codebase:

``original``
    Pixel coordinates of the uploaded file (before any transformation).
``page``
    Pixel coordinates of the *processed* page image actually analysed
    (``PageIR.width_px`` x ``PageIR.height_px``). Detection and OCR work here.
``normalized``
    ``0..1`` fractions of the processed page. Used for storage and for
    renderer-independent geometry.

Millimetre values are derived from the page format and are what the renderers
consume. ``PageIR.transforms`` records every geometric operation applied to the
original so that boxes can be mapped back with
:func:`painter.geometry.map_bbox_original_to_page`.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class ElementType(str, Enum):
    """Element classes targeted by the reconstruction engine."""

    HEADING = "heading"
    SUBHEADING = "subheading"
    PARAGRAPH = "paragraph"
    TEXT_BLOCK = "text_block"
    LIST = "list"
    IMAGE = "image"
    LOGO = "logo"
    TABLE = "table"
    TABLE_CAPTION = "table_caption"
    FIGURE_CAPTION = "figure_caption"
    SHAPE = "shape"
    LINE = "line"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    CHECKBOX = "checkbox"
    FORM_FIELD = "form_field"
    SIGNATURE = "signature"
    STAMP = "stamp"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    FORMULA = "formula"
    UNKNOWN = "unknown"


TEXT_ELEMENT_TYPES: frozenset[ElementType] = frozenset(
    {
        ElementType.HEADING,
        ElementType.SUBHEADING,
        ElementType.PARAGRAPH,
        ElementType.TEXT_BLOCK,
        ElementType.LIST,
        ElementType.TABLE_CAPTION,
        ElementType.FIGURE_CAPTION,
        ElementType.HEADER,
        ElementType.FOOTER,
        ElementType.PAGE_NUMBER,
        ElementType.FORMULA,
        ElementType.FORM_FIELD,
    }
)

IMAGE_ELEMENT_TYPES: frozenset[ElementType] = frozenset(
    {ElementType.IMAGE, ElementType.LOGO, ElementType.SIGNATURE, ElementType.STAMP}
)

CONTAINER_ELEMENT_TYPES: frozenset[ElementType] = frozenset({ElementType.TABLE})

SHAPE_ELEMENT_TYPES: frozenset[ElementType] = frozenset(
    {
        ElementType.SHAPE,
        ElementType.LINE,
        ElementType.RECTANGLE,
        ElementType.CIRCLE,
        ElementType.CHECKBOX,
    }
)


class Alignment(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFY = "justify"


class AssetOrigin(str, Enum):
    EXTRACTED = "extracted"
    RECOVERED = "recovered"
    RENDERED = "rendered"
    NONE = "none"


class ReviewFlag(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    OCR_FAILED = "ocr_failed"
    OCR_UNCERTAIN = "ocr_uncertain"
    TEXT_MISSING = "text_missing"
    IMAGE_UNMATCHED = "image_unmatched"
    GEOMETRY_ESTIMATED = "geometry_estimated"
    TYPOGRAPHY_ESTIMATED = "typography_estimated"
    TABLE_UNCERTAIN = "table_uncertain"
    DETECTION_FALLBACK = "detection_fallback"
    EXTERNAL_MATCH_LOW_CONFIDENCE = "external_match_low_confidence"
    SHAPE_IMAGE_FALLBACK = "shape_image_fallback"
    NEEDS_REVIEW = "needs_review"


class BBox(BaseModel):
    """Axis-aligned box in pixel space."""

    model_config = ConfigDict(frozen=False)

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, values: list[float] | tuple[float, ...]) -> "BBox":
        x1, y1, x2, y2 = (float(v) for v in values[:4])
        return cls(x1=min(x1, x2), y1=min(y1, y2), x2=max(x1, x2), y2=max(y1, y2))

    def clip(self, width: float, height: float) -> "BBox":
        return BBox(
            x1=min(max(self.x1, 0.0), width),
            y1=min(max(self.y1, 0.0), height),
            x2=min(max(self.x2, 0.0), width),
            y2=min(max(self.y2, 0.0), height),
        )

    def pad(self, amount: float, width: float, height: float) -> "BBox":
        return BBox(
            x1=self.x1 - amount, y1=self.y1 - amount, x2=self.x2 + amount, y2=self.y2 + amount
        ).clip(width, height)

    def iou(self, other: "BBox") -> float:
        ix = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def overlap_fraction(self, other: "BBox") -> float:
        """Fraction of *this* box covered by ``other``."""
        ix = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        return (ix * iy) / self.area if self.area > 0 else 0.0


class NormBBox(BaseModel):
    """Box expressed as 0..1 fractions of the page."""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_bbox(cls, box: BBox, width: float, height: float) -> "NormBBox":
        if width <= 0 or height <= 0:
            return cls(x1=0.0, y1=0.0, x2=0.0, y2=0.0)
        return cls(
            x1=round(box.x1 / width, 6),
            y1=round(box.y1 / height, 6),
            x2=round(box.x2 / width, 6),
            y2=round(box.y2 / height, 6),
        )

    def to_bbox(self, width: float, height: float) -> BBox:
        return BBox(
            x1=self.x1 * width, y1=self.y1 * height, x2=self.x2 * width, y2=self.y2 * height
        )

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


class TextLine(BaseModel):
    """A single recognised line of text inside an element."""

    text: str
    bbox: BBox
    confidence: float | None = None
    words: list[str] = Field(default_factory=list)
    source: str | None = None


class Typography(BaseModel):
    """Estimated typography for a text element.

    ``estimated`` is always true for automatically derived values; a manual edit
    sets it to false for the fields the user touched.
    """

    font_family: str | None = None
    font_family_candidates: list[str] = Field(default_factory=list)
    font_size_pt: float | None = None
    font_weight: int | None = None  # CSS numeric weight, 100..900
    italic: bool | None = None
    underline: bool = False
    color: str | None = None  # "#rrggbb"
    line_height_pt: float | None = None
    line_spacing_ratio: float | None = None
    paragraph_spacing_pt: float | None = None
    letter_spacing_em: float | None = None
    word_spacing_em: float | None = None
    indent_first_line_mm: float | None = None
    indent_left_mm: float | None = None
    estimated: bool = True
    confidence: float | None = None
    notes: list[str] = Field(default_factory=list)


class ListInfo(BaseModel):
    ordered: bool = False
    marker: str | None = None
    items: list[str] = Field(default_factory=list)
    item_count: int = 0


class ImageCandidate(BaseModel):
    """A candidate replacement asset proposed by an external image search."""

    url: str
    title: str | None = None
    source: str | None = None
    width: int | None = None
    height: int | None = None
    score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    local_path: str | None = None
    downloaded: bool = False
    rejected_reason: str | None = None
    license_note: str | None = None
    provider: str | None = None
    retrieved_at: datetime | None = None


class ImageContent(BaseModel):
    """Image/logo payload attached to an element."""

    asset_path: str | None = None  # path relative to the document workspace
    asset_origin: AssetOrigin = AssetOrigin.NONE
    width_px: int | None = None
    height_px: int | None = None
    aspect_ratio: float | None = None
    perceptual_hash: str | None = None
    source_url: str | None = None
    source_title: str | None = None
    search_query_used: bool = False
    candidates: list[ImageCandidate] = Field(default_factory=list)
    matched: bool = False
    match_confidence: float | None = None
    notes: list[str] = Field(default_factory=list)


class TableCell(BaseModel):
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    text: str = ""
    bbox: BBox | None = None
    alignment: Alignment | None = None
    confidence: float | None = None
    is_header: bool = False
    background_color: str | None = None
    typography: Typography | None = None
    flags: list[ReviewFlag] = Field(default_factory=list)


class TableContent(BaseModel):
    rows: int = 0
    cols: int = 0
    cells: list[TableCell] = Field(default_factory=list)
    column_widths_mm: list[float] = Field(default_factory=list)
    row_heights_mm: list[float] = Field(default_factory=list)
    border_width_mm: float | None = None
    has_outer_border: bool = True
    header_rows: int = 0
    detection_method: str | None = None
    structure_confidence: float | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_grid(self) -> "TableContent":
        if self.cells and self.rows and self.cols:
            for cell in self.cells:
                if cell.row >= self.rows or cell.col >= self.cols:
                    raise ValueError(
                        f"cell ({cell.row},{cell.col}) outside {self.rows}x{self.cols} grid"
                    )
        return self


class ShapeContent(BaseModel):
    kind: str = "rectangle"  # rectangle | line | circle | arrow | checkbox | shaded
    stroke_color: str | None = "#000000"
    stroke_width_mm: float | None = None
    fill_color: str | None = None
    corner_radius_mm: float | None = None
    orientation: Literal["horizontal", "vertical", "diagonal"] | None = None
    checked: bool | None = None
    rendered_asset_path: str | None = None
    notes: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    """Where a piece of information came from."""

    stage: str
    method: str
    provider: str | None = None
    model: str | None = None
    confidence: float | None = None
    timestamp: datetime = Field(default_factory=utcnow)
    notes: str | None = None


class EditRecord(BaseModel):
    field: str
    old_value: Any = None
    new_value: Any = None
    editor: str = "user"
    timestamp: datetime = Field(default_factory=utcnow)


class Element(BaseModel):
    """One reconstructed document element."""

    id: str = Field(default_factory=lambda: new_id("el"))
    type: ElementType
    bbox: BBox
    norm_bbox: NormBBox | None = None
    original_bbox: BBox | None = None
    confidence: float = 0.0
    detection_class: str | None = None
    z: int = 0
    rotation: float = 0.0
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)
    order_index: int | None = None

    text: str | None = None
    lines: list[TextLine] = Field(default_factory=list)
    typography: Typography | None = None
    alignment: Alignment | None = None
    list_info: ListInfo | None = None

    image: ImageContent | None = None
    table: TableContent | None = None
    shape: ShapeContent | None = None

    flags: list[ReviewFlag] = Field(default_factory=list)
    provenance: list[Provenance] = Field(default_factory=list)
    edits: list[EditRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @property
    def needs_review(self) -> bool:
        return bool(self.flags)

    def add_flag(self, flag: ReviewFlag) -> None:
        if flag not in self.flags:
            self.flags.append(flag)


class MarginMM(BaseModel):
    top: float = 0.0
    right: float = 0.0
    bottom: float = 0.0
    left: float = 0.0

    @property
    def uniform(self) -> bool:
        return max(self.top, self.right, self.bottom, self.left) - min(
            self.top, self.right, self.bottom, self.left
        ) < 0.6


class PageFormat(BaseModel):
    name: str = "A4"
    width_mm: float = 210.0
    height_mm: float = 297.0
    estimated: bool = True
    #: Stated assumptions behind an inferred format: a scan cannot prove paper
    #: size, so the assumption is recorded rather than hidden.
    notes: list[str] = Field(default_factory=list)


class PageQuality(BaseModel):
    similarity: float | None = None
    ssim: float | None = None
    pixel_diff_ratio: float | None = None
    edge_iou: float | None = None
    projection_correlation: float | None = None
    text_similarity: float | None = None
    table_geometry_score: float | None = None
    metrics_used: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    iterations: int = 0


class PageTransform(BaseModel):
    """A geometric operation applied to the original image, with parameters."""

    kind: str  # crop | resize | perspective | rotate | deskew
    params: dict[str, Any] = Field(default_factory=dict)
    note: str | None = None


class PageIR(BaseModel):
    index: int = 0
    original_path: str | None = None
    image_path: str | None = None  # processed page used for analysis
    preview_path: str | None = None
    width_px: int = 0
    height_px: int = 0
    dpi: float = 200.0
    page_format: PageFormat = Field(default_factory=PageFormat)
    orientation: Literal["portrait", "landscape"] = "portrait"
    margins_mm: MarginMM = Field(default_factory=MarginMM)
    elements: list[Element] = Field(default_factory=list)
    reading_order: list[str] = Field(default_factory=list)
    transforms: list[PageTransform] = Field(default_factory=list)
    quality: PageQuality = Field(default_factory=PageQuality)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def element(self, element_id: str) -> Element | None:
        for element in self.elements:
            if element.id == element_id:
                return element
        return None

    def elements_of_type(self, *types: ElementType) -> list[Element]:
        wanted = set(types)
        return [e for e in self.elements if e.type in wanted]

    @property
    def mm_per_px_x(self) -> float:
        return self.page_format.width_mm / self.width_px if self.width_px else 0.0

    @property
    def mm_per_px_y(self) -> float:
        return self.page_format.height_mm / self.height_px if self.height_px else 0.0

    def px_to_mm(self, box: BBox) -> dict[str, float]:
        return {
            "x": round(box.x1 * self.mm_per_px_x, 3),
            "y": round(box.y1 * self.mm_per_px_y, 3),
            "width": round(box.width * self.mm_per_px_x, 3),
            "height": round(box.height * self.mm_per_px_y, 3),
        }


class DocumentSource(BaseModel):
    filename: str
    stored_path: str
    sha256: str
    content_type: str | None = None
    size_bytes: int = 0
    page_count: int = 0
    uploaded_at: datetime = Field(default_factory=utcnow)


class DocumentMetadata(BaseModel):
    title: str | None = None
    language: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class DocumentIR(BaseModel):
    """Root of the intermediate representation."""

    schema_version: str = SCHEMA_VERSION
    document_id: str = Field(default_factory=lambda: new_id("doc"))
    job_id: str | None = None
    source: DocumentSource | None = None
    pages: list[PageIR] = Field(default_factory=list)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    warnings: list[str] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    revision: int = 1
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def page(self, index: int) -> PageIR | None:
        for page in self.pages:
            if page.index == index:
                return page
        return None

    def find_element(self, element_id: str) -> tuple[PageIR, Element] | None:
        for page in self.pages:
            element = page.element(element_id)
            if element is not None:
                return page, element
        return None

    def touch(self) -> None:
        self.updated_at = utcnow()

    def recompute_stats(self) -> None:
        elements = [e for page in self.pages for e in page.elements]
        by_type: dict[str, int] = {}
        for element in elements:
            by_type[element.type.value] = by_type.get(element.type.value, 0) + 1
        self.stats = {
            "pages": len(self.pages),
            "elements": len(elements),
            "elements_by_type": by_type,
            "elements_needing_review": sum(1 for e in elements if e.needs_review),
            "text_characters": sum(len(e.text or "") for e in elements),
        }


class ValidationIssue(BaseModel):
    level: Literal["error", "warning"] = "warning"
    code: str
    message: str
    element_id: str | None = None
    page_index: int | None = None


class ValidationReport(BaseModel):
    ok: bool = True
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]


def validate_document(document: DocumentIR) -> ValidationReport:
    """Validate the IR before it reaches a renderer.

    Checks structural integrity (ids, parents, reading order), geometric sanity
    (boxes inside the page, no unexplained overlap between flow elements) and
    content sanity (text elements must carry text or an explicit flag).
    """
    report = ValidationReport()

    def issue(level: str, code: str, message: str, element_id: str | None = None, page: int | None = None) -> None:
        report.issues.append(
            ValidationIssue(level=level, code=code, message=message, element_id=element_id, page_index=page)
        )

    if not document.pages:
        issue("error", "no_pages", "document has no pages")

    for page in document.pages:
        seen_ids: set[str] = set()
        for element in page.elements:
            if element.id in seen_ids:
                issue("error", "duplicate_element_id", f"duplicate element id {element.id}", element.id, page.index)
            seen_ids.add(element.id)

            box = element.bbox
            if box.width <= 0 or box.height <= 0:
                issue("error", "empty_bbox", f"element {element.id} has an empty bounding box", element.id, page.index)
            if page.width_px and page.height_px:
                tolerance = 1.5
                if (
                    box.x1 < -tolerance
                    or box.y1 < -tolerance
                    or box.x2 > page.width_px + tolerance
                    or box.y2 > page.height_px + tolerance
                ):
                    issue(
                        "error",
                        "bbox_out_of_page",
                        f"element {element.id} bbox {box.as_list()} outside page "
                        f"{page.width_px}x{page.height_px}",
                        element.id,
                        page.index,
                    )
            if element.type in TEXT_ELEMENT_TYPES and not (element.text or "").strip():
                if ReviewFlag.TEXT_MISSING not in element.flags and ReviewFlag.OCR_FAILED not in element.flags:
                    # flag it here, so the warning is never about an unflagged element
                    element.add_flag(ReviewFlag.TEXT_MISSING)
                    issue(
                        "warning",
                        "text_missing",
                        f"text element {element.id} has no text; flagged for review",
                        element.id,
                        page.index,
                    )
            if element.type in IMAGE_ELEMENT_TYPES:
                content = element.image
                if content is None or (not content.asset_path and not content.source_url):
                    issue(
                        "warning",
                        "image_without_asset",
                        f"image element {element.id} has no asset path",
                        element.id,
                        page.index,
                    )
            if element.type == ElementType.TABLE:
                table = element.table
                if table is None or not table.cells:
                    issue("warning", "table_without_cells", f"table {element.id} has no cells", element.id, page.index)
                elif table.rows * table.cols > 0 and len(table.cells) < max(1, table.rows + table.cols - 1):
                    issue(
                        "warning",
                        "table_sparse",
                        f"table {element.id} has {len(table.cells)} cells for a "
                        f"{table.rows}x{table.cols} grid",
                        element.id,
                        page.index,
                    )
            if element.parent_id and element.parent_id not in seen_ids:
                issue(
                    "warning",
                    "dangling_parent",
                    f"element {element.id} references missing parent {element.parent_id}",
                    element.id,
                    page.index,
                )

        for element_id in page.reading_order:
            if element_id not in seen_ids:
                issue(
                    "error",
                    "reading_order_unknown_id",
                    f"reading order references unknown element {element_id}",
                    element_id,
                    page.index,
                )
        flow_ids = {
            e.id for e in page.elements if e.type in TEXT_ELEMENT_TYPES or e.type in IMAGE_ELEMENT_TYPES
        }
        missing_order = flow_ids - set(page.reading_order)
        if missing_order:
            issue(
                "warning",
                "reading_order_incomplete",
                f"{len(missing_order)} element(s) missing from reading order",
                None,
                page.index,
            )

    report.ok = not report.errors
    return report
