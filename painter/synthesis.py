"""Physically plausible camera-capture simulation with exact labels.

The point of this module is labels that are *known*, not guessed. A clean digital
page is placed on a photographed surface, warped, lit, blurred and compressed,
and because every step is a transformation we applied ourselves, the page
corners, the layout boxes, the text and the degradation parameters are all exact.

That matters because the failure this exists to fix was a page-boundary detector
returning ``None`` on a certificate photographed on a concrete floor. Labelling
photographs of documents by hand at that scale is not feasible, and "random
labelling" produces a model that has learned nothing: here each photograph is
paired with the very page it shows, so the ground truth is not an annotation but
a record of the transformation.

Coordinates are always ``(x, y)`` floats in the photograph's pixel space, and
page corners are ordered top-left, top-right, bottom-right, bottom-left — the
order SmartDoc's own ground truth uses, so real and synthetic samples are
interchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

#: Corner order used everywhere in this project, matching SmartDoc.
CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")


@dataclass
class CaptureParams:
    """The physical conditions of one simulated capture."""

    #: How far the page is bent, 0 for a flat sheet.
    curl: float = 0.0
    #: Whether a fold crease was applied.
    fold: bool = False
    #: Gaussian blur sigma in pixels.
    blur: float = 0.0
    #: Directional motion blur length in pixels.
    motion_blur: float = 0.0
    #: Shadow strength, 0..1.
    shadow: float = 0.0
    #: Specular glare strength, 0..1.
    glare: float = 0.0
    #: Fraction of the page that falls outside the frame.
    crop: float = 0.0
    #: Corner displacement of the perspective warp, in pixels.
    perspective: float = 0.0
    #: In-plane rotation in degrees.
    rotation: float = 0.0
    #: JPEG quality used for the final encode.
    jpeg_quality: int = 92
    #: Additive sensor noise sigma.
    noise: float = 0.0
    #: Per-channel gain, standing in for an ambient colour cast.
    colour_cast: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def distortion_labels(self, crop_threshold: float = 0.01) -> list[str]:
        """Return the taxonomy tags this capture earns.

        The thresholds are deliberately conservative: a tag should mean the
        condition is actually present, otherwise the labels stop being useful
        supervision and become noise.
        """
        labels: list[str] = []
        if self.fold:
            labels.append("fold")
        elif self.curl > 0.08:
            labels.append("curl")
        if self.blur > 1.2 or self.motion_blur > 6:
            labels.append("blur")
        if self.shadow > 0.18:
            labels.append("shadow")
        if self.glare > 0.12:
            labels.append("glare")
        if self.crop > crop_threshold:
            labels.append("crop")
        if self.perspective > 12:
            labels.append("perspective")
        if abs(self.rotation) > 6:
            labels.append("rotation")
        return labels

    def as_dict(self) -> dict[str, Any]:
        return {
            "curl": round(self.curl, 4),
            "fold": self.fold,
            "blur": round(self.blur, 3),
            "motion_blur": round(self.motion_blur, 3),
            "shadow": round(self.shadow, 4),
            "glare": round(self.glare, 4),
            "crop": round(self.crop, 4),
            "perspective": round(self.perspective, 2),
            "rotation": round(self.rotation, 3),
            "jpeg_quality": self.jpeg_quality,
            "noise": round(self.noise, 3),
            "colour_cast": [round(value, 4) for value in self.colour_cast],
            "labels": self.distortion_labels(),
        }


@dataclass
class PhotoSample:
    """A simulated photograph plus everything needed to label it."""

    photo: np.ndarray
    page_quad: np.ndarray
    homography: np.ndarray
    params: CaptureParams
    #: The displacement field applied for page curl, or ``None`` when flat.
    curl_field: np.ndarray | None = None
    page_size: tuple[int, int] = (0, 0)
    metadata: dict[str, Any] = field(default_factory=dict)

    def map_page_points(self, points: np.ndarray) -> np.ndarray:
        """Map points from clean-page coordinates into the photograph.

        Layout boxes and text lines are known in page space, so this is how they
        become labels in photo space without anyone drawing a box by hand.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(points, self.homography).reshape(-1, 2)
        if self.curl_field is not None:
            projected = _apply_curl_to_points(projected, self.curl_field)
        return projected

    def map_page_boxes(self, boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
        """Map ``(x1, y1, x2, y2)`` page boxes to photo-space boxes.

        A curved page is not a rectangle in the photograph, so the result is the
        axis-aligned box that contains the warped corners: the same convention the
        project uses for detected regions.
        """
        out = np.zeros((len(boxes), 4), dtype=np.float64)
        for index, (x1, y1, x2, y2) in enumerate(boxes):
            corners = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64
            )
            mapped = self.map_page_points(corners)
            out[index] = [
                mapped[:, 0].min(),
                mapped[:, 1].min(),
                mapped[:, 0].max(),
                mapped[:, 1].max(),
            ]
        return out


def _apply_curl_to_points(points: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Sample a displacement field at ``points`` and add it.

    Bilinear sampling keeps the label consistent with the pixels: the page layer
    is displaced by resampling the same field, so a corner that lands on a page
    edge in the image lands on the same edge in the label.
    """
    height, width = field.shape[:2]
    xs = np.clip(points[:, 0], 0, width - 1.001)
    ys = np.clip(points[:, 1], 0, height - 1.001)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    fx = (xs - x0)[:, None]
    fy = (ys - y0)[:, None]

    def sample(channel: int) -> np.ndarray:
        top = field[y0, x0, channel][:, None] * (1 - fx) + field[y0, x1, channel][:, None] * fx
        bottom = field[y1, x0, channel][:, None] * (1 - fx) + field[y1, x1, channel][:, None] * fx
        return top * (1 - fy) + bottom * fy

    return points + np.hstack([sample(0), sample(1)])


def sample_page_quad(
    frame_width: int,
    frame_height: int,
    rng: np.random.Generator,
    min_scale: float = 0.35,
    max_scale: float = 0.95,
    max_rotation_deg: float = 14.0,
    max_perspective: float = 45.0,
) -> tuple[np.ndarray, float, float]:
    """Sample a plausible page placement in the camera frame.

    Returns the quad, the rotation in degrees and the largest corner offset, so
    the caller can record them as labels instead of re-deriving them.
    """
    # The page keeps its aspect ratio; only its size, angle and perspective vary.
    scale = float(rng.uniform(min_scale, max_scale))
    target_width = frame_width * scale
    target_height = target_width * rng.uniform(1.2, 1.45)
    if target_height > frame_height * max_scale:
        shrink = (frame_height * max_scale) / target_height
        target_width *= shrink
        target_height *= shrink

    half_w, half_h = target_width / 2, target_height / 2
    rotation = float(rng.uniform(-max_rotation_deg, max_rotation_deg))
    angle = np.deg2rad(rotation)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    base = np.array(
        [[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]],
        dtype=np.float64,
    )
    rotated = base @ np.array([[cos_a, -sin_a], [sin_a, cos_a]]).T

    # Perspective: push each corner along the direction from the centre.
    offsets = rng.uniform(0.0, max_perspective, size=(4, 1))
    directions = rotated / np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-6)
    quad = rotated + directions * offsets
    perspective = float(offsets.max())

    # Place it, allowing the page to run off the frame edge only sometimes.
    centre_x = frame_width / 2 + rng.uniform(-0.08, 0.08) * frame_width
    centre_y = frame_height / 2 + rng.uniform(-0.08, 0.08) * frame_height
    quad = quad + np.array([centre_x, centre_y])
    return quad.astype(np.float64), rotation, perspective


def build_curl_field(
    width: int,
    height: int,
    rng: np.random.Generator,
    fold: bool,
) -> tuple[np.ndarray, float]:
    """Build a displacement field standing in for a bent or folded sheet.

    A real page curves because paper resists bending: the displacement is a smooth
    sinusoid along one axis, not random jitter. A fold adds a crease — a sharp,
    localised slope reversal — which is the case that breaks line-based
    rectification and the reason ``fold`` is its own label.
    """
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    amplitude = rng.uniform(0.02, 0.075) * width
    if rng.random() < 0.5:
        # bend around the vertical axis: displacement grows along x
        phase = rng.uniform(0, np.pi)
        frequency = rng.uniform(0.6, 1.4) * np.pi / max(width, 1)
        horizontal = amplitude * np.sin(frequency * xs + phase)
        vertical = np.zeros_like(horizontal)
        strength = amplitude / max(width, 1)
    else:
        # bend around the horizontal axis
        phase = rng.uniform(0, np.pi)
        frequency = rng.uniform(0.6, 1.4) * np.pi / max(height, 1)
        vertical = amplitude * np.sin(frequency * ys + phase)
        horizontal = np.zeros_like(vertical)
        strength = amplitude / max(height, 1)

    if fold:
        # a crease: one narrow band where the surface tilts sharply
        if rng.random() < 0.5:
            crease = rng.uniform(0.25, 0.75) * width
            band = np.exp(-((xs - crease) ** 2) / (2 * (width * 0.02) ** 2))
            horizontal = horizontal + band * rng.uniform(0.02, 0.05) * width * rng.choice([-1, 1])
        else:
            crease = rng.uniform(0.25, 0.75) * height
            band = np.exp(-((ys - crease) ** 2) / (2 * (height * 0.02) ** 2))
            vertical = vertical + band * rng.uniform(0.02, 0.05) * height * rng.choice([-1, 1])
        strength = max(strength, 0.04)

    field = np.dstack([horizontal, vertical]).astype(np.float32)
    return field, float(strength)


def _warp_page(page: np.ndarray, quad: np.ndarray, frame_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp a clean page onto ``quad`` inside a frame, returning layer and mask."""
    frame_width, frame_height = frame_size
    page_height, page_width = page.shape[:2]
    source = np.array(
        [[0, 0], [page_width - 1, 0], [page_width - 1, page_height - 1], [0, page_height - 1]],
        dtype=np.float64,
    )
    # OpenCV 5 insists on CV_32F point vectors here, not the float64 a caller
    # naturally reaches for.
    homography = cv2.getPerspectiveTransform(
        source.astype(np.float32), quad.astype(np.float32)
    )
    layer = cv2.warpPerspective(
        page, homography, (frame_width, frame_height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )
    mask = cv2.warpPerspective(
        np.full((page_height, page_width), 255, np.uint8),
        homography,
        (frame_width, frame_height),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    return layer, mask, homography


def _apply_field(image: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Resample ``image`` through a displacement field."""
    height, width = image.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    map_x = (xs + field[:, :, 0]).astype(np.float32)
    map_y = (ys + field[:, :, 1]).astype(np.float32)
    return cv2.remap(
        image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def simulate_capture(
    page: np.ndarray,
    background: np.ndarray,
    rng: np.random.Generator,
    *,
    curl_probability: float = 0.55,
    fold_probability: float = 0.18,
    crop_probability: float = 0.12,
    max_perspective: float = 45.0,
) -> PhotoSample:
    """Photograph a clean ``page`` lying on ``background``.

    ``background`` is a real surface texture — a floor, a desk, clutter — so the
    page has to be found against something that genuinely competes with it, which
    is the condition the heuristic detector failed.
    """
    frame_height, frame_width = background.shape[:2]
    quad, rotation, perspective = sample_page_quad(
        frame_width, frame_height, rng, max_perspective=max_perspective
    )

    # A page that runs off the frame is a real capture condition worth labelling.
    crop = 0.0
    if rng.random() < crop_probability:
        shift = rng.uniform(0.02, 0.10) * frame_width * rng.choice([-1, 1])
        quad = quad + np.array([shift, rng.uniform(-0.03, 0.03) * frame_height])
        outside = (
            (quad[:, 0] < 0) | (quad[:, 0] > frame_width) | (quad[:, 1] < 0) | (quad[:, 1] > frame_height)
        )
        crop = float(outside.mean()) * 0.25 if outside.any() else 0.0

    layer, mask, homography = _warp_page(page, quad, (frame_width, frame_height))

    params = CaptureParams(rotation=rotation, perspective=perspective, crop=crop)
    curl_field: np.ndarray | None = None
    if rng.random() < curl_probability:
        fold = rng.random() < fold_probability
        curl_field, strength = build_curl_field(frame_width, frame_height, rng, fold)
        layer = _apply_field(layer, curl_field)
        mask = _apply_field(mask, curl_field)
        params.curl = strength
        params.fold = bool(fold)

    canvas = background.astype(np.float32).copy()

    # Shadow: the page occludes the light, so the surface around it darkens. Blur
    # the mask, offset it along the light direction and use it as an attenuation.
    if rng.random() < 0.85:
        shadow_strength = float(rng.uniform(0.05, 0.45))
        offset = int(rng.uniform(0.01, 0.05) * frame_width)
        kernel = max(3, int(rng.uniform(0.01, 0.05) * frame_width) | 1)
        shadow = cv2.GaussianBlur(mask, (kernel, kernel), 0)
        shadow = np.roll(shadow, (offset, offset), axis=(0, 1)).astype(np.float32) / 255.0
        canvas *= 1.0 - shadow_strength * shadow[:, :, None]
        params.shadow = shadow_strength

    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    canvas = canvas * (1.0 - alpha) + layer.astype(np.float32) * alpha

    # Glare: a soft highlight, as a sheet of paper under a lamp produces.
    if rng.random() < 0.3:
        glare_strength = float(rng.uniform(0.05, 0.35))
        gx = int(rng.uniform(0.2, 0.8) * frame_width)
        gy = int(rng.uniform(0.2, 0.8) * frame_height)
        radius = int(rng.uniform(0.15, 0.4) * frame_width) | 1
        highlight = np.zeros((frame_height, frame_width), np.float32)
        cv2.circle(highlight, (gx, gy), radius, 1.0, -1)
        highlight = cv2.GaussianBlur(highlight, (radius | 1, radius | 1), 0)
        canvas += glare_strength * 255.0 * highlight[:, :, None] * alpha
        params.glare = glare_strength

    # Ambient colour cast, which is what turned "Department" into "Vepartment".
    if rng.random() < 0.5:
        cast = tuple(float(value) for value in rng.uniform(0.9, 1.1, size=3))
        canvas *= np.array(cast, dtype=np.float32)[None, None, :]
        params.colour_cast = cast

    # Illumination gradient across the frame.
    low, high = float(rng.uniform(0.88, 1.0)), float(rng.uniform(0.88, 1.0))
    # Shape matters here: a transposed 1-D ramp no longer broadcasts against a
    # non-square frame, which silently produced a ValueError on tall captures.
    if rng.random() < 0.5:
        gradient = np.linspace(low, high, frame_width)[None, :]
    else:
        gradient = np.linspace(low, high, frame_height)[:, None]
    canvas *= gradient[:, :, None]

    canvas = np.clip(canvas, 0, 255)

    if rng.random() < 0.35:
        params.blur = float(rng.uniform(0.6, 2.2))
        canvas = cv2.GaussianBlur(canvas, (0, 0), params.blur)
    if rng.random() < 0.3:
        params.motion_blur = float(rng.uniform(4, 13))
        size = int(params.motion_blur) | 1
        kernel = np.zeros((size, size), np.float32)
        angle = rng.uniform(0, np.pi)
        cv2.line(
            kernel,
            (size // 2 - int(np.cos(angle) * size / 2), size // 2 - int(np.sin(angle) * size / 2)),
            (size // 2 + int(np.cos(angle) * size / 2), size // 2 + int(np.sin(angle) * size / 2)),
            1.0,
            1,
        )
        kernel /= max(kernel.sum(), 1e-6)
        canvas = cv2.filter2D(canvas, -1, kernel)

    if rng.random() < 0.5:
        params.noise = float(rng.uniform(1.0, 6.0))
        canvas += rng.normal(0, params.noise, canvas.shape).astype(np.float32)

    params.jpeg_quality = int(rng.integers(68, 96))
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), params.jpeg_quality]
    ok, buffer = cv2.imencode(".jpg", np.clip(canvas, 0, 255).astype(np.uint8), encode_params)
    photo = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if ok else np.clip(canvas, 0, 255).astype(np.uint8)

    return PhotoSample(
        photo=photo,
        page_quad=quad.astype(np.float32),
        homography=homography,
        params=params,
        curl_field=curl_field,
        page_size=(page.shape[1], page.shape[0]),
        metadata={"frame_size": [frame_width, frame_height]},
    )


def background_from_photo(photo: np.ndarray, page_mask: np.ndarray | None = None) -> np.ndarray:
    """Recover a surface texture from a real photograph.

    This is how the user's own floor enters the training set: the concrete the
    certificate was lying on becomes a background the model is trained against,
    so the next certificate on that floor is not a surprise.
    """
    if page_mask is not None and page_mask.shape[:2] == photo.shape[:2]:
        surface = photo.copy()
        # Inpaint the page area so the texture does not contain a document.
        surface = cv2.inpaint(surface, (page_mask > 0).astype(np.uint8), 7, cv2.INPAINT_TELEA)
        return surface
    return photo.copy()
