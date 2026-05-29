"""Zoom-in crop tool for large remote-sensing images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from PIL import Image


BBox = Sequence[float | int]


@dataclass(frozen=True)
class ZoomObservation:
    """Result returned by the zoom-in tool."""

    image: Image.Image | None
    bbox_xyxy: tuple[int, int, int, int]
    original_size: tuple[int, int]
    crop_size: tuple[int, int]
    source: str | None = None
    valid: bool = True
    clipped: bool = False
    error: str | None = None
    area_ratio: float = 0.0

    def metadata(self) -> dict[str, object]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "original_size": list(self.original_size),
            "crop_size": list(self.crop_size),
            "source": self.source,
            "valid": self.valid,
            "clipped": self.clipped,
            "error": self.error,
            "area_ratio": self.area_ratio,
        }


def _as_pil_image(image: str | Path | Image.Image) -> tuple[Image.Image, str | None]:
    if isinstance(image, Image.Image):
        return image, None
    path = Path(image)
    return Image.open(path).convert("RGB"), str(path)


def _convert_bbox(
    bbox: BBox,
    image_size: tuple[int, int],
    coord_type: Literal["pixel", "normalized"] = "pixel",
) -> list[float]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must contain 4 values, got {len(bbox)}")
    values = [float(v) for v in bbox]
    if coord_type == "pixel":
        return values
    if coord_type != "normalized":
        raise ValueError("coord_type must be 'pixel' or 'normalized'")
    width, height = image_size
    x1, y1, x2, y2 = values
    return [x1 * width, y1 * height, x2 * width, y2 * height]


def _clip_bbox(
    bbox: BBox,
    image_size: tuple[int, int],
    padding: int = 0,
    min_crop_size: int = 1,
    coord_type: Literal["pixel", "normalized"] = "pixel",
) -> tuple[tuple[int, int, int, int], bool]:
    if padding < 0:
        raise ValueError("padding must be non-negative")
    if min_crop_size < 1:
        raise ValueError("min_crop_size must be at least 1")

    width, height = image_size
    x1, y1, x2, y2 = _convert_bbox(bbox, image_size, coord_type)
    before_clip = (x1 - padding, y1 - padding, x2 + padding, y2 + padding)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding

    left = max(0, int(x1))
    top = max(0, int(y1))
    right = min(width, int(round(x2)))
    bottom = min(height, int(round(y2)))

    if right - left < min_crop_size:
        center = (left + right) // 2
        half = max(1, min_crop_size) // 2
        left = max(0, center - half)
        right = min(width, left + min_crop_size)
        left = max(0, right - min_crop_size)
    if bottom - top < min_crop_size:
        center = (top + bottom) // 2
        half = max(1, min_crop_size) // 2
        top = max(0, center - half)
        bottom = min(height, top + min_crop_size)
        top = max(0, bottom - min_crop_size)

    if right <= left or bottom <= top:
        raise ValueError(
            f"bbox {list(bbox)} is outside image bounds or empty for image size {image_size}"
        )
    clipped = (left, top, right, bottom) != tuple(int(round(v)) for v in before_clip)
    return (left, top, right, bottom), clipped


def zoom_in(
    image: str | Path | Image.Image,
    bbox: BBox,
    *,
    padding: int = 0,
    min_crop_size: int = 1,
    output_size: tuple[int, int] | int | None = None,
    coord_type: Literal["pixel", "normalized"] = "pixel",
    min_bbox_area_ratio: float = 0.0,
    max_bbox_area_ratio: float = 1.0,
) -> ZoomObservation:
    """Crop a bounded region from an image and return the crop plus metadata."""

    pil_image, source = _as_pil_image(image)
    crop_box, clipped = _clip_bbox(
        bbox,
        pil_image.size,
        padding=padding,
        min_crop_size=min_crop_size,
        coord_type=coord_type,
    )
    width, height = pil_image.size
    area_ratio = ((crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])) / float(width * height)
    if area_ratio < min_bbox_area_ratio or area_ratio > max_bbox_area_ratio:
        return ZoomObservation(
            image=None,
            bbox_xyxy=crop_box,
            original_size=pil_image.size,
            crop_size=(0, 0),
            source=source,
            valid=False,
            clipped=clipped,
            error=f"bbox area ratio {area_ratio:.6f} outside [{min_bbox_area_ratio}, {max_bbox_area_ratio}]",
            area_ratio=area_ratio,
        )
    crop = pil_image.crop(crop_box)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    if output_size is not None:
        if output_size[0] < 1 or output_size[1] < 1:
            raise ValueError("output_size values must be positive")
        crop = crop.resize(output_size, Image.Resampling.BICUBIC)
    return ZoomObservation(
        image=crop,
        bbox_xyxy=crop_box,
        original_size=pil_image.size,
        crop_size=crop.size,
        source=source,
        valid=True,
        clipped=clipped,
        error=None,
        area_ratio=area_ratio,
    )
