from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter
from skimage import measure
import svgwrite


@dataclass
class VectorizationSettings:
    """Configurable settings that shape how raster images are vectorized."""

    palette_size: int = 12
    blur_radius: float = 0.8
    contour_tolerance: float = 1.4
    min_contour_area: float = 18.0
    alpha_threshold: int = 10

    @property
    def use_blur(self) -> bool:
        return self.blur_radius > 0


@dataclass
class VectorizedShape:
    """Represents a single polygon ready to be written to SVG."""

    points: List[Tuple[float, float]]
    fill: Tuple[int, int, int, int]
    area: float


@dataclass
class VectorizationReport:
    """Statistics about the vectorization run."""

    colors_processed: int
    shapes_written: int
    width: int
    height: int


class VectorizationError(RuntimeError):
    pass


def _load_image(path: Path, settings: VectorizationSettings) -> Image.Image:
    try:
        image = Image.open(path).convert("RGBA")
    except OSError as exc:
        raise VectorizationError(f"Unable to open image '{path}': {exc}") from exc

    if settings.use_blur:
        image = image.filter(ImageFilter.GaussianBlur(radius=settings.blur_radius))

    return image


def _quantize(image: Image.Image, palette_size: int) -> Image.Image:
    # Pillow's quantize does a good job keeping edges while reducing palette size.
    return image.convert("RGB").quantize(colors=palette_size, method=Image.MEDIANCUT).convert("RGBA")


def _mask_for_color(data: np.ndarray, color: Sequence[int], alpha_threshold: int) -> np.ndarray:
    rgb = data[:, :, :3]
    alpha = data[:, :, 3]
    mask = np.all(rgb == np.array(color[:3], dtype=np.uint8), axis=-1)
    return np.logical_and(mask, alpha > alpha_threshold)


def _polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _extract_contours(mask: np.ndarray, tolerance: float, min_area: float) -> List[List[Tuple[float, float]]]:
    contours = measure.find_contours(mask.astype(float), 0.5)
    polygons: List[List[Tuple[float, float]]] = []
    for contour in contours:
        approx = measure.approximate_polygon(contour, tolerance=tolerance)
        # convert to (x, y) pairs and clip to ensure within bounds
        polygon = [(float(x), float(y)) for y, x in approx]
        area = _polygon_area(polygon)
        if area >= min_area:
            polygons.append(polygon)
    return polygons


def _to_hex(color: Sequence[int]) -> str:
    r, g, b, _ = color
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_shapes(image: Image.Image, settings: VectorizationSettings) -> List[VectorizedShape]:
    quantized = _quantize(image, settings.palette_size)
    data = np.array(quantized)

    unique_colors = np.unique(data.reshape(-1, data.shape[2]), axis=0)
    shapes: List[VectorizedShape] = []
    for color in unique_colors:
        mask = _mask_for_color(data, color, settings.alpha_threshold)
        if not np.any(mask):
            continue
        polygons = _extract_contours(mask, settings.contour_tolerance, settings.min_contour_area)
        for polygon in polygons:
            area = _polygon_area(polygon)
            shapes.append(VectorizedShape(points=polygon, fill=tuple(int(c) for c in color), area=area))

    # Largest areas first so dominant shapes are laid down first.
    return sorted(shapes, key=lambda s: s.area, reverse=True)


def vectorize_image(source: Path, output: Path, settings: VectorizationSettings | None = None) -> VectorizationReport:
    settings = settings or VectorizationSettings()

    image = _load_image(source, settings)
    shapes = _build_shapes(image, settings)

    width, height = image.size
    dwg = svgwrite.Drawing(filename=str(output), size=(width, height))
    for shape in shapes:
        dwg.add(dwg.polygon(shape.points, fill=_to_hex(shape.fill), fill_opacity=shape.fill[3] / 255))

    output.parent.mkdir(parents=True, exist_ok=True)
    dwg.save()

    return VectorizationReport(
        colors_processed=len({shape.fill for shape in shapes}),
        shapes_written=len(shapes),
        width=width,
        height=height,
    )


__all__ = [
    "VectorizationSettings",
    "VectorizationReport",
    "VectorizationError",
    "vectorize_image",
]
