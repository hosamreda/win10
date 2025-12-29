from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import VectorizationError, VectorizationSettings, vectorize_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vectorize raster images into clean SVG polygons.")
    parser.add_argument("source", type=Path, help="Path to the source raster image (PNG, JPG, etc.)")
    parser.add_argument("--output", "-o", type=Path, help="Where to write the SVG. Defaults to <source>.svg")
    parser.add_argument("--palette-size", type=int, default=12, help="Number of dominant colors to keep (default: 12)")
    parser.add_argument("--blur-radius", type=float, default=0.8, help="Gaussian blur radius to soften noise (default: 0.8)")
    parser.add_argument("--contour-tolerance", type=float, default=1.4, help="Higher = smoother polygons (default: 1.4)")
    parser.add_argument("--min-area", type=float, default=18.0, help="Minimum polygon area to keep (default: 18)")
    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=10,
        help="Drop pixels with alpha under this level before tracing (default: 10)",
    )
    parser.add_argument(
        "--no-blur",
        action="store_true",
        help="Disable the pre-quantization Gaussian blur to preserve very sharp edges",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_path = args.output or args.source.with_suffix(".svg")
    settings = VectorizationSettings(
        palette_size=args.palette_size,
        blur_radius=0 if args.no_blur else args.blur_radius,
        contour_tolerance=args.contour_tolerance,
        min_contour_area=args.min_area,
        alpha_threshold=args.alpha_threshold,
    )

    try:
        report = vectorize_image(args.source, output_path, settings)
    except VectorizationError as exc:
        parser.error(str(exc))
        return 2

    print(
        f"Vectorized {args.source} -> {output_path} "
        f"({report.shapes_written} shapes across {report.colors_processed} colors)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
