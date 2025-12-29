from pathlib import Path

from PIL import Image

from vectorizer.pipeline import VectorizationSettings, vectorize_image


def test_vectorize_simple_shapes(tmp_path: Path) -> None:
    # Build a simple two-color image with a rectangle and a diagonal stripe.
    img = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
    for x in range(16, 48):
        for y in range(16, 48):
            img.putpixel((x, y), (200, 30, 30, 255))
    for i in range(64):
        img.putpixel((i, i), (20, 20, 200, 255))

    source = tmp_path / "source.png"
    img.save(source)

    output = tmp_path / "vectorized.svg"
    settings = VectorizationSettings(palette_size=4, blur_radius=0, contour_tolerance=0.8, min_contour_area=5)
    report = vectorize_image(source, output, settings)

    assert output.exists()
    assert report.shapes_written >= 2
    with output.open() as f:
        content = f.read()
    assert "polygon" in content
    assert "#c81e1e" in content or "#1414c8" in content
