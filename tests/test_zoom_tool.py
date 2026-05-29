from PIL import Image

from src.tools.zoom_tool import zoom_in


def test_zoom_in_crops_requested_region():
    image = Image.new("RGB", (100, 80), "white")

    obs = zoom_in(image, [10, 20, 40, 50])

    assert obs.bbox_xyxy == (10, 20, 40, 50)
    assert obs.original_size == (100, 80)
    assert obs.crop_size == (30, 30)
    assert obs.image.size == (30, 30)


def test_zoom_in_clips_and_pads_bbox():
    image = Image.new("RGB", (100, 80), "white")

    obs = zoom_in(image, [-5, 10, 20, 30], padding=10)

    assert obs.bbox_xyxy == (0, 0, 30, 40)
    assert obs.crop_size == (30, 40)


def test_zoom_in_supports_resize():
    image = Image.new("RGB", (100, 80), "white")

    obs = zoom_in(image, [10, 20, 40, 50], output_size=(64, 64))

    assert obs.bbox_xyxy == (10, 20, 40, 50)
    assert obs.crop_size == (64, 64)
    assert obs.image.size == (64, 64)


def test_zoom_in_metadata_is_json_serializable_shape():
    image = Image.new("RGB", (100, 80), "white")

    meta = zoom_in(image, [10, 20, 40, 50]).metadata()

    assert meta == {
        "bbox_xyxy": [10, 20, 40, 50],
        "original_size": [100, 80],
        "crop_size": [30, 30],
        "source": None,
        "valid": True,
        "clipped": False,
        "error": None,
        "area_ratio": 0.1125,
    }
