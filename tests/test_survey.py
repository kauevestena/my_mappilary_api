"""
Offline end-to-end test of scripts/segmentation_survey.py against a fake API.
"""

import io
import zlib
import os
import sys

import pandas as pd
from PIL import Image
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))

import segmentation_survey as survey  # noqa: E402
from test_segmentation import EXTENT, FakeResponse, TOKEN, detection  # noqa: E402


class FakeThumb:
    def __init__(self):
        buffer = io.BytesIO()
        Image.new("RGB", (64, 48), (120, 120, 120)).save(buffer, format="JPEG")
        self.content = buffer.getvalue()

    def raise_for_status(self):
        pass


def fake_get(url, params=None, timeout=None):
    if url.startswith("https://thumbs.example"):
        return FakeThumb()
    if url.endswith("/images"):
        if params["start_captured_at"].startswith("2014"):
            return FakeResponse({"data": []})  # no imagery in this epoch
        year = params["start_captured_at"][:4]
        place = zlib.crc32(params["bbox"].encode()) % 1000  # unique image IDs per location
        return FakeResponse(
            {
                "data": [
                    {"id": f"{year}{place:03d}{i}", "captured_at": 1600000000000, "sequence": f"s{i}", "width": 64, "height": 48, "thumb_1024_url": "https://thumbs.example/x.jpg"}
                    for i in range(5)
                ]
            }
        )
    if url.endswith("/detections"):
        image_id = url.split("/")[-2]
        if image_id.startswith("2016"):
            return FakeResponse({"data": []})  # no detections in this epoch
        if image_id.startswith("2018"):
            return FakeResponse({"error": {"message": f"Invalid token {TOKEN}"}}, 400)
        return FakeResponse(
            {
                "data": [
                    detection("nature--sky", box(0, 0, EXTENT, 1000)),
                    detection("construction--flat--road", box(0, 3000, EXTENT, EXTENT)),
                ]
            }
        )
    raise AssertionError(f"unexpected url {url}")


def test_survey_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(survey.mly.requests, "get", fake_get)
    out, raw = str(tmp_path / "survey"), str(tmp_path / "raw")

    summary = survey.run_survey(
        out, raw, per_cell=2, seed=1, max_samples=3,
        locations=survey.LOCATIONS[:2],
        epochs=[e for e in survey.EPOCHS if e[0] in ("2014-2015", "2016-2017", "2018", "2019")],
        token=TOKEN, pause=0,
    )

    df = pd.read_csv(os.path.join(out, "segmentation_availability.csv"), keep_default_na=False)
    assert len(df) == 2 * 3 * 2  # 2 locations × 3 epochs with imagery × 2 images
    assert summary["cells"] == 8 and summary["cells_with_images"] == 6
    assert summary["requests_failed"] == 4
    assert summary["images_with_detections"] == 4
    assert summary["images_with_surface_classes"] == 4
    assert summary["sky_above_road"] == {"True": 4}

    report = open(os.path.join(out, "SEGMENTATION_AVAILABILITY.md"), encoding="utf-8").read()
    assert "## By capture epoch" in report and "samples/" in report
    assert len(os.listdir(os.path.join(out, "samples"))) == 3
    assert len(os.listdir(os.path.join(raw, "detections"))) == 4

    # the token never reaches the outputs
    for folder in (out, raw):
        for root, _, files in os.walk(folder):
            for name in files:
                with open(os.path.join(root, name), "rb") as f:
                    assert b"secret" not in f.read()
