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
        if "start_captured_at" not in params:
            return FakeResponse({"data": [{"id": "probe"}]})  # preflight probe
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


def fake_get_without_date_filters(url, params=None, timeout=None):
    """Fake API that fails on date filters (HTTP 500) and on the 'sequence' field."""
    if url.endswith("/images"):
        if "start_captured_at" in params:
            return FakeResponse({"error": {"message": "An unknown error has occurred", "code": 1}}, 500)
        if "is_pano" in params["fields"]:
            return FakeResponse({"error": {"message": "Unknown field"}}, 500)
        place = zlib.crc32(params["bbox"].encode()) % 1000
        # one image per year, 2014 to 2021 (captured_at in epoch milliseconds, mid-year)
        return FakeResponse(
            {
                "data": [
                    {"id": f"{place:03d}{year}", "captured_at": int((year - 1970) * 365.25 * 86400000 + 180 * 86400000), "sequence": f"s{year}", "thumb_1024_url": "https://thumbs.example/x.jpg"}
                    for year in range(2014, 2022)
                ]
            }
        )
    return fake_get(url, params, timeout)


def test_survey_falls_back_to_client_side_epochs(tmp_path, monkeypatch):
    monkeypatch.setattr(survey.mly.requests, "get", fake_get_without_date_filters)
    monkeypatch.setattr(survey, "sleep", lambda s: None)
    out = str(tmp_path / "survey")

    summary = survey.run_survey(
        out, "", per_cell=2, seed=1, max_samples=0,
        locations=survey.LOCATIONS[:2], epochs=survey.EPOCHS, token=TOKEN, pause=0,
    )

    assert summary["search_strategy"].startswith("client-side")
    assert [p["ok"] for p in summary["api_probes"]] == [False, True, False, False]
    assert "Unknown field" in summary["api_probes"][0]["error"]

    cells = pd.read_csv(os.path.join(out, "cells.csv"), keep_default_na=False)
    found = dict(zip(cells[cells["location"] == "Curitiba"]["epoch"], cells[cells["location"] == "Curitiba"]["images_found"]))
    assert found["2014-2015"] == 2 and found["2016-2017"] == 2 and found["2021"] == 1 and found["2022"] == 0
    # 2019 images get no detections from fake_get (IDs starting with 2016/2018 are special there, not these)
    assert summary["images_sampled"] == 2 * (2 + 2 + 1 + 1 + 1 + 1)
    report = open(os.path.join(out, "SEGMENTATION_AVAILABILITY.md"), encoding="utf-8").read()
    assert "## API probes" in report and "client-side" in report


def test_survey_aborts_when_every_search_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(survey.mly.requests, "get", lambda *a, **k: FakeResponse({"error": {"message": "boom"}}, 500))
    summary = survey.run_survey(
        str(tmp_path), "", per_cell=2, seed=1, max_samples=0,
        locations=survey.LOCATIONS[:2], epochs=survey.EPOCHS, token=TOKEN, pause=0,
    )
    assert summary["search_strategy"].startswith("none")
    assert summary["images_sampled"] == 0 and len(summary["api_probes"]) == 3
