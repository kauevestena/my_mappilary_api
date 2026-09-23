"""
Offline end-to-end tests of scripts/segmentation_survey.py against a fake API.
"""

import io
import os
import re
import sys
from datetime import datetime, timezone

import pandas as pd
from PIL import Image
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))

import segmentation_survey as survey  # noqa: E402
from test_segmentation import (  # noqa: E402
    EXTENT, TOKEN, FakeResponse, FakeTileResponse, detection, encode_coverage_tile,
)

YEARS = range(2014, 2022)  # one image per year and location in the fake coverage tiles


class FakeThumb:
    def __init__(self):
        buffer = io.BytesIO()
        Image.new("RGB", (64, 48), (120, 120, 120)).save(buffer, format="JPEG")
        self.content = buffer.getvalue()

    def raise_for_status(self):
        pass


def mid_year_ms(year):
    return int(datetime(year, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)


def fake_get(url, params=None, timeout=None):
    if url.startswith("https://thumbs.example"):
        return FakeThumb()
    if url.startswith("https://tiles.mapillary.com"):
        x = url.split("/")[-2]
        return FakeTileResponse(
            encode_coverage_tile(
                [(100 * i, 100 * i, {"id": int(f"{year}{x}"), "captured_at": mid_year_ms(year), "sequence_id": str(year)})
                 for i, year in enumerate(YEARS)]
            )
        )
    if url.endswith("/images"):
        minx, _, maxx, _ = map(float, params["bbox"].split(","))
        if maxx - minx > 0.01:
            return FakeResponse({"error": {"message": "Please reduce the amount of data you're asking for, then retry your request"}}, 500)
        return FakeResponse({"data": [{"id": "1"}]})
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
    if re.search(r"graph.mapillary.com/\d+$", url):
        return FakeResponse({"thumb_1024_url": "https://thumbs.example/x.jpg", "id": url.split("/")[-1]})
    raise AssertionError(f"unexpected url {url}")


def test_survey_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(survey.mly.requests, "get", fake_get)
    out, raw = str(tmp_path / "survey"), str(tmp_path / "raw")

    summary = survey.run_survey(
        out, raw, per_cell=2, seed=1, max_samples=3,
        locations=survey.LOCATIONS[:2], epochs=survey.EPOCHS, token=TOKEN, pause=0,
    )

    assert summary["search_strategy"].startswith("all images of the zoom-14 coverage vector tile")
    assert [p["ok"] for p in summary["api_probes"]] == [True, False, True, True]
    assert "reduce the amount of data" in summary["api_probes"][1]["error"]

    cells = pd.read_csv(os.path.join(out, "cells.csv"), keep_default_na=False)
    assert len(cells) == 2 * len(survey.EPOCHS)
    found = dict(zip(cells["epoch"][:len(survey.EPOCHS)], cells["images_found"][:len(survey.EPOCHS)]))
    assert found["2014-2015"] == 2 and found["2016-2017"] == 2 and found["2021"] == 1 and found["2022"] == 0

    df = pd.read_csv(os.path.join(out, "segmentation_availability.csv"), keep_default_na=False)
    assert len(df) == summary["images_sampled"] == 2 * (2 + 2 + 1 + 1 + 1 + 1)
    assert summary["requests_failed"] == 2  # the 2018 images
    assert summary["images_with_detections"] == summary["images_with_surface_classes"] == 2 * 6  # 2014, 2015, 2017, 2019, 2020, 2021
    assert summary["sky_above_road"] == {"True": 12}

    report = open(os.path.join(out, "SEGMENTATION_AVAILABILITY.md"), encoding="utf-8").read()
    for section in ("## API probes", "## By capture epoch", "## By continent", "samples/"):
        assert section in report
    assert len(os.listdir(os.path.join(out, "samples"))) == 3
    assert len(os.listdir(os.path.join(raw, "detections"))) == 12

    # the token never reaches the outputs
    for folder in (out, raw):
        for root, _, files in os.walk(folder):
            for name in files:
                with open(os.path.join(root, name), "rb") as f:
                    assert b"secret" not in f.read()


def test_survey_aborts_when_tiles_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(survey.mly.requests, "get", lambda *a, **k: FakeResponse({"error": {"message": "boom"}}, 500))
    summary = survey.run_survey(
        str(tmp_path), "", per_cell=2, seed=1, max_samples=0,
        locations=survey.LOCATIONS[:2], epochs=survey.EPOCHS, token=TOKEN, pause=0,
    )
    assert summary["search_strategy"].startswith("none")
    assert summary["images_sampled"] == 0 and len(summary["api_probes"]) == 4
    report = open(os.path.join(tmp_path, "SEGMENTATION_AVAILABILITY.md"), encoding="utf-8").read()
    assert "## By capture epoch\n\n_No data._" in report
