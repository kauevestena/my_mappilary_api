"""
Offline tests for the semantic segmentation (detections) helpers.

Detection geometries are synthesized with mapbox_vector_tile.encode, so no
network access or API token is needed.
"""

import base64
import os
import sys

import numpy as np
import pytest
import requests
import mapbox_vector_tile
from shapely.geometry import Polygon, box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mapillary_api as mly  # noqa: E402

EXTENT = 4096
TOKEN = "MLY|123|secret"


def encode_geometry(*geoms, layer="mpy-or"):
    """Encode polygons given in tile coordinates (y down) as a detection geometry."""
    tile = mapbox_vector_tile.encode(
        [{"name": layer, "features": [{"geometry": g.wkt, "properties": {}} for g in geoms]}],
        default_options={"y_coord_down": True, "extents": EXTENT},
    )
    return base64.b64encode(tile).decode("ascii")


def detection(value, *geoms, id_="1"):
    return {"id": id_, "value": value, "geometry": encode_geometry(*geoms), "created_at": "2021-01-01T00:00:00+0000"}


SKY = box(0, 0, EXTENT, EXTENT // 4)  # top quarter
ROAD = box(0, 3 * EXTENT // 4, EXTENT, EXTENT)  # bottom quarter


def test_decode_normalized_and_scaled():
    geom = mly.decode_detection_geometry(encode_geometry(SKY))
    minx, miny, maxx, maxy = geom.bounds
    assert (minx, miny, maxx, maxy) == pytest.approx((0, 0, 1, 0.25))

    geom = mly.decode_detection_geometry(encode_geometry(SKY), width=2000, height=1000)
    assert geom.bounds == pytest.approx((0, 0, 2000, 250))


def test_decode_keeps_y_axis_pointing_down():
    sky = mly.decode_detection_geometry(encode_geometry(SKY), 100, 100)
    road = mly.decode_detection_geometry(encode_geometry(ROAD), 100, 100)
    assert sky.centroid.y < road.centroid.y


def test_decode_all_features_and_layers():
    a, b = box(0, 0, 100, 100), box(1000, 1000, 1100, 1100)
    geom = mly.decode_detection_geometry(encode_geometry(a, b), EXTENT, EXTENT)
    assert geom.area == pytest.approx(a.area + b.area)

    tile = mapbox_vector_tile.encode(
        [
            {"name": "l1", "features": [{"geometry": a.wkt, "properties": {}}]},
            {"name": "l2", "features": [{"geometry": b.wkt, "properties": {}}]},
        ],
        default_options={"y_coord_down": True},
    )
    geom = mly.decode_detection_geometry(base64.b64encode(tile).decode(), EXTENT, EXTENT)
    assert geom.area == pytest.approx(a.area + b.area)


def test_decode_invalid_geometry_raises():
    with pytest.raises(ValueError):
        mly.decode_detection_geometry("not a vector tile")


def test_detections_to_gdf():
    gdf = mly.detections_to_gdf(
        [detection("nature--sky", SKY, id_="1"), detection("regulatory--stop--g1", box(0, 0, 10, 10), id_="2")],
        width=400,
        height=300,
    )
    assert list(gdf["value"]) == ["nature--sky", "regulatory--stop--g1"]
    assert list(gdf["group"]) == ["surface", "traffic_sign"]
    assert gdf.crs is None
    assert gdf.geometry.iloc[0].bounds == pytest.approx((0, 0, 400, 75))

    assert mly.detections_to_gdf([]).empty


def test_detections_to_mask_labels_and_order():
    car = box(1000, 3200, 2000, 3800)  # small object on top of the road
    detections = [
        detection("construction--flat--road", ROAD),
        detection("nature--sky", SKY),
        detection("object--vehicle--car", car),
    ]
    mask, classes = mly.detections_to_mask(detections, 64, 64)

    assert mask.shape == (64, 64) and mask.dtype == np.uint16
    assert set(classes) == {"construction--flat--road", "nature--sky", "object--vehicle--car"}
    assert mask[2, 32] == classes["nature--sky"]
    assert mask[62, 5] == classes["construction--flat--road"]
    assert mask[54, 23] == classes["object--vehicle--car"]  # not hidden by the road
    assert mask[32, 32] == 0  # middle band is unlabeled


def test_detections_to_mask_holes_and_class_index():
    ring = Polygon(
        [(0, 0), (EXTENT, 0), (EXTENT, EXTENT), (0, EXTENT)],
        [[(1024, 1024), (3072, 1024), (3072, 3072), (1024, 3072)]],
    )
    building = box(1500, 1500, 2500, 2500)  # inside the hole
    mask, classes = mly.detections_to_mask(
        [detection("nature--vegetation", ring), detection("construction--structure--building", building)],
        64,
        64,
        class_index={"nature--sky": 1},
    )
    assert classes["nature--sky"] == 1  # existing labels are kept
    assert mask[2, 2] == classes["nature--vegetation"]
    assert mask[20, 20] == 0  # hole, not covered by the building
    assert mask[32, 32] == classes["construction--structure--building"]


def test_colorize_mask():
    mask = np.array([[0, 1], [2, 1]], dtype=np.uint16)
    rgb = mly.colorize_mask(mask, {"nature--sky": 1, "construction--flat--road": 2})
    assert rgb.shape == (2, 2, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[0, 0]) == (0, 0, 0)
    assert tuple(rgb[0, 1]) == mly.class_color("nature--sky")


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} Error for url: https://graph.mapillary.com/1?access_token={requests.utils.quote(TOKEN, safe='')}",
                response=self,
            )


def test_get_image_detections_follows_paging(monkeypatch):
    calls = []
    pages = [
        {"data": [{"id": "a"}], "paging": {"next": f"https://graph.mapillary.com/next?access_token={TOKEN}"}},
        {"data": [{"id": "b"}]},
    ]

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        return FakeResponse(pages[len(calls) - 1])

    monkeypatch.setattr(mly.requests, "get", fake_get)
    detections = mly.get_image_detections("123", token=TOKEN)

    assert [d["id"] for d in detections] == ["a", "b"]
    assert calls[0][0] == "https://graph.mapillary.com/123/detections"
    assert calls[0][1]["access_token"] == TOKEN
    assert "access_token" not in calls[1][1]


def test_errors_do_not_leak_token(monkeypatch):
    monkeypatch.setattr(mly.requests, "get", lambda *a, **k: FakeResponse(None, 500))
    with pytest.raises(requests.exceptions.RequestException) as excinfo:
        mly.get_image_detections("1", token=TOKEN)
    assert "secret" not in str(excinfo.value)

    error = {"error": {"message": f"Invalid OAuth access token {TOKEN}"}}
    monkeypatch.setattr(mly.requests, "get", lambda *a, **k: FakeResponse(error, 400))
    with pytest.raises(ValueError) as excinfo:
        mly.get_image_detections("1", token=TOKEN)
    assert "secret" not in str(excinfo.value)
    assert "Invalid OAuth access token" in str(excinfo.value)


def test_detection_class_group():
    assert mly.detection_class_group("construction--flat--road") == "surface"
    assert mly.detection_class_group("marking--discrete--crosswalk-zebra") == "marking"
    assert mly.detection_class_group("human--person--individual") == "object"
    assert mly.detection_class_group("warning--pedestrians-crossing--g4") == "traffic_sign"
    assert mly.detection_class_group("something-new") == "other"


def test_download_segmentation_masks_from_gdf(tmp_path, monkeypatch):
    import geopandas as gpd
    from PIL import Image

    def fake_get(url, params=None, timeout=None):
        if url.endswith("/empty/detections"):
            return FakeResponse({"data": []})
        if url.endswith("/nosize"):
            return FakeResponse({"width": 80, "height": 60})
        return FakeResponse({"data": [detection("nature--sky", SKY), detection("construction--flat--road", ROAD)]})

    monkeypatch.setattr(mly.requests, "get", fake_get)
    gdf = gpd.GeoDataFrame({"id": ["full", "empty", "nosize"], "width": [400, 400, None], "height": [300, 300, None]})

    summary = mly.download_segmentation_masks_from_gdf(
        gdf, str(tmp_path), token=TOKEN, scale_factor=0.5, save_colorized=True
    )
    assert (summary["success"], summary["empty"], summary["failed"]) == (2, 1, 0)

    mask = np.array(Image.open(tmp_path / "full_mask.png"))
    assert mask.shape == (150, 200)
    classes = mly.read_json(str(tmp_path / "full_classes.json"))
    assert mask[0, 0] == classes["nature--sky"] and mask[-1, 0] == classes["construction--flat--road"]
    assert (tmp_path / "full_detections.json").exists() and (tmp_path / "full_mask_color.png").exists()
    assert np.array(Image.open(tmp_path / "nosize_mask.png")).shape == (30, 40)
    assert not (tmp_path / "empty_mask.png").exists()


class FakeTileResponse:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Error for url: x?access_token={TOKEN}")


def encode_coverage_tile(images):
    """Encode (px, py, properties) tuples as the 'image' layer of a coverage tile."""
    from shapely.geometry import Point

    return mapbox_vector_tile.encode(
        [{"name": "image", "features": [{"geometry": Point(px, py).wkt, "properties": props} for px, py, props in images]}],
        default_options={"y_coord_down": True, "extents": EXTENT},
    )


def test_get_coverage_tile_images(monkeypatch):
    import gzip
    import mercantile

    tile = mercantile.tile(-49.2733, -25.4284, 14)
    content = encode_coverage_tile(
        [
            (0, 0, {"id": 1, "captured_at": 1600000000000, "sequence_id": "a"}),
            (EXTENT // 2, EXTENT // 2, {"id": 2, "captured_at": 1400000000000, "sequence_id": "b"}),
        ]
    )
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        return FakeTileResponse(gzip.compress(content))

    monkeypatch.setattr(mly.requests, "get", fake_get)
    images = mly.get_coverage_tile_images(tile, token=TOKEN)

    assert calls[0][0] == f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/14/{tile.x}/{tile.y}"
    assert [i["id"] for i in images] == [1, 2]
    assert images[0]["captured_at"] == 1600000000000 and images[1]["sequence_id"] == "b"

    # pixel (0, 0) is the north-west corner of the tile, the center maps to the mercator center
    bounds = mercantile.bounds(tile)
    assert images[0]["geometry"]["coordinates"] == pytest.approx([bounds.west, bounds.north])
    left, bottom, right, top = mercantile.xy_bounds(tile)
    center = mercantile.lnglat((left + right) / 2, (bottom + top) / 2)
    assert images[1]["geometry"]["coordinates"] == pytest.approx([center.lng, center.lat])

    gdf = mly.mapillary_data_to_gdf({"data": images})
    assert len(gdf) == 2 and gdf.crs == "EPSG:4326"

    monkeypatch.setattr(mly.requests, "get", lambda *a, **k: FakeTileResponse(b""))
    assert mly.get_coverage_tile_images(tile, token=TOKEN) == []

    monkeypatch.setattr(mly.requests, "get", lambda *a, **k: FakeTileResponse(b"", 500))
    with pytest.raises(requests.exceptions.RequestException) as excinfo:
        mly.get_coverage_tile_images(tile, token=TOKEN)
    assert "secret" not in str(excinfo.value)


def test_detections_summary():
    summary = mly.detections_summary(
        [
            detection("nature--sky", SKY),
            detection("construction--flat--road", ROAD),
            detection("object--vehicle--car", box(0, 0, 1024, 1024), box(2048, 2048, 3072, 3072)),  # 2 features
            detection("object--vehicle--car", box(1024, 1024, 2048, 2048), id_="2"),  # same class again
        ]
    )
    assert summary["present"] is True
    assert summary["number_available_classes"] == 3
    assert summary["class_percents"] == {
        "nature--sky": 25.0,
        "construction--flat--road": 25.0,
        "object--vehicle--car": 18.75,
    }
    assert list(summary["class_percents"].values()) == sorted(summary["class_percents"].values(), reverse=True)

    # holes are subtracted
    hole = Polygon(
        [(0, 0), (EXTENT, 0), (EXTENT, EXTENT), (0, EXTENT)],
        [[(1024, 1024), (3072, 1024), (3072, 3072), (1024, 3072)]],
    )
    assert mly.detections_summary([detection("nature--vegetation", hole)])["class_percents"] == {"nature--vegetation": 75.0}

    assert mly.detections_summary([]) == {"present": False, "number_available_classes": 0, "class_percents": {}}


def test_detections_summary_matches_mask():
    detections = [
        detection("nature--sky", Polygon([(0, 0), (4096, 0), (4096, 900), (2000, 1400), (0, 1100)])),
        detection("construction--flat--road", Polygon([(0, 4096), (4096, 4096), (2500, 2600), (1500, 2600)])),
    ]
    mask, classes = mly.detections_to_mask(detections, 256, 256)
    percents = mly.detections_summary(detections)["class_percents"]
    for value, label in classes.items():
        assert percents[value] == pytest.approx(100 * (mask == label).mean(), abs=1)


def test_add_detections_summary_builds_no_geometry(monkeypatch):
    import geopandas as gpd
    from shapely.geometry import Point

    def forbidden(*args, **kwargs):
        raise AssertionError("geometry objects must not be built")

    for name in ("shape", "decode_detection_geometry", "detections_to_gdf", "detections_to_mask"):
        monkeypatch.setattr(mly, name, forbidden)

    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        if url.endswith("/empty/detections"):
            return FakeResponse({"data": []})
        if url.endswith("/broken/detections"):
            return FakeResponse({"error": {"message": f"boom {TOKEN}"}}, 500)
        return FakeResponse({"data": [detection("nature--sky", SKY), detection("construction--flat--road", ROAD)]})

    monkeypatch.setattr(mly.requests, "get", fake_get)
    points = [Point(-49.27, -25.43), Point(-49.26, -25.42), Point(-49.25, -25.41)]
    gdf = gpd.GeoDataFrame({"id": ["full", "empty", "broken"]}, geometry=points, crs="EPSG:4326", index=[0, 0, 1])

    result = mly.add_detections_summary(gdf, token=TOKEN)

    assert result is gdf
    assert list(gdf.geometry) == points  # photo locations are untouched
    full, empty, broken = gdf["detections_summary"]
    assert full == {
        "present": True,
        "number_available_classes": 2,
        "class_percents": {"nature--sky": 25.0, "construction--flat--road": 25.0},
    }
    assert empty == {"present": False, "number_available_classes": 0, "class_percents": {}}
    assert broken is None
    assert all(p["fields"] == "value,geometry" for p in calls)


def test_mapillary_data_to_gdf_with_detections_summary(tmp_path, monkeypatch):
    import json

    import geopandas as gpd

    monkeypatch.setattr(
        mly.requests, "get",
        lambda url, params=None, timeout=None: FakeResponse({"data": [detection("nature--sky", SKY)]}),
    )
    data = {"data": [
        {"id": "1", "geometry": {"type": "Point", "coordinates": [-49.27, -25.43]}},
        {"id": "2", "geometry": {"type": "Point", "coordinates": [-49.26, -25.42]}},
    ]}
    for name in ("images.geojson", "images.gpkg"):
        outpath = str(tmp_path / name)
        gdf = mly.mapillary_data_to_gdf(data, outpath=outpath, detections_summary=True, token=TOKEN)

        assert isinstance(gdf["detections_summary"].iloc[0], dict)  # real dicts in memory
        assert gdf["detections_summary"].iloc[0]["class_percents"] == {"nature--sky": 25.0}

        # saved as JSON text (the GeoJSON reader parses it back into a dict)
        saved = gpd.read_file(outpath)["detections_summary"].iloc[1]
        assert (json.loads(saved) if isinstance(saved, str) else saved) == gdf["detections_summary"].iloc[1]

    # without the flag, no request is made and no column is added
    monkeypatch.setattr(mly.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no request expected")))
    assert "detections_summary" not in mly.mapillary_data_to_gdf(data).columns


def test_detections_summary_uses_plain_floats():
    percents = mly.detections_summary([detection("nature--sky", SKY)])["class_percents"]
    assert type(percents["nature--sky"]) is float
