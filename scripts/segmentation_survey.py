"""
Survey of the availability of Mapillary's server-side semantic segmentations
(image detections) across the globe and across capture epochs.

For every (location, epoch) cell, a few images captured in that epoch are
sampled near the location, their detections are fetched and decoded, and the
results are summarized in a CSV, a JSON summary and a Markdown report, along
with a few sample overlays (image + colorized mask).

Usage:
    python scripts/segmentation_survey.py --out survey --raw-out survey_raw

The API token is read like in mapillary_api (API_TOKEN environment variable,
etc.) and is never written to the outputs.
"""

import argparse
import io
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from time import sleep

import numpy as np
import pandas as pd
import requests
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mapillary_api as mly  # noqa: E402

# (name, continent, lon, lat)
LOCATIONS = [
    ("Curitiba", "South America", -49.2733, -25.4284),
    ("São Paulo", "South America", -46.6333, -23.5505),
    ("Buenos Aires", "South America", -58.3816, -34.6037),
    ("Bogotá", "South America", -74.0721, 4.7110),
    ("Lima", "South America", -77.0428, -12.0464),
    ("Mexico City", "North America", -99.1332, 19.4326),
    ("San Francisco", "North America", -122.4194, 37.7749),
    ("New York", "North America", -73.9857, 40.7484),
    ("Toronto", "North America", -79.3832, 43.6532),
    ("London", "Europe", -0.1276, 51.5072),
    ("Paris", "Europe", 2.3522, 48.8566),
    ("Berlin", "Europe", 13.4050, 52.5200),
    ("Madrid", "Europe", -3.7038, 40.4168),
    ("Stockholm", "Europe", 18.0686, 59.3293),
    ("Warsaw", "Europe", 21.0122, 52.2297),
    ("Rome", "Europe", 12.4964, 41.9028),
    ("Istanbul", "Europe", 28.9784, 41.0082),
    ("Cairo", "Africa", 31.2357, 30.0444),
    ("Casablanca", "Africa", -7.5898, 33.5731),
    ("Lagos", "Africa", 3.3792, 6.5244),
    ("Nairobi", "Africa", 36.8219, -1.2921),
    ("Johannesburg", "Africa", 28.0473, -26.2041),
    ("Dubai", "Asia", 55.2708, 25.2048),
    ("Mumbai", "Asia", 72.8777, 19.0760),
    ("Bangkok", "Asia", 100.5018, 13.7563),
    ("Jakarta", "Asia", 106.8456, -6.2088),
    ("Singapore", "Asia", 103.8198, 1.3521),
    ("Manila", "Asia", 120.9842, 14.5995),
    ("Seoul", "Asia", 126.9780, 37.5665),
    ("Tokyo", "Asia", 139.6917, 35.6895),
    ("Sydney", "Oceania", 151.2093, -33.8688),
    ("Auckland", "Oceania", 174.7633, -36.8485),
]

# (label, start_captured_at, end_captured_at)
EPOCHS = [("2014-2015", "2014-01-01", "2016-01-01"), ("2016-2017", "2016-01-01", "2018-01-01")] + [
    (str(year), f"{year}-01-01", f"{year + 1}-01-01") for year in range(2018, 2027)
]

IMAGE_FIELDS = ["id", "captured_at", "geometry", "width", "height", "is_pano", "sequence", "thumb_1024_url"]

# bbox half-sizes (degrees) tried in order until images are found
SEARCH_RADII = [0.01, 0.03]


def iso_date(value):
    """Convert a Mapillary timestamp (epoch milliseconds or ISO string) to an ISO date string."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    return str(value)[:10]


def pick_images(images, k, rng):
    """Randomly pick up to k images, preferring distinct sequences."""
    images = list(images)
    rng.shuffle(images)
    picked, seen = [], set()
    for image in images:
        if image.get("sequence") not in seen:
            picked.append(image)
            seen.add(image.get("sequence"))
        if len(picked) == k:
            return picked
    for image in images:
        if image not in picked:
            picked.append(image)
        if len(picked) == k:
            break
    return picked


def search_images(lon, lat, start, end, token):
    """Search images captured in [start, end) around a location, widening the bbox if needed."""
    for radius in SEARCH_RADII:
        data = mly.get_mapillary_images_metadata(
            round(lon - radius, 6),
            round(lat - radius, 6),
            round(lon + radius, 6),
            round(lat + radius, 6),
            fields=IMAGE_FIELDS,
            token=token,
            limit=100,
            timeout=120,
            start_captured_at=f"{start}T00:00:00Z",
            end_captured_at=f"{end}T00:00:00Z",
        )
        images = data.get("data", [])
        if images:
            return images, radius
    return [], SEARCH_RADII[-1]


def orientation_check(gdf):
    """True if the sky lies above the road (as expected), False if below, None if not testable."""
    sky = gdf[gdf["value"] == "nature--sky"]
    road = gdf[gdf["value"] == "construction--flat--road"]
    if sky.empty or road.empty:
        return None
    return bool(sky.geometry.union_all().centroid.y < road.geometry.union_all().centroid.y)


def analyze_image(image, token, raw_out):
    """Fetch and analyze the detections of one image. Returns (record, detections)."""
    record = {
        "image_id": image.get("id"),
        "captured_at": iso_date(image.get("captured_at")),
        "is_pano": image.get("is_pano"),
        "width": image.get("width"),
        "height": image.get("height"),
        "n_detections": 0,
        "n_classes": 0,
        "n_surface": 0,
        "n_object": 0,
        "n_traffic_sign": 0,
        "n_other": 0,
        "has_surface": False,
        "detections_created_min": None,
        "detections_created_max": None,
        "decode_failures": 0,
        "sky_above_road": None,
        "top_classes": "",
        "error": "",
    }
    try:
        detections = mly.get_image_detections(image["id"], token=token)
    except Exception as e:
        record["error"] = mly.redact_token(e, token)[:300]
        return record, []

    record["n_detections"] = len(detections)
    if not detections:
        return record, []

    if raw_out:
        mly.dump_json(detections, os.path.join(raw_out, "detections", f"{image['id']}.json"))

    gdf = mly.detections_to_gdf(detections)
    record["decode_failures"] = len([d for d in detections if d.get("geometry")]) - len(gdf)

    values = [d.get("value") for d in detections]
    groups = Counter(mly.detection_class_group(v) for v in values)
    for group in ("surface", "object", "traffic_sign", "other"):
        record[f"n_{group}"] = groups.get(group, 0)
    record["n_classes"] = len(set(values))
    record["has_surface"] = groups.get("surface", 0) > 0
    record["top_classes"] = ";".join(v for v, _ in Counter(values).most_common(8))

    created = sorted(str(d["created_at"]) for d in detections if d.get("created_at"))
    if created:
        record["detections_created_min"] = created[0][:10]
        record["detections_created_max"] = created[-1][:10]

    if not gdf.empty:
        record["sky_above_road"] = orientation_check(gdf)

    return record, detections


def render_sample(image, detections, outpath, token):
    """Save the image thumbnail next to its colorized segmentation overlay."""
    url = image.get("thumb_1024_url")
    if not url:
        return False
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    thumb = Image.open(io.BytesIO(response.content)).convert("RGB")

    mask, class_index = mly.detections_to_mask(detections, thumb.width, thumb.height)
    color = Image.fromarray(mly.colorize_mask(mask, class_index))
    overlay = Image.blend(thumb, color, 0.55)
    overlay.paste(thumb, mask=Image.fromarray(np.where(mask == 0, 255, 0).astype(np.uint8)))

    side = Image.new("RGB", (thumb.width * 2, thumb.height))
    side.paste(thumb, (0, 0))
    side.paste(overlay, (thumb.width, 0))
    side.thumbnail((1280, 1280))
    side.save(outpath, quality=80)
    return True


def percent(series):
    return round(100 * series.mean(), 1) if len(series) else None


def availability_table(df, by):
    """Availability statistics grouped by a column. Percentages exclude failed requests."""

    def stats(group):
        ok = group[group["error"] == ""]
        created = ok["detections_created_max"].dropna()
        return pd.Series(
            {
                "Images": len(ok),
                "Failed requests": int((group["error"] != "").sum()),
                "% with detections": percent(ok["n_detections"] > 0),
                "% with full-scene classes": percent(ok["has_surface"].astype(bool)),
                "Median classes": ok["n_classes"].median() if len(ok) else None,
                "Latest detection created_at": created.max() if len(created) else "-",
            }
        )

    return df.groupby(by, sort=False, observed=True)[list(df.columns)].apply(stats)


def write_report(df, cells, samples, out, started, finished):
    ok = df[df["error"] == ""]
    epoch_order = [e[0] for e in EPOCHS]

    by_epoch = availability_table(df.assign(epoch=pd.Categorical(df["epoch"], epoch_order, ordered=True)).sort_values("epoch"), "epoch")
    by_continent = availability_table(df, "continent")

    epochs_run = [e for e in epoch_order if e in set(cells["epoch"])]
    matrix = (
        ok.assign(avail=ok["n_detections"] > 0)
        .pivot_table(index="location", columns="epoch", values="avail", aggfunc="mean")
        .reindex(index=list(dict.fromkeys(cells["location"])), columns=epochs_run)
        .map(lambda v: None if pd.isna(v) else f"{round(100 * v)}%")
    )
    # distinguish cells without imagery from cells whose requests failed
    for cell in cells.itertuples():
        if matrix.at[cell.location, cell.epoch] is None:
            matrix.at[cell.location, cell.epoch] = "err" if cell.error or cell.images_found else "·"

    processing_years = (
        ok["detections_created_max"].dropna().str[:4].value_counts().sort_index().rename("Images")
    )
    processing_years.index.name = "Year"
    orientation = ok["sky_above_road"].value_counts(dropna=True)

    summary = {
        "started": started,
        "finished": finished,
        "cells": len(cells),
        "cells_with_images": int((cells["images_found"] > 0).sum()),
        "images_sampled": int(len(df)),
        "requests_failed": int((df["error"] != "").sum()),
        "images_with_detections": int((ok["n_detections"] > 0).sum()),
        "images_with_surface_classes": int(ok["has_surface"].sum()),
        "decode_failures": int(ok["decode_failures"].sum()),
        "latest_detection_created_at": ok["detections_created_max"].dropna().max() if ok["detections_created_max"].notna().any() else None,
        "sky_above_road": {str(k): int(v) for k, v in orientation.items()},
        "by_epoch": json.loads(by_epoch.to_json(orient="index")),
        "by_continent": json.loads(by_continent.to_json(orient="index")),
    }
    mly.dump_json(summary, os.path.join(out, "summary.json"))

    errors = df.loc[df["error"] != "", "error"].str[:120].value_counts().head(10)
    cell_errors = cells.loc[cells["error"] != "", "error"].str[:120].value_counts().head(10)

    lines = [
        "# Mapillary semantic segmentation availability survey",
        "",
        f"Run: {started} → {finished} (UTC). Generated by `scripts/segmentation_survey.py`.",
        "",
        "For each location × capture epoch, up to a few images captured in that epoch were sampled near the "
        "location (bbox ±0.01°, widened to ±0.03° when empty) and their detections were requested from "
        "`https://graph.mapillary.com/{image_id}/detections`.",
        "",
        "- **% with detections**: images for which the API returned at least one detection.",
        "- **% with full-scene classes**: images with at least one *surface* class "
        "(`construction--*`, `nature--*`, `marking--*`, `void--*`), i.e. a real semantic segmentation "
        "rather than only object or traffic-sign detections.",
        "- **Latest detection created_at**: when Mapillary produced the newest detection of the group "
        "(processing date, not capture date).",
        "",
        "## Overview",
        "",
        f"- Cells (location × epoch): {summary['cells']}, with imagery: {summary['cells_with_images']}",
        f"- Images sampled: {summary['images_sampled']} (failed requests: {summary['requests_failed']})",
        f"- Images with detections: {summary['images_with_detections']}",
        f"- Images with full-scene segmentation classes: {summary['images_with_surface_classes']}",
        f"- Geometry decode failures: {summary['decode_failures']}",
        f"- Newest detection created_at: {summary['latest_detection_created_at']}",
        f"- Orientation check (sky centroid above road centroid): {summary['sky_above_road'] or 'not testable'}",
        "",
        "## By capture epoch",
        "",
        by_epoch.to_markdown(),
        "",
        "## By continent",
        "",
        by_continent.to_markdown(),
        "",
        "## Detection processing year (newest detection per image)",
        "",
        processing_years.to_frame().to_markdown() if not processing_years.empty else "_No detections._",
        "",
        "## Share of sampled images with detections, per location and epoch",
        "",
        "`·` = no imagery found for that epoch, `err` = requests failed.",
        "",
        matrix.to_markdown() if not matrix.empty else "_No data._",
        "",
    ]
    if len(errors) or len(cell_errors):
        lines += ["## Errors", ""]
        lines += [f"- {n}× image request: `{e}`" for e, n in errors.items()]
        lines += [f"- {n}× image search: `{e}`" for e, n in cell_errors.items()]
        lines += [""]
    if samples:
        lines += ["## Samples", "", "Left: image thumbnail. Right: decoded segmentation overlay.", ""]
        for sample in samples:
            lines += [
                f"**{sample['location']}, captured {sample['captured_at']} (detections created {sample['created']})**, "
                f"image `{sample['image_id']}`, {sample['n_classes']} classes",
                "",
                f"![{sample['image_id']}](samples/{os.path.basename(sample['path'])})",
                "",
            ]

    with open(os.path.join(out, "SEGMENTATION_AVAILABILITY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return summary


def run_survey(out, raw_out, per_cell, seed, max_samples, locations, epochs, token, pause=0.1):
    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    rng = random.Random(seed)
    mly.create_dir_if_not_exists(os.path.join(out, "samples"))
    if raw_out:
        mly.create_dir_if_not_exists(os.path.join(raw_out, "detections"))

    records, cells = [], []
    candidates = {}  # epoch -> list of (image, detections, record) with full-scene classes

    for name, continent, lon, lat in locations:
        for label, start, end in epochs:
            cell = {"location": name, "continent": continent, "epoch": label, "images_found": 0, "radius": None, "error": ""}
            try:
                images, radius = search_images(lon, lat, start, end, token)
                cell.update(images_found=len(images), radius=radius)
            except Exception as e:
                cell["error"] = mly.redact_token(e, token)[:300]
                images = []
            cells.append(cell)
            print(f"{name:14s} {label:9s} images={cell['images_found']:3d} {cell['error'][:80]}", flush=True)

            for image in pick_images(images, per_cell, rng):
                record, detections = analyze_image(image, token, raw_out)
                record.update(location=name, continent=continent, epoch=label, lon=lon, lat=lat)
                records.append(record)
                if record["has_surface"]:
                    candidates.setdefault(label, []).append((image, detections, record))
                sleep(pause)
            sleep(pause)

    columns = ["location", "continent", "epoch", "lon", "lat"]
    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=columns + ["image_id", "n_detections", "n_classes", "has_surface", "detections_created_max", "decode_failures", "sky_above_road", "error"])
    df = df[columns + [c for c in df.columns if c not in columns]]
    df.to_csv(os.path.join(out, "segmentation_availability.csv"), index=False)
    cells = pd.DataFrame(cells, columns=["location", "continent", "epoch", "images_found", "radius", "error"])
    cells.to_csv(os.path.join(out, "cells.csv"), index=False)

    # one sample per epoch first, then round-robin over the remaining candidates
    samples, queue = [], [list(v) for _, v in sorted(candidates.items())]
    for items in queue:
        rng.shuffle(items)
    while len(samples) < max_samples and any(queue):
        for items in queue:
            if items and len(samples) < max_samples:
                image, detections, record = items.pop()
                path = os.path.join(out, "samples", f"{len(samples) + 1:02d}_{record['epoch']}_{record['image_id']}.jpg")
                try:
                    if render_sample(image, detections, path, token):
                        samples.append(
                            {
                                "path": path,
                                "location": record["location"],
                                "captured_at": record["captured_at"],
                                "created": record["detections_created_max"],
                                "image_id": record["image_id"],
                                "n_classes": record["n_classes"],
                            }
                        )
                except Exception as e:
                    print(f"⚠️  Could not render sample {record['image_id']}: {mly.redact_token(e, token)}")

    finished = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    return write_report(df, cells, samples, out, started, finished)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="survey", help="Output folder for the report, CSVs and samples")
    parser.add_argument("--raw-out", default="survey_raw", help="Output folder for raw detections ('' to skip)")
    parser.add_argument("--per-cell", type=int, default=3, help="Images sampled per location × epoch")
    parser.add_argument("--samples", type=int, default=12, help="Maximum number of rendered sample overlays")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--locations", default="", help="Comma-separated subset of location names")
    parser.add_argument("--epochs", default="", help="Comma-separated subset of epoch labels")
    args = parser.parse_args(argv)

    token = mly.get_mapillary_token()
    if not token:
        parser.error("No Mapillary API token found (set the API_TOKEN environment variable)")

    locations = [l for l in LOCATIONS if not args.locations or l[0] in args.locations.split(",")]
    epochs = [e for e in EPOCHS if not args.epochs or e[0] in args.epochs.split(",")]

    summary = run_survey(args.out, args.raw_out, args.per_cell, args.seed, args.samples, locations, epochs, token)
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("by_")}, indent=2))


if __name__ == "__main__":
    main()
