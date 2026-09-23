# My Mapillary API

A lightweight Python library for accessing the Mapillary API speaking GeoDataframes directly.

## Features

- 🗺️ Fetch street-level imagery metadata from Mapillary
- 📍 Query by bounding box or place name
- 🧩 Tiled querying for large areas
- 📊 Export data to GeoJSON and other formats
- 🎨 Semantic segmentation masks computed by Mapillary (decoded from the API's detections)
- 🌍 Interactive map visualization with Folium
- 🤖 Automated examples with GitHub Actions

## Installation

```bash
pip install -r requirements.txt
```

For interactive map visualizations, also install:
```bash
pip install folium
```

## Quick Start

### Setup API Token

Set your Mapillary API token as an environment variable:

```bash
export API_TOKEN="your_mapillary_api_token_here"
# or
export MAPPILLARY_API_TOKEN="your_mapillary_api_token_here"
```

Alternatively, create a file named `mapillary_token` with your token.

### Basic Usage

```python
from mapillary_api import *

# Query images in a bounding box (minLon, minLat, maxLon, maxLat)
metadata = get_mapillary_images_metadata(-49.28, -25.44, -49.27, -25.43)

# Convert to GeoDataFrame
gdf = mapillary_data_to_gdf(metadata)

# Save to file
gdf.to_file("mapillary_data.geojson")
```

### Semantic Segmentation Masks

Mapillary runs semantic segmentation on its servers and serves it per image as
*detections*: one entry per region, with the class in `value` (e.g.
`construction--flat--road`, `nature--sky`, `object--vehicle--car`) and the
region polygon in `geometry`, encoded as a base64 Mapbox Vector Tile. The
library decodes them into polygons or label masks:

```python
detections = get_image_detections(image_id)

# Polygons in pixel coordinates (origin at the top-left corner)
detections_gdf = detections_to_gdf(detections, width=2048, height=1536)

# Label mask (uint16, 0 = unlabeled) and its {class: label} mapping
mask, class_index = detections_to_mask(detections, 2048, 1536)
rgb = colorize_mask(mask, class_index)

# Bulk download for all images of a GeoDataFrame (16-bit PNG masks + class legends)
download_segmentation_masks_from_gdf(gdf, "masks", scale_factor=0.25)
```

For a quick overview of many images, add a `detections_summary` column to the
images GeoDataFrame (one dict per photo). It is computed directly from the
encoded polygons, without building any geometry:

```python
gdf = mapillary_data_to_gdf(metadata, detections_summary=True)
# or, for an existing images GeoDataFrame:
gdf = add_detections_summary(gdf)

gdf["detections_summary"].iloc[0]
# {'present': True,
#  'number_available_classes': 21,
#  'class_percents': {'construction--flat--road': 31.2, 'nature--sky': 18.7, ...}}
```

`class_percents` gives the share of the image (in %) covered by each class;
images whose detections could not be requested get `None`. When saved to a
file, the column is written as JSON text.

#### Availability (survey of September 2026)

The feature is **not discontinued**: Mapillary still segments new uploads, and
the newest detection found was created on 2026-09-17. The
[availability survey](survey/SEGMENTATION_AVAILABILITY.md) sampled 865 images
in 32 cities on every inhabited continent, captured from 2014 to 2026:

| Capture epoch | Images with detections | Images with full-scene segmentation |
|:--|--:|--:|
| 2014 – 2020 | 64 – 74% | 15 – 24% |
| 2021 | 68% | 67% |
| 2022 – 2026 | 100% | 100% (median ~22 classes per image) |

- Images captured since 2022 all have a full semantic segmentation (road,
  sidewalk, building, sky, vegetation, vehicles, poles, signs, ...), usually
  produced within days of the capture (median: 1 day).
- Most older images only kept a few object, road-marking or traffic-sign
  detections, created during the 2021 migration to API v4. Some of them
  (about 1 in 5) were re-processed in 2023 – 2024 and have a full segmentation.
- All 173,660 detection geometries decoded without errors, and the masks are
  upright (the sky was above the road in all 451 testable images).
- The `/images` bbox search refuses busy areas even for tiny requests
  (`fields=id`, `limit=10`, a ±0.01° box: *"Please reduce the amount of data
  you're asking for"*). Use small boxes (±0.001° works),
  `tiled_mapillary_data_to_gdf`, or `get_coverage_tile_images`.

The survey can be re-run with the *Segmentation Availability Survey* GitHub
Action (manual trigger), which runs
[`scripts/segmentation_survey.py`](scripts/segmentation_survey.py) and commits
its report to [`survey/`](survey/).

### Interactive Examples

Check out the **[examples.ipynb](examples.ipynb)** notebook for comprehensive examples including:

- 📍 Querying images by geographic area
- 🗺️ Interactive map visualization with Folium
- 🏘️ Territory-based queries using place names
- 📊 Data exploration and analysis

The examples notebook is automatically updated via GitHub Actions to ensure fresh data and working code.

## API Reference

### Core Functions

- `get_mapillary_images_metadata(minLon, minLat, maxLon, maxLat, ...)` - Fetch image metadata for a bounding box
- `mapillary_data_to_gdf(data, ...)` - Convert API response to GeoDataFrame
- `get_territory_polygon(place_name, ...)` - Get polygon for a named place
- `tiled_mapillary_data_to_gdf(polygon, ...)` - Query large areas using tiles
- `download_all_pictures_from_gdf(gdf, folder, ...)` - Download actual images
- `get_coverage_tile_images(tile, ...)` - List every image of a zoom-14 coverage vector tile (with capture dates), for areas where the bbox search refuses to answer

### Semantic Segmentation Functions

- `get_image_detections(image_id, ...)` - Fetch the detections (segmented regions) of an image
- `decode_detection_geometry(geometry, width, height)` - Decode a detection's base64 vector tile into a polygon
- `detections_to_gdf(detections, width, height)` - Detections as a GeoDataFrame of image-space polygons
- `add_detections_summary(gdf, ...)` - Add a `detections_summary` column (presence, number of classes, % of the image per class) to an images GeoDataFrame
- `detections_summary(detections)` - The summary dict of one image's detections, without building geometries
- `detections_to_mask(detections, width, height)` - Rasterize detections into a label mask
- `colorize_mask(mask, class_index)` - RGB visualization of a label mask
- `download_segmentation_masks_from_gdf(gdf, folder, ...)` - Download masks for all images of a GeoDataFrame

### Utility Functions

- `get_mapillary_token()` - Discover API token from environment or file
- `get_bounding_box(lon, lat, radius)` - Create bounding box around point
- `radius_to_degrees(radius, lat)` - Convert radius in meters to degrees

## GitHub Actions Integration

This repository includes automated GitHub Actions that:

- 🔄 Execute the examples notebook automatically
- 📊 Update visualizations with fresh data
- ✅ Ensure code examples stay working
- 📈 Run weekly to keep content current

### Setting up the API Token Secret

1. Go to your repository Settings
2. Navigate to Secrets and variables → Actions
3. Add a new repository secret named `API_TOKEN`
4. Paste your Mapillary API token as the value

The workflow will automatically use this secret when running the examples.

## Dependencies

Core dependencies (see [requirements.txt](requirements.txt)):
- `geopandas` - Geospatial data handling
- `requests` - HTTP requests to Mapillary API
- `wget` - File downloads
- `mercantile` - Map tile utilities
- `tqdm` - Progress bars
- `pillow`, `numpy` - Image handling and masks
- `mapbox-vector-tile` - Decoding of the detection geometries

Optional dependencies:
- `folium` - Interactive map visualizations (not in requirements.txt)
- `jupyter` - For running notebooks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Test with the examples notebook
5. Submit a pull request

## License

See [LICENSE](LICENSE) file for details.
