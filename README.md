# My Mapillary API

A lightweight Python library for accessing the Mapillary API speaking GeoDataframes directly.

## Features

- 🗺️ Fetch street-level imagery metadata from Mapillary
- 📍 Query by bounding box or place name
- 🧩 Tiled querying for large areas
- 📊 Export data to GeoJSON and other formats
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
