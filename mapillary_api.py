from math import cos, pi
from time import sleep
import os
import json
import base64
import zlib
import requests
import wget
import geopandas as gpd
import pandas as pd
from shapely import Point, box
from shapely.geometry import Polygon, shape
from shapely.affinity import scale as scale_geometry
import mercantile
import mapbox_vector_tile
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

ZOOM_LEVEL = 18

default_fields = [
    "altitude",
    "atomic_scale",
    "camera_parameters",
    "camera_type",
    "captured_at",
    "compass_angle",
    "computed_altitude",
    "computed_compass_angle",
    "computed_geometry",
    "computed_rotation",
    "creator",
    "exif_orientation",
    "geometry",
    "height",
    "is_pano",
    "make",
    "model",
    "thumb_original_url",
    "merge_cc",
    "sequence",
    "width",
]


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def download_all_pictures_from_gdf(
    gdf, outfolderpath, id_field="id", url_field="thumb_original_url", scale_factor=1.0
):
    """
    Downloads all the pictures from a GeoDataFrame (gdf) and saves them to the
    specified output folder.
    Optionally rescales downloaded images if scale_factor != 1.0.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing the data.
        outfolderpath (str): The path to the output folder where the pictures
            will be saved.
        id_field (str, optional): The name of the field in the GeoDataFrame
            that contains the unique identifier for each picture. Default is 'id'.
        url_field (str, optional): The name of the field in the GeoDataFrame
            that contains the URL of the picture. Default is 'thumb_original_url'.
        scale_factor (float, optional): Scaling factor to resize the images. Default is 1.0.

    Returns:
        dict: Summary of download results with success/failure counts
    """
    if gdf.empty:
        print("⚠️  Warning: Empty GeoDataFrame provided - nothing to download")
        return {"success": 0, "failed": 0, "errors": []}

    # Validate required fields exist
    if id_field not in gdf.columns:
        raise ValueError(f"ID field '{id_field}' not found in GeoDataFrame columns")
    if url_field not in gdf.columns:
        raise ValueError(f"URL field '{url_field}' not found in GeoDataFrame columns")

    # Create output directory if it doesn't exist
    create_dir_if_not_exists(outfolderpath)

    success_count = 0
    failed_count = 0
    errors = []

    for row in tqdm(gdf.itertuples(), total=len(gdf), desc="Downloading images"):
        try:
            image_id = getattr(row, id_field)
            image_url = getattr(row, url_field)

            if not image_url:
                errors.append(f"Empty URL for image ID: {image_id}")
                failed_count += 1
                continue

            outfilepath = os.path.join(outfolderpath, str(image_id) + ".jpg")
            download_mapillary_image(
                image_url,
                outfilepath,
            )
            if scale_factor != 1.0 and os.path.exists(outfilepath):
                with Image.open(outfilepath) as img:
                    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
                    img_resized.save(outfilepath)
            success_count += 1
        except Exception as e:
            error_msg = (
                f"Failed to download image ID {getattr(row, id_field, 'unknown')}: {e}"
            )
            errors.append(error_msg)
            failed_count += 1

    print(f"✅ Download completed: {success_count} successful, {failed_count} failed")
    if errors:
        print(f"❌ Errors encountered: {len(errors)}")
        for error in errors[:5]:  # Show first 5 errors
            print(f"   - {error}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more errors")

    return {"success": success_count, "failed": failed_count, "errors": errors}


def tile_bbox_to_box(tile_bbox, swap_latlon=False):
    if swap_latlon:
        return box(tile_bbox.south, tile_bbox.west, tile_bbox.north, tile_bbox.east)
    else:
        return box(tile_bbox.west, tile_bbox.south, tile_bbox.east, tile_bbox.north)


def tilebboxes_from_bbox(
    minlat, minlon, maxlat, maxlon, zoom=ZOOM_LEVEL, as_list=False
):
    if as_list:
        return [
            list(mercantile.bounds(tile))
            for tile in mercantile.tiles(minlon, minlat, maxlon, maxlat, zoom)
        ]
    else:
        return [
            mercantile.bounds(tile)
            for tile in mercantile.tiles(minlon, minlat, maxlon, maxlat, zoom)
        ]


def check_type_by_first_valid(input_iterable):
    for item in input_iterable:
        if item:
            return type(item)


def selected_columns_to_str(df, desired_type=list):
    for column in df.columns:
        c_type = check_type_by_first_valid(df[column])

        if c_type == desired_type:
            # print(column)
            df[column] = df[column].apply(lambda x: str(x))


def dump_json(data, path):
    """
    Save data as JSON file with error handling.

    Parameters:
        data: Data to save as JSON
        path (str): File path where to save the JSON

    Raises:
        IOError: If file cannot be written
        TypeError: If data cannot be serialized to JSON
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except (IOError, OSError) as e:
        raise IOError(f"Cannot write JSON file '{path}': {e}")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Cannot serialize data to JSON: {e}")


def read_json(path):
    """
    Read JSON file with error handling.

    Parameters:
        path (str): Path to JSON file to read

    Returns:
        Data loaded from JSON file

    Raises:
        IOError: If file cannot be read
        ValueError: If file contains invalid JSON
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (IOError, OSError) as e:
        raise IOError(f"Cannot read JSON file '{path}': {e}")
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Invalid JSON in file '{path}': {e}")


def get_coordinates_as_point(inputdict):

    return Point(inputdict["coordinates"])


# from tqdm import tqdm
def get_mapillary_token(token_file="mapillary_token", verbose=False):
    """
    Discover Mapillary API token from multiple sources in priority order:
    1. Environment variables (API_TOKEN, MAPPILLARY_API_TOKEN, MAPILLARY_TOKEN)
    2. Token file (default: "mapillary_token")
    Returns:
        str: The API token, or empty string if none found
    """
    # List of environment variables to check in priority order
    env_vars = ["API_TOKEN", "MAPPILLARY_API_TOKEN", "MAPILLARY_TOKEN"]
    # Check environment variables first
    for env_var in env_vars:
        token = os.environ.get(env_var)
        if token and token.strip():
            if verbose:
                print(f"✅ Found API token in environment variable: {env_var}")
            return token.strip()

    # Fallback to file-based token discovery
    if os.path.exists(token_file):
        try:
            with open(token_file, "r") as f:
                token = f.readline().strip()
                if token:
                    if verbose:
                        print(f"✅ Found API token in file: {token_file}")
                    return token
        except (IOError, OSError) as e:
            if verbose:
                print(f"⚠️  Warning: Could not read token file {token_file}: {e}")

    if verbose:
        print(
            "⚠️  Warning: No API token found. Please set one of these environment variables:"
        )
        print("   - API_TOKEN")
        print("   - MAPPILLARY_API_TOKEN")
        print("   - MAPILLARY_TOKEN")
        print(f"   Or create a file named '{token_file}' with your token.")

    return ""


def redact_token(message, token):
    """
    Replace every occurrence of the API token in a message (e.g. an exception
    whose text contains the request URL) with '***'.
    """
    message = str(message)
    if token:
        message = message.replace(token, "***")
        # requests URL-encodes the "|" separators of Mapillary tokens
        message = message.replace(requests.utils.quote(token, safe=""), "***")
    return message


# right after the function definition
MAPPILARY_TOKEN = get_mapillary_token()

# Warn if no token is found at import time
if not MAPPILARY_TOKEN:
    print("⚠️  Warning: No Mapillary API token found at import time.")
    print("   Functions requiring authentication will fail unless a token is provided.")


def get_mapillary_images_metadata(
    minLon,
    minLat,
    maxLon,
    maxLat,
    fields=default_fields,
    token=MAPPILARY_TOKEN,
    outpath=None,
    limit=2000,
    timeout=300,
    start_captured_at=None,
    end_captured_at=None,
):
    """
    Request images from Mapillary API given a bbox

    Parameters:
        minLat (float): The latitude of the first coordinate.
        minLon (float): The longitude of the first coordinate.
        maxLat (float): The latitude of the second coordinate.
        maxLon (float): The longitude of the second coordinate.
        token (str): The Mapillary API token.
        start_captured_at (str, optional): Only images captured at or after
            this ISO 8601 datetime (e.g. '2019-01-01T00:00:00Z').
        end_captured_at (str, optional): Only images captured before this
            ISO 8601 datetime.

    Returns:
        dict: A dictionary containing the response from the API.

    Raises:
        requests.exceptions.RequestException: For network-related errors
        ValueError: For invalid API responses or missing token
    """
    # Input validation
    if not all(
        isinstance(coord, (int, float)) for coord in [minLon, minLat, maxLon, maxLat]
    ):
        raise ValueError("All coordinate parameters must be numbers")

    if not (-180 <= minLon <= 180) or not (-180 <= maxLon <= 180):
        raise ValueError("Longitude values must be between -180 and 180")

    if not (-90 <= minLat <= 90) or not (-90 <= maxLat <= 90):
        raise ValueError("Latitude values must be between -90 and 90")

    if minLon >= maxLon or minLat >= maxLat:
        raise ValueError(
            "Invalid bounding box: min coordinates must be less than max coordinates"
        )

    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("Limit must be a positive integer")

    if not token:
        raise ValueError(
            "No valid Mapillary API token provided. Please set API_TOKEN environment variable or create a mapillary_token file."
        )

    url = "https://graph.mapillary.com/images"
    params = {
        "bbox": f"{minLon},{minLat},{maxLon},{maxLat}",
        "limit": limit,
        "access_token": token,
        "fields": ",".join(fields),
    }
    if start_captured_at:
        params["start_captured_at"] = start_captured_at
    if end_captured_at:
        params["end_captured_at"] = end_captured_at

    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()  # Raises HTTPError for bad status codes
    except requests.exceptions.RequestException as e:
        # include the API's own error message, if any
        detail = ""
        if getattr(e, "response", None) is not None:
            try:
                detail = f" ({e.response.json()['error']['message']})"
            except Exception:
                pass
        raise requests.exceptions.RequestException(
            f"Failed to fetch data from Mapillary API: {redact_token(str(e) + detail, token)}"
        )

    try:
        as_dict = response.json()
    except ValueError as e:
        raise ValueError(f"Invalid JSON response from Mapillary API: {e}")

    # Check for API error responses
    if "error" in as_dict:
        error_msg = as_dict.get("error", {}).get("message", "Unknown API error")
        raise ValueError(f"Mapillary API error: {error_msg}")

    # Warn if results might be truncated
    if as_dict.get("data") and len(as_dict["data"]) == limit:
        print(
            f"⚠️  Warning: Query returned exactly {limit} results - there may be more images available. Consider using tiled querying for complete coverage."
        )

    if outpath:
        dump_json(as_dict, outpath)

    return as_dict


def radius_to_degrees(radius, lat):
    """
    Convert a radius in meters to degrees.
    """
    return radius / (111320 * cos(lat * pi / 180))


def degrees_to_radius(degrees, lat):
    """
    Convert a radius in degrees to meters.
    """
    return degrees * 111320 * cos(lat * pi / 180)


def get_bounding_box(lon, lat, radius):
    """
    Return a bounding box tuple as (minLon, minLat, maxLon, maxLat) from a pair
    of coordinates and a radius, using shapely.

    Parameters:
        lon (float): The longitude of the center of the bounding box.
        lat (float): The latitude of the center of the bounding box.
        radius (float): The radius of the bounding box in meters.

    Returns:
        tuple: A tuple containing the minimum and maximum longitude and latitude
            of the bounding box.

    Raises:
        ValueError: For invalid coordinate or radius values
    """
    # Input validation
    if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
        raise ValueError("Longitude and latitude must be numbers")

    if not (-180 <= lon <= 180):
        raise ValueError("Longitude must be between -180 and 180")

    if not (-90 <= lat <= 90):
        raise ValueError("Latitude must be between -90 and 90")

    if not isinstance(radius, (int, float)) or radius <= 0:
        raise ValueError("Radius must be a positive number")

    # Convert radius from meters to degrees
    radius_deg = radius_to_degrees(radius, lat)

    point = Point(lon, lat)
    return box(
        point.x - radius_deg,
        point.y - radius_deg,
        point.x + radius_deg,
        point.y + radius_deg,
    ).bounds


# function to download an image from a url:
def download_mapillary_image(url, outfilepath, cooldown=1):
    """
    Download an image from a URL and save it to the specified path.

    Parameters:
        url (str): The URL of the image to download.
        outfilepath (str): The path where the image should be saved.
        cooldown (int): Time to wait after download in seconds.

    Raises:
        Exception: Re-raises any download errors for proper error propagation
    """
    try:
        wget.download(url, out=outfilepath)
        if cooldown:
            sleep(cooldown)
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        raise  # Re-raise the exception so calling code can handle it


def mapillary_data_to_gdf(
    data,
    outpath=None,
    filtering_polygon=None,
    detections_summary=False,
    token=MAPPILARY_TOKEN,
):
    """
    Convert Mapillary API response data to a GeoDataFrame.

    Parameters:
        data (dict): Mapillary API response containing image metadata
        outpath (str, optional): Path to save the GeoDataFrame
        filtering_polygon (optional): Polygon to filter results spatially
        detections_summary (bool, optional): Add a 'detections_summary'
            column (see add_detections_summary()). Requests the detections of
            every image. Default is False.
        token (str, optional): The Mapillary API token, used when
            detections_summary is True.

    Returns:
        GeoDataFrame: Processed image data with geometry

    Raises:
        ValueError: If data format is invalid
    """
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary (Mapillary API response)")

    if data.get("data"):
        try:
            as_df = pd.DataFrame.from_records(data["data"])

            # Check if geometry column exists
            if "geometry" not in as_df.columns:
                raise ValueError("No 'geometry' field found in data records")

            as_df.geometry = as_df.geometry.apply(get_coordinates_as_point)

            as_gdf = gpd.GeoDataFrame(as_df, crs="EPSG:4326", geometry="geometry")

            selected_columns_to_str(as_gdf)

            if filtering_polygon:
                as_gdf = as_gdf[as_gdf.intersects(filtering_polygon)].copy()

            if detections_summary:
                add_detections_summary(as_gdf, token=token)

            if outpath:
                try:
                    save_gdf(as_gdf, outpath)
                except Exception as e:
                    print(f"⚠️  Warning: Could not save to {outpath}: {e}")

            return as_gdf
        except Exception as e:
            print(f"⚠️  Warning: Error processing data: {e}")
            return gpd.GeoDataFrame()
    else:
        return gpd.GeoDataFrame()


def tiled_mapillary_data_to_gdf(
    input_polygon, token, zoom=ZOOM_LEVEL, outpath=None, detections_summary=False
):

    # get the bbox of the input polygon:
    minLon, minLat, maxLon, maxLat = input_polygon.bounds

    # get the bboxes of the tiles:
    bboxes = tilebboxes_from_bbox(minLat, minLon, maxLat, maxLon, zoom)

    # get the metadata for each tile:
    gdfs_list = []

    for bbox in tqdm(bboxes):
        # for i, bbox in enumerate(tqdm(bboxes)):

        # get the tile as geometry:
        bbox_geom = tile_bbox_to_box(bbox)

        # check if the tile intersects the input polygon:
        if not bbox_geom.disjoint(input_polygon):
            # get the metadata for the tile:
            data = get_mapillary_images_metadata(
                *resort_bbox(bbox), token
            )  # ,outpath=f'tests\small_city_tiles\{i}.json')

            if data.get("data"):
                # convert the metadata to a GeoDataFrame:
                gdfs_list.append(mapillary_data_to_gdf(data, outpath, input_polygon))

    # concatenate the GeoDataFrames:
    as_gdf = pd.concat(gdfs_list)

    if detections_summary:
        add_detections_summary(as_gdf, token=token)

    if outpath:
        save_gdf(as_gdf, outpath)

    return as_gdf


def resort_bbox(bbox):
    return [bbox[1], bbox[0], bbox[3], bbox[2]]


def get_territory_polygon(place_name, outpath=None):
    """
    Get polygon for a named place using OpenStreetMap Nominatim API.

    Parameters:
        place_name (str): Name of the place to search for.
        outpath (str, optional): Path to save the polygon as JSON.

    Returns:
        dict: GeoJSON polygon object, or None if not found.

    Raises:
        requests.exceptions.RequestException: For network-related errors
        ValueError: For invalid responses or no results found
    """
    # Make a request to Nominatim API with the place name
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place_name, "format": "json", "polygon_geojson": 1}

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(
            f"Failed to fetch territory data from Nominatim API: {e}"
        )

    try:
        data = response.json()
    except ValueError as e:
        raise ValueError(f"Invalid JSON response from Nominatim API: {e}")

    if not data:
        raise ValueError(f"No results found for place name: '{place_name}'")

    # sort data by "importance", that is a key in each dictionary of the list:
    data.sort(key=lambda x: x.get("importance", 0), reverse=True)

    # removing all non-polygon objects:
    polygon_data = [d for d in data if d.get("geojson", {}).get("type") == "Polygon"]

    if not polygon_data:
        raise ValueError(f"No polygon geometry found for place name: '{place_name}'")

    # Get the polygon of the territory as a GeoJSON object
    polygon = polygon_data[0]["geojson"]

    if outpath:
        dump_json(polygon, outpath)

    # Return the polygon
    return polygon


def filter_metadata_with_polygon(data, polygon, anti_rounding_factor=1000000):
    """
    Filter metadata by keeping only points that are contained within the polygon.

    Parameters:
        data (dict): Mapillary API response containing image metadata.
        polygon: Shapely polygon object for filtering.
        anti_rounding_factor (int): Factor to handle coordinate precision issues.

    Returns:
        dict: Filtered metadata dictionary.
    """
    if not data.get("data"):
        return data

    # Create a copy to avoid modifying the original data
    filtered_data = data.copy()
    filtered_data["data"] = []

    # Iterate through items and keep only those within the polygon
    for item in data["data"]:
        try:
            point = Point(item["geometry"]["coordinates"])
            if polygon.contains(point):
                filtered_data["data"].append(item)
        except (KeyError, TypeError, ValueError) as e:
            # Skip malformed entries but continue processing
            print(f"⚠️  Warning: Skipping malformed geometry in item: {e}")
            continue

    return filtered_data


# ---------------------------------------------------------------------------
# Coverage vector tiles
# ---------------------------------------------------------------------------
#
# The /images bbox search refuses areas with many images ("Please reduce the
# amount of data you're asking for"). The coverage vector tiles list every
# image of a zoom-14 tile (~2.4 km) with its capture date instead.

COVERAGE_TILES_URL = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}"
IMAGES_TILE_ZOOM = 14


def get_coverage_tile_images(tile, token=MAPPILARY_TOKEN, timeout=120):
    """
    Get the images of a Mapillary coverage vector tile ('image' layer).

    Parameters:
        tile (mercantile.Tile): The tile, at zoom 14 (the only zoom level
            with the image layer), e.g. mercantile.tile(lon, lat, 14).
        token (str): The Mapillary API token.
        timeout (int, optional): Request timeout in seconds.

    Returns:
        list: Image dictionaries shaped like the /images API data (so that
            mapillary_data_to_gdf({"data": images}) works), with the tile's
            properties: id, captured_at (epoch milliseconds), sequence_id,
            is_pano, compass_angle, creator_id, organization_id.

    Raises:
        requests.exceptions.RequestException: For network-related errors
        ValueError: For a missing token or an undecodable tile
    """
    if not token:
        raise ValueError(
            "No valid Mapillary API token provided. Please set API_TOKEN environment variable or create a mapillary_token file."
        )

    url = COVERAGE_TILES_URL.format(z=tile.z, x=tile.x, y=tile.y)
    try:
        response = requests.get(url, params={"access_token": token}, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(
            f"Failed to fetch coverage tile from Mapillary: {redact_token(e, token)}"
        )

    data = response.content
    if not data:
        return []
    try:
        if data[:2] == b"\x1f\x8b":
            data = zlib.decompress(data, 16 + zlib.MAX_WBITS)
        decoded = mapbox_vector_tile.decode(data, default_options={"y_coord_down": True})
    except Exception as e:
        raise ValueError(f"Cannot decode coverage tile {tuple(tile)}: {e}")

    layer = decoded.get("image")
    if not layer:
        return []

    # tile pixel coordinates -> web mercator -> lon/lat
    extent = layer.get("extent", 4096)
    left, bottom, right, top = mercantile.xy_bounds(tile)

    images = []
    for feature in layer.get("features", []):
        if feature["geometry"]["type"] != "Point":
            continue
        px, py = feature["geometry"]["coordinates"]
        lon, lat = mercantile.lnglat(
            left + (right - left) * px / extent, top - (top - bottom) * py / extent
        )
        image = dict(feature.get("properties", {}))
        image.setdefault("id", feature.get("id"))
        image["geometry"] = {"type": "Point", "coordinates": [lon, lat]}
        images.append(image)

    return images


# ---------------------------------------------------------------------------
# Semantic segmentation (detections)
# ---------------------------------------------------------------------------
#
# Mapillary runs semantic segmentation on its servers and exposes the result
# per image as "detections": one entry per segmented region, where "value" is
# the class (e.g. "construction--flat--road", "nature--sky") and "geometry" is
# a base64-encoded Mapbox Vector Tile containing the region polygon(s) in
# image space, normalized to the tile extent (usually 4096).

GRAPH_API_URL = "https://graph.mapillary.com"

DETECTION_FIELDS = ["id", "value", "geometry", "created_at"]

# Top-level prefixes of the detection classes, grouped by kind
SURFACE_CLASS_PREFIXES = ("construction", "nature", "void")
MARKING_CLASS_PREFIXES = ("marking",)
OBJECT_CLASS_PREFIXES = ("object", "human", "animal")
TRAFFIC_SIGN_CLASS_PREFIXES = ("regulatory", "warning", "information", "complementary")


def detection_class_group(value):
    """
    Return the group of a detection class: 'surface' (full-scene classes such
    as road, sky, building), 'marking' (road markings), 'object',
    'traffic_sign' or 'other'.
    """
    prefix = str(value).split("--")[0]
    if prefix in SURFACE_CLASS_PREFIXES:
        return "surface"
    if prefix in MARKING_CLASS_PREFIXES:
        return "marking"
    if prefix in OBJECT_CLASS_PREFIXES:
        return "object"
    if prefix in TRAFFIC_SIGN_CLASS_PREFIXES:
        return "traffic_sign"
    return "other"


def _graph_api_get(url, params=None, token=MAPPILARY_TOKEN, timeout=60):
    """
    GET a Mapillary Graph API URL and return the parsed JSON.

    The token is never included in raised error messages.

    Raises:
        requests.exceptions.RequestException: For network-related errors
        ValueError: For API errors, invalid JSON or a missing token
    """
    if not token:
        raise ValueError(
            "No valid Mapillary API token provided. Please set API_TOKEN environment variable or create a mapillary_token file."
        )

    params = dict(params or {})
    if "access_token=" not in url:
        params["access_token"] = token

    try:
        response = requests.get(url, params=params, timeout=timeout)
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(
            f"Failed to fetch data from Mapillary API: {redact_token(e, token)}"
        )

    try:
        as_dict = response.json()
    except ValueError:
        as_dict = None

    # API errors come with a JSON body, often along with a 4xx status code
    if isinstance(as_dict, dict) and "error" in as_dict:
        error = as_dict["error"]
        error_msg = error.get("message", "Unknown API error") if isinstance(error, dict) else error
        raise ValueError(
            f"Mapillary API error (HTTP {response.status_code}): {redact_token(error_msg, token)}"
        )

    try:
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(
            f"Failed to fetch data from Mapillary API: {redact_token(e, token)}"
        )

    if as_dict is None:
        raise ValueError("Invalid JSON response from Mapillary API")

    return as_dict


def get_image_detections(
    image_id, token=MAPPILARY_TOKEN, fields=DETECTION_FIELDS, timeout=60, max_pages=100
):
    """
    Get the detections (semantic segmentation regions) of a Mapillary image.

    Parameters:
        image_id (str or int): The Mapillary image ID.
        token (str): The Mapillary API token.
        fields (list, optional): Detection fields to request. Default is
            ['id', 'value', 'geometry', 'created_at'].
        timeout (int, optional): Request timeout in seconds.
        max_pages (int, optional): Maximum number of result pages to follow.

    Returns:
        list: Detection dictionaries. Empty if the image has no detections.

    Raises:
        requests.exceptions.RequestException: For network-related errors
        ValueError: For invalid API responses or missing token
    """
    url = f"{GRAPH_API_URL}/{image_id}/detections"
    params = {"fields": ",".join(fields)}

    detections = []
    for _ in range(max_pages):
        as_dict = _graph_api_get(url, params, token=token, timeout=timeout)
        detections.extend(as_dict.get("data", []))

        next_url = as_dict.get("paging", {}).get("next")
        if not next_url:
            break
        # the "next" URL already carries all the query parameters
        url, params = next_url, None

    return detections


def get_image_size(image_id, token=MAPPILARY_TOKEN, timeout=60):
    """
    Get the (width, height) in pixels of the original Mapillary image.
    """
    as_dict = _graph_api_get(
        f"{GRAPH_API_URL}/{image_id}", {"fields": "width,height"}, token=token, timeout=timeout
    )
    return int(as_dict["width"]), int(as_dict["height"])


def _decode_detection_tile(geometry, y_coord_down=True):
    """
    Decode the base64-encoded vector tile of a detection into the plain
    {layer: {"extent": ..., "features": [...]}} structure of mapbox_vector_tile,
    where feature geometries are GeoJSON-like coordinate lists.
    """
    try:
        data = base64.b64decode(geometry)
        # tolerate gzip-compressed tiles
        if data[:2] == b"\x1f\x8b":
            data = zlib.decompress(data, 16 + zlib.MAX_WBITS)
        return mapbox_vector_tile.decode(
            data, default_options={"y_coord_down": y_coord_down}
        )
    except Exception as e:
        raise ValueError(f"Cannot decode detection geometry: {e}")


def decode_detection_geometry(geometry, width=1, height=1, y_coord_down=True):
    """
    Decode the base64-encoded vector tile geometry of a Mapillary detection.

    Parameters:
        geometry (str): The 'geometry' field of a detection.
        width (float, optional): Image width used to scale the coordinates.
            Default is 1 (normalized coordinates).
        height (float, optional): Image height used to scale the coordinates.
            Default is 1 (normalized coordinates).
        y_coord_down (bool, optional): Keep the tile's native y axis, which
            points down like image rows. Default is True.

    Returns:
        Polygon or MultiPolygon: The region in image coordinates, with the
            origin at the top-left corner of the image.

    Raises:
        ValueError: If the geometry cannot be decoded
    """
    tile = _decode_detection_tile(geometry, y_coord_down)

    parts = []
    for layer in tile.values():
        extent = layer.get("extent", 4096)
        for feature in layer.get("features", []):
            geom = shape(feature["geometry"])
            if geom.is_empty:
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            parts.append(
                scale_geometry(
                    geom, xfact=width / extent, yfact=height / extent, origin=(0, 0)
                )
            )

    if not parts:
        raise ValueError("Detection geometry contains no features")

    if len(parts) == 1:
        return parts[0]
    return gpd.GeoSeries(parts).union_all()


def _ring_area(ring):
    """Area of a ring given as a list of [x, y] (shoelace formula)."""
    coords = np.asarray(ring, dtype=float)
    if len(coords) < 3:
        return 0.0
    x, y = coords[:, 0], coords[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def _encoded_geometry_fraction(geometry):
    """
    Fraction of the image covered by an encoded detection geometry, computed
    directly from the decoded coordinate lists (no geometry objects).
    """
    fraction = 0.0
    for layer in _decode_detection_tile(geometry).values():
        extent = layer.get("extent", 4096)
        for feature in layer.get("features", []):
            geom = feature["geometry"]
            if geom["type"] == "Polygon":
                polygons = [geom["coordinates"]]
            elif geom["type"] == "MultiPolygon":
                polygons = geom["coordinates"]
            else:
                continue
            for rings in polygons:
                if not rings:
                    continue
                area = _ring_area(rings[0]) - sum(_ring_area(r) for r in rings[1:])
                fraction += max(area, 0.0) / extent**2
    return fraction


def detections_summary(detections):
    """
    Summarize the detections of an image without building any geometry.

    Parameters:
        detections (list): Detections from get_image_detections() (the
            'value' and 'geometry' fields are used).

    Returns:
        dict: {
            "present": bool, whether the image has detections,
            "number_available_classes": int, number of distinct classes,
            "class_percents": {class_value: percent of the image area},
                sorted from the largest to the smallest class,
        }
        The percentages are areas of Mapillary's polygons; regions of a
        semantic segmentation don't overlap, so they sum to at most ~100 and
        the rest of the image is unlabeled.
    """
    percents = {}
    for detection in detections:
        value = detection.get("value")
        if not detection.get("geometry"):
            continue
        try:
            fraction = _encoded_geometry_fraction(detection["geometry"])
        except ValueError as e:
            print(f"⚠️  Warning: Skipping detection {detection.get('id')}: {e}")
            continue
        percents[value] = percents.get(value, 0.0) + 100 * fraction

    return {
        "present": len(detections) > 0,
        "number_available_classes": len({d.get("value") for d in detections}),
        "class_percents": {
            value: round(float(percent), 4)
            for value, percent in sorted(percents.items(), key=lambda item: item[1], reverse=True)
        },
    }


def add_detections_summary(
    gdf, token=MAPPILARY_TOKEN, id_field="id", column="detections_summary", cooldown=0
):
    """
    Add a column with the detections summary (see detections_summary()) of
    each image of a GeoDataFrame, e.g. one from mapillary_data_to_gdf().

    Only the detection classes and encoded geometries are requested, and the
    geometries are never turned into geometry objects.

    Parameters:
        gdf (GeoDataFrame): The images, one row per image.
        token (str): The Mapillary API token.
        id_field (str, optional): Field with the image ID. Default is 'id'.
        column (str, optional): Name of the new column. Default is
            'detections_summary'.
        cooldown (float, optional): Seconds to wait between images.

    Returns:
        GeoDataFrame: The same GeoDataFrame, with the new column. Images whose
            detections could not be requested get None.
    """
    if id_field not in gdf.columns:
        raise ValueError(f"ID field '{id_field}' not found in GeoDataFrame columns")

    summaries = []
    errors = []
    for image_id in tqdm(gdf[id_field], total=len(gdf), desc="Summarizing detections"):
        try:
            detections = get_image_detections(
                image_id, token=token, fields=["value", "geometry"]
            )
            summaries.append(detections_summary(detections))
        except Exception as e:
            summaries.append(None)
            errors.append(
                f"Failed to get detections for image ID {image_id}: {redact_token(e, token)}"
            )
        if cooldown:
            sleep(cooldown)

    # a plain object array, so that duplicate index labels (e.g. after
    # pd.concat) don't get in the way
    values = np.empty(len(summaries), dtype=object)
    values[:] = summaries
    gdf[column] = values

    with_detections = sum(1 for s in summaries if s and s["present"])
    print(
        f"✅ Detections summarized: {with_detections} with detections, {len(summaries) - with_detections - len(errors)} without, {len(errors)} failed"
    )
    if errors:
        print(f"❌ Errors encountered: {len(errors)}")
        for error in errors[:5]:
            print(f"   - {error}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more errors")

    return gdf


def save_gdf(gdf, outpath):
    """
    Save a GeoDataFrame, serializing dict columns (e.g. detections_summary)
    as JSON text, since vector file formats cannot store them.
    """
    to_save = gdf.copy()
    for column in to_save.columns:
        if column != to_save.geometry.name and any(isinstance(v, dict) for v in to_save[column]):
            to_save[column] = to_save[column].apply(
                lambda v: json.dumps(v, ensure_ascii=False) if v is not None else None
            )
    to_save.to_file(outpath)


def detections_to_gdf(detections, width=1, height=1, y_coord_down=True):
    """
    Convert Mapillary detections to a GeoDataFrame of image-space polygons.

    The geometries are in pixel coordinates (origin at the top-left corner,
    y pointing down) when width and height are given, or normalized to [0, 1]
    otherwise. The GeoDataFrame has no CRS.

    Parameters:
        detections (list): Detections from get_image_detections().
        width (float, optional): Image width. Default is 1 (normalized).
        height (float, optional): Image height. Default is 1 (normalized).
        y_coord_down (bool, optional): See decode_detection_geometry().

    Returns:
        GeoDataFrame: One row per detection, plus a 'group' column
            (see detection_class_group()).
    """
    records = []
    for detection in detections:
        if not detection.get("geometry"):
            continue
        try:
            geom = decode_detection_geometry(
                detection["geometry"], width, height, y_coord_down
            )
        except ValueError as e:
            print(f"⚠️  Warning: Skipping detection {detection.get('id')}: {e}")
            continue

        record = {k: v for k, v in detection.items() if k != "geometry"}
        record["group"] = detection_class_group(detection.get("value"))
        record["geometry"] = geom
        records.append(record)

    if not records:
        return gpd.GeoDataFrame(
            columns=["id", "value", "created_at", "group", "geometry"],
            geometry="geometry",
        )

    as_gdf = gpd.GeoDataFrame(records, geometry="geometry")
    selected_columns_to_str(as_gdf, dict)
    return as_gdf


def _polygon_parts(geom):
    if geom.geom_type == "Polygon":
        return [geom]
    if hasattr(geom, "geoms"):
        return [p for g in geom.geoms for p in _polygon_parts(g)]
    return []


def detections_to_mask(
    detections, width, height, class_index=None, y_coord_down=True
):
    """
    Rasterize Mapillary detections into a semantic segmentation label mask.

    Regions are drawn from the largest to the smallest outline, so smaller
    objects stay on top of larger regions. Holes are left unlabeled unless
    another region fills them.

    Parameters:
        detections (list): Detections from get_image_detections().
        width (int): Mask width in pixels.
        height (int): Mask height in pixels.
        class_index (dict, optional): Existing {class_value: label} mapping to
            extend, so that labels are consistent across images.
        y_coord_down (bool, optional): See decode_detection_geometry().

    Returns:
        tuple: (mask, class_index), where mask is a (height, width) uint16
            array with 0 meaning unlabeled, and class_index maps each class
            value to its label.
    """
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("Mask width and height must be positive")

    class_index = dict(class_index or {})
    gdf = detections_to_gdf(detections, width, height, y_coord_down)

    for value in sorted(gdf["value"].unique()) if not gdf.empty else []:
        if value not in class_index:
            class_index[value] = max(class_index.values(), default=0) + 1

    if max(class_index.values(), default=0) > np.iinfo(np.uint16).max:
        raise ValueError("Too many classes for a uint16 mask")

    canvas = Image.new("I", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    shapes = []
    for value, geom in zip(gdf["value"], gdf.geometry):
        for part in _polygon_parts(geom):
            outline_area = abs(Polygon(part.exterior).area)
            shapes.append((outline_area, class_index[value], part))

    for _, label, part in sorted(shapes, key=lambda s: s[0], reverse=True):
        draw.polygon(list(part.exterior.coords), fill=label)
        for interior in part.interiors:
            draw.polygon(list(interior.coords), fill=0)

    mask = np.array(canvas, dtype=np.int32).astype(np.uint16)
    return mask, class_index


def class_color(value):
    """
    Deterministic RGB color for a detection class.
    """
    digest = zlib.crc32(str(value).encode("utf-8"))
    return (64 + digest % 192, 64 + (digest >> 8) % 192, 64 + (digest >> 16) % 192)


def colorize_mask(mask, class_index):
    """
    Convert a label mask into an RGB image (uint8 array) for visualization.
    Unlabeled pixels are black.
    """
    lut = np.zeros((max(class_index.values(), default=0) + 1, 3), dtype=np.uint8)
    for value, label in class_index.items():
        lut[label] = class_color(value)
    return lut[mask]


def save_mask(mask, outpath):
    """
    Save a label mask as a 16-bit grayscale PNG (pixel value = class label).
    """
    Image.fromarray(mask.astype(np.uint16)).save(outpath)


def download_segmentation_masks_from_gdf(
    gdf,
    outfolderpath,
    id_field="id",
    width_field="width",
    height_field="height",
    token=MAPPILARY_TOKEN,
    scale_factor=1.0,
    save_detections=True,
    save_colorized=False,
    cooldown=0,
):
    """
    Download the semantic segmentation (detections) of every image in a
    GeoDataFrame and save them as label masks.

    For each image with detections, writes:
        {id}_mask.png: 16-bit label mask (see detections_to_mask())
        {id}_classes.json: {class_value: label} mapping of the mask
        {id}_detections.json: raw detections (if save_detections)
        {id}_mask_color.png: colorized mask (if save_colorized)

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing the images.
        outfolderpath (str): The path to the output folder.
        id_field (str, optional): Field with the image ID. Default is 'id'.
        width_field (str, optional): Field with the image width. Default is
            'width'. The size is requested from the API if the field is missing.
        height_field (str, optional): Field with the image height. Default is
            'height'.
        token (str): The Mapillary API token.
        scale_factor (float, optional): Scaling factor of the masks relative
            to the original image size. Default is 1.0.
        save_detections (bool, optional): Save raw detections. Default is True.
        save_colorized (bool, optional): Save colorized masks. Default is False.
        cooldown (float, optional): Seconds to wait between images.

    Returns:
        dict: Summary with success/empty/failed counts and errors. 'empty'
            counts images for which Mapillary returned no detections.
    """
    if gdf.empty:
        print("⚠️  Warning: Empty GeoDataFrame provided - nothing to download")
        return {"success": 0, "empty": 0, "failed": 0, "errors": []}

    if id_field not in gdf.columns:
        raise ValueError(f"ID field '{id_field}' not found in GeoDataFrame columns")

    create_dir_if_not_exists(outfolderpath)

    success_count = 0
    empty_count = 0
    failed_count = 0
    errors = []

    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Downloading masks"):
        image_id = row.get(id_field, "unknown")
        try:
            detections = get_image_detections(image_id, token=token)
            if not detections:
                empty_count += 1
                continue

            width, height = row.get(width_field), row.get(height_field)
            if pd.isna(width) or pd.isna(height) or not width or not height:
                width, height = get_image_size(image_id, token=token)

            mask, class_index = detections_to_mask(
                detections,
                max(1, round(float(width) * scale_factor)),
                max(1, round(float(height) * scale_factor)),
            )

            basepath = os.path.join(outfolderpath, str(image_id))
            save_mask(mask, basepath + "_mask.png")
            dump_json(class_index, basepath + "_classes.json")
            if save_detections:
                dump_json(detections, basepath + "_detections.json")
            if save_colorized:
                Image.fromarray(colorize_mask(mask, class_index)).save(
                    basepath + "_mask_color.png"
                )
            success_count += 1
        except Exception as e:
            errors.append(
                f"Failed to download mask for image ID {image_id}: {redact_token(e, token)}"
            )
            failed_count += 1

        if cooldown:
            sleep(cooldown)

    print(
        f"✅ Download completed: {success_count} successful, {empty_count} without detections, {failed_count} failed"
    )
    if errors:
        print(f"❌ Errors encountered: {len(errors)}")
        for error in errors[:5]:
            print(f"   - {error}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more errors")

    return {
        "success": success_count,
        "empty": empty_count,
        "failed": failed_count,
        "errors": errors,
    }
