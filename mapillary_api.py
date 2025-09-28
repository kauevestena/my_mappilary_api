from math import cos, pi
from time import sleep
import os
import json
import requests
import wget
import geopandas as gpd
import pandas as pd
from shapely import Point, box
import mercantile
from tqdm import tqdm

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
    gdf, outfolderpath, id_field="id", url_field="thumb_original_url"
):
    """
    Downloads all the pictures from a GeoDataFrame (gdf) and saves them to the
    specified output folder.

    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame containing the data.
        outfolderpath (str): The path to the output folder where the pictures
            will be saved.
        id_field (str, optional): The name of the field in the GeoDataFrame
            that contains the unique identifier for each picture. Default is 'id'.
        url_field (str, optional): The name of the field in the GeoDataFrame
            that contains the URL of the picture. Default is 'thumb_original_url'.

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
                
            download_mapillary_image(
                image_url,
                os.path.join(outfolderpath, str(image_id) + ".jpg"),
            )
            success_count += 1
        except Exception as e:
            error_msg = f"Failed to download image ID {getattr(row, id_field, 'unknown')}: {e}"
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
        print("⚠️  Warning: No API token found. Please set one of these environment variables:")
        print("   - API_TOKEN")
        print("   - MAPPILLARY_API_TOKEN") 
        print("   - MAPILLARY_TOKEN")
        print(f"   Or create a file named '{token_file}' with your token.")
    
    return ""


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
    limit=5000,
):
    """
    Request images from Mapillary API given a bbox

    Parameters:
        minLat (float): The latitude of the first coordinate.
        minLon (float): The longitude of the first coordinate.
        maxLat (float): The latitude of the second coordinate.
        maxLon (float): The longitude of the second coordinate.
        token (str): The Mapillary API token.

    Returns:
        dict: A dictionary containing the response from the API.
        
    Raises:
        requests.exceptions.RequestException: For network-related errors
        ValueError: For invalid API responses or missing token
    """
    # Input validation
    if not all(isinstance(coord, (int, float)) for coord in [minLon, minLat, maxLon, maxLat]):
        raise ValueError("All coordinate parameters must be numbers")
    
    if not (-180 <= minLon <= 180) or not (-180 <= maxLon <= 180):
        raise ValueError("Longitude values must be between -180 and 180")
    
    if not (-90 <= minLat <= 90) or not (-90 <= maxLat <= 90):
        raise ValueError("Latitude values must be between -90 and 90")
    
    if minLon >= maxLon or minLat >= maxLat:
        raise ValueError("Invalid bounding box: min coordinates must be less than max coordinates")
    
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("Limit must be a positive integer")
        
    if not token:
        raise ValueError("No valid Mapillary API token provided. Please set API_TOKEN environment variable or create a mapillary_token file.")
    
    url = "https://graph.mapillary.com/images"
    params = {
        "bbox": f"{minLon},{minLat},{maxLon},{maxLat}",
        "limit": limit,
        "access_token": token,
        "fields": ",".join(fields),
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()  # Raises HTTPError for bad status codes
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Failed to fetch data from Mapillary API: {e}")

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
        print(f"⚠️  Warning: Query returned exactly {limit} results - there may be more images available. Consider using tiled querying for complete coverage.")

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


def mapillary_data_to_gdf(data, outpath=None, filtering_polygon=None):
    """
    Convert Mapillary API response data to a GeoDataFrame.
    
    Parameters:
        data (dict): Mapillary API response containing image metadata
        outpath (str, optional): Path to save the GeoDataFrame
        filtering_polygon (optional): Polygon to filter results spatially
        
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
                as_gdf = as_gdf[as_gdf.intersects(filtering_polygon)]

            if outpath:
                try:
                    as_gdf.to_file(outpath)
                except Exception as e:
                    print(f"⚠️  Warning: Could not save to {outpath}: {e}")

            return as_gdf
        except Exception as e:
            print(f"⚠️  Warning: Error processing data: {e}")
            return gpd.GeoDataFrame()
    else:
        return gpd.GeoDataFrame()


def tiled_mapillary_data_to_gdf(input_polygon, token, zoom=ZOOM_LEVEL, outpath=None):

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

    if outpath:
        as_gdf.to_file(outpath)

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
        raise requests.exceptions.RequestException(f"Failed to fetch territory data from Nominatim API: {e}")

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
    if not data.get('data'):
        return data
    
    # Create a copy to avoid modifying the original data
    filtered_data = data.copy()
    filtered_data['data'] = []
    
    # Iterate through items and keep only those within the polygon
    for item in data['data']:
        try:
            point = Point(item['geometry']['coordinates'])
            if polygon.contains(point):
                filtered_data['data'].append(item)
        except (KeyError, TypeError, ValueError) as e:
            # Skip malformed entries but continue processing
            print(f"⚠️  Warning: Skipping malformed geometry in item: {e}")
            continue
    
    return filtered_data
