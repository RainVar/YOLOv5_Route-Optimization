import os
import csv
import time
import requests
from shapely.geometry import LineString
from io import BytesIO
from PIL import Image
import osmnx as ox
import networkx as nx

# Note: 
# - replace api key
# - modify pa ang pitch ug size sa image

# Config
API_KEY = "YOUR_GOOGLE_API_KEY" ### atoa key frfr
IMAGE_DIR = "road_images"
METADATA_FILE = "image_metadata.csv" ### trace ang images where na belong
SPACING_METERS = 10
HEADINGS = [0]  # can be [0, 90, 180, 270] for full coverage

# Ensure folders exist
os.makedirs(IMAGE_DIR, exist_ok=True)

# Metadata CSV header
CSV_HEADER = [
    "segment_id", "u", "v", "k", "index", "lat", "lng", 
    "heading", "image_path"
]

# Initialize metadata file
if not os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)


def sample_points_on_edge(u, v, k, data, spacing=10):
    """Samples points along the edge geometry at fixed spacing."""
    geom = data.get('geometry')
    if not geom:
        # Use straight line
        point_u = (data['x'], data['y'])
        point_v = (data['x'], data['y'])
        geom = LineString([point_u, point_v])
    
    length = geom.length
    if length < spacing:
        return [geom.interpolate(0.5, normalized=True)]
    
    return [geom.interpolate(i * spacing) for i in range(1, int(length // spacing))]


def fetch_street_view_image(lat, lng, heading, api_key=API_KEY):
    """Fetch image bytes from Google Street View Static API."""
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        'size': '640x640',
        'location': f'{lat},{lng}',
        'fov': 90,
        'heading': heading,
        'pitch': 0,
        'key': api_key
    }
    response = requests.get(base_url, params=params)
    return response.content if response.status_code == 200 else None


def save_image(image_bytes, save_path):
    """Save image to disk."""
    with open(save_path, 'wb') as f:
        f.write(image_bytes)


def process_graph_and_download(G, sample_spacing=SPACING_METERS):
    """Main function to fetch and save street view images for each edge."""
    for u, v, k, data in G.edges(keys=True, data=True):
        segment_id = f"{u}_{v}_{k}"
        segment_folder = os.path.join(IMAGE_DIR, segment_id)
        os.makedirs(segment_folder, exist_ok=True)

        try:
            points = sample_points_on_edge(u, v, k, data, spacing=sample_spacing)
        except Exception as e:
            print(f"Failed sampling for edge {segment_id}: {e}")
            continue

        for i, point in enumerate(points):
            lat, lng = point.y, point.x
            for heading in HEADINGS:
                image_id = f"{i}_{heading}"
                filename = f"{image_id}.jpg"
                save_path = os.path.join(segment_folder, filename)

                if os.path.exists(save_path):
                    continue  # skip if already exists

                try:
                    img_bytes = fetch_street_view_image(lat, lng, heading)
                    if img_bytes:
                        save_image(img_bytes, save_path)

                        # Append to metadata
                        with open(METADATA_FILE, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                segment_id, u, v, k, i, lat, lng,
                                heading, save_path
                            ])
                        time.sleep(0.1)  # slight delay to avoid rate limits
                    else:
                        print(f"Image not found at {lat}, {lng}")
                except Exception as e:
                    print(f"Error fetching image for {segment_id} at {lat},{lng}: {e}")
                    continue
