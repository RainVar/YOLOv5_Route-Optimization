import os
import csv
import time
import requests
from shapely.geometry import LineString
import osmnx as ox
import networkx as nx
from dotenv import load_dotenv  # Add this import

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# CONFIGURATION
# -----------------------------
# The .env file should contain a line: GOOGLE_API_KEY=your_actual_key_here
API_KEY = os.getenv("GOOGLE_API_KEY")
IMAGE_DIR = os.path.join("data", "road_images")        # Directory to store images
METADATA_FILE = os.path.join("data", "image_metadata.csv")  # CSV to store image metadata
SPACING_METERS = 10              # Distance between sampled points on each edge
HEADINGS = [0]                   # List of headings (can be [0, 90, 180, 270] for more coverage)

# Ensure the data and image directories exist
os.makedirs("data", exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Metadata CSV header
CSV_HEADER = [
    "segment_id", "u", "v", "k", "index", "lat", "lng", "heading", "image_path"
]

# Initialize metadata file if it doesn't exist
if not os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)


def sample_points_on_edge(u, v, k, data, spacing=10):
    """
    Samples points along the edge geometry at fixed spacing (in meters).
    Returns a list of shapely Point objects.
    """
    geom = data.get('geometry')
    if not geom:
        # If no geometry, use a straight line between nodes
        point_u = (data['x'], data['y'])
        point_v = (data['x'], data['y'])
        geom = LineString([point_u, point_v])
    length = geom.length
    if length < spacing:
        return [geom.interpolate(0.5, normalized=True)]
    return [geom.interpolate(i * spacing) for i in range(1, int(length // spacing))]


def fetch_street_view_image(lat, lng, heading, api_key=API_KEY):
    """
    Fetch image bytes from Google Street View Static API for a given lat/lng/heading.
    Returns image bytes if successful, else None.
    """
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
    """
    Save image bytes to disk at the specified path.
    """
    with open(save_path, 'wb') as f:
        f.write(image_bytes)


def process_graph_and_download(G, sample_spacing=SPACING_METERS):
    """
    For each edge in the OSMnx graph G, sample points, download images, and record metadata.
    Images are stored in road_images/{segment_id}/{index}_{heading}.jpg
    Metadata is appended to image_metadata.csv
    """
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
                    # Skip if image already exists
                    continue

                try:
                    img_bytes = fetch_street_view_image(lat, lng, heading)
                    if img_bytes:
                        save_image(img_bytes, save_path)
                        # Append metadata row
                        with open(METADATA_FILE, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                segment_id, u, v, k, i, lat, lng, heading, save_path
                            ])
                        time.sleep(0.1)  # Delay to avoid API rate limits
                    else:
                        print(f"Image not found at {lat}, {lng}, heading {heading}")
                except Exception as e:
                    print(f"Error fetching image for {segment_id} at {lat},{lng}, heading {heading}: {e}")
                    continue

# Example usage (uncomment and provide your own OSMnx graph):
# G = ox.load_graphml('data/road_network.graphml')
# process_graph_and_download(G)
