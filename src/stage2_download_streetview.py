import os
import csv
import time
import requests
from shapely.geometry import LineString
import osmnx as ox
import networkx as nx
from dotenv import load_dotenv
import math

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
PITCH = -20                      # Camera pitch for pavement assessment (negative looks downward)
OVERWRITE = False                # If True, overwrite existing images

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


def sample_points_on_edge(G, u, v, k, data, spacing=10):
    """
    Samples points along the edge geometry at fixed spacing (in meters).
    Returns a list of shapely Point objects.
    """
    geom = data.get('geometry')
    if not geom:
        # If no geometry, use a straight line between node coordinates
        point_u = (G.nodes[u]['x'], G.nodes[u]['y'])
        point_v = (G.nodes[v]['x'], G.nodes[v]['y'])
        geom = LineString([point_u, point_v])
    length = geom.length
    if length < spacing:
        return [geom.interpolate(0.5, normalized=True)]
    return [geom.interpolate(i * spacing) for i in range(1, int(length // spacing))]


def fetch_street_view_image(lat, lng, heading, api_key=API_KEY, pitch=PITCH):
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
        'pitch': pitch,
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


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing between two points (in degrees, 0 = north, clockwise).
    """
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    diff_long = math.radians(lon2 - lon1)
    x = math.sin(diff_long) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (
        math.sin(lat1) * math.cos(lat2) * math.cos(diff_long)
    )
    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing


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
            points = sample_points_on_edge(G, u, v, k, data, spacing=sample_spacing)
        except Exception as e:
            print(f"Failed sampling for edge {segment_id}: {e}")
            continue

        # For bearing calculation, get all points including endpoints
        geom = data.get('geometry')
        if not geom:
            point_u = (G.nodes[u]['x'], G.nodes[u]['y'])
            point_v = (G.nodes[v]['x'], G.nodes[v]['y'])
            geom = LineString([point_u, point_v])
        full_points = list(geom.coords)
        # If only one point, treat as midpoint between u and v
        if len(points) == 1:
            # Use u and v as endpoints
            endpoints = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
            bearings = [calculate_bearing(*endpoints[0], *endpoints[1])]
        else:
            # For each sampled point, find the closest segment in the geometry and use its direction
            bearings = []
            for pt in points:
                min_dist = float('inf')
                best_bearing = 0
                for i in range(len(full_points) - 1):
                    x1, y1 = full_points[i]
                    x2, y2 = full_points[i + 1]
                    # Distance from pt to segment midpoint
                    midx = (x1 + x2) / 2
                    midy = (y1 + y2) / 2
                    dist = (pt.x - midx) ** 2 + (pt.y - midy) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        best_bearing = calculate_bearing(y1, x1, y2, x2)
                bearings.append(best_bearing)

        for i, (point, heading) in enumerate(zip(points, bearings)):
            lat, lng = point.y, point.x
            image_id = f"{i}_{int(round(heading))}"
            filename = f"{image_id}.jpg"
            save_path = os.path.join(segment_folder, filename)

            if os.path.exists(save_path) and not OVERWRITE:
                # Skip if image already exists and not overwriting
                continue

            try:
                img_bytes = fetch_street_view_image(lat, lng, heading, pitch=PITCH)
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

if __name__ == "__main__":
    import osmnx as ox
    # Load the road network graph
    graph_path = os.path.join("data", "road_network.graphml")
    if not os.path.exists(graph_path):
        print(f"GraphML file not found at {graph_path}. Run stage1 to generate it.")
    else:
        G = ox.load_graphml(graph_path)
        process_graph_and_download(G)
        print("Street View image download and metadata collection complete.")
