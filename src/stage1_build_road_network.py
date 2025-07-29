"""
Stage 1: Build Road Network
---------------------------
Downloads and processes a road network graph from OpenStreetMap using OSMnx.
Adds speed, travel time, and elevation data from an SRTM raster.
Ensures bidirectionality and computes elevation gain for each edge.

Configuration is set via constants at the top of the script.
"""

import os
import osmnx as ox
import requests
from dotenv import load_dotenv

# -----------------------------
# CONFIGURATION
# -----------------------------
CENTER_POINT = (10.299848, 123.871968)  # (lat, lon) for Tisa, Cebu City
DIST = 200                              # Distance in meters for graph radius
SRTM_PATH = 'data/srtm.tif'             # Path to SRTM raster for elevation
OUTPUT_PATH = os.path.join('data', 'road_network.graphml')  # Output GraphML file

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def fetch_elevations_google(coords, api_key):
    """
    Fetch elevations for a list of (lat, lon) tuples using Google Elevation API.
    Returns a list of elevations (meters).
    """
    elevations = []
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    # Google API allows up to 512 locations per request
    BATCH_SIZE = 100  # Stay well below the limit
    for i in range(0, len(coords), BATCH_SIZE):
        batch = coords[i:i+BATCH_SIZE]
        locations = "|".join([f"{lat},{lon}" for lat, lon in batch])
        params = {"locations": locations, "key": api_key}
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "OK":
                elevations.extend([result["elevation"] for result in data["results"]])
            else:
                print(f"Google Elevation API error: {data.get('status')}")
                elevations.extend([0] * len(batch))
        else:
            print(f"HTTP error from Google Elevation API: {resp.status_code}")
            elevations.extend([0] * len(batch))
    return elevations


def build_road_network(center_point, dist=800, srtm_path=None):
    """
    Downloads and builds a road network graph for a given center point and distance.
    Adds edge speeds, travel times, and elevation data from Google Elevation API.
    Ensures bidirectionality and computes elevation gain for each edge.
    """
    print(f"Building road network for point: {center_point} with dist={dist}m")
    try:
        # Download road network from OSM
        graph = ox.graph_from_point(center_point, dist=dist, network_type='drive')
    except Exception as e:
        print("Error: Could not download road network. Check the coordinates or your internet connection.")
        raise e

    if len(graph.nodes) == 0:
        print("Error: Found no graph nodes within the requested area. Try a broader location or larger distance.")
        return None

    print("Downloaded road network.")

    # Add speeds and travel times to edges
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    print("Added edge speeds and travel times.")

    # Add elevation from Google Elevation API
    print("Adding elevation using Google Elevation API...")
    node_coords = [(data['y'], data['x']) for node, data in graph.nodes(data=True)]
    elevations = fetch_elevations_google(node_coords, GOOGLE_API_KEY)
    for (node, data), elev in zip(graph.nodes(data=True), elevations):
        data['elevation'] = elev
    print("Successfully added elevation data to nodes.")

    # Make graph bidirectional manually and compute elevation gain
    edges_to_add = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        elev_u = graph.nodes[u].get('elevation', 0)
        elev_v = graph.nodes[v].get('elevation', 0)
        try:
            elev_u = float(elev_u)
            elev_v = float(elev_v)
            gain = elev_v - elev_u
        except (TypeError, ValueError):
            gain = 0
        data['distance'] = data.get('length', 0)
        data['elevation_gain'] = gain
        # If reverse edge does not exist, add it
        if not graph.has_edge(v, u, k):
            rev_data = data.copy()
            rev_data['elevation_gain'] = -gain
            edges_to_add.append((v, u, k, rev_data))

    # Add missing reverse edges
    for v, u, k, rev_data in edges_to_add:
        graph.add_edge(v, u, key=k, **rev_data)

    print("Processed edge attributes and ensured bidirectionality.")
    return graph

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    try:
        road_network = build_road_network(CENTER_POINT, dist=DIST, srtm_path=SRTM_PATH)
        if road_network is not None:
            ox.save_graphml(road_network, filepath=OUTPUT_PATH)
            print(f"Graph saved to {OUTPUT_PATH}")
            # Display node IDs as labels
            import matplotlib.pyplot as plt
            import networkx as nx
            fig, ax = ox.plot_graph(road_network, edge_linewidth=1, edge_color="gray", show=False, close=False)
            pos = {node: (data['x'], data['y']) for node, data in road_network.nodes(data=True)}
            node_labels = {node: str(node) for node in road_network.nodes()}
            nx.draw_networkx_labels(road_network, pos=pos, labels=node_labels, ax=ax, font_size=8, font_color="white")
            plt.show()
    except Exception as e:
        print(f"Error: {e}")
