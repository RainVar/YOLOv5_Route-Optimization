import os
import osmnx as ox

def build_road_network(center_point, dist=800):
    print(f"Building road network for point: {center_point} with dist={dist}m")
    try:
        graph = ox.graph_from_point(center_point, dist=dist, network_type='drive')
    except Exception as e:
        print("Error: Could not download road network. Check the coordinates or your internet connection.")
        raise e
    if len(graph.nodes) == 0:
        print("Error: Found no graph nodes within the requested area. Try a broader location or larger distance.")
        return None
    print("Downloaded road network.")
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    print("Added edge speeds and travel times.")
    srtm_path = 'data/srtm.tif'
    if os.path.exists(srtm_path):
        graph = ox.elevation.add_node_elevations_raster(graph, srtm_path, cpus=1)
        graph = ox.elevation.add_edge_grades(graph)
        print("Added elevation and edge grades.")
    else:
        print("Elevation raster not found. Skipping elevation.")
    for u, v, k, data in graph.edges(keys=True, data=True):
        data['distance'] = data.get('length', 0)
        data['elevation_gain'] = data.get('grade', 0)
    print("Processed edge attributes.")
    return graph

if __name__ == "__main__":
    center_point = (10.299848, 123.871968)  # Approximate coordinates of Tisa, Cebu City
    os.makedirs('data', exist_ok=True)
    try:
        road_network = build_road_network(center_point, dist=1000)
        if road_network is not None:
            ox.save_graphml(road_network, filepath=os.path.join('data', 'road_network.graphml'))
            print("Graph saved to data/road_network.graphml")
            # Visualize the graph
            ox.plot_graph(road_network)
    except Exception as e:
        print(f"Error: {e}")
