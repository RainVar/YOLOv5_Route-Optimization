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

    # Add speeds and travel times
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    print("Added edge speeds and travel times.")

    # Elevation from SRTM raster in data/
    srtm_path = 'data/srtm.tif'
    if os.path.exists(srtm_path):
        try:
            print("Adding elevation using SRTM raster...")
            graph = ox.elevation.add_node_elevations_raster(graph, filepath=srtm_path, cpus=1)
            print("Successfully added elevation data to nodes.")
        except Exception as e:
            print(f"Error loading elevation: {e}")
    else:
        print("Elevation raster not found at data/srtm.tif. Skipping elevation.")

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

    for v, u, k, rev_data in edges_to_add:
        graph.add_edge(v, u, key=k, **rev_data)

    print("Processed edge attributes and ensured bidirectionality.")
    return graph

if __name__ == "__main__":
    center_point = (10.299848, 123.871968)  # Tisa, Cebu City
    os.makedirs('data', exist_ok=True)
    try:
        road_network = build_road_network(center_point, dist=200)
        if road_network is not None:
            ox.save_graphml(road_network, filepath=os.path.join('data', 'road_network.graphml'))
            print("Graph saved to data/road_network.graphml")
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
