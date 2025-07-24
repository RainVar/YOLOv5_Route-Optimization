import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import os

def display_graph(graph, node_label_color="white", edge_color="gray"):
    fig, ax = ox.plot_graph(graph, edge_linewidth=1, edge_color=edge_color, show=False, close=False)
    pos = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}
    node_labels = {node: str(node) for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos=pos, labels=node_labels, ax=ax, font_size=8, font_color=node_label_color)
    plt.show()

if __name__ == "__main__":
    graph_path = os.path.join('data', 'road_network.graphml')
    if not os.path.exists(graph_path):
        print(f"Graph file not found: {graph_path}")
    else:
        try:
            G = ox.load_graphml(graph_path)
            print("Loaded graph from data/road_network.graphml")
            fig, ax = ox.plot_graph(G, edge_linewidth=1, edge_color="gray", show=False, close=False)
            pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
            node_labels = {node: str(node) for node in G.nodes()}
            nx.draw_networkx_labels(G, pos=pos, labels=node_labels, ax=ax, font_size=8, font_color="white")
            plt.show()
        except Exception as e:
            print(f"Error loading or displaying graph: {e}")
