"""
Stage 6: Route Optimization using Modified Dijkstra's Algorithm
--------------------------------------------------------------
Implements route optimization that considers pavement quality (proxy PASER scores), 
elevation gain, and distance using a composite weight approach. The algorithm uses 
Rank Order Centroid (ROC) method to prioritize factors and applies modified Dijkstra's 
algorithm for cyclist-friendly route planning.

Configuration is set via constants at the top of the script.
"""

import os
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

# -----------------------------
# CONFIGURATION
# -----------------------------
UPDATED_NETWORK_PATH = "data/updated_road_network.graphml"  # Input: Graph with PASER scores
OUTPUT_PATH = "data/optimized_route.json"                   # Output: Optimized route data

# ROC-based weights (Rank Order Centroid method)
# Based on priority: 1) Pavement condition (PASER), 2) Elevation gain, 3) Distance
ALPHA = 0.611  # Weight for normalized PASER (pavement condition) - highest priority
BETA = 0.278   # Weight for normalized elevation gain - medium priority  
GAMMA = 0.111  # Weight for normalized distance - lowest priority

# Route optimization parameters
START_NODE = None    # Will be set dynamically or by user input
END_NODE = None      # Will be set dynamically or by user input

def load_road_network_with_paser(graph_path: str) -> nx.MultiDiGraph:
    """
    Load the road network graph with PASER scores from Stage 5.
    """
    print(f"Loading road network with PASER scores from {graph_path}")
    
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    
    try:
        graph = ox.load_graphml(graph_path)
        print(f"Loaded graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
        return graph
    except Exception as e:
        raise Exception(f"Error loading graph: {e}")

def calculate_elevation_gain(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Calculate elevation gain for each edge, considering only uphill segments.
    Negative elevation gains (downhill) are set to zero.
    """
    print("Calculating elevation gain for edges...")
    
    for u, v, k, data in graph.edges(keys=True, data=True):
        # Get elevation data from nodes
        elev_u = graph.nodes[u].get('elevation', 0)
        elev_v = graph.nodes[v].get('elevation', 0)
        
        try:
            elev_u = float(elev_u)
            elev_v = float(elev_v)
            
            # Calculate elevation gain (only positive gains considered)
            elevation_gain = max(0, elev_v - elev_u)
            data['elevation_gain'] = elevation_gain
            
        except (TypeError, ValueError):
            # If elevation data is missing or invalid, set to 0
            data['elevation_gain'] = 0.0
    
    print("Calculated elevation gain for all edges")
    return graph

def normalize_edge_attributes(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Normalize inverted PASER scores, elevation gain, and distance using Min-Max normalization.
    """
    print("Normalizing edge attributes...")
    
    # Collect all values for normalization
    paser_values = []
    elev_gain_values = []
    distance_values = []
    
    for u, v, k, data in graph.edges(keys=True, data=True):
        # Convert inverted PASER to float to handle string values from GraphML
        inverted_paser = data.get('inverted_paser', 6.0)
        try:
            inverted_paser = float(inverted_paser)
        except (TypeError, ValueError):
            inverted_paser = 6.0  # Default inverted PASER = 6 (fair)
        
        paser_values.append(inverted_paser)
        elev_gain_values.append(data.get('elevation_gain', 0.0))
        # Handle both 'length' and 'distance' attributes, convert to float
        distance = data.get('length', data.get('distance', 0.0))
        try:
            distance = float(distance)
        except (TypeError, ValueError):
            distance = 0.0
        distance_values.append(distance)
    
    # Calculate min and max values
    min_paser, max_paser = min(paser_values), max(paser_values)
    min_elev, max_elev = min(elev_gain_values), max(elev_gain_values)
    min_dist, max_dist = min(distance_values), max(distance_values)
    
    print(f"Inverted PASER range: {min_paser:.2f} - {max_paser:.2f}")
    print(f"Elevation gain range: {min_elev:.2f} - {max_elev:.2f} meters")
    print(f"Distance range: {min_dist:.2f} - {max_dist:.2f} meters")
    
    # Normalize values for each edge
    for u, v, k, data in graph.edges(keys=True, data=True):
        inverted_paser = data.get('inverted_paser', 6.0)
        # Convert to float to handle string values from GraphML
        try:
            inverted_paser = float(inverted_paser)
        except (TypeError, ValueError):
            inverted_paser = 6.0
            
        elev_gain = data.get('elevation_gain', 0.0)
        distance = data.get('length', data.get('distance', 0.0))
        
        # Convert to float to handle string values
        try:
            distance = float(distance)
        except (TypeError, ValueError):
            distance = 0.0
        
        # Min-Max normalization
        # Inverted PASER: normalize to [0,1] where 0=excellent, 1=worst
        if max_paser > min_paser:
            norm_paser = (inverted_paser - min_paser) / (max_paser - min_paser)
        else:
            norm_paser = 0.0
        
        # Elevation gain: normalize to [0,1] where 0=flat, 1=steepest
        norm_elev = elev_gain / max_elev if max_elev > 0 else 0.0
        
        # Distance: normalize to [0,1] where 0=shortest, 1=longest
        if max_dist > min_dist:
            norm_dist = (distance - min_dist) / (max_dist - min_dist)
        else:
            norm_dist = 0.0
        
        # Store normalized values
        data['norm_paser'] = norm_paser
        data['norm_elev'] = norm_elev
        data['norm_dist'] = norm_dist
    
    print("Completed normalization of edge attributes")
    return graph

def calculate_composite_weights(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Calculate composite weights using ROC-based coefficients.
    composite_weight = α × norm_PASER + β × norm_elev + γ × norm_dist
    where α=0.611, β=0.278, γ=0.111
    """
    print("Calculating composite weights using ROC method...")
    print(f"ROC coefficients: α={ALPHA:.3f} (PASER), β={BETA:.3f} (elevation), γ={GAMMA:.3f} (distance)")
    
    composite_weights = []
    
    for u, v, k, data in graph.edges(keys=True, data=True):
        norm_paser = data.get('norm_paser', 0.0)
        norm_elev = data.get('norm_elev', 0.0)
        norm_dist = data.get('norm_dist', 0.0)
        
        # Calculate composite weight using ROC coefficients
        composite_weight = (ALPHA * norm_paser + 
                          BETA * norm_elev + 
                          GAMMA * norm_dist)
        
        data['composite_weight'] = composite_weight
        composite_weights.append(composite_weight)
    
    # Print statistics
    avg_weight = np.mean(composite_weights)
    min_weight = np.min(composite_weights)
    max_weight = np.max(composite_weights)
    
    print(f"Composite weight statistics:")
    print(f"  Average: {avg_weight:.4f}")
    print(f"  Range: {min_weight:.4f} - {max_weight:.4f}")
    print(f"  Total edges with weights: {len(composite_weights)}")
    
    return graph

def find_optimal_route(graph: nx.MultiDiGraph, start_node: str, end_node: str) -> Tuple[List[str], float]:
    """
    Find the optimal route using modified Dijkstra's algorithm with composite weights.
    """
    print(f"Finding optimal route from {start_node} to {end_node}...")
    
    try:
        # Use NetworkX shortest path with composite weights
        path = nx.shortest_path(graph, start_node, end_node, weight='composite_weight')
        
        # Calculate total route cost
        total_cost = nx.shortest_path_length(graph, start_node, end_node, weight='composite_weight')
        
        print(f"Optimal route found with {len(path)} nodes and total cost: {total_cost:.4f}")
        return path, total_cost
        
    except nx.NetworkXNoPath:
        print(f"No path found between {start_node} and {end_node}")
        return [], float('inf')
    except Exception as e:
        print(f"Error finding route: {e}")
        return [], float('inf')

def analyze_route_composition(graph: nx.MultiDiGraph, path: List[str]) -> dict:
    """
    Analyze the composition of the optimal route in terms of PASER scores, elevation, and distance.
    """
    if len(path) < 2:
        return {}
    
    print("Analyzing route composition...")
    
    total_distance = 0.0
    total_elevation_gain = 0.0
    paser_scores = []
    composite_weights = []
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        
        # Find the edge data (handle multi-edges)
        edge_data = None
        if graph.has_edge(u, v):
            # Get the first edge if multiple exist
            edge_data = graph[u][v][0]
        
        if edge_data:
            # Convert distance to float to handle string values
            distance = edge_data.get('length', edge_data.get('distance', 0.0))
            try:
                distance = float(distance)
            except (TypeError, ValueError):
                distance = 0.0
            
            total_distance += distance
            total_elevation_gain += edge_data.get('elevation_gain', 0.0)
            
            # Convert PASER score to float
            paser_score = edge_data.get('paser_score', 5.0)
            try:
                paser_score = float(paser_score)
            except (TypeError, ValueError):
                paser_score = 5.0
            
            paser_scores.append(paser_score)
            composite_weights.append(edge_data.get('composite_weight', 0.0))
    
    analysis = {
        'total_distance_m': total_distance,
        'total_elevation_gain_m': total_elevation_gain,
        'average_paser_score': np.mean(paser_scores) if paser_scores else 5.0,
        'average_composite_weight': np.mean(composite_weights) if composite_weights else 0.0,
        'num_segments': len(path) - 1
    }
    
    print(f"Route Analysis:")
    print(f"  Total distance: {analysis['total_distance_m']:.2f} meters")
    print(f"  Total elevation gain: {analysis['total_elevation_gain_m']:.2f} meters")
    print(f"  Average PASER score: {analysis['average_paser_score']:.2f}")
    print(f"  Average composite weight: {analysis['average_composite_weight']:.4f}")
    print(f"  Number of segments: {analysis['num_segments']}")
    
    return analysis

def get_route_nodes_interactive(graph: nx.MultiDiGraph) -> Tuple[Optional[str], Optional[str]]:
    """
    Get start and end nodes interactively from user or use predefined nodes.
    """
    nodes = list(graph.nodes())
    
    print(f"Available nodes in the graph: {len(nodes)}")
    print("Sample nodes:", nodes[:10])
    
    # For demonstration, use first and last nodes if not specified
    if len(nodes) >= 2:
        start_node = nodes[0]
        end_node = nodes[-1]
        print(f"Using demo route: {start_node} -> {end_node}")
        return start_node, end_node
    else:
        print("Insufficient nodes in graph for route planning")
        return None, None

def optimize_cycling_route():
    """
    Main function to perform route optimization for cycling.
    """
    print("=== Stage 6: Route Optimization using Modified Dijkstra's Algorithm ===")
    
    try:
        # Load road network with PASER scores (including inverted scores from Stage 5)
        graph = load_road_network_with_paser(UPDATED_NETWORK_PATH)
        
        # Calculate elevation gain for edges
        graph = calculate_elevation_gain(graph)
        
        # Normalize edge attributes
        graph = normalize_edge_attributes(graph)
        
        # Calculate composite weights using ROC method
        graph = calculate_composite_weights(graph)
        
        # Get route start and end points
        start_node, end_node = get_route_nodes_interactive(graph)
        
        if start_node is None or end_node is None:
            print("Cannot proceed without valid start and end nodes")
            return
        
        # Find optimal route
        optimal_path, total_cost = find_optimal_route(graph, start_node, end_node)
        
        if not optimal_path:
            print("No optimal route found")
            return
        
        # Analyze route composition
        route_analysis = analyze_route_composition(graph, optimal_path)
        
        # Save results
        save_route_results(optimal_path, total_cost, route_analysis, start_node, end_node)
        
        print("\n=== Route Optimization Completed Successfully ===")
        
    except Exception as e:
        print(f"Error during route optimization: {e}")

def save_route_results(path: List[str], cost: float, analysis: dict, start: str, end: str):
    """
    Save route optimization results to JSON file.
    """
    import json
    
    results = {
        'route': {
            'start_node': start,
            'end_node': end,
            'path': path,
            'total_cost': cost
        },
        'analysis': analysis,
        'methodology': {
            'weights': {
                'alpha_paser': ALPHA,
                'beta_elevation': BETA,
                'gamma_distance': GAMMA
            },
            'method': 'ROC-based composite weight with modified Dijkstra algorithm using proxy PASER scores'
        }
    }
    
    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Route results saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    optimize_cycling_route()
