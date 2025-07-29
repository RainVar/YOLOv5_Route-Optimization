"""
Stage 5: Update Graph with PASER Scores
---------------------------------------
Updates the road network graph with proxy PASER (Pavement Surface Evaluation and Rating) scores
from the regression inference stage. These scores are added as edge attributes to enable
route optimization that considers pavement quality.

Configuration is set via constants at the top of the script.
"""

import os
import csv
import osmnx as ox
import networkx as nx
from collections import defaultdict

# -----------------------------
# CONFIGURATION
# -----------------------------
ROAD_NETWORK_PATH = "data/road_network.graphml"       # Input: Road network graph
PASER_SCORES_CSV = "data/proxy_paser_scores.csv"     # Input: PASER scores per image
OUTPUT_PATH = "data/updated_road_network.graphml" # Output: Updated graph with PASER scores

def load_paser_scores(csv_path):
    """
    Load PASER scores from CSV and organize by edge segments.
    Returns a dictionary mapping (u, v, k) tuples to PASER scores.
    """
    print(f"Loading PASER scores from {csv_path}")
    paser_scores = defaultdict(list)
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = row['u']
            v = row['v'] 
            k = int(row['k'])
            score = float(row['proxy_paser_score'])
            
            # Group scores by edge (u, v, k)
            edge_key = (u, v, k)
            paser_scores[edge_key].append(score)
    
    # Calculate average score for each edge
    avg_scores = {}
    for edge_key, scores in paser_scores.items():
        avg_scores[edge_key] = sum(scores) / len(scores)
        print(f"Edge {edge_key}: {len(scores)} images, avg PASER score: {avg_scores[edge_key]:.2f}")
    
    print(f"Loaded PASER scores for {len(avg_scores)} edges")
    return avg_scores

def update_graph_with_paser(graph, paser_scores):
    """
    Update the graph with PASER scores as edge attributes.
    Handles both string and integer node IDs.
    """
    print("Updating graph edges with PASER scores...")
    updated_edges = 0
    total_edges = len(graph.edges())
    
    for u, v, k, data in graph.edges(keys=True, data=True):
        # Try both string and integer versions of node IDs
        edge_keys_to_try = [
            (str(u), str(v), k),
            (u, v, k),
            (int(u) if str(u).isdigit() else u, int(v) if str(v).isdigit() else v, k)
        ]
        
        paser_score = None
        for edge_key in edge_keys_to_try:
            if edge_key in paser_scores:
                paser_score = paser_scores[edge_key]
                break
        
        if paser_score is not None:
            data['paser_score'] = paser_score
            updated_edges += 1
        else:
            # Default PASER score for edges without data (neutral/good condition)
            data['paser_score'] = 5.0  # Middle score (1-10 scale, 5 = fair condition)
    
    print(f"Updated {updated_edges}/{total_edges} edges with PASER scores")
    print(f"Remaining {total_edges - updated_edges} edges assigned default score of 5.0")
    return graph

def calculate_weighted_travel_time(graph):
    """
    Calculate weighted travel time based on PASER scores.
    Lower PASER scores (worse pavement) increase travel time.
    """
    print("Calculating weighted travel times based on PASER scores...")
    
    for u, v, k, data in graph.edges(keys=True, data=True):
        base_time = data.get('travel_time', 0)
        paser_score = data.get('paser_score', 5.0)
        
        # Weight factor: worse pavement (lower PASER) increases travel time
        # PASER scale: 1 (very poor) to 10 (excellent)
        # Weight factor ranges from 1.5 (poor) to 1.0 (excellent)
        weight_factor = 2.0 - (paser_score / 10.0)
        weighted_time = base_time * weight_factor
        
        data['weighted_travel_time'] = weighted_time
    
    print("Calculated weighted travel times for all edges")

def update_road_network_with_paser():
    """
    Main function to update the road network with PASER scores.
    """
    # Check if required files exist
    if not os.path.exists(ROAD_NETWORK_PATH):
        print(f"Error: Road network file not found: {ROAD_NETWORK_PATH}")
        print("Please run stage1_build_road_network.py first")
        return
    
    if not os.path.exists(PASER_SCORES_CSV):
        print(f"Error: PASER scores file not found: {PASER_SCORES_CSV}")
        print("Please run stage4_regression_inference.py first")
        return
    
    # Load the road network graph
    print(f"Loading road network from {ROAD_NETWORK_PATH}")
    try:
        graph = ox.load_graphml(ROAD_NETWORK_PATH)
        print(f"Loaded graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    except Exception as e:
        print(f"Error loading road network: {e}")
        return
    
    # Load PASER scores
    try:
        paser_scores = load_paser_scores(PASER_SCORES_CSV)
    except Exception as e:
        print(f"Error loading PASER scores: {e}")
        return
    
    # Update graph with PASER scores
    try:
        graph = update_graph_with_paser(graph, paser_scores)
    except Exception as e:
        print(f"Error updating graph with PASER scores: {e}")
        return
    
    # Calculate weighted travel times
    try:
        calculate_weighted_travel_time(graph)
    except Exception as e:
        print(f"Error calculating weighted travel times: {e}")
        return
    
    # Save the updated graph
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        ox.save_graphml(graph, filepath=OUTPUT_PATH)
        print(f"Updated graph saved to {OUTPUT_PATH}")
        
        # Print summary statistics
        paser_scores_list = [data.get('paser_score', 5.0) for u, v, k, data in graph.edges(keys=True, data=True)]
        avg_paser = sum(paser_scores_list) / len(paser_scores_list)
        min_paser = min(paser_scores_list)
        max_paser = max(paser_scores_list)
        
        print(f"\nPASER Score Statistics:")
        print(f"Average: {avg_paser:.2f}")
        print(f"Range: {min_paser:.2f} - {max_paser:.2f}")
        print(f"Total edges: {len(paser_scores_list)}")
        
    except Exception as e:
        print(f"Error saving updated graph: {e}")
        return

if __name__ == "__main__":
    update_road_network_with_paser()