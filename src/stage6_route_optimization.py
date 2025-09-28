"""
Stage 6: Route Optimization POC Application
------------------------------------------
Interactive Proof of Concept application that demonstrates the integration of
computer vision-based pavement assessment with route optimization.

This application allows users to:
- Select start and end nodes interactively
- Choose from multiple pathfinding algorithms
- Create manual routes
- Analyze routes comprehensively
- Compare different routing approaches
- Export results for further analysis

Author: YOLOv5 Route Optimization System
"""

import os
import json
import csv
import networkx as nx
import osmnx as ox
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import statistics
import math
import time

class RouteOptimizationApp:
    """
    Interactive Route Optimization Application

    This POC demonstrates the value of CV-based pavement assessment
    by providing multiple routing algorithms and comprehensive analysis.
    """

    def __init__(self, graph_path: str = "../data/updated_road_network.graphml"):
        """
        Initialize the Route Optimization Application

        Args:
            graph_path: Path to the graph with PASER scores
        """
        self.graph = None
        self.current_route = None
        self.route_history = []
        self.predefined_routes = {}

        print("🚀 Initializing Route Optimization POC Application...")
        self._ensure_stage5_completed()
        self._load_graph(graph_path)
        self._setup_predefined_routes()
        print("✅ Application initialized successfully!")

    def _ensure_stage5_completed(self) -> None:
        """Ensure Stage 5 has been completed and required data files exist"""
        print("🔗 Checking Stage 5 dependencies...")

        # Check if Stage 5 output exists
        updated_graph_path = "../data/updated_road_network.graphml"
        if not os.path.exists(updated_graph_path):
            print("❌ Stage 5 not completed. Updated road network not found.")
            print("   python src/stage5_update_graph.py")
            raise FileNotFoundError(f"Stage 5 output not found: {updated_graph_path}")

        # Also check if the original dependencies exist
        road_network_path = "../data/road_network.graphml"
        if not os.path.exists(road_network_path):
            print("⚠️ Warning: Original road network not found")
            print(f"   Expected at: {road_network_path}")

        if not os.path.exists(paser_scores_path):
            print("⚠️ Warning: PASER scores not found")
            print(f"   Expected at: {paser_scores_path}")

        print("✅ Stage 5 dependencies verified")

    def _load_graph(self, graph_path: str) -> None:
        """Load the road network graph with PASER scores"""
        try:
            if not os.path.exists(graph_path):
                raise FileNotFoundError(f"Graph file not found: {graph_path}")

            print(f"📊 Loading graph from {graph_path}...")
            self.graph = ox.load_graphml(graph_path)

            # Print basic statistics
            n_nodes = len(self.graph.nodes())
            n_edges = len(self.graph.edges())
            print(f"✅ Graph loaded: {n_nodes} nodes, {n_edges} edges")

            # Validate PASER scores are present
            edges_with_paser = 0
            for u, v, k, data in self.graph.edges(keys=True, data=True):
                if 'paser_score' in data:
                    edges_with_paser += 1

            print(f"✅ PASER scores found on {edges_with_paser}/{n_edges} edges")

        except Exception as e:
            raise

    def _setup_predefined_routes(self) -> None:
        """Setup some predefined routes for demonstration"""
        self.predefined_routes = {
            "demo_route_1": {
                "name": "Demo Route 1",
                "description": "Sample route for testing",
                "start_node": None,  # Will be set dynamically
                "end_node": None,    # Will be set dynamically
            },
            "demo_route_2": {
                "name": "Demo Route 2",
                "description": "Another sample route for testing",
                "start_node": None,
                "end_node": None,
            },
            "demo_route_3": {
                "name": "Demo Route 3",
                "description": "Third sample route for testing",
                "start_node": None,
                "end_node": None,
            }
        }

        # Try to find some representative nodes
        self._find_representative_nodes()

    def _find_representative_nodes(self) -> None:
        """Find representative nodes for predefined routes"""
        try:
            # Get some nodes with high connectivity (intersections)
            node_degrees = dict(self.graph.degree())
            high_degree_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)

            if len(high_degree_nodes) >= 6:
                # Assign nodes to predefined routes
                nodes = [node for node, degree in high_degree_nodes[:6]]

                self.predefined_routes["demo_route_1"]["start_node"] = nodes[0]
                self.predefined_routes["demo_route_1"]["end_node"] = nodes[1]

                self.predefined_routes["demo_route_2"]["start_node"] = nodes[2]
                self.predefined_routes["demo_route_2"]["end_node"] = nodes[3]

                self.predefined_routes["demo_route_3"]["start_node"] = nodes[4]
                self.predefined_routes["demo_route_3"]["end_node"] = nodes[5]

                print("✅ Representative nodes assigned to predefined routes")

        except Exception as e:
            print(f"⚠️ Could not assign representative nodes: {e}")

    def get_available_nodes(self) -> List[int]:
        """Get list of available node IDs for selection"""
        return list(self.graph.nodes())

    def validate_node(self, node_id) -> bool:
        """Check if a node ID exists in the graph"""
        return node_id in self.graph.nodes()

    def display_menu(self) -> None:
        """Display the main application menu"""
        print("\n" + "="*60)
        print("🚴 CYCLING ROUTE OPTIMIZATION - PROOF OF CONCEPT")
        print("="*60)
        print("🧠 Powered by Computer Vision Pavement Assessment")
        print("="*60)
        print()
        print("📋 MENU OPTIONS:")
        print("1. Select Start & End Nodes")
        print("2. Run Pathfinding Algorithms")
        print("3. Create Manual Route")
        print("4. Analyze Current Route")
        print("5. Compare Multiple Routes")
        print("6. View Route History")
        print("7. Export Results")
        print("8. Predefined Routes Demo")
        print("9. Exit")
        print()
        print("Current Route:", self.current_route["name"] if self.current_route else "None")
        print("="*60)

    def select_start_end_nodes(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Interactive node selection interface

        Returns:
            Tuple of (start_node, end_node) or (None, None) if cancelled
        """
        print("\n🔍 NODE SELECTION")
        print("-" * 30)

        # Display options
        print("1. Manual Node ID Entry")
        print("2. Choose from Predefined Routes")
        print("3. Show Available Nodes")
        print("4. Cancel")

        while True:
            try:
                choice = input("\nSelect option (1-4): ").strip()

                if choice == "1":
                    return self._manual_node_selection()
                elif choice == "2":
                    return self._predefined_route_selection()
                elif choice == "3":
                    self._show_available_nodes()
                    continue
                elif choice == "4":
                    return None, None
                else:
                    print("❌ Invalid choice. Please try again.")

            except KeyboardInterrupt:
                print("\n⚠️ Operation cancelled.")
                return None, None
            except Exception as e:
                print(f"❌ Error: {e}")

    def _manual_node_selection(self) -> Tuple[Optional[int], Optional[int]]:
        """Manual node ID entry"""
        print("\n📝 MANUAL NODE SELECTION")
        print("Enter node IDs (must exist in the graph)")

        while True:
            try:
                start_input = input("Start node ID: ").strip()
                end_input = input("End node ID: ").strip()

                if not start_input or not end_input:
                    print("❌ Both start and end nodes are required.")
                    continue

                start_node = int(start_input)
                end_node = int(end_input)

                if not self.validate_node(start_node):
                    print(f"❌ Start node {start_node} not found in graph.")
                    continue

                if not self.validate_node(end_node):
                    print(f"❌ End node {end_node} not found in graph.")
                    continue

                if start_node == end_node:
                    print("❌ Start and end nodes must be different.")
                    continue

                print(f"✅ Selected route: {start_node} → {end_node}")
                return start_node, end_node

            except ValueError:
                print("❌ Please enter valid integer node IDs.")
            except KeyboardInterrupt:
                return None, None

    def _predefined_route_selection(self) -> Tuple[Optional[int], Optional[int]]:
        """Choose from predefined routes"""
        print("\n🗺️ PREDEFINED ROUTES")
        print("-" * 25)

        for i, (key, route) in enumerate(self.predefined_routes.items(), 1):
            start = route["start_node"]
            end = route["end_node"]
            if start and end:
                print(f"{i}. {route['name']}")
                print(f"   {route['description']}")
                print(f"   Route: {start} → {end}")
            else:
                print(f"{i}. {route['name']} (nodes not available)")

        print(f"{len(self.predefined_routes) + 1}. Cancel")

        while True:
            try:
                choice = input(f"\nSelect route (1-{len(self.predefined_routes) + 1}): ").strip()

                if choice == str(len(self.predefined_routes) + 1):
                    return None, None

                route_num = int(choice) - 1
                if 0 <= route_num < len(self.predefined_routes):
                    route_key = list(self.predefined_routes.keys())[route_num]
                    route = self.predefined_routes[route_key]

                    if route["start_node"] and route["end_node"]:
                        print(f"✅ Selected: {route['name']}")
                        return route["start_node"], route["end_node"]
                    else:
                        print("❌ This route is not available (nodes not found).")
                        continue
                else:
                    print("❌ Invalid choice.")

            except ValueError:
                print("❌ Please enter a valid number.")
            except KeyboardInterrupt:
                return None, None

    def _show_available_nodes(self, limit: int = 20) -> None:
        """Show some available nodes for reference"""
        print(f"\n📋 SAMPLE AVAILABLE NODES (showing {limit} of {len(self.graph.nodes())})")
        print("-" * 50)

        nodes = list(self.graph.nodes())[:limit]
        for i, node in enumerate(nodes, 1):
            print(f"{i:2d}. {node}")

        print("...")
        print("💡 Use these node IDs when selecting routes manually")

    def run_pathfinding_algorithms(self, start_node: int, end_node: int) -> Dict:
        """
        Run multiple pathfinding algorithms and return results

        Args:
            start_node: Starting node ID
            end_node: Ending node ID

        Returns:
            Dictionary with results from all algorithms
        """
        print(f"\n🔍 RUNNING PATHFINDING ALGORITHMS")
        print(f"From: {start_node} → To: {end_node}")
        print("-" * 40)

        results = {}

        # Algorithm 1: Shortest Path (Distance)
        print("1️⃣ Running Shortest Path (Distance-based)...")
        try:
            shortest_path = self._shortest_path_algorithm(start_node, end_node)
            results["shortest_path"] = {
                "name": "Shortest Path (Distance)",
                "path": shortest_path,
                "algorithm": "Dijkstra (distance only)"
            }
            print(f"   ✅ Found path with {len(shortest_path)} nodes")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results["shortest_path"] = None

        # Algorithm 2: Pavement Optimized
        print("2️⃣ Running Pavement-Optimized Path...")
        try:
            pavement_path = self._pavement_optimized_path(start_node, end_node)
            results["pavement_optimized"] = {
                "name": "Pavement-Optimized Path",
                "path": pavement_path,
                "algorithm": "Dijkstra (inverted PASER scores)"
            }
            print(f"   ✅ Found path with {len(pavement_path)} nodes")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results["pavement_optimized"] = None

        # Algorithm 3: Multi-Criteria Optimization
        print("3️⃣ Running Multi-Criteria Optimization...")
        try:
            # Custom weights for pavement (60%), elevation (30%), distance (10%)
            weights = {"paser": 0.6, "elevation": 0.3, "distance": 0.1}
            multi_path = self._multi_criteria_path(start_node, end_node, weights)
            results["multi_criteria"] = {
                "name": "Multi-Criteria Optimized",
                "path": multi_path,
                "algorithm": "Dijkstra (weighted: PASER 60%, Elevation 30%, Distance 10%)",
                "weights": weights
            }
            print(f"   ✅ Found path with {len(multi_path)} nodes")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results["multi_criteria"] = None

        return results

    def _shortest_path_algorithm(self, start: int, end: int) -> List[int]:
        """Traditional shortest path by distance"""
        try:
            # Use NetworkX shortest path with distance as weight
            path = nx.shortest_path(
                self.graph,
                source=start,
                target=end,
                weight='length'  # Use edge length (distance)
            )
            return path
        except nx.NetworkXNoPath:
            raise ValueError("No path found between nodes")
        except Exception as e:
            raise Exception(f"Shortest path algorithm failed: {e}")

    def _pavement_optimized_path(self, start: int, end: int) -> List[int]:
        """Path optimized for pavement quality (using inverted PASER scores)"""
        try:
            # Use inverted PASER scores as cost (lower PASER = higher cost)
            path = nx.shortest_path(
                self.graph,
                source=start,
                target=end,
                weight='inverted_paser'  # Use inverted PASER as cost
            )
            return path
        except nx.NetworkXNoPath:
            raise ValueError("No path found between nodes")
        except Exception as e:
            raise Exception(f"Pavement optimization failed: {e}")

    def _multi_criteria_path(self, start: int, end: int, weights: Dict) -> List[int]:
        """Multi-criteria path optimization"""
        try:
            # Create composite weight for each edge
            for u, v, k, data in self.graph.edges(keys=True, data=True):
                # Normalize components and apply weights
                distance = data.get('length', 100) / 1000  # Normalize to km
                elevation = abs(data.get('elevation_gain', 0))  # Absolute elevation change
                paser_cost = data.get('inverted_paser', 6)  # Inverted PASER cost

                # Composite cost
                composite_cost = (
                    weights.get('distance', 0.1) * distance +
                    weights.get('elevation', 0.1) * elevation +
                    weights.get('paser', 0.8) * paser_cost
                )

                data['composite_cost'] = composite_cost

            # Find path using composite cost
            path = nx.shortest_path(
                self.graph,
                source=start,
                target=end,
                weight='composite_cost'
            )

            return path

        except nx.NetworkXNoPath:
            raise ValueError("No path found between nodes")
        except Exception as e:
            raise Exception(f"Multi-criteria optimization failed: {e}")

    def create_manual_route(self) -> Optional[List[int]]:
        """
        Interactive manual route creation

        Returns:
            List of node IDs for the route, or None if cancelled
        """
        print("\n✏️ MANUAL ROUTE CREATION")
        print("-" * 30)
        print("Enter node IDs in sequence to create a custom route.")
        print("Enter 'done' when finished, 'cancel' to abort.")

        route = []

        while True:
            try:
                node_input = input("Add node (or 'done'/'cancel'): ").strip().lower()

                if node_input == 'cancel':
                    return None
                elif node_input == 'done':
                    if len(route) < 2:
                        print("❌ Route must have at least 2 nodes.")
                        continue
                    break
                else:
                    try:
                        node_id = int(node_input)
                        if not self.validate_node(node_id):
                            print(f"❌ Node {node_id} not found in graph.")
                            continue

                        if route and node_id == route[-1]:
                            print("❌ Cannot add the same node twice in a row.")
                            continue

                        route.append(node_id)
                        print(f"✅ Added node {node_id} (route length: {len(route)})")

                    except ValueError:
                        print("❌ Please enter a valid node ID.")

            except KeyboardInterrupt:
                return None

        # Validate that the route forms a valid path
        if not self._validate_manual_route(route):
            print("❌ The entered route is not a valid path in the graph.")
            return None

        print(f"✅ Manual route created with {len(route)} nodes")
        return route

    def _validate_manual_route(self, route: List[int]) -> bool:
        """Validate that a manual route forms valid edges in the graph"""
        for i in range(len(route) - 1):
            start_node = route[i]
            end_node = route[i + 1]

            # Check if there's an edge between consecutive nodes
            if not self.graph.has_edge(start_node, end_node):
                return False

        return True

    def analyze_route(self, route_data: Dict, route_name: str = "Current Route") -> Dict:
        """
        Comprehensive analysis of a route

        Args:
            route_data: Route information from algorithms
            route_name: Name for the route

        Returns:
            Detailed analysis dictionary
        """
        print(f"\n📊 ANALYZING ROUTE: {route_name}")
        print("-" * 40)

        path = route_data["path"]
        if not path or len(path) < 2:
            return {"error": "Invalid route - no path found"}

        analysis = {
            "route_name": route_name,
            "algorithm": route_data.get("algorithm", "Unknown"),
            "path_length": len(path),
            "total_distance_m": 0.0,
            "elevation_gain_m": 0.0,
            "paser_scores": [],
            "segment_details": []
        }

        # Analyze each segment
        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]

            # Get edge data
            edge_data = self.graph.get_edge_data(start_node, end_node)

            if edge_data:
                # Get the first edge (if multiple)
                edge_key = list(edge_data.keys())[0]
                edge_attrs = edge_data[edge_key]

                # Distance
                distance = edge_attrs.get('length', 0)
                analysis["total_distance_m"] += distance

                # Elevation
                elevation = edge_attrs.get('elevation_gain', 0)
                analysis["elevation_gain_m"] += abs(elevation)

                # PASER score
                paser_score = edge_attrs.get('paser_score', 5.0)
                analysis["paser_scores"].append(paser_score)

                # Segment details
                segment = {
                    "start_node": start_node,
                    "end_node": end_node,
                    "distance_m": distance,
                    "elevation_m": elevation,
                    "paser_score": paser_score,
                    "segment_index": i + 1
                }
                analysis["segment_details"].append(segment)

        # Calculate derived metrics
        if analysis["paser_scores"]:
            analysis["average_paser_score"] = statistics.mean(analysis["paser_scores"])
            analysis["min_paser_score"] = min(analysis["paser_scores"])
            analysis["max_paser_score"] = max(analysis["paser_scores"])
            analysis["paser_std_dev"] = statistics.stdev(analysis["paser_scores"]) if len(analysis["paser_scores"]) > 1 else 0

        # Calculate efficiency metrics
        if analysis["total_distance_m"] > 0:
            analysis["elevation_per_km"] = analysis["elevation_gain_m"] / (analysis["total_distance_m"] / 1000)
            analysis["distance_per_segment"] = analysis["total_distance_m"] / (len(path) - 1)

        # Store route for history
        self.current_route = {
            "name": route_name,
            "data": route_data,
            "analysis": analysis
        }

        # Print summary
        self._print_route_summary(analysis)

        return analysis

    def _print_route_summary(self, analysis: Dict) -> None:
        """Print a formatted summary of route analysis"""
        print("📈 ROUTE ANALYSIS SUMMARY")
        print(f"Route Name: {analysis['route_name']}")
        print(f"Algorithm: {analysis['algorithm']}")
        print(f"Path Length: {analysis['path_length']} nodes")
        print(f"Total Distance: {analysis['total_distance_m']:.1f} meters ({analysis['total_distance_m']/1000:.2f} km)")
        print(f"Total Elevation Gain: {analysis['elevation_gain_m']:.1f} meters")
        print(f"Average PASER Score: {analysis['average_paser_score']:.2f}" if 'average_paser_score' in analysis else "N/A")
        print(f"Best Pavement Segment: {analysis['max_paser_score']:.2f}" if 'max_paser_score' in analysis else "N/A")
        print(f"Worst Pavement Segment: {analysis['min_paser_score']:.2f}" if 'min_paser_score' in analysis else "N/A")

        if 'elevation_per_km' in analysis:
            print(f"Efficiency: {analysis['elevation_per_km']:.1f}m elevation per km")

    def compare_routes(self, routes: List[Dict]) -> None:
        """
        Compare multiple routes side by side

        Args:
            routes: List of route analysis dictionaries
        """
        print("\n⚖️ ROUTE COMPARISON")
        print("-" * 60)

        if len(routes) < 2:
            print("❌ Need at least 2 routes to compare")
            return

        # Create comparison table
        print(f"{'Route Name':<25} {'Distance (km)':<15} {'Elev (m)':<12} {'Avg PASER':<12} {'Efficiency':<15}")
        print("-" * 85)

        for route in routes:
            if "error" in route:
                continue

            name = route["route_name"][:23]  # Truncate long names
            distance_km = route["total_distance_m"] / 1000
            elevation = route["elevation_gain_m"]
            avg_paser = route.get("average_paser_score", 0)
            efficiency = route.get("elevation_per_km", 0)

            print(f"{name:<25} {distance_km:<15.2f} {elevation:<12.1f} {avg_paser:<12.2f} {efficiency:<15.1f}")

        print("-" * 85)
        print("\n💡 TIP: Lower elevation and higher PASER scores indicate better cycling routes!")

    def export_results(self, filename: str = None) -> None:
        """Export current route and analysis to JSON file"""
        if not self.current_route:
            print("❌ No current route to export")
            return

        if not filename:
            timestamp = str(int(time.time()))
            filename = f"route_analysis_{timestamp}.json"

        try:
            export_data = {
                "route": self.current_route["data"],
                "analysis": self.current_route["analysis"],
                "export_timestamp": str(int(time.time())),
                "app_version": "Route Optimization POC v1.0"
            }

            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)

            print(f"✅ Route exported to {filename}")

        except Exception as e:
            print(f"❌ Export failed: {e}")

    def run_demo(self) -> None:
        """Run a demonstration of the application's capabilities"""
        print("\n🎬 RUNNING DEMONSTRATION")
        print("-" * 40)

        # Try to run demo with predefined routes
        demo_routes = []

        for route_key, route_info in self.predefined_routes.items():
            start = route_info["start_node"]
            end = route_info["end_node"]

            if start and end:
                print(f"\n🧪 Testing {route_info['name']}...")
                try:
                    results = self.run_pathfinding_algorithms(start, end)

                    # Analyze each successful algorithm
                    for alg_name, result in results.items():
                        if result and result["path"]:
                            analysis = self.analyze_route(result, f"{route_info['name']} - {result['name']}")
                            demo_routes.append(analysis)

                except Exception as e:
                    print(f"   ❌ Demo failed for {route_info['name']}: {e}")

        # Compare all demo routes
        if len(demo_routes) >= 2:
            print("\n📊 DEMONSTRATION RESULTS")
            self.compare_routes(demo_routes)

        print("\n✅ Demonstration complete!")
    def show_history(self) -> None:
        """Show route history"""
        if not self.route_history:
            print("📚 No routes in history")
            return

        print("\n📚 ROUTE HISTORY")
        print("-" * 40)

        for i, route in enumerate(self.route_history, 1):
            print(f"{i}. {route['name']}")
            print(f"   Algorithm: {route['algorithm']}")
            print(f"   Distance: {route['total_distance_m']/1000:.2f} km")
            print(f"   Avg PASER: {route.get('average_paser_score', 'N/A')}")
            print()

    def interactive_mode(self) -> None:
        """Run the application in interactive mode"""
        print("🚀 Welcome to the Cycling Route Optimization POC!")
        print("This application demonstrates the value of computer vision-based")
        print("pavement assessment for creating better cycling routes.")

        while True:
            try:
                self.display_menu()
                choice = input("Select option (1-9): ").strip()

                if choice == "1":
                    start, end = self.select_start_end_nodes()
                    if start and end:
                        print(f"Selected: {start} → {end}")
                elif choice == "2":
                    # This would need start/end nodes to be set first
                    print("Please select start/end nodes first (option 1)")
                elif choice == "3":
                    manual_route = self.create_manual_route()
                    if manual_route:
                        print(f"Manual route with {len(manual_route)} nodes created")
                elif choice == "4":
                    # This would need a route to be set first
                    print("Please create or generate a route first")
                elif choice == "5":
                    print("Please generate multiple routes first (option 2)")
                elif choice == "6":
                    self.show_history()
                elif choice == "7":
                    filename = input("Enter filename (or press Enter for auto): ").strip()
                    self.export_results(filename if filename else None)
                elif choice == "8":
                    self.run_demo()
                elif choice == "9":
                    print("👋 Thank you for using the Route Optimization POC!")
                    break
                else:
                    print("❌ Invalid option. Please try again.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function to run the Route Optimization POC"""
    try:
        # Initialize the application
        app = RouteOptimizationApp()

        # Run in interactive mode
        app.interactive_mode()

    except Exception as e:
        print(f"❌ Application failed to start: {e}")


if __name__ == "__main__":
    main()