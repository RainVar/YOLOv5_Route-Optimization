"""
Stage 6: Visual Route Optimization POC Application
--------------------------------------------------
Desktop GUI application that demonstrates the integration of
computer vision-based pavement assessment with route optimization.

Features:
- Interactive graph visualization
- Click-to-select start/end nodes
- Multiple pathfinding algorithms
- Visual path comparison
- Real-time metrics display
- Export functionality

Author: YOLOv5 Route Optimization System
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
import networkx as nx
import osmnx as ox
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import json
import time
import statistics
import math
from datetime import datetime

class RouteOptimizationGUI:
    """
    Desktop GUI for Route Optimization POC
    """

    def __init__(self, graph_path: str = "../data/updated_road_network.graphml"):
        """
        Initialize the GUI application

        Args:
            graph_path: Path to the graph with PASER scores
        """
        self.graph = None
        self.node_positions = {}
        self.selected_start = None
        self.selected_end = None
        self.current_paths = {}
        self.route_analyses = {}

        # Colors for different algorithms
        self.colors = {
            'shortest': 'blue',
            'pavement': 'green',
            'multi_criteria': 'red'
        }

        self._load_graph(graph_path)
        self._setup_gui()

    def _load_graph(self, graph_path: str) -> None:
        """Load the road network graph with PASER scores"""
        try:
            print(f"📊 Loading graph from {graph_path}...")
            self.graph = ox.load_graphml(graph_path)

            # Get node positions for plotting
            self.node_positions = {node: (data['x'], data['y'])
                                 for node, data in self.graph.nodes(data=True)}

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
            messagebox.showerror("Error", f"Failed to load graph: {e}")
            raise

    def _setup_gui(self) -> None:
        """Setup the main GUI interface"""
        self.root = tk.Tk()
        self.root.title("🚴 Cycling Route Optimization - Visual POC")
        self.root.geometry("1400x900")
        self.root.state('zoomed')

        # Create main frames
        self._create_menu_bar()
        self._create_main_layout()

        # Initialize plot
        self._initialize_plot()

        # Setup event handlers
        self._setup_event_handlers()

    def _create_menu_bar(self) -> None:
        """Create the menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export Results", command=self._export_results)
        file_menu.add_command(label="Exit", command=self.root.quit)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Reset View", command=self._reset_view)
        view_menu.add_command(label="Toggle Node Labels", command=self._toggle_node_labels)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_main_layout(self) -> None:
        """Create the main layout with panels"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Controls
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        # Right panel - Visualization
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Controls panel
        self._create_controls_panel(left_panel)

        # Visualization panel
        self._create_visualization_panel(right_panel)

    def _create_controls_panel(self, parent) -> None:
        """Create the controls panel"""
        # Title
        title_label = ttk.Label(parent, text="🚴 Route Optimization Controls",
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))

        # Node selection frame
        selection_frame = ttk.LabelFrame(parent, text="Node Selection", padding=10)
        selection_frame.pack(fill=tk.X, pady=(0, 15))

        # Start node
        start_frame = ttk.Frame(selection_frame)
        start_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(start_frame, text="Start Node:").pack(side=tk.LEFT)
        self.start_var = tk.StringVar()
        self.start_entry = ttk.Entry(start_frame, textvariable=self.start_var, width=15)
        self.start_entry.pack(side=tk.RIGHT)

        # End node
        end_frame = ttk.Frame(selection_frame)
        end_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(end_frame, text="End Node:").pack(side=tk.LEFT)
        self.end_var = tk.StringVar()
        self.end_entry = ttk.Entry(end_frame, textvariable=self.end_var, width=15)
        self.end_entry.pack(side=tk.RIGHT)

        # Algorithm selection
        algo_frame = ttk.LabelFrame(parent, text="Pathfinding Algorithms", padding=10)
        algo_frame.pack(fill=tk.X, pady=(0, 15))

        self.algo_vars = {}
        algorithms = [
            ("Shortest Path", "shortest", "Distance-based routing"),
            ("Pavement Optimized", "pavement", "Best pavement quality"),
            ("Multi-Criteria", "multi_criteria", "Balanced optimization")
        ]

        for name, key, desc in algorithms:
            var = tk.BooleanVar(value=True)
            self.algo_vars[key] = var

            frame = ttk.Frame(algo_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Checkbutton(frame, text=name, variable=var).pack(side=tk.LEFT)
            ttk.Label(frame, text=desc, font=("Arial", 8), foreground="gray").pack(side=tk.RIGHT, padx=(10, 0))

        # Action buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Button(button_frame, text="🎯 Select on Map",
                  command=self._enter_selection_mode).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="🔍 Find Paths",
                  command=self._find_paths).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="📊 Compare Routes",
                  command=self._compare_routes).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="🎬 Run Demo",
                  command=self._run_demo).pack(fill=tk.X, pady=2)

        # Metrics display
        metrics_frame = ttk.LabelFrame(parent, text="Route Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, pady=(0, 15))

        # Create text widget for metrics
        self.metrics_text = tk.Text(metrics_frame, height=10, width=35,
                                   font=("Consolas", 9), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(metrics_frame, orient=tk.VERTICAL,
                                 command=self.metrics_text.yview)
        self.metrics_text.configure(yscrollcommand=scrollbar.set)

        self.metrics_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(parent, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(15, 0))

    def _create_visualization_panel(self, parent) -> None:
        """Create the visualization panel with matplotlib"""
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.patch.set_facecolor('#f0f0f0')

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(toolbar_frame, text="🔍 Zoom In",
                  command=lambda: self._zoom(1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="🔍 Zoom Out",
                  command=lambda: self._zoom(0.8)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="🏠 Reset View",
                  command=self._reset_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="📸 Save Image",
                  command=self._save_image).pack(side=tk.LEFT, padx=2)

    def _initialize_plot(self) -> None:
        """Initialize the plot with the road network"""
        self.ax.clear()

        # Draw the road network
        edges = []
        for u, v, data in self.graph.edges(data=True):
            if u in self.node_positions and v in self.node_positions:
                x1, y1 = self.node_positions[u]
                x2, y2 = self.node_positions[v]
                edges.append([(x1, y1), (x2, y2)])

        # Create LineCollection for edges
        from matplotlib.collections import LineCollection
        if edges:
            lc = LineCollection(edges, colors='lightgray', linewidths=0.5, alpha=0.7)
            self.ax.add_collection(lc)

        # Draw nodes
        if self.node_positions:
            positions = np.array(list(self.node_positions.values()))
            self.ax.scatter(positions[:, 0], positions[:, 1],
                          c='blue', s=5, alpha=0.6, picker=True)

        self.ax.set_aspect('equal')
        self.ax.set_title("🚴 Cycling Route Optimization - Interactive Map", fontsize=14, pad=20)
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")

        # Add legend
        legend_elements = [
            plt.scatter([], [], c='blue', s=20, label='Road Nodes'),
            plt.scatter([], [], c='red', s=30, label='Start Node'),
            plt.scatter([], [], c='green', s=30, label='End Node'),
            plt.Line2D([], [], color='blue', label='Shortest Path'),
            plt.Line2D([], [], color='green', label='Pavement Optimized'),
            plt.Line2D([], [], color='red', label='Multi-Criteria')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')

        self.canvas.draw()

    def _setup_event_handlers(self) -> None:
        """Setup event handlers for interactive features"""
        self.canvas.mpl_connect('button_press_event', self._on_map_click)
        self.canvas.mpl_connect('pick_event', self._on_node_pick)

    def _on_map_click(self, event) -> None:
        """Handle map click events for node selection"""
        if hasattr(self, '_selection_mode') and self._selection_mode:
            if event.xdata is not None and event.ydata is not None:
                # Find nearest node
                nearest_node = self._find_nearest_node(event.xdata, event.ydata)

                if self.selected_start is None:
                    self.selected_start = nearest_node
                    self.start_var.set(str(nearest_node))
                    self._update_plot()
                    self.status_var.set(f"Start node selected: {nearest_node}")
                elif self.selected_end is None:
                    self.selected_end = nearest_node
                    self.end_var.set(str(nearest_node))
                    self._update_plot()
                    self.status_var.set(f"End node selected: {nearest_end}")
                    self._selection_mode = False
                else:
                    # Both nodes selected, exit selection mode
                    self._selection_mode = False

    def _on_node_pick(self, event) -> None:
        """Handle node pick events"""
        if event.artist:
            node_id = list(self.node_positions.keys())[event.ind[0]]
            print(f"Node picked: {node_id}")

    def _find_nearest_node(self, x: float, y: float) -> int:
        """Find the nearest node to clicked coordinates"""
        min_distance = float('inf')
        nearest_node = None

        for node, (nx, ny) in self.node_positions.items():
            distance = math.sqrt((x - nx)**2 + (y - ny)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node

        return nearest_node

    def _enter_selection_mode(self) -> None:
        """Enter interactive node selection mode"""
        self._selection_mode = True
        self.selected_start = None
        self.selected_end = None
        self.start_var.set("")
        self.end_var.set("")
        self._clear_paths()
        self._update_plot()
        self.status_var.set("Click on map to select start node...")

    def _find_paths(self) -> None:
        """Find paths using selected algorithms"""
        try:
            start_node = self.selected_start
            end_node = self.selected_end

            if not start_node or not end_node:
                messagebox.showwarning("Warning", "Please select start and end nodes first")
                return

            self.status_var.set(f"Finding paths from {start_node} to {end_node}...")
            self._clear_paths()

            # Run selected algorithms
            for algo_key, var in self.algo_vars.items():
                if var.get():
                    try:
                        if algo_key == "shortest":
                            path = self._shortest_path_algorithm(start_node, end_node)
                            self.current_paths[algo_key] = path
                        elif algo_key == "pavement":
                            path = self._pavement_optimized_path(start_node, end_node)
                            self.current_paths[algo_key] = path
                        elif algo_key == "multi_criteria":
                            path = self._multi_criteria_path(start_node, end_node)
                            self.current_paths[algo_key] = path

                        print(f"✅ {algo_key} path found with {len(path)} nodes")

                    except Exception as e:
                        print(f"❌ {algo_key} algorithm failed: {e}")
                        messagebox.showerror("Error", f"{algo_key} algorithm failed: {e}")
                        import traceback
                        traceback.print_exc()

            self._update_plot()
            self.status_var.set("Paths found successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to find paths: {e}")

    def _compare_routes(self) -> None:
        """Compare the found routes and show analysis"""
        if not self.current_paths:
            messagebox.showwarning("Warning", "No paths to compare. Please find paths first.")
            return

        # Analyze each path
        self.route_analyses = {}
        for algo_key, path in self.current_paths.items():
            analysis = self._analyze_route(path, f"{algo_key.title()} Path")
            self.route_analyses[algo_key] = analysis

        # Update metrics display
        self._update_metrics_display()

        # Show comparison dialog
        self._show_comparison_dialog()

    def _analyze_route(self, path: List[int], route_name: str) -> Dict:
        """Analyze a route path"""
        analysis = {
            "route_name": route_name,
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

        return analysis

    def _update_metrics_display(self) -> None:
        """Update the metrics display with current route analyses"""
        self.metrics_text.delete(1.0, tk.END)

        if not self.route_analyses:
            self.metrics_text.insert(tk.END, "No route analyses available")
            return

        for algo_key, analysis in self.route_analyses.items():
            color = self.colors.get(algo_key, 'black')
            self.metrics_text.insert(tk.END, f"\n{'='*40}\n", 'header')
            self.metrics_text.insert(tk.END, f"📊 {analysis['route_name']}\n", f'color_{color}')
            self.metrics_text.insert(tk.END, f"{'='*40}\n", 'header')

            self.metrics_text.insert(tk.END, f"Path Length: {analysis['path_length']} nodes\n")
            self.metrics_text.insert(tk.END, f"Distance: {analysis['total_distance_m']:.1f}m ({analysis['total_distance_m']/1000:.2f}km)\n")
            self.metrics_text.insert(tk.END, f"Elevation Gain: {analysis['elevation_gain_m']:.1f}m\n")

            if 'average_paser_score' in analysis:
                self.metrics_text.insert(tk.END, f"Avg PASER Score: {analysis['average_paser_score']:.2f}\n")
                self.metrics_text.insert(tk.END, f"Best Segment: {analysis['max_paser_score']:.2f}\n")
                self.metrics_text.insert(tk.END, f"Worst Segment: {analysis['min_paser_score']:.2f}\n")

            if 'elevation_per_km' in analysis:
                self.metrics_text.insert(tk.END, f"Efficiency: {analysis['elevation_per_km']:.1f}m/km\n")

        self.metrics_text.tag_configure('header', font=("Arial", 10, "bold"))
        self.metrics_text.tag_configure('color_blue', foreground='blue')
        self.metrics_text.tag_configure('color_green', foreground='green')
        self.metrics_text.tag_configure('color_red', foreground='red')

    def _show_comparison_dialog(self) -> None:
        """Show a dialog with route comparison"""
        if len(self.route_analyses) < 2:
            messagebox.showinfo("Info", "Need at least 2 routes to compare")
            return

        # Create comparison window
        comp_window = tk.Toplevel(self.root)
        comp_window.title("Route Comparison")
        comp_window.geometry("800x600")

        # Create treeview for comparison
        columns = ('Route', 'Distance (km)', 'Elevation (m)', 'Avg PASER', 'Efficiency')
        tree = ttk.Treeview(comp_window, columns=columns, show='headings', height=10)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor=tk.CENTER)

        # Add data
        for algo_key, analysis in self.route_analyses.items():
            color = self.colors.get(algo_key, 'black')
            distance_km = analysis['total_distance_m'] / 1000
            elevation = analysis['elevation_gain_m']
            avg_paser = analysis.get('average_paser_score', 0)
            efficiency = analysis.get('elevation_per_km', 0)

            tree.insert('', tk.END, values=(
                analysis['route_name'],
                f"{distance_km:.2f}",
                f"{elevation:.1f}",
                f"{avg_paser:.2f}",
                f"{efficiency:.1f}"
            ), tags=(f'color_{color}',))

        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configure tags for colors
        for color in ['blue', 'green', 'red']:
            tree.tag_configure(f'color_{color}', foreground=color)

        # Add summary
        summary_frame = ttk.Frame(comp_window)
        summary_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Label(summary_frame, text="💡 Lower elevation and higher PASER scores indicate better cycling routes",
                 font=("Arial", 9, "italic")).pack()

    def _clear_paths(self) -> None:
        """Clear all drawn paths from the plot"""
        # Remove path lines and markers
        for collection in self.ax.collections[:]:
            if hasattr(collection, 'get_label') and collection.get_label() in ['path', 'start', 'end']:
                collection.remove()
        for patch in self.ax.patches[:]:
            patch.remove()

    def _update_plot(self) -> None:
        """Update the plot with current selections and paths"""
        self._clear_paths()

        # Draw start and end nodes
        if self.selected_start and self.selected_start in self.node_positions:
            x, y = self.node_positions[self.selected_start]
            self.ax.scatter(x, y, c='red', s=100, marker='o', edgecolors='darkred',
                          linewidth=2, label='start', zorder=10)

        if self.selected_end and self.selected_end in self.node_positions:
            x, y = self.node_positions[self.selected_end]
            self.ax.scatter(x, y, c='green', s=100, marker='X', edgecolors='darkgreen',
                          linewidth=2, label='end', zorder=10)

        # Draw paths
        for algo_key, path in self.current_paths.items():
            if path and len(path) > 1:
                color = self.colors.get(algo_key, 'black')
                path_coords = []

                for node in path:
                    if node in self.node_positions:
                        path_coords.append(self.node_positions[node])

                if len(path_coords) > 1:
                    path_coords = np.array(path_coords)
                    self.ax.plot(path_coords[:, 0], path_coords[:, 1],
                               color=color, linewidth=3, alpha=0.8,
                               label=f'path_{algo_key}', zorder=5)

        self.canvas.draw()

    def _zoom(self, factor: float) -> None:
        """Zoom in or out of the plot"""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2

        x_range = (xlim[1] - xlim[0]) / 2
        y_range = (ylim[1] - ylim[0]) / 2

        self.ax.set_xlim(x_center - x_range / factor, x_center + x_range / factor)
        self.ax.set_ylim(y_center - y_range / factor, y_center + y_range / factor)

        self.canvas.draw()

    def _reset_view(self) -> None:
        """Reset the plot view to show all nodes"""
        if self.node_positions:
            positions = np.array(list(self.node_positions.values()))
            margin = 0.02
            self.ax.set_xlim(positions[:, 0].min() - margin, positions[:, 0].max() + margin)
            self.ax.set_ylim(positions[:, 1].min() - margin, positions[:, 1].max() + margin)
            self.canvas.draw()

    def _toggle_node_labels(self) -> None:
        """Toggle node labels on/off"""
        # This would require storing label objects and showing/hiding them
        pass

    def _save_image(self) -> None:
        """Save the current plot as an image"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if filename:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Image saved as {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")

    def _export_results(self) -> None:
        """Export route results to JSON"""
        if not self.route_analyses:
            messagebox.showwarning("Warning", "No route analyses to export")
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filename:
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "routes": self.route_analyses,
                    "app_version": "Visual Route Optimization POC v1.0"
                }

                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)

                messagebox.showinfo("Success", f"Results exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {e}")

    def _run_demo(self) -> None:
        """Run a demonstration with sample routes"""
        try:
            # Find some high-degree nodes for demo
            node_degrees = dict(self.graph.degree())
            high_degree_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)

            if len(high_degree_nodes) < 4:
                messagebox.showwarning("Warning", "Not enough nodes for demo")
                return

            # Select demo nodes
            demo_nodes = [node for node, degree in high_degree_nodes[:4]]
            self.selected_start = demo_nodes[0]
            self.selected_end = demo_nodes[1]

            self.start_var.set(str(self.selected_start))
            self.end_var.set(str(self.selected_end))

            self._find_paths()

            messagebox.showinfo("Demo Complete", "Demo routes calculated and displayed!")

        except Exception as e:
            messagebox.showerror("Error", f"Demo failed: {e}")

    def _show_about(self) -> None:
        """Show about dialog"""
        messagebox.showinfo("About",
            "🚴 Cycling Route Optimization - Visual POC\n\n"
            "This application demonstrates the integration of\n"
            "computer vision-based pavement assessment with\n"
            "advanced route optimization algorithms.\n\n"
            "Features:\n"
            "- Interactive graph visualization\n"
            "- Multiple pathfinding algorithms\n"
            "- Real-time route comparison\n"
            "- PASER score optimization\n"
            "- Elevation and distance analysis"
        )

    # Pathfinding algorithms (adapted from original)
    def _shortest_path_algorithm(self, start: int, end: int) -> List[int]:
        """Traditional shortest path by distance"""
        try:
            path = nx.shortest_path(
                self.graph, source=start, target=end, weight='length'
            )
            return path
        except nx.NetworkXNoPath:
            raise ValueError("No path found between nodes")

    def _pavement_optimized_path(self, start: int, end: int) -> List[int]:
        """Path optimized for pavement quality"""
        try:
            # Check if edges have inverted_paser attribute
            has_inverted_paser = any(
                'inverted_paser' in data
                for u, v, k, data in self.graph.edges(keys=True, data=True)
            )

            if not has_inverted_paser:
                # Fall back to using paser_score (invert it on the fly)
                def get_pavement_weight(u, v, data):
                    paser_score = data.get('paser_score', 5.0)
                    # Invert PASER score (lower PASER = higher cost = worse pavement)
                    return 10 - paser_score  # Higher return value = worse pavement

                path = nx.shortest_path(
                    self.graph, source=start, target=end, weight=get_pavement_weight
                )
            else:
                path = nx.shortest_path(
                    self.graph, source=start, target=end, weight='inverted_paser'
                )
            return path
        except nx.NetworkXNoPath:
            raise ValueError("No path found between nodes")
        except Exception as e:
            print(f"Debug: Pavement algorithm error: {e}")
            # Try fallback to distance if pavement optimization fails
            try:
                return self._shortest_path_algorithm(start, end)
            except:
                raise Exception(f"Pavement optimization failed: {e}")

    def _multi_criteria_path(self, start: int, end: int) -> List[int]:
        """Multi-criteria path optimization"""
        try:
            # Create composite weight for each edge
            for u, v, k, data in self.graph.edges(keys=True, data=True):
                try:
                    # Get attributes with safe defaults
                    distance = data.get('length', 100)  # Default 100m if missing
                    elevation_gain = data.get('elevation_gain', 0)

                    # Handle PASER cost - check both inverted_paser and paser_score
                    if 'inverted_paser' in data:
                        paser_cost = data['inverted_paser']
                    elif 'paser_score' in data:
                        paser_score = data['paser_score']
                        # Invert PASER score (lower PASER = higher cost = worse pavement)
                        paser_cost = max(1.0, 10 - paser_score)  # Ensure positive cost
                    else:
                        paser_cost = 6  # Default neutral cost

                    # Normalize values to prevent extreme scaling
                    distance_km = distance / 1000  # Convert to km
                    elevation = abs(elevation_gain)  # Absolute elevation change

                    # Ensure all values are positive and reasonable
                    distance_km = max(0.01, distance_km)  # Minimum 10m
                    elevation = max(0, elevation)  # No negative elevation
                    paser_cost = max(0.1, paser_cost)  # Minimum cost

                    # Create composite cost with normalized weights
                    composite_cost = (
                        0.1 * distance_km +      # 10% distance
                        0.1 * elevation +        # 10% elevation
                        0.8 * paser_cost         # 80% pavement quality
                    )

                    # Ensure composite cost is reasonable
                    composite_cost = max(0.1, composite_cost)
                    data['composite_cost'] = composite_cost

                except Exception as e:
                    # If calculation fails for this edge, use default cost
                    data['composite_cost'] = 5.0
                    print(f"Warning: Could not calculate composite cost for edge {u}-{v}: {e}")

            path = nx.shortest_path(
                self.graph, source=start, target=end, weight='composite_cost'
            )
            return path
        except nx.NetworkXNoPath:
            raise ValueError("No path found between nodes")
        except Exception as e:
            print(f"Debug: Multi-criteria algorithm error: {e}")
            # Try fallback to distance if multi-criteria optimization fails
            try:
                return self._shortest_path_algorithm(start, end)
            except:
                raise Exception(f"Multi-criteria optimization failed: {e}")

    def run(self) -> None:
        """Run the GUI application"""
        self.root.mainloop()

def main():
    """Main function to run the Visual Route Optimization POC"""
    try:
        # Check if required files exist
        graph_path = "../data/updated_road_network.graphml"
        if not os.path.exists(graph_path):
            print("❌ Updated road network not found.")
            print("💡 Please run Stage 5 first: python src/stage5_update_graph.py")
            return

        # Initialize and run the GUI
        app = RouteOptimizationGUI(graph_path)
        print("🚀 Starting Visual Route Optimization POC...")
        app.run()

    except Exception as e:
        print(f"❌ Application failed to start: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
