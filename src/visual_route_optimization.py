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
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
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
        self.path_lines = {}  # Store references to path lines for better management

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
            print(f"[Graph] Loading graph from {graph_path}...")
            self.graph = ox.load_graphml(graph_path)

            # Get node positions for plotting
            self.node_positions = {node: (data['x'], data['y'])
                                 for node, data in self.graph.nodes(data=True)}

            # Print basic statistics
            n_nodes = len(self.graph.nodes())
            n_edges = len(self.graph.edges())
            print(f"[OK] Graph loaded: {n_nodes} nodes, {n_edges} edges")

            # Validate PASER scores are present
            edges_with_paser = 0
            for u, v, k, data in self.graph.edges(keys=True, data=True):
                if 'paser_score' in data:
                    edges_with_paser += 1

            print(f"[OK] PASER scores found on {edges_with_paser}/{n_edges} edges")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load graph: {e}")
            raise

    def _setup_gui(self) -> None:
        """Setup the main GUI interface"""
        self.root = tk.Tk()
        self.root.title("🚴 Cycling Route Optimization - Visual POC")
        self.root.geometry("1400x900")
        self.root.state('zoomed')

        # Initialize variables requiring root
        self.color_by_paser_var = tk.BooleanVar(value=False)

        # Create main frames
        self._create_menu_bar()
        self._create_main_layout()

        # Initialize plot
        self._initialize_plot()

        # Setup event handlers
        self._setup_event_handlers()

        # Handle window close properly
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self) -> None:
        """Handle application exit"""
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
        finally:
            sys.exit(0)

    def _create_menu_bar(self) -> None:
        """Create the menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export Results", command=self._export_results)
        file_menu.add_command(label="Exit", command=self._on_closing)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Reset View", command=self._reset_view)
        view_menu.add_separator()
        view_menu.add_command(label="Toggle All Paths", command=self._toggle_all_paths)
        view_menu.add_command(label="Show Only Shortest Path", command=self._show_only_shortest)
        view_menu.add_command(label="Show Only Pavement Path", command=self._show_only_pavement)
        view_menu.add_command(label="Show Only Multi-Criteria Path", command=self._show_only_multi_criteria)
        view_menu.add_separator()
        view_menu.add_checkbutton(label="Color Paths by PASER Score",
                                variable=self.color_by_paser_var,
                                command=self._update_plot)
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
        self.path_visibility_vars = {}
        algorithms = [
            ("Shortest Path", "shortest", "Distance-based routing"),
            ("Pavement Optimized", "pavement", "Best pavement quality"),
            ("Multi-Criteria", "multi_criteria", "Balanced optimization")
        ]

        for name, key, desc in algorithms:
            var = tk.BooleanVar(value=True)
            visibility_var = tk.BooleanVar(value=True)
            self.algo_vars[key] = var
            self.path_visibility_vars[key] = visibility_var

            frame = ttk.Frame(algo_frame)
            frame.pack(fill=tk.X, pady=2)

            # Algorithm selection checkbox
            ttk.Checkbutton(frame, text=name, variable=var).pack(side=tk.LEFT)

            # Path visibility toggle (only show if path exists)
            # Use a closure to capture the current key value
            def make_visibility_command(algo_key):
                return lambda: self._on_visibility_toggle(algo_key)

            visibility_cb = ttk.Checkbutton(frame, text="Show", variable=visibility_var,
                                          command=make_visibility_command(key))
            visibility_cb.pack(side=tk.LEFT, padx=(10, 0))

            # Store reference for later enabling/disabling
            setattr(self, f"visibility_cb_{key}", visibility_cb)
            visibility_cb.configure(state="disabled")  # Initially disabled

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
        ttk.Button(button_frame, text="🧹 Clear Paths",
                  command=self._clear_all_paths).pack(fill=tk.X, pady=2)
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
                    self.status_var.set(f"End node selected: {nearest_node}")
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

        # Clear paths and disable visibility checkboxes
        self.current_paths = {}
        self.route_analyses = {}
        self.path_lines = {}  # Clear path line references
        for key in self.path_visibility_vars:
            visibility_cb = getattr(self, f"visibility_cb_{key}")
            visibility_cb.configure(state="disabled")

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

                        print(f"[OK] {algo_key} path found with {len(path)} nodes")

                        # Debug: Show path differences
                        if algo_key != "shortest":
                            # Compare with the stored shortest path
                            if 'shortest' in self.current_paths:
                                shortest_path = self.current_paths['shortest']
                                if path != shortest_path:
                                    print(f"   [Diff] {algo_key} path differs from shortest path!")
                                else:
                                    print(f"   [Same] {algo_key} path is identical to shortest path")

                    except Exception as e:
                        print(f"[Error] {algo_key} algorithm failed: {e}")
                        messagebox.showerror("Error", f"{algo_key} algorithm failed: {e}")
                        import traceback
                        traceback.print_exc()

            # Enable visibility checkboxes for found paths
            for algo_key in self.current_paths:
                if self.current_paths[algo_key]:  # Path found
                    visibility_cb = getattr(self, f"visibility_cb_{algo_key}")
                    visibility_cb.configure(state="normal")

            self._update_plot()
            self.status_var.set("Paths found successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to find paths: {e}")

    def _compare_routes(self) -> None:
        """Compare the found routes and show analysis"""
        print(f"Debug: Comparing routes. Current paths: {list(self.current_paths.keys())}")

        if not self.current_paths:
            messagebox.showwarning("Warning", "No paths to compare. Please find paths first.")
            return

        if len(self.current_paths) < 2:
            messagebox.showwarning("Warning", "Need at least 2 paths to compare. Please ensure multiple algorithms found paths.")
            return

        try:
            self.status_var.set("Analyzing routes for comparison...")

            # Analyze each path
            self.route_analyses = {}
            successful_analyses = 0

            for algo_key, path in self.current_paths.items():
                if path and len(path) > 1:
                    try:
                        print(f"Debug: Analyzing {algo_key} path with {len(path)} nodes")
                        analysis = self._analyze_route(path, f"{algo_key.title()} Path")
                        self.route_analyses[algo_key] = analysis
                        successful_analyses += 1
                        print(f"Debug: Successfully analyzed {algo_key} path")
                    except Exception as e:
                        print(f"Debug: Failed to analyze {algo_key} path: {e}")
                        messagebox.showwarning("Warning", f"Could not analyze {algo_key} path: {e}")

            if successful_analyses == 0:
                messagebox.showerror("Error", "Could not analyze any paths for comparison.")
                return

            print(f"Debug: Successfully analyzed {successful_analyses} routes")

            # Update metrics display
            self._update_metrics_display()

            # Show comparison dialog
            self._show_comparison_dialog()

            self.status_var.set(f"Comparison complete! Analyzed {successful_analyses} routes")

        except Exception as e:
            print(f"Debug: Comparison failed: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to compare routes: {e}")

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
                distance_raw = edge_attrs.get('length', 0)
                try:
                    # Handle case where distance might be stored as string
                    distance = float(distance_raw) if distance_raw != 0 else 0
                except (ValueError, TypeError):
                    distance = 0
                    print(f"Warning: Invalid distance value '{distance_raw}' for edge {start_node}-{end_node}")

                analysis["total_distance_m"] += distance

                # Elevation
                elevation_raw = edge_attrs.get('elevation_gain', 0)
                try:
                    # Handle case where elevation might be stored as string
                    elevation = float(elevation_raw) if elevation_raw != 0 else 0
                except (ValueError, TypeError):
                    elevation = 0
                    print(f"Warning: Invalid elevation value '{elevation_raw}' for edge {start_node}-{end_node}")

                analysis["elevation_gain_m"] += abs(elevation)

                # PASER score
                paser_raw = edge_attrs.get('paser_score', 5.0)
                try:
                    # Handle case where paser_score might be stored as string
                    if isinstance(paser_raw, str):
                        # Remove any whitespace and potential invisible characters
                        clean_paser = paser_raw.strip().replace('\x00', '').replace('\ufeff', '')
                        paser_score = float(clean_paser) if clean_paser != '' else 5.0
                    else:
                        paser_score = float(paser_raw) if paser_raw != 0 else 5.0
                except (ValueError, TypeError):
                    paser_score = 5.0  # Default neutral PASER score
                    print(f"Warning: Invalid paser_score value '{paser_raw}' for edge {start_node}-{end_node}")

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

        try:
            # Create comparison window
            comp_window = tk.Toplevel(self.root)
            comp_window.title("🚴 Route Comparison Results")
            comp_window.geometry("900x700")
            comp_window.lift()  # Bring window to front
            comp_window.focus_force()  # Force focus
            comp_window.grab_set()  # Make modal

            # Add window close handler
            def on_closing():
                comp_window.grab_release()
                comp_window.destroy()

            comp_window.protocol("WM_DELETE_WINDOW", on_closing)

            # Title
            title_label = ttk.Label(comp_window, text="🚴 Route Comparison Analysis",
                                   font=("Arial", 16, "bold"))
            title_label.pack(pady=(20, 10))

            # Create main frame
            main_frame = ttk.Frame(comp_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

            # Create treeview for comparison
            columns = ('Route', 'Distance (km)', 'Elevation (m)', 'Avg PASER', 'Efficiency', 'Best Segment', 'Worst Segment')
            tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=8)

            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=120, anchor=tk.CENTER)

            # Add data
            for algo_key, analysis in self.route_analyses.items():
                color = self.colors.get(algo_key, 'black')
                distance_km = analysis['total_distance_m'] / 1000
                elevation = analysis['elevation_gain_m']
                avg_paser = analysis.get('average_paser_score', 0)
                efficiency = analysis.get('elevation_per_km', 0)
                best_segment = analysis.get('max_paser_score', 0)
                worst_segment = analysis.get('min_paser_score', 0)

                tree.insert('', tk.END, values=(
                    analysis['route_name'],
                    f"{distance_km:.2f}",
                    f"{elevation:.1f}",
                    f"{avg_paser:.2f}",
                    f"{efficiency:.1f}",
                    f"{best_segment:.1f}",
                    f"{worst_segment:.1f}"
                ), tags=(f'color_{color}',))

            tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Configure tags for colors
            for color in ['blue', 'green', 'red']:
                tree.tag_configure(f'color_{color}', foreground=color)

            # Add summary at bottom
            summary_frame = ttk.Frame(comp_window)
            summary_frame.pack(fill=tk.X, padx=20, pady=(10, 20))

            summary_text = tk.Text(summary_frame, height=4, width=80,
                                  font=("Arial", 10), wrap=tk.WORD)
            summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            summary_text.insert(tk.END, "💡 Analysis Summary:\n\n", "bold")
            summary_text.insert(tk.END, "• Lower elevation and higher PASER scores indicate better cycling routes\n")
            summary_text.insert(tk.END, "• Green path (Pavement Optimized) typically has the best road surface quality\n")
            summary_text.insert(tk.END, "• Red path (Multi-Criteria) balances distance, elevation, and pavement quality\n")
            summary_text.insert(tk.END, "• Blue path (Shortest) is fastest but may have poorer road conditions")

            summary_text.tag_configure("bold", font=("Arial", 10, "bold"))
            summary_text.config(state=tk.DISABLED)

            # Close button
            close_btn = ttk.Button(comp_window, text="Close Comparison",
                                  command=on_closing)
            close_btn.pack(pady=(0, 20))

            print(f"Debug: Comparison dialog created with {len(self.route_analyses)} routes")

        except Exception as e:
            print(f"Debug: Failed to create comparison dialog: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to show comparison: {e}")

    def _clear_paths(self) -> None:
        """Clear all drawn paths from the plot"""
        # Remove all lines that are not the base network
        lines_to_remove = []
        for line in self.ax.lines:
            if hasattr(line, '_path_type'):  # Our custom attribute for path lines
                lines_to_remove.append(line)

        for line in lines_to_remove:
            line.remove()

        # Also clear any collections with path labels
        collections_to_remove = []
        for collection in self.ax.collections:
            if hasattr(collection, 'get_label'):
                label = collection.get_label()
                if label and ('path' in label or label in ['start', 'end']):
                    collections_to_remove.append(collection)

        for collection in collections_to_remove:
            collection.remove()

    def _clear_all_paths(self) -> None:
        """Clear all paths and reset selections"""
        self.current_paths = {}
        self.route_analyses = {}
        self.path_lines = {}  # Clear path line references
        self.selected_start = None
        self.selected_end = None
        self.start_var.set("")
        self.end_var.set("")

        # Disable all visibility checkboxes
        for key in self.path_visibility_vars:
            visibility_cb = getattr(self, f"visibility_cb_{key}")
            visibility_cb.configure(state="disabled")
            self.path_visibility_vars[key].set(True)  # Reset to visible

        # Clear metrics display
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, "No routes calculated")

        # Update plot
        self._clear_paths()
        self._initialize_plot()

        self.status_var.set("All paths cleared")

    def _get_edge_paser(self, u: int, v: int) -> float:
        """Get PASER score for an edge with robust error handling"""
        edge_data = self.graph.get_edge_data(u, v)
        if not edge_data:
            return 5.0

        # Get the first edge key
        key = list(edge_data.keys())[0]
        data = edge_data[key]

        paser_raw = data.get('paser_score', 5.0)
        try:
            if isinstance(paser_raw, str):
                clean_paser = paser_raw.strip().replace('\x00', '').replace('\ufeff', '')
                return float(clean_paser) if clean_paser else 5.0
            return float(paser_raw) if paser_raw != 0 else 5.0
        except (ValueError, TypeError):
            return 5.0

    def _update_plot(self) -> None:
        """Update the plot with current selections and paths"""
        self._clear_paths()

        # Draw start and end nodes
        if self.selected_start and self.selected_start in self.node_positions:
            x, y = self.node_positions[self.selected_start]
            scatter = self.ax.scatter(x, y, c='red', s=100, marker='o', edgecolors='darkred',
                          linewidth=2, zorder=10)
            scatter.set_label('start')

        if self.selected_end and self.selected_end in self.node_positions:
            x, y = self.node_positions[self.selected_end]
            scatter = self.ax.scatter(x, y, c='green', s=100, marker='X', edgecolors='darkgreen',
                          linewidth=2, zorder=10)
            scatter.set_label('end')

        # Clear stored path line references
        self.path_lines = {}

        # Draw paths (only if visible)
        for algo_key, path in self.current_paths.items():
            if (path and len(path) > 1 and
                algo_key in self.path_visibility_vars and
                self.path_visibility_vars[algo_key].get()):

                # Determine z-order
                zorder = 5 + len(self.current_paths) - list(self.current_paths.keys()).index(algo_key)

                path_coords = []
                segments = []
                colors_list = []

                # Prepare segments
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    if u in self.node_positions and v in self.node_positions:
                        p1 = self.node_positions[u]
                        p2 = self.node_positions[v]

                        if self.color_by_paser_var.get():
                            segments.append([p1, p2])
                            paser = self._get_edge_paser(u, v)
                            # Map PASER 1-10 to colormap (RdYlGn)
                            # Normalize: 1 -> 0.0 (Red), 10 -> 1.0 (Green)
                            norm_score = max(0.0, min(1.0, (paser - 1) / 9.0))
                            colors_list.append(cm.RdYlGn(norm_score))
                        else:
                            path_coords.append(p1)

                # Draw based on mode
                if self.color_by_paser_var.get() and segments:
                    lc = LineCollection(segments, colors=colors_list, linewidths=4, alpha=0.8, zorder=zorder)
                    lc.set_label(f'{algo_key}_path')
                    lc._path_type = algo_key
                    self.ax.add_collection(lc)
                    self.path_lines[algo_key] = lc

                elif not self.color_by_paser_var.get() and path_coords:
                    # Include last point for continuous line
                    if path[-1] in self.node_positions:
                        path_coords.append(self.node_positions[path[-1]])

                    if len(path_coords) > 1:
                        path_coords = np.array(path_coords)
                        color = self.colors.get(algo_key, 'black')

                        line, = self.ax.plot(path_coords[:, 0], path_coords[:, 1],
                                           color=color, linewidth=3, alpha=0.8,
                                           zorder=zorder)

                        line._path_type = algo_key
                        line._algo_key = algo_key
                        line.set_label(f'{algo_key}_path')
                        self.path_lines[algo_key] = line

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

    def _toggle_all_paths(self) -> None:
        """Toggle visibility of all paths"""
        if not self.current_paths:
            messagebox.showinfo("Info", "No paths to toggle. Please find paths first.")
            return

        # Check if all paths are currently visible
        all_visible = all(self.path_visibility_vars[key].get()
                         for key in self.path_visibility_vars
                         if key in self.current_paths)

        # Toggle all paths
        new_state = not all_visible
        for key in self.current_paths:
            if key in self.path_visibility_vars:
                self.path_visibility_vars[key].set(new_state)

        self._update_plot()
        state_text = "shown" if new_state else "hidden"
        self.status_var.set(f"All paths {state_text}")

    def _show_only_shortest(self) -> None:
        """Show only the shortest path"""
        self._show_only_path("shortest", "Shortest Path")

    def _show_only_pavement(self) -> None:
        """Show only the pavement optimized path"""
        self._show_only_path("pavement", "Pavement Optimized Path")

    def _show_only_multi_criteria(self) -> None:
        """Show only the multi-criteria path"""
        self._show_only_path("multi_criteria", "Multi-Criteria Path")

    def _show_only_path(self, target_key: str, path_name: str) -> None:
        """Show only the specified path type"""
        if target_key not in self.current_paths:
            messagebox.showinfo("Info", f"{path_name} not available. Please calculate paths first.")
            return

        # Hide all paths except the target
        for key in self.path_visibility_vars:
            if key in self.current_paths:
                self.path_visibility_vars[key].set(key == target_key)

        self._update_plot()
        self.status_var.set(f"Showing only {path_name}")

    def _on_visibility_toggle(self, algo_key: str = None) -> None:
        """Handle individual path visibility toggle"""
        if not self.current_paths:
            return

        print(f"Debug: Visibility toggled for {algo_key}")  # Debug output

        # Count visible paths
        visible_paths = []
        for key in self.current_paths:
            if key in self.path_visibility_vars and self.path_visibility_vars[key].get():
                visible_paths.append(key)

        # Update plot
        self._update_plot()

        # Update status
        if len(visible_paths) == 0:
            self.status_var.set("All paths hidden")
        elif len(visible_paths) == 1:
            path_name = visible_paths[0].replace("_", " ").title()
            self.status_var.set(f"Showing only {path_name} path")
        else:
            self.status_var.set(f"Showing {len(visible_paths)} paths")

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
            # Check if edges have usable inverted_paser attribute (numeric values)
            has_usable_inverted_paser = False
            sample_inverted_values = []

            for u, v, k, data in self.graph.edges(keys=True, data=True):
                if 'inverted_paser' in data:
                    try:
                        inverted_val = data['inverted_paser']
                        # Test conversion with more robust cleaning
                        if isinstance(inverted_val, str):
                            clean_val = inverted_val.strip().replace('\x00', '').replace('\ufeff', '').replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
                            if clean_val:
                                float(clean_val)
                        else:
                            float(inverted_val)

                        has_usable_inverted_paser = True
                        sample_inverted_values.append((u, v, inverted_val))
                        if len(sample_inverted_values) >= 3:  # Just need a few samples
                            break
                    except (ValueError, TypeError, AttributeError):
                        print(f"Debug: Found unusable inverted_paser value '{inverted_val}' for edge {u}-{v}")
                        continue

            print(f"Debug: Found usable inverted_paser: {has_usable_inverted_paser}")
            if sample_inverted_values:
                print(f"Debug: Sample inverted_paser values: {sample_inverted_values[:3]}")

            if not has_usable_inverted_paser:
                # Fall back to using paser_score (invert it on the fly)
                print(f"Debug: No usable inverted_paser found, using paser_score fallback")
                def get_pavement_weight(u, v, data):
                    paser_raw = data.get('paser_score', 5.0)
                    try:
                        # Handle case where paser_score might be stored as string
                        if isinstance(paser_raw, str):
                            # Remove any whitespace and potential invisible characters
                            clean_paser = paser_raw.strip().replace('\x00', '').replace('\ufeff', '')
                            paser_score = float(clean_paser) if clean_paser != '' else 5.0
                        else:
                            paser_score = float(paser_raw) if paser_raw != 0 else 5.0
                    except (ValueError, TypeError) as e:
                        print(f"Debug: Failed to convert paser_score '{repr(paser_raw)}' for edge {u}-{v}: {e}")
                        paser_score = 5.0  # Default neutral PASER score

                    # Invert PASER score (lower PASER = higher cost = worse pavement)
                    # Multiply by length to penalize long stretches of bad road
                    paser_cost = max(0.1, 10 - paser_score)
                    length = data.get('length', 100)
                    return paser_cost * length

                try:
                    path = nx.shortest_path(
                        self.graph, source=start, target=end, weight=get_pavement_weight
                    )
                    print(f"Debug: Pavement algorithm succeeded using paser_score fallback")
                except Exception as e:
                    print(f"Debug: Pavement algorithm failed even with fallback: {e}")
                    # Last resort: fall back to distance
                    path = self._shortest_path_algorithm(start, end)
            else:
                print(f"Debug: Using inverted_paser for pavement optimization")
                # Define weight function for inverted_paser strategy as well to include length
                def get_inverted_paser_weight(u, v, data):
                    inverted_val = data.get('inverted_paser', 5.0)
                    length = data.get('length', 100)
                    try:
                        # Ensure numeric
                        val = float(inverted_val) if isinstance(inverted_val, (int, float, str)) else 5.0
                        dist = float(length) if length else 100.0
                        return val * dist
                    except:
                        return 5.0 * 100.0

                try:
                    path = nx.shortest_path(
                        self.graph, source=start, target=end, weight=get_inverted_paser_weight
                    )
                    print(f"Debug: Pavement algorithm succeeded using weighted inverted_paser")
                except Exception as e:
                    print(f"Debug: Pavement algorithm failed with inverted_paser: {e}")
                    print(f"Debug: Error type: {type(e).__name__}")
                    # Fall back to paser_score method
                    try:
                        def get_pavement_weight(u, v, data):
                            paser_raw = data.get('paser_score', 5.0)
                            try:
                                if isinstance(paser_raw, str):
                                    clean_paser = paser_raw.strip().replace('\x00', '').replace('\ufeff', '').replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
                                    paser_score = float(clean_paser) if clean_paser != '' else 5.0
                                else:
                                    paser_score = float(paser_raw) if paser_raw != 0 else 5.0
                            except (ValueError, TypeError):
                                paser_score = 5.0

                            return max(0.1, 10 - paser_score)

                        path = nx.shortest_path(
                            self.graph, source=start, target=end, weight=get_pavement_weight
                        )
                        print(f"Debug: Pavement algorithm succeeded with paser_score fallback")
                    except Exception as e2:
                        print(f"Debug: Pavement algorithm failed even with fallback: {e2}")
                        path = self._shortest_path_algorithm(start, end)
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
        """
        Multi-criteria path optimization using methodology-specified normalization and ROC weights.
        
        Normalization formulas (from methodology):
        - Inverted PASER Score: norm_PPS = PPS / 10 (where PPS is inverted, 0=excellent, 10=worst)
        - Elevation Gain: norm_elev = elev_gain / max_elev_gain
        - Distance: norm_dist = (distance - min_dist) / (max_dist - min_dist)
                    Simplified to: norm_dist = distance / max_dist (when min_dist ≈ 0)
        
        Composite weight formula:
        composite_weight = 0.611 × norm_PPS + 0.278 × norm_elev + 0.111 × norm_dist
        
        Edge cost calculation (for Dijkstra's algorithm):
        edge_cost = composite_weight × edge_length
        
        The multiplication by edge_length is essential because:
        1. It makes the cost proportional to distance traveled (longer bad roads cost more)
        2. It ensures paths are compared fairly regardless of edge count
        3. Without it, Dijkstra would favor paths with fewer edges, not better quality
        """
        try:
            # First pass: Calculate max values for normalization (as per methodology)
            print(f"Debug: Calculating normalization ranges for {len(list(self.graph.edges()))} edges")

            # Collect values for normalization ranges
            elevation_gains = []
            distances = []

            for u, v, k, data in self.graph.edges(keys=True, data=True):
                # Elevation gains (only positive values as per methodology)
                if 'elevation_gain' in data:
                    try:
                        elev_val = float(data['elevation_gain'])
                        if elev_val > 0:  # Only positive elevation gains (uphill)
                            elevation_gains.append(elev_val)
                    except:
                        pass

                # Distances
                if 'length' in data:
                    try:
                        dist_val = float(data['length'])
                        if dist_val > 0:
                            distances.append(dist_val)
                    except:
                        pass

            # Calculate normalization ranges as per methodology
            # PASER: Fixed scale 0-10, so we just divide by 10 (no need for min/max)
            max_elev_gain = max(elevation_gains) if elevation_gains else 1.0
            min_dist = min(distances) if distances else 0.0
            max_dist = max(distances) if distances else 1000.0

            print(f"Debug: Normalization ranges (methodology) - PASER: fixed 0-10 scale, "
                  f"max_elev_gain: {max_elev_gain:.2f}m, dist: {min_dist:.2f}-{max_dist:.2f}m")

            # ROC weights from methodology (Rank Order Centroid method)
            # Based on ranking: pavement condition > elevation gain > distance
            alpha = 0.611  # Pavement quality (61.1%)
            beta = 0.278   # Elevation (27.8%)
            gamma = 0.111  # Distance (11.1%)

            # Second pass: Calculate composite weights for each edge
            successful_calculations = 0
            for u, v, k, data in self.graph.edges(keys=True, data=True):
                try:
                    # Get raw values
                    distance_raw = data.get('length', 100)
                    elevation_gain_raw = data.get('elevation_gain', 0)

                    # Get PASER score - prefer inverted_paser if available, otherwise invert paser_score
                    if 'inverted_paser' in data:
                        pps_raw = data.get('inverted_paser', 5.0)
                    else:
                        # Invert PASER score: inverted = 10 - original
                        # So PASER 10 (best) becomes 0, PASER 1 (worst) becomes 9
                        paser_raw = data.get('paser_score', 5.0)
                        try:
                            paser_val = float(paser_raw) if not isinstance(paser_raw, str) else float(paser_raw.strip())
                            pps_raw = 10.0 - paser_val
                        except:
                            pps_raw = 5.0

                    # Convert to numbers with fallbacks
                    distance = float(distance_raw) if distance_raw != 0 else 100.0
                    elevation_gain = float(elevation_gain_raw) if elevation_gain_raw != 0 else 0.0

                    # Handle inverted PASER score conversion
                    try:
                        if isinstance(pps_raw, str):
                            pps = float(pps_raw.strip())
                        else:
                            pps = float(pps_raw) if pps_raw != 0 else 5.0
                    except:
                        pps = 5.0

                    # Apply normalization as per methodology

                    # 1. Inverted PASER Score: norm_PPS = PPS / 10
                    #    Where PPS ranges 0-10 (0=excellent pavement, 10=worst)
                    norm_pps = pps / 10.0
                    # Clamp to valid range [0, 1]
                    norm_pps = max(0.0, min(1.0, norm_pps))

                    # 2. Elevation Gain: norm_elev = elev_gain / max_elev_gain
                    #    Only positive elevation gains considered (uphill)
                    elev_gain_positive = max(0.0, elevation_gain)
                    norm_elev = elev_gain_positive / max_elev_gain if max_elev_gain > 0 else 0.0

                    # 3. Distance: norm_dist = (distance - min_dist) / (max_dist - min_dist)
                    #    Simplified to distance / max_dist when min_dist ≈ 0
                    if min_dist < 1.0:  # min_dist approximately 0
                        norm_dist = distance / max_dist if max_dist > 0 else 0.5
                    else:
                        norm_dist = (distance - min_dist) / (max_dist - min_dist) if max_dist > min_dist else 0.5

                    # Calculate composite weight using ROC weights (methodology formula)
                    # composite_weight = α × norm_PPS + β × norm_elev + γ × norm_dist
                    composite_weight = (
                        alpha * norm_pps +      # 0.611 × normalized inverted PASER
                        beta * norm_elev +      # 0.278 × normalized elevation gain
                        gamma * norm_dist       # 0.111 × normalized distance
                    )

                    # Ensure positive weight (avoid zero-weight edges)
                    composite_weight = max(0.001, composite_weight)

                    # CRITICAL: Multiply by edge length to create proper path cost
                    # Without this, Dijkstra favors paths with fewer edges regardless of quality
                    # This makes the cost proportional to distance: longer bad roads cost more
                    edge_cost = composite_weight * distance

                    # Store in graph
                    data['composite_weight'] = edge_cost
                    successful_calculations += 1

                except Exception as e:
                    # Fallback: use mid-range default weight
                    data['composite_weight'] = 0.5
                    print(f"Warning: Could not calculate composite weight for edge {u}-{v}: {e}")

            print(f"Debug: Multi-criteria - successfully calculated weights for "
                  f"{successful_calculations}/{len(list(self.graph.edges()))} edges")

            # Find path using composite weights (Dijkstra's algorithm via NetworkX)
            path = nx.shortest_path(
                self.graph, source=start, target=end, weight='composite_weight'
            )
            print(f"Debug: Multi-criteria algorithm succeeded with methodology-specified ROC weighting")
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
        # Use paths relative to the script location to ensure they work from any working directory
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        graph_path = os.path.join(BASE_DIR, "data", "updated_road_network.graphml")

        # Check if required files exist
        if not os.path.exists(graph_path):
            print("[Error] Updated road network not found.")
            print("[Info] Please run Stage 5 first: python src/stage5_update_graph.py")
            return

        # Initialize and run the GUI
        app = RouteOptimizationGUI(graph_path)
        print("[Start] Starting Visual Route Optimization POC...")
        app.run()

    except Exception as e:
        print(f"[Error] Application failed to start: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
