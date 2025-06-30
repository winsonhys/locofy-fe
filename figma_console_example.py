#!/usr/bin/env python3
"""
Advanced Figma Console API Examples with Depth-First Search

This script demonstrates various ways to use the Figma Console APIs
to perform depth-first search operations on Figma nodes.
"""

import requests
import json
from typing import Dict, List, Any, Optional, Callable
from figma_dfs import FigmaDFS, FigmaNode, NodeType
import time


class FigmaConsoleAPI:
    """Advanced Figma Console API wrapper with DFS capabilities"""

    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://api.figma.com/v1"
        self.headers = {
            "X-Figma-Token": access_token,
            "Content-Type": "application/json",
        }

    def get_file_components(self, file_key: str) -> Dict[str, Any]:
        """Get all components in a file"""
        url = f"{self.base_url}/files/{file_key}/components"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_file_styles(self, file_key: str) -> Dict[str, Any]:
        """Get all styles in a file"""
        url = f"{self.base_url}/files/{file_key}/styles"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_file_images(
        self,
        file_key: str,
        node_ids: List[str],
        format: str = "png",
        scale: float = 1.0,
    ) -> Dict[str, Any]:
        """Get image URLs for specific nodes"""
        ids_param = ",".join(node_ids)
        url = f"{self.base_url}/images/{file_key}?ids={ids_param}&format={format}&scale={scale}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_comments(self, file_key: str) -> Dict[str, Any]:
        """Get all comments in a file"""
        url = f"{self.base_url}/files/{file_key}/comments"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_team_projects(self, team_id: str) -> Dict[str, Any]:
        """Get all projects in a team"""
        url = f"{self.base_url}/teams/{team_id}/projects"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_project_files(self, project_id: str) -> Dict[str, Any]:
        """Get all files in a project"""
        url = f"{self.base_url}/projects/{project_id}/files"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()


class AdvancedFigmaDFS(FigmaDFS):
    """Extended FigmaDFS with advanced search capabilities"""

    def __init__(self, access_token: str):
        super().__init__(access_token)
        self.console_api = FigmaConsoleAPI(access_token)

    def search_by_properties(
        self, root_node: FigmaNode, properties: Dict[str, Any]
    ) -> List[FigmaNode]:
        """
        Search for nodes with specific properties using DFS

        Args:
            root_node: Starting node
            properties: Dictionary of properties to match

        Returns:
            List of nodes matching the properties
        """
        matching_nodes = []

        def property_callback(node: FigmaNode, depth: int) -> bool:
            node_data = node.data

            # Check if all properties match
            matches = True
            for key, value in properties.items():
                if key not in node_data or node_data[key] != value:
                    matches = False
                    break

            if matches:
                matching_nodes.append(node)
            return True

        self.depth_first_search(root_node, property_callback)
        return matching_nodes

    def search_by_constraints(
        self, root_node: FigmaNode, constraint_type: str = None
    ) -> List[FigmaNode]:
        """
        Search for nodes with specific constraints

        Args:
            root_node: Starting node
            constraint_type: Type of constraint to search for

        Returns:
            List of nodes with constraints
        """
        matching_nodes = []

        def constraint_callback(node: FigmaNode, depth: int) -> bool:
            node_data = node.data

            # Check for constraints
            if "constraints" in node_data:
                constraints = node_data["constraints"]
                if (
                    constraint_type is None
                    or constraints.get("type") == constraint_type
                ):
                    matching_nodes.append(node)
            return True

        self.depth_first_search(root_node, constraint_callback)
        return matching_nodes

    def search_by_fills(
        self, root_node: FigmaNode, fill_type: str = None
    ) -> List[FigmaNode]:
        """
        Search for nodes with specific fill types

        Args:
            root_node: Starting node
            fill_type: Type of fill to search for (e.g., "SOLID", "GRADIENT")

        Returns:
            List of nodes with matching fills
        """
        matching_nodes = []

        def fill_callback(node: FigmaNode, depth: int) -> bool:
            node_data = node.data

            # Check for fills
            if "fills" in node_data and node_data["fills"]:
                for fill in node_data["fills"]:
                    if fill_type is None or fill.get("type") == fill_type:
                        matching_nodes.append(node)
                        break
            return True

        self.depth_first_search(root_node, fill_callback)
        return matching_nodes

    def get_node_hierarchy(
        self, root_node: FigmaNode, target_node_id: str
    ) -> Optional[List[FigmaNode]]:
        """
        Get the path from root to a specific node

        Args:
            root_node: Starting node
            target_node_id: ID of the target node

        Returns:
            List of nodes representing the path, or None if not found
        """
        path = []

        def path_callback(node: FigmaNode, depth: int) -> bool:
            path.append(node)

            if node.id == target_node_id:
                return False  # Stop searching

            return True

        self.depth_first_search(root_node, path_callback)

        # Check if we found the target
        if path and path[-1].id == target_node_id:
            return path
        return None

    def count_nodes_by_type(self, root_node: FigmaNode) -> Dict[str, int]:
        """
        Count nodes by type using DFS

        Args:
            root_node: Starting node

        Returns:
            Dictionary with node types as keys and counts as values
        """
        type_counts = {}

        def count_callback(node: FigmaNode, depth: int) -> bool:
            node_type = node.type
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
            return True

        self.depth_first_search(root_node, count_callback)
        return type_counts

    def find_largest_nodes(
        self, root_node: FigmaNode, limit: int = 10
    ) -> List[FigmaNode]:
        """
        Find the largest nodes by area using DFS

        Args:
            root_node: Starting node
            limit: Maximum number of nodes to return

        Returns:
            List of largest nodes sorted by area
        """
        nodes_with_area = []

        def area_callback(node: FigmaNode, depth: int) -> bool:
            node_data = node.data

            # Calculate area if dimensions are available
            if "absoluteBoundingBox" in node_data:
                bbox = node_data["absoluteBoundingBox"]
                width = bbox.get("width", 0)
                height = bbox.get("height", 0)
                area = width * height

                if area > 0:
                    nodes_with_area.append((node, area))
            return True

        self.depth_first_search(root_node, area_callback)

        # Sort by area and return top nodes
        nodes_with_area.sort(key=lambda x: x[1], reverse=True)
        return [node for node, area in nodes_with_area[:limit]]


def advanced_example_usage():
    """Demonstrate advanced Figma Console API usage with DFS"""

    # Replace with your actual credentials
    ACCESS_TOKEN = "your_figma_access_token_here"
    FILE_KEY = "your_file_key_here"

    # Initialize advanced Figma DFS
    figma_dfs = AdvancedFigmaDFS(ACCESS_TOKEN)

    try:
        print("🔍 Advanced Figma Console API Examples")
        print("=" * 60)

        # Get file data
        print("📁 Fetching Figma file...")
        file_data = figma_dfs.get_file(FILE_KEY)
        document_data = file_data.get("document", {})
        root_node = figma_dfs.parse_node(document_data)

        print(f"✅ Successfully loaded: {file_data.get('name', 'Unknown')}")
        print(f"📊 Root node: {root_node.name} ({root_node.type})")

        # Example 1: Count nodes by type
        print("\n📈 NODE TYPE STATISTICS")
        print("-" * 30)
        type_counts = figma_dfs.count_nodes_by_type(root_node)
        for node_type, count in sorted(type_counts.items()):
            print(f"{node_type}: {count}")

        # Example 2: Find nodes with specific properties
        print("\n🎨 NODES WITH SOLID FILLS")
        print("-" * 30)
        solid_fill_nodes = figma_dfs.search_by_fills(root_node, "SOLID")
        for i, node in enumerate(solid_fill_nodes[:5], 1):  # Show first 5
            print(f"{i}. {node.name} ({node.type})")

        # Example 3: Find largest nodes
        print("\n📏 LARGEST NODES BY AREA")
        print("-" * 30)
        largest_nodes = figma_dfs.find_largest_nodes(root_node, limit=5)
        for i, node in enumerate(largest_nodes, 1):
            bbox = node.data.get("absoluteBoundingBox", {})
            width = bbox.get("width", 0)
            height = bbox.get("height", 0)
            area = width * height
            print(
                f"{i}. {node.name} ({node.type}) - {width:.0f}x{height:.0f} = {area:.0f}px²"
            )

        # Example 4: Get file components
        print("\n🧩 FILE COMPONENTS")
        print("-" * 30)
        components = figma_dfs.console_api.get_file_components(FILE_KEY)
        component_count = len(components.get("meta", {}).get("components", {}))
        print(f"Total components: {component_count}")

        # Example 5: Get file styles
        print("\n🎨 FILE STYLES")
        print("-" * 30)
        styles = figma_dfs.console_api.get_file_styles(FILE_KEY)
        style_count = len(styles.get("meta", {}).get("styles", {}))
        print(f"Total styles: {style_count}")

        # Example 6: Custom DFS with complex filtering
        print("\n🔍 CUSTOM DFS - FRAMES WITH CHILDREN")
        print("-" * 30)

        def complex_callback(node: FigmaNode, depth: int) -> bool:
            if node.type == "FRAME" and node.children:
                child_count = len(node.children)
                if child_count > 2:  # Only frames with more than 2 children
                    indent = "  " * depth
                    print(f"{indent}Frame: {node.name} ({child_count} children)")
            return True

        figma_dfs.depth_first_search(root_node, complex_callback, max_depth=3)

        # Example 7: Search by name pattern with case sensitivity
        print("\n🔤 NODES CONTAINING 'TEXT' (CASE SENSITIVE)")
        print("-" * 30)
        text_nodes = figma_dfs.search_by_name(root_node, "TEXT", case_sensitive=True)
        for i, node in enumerate(text_nodes[:5], 1):  # Show first 5
            print(f"{i}. {node.name} ({node.type})")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error making request to Figma API: {e}")
    except KeyError as e:
        print(f"❌ Error parsing Figma data: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def interactive_search():
    """Interactive search interface"""

    ACCESS_TOKEN = "your_figma_access_token_here"
    FILE_KEY = "your_file_key_here"

    figma_dfs = AdvancedFigmaDFS(ACCESS_TOKEN)

    try:
        # Load file
        file_data = figma_dfs.get_file(FILE_KEY)
        document_data = file_data.get("document", {})
        root_node = figma_dfs.parse_node(document_data)

        print("🔍 Interactive Figma Node Search")
        print("=" * 40)
        print("Available commands:")
        print("1. search type <node_type> - Search by node type")
        print("2. search name <pattern> - Search by name pattern")
        print("3. search fills <fill_type> - Search by fill type")
        print("4. tree <max_depth> - Show node tree")
        print("5. stats - Show node statistics")
        print("6. largest <count> - Show largest nodes")
        print("7. quit - Exit")

        while True:
            try:
                command = input("\nEnter command: ").strip().lower()

                if command == "quit":
                    break
                elif command.startswith("search type "):
                    node_type = command.split(" ", 2)[2].upper()
                    nodes = figma_dfs.search_by_type(root_node, node_type)
                    print(f"Found {len(nodes)} nodes of type '{node_type}':")
                    for i, node in enumerate(nodes[:10], 1):  # Show first 10
                        print(f"  {i}. {node.name} (ID: {node.id})")

                elif command.startswith("search name "):
                    pattern = command.split(" ", 2)[2]
                    nodes = figma_dfs.search_by_name(root_node, pattern)
                    print(f"Found {len(nodes)} nodes with name pattern '{pattern}':")
                    for i, node in enumerate(nodes[:10], 1):  # Show first 10
                        print(f"  {i}. {node.name} ({node.type})")

                elif command.startswith("search fills "):
                    fill_type = command.split(" ", 2)[2].upper()
                    nodes = figma_dfs.search_by_fills(root_node, fill_type)
                    print(f"Found {len(nodes)} nodes with fill type '{fill_type}':")
                    for i, node in enumerate(nodes[:10], 1):  # Show first 10
                        print(f"  {i}. {node.name} ({node.type})")

                elif command.startswith("tree "):
                    try:
                        max_depth = int(command.split(" ", 1)[1])
                        figma_dfs.print_node_tree(root_node, max_depth)
                    except (IndexError, ValueError):
                        figma_dfs.print_node_tree(root_node, 3)

                elif command == "stats":
                    type_counts = figma_dfs.count_nodes_by_type(root_node)
                    print("Node type statistics:")
                    for node_type, count in sorted(type_counts.items()):
                        print(f"  {node_type}: {count}")

                elif command.startswith("largest "):
                    try:
                        count = int(command.split(" ", 1)[1])
                        largest_nodes = figma_dfs.find_largest_nodes(root_node, count)
                        print(f"Top {count} largest nodes:")
                        for i, node in enumerate(largest_nodes, 1):
                            bbox = node.data.get("absoluteBoundingBox", {})
                            width = bbox.get("width", 0)
                            height = bbox.get("height", 0)
                            area = width * height
                            print(
                                f"  {i}. {node.name} ({node.type}) - {width:.0f}x{height:.0f} = {area:.0f}px²"
                            )
                    except (IndexError, ValueError):
                        print("Usage: largest <count>")

                else:
                    print("Unknown command. Type 'quit' to exit.")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

    except Exception as e:
        print(f"Error loading file: {e}")


if __name__ == "__main__":
    print("Choose an example:")
    print("1. Advanced API examples")
    print("2. Interactive search")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "2":
        interactive_search()
    else:
        advanced_example_usage()
