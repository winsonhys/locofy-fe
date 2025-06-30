#!/usr/bin/env python3
"""
Example script demonstrating input UI structure detection from Figma nodes
"""

from figma_dfs import FigmaDFS
from typing import List, Dict, Any


def example_input_detection():
    """Example of detecting input structures from Figma nodes"""

    # Sample data - replace with your actual Figma access token and file key
    ACCESS_TOKEN = "your_figma_access_token_here"
    FILE_KEY = "your_file_key_here"

    # Initialize Figma DFS
    figma_dfs = FigmaDFS(ACCESS_TOKEN)

    # Example 1: Process a list of nodes with basic information
    sample_nodes = [
        {
            "node_id": "1:2",
            "type": "FRAME",
            "x": 100,
            "y": 200,
            "name": "Search Input Field",
        },
        {
            "node_id": "1:3",
            "type": "TEXT",
            "x": 150,
            "y": 250,
            "name": "Enter search term",
        },
        {
            "node_id": "1:4",
            "type": "RECTANGLE",
            "x": 200,
            "y": 300,
            "name": "Regular Button",
        },
        {
            "node_id": "1:5",
            "type": "GROUP",
            "x": 300,
            "y": 400,
            "name": "Email Input Container",
        },
    ]

    print("Sample nodes:")
    for node in sample_nodes:
        print(f"  - {node['name']} ({node['type']}) at ({node['x']}, {node['y']})")

    # Detect input structures
    input_nodes = figma_dfs.identify_input_structures(sample_nodes)

    print(f"\nDetected {len(input_nodes)} input structures:")
    for node in input_nodes:
        print(
            f"  - {node['node_id']}: {node['type']} at ({node['x']}, {node['y']}) - {node['input_type']}"
        )

    # Example 2: Process nodes from actual Figma API
    # Uncomment the following code if you have valid Figma credentials

    """
    try:
        # Get specific nodes from Figma
        node_ids = ["1:2", "1:3", "1:4", "1:5"]  # Replace with actual node IDs
        
        print(f"\nAnalyzing nodes from Figma file: {FILE_KEY}")
        figma_input_nodes = figma_dfs.get_input_nodes_from_figma(FILE_KEY, node_ids)
        
        print(f"Found {len(figma_input_nodes)} input structures from Figma:")
        for node in figma_input_nodes:
            print(f"  - {node['node_id']}: {node['type']} at ({node['x']}, {node['y']}) - {node['input_type']}")
            
    except Exception as e:
        print(f"Error analyzing Figma nodes: {e}")
    """


def advanced_input_detection_example():
    """Example showing more advanced input detection scenarios"""

    # Sample nodes with more detailed properties
    detailed_nodes = [
        {
            "node_id": "search:1",
            "type": "FRAME",
            "x": 50,
            "y": 50,
            "name": "Search Bar",
            "data": {
                "children": [
                    {
                        "type": "TEXT",
                        "characters": "Search products...",
                        "fills": [{"type": "SOLID", "opacity": 0.5}],
                        "name": "Placeholder Text",
                    },
                    {
                        "type": "VECTOR",
                        "name": "Search Icon",
                        "fills": [{"type": "SOLID", "opacity": 1.0}],
                    },
                ]
            },
        },
        {
            "node_id": "email:1",
            "type": "GROUP",
            "x": 50,
            "y": 100,
            "name": "Email Input",
            "data": {
                "children": [
                    {
                        "type": "TEXT",
                        "characters": "Enter your email",
                        "fills": [{"type": "SOLID", "opacity": 0.6}],
                        "name": "Email Placeholder",
                    },
                    {
                        "type": "RECTANGLE",
                        "name": "Text Cursor",
                        "absoluteBoundingBox": {"width": 1, "height": 20},
                    },
                ]
            },
        },
        {
            "node_id": "button:1",
            "type": "RECTANGLE",
            "x": 50,
            "y": 150,
            "name": "Submit Button",
            "data": {
                "children": [
                    {
                        "type": "TEXT",
                        "characters": "Submit",
                        "fills": [{"type": "SOLID", "opacity": 1.0}],
                    }
                ]
            },
        },
    ]

    print("\n" + "=" * 60)
    print("ADVANCED INPUT DETECTION EXAMPLE")
    print("=" * 60)

    # Create a mock FigmaDFS instance for demonstration
    class MockFigmaDFS:
        def identify_input_structures(self, nodes):
            input_nodes = []
            for node in nodes:
                # Check for input indicators
                if self._has_input_indicators(node):
                    input_node = {
                        "node_id": node["node_id"],
                        "type": node["type"],
                        "x": node["x"],
                        "y": node["y"],
                        "input_type": "input",
                    }
                    input_nodes.append(input_node)
            return input_nodes

        def _has_input_indicators(self, node):
            name = node.get("name", "").lower()
            data = node.get("data", {})

            # Check name for input keywords
            input_keywords = ["search", "input", "email", "text field", "placeholder"]
            for keyword in input_keywords:
                if keyword in name:
                    return True

            # Check for placeholder text in children
            if self._has_placeholder_in_children(data):
                return True

            # Check for search icons
            if self._has_search_icon_in_children(data):
                return True

            # Check for cursor rectangles
            if self._has_cursor_in_children(data):
                return True

            return False

        def _has_placeholder_in_children(self, data):
            children = data.get("children", [])
            for child in children:
                if child.get("type") == "TEXT":
                    characters = child.get("characters", "").lower()
                    if any(
                        phrase in characters
                        for phrase in ["search", "enter", "type", "placeholder"]
                    ):
                        return True
            return False

        def _has_search_icon_in_children(self, data):
            children = data.get("children", [])
            for child in children:
                if (
                    child.get("type") == "VECTOR"
                    and "search" in child.get("name", "").lower()
                ):
                    return True
            return False

        def _has_cursor_in_children(self, data):
            children = data.get("children", [])
            for child in children:
                if child.get("type") == "RECTANGLE":
                    bbox = child.get("absoluteBoundingBox", {})
                    width = bbox.get("width", 0)
                    height = bbox.get("height", 0)
                    if (width < 2 and height > 10) or (height < 2 and width > 10):
                        return True
            return False

    mock_dfs = MockFigmaDFS()

    print("Detailed nodes:")
    for node in detailed_nodes:
        print(f"  - {node['name']} ({node['type']}) at ({node['x']}, {node['y']})")

    # Detect input structures
    input_nodes = mock_dfs.identify_input_structures(detailed_nodes)

    print(f"\nDetected {len(input_nodes)} input structures:")
    for node in input_nodes:
        print(
            f"  - {node['node_id']}: {node['type']} at ({node['x']}, {node['y']}) - {node['input_type']}"
        )


if __name__ == "__main__":
    print("FIGMA INPUT DETECTION EXAMPLES")
    print("=" * 50)

    example_input_detection()
    advanced_input_detection_example()

    print("\n" + "=" * 50)
    print("USAGE NOTES:")
    print("=" * 50)
    print(
        "1. Replace 'your_figma_access_token_here' with your actual Figma access token"
    )
    print("2. Replace 'your_file_key_here' with your actual Figma file key")
    print("3. Replace node IDs with actual node IDs from your Figma file")
    print("4. The detection works by analyzing:")
    print("   - Node names for input-related keywords")
    print("   - Text content for placeholder phrases")
    print("   - Opacity levels (reduced opacity = placeholder)")
    print("   - Search icons and cursor-like rectangles")
    print("   - Container types (FRAME, GROUP, etc.) with input indicators")
