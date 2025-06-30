#!/usr/bin/env python3
"""
Test script for Figma DFS functionality

This script demonstrates the DFS capabilities using mock Figma data
to show how the code works without requiring actual Figma API access.
"""

import json
from figma_dfs import FigmaDFS, FigmaNode


def create_mock_figma_data():
    """Create mock Figma data for testing"""
    return {
        "document": {
            "id": "0:0",
            "name": "Document",
            "type": "DOCUMENT",
            "children": [
                {
                    "id": "1:0",
                    "name": "Page 1",
                    "type": "CANVAS",
                    "children": [
                        {
                            "id": "2:0",
                            "name": "Main Frame",
                            "type": "FRAME",
                            "absoluteBoundingBox": {
                                "x": 0,
                                "y": 0,
                                "width": 800,
                                "height": 600,
                            },
                            "children": [
                                {
                                    "id": "3:0",
                                    "name": "Header",
                                    "type": "FRAME",
                                    "absoluteBoundingBox": {
                                        "x": 0,
                                        "y": 0,
                                        "width": 800,
                                        "height": 100,
                                    },
                                    "children": [
                                        {
                                            "id": "4:0",
                                            "name": "Logo",
                                            "type": "RECTANGLE",
                                            "absoluteBoundingBox": {
                                                "x": 20,
                                                "y": 20,
                                                "width": 60,
                                                "height": 60,
                                            },
                                            "fills": [
                                                {
                                                    "type": "SOLID",
                                                    "color": {
                                                        "r": 0.2,
                                                        "g": 0.4,
                                                        "b": 0.8,
                                                    },
                                                }
                                            ],
                                        },
                                        {
                                            "id": "4:1",
                                            "name": "Navigation",
                                            "type": "FRAME",
                                            "absoluteBoundingBox": {
                                                "x": 600,
                                                "y": 30,
                                                "width": 180,
                                                "height": 40,
                                            },
                                            "children": [
                                                {
                                                    "id": "5:0",
                                                    "name": "Home Button",
                                                    "type": "TEXT",
                                                    "absoluteBoundingBox": {
                                                        "x": 610,
                                                        "y": 35,
                                                        "width": 40,
                                                        "height": 20,
                                                    },
                                                },
                                                {
                                                    "id": "5:1",
                                                    "name": "About Button",
                                                    "type": "TEXT",
                                                    "absoluteBoundingBox": {
                                                        "x": 670,
                                                        "y": 35,
                                                        "width": 50,
                                                        "height": 20,
                                                    },
                                                },
                                                {
                                                    "id": "5:2",
                                                    "name": "Contact Button",
                                                    "type": "TEXT",
                                                    "absoluteBoundingBox": {
                                                        "x": 740,
                                                        "y": 35,
                                                        "width": 60,
                                                        "height": 20,
                                                    },
                                                },
                                            ],
                                        },
                                    ],
                                },
                                {
                                    "id": "3:1",
                                    "name": "Content",
                                    "type": "FRAME",
                                    "absoluteBoundingBox": {
                                        "x": 0,
                                        "y": 100,
                                        "width": 800,
                                        "height": 400,
                                    },
                                    "children": [
                                        {
                                            "id": "4:2",
                                            "name": "Main Heading",
                                            "type": "TEXT",
                                            "absoluteBoundingBox": {
                                                "x": 50,
                                                "y": 120,
                                                "width": 300,
                                                "height": 40,
                                            },
                                        },
                                        {
                                            "id": "4:3",
                                            "name": "Description",
                                            "type": "TEXT",
                                            "absoluteBoundingBox": {
                                                "x": 50,
                                                "y": 180,
                                                "width": 400,
                                                "height": 100,
                                            },
                                        },
                                        {
                                            "id": "4:4",
                                            "name": "Call to Action",
                                            "type": "RECTANGLE",
                                            "absoluteBoundingBox": {
                                                "x": 50,
                                                "y": 300,
                                                "width": 150,
                                                "height": 50,
                                            },
                                            "fills": [
                                                {
                                                    "type": "SOLID",
                                                    "color": {
                                                        "r": 0.2,
                                                        "g": 0.8,
                                                        "b": 0.2,
                                                    },
                                                }
                                            ],
                                        },
                                    ],
                                },
                                {
                                    "id": "3:2",
                                    "name": "Footer",
                                    "type": "FRAME",
                                    "absoluteBoundingBox": {
                                        "x": 0,
                                        "y": 500,
                                        "width": 800,
                                        "height": 100,
                                    },
                                    "children": [
                                        {
                                            "id": "4:5",
                                            "name": "Footer Text",
                                            "type": "TEXT",
                                            "absoluteBoundingBox": {
                                                "x": 50,
                                                "y": 520,
                                                "width": 200,
                                                "height": 20,
                                            },
                                        }
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    }


def test_basic_dfs():
    """Test basic depth-first search functionality"""
    print("🧪 Testing Basic DFS Functionality")
    print("=" * 50)

    # Create mock Figma DFS instance
    figma_dfs = FigmaDFS("mock_token")

    # Create mock data
    mock_data = create_mock_figma_data()
    root_node = figma_dfs.parse_node(mock_data["document"])

    print(f"Root node: {root_node.name} ({root_node.type})")
    print(f"Total children: {len(root_node.children) if root_node.children else 0}")

    # Test basic DFS
    print("\n📋 DFS Traversal Order:")
    visited_nodes = figma_dfs.depth_first_search(root_node)
    for i, node in enumerate(visited_nodes, 1):
        print(f"{i:2d}. {node.name} ({node.type})")

    return root_node, figma_dfs


def test_search_functionality(root_node, figma_dfs):
    """Test search functionality"""
    print("\n🔍 Testing Search Functionality")
    print("=" * 50)

    # Test search by type
    print("\n📝 Finding all TEXT nodes:")
    text_nodes = figma_dfs.search_by_type(root_node, "TEXT")
    for i, node in enumerate(text_nodes, 1):
        print(f"{i}. {node.name} (ID: {node.id})")

    print("\n🖼️ Finding all FRAME nodes:")
    frame_nodes = figma_dfs.search_by_type(root_node, "FRAME")
    for i, node in enumerate(frame_nodes, 1):
        print(f"{i}. {node.name} (ID: {node.id})")

    print("\n🔲 Finding all RECTANGLE nodes:")
    rect_nodes = figma_dfs.search_by_type(root_node, "RECTANGLE")
    for i, node in enumerate(rect_nodes, 1):
        print(f"{i}. {node.name} (ID: {node.id})")

    # Test search by name
    print("\n🔤 Finding nodes with 'Button' in name:")
    button_nodes = figma_dfs.search_by_name(root_node, "Button", case_sensitive=False)
    for i, node in enumerate(button_nodes, 1):
        print(f"{i}. {node.name} ({node.type})")


def test_advanced_functionality(root_node, figma_dfs):
    """Test advanced functionality"""
    print("\n🚀 Testing Advanced Functionality")
    print("=" * 50)

    # Test custom callback
    print("\n🎯 Custom DFS - Only FRAME nodes at depth <= 2:")

    def frame_callback(node, depth):
        if node.type == "FRAME" and depth <= 2:
            indent = "  " * depth
            print(f"{indent}Frame: {node.name} (depth: {depth})")
        return True

    figma_dfs.depth_first_search(root_node, frame_callback, max_depth=3)

    # Test node tree printing
    print("\n🌳 Node Tree (max depth 3):")
    figma_dfs.print_node_tree(root_node, max_depth=3)


def test_node_properties(root_node, figma_dfs):
    """Test node property access"""
    print("\n📊 Testing Node Properties")
    print("=" * 50)

    # Find nodes with bounding boxes
    nodes_with_bbox = []

    def bbox_callback(node, depth):
        if "absoluteBoundingBox" in node.data:
            bbox = node.data["absoluteBoundingBox"]
            width = bbox.get("width", 0)
            height = bbox.get("height", 0)
            area = width * height
            if area > 0:
                nodes_with_bbox.append((node, area))
        return True

    figma_dfs.depth_first_search(root_node, bbox_callback)

    # Sort by area
    nodes_with_bbox.sort(key=lambda x: x[1], reverse=True)

    print("📏 Nodes with largest areas:")
    for i, (node, area) in enumerate(nodes_with_bbox[:5], 1):
        bbox = node.data["absoluteBoundingBox"]
        width = bbox.get("width", 0)
        height = bbox.get("height", 0)
        print(
            f"{i}. {node.name} ({node.type}) - {width:.0f}x{height:.0f} = {area:.0f}px²"
        )

    # Test fill properties
    print("\n🎨 Nodes with fills:")
    nodes_with_fills = []

    def fill_callback(node, depth):
        if "fills" in node.data and node.data["fills"]:
            nodes_with_fills.append(node)
        return True

    figma_dfs.depth_first_search(root_node, fill_callback)

    for i, node in enumerate(nodes_with_fills, 1):
        fills = node.data["fills"]
        fill_types = [fill.get("type", "unknown") for fill in fills]
        print(f"{i}. {node.name} ({node.type}) - Fills: {fill_types}")


def main():
    """Run all tests"""
    print("🎨 Figma DFS Test Suite")
    print("=" * 60)

    try:
        # Run basic tests
        root_node, figma_dfs = test_basic_dfs()

        # Run search tests
        test_search_functionality(root_node, figma_dfs)

        # Run advanced tests
        test_advanced_functionality(root_node, figma_dfs)

        # Run property tests
        test_node_properties(root_node, figma_dfs)

        print("\n✅ All tests completed successfully!")
        print("\n💡 To test with real Figma data:")
        print("1. Get your Figma access token")
        print("2. Get your file key from the Figma URL")
        print("3. Update the credentials in figma_dfs.py or figma_console_example.py")
        print("4. Run: python figma_dfs.py")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
