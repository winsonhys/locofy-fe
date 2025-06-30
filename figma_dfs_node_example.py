#!/usr/bin/env python3
"""
Example script demonstrating DFS starting from a specific node ID
"""

from figma_dfs import FigmaDFS
from config import FIGMA_ACCESS_TOKEN, FIGMA_FILE_KEY, START_NODE_ID


def main():
    """Main function demonstrating node-based DFS"""

    # Initialize Figma DFS
    figma_dfs = FigmaDFS(FIGMA_ACCESS_TOKEN)

    # Use the START_NODE_ID from config
    print("Figma DFS - Starting from Specific Node ID")
    print("=" * 50)
    print(f"File Key: {FIGMA_FILE_KEY}")
    print(f"Starting Node ID: {START_NODE_ID}")
    print()

    try:
        # 1. Print the tree starting from this specific node
        print("1. NODE TREE STARTING FROM SPECIFIC NODE:")
        print("-" * 40)
        figma_dfs.print_node_tree_from_node_id(
            FIGMA_FILE_KEY, START_NODE_ID, max_depth=2
        )
        print()

        # 2. Search for all TEXT nodes starting from this node
        print("2. TEXT NODES FOUND FROM THIS STARTING POINT:")
        print("-" * 40)
        text_nodes = figma_dfs.search_by_type_from_node_id(
            FIGMA_FILE_KEY, START_NODE_ID, "TEXT"
        )
        if text_nodes:
            for i, node in enumerate(text_nodes, 1):
                print(f"  {i}. {node.name} (ID: {node.id})")
        else:
            print("  No TEXT nodes found")
        print()

        # 3. Search for all FRAME nodes starting from this node
        print("3. FRAME NODES FOUND FROM THIS STARTING POINT:")
        print("-" * 40)
        frame_nodes = figma_dfs.search_by_type_from_node_id(
            FIGMA_FILE_KEY, START_NODE_ID, "FRAME"
        )
        if frame_nodes:
            for i, node in enumerate(frame_nodes, 1):
                print(f"  {i}. {node.name} (ID: {node.id})")
        else:
            print("  No FRAME nodes found")
        print()

        # 4. Search for nodes containing "button" in name
        print("4. NODES WITH 'BUTTON' IN NAME:")
        print("-" * 40)
        button_nodes = figma_dfs.search_by_name_from_node_id(
            FIGMA_FILE_KEY, START_NODE_ID, "button", case_sensitive=False
        )
        if button_nodes:
            for i, node in enumerate(button_nodes, 1):
                print(f"  {i}. {node.name} ({node.type}) - ID: {node.id}")
        else:
            print("  No nodes with 'button' in name found")
        print()

        # 5. Custom DFS with callback - find all RECTANGLE nodes
        print("5. RECTANGLE NODES (CUSTOM DFS):")
        print("-" * 40)
        rectangle_nodes = []

        def rectangle_callback(node, depth):
            if node.type == "RECTANGLE":
                rectangle_nodes.append((node, depth))
            return True

        figma_dfs.depth_first_search_from_node_id(
            FIGMA_FILE_KEY, START_NODE_ID, rectangle_callback, max_depth=3
        )

        if rectangle_nodes:
            for i, (node, depth) in enumerate(rectangle_nodes, 1):
                indent = "  " * depth
                print(f"  {i}. {indent}{node.name} (ID: {node.id})")
        else:
            print("  No RECTANGLE nodes found")
        print()

    except KeyError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the node ID exists in the file")
        print(
            "💡 You can get node IDs by right-clicking on nodes in Figma and selecting 'Copy link'"
        )
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        print("💡 Check your Figma access token and file key")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
