#!/usr/bin/env python3
"""
Concise example demonstrating Gemini text input detection with all node types
"""

import logging
from figma_dfs import FigmaDFS
from config import (
    FIGMA_ACCESS_TOKEN,
    FIGMA_FILE_KEY,
    START_NODE_ID,
    LOG_LEVEL,
    LOG_FORMAT,
)

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating concise Gemini text input detection"""

    logger.info("Starting concise Gemini text input detection example")
    logger.info(f"File Key: {FIGMA_FILE_KEY}")
    logger.info(f"Starting Node ID: {START_NODE_ID}")

    # Initialize Figma DFS
    figma_dfs = FigmaDFS(FIGMA_ACCESS_TOKEN)

    try:
        # Step 1: First, let's explore the starting node to understand the structure
        logger.info("Step 1: Exploring the starting node structure")
        print("=" * 80)
        print("STEP 1: EXPLORING STARTING NODE STRUCTURE")
        print("=" * 80)

        figma_dfs.print_node_tree_from_node_id(
            FIGMA_FILE_KEY, START_NODE_ID, max_depth=3
        )
        print()

        # Step 2: Get comprehensive node statistics
        logger.info("Step 2: Getting comprehensive node statistics")
        print("=" * 80)
        print("STEP 2: COMPREHENSIVE NODE STATISTICS")
        print("=" * 80)

        # Get all nodes using DFS to understand the structure
        all_nodes = figma_dfs.depth_first_search_from_node_id(
            FIGMA_FILE_KEY, START_NODE_ID, max_depth=None
        )

        # Count nodes by type
        node_type_counts = {}
        for node in all_nodes:
            node_type = node.type
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

        print(f"Total nodes found: {len(all_nodes)}")
        print("Node types breakdown:")
        for node_type, count in sorted(node_type_counts.items()):
            print(f"  - {node_type}: {count}")
        print()

        # Step 3: Use Gemini to analyze all nodes for text inputs only
        logger.info("Step 3: Using Gemini to detect text inputs only")
        print("=" * 80)
        print("STEP 3: GEMINI TEXT INPUT DETECTION")
        print("=" * 80)

        print("Sending all nodes to Gemini for text input detection...")
        print("Focus: ONLY text input elements (search bars, form fields, etc.)")
        print()

        # Use the comprehensive method
        text_input_nodes = figma_dfs.get_all_text_inputs_from_figma_with_gemini(
            FIGMA_FILE_KEY, START_NODE_ID, max_depth=None
        )

        # Step 4: Display results
        logger.info("Step 4: Displaying Gemini text input detection results")
        print("=" * 80)
        print("STEP 4: GEMINI TEXT INPUT DETECTION RESULTS")
        print("=" * 80)

        if text_input_nodes:
            print(f"Gemini identified {len(text_input_nodes)} text input elements:")
            print()
            for i, (node_id, node_info) in enumerate(text_input_nodes.items(), 1):
                print(f"{i}. Text Input:")
                print(f"   - Node ID: {node_id}")
                print(f"   - Tag: {node_info.get('tag', 'input')}")
                print()
        else:
            print(
                "Gemini did not identify any text input elements in the analyzed nodes."
            )
            print("This could mean:")
            print("- The design doesn't contain text input UI elements")
            print("- The text inputs are structured differently than expected")
            print("- The analysis needs different parameters")

        # Step 5: Summary
        logger.info("Step 5: Generating summary")
        print("=" * 80)
        print("STEP 5: SUMMARY")
        print("=" * 80)

        print(f"Text Input Detection Summary:")
        print(f"- Starting Node ID: {START_NODE_ID}")
        print(f"- Total nodes analyzed: {len(all_nodes)}")
        print(f"- Node types found: {len(node_type_counts)}")
        print(f"- Text input elements identified by Gemini: {len(text_input_nodes)}")
        print()
        print("Node Type Distribution:")
        for node_type, count in sorted(node_type_counts.items()):
            percentage = (count / len(all_nodes)) * 100
            print(f"  - {node_type}: {count} ({percentage:.1f}%)")

        logger.info(
            "Concise Gemini text input detection example completed successfully"
        )

    except KeyError as e:
        logger.error(f"Node ID not found: {e}")
        print(f"❌ Error: Node ID not found: {e}")
        print("💡 Make sure the START_NODE_ID in config.py exists in your Figma file")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
