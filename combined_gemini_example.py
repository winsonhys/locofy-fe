#!/usr/bin/env python3
"""
Simple example demonstrating combined Gemini analysis with single API call
"""

import asyncio
import logging
import time
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


async def main():
    """Main async example demonstrating combined Gemini analysis capabilities"""

    print("🚀 Combined Gemini Analysis Example")
    print("=" * 50)

    figma_dfs = FigmaDFS(FIGMA_ACCESS_TOKEN)
    try:
        print("🔍 Analyzing all element types with single Gemini call...")
        start_time = time.time()

        # Use the new combined method
        all_results = await figma_dfs.get_all_elements_from_figma_with_gemini_combined(
            FIGMA_FILE_KEY, START_NODE_ID, max_depth=None
        )

        end_time = time.time()
        analysis_time = end_time - start_time

        print(f"⏱️  Analysis completed in {analysis_time:.2f} seconds")
        print(f"📊 Found {len(all_results)} total elements")
        print()

        # Display the raw results
        print("📋 Raw Detection Results:")
        print(all_results)
        print()

        # Group results by tag type
        results_by_tag = {"input": [], "button": [], "select": [], "link": []}
        for node_id, node_info in all_results.items():
            tag = node_info.get("tag", "unknown")
            if tag in results_by_tag:
                results_by_tag[tag].append((node_id, node_info))

        print("🎯 Detection Results by Element Type:")
        for tag in ["input", "button", "select", "link"]:
            print(f"{tag.upper()} Elements:")
            nodes = results_by_tag[tag]
            if nodes:
                # Extract just the node IDs for a clean list
                node_ids = [node_id for node_id, node_info in nodes]
                print(f"  Node IDs: {node_ids}")
                print(f"  Count: {len(node_ids)}")
            else:
                print("  Node IDs: []")
                print("  Count: 0")
            print()

        # Show performance metrics
        print("⚡ Performance Metrics:")
        print(f"  - Single API call approach")
        print(f"  - Analysis time: {analysis_time:.2f} seconds")
        print(f"  - Total elements detected: {len(all_results)}")
        print(f"  - Elements by type:")
        for tag, nodes in results_by_tag.items():
            print(f"    * {tag}: {len(nodes)}")
        print()

    except Exception as e:
        logger.error(f"Error during combined analysis: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Run the main combined example
    asyncio.run(main())

    # Run additional tests
    # asyncio.run(test_combined_vs_separate())
    # asyncio.run(test_detailed_analysis())
    # asyncio.run(test_element_specific_analysis())
