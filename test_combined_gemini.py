#!/usr/bin/env python3
"""
Test script for combined Gemini analysis with single API call
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


async def test_combined_gemini_analysis():
    """Test the new combined Gemini analysis functionality"""

    print("🚀 Testing Combined Gemini Analysis (Single API Call)")
    print("=" * 70)

    # Initialize Figma DFS
    figma_dfs = FigmaDFS(FIGMA_ACCESS_TOKEN)

    try:
        print(f"📁 File Key: {FIGMA_FILE_KEY}")
        print(f"🎯 Starting Node ID: {START_NODE_ID}")
        print()

        # Test combined analysis
        print("🔍 Combined Analysis: Single Gemini Call for All Element Types")
        print("-" * 70)
        print("This will detect: input, button, select, link in one API call!")
        print("Gemini will make better decisions by considering all types together.")
        print()

        start_time = time.time()

        # Use the new combined method
        combined_results = (
            await figma_dfs.get_all_elements_from_figma_with_gemini_combined(
                FIGMA_FILE_KEY, START_NODE_ID, max_depth=None
            )
        )

        end_time = time.time()
        total_time = end_time - start_time

        print(f"⏱️  Total analysis time: {total_time:.2f} seconds")
        print(f"📊 Total unique nodes found: {len(combined_results)}")
        print()

        # Group results by tag type
        results_by_tag = {"input": [], "button": [], "select": [], "link": []}
        for node_id, node_info in combined_results.items():
            tag = node_info.get("tag", "unknown")
            if tag in results_by_tag:
                results_by_tag[tag].append(node_id)

        # Display results by tag type
        for tag, node_ids in results_by_tag.items():
            print(f"🎯 {tag.upper()} Elements ({len(node_ids)} nodes):")
            if node_ids:
                for i, node_id in enumerate(node_ids[:5], 1):  # Show first 5
                    print(f"  {i}. Node {node_id}")
                if len(node_ids) > 5:
                    print(f"  ... and {len(node_ids) - 5} more")
            else:
                print("  (none)")
            print()

        # Show the combined dictionary format
        print("📋 COMBINED DETECTION DICTIONARY:")
        combined_dict = {}
        all_node_ids = []
        for node_id, node_info in combined_results.items():
            tag = node_info.get("tag", "unknown")
            combined_dict[node_id] = {"tag": tag}
            all_node_ids.append(node_id)

        print(f"Combined Dictionary: {combined_dict}")
        print(f"All Node IDs: {all_node_ids}")
        print()

        # Performance comparison
        print("⚡ PERFORMANCE COMPARISON:")
        print("-" * 30)
        print("✅ Single API call approach:")
        print(f"   - Time: {total_time:.2f} seconds")
        print(f"   - API calls: 1")
        print(f"   - Elements detected: {len(combined_results)}")
        print()
        print("💡 Benefits of combined approach:")
        print("   - Faster execution (single API call)")
        print("   - Better decision making (Gemini sees all context)")
        print("   - Reduced API costs")
        print("   - More consistent results")
        print("   - Handles edge cases better")

    except Exception as e:
        logger.error(f"Error during combined testing: {e}")
        print(f"❌ Error: {e}")


async def compare_approaches():
    """Compare the combined approach vs multiple separate calls"""

    print("\n🔄 COMPARING APPROACHES")
    print("=" * 50)

    figma_dfs = FigmaDFS(FIGMA_ACCESS_TOKEN)

    try:
        # Test combined approach
        print("1. Testing Combined Approach (Single API Call)...")
        start_time = time.time()
        combined_results = (
            await figma_dfs.get_all_elements_from_figma_with_gemini_combined(
                FIGMA_FILE_KEY, START_NODE_ID, max_depth=2
            )
        )
        combined_time = time.time() - start_time

        print(
            f"   Combined approach: {combined_time:.2f}s, {len(combined_results)} elements"
        )
        print()

        # Test multiple separate calls approach
        print("2. Testing Multiple Separate Calls...")
        start_time = time.time()

        # Get nodes first
        all_nodes = figma_dfs.depth_first_search_from_node_id(
            FIGMA_FILE_KEY, START_NODE_ID, max_depth=2
        )

        # Convert to format expected by Gemini
        nodes_with_data = []
        for node in all_nodes:
            node_dict = {
                "node_id": figma_dfs._extract_base_node_id(node.id),
                "type": node.type,
                "x": node.data.get("absoluteBoundingBox", {}).get("x", 0),
                "y": node.data.get("absoluteBoundingBox", {}).get("y", 0),
                "name": node.name,
                "data": node.data,
                "parent_id": (
                    figma_dfs._extract_base_node_id(node.parent_id)
                    if node.parent_id
                    else None
                ),
            }
            nodes_with_data.append(node_dict)

        # Multiple separate calls
        separate_results = await figma_dfs.gemini_analyzer.analyze_multiple_types(
            nodes_with_data, ["text_input", "button", "link", "select"]
        )
        separate_time = time.time() - start_time

        print(
            f"   Separate calls: {separate_time:.2f}s, {len(separate_results)} elements"
        )
        print()

        # Summary
        print("📊 COMPARISON SUMMARY:")
        print(f"Combined approach: {combined_time:.2f}s")
        print(f"Separate calls: {separate_time:.2f}s")
        print(
            f"Speed improvement: {((separate_time - combined_time) / separate_time * 100):.1f}% faster"
        )
        print()

    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_combined_gemini_analysis())
    asyncio.run(compare_approaches())
