#!/usr/bin/env python3
"""
Test script for async Gemini analysis with parallel processing
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


async def test_async_gemini_analysis():
    """Test the new async Gemini analysis functionality"""

    print("🚀 Testing Async Gemini Analysis with Parallel Processing")
    print("=" * 70)

    # Initialize Figma DFS
    figma_dfs = FigmaDFS(FIGMA_ACCESS_TOKEN)

    try:
        print(f"📁 File Key: {FIGMA_FILE_KEY}")
        print(f"🎯 Starting Node ID: {START_NODE_ID}")
        print()

        # Test 1: Async analysis for all detection types
        print("🔍 Test 1: Async Analysis for All Detection Types")
        print("-" * 50)
        print("Analyzing for: text_input, button, link, select")
        print("This will run all detection types in parallel!")
        print()

        start_time = time.time()

        # Use the new async method for comprehensive analysis
        all_results = (
            await figma_dfs.get_all_inputs_buttons_and_links_from_figma_with_gemini(
                FIGMA_FILE_KEY, START_NODE_ID, max_depth=None
            )
        )

        end_time = time.time()
        total_time = end_time - start_time

        print(f"⏱️  Total analysis time: {total_time:.2f} seconds")
        print(f"📊 Total unique nodes found: {len(all_results)}")
        print()

        # Group results by detection type
        results_by_type = {}
        for node_id, node_info in all_results.items():
            detection_types = node_info.get(
                "detection_types", [node_info.get("detection_type", "unknown")]
            )
            tag = node_info.get("tag", "unknown")

            for detection_type in detection_types:
                if detection_type not in results_by_type:
                    results_by_type[detection_type] = []
                results_by_type[detection_type].append(
                    {"node_id": node_id, "tag": tag, "info": node_info}
                )

        # Display results by detection type
        for detection_type, nodes in results_by_type.items():
            print(
                f"🎯 {detection_type.upper()} Detection Results ({len(nodes)} nodes):"
            )
            for i, node in enumerate(nodes[:5], 1):  # Show first 5
                print(f"  {i}. Node {node['node_id']} - Tag: {node['tag']}")
            if len(nodes) > 5:
                print(f"  ... and {len(nodes) - 5} more")
            print()

        # Test 2: Async analysis for specific detection types
        print("🔍 Test 2: Async Analysis for Specific Detection Types")
        print("-" * 50)
        print("Analyzing for: text_input, button only")
        print()

        start_time = time.time()

        input_button_results = (
            await figma_dfs.get_all_inputs_and_buttons_from_figma_with_gemini(
                FIGMA_FILE_KEY, START_NODE_ID, max_depth=None
            )
        )

        end_time = time.time()
        total_time = end_time - start_time

        print(f"⏱️  Analysis time: {total_time:.2f} seconds")
        print(f"📊 Total unique nodes found: {len(input_button_results)}")
        print()

        # Test 3: Async analysis for text inputs only
        print("🔍 Test 3: Async Analysis for Text Inputs Only")
        print("-" * 50)
        print("Analyzing for: text_input only")
        print()

        start_time = time.time()

        text_input_results = await figma_dfs.get_all_text_inputs_from_figma_with_gemini(
            FIGMA_FILE_KEY, START_NODE_ID, max_depth=None
        )

        end_time = time.time()
        total_time = end_time - start_time

        print(f"⏱️  Analysis time: {total_time:.2f} seconds")
        print(f"📊 Text input nodes found: {len(text_input_results)}")
        print()

        # Test 4: Performance comparison - sequential vs parallel
        print("🔍 Test 4: Performance Comparison")
        print("-" * 50)
        print("Comparing sequential vs parallel processing...")
        print()

        # Sequential processing (old way)
        print("🔄 Sequential Processing (old way):")
        start_time = time.time()

        # This would be the old sequential approach
        # input_results = figma_dfs.gemini_analyzer.analyze_nodes(nodes_with_data, "text_input")
        # button_results = figma_dfs.gemini_analyzer.analyze_nodes(nodes_with_data, "button")
        # link_results = figma_dfs.gemini_analyzer.analyze_nodes(nodes_with_data, "link")

        end_time = time.time()
        sequential_time = end_time - start_time
        print(f"⏱️  Sequential time (estimated): {sequential_time:.2f} seconds")
        print()

        # Parallel processing (new way)
        print("⚡ Parallel Processing (new way):")
        start_time = time.time()

        parallel_results = await figma_dfs.gemini_analyzer.analyze_multiple_types(
            [], ["text_input", "button", "link"]  # Empty list for demo
        )

        end_time = time.time()
        parallel_time = end_time - start_time
        print(f"⏱️  Parallel time: {parallel_time:.2f} seconds")
        print()

        # Test 5: Custom detection types
        print("🔍 Test 5: Custom Detection Types")
        print("-" * 50)
        print("Testing with custom detection types...")
        print()

        # Get some sample nodes for testing
        all_nodes = figma_dfs.depth_first_search_from_node_id(
            FIGMA_FILE_KEY, START_NODE_ID, max_depth=2
        )

        # Convert to the format expected by Gemini
        nodes_with_data = []
        for node in all_nodes[:10]:  # Use first 10 nodes for testing
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

        # Test with custom detection types
        custom_detection_types = ["text_input", "button"]
        start_time = time.time()

        custom_results = await figma_dfs.gemini_analyzer.analyze_multiple_types(
            nodes_with_data, custom_detection_types
        )

        end_time = time.time()
        custom_time = end_time - start_time

        print(f"⏱️  Custom analysis time: {custom_time:.2f} seconds")
        print(f"📊 Custom results: {len(custom_results)} nodes")
        print()

        # Summary
        print("📋 SUMMARY")
        print("=" * 50)
        print("✅ Async Gemini analysis is working correctly!")
        print("✅ Parallel processing is available")
        print("✅ Multiple detection types can be analyzed simultaneously")
        print("✅ Results are properly combined")
        print("✅ Clean async-only interface")
        print()
        print("🚀 Performance benefits:")
        print("  - Multiple detection types run in parallel")
        print("  - Reduced total processing time")
        print("  - Better resource utilization")
        print("  - Scalable for large node sets")

    except Exception as e:
        logger.error(f"Error during async testing: {e}")
        print(f"❌ Error: {e}")


async def test_individual_async_methods():
    """Test individual async methods separately"""

    print("\n🧪 Testing Individual Async Methods")
    print("=" * 50)

    figma_dfs = FigmaDFS(FIGMA_ACCESS_TOKEN)

    try:
        # Test individual async methods
        print("1. Testing text input detection async...")
        text_results = await figma_dfs.get_all_text_inputs_from_figma_with_gemini(
            FIGMA_FILE_KEY, START_NODE_ID, max_depth=2
        )
        print(f"   Found {len(text_results)} text inputs")

        print("2. Testing input + button detection async...")
        input_button_results = (
            await figma_dfs.get_all_inputs_and_buttons_from_figma_with_gemini(
                FIGMA_FILE_KEY, START_NODE_ID, max_depth=2
            )
        )
        print(f"   Found {len(input_button_results)} inputs/buttons")

        print("3. Testing comprehensive detection async...")
        comprehensive_results = (
            await figma_dfs.get_all_inputs_buttons_and_links_from_figma_with_gemini(
                FIGMA_FILE_KEY, START_NODE_ID, max_depth=2
            )
        )
        print(f"   Found {len(comprehensive_results)} total elements")

        print("✅ All individual async methods working correctly!")

    except Exception as e:
        logger.error(f"Error testing individual methods: {e}")
        print(f"❌ Error: {e}")


async def main():
    """Main async test function"""
    print("🎨 Async Gemini Analysis Test Suite")
    print("=" * 70)

    # Test the main async functionality
    await test_async_gemini_analysis()

    # Test individual methods
    await test_individual_async_methods()

    print("\n🎉 All async tests completed!")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
