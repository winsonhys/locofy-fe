#!/usr/bin/env python3
"""
Simple example demonstrating async Gemini analysis
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
    """Main async example demonstrating Gemini analysis capabilities"""

    figma_dfs = FigmaDFS(FIGMA_ACCESS_TOKEN)
    try:
        all_results = (
            await figma_dfs.get_all_inputs_buttons_and_links_from_figma_with_gemini(
                FIGMA_FILE_KEY, START_NODE_ID, max_depth=None
            )
        )
        print(all_results)
        # Group results by detection type
        results_by_type = {"text_input": [], "button": [], "link": []}
        for node_id, node_info in all_results.items():
            detection_types = node_info.get(
                "detection_types", [node_info.get("detection_type", "unknown")]
            )
            tag = node_info.get("tag", "unknown")
            for detection_type in detection_types:
                if detection_type in results_by_type:
                    results_by_type[detection_type].append((node_id, tag))

        print("Detection Results:")
        for dtype in ["text_input", "button", "link"]:
            print(f"{dtype}:")
            nodes = results_by_type[dtype]
            if nodes:
                # Extract just the node IDs for a clean list
                node_ids = [node_id for node_id, tag in nodes]
                print(f"  Node IDs: {node_ids}")
                print(f"  Count: {len(node_ids)}")
            else:
                print("  Node IDs: []")
                print("  Count: 0")

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


if __name__ == "__main__":
    # Run the async example
    asyncio.run(main())
