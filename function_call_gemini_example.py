#!/usr/bin/env python3
"""
Function Calling Gemini Example - Demonstrates using Gemini function calling instead of passing Figma tokens directly
This example shows how to analyze Figma designs using Gemini's function calling capabilities
"""

import asyncio
import logging
import time
import json
from figma_dfs_function_calling import FigmaDFSFunctionCalling
from gemini_function_caller import GeminiFunctionCaller
from config import (
    FIGMA_FILE_KEY,
    START_NODE_ID,
    LOG_LEVEL,
    LOG_FORMAT,
)

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


async def main():
    """Main function demonstrating the hybrid workflow: DFS + function calling backup"""
    print("🚀 Starting Hybrid Workflow Demo: DFS + Function Calling Backup")
    print("=" * 60)

    # Initialize the hybrid analyzer
    analyzer = FigmaDFSFunctionCalling()

    # Test file and node
    file_key = "NnmJQ6LgSUJn08LLXkylSp"
    node_id = "1:2"

    print(f"\n📁 Analyzing Figma file: {file_key}")
    print(f"🎯 Starting from node: {node_id}")

    try:
        # Step 1: Get all interactive elements using hybrid approach with limited depth
        print(
            "\n🔍 Step 1: Getting all interactive elements using hybrid approach (limited depth)..."
        )
        all_elements = await analyzer.get_all_elements_from_figma_with_gemini_combined(
            file_key, node_id, max_depth=2  # Limit depth to avoid rate limits
        )

        print(f"\n✅ Found {len(all_elements)} interactive elements:")
        for node_id, detection in all_elements.items():
            print(f"  - {node_id}: {detection}")

        # Step 2: Get text inputs specifically (also with limited depth)
        print(
            "\n🔍 Step 2: Getting text inputs using hybrid approach (limited depth)..."
        )
        text_inputs = await analyzer.get_all_text_inputs_from_figma_with_gemini(
            file_key, node_id, max_depth=2
        )

        print(f"\n✅ Found {len(text_inputs)} text inputs:")
        for node_id, detection in text_inputs.items():
            print(f"  - {node_id}: {detection}")

        # Step 3: Get inputs, buttons, and links (also with limited depth)
        print(
            "\n🔍 Step 3: Getting inputs, buttons, and links using hybrid approach (limited depth)..."
        )
        inputs_buttons_links = (
            await analyzer.get_all_inputs_buttons_and_links_from_figma_with_gemini(
                file_key, node_id, max_depth=2
            )
        )

        print(f"\n✅ Found {len(inputs_buttons_links)} inputs, buttons, and links:")
        for node_id, detection in inputs_buttons_links.items():
            print(f"  - {node_id}: {detection}")

        # Step 4: Print the node tree for context (limited depth)
        print("\n🌳 Step 4: Printing node tree structure (limited depth)...")
        await analyzer.print_node_tree(file_key, node_id, max_depth=2)

        print("\n🎉 Hybrid workflow demo completed successfully!")
        print("\n📊 Summary:")
        print(f"  - Total interactive elements found: {len(all_elements)}")
        print(f"  - Text inputs found: {len(text_inputs)}")
        print(f"  - Inputs, buttons, and links found: {len(inputs_buttons_links)}")
        print(
            "\n💡 Note: This demo used limited depth (max_depth=2) to avoid rate limits."
        )
        print(
            "   In production, you can increase the depth for more comprehensive analysis."
        )

    except Exception as e:
        print(f"\n❌ Error in hybrid workflow demo: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")


async def demonstrate_function_calling_capabilities():
    """Demonstrate the various function calling capabilities"""

    print("\n🔧 Function Calling Capabilities Demo")
    print("=" * 50)

    caller = GeminiFunctionCaller()

    try:
        print("📁 1. Getting complete file structure...")
        file_data = await caller.get_figma_file(FIGMA_FILE_KEY)
        if "error" not in file_data:
            print(
                f"   ✅ Retrieved file with {len(file_data.get('document', {}).get('children', []))} top-level nodes"
            )
        else:
            print(f"   ❌ Error: {file_data['error']}")

        print("\n🎯 2. Getting specific node data...")
        node_data = await caller.get_figma_node(FIGMA_FILE_KEY, START_NODE_ID)
        if "error" not in node_data:
            print("   ✅ Retrieved node data successfully")
        else:
            print(f"   ❌ Error: {node_data['error']}")

        print("\n🔍 3. Searching for TEXT nodes...")
        text_nodes = await caller.search_nodes_by_type(FIGMA_FILE_KEY, "TEXT")
        if "error" not in text_nodes:
            print(f"   ✅ Found {text_nodes.get('count', 0)} TEXT nodes")
        else:
            print(f"   ❌ Error: {text_nodes['error']}")

        print("\n🔍 4. Searching for nodes with 'button' in name...")
        button_nodes = await caller.search_nodes_by_name(FIGMA_FILE_KEY, "button")
        if "error" not in button_nodes:
            print(
                f"   ✅ Found {button_nodes.get('count', 0)} nodes with 'button' in name"
            )
        else:
            print(f"   ❌ Error: {button_nodes['error']}")

        print("\n✅ All function calling capabilities working!")

    except Exception as e:
        logger.error(f"Error demonstrating function calling: {e}")
        print(f"❌ Error: {e}")


async def compare_approaches():
    """Compare the old approach vs function calling approach"""

    print("\n🔄 Approach Comparison")
    print("=" * 40)

    print("OLD APPROACH:")
    print("  ❌ figma_dfs = FigmaDFS(FIGMA_ACCESS_TOKEN)")
    print("  ❌ Direct token exposure to external APIs")
    print("  ❌ Less secure and controlled")
    print()

    print("NEW FUNCTION CALLING APPROACH:")
    print("  ✅ figma_dfs = FigmaDFSFunctionCalling()")
    print("  ✅ No direct token exposure")
    print("  ✅ Gemini controls API access")
    print("  ✅ More secure and intelligent")
    print("  ✅ Better error handling and retry logic")
    print()

    print("BENEFITS:")
    print("  🔒 Security: No direct token exposure")
    print("  🧠 Intelligence: Gemini makes smart decisions about data fetching")
    print("  🛡️  Control: Gemini controls when and how to access Figma data")
    print("  🔄 Resilience: Better handling of rate limits and errors")
    print("  📊 Analytics: Gemini can optimize data requests")


if __name__ == "__main__":
    # Run the hybrid workflow demo
    asyncio.run(main())

    # # Demonstrate function calling capabilities
    # asyncio.run(demonstrate_function_calling_capabilities())

    # # Compare approaches
    # asyncio.run(compare_approaches())
