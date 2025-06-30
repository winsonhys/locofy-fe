#!/usr/bin/env python3
"""
Test script for concise Gemini text input detection
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


def test_concise_text_input_detection():
    """Test the concise Gemini text input detection approach"""

    print("🧪 Testing Concise Gemini Text Input Detection")
    print("=" * 60)

    # Initialize Figma DFS
    figma_dfs = FigmaDFS(FIGMA_ACCESS_TOKEN)

    try:
        print(f"📁 File Key: {FIGMA_FILE_KEY}")
        print(f"🎯 Starting Node ID: {START_NODE_ID}")
        print()

        # Get all nodes with structure and analyze with Gemini for text inputs only
        print("🔍 Analyzing all nodes for text inputs only with Gemini...")
        print("Focus: ONLY text input elements (search bars, form fields, etc.)")
        print()

        text_input_nodes = figma_dfs.get_all_text_inputs_from_figma_with_gemini(
            FIGMA_FILE_KEY, START_NODE_ID, max_depth=None
        )

        # Display results
        print("📊 RESULTS:")
        print("=" * 40)

        if text_input_nodes:
            print(f"✅ Gemini identified {len(text_input_nodes)} text input elements:")
            print()

            for i, (node_id, node_info) in enumerate(text_input_nodes.items(), 1):
                print(f"{i}. Node ID: {node_id}")
                print(f"   Tag: {node_info.get('tag', 'input')}")
                print()
        else:
            print("❌ No text input elements identified by Gemini")
            print("This could mean:")
            print("   - The design doesn't contain text input UI elements")
            print("   - The text inputs are structured differently than expected")
            print("   - The analysis needs different parameters")

        print("🎉 Test completed!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_concise_text_input_detection()
