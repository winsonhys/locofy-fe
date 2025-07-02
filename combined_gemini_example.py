#!/usr/bin/env python3
"""
Example demonstrating the Figma Agentic Workflow using LangGraph
"""

import asyncio
import logging
from figma_agentic_workflow import FigmaAgenticWorkflow
from config import FIGMA_FILE_KEY, START_NODE_ID

# Disable console logging - all logs will go to the timestamped file
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("figma_dfs").setLevel(logging.WARNING)
logging.getLogger("input_detection_prompts").setLevel(logging.WARNING)


async def main():
    """Main example function"""

    print("🤖 Figma Agentic Workflow Example")
    print("=" * 50)

    # Initialize the agentic workflow
    workflow = FigmaAgenticWorkflow()

    # Run the analysis
    results = await workflow.analyze(
        file_key=FIGMA_FILE_KEY,
        node_id=START_NODE_ID,
        max_depth=None,  # Limit depth for faster testing
        verbose=True,
    )

    print("\n📋 Final Results Summary:")
    print(f"Total classified elements: {len(results)}")

    if results:
        for node_id, classification in results.items():
            tag = classification.get("tag", "unknown")
            print(f"  {node_id}: {tag}")
    else:
        print("  No elements classified")

    print("\n✅ Example completed!")


if __name__ == "__main__":
    asyncio.run(main())
