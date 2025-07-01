#!/usr/bin/env python3
"""
Demonstration of Chunked Verification Approach
This script shows how breaking down verification into smaller chunks
prevents Gemini from hitting token limits and returning empty responses.
"""

import asyncio
import logging
from figma_agentic_verification import FigmaAgenticVerificationWorkflow
from config import FIGMA_FILE_KEY, START_NODE_ID, LOG_LEVEL, LOG_FORMAT

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


async def demonstrate_chunked_verification():
    """Demonstrate the chunked verification approach"""

    print("🎨 Chunked Verification Demonstration")
    print("=" * 50)
    print("This demonstrates how chunked processing prevents empty responses")
    print()

    workflow = FigmaAgenticVerificationWorkflow()

    try:
        print("🔍 Running chunked verification...")
        print("   - Processing nodes in chunks of 8")
        print("   - Each chunk gets its own Gemini conversation")
        print("   - Prevents token limit issues")
        print()

        # Run the verification
        verification_result = await workflow.run_agentic_verification(
            FIGMA_FILE_KEY, START_NODE_ID, max_depth=2
        )

        print("✅ Verification completed!")
        print()

        # Show results
        print("📊 Results Summary:")
        print(f"   Total nodes: {len(verification_result.original_results)}")
        print(f"   Verified nodes: {len(verification_result.verified_results)}")
        print(f"   API queries made: {len(verification_result.api_queries_made)}")

        # Show some sample results
        if verification_result.verified_results:
            print()
            print("🎯 Sample Verification Results:")
            sample_items = list(verification_result.verified_results.items())[:5]
            for node_id, result in sample_items:
                confidence = verification_result.confidence_scores.get(node_id, 0.0)
                print(
                    f"   Node {node_id}: {result['tag']} (confidence: {confidence:.2f})"
                )

        # Check for empty response errors
        empty_errors = [
            note
            for note in verification_result.verification_notes
            if "empty response" in note.lower()
        ]

        if empty_errors:
            print()
            print("❌ Empty response errors found:")
            for error in empty_errors:
                print(f"   - {error}")
        else:
            print()
            print("✅ No empty response errors detected!")
            print("   Chunked approach successfully prevented token limit issues")

        print()
        print("🎉 Demonstration completed successfully!")

    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(demonstrate_chunked_verification())
