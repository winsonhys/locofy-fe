#!/usr/bin/env python3
"""
Test script for the enhanced FigmaAgenticVerificationWorkflow
Demonstrates the new functionality with bundled confidence checking and specified return format.
"""

import asyncio
import json
from figma_agentic_verification import FigmaAgenticVerificationWorkflow
from config import FIGMA_FILE_KEY, START_NODE_ID


async def test_simple_usage():
    """Test the simple usage - just get final results"""
    print("🧪 Testing Simple Usage")
    print("=" * 40)

    workflow = FigmaAgenticVerificationWorkflow()

    try:
        # Get final results in the specified format
        final_results = await workflow.run_and_get_final_results(
            FIGMA_FILE_KEY, START_NODE_ID, max_depth=None  # No depth limit
        )

        print(f"✅ Success! Got {len(final_results)} results")
        print()
        print("📄 Final Results Format:")
        print(json.dumps(final_results, indent=2))

        return final_results

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


async def test_detailed_usage():
    """Test the detailed usage - get both final results and verification details"""
    print("\n🧪 Testing Detailed Usage")
    print("=" * 40)

    workflow = FigmaAgenticVerificationWorkflow()

    try:
        # Get both final results and detailed verification info
        final_results, verification_result = await workflow.run_with_detailed_results(
            FIGMA_FILE_KEY, START_NODE_ID, max_depth=None  # No depth limit
        )

        print(f"✅ Success! Got {len(final_results)} final results")
        print()

        # Show final results
        print("📄 Final Results:")
        print(json.dumps(final_results, indent=2))
        print()

        # Show detailed verification info
        print("🔍 Verification Details:")
        print(f"  - Original elements: {len(verification_result.original_results)}")
        print(f"  - Verified elements: {len(verification_result.verified_results)}")
        print(f"  - API queries: {len(verification_result.api_queries_made)}")

        if verification_result.confidence_scores:
            avg_conf = sum(verification_result.confidence_scores.values()) / len(
                verification_result.confidence_scores
            )
            print(f"  - Average confidence: {avg_conf:.2f}")

        if verification_result.verification_notes:
            print("  - Notes:")
            for note in verification_result.verification_notes:
                print(f"    • {note}")

        return final_results, verification_result

    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None


async def test_confidence_logic():
    """Test the confidence logic by examining how results are chosen"""
    print("\n🧪 Testing Confidence Logic")
    print("=" * 40)

    workflow = FigmaAgenticVerificationWorkflow()

    try:
        # Get detailed results to examine confidence logic
        final_results, verification_result = await workflow.run_with_detailed_results(
            FIGMA_FILE_KEY, START_NODE_ID, max_depth=None
        )

        print("🔍 Confidence Analysis:")
        print()

        for node_id in final_results.keys():
            final_tag = final_results[node_id]["tag"]
            original_tag = verification_result.original_results[node_id].get(
                "tag", "none"
            )

            if node_id in verification_result.verified_results:
                verified_data = verification_result.verified_results[node_id]
                verified_tag = verified_data.get("tag", "none")
                confidence = verified_data.get("confidence", 0.0)

                if confidence >= 0.5:
                    print(
                        f"  {node_id}: Using VERIFIED '{verified_tag}' (confidence: {confidence:.2f})"
                    )
                else:
                    print(
                        f"  {node_id}: Using ORIGINAL '{original_tag}' (verified confidence too low: {confidence:.2f})"
                    )
            else:
                print(
                    f"  {node_id}: Using ORIGINAL '{original_tag}' (no verification available)"
                )

        return final_results

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


async def main():
    """Run all tests"""
    print("🚀 Figma Agentic Verification Workflow Tests")
    print("=" * 60)

    # Test 1: Simple usage
    simple_results = await test_simple_usage()

    # Test 2: Detailed usage
    detailed_results, verification_result = await test_detailed_usage()

    # Test 3: Confidence logic
    confidence_results = await test_confidence_logic()

    print("\n" + "=" * 60)
    print("✅ All tests completed!")

    # Summary
    if simple_results and detailed_results and confidence_results:
        print(
            f"📊 Summary: All tests passed with {len(simple_results)} elements analyzed"
        )
    else:
        print("⚠️  Some tests failed")


if __name__ == "__main__":
    asyncio.run(main())
