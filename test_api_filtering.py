#!/usr/bin/env python3
"""
Test script for API response filtering optimizations
"""

import json
from figma_agentic_verification import FigmaAgenticVerificationWorkflow


def test_api_filtering():
    """Test the API response filtering optimizations"""

    # Sample API response data (simplified version of what we see in the logs)
    sample_api_data = {
        "nodes": {
            "1:5": {
                "document": {
                    "id": "1:5",
                    "name": "Sign up and get 20% off to your first order. Sign Up Now",
                    "type": "TEXT",
                    "characters": "Sign up and get 20% off to your first order. Sign Up Now",
                    "fills": [
                        {
                            "blendMode": "NORMAL",
                            "type": "SOLID",
                            "color": {
                                "r": 0.9709374904632568,
                                "g": 0.9759166836738586,
                                "b": 0.9958333373069763,
                                "a": 1.0,
                            },
                        }
                    ],
                    "strokes": [
                        {
                            "blendMode": "NORMAL",
                            "type": "SOLID",
                            "color": {"r": 0.2, "g": 0.2, "b": 0.2, "a": 1.0},
                        }
                    ],
                    "children": [],
                }
            },
            "1:15": {
                "document": {
                    "id": "1:15",
                    "name": "Frame 3",
                    "type": "FRAME",
                    "cornerRadius": 62.0,
                    "fills": [
                        {
                            "blendMode": "NORMAL",
                            "type": "SOLID",
                            "color": {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0},
                        }
                    ],
                    "children": [{"type": "TEXT"}, {"type": "FRAME"}],
                }
            },
            "1:21": {
                "document": {
                    "id": "1:21",
                    "name": "Frame 4",
                    "type": "FRAME",
                    "children": [{"type": "TEXT"}, {"type": "FRAME"}],
                }
            },
        }
    }

    # Create workflow instance
    workflow = FigmaAgenticVerificationWorkflow()

    # Test filtering
    filtered_data = workflow._filter_api_response(sample_api_data)

    # Test compact JSON serialization
    compact_json = workflow._create_compact_json(filtered_data)
    regular_json = json.dumps(filtered_data, indent=2)
    compact_regular_json = json.dumps(filtered_data, separators=(",", ":"))

    # Print results
    print("=" * 60)
    print("API RESPONSE FILTERING TEST")
    print("=" * 60)

    print("\nORIGINAL DATA SIZE:")
    original_json = json.dumps(sample_api_data, indent=2)
    print(f"Characters: {len(original_json)}")
    print(f"Lines: {len(original_json.split(chr(10)))}")

    print("\nFILTERED DATA SIZE:")
    print(f"Characters: {len(regular_json)}")
    print(f"Lines: {len(regular_json.split(chr(10)))}")

    print("\nCOMPACT JSON SIZE:")
    print(f"Characters: {len(compact_json)}")
    print(f"Lines: {len(compact_json.split(chr(10)))}")

    print("\nCOMPACT REGULAR JSON SIZE:")
    print(f"Characters: {len(compact_regular_json)}")
    print(f"Lines: {len(compact_regular_json.split(chr(10)))}")

    print(f"\nREDUCTION (Original -> Filtered):")
    char_reduction = (
        (len(original_json) - len(regular_json)) / len(original_json)
    ) * 100
    line_reduction = (
        (len(original_json.split(chr(10))) - len(regular_json.split(chr(10))))
        / len(original_json.split(chr(10)))
    ) * 100
    print(f"Characters: {char_reduction:.1f}% reduction")
    print(f"Lines: {line_reduction:.1f}% reduction")

    print(f"\nREDUCTION (Filtered -> Compact):")
    char_reduction_compact = (
        (len(regular_json) - len(compact_json)) / len(regular_json)
    ) * 100
    line_reduction_compact = (
        (len(regular_json.split(chr(10))) - len(compact_json.split(chr(10))))
        / len(regular_json.split(chr(10)))
    ) * 100
    print(f"Characters: {char_reduction_compact:.1f}% reduction")
    print(f"Lines: {line_reduction_compact:.1f}% reduction")

    print(f"\nTOTAL REDUCTION (Original -> Compact):")
    total_char_reduction = (
        (len(original_json) - len(compact_json)) / len(original_json)
    ) * 100
    total_line_reduction = (
        (len(original_json.split(chr(10))) - len(compact_json.split(chr(10))))
        / len(original_json.split(chr(10)))
    ) * 100
    print(f"Characters: {total_char_reduction:.1f}% reduction")
    print(f"Lines: {total_line_reduction:.1f}% reduction")

    print("\nFILTERED DATA STRUCTURE:")
    print(json.dumps(filtered_data, indent=2))

    print("\nCOMPACT JSON OUTPUT:")
    print(compact_json)

    # Test with empty fields to ensure they're properly removed
    print("\n" + "=" * 60)
    print("TESTING EMPTY FIELD REMOVAL")
    print("=" * 60)

    test_data = {
        "n": {
            "1:5": ["Button", "INSTANCE", "", {}, "", [], {}, {"c": 0, "t": []}],
            "1:15": [
                "Frame 3",
                "FRAME",
                "",
                {},
                "",
                [],
                {"r": 62.0, "f": 1},
                {"c": 2, "t": ["TEXT", "FRAME"]},
            ],
        }
    }

    compact_test = workflow._create_compact_json(test_data)
    print("Original test data:")
    print(json.dumps(test_data, indent=2))
    print("\nCompact test data:")
    print(compact_test)

    # Verify that empty fields are removed
    compact_parsed = json.loads(compact_test)
    print("\nParsed compact data:")
    print(json.dumps(compact_parsed, indent=2))


if __name__ == "__main__":
    test_api_filtering()
