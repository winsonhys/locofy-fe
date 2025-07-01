#!/usr/bin/env python3
"""
Test script to verify API response filtering for the verification agent
"""

import json
from figma_agentic_verification import FigmaAgenticVerificationWorkflow


def test_api_filtering():
    """Test the API response filtering function"""

    # Sample API response data (simplified version of what we see in the logs)
    sample_api_data = {
        "nodes": {
            "1:13168": {
                "id": "1:13168",
                "name": "Input",
                "type": "INSTANCE",
                "visible": True,
                "locked": False,
                "x": -153.0,
                "y": 24.0,
                "width": 306.0,
                "height": 56.0,
                "componentId": "1:263",
                "componentProperties": {
                    "Type": {
                        "value": "Icon Right",
                        "type": "VARIANT",
                        "boundVariables": {},
                    },
                    "Size": {
                        "value": "Default",
                        "type": "VARIANT",
                        "boundVariables": {},
                    },
                    "Label": {"value": "Top", "type": "VARIANT", "boundVariables": {}},
                    "Help Text": {
                        "value": "False",
                        "type": "VARIANT",
                        "boundVariables": {},
                    },
                    "Hover": {"value": "True", "type": "VARIANT", "boundVariables": {}},
                    "Disabled": {
                        "value": "False",
                        "type": "VARIANT",
                        "boundVariables": {},
                    },
                    "Focus": {"value": "None", "type": "VARIANT", "boundVariables": {}},
                    "Unnecessary Field": {
                        "value": "Should be filtered out",
                        "type": "VARIANT",
                        "boundVariables": {},
                    },
                },
                "cornerRadius": 4.0,
                "layoutMode": "VERTICAL",
                "interactions": [
                    {
                        "trigger": {"type": "ON_CLICK"},
                        "actions": [
                            {
                                "type": "NODE",
                                "destinationId": "1:257",
                                "navigation": "CHANGE_TO",
                                "transition": {
                                    "type": "DISSOLVE",
                                    "easing": {"type": "EASE_IN_AND_OUT"},
                                    "duration": 0.05999999865889549,
                                },
                                "preserveScrollPosition": True,
                            }
                        ],
                    }
                ],
                "children_count": 1,
                "children_types": ["INSTANCE"],
                "children_names": [".Atom / Input Base"],
                "overrides": [
                    {
                        "id": "1:13168",
                        "overriddenFields": [
                            "transitionDuration",
                            "transitionEasing",
                            "transitionNodeID",
                        ],
                    }
                ],
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
                    },
                    {
                        "blendMode": "NORMAL",
                        "type": "SOLID",
                        "color": {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0},
                    },
                ],
                "strokes": [
                    {
                        "blendMode": "NORMAL",
                        "type": "SOLID",
                        "color": {"r": 0.2, "g": 0.2, "b": 0.2, "a": 1.0},
                    }
                ],
            }
        }
    }

    # Create workflow instance
    workflow = FigmaAgenticVerificationWorkflow()

    # Test filtering
    filtered_data = workflow._filter_api_response(sample_api_data)

    # Print results
    print("=" * 60)
    print("API RESPONSE FILTERING TEST")
    print("=" * 60)

    print("\nORIGINAL DATA SIZE:")
    original_json = json.dumps(sample_api_data, indent=2)
    print(f"Characters: {len(original_json)}")
    print(f"Lines: {len(original_json.split(chr(10)))}")

    print("\nFILTERED DATA SIZE:")
    filtered_json = json.dumps(filtered_data, indent=2)
    print(f"Characters: {len(filtered_json)}")
    print(f"Lines: {len(filtered_json.split(chr(10)))}")

    print(f"\nREDUCTION:")
    char_reduction = (
        (len(original_json) - len(filtered_json)) / len(original_json)
    ) * 100
    line_reduction = (
        (len(original_json.split(chr(10))) - len(filtered_json.split(chr(10))))
        / len(original_json.split(chr(10)))
    ) * 100
    print(f"Characters: {char_reduction:.1f}% reduction")
    print(f"Lines: {line_reduction:.1f}% reduction")

    print("\nCOMPACT DATA STRUCTURE:")
    print(json.dumps(filtered_data, indent=2))

    # Verify essential information is preserved in compact format
    print("\nVERIFICATION:")
    node_data = filtered_data["n"]["1:13168"]  # New compact format uses "n" key

    # Compact format: [name, type, component_id, props, text, interactions, visual, children]
    print(f"Compact format array length: {len(node_data)}")
    print(
        f"Expected: 8 elements [name, type, component_id, props, text, interactions, visual, children]"
    )

    if len(node_data) >= 8:
        (
            name,
            element_type,
            component_id,
            props,
            text,
            interactions,
            visual,
            children,
        ) = node_data

        print(f"\nExtracted data:")
        print(f"  Name: {name}")
        print(f"  Type: {element_type}")
        print(f"  Component ID: {component_id}")
        print(f"  Properties: {props}")
        print(f"  Text: {text}")
        print(f"  Interactions: {interactions}")
        print(f"  Visual: {visual}")
        print(f"  Children: {children}")

        # Verify essential information is preserved
        essential_info = {
            "name": name == "Input",
            "type": element_type == "INSTANCE",
            "component_id": component_id == "1:263",
            "has_props": props is not None and len(props) > 0,
            "has_interactions": interactions is not None and len(interactions) > 0,
            "has_visual": visual is not None,
        }

        print(f"\nEssential information preserved:")
        for key, preserved in essential_info.items():
            status = "✓" if preserved else "✗"
            print(f"  {key}: {status}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_api_filtering()
