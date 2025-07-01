#!/usr/bin/env python3
"""
Test that simulates the full verification workflow including API queries
"""

import asyncio
import logging
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import GEMINI_API_KEY, GEMINI_MODEL_NAME, LOG_LEVEL, LOG_FORMAT

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


async def test_full_verification_workflow():
    """Test the full verification workflow including API queries"""

    print("🧪 Testing Full Verification Workflow")
    print("=" * 60)

    try:
        # Initialize Gemini with same config as verification workflow
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
            max_output_tokens=2048,
        )

        # Step 1: Initial prompt
        print("🔍 Step 1: Initial prompt")
        initial_prompt = """You are a Figma UI element verification expert. You have access to the Figma API to verify the classification of UI elements.

CHUNK 1 ANALYSIS RESULTS:
Node 1:2055: input
Node 1:13168: input
Node 2:54: button
Node 4:67: input
Node 8:1722: select

FILE KEY: ks1iOmxKBMeLJoqbtCMIU4

YOUR TASK:
1. Review each classification (link, button, input, select, none) for the 5 nodes above
2. If you need more information about a specific node, you can query the Figma API using this format:
   QUERY_API: {"node_id": "node_id_here"}
3. Provide your verification with confidence scores (0.0-1.0)

DETECTION GUIDELINES:
- BUTTON: Look for button_like proportions, button_text patterns, button_name in component name, primary/secondary button styles, interactions with CLICK triggers
- INPUT: Look for input_like proportions, input_placeholder text patterns, input_name in component name, input_type properties, form-related properties
- SELECT: Look for select_name in component name, dropdown/combobox properties, options/selected properties, menu-related interactions
- LINK: Look for link_text patterns, link_name in component name, href/url properties, external/internal indicators, navigation interactions
- Use ui_hints to cross-reference with component properties and visual indicators for higher confidence

AVAILABLE FIGMA API ENDPOINTS:
- GET /files/ks1iOmxKBMeLJoqbtCMIU4/nodes?ids={node_id} - Get specific node data

COMPACT API RESPONSE FORMAT:
API responses use enhanced compact format: {"n": {"node_id": [name, type, component_id, props, text, interactions, visual, children, ui_hints]}}
- name: element name (string)
- type: element type (INSTANCE, FRAME, TEXT, etc.)
- component_id: component identifier (string, empty if none)
- props: enhanced component properties for UI detection (object, empty {} if none)
- text: text content (string, empty if none)
- interactions: enhanced interaction details [{"trigger": type, "action": type}] (array, empty [] if none)
- visual: {"r": cornerRadius, "f": hasFills, "fc": fillColor, "s": hasStrokes, "sc": strokeColor, "e": hasEffects, "o": opacity} (object, empty {} if none)
- children: {"c": count, "t": [types]} (object, always present)
- ui_hints: UI detection hints (object, empty {} if none)

IMPORTANT: You must respond with valid JSON only. Do not include any text before or after the JSON.

RESPONSE FORMAT (respond with this exact JSON structure):
{
  "verifications": {
    "node_id": {
      "original_tag": "original_tag",
      "verified_tag": "verified_tag", 
      "confidence": 0.95,
      "notes": "verification notes"
    }
  },
  "api_queries": ["list of node_ids that were queried"],
  "overall_confidence": 0.92
}

CRITICAL: You MUST provide a complete JSON response. Do not return empty responses or partial responses. If you cannot verify a node, mark it with low confidence (0.1-0.3) but still include it in the response.

Start by reviewing the results. If you need more details about specific nodes, use QUERY_API format. When you're ready to provide verification, respond with ONLY the JSON object above."""

        messages = [
            SystemMessage(
                content="You are a Figma UI element verification expert with API access. Process the nodes in this chunk and provide verification results."
            ),
            HumanMessage(content=initial_prompt),
        ]

        print("Calling Gemini with initial prompt...")
        response = await llm.ainvoke(messages)
        content = response.content.strip()

        print(f"Initial response: '{content}'")
        print(f"Response length: {len(content)}")
        print(f"Contains QUERY_API: {'QUERY_API:' in content}")

        # Step 2: Process API queries if present
        if "QUERY_API:" in content:
            print("\n🔍 Step 2: Processing API queries")

            # Extract API queries
            lines = content.split("\n")
            new_messages = []

            for line in lines:
                if line.strip().startswith("QUERY_API:"):
                    try:
                        json_str = line.replace("QUERY_API:", "").strip()
                        query_data = json.loads(json_str)
                        node_id = query_data.get("node_id")

                        if node_id:
                            print(f"Processing API query for node: {node_id}")

                            # Simulate API response (mock data)
                            mock_api_response = {
                                "n": {
                                    "1:2055": [
                                        "Input Field",
                                        "FRAME",
                                        "",
                                        {},
                                        "Enter text here",
                                        [],
                                        {
                                            "r": 8,
                                            "f": 1,
                                            "fc": {"r": 1, "g": 1, "b": 1, "a": 1},
                                        },
                                        {"c": 1, "t": ["TEXT"]},
                                        {
                                            "size": {"w": 200, "h": 40},
                                            "input_like": True,
                                            "input_placeholder": True,
                                        },
                                    ],
                                    "1:13168": [
                                        "Search Box",
                                        "FRAME",
                                        "",
                                        {},
                                        "Search...",
                                        [],
                                        {
                                            "r": 8,
                                            "f": 1,
                                            "fc": {
                                                "r": 0.95,
                                                "g": 0.95,
                                                "b": 0.95,
                                                "a": 1,
                                            },
                                        },
                                        {"c": 1, "t": ["TEXT"]},
                                        {
                                            "size": {"w": 300, "h": 40},
                                            "input_like": True,
                                            "input_placeholder": True,
                                        },
                                    ],
                                    "2:54": [
                                        "Submit Button",
                                        "FRAME",
                                        "",
                                        {},
                                        "Submit",
                                        [],
                                        {
                                            "r": 8,
                                            "f": 1,
                                            "fc": {"r": 0.2, "g": 0.6, "b": 1, "a": 1},
                                        },
                                        {"c": 1, "t": ["TEXT"]},
                                        {
                                            "size": {"w": 120, "h": 40},
                                            "button_like": True,
                                            "button_text": True,
                                        },
                                    ],
                                    "4:67": [
                                        "Email Input",
                                        "FRAME",
                                        "",
                                        {},
                                        "Enter email",
                                        [],
                                        {
                                            "r": 8,
                                            "f": 1,
                                            "fc": {"r": 1, "g": 1, "b": 1, "a": 1},
                                        },
                                        {"c": 1, "t": ["TEXT"]},
                                        {
                                            "size": {"w": 250, "h": 40},
                                            "input_like": True,
                                            "input_placeholder": True,
                                        },
                                    ],
                                    "8:1722": [
                                        "Dropdown",
                                        "FRAME",
                                        "",
                                        {},
                                        "Select option",
                                        [],
                                        {
                                            "r": 8,
                                            "f": 1,
                                            "fc": {"r": 1, "g": 1, "b": 1, "a": 1},
                                        },
                                        {"c": 2, "t": ["TEXT", "FRAME"]},
                                        {
                                            "size": {"w": 200, "h": 40},
                                            "button_like": True,
                                        },
                                    ],
                                }
                            }

                            compact_json = json.dumps(
                                mock_api_response,
                                separators=(",", ":"),
                                ensure_ascii=False,
                            )
                            api_response = (
                                f"API DATA for node {node_id}: {compact_json}"
                            )
                            new_messages.append(HumanMessage(content=api_response))
                            print(f"Added API response for {node_id}")

                    except json.JSONDecodeError as e:
                        print(f"Failed to parse API query: {line}")
                        continue
                else:
                    new_messages.append(HumanMessage(content=line))

            # Add new messages to conversation
            messages.extend(new_messages)

            # Add follow-up message
            follow_up_message = "Please continue with your verification analysis for chunk 1. Remember to provide a complete JSON response."
            messages.append(HumanMessage(content=follow_up_message))

            print(f"Added {len(new_messages)} new messages to conversation")
            print(f"Total messages: {len(messages)}")

            # Step 3: Get final response
            print("\n🔍 Step 3: Getting final response")
            print("Calling Gemini with API data...")

            response = await llm.ainvoke(messages)
            content = response.content.strip()

            print(f"Final response: '{content}'")
            print(f"Response length: {len(content)}")
            print(f"Empty response: {not content.strip()}")

            if not content.strip():
                print("❌ Empty response detected!")
            else:
                print("✅ Response received successfully")

                # Check if response contains JSON
                if "{" in content and "}" in content:
                    print("✅ JSON structure detected in response")

                    # Try to parse JSON
                    try:
                        # Extract JSON from response
                        json_content = content
                        if "```json" in content:
                            start = content.find("```json") + 7
                            end = content.find("```", start)
                            if end != -1:
                                json_content = content[start:end].strip()
                        elif "```" in content:
                            start = content.find("```") + 3
                            end = content.find("```", start)
                            if end != -1:
                                json_content = content[start:end].strip()

                        # Try to find JSON object in the response
                        if not json_content.strip().startswith("{"):
                            brace_start = content.find("{")
                            brace_end = content.rfind("}")
                            if (
                                brace_start != -1
                                and brace_end != -1
                                and brace_end > brace_start
                            ):
                                json_content = content[brace_start : brace_end + 1]

                        if json_content.strip():
                            verification_data = json.loads(json_content)
                            print("✅ JSON parsed successfully")
                            print(
                                f"Verification data: {json.dumps(verification_data, indent=2)}"
                            )
                        else:
                            print("❌ No JSON content found")

                    except json.JSONDecodeError as e:
                        print(f"❌ JSON parsing error: {e}")
                else:
                    print("❌ No JSON structure found in response")

        else:
            print("No API queries detected, processing direct response...")
            # Try to parse the response as verification data
            if "{" in content and "}" in content:
                print("✅ JSON structure detected in direct response")
            else:
                print("❌ No JSON structure found in direct response")

    except Exception as e:
        logger.error(f"Error in full verification workflow test: {e}")
        print(f"❌ Error: {e}")


async def main():
    """Main test function"""
    print("🎨 Full Verification Workflow Test")
    print("=" * 70)

    await test_full_verification_workflow()

    print("\n🎉 Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
