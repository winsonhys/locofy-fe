#!/usr/bin/env python3
"""
Test that mimics the exact verification workflow to identify the issue
"""

import asyncio
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import GEMINI_API_KEY, GEMINI_MODEL_NAME, LOG_LEVEL, LOG_FORMAT

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


async def test_verification_workflow_exact():
    """Test the exact verification workflow setup"""

    print("🧪 Testing Exact Verification Workflow Setup")
    print("=" * 60)

    try:
        # Use the exact same configuration as the verification workflow
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
            max_output_tokens=2048,  # Same as verification workflow
        )

        # Test with the exact same prompt structure
        print("🔍 Test 1: Exact verification prompt")

        # Create the exact prompt from the chunk log
        prompt = """You are a Figma UI element verification expert. You have access to the Figma API to verify the classification of UI elements.

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
            HumanMessage(content=prompt),
        ]

        print(f"Prompt length: {len(prompt)} characters")
        print(f"Messages count: {len(messages)}")
        print()

        # Call Gemini
        print("Calling Gemini...")
        response = await llm.ainvoke(messages)
        content = response.content.strip()

        print(f"Response: '{content}'")
        print(f"Response length: {len(content)}")
        print(f"Empty response: {not content.strip()}")

        if not content.strip():
            print("❌ Empty response detected!")
        else:
            print("✅ Response received successfully")

        # Check if response contains JSON
        if "{" in content and "}" in content:
            print("✅ JSON structure detected in response")
        else:
            print("❌ No JSON structure found in response")

    except Exception as e:
        logger.error(f"Error in verification workflow test: {e}")
        print(f"❌ Error: {e}")


async def test_different_token_limits():
    """Test different token limits to see if that's the issue"""

    print("\n🔍 Testing Different Token Limits")
    print("=" * 50)

    token_limits = [1024, 2048, 4096, 8192]

    for max_tokens in token_limits:
        print(f"\nTesting with max_output_tokens: {max_tokens}")

        try:
            llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL_NAME,
                google_api_key=GEMINI_API_KEY,
                temperature=0.1,
                max_output_tokens=max_tokens,
            )

            messages = [
                SystemMessage(content="You are a UI verification expert."),
                HumanMessage(
                    content="""Analyze these UI elements and return JSON:

Node 1: button
Node 2: input

Return this exact JSON format:
{
  "verifications": {
    "1": {"tag": "button", "confidence": 0.9},
    "2": {"tag": "input", "confidence": 0.8}
  }
}"""
                ),
            ]

            response = await llm.ainvoke(messages)
            content = response.content.strip()

            print(f"  Response length: {len(content)}")
            print(f"  Empty: {not content.strip()}")
            print(f"  Has JSON: {'{' in content and '}' in content}")

        except Exception as e:
            print(f"  ❌ Error: {e}")


async def test_prompt_complexity():
    """Test different prompt complexities"""

    print("\n🔍 Testing Prompt Complexity")
    print("=" * 50)

    # Simple prompt
    print("Testing simple prompt...")
    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
            max_output_tokens=2048,
        )

        simple_messages = [
            SystemMessage(content="You are a UI verification expert."),
            HumanMessage(content='Return JSON: {"test": "simple"}'),
        ]

        response = await llm.ainvoke(simple_messages)
        content = response.content.strip()
        print(f"  Simple prompt - Empty: {not content.strip()}")

    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Complex prompt (like verification)
    print("Testing complex prompt...")
    try:
        complex_prompt = """You are a Figma UI element verification expert. You have access to the Figma API to verify the classification of UI elements.

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

        complex_messages = [
            SystemMessage(
                content="You are a Figma UI element verification expert with API access."
            ),
            HumanMessage(content=complex_prompt),
        ]

        response = await llm.ainvoke(complex_messages)
        content = response.content.strip()
        print(f"  Complex prompt - Empty: {not content.strip()}")
        print(f"  Complex prompt - Length: {len(content)}")

    except Exception as e:
        print(f"  ❌ Error: {e}")


async def main():
    """Main test function"""
    print("🎨 Verification Workflow Test Suite")
    print("=" * 70)

    await test_verification_workflow_exact()
    await test_different_token_limits()
    await test_prompt_complexity()

    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
