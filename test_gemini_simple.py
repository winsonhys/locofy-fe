#!/usr/bin/env python3
"""
Simple test to check if Gemini API is working correctly
"""

import asyncio
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import GEMINI_API_KEY, GEMINI_MODEL_NAME, LOG_LEVEL, LOG_FORMAT

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


async def test_gemini_basic():
    """Test basic Gemini functionality"""

    print("🧪 Testing Basic Gemini Functionality")
    print("=" * 50)

    try:
        # Initialize Gemini
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
            max_output_tokens=8192,
        )

        # Test 1: Simple prompt
        print("🔍 Test 1: Simple prompt")
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Say 'Hello, world!' and nothing else."),
        ]

        response = await llm.ainvoke(messages)
        content = response.content.strip()

        print(f"Response: '{content}'")
        print(f"Response length: {len(content)}")
        print(f"Empty response: {not content.strip()}")
        print()

        # Test 2: JSON response
        print("🔍 Test 2: JSON response")
        messages = [
            SystemMessage(
                content="You are a helpful assistant that responds with JSON."
            ),
            HumanMessage(
                content='Respond with this exact JSON: {"message": "test", "status": "success"}'
            ),
        ]

        response = await llm.ainvoke(messages)
        content = response.content.strip()

        print(f"Response: '{content}'")
        print(f"Response length: {len(content)}")
        print(f"Empty response: {not content.strip()}")
        print()

        # Test 3: Verification-like prompt
        print("🔍 Test 3: Verification-like prompt")
        messages = [
            SystemMessage(content="You are a UI element verification expert."),
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

        print(f"Response: '{content}'")
        print(f"Response length: {len(content)}")
        print(f"Empty response: {not content.strip()}")
        print()

        # Test 4: Empty response detection
        print("🔍 Test 4: Empty response detection")
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
        logger.error(f"Error testing Gemini: {e}")
        print(f"❌ Error: {e}")


async def test_gemini_with_retry():
    """Test Gemini with retry logic"""

    print("\n🔄 Testing Gemini with Retry Logic")
    print("=" * 50)

    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
            max_output_tokens=8192,
        )

        messages = [
            SystemMessage(
                content="You are a UI verification expert. Always respond with valid JSON."
            ),
            HumanMessage(
                content="""Verify these elements and return JSON:

Node A: button
Node B: input

Format: {"results": {"A": "button", "B": "input"}}"""
            ),
        ]

        max_retries = 3
        for attempt in range(max_retries):
            print(f"Attempt {attempt + 1}/{max_retries}")

            try:
                response = await llm.ainvoke(messages)
                content = response.content.strip()

                print(f"Response: '{content}'")
                print(f"Empty: {not content.strip()}")

                if content.strip():
                    print("✅ Success!")
                    break
                else:
                    print("❌ Empty response, retrying...")
                    if attempt < max_retries - 1:
                        messages.append(
                            HumanMessage(
                                content="Please provide a JSON response. Do not return empty."
                            )
                        )

            except Exception as e:
                print(f"❌ Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise

    except Exception as e:
        logger.error(f"Error in retry test: {e}")
        print(f"❌ Error: {e}")


async def main():
    """Main test function"""
    print("🎨 Gemini API Test Suite")
    print("=" * 70)

    await test_gemini_basic()
    await test_gemini_with_retry()

    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
