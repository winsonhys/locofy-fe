#!/usr/bin/env python3
"""
Agentic Workflow for Figma Node Analysis with Double Verification
1. Runs get_all_elements_from_figma_with_gemini_combined
2. Sends results to Gemini for double-checking with Figma API access
"""

import asyncio
import json
import logging
import time
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from figma_dfs import FigmaDFS
from config import (
    FIGMA_ACCESS_TOKEN,
    FIGMA_FILE_KEY,
    START_NODE_ID,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    LOG_LEVEL,
    LOG_FORMAT,
)

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of the verification process"""

    original_results: Dict[str, Dict[str, str]]
    verified_results: Dict[str, Dict[str, str]]
    verification_notes: List[str]
    confidence_scores: Dict[str, float]
    api_queries_made: List[str]


class FigmaAgenticVerificationWorkflow:
    """Agentic workflow with double verification using Figma API access"""

    def __init__(self):
        """Initialize the verification workflow"""
        self.figma_dfs = FigmaDFS(FIGMA_ACCESS_TOKEN)
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
            max_output_tokens=2048,
        )
        self.figma_api_token = FIGMA_ACCESS_TOKEN
        self.base_url = "https://api.figma.com/v1"
        self.headers = {
            "X-Figma-Token": self.figma_api_token,
            "Content-Type": "application/json",
        }

    async def _query_figma_api(self, file_key: str, node_id: str) -> Dict[str, Any]:
        """Query Figma API for specific node data"""
        try:
            url = f"{self.base_url}/files/{file_key}/nodes?ids={node_id}"
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: requests.get(url, headers=self.headers)
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error querying Figma API for node {node_id}: {e}")
            return {}

    def _create_verification_prompt(
        self, original_results: Dict[str, Dict[str, str]], file_key: str
    ) -> str:
        """Create a prompt for Gemini to verify results with Figma API access"""

        # Convert results to a more readable format
        results_summary = []
        for node_id, result in original_results.items():
            tag = result.get("tag", "unknown")
            results_summary.append(f"Node {node_id}: {tag}")

        prompt = f"""You are a Figma UI element verification expert. You have access to the Figma API to verify the classification of UI elements.

ORIGINAL ANALYSIS RESULTS:
{chr(10).join(results_summary)}

FILE KEY: {file_key}

YOUR TASK:
1. Review each classification (link, button, input, select, none)
2. If you need more information about a specific node, you can query the Figma API using this format:
   QUERY_API: {{"node_id": "node_id_here"}}
3. Provide your verification with confidence scores (0.0-1.0)

AVAILABLE FIGMA API ENDPOINTS:
- GET /files/{{file_key}}/nodes?ids={{node_id}} - Get specific node data

RESPONSE FORMAT:
{{
  "verifications": {{
    "node_id": {{
      "original_tag": "original_tag",
      "verified_tag": "verified_tag", 
      "confidence": 0.95,
      "notes": "verification notes"
    }}
  }},
  "api_queries": ["list of node_ids that were queried"],
  "overall_confidence": 0.92
}}

Start by reviewing the results. If you need more details about specific nodes, use QUERY_API format."""

        return prompt

    async def _handle_verification_with_api_access(
        self, original_results: Dict[str, Dict[str, str]], file_key: str
    ) -> VerificationResult:
        """Handle the verification process with Figma API access"""

        verification_notes = []
        api_queries_made = []
        confidence_scores = {}
        verified_results = {}

        # Create initial verification prompt
        prompt = self._create_verification_prompt(original_results, file_key)

        # Start conversation with Gemini
        messages = [
            SystemMessage(
                content="You are a Figma UI element verification expert with API access."
            ),
            HumanMessage(content=prompt),
        ]

        max_iterations = 5  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            try:
                # Get response from Gemini
                response = await self.llm.ainvoke(messages)
                content = response.content.strip()

                # Check if Gemini wants to query the API
                if "QUERY_API:" in content:
                    # Extract API queries
                    lines = content.split("\n")
                    new_messages = []

                    for line in lines:
                        if line.strip().startswith("QUERY_API:"):
                            try:
                                # Extract the JSON part after QUERY_API:
                                json_str = line.replace("QUERY_API:", "").strip()
                                query_data = json.loads(json_str)
                                node_id = query_data.get("node_id")

                                if node_id:
                                    logger.info(
                                        f"Gemini requesting API data for node: {node_id}"
                                    )
                                    api_queries_made.append(node_id)

                                    # Query Figma API
                                    api_data = await self._query_figma_api(
                                        file_key, node_id
                                    )

                                    # Add API response to conversation
                                    api_response = f"API DATA for node {node_id}: {json.dumps(api_data, indent=2)}"
                                    new_messages.append(
                                        HumanMessage(content=api_response)
                                    )

                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse API query: {line}")
                                continue
                        else:
                            new_messages.append(HumanMessage(content=line))

                    # Add new messages to conversation
                    messages.extend(new_messages)
                    messages.append(
                        HumanMessage(
                            content="Please continue with your verification analysis."
                        )
                    )

                else:
                    # Try to parse the final verification result
                    try:
                        # Extract JSON from response
                        if "```json" in content:
                            start = content.find("```json") + 7
                            end = content.find("```", start)
                            json_content = content[start:end].strip()
                        elif "```" in content:
                            start = content.find("```") + 3
                            end = content.find("```", start)
                            json_content = content[start:end].strip()
                        else:
                            json_content = content

                        verification_data = json.loads(json_content)

                        # Extract verification results
                        verifications = verification_data.get("verifications", {})
                        for node_id, verification in verifications.items():
                            verified_results[node_id] = {
                                "tag": verification.get("verified_tag", "none"),
                                "confidence": verification.get("confidence", 0.5),
                                "notes": verification.get("notes", ""),
                            }
                            confidence_scores[node_id] = verification.get(
                                "confidence", 0.5
                            )

                        # Extract overall confidence
                        overall_confidence = verification_data.get(
                            "overall_confidence", 0.5
                        )
                        verification_notes.append(
                            f"Overall confidence: {overall_confidence}"
                        )

                        # Add API queries to notes
                        api_queries = verification_data.get("api_queries", [])
                        if api_queries:
                            verification_notes.append(
                                f"API queries made: {', '.join(api_queries)}"
                            )

                        break  # Exit the loop

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse verification response: {e}")
                        verification_notes.append(
                            f"Failed to parse response: {content[:100]}..."
                        )
                        break

                iteration += 1

            except Exception as e:
                logger.error(f"Error in verification iteration {iteration}: {e}")
                verification_notes.append(f"Error: {str(e)}")
                break

        return VerificationResult(
            original_results=original_results,
            verified_results=verified_results,
            verification_notes=verification_notes,
            confidence_scores=confidence_scores,
            api_queries_made=api_queries_made,
        )

    async def run_agentic_verification(
        self, file_key: str, node_id: str, max_depth: Optional[int] = None
    ) -> VerificationResult:
        """
        Run the complete agentic verification workflow

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            max_depth: Maximum depth for DFS traversal

        Returns:
            VerificationResult with original and verified results
        """
        logger.info(f"🚀 Starting agentic verification workflow")
        logger.info(f"File: {file_key}, Node: {node_id}, Max Depth: {max_depth}")

        start_time = time.time()

        try:
            # Step 1: Run the original combined analysis
            logger.info("📋 Step 1: Running original combined analysis...")
            original_results = (
                await self.figma_dfs.get_all_elements_from_figma_with_gemini_combined(
                    file_key, node_id, max_depth
                )
            )

            logger.info(
                f"✅ Original analysis completed: {len(original_results)} elements found"
            )

            # Step 2: Double-check with Gemini and Figma API access
            logger.info("🔍 Step 2: Running verification with Figma API access...")
            verification_result = await self._handle_verification_with_api_access(
                original_results, file_key
            )

            end_time = time.time()
            total_time = end_time - start_time

            logger.info(f"✅ Verification completed in {total_time:.2f} seconds")
            logger.info(
                f"📊 API queries made: {len(verification_result.api_queries_made)}"
            )

            return verification_result

        except Exception as e:
            logger.error(f"❌ Agentic verification failed: {e}")
            return VerificationResult(
                original_results={},
                verified_results={},
                verification_notes=[f"Error: {str(e)}"],
                confidence_scores={},
                api_queries_made=[],
            )


async def main():
    """Example usage of the agentic verification workflow"""
    print("🚀 Figma Agentic Verification Workflow")
    print("=" * 50)

    workflow = FigmaAgenticVerificationWorkflow()

    try:
        print("🔍 Starting agentic verification...")
        start_time = time.time()

        result = await workflow.run_agentic_verification(
            FIGMA_FILE_KEY, START_NODE_ID, max_depth=2
        )

        end_time = time.time()
        analysis_time = end_time - start_time

        print(f"⏱️  Analysis completed in {analysis_time:.2f} seconds")
        print()

        if result.original_results:
            print("✅ Agentic verification successful!")
            print()

            # Display original results
            print("📋 ORIGINAL ANALYSIS RESULTS:")
            original_count = len(result.original_results)
            print(f"Total elements: {original_count}")

            original_tags = {}
            for node_id, data in result.original_results.items():
                tag = data.get("tag", "none")
                original_tags[tag] = original_tags.get(tag, 0) + 1
                print(f"  {node_id}: {tag}")

            print()
            print("🎯 Original Distribution:")
            for tag, count in original_tags.items():
                print(f"  {tag.upper()}: {count}")

            print()

            # Display verified results
            if result.verified_results:
                print("🔍 VERIFIED RESULTS:")
                verified_count = len(result.verified_results)
                print(f"Total verified: {verified_count}")

                verified_tags = {}
                changes = []

                for node_id, data in result.verified_results.items():
                    original_tag = result.original_results.get(node_id, {}).get(
                        "tag", "none"
                    )
                    verified_tag = data.get("tag", "none")
                    confidence = data.get("confidence", 0.0)
                    notes = data.get("notes", "")

                    verified_tags[verified_tag] = verified_tags.get(verified_tag, 0) + 1

                    if original_tag != verified_tag:
                        changes.append(
                            f"  {node_id}: {original_tag} → {verified_tag} (confidence: {confidence:.2f})"
                        )
                    else:
                        print(
                            f"  {node_id}: {verified_tag} ✓ (confidence: {confidence:.2f})"
                        )

                print()
                print("🎯 Verified Distribution:")
                for tag, count in verified_tags.items():
                    print(f"  {tag.upper()}: {count}")

                if changes:
                    print()
                    print("🔄 CHANGES MADE:")
                    for change in changes:
                        print(change)

            # Display verification notes
            if result.verification_notes:
                print()
                print("📝 VERIFICATION NOTES:")
                for note in result.verification_notes:
                    print(f"  • {note}")

            # Display API usage
            if result.api_queries_made:
                print()
                print("🔗 FIGMA API QUERIES:")
                for query in result.api_queries_made:
                    print(f"  • {query}")

            # Performance metrics
            print()
            print("⚡ PERFORMANCE METRICS:")
            print(f"  - Total time: {analysis_time:.2f} seconds")
            print(f"  - API queries: {len(result.api_queries_made)}")
            print(f"  - Elements per second: {original_count / analysis_time:.2f}")

        else:
            print("❌ No results obtained")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
