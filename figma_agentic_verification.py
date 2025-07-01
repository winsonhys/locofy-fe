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

    def _filter_api_response(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ultra-compact API response format for minimal token usage"""
        if not api_data or "nodes" not in api_data:
            return {}

        # Use compact format: node_id -> [name, type, component_id, props, text, interactions]
        compact_data = {}

        for node_id, node_data in api_data["nodes"].items():
            # Handle both direct node data and nested document structure
            if "document" in node_data:
                doc = node_data["document"]
            else:
                doc = node_data

            # Extract only the most critical fields in compact format
            compact_node = []

            # 0: name
            compact_node.append(doc.get("name", ""))

            # 1: type
            compact_node.append(doc.get("type", ""))

            # 2: component_id (or None)
            compact_node.append(doc.get("componentId"))

            # 3: essential component properties (filtered and simplified)
            props = {}
            if "componentProperties" in doc:
                for key, value in doc["componentProperties"].items():
                    # Only keep properties that directly indicate UI element type
                    if any(
                        keyword in key.lower()
                        for keyword in [
                            "type",
                            "state",
                            "variant",
                            "placeholder",
                            "text",
                            "label",
                            "button",
                            "input",
                            "select",
                            "disabled",
                        ]
                    ):
                        # Extract just the value, not the full object
                        if isinstance(value, dict) and "value" in value:
                            props[key] = value["value"]
                        else:
                            props[key] = str(value)
            compact_node.append(props if props else None)

            # 4: text content
            compact_node.append(doc.get("characters", ""))

            # 5: interactions (simplified to just trigger types)
            interactions = []
            if "interactions" in doc and doc["interactions"]:
                for interaction in doc["interactions"]:
                    trigger = interaction.get("trigger", {})
                    trigger_type = trigger.get("type", "")
                    if trigger_type:
                        interactions.append(trigger_type)
            compact_node.append(interactions if interactions else None)

            # 6: visual indicators (corner radius, fills, strokes)
            visual = {}
            if "cornerRadius" in doc:
                visual["r"] = doc["cornerRadius"]
            if "fills" in doc and doc["fills"]:
                visual["f"] = 1  # Has fills
            if "strokes" in doc and doc["strokes"]:
                visual["s"] = 1  # Has strokes
            compact_node.append(visual if visual else None)

            # 7: children summary (just count and main types)
            children_info = None
            if "children" in doc and doc["children"]:
                child_types = list(set(child.get("type") for child in doc["children"]))
                children_info = {
                    "c": len(doc["children"]),
                    "t": child_types[:2],  # Only first 2 types
                }
            compact_node.append(children_info)

            compact_data[node_id] = compact_node

        return {"n": compact_data}  # "n" for nodes

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

COMPACT API RESPONSE FORMAT:
API responses use ultra-compact format: {{"n": {{"node_id": [name, type, component_id, props, text, interactions, visual, children]}}}}
- name: element name
- type: element type (INSTANCE, FRAME, etc.)
- component_id: component identifier (if any)
- props: essential component properties (type, state, variant, placeholder, text, label, button, input, select, disabled)
- text: text content
- interactions: list of interaction trigger types (ON_CLICK, ON_HOVER, etc.)
- visual: {{"r": cornerRadius, "f": hasFills, "s": hasStrokes}}
- children: {{"c": count, "t": [types]}}

IMPORTANT: You must respond with valid JSON only. Do not include any text before or after the JSON.

RESPONSE FORMAT (respond with this exact JSON structure):
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

Start by reviewing the results. If you need more details about specific nodes, use QUERY_API format. When you're ready to provide verification, respond with ONLY the JSON object above."""

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

        # Write prompts to file for transparency
        prompt_log = []
        prompt_log.append("=" * 80)
        prompt_log.append("FIGMA AGENTIC VERIFICATION PROMPTS LOG")
        prompt_log.append("=" * 80)
        prompt_log.append(f"File Key: {file_key}")
        prompt_log.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        prompt_log.append("")

        # Start conversation with Gemini
        messages = [
            SystemMessage(
                content="You are a Figma UI element verification expert with API access."
            ),
            HumanMessage(content=prompt),
        ]

        # Log initial prompt
        prompt_log.append("INITIAL PROMPT:")
        prompt_log.append("-" * 40)
        prompt_log.append("System Message:")
        prompt_log.append(
            "You are a Figma UI element verification expert with API access."
        )
        prompt_log.append("")
        prompt_log.append("User Message:")
        prompt_log.append(prompt)
        prompt_log.append("")

        max_iterations = 5  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            try:
                # Get response from Gemini
                response = await self.llm.ainvoke(messages)
                content = response.content.strip()

                # Log Gemini's response
                prompt_log.append(f"ITERATION {iteration + 1} - GEMINI RESPONSE:")
                prompt_log.append("-" * 40)
                prompt_log.append(content)
                prompt_log.append("")

                # Check if Gemini wants to query the API
                if "QUERY_API:" in content:
                    # Extract API queries
                    lines = content.split("\n")
                    new_messages = []

                    prompt_log.append(
                        f"ITERATION {iteration + 1} - API QUERIES DETECTED:"
                    )
                    prompt_log.append("-" * 40)

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

                                    # Log API query
                                    prompt_log.append(f"API Query: {json_str}")

                                    # Query Figma API
                                    api_data = await self._query_figma_api(
                                        file_key, node_id
                                    )

                                    # Filter API response to reduce token usage
                                    filtered_api_data = self._filter_api_response(
                                        api_data
                                    )

                                    # Log API response
                                    prompt_log.append("API Response:")
                                    prompt_log.append(
                                        json.dumps(filtered_api_data, indent=2)
                                    )
                                    prompt_log.append("")

                                    # Add filtered API response to conversation
                                    api_response = f"API DATA for node {node_id}: {json.dumps(filtered_api_data, indent=2)}"
                                    new_messages.append(
                                        HumanMessage(content=api_response)
                                    )

                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse API query: {line}")
                                prompt_log.append(
                                    f"ERROR: Failed to parse API query: {line}"
                                )
                                continue
                        else:
                            new_messages.append(HumanMessage(content=line))

                    # Add new messages to conversation
                    messages.extend(new_messages)
                    follow_up_message = (
                        "Please continue with your verification analysis."
                    )
                    messages.append(HumanMessage(content=follow_up_message))

                    # Log follow-up message
                    prompt_log.append("Follow-up Message:")
                    prompt_log.append(follow_up_message)
                    prompt_log.append("")

                else:
                    # Try to parse the final verification result
                    try:
                        # Check if response is empty
                        if not content.strip():
                            logger.error("Gemini returned empty response")
                            verification_notes.append(
                                "Error: Gemini returned empty response"
                            )
                            break

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
                            # Look for JSON object in the response
                            brace_start = content.find("{")
                            brace_end = content.rfind("}")
                            if (
                                brace_start != -1
                                and brace_end != -1
                                and brace_end > brace_start
                            ):
                                json_content = content[brace_start : brace_end + 1]

                        if not json_content.strip():
                            logger.error("No JSON content found in response")
                            verification_notes.append(
                                "Error: No JSON content found in response"
                            )
                            break

                        verification_data = json.loads(json_content)

                        # Log final verification result
                        prompt_log.append("FINAL VERIFICATION RESULT:")
                        prompt_log.append("-" * 40)
                        prompt_log.append(json.dumps(verification_data, indent=2))
                        prompt_log.append("")

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
                        prompt_log.append(
                            "ERROR: Failed to parse verification response"
                        )
                        prompt_log.append(f"Error: {str(e)}")
                        prompt_log.append("Raw response:")
                        prompt_log.append(content)

                        # Try to create a fallback verification based on original results
                        logger.info(
                            "Creating fallback verification based on original results"
                        )
                        for node_id, original_data in original_results.items():
                            verified_results[node_id] = {
                                "tag": original_data.get("tag", "none"),
                                "confidence": 0.7,  # Lower confidence due to parsing failure
                                "notes": "Fallback verification due to parsing error",
                            }
                            confidence_scores[node_id] = 0.7

                        verification_notes.append(
                            "Fallback verification applied due to parsing error"
                        )
                        break

                iteration += 1

            except Exception as e:
                logger.error(f"Error in verification iteration {iteration}: {e}")
                verification_notes.append(f"Error: {str(e)}")
                prompt_log.append(f"ERROR in iteration {iteration}: {str(e)}")
                break

        # Write prompts to file
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"verification_prompts_{timestamp}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(prompt_log))
            logger.info(f"Verification prompts written to: {filename}")
            verification_notes.append(f"Prompts logged to: {filename}")
        except Exception as e:
            logger.error(f"Failed to write prompts to file: {e}")
            verification_notes.append(f"Failed to write prompts: {str(e)}")

        return VerificationResult(
            original_results=original_results,
            verified_results=verified_results,
            verification_notes=verification_notes,
            confidence_scores=confidence_scores,
            api_queries_made=api_queries_made,
        )

    def _log_verification_results(
        self, result: VerificationResult, file_key: str
    ) -> None:
        """Log verification results in the specified format"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"verification_results_{file_key}_{timestamp}.json"

            # Filter and format results
            high_confidence_results = {}
            low_confidence_results = {}

            for node_id, data in result.verified_results.items():
                verified_tag = data.get("tag", "none")
                confidence = data.get("confidence", 0.0)

                # Format: {node_id: {"tag": type}}
                formatted_result = {node_id: {"tag": verified_tag}}

                if confidence >= 0.5:
                    high_confidence_results[node_id] = formatted_result
                else:
                    low_confidence_results[node_id] = {
                        "result": formatted_result,
                        "confidence": confidence,
                        "notes": data.get("notes", ""),
                    }

            # Create log data
            log_data = {
                "file_key": file_key,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "high_confidence_results": high_confidence_results,
                "low_confidence_results": low_confidence_results,
                "summary": {
                    "total_elements": len(result.verified_results),
                    "high_confidence_count": len(high_confidence_results),
                    "low_confidence_count": len(low_confidence_results),
                    "overall_confidence": (
                        sum(result.confidence_scores.values())
                        / len(result.confidence_scores)
                        if result.confidence_scores
                        else 0.0
                    ),
                },
            }

            # Write to file
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2)

            logger.info(f"Verification results logged to: {filename}")

        except Exception as e:
            logger.error(f"Failed to log verification results: {e}")

    def get_final_results(
        self, verification_result: VerificationResult
    ) -> Dict[str, Dict[str, str]]:
        """
        Get final results in the specified format, bundling confidence check into the workflow.

        This method combines original and verified results, using verified results for high-confidence
        classifications and falling back to original results for low-confidence ones.

        Args:
            verification_result: The VerificationResult from run_agentic_verification

        Returns:
            Dictionary in format: {node_id: {"tag": detected_tag}}
        """
        final_results = {}

        # Process each node
        for node_id in verification_result.original_results.keys():
            original_tag = verification_result.original_results[node_id].get(
                "tag", "none"
            )

            # Check if we have verified results for this node
            if node_id in verification_result.verified_results:
                verified_data = verification_result.verified_results[node_id]
                verified_tag = verified_data.get("tag", "none")
                confidence = verified_data.get("confidence", 0.0)

                # Use verified result if confidence is high enough (≥50%)
                if confidence >= 0.5:
                    final_results[node_id] = {"tag": verified_tag}
                    logger.debug(
                        f"Node {node_id}: Using verified result '{verified_tag}' (confidence: {confidence:.2f})"
                    )
                else:
                    # Fall back to original result for low confidence
                    final_results[node_id] = {"tag": original_tag}
                    logger.debug(
                        f"Node {node_id}: Using original result '{original_tag}' (verified confidence too low: {confidence:.2f})"
                    )
            else:
                # No verification available, use original result
                final_results[node_id] = {"tag": original_tag}
                logger.debug(
                    f"Node {node_id}: Using original result '{original_tag}' (no verification available)"
                )

        logger.info(f"Final results generated: {len(final_results)} elements")
        return final_results

    async def run_and_get_final_results(
        self, file_key: str, node_id: str, max_depth: Optional[int] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Run the complete agentic verification workflow and return final results in the specified format.

        This is a convenience method that combines run_agentic_verification and get_final_results.

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            max_depth: Maximum depth for DFS traversal

        Returns:
            Dictionary in format: {node_id: {"tag": detected_tag}}
        """
        logger.info(f"🚀 Running complete workflow with final results output")

        # Run the verification workflow
        verification_result = await self.run_agentic_verification(
            file_key, node_id, max_depth
        )

        # Get final results in the specified format
        final_results = self.get_final_results(verification_result)

        logger.info(
            f"✅ Complete workflow finished: {len(final_results)} final results"
        )
        return final_results

    async def run_with_detailed_results(
        self, file_key: str, node_id: str, max_depth: Optional[int] = None
    ) -> tuple[Dict[str, Dict[str, str]], VerificationResult]:
        """
        Run the complete agentic verification workflow and return both final results and detailed verification info.

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            max_depth: Maximum depth for DFS traversal

        Returns:
            Tuple of (final_results, verification_result) where:
            - final_results: Dictionary in format: {node_id: {"tag": detected_tag}}
            - verification_result: Detailed VerificationResult with confidence scores, notes, etc.
        """
        logger.info(f"🚀 Running complete workflow with detailed results output")

        # Run the verification workflow
        verification_result = await self.run_agentic_verification(
            file_key, node_id, max_depth
        )

        # Get final results in the specified format
        final_results = self.get_final_results(verification_result)

        logger.info(
            f"✅ Complete workflow finished: {len(final_results)} final results"
        )
        return final_results, verification_result

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

            # Log verification results
            self._log_verification_results(verification_result, file_key)

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

        # Get final results in the specified format
        final_results, verification_result = await workflow.run_with_detailed_results(
            FIGMA_FILE_KEY,
            START_NODE_ID,
            max_depth=None,  # Use smaller depth for faster testing
        )

        end_time = time.time()
        analysis_time = end_time - start_time

        print(f"⏱️  Analysis completed in {analysis_time:.2f} seconds")
        print()

        if final_results:
            print("✅ Agentic verification successful!")
            print()

            # Display final results
            print("📋 FINAL RESULTS:")
            total_count = len(final_results)
            print(f"Total elements: {total_count}")

            # Count tags
            tag_distribution = {}
            for node_id, data in final_results.items():
                tag = data["tag"]
                tag_distribution[tag] = tag_distribution.get(tag, 0) + 1
                print(f"  {node_id}: {tag}")

            print()
            print("🎯 Final Distribution:")
            for tag, count in tag_distribution.items():
                print(f"  {tag.upper()}: {count}")

            # Show detailed verification information
            print()
            print("🔍 DETAILED VERIFICATION INFO:")
            print(
                f"  - Original elements analyzed: {len(verification_result.original_results)}"
            )
            print(f"  - Elements verified: {len(verification_result.verified_results)}")
            print(f"  - API queries made: {len(verification_result.api_queries_made)}")

            # Show confidence statistics
            if verification_result.confidence_scores:
                avg_confidence = sum(
                    verification_result.confidence_scores.values()
                ) / len(verification_result.confidence_scores)
                print(f"  - Average confidence: {avg_confidence:.2f}")

                high_conf_count = sum(
                    1
                    for conf in verification_result.confidence_scores.values()
                    if conf >= 0.5
                )
                print(f"  - High confidence (≥50%): {high_conf_count}")
                print(
                    f"  - Low confidence (<50%): {len(verification_result.confidence_scores) - high_conf_count}"
                )

            # Display verification notes
            if verification_result.verification_notes:
                print()
                print("📝 VERIFICATION NOTES:")
                for note in verification_result.verification_notes:
                    print(f"  • {note}")

            # Performance metrics
            print()
            print("⚡ PERFORMANCE METRICS:")
            print(f"  - Total time: {analysis_time:.2f} seconds")
            print(f"  - Elements per second: {total_count / analysis_time:.2f}")

            # Example of the returned format
            print()
            print("📄 RETURNED FORMAT EXAMPLE:")
            print("The function returns a dictionary in this format:")
            print("{")
            for i, (node_id, data) in enumerate(
                list(final_results.items())[:3]
            ):  # Show first 3
                tag = data["tag"]
                print(f'  "{node_id}": {{"tag": "{tag}"}}' + ("," if i < 2 else ""))
            if len(final_results) > 3:
                print("  ...")
            print("}")

            # Show usage examples
            print()
            print("💡 USAGE EXAMPLES:")
            print("  # Simple usage - just get final results")
            print(
                "  final_results = await workflow.run_and_get_final_results(file_key, node_id)"
            )
            print()
            print(
                "  # Detailed usage - get both final results and verification details"
            )
            print(
                "  final_results, verification_result = await workflow.run_with_detailed_results(file_key, node_id)"
            )
            print("  # Access confidence scores: verification_result.confidence_scores")
            print(
                "  # Access verification notes: verification_result.verification_notes"
            )

        else:
            print("❌ No results obtained")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
