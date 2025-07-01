#!/usr/bin/env python3
"""
Advanced Agentic Workflow for Figma Node Analysis using LangGraph
Enhanced version with parallel processing, better error handling, and sophisticated analysis
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import aiohttp

from langgraph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

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


# Pydantic models for structured output
class NodeAnalysis(BaseModel):
    """Structured analysis result for a single node"""

    tag: str = Field(description="Element type: link, button, input, select, or none")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Explanation for the classification")
    visual_indicators: List[str] = Field(
        description="Visual cues that led to this classification"
    )
    interaction_pattern: Optional[str] = Field(
        description="Expected interaction pattern"
    )


class AnalysisResults(BaseModel):
    """Structured results for all nodes"""

    nodes: Dict[str, NodeAnalysis] = Field(description="Analysis results for each node")


# State definition for the advanced agentic workflow
class AdvancedAgentState(TypedDict):
    """Enhanced state for the advanced agentic workflow"""

    file_key: str
    node_id: str
    max_depth: Optional[int]
    nodes_data: List[Dict[str, Any]]
    analysis_results: Dict[str, Dict[str, Any]]
    current_step: str
    error: Optional[str]
    metadata: Dict[str, Any]
    processing_batches: List[List[Dict[str, Any]]]
    batch_results: List[Dict[str, Any]]
    retry_count: int
    max_retries: int


@dataclass
class WorkflowConfig:
    """Configuration for the workflow"""

    batch_size: int = 10
    max_parallel_batches: int = 3
    max_retries: int = 3
    timeout_seconds: int = 30
    confidence_threshold: float = 0.7


class FigmaAdvancedAgenticWorkflow:
    """Advanced agentic workflow for analyzing Figma nodes using LangGraph"""

    def __init__(self, config: Optional[WorkflowConfig] = None):
        """Initialize the advanced agentic workflow"""
        self.config = config or WorkflowConfig()
        self.figma_dfs = FigmaDFS(FIGMA_ACCESS_TOKEN)

        # Initialize multiple LLM instances for parallel processing
        self.llm_primary = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
            max_output_tokens=8192,
        )

        self.llm_secondary = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0.05,
            max_output_tokens=4096,
        )

        self.memory = MemorySaver()
        self.workflow = self._create_workflow()
        self.output_parser = JsonOutputParser(pydantic_object=AnalysisResults)

    def _create_workflow(self) -> StateGraph:
        """Create the advanced LangGraph workflow"""
        workflow = StateGraph(AdvancedAgentState)

        # Add nodes to the workflow
        workflow.add_node("extract_nodes", self._extract_nodes)
        workflow.add_node("prepare_batches", self._prepare_batches)
        workflow.add_node("analyze_batches_parallel", self._analyze_batches_parallel)
        workflow.add_node("merge_results", self._merge_results)
        workflow.add_node("validate_and_enhance", self._validate_and_enhance)
        workflow.add_node("format_output", self._format_output)
        workflow.add_node("handle_error", self._handle_error)

        # Define the workflow edges
        workflow.set_entry_point("extract_nodes")
        workflow.add_edge("extract_nodes", "prepare_batches")
        workflow.add_edge("prepare_batches", "analyze_batches_parallel")
        workflow.add_edge("analyze_batches_parallel", "merge_results")
        workflow.add_edge("merge_results", "validate_and_enhance")
        workflow.add_edge("validate_and_enhance", "format_output")
        workflow.add_edge("format_output", END)

        # Add conditional edges for error handling and retries
        workflow.add_conditional_edges(
            "extract_nodes",
            self._should_continue_or_retry,
            {
                "continue": "prepare_batches",
                "retry": "extract_nodes",
                "error": "handle_error",
            },
        )

        workflow.add_conditional_edges(
            "analyze_batches_parallel",
            self._should_continue_or_retry,
            {
                "continue": "merge_results",
                "retry": "analyze_batches_parallel",
                "error": "handle_error",
            },
        )

        workflow.add_conditional_edges(
            "validate_and_enhance",
            self._should_continue_or_retry,
            {
                "continue": "format_output",
                "retry": "validate_and_enhance",
                "error": "handle_error",
            },
        )

        workflow.add_edge("handle_error", END)

        return workflow.compile(checkpointer=self.memory)

    def _should_continue_or_retry(self, state: AdvancedAgentState) -> str:
        """Determine if workflow should continue, retry, or end due to error"""
        if state.get("error"):
            retry_count = state.get("retry_count", 0)
            max_retries = state.get("max_retries", self.config.max_retries)

            if retry_count < max_retries:
                state["retry_count"] = retry_count + 1
                state["error"] = None  # Clear error for retry
                return "retry"
            else:
                return "error"
        return "continue"

    async def _extract_nodes(self, state: AdvancedAgentState) -> AdvancedAgentState:
        """Extract nodes from Figma using DFS with enhanced error handling"""
        try:
            logger.info(
                f"Extracting nodes from Figma file {state['file_key']}, node {state['node_id']}"
            )
            state["current_step"] = "extract_nodes"

            # Get nodes using DFS
            nodes = self.figma_dfs.depth_first_search_from_node_id(
                state["file_key"], state["node_id"], max_depth=state.get("max_depth")
            )

            # Convert nodes to serializable format with enhanced data
            nodes_data = []
            for node in nodes:
                node_data = {
                    "id": node.id,
                    "name": node.name,
                    "type": node.type,
                    "visible": node.visible,
                    "locked": node.locked,
                    "parent_id": node.parent_id,
                    "data": node.data,
                    "extracted_at": time.time(),
                }

                # Extract additional useful information
                if hasattr(node, "data") and node.data:
                    node_data.update(
                        {
                            "absolute_bounding_box": node.data.get(
                                "absoluteBoundingBox"
                            ),
                            "fills": node.data.get("fills"),
                            "strokes": node.data.get("strokes"),
                            "stroke_weight": node.data.get("strokeWeight"),
                            "corner_radius": node.data.get("cornerRadius"),
                            "characters": node.data.get("characters"),
                            "style": node.data.get("style"),
                            "effects": node.data.get("effects"),
                        }
                    )

                nodes_data.append(node_data)

            state["nodes_data"] = nodes_data
            state["metadata"] = {
                "total_nodes": len(nodes_data),
                "extraction_time": time.time(),
                "node_types": self._count_node_types(nodes_data),
            }

            logger.info(f"Extracted {len(nodes_data)} nodes successfully")
            return state

        except Exception as e:
            logger.error(f"Error extracting nodes: {e}")
            state["error"] = str(e)
            return state

    async def _prepare_batches(self, state: AdvancedAgentState) -> AdvancedAgentState:
        """Prepare nodes into batches for parallel processing"""
        try:
            logger.info("Preparing nodes into batches for parallel processing")
            state["current_step"] = "prepare_batches"

            nodes_data = state.get("nodes_data", [])
            batch_size = self.config.batch_size

            # Create batches
            batches = []
            for i in range(0, len(nodes_data), batch_size):
                batch = nodes_data[i : i + batch_size]
                batches.append(batch)

            state["processing_batches"] = batches
            state["batch_results"] = []
            state["metadata"]["batch_count"] = len(batches)
            state["metadata"]["batch_size"] = batch_size

            logger.info(f"Prepared {len(batches)} batches for processing")
            return state

        except Exception as e:
            logger.error(f"Error preparing batches: {e}")
            state["error"] = str(e)
            return state

    async def _analyze_batches_parallel(
        self, state: AdvancedAgentState
    ) -> AdvancedAgentState:
        """Analyze batches in parallel using multiple LLM instances"""
        try:
            logger.info("Starting parallel batch analysis")
            state["current_step"] = "analyze_batches_parallel"

            batches = state.get("processing_batches", [])
            if not batches:
                raise ValueError("No batches to process")

            # Create tasks for parallel processing
            tasks = []
            for i, batch in enumerate(batches):
                # Use different LLM instances for different batches
                llm = self.llm_primary if i % 2 == 0 else self.llm_secondary
                task = self._analyze_single_batch(batch, llm, i)
                tasks.append(task)

            # Process batches with concurrency limit
            semaphore = asyncio.Semaphore(self.config.max_parallel_batches)

            async def process_with_semaphore(task):
                async with semaphore:
                    return await task

            batch_results = await asyncio.gather(
                *[process_with_semaphore(task) for task in tasks],
                return_exceptions=True,
            )

            # Process results
            valid_results = []
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch {i} failed: {result}")
                    # Create empty result for failed batch
                    valid_results.append({})
                else:
                    valid_results.append(result)

            state["batch_results"] = valid_results
            state["metadata"]["analysis_time"] = time.time()

            logger.info(f"Parallel analysis completed for {len(valid_results)} batches")
            return state

        except Exception as e:
            logger.error(f"Error in parallel batch analysis: {e}")
            state["error"] = str(e)
            return state

    async def _analyze_single_batch(
        self, batch: List[Dict[str, Any]], llm, batch_index: int
    ) -> Dict[str, Any]:
        """Analyze a single batch of nodes"""
        try:
            # Create analysis prompt for this batch
            analysis_prompt = self._create_enhanced_analysis_prompt(batch, batch_index)

            # Get AI response
            messages = [
                SystemMessage(
                    content="You are an expert UI/UX analyst specializing in Figma design analysis."
                ),
                HumanMessage(content=analysis_prompt),
            ]

            response = await llm.ainvoke(messages)

            # Parse the response
            parsed_results = self._parse_enhanced_response(response.content, batch)

            return {
                "batch_index": batch_index,
                "results": parsed_results,
                "processed_at": time.time(),
            }

        except Exception as e:
            logger.error(f"Error analyzing batch {batch_index}: {e}")
            raise

    async def _merge_results(self, state: AdvancedAgentState) -> AdvancedAgentState:
        """Merge results from all batches"""
        try:
            logger.info("Merging results from all batches")
            state["current_step"] = "merge_results"

            batch_results = state.get("batch_results", [])
            merged_results = {}

            for batch_result in batch_results:
                if batch_result and "results" in batch_result:
                    merged_results.update(batch_result["results"])

            state["analysis_results"] = merged_results
            state["metadata"]["merged_results_count"] = len(merged_results)

            logger.info(
                f"Merged {len(merged_results)} results from {len(batch_results)} batches"
            )
            return state

        except Exception as e:
            logger.error(f"Error merging results: {e}")
            state["error"] = str(e)
            return state

    async def _validate_and_enhance(
        self, state: AdvancedAgentState
    ) -> AdvancedAgentState:
        """Validate and enhance analysis results"""
        try:
            logger.info("Validating and enhancing analysis results")
            state["current_step"] = "validate_and_enhance"

            results = state.get("analysis_results", {})
            enhanced_results = {}

            for node_id, result in results.items():
                # Validate result
                if self._is_valid_enhanced_result(result):
                    # Enhance result with additional analysis
                    enhanced_result = await self._enhance_result(result, node_id)
                    enhanced_results[node_id] = enhanced_result
                else:
                    logger.warning(f"Invalid result for node {node_id}: {result}")
                    # Create default result
                    enhanced_results[node_id] = {
                        "tag": "none",
                        "confidence": 0.0,
                        "reasoning": "Invalid or unclear result",
                        "visual_indicators": [],
                        "interaction_pattern": None,
                    }

            state["analysis_results"] = enhanced_results
            state["metadata"]["validation_time"] = time.time()

            logger.info(
                f"Validation and enhancement completed for {len(enhanced_results)} results"
            )
            return state

        except Exception as e:
            logger.error(f"Error validating and enhancing results: {e}")
            state["error"] = str(e)
            return state

    async def _enhance_result(
        self, result: Dict[str, Any], node_id: str
    ) -> Dict[str, Any]:
        """Enhance a single result with additional analysis"""
        try:
            # Add confidence scoring based on reasoning quality
            reasoning = result.get("reasoning", "")
            confidence = result.get("confidence", 0.0)

            # Enhance confidence based on reasoning length and quality
            if len(reasoning) > 50:
                confidence = min(1.0, confidence + 0.1)

            # Add visual indicators if not present
            if "visual_indicators" not in result:
                result["visual_indicators"] = self._extract_visual_indicators(reasoning)

            # Add interaction pattern if not present
            if "interaction_pattern" not in result:
                result["interaction_pattern"] = self._determine_interaction_pattern(
                    result.get("tag", "none")
                )

            result["confidence"] = confidence
            result["enhanced_at"] = time.time()

            return result

        except Exception as e:
            logger.error(f"Error enhancing result for node {node_id}: {e}")
            return result

    async def _format_output(self, state: AdvancedAgentState) -> AdvancedAgentState:
        """Format the final output with comprehensive statistics"""
        try:
            logger.info("Formatting final output")
            state["current_step"] = "format_output"

            results = state.get("analysis_results", {})

            # Calculate comprehensive statistics
            tag_counts = {"link": 0, "button": 0, "input": 0, "select": 0, "none": 0}
            confidence_stats = {"high": 0, "medium": 0, "low": 0}
            total_confidence = 0.0

            for result in results.values():
                tag = result.get("tag", "none")
                confidence = result.get("confidence", 0.0)

                if tag in tag_counts:
                    tag_counts[tag] += 1

                total_confidence += confidence

                if confidence >= 0.8:
                    confidence_stats["high"] += 1
                elif confidence >= 0.5:
                    confidence_stats["medium"] += 1
                else:
                    confidence_stats["low"] += 1

            avg_confidence = total_confidence / len(results) if results else 0.0

            state["metadata"].update(
                {
                    "tag_counts": tag_counts,
                    "confidence_stats": confidence_stats,
                    "average_confidence": avg_confidence,
                    "completion_time": time.time(),
                }
            )

            logger.info(f"Output formatting completed. Tag counts: {tag_counts}")
            return state

        except Exception as e:
            logger.error(f"Error formatting output: {e}")
            state["error"] = str(e)
            return state

    async def _handle_error(self, state: AdvancedAgentState) -> AdvancedAgentState:
        """Handle errors in the workflow"""
        logger.error(
            f"Workflow error at step {state.get('current_step', 'unknown')}: {state.get('error', 'Unknown error')}"
        )
        return state

    def _create_enhanced_analysis_prompt(
        self, nodes_data: List[Dict[str, Any]], batch_index: int
    ) -> str:
        """Create enhanced analysis prompt for Gemini AI"""
        prompt = f"""
You are an expert UI/UX analyst specializing in Figma design analysis. Your task is to analyze a batch of Figma nodes and determine what type of interactive element each node represents.

For each node, you must classify it as one of the following types:
- "link": Clickable links, navigation elements, or text that appears to be a hyperlink
- "button": Clickable buttons, CTAs, or interactive elements that trigger actions
- "input": Form input fields, text boxes, search bars, or data entry elements
- "select": Dropdown menus, select boxes, or choice selection elements
- "none": Non-interactive elements like text, images, or decorative elements

IMPORTANT: You must respond with a valid JSON object where:
- Keys are the node IDs
- Values are objects with "tag", "confidence" (0.0-1.0), "reasoning", "visual_indicators", and "interaction_pattern" fields

Example response format:
{{
  "node_id_1": {{
    "tag": "button",
    "confidence": 0.95,
    "reasoning": "This node has button-like styling with rounded corners, background color, and contains action text 'Submit'",
    "visual_indicators": ["rounded_corners", "background_color", "action_text"],
    "interaction_pattern": "click_to_submit"
  }},
  "node_id_2": {{
    "tag": "input",
    "confidence": 0.88,
    "reasoning": "This appears to be a text input field with placeholder text 'Enter your email' and border styling",
    "visual_indicators": ["placeholder_text", "border", "rectangular_shape"],
    "interaction_pattern": "type_text"
  }}
}}

Here are the nodes in batch {batch_index} to analyze:

{json.dumps(nodes_data, indent=2)}

Analyze each node carefully and provide your classification. Consider:
1. Node name and type
2. Visual properties (fills, strokes, corner radius, effects)
3. Text content and styling
4. Context and positioning
5. Parent-child relationships
6. Common UI patterns and conventions

Respond only with the JSON object, no additional text.
"""
        return prompt

    def _parse_enhanced_response(
        self, response: str, nodes_data: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, str]]:
        """Parse the enhanced AI response into structured results"""
        try:
            # Extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]

            parsed_response = json.loads(response)

            # Validate and clean the results
            cleaned_results = {}
            for node_id, result in parsed_response.items():
                if isinstance(result, dict):
                    tag = result.get("tag", "none")
                    confidence = result.get("confidence", 0.0)
                    reasoning = result.get("reasoning", "")
                    visual_indicators = result.get("visual_indicators", [])
                    interaction_pattern = result.get("interaction_pattern")

                    # Ensure tag is valid
                    if tag not in ["link", "button", "input", "select", "none"]:
                        tag = "none"

                    cleaned_results[node_id] = {
                        "tag": tag,
                        "confidence": float(confidence),
                        "reasoning": str(reasoning),
                        "visual_indicators": (
                            list(visual_indicators)
                            if isinstance(visual_indicators, list)
                            else []
                        ),
                        "interaction_pattern": (
                            str(interaction_pattern) if interaction_pattern else None
                        ),
                    }

            return cleaned_results

        except Exception as e:
            logger.error(f"Error parsing enhanced AI response: {e}")
            return {}

    def _is_valid_enhanced_result(self, result: Dict[str, Any]) -> bool:
        """Validate if an enhanced result is properly formatted"""
        if not isinstance(result, dict):
            return False

        required_fields = ["tag", "confidence", "reasoning"]
        for field in required_fields:
            if field not in result:
                return False

        tag = result.get("tag")
        if tag not in ["link", "button", "input", "select", "none"]:
            return False

        confidence = result.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            return False

        return True

    def _count_node_types(self, nodes_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count the types of nodes in the data"""
        type_counts = {}
        for node in nodes_data:
            node_type = node.get("type", "unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts

    def _extract_visual_indicators(self, reasoning: str) -> List[str]:
        """Extract visual indicators from reasoning text"""
        indicators = []
        reasoning_lower = reasoning.lower()

        # Common visual indicators
        visual_keywords = [
            "rounded",
            "corner",
            "border",
            "background",
            "color",
            "shadow",
            "effect",
            "text",
            "font",
            "size",
            "padding",
            "margin",
            "width",
            "height",
            "icon",
            "image",
            "shape",
            "rectangle",
            "circle",
            "ellipse",
        ]

        for keyword in visual_keywords:
            if keyword in reasoning_lower:
                indicators.append(keyword)

        return indicators

    def _determine_interaction_pattern(self, tag: str) -> str:
        """Determine interaction pattern based on tag"""
        patterns = {
            "link": "click_to_navigate",
            "button": "click_to_action",
            "input": "type_text",
            "select": "select_option",
            "none": "no_interaction",
        }
        return patterns.get(tag, "unknown")

    async def analyze_figma_nodes(
        self, file_key: str, node_id: str, max_depth: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main method to analyze Figma nodes using the advanced agentic workflow

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            max_depth: Maximum depth for DFS traversal

        Returns:
            Dictionary containing analysis results and metadata
        """
        logger.info(
            f"Starting advanced agentic analysis for file {file_key}, node {node_id}"
        )

        # Initialize state
        initial_state = AdvancedAgentState(
            file_key=file_key,
            node_id=node_id,
            max_depth=max_depth,
            nodes_data=[],
            analysis_results={},
            current_step="",
            error=None,
            metadata={},
            processing_batches=[],
            batch_results=[],
            retry_count=0,
            max_retries=self.config.max_retries,
        )

        # Run the workflow
        try:
            final_state = await self.workflow.ainvoke(initial_state)

            if final_state.get("error"):
                logger.error(f"Workflow failed: {final_state['error']}")
                return {
                    "success": False,
                    "error": final_state["error"],
                    "results": {},
                    "metadata": final_state.get("metadata", {}),
                }

            return {
                "success": True,
                "results": final_state.get("analysis_results", {}),
                "metadata": final_state.get("metadata", {}),
                "current_step": final_state.get("current_step", ""),
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {"success": False, "error": str(e), "results": {}, "metadata": {}}


async def main():
    """Example usage of the advanced agentic workflow"""
    print("🚀 Advanced Figma Agentic Workflow Example")
    print("=" * 60)

    # Create workflow with custom configuration
    config = WorkflowConfig(
        batch_size=8,
        max_parallel_batches=2,
        max_retries=2,
        timeout_seconds=45,
        confidence_threshold=0.7,
    )

    workflow = FigmaAdvancedAgenticWorkflow(config)

    try:
        print("🔍 Starting advanced agentic analysis...")
        start_time = time.time()

        result = await workflow.analyze_figma_nodes(
            FIGMA_FILE_KEY, START_NODE_ID, max_depth=5
        )

        end_time = time.time()
        analysis_time = end_time - start_time

        print(f"⏱️  Analysis completed in {analysis_time:.2f} seconds")
        print()

        if result["success"]:
            print("✅ Analysis successful!")
            print()

            # Display results
            results = result["results"]
            metadata = result["metadata"]

            print("📊 Analysis Results:")
            print(f"Total nodes analyzed: {metadata.get('total_nodes', 0)}")
            print(f"Batches processed: {metadata.get('batch_count', 0)}")
            print(f"Average confidence: {metadata.get('average_confidence', 0):.2f}")
            print()

            # Show tag counts
            tag_counts = metadata.get("tag_counts", {})
            print("🎯 Element Type Distribution:")
            for tag, count in tag_counts.items():
                if count > 0:
                    print(f"  {tag.upper()}: {count}")
            print()

            # Show confidence statistics
            confidence_stats = metadata.get("confidence_stats", {})
            print("📈 Confidence Statistics:")
            for level, count in confidence_stats.items():
                if count > 0:
                    print(f"  {level.upper()}: {count}")
            print()

            # Show detailed results (first 10)
            print("📋 Detailed Results (first 10):")
            count = 0
            for node_id, node_result in results.items():
                if count >= 10:
                    break
                tag = node_result.get("tag", "none")
                confidence = node_result.get("confidence", 0.0)
                reasoning = node_result.get("reasoning", "")
                indicators = node_result.get("visual_indicators", [])
                pattern = node_result.get("interaction_pattern", "")

                print(f"  Node {node_id}: {tag} (confidence: {confidence:.2f})")
                print(f"    Pattern: {pattern}")
                print(
                    f"    Indicators: {', '.join(indicators) if indicators else 'None'}"
                )
                if reasoning:
                    print(
                        f"    Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}"
                    )
                print()
                count += 1

            # Show performance metrics
            print("⚡ Performance Metrics:")
            print(f"  - Workflow steps: {result.get('current_step', 'unknown')}")
            print(f"  - Total time: {analysis_time:.2f} seconds")
            print(
                f"  - Nodes per second: {metadata.get('total_nodes', 0) / analysis_time:.2f}"
            )
            print(f"  - Batch size: {config.batch_size}")
            print(f"  - Parallel batches: {config.max_parallel_batches}")

        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
