#!/usr/bin/env python3
"""
Agentic Workflow for Figma Node Analysis using LangGraph
Parses Figma nodes and tags them with appropriate element types (link, button, input, select)
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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


# State definition for the agentic workflow
class AgentState(TypedDict):
    """State for the agentic workflow"""

    thread_id: str
    file_key: str
    node_id: str
    max_depth: Optional[int]
    nodes_data: List[Dict[str, Any]]
    analysis_results: Dict[str, Dict[str, str]]
    current_step: str
    error: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class NodeAnalysisResult:
    """Result of node analysis"""

    node_id: str
    tag: str  # "link", "button", "input", "select", or "none"
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]


class FigmaAgenticWorkflow:
    """Agentic workflow for analyzing Figma nodes using LangGraph"""

    def __init__(self):
        """Initialize the agentic workflow"""
        self.figma_dfs = FigmaDFS(FIGMA_ACCESS_TOKEN)
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
            max_output_tokens=8192,
        )
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes to the workflow
        workflow.add_node("extract_nodes", self._extract_nodes)
        workflow.add_node("analyze_nodes", self._analyze_nodes)
        workflow.add_node("validate_results", self._validate_results)
        workflow.add_node("format_output", self._format_output)

        # Define the workflow edges
        workflow.set_entry_point("extract_nodes")
        workflow.add_edge("extract_nodes", "analyze_nodes")
        workflow.add_edge("analyze_nodes", "validate_results")
        workflow.add_edge("validate_results", "format_output")
        workflow.add_edge("format_output", END)

        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "extract_nodes",
            self._should_continue,
            {"continue": "analyze_nodes", "error": END},
        )

        workflow.add_conditional_edges(
            "analyze_nodes",
            self._should_continue,
            {"continue": "validate_results", "error": END},
        )

        workflow.add_conditional_edges(
            "validate_results",
            self._should_continue,
            {"continue": "format_output", "error": END},
        )

        return workflow.compile()

    def _should_continue(self, state: AgentState) -> str:
        """Determine if workflow should continue or end due to error"""
        if state.get("error"):
            return "error"
        return "continue"

    async def _extract_nodes(self, state: AgentState) -> AgentState:
        """Extract nodes from Figma using DFS"""
        try:
            logger.info(
                f"Extracting nodes from Figma file {state['file_key']}, node {state['node_id']}"
            )
            state["current_step"] = "extract_nodes"

            # Get nodes using DFS
            nodes = self.figma_dfs.depth_first_search_from_node_id(
                state["file_key"], state["node_id"], max_depth=state.get("max_depth")
            )

            # Convert nodes to serializable format
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
                }
                nodes_data.append(node_data)

            state["nodes_data"] = nodes_data
            state["metadata"] = {
                "total_nodes": len(nodes_data),
                "extraction_time": time.time(),
            }

            logger.info(f"Extracted {len(nodes_data)} nodes successfully")
            return state

        except Exception as e:
            logger.error(f"Error extracting nodes: {e}")
            state["error"] = str(e)
            return state

    async def _analyze_nodes(self, state: AgentState) -> AgentState:
        """Analyze nodes using Gemini AI to determine element types"""
        try:
            logger.info("Starting node analysis with Gemini AI")
            state["current_step"] = "analyze_nodes"

            if not state.get("nodes_data"):
                raise ValueError("No nodes data available for analysis")

            # Create analysis prompt
            analysis_prompt = self._create_analysis_prompt(state["nodes_data"])

            # Get AI response
            messages = [HumanMessage(content=analysis_prompt)]
            response = await self.llm.ainvoke(messages)

            # Parse the response
            analysis_results = self._parse_analysis_response(
                response.content, state["nodes_data"]
            )

            state["analysis_results"] = analysis_results
            state["metadata"]["analysis_time"] = time.time()

            logger.info(f"Analysis completed for {len(analysis_results)} nodes")
            return state

        except Exception as e:
            logger.error(f"Error analyzing nodes: {e}")
            state["error"] = str(e)
            return state

    async def _validate_results(self, state: AgentState) -> AgentState:
        """Validate and clean up analysis results"""
        try:
            logger.info("Validating analysis results")
            state["current_step"] = "validate_results"

            results = state.get("analysis_results", {})
            validated_results = {}

            # Validate each result
            for node_id, result in results.items():
                if self._is_valid_result(result):
                    validated_results[node_id] = result
                else:
                    logger.warning(f"Invalid result for node {node_id}: {result}")
                    # Set default tag for invalid results
                    validated_results[node_id] = {
                        "tag": "none",
                        "confidence": 0.0,
                        "reasoning": "Invalid or unclear result",
                    }

            state["analysis_results"] = validated_results
            state["metadata"]["validation_time"] = time.time()

            logger.info(f"Validation completed. {len(validated_results)} valid results")
            return state

        except Exception as e:
            logger.error(f"Error validating results: {e}")
            state["error"] = str(e)
            return state

    async def _format_output(self, state: AgentState) -> AgentState:
        """Format the final output"""
        try:
            logger.info("Formatting final output")
            state["current_step"] = "format_output"

            # Add summary statistics
            results = state.get("analysis_results", {})
            tag_counts = {"link": 0, "button": 0, "input": 0, "select": 0, "none": 0}

            for result in results.values():
                tag = result.get("tag", "none")
                if tag in tag_counts:
                    tag_counts[tag] += 1

            state["metadata"]["tag_counts"] = tag_counts
            state["metadata"]["completion_time"] = time.time()

            logger.info(f"Output formatting completed. Tag counts: {tag_counts}")
            return state

        except Exception as e:
            logger.error(f"Error formatting output: {e}")
            state["error"] = str(e)
            return state

    def _create_analysis_prompt(self, nodes_data: List[Dict[str, Any]]) -> str:
        """Create the analysis prompt for Gemini AI"""
        prompt = f"""
You are an expert UI/UX analyst specializing in Figma design analysis. Your task is to analyze a list of Figma nodes and determine what type of interactive element each node represents.

For each node, you must classify it as one of the following types:
- "link": Clickable links, navigation elements, or text that appears to be a hyperlink
- "button": Clickable buttons, CTAs, or interactive elements that trigger actions
- "input": Form input fields, text boxes, search bars, or data entry elements
- "select": Dropdown menus, select boxes, or choice selection elements
- "none": Non-interactive elements like text, images, or decorative elements

IMPORTANT: You must respond with a valid JSON object where:
- Keys are the node IDs
- Values are objects with "tag", "confidence" (0.0-1.0), and "reasoning" fields

Example response format:
{{
  "node_id_1": {{
    "tag": "button",
    "confidence": 0.95,
    "reasoning": "This node has button-like styling and contains action text"
  }},
  "node_id_2": {{
    "tag": "input",
    "confidence": 0.88,
    "reasoning": "This appears to be a text input field with placeholder text"
  }}
}}

Here are the nodes to analyze:

{json.dumps(nodes_data, indent=2)}

Analyze each node carefully and provide your classification. Consider:
1. Node name and type
2. Visual properties and styling
3. Context and positioning
4. Text content if present
5. Parent-child relationships

Respond only with the JSON object, no additional text.
"""
        return prompt

    def _parse_analysis_response(
        self, response: str, nodes_data: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, str]]:
        """Parse the AI response into structured results"""
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

                    # Ensure tag is valid
                    if tag not in ["link", "button", "input", "select", "none"]:
                        tag = "none"

                    cleaned_results[node_id] = {
                        "tag": tag,
                        "confidence": float(confidence),
                        "reasoning": str(reasoning),
                    }

            return cleaned_results

        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            # Return empty results if parsing fails
            return {}

    def _is_valid_result(self, result: Dict[str, Any]) -> bool:
        """Validate if a result is properly formatted"""
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

    async def analyze_figma_nodes(
        self, file_key: str, node_id: str, max_depth: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main method to analyze Figma nodes using the agentic workflow

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            max_depth: Maximum depth for DFS traversal

        Returns:
            Dictionary containing analysis results and metadata
        """
        logger.info(f"Starting agentic analysis for file {file_key}, node {node_id}")

        # Initialize state
        initial_state = AgentState(
            thread_id=str(uuid.uuid4()),
            file_key=file_key,
            node_id=node_id,
            max_depth=max_depth,
            nodes_data=[],
            analysis_results={},
            current_step="",
            error=None,
            metadata={},
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
    """Example usage of the agentic workflow"""
    print("🚀 Figma Agentic Workflow Example")
    print("=" * 50)

    workflow = FigmaAgenticWorkflow()

    try:
        print("🔍 Starting agentic analysis...")
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
            print()

            # Show tag counts
            tag_counts = metadata.get("tag_counts", {})
            print("🎯 Element Type Distribution:")
            for tag, count in tag_counts.items():
                if count > 0:
                    print(f"  {tag.upper()}: {count}")
            print()

            # Show detailed results
            print("📋 Detailed Results:")
            for node_id, node_result in results.items():
                tag = node_result.get("tag", "none")
                confidence = node_result.get("confidence", 0.0)
                reasoning = node_result.get("reasoning", "")
                print(f"  Node {node_id}: {tag} (confidence: {confidence:.2f})")
                if reasoning:
                    print(f"    Reasoning: {reasoning}")
            print()

            # Show performance metrics
            print("⚡ Performance Metrics:")
            print(f"  - Workflow steps: {result.get('current_step', 'unknown')}")
            print(f"  - Total time: {analysis_time:.2f} seconds")
            print(
                f"  - Nodes per second: {metadata.get('total_nodes', 0) / analysis_time:.2f}"
            )

        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
