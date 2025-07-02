#!/usr/bin/env python3
"""
Figma Agentic Workflow using LangGraph - Allows Gemini to call Figma APIs as tools
"""

import json
import logging
import time
import requests
from typing import Dict, List, Any, Optional, Annotated
from datetime import datetime
import os

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from figma_dfs import FigmaDFS
from input_detection_prompts import AgenticDetectionPromptCreator
from config import (
    FIGMA_ACCESS_TOKEN,
    FIGMA_FILE_KEY,
    START_NODE_ID,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
)

# Set up logging
logger = logging.getLogger(__name__)

# Create a custom file handler for Gemini interactions
gemini_logger = logging.getLogger("gemini_agentic")
gemini_logger.setLevel(logging.INFO)

# Remove any existing handlers to avoid duplicates
for handler in gemini_logger.handlers[:]:
    gemini_logger.removeHandler(handler)

# Create file handler with timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"gemini_agentic_{timestamp}.txt"
file_handler = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)

# Add handler to logger (only file handler, no console)
gemini_logger.addHandler(file_handler)

# Disable propagation to root logger to prevent console output
gemini_logger.propagate = False


def log_gemini_interaction(
    interaction_type: str, content: str, metadata: Dict[str, Any] = None
):
    """
    Log Gemini interactions to gemini_agentic.txt

    Args:
        interaction_type: Type of interaction (QUERY, RESPONSE, TOOL_CALL, TOOL_RESPONSE)
        content: The content to log
        metadata: Additional metadata about the interaction
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"\n{'='*80}\n"
    log_entry += f"GEMINI INTERACTION - {interaction_type}\n"
    log_entry += f"Timestamp: {timestamp}\n"

    if metadata:
        log_entry += f"Metadata: {json.dumps(metadata, indent=2)}\n"

    log_entry += f"{'='*80}\n"
    log_entry += f"{content}\n"
    log_entry += f"{'='*80}\n"

    gemini_logger.info(log_entry)


# Define the state for our graph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    nodes_data: List[Dict[str, Any]]
    analysis_results: Dict[str, Dict[str, str]]


@tool
def get_multiple_figma_nodes(file_key: str, node_ids: List[str]) -> Dict[str, Any]:
    """
    Get multiple nodes data from Figma in a single call, filtered for UI-relevant fields

    Args:
        file_key: The Figma file key
        node_ids: List of node IDs to fetch

    Returns:
        Filtered data with only UI-relevant fields for all requested nodes
    """
    # Log the tool call
    log_gemini_interaction(
        "TOOL_CALL",
        f"Tool: get_multiple_figma_nodes\nArguments: file_key={file_key}, node_ids={node_ids}",
        {
            "tool_name": "get_multiple_figma_nodes",
            "file_key": file_key,
            "node_ids_count": len(node_ids),
            "node_ids": node_ids,
        },
    )

    try:
        # Figma API allows multiple node IDs separated by commas
        ids_param = ",".join(node_ids)
        url = f"https://api.figma.com/v1/files/{file_key}/nodes?ids={ids_param}"
        headers = {
            "X-Figma-Token": FIGMA_ACCESS_TOKEN,
            "Content-Type": "application/json",
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        raw_data = response.json()

        # Filter and optimize the data for UI analysis
        filtered_data = {}

        if "nodes" in raw_data:
            for node_id, node_data in raw_data["nodes"].items():
                if "document" in node_data:
                    doc = node_data["document"]
                    filtered_node = _filter_node_for_ui_analysis(doc)
                    if filtered_node:
                        filtered_data[node_id] = filtered_node

        logger.info(f"Successfully fetched and filtered {len(filtered_data)} nodes")
        return {"nodes": filtered_data}
    except Exception as e:
        logger.error(f"Error fetching multiple nodes: {e}")
        return {"error": str(e)}


def _filter_node_for_ui_analysis(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter a Figma node to only include UI-relevant fields for analysis

    Args:
        node: Raw Figma node data

    Returns:
        Filtered node with only UI-relevant fields in token-efficient format
    """
    # Essential fields for UI analysis
    filtered = {
        "id": node.get("id"),
        "name": node.get("name"),
        "type": node.get("type"),
    }

    # Bounding box for positioning and sizing
    if "absoluteBoundingBox" in node:
        bbox = node["absoluteBoundingBox"]
        filtered["pos"] = [bbox.get("x", 0), bbox.get("y", 0)]
        filtered["size"] = [bbox.get("width", 0), bbox.get("height", 0)]

    # Text content for inputs, buttons, links
    if "characters" in node:
        text = node.get("characters", "").strip()
        if text:  # Only include non-empty text
            filtered["text"] = text

    # Text styling for better UI analysis
    if "style" in node:
        style = node["style"]
        if "fontSize" in style:
            filtered["fontSize"] = style.get("fontSize")
        if "fontWeight" in style:
            filtered["fontWeight"] = style.get("fontWeight")
        if "textAlignHorizontal" in style:
            filtered["textAlign"] = style.get("textAlignHorizontal")
        if "textAlignVertical" in style:
            filtered["textAlignV"] = style.get("textAlignVertical")
        if "letterSpacing" in style:
            filtered["letterSpacing"] = style.get("letterSpacing")
        if "lineHeightPx" in style:
            filtered["lineHeight"] = style.get("lineHeightPx")
        if "textDecoration" in style:
            filtered["textDecoration"] = style.get("textDecoration")

    # Corner radius for input/button styling
    if "cornerRadius" in node and node.get("cornerRadius", 0) > 0:
        filtered["radius"] = node.get("cornerRadius")

    # Individual corner radius values
    if "rectangleCornerRadii" in node:
        radii = node["rectangleCornerRadii"]
        if any(r > 0 for r in radii):
            filtered["cornerRadii"] = radii

    # Fill colors for styling analysis (only if visible)
    if "fills" in node and node["fills"]:
        fills = node["fills"]
        if fills and len(fills) > 0:
            fill = fills[0]
            if "color" in fill and fill.get("visible", True):
                color = fill["color"]
                filtered["fill"] = [
                    color.get("r", 0),
                    color.get("g", 0),
                    color.get("b", 0),
                ]
                # Include opacity if not 1.0
                if "opacity" in fill and fill.get("opacity", 1.0) != 1.0:
                    filtered["fillOpacity"] = fill.get("opacity")
                # Include blend mode if not normal
                if "blendMode" in fill and fill.get("blendMode") != "NORMAL":
                    filtered["fillBlendMode"] = fill.get("blendMode")

    # Stroke info for underlining (links) and borders
    if "strokes" in node and node["strokes"]:
        strokes = node["strokes"]
        if strokes and len(strokes) > 0 and strokes[0].get("visible", True):
            stroke = strokes[0]
            filtered["stroke"] = True
            # Include stroke color
            if "color" in stroke:
                color = stroke["color"]
                filtered["strokeColor"] = [
                    color.get("r", 0),
                    color.get("g", 0),
                    color.get("b", 0),
                ]
            # Include stroke weight
            if "strokeWeight" in stroke:
                filtered["strokeWeight"] = stroke.get("strokeWeight")
            # Include stroke opacity
            if "opacity" in stroke and stroke.get("opacity", 1.0) != 1.0:
                filtered["strokeOpacity"] = stroke.get("opacity")

    # Stroke alignment
    if "strokeAlign" in node:
        filtered["strokeAlign"] = node.get("strokeAlign")

    # Effects (shadows, blurs)
    if "effects" in node and node["effects"]:
        effects = node["effects"]
        visible_effects = [e for e in effects if e.get("visible", True)]
        if visible_effects:
            filtered["effects"] = []
            for effect in visible_effects:
                effect_info = {
                    "type": effect.get("type"),
                    "visible": effect.get("visible", True),
                }
                if "color" in effect:
                    color = effect["color"]
                    effect_info["color"] = [
                        color.get("r", 0),
                        color.get("g", 0),
                        color.get("b", 0),
                    ]
                if "opacity" in effect:
                    effect_info["opacity"] = effect.get("opacity")
                if "radius" in effect:
                    effect_info["radius"] = effect.get("radius")
                if "offset" in effect:
                    offset = effect["offset"]
                    effect_info["offset"] = [offset.get("x", 0), offset.get("y", 0)]
                if "spread" in effect:
                    effect_info["spread"] = effect.get("spread")
                filtered["effects"].append(effect_info)

    # Layout constraints for responsive design
    if "constraints" in node:
        constraints = node["constraints"]
        filtered["constraints"] = {
            "horizontal": constraints.get("horizontal"),
            "vertical": constraints.get("vertical"),
        }

    # Layout mode for auto-layout frames
    if "layoutMode" in node:
        filtered["layoutMode"] = node.get("layoutMode")
        # Include layout-specific properties
        if "primaryAxisSizingMode" in node:
            filtered["primaryAxisSizing"] = node.get("primaryAxisSizingMode")
        if "counterAxisSizingMode" in node:
            filtered["counterAxisSizing"] = node.get("counterAxisSizingMode")
        if "primaryAxisAlignItems" in node:
            filtered["primaryAxisAlign"] = node.get("primaryAxisAlignItems")
        if "counterAxisAlignItems" in node:
            filtered["counterAxisAlign"] = node.get("counterAxisAlignItems")
        if "paddingLeft" in node:
            filtered["paddingLeft"] = node.get("paddingLeft")
        if "paddingRight" in node:
            filtered["paddingRight"] = node.get("paddingRight")
        if "paddingTop" in node:
            filtered["paddingTop"] = node.get("paddingTop")
        if "paddingBottom" in node:
            filtered["paddingBottom"] = node.get("paddingBottom")
        if "itemSpacing" in node:
            filtered["itemSpacing"] = node.get("itemSpacing")

    # Opacity for transparency
    if "opacity" in node and node.get("opacity", 1.0) != 1.0:
        filtered["opacity"] = node.get("opacity")

    # Blend mode for layering effects
    if "blendMode" in node and node.get("blendMode") != "NORMAL":
        filtered["blendMode"] = node.get("blendMode")

    # Clips content (important for input fields)
    if "clipsContent" in node:
        filtered["clipsContent"] = node.get("clipsContent")

    # Background blur
    if "backgroundBlur" in node and node.get("backgroundBlur", 0) > 0:
        filtered["backgroundBlur"] = node.get("backgroundBlur")

    # Component properties for instances
    if "componentProperties" in node:
        filtered["componentProperties"] = node.get("componentProperties")

    # Component ID for instances
    if "componentId" in node:
        filtered["componentId"] = node.get("componentId")

    # Variant properties for component sets
    if "variantProperties" in node:
        filtered["variantProperties"] = node.get("variantProperties")

    # Prototype interactions
    if "prototypeInteractions" in node and node["prototypeInteractions"]:
        filtered["prototypeInteractions"] = node.get("prototypeInteractions")

    # Prototype starting point
    if "prototypeStartNodeID" in node:
        filtered["prototypeStartNodeID"] = node.get("prototypeStartNodeID")

    # Export settings
    if "exportSettings" in node and node["exportSettings"]:
        filtered["exportSettings"] = node.get("exportSettings")

    # Children for hierarchical analysis (only include relevant children)
    if "children" in node and node["children"]:
        relevant_children = []
        for child in node["children"]:
            # Skip invisible children
            if child.get("visible", True):
                filtered_child = _filter_node_for_ui_analysis(child)
                if filtered_child:  # Only include non-empty filtered children
                    relevant_children.append(filtered_child)

        if relevant_children:
            filtered["children"] = relevant_children

    return filtered


# Define the tools available to the agent
tools = [get_multiple_figma_nodes]


def create_analysis_prompt(
    nodes_data: List[Dict[str, Any]], file_key: str, start_node_id: str
) -> str:
    """Create the prompt for Gemini analysis with available tools"""

    # Use the AgenticDetectionPromptCreator to generate the prompt
    prompt_creator = AgenticDetectionPromptCreator()
    return prompt_creator.create_prompt(nodes_data, file_key, start_node_id)


def analyzer_node(state: State) -> State:
    """Node that performs the Gemini analysis with tool access"""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
    )
    llm_with_tools = llm.bind_tools(tools)
    if not state["messages"]:
        prompt = create_analysis_prompt(
            state["nodes_data"], FIGMA_FILE_KEY, START_NODE_ID
        )
        messages = [
            SystemMessage(
                content="You are an expert UI analyst with access to Figma API tools."
            ),
            HumanMessage(content=prompt),
        ]
        query_content = f"System Message: {messages[0].content}\n\nHuman Message: {messages[1].content}"
        log_gemini_interaction(
            "QUERY",
            query_content,
            {
                "model": GEMINI_MODEL_NAME,
                "temperature": 0.1,
                "nodes_count": len(state["nodes_data"]),
                "file_key": FIGMA_FILE_KEY,
                "start_node_id": START_NODE_ID,
            },
        )
        state["messages"].extend(messages)
    response = llm_with_tools.invoke(state["messages"])
    state["messages"].append(response)
    response_content = (
        response.content if hasattr(response, "content") else str(response)
    )
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    log_gemini_interaction(
        "RESPONSE",
        response_content,
        {
            "has_tool_calls": len(tool_calls) > 0,
            "tool_calls_count": len(tool_calls),
            "tool_calls": (
                [{"name": tc.get("name"), "args": tc.get("args")} for tc in tool_calls]
                if tool_calls
                else []
            ),
        },
    )
    return state


def tool_node_wrapper(state: State) -> State:
    """Wrapper for the tool node to handle tool calls"""
    tool_node = ToolNode(tools)
    result = tool_node.invoke(state)

    # If we have tool results, convert them to optimized format and update nodes_data
    if "messages" in result and result["messages"]:
        last_message = result["messages"][-1]
        if hasattr(last_message, "content") and isinstance(last_message.content, dict):
            # Check if this is a response from get_multiple_figma_nodes
            if "nodes" in last_message.content:
                # Log the tool response
                log_gemini_interaction(
                    "TOOL_RESPONSE",
                    f"Tool: get_multiple_figma_nodes\nResponse: {json.dumps(last_message.content, indent=2)}",
                    {
                        "tool_name": "get_multiple_figma_nodes",
                        "nodes_returned": len(last_message.content.get("nodes", {})),
                        "has_error": "error" in last_message.content,
                    },
                )

                # Convert API response to optimized format
                workflow_instance = FigmaAgenticWorkflow()
                optimized_nodes = (
                    workflow_instance._convert_api_response_to_optimized_format(
                        last_message.content
                    )
                )

                # Update the nodes_data in the state with the new optimized data
                if optimized_nodes:
                    result["nodes_data"] = optimized_nodes
                    logger.info(
                        f"Updated nodes_data with {len(optimized_nodes)} optimized nodes from API"
                    )

    return result


def should_continue(state: State) -> str:
    """Determine if we should continue with tool calls or finish"""
    last_message = state["messages"][-1]

    # If the last message has tool calls, we should run tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, we're done
    return "end"


def extract_results(state: State) -> State:
    """Extract the final analysis results from the conversation"""

    # Look for JSON in the last few messages
    results = {}

    for message in reversed(state["messages"][-5:]):  # Check last 5 messages
        if hasattr(message, "content") and message.content:
            content = message.content

            # Try to find JSON in the content
            try:
                # Look for JSON block
                start = content.find("{")
                end = content.rfind("}") + 1

                if start != -1 and end > start:
                    json_str = content[start:end]
                    parsed_results = json.loads(json_str)

                    # Validate the structure and extract clean node IDs
                    if isinstance(parsed_results, dict):
                        for node_id, node_info in parsed_results.items():
                            if isinstance(node_info, dict) and "tag" in node_info:
                                # Extract clean node ID from verbose format (e.g., "5:158|INSTANCE|Button Secondary" -> "5:158")
                                clean_node_id = (
                                    node_id.split("|")[0] if "|" in node_id else node_id
                                )
                                results[clean_node_id] = {"tag": node_info["tag"]}

                    if results:
                        break

            except (json.JSONDecodeError, ValueError):
                continue

    state["analysis_results"] = results

    # Log the final results
    log_gemini_interaction(
        "FINAL_RESULTS",
        f"Analysis completed\nResults: {json.dumps(results, indent=2)}",
        {
            "results_count": len(results),
            "detected_elements": list(results.keys()),
            "element_types": list(
                set(info.get("tag", "unknown") for info in results.values())
            ),
        },
    )

    return state


class FigmaAgenticWorkflow:
    """Main class for the Figma Agentic Workflow"""

    def __init__(self):
        """Initialize the workflow"""
        self.figma_dfs = FigmaDFS(FIGMA_ACCESS_TOKEN)
        self.workflow = None
        self._setup_workflow()

    def _setup_workflow(self):
        """Set up the LangGraph workflow"""
        # Create the graph
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("analyzer", analyzer_node)
        workflow.add_node("tools", tool_node_wrapper)
        workflow.add_node("extract", extract_results)

        # Set entry point
        workflow.set_entry_point("analyzer")

        # Add conditional edges
        workflow.add_conditional_edges(
            "analyzer", should_continue, {"tools": "tools", "end": "extract"}
        )

        # Add edge from tools back to analyzer
        workflow.add_edge("tools", "analyzer")

        # Add final edge
        workflow.add_edge("extract", END)

        # Compile the graph
        self.workflow = workflow.compile()

    def _prepare_nodes_data(self, all_nodes: List[Any]) -> List[Dict[str, Any]]:
        """Convert FigmaNode objects to optimized format for analysis"""
        nodes_data = []

        for node in all_nodes:
            try:
                # Extract base node ID
                base_id = self.figma_dfs._extract_base_node_id(node.id)

                # Get bounding box data
                absolute_bounding_box = node.data.get("absoluteBoundingBox", {})
                x = absolute_bounding_box.get("x", 0)
                y = absolute_bounding_box.get("y", 0)
                width = absolute_bounding_box.get("width", 0)
                height = absolute_bounding_box.get("height", 0)

                # Get children info
                children = node.children if node.children is not None else []
                child_names = [child.name for child in children]
                child_types = [child.type for child in children]

                # Create optimized node dict with minimal token usage
                node_dict = {
                    "id": base_id,
                    "t": node.type,  # type
                    "n": node.name,  # name
                    "p": [x, y],  # position [x, y]
                    "s": [width, height],  # size [width, height]
                    "c": child_names,  # child names
                    "ct": child_types,  # child types
                }

                # Add optional fields only if they exist and are relevant
                if "characters" in node.data:
                    text = node.data.get("characters", "").strip()
                    if text:  # Only include non-empty text
                        node_dict["tx"] = text

                # Text styling
                if "style" in node.data:
                    style = node.data["style"]
                    if "fontSize" in style:
                        node_dict["fs"] = style.get("fontSize")  # fontSize
                    if "fontWeight" in style:
                        node_dict["fw"] = style.get("fontWeight")  # fontWeight
                    if "textAlignHorizontal" in style:
                        node_dict["ta"] = style.get("textAlignHorizontal")  # textAlign
                    if "textAlignVertical" in style:
                        node_dict["tav"] = style.get(
                            "textAlignVertical"
                        )  # textAlignVertical
                    if "letterSpacing" in style:
                        node_dict["ls"] = style.get("letterSpacing")  # letterSpacing
                    if "lineHeightPx" in style:
                        node_dict["lh"] = style.get("lineHeightPx")  # lineHeight
                    if "textDecoration" in style:
                        node_dict["td"] = style.get("textDecoration")  # textDecoration

                if "cornerRadius" in node.data and node.data.get("cornerRadius", 0) > 0:
                    node_dict["r"] = node.data.get("cornerRadius")

                # Individual corner radius values
                if "rectangleCornerRadii" in node.data:
                    radii = node.data["rectangleCornerRadii"]
                    if any(r > 0 for r in radii):
                        node_dict["cr"] = radii  # cornerRadii

                # Add fill color info (only if visible)
                if "fills" in node.data and node.data["fills"]:
                    fills = node.data["fills"]
                    if fills and len(fills) > 0:
                        fill = fills[0]
                        if "color" in fill and fill.get("visible", True):
                            color = fill["color"]
                            node_dict["f"] = [
                                color.get("r", 0),
                                color.get("g", 0),
                                color.get("b", 0),
                            ]
                            # Include fill opacity if not 1.0
                            if "opacity" in fill and fill.get("opacity", 1.0) != 1.0:
                                node_dict["fo"] = fill.get("opacity")  # fillOpacity
                            # Include fill blend mode if not normal
                            if (
                                "blendMode" in fill
                                and fill.get("blendMode") != "NORMAL"
                            ):
                                node_dict["fbm"] = fill.get(
                                    "blendMode"
                                )  # fillBlendMode

                # Add stroke info (only if visible)
                if "strokes" in node.data and node.data["strokes"]:
                    strokes = node.data["strokes"]
                    if strokes and len(strokes) > 0 and strokes[0].get("visible", True):
                        stroke = strokes[0]
                        node_dict["st"] = True
                        # Include stroke color
                        if "color" in stroke:
                            color = stroke["color"]
                            node_dict["sc"] = [
                                color.get("r", 0),
                                color.get("g", 0),
                                color.get("b", 0),
                            ]  # strokeColor
                        # Include stroke weight
                        if "strokeWeight" in stroke:
                            node_dict["sw"] = stroke.get("strokeWeight")  # strokeWeight
                        # Include stroke opacity
                        if "opacity" in stroke and stroke.get("opacity", 1.0) != 1.0:
                            node_dict["so"] = stroke.get("opacity")  # strokeOpacity

                # Stroke alignment
                if "strokeAlign" in node.data:
                    node_dict["sa"] = node.data.get("strokeAlign")  # strokeAlign

                # Effects (shadows, blurs)
                if "effects" in node.data and node.data["effects"]:
                    effects = node.data["effects"]
                    visible_effects = [e for e in effects if e.get("visible", True)]
                    if visible_effects:
                        node_dict["ef"] = []  # effects
                        for effect in visible_effects:
                            effect_info = {
                                "t": effect.get("type"),  # type
                                "v": effect.get("visible", True),  # visible
                            }
                            if "color" in effect:
                                color = effect["color"]
                                effect_info["c"] = [
                                    color.get("r", 0),
                                    color.get("g", 0),
                                    color.get("b", 0),
                                ]  # color
                            if "opacity" in effect:
                                effect_info["o"] = effect.get("opacity")  # opacity
                            if "radius" in effect:
                                effect_info["r"] = effect.get("radius")  # radius
                            if "offset" in effect:
                                offset = effect["offset"]
                                effect_info["of"] = [
                                    offset.get("x", 0),
                                    offset.get("y", 0),
                                ]  # offset
                            if "spread" in effect:
                                effect_info["s"] = effect.get("spread")  # spread
                            node_dict["ef"].append(effect_info)

                # Layout constraints
                if "constraints" in node.data:
                    constraints = node.data["constraints"]
                    node_dict["co"] = {  # constraints
                        "h": constraints.get("horizontal"),  # horizontal
                        "v": constraints.get("vertical"),  # vertical
                    }

                # Layout mode for auto-layout frames
                if "layoutMode" in node.data:
                    node_dict["lm"] = node.data.get("layoutMode")  # layoutMode
                    # Include layout-specific properties
                    if "primaryAxisSizingMode" in node.data:
                        node_dict["pas"] = node.data.get(
                            "primaryAxisSizingMode"
                        )  # primaryAxisSizing
                    if "counterAxisSizingMode" in node.data:
                        node_dict["cas"] = node.data.get(
                            "counterAxisSizingMode"
                        )  # counterAxisSizing
                    if "primaryAxisAlignItems" in node.data:
                        node_dict["paa"] = node.data.get(
                            "primaryAxisAlignItems"
                        )  # primaryAxisAlign
                    if "counterAxisAlignItems" in node.data:
                        node_dict["caa"] = node.data.get(
                            "counterAxisAlignItems"
                        )  # counterAxisAlign
                    if "paddingLeft" in node.data:
                        node_dict["pl"] = node.data.get("paddingLeft")  # paddingLeft
                    if "paddingRight" in node.data:
                        node_dict["pr"] = node.data.get("paddingRight")  # paddingRight
                    if "paddingTop" in node.data:
                        node_dict["pt"] = node.data.get("paddingTop")  # paddingTop
                    if "paddingBottom" in node.data:
                        node_dict["pb"] = node.data.get(
                            "paddingBottom"
                        )  # paddingBottom
                    if "itemSpacing" in node.data:
                        node_dict["is"] = node.data.get("itemSpacing")  # itemSpacing

                # Opacity for transparency
                if "opacity" in node.data and node.data.get("opacity", 1.0) != 1.0:
                    node_dict["op"] = node.data.get("opacity")  # opacity

                # Blend mode for layering effects
                if "blendMode" in node.data and node.data.get("blendMode") != "NORMAL":
                    node_dict["bm"] = node.data.get("blendMode")  # blendMode

                # Clips content (important for input fields)
                if "clipsContent" in node.data:
                    node_dict["cc"] = node.data.get("clipsContent")  # clipsContent

                # Background blur
                if (
                    "backgroundBlur" in node.data
                    and node.data.get("backgroundBlur", 0) > 0
                ):
                    node_dict["bb"] = node.data.get("backgroundBlur")  # backgroundBlur

                # Component properties for instances
                if "componentProperties" in node.data:
                    node_dict["cp"] = node.data.get(
                        "componentProperties"
                    )  # componentProperties

                # Component ID for instances
                if "componentId" in node.data:
                    node_dict["ci"] = node.data.get("componentId")  # componentId

                # Variant properties for component sets
                if "variantProperties" in node.data:
                    node_dict["vp"] = node.data.get(
                        "variantProperties"
                    )  # variantProperties

                # Prototype interactions
                if (
                    "prototypeInteractions" in node.data
                    and node.data["prototypeInteractions"]
                ):
                    node_dict["pi"] = node.data.get(
                        "prototypeInteractions"
                    )  # prototypeInteractions

                # Prototype starting point
                if "prototypeStartNodeID" in node.data:
                    node_dict["ps"] = node.data.get(
                        "prototypeStartNodeID"
                    )  # prototypeStartNodeID

                # Export settings
                if "exportSettings" in node.data and node.data["exportSettings"]:
                    node_dict["es"] = node.data.get("exportSettings")  # exportSettings

                nodes_data.append(node_dict)

            except Exception as e:
                logger.error(f"Error processing node {node.id}: {e}")
                continue

        return nodes_data

    def _prepare_filtered_nodes_data(
        self, filtered_nodes: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert filtered API nodes to optimized format for analysis"""
        nodes_data = []

        for node_id, node in filtered_nodes.items():
            try:
                # Extract base node ID
                base_id = self.figma_dfs._extract_base_node_id(node.get("id", node_id))

                # Get position and size
                pos = node.get("pos", [0, 0])
                size = node.get("size", [0, 0])

                # Get children info
                children = node.get("children", [])
                child_names = [child.get("name", "") for child in children]
                child_types = [child.get("type", "") for child in children]

                # Create optimized node dict with minimal token usage
                node_dict = {
                    "id": base_id,
                    "t": node.get("type", ""),  # type
                    "n": node.get("name", ""),  # name
                    "p": pos,  # position [x, y]
                    "s": size,  # size [width, height]
                    "c": child_names,  # child names
                    "ct": child_types,  # child types
                }

                # Add optional fields only if they exist
                if "text" in node:
                    node_dict["tx"] = node["text"]  # text content

                # Text styling fields
                if "fontSize" in node:
                    node_dict["fs"] = node["fontSize"]  # fontSize
                if "fontWeight" in node:
                    node_dict["fw"] = node["fontWeight"]  # fontWeight
                if "textAlign" in node:
                    node_dict["ta"] = node["textAlign"]  # textAlign
                if "textAlignV" in node:
                    node_dict["tav"] = node["textAlignV"]  # textAlignVertical
                if "letterSpacing" in node:
                    node_dict["ls"] = node["letterSpacing"]  # letterSpacing
                if "lineHeight" in node:
                    node_dict["lh"] = node["lineHeight"]  # lineHeight
                if "textDecoration" in node:
                    node_dict["td"] = node["textDecoration"]  # textDecoration

                if "radius" in node:
                    node_dict["r"] = node["radius"]  # corner radius

                if "cornerRadii" in node:
                    node_dict["cr"] = node["cornerRadii"]  # cornerRadii

                if "fill" in node:
                    node_dict["f"] = node["fill"]  # fill color

                if "fillOpacity" in node:
                    node_dict["fo"] = node["fillOpacity"]  # fillOpacity

                if "fillBlendMode" in node:
                    node_dict["fbm"] = node["fillBlendMode"]  # fillBlendMode

                if "stroke" in node:
                    node_dict["st"] = True  # stroke

                if "strokeColor" in node:
                    node_dict["sc"] = node["strokeColor"]  # strokeColor

                if "strokeWeight" in node:
                    node_dict["sw"] = node["strokeWeight"]  # strokeWeight

                if "strokeOpacity" in node:
                    node_dict["so"] = node["strokeOpacity"]  # strokeOpacity

                if "strokeAlign" in node:
                    node_dict["sa"] = node["strokeAlign"]  # strokeAlign

                if "effects" in node:
                    node_dict["ef"] = node["effects"]  # effects

                if "constraints" in node:
                    node_dict["co"] = node["constraints"]  # constraints

                if "layoutMode" in node:
                    node_dict["lm"] = node["layoutMode"]  # layoutMode

                if "primaryAxisSizing" in node:
                    node_dict["pas"] = node["primaryAxisSizing"]  # primaryAxisSizing

                if "counterAxisSizing" in node:
                    node_dict["cas"] = node["counterAxisSizing"]  # counterAxisSizing

                if "primaryAxisAlign" in node:
                    node_dict["paa"] = node["primaryAxisAlign"]  # primaryAxisAlign

                if "counterAxisAlign" in node:
                    node_dict["caa"] = node["counterAxisAlign"]  # counterAxisAlign

                if "paddingLeft" in node:
                    node_dict["pl"] = node["paddingLeft"]  # paddingLeft

                if "paddingRight" in node:
                    node_dict["pr"] = node["paddingRight"]  # paddingRight

                if "paddingTop" in node:
                    node_dict["pt"] = node["paddingTop"]  # paddingTop

                if "paddingBottom" in node:
                    node_dict["pb"] = node["paddingBottom"]  # paddingBottom

                if "itemSpacing" in node:
                    node_dict["is"] = node["itemSpacing"]  # itemSpacing

                if "opacity" in node:
                    node_dict["op"] = node["opacity"]  # opacity

                if "blendMode" in node:
                    node_dict["bm"] = node["blendMode"]  # blendMode

                if "clipsContent" in node:
                    node_dict["cc"] = node["clipsContent"]  # clipsContent

                if "backgroundBlur" in node:
                    node_dict["bb"] = node["backgroundBlur"]  # backgroundBlur

                if "componentProperties" in node:
                    node_dict["cp"] = node["componentProperties"]  # componentProperties

                if "componentId" in node:
                    node_dict["ci"] = node["componentId"]  # componentId

                if "variantProperties" in node:
                    node_dict["vp"] = node["variantProperties"]  # variantProperties

                if "prototypeInteractions" in node:
                    node_dict["pi"] = node[
                        "prototypeInteractions"
                    ]  # prototypeInteractions

                if "prototypeStartNodeID" in node:
                    node_dict["ps"] = node[
                        "prototypeStartNodeID"
                    ]  # prototypeStartNodeID

                if "exportSettings" in node:
                    node_dict["es"] = node["exportSettings"]  # exportSettings

                nodes_data.append(node_dict)

            except Exception as e:
                logger.error(f"Error processing filtered node {node_id}: {e}")
                continue

        return nodes_data

    def _convert_api_response_to_optimized_format(
        self, api_response: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Convert API response from get_multiple_figma_nodes to optimized format

        Args:
            api_response: Response from get_multiple_figma_nodes tool

        Returns:
            List of nodes in optimized format for analysis
        """
        if "error" in api_response:
            logger.error(f"API response contains error: {api_response['error']}")
            return []

        if "nodes" not in api_response:
            logger.warning("API response does not contain 'nodes' key")
            return []

        filtered_nodes = api_response["nodes"]
        return self._prepare_filtered_nodes_data(filtered_nodes)

    async def analyze(
        self,
        file_key: str,
        node_id: str,
        max_depth: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, str]]:
        """
        Run the agentic analysis workflow

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            max_depth: Maximum depth for DFS
            verbose: Whether to print progress information

        Returns:
            Dictionary with node_id as keys and {"tag": "type"} as values
        """

        # Log the start of analysis
        log_gemini_interaction(
            "ANALYSIS_START",
            f"Starting Figma Agentic Analysis\nFile: {file_key}\nNode: {node_id}\nMax Depth: {max_depth}",
            {
                "file_key": file_key,
                "node_id": node_id,
                "max_depth": max_depth,
                "verbose": verbose,
            },
        )

        if verbose:
            print("🚀 Starting Figma Agentic Analysis with LangGraph")
            print("=" * 60)

        # Step 1: Perform DFS to get node structure
        if verbose:
            print("📊 Step 1: Performing DFS to gather node structure...")

        all_nodes = self.figma_dfs.depth_first_search_from_node_id(
            file_key, node_id, max_depth=max_depth
        )

        if verbose:
            print(f"   Found {len(all_nodes)} nodes in structure")

        # Convert to structured data
        nodes_data = self._prepare_nodes_data(all_nodes)

        if verbose:
            print(f"   Processed {len(nodes_data)} nodes for analysis")

        # Log the initial data preparation
        log_gemini_interaction(
            "DATA_PREPARATION",
            f"Data preparation completed\nNodes found: {len(all_nodes)}\nProcessed nodes: {len(nodes_data)}",
            {
                "total_nodes": len(all_nodes),
                "processed_nodes": len(nodes_data),
                "node_types": list(
                    set(node.get("t", "unknown") for node in nodes_data)
                ),
            },
        )

        # Step 2: Set up the workflow
        if verbose:
            print("🤖 Step 2: Setting up agentic workflow...")

        # Step 3: Run the agentic workflow
        if verbose:
            print("🔍 Step 3: Running agentic analysis...")

        initial_state = {
            "messages": [],
            "nodes_data": nodes_data,
            "analysis_results": {},
        }

        start_time = time.time()

        # Run the workflow
        final_state = await self.workflow.ainvoke(initial_state)

        end_time = time.time()
        analysis_time = end_time - start_time

        # Step 4: Extract and display results
        results = final_state.get("analysis_results", {})

        # Log the analysis completion
        log_gemini_interaction(
            "ANALYSIS_COMPLETE",
            f"Analysis completed successfully\nTime: {analysis_time:.2f} seconds\nResults: {len(results)} elements found",
            {
                "analysis_time": analysis_time,
                "results_count": len(results),
                "success": True,
            },
        )

        if verbose:
            print(f"⏱️  Analysis completed in {analysis_time:.2f} seconds")
            print(f"📊 Found {len(results)} classified elements")
            print()

            # Display results by type
            if results:
                results_by_tag = {"input": [], "button": [], "select": [], "link": []}
                for node_id, node_info in results.items():
                    tag = node_info.get("tag", "unknown")
                    if tag in results_by_tag:
                        results_by_tag[tag].append(node_id)

                print("🎯 Detection Results by Element Type:")
                for tag in ["input", "button", "select", "link"]:
                    nodes = results_by_tag[tag]
                    print(f"{tag.upper()} Elements:")
                    if nodes:
                        print(f"  Node IDs: {nodes}")
                        print(f"  Count: {len(nodes)}")
                    else:
                        print("  Node IDs: []")
                        print("  Count: 0")
                    print()

            print("✅ Agentic analysis completed!")

        return results
