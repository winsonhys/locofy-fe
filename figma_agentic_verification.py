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
            max_output_tokens=20000,  # Increased to prevent truncation
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

    def _create_compact_json(self, data: Dict[str, Any]) -> str:
        """Create ultra-compact JSON with minimal token usage"""

        def compact_serialize(obj):
            if isinstance(obj, dict):
                # Use shorter keys and remove empty values
                result = {}
                for key, value in obj.items():
                    if (
                        value is not None
                        and value != ""
                        and value != []
                        and value != {}
                    ):
                        if isinstance(value, dict):
                            compacted = compact_serialize(value)
                            if compacted:  # Only add if not empty after compaction
                                result[key] = compacted
                        elif isinstance(value, list):
                            compacted = [
                                compact_serialize(item) for item in value if item
                            ]
                            if compacted:  # Only add if not empty after compaction
                                result[key] = compacted
                        else:
                            result[key] = value
                return result
            elif isinstance(obj, list):
                return [compact_serialize(item) for item in obj if item]
            else:
                return obj

        compacted = compact_serialize(data)
        return json.dumps(compacted, separators=(",", ":"), ensure_ascii=False)

    def _filter_api_response(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ultra-compact API response format with enhanced fields for UI element detection"""
        if not api_data or "nodes" not in api_data:
            return {}

        # Enhanced format: node_id -> [name, type, component_id, props, text, interactions, visual, children, ui_hints]
        compact_data = {}

        for node_id, node_data in api_data["nodes"].items():
            # Handle both direct node data and nested document structure
            if "document" in node_data:
                doc = node_data["document"]
            else:
                doc = node_data

            # Extract enhanced fields for UI element detection
            compact_node = []

            # 0: name
            name = doc.get("name", "")
            compact_node.append(name)

            # 1: type
            node_type = doc.get("type", "")
            compact_node.append(node_type)

            # 2: component_id (only if exists)
            component_id = doc.get("componentId")
            compact_node.append(component_id if component_id else "")

            # 3: enhanced component properties for UI detection
            props = {}
            if "componentProperties" in doc:
                for key, value in doc["componentProperties"].items():
                    # Enhanced property detection for UI elements
                    if any(
                        keyword in key.lower()
                        for keyword in [
                            # General UI properties
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
                            "enabled",
                            "active",
                            "hover",
                            "focus",
                            "pressed",
                            "loading",
                            "error",
                            "success",
                            # Input-specific properties
                            "input_type",
                            "inputmode",
                            "autocomplete",
                            "required",
                            "readonly",
                            "min",
                            "max",
                            "step",
                            "pattern",
                            "maxlength",
                            "minlength",
                            # Select-specific properties
                            "options",
                            "selected",
                            "multiple",
                            "dropdown",
                            "combobox",
                            # Button-specific properties
                            "button_type",
                            "button_style",
                            "primary",
                            "secondary",
                            "tertiary",
                            "icon",
                            "icon_position",
                            "size",
                            "width",
                            "height",
                            # Link-specific properties
                            "href",
                            "url",
                            "target",
                            "rel",
                            "external",
                            "internal",
                            # Form properties
                            "form",
                            "name",
                            "id",
                            "value",
                            "default_value",
                            "validation",
                            # Accessibility properties
                            "aria_label",
                            "aria_describedby",
                            "aria_required",
                            "role",
                            # Visual properties
                            "color",
                            "background",
                            "border",
                            "shadow",
                            "opacity",
                            # Layout properties
                            "width",
                            "height",
                            "padding",
                            "margin",
                            "flex",
                            "grid",
                        ]
                    ):
                        # Extract just the value, not the full object
                        if isinstance(value, dict) and "value" in value:
                            props[key] = value["value"]
                        else:
                            props[key] = str(value)
            # Only include props if not empty
            compact_node.append(props if props else {})

            # 4: text content (only if exists)
            text = doc.get("characters", "")
            compact_node.append(text)

            # 5: enhanced interactions with more detail
            interactions = []
            if "interactions" in doc and doc["interactions"]:
                for interaction in doc["interactions"]:
                    trigger = interaction.get("trigger", {})
                    trigger_type = trigger.get("type", "")
                    action = interaction.get("action", {})
                    action_type = action.get("type", "")

                    if trigger_type:
                        interaction_info = {"trigger": trigger_type}
                        if action_type:
                            interaction_info["action"] = action_type
                        interactions.append(interaction_info)
            # Only include interactions if not empty
            compact_node.append(interactions if interactions else [])

            # 6: enhanced visual indicators
            visual = {}
            if "cornerRadius" in doc and doc["cornerRadius"]:
                visual["r"] = doc["cornerRadius"]
            if "fills" in doc and doc["fills"]:
                visual["f"] = 1  # Has fills
                # Add fill color info for button/link detection
                if doc["fills"] and len(doc["fills"]) > 0:
                    fill = doc["fills"][0]
                    if "color" in fill:
                        visual["fc"] = fill["color"]  # fill color
            if "strokes" in doc and doc["strokes"]:
                visual["s"] = 1  # Has strokes
                # Add stroke info for button detection
                if doc["strokes"] and len(doc["strokes"]) > 0:
                    stroke = doc["strokes"][0]
                    if "color" in stroke:
                        visual["sc"] = stroke["color"]  # stroke color
            if "effects" in doc and doc["effects"]:
                visual["e"] = 1  # Has effects (shadows, etc.)
            if "opacity" in doc and doc["opacity"] != 1:
                visual["o"] = doc["opacity"]
            # Only include visual if not empty
            compact_node.append(visual if visual else {})

            # 7: children summary (always present but minimal)
            children_info = {"c": 0, "t": []}
            if "children" in doc and doc["children"]:
                child_types = list(set(child.get("type") for child in doc["children"]))
                children_info = {
                    "c": len(doc["children"]),
                    "t": child_types[:2],  # Only first 2 types
                }
            compact_node.append(children_info)

            # 8: UI-specific hints and metadata
            ui_hints = {}

            # Size hints for button/input detection
            if "absoluteBoundingBox" in doc:
                bbox = doc["absoluteBoundingBox"]
                width = bbox.get("width", 0)
                height = bbox.get("height", 0)
                if width > 0 and height > 0:
                    ui_hints["size"] = {"w": width, "h": height}
                    # Button-like proportions (typically wider than tall, but not too wide)
                    if 2 <= width / height <= 8 and height >= 20:
                        ui_hints["button_like"] = True
                    # Input-like proportions (typically wider than tall)
                    elif width / height >= 3 and height >= 20:
                        ui_hints["input_like"] = True

            # Text analysis for link/button detection
            if text:
                text_lower = text.lower()
                # Common button text patterns
                if any(
                    word in text_lower
                    for word in [
                        "submit",
                        "save",
                        "cancel",
                        "delete",
                        "edit",
                        "add",
                        "create",
                        "update",
                        "confirm",
                        "ok",
                        "yes",
                        "no",
                    ]
                ):
                    ui_hints["button_text"] = True
                # Common link text patterns
                if any(
                    word in text_lower
                    for word in [
                        "learn more",
                        "read more",
                        "view",
                        "see",
                        "click here",
                        "link",
                        "url",
                        "website",
                    ]
                ):
                    ui_hints["link_text"] = True
                # Common input placeholder patterns
                if any(
                    word in text_lower
                    for word in [
                        "enter",
                        "type",
                        "input",
                        "search",
                        "email",
                        "password",
                        "username",
                        "name",
                        "phone",
                    ]
                ):
                    ui_hints["input_placeholder"] = True

            # Component name analysis
            name_lower = name.lower()
            if any(word in name_lower for word in ["button", "btn", "cta"]):
                ui_hints["button_name"] = True
            elif any(
                word in name_lower for word in ["input", "field", "textbox", "textarea"]
            ):
                ui_hints["input_name"] = True
            elif any(
                word in name_lower
                for word in ["select", "dropdown", "combobox", "menu"]
            ):
                ui_hints["select_name"] = True
            elif any(word in name_lower for word in ["link", "href", "url", "anchor"]):
                ui_hints["link_name"] = True

            # Layout hints
            if "layoutMode" in doc:
                ui_hints["layout"] = doc["layoutMode"]

            # Auto-layout hints for button/input detection
            if "primaryAxisAlignItems" in doc:
                ui_hints["primary_align"] = doc["primaryAxisAlignItems"]
            if "counterAxisAlignItems" in doc:
                ui_hints["counter_align"] = doc["counterAxisAlignItems"]

            # Only include ui_hints if not empty
            compact_node.append(ui_hints if ui_hints else {})

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

WORKFLOW INSTRUCTIONS:
1. FIRST: Query the Figma API for detailed information about ALL nodes using this format:
   QUERY_API: {{"node_id": "node_id_here"}}
   You can query multiple nodes at once by separating node IDs with commas.
2. SECOND: After receiving the API data, analyze each node's properties, name, text content, and visual characteristics
3. THIRD: Provide your verification with confidence scores based on the actual Figma data

CRITICAL: You MUST query the Figma API FIRST before providing any verification results. Do not provide confidence scores without examining the actual node data.

DETECTION GUIDELINES:
- BUTTON: Look for button_like proportions, button_text patterns, button_name in component name, primary/secondary button styles, interactions with CLICK triggers
- INPUT: Look for input_like proportions, input_placeholder text patterns, input_name in component name, input_type properties, form-related properties
- SELECT: Look for select_name in component name, dropdown/combobox properties, options/selected properties, menu-related interactions
- LINK: Look for link_text patterns, link_name in component name, href/url properties, external/internal indicators, navigation interactions
- Use ui_hints to cross-reference with component properties and visual indicators for higher confidence

AVAILABLE FIGMA API ENDPOINTS:
- GET /files/{file_key}/nodes?ids={{node_id}} - Get specific node data

COMPACT API RESPONSE FORMAT:
API responses use enhanced compact format: {{"n": {{"node_id": [name, type, component_id, props, text, interactions, visual, children, ui_hints]}}}}
- name: element name (string)
- type: element type (INSTANCE, FRAME, TEXT, etc.)
- component_id: component identifier (string, empty if none)
- props: enhanced component properties for UI detection (object, empty {{}} if none)
  * Input properties: input_type, placeholder, required, readonly, autocomplete, etc.
  * Select properties: options, selected, multiple, dropdown, etc.
  * Button properties: button_type, button_style, primary, secondary, etc.
  * Link properties: href, url, target, external, etc.
  * Form properties: form, name, id, validation, etc.
  * Accessibility: aria_label, role, etc.
- text: text content (string, empty if none)
- interactions: enhanced interaction details [{{"trigger": type, "action": type}}] (array, empty [] if none)
- visual: {{"r": cornerRadius, "f": hasFills, "fc": fillColor, "s": hasStrokes, "sc": strokeColor, "e": hasEffects, "o": opacity}} (object, empty {{}} if none)
- children: {{"c": count, "t": [types]}} (object, always present)
- ui_hints: UI detection hints (object, empty {{}} if none)
  * size: {{"w": width, "h": height}} - element dimensions
  * button_like: true if proportions suggest button (2-8x wider than tall, height≥20)
  * input_like: true if proportions suggest input (≥3x wider than tall, height≥20)
  * button_text: true if text contains common button words (submit, save, cancel, etc.)
  * link_text: true if text contains common link words (learn more, view, see, etc.)
  * input_placeholder: true if text suggests input placeholder (enter, type, search, etc.)
  * button_name: true if name contains button-related words
  * input_name: true if name contains input-related words
  * select_name: true if name contains select-related words
  * link_name: true if name contains link-related words
  * layout: auto-layout mode (HORIZONTAL, VERTICAL, etc.)
  * primary_align: primary axis alignment
  * counter_align: counter axis alignment

RESPONSE FORMAT (provide this JSON structure after analyzing API data):
{{
  "verifications": [
    {{
      "node_id": "node_id",
      "classification": "input|button|select|link",
      "confidence": 0.95,
      "notes": "verification notes based on actual Figma data"
    }}
  ]
}}

IMPORTANT CLASSIFICATION RULES:
- Only classify elements as one of the 4 types: "input", "button", "select", "link"
- For every node, assign the most plausible tag and a confidence score based on actual Figma data
- Do not use "none" as a classification and do not exclude any node
- Focus on elements that are clearly interactive UI components
- Exclude decorative elements, icons that are part of other components, and non-interactive elements
- Base your confidence scores on the actual properties, names, and characteristics from the Figma API data

STEP 1: Query the Figma API for all nodes to get detailed information.
STEP 2: Analyze the API response data for each node.
STEP 3: Provide verification results with confidence scores based on the actual data."""

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
        retry_count = 0
        max_retries = 3

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

                # Check if response is empty
                if not content.strip():
                    logger.warning(
                        f"Empty response from Gemini on iteration {iteration + 1}"
                    )
                    retry_count += 1

                    if retry_count >= max_retries:
                        logger.error("Max retries reached for empty responses")
                        verification_notes.append(
                            "Error: Gemini returned empty responses after multiple attempts"
                        )
                        break

                    # Add a more explicit follow-up message
                    follow_up_message = "You must provide a JSON response. Please analyze the nodes and return the verification results in the specified JSON format."
                    messages.append(HumanMessage(content=follow_up_message))
                    prompt_log.append("Follow-up Message (retry):")
                    prompt_log.append(follow_up_message)
                    prompt_log.append("")
                    continue

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
                                    compact_json = self._create_compact_json(
                                        filtered_api_data
                                    )
                                    prompt_log.append(compact_json)
                                    prompt_log.append("")

                                    # Add filtered API response to conversation
                                    api_response = (
                                        f"API DATA for node {node_id}: {compact_json}"
                                    )
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
                    follow_up_message = "Please continue with your verification analysis. Remember to provide a complete JSON response."
                    messages.append(HumanMessage(content=follow_up_message))

                    # Log follow-up message
                    prompt_log.append("Follow-up Message:")
                    prompt_log.append(follow_up_message)
                    prompt_log.append("")

                else:
                    # Try to parse the final verification result
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
                            retry_count += 1

                            if retry_count >= max_retries:
                                verification_notes.append(
                                    "Error: No JSON content found in response after multiple attempts"
                                )
                                break

                            # Add explicit instruction to provide JSON
                            follow_up_message = "Please provide your verification results in the exact JSON format specified. Do not include any text outside the JSON object."
                            messages.append(HumanMessage(content=follow_up_message))
                            prompt_log.append("Follow-up Message (JSON required):")
                            prompt_log.append(follow_up_message)
                            prompt_log.append("")
                            continue

                        verification_data = json.loads(json_content)

                        # Log final verification result
                        prompt_log.append("FINAL VERIFICATION RESULT:")
                        prompt_log.append("-" * 40)
                        prompt_log.append(json.dumps(verification_data, indent=2))
                        prompt_log.append("")

                        # Extract verification results
                        verifications = verification_data.get("verifications", {})

                        # Handle both list and dictionary formats
                        if isinstance(verifications, list):
                            # Convert list format to dictionary format
                            for verification in verifications:
                                node_id = verification.get("node_id")
                                if node_id:
                                    verified_results[node_id] = {
                                        "tag": verification.get(
                                            "classification",
                                            verification.get("verified_tag", "none"),
                                        ),
                                        "confidence": verification.get(
                                            "confidence", 0.5
                                        ),
                                        "notes": verification.get("notes", ""),
                                    }
                                    confidence_scores[node_id] = verification.get(
                                        "confidence", 0.5
                                    )
                        else:
                            # Handle dictionary format
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
                        retry_count += 1

                        if retry_count >= max_retries:
                            verification_notes.append(
                                f"Failed to parse response after multiple attempts: {content[:100]}..."
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
                                plausible_tag = original_data.get("tag", "input")
                                if plausible_tag not in [
                                    "input",
                                    "button",
                                    "select",
                                    "link",
                                ]:
                                    plausible_tag = "input"  # default fallback
                                verified_results[node_id] = {
                                    "tag": plausible_tag,
                                    "confidence": 0.2,  # Very low confidence due to parsing failure
                                    "notes": "Fallback verification due to parsing error; assigned most plausible tag with low confidence.",
                                }
                                confidence_scores[node_id] = 0.2

                            verification_notes.append(
                                "Fallback verification applied due to parsing error"
                            )
                            break

                        # Add instruction to fix JSON format
                        follow_up_message = f"Your response contains invalid JSON. Please fix the format and provide a valid JSON response. Error: {str(e)}"
                        messages.append(HumanMessage(content=follow_up_message))
                        prompt_log.append("Follow-up Message (JSON error):")
                        prompt_log.append(follow_up_message)
                        prompt_log.append("")
                        continue

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
        self,
        file_key: str,
        node_id: str,
        max_depth: Optional[int] = None,
        use_chunked_verification: bool = True,
        chunk_size: int = 8,
    ) -> Dict[str, Dict[str, str]]:
        """
        Run the complete agentic verification workflow and return final results in the specified format.

        This is a convenience method that combines run_agentic_verification and get_final_results.

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            max_depth: Maximum depth for DFS traversal
            use_chunked_verification: Whether to use chunked verification (default: True)
            chunk_size: Number of nodes to process in each chunk when using chunked verification (default: 8)

        Returns:
            Dictionary in format: {node_id: {"tag": detected_tag}}
        """
        logger.info(f"🚀 Running complete workflow with final results output")
        logger.info(
            f"Verification mode: {'Chunked' if use_chunked_verification else 'Non-chunked'}"
        )

        # Run the verification workflow
        verification_result = await self.run_agentic_verification(
            file_key, node_id, max_depth, use_chunked_verification, chunk_size
        )

        # Get final results in the specified format
        final_results = self.get_final_results(verification_result)

        logger.info(
            f"✅ Complete workflow finished: {len(final_results)} final results"
        )
        return final_results

    async def run_with_detailed_results(
        self,
        file_key: str,
        node_id: str,
        max_depth: Optional[int] = None,
        use_chunked_verification: bool = True,
        chunk_size: int = 8,
    ) -> tuple[Dict[str, Dict[str, str]], VerificationResult]:
        """
        Run the complete agentic verification workflow and return both final results and detailed verification info.

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            max_depth: Maximum depth for DFS traversal
            use_chunked_verification: Whether to use chunked verification (default: True)
            chunk_size: Number of nodes to process in each chunk when using chunked verification (default: 8)

        Returns:
            Tuple of (final_results, verification_result) where:
            - final_results: Dictionary in format: {node_id: {"tag": detected_tag}}
            - verification_result: Detailed VerificationResult with confidence scores, notes, etc.
        """
        logger.info(f"🚀 Running complete workflow with detailed results output")
        logger.info(
            f"Verification mode: {'Chunked' if use_chunked_verification else 'Non-chunked'}"
        )

        # Run the verification workflow
        verification_result = await self.run_agentic_verification(
            file_key, node_id, max_depth, use_chunked_verification, chunk_size
        )

        # Get final results in the specified format
        final_results = self.get_final_results(verification_result)

        logger.info(
            f"✅ Complete workflow finished: {len(final_results)} final results"
        )
        return final_results, verification_result

    async def run_agentic_verification(
        self,
        file_key: str,
        node_id: str,
        max_depth: Optional[int] = None,
        use_chunked_verification: bool = True,
        chunk_size: int = 8,
    ) -> VerificationResult:
        """
        Run the complete agentic verification workflow

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            max_depth: Maximum depth for DFS traversal
            use_chunked_verification: Whether to use chunked verification (default: True)
            chunk_size: Number of nodes to process in each chunk when using chunked verification (default: 8)

        Returns:
            VerificationResult with original and verified results
        """
        logger.info(f"🚀 Starting agentic verification workflow")
        logger.info(f"File: {file_key}, Node: {node_id}, Max Depth: {max_depth}")
        logger.info(
            f"Verification mode: {'Chunked' if use_chunked_verification else 'Non-chunked'}"
        )

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
            if use_chunked_verification:
                logger.info(
                    f"🔍 Step 2: Running chunked verification with Figma API access (chunk_size: {chunk_size})..."
                )
                verification_result = (
                    await self._handle_verification_with_api_access_chunked(
                        original_results,
                        file_key,
                        chunk_size=chunk_size,
                    )
                )
            else:
                logger.info(
                    "🔍 Step 2: Running non-chunked verification with Figma API access..."
                )
                verification_result = await self._handle_verification_with_api_access(
                    original_results,
                    file_key,
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

    async def _handle_verification_with_api_access_chunked(
        self,
        original_results: Dict[str, Dict[str, str]],
        file_key: str,
        chunk_size: int = 10,
    ) -> VerificationResult:
        """
        Handle the verification process with Figma API access using chunked processing

        Args:
            original_results: Original analysis results
            file_key: Figma file key
            chunk_size: Number of nodes to process in each chunk (default: 10)
        """
        verification_notes = []
        api_queries_made = []
        confidence_scores = {}
        verified_results = {}

        # Convert results to list for chunking
        node_items = list(original_results.items())
        total_nodes = len(node_items)

        logger.info(
            f"Starting chunked verification for {total_nodes} nodes with chunk size {chunk_size}"
        )
        verification_notes.append(
            f"Processing {total_nodes} nodes in chunks of {chunk_size}"
        )

        # Process nodes in chunks
        for chunk_idx, i in enumerate(range(0, total_nodes, chunk_size)):
            chunk_end = min(i + chunk_size, total_nodes)
            chunk_items = node_items[i:chunk_end]
            chunk_results = dict(chunk_items)

            logger.info(
                f"Processing chunk {chunk_idx + 1}/{(total_nodes + chunk_size - 1) // chunk_size}: nodes {i+1}-{chunk_end}"
            )
            verification_notes.append(
                f"Chunk {chunk_idx + 1}: Processing nodes {i+1}-{chunk_end}"
            )

            # Process this chunk
            chunk_verification = await self._process_verification_chunk(
                chunk_results, file_key, chunk_idx + 1
            )

            # Merge results
            verified_results.update(chunk_verification.verified_results)
            confidence_scores.update(chunk_verification.confidence_scores)
            api_queries_made.extend(chunk_verification.api_queries_made)
            verification_notes.extend(chunk_verification.verification_notes)

        # Calculate overall confidence
        if confidence_scores:
            overall_confidence = sum(confidence_scores.values()) / len(
                confidence_scores
            )
        else:
            overall_confidence = 0.0

        verification_notes.append(
            f"Chunked verification completed. Overall confidence: {overall_confidence:.2f}"
        )

        return VerificationResult(
            original_results=original_results,
            verified_results=verified_results,
            verification_notes=verification_notes,
            confidence_scores=confidence_scores,
            api_queries_made=api_queries_made,
        )

    async def _process_verification_chunk(
        self, chunk_results: Dict[str, Dict[str, str]], file_key: str, chunk_number: int
    ) -> VerificationResult:
        """
        Process a single chunk of nodes for verification

        Args:
            chunk_results: Results for this chunk of nodes
            file_key: Figma file key
            chunk_number: Current chunk number for logging

        Returns:
            VerificationResult for this chunk
        """
        verification_notes = []
        api_queries_made = []
        confidence_scores = {}
        verified_results = {}

        # Create chunk-specific prompt
        prompt = self._create_chunk_verification_prompt(
            chunk_results, file_key, chunk_number
        )

        # Write chunk prompts to file for transparency
        prompt_log = []
        prompt_log.append(f"CHUNK {chunk_number} VERIFICATION LOG")
        prompt_log.append("=" * 60)
        prompt_log.append(f"File Key: {file_key}")
        prompt_log.append(f"Chunk Number: {chunk_number}")
        prompt_log.append(f"Nodes in chunk: {len(chunk_results)}")
        prompt_log.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        prompt_log.append("")

        # Start conversation with Gemini for this chunk
        messages = [
            SystemMessage(
                content="You are a Figma UI element verification expert with API access. Process the nodes in this chunk and provide verification results."
            ),
            HumanMessage(content=prompt),
        ]

        # Log chunk prompt
        prompt_log.append("CHUNK PROMPT:")
        prompt_log.append("-" * 40)
        prompt_log.append(prompt)
        prompt_log.append("")

        max_iterations = 3  # Reduced for chunks
        iteration = 0
        retry_count = 0
        max_retries = 2

        while iteration < max_iterations:
            try:
                # Get response from Gemini
                response = await self.llm.ainvoke(messages)
                content = response.content.strip()

                # Log Gemini's response
                prompt_log.append(
                    f"CHUNK {chunk_number} - ITERATION {iteration + 1} - GEMINI RESPONSE:"
                )
                prompt_log.append("-" * 40)
                prompt_log.append(content)
                prompt_log.append("")

                # Check if response is empty
                if not content.strip():
                    logger.warning(
                        f"Empty response from Gemini for chunk {chunk_number}, iteration {iteration + 1}"
                    )
                    retry_count += 1

                    if retry_count >= max_retries:
                        logger.error(f"Max retries reached for chunk {chunk_number}")
                        verification_notes.append(
                            f"Error: Gemini returned empty responses for chunk {chunk_number} after multiple attempts"
                        )
                        # Assign low confidence to all nodes in this chunk
                        for node_id in chunk_results.keys():
                            verified_results[node_id] = {
                                "tag": chunk_results[node_id].get("tag", "none"),
                                "confidence": 0.1,
                                "notes": "Low confidence due to empty response from Gemini",
                            }
                            confidence_scores[node_id] = 0.1
                        break

                    # Add a more explicit follow-up message
                    follow_up_message = f"You must provide a JSON response for chunk {chunk_number}. Please analyze the {len(chunk_results)} nodes and return the verification results in the specified JSON format."
                    messages.append(HumanMessage(content=follow_up_message))
                    prompt_log.append("Follow-up Message (retry):")
                    prompt_log.append(follow_up_message)
                    prompt_log.append("")
                    continue

                # Check if Gemini wants to query the API
                if "QUERY_API:" in content:
                    # Process API queries for this chunk
                    chunk_api_queries, chunk_messages = (
                        await self._process_chunk_api_queries(
                            content, file_key, prompt_log, chunk_number
                        )
                    )
                    api_queries_made.extend(chunk_api_queries)
                    messages.extend(chunk_messages)

                    follow_up_message = f"Please continue with your verification analysis for chunk {chunk_number}. Remember to provide a complete JSON response."
                    messages.append(HumanMessage(content=follow_up_message))
                    prompt_log.append("Follow-up Message:")
                    prompt_log.append(follow_up_message)
                    prompt_log.append("")

                else:
                    # Try to parse the final verification result for this chunk
                    chunk_verified_results, chunk_confidence_scores = (
                        self._parse_chunk_verification_response(
                            content, chunk_results, chunk_number, prompt_log
                        )
                    )

                    if chunk_verified_results:
                        verified_results.update(chunk_verified_results)
                        confidence_scores.update(chunk_confidence_scores)
                        verification_notes.append(
                            f"Chunk {chunk_number} verification completed successfully"
                        )
                        break
                    else:
                        retry_count += 1
                        if retry_count >= max_retries:
                            verification_notes.append(
                                f"Failed to parse verification response for chunk {chunk_number} after multiple attempts"
                            )
                            # Assign low confidence to all nodes in this chunk
                            for node_id in chunk_results.keys():
                                verified_results[node_id] = {
                                    "tag": chunk_results[node_id].get("tag", "none"),
                                    "confidence": 0.1,
                                    "notes": "Low confidence due to parsing failure",
                                }
                                confidence_scores[node_id] = 0.1
                            break

                        # Add explicit instruction to provide JSON
                        follow_up_message = f"Please provide your verification results for chunk {chunk_number} in the exact JSON format specified. Do not include any text outside the JSON object."
                        messages.append(HumanMessage(content=follow_up_message))
                        prompt_log.append("Follow-up Message (JSON required):")
                        prompt_log.append(follow_up_message)
                        prompt_log.append("")

                iteration += 1

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_number}: {e}")
                verification_notes.append(
                    f"Error processing chunk {chunk_number}: {str(e)}"
                )
                # Assign low confidence to all nodes in this chunk
                for node_id in chunk_results.keys():
                    verified_results[node_id] = {
                        "tag": chunk_results[node_id].get("tag", "none"),
                        "confidence": 0.1,
                        "notes": f"Error: {str(e)}",
                    }
                    confidence_scores[node_id] = 0.1
                break

        # Write chunk log to file
        self._write_chunk_log_to_file(prompt_log, file_key, chunk_number)

        return VerificationResult(
            original_results=chunk_results,
            verified_results=verified_results,
            verification_notes=verification_notes,
            confidence_scores=confidence_scores,
            api_queries_made=api_queries_made,
        )

    def _create_chunk_verification_prompt(
        self, chunk_results: Dict[str, Dict[str, str]], file_key: str, chunk_number: int
    ) -> str:
        """Create a prompt for verifying a chunk of nodes"""

        # Convert results to a more readable format
        results_summary = []
        for node_id, result in chunk_results.items():
            tag = result.get("tag", "unknown")
            results_summary.append(f"Node {node_id}: {tag}")

        prompt = f"""You are a Figma UI element verification expert. You have access to the Figma API to verify the classification of UI elements.

CHUNK {chunk_number} ANALYSIS RESULTS:
{chr(10).join(results_summary)}

FILE KEY: {file_key}

WORKFLOW INSTRUCTIONS:
1. FIRST: Query the Figma API for detailed information about ALL {len(chunk_results)} nodes using this format:
   QUERY_API: {{"node_id": "node_id_here"}}
   You can query multiple nodes at once by separating node IDs with commas.
2. SECOND: After receiving the API data, analyze each node's properties, name, text content, and visual characteristics
3. THIRD: Provide your verification with confidence scores based on the actual Figma data

CRITICAL: You MUST query the Figma API FIRST before providing any verification results. Do not provide confidence scores without examining the actual node data.

DETECTION GUIDELINES:
- BUTTON: Look for button_like proportions, button_text patterns, button_name in component name, primary/secondary button styles, interactions with CLICK triggers
- INPUT: Look for input_like proportions, input_placeholder text patterns, input_name in component name, input_type properties, form-related properties
- SELECT: Look for select_name in component name, dropdown/combobox properties, options/selected properties, menu-related interactions
- LINK: Look for link_text patterns, link_name in component name, href/url properties, external/internal indicators, navigation interactions
- Use ui_hints to cross-reference with component properties and visual indicators for higher confidence

AVAILABLE FIGMA API ENDPOINTS:
- GET /files/{file_key}/nodes?ids={{node_id}} - Get specific node data

COMPACT API RESPONSE FORMAT:
API responses use enhanced compact format: {{"n": {{"node_id": [name, type, component_id, props, text, interactions, visual, children, ui_hints]}}}}
- name: element name (string)
- type: element type (INSTANCE, FRAME, TEXT, etc.)
- component_id: component identifier (string, empty if none)
- props: enhanced component properties for UI detection (object, empty {{}} if none)
- text: text content (string, empty if none)
- interactions: enhanced interaction details [{{"trigger": type, "action": type}}] (array, empty [] if none)
- visual: {{"r": cornerRadius, "f": hasFills, "fc": fillColor, "s": hasStrokes, "sc": strokeColor, "e": hasEffects, "o": opacity}} (object, empty {{}} if none)
- children: {{"c": count, "t": [types]}} (object, always present)
- ui_hints: UI detection hints (object, empty {{}} if none)

RESPONSE FORMAT (provide this JSON structure after analyzing API data):
{{
  "verifications": {{
    "node_id": {{
      "original_tag": "original_tag",
      "verified_tag": "verified_tag", 
      "confidence": 0.95,
      "notes": "verification notes based on actual Figma data"
    }}
  }},
  "api_queries": ["list of node_ids that were queried"],
  "overall_confidence": 0.92
}}

IMPORTANT CLASSIFICATION RULES:
- Only classify elements as one of the 4 types: "input", "button", "select", "link"
- For every node, assign the most plausible tag and a confidence score based on actual Figma data
- Do not use "none" as a classification and do not exclude any node
- Focus on elements that are clearly interactive UI components
- Exclude decorative elements, icons that are part of other components, and non-interactive elements
- Base your confidence scores on the actual properties, names, and characteristics from the Figma API data

STEP 1: Query the Figma API for all {len(chunk_results)} nodes to get detailed information.
STEP 2: Analyze the API response data for each node.
STEP 3: Provide verification results with confidence scores based on the actual data."""

        return prompt

    async def _process_chunk_api_queries(
        self, content: str, file_key: str, prompt_log: List[str], chunk_number: int
    ) -> tuple[List[str], List[HumanMessage]]:
        """Process API queries for a chunk"""

        api_queries_made = []
        new_messages = []

        prompt_log.append(f"CHUNK {chunk_number} - API QUERIES DETECTED:")
        prompt_log.append("-" * 40)

        lines = content.split("\n")
        for line in lines:
            if line.strip().startswith("QUERY_API:"):
                try:
                    # Extract the JSON part after QUERY_API:
                    json_str = line.replace("QUERY_API:", "").strip()
                    query_data = json.loads(json_str)
                    node_id = query_data.get("node_id")

                    if node_id:
                        logger.info(
                            f"Gemini requesting API data for node: {node_id} (chunk {chunk_number})"
                        )
                        api_queries_made.append(node_id)

                        # Log API query
                        prompt_log.append(f"API Query: {json_str}")

                        # Query Figma API
                        api_data = await self._query_figma_api(file_key, node_id)

                        # Filter API response to reduce token usage
                        filtered_api_data = self._filter_api_response(api_data)

                        # Log API response
                        prompt_log.append("API Response:")
                        compact_json = self._create_compact_json(filtered_api_data)
                        prompt_log.append(compact_json)
                        prompt_log.append("")

                        # Add filtered API response to conversation
                        api_response = f"API DATA for node {node_id}: {compact_json}"
                        new_messages.append(HumanMessage(content=api_response))

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse API query: {line}")
                    prompt_log.append(f"ERROR: Failed to parse API query: {line}")
                    continue
            else:
                new_messages.append(HumanMessage(content=line))

        return api_queries_made, new_messages

    def _parse_chunk_verification_response(
        self,
        content: str,
        chunk_results: Dict[str, Dict[str, str]],
        chunk_number: int,
        prompt_log: List[str],
    ) -> tuple[Dict[str, Dict[str, str]], Dict[str, float]]:
        """Parse verification response for a chunk"""

        verified_results = {}
        confidence_scores = {}

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
                # Look for JSON object in the response
                brace_start = content.find("{")
                brace_end = content.rfind("}")
                if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                    json_content = content[brace_start : brace_end + 1]

            if not json_content.strip():
                logger.error(f"No JSON content found in chunk {chunk_number} response")
                return {}, {}

            verification_data = json.loads(json_content)

            # Log final verification result
            prompt_log.append(f"CHUNK {chunk_number} - FINAL VERIFICATION RESULT:")
            prompt_log.append("-" * 40)
            prompt_log.append(json.dumps(verification_data, indent=2))
            prompt_log.append("")

            # Extract verification results
            verifications = verification_data.get("verifications", {})

            # Handle both list and dictionary formats
            if isinstance(verifications, list):
                # Convert list format to dictionary format
                for verification in verifications:
                    node_id = verification.get("node_id")
                    if node_id:
                        verified_results[node_id] = {
                            "tag": verification.get(
                                "classification",
                                verification.get("verified_tag", "none"),
                            ),
                            "confidence": verification.get("confidence", 0.5),
                            "notes": verification.get("notes", ""),
                        }
                        confidence_scores[node_id] = verification.get("confidence", 0.5)
            else:
                # Handle dictionary format
                for node_id, verification in verifications.items():
                    verified_results[node_id] = {
                        "tag": verification.get("verified_tag", "none"),
                        "confidence": verification.get("confidence", 0.5),
                        "notes": verification.get("notes", ""),
                    }
                    confidence_scores[node_id] = verification.get("confidence", 0.5)

            return verified_results, confidence_scores

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse chunk {chunk_number} verification response: {e}"
            )
            prompt_log.append("ERROR: Failed to parse verification response")
            prompt_log.append(f"Error: {str(e)}")
            prompt_log.append("Raw response:")
            prompt_log.append(content)
            return {}, {}

    def _write_chunk_log_to_file(
        self, prompt_log: List[str], file_key: str, chunk_number: int
    ):
        """Write chunk verification log to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"verification_chunk_{chunk_number}_{file_key}_{timestamp}.txt"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(prompt_log))
            logger.info(f"Chunk {chunk_number} verification log written to {filename}")
        except Exception as e:
            logger.error(f"Failed to write chunk {chunk_number} log: {e}")


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
            use_chunked_verification=False,  # Use chunked verification (default)
            chunk_size=8,  # Process 8 nodes at a time
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

        else:
            print("❌ No results obtained")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
