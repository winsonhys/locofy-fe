#!/usr/bin/env python3
"""
Gemini Analyzer - Modular AI analysis for Figma nodes
Handles prompt creation and API calling for different detection types
"""

import json
import logging
import asyncio
import aiohttp
import google.generativeai as genai
from typing import Dict, List, Any, Optional
from config import GEMINI_API_KEY, GEMINI_MODEL_NAME
from input_detection_prompts import (
    InputDetectionPromptCreator,
    ButtonDetectionPromptCreator,
    LinkDetectionPromptCreator,
    CombinedDetectionPromptCreator,
)

logger = logging.getLogger(__name__)


class GeminiAnalyzer:
    """Handles Gemini AI analysis for Figma nodes with different detection types"""

    def __init__(self):
        """Initialize Gemini Analyzer with API configuration"""
        self._configure_gemini()
        self.input_prompt_creator = InputDetectionPromptCreator()
        self.button_prompt_creator = ButtonDetectionPromptCreator()
        self.link_prompt_creator = LinkDetectionPromptCreator()
        self.combined_prompt_creator = CombinedDetectionPromptCreator()

    def _configure_gemini(self):
        """Configure Gemini API"""
        logger.debug("Configuring Gemini API...")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        self.api_key = GEMINI_API_KEY  # Store API key for async HTTP calls
        logger.debug("Gemini API configured successfully")

    async def analyze_nodes(
        self, nodes: List[Dict[str, Any]], detection_type: str = "text_input"
    ) -> Dict[str, Dict[str, str]]:
        """
        Analyze nodes using Gemini AI asynchronously for specified detection type

        Args:
            nodes: List of node dictionaries with complete structure
            detection_type: Type of detection to perform (e.g., "text_input", "button", "form")

        Returns:
            Dictionary with node_id as keys and detection results as values
        """
        logger.info(
            f"Starting async Gemini {detection_type} detection for {len(nodes)} nodes"
        )

        # Create prompt based on detection type
        prompt = self._create_prompt(nodes, detection_type)

        # Write prompt to file for debugging
        self._write_prompt_to_file(prompt, detection_type)

        try:
            # Send request to Gemini asynchronously
            logger.info(f"Sending async request to Gemini API for {detection_type}...")
            response = await self._generate_content_async(prompt)
            logger.info(
                f"Gemini API response received successfully for {detection_type}"
            )

            # Parse response
            result = self._parse_response(response, detection_type)
            logger.info(
                f"Successfully parsed {len(result)} {detection_type} nodes from Gemini response"
            )

            return result

        except Exception as e:
            logger.error(f"Error using Gemini for {detection_type} detection: {e}")
            logger.info(
                f"Gemini {detection_type} detection failed - returning empty result"
            )
            return {}

    async def _generate_content_async(self, prompt: str) -> str:
        """
        Generate content asynchronously using Gemini API

        Args:
            prompt: The prompt to send to Gemini

        Returns:
            Response text from Gemini
        """
        # Use aiohttp for truly async API calls
        import aiohttp

        # Configure the request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": 8192,
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    error_text = await response.text()
                    raise Exception(f"Gemini API error {response.status}: {error_text}")

    async def analyze_multiple_types(
        self, nodes: List[Dict[str, Any]], detection_types: List[str] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Analyze nodes for multiple detection types concurrently and combine results

        Args:
            nodes: List of node dictionaries with complete structure
            detection_types: List of detection types to analyze (default: ["text_input", "button", "link"])

        Returns:
            Combined dictionary with node_id as keys and detection results as values
        """
        if detection_types is None:
            detection_types = ["text_input", "button", "link"]

        logger.info(
            f"Starting concurrent analysis for {len(detection_types)} detection types: {detection_types}"
        )

        # Create tasks for all detection types to run concurrently
        tasks = []
        for detection_type in detection_types:
            task = self._analyze_detection_type(nodes, detection_type)
            tasks.append(task)

        # Wait for all tasks to complete concurrently
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        results = {}
        for i, result in enumerate(results_list):
            detection_type = detection_types[i]

            if isinstance(result, Exception):
                logger.error(f"Error in {detection_type} detection: {result}")
                results[detection_type] = {}
            else:
                results[detection_type] = result
                logger.info(
                    f"{detection_type} detection completed with {len(result)} results"
                )

                # Print detailed results for this detection type
                if result:
                    print(f"\n🔍 {detection_type.upper()} DETECTION RESULTS:")
                    print(f"Found {len(result)} {detection_type} elements:")

                    # Print the detection results dictionary
                    detection_dict = {}
                    node_ids = []
                    for node_id, node_info in result.items():
                        tag = node_info.get("tag", "unknown")
                        detection_dict[node_id] = {"tag": tag}
                        node_ids.append(node_id)

                    print(f"Detection Dictionary: {detection_dict}")
                    print(f"Node IDs: {node_ids}")
                    print()
                else:
                    print(f"\n🔍 {detection_type.upper()} DETECTION RESULTS:")
                    print(f"No {detection_type} elements found")
                    print()

        # Combine all results
        combined_results = self._combine_results(results)
        logger.info(
            f"Combined results: {len(combined_results)} total unique nodes from {len(detection_types)} detection types"
        )

        # Print combined results summary
        if combined_results:
            print(f"\n📊 COMBINED DETECTION RESULTS:")
            print(f"Total unique nodes detected: {len(combined_results)}")

            # Create simplified combined dictionary with only node_id and tag
            combined_dict = {}
            all_node_ids = []
            for node_id, node_info in combined_results.items():
                tag = node_info.get("tag", "unknown")
                combined_dict[node_id] = {"tag": tag}
                all_node_ids.append(node_id)

            print(f"Combined Dictionary: {combined_dict}")
            print(f"All Node IDs: {all_node_ids}")
            print()
        else:
            print(f"\n📊 COMBINED DETECTION RESULTS:")
            print("No elements detected across all detection types")
            print()

        return combined_results

    async def _analyze_detection_type(
        self, nodes: List[Dict[str, Any]], detection_type: str
    ) -> Dict[str, Dict[str, str]]:
        """
        Analyze nodes for a specific detection type (wrapper for analyze_nodes)

        Args:
            nodes: List of node dictionaries
            detection_type: Type of detection to perform

        Returns:
            Detection results for the specified type
        """
        logger.info(f"Starting {detection_type} detection...")
        try:
            result = await self.analyze_nodes(nodes, detection_type)
            logger.info(f"{detection_type} detection completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {detection_type} detection: {e}")
            raise

    def _combine_results(
        self, results: Dict[str, Dict[str, Dict[str, str]]]
    ) -> Dict[str, Dict[str, str]]:
        """
        Combine results from multiple detection types

        Args:
            results: Dictionary with detection_type as keys and results as values

        Returns:
            Combined dictionary with node_id as keys and {"tag": "value"} as values
        """
        combined = {}

        for detection_type, detection_results in results.items():
            for node_id, node_info in detection_results.items():
                # Extract clean node ID from verbose format if present
                clean_node_id = node_id.split("|")[0] if "|" in node_id else node_id

                # Get the tag from node_info
                if isinstance(node_info, dict):
                    tag = node_info.get("tag", "unknown")
                else:
                    tag = str(node_info)

                # If node already exists, keep the existing tag (first detection wins)
                if clean_node_id not in combined:
                    combined[clean_node_id] = {"tag": tag}

        return combined

    def _create_prompt(self, nodes: List[Dict[str, Any]], detection_type: str) -> str:
        """
        Create prompt for specific detection type

        Args:
            nodes: List of node dictionaries
            detection_type: Type of detection to perform

        Returns:
            Formatted prompt string for Gemini
        """
        logger.info(
            f"Creating Gemini prompt for {detection_type} detection with {len(nodes)} nodes..."
        )

        # Use specialized prompt creator for input detection
        if detection_type == "text_input":
            return self.input_prompt_creator.create_prompt(nodes)
        elif detection_type == "button":
            return self.button_prompt_creator.create_prompt(nodes)
        elif detection_type == "link":
            return self.link_prompt_creator.create_prompt(nodes)

        # For other detection types, use the generic approach
        config = self._get_detection_config(detection_type)

        prompt = f"""You are a UI analyst. Identify {config['description']} from Figma nodes.

## Detection Criteria:
{config['criteria']}

## Output Format:
```json
{config['output_format']}
```

## Nodes to Analyze:"""

        # Add filtered nodes to prompt
        filtered_nodes = self._filter_nodes_for_detection(nodes, detection_type)

        for node in filtered_nodes:
            node_id = node.get("node_id", "N/A")
            node_type = node.get("type", "N/A")
            node_name = node.get("name", "N/A")

            prompt += f"\n{node_id}|{node_type}|{node_name}"

            # Add additional data based on detection type
            if "data" in node and node["data"]:
                data = node["data"]

                # Add corner radius for input-like styling
                if "cornerRadius" in data and data.get("cornerRadius", 0) > 0:
                    prompt += f"|r{data.get('cornerRadius')}"

                # Add text content for relevant nodes
                if (
                    node_type in ["FRAME", "RECTANGLE", "INSTANCE"]
                    and "characters" in data
                ):
                    chars = data.get("characters", "").strip()
                    if chars and len(chars) < 50:
                        prompt += f"|t{chars}"

        prompt += f"""
Analyze the nodes above and return your JSON response:"""

        logger.info(
            f"Gemini prompt created successfully (length: {len(prompt)} characters)"
        )
        return prompt

    def _get_detection_config(self, detection_type: str) -> Dict[str, str]:
        """
        Get configuration for specific detection type

        Args:
            detection_type: Type of detection

        Returns:
            Configuration dictionary
        """
        configs = {
            "text_input": {
                "description": "main text input elements",
                "criteria": """- Main input containers (FRAME, RECTANGLE, INSTANCE) with input functionality
- Named like "Search Bar", "Input Field", "Text Input", "Search"
- MUST be the main input container, not child elements
- Exclude: icons, buttons, labels, help text, placeholder text, backgrounds
- Exclude: any node that is a child of another input container""",
                "output_format": """{
  "<node_id>": {"tag": "input"},
  "<node_id>": {"tag": "input"}
}""",
            },
            "select": {
                "description": "dropdown/select elements",
                "criteria": """- Dropdown/select containers (FRAME, RECTANGLE, INSTANCE) with dropdown functionality
- Named like "Select", "Dropdown", "Menu", "Choose", "Filter"
- Contains text + dropdown icon on the right (arrows, chevrons)
- MUST be the main select container, not child elements
- Include: dropdown buttons, select menus, filter dropdowns
- Exclude: regular action buttons without dropdown indicators""",
                "output_format": """{
  "<node_id>": {"tag": "select"},
  "<node_id>": {"tag": "select"}
}""",
            },
            "button": {
                "description": "button elements",
                "criteria": """- Button containers (FRAME, RECTANGLE, INSTANCE) with button functionality
- Named like "Button", "IconButton", "Submit", "Add", "Delete"
- MUST be the main button container, not child elements
- Exclude: icons, labels, text elements that are children of buttons
- Include: primary buttons, secondary buttons, icon buttons""",
                "output_format": """{
  "<node_id>": {"tag": "button"},
  "<node_id>": {"tag": "button"}
}""",
            },
            "form": {
                "description": "form containers",
                "criteria": """- Form containers (FRAME, GROUP) that contain multiple input elements
- Named like "Form", "Login", "Signup", "Contact"
- MUST be the main form container that groups multiple inputs
- Include: forms with multiple fields, login forms, contact forms
- Exclude: individual input elements (those are detected separately)""",
                "output_format": """{
  "<node_id>": {"tag": "form"},
  "<node_id>": {"tag": "form"}
}""",
            },
        }

        return configs.get(detection_type, configs["text_input"])

    def _filter_nodes_for_detection(
        self, nodes: List[Dict[str, Any]], detection_type: str
    ) -> List[Dict[str, Any]]:
        """
        Filter nodes based on detection type

        Args:
            nodes: List of all nodes
            detection_type: Type of detection

        Returns:
            Filtered list of nodes
        """
        filtered_nodes = []

        for node in nodes:
            node_id = node.get("node_id", "N/A")
            node_type = node.get("type", "N/A")
            node_name = node.get("name", "N/A")
            parent_id = node.get("parent_id", None)

            # Skip nodes that are clearly not candidates
            if node_type in ["TEXT", "VECTOR", "LINE", "ELLIPSE", "STAR"]:
                continue

            # Apply detection-specific filtering
            if detection_type == "button":
                if self._should_skip_for_button(node_name, parent_id):
                    continue
            elif detection_type == "form":
                if self._should_skip_for_form(node_name, parent_id):
                    continue

            filtered_nodes.append(node)

        return filtered_nodes

    def _should_skip_for_button(self, node_name: str, parent_id: Optional[str]) -> bool:
        """Check if node should be skipped for button detection"""
        if parent_id and any(
            skip_name in node_name.lower()
            for skip_name in ["icon", "label", "text", "inner", "base"]
        ):
            return True

        return False

    def _should_skip_for_form(self, node_name: str, parent_id: Optional[str]) -> bool:
        """Check if node should be skipped for form detection"""
        if parent_id and any(
            skip_name in node_name.lower()
            for skip_name in [
                "icon",
                "label",
                "help",
                "placeholder",
                "button",
                "inner",
                "text",
                "base",
                "fill",
            ]
        ):
            return True

        return False

    def _write_prompt_to_file(self, prompt: str, detection_type: str):
        """Write prompt to file for debugging"""
        try:
            filename = f"gemini_prompt_{detection_type}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(prompt)
            logger.info(f"Full Gemini prompt written to {filename}")
        except Exception as e:
            logger.error(f"Error writing prompt to file: {e}")

    def _parse_response(
        self, response_text: str, detection_type: str
    ) -> Dict[str, Dict[str, str]]:
        """
        Parse Gemini response and extract detection results

        Args:
            response_text: Raw response text from Gemini
            detection_type: Type of detection performed

        Returns:
            Dictionary with node_id as keys and detection results as values
        """
        logger.info(f"Parsing Gemini response for {detection_type} detection...")

        try:
            # Try to extract JSON from the response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                logger.warning(f"No JSON found in Gemini response for {detection_type}")
                return {}

            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)

            # Validate the result structure
            if not isinstance(result, dict):
                logger.warning(f"Invalid result structure for {detection_type}")
                return {}

            # Clean the result by extracting clean node IDs and simplifying the structure
            cleaned_result = {}
            for node_id, node_info in result.items():
                # Extract clean node ID from verbose format (e.g., "5:158|INSTANCE|Button Secondary" -> "5:158")
                clean_node_id = node_id.split("|")[0] if "|" in node_id else node_id

                if isinstance(node_info, dict):
                    tag = node_info.get("tag", "unknown")
                    cleaned_result[clean_node_id] = {"tag": tag}
                else:
                    # If node_info is not a dict, convert it
                    tag = str(node_info)
                    cleaned_result[clean_node_id] = {"tag": tag}

            logger.info(
                f"Successfully parsed {len(cleaned_result)} results for {detection_type}"
            )
            return cleaned_result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for {detection_type}: {e}")
            logger.debug(f"Raw response: {response_text}")
            return {}
        except Exception as e:
            logger.error(f"Error parsing response for {detection_type}: {e}")
            return {}

    async def analyze_nodes_combined(
        self, nodes: List[Dict[str, Any]], detection_types: List[str] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Analyze nodes using a single combined prompt for all detection types

        Args:
            nodes: List of node dictionaries with complete structure
            detection_types: List of detection types to analyze (default: ["text_input", "button", "link", "select"])

        Returns:
            Dictionary with node_id as keys and detection results as values
        """
        if detection_types is None:
            detection_types = ["text_input", "button", "link", "select"]

        logger.info(
            f"Starting combined Gemini analysis for {len(detection_types)} detection types: {detection_types}"
        )

        # Create combined prompt for all detection types
        prompt = self.combined_prompt_creator.create_prompt(nodes, detection_types)

        # Write prompt to file for debugging
        self._write_prompt_to_file(prompt, "combined")

        try:
            # Send request to Gemini asynchronously
            logger.info(f"Sending combined request to Gemini API...")
            response = await self._generate_content_async(prompt)
            logger.info(
                f"Gemini API response received successfully for combined analysis"
            )

            # Parse response
            result = self._parse_combined_response(response, detection_types)
            logger.info(
                f"Successfully parsed {len(result)} results from combined Gemini response"
            )

            # Post-process and log results grouped by tag type
            self._log_results_by_tag_type(result)

            return result

        except Exception as e:
            logger.error(f"Error using Gemini for combined detection: {e}")
            logger.info(f"Gemini combined detection failed - returning empty result")
            return {}

    def _parse_combined_response(
        self, response_text: str, detection_types: List[str]
    ) -> Dict[str, Dict[str, str]]:
        """
        Parse combined Gemini response and extract detection results

        Args:
            response_text: Raw response text from Gemini
            detection_types: List of detection types that were analyzed

        Returns:
            Dictionary with node_id as keys and detection results as values
        """
        logger.info(f"Parsing combined Gemini response...")

        try:
            # Try to extract JSON from the response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                logger.warning(f"No JSON found in combined Gemini response")
                return {}

            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)

            # Validate the result structure
            if not isinstance(result, dict):
                logger.warning(f"Invalid result structure for combined detection")
                return {}

            # Simplify the result to only include tag and extract clean node IDs
            simplified_result = {}
            for node_id, node_info in result.items():
                # Extract clean node ID from verbose format (e.g., "5:158|INSTANCE|Button Secondary" -> "5:158")
                clean_node_id = node_id.split("|")[0] if "|" in node_id else node_id

                if isinstance(node_info, dict):
                    tag = node_info.get("tag", "unknown")
                    simplified_result[clean_node_id] = {"tag": tag}
                else:
                    # If node_info is not a dict, convert it
                    tag = str(node_info)
                    simplified_result[clean_node_id] = {"tag": tag}

            logger.info(
                f"Successfully parsed {len(simplified_result)} results from combined detection"
            )
            return simplified_result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for combined detection: {e}")
            logger.debug(f"Raw response: {response_text}")
            return {}
        except Exception as e:
            logger.error(f"Error parsing combined response: {e}")
            return {}

    def _log_results_by_tag_type(self, result: Dict[str, Dict[str, str]]):
        """
        Post-process and log results grouped by tag type

        Args:
            result: Dictionary with node_id as keys and {"tag": "value"} as values
        """
        if not result:
            logger.info("No elements detected - no results to group by tag type")
            return

        # Group node IDs by tag type
        tag_groups = {"input": [], "button": [], "select": [], "link": []}

        for node_id, node_info in result.items():
            tag = node_info.get("tag", "unknown")
            if tag in tag_groups:
                tag_groups[tag].append(node_id)
            else:
                # Handle unknown tags
                if "unknown" not in tag_groups:
                    tag_groups["unknown"] = []
                tag_groups["unknown"].append(node_id)

        # Log the grouped results
        logger.info("=" * 60)
        logger.info("POST-PROCESSED RESULTS GROUPED BY TAG TYPE:")
        logger.info("=" * 60)

        total_elements = len(result)
        logger.info(f"Total elements detected: {total_elements}")
        logger.info("")

        for tag, node_ids in tag_groups.items():
            if node_ids:  # Only log groups that have elements
                logger.info(f"🎯 {tag.upper()} ELEMENTS ({len(node_ids)} nodes):")
                logger.info(f"   Node IDs: {node_ids}")
                logger.info("")

        # Log summary
        logger.info("📊 SUMMARY:")
        for tag, node_ids in tag_groups.items():
            if node_ids:
                percentage = (len(node_ids) / total_elements) * 100
                logger.info(f"   {tag}: {len(node_ids)} elements ({percentage:.1f}%)")

        logger.info("=" * 60)

    def _map_tag_to_detection_type(self, tag: str) -> str:
        """
        Map a detected tag to its corresponding detection type

        Args:
            tag: The detected tag (input, button, select, link)

        Returns:
            The corresponding detection type
        """
        tag_mapping = {
            "input": "text_input",
            "button": "button",
            "select": "select",
            "link": "link",
        }
        return tag_mapping.get(tag, "unknown")
