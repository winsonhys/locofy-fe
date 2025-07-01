#!/usr/bin/env python3
"""
Gemini Function Caller - Uses Gemini's function calling to interact with Figma API
Instead of passing tokens directly, Gemini calls functions to get Figma data
"""

import json
import logging
import asyncio
import aiohttp
import google.generativeai as genai
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from config import GEMINI_API_KEY, GEMINI_MODEL_NAME, FIGMA_ACCESS_TOKEN
from input_detection_prompts import FunctionCallingDetectionPromptCreator

logger = logging.getLogger(__name__)


@dataclass
class FigmaNode:
    """Represents a Figma node with its properties"""

    id: str
    name: str
    type: str
    children: Optional[List["FigmaNode"]] = None
    visible: bool = True
    locked: bool = False
    data: Dict[str, Any] = None
    parent_id: Optional[str] = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}


class FigmaAPIClient:
    """Handles direct Figma API calls - used by function calling"""

    def __init__(self):
        self.base_url = "https://api.figma.com/v1"
        self.headers = {
            "X-Figma-Token": FIGMA_ACCESS_TOKEN,
            "Content-Type": "application/json",
        }

    async def get_file(self, file_key: str) -> Dict[str, Any]:
        """Get Figma file data"""
        url = f"{self.base_url}/files/{file_key}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                return await response.json()

    async def get_node_by_id(self, file_key: str, node_id: str) -> Dict[str, Any]:
        """Get specific node by ID"""
        url = f"{self.base_url}/files/{file_key}/nodes?ids={node_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                return await response.json()


class GeminiFunctionCaller:
    """Uses Gemini's function calling to analyze Figma designs"""

    def __init__(self):
        self._configure_gemini()
        self.figma_client = FigmaAPIClient()
        self.function_definitions = self._create_function_definitions()
        self.prompt_creator = FunctionCallingDetectionPromptCreator()

    def _configure_gemini(self):
        """Configure Gemini API"""
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        self.api_key = GEMINI_API_KEY

    def _create_function_definitions(self) -> List[Dict[str, Any]]:
        """Create function definitions for Gemini to call"""
        return [
            {
                "name": "get_figma_file",
                "description": "Get the complete structure of a Figma file",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "file_key": {
                            "type": "STRING",
                            "description": "The Figma file key from the URL",
                        }
                    },
                    "required": ["file_key"],
                },
            },
            {
                "name": "get_figma_node",
                "description": "Get a specific node and its children from a Figma file",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "file_key": {
                            "type": "STRING",
                            "description": "The Figma file key from the URL",
                        },
                        "node_id": {
                            "type": "STRING",
                            "description": "The ID of the node to retrieve",
                        },
                    },
                    "required": ["file_key", "node_id"],
                },
            },
            {
                "name": "search_nodes_by_type",
                "description": "Search for nodes of a specific type in the Figma file",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "file_key": {
                            "type": "STRING",
                            "description": "The Figma file key from the URL",
                        },
                        "node_type": {
                            "type": "STRING",
                            "description": "Type of nodes to search for (e.g., TEXT, RECTANGLE, FRAME)",
                        },
                    },
                    "required": ["file_key", "node_type"],
                },
            },
            {
                "name": "search_nodes_by_name",
                "description": "Search for nodes by name pattern in the Figma file",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "file_key": {
                            "type": "STRING",
                            "description": "The Figma file key from the URL",
                        },
                        "name_pattern": {
                            "type": "STRING",
                            "description": "Name pattern to search for",
                        },
                    },
                    "required": ["file_key", "name_pattern"],
                },
            },
        ]

    async def get_figma_file(self, file_key: str) -> Dict[str, Any]:
        """Function that Gemini can call to get Figma file data"""
        try:
            logger.info(f"Getting Figma file: {file_key}")
            result = await self.figma_client.get_file(file_key)
            logger.info(
                f"Successfully retrieved Figma file with {len(result.get('document', {}).get('children', []))} top-level nodes"
            )
            return result
        except Exception as e:
            logger.error(f"Error getting Figma file: {e}")
            return {"error": str(e)}

    async def get_figma_node(self, file_key: str, node_id: str) -> Dict[str, Any]:
        """Function that Gemini can call to get specific node data"""
        try:
            logger.info(f"Getting Figma node: {node_id} from file: {file_key}")
            result = await self.figma_client.get_node_by_id(file_key, node_id)
            logger.info(f"Successfully retrieved node data")
            return result
        except Exception as e:
            logger.error(f"Error getting Figma node: {e}")
            return {"error": str(e)}

    async def search_nodes_by_type(
        self, file_key: str, node_type: str
    ) -> Dict[str, Any]:
        """Function that Gemini can call to search nodes by type"""
        try:
            logger.info(f"Searching for nodes of type: {node_type} in file: {file_key}")
            # Get the full file first
            file_data = await self.figma_client.get_file(file_key)

            # Search for nodes of the specified type
            matching_nodes = []
            self._search_nodes_recursive(
                file_data.get("document", {}), node_type, matching_nodes
            )

            result = {
                "file_key": file_key,
                "node_type": node_type,
                "matching_nodes": matching_nodes,
                "count": len(matching_nodes),
            }

            logger.info(f"Found {len(matching_nodes)} nodes of type {node_type}")
            return result
        except Exception as e:
            logger.error(f"Error searching nodes by type: {e}")
            return {"error": str(e)}

    async def search_nodes_by_name(
        self, file_key: str, name_pattern: str
    ) -> Dict[str, Any]:
        """Function that Gemini can call to search nodes by name"""
        try:
            logger.info(
                f"Searching for nodes with name pattern: {name_pattern} in file: {file_key}"
            )
            # Get the full file first
            file_data = await self.figma_client.get_file(file_key)

            # Search for nodes matching the name pattern
            matching_nodes = []
            self._search_nodes_by_name_recursive(
                file_data.get("document", {}), name_pattern, matching_nodes
            )

            result = {
                "file_key": file_key,
                "name_pattern": name_pattern,
                "matching_nodes": matching_nodes,
                "count": len(matching_nodes),
            }

            logger.info(
                f"Found {len(matching_nodes)} nodes matching name pattern {name_pattern}"
            )
            return result
        except Exception as e:
            logger.error(f"Error searching nodes by name: {e}")
            return {"error": str(e)}

    def _search_nodes_recursive(
        self,
        node: Dict[str, Any],
        target_type: str,
        matching_nodes: List[Dict[str, Any]],
    ):
        """Recursively search for nodes of a specific type"""
        if node.get("type") == target_type:
            matching_nodes.append(
                {
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "type": node.get("type"),
                    "visible": node.get("visible", True),
                }
            )

        # Search children
        for child in node.get("children", []):
            self._search_nodes_recursive(child, target_type, matching_nodes)

    def _search_nodes_by_name_recursive(
        self,
        node: Dict[str, Any],
        name_pattern: str,
        matching_nodes: List[Dict[str, Any]],
    ):
        """Recursively search for nodes by name pattern"""
        node_name = node.get("name", "").lower()
        if name_pattern.lower() in node_name:
            matching_nodes.append(
                {
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "type": node.get("type"),
                    "visible": node.get("visible", True),
                }
            )

        # Search children
        for child in node.get("children", []):
            self._search_nodes_by_name_recursive(child, name_pattern, matching_nodes)

    async def analyze_nodes_with_function_calling_backup(
        self,
        nodes: List[Dict[str, Any]],
        file_key: str,
        node_id: Optional[str] = None,
        analysis_type: str = "all",
    ) -> Dict[str, Any]:
        """
        Analyze nodes using Gemini with function calling as backup for additional context

        Args:
            nodes: List of node dictionaries from DFS
            file_key: Figma file key (for function calling if needed)
            node_id: Optional specific node ID (for function calling if needed)
            analysis_type: Type of analysis to perform

        Returns:
            Analysis results from Gemini
        """
        # Create the prompt using the FunctionCallingDetectionPromptCreator
        prompt = self.prompt_creator.create_prompt(
            nodes=nodes,
            file_key=file_key,
            node_id=node_id,
            detection_types=[analysis_type] if analysis_type != "all" else None,
        )

        try:
            # Create the model with function calling capabilities
            model = genai.GenerativeModel(
                GEMINI_MODEL_NAME, tools=self.function_definitions
            )

            # Start the conversation
            chat = model.start_chat()

            # Send the initial prompt with nodes
            response = chat.send_message(prompt)

            # Handle function calls if needed
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]

                if hasattr(candidate, "content") and candidate.content.parts:
                    part = candidate.content.parts[0]

                    # Check if this is a function call
                    if hasattr(part, "function_call") and part.function_call:
                        function_call = part.function_call
                        logger.info(f"Function call detected: {function_call.name}")

                        # Execute the function call
                        if function_call.name == "get_figma_file":
                            result = await self.get_figma_file(
                                function_call.args["file_key"]
                            )
                        elif function_call.name == "get_figma_node":
                            result = await self.get_figma_node(
                                function_call.args["file_key"],
                                function_call.args["node_id"],
                            )
                        elif function_call.name == "search_nodes_by_type":
                            result = await self.search_nodes_by_type(
                                function_call.args["file_key"],
                                function_call.args["node_type"],
                            )
                        elif function_call.name == "search_nodes_by_name":
                            result = await self.search_nodes_by_name(
                                function_call.args["file_key"],
                                function_call.args["name_pattern"],
                            )
                        else:
                            result = {
                                "error": f"Unknown function: {function_call.name}"
                            }

                        # Send the function result back to Gemini as a new message
                        function_response = chat.send_message(
                            f"Function {function_call.name} returned: {json.dumps(result, indent=2)}"
                        )

                        return {
                            "initial_response": "Function call detected",
                            "function_result": result,
                            "final_response": function_response.text,
                        }
                    else:
                        # No function call, just return the response
                        logger.info(
                            "No function call detected, returning direct response"
                        )
                        return {"response": response.text, "function_calls": []}

            # If no function call was made, return the direct response
            return {"response": response.text, "function_calls": []}

        except Exception as e:
            logger.error(f"Error in function calling analysis: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}

    async def get_text_inputs_with_function_calling(
        self, file_key: str, node_id: Optional[str] = None
    ) -> Dict[str, Dict[str, str]]:
        """Get text input elements using function calling"""
        result = await self.analyze_nodes_with_function_calling_backup(
            [], file_key, node_id, "inputs"
        )
        return self._parse_function_calling_result(result, "input")

    async def get_buttons_with_function_calling(
        self, file_key: str, node_id: Optional[str] = None
    ) -> Dict[str, Dict[str, str]]:
        """Get button elements using function calling"""
        result = await self.analyze_nodes_with_function_calling_backup(
            [], file_key, node_id, "buttons"
        )
        return self._parse_function_calling_result(result, "button")

    async def get_links_with_function_calling(
        self, file_key: str, node_id: Optional[str] = None
    ) -> Dict[str, Dict[str, str]]:
        """Get link elements using function calling"""
        result = await self.analyze_nodes_with_function_calling_backup(
            [], file_key, node_id, "links"
        )
        return self._parse_function_calling_result(result, "link")

    async def get_all_elements_with_function_calling(
        self, file_key: str, node_id: Optional[str] = None
    ) -> Dict[str, Dict[str, str]]:
        """Get all interactive elements using function calling"""
        result = await self.analyze_nodes_with_function_calling_backup(
            [], file_key, node_id, "all"
        )
        return self._parse_function_calling_result(result, "all")

    def _parse_function_calling_result(
        self, result: Dict[str, Any], element_type: str
    ) -> Dict[str, Dict[str, str]]:
        """Parse the function calling result into the expected format"""
        if "error" in result:
            logger.error(f"Function calling error: {result['error']}")
            return {}

        parsed_result = {}

        # Try to extract the final response and parse it
        final_response = result.get("final_response", "")
        if final_response:
            try:
                # Look for JSON in the response
                import re

                json_match = re.search(r"\{.*\}", final_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    parsed_data = json.loads(json_str)

                    # Extract interactive elements
                    interactive_elements = parsed_data.get("interactive_elements", [])
                    for element in interactive_elements:
                        node_id = element.get("node_id")
                        if node_id:
                            parsed_result[node_id] = {
                                "tag": element.get("type", "unknown"),
                                "name": element.get("name", ""),
                                "description": element.get("description", ""),
                                "confidence": element.get("confidence", "medium"),
                            }
            except Exception as e:
                logger.error(f"Error parsing function calling result: {e}")

        return parsed_result


# Example usage
async def example_usage():
    """Example of using the new function calling approach"""
    print("🚀 Gemini Function Calling Example")
    print("=" * 50)

    caller = GeminiFunctionCaller()

    try:
        # Test direct function calls first
        print("🔍 Testing direct function calls...")

        # Test getting file data
        print("📁 Getting file data...")
        file_data = await caller.get_figma_file("NnmJQ6LgSUJn08LLXkylSp")
        if "error" not in file_data:
            print(
                f"✅ File retrieved successfully with {len(file_data.get('document', {}).get('children', []))} top-level nodes"
            )
        else:
            print(f"❌ Error getting file: {file_data['error']}")

        # Test getting node data
        print("🎯 Getting node data...")
        node_data = await caller.get_figma_node("NnmJQ6LgSUJn08LLXkylSp", "1:2")
        if "error" not in node_data:
            print("✅ Node data retrieved successfully")
        else:
            print(f"❌ Error getting node: {node_data['error']}")

        # Test searching by type
        print("🔍 Searching for TEXT nodes...")
        text_nodes = await caller.search_nodes_by_type("NnmJQ6LgSUJn08LLXkylSp", "TEXT")
        if "error" not in text_nodes:
            print(f"✅ Found {text_nodes.get('count', 0)} TEXT nodes")
        else:
            print(f"❌ Error searching by type: {text_nodes['error']}")

        # Now test the full function calling analysis
        print("\n🔍 Testing full function calling analysis...")
        result = await caller.analyze_nodes_with_function_calling_backup(
            [], "NnmJQ6LgSUJn08LLXkylSp", "1:2", "all"
        )

        print("📊 Analysis Result:")
        print(json.dumps(result, indent=2))

        # Try getting specific elements
        print("\n🔍 Getting text inputs...")
        inputs = await caller.get_text_inputs_with_function_calling(
            file_key="NnmJQ6LgSUJn08LLXkylSp", node_id="1:2"
        )
        print(f"Found {len(inputs)} text inputs")

    except Exception as e:
        logger.error(f"Error in example: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(example_usage())
