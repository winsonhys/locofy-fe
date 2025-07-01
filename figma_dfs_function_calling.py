#!/usr/bin/env python3
"""
Modified FigmaDFS that uses Gemini function calling instead of direct token access
This version delegates Figma API calls to Gemini through function calling
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from gemini_function_caller import GeminiFunctionCaller
from config import (
    GEMINI_API_KEY,
    LOG_LEVEL,
    LOG_FORMAT,
    FIGMA_FILE_KEY,
    START_NODE_ID,
)

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Figma node types"""

    DOCUMENT = "DOCUMENT"
    CANVAS = "CANVAS"
    FRAME = "FRAME"
    GROUP = "GROUP"
    VECTOR = "VECTOR"
    BOOLEAN_OPERATION = "BOOLEAN_OPERATION"
    STAR = "STAR"
    LINE = "LINE"
    ELLIPSE = "ELLIPSE"
    REGULAR_POLYGON = "REGULAR_POLYGON"
    RECTANGLE = "RECTANGLE"
    TABLE = "TABLE"
    TABLE_CELL = "TABLE_CELL"
    TEXT = "TEXT"
    SLICE = "SLICE"
    INSTANCE = "INSTANCE"
    COMPONENT = "COMPONENT"
    COMPONENT_SET = "COMPONENT_SET"


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


class FigmaDFSFunctionCalling:
    """
    Modified FigmaDFS that uses Gemini function calling instead of direct token access
    This approach is more secure as it doesn't expose tokens directly to external APIs
    """

    def __init__(self):
        """
        Initialize FigmaDFS with function calling approach
        No token needed in constructor - Gemini handles API access
        """
        self.gemini_caller = GeminiFunctionCaller()
        logger.info("FigmaDFS initialized with function calling approach")

    async def get_file(self, file_key: str) -> Dict[str, Any]:
        """
        Get Figma file data through Gemini function calling

        Args:
            file_key: Figma file key from URL

        Returns:
            File data as dictionary
        """
        logger.info(f"Getting Figma file {file_key} through function calling")
        return await self.gemini_caller.get_figma_file(file_key)

    async def get_node_by_id(self, file_key: str, node_id: str) -> Dict[str, Any]:
        """
        Get specific node by ID through Gemini function calling

        Args:
            file_key: Figma file key
            node_id: Node ID to retrieve

        Returns:
            Node data as dictionary
        """
        logger.info(
            f"Getting node {node_id} from file {file_key} through function calling"
        )
        return await self.gemini_caller.get_figma_node(file_key, node_id)

    def parse_node(self, node_data: Dict[str, Any]) -> FigmaNode:
        """
        Parse raw node data into FigmaNode object

        Args:
            node_data: Raw node data from Figma API

        Returns:
            FigmaNode object
        """
        node = FigmaNode(
            id=node_data.get("id", ""),
            name=node_data.get("name", ""),
            type=node_data.get("type", ""),
            visible=node_data.get("visible", True),
            locked=node_data.get("locked", False),
            data=node_data,
        )

        # Parse children if they exist
        if "children" in node_data and node_data["children"]:
            node.children = [self.parse_node(child) for child in node_data["children"]]

        return node

    async def depth_first_search(
        self,
        file_key: str,
        node_id: str,
        visit_callback: Optional[Callable[[FigmaNode, int], bool]] = None,
        max_depth: Optional[int] = None,
    ) -> List[FigmaNode]:
        """
        Perform depth-first search on Figma nodes using function calling

        Args:
            file_key: Figma file key
            node_id: Starting node ID for DFS
            visit_callback: Optional callback function(node, depth) -> bool
                          Return True to continue searching, False to stop
            max_depth: Maximum depth to search (None for unlimited)

        Returns:
            List of visited nodes in DFS order
        """
        logger.info(f"Starting DFS on node {node_id} in file {file_key}")

        # Get the node data through function calling
        node_data = await self.get_node_by_id(file_key, node_id)

        if "error" in node_data:
            logger.error(f"Error getting node data: {node_data['error']}")
            return []

        # Extract the actual node from the response
        actual_node_data = (
            node_data.get("nodes", {}).get(node_id, {}).get("document", {})
        )
        if not actual_node_data:
            logger.error("No node data found in response")
            return []

        # Parse the root node
        root_node = self.parse_node(actual_node_data)

        # Perform DFS
        visited_nodes = []

        def dfs_recursive(
            node: FigmaNode, depth: int = 0, parent_id: Optional[str] = None
        ) -> bool:
            # Check max depth
            if max_depth is not None and depth > max_depth:
                return True

            # Set parent_id for this node
            node.parent_id = parent_id

            # Add node to visited list
            visited_nodes.append(node)

            # Call visit callback if provided
            if visit_callback:
                should_continue = visit_callback(node, depth)
                if not should_continue:
                    return False

            # Recursively visit children
            if node.children:
                for child in node.children:
                    if not dfs_recursive(child, depth + 1, node.id):
                        return False

            return True

        dfs_recursive(root_node)
        logger.info(f"DFS completed: visited {len(visited_nodes)} nodes")
        return visited_nodes

    async def search_by_type(
        self, file_key: str, node_id: str, target_type: str
    ) -> List[FigmaNode]:
        """
        Search for nodes of a specific type using function calling

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            target_type: Type of nodes to find

        Returns:
            List of nodes matching the type
        """
        logger.info(f"Searching for {target_type} nodes in file {file_key}")

        # Use Gemini's function calling to search by type
        search_result = await self.gemini_caller.search_nodes_by_type(
            file_key, target_type
        )

        if "error" in search_result:
            logger.error(f"Error searching by type: {search_result['error']}")
            return []

        # Convert the search results to FigmaNode objects
        matching_nodes = []
        for node_data in search_result.get("matching_nodes", []):
            node = FigmaNode(
                id=node_data.get("id", ""),
                name=node_data.get("name", ""),
                type=node_data.get("type", ""),
                visible=node_data.get("visible", True),
                data=node_data,
            )
            matching_nodes.append(node)

        logger.info(f"Found {len(matching_nodes)} nodes of type {target_type}")
        return matching_nodes

    async def search_by_name(
        self,
        file_key: str,
        node_id: str,
        name_pattern: str,
        case_sensitive: bool = False,
    ) -> List[FigmaNode]:
        """
        Search for nodes by name pattern using function calling

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            name_pattern: Name pattern to search for
            case_sensitive: Whether search should be case sensitive

        Returns:
            List of nodes matching the name pattern
        """
        logger.info(
            f"Searching for nodes with name pattern '{name_pattern}' in file {file_key}"
        )

        # Use Gemini's function calling to search by name
        search_result = await self.gemini_caller.search_nodes_by_name(
            file_key, name_pattern
        )

        if "error" in search_result:
            logger.error(f"Error searching by name: {search_result['error']}")
            return []

        # Convert the search results to FigmaNode objects
        matching_nodes = []
        for node_data in search_result.get("matching_nodes", []):
            node = FigmaNode(
                id=node_data.get("id", ""),
                name=node_data.get("name", ""),
                type=node_data.get("type", ""),
                visible=node_data.get("visible", True),
                data=node_data,
            )
            matching_nodes.append(node)

        logger.info(
            f"Found {len(matching_nodes)} nodes matching name pattern '{name_pattern}'"
        )
        return matching_nodes

    async def get_text_input_nodes_from_figma_with_gemini(
        self, file_key: str, node_ids: List[str]
    ) -> Dict[str, Dict[str, str]]:
        """
        Get text input nodes using Gemini function calling

        Args:
            file_key: Figma file key
            node_ids: List of node IDs to analyze

        Returns:
            Dictionary with node_id as keys and detection results as values
        """
        logger.info(
            f"Getting text input nodes for {len(node_ids)} nodes using function calling"
        )

        # Use the first node ID as the starting point for analysis
        if node_ids:
            result = await self.gemini_caller.get_text_inputs_with_function_calling(
                file_key, node_ids[0]
            )
            return result
        else:
            logger.warning("No node IDs provided for text input analysis")
            return {}

    async def get_all_text_inputs_from_figma_with_gemini(
        self, file_key: str, node_id: str, max_depth: Optional[int] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Get all text inputs from Figma using Gemini function calling

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            max_depth: Maximum depth to search

        Returns:
            Dictionary with node_id as keys and detection results as values
        """
        logger.info(
            f"Getting all text inputs from node {node_id} using function calling"
        )

        result = await self.gemini_caller.get_text_inputs_with_function_calling(
            file_key, node_id
        )
        return result

    async def get_all_inputs_buttons_and_links_from_figma_with_gemini(
        self, file_key: str, node_id: str, max_depth: Optional[int] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Get all inputs, buttons, and links from Figma using Gemini function calling

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            max_depth: Maximum depth to search

        Returns:
            Dictionary with node_id as keys and detection results as values
        """
        logger.info(
            f"Getting all inputs, buttons, and links from node {node_id} using function calling"
        )

        # Use the combined analysis approach
        result = await self.gemini_caller.analyze_nodes_with_function_calling_backup(
            [], file_key, node_id, "all"
        )

        # Parse the result into the expected format
        return self.gemini_caller._parse_function_calling_result(result, "all")

    async def get_all_elements_from_figma_with_gemini_combined(
        self, file_key: str, node_id: str, max_depth: Optional[int] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Get all interactive elements using hybrid approach: DFS + function calling backup

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            max_depth: Maximum depth to search

        Returns:
            Dictionary with node_id as keys and detection results as values
        """
        logger.info(
            f"Getting all interactive elements from node {node_id} using hybrid approach"
        )

        try:
            # Step 1: Follow normal DFS workflow to get nodes
            logger.info("Step 1: Performing DFS to get nodes...")
            nodes = await self.depth_first_search(
                file_key, node_id, max_depth=max_depth
            )

            # Convert FigmaNode objects to dictionaries for Gemini
            node_dicts = []
            for node in nodes:
                node_dict = {
                    "id": node.id,
                    "name": node.name,
                    "type": node.type,
                    "visible": node.visible,
                    "locked": node.locked,
                    "parent_id": node.parent_id,
                    "data": node.data,
                }
                node_dicts.append(node_dict)

            logger.info(f"DFS completed: collected {len(node_dicts)} nodes")

            # Step 2: Send nodes to Gemini with function calling as backup
            logger.info(
                "Step 2: Sending nodes to Gemini with function calling backup..."
            )
            result = (
                await self.gemini_caller.analyze_nodes_with_function_calling_backup(
                    node_dicts, file_key, node_id, "all"
                )
            )

            # Step 3: Parse the result
            return self.gemini_caller._parse_function_calling_result(result, "all")

        except Exception as e:
            logger.error(f"Error in hybrid analysis: {e}")
            return {}

    async def print_node_tree(
        self, file_key: str, node_id: str, max_depth: Optional[int] = None
    ):
        """
        Print the node tree structure using function calling

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            max_depth: Maximum depth to print
        """
        logger.info(f"Printing node tree for node {node_id} in file {file_key}")

        def print_callback(node: FigmaNode, depth: int) -> bool:
            indent = "  " * depth
            print(f"{indent}- {node.name} ({node.type}) [ID: {node.id}]")
            return True

        await self.depth_first_search(file_key, node_id, print_callback, max_depth)


# Example usage
async def example_usage():
    """Example usage of the function calling approach"""
    print("🚀 FigmaDFS Function Calling Example")
    print("=" * 50)

    # Initialize without passing tokens
    figma_dfs = FigmaDFSFunctionCalling()

    try:
        # Get file data
        print("📁 Getting file data...")
        file_data = await figma_dfs.get_file(FIGMA_FILE_KEY)
        print(f"File retrieved successfully")

        # Get specific node
        print("🎯 Getting specific node...")
        node_data = await figma_dfs.get_node_by_id(FIGMA_FILE_KEY, START_NODE_ID)
        print(f"Node retrieved successfully")

        # Search by type
        print("🔍 Searching for TEXT nodes...")
        text_nodes = await figma_dfs.search_by_type(
            FIGMA_FILE_KEY, START_NODE_ID, "TEXT"
        )
        print(f"Found {len(text_nodes)} TEXT nodes")

        # Search by name
        print("🔍 Searching for nodes with 'button' in name...")
        button_nodes = await figma_dfs.search_by_name(
            FIGMA_FILE_KEY, START_NODE_ID, "button"
        )
        print(f"Found {len(button_nodes)} nodes with 'button' in name")

        # Get text inputs
        print("📝 Getting text inputs...")
        inputs = await figma_dfs.get_all_text_inputs_from_figma_with_gemini(
            FIGMA_FILE_KEY, START_NODE_ID
        )
        print(f"Found {len(inputs)} text inputs")

        # Get all elements
        print("🎯 Getting all interactive elements...")
        all_elements = await figma_dfs.get_all_elements_from_figma_with_gemini_combined(
            FIGMA_FILE_KEY, START_NODE_ID
        )
        print(f"Found {len(all_elements)} total interactive elements")

        print("✅ Function calling approach completed successfully!")
        print("   No tokens were passed directly to external APIs.")

    except Exception as e:
        logger.error(f"Error in example: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(example_usage())
