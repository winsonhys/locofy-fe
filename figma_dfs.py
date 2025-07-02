import requests
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import logging
from gemini_analyzer import GeminiAnalyzer
from config import (
    LOG_LEVEL,
    LOG_FORMAT,
)

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
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


class FigmaDFS:
    """Minimal FigmaDFS implementation for agentic workflow"""

    def __init__(self, access_token: str):
        """
        Initialize Figma DFS with access token

        Args:
            access_token: Figma personal access token
        """
        self.access_token = access_token
        self.base_url = "https://api.figma.com/v1"
        self.headers = {
            "X-Figma-Token": access_token,
            "Content-Type": "application/json",
        }
        self.gemini_analyzer = GeminiAnalyzer()

    def get_node_by_id(self, file_key: str, node_id: str) -> Dict[str, Any]:
        """
        Get specific node by ID

        Args:
            file_key: Figma file key
            node_id: Node ID to retrieve

        Returns:
            Node data as dictionary
        """
        url = f"{self.base_url}/files/{file_key}/nodes?ids={node_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

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

    def depth_first_search(
        self,
        root_node: FigmaNode,
        visit_callback: Optional[Callable[[FigmaNode, int], bool]] = None,
        max_depth: Optional[int] = None,
    ) -> List[FigmaNode]:
        """
        Perform depth-first search on Figma nodes

        Args:
            root_node: Starting node for DFS
            visit_callback: Optional callback function(node, depth) -> bool
                          Return True to continue searching, False to stop
            max_depth: Maximum depth to search (None for unlimited)

        Returns:
            List of visited nodes in DFS order
        """
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
        return visited_nodes

    def _extract_base_node_id(self, node_id: str) -> str:
        """
        Extract the base node ID from a Figma node ID string.
        Figma node IDs can include component and variant information separated by semicolons,
        and instance nodes may have an "I" prefix.
        This method returns only the base node ID (before the first semicolon and without "I" prefix).

        Args:
            node_id: Full node ID string (e.g., "290:7308;216:4169" or "I5:159")

        Returns:
            Base node ID string (e.g., "290:7308" or "5:159")
        """
        # Remove "I" prefix if present
        if node_id.startswith("I"):
            node_id = node_id[1:]

        # Remove semicolon-separated parts if present
        return node_id.split(";")[0] if ";" in node_id else node_id

    def depth_first_search_from_node_id(
        self,
        file_key: str,
        node_id: str,
        visit_callback: Optional[Callable[[FigmaNode, int], bool]] = None,
        max_depth: Optional[int] = None,
    ) -> List[FigmaNode]:
        """
        Perform depth-first search starting from a specific node ID

        Args:
            file_key: Figma file key
            node_id: Starting node ID for DFS
            visit_callback: Optional callback function(node, depth) -> bool
                          Return True to continue searching, False to stop
            max_depth: Maximum depth to search (None for unlimited)

        Returns:
            List of visited nodes in DFS order

        Raises:
            requests.exceptions.RequestException: If API request fails
            KeyError: If node_id is not found in the file
        """
        logger.info(f"Starting DFS from node ID: {node_id}")
        logger.info(f"File key: {file_key}")
        logger.info(f"Max depth: {max_depth}")

        # Get the specific node data
        logger.debug(f"Fetching node data for node ID: {node_id}")
        node_data = self.get_node_by_id(file_key, node_id)
        nodes = node_data.get("nodes", {})
        logger.debug(f"Received {len(nodes)} nodes from API")

        if node_id not in nodes:
            logger.error(f"Node ID '{node_id}' not found in file")
            raise KeyError(f"Node ID '{node_id}' not found in file")

        # Parse the node
        logger.debug("Parsing node data...")
        node_info = nodes[node_id]
        document = node_info.get("document", {})
        root_node = self.parse_node(document)
        logger.info(
            f"Successfully parsed root node: {root_node.name} ({root_node.type})"
        )

        # Perform DFS from this node
        logger.info("Starting DFS traversal...")
        result = self.depth_first_search(root_node, visit_callback, max_depth)
        logger.info(f"DFS completed. Visited {len(result)} nodes")
        return result
