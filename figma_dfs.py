import requests
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time
import logging
import google.generativeai as genai
from config import (
    GEMINI_API_KEY,
    LOG_LEVEL,
    LOG_FORMAT,
    FIGMA_ACCESS_TOKEN,
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


class FigmaDFS:
    """Depth-First Search implementation for Figma nodes"""

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

    def get_file(self, file_key: str) -> Dict[str, Any]:
        """
        Get Figma file data

        Args:
            file_key: Figma file key from URL

        Returns:
            File data as dictionary
        """
        url = f"{self.base_url}/files/{file_key}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

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

    def search_by_type(self, root_node: FigmaNode, target_type: str) -> List[FigmaNode]:
        """
        Search for nodes of a specific type using DFS

        Args:
            root_node: Starting node
            target_type: Type of nodes to find

        Returns:
            List of nodes matching the type
        """
        matching_nodes = []

        def type_callback(node: FigmaNode, depth: int) -> bool:
            if node.type == target_type:
                matching_nodes.append(node)
            return True

        self.depth_first_search(root_node, type_callback)
        return matching_nodes

    def search_by_name(
        self, root_node: FigmaNode, name_pattern: str, case_sensitive: bool = False
    ) -> List[FigmaNode]:
        """
        Search for nodes by name pattern using DFS

        Args:
            root_node: Starting node
            name_pattern: Name pattern to search for
            case_sensitive: Whether search should be case sensitive

        Returns:
            List of nodes matching the name pattern
        """
        matching_nodes = []

        def name_callback(node: FigmaNode, depth: int) -> bool:
            node_name = node.name
            pattern = name_pattern

            if not case_sensitive:
                node_name = node_name.lower()
                pattern = pattern.lower()

            if pattern in node_name:
                matching_nodes.append(node)
            return True

        self.depth_first_search(root_node, name_callback)
        return matching_nodes

    def print_node_tree(self, root_node: FigmaNode, max_depth: Optional[int] = None):
        """
        Print a visual tree representation of nodes

        Args:
            root_node: Root node to print
            max_depth: Maximum depth to print
        """

        def print_callback(node: FigmaNode, depth: int) -> bool:
            indent = "  " * depth
            print(f"{indent}├─ {node.name} ({node.type}) [ID: {node.id}]")
            return True

        print(f"Node Tree for: {root_node.name}")
        print("=" * 50)
        self.depth_first_search(root_node, print_callback, max_depth)

    def _extract_base_node_id(self, node_id: str) -> str:
        """
        Extract the base node ID from a Figma node ID string.
        Figma node IDs can include component and variant information separated by semicolons.
        This method returns only the base node ID (before the first semicolon).

        Args:
            node_id: Full node ID string (e.g., "290:7308;216:4169")

        Returns:
            Base node ID string (e.g., "290:7308")
        """
        return node_id.split(";")[0] if ";" in node_id else node_id

    def identify_text_inputs_with_gemini(
        self, nodes: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, str]]:
        """
        Identify text input elements using Gemini AI analysis ONLY

        Args:
            nodes: List of dictionaries with node_id, type, x, y properties and optional data

        Returns:
            Dictionary with node_id as keys and {"tag": "input"} as values
        """
        logger.info(f"Starting Gemini text input identification for {len(nodes)} nodes")

        # Configure Gemini
        logger.debug("Configuring Gemini API...")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        logger.debug("Gemini API configured successfully")

        # Prepare the prompt for Gemini
        logger.info("Creating Gemini prompt...")
        prompt = self._create_gemini_input_detection_prompt(nodes)
        logger.debug(f"Gemini prompt created (length: {len(prompt)} characters)")
        logger.debug(f"Prompt preview: {prompt[:500]}...")

        # Write the full prompt to a file
        try:
            with open("gemini_prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
            logger.info("Full Gemini prompt written to gemini_prompt.txt")
        except Exception as e:
            logger.error(f"Error writing prompt to file: {e}")

        try:
            # Send request to Gemini
            logger.info("Sending request to Gemini API...")
            logger.debug("Making Gemini API call...")
            response = model.generate_content(prompt)
            logger.info("Gemini API response received successfully")
            logger.debug(f"Gemini response length: {len(response.text)} characters")
            logger.debug(f"Gemini response preview: {response.text[:500]}...")

            # Parse Gemini's response
            logger.info("Parsing Gemini response...")
            result = self._parse_gemini_response(response.text)
            logger.info(
                f"Successfully parsed {len(result)} text input nodes from Gemini response"
            )

            return result

        except Exception as e:
            logger.error(f"Error using Gemini for text input detection: {e}")
            logger.info("Gemini detection failed - returning empty result")
            # Return empty result instead of falling back to local detection
            return {}

    def _create_gemini_input_detection_prompt(self, nodes: List[Dict[str, Any]]) -> str:
        """
        Create a concise prompt for Gemini to detect main text input elements

        Args:
            nodes: List of node dictionaries with complete structure

        Returns:
            Formatted prompt string for Gemini
        """
        logger.info(f"Creating concise Gemini prompt for {len(nodes)} nodes...")

        prompt = """You are a UI analyst. Identify main text input elements from Figma nodes.

## Detection Criteria:
- Main input containers (FRAME, RECTANGLE, INSTANCE) with input functionality
- Named like "Search Bar", "Input Field", "Text Input", "Search"
- MUST be the main input container, not child elements
- Exclude: icons, buttons, labels, help text, placeholder text, backgrounds
- Exclude: any node that is a child of another input container

## Output Format:
```json
{
  "<node_id>": {"tag": "input"},
  "<node_id>": {"tag": "input"}
}
```

## Nodes to Analyze:"""

        # Add concise node information to the prompt
        logger.debug("Adding concise node data to prompt...")
        for i, node in enumerate(nodes, 1):
            logger.debug(
                f"Adding node {i+1}/{len(nodes)}: {node.get('node_id', 'N/A')} - {node.get('name', 'N/A')}"
            )

            # Only include essential information for text input detection
            node_id = node.get("node_id", "N/A")
            node_type = node.get("type", "N/A")
            node_name = node.get("name", "N/A")
            parent_id = node.get("parent_id", None)

            # Skip nodes that are clearly not input candidates
            if node_type in ["TEXT", "VECTOR", "LINE", "ELLIPSE", "STAR"]:
                continue

            # Skip child elements that are clearly not main inputs (but allow main input containers)
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
                continue

            # Skip nodes that are children of other input containers (but allow the main containers themselves)
            if (
                parent_id
                and parent_id != "1:13180"
                and any(
                    parent_name in node_name.lower()
                    for parent_name in ["input", "search", "field"]
                )
            ):
                continue

            prompt += f"\n{node_id}|{node_type}|{node_name}"

            # Add only critical data for input detection
            if "data" in node and node["data"]:
                data = node["data"]

                # Include corner radius for input-like styling
                if "cornerRadius" in data and data.get("cornerRadius", 0) > 0:
                    prompt += f"|r{data.get('cornerRadius')}"

                # Include text content only for nodes that might be inputs
                if (
                    node_type in ["FRAME", "RECTANGLE", "INSTANCE"]
                    and "characters" in data
                ):
                    chars = data.get("characters", "").strip()
                    if chars and len(chars) < 50:  # Only short text
                        prompt += f"|t{chars}"

        prompt += """
Analyze the nodes above and return your JSON response:"""

        logger.info(
            f"Concise Gemini prompt created successfully (length: {len(prompt)} characters)"
        )
        return prompt

    def _parse_gemini_response(self, response_text: str) -> Dict[str, Dict[str, str]]:
        """
        Parse Gemini's response and extract text input nodes

        Args:
            response_text: Raw response from Gemini

        Returns:
            Dictionary with node_id as keys and {"tag": "input"} as values
        """
        logger.info("Starting to parse Gemini response...")
        logger.debug(f"Response text length: {len(response_text)} characters")

        try:
            # Extract JSON from response (handle markdown code blocks)
            logger.debug("Extracting JSON from Gemini response...")

            if "```json" in response_text:
                logger.debug("Found ```json code block, extracting...")
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
                logger.debug(
                    f"Extracted JSON from ```json block (length: {len(json_str)})"
                )
            elif "```" in response_text:
                logger.debug("Found ``` code block, extracting...")
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
                logger.debug(f"Extracted JSON from ``` block (length: {len(json_str)})")
            else:
                logger.debug("No code blocks found, looking for JSON object...")
                # Try to find JSON object in the response
                start_brace = response_text.find("{")
                end_brace = response_text.rfind("}")
                if start_brace != -1 and end_brace != -1:
                    json_str = response_text[start_brace : end_brace + 1]
                    logger.debug(f"Extracted JSON object (length: {len(json_str)})")
                else:
                    logger.error("No JSON found in response")
                    raise ValueError("No JSON found in response")

            logger.debug(f"Extracted JSON string: {json_str[:200]}...")

            # Parse the JSON
            logger.debug("Parsing JSON string...")
            input_nodes_dict = json.loads(json_str)
            logger.info(
                f"Successfully parsed JSON, found {len(input_nodes_dict)} text input nodes"
            )

            return input_nodes_dict

        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            logger.error(f"Raw response: {response_text}")
            return {}

    def get_text_input_nodes_from_figma_with_gemini(
        self, file_key: str, node_ids: List[str]
    ) -> Dict[str, Dict[str, str]]:
        """
        Get text input nodes from Figma API and identify them using Gemini ONLY

        Args:
            file_key: Figma file key
            node_ids: List of node IDs to analyze

        Returns:
            Dictionary with node_id as keys and {"tag": "input"} as values
        """
        logger.info(
            f"Starting Gemini text input detection for {len(node_ids)} node IDs: {node_ids}"
        )
        logger.info(f"File key: {file_key}")

        # Get node data from Figma
        nodes_with_data = []
        logger.info("Fetching node data from Figma API...")

        for i, node_id in enumerate(node_ids, 1):
            try:
                logger.info(f"Processing node {i}/{len(node_ids)}: {node_id}")

                # Get node data from Figma API
                logger.debug(f"Making API call to get node {node_id}")
                node_data = self.get_node_by_id(file_key, node_id)
                nodes = node_data.get("nodes", {})
                logger.debug(f"API response received for node {node_id}")

                if node_id in nodes:
                    node_info = nodes[node_id]
                    document = node_info.get("document", {})
                    logger.debug(f"Node {node_id} document data extracted")

                    # Extract position information
                    absolute_bounding_box = document.get("absoluteBoundingBox", {})
                    x = absolute_bounding_box.get("x", 0)
                    y = absolute_bounding_box.get("y", 0)
                    logger.debug(f"Node {node_id} position: x={x}, y={y}")

                    # Create node dictionary with full data
                    node_dict = {
                        "node_id": self._extract_base_node_id(node_id),
                        "type": document.get("type", ""),
                        "x": x,
                        "y": y,
                        "name": document.get("name", ""),
                        "data": document,
                        "parent_id": None,  # No parent tracking for individual node analysis
                    }
                    nodes_with_data.append(node_dict)
                    logger.info(
                        f"Node {node_id} processed successfully: {node_dict['name']} ({node_dict['type']})"
                    )

                else:
                    logger.warning(f"Node {node_id} not found in API response")

            except Exception as e:
                logger.error(f"Error processing node {node_id}: {e}")
                continue

        logger.info(
            f"Successfully processed {len(nodes_with_data)} nodes out of {len(node_ids)} requested"
        )
        logger.debug(
            f"Final nodes_with_data structure: {json.dumps([{'node_id': n['node_id'], 'name': n['name'], 'type': n['type']} for n in nodes_with_data], indent=2)}"
        )

        # Use Gemini to analyze the nodes
        logger.info("Starting Gemini analysis...")
        result = self.identify_text_inputs_with_gemini(nodes_with_data)
        logger.info(
            f"Gemini analysis completed. Found {len(result)} text input structures"
        )
        return result

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
        logger.info(f"Visit callback provided: {visit_callback is not None}")

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

    def search_by_type_from_node_id(
        self, file_key: str, node_id: str, target_type: str
    ) -> List[FigmaNode]:
        """
        Search for nodes of a specific type using DFS starting from a specific node ID

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            target_type: Type of nodes to find

        Returns:
            List of nodes matching the type
        """
        logger.info(
            f"Searching for nodes of type '{target_type}' starting from node ID: {node_id}"
        )

        matching_nodes = []

        def type_callback(node: FigmaNode, depth: int) -> bool:
            if node.type == target_type:
                matching_nodes.append(node)
                logger.debug(
                    f"Found matching node: {node.name} (ID: {node.id}) at depth {depth}"
                )
            return True

        self.depth_first_search_from_node_id(file_key, node_id, type_callback)
        logger.info(f"Found {len(matching_nodes)} nodes of type '{target_type}'")
        return matching_nodes

    def search_by_name_from_node_id(
        self,
        file_key: str,
        node_id: str,
        name_pattern: str,
        case_sensitive: bool = False,
    ) -> List[FigmaNode]:
        """
        Search for nodes by name pattern using DFS starting from a specific node ID

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            name_pattern: Name pattern to search for
            case_sensitive: Whether search should be case sensitive

        Returns:
            List of nodes matching the name pattern
        """
        logger.info(
            f"Searching for nodes with name pattern '{name_pattern}' (case_sensitive: {case_sensitive}) starting from node ID: {node_id}"
        )

        matching_nodes = []

        def name_callback(node: FigmaNode, depth: int) -> bool:
            node_name = node.name
            pattern = name_pattern

            if not case_sensitive:
                node_name = node_name.lower()
                pattern = pattern.lower()

            if pattern in node_name:
                matching_nodes.append(node)
                logger.debug(
                    f"Found matching node: {node.name} (ID: {node.id}) at depth {depth}"
                )
            return True

        self.depth_first_search_from_node_id(file_key, node_id, name_callback)
        logger.info(
            f"Found {len(matching_nodes)} nodes matching name pattern '{name_pattern}'"
        )
        return matching_nodes

    def print_node_tree_from_node_id(
        self, file_key: str, node_id: str, max_depth: Optional[int] = None
    ):
        """
        Print the node tree starting from a specific node ID

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            max_depth: Maximum depth to print (None for unlimited)
        """
        logger.info(f"Printing node tree starting from node ID: {node_id}")
        logger.info(f"Max depth: {max_depth}")

        def print_callback(node: FigmaNode, depth: int) -> bool:
            indent = "  " * depth
            print(f"{indent}{node.name} ({node.type}) - ID: {node.id}")
            logger.debug(
                f"Printed node: {node.name} ({node.type}) - ID: {node.id} at depth {depth}"
            )
            return True

        self.depth_first_search_from_node_id(
            file_key, node_id, print_callback, max_depth
        )
        logger.info("Node tree printing completed")

    def get_all_text_inputs_from_figma_with_gemini(
        self, file_key: str, node_id: str, max_depth: Optional[int] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Get all nodes with their structure starting from a specific node ID and analyze for text inputs with Gemini ONLY

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            max_depth: Maximum depth to traverse (None for unlimited)

        Returns:
            Dictionary with node_id as keys and {"tag": "input"} as values
        """
        logger.info(
            f"Starting comprehensive text input analysis with Gemini from node ID: {node_id}"
        )
        logger.info(f"File key: {file_key}")
        logger.info(f"Max depth: {max_depth}")

        # Get all nodes using DFS
        all_nodes = self.depth_first_search_from_node_id(
            file_key, node_id, max_depth=max_depth
        )
        logger.info(f"Found {len(all_nodes)} total nodes in the structure")

        # Convert FigmaNode objects to dictionaries with full structure
        nodes_with_data = []
        logger.info("Converting nodes to structured data for Gemini analysis...")

        for i, node in enumerate(all_nodes, 1):
            try:
                logger.debug(
                    f"Processing node {i}/{len(all_nodes)}: {node.name} (ID: {node.id})"
                )

                # Extract position information from the node's data
                absolute_bounding_box = node.data.get("absoluteBoundingBox", {})
                x = absolute_bounding_box.get("x", 0)
                y = absolute_bounding_box.get("y", 0)

                # Create comprehensive node dictionary using existing data
                node_dict = {
                    "node_id": self._extract_base_node_id(node.id),
                    "type": node.type,
                    "x": x,
                    "y": y,
                    "name": node.name,
                    "data": node.data,
                    "parent_id": (
                        self._extract_base_node_id(node.parent_id)
                        if node.parent_id
                        else None
                    ),
                }
                nodes_with_data.append(node_dict)
                logger.debug(
                    f"Node {node.id} processed successfully: {node_dict['name']} ({node_dict['type']})"
                )

            except Exception as e:
                logger.error(f"Error processing node {node.id}: {e}")
                continue

        logger.info(
            f"Successfully processed {len(nodes_with_data)} nodes out of {len(all_nodes)} found"
        )
        logger.info("Sending all nodes to Gemini for text input analysis...")

        # Use Gemini to analyze all nodes
        result = self.identify_text_inputs_with_gemini(nodes_with_data)
        logger.info(
            f"Gemini analysis completed. Found {len(result)} text input structures"
        )
        return result


def example_usage():
    """Example usage of the FigmaDFS class"""

    # Replace with your actual Figma access token
    ACCESS_TOKEN = "your_figma_access_token_here"

    # Replace with your actual Figma file key (from URL)
    FILE_KEY = "your_file_key_here"

    # Optional: Replace with a specific node ID to start DFS from
    # You can get this from Figma by right-clicking on a node and selecting "Copy/Paste as > Copy link"
    # The node ID is the last part of the URL
    STARTING_NODE_ID = None  # Set to a specific node ID if you want to start from there

    # Initialize Figma DFS
    figma_dfs = FigmaDFS(ACCESS_TOKEN)

    try:
        # Get file data
        print("Fetching Figma file...")
        file_data = figma_dfs.get_file(FILE_KEY)

        # Parse the document node
        document_data = file_data.get("document", {})
        root_node = figma_dfs.parse_node(document_data)

        print(f"Successfully loaded file: {file_data.get('name', 'Unknown')}")
        print(f"Root node: {root_node.name} ({root_node.type})")

        # Example 1: Print the entire node tree (from root)
        print("\n" + "=" * 60)
        print("ENTIRE NODE TREE (FROM ROOT)")
        print("=" * 60)
        figma_dfs.print_node_tree(root_node, max_depth=3)

        # Example 2: If a specific node ID is provided, demonstrate DFS from that node
        if STARTING_NODE_ID:
            print("\n" + "=" * 60)
            print(f"NODE TREE STARTING FROM NODE ID: {STARTING_NODE_ID}")
            print("=" * 60)
            try:
                figma_dfs.print_node_tree_from_node_id(
                    FILE_KEY, STARTING_NODE_ID, max_depth=3
                )

                # Example 3: Find all text nodes starting from the specific node
                print("\n" + "=" * 60)
                print(
                    f"FINDING ALL TEXT NODES STARTING FROM NODE ID: {STARTING_NODE_ID}"
                )
                print("=" * 60)
                text_nodes = figma_dfs.search_by_type_from_node_id(
                    FILE_KEY, STARTING_NODE_ID, "TEXT"
                )
                for i, node in enumerate(text_nodes, 1):
                    print(f"{i}. {node.name} (ID: {node.id})")

                # Example 4: Find nodes containing "button" in name starting from the specific node
                print("\n" + "=" * 60)
                print(
                    f"FINDING NODES WITH 'BUTTON' IN NAME STARTING FROM NODE ID: {STARTING_NODE_ID}"
                )
                print("=" * 60)
                button_nodes = figma_dfs.search_by_name_from_node_id(
                    FILE_KEY, STARTING_NODE_ID, "button", case_sensitive=False
                )
                for i, node in enumerate(button_nodes, 1):
                    print(f"{i}. {node.name} ({node.type}) - ID: {node.id}")

            except KeyError as e:
                print(f"Error: {e}")
                print("Make sure the node ID exists in the file")
            except requests.exceptions.RequestException as e:
                print(f"Error making request to Figma API: {e}")

        # Example 5: Find all text nodes (from root)
        print("\n" + "=" * 60)
        print("FINDING ALL TEXT NODES (FROM ROOT)")
        print("=" * 60)
        text_nodes = figma_dfs.search_by_type(root_node, "TEXT")
        for i, node in enumerate(text_nodes, 1):
            print(f"{i}. {node.name} (ID: {node.id})")

        # Example 6: Find nodes containing "button" in name (from root)
        print("\n" + "=" * 60)
        print("FINDING NODES WITH 'BUTTON' IN NAME (FROM ROOT)")
        print("=" * 60)
        button_nodes = figma_dfs.search_by_name(
            root_node, "button", case_sensitive=False
        )
        for i, node in enumerate(button_nodes, 1):
            print(f"{i}. {node.name} ({node.type}) - ID: {node.id}")

        # Example 7: Custom DFS with callback (from root)
        print("\n" + "=" * 60)
        print("CUSTOM DFS - FRAMES ONLY (MAX DEPTH 2) (FROM ROOT)")
        print("=" * 60)

        def frame_callback(node: FigmaNode, depth: int) -> bool:
            if node.type == "FRAME":
                indent = "  " * depth
                print(f"{indent}Frame: {node.name} (ID: {node.id})")
            return True

        figma_dfs.depth_first_search(root_node, frame_callback, max_depth=2)

    except requests.exceptions.RequestException as e:
        print(f"Error making request to Figma API: {e}")
    except KeyError as e:
        print(f"Error parsing Figma data: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def example_usage_with_node_id():
    """Example usage demonstrating DFS from a specific node ID"""

    # Initialize Figma DFS
    figma_dfs = FigmaDFS(FIGMA_ACCESS_TOKEN)

    # Use the START_NODE_ID from config
    logger.info(f"Starting example with node ID: {START_NODE_ID}")

    try:
        print(f"Starting DFS from node ID: {START_NODE_ID}")
        print("=" * 60)

        # Print the tree starting from this node
        figma_dfs.print_node_tree_from_node_id(
            FIGMA_FILE_KEY, START_NODE_ID, max_depth=2
        )

        # Search for specific types starting from this node
        print("\n" + "=" * 60)
        print("Searching for TEXT nodes from this starting point:")
        print("=" * 60)
        text_nodes = figma_dfs.search_by_type_from_node_id(
            FIGMA_FILE_KEY, START_NODE_ID, "TEXT"
        )
        for i, node in enumerate(text_nodes, 1):
            print(f"{i}. {node.name} (ID: {node.id})")

    except KeyError as e:
        logger.error(f"Node ID not found: {e}")
        print(f"Node ID not found: {e}")
        print("Please provide a valid node ID from your Figma file")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    example_usage()
