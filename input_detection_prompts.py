#!/usr/bin/env python3
"""
Input Detection Prompts - Prompt creation logic for text input detection
Handles the creation of optimized prompts for Gemini AI input detection
"""

import logging
from typing import Dict, List, Any, Optional
from config import FIGMA_ACCESS_TOKEN, FIGMA_FILE_KEY, START_NODE_ID

logger = logging.getLogger(__name__)


class InputDetectionPromptCreator:
    """Creates optimized prompts for text input detection using Gemini AI"""

    def create_prompt(self, nodes: List[Dict[str, Any]]) -> str:
        """
        Create an optimized prompt for text input detection

        Args:
            nodes: List of node dictionaries with complete structure

        Returns:
            Formatted prompt string for Gemini
        """
        logger.info(f"Creating input detection prompt for {len(nodes)} nodes...")

        prompt = self._get_base_prompt()

        # Add filtered nodes to prompt
        filtered_nodes = self._filter_nodes_for_input_detection(nodes)

        for node in filtered_nodes:
            node_line = self._format_node_for_prompt(node)
            prompt += f"\n{node_line}"

        prompt += """
Analyze the nodes above and return your JSON response:"""

        logger.info(
            f"Input detection prompt created successfully (length: {len(prompt)} characters)"
        )
        return prompt

    def _get_base_prompt(self) -> str:
        """Get the base prompt template for input detection"""
        return """You are a UI analyst. Identify main text input elements from Figma nodes.

## Input Field Characteristics:
- Purpose: Allow users to enter text data
- Shape: Typically rectangular and wider than tall (to accommodate text input)
- Content: May contain placeholder text, labels, or help text
- Naming: Often named 'Input', 'Search Bar', 'Text Field', 'Form Field', 'Email', 'Password'
- Structure: Usually contains text elements, labels, and sometimes icons
- Styling: May have borders, background fills, and rounded corners
- Size: Generally longer horizontally to provide space for user typing

## Icon Rules:
- Inputs may have a left icon (e.g., search, user, email, etc.)—this is common and does NOT affect input classification.
- Inputs do NOT require a right icon.
- Only classify as select if:
  1. The node is explicitly named 'Select', 'Dropdown', 'Menu', 'Choose', 'Filter', OR
  2. The right icon is clearly a dropdown indicator (named like 'arrow-down', 'chevron-down', 'ep:arrow-down').
- All other right-side icons (including generic names like 'Icon / Right', 'Right Icon', 'Arrow' without 'down') should NOT change an input to a select.

## Node Context:
- For each node, you are provided with left and right icon information (names and positions if present).

## What to Look For:
- Nodes named with input-specific terms: 'Input', 'Search Bar', 'Text Field', 'Form Field'
- Rectangular containers that are wider than tall
- Containers that include text elements, labels, or placeholder text
- Elements designed for user text entry

## Naming Patterns:
- 'Search Bar' = Input field (include)
- 'Input' = Input field (include)
- 'Search' (standalone) = Usually an icon, not an input field
- 'Search Icon' = Icon, not an input field

## IMPORTANT - Right Icon Classification:
- Inputs with right-side icons are STILL inputs unless:
  1. The node is explicitly named 'Select', 'Dropdown', 'Menu', 'Choose', 'Filter'
  2. The right icon is clearly a dropdown arrow (named like 'arrow-down', 'chevron-down', 'ep:arrow-down')
- Generic right icons like 'Icon / Right', 'Right Icon', 'Arrow' (without 'down') should NOT make an input a select
- Only classify as select if the name OR the right icon name clearly indicates a dropdown/select

## Output Format:
```json
{
  "<node_id>": {"tag": "input"},
  "<node_id>": {"tag": "input"}
}
```

## Nodes to Analyze:"""

    def _filter_nodes_for_input_detection(
        self, nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter nodes specifically for input detection

        Args:
            nodes: List of all nodes

        Returns:
            Filtered list of nodes relevant for input detection
        """
        filtered_nodes = []
        for node in nodes:
            node_type = node.get("type", "N/A")
            # Only filter out obvious non-UI types
            if node_type in ["TEXT", "VECTOR", "LINE", "ELLIPSE", "STAR"]:
                continue
            filtered_nodes.append(node)
        return filtered_nodes

    def _should_skip_child_element(
        self, node_name: str, parent_id: Optional[str]
    ) -> bool:
        # No longer used, but kept for reference
        return False

    def _format_node_for_prompt(self, node: Dict[str, Any]) -> str:
        """
        Format a single node for inclusion in the prompt

        Args:
            node: Node dictionary

        Returns:
            Formatted node string for prompt
        """
        node_id = node.get("node_id", "N/A")
        node_type = node.get("type", "N/A")
        node_name = node.get("name", "N/A")
        parent_name = node.get("parent_name", "")
        parent_type = node.get("parent_type", "")
        child_names = ",".join(node.get("child_names", []))
        child_types = ",".join(node.get("child_types", []))
        sibling_names = ",".join(node.get("sibling_names", []))
        sibling_types = ",".join(node.get("sibling_types", []))
        has_right_icon = node.get("has_right_icon", False)
        right_icon_name = node.get("right_icon_name", "")
        is_wider_than_tall = node.get("is_wider_than_tall", False)
        text_content = node.get("text_content", "")

        node_line = f"{node_id}|{node_type}|{node_name}|parent:{parent_name}({parent_type})|children:{child_names}({child_types})|siblings:{sibling_names}({sibling_types})|right_icon:{has_right_icon}:{right_icon_name}|wider_than_tall:{is_wider_than_tall}|text:{text_content}"

        # Add additional data that's relevant for input detection
        if "data" in node and node["data"]:
            data = node["data"]

            # Add corner radius for input-like styling
            if "cornerRadius" in data and data.get("cornerRadius", 0) > 0:
                node_line += f"|r{data.get('cornerRadius')}"

            # Add text content for relevant nodes
            if node_type in ["FRAME", "RECTANGLE", "INSTANCE"] and "characters" in data:
                chars = data.get("characters", "").strip()
                if chars and len(chars) < 50:  # Only short text
                    node_line += f"|t{chars}"

        return node_line

    def get_prompt_statistics(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the prompt creation process

        Args:
            nodes: List of all nodes

        Returns:
            Dictionary with statistics
        """
        filtered_nodes = self._filter_nodes_for_input_detection(nodes)

        return {
            "total_nodes": len(nodes),
            "filtered_nodes": len(filtered_nodes),
            "filtered_percentage": (
                (len(filtered_nodes) / len(nodes)) * 100 if nodes else 0
            ),
            "node_types": list(set(node.get("type", "N/A") for node in filtered_nodes)),
        }


class ButtonDetectionPromptCreator:
    """Creates optimized prompts for button detection using Gemini AI"""

    def create_prompt(self, nodes: List[Dict[str, Any]]) -> str:
        """
        Create an optimized prompt for button detection

        Args:
            nodes: List of node dictionaries with complete structure

        Returns:
            Formatted prompt string for Gemini
        """
        logger.info(f"Creating button detection prompt for {len(nodes)} nodes...")

        prompt = self._get_base_prompt()

        filtered_nodes = self._filter_nodes_for_button_detection(nodes)
        for node in filtered_nodes:
            node_line = self._format_node_for_prompt(node)
            prompt += f"\n{node_line}"

        prompt += """
Analyze the nodes above and return your JSON response:"""

        logger.info(
            f"Button detection prompt created successfully (length: {len(prompt)} characters)"
        )
        return prompt

    def _get_base_prompt(self) -> str:
        return """You are a UI analyst. Identify button and dropdown/select elements from Figma nodes.

## Button Characteristics:
- Purpose: Trigger actions when clicked
- Shape: Can be rectangular, rounded, or pill-shaped
- Content: Usually contains text labels, icons, or both
- Naming: Often named 'Button', 'Submit', 'Save', 'Cancel', 'Primary', 'Secondary'
- Styling: May have background fills, borders, and hover states
- Size: Varies but typically compact and clickable

## Text as Buttons:
- Text elements in headers/footers that look like call-to-action
- Text that describes actions you can do when clicked (e.g., "Login", "Sign Up", "Settings", "Profile")
- Tab selection text (e.g., "Home", "About", "Contact", "Products")
- Section selection text (e.g., "All", "Recent", "Popular", "Favorites")
- Navigation text that implies action (e.g., "Next", "Previous", "Back", "Continue")
- Text that appears clickable and action-oriented

## Dropdown/Select Characteristics:
- Purpose: Allow users to choose from a list of options
- Shape: Typically rectangular with a dropdown arrow
- Content: Shows selected value and has a dropdown indicator
- Naming: Often named 'Select', 'Dropdown', 'Menu', 'Choose', 'Filter'
- Structure: Contains text + dropdown icon on the right
- Behavior: Opens a list of options when clicked

## What to Look For:

### Buttons:
- Nodes named with button-specific terms: 'Button', 'Submit', 'Save', 'Cancel'
- Elements designed for user interaction and action triggering
- Containers with button-like styling and text content
- Text elements in headers/footers that are action-oriented
- Tab/section selection text that implies clickable behavior

### Dropdowns/Selects:
- Nodes named 'Select', 'Dropdown', 'Menu', 'Choose', 'Filter'
- Elements with dropdown arrows (named like 'arrow-down', 'chevron-down', 'ep:arrow-down')
- Containers that appear to show selected values with dropdown indicators

## IMPORTANT - Right Icon Classification:
- Only classify as select if:
  1. The node is explicitly named 'Select', 'Dropdown', 'Menu', 'Choose', 'Filter'
  2. The right icon is clearly a dropdown arrow (named like 'arrow-down', 'chevron-down', 'ep:arrow-down')
- Generic right icons like 'Icon / Right', 'Right Icon', 'Arrow' (without 'down') should NOT make an element a select
- Input elements with generic right icons should remain as inputs, not selects

## Output Format:
```json
{
  "<node_id>": {"tag": "button"},
  "<node_id>": {"tag": "select"}
}
```"""

    def _filter_nodes_for_button_detection(
        self, nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        filtered_nodes = []
        for node in nodes:
            node_type = node.get("type", "N/A")
            if node_type in ["TEXT", "VECTOR", "LINE", "ELLIPSE", "STAR"]:
                continue
            filtered_nodes.append(node)
        return filtered_nodes

    def _format_node_for_prompt(self, node: Dict[str, Any]) -> str:
        node_id = node.get("node_id", "N/A")
        node_type = node.get("type", "N/A")
        node_name = node.get("name", "N/A")
        parent_name = node.get("parent_name", "")
        parent_type = node.get("parent_type", "")
        child_names = ",".join(node.get("child_names", []))
        child_types = ",".join(node.get("child_types", []))
        sibling_names = ",".join(node.get("sibling_names", []))
        sibling_types = ",".join(node.get("sibling_types", []))
        has_right_icon = node.get("has_right_icon", False)
        right_icon_name = node.get("right_icon_name", "")
        is_wider_than_tall = node.get("is_wider_than_tall", False)
        text_content = node.get("text_content", "")

        node_line = f"{node_id}|{node_type}|{node_name}|parent:{parent_name}({parent_type})|children:{child_names}({child_types})|siblings:{sibling_names}({sibling_types})|right_icon:{has_right_icon}:{right_icon_name}|wider_than_tall:{is_wider_than_tall}|text:{text_content}"

        if "data" in node and node["data"]:
            data = node["data"]
            if "cornerRadius" in data and data.get("cornerRadius", 0) > 0:
                node_line += f"|r{data.get('cornerRadius')}"
        return node_line


class LinkDetectionPromptCreator:
    """Creates optimized prompts for link detection using Gemini AI"""

    def create_prompt(self, nodes: List[Dict[str, Any]]) -> str:
        """
        Create an optimized prompt for link detection

        Args:
            nodes: List of node dictionaries with complete structure

        Returns:
            Formatted prompt string for Gemini
        """
        logger.info(f"Creating link detection prompt for {len(nodes)} nodes...")

        prompt = self._get_base_prompt()

        filtered_nodes = self._filter_nodes_for_link_detection(nodes)
        for node in filtered_nodes:
            node_line = self._format_node_for_prompt(node)
            prompt += f"\n{node_line}"

        prompt += """
Analyze the nodes above and return your JSON response:"""

        logger.info(
            f"Link detection prompt created successfully (length: {len(prompt)} characters)"
        )
        return prompt

    def _get_base_prompt(self) -> str:
        return """You are a UI analyst. Identify main link elements from Figma nodes.

## Link Characteristics:
- Purpose: Navigate to external URLs or internal pages when clicked
- Content: Contains actual URL text (starting with http://, https://, ftp://, mailto:, tel:, etc.)
- Naming: Often named "Link", "URL", "Website", "External Link", "Navigation", "Visit", "Go to"
- Structure: Usually contains text elements with URL content
- Styling: ALWAYS underlined and has different color from surrounding text (typically blue, purple, or brand colors)
- Visual Indicators: May have hover states, different colors to indicate tappability
- Behavior: Designed for navigation, not form submission or data entry
- Text Content: Contains actual URLs, domain names, or explicit link text like "Visit website", "Go to page"

## CRITICAL: Links vs Buttons
- Links contain ACTUAL URLs or explicit navigation text
- Buttons contain ACTION text (Submit, Buy, Add, Delete, etc.) - these are NOT links
- "Buy tickets" = BUTTON (action), not a link
- "Visit website" = LINK (navigation)
- "Submit" = BUTTON (action), not a link
- "http://example.com" = LINK (actual URL)
- "example.com" = LINK (domain name)

## What to Look For:
- Nodes named with link-specific terms: "Link", "URL", "Website", "External", "Navigation", "Visit"
- Text elements containing actual URL schemes (http://, https://, ftp://, mailto:, tel:)
- Text elements containing domain names (example.com, www.example.com)
- Elements with ALWAYS underlined styling and different colors
- Explicit navigation text like "Visit website", "Go to page", "Learn more"

## What to EXCLUDE:
- Action buttons (Submit, Buy, Add, Delete, etc.)
- Form submission buttons
- Interactive elements that perform actions rather than navigation
- Elements named "Button" or containing action words

## URL Patterns (include as links):
- http://example.com
- https://example.com
- ftp://example.com
- mailto:user@example.com
- tel:+1234567890
- www.example.com
- example.com

## Action Patterns (exclude as links):
- "Buy tickets" = BUTTON
- "Submit" = BUTTON
- "Add to cart" = BUTTON
- "Delete" = BUTTON
- "Save" = BUTTON

## Output Format:
```json
{
  "<node_id>": {"tag": "link"},
  "<node_id>": {"tag": "link"}
}
```

## Nodes to Analyze:"""

    def _filter_nodes_for_link_detection(
        self, nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter nodes for link detection

        Args:
            nodes: List of all nodes

        Returns:
            Filtered list of nodes for link detection
        """
        filtered_nodes = []

        for node in nodes:
            node_id = node.get("node_id", "N/A")
            node_type = node.get("type", "N/A")
            node_name = node.get("name", "N/A")
            parent_id = node.get("parent_id", None)

            # Skip nodes that are clearly not candidates
            if node_type in ["VECTOR", "LINE", "ELLIPSE", "STAR"]:
                continue

            # Skip child elements that are likely not main link containers
            if parent_id and any(
                skip_name in node_name.lower()
                for skip_name in [
                    "icon",
                    "label",
                    "inner",
                    "base",
                    "fill",
                    "background",
                    "border",
                ]
            ):
                continue

            filtered_nodes.append(node)

        logger.info(
            f"Link detection filtering: {len(nodes)} total nodes -> {len(filtered_nodes)} filtered nodes"
        )
        return filtered_nodes

    def _format_node_for_prompt(self, node: Dict[str, Any]) -> str:
        """
        Format a node for inclusion in the prompt

        Args:
            node: Node dictionary

        Returns:
            Formatted node string
        """
        node_id = node.get("node_id", "N/A")
        node_type = node.get("type", "N/A")
        node_name = node.get("name", "N/A")

        # Start with basic node info
        node_line = f"{node_id}|{node_type}|{node_name}"

        # Add additional data if available
        data = node.get("data", {})
        if data:
            # Add corner radius for styling info
            if "cornerRadius" in data and data.get("cornerRadius", 0) > 0:
                node_line += f"|r{data.get('cornerRadius')}"

            # Add text content for TEXT nodes
            if node_type == "TEXT" and "characters" in data:
                chars = data.get("characters", "").strip()
                if chars and len(chars) < 100:  # Limit text length
                    node_line += f"|t{chars}"

            # Add fill color info for styling
            if "fills" in data and data["fills"]:
                fills = data["fills"]
                if fills and len(fills) > 0:
                    fill = fills[0]
                    if "color" in fill:
                        color = fill["color"]
                        node_line += f"|c{color.get('r', 0):.2f},{color.get('g', 0):.2f},{color.get('b', 0):.2f}"

            # Add stroke info for underlining
            if "strokes" in data and data["strokes"]:
                node_line += "|s"

        # Add child nodes with indentation
        children = node.get("children", {})
        for child_id, child_info in children.items():
            child_line = f"I{child_id}|{child_info['type']}|{child_info['name']}"

            # Add child text content
            child_data = child_info.get("data", {})
            if child_info["type"] == "TEXT" and "characters" in child_data:
                chars = child_data.get("characters", "").strip()
                if chars and len(chars) < 100:
                    child_line += f"|t{chars}"

            node_line += f"\n{child_line}"

        return node_line


class CombinedDetectionPromptCreator:
    """Creates optimized prompts for combined detection of all element types using Gemini AI"""

    def create_prompt(
        self, nodes: List[Dict[str, Any]], detection_types: List[str] = None
    ) -> str:
        """
        Create a comprehensive prompt for combined detection of all element types

        Args:
            nodes: List of node dictionaries with complete structure
            detection_types: List of detection types to analyze (default: ["text_input", "button", "link", "select"])

        Returns:
            Formatted combined prompt string for Gemini
        """
        if detection_types is None:
            detection_types = ["text_input", "button", "link", "select"]

        logger.info(
            f"Creating combined detection prompt for {detection_types} with {len(nodes)} nodes..."
        )

        prompt = self._get_base_prompt()

        # Add filtered nodes to prompt
        filtered_nodes = self._filter_nodes_for_combined_detection(nodes)

        for node in filtered_nodes:
            node_line = self._format_node_for_prompt(node)
            prompt += f"\n{node_line}"

        prompt += f"""
Analyze each node above and determine if it represents an input field, button, select/dropdown, or link. Return your analysis in the JSON format specified above. Only include nodes that clearly match one of the detection categories."""

        logger.info(
            f"Combined detection prompt created successfully (length: {len(prompt)} characters)"
        )
        return prompt

    def _get_base_prompt(self) -> str:
        """Get the comprehensive base prompt template for combined detection"""
        return f"""You are an expert UI analyst specializing in identifying interactive elements from Figma design nodes. Your task is to analyze each node and determine if it represents an input field, button, select/dropdown, or link.

## Figma Access Information (Optional - You may use this to make better decisions):
- Figma Access Token: {FIGMA_ACCESS_TOKEN}
- Figma File Key: {FIGMA_FILE_KEY}
- Start Node ID: {START_NODE_ID}
- You can optionally access the Figma file directly to get additional context about the design, but this is not required. You may ignore this information if you prefer to rely solely on the provided node data.

## Detection Guidelines:

### TEXT INPUT ELEMENTS:
- Purpose: Allow users to enter text data
- Shape: Typically rectangular and wider than tall (to accommodate text input)
- Content: May contain placeholder text, labels, or help text
- Naming: Often named 'Input', 'Search Bar', 'Text Field', 'Form Field', 'Email', 'Password', 'Search'
- Structure: Usually contains text elements, labels, and sometimes icons
- Styling: May have borders, background fills, and rounded corners
- Size: Generally longer horizontally to provide space for user typing
- Main input containers (FRAME, RECTANGLE, INSTANCE) designed for text entry
- Exclude: icons, buttons, labels, help text that are children of input containers

### BUTTON ELEMENTS:
- Purpose: Trigger actions when clicked
- Shape: Can be rectangular, rounded, or pill-shaped
- Content: Usually contains text labels, icons, or both
- Naming: Often named 'Button', 'Submit', 'Save', 'Cancel', 'Primary', 'Secondary', 'IconButton', 'Add', 'Delete'
- Styling: May have background fills, borders, and hover states
- Size: Varies but typically compact and clickable
- Interactive button containers (FRAME, RECTANGLE, INSTANCE) with button functionality
- Exclude: icons, labels, text elements that are children of buttons

### STANDALONE ICONS AS BUTTONS:
- **CRITICAL RULE**: Standalone icons (VECTOR, INSTANCE, FRAME with icon content) are generally BUTTONS unless they are children of input containers
- Icons that appear to be clickable/interactive should be classified as buttons
- Common standalone icon buttons: search icons, menu icons, close icons, settings icons, profile icons
- Only exclude icons that are clearly decorative or part of input fields
- If an icon appears to serve an interactive purpose, classify it as a button

### SELECT/DROPDOWN ELEMENTS:
- Purpose: Allow users to choose from a list of options
- Shape: Typically rectangular with a dropdown arrow
- Content: Shows selected value and has a dropdown indicator
- Naming: Often named 'Select', 'Dropdown', 'Menu', 'Choose', 'Filter', 'Category'
- Structure: Contains text + dropdown icon on the right
- Behavior: Opens a list of options when clicked
- Dropdown/select containers with selection functionality
- Contains text + dropdown indicator (arrows, chevrons, down arrow)
- May show current selection or placeholder text
- Exclude: regular action buttons without dropdown indicators

### LINK ELEMENTS:
- Purpose: Navigate to external URLs or internal pages when clicked
- Content: Contains actual URL text (starting with http://, https://, ftp://, mailto:, tel:, etc.). Any text that fulfils rfc1738 and is not a child of a button or select is a link
- Naming: Often named "Link", "URL", "Website", "External Link", "Navigation", "Visit", "Go to", "More", "Read More", "Learn More"
- Structure: Usually contains text elements with URL content
- Styling: ALWAYS underlined and has different color from surrounding text (typically blue, purple, or brand colors)
- Visual Indicators: May have hover states, different colors to indicate tappability
- Behavior: Designed for navigation, not form submission or data entry
- Text Content: Contains actual URLs, domain names, or explicit link text like "Visit website", "Go to page"
- May contain icons (external link, arrow)
- Exclude: regular text that's not meant to be clicked

## CRITICAL: Links vs Buttons
- Links contain ACTUAL URLs or explicit navigation text
- Buttons contain ACTION text (Submit, Buy, Add, Delete, etc.) - these are NOT links
- "Buy tickets" = BUTTON (action), not a link
- "Visit website" = LINK (navigation)
- "Submit" = BUTTON (action), not a link
- "http://example.com" = LINK (actual URL)
- "example.com" = LINK (domain name)

## Text as Buttons:
- Text elements in headers/footers that look like call-to-action
- Text that describes actions you can do when clicked (e.g., "Login", "Sign Up", "Settings", "Profile")
- Tab selection text (e.g., "Home", "About", "Contact", "Products")
- Section selection text (e.g., "All", "Recent", "Popular", "Favorites")
- Navigation text that implies action (e.g., "Next", "Previous", "Back", "Continue")
- Text that appears clickable and action-oriented
- Text elements that contains words that appears to be clickable and action-oriented

## IMPORTANT - Right Icon Classification:
- Inputs may have a left icon (e.g., search, user, email, etc.)—this is common and does NOT affect input classification.
- Inputs do NOT require a right icon.
- Only classify as select if:
  1. The node is explicitly named 'Select', 'Dropdown', 'Menu', 'Choose', 'Filter', OR
  2. The right icon is clearly a dropdown indicator (named like 'arrow-down', 'chevron-down', 'ep:arrow-down').
- All other right-side icons (including generic names like 'Icon / Right', 'Right Icon', 'Arrow' without 'down') should NOT change an input to a select.
- Generic right icons like 'Icon / Right', 'Right Icon', 'Arrow' (without 'down') should NOT make an element a select
- Input elements with generic right icons should remain as inputs, not selects

## URL Patterns (include as links):
- http://example.com
- https://example.com
- ftp://example.com
- mailto:user@example.com
- tel:+1234567890
- www.example.com
- example.com

## Action Patterns (exclude as links):
- "Buy tickets" = BUTTON
- "Submit" = BUTTON
- "Add to cart" = BUTTON
- "Delete" = BUTTON
- "Save" = BUTTON

## Decision Process:
1. First, determine if the node is a main container (not a child element)
2. Analyze the node's name, type, and context
3. Consider the node's styling and visual characteristics
4. Look at child elements to understand the node's purpose
5. **Pay special attention to standalone icons - they are usually buttons unless part of an input**
6. Choose the most appropriate tag: "input", "button", "select", or "link"
7. If the node doesn't clearly fit any category, exclude it

## Output Format:
```json
{{
  "<node_id>": {{"tag": "<detected_tag>"}},
  "<node_id>": {{"tag": "<detected_tag>"}}
}}
```

Where <detected_tag> is one of: "input", "button", "select", "link"

## Nodes to Analyze:"""

    def _filter_nodes_for_combined_detection(
        self, nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter nodes for combined detection (less restrictive than individual filters)

        Args:
            nodes: List of all nodes

        Returns:
            Filtered list of nodes for combined detection
        """
        filtered_nodes = []

        for node in nodes:
            node_id = node.get("node_id", "N/A")
            node_type = node.get("type", "N/A")
            node_name = node.get("name", "N/A")
            parent_id = node.get("parent_id", None)

            # Skip nodes that are clearly not candidates
            if node_type in ["LINE", "ELLIPSE", "STAR"]:
                continue

            # Include VECTOR nodes (icons) as they can be standalone buttons
            # Skip very small decorative text elements
            if node_type == "TEXT" and len(node_name) < 3:
                continue

            # Skip child elements that are clearly not main containers, but be less restrictive for icons
            if parent_id and any(
                skip_name in node_name.lower()
                for skip_name in [
                    "label",
                    "inner",
                    "base",
                    "fill",
                    "background",
                    "border",
                    "shadow",
                    "overlay",
                ]
            ):
                continue

            # Special handling for icons - include them unless they're clearly decorative
            if node_type == "VECTOR" or "icon" in node_name.lower():
                # Include standalone icons as they can be buttons
                # Only skip if they're clearly decorative child elements
                if parent_id and any(
                    skip_name in node_name.lower()
                    for skip_name in [
                        "decorative",
                        "ornament",
                        "pattern",
                        "background",
                    ]
                ):
                    continue

            filtered_nodes.append(node)

        logger.info(
            f"Combined detection filtering: {len(nodes)} total nodes -> {len(filtered_nodes)} filtered nodes"
        )
        return filtered_nodes

    def _format_node_for_prompt(self, node: Dict[str, Any]) -> str:
        """
        Format a single node for inclusion in the prompt

        Args:
            node: Node dictionary

        Returns:
            Formatted node string for prompt
        """
        node_id = node.get("node_id", "N/A")
        node_type = node.get("type", "N/A")
        node_name = node.get("name", "N/A")
        parent_name = node.get("parent_name", "")
        parent_type = node.get("parent_type", "")
        child_names = ",".join(node.get("child_names", []))
        child_types = ",".join(node.get("child_types", []))
        sibling_names = ",".join(node.get("sibling_names", []))
        sibling_types = ",".join(node.get("sibling_types", []))
        has_right_icon = node.get("has_right_icon", False)
        right_icon_name = node.get("right_icon_name", "")
        is_wider_than_tall = node.get("is_wider_than_tall", False)
        text_content = node.get("text_content", "")

        node_line = f"{node_id}|{node_type}|{node_name}|parent:{parent_name}({parent_type})|children:{child_names}({child_types})|siblings:{sibling_names}({sibling_types})|right_icon:{has_right_icon}:{right_icon_name}|wider_than_tall:{is_wider_than_tall}|text:{text_content}"

        # Add additional data that's relevant for detection
        if "data" in node and node["data"]:
            data = node["data"]

            # Add corner radius for input-like styling
            if "cornerRadius" in data and data.get("cornerRadius", 0) > 0:
                node_line += f"|r{data.get('cornerRadius')}"

            # Add text content for relevant nodes
            if node_type in ["FRAME", "RECTANGLE", "INSTANCE"] and "characters" in data:
                chars = data.get("characters", "").strip()
                if chars and len(chars) < 50:  # Only short text
                    node_line += f"|t{chars}"

            # Add fill color info for styling
            if "fills" in data and data["fills"]:
                fills = data["fills"]
                if fills and len(fills) > 0:
                    fill = fills[0]
                    if "color" in fill:
                        color = fill["color"]
                        node_line += f"|c{color.get('r', 0):.2f},{color.get('g', 0):.2f},{color.get('b', 0):.2f}"

            # Add stroke info for underlining
            if "strokes" in data and data["strokes"]:
                node_line += "|s"

        return node_line

    def get_prompt_statistics(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the prompt creation process

        Args:
            nodes: List of all nodes

        Returns:
            Dictionary with statistics
        """
        filtered_nodes = self._filter_nodes_for_combined_detection(nodes)

        return {
            "total_nodes": len(nodes),
            "filtered_nodes": len(filtered_nodes),
            "filtered_percentage": (
                (len(filtered_nodes) / len(nodes)) * 100 if nodes else 0
            ),
            "node_types": list(set(node.get("type", "N/A") for node in filtered_nodes)),
        }


class FunctionCallingDetectionPromptCreator:
    """Creates optimized prompts for detection using hybrid workflow: DFS + function calling backup"""

    def create_prompt(
        self,
        nodes: List[Dict[str, Any]],
        file_key: str,
        node_id: Optional[str] = None,
        detection_types: List[str] = None,
    ) -> str:
        """
        Create a comprehensive prompt for detection using hybrid workflow

        Args:
            nodes: List of node dictionaries from DFS
            file_key: Figma file key (for function calling if needed)
            node_id: Optional specific node ID (for function calling if needed)
            detection_types: List of detection types to analyze (default: ["text_input", "button", "link", "select"])

        Returns:
            Formatted prompt string for Gemini with function calling capabilities
        """
        if detection_types is None:
            detection_types = ["text_input", "button", "link", "select"]

        logger.info(
            f"Creating function calling detection prompt for {detection_types} with {len(nodes)} nodes..."
        )

        prompt = self._get_base_prompt(file_key, node_id)

        # Add filtered nodes to prompt
        filtered_nodes = self._filter_nodes_for_function_calling_detection(nodes)

        for node in filtered_nodes:
            node_line = self._format_node_for_prompt(node)
            prompt += f"\n{node_line}"

        prompt += f"""
Analyze each node above and determine if it represents an input field, button, select/dropdown, or link. Return your analysis in the JSON format specified above. Only include nodes that clearly match one of the detection categories.

If you need additional context about specific nodes, file structure, or want to search for specific elements,
you can call the following functions:
- get_figma_file: Get the complete file structure
- get_figma_node: Get a specific node and its children
- search_nodes_by_type: Search for nodes of a specific type
- search_nodes_by_name: Search for nodes by name pattern"""

        logger.info(
            f"Function calling detection prompt created successfully (length: {len(prompt)} characters)"
        )
        return prompt

    def _get_base_prompt(self, file_key: str, node_id: Optional[str] = None) -> str:
        """Get the comprehensive base prompt template for function calling detection"""
        node_info = f"Node ID: {node_id}" if node_id else "Complete file analysis"

        return f"""You are an expert UI analyst specializing in identifying interactive elements from Figma design nodes. Your task is to analyze each node and determine if it represents an input field, button, select/dropdown, or link.

## Figma Access Information:
- Figma File Key: {file_key}
- {node_info}
- You have access to nodes from a depth-first search of the Figma design.

## Detection Guidelines:

### TEXT INPUT ELEMENTS:
- Purpose: Allow users to enter text data
- Shape: Typically rectangular and wider than tall (to accommodate text input)
- Content: May contain placeholder text, labels, or help text
- Naming: Often named 'Input', 'Search Bar', 'Text Field', 'Form Field', 'Email', 'Password', 'Search'
- Structure: Usually contains text elements, labels, and sometimes icons
- Styling: May have borders, background fills, and rounded corners
- Size: Generally longer horizontally to provide space for user typing
- Main input containers (FRAME, RECTANGLE, INSTANCE) designed for text entry
- Exclude: icons, buttons, labels, help text that are children of input containers

### BUTTON ELEMENTS:
- Purpose: Trigger actions when clicked
- Shape: Can be rectangular, rounded, or pill-shaped
- Content: Usually contains text labels, icons, or both
- Naming: Often named 'Button', 'Submit', 'Save', 'Cancel', 'Primary', 'Secondary', 'IconButton', 'Add', 'Delete'
- Styling: May have background fills, borders, and hover states
- Size: Varies but typically compact and clickable
- Interactive button containers (FRAME, RECTANGLE, INSTANCE) with button functionality
- Exclude: icons, labels, text elements that are children of buttons

### STANDALONE ICONS AS BUTTONS:
- **CRITICAL RULE**: Standalone icons (VECTOR, INSTANCE, FRAME with icon content) are generally BUTTONS unless they are children of input containers
- Icons that appear to be clickable/interactive should be classified as buttons
- Common standalone icon buttons: search icons, menu icons, close icons, settings icons, profile icons
- Only exclude icons that are clearly decorative or part of input fields
- If an icon appears to serve an interactive purpose, classify it as a button

### SELECT/DROPDOWN ELEMENTS:
- Purpose: Allow users to choose from a list of options
- Shape: Typically rectangular with a dropdown arrow
- Content: Shows selected value and has a dropdown indicator
- Naming: Often named 'Select', 'Dropdown', 'Menu', 'Choose', 'Filter', 'Category'
- Structure: Contains text + dropdown icon on the right
- Behavior: Opens a list of options when clicked
- Dropdown/select containers with selection functionality
- Contains text + dropdown indicator (arrows, chevrons, down arrow)
- May show current selection or placeholder text
- Exclude: regular action buttons without dropdown indicators

### LINK ELEMENTS:
- Purpose: Navigate to external URLs or internal pages when clicked
- Content: Contains actual URL text (starting with http://, https://, ftp://, mailto:, tel:, etc.). Any text that fulfils rfc1738 and is not a child of a button or select is a link
- Naming: Often named "Link", "URL", "Website", "External Link", "Navigation", "Visit", "Go to", "More", "Read More", "Learn More"
- Structure: Usually contains text elements with URL content
- Styling: ALWAYS underlined and has different color from surrounding text (typically blue, purple, or brand colors)
- Visual Indicators: May have hover states, different colors to indicate tappability
- Behavior: Designed for navigation, not form submission or data entry
- Text Content: Contains actual URLs, domain names, or explicit link text like "Visit website", "Go to page"
- May contain icons (external link, arrow)
- Exclude: regular text that's not meant to be clicked

## CRITICAL: Links vs Buttons
- Links contain ACTUAL URLs or explicit navigation text
- Buttons contain ACTION text (Submit, Buy, Add, Delete, etc.) - these are NOT links
- "Buy tickets" = BUTTON (action), not a link
- "Visit website" = LINK (navigation)
- "Submit" = BUTTON (action), not a link
- "http://example.com" = LINK (actual URL)
- "example.com" = LINK (domain name)

## Text as Buttons:
- Text elements in headers/footers that look like call-to-action
- Text that describes actions you can do when clicked (e.g., "Login", "Sign Up", "Settings", "Profile")
- Tab selection text (e.g., "Home", "About", "Contact", "Products")
- Section selection text (e.g., "All", "Recent", "Popular", "Favorites")
- Navigation text that implies action (e.g., "Next", "Previous", "Back", "Continue")
- Text that appears clickable and action-oriented
- Text elements that contains words that appears to be clickable and action-oriented

## IMPORTANT - Right Icon Classification:
- Inputs may have a left icon (e.g., search, user, email, etc.)—this is common and does NOT affect input classification.
- Inputs do NOT require a right icon.
- Only classify as select if:
  1. The node is explicitly named 'Select', 'Dropdown', 'Menu', 'Choose', 'Filter', OR
  2. The right icon is clearly a dropdown indicator (named like 'arrow-down', 'chevron-down', 'ep:arrow-down').
- All other right-side icons (including generic names like 'Icon / Right', 'Right Icon', 'Arrow' without 'down') should NOT change an input to a select.
- Generic right icons like 'Icon / Right', 'Right Icon', 'Arrow' (without 'down') should NOT make an element a select
- Input elements with generic right icons should remain as inputs, not selects

## URL Patterns (include as links):
- http://example.com
- https://example.com
- ftp://example.com
- mailto:user@example.com
- tel:+1234567890
- www.example.com
- example.com

## Action Patterns (exclude as links):
- "Buy tickets" = BUTTON
- "Submit" = BUTTON
- "Add to cart" = BUTTON
- "Delete" = BUTTON
- "Save" = BUTTON

## Decision Process:
1. First, determine if the node is a main container (not a child element)
2. Analyze the node's name, type, and context
3. Consider the node's styling and visual characteristics
4. Look at child elements to understand the node's purpose
5. **Pay special attention to standalone icons - they are usually buttons unless part of an input**
6. Choose the most appropriate tag: "input", "button", "select", or "link"
7. If the node doesn't clearly fit any category, exclude it

## Output Format:
```json
{{
  "<node_id>": {{"tag": "<detected_tag>"}},
  "<node_id>": {{"tag": "<detected_tag>"}}
}}
```

Where <detected_tag> is one of: "input", "button", "select", "link"

## Nodes to Analyze:"""

    def _filter_nodes_for_function_calling_detection(
        self, nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter nodes for function calling detection (similar to combined detection)

        Args:
            nodes: List of all nodes

        Returns:
            Filtered list of nodes for function calling detection
        """
        filtered_nodes = []

        for node in nodes:
            node_id = node.get("node_id", "N/A")
            node_type = node.get("type", "N/A")
            node_name = node.get("name", "N/A")
            parent_id = node.get("parent_id", None)

            # Skip nodes that are clearly not candidates
            if node_type in ["LINE", "ELLIPSE", "STAR"]:
                continue

            # Include VECTOR nodes (icons) as they can be standalone buttons
            # Skip very small decorative text elements
            if node_type == "TEXT" and len(node_name) < 3:
                continue

            # Skip child elements that are clearly not main containers, but be less restrictive for icons
            if parent_id and any(
                skip_name in node_name.lower()
                for skip_name in [
                    "label",
                    "inner",
                    "base",
                    "fill",
                    "background",
                    "border",
                    "shadow",
                    "overlay",
                ]
            ):
                continue

            # Special handling for icons - include them unless they're clearly decorative
            if node_type == "VECTOR" or "icon" in node_name.lower():
                # Include standalone icons as they can be buttons
                # Only skip if they're clearly decorative child elements
                if parent_id and any(
                    skip_name in node_name.lower()
                    for skip_name in [
                        "decorative",
                        "ornament",
                        "pattern",
                        "background",
                    ]
                ):
                    continue

            filtered_nodes.append(node)

        logger.info(
            f"Function calling detection filtering: {len(nodes)} total nodes -> {len(filtered_nodes)} filtered nodes"
        )
        return filtered_nodes

    def _format_node_for_prompt(self, node: Dict[str, Any]) -> str:
        """
        Format a single node for inclusion in the prompt

        Args:
            node: Node dictionary

        Returns:
            Formatted node string for prompt
        """
        node_id = node.get("node_id", "N/A")
        node_type = node.get("type", "N/A")
        node_name = node.get("name", "N/A")
        parent_name = node.get("parent_name", "")
        parent_type = node.get("parent_type", "")
        child_names = ",".join(node.get("child_names", []))
        child_types = ",".join(node.get("child_types", []))
        sibling_names = ",".join(node.get("sibling_names", []))
        sibling_types = ",".join(node.get("sibling_types", []))
        has_right_icon = node.get("has_right_icon", False)
        right_icon_name = node.get("right_icon_name", "")
        is_wider_than_tall = node.get("is_wider_than_tall", False)
        text_content = node.get("text_content", "")

        node_line = f"{node_id}|{node_type}|{node_name}|parent:{parent_name}({parent_type})|children:{child_names}({child_types})|siblings:{sibling_names}({sibling_types})|right_icon:{has_right_icon}:{right_icon_name}|wider_than_tall:{is_wider_than_tall}|text:{text_content}"

        # Add additional data that's relevant for detection
        if "data" in node and node["data"]:
            data = node["data"]

            # Add corner radius for input-like styling
            if "cornerRadius" in data and data.get("cornerRadius", 0) > 0:
                node_line += f"|r{data.get('cornerRadius')}"

            # Add text content for relevant nodes
            if node_type in ["FRAME", "RECTANGLE", "INSTANCE"] and "characters" in data:
                chars = data.get("characters", "").strip()
                if chars and len(chars) < 50:  # Only short text
                    node_line += f"|t{chars}"

            # Add fill color info for styling
            if "fills" in data and data["fills"]:
                fills = data["fills"]
                if fills and len(fills) > 0:
                    fill = fills[0]
                    if "color" in fill:
                        color = fill["color"]
                        node_line += f"|c{color.get('r', 0):.2f},{color.get('g', 0):.2f},{color.get('b', 0):.2f}"

            # Add stroke info for underlining
            if "strokes" in data and data["strokes"]:
                node_line += "|s"

        return node_line

    def get_prompt_statistics(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the prompt creation process

        Args:
            nodes: List of all nodes

        Returns:
            Dictionary with statistics
        """
        filtered_nodes = self._filter_nodes_for_function_calling_detection(nodes)

        return {
            "total_nodes": len(nodes),
            "filtered_nodes": len(filtered_nodes),
            "filtered_percentage": (
                (len(filtered_nodes) / len(nodes)) * 100 if nodes else 0
            ),
            "node_types": list(set(node.get("type", "N/A") for node in filtered_nodes)),
        }
