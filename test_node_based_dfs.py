#!/usr/bin/env python3
"""
Test script for node ID-based DFS functionality
"""

import unittest
from unittest.mock import Mock, patch
from figma_dfs import FigmaDFS, FigmaNode


class TestNodeBasedDFS(unittest.TestCase):
    """Test cases for node ID-based DFS functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.figma_dfs = FigmaDFS("test_token")

        # Create mock nodes for testing
        self.mock_root_node = FigmaNode(
            id="1:1",
            name="Root Frame",
            type="FRAME",
            children=[
                FigmaNode(
                    id="1:2",
                    name="Child Frame",
                    type="FRAME",
                    children=[
                        FigmaNode(id="1:3", name="Text Node", type="TEXT"),
                        FigmaNode(id="1:4", name="Button", type="RECTANGLE"),
                    ],
                ),
                FigmaNode(
                    id="1:5",
                    name="Another Frame",
                    type="FRAME",
                    children=[FigmaNode(id="1:6", name="Another Text", type="TEXT")],
                ),
            ],
        )

    @patch("figma_dfs.requests.get")
    def test_depth_first_search_from_node_id(self, mock_get):
        """Test DFS starting from a specific node ID"""
        # Mock the API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "nodes": {
                "1:2": {
                    "document": {
                        "id": "1:2",
                        "name": "Child Frame",
                        "type": "FRAME",
                        "children": [
                            {"id": "1:3", "name": "Text Node", "type": "TEXT"},
                            {"id": "1:4", "name": "Button", "type": "RECTANGLE"},
                        ],
                    }
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test DFS from node ID
        visited_nodes = self.figma_dfs.depth_first_search_from_node_id(
            "test_file", "1:2"
        )

        # Should have visited 3 nodes: Child Frame, Text Node, Button
        self.assertEqual(len(visited_nodes), 3)
        self.assertEqual(visited_nodes[0].name, "Child Frame")
        self.assertEqual(visited_nodes[1].name, "Text Node")
        self.assertEqual(visited_nodes[2].name, "Button")

    @patch("figma_dfs.requests.get")
    def test_search_by_type_from_node_id(self, mock_get):
        """Test searching by type from a specific node ID"""
        # Mock the API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "nodes": {
                "1:2": {
                    "document": {
                        "id": "1:2",
                        "name": "Child Frame",
                        "type": "FRAME",
                        "children": [
                            {"id": "1:3", "name": "Text Node", "type": "TEXT"},
                            {"id": "1:4", "name": "Button", "type": "RECTANGLE"},
                        ],
                    }
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test searching for TEXT nodes from node ID
        text_nodes = self.figma_dfs.search_by_type_from_node_id(
            "test_file", "1:2", "TEXT"
        )

        # Should find 1 TEXT node
        self.assertEqual(len(text_nodes), 1)
        self.assertEqual(text_nodes[0].name, "Text Node")
        self.assertEqual(text_nodes[0].type, "TEXT")

    @patch("figma_dfs.requests.get")
    def test_search_by_name_from_node_id(self, mock_get):
        """Test searching by name from a specific node ID"""
        # Mock the API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "nodes": {
                "1:2": {
                    "document": {
                        "id": "1:2",
                        "name": "Child Frame",
                        "type": "FRAME",
                        "children": [
                            {"id": "1:3", "name": "Text Node", "type": "TEXT"},
                            {"id": "1:4", "name": "Button", "type": "RECTANGLE"},
                        ],
                    }
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test searching for nodes with "Button" in name
        button_nodes = self.figma_dfs.search_by_name_from_node_id(
            "test_file", "1:2", "Button"
        )

        # Should find 1 node with "Button" in name
        self.assertEqual(len(button_nodes), 1)
        self.assertEqual(button_nodes[0].name, "Button")

    @patch("figma_dfs.requests.get")
    def test_node_id_not_found(self, mock_get):
        """Test error handling when node ID is not found"""
        # Mock the API response for non-existent node
        mock_response = Mock()
        mock_response.json.return_value = {"nodes": {}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test that KeyError is raised for non-existent node ID
        with self.assertRaises(KeyError):
            self.figma_dfs.depth_first_search_from_node_id(
                "test_file", "non_existent_id"
            )

    def test_max_depth_parameter(self):
        """Test that max_depth parameter works correctly"""
        # Test with max_depth=1 (should only visit root and immediate children)
        visited_nodes = self.figma_dfs.depth_first_search(
            self.mock_root_node, max_depth=1
        )

        # Should have visited 3 nodes: Root Frame, Child Frame, Another Frame
        self.assertEqual(len(visited_nodes), 3)

        # Test with max_depth=0 (should only visit root)
        visited_nodes = self.figma_dfs.depth_first_search(
            self.mock_root_node, max_depth=0
        )
        self.assertEqual(len(visited_nodes), 1)
        self.assertEqual(visited_nodes[0].name, "Root Frame")


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
