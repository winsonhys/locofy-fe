#!/usr/bin/env python3
"""
Simple Agentic Workflow for Figma Node Analysis
Minimizes token usage while maintaining core functionality
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI

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
class SimpleNode:
    """Minimal node representation to reduce token usage"""

    id: str
    name: str
    type: str
    characters: Optional[str] = None


class SimpleFigmaAgenticWorkflow:
    """Simple agentic workflow with minimal token usage"""

    def __init__(self):
        """Initialize the simple workflow"""
        self.figma_dfs = FigmaDFS(FIGMA_ACCESS_TOKEN)
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
            max_output_tokens=1024,  # Reduced significantly
        )

    def _extract_minimal_nodes(
        self, nodes_data: List[Dict[str, Any]]
    ) -> List[SimpleNode]:
        """Extract only essential node information to minimize tokens"""
        minimal_nodes = []
        for node in nodes_data:
            # Only extract the most important fields
            simple_node = SimpleNode(
                id=node.get("id", ""),
                name=node.get("name", ""),
                type=node.get("type", ""),
                characters=node.get("data", {}).get("characters", ""),
            )
            minimal_nodes.append(simple_node)
        return minimal_nodes

    def _create_minimal_prompt(self, nodes: List[SimpleNode]) -> str:
        """Create a minimal prompt to reduce token usage"""
        # Convert nodes to minimal JSON format
        nodes_json = []
        for node in nodes:
            node_data = {"id": node.id, "name": node.name, "type": node.type}
            if node.characters:
                node_data["text"] = node.characters[:50]  # Limit text length
            nodes_json.append(node_data)

        prompt = f"""Analyze these Figma nodes. Classify each as: link, button, input, select, or none.

Respond with JSON only:
{{
  "node_id": "tag"
}}

Nodes: {json.dumps(nodes_json, separators=(',', ':'))}"""

        return prompt

    async def _analyze_batch(self, nodes: List[SimpleNode]) -> Dict[str, str]:
        """Analyze a batch of nodes with minimal token usage"""
        try:
            prompt = self._create_minimal_prompt(nodes)
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])

            # Parse simple response format
            try:
                content = response.content.strip()

                # Remove markdown code blocks if present
                if content.startswith("```json"):
                    content = content[7:]
                elif content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]

                result = json.loads(content.strip())
                return result
            except json.JSONDecodeError:
                logger.error(f"Failed to parse response: {response.content[:100]}...")
                return {}

        except Exception as e:
            logger.error(f"Error analyzing batch: {e}")
            return {}

    async def analyze_figma_nodes(
        self,
        file_key: str,
        node_id: str,
        max_depth: Optional[int] = None,
        batch_size: int = 3,  # Smaller batches
    ) -> Dict[str, Any]:
        """
        Analyze Figma nodes with minimal token usage

        Args:
            file_key: Figma file key
            node_id: Starting node ID
            max_depth: Maximum depth for DFS traversal
            batch_size: Number of nodes per batch

        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Starting simple analysis for file {file_key}, node {node_id}")
        start_time = time.time()

        try:
            # Extract nodes
            nodes = self.figma_dfs.depth_first_search_from_node_id(
                file_key, node_id, max_depth=max_depth
            )

            # Convert to minimal format
            nodes_data = []
            for node in nodes:
                node_data = {
                    "id": node.id,
                    "name": node.name,
                    "type": node.type,
                    "data": node.data,
                }
                nodes_data.append(node_data)

            logger.info(f"Extracted {len(nodes_data)} nodes")

            # Process in small batches
            all_results = {}
            total_batches = (len(nodes_data) + batch_size - 1) // batch_size

            for i in range(0, len(nodes_data), batch_size):
                batch_data = nodes_data[i : i + batch_size]
                batch_nodes = self._extract_minimal_nodes(batch_data)

                batch_num = i // batch_size + 1
                logger.info(f"Processing batch {batch_num}/{total_batches}")

                batch_results = await self._analyze_batch(batch_nodes)
                all_results.update(batch_results)

                # Small delay between batches
                if i + batch_size < len(nodes_data):
                    await asyncio.sleep(1.0)

            # Calculate statistics
            tag_counts = {"link": 0, "button": 0, "input": 0, "select": 0, "none": 0}
            for tag in all_results.values():
                if tag in tag_counts:
                    tag_counts[tag] += 1

            end_time = time.time()

            return {
                "success": True,
                "results": all_results,
                "metadata": {
                    "total_nodes": len(nodes_data),
                    "total_batches": total_batches,
                    "batch_size": batch_size,
                    "analysis_time": end_time - start_time,
                    "tag_counts": tag_counts,
                },
            }

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"success": False, "error": str(e), "results": {}, "metadata": {}}


async def main():
    """Example usage of the simple agentic workflow"""
    print("🚀 Simple Figma Agentic Workflow")
    print("=" * 40)

    workflow = SimpleFigmaAgenticWorkflow()

    try:
        print("🔍 Starting simple analysis...")
        start_time = time.time()

        result = await workflow.analyze_figma_nodes(
            FIGMA_FILE_KEY,
            START_NODE_ID,
            max_depth=2,  # Reduced depth
            batch_size=2,  # Very small batches
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
            print(f"Batches processed: {metadata.get('total_batches', 0)}")
            print()

            # Show tag counts
            tag_counts = metadata.get("tag_counts", {})
            print("🎯 Element Type Distribution:")
            for tag, count in tag_counts.items():
                if count > 0:
                    print(f"  {tag.upper()}: {count}")
            print()

            # Show detailed results
            print("📋 Results:")
            for node_id, tag in results.items():
                print(f"  Node {node_id}: {tag}")

            # Show performance metrics
            print("\n⚡ Performance Metrics:")
            print(f"  - Total time: {analysis_time:.2f} seconds")
            print(f"  - Batch size: {metadata.get('batch_size', 0)}")
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
