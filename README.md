# Figma Agentic Workflow with LangGraph

A sophisticated Python library that uses Google's Gemini AI and LangGraph to intelligently analyze Figma designs and detect UI elements (inputs, buttons, links, selects) through an agentic workflow.

## 🚀 Features

- **Agentic AI Workflow**: Uses LangGraph to create intelligent agents that can call Figma APIs as tools
- **Gemini AI Integration**: Leverages Google's Gemini model for intelligent UI element detection
- **Dynamic Tool Calling**: AI agents can request additional Figma data when needed
- **Comprehensive Detection**: Identifies text inputs, buttons, links, and select elements
- **Optimized Data Processing**: Token-efficient data formatting for better AI performance
- **Detailed Logging**: Complete interaction logging with timestamped files
- **Async Processing**: Non-blocking operations for better performance

## 🏗️ Architecture Overview

The system uses a LangGraph-based workflow where Gemini AI acts as an intelligent agent that can:
1. Analyze initial Figma node data
2. Request additional data from Figma APIs when needed
3. Make informed decisions about UI element classification
4. Iteratively refine its analysis

```mermaid
graph TB
    %% External Services
    subgraph "External Services"
        FIGMA[Figma API]
        GEMINI[Gemini AI]
    end

    %% Main Components
    subgraph "Core System"
        WORKFLOW[FigmaAgenticWorkflow]
        DFS[FigmaDFS]
        ANALYZER[GeminiAnalyzer]
        PROMPTS[PromptCreator]
    end

    %% LangGraph Workflow
    subgraph "LangGraph Workflow"
        ANALYZER_NODE[Analyzer Node]
        TOOL_NODE[Tool Node]
        EXTRACT_NODE[Extract Node]
        DECISION{Continue?}
    end

    %% Data Flow
    subgraph "Data Processing"
        NODE_DATA[Node Data]
        FILTERED_DATA[Filtered Data]
        RESULTS[Analysis Results]
    end

    %% User Interface
    subgraph "User Interface"
        USER[User Input]
        CONFIG[Config.py]
        OUTPUT[Results Output]
    end

    %% Connections - User to System
    USER --> CONFIG
    CONFIG --> WORKFLOW

    %% Connections - System to External
    WORKFLOW --> FIGMA
    WORKFLOW --> GEMINI

    %% Connections - Core Components
    WORKFLOW --> DFS
    WORKFLOW --> ANALYZER
    WORKFLOW --> PROMPTS

    %% Connections - LangGraph Flow
    WORKFLOW --> ANALYZER_NODE
    ANALYZER_NODE --> DECISION
    DECISION -->|Need More Data| TOOL_NODE
    DECISION -->|Complete| EXTRACT_NODE
    TOOL_NODE --> ANALYZER_NODE
    EXTRACT_NODE --> RESULTS

    %% Connections - Data Flow
    DFS --> NODE_DATA
    NODE_DATA --> FILTERED_DATA
    FILTERED_DATA --> ANALYZER_NODE
    RESULTS --> OUTPUT

    %% Styling
    classDef external fill:#e1f5fe
    classDef core fill:#f3e5f5
    classDef workflow fill:#e8f5e8
    classDef data fill:#fff3e0
    classDef ui fill:#fce4ec

    class FIGMA,GEMINI external
    class WORKFLOW,DFS,ANALYZER,PROMPTS core
    class ANALYZER_NODE,TOOL_NODE,EXTRACT_NODE,DECISION workflow
    class NODE_DATA,FILTERED_DATA,RESULTS data
    class USER,CONFIG,OUTPUT ui
```

### Workflow Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant W as Workflow
    participant D as DFS
    participant F as Figma API
    participant A as Analyzer Node
    participant T as Tool Node
    participant G as Gemini AI
    participant E as Extract Node

    U->>W: analyze(file_key, node_id)
    W->>D: depth_first_search_from_node_id()
    D->>F: get_node_by_id()
    F-->>D: node data
    D-->>W: visited nodes
    
    W->>A: analyzer_node(state)
    A->>G: analyze with initial data
    G-->>A: analysis response
    
    alt Need more data
        A->>T: tool_node_wrapper()
        T->>F: get_multiple_figma_nodes()
        F-->>T: additional node data
        T-->>A: enhanced data
        A->>G: re-analyze with new data
        G-->>A: refined analysis
    end
    
    A->>E: extract_results()
    E-->>W: structured results
    W-->>U: classified UI elements
```

### System Architecture Diagram

```mermaid
graph LR
    %% Main System Components
    subgraph "Figma Agentic Workflow System"
        subgraph "Data Layer"
            FIGMA_API[Figma API Client]
            DFS_TRAVERSAL[DFS Traversal]
            NODE_FILTERING[Node Filtering]
        end
        
        subgraph "AI Layer"
            LANGGRAPH[LangGraph Workflow]
            GEMINI_AI[Gemini AI Model]
            PROMPT_ENGINE[Prompt Engine]
        end
        
        subgraph "Analysis Layer"
            ELEMENT_DETECTION[UI Element Detection]
            CLASSIFICATION[Element Classification]
            RESULT_FORMATTING[Result Formatting]
        end
        
        subgraph "Tools Layer"
            FIGMA_TOOLS[Figma API Tools]
            DYNAMIC_CALLS[Dynamic API Calls]
            DATA_ENHANCEMENT[Data Enhancement]
        end
    end
    
    %% External Dependencies
    subgraph "External Services"
        FIGMA_SERVICE[Figma Service]
        GEMINI_SERVICE[Gemini Service]
    end
    
    %% Data Flow
    FIGMA_API --> DFS_TRAVERSAL
    DFS_TRAVERSAL --> NODE_FILTERING
    NODE_FILTERING --> LANGGRAPH
    
    LANGGRAPH --> GEMINI_AI
    GEMINI_AI --> PROMPT_ENGINE
    PROMPT_ENGINE --> ELEMENT_DETECTION
    
    ELEMENT_DETECTION --> CLASSIFICATION
    CLASSIFICATION --> RESULT_FORMATTING
    
    %% Tool Integration
    LANGGRAPH --> FIGMA_TOOLS
    FIGMA_TOOLS --> DYNAMIC_CALLS
    DYNAMIC_CALLS --> DATA_ENHANCEMENT
    DATA_ENHANCEMENT --> GEMINI_AI
    
    %% External Connections
    FIGMA_API --> FIGMA_SERVICE
    GEMINI_AI --> GEMINI_SERVICE
    
    %% Styling
    classDef dataLayer fill:#e3f2fd
    classDef aiLayer fill:#f3e5f5
    classDef analysisLayer fill:#e8f5e8
    classDef toolsLayer fill:#fff3e0
    classDef external fill:#ffebee
    
    class FIGMA_API,DFS_TRAVERSAL,NODE_FILTERING dataLayer
    class LANGGRAPH,GEMINI_AI,PROMPT_ENGINE aiLayer
    class ELEMENT_DETECTION,CLASSIFICATION,RESULT_FORMATTING analysisLayer
    class FIGMA_TOOLS,DYNAMIC_CALLS,DATA_ENHANCEMENT toolsLayer
    class FIGMA_SERVICE,GEMINI_SERVICE external
```

## 📋 Workflow Steps

### 1. **Data Preparation Phase**
- **DFS Traversal**: Performs depth-first search on Figma file structure
- **Node Optimization**: Converts Figma nodes to token-efficient format
- **Data Filtering**: Extracts only UI-relevant properties (position, styling, text, etc.)

### 2. **Agentic Analysis Phase**
- **Initial Analysis**: Gemini AI analyzes the prepared node data
- **Tool Decision**: AI determines if additional Figma data is needed
- **Dynamic API Calls**: Uses `get_multiple_figma_nodes` tool to fetch specific nodes
- **Iterative Refinement**: Continues analysis with enhanced data

### 3. **Result Extraction Phase**
- **JSON Parsing**: Extracts structured results from AI responses
- **Node ID Cleaning**: Converts verbose node IDs to clean format
- **Classification**: Organizes results by element type (input, button, link, select)

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd locofy
```

2. **Install dependencies**:
```bash
pipenv install
```

3. **Activate the virtual environment**:
```bash
pipenv shell
```

4. **Configure your API keys** in `config.py`:
```python
FIGMA_ACCESS_TOKEN = "your_figma_access_token"
GEMINI_API_KEY = "your_gemini_api_key"
FIGMA_FILE_KEY = "your_figma_file_key"
START_NODE_ID = "your_starting_node_id" # ie The page/screen node.
```

**⚠️ Security Note**: The `config.py` file contains placeholder values. Replace them with your actual API keys:
- **Figma Access Token**: Get from [Figma Account Settings](https://www.figma.com/developers/api#access-tokens)
- **Gemini API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Figma File Key**: Extract from your Figma file URL (the part after `/file/`)

**For production use**, consider using environment variables instead of hardcoding API keys:
```python
import os
FIGMA_ACCESS_TOKEN = os.getenv("FIGMA_ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
```

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from figma_agentic_workflow import FigmaAgenticWorkflow

async def main():
    # Initialize the agentic workflow
    workflow = FigmaAgenticWorkflow()
    
    # Run the analysis
    results = await workflow.analyze(
        file_key="your_file_key",
        node_id="your_node_id",
        max_depth=None,  # No depth limit
        verbose=True,    # Show progress
    )
    
    print(f"Found {len(results)} classified elements")
    print(results)

# Run the example
asyncio.run(main())
```

### Running the Example

```bash
# Make sure you're in the virtual environment
pipenv shell

# Run the example script
python combined_gemini_example.py
```

## 📁 Project Structure

```
locofy/
├── figma_agentic_workflow.py    # Main LangGraph workflow implementation
├── combined_gemini_example.py   # Example usage script
├── figma_dfs.py                 # Figma DFS traversal utilities
├── gemini_analyzer.py           # Gemini AI analysis utilities
├── input_detection_prompts.py   # AI prompt generation
├── config.py                    # Configuration and API keys
├── config_template.py           # Template for API key setup
├── Pipfile                      # Python dependencies (pipenv)
├── Pipfile.lock                 # Locked dependencies
└── README.md                    # This file
```

## 🔧 Core Components

### FigmaAgenticWorkflow Class

The main class that orchestrates the entire agentic workflow:

```python
class FigmaAgenticWorkflow:
    def __init__(self):
        """Initialize the workflow with LangGraph setup"""
        
    async def analyze(self, file_key, node_id, max_depth=None, verbose=True):
        """Run the complete agentic analysis workflow"""
```

### LangGraph Workflow Nodes

1. **Analyzer Node**: Processes initial data and makes AI analysis decisions
2. **Tool Node**: Handles Figma API calls when AI needs more data
3. **Extract Node**: Parses and formats final results

### Available Tools

- **`get_multiple_figma_nodes`**: Fetches specific Figma nodes with UI-relevant filtering

## 🎯 Detection Capabilities

The system can identify:

- **Text Inputs**: Search bars, form fields, input containers
- **Buttons**: Primary buttons, icon buttons, action buttons
- **Links**: Clickable text links, navigation elements
- **Select Elements**: Dropdown menus, selection controls
