"""
Template configuration file for Figma API credentials and settings
Copy this file to config.py and replace the placeholder values with your actual API keys
"""

# Figma API Configuration
FIGMA_ACCESS_TOKEN = "your_figma_access_token_here"  # Get from Figma Account Settings
FIGMA_FILE_KEY = "your_figma_file_key_here"  # Extract from Figma file URL
START_NODE_ID = "your_starting_node_id_here"  # The page/screen node ID

GEMINI_API_KEY = "your_gemini_api_key_here"  # Get from Google AI Studio
GEMINI_MODEL_NAME = "gemini-2.5-pro"

# API Settings
FIGMA_API_BASE_URL = "https://api.figma.com/v1"
FIGMA_API_TIMEOUT = 30  # seconds
FIGMA_API_RETRY_ATTEMPTS = 3

# DFS Settings
DEFAULT_MAX_DEPTH = 10
DEFAULT_SEARCH_LIMIT = 100

# Output Settings
PRINT_INDENT_SIZE = 2
SHOW_NODE_IDS = True
SHOW_NODE_TYPES = True

# Search Settings
DEFAULT_CASE_SENSITIVE = False
DEFAULT_SEARCH_LIMIT = 50

# Image Export Settings
DEFAULT_IMAGE_FORMAT = "png"
DEFAULT_IMAGE_SCALE = 1.0
DEFAULT_IMAGE_QUALITY = 80

# Rate Limiting
REQUEST_DELAY = 0.1  # seconds between requests
MAX_REQUESTS_PER_MINUTE = 60

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Example usage:
# from config import FIGMA_ACCESS_TOKEN, FIGMA_FILE_KEY
#
# figma_dfs = FigmaDFS(FIGMA_ACCESS_TOKEN)
# file_data = figma_dfs.get_file(FIGMA_FILE_KEY)
