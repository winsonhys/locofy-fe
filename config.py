"""
Configuration file for Figma API credentials and settings
"""

# Figma API Configuration
FIGMA_ACCESS_TOKEN = "secret_token_here"
FIGMA_FILE_KEY = "NnmJQ6LgSUJn08LLXkylSp"
START_NODE_ID = "1:2"

GEMINI_API_KEY = "secret_token_here"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

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
