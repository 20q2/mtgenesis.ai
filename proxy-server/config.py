"""
Configuration constants for MTGenesis AI card generation system.
"""

import os

# ===== PERFORMANCE TOGGLE =====
# Set to False to force CPU-only mode (slower but won't destroy your GPU)
# Set to True to use CUDA if available (faster but may lag your system)
USE_CUDA = True  # <-- Change this to True when you want GPU mode

# ===== MODEL SELECTION =====
# Using SDXL-Turbo for best quality image generation
ENABLE_IMAGE_GENERATION = True  # Set to False to disable image generation (placeholder mode)

# ===== TIMEOUT CONFIGURATION =====
# Global timeout settings for all operations (in seconds)
COLD_START_TIMEOUT = 180    # 3 minutes for first-time model loading
WARM_RUN_TIMEOUT = 180      # 3 minutes for subsequent generations
MAX_REQUEST_AGE = 300       # 5 minutes max age before cleanup (was 600)
CLEANUP_INTERVAL = 60       # Check for old requests every 60 seconds
DELAYED_CLEANUP = 30        # Wait 30 seconds before cleaning completed requests

# ===== QUEUE CONFIGURATION =====
# Request queue management settings
MAX_CONCURRENT_REQUESTS = 1  # Maximum concurrent card generation requests

# ===== MODEL PATHS =====
# Model identifier for image generation
IMAGE_MODEL = 'stabilityai/sdxl-turbo'

# Content generation model
CONTENT_MODEL = "mistral:7b"  # Ollama model for text generation

# ===== GENERATION SETTINGS =====
# Image generation parameters
DEFAULT_IMAGE_STEPS = 1  # For SDXL-Turbo
DEFAULT_GUIDANCE_SCALE = 0.0  # For SDXL-Turbo
# DEFAULT_IMAGE_SIZE = (408, 336)  # Magic card art dimensions

#Reduced for speed
DEFAULT_IMAGE_SIZE = (304, 248)

# Text generation parameters
MAX_PROMPT_TOKENS = 75  # Token limit for prompts
DEFAULT_CARD_WIDTH = 408
DEFAULT_CARD_HEIGHT = 336

# ===== FLYING KEYWORD RESTRICTIONS =====
# Creature types that should be discouraged from getting flying
FLYING_RESTRICTED_TYPES = [
    'human', 'dwarf', 'orc', 'goblin', 'zombie', 'skeleton', 'troll', 'giant',
    'minotaur', 'centaur', 'beast', 'wolf', 'bear', 'boar', 'ox', 'horse',
    'elephant', 'rhino', 'hippo', 'crocodile', 'snake', 'lizard', 'frog',
    'fish', 'octopus', 'crab', 'spider', 'worm', 'slug', 'turtle', 'ape',
    'plant', 'fungus', 'tree', 'wall', 'golem', 'construct', 'vehicle'
]

# Creature types that should be encouraged to get flying
FLYING_ENCOURAGED_TYPES = [
    'bird', 'dragon', 'angel', 'demon', 'devil', 'bat', 'pegasus', 'sphinx',
    'griffin', 'phoenix', 'drake', 'wyvern', 'fairy', 'spirit', 'elemental',
    'djinn', 'efreet', 'gargoyle', 'nightmare', 'specter', 'wraith', 'ghost'
]

# ===== DEBUGGING FLAGS =====
# Control debug output verbosity
DEBUG_QUEUE = False
DEBUG_TEXT_PROCESSING = True
DEBUG_IMAGE_GENERATION = True
DEBUG_CONTENT_GENERATION = True

# ===== GLOBAL STATE TRACKING =====
# Track model loading state for cold vs warm timeout detection
# NOTE: This is mutable state that gets modified at runtime
_models_loaded = {
    'image': False,
    'content': False
}

# Global flag for first job completion (for dynamic loading times)
first_job_completed = False