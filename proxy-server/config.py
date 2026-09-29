"""
Server configuration toggles. Edit these values, then restart `python app.py`.
"""
from pathlib import Path

# ===== PERFORMANCE TOGGLE =====
# Set to False to force CPU-only mode (slower but won't stress your GPU)
# Set to True to use CUDA if available
USE_CUDA = True

# ===== MODEL SELECTION =====
# "heavy"       - IMAGE_MODEL_ID below (SDXL fine-tune; needs a good GPU)
# "medium"      - runwayml/stable-diffusion-v1-5 (balanced quality/performance)
# "light"       - CompVis/stable-diffusion-v1-4 (lighter, works better on CPU)
# "placeholder" - disable image generation entirely (for testing/debugging)
MODEL_SIZE = "heavy"

# ===== IMAGE GENERATION =====
IMAGE_MODEL_ID = "Lykon/dreamshaper-xl-lightning"
IMAGE_STEPS = 6
IMAGE_GUIDANCE = 2.0
IMAGE_GEN_SIZE = (1088, 896)   # SDXL render size; same 1.214 aspect ratio as the art box
ART_BOX_SIZE = (408, 336)      # size of the art window on the rendered card

# ===== AI NIGHT =====
ADMIN_PIN = "1234"             # change before the event; required for /admin actions
DATA_DIR = Path(__file__).parent / "data"

# ===== TIMEOUT CONFIGURATION =====
# Global timeout settings for all operations (in seconds)
COLD_START_TIMEOUT = 180    # 3 minutes for first-time model loading
WARM_RUN_TIMEOUT = 180      # 3 minutes for subsequent generations
MAX_REQUEST_AGE = 300       # 5 minutes max age before cleanup
CLEANUP_INTERVAL = 60       # Check for old requests every 60 seconds
DELAYED_CLEANUP = 30        # Wait 30 seconds before cleaning completed requests
