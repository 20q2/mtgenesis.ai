"""
Server configuration toggles. Edit these values, then restart `python app.py`.
"""
import os
from pathlib import Path

# ===== PERFORMANCE TOGGLE =====
# Set to False to force CPU-only mode (slower but won't stress your GPU)
# Set to True to use CUDA if available
USE_CUDA = True

# ===== MODEL SELECTION =====
# "placeholder" - disable image generation entirely: solid gray art, torch is never loaded
#                 (for testing/debugging)
# anything else - IMAGE_MODEL_ID below (SDXL fine-tune; needs a good GPU). "heavy" is the
#                 normal value; "medium" and "light" are legacy names that now behave
#                 exactly like "heavy" (the old SD 1.5 / SD 1.4 options were removed).
MODEL_SIZE = "heavy"

# ===== IMAGE GENERATION =====
IMAGE_MODEL_ID = "Lykon/dreamshaper-xl-lightning"
IMAGE_STEPS = 6
IMAGE_GUIDANCE = 1.5           # higher pushes Lightning models toward harsh contrast and blown whites
IMAGE_GEN_SIZE = (1088, 896)   # SDXL render size; same 1.214 aspect ratio as the art box
ART_BOX_SIZE = (408, 336)      # size of the art window on the rendered card
# fp16-safe SDXL VAE: decodes in fp16 instead of upcasting to fp32 (8.3s -> 1.9s per image,
# ~1.5 GB less VRAM, visually identical). None = use the model's own VAE.
IMAGE_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
# Style LoRAs fused into the model at load: (Hugging Face repo, weight file, strength).
# Needs `peft`. Empty = the base model's own look. The oil-painting slider (MIT, 9 MB) at 3
# gives visible brushwork instead of DreamShaper's smooth digital look; compare strengths
# with tools/e2e_art.py --lora.
IMAGE_LORAS: list[tuple[str, str, float]] = [
    ("ntc-ai/SDXL-LoRA-slider.oil-painting", "oil painting.safetensors", 3.0),
]
# Keep each SDXL component in RAM and move it to the GPU only while it runs (peak ~5.6 GB
# instead of ~9.4 GB). Needed on a 12 GB card while Ollama keeps the text model (~5.3 GB) resident:
# without it the driver spills to system RAM and one image takes ~280s instead of ~4s.
# Set False on a 16 GB+ GPU (or with no LLM on the GPU) for ~2s instead of ~4s per image.
IMAGE_CPU_OFFLOAD = True

# ===== RULES TEXT (LLM) =====
# Ollama model that writes rules text (run `ollama pull <model>` first). It shares the GPU
# with SDXL, so keep it around 5 GB. MTG_TEXT_MODEL overrides it (tools/e2e_rules_text.py).
TEXT_MODEL = os.environ.get("MTG_TEXT_MODEL", "qwen3:8b")
TEXT_ATTEMPTS = 3              # regenerate (up to twice) while the cleaned text still has lint errors
TEXT_THINK = os.environ.get("MTG_TEXT_THINK") == "1"  # reasoning models: think before answering (slower)
# Card director (director.py): a brief per card that steers rules text and art.
# MTG_DIRECTOR=0 switches it off (cards are then generated as before); MTG_DIRECTOR_MODEL tries
# another model, but a second resident model needs VRAM that SDXL shares on a 12 GB card.
DIRECTOR_ENABLED = os.environ.get("MTG_DIRECTOR", "1") != "0"
DIRECTOR_MODEL = os.environ.get("MTG_DIRECTOR_MODEL", TEXT_MODEL)
# Art waits for the brief, so the director gets its own short timeout (a hung Ollama costs
# one brief, not minutes per card).
DIRECTOR_TIMEOUT_SECONDS = 20

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
