# Import all configuration constants
from config import (
    USE_CUDA, IMAGE_MODEL, ENABLE_IMAGE_GENERATION,
    COLD_START_TIMEOUT, WARM_RUN_TIMEOUT, MAX_REQUEST_AGE, 
    CLEANUP_INTERVAL, DELAYED_CLEANUP, MAX_CONCURRENT_REQUESTS, DEFAULT_IMAGE_SIZE,
    FLYING_RESTRICTED_TYPES as flying_restricted_types,
    FLYING_ENCOURAGED_TYPES as flying_encouraged_types,
    _models_loaded, first_job_completed
)

# Import text processing modules
from text_processing import (
    parse_abilities, classify_ability, reorder_abilities_properly, reorder_abilities_properly_array,
    strip_non_rules_text, fix_markdown_bullet_points, clean_ability_text,
    clean_ability_arrays, format_ability_newlines, smart_split_by_periods,
    clean_ability_quotes, generate_creature_stats, should_generate_asterisk_pt,
    choose_asterisk_pattern, generate_asterisk_stats, validate_asterisk_abilities
)

# Import prompt generation utilities
from prompt_generator import (
    estimate_tokens, truncate_prompt_smartly, generate_color_palette,
    generate_art_type_context, build_enhanced_art_prompt
)

# Import rules text processing utilities
from rules_text_processor import (
    process_card_description_text, limit_creature_active_abilities, 
    remove_typeline_contamination, ensure_periods_on_abilities, 
    sanitize_planeswalker_abilities, sanitize_spell_abilities, 
    sanitize_land_abilities, sanitize_permanent_abilities, 
    apply_universal_complexity_limits, validate_rules_text
)

# Import card content generation
from card_content_generator import createCardContent

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import requests
import base64
import io
import json
import re
from PIL import Image, ImageDraw
import tempfile
import ollama
# Try to import diffusers, but don't fail if it's not available
try:
    from diffusers import AutoPipelineForText2Image
    DIFFUSERS_AVAILABLE = True
    print("[SUCCESS] diffusers imported successfully")
except ImportError as e:
    print(f"[WARNING] diffusers import failed: {e}")
    print("[INFO] Image generation will be disabled, but text-only generation will work")
    AutoPipelineForText2Image = None
    DIFFUSERS_AVAILABLE = False
print(f"[DEBUG] Python executable: {sys.executable}")
print(f"[DEBUG] Python version: {sys.version}")
print(f"[DEBUG] Python path: {sys.path[:3]}...")  # Show first 3 paths
try:
    import torch
    print(f"[SUCCESS] torch imported successfully: {torch.__version__}")
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"[WARNING] torch import failed: {e}")
    print("[INFO] Torch not available - image generation will be disabled")
    torch = None
    TORCH_AVAILABLE = False
import threading
import concurrent.futures
from card_renderer import card_renderer
import queue
import time
import uuid

# Global queuing system for handling concurrent requests
class RequestQueue:
    def __init__(self):
        self.queue = queue.Queue()
        self.active_requests = {}
        self.max_concurrent = MAX_CONCURRENT_REQUESTS
        self.current_concurrent = 0
        self.lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        
        # Start cleanup thread for periodic maintenance
        self.cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self.cleanup_thread.start()
        
        print(f"[INIT] Request queue initialized with max {MAX_CONCURRENT_REQUESTS} concurrent requests")
    
    def add_request(self, request_id, process_func, *args, **kwargs):
        """Add a request to the queue"""
        import time
        request_item = {
            'id': request_id,
            'func': process_func,
            'args': args,
            'kwargs': kwargs,
            'result': None,
            'error': None,
            'completed': False,
            'started': False,
            'created_at': time.time(),
            'started_at': None
        }
        
        with self.lock:
            self.active_requests[request_id] = request_item
        
        self.queue.put(request_item)
        return request_id
    
    def get_status(self, request_id):
        """Get the status of a request"""
        import time
        with self.lock:
            if request_id in self.active_requests:
                req = self.active_requests[request_id]
                current_time = time.time()
                
                # Check for timeout using global config
                if current_time - req['created_at'] > MAX_REQUEST_AGE:
                    if not req['completed']:
                        req['completed'] = True
                        req['error'] = f'Request timed out after {MAX_REQUEST_AGE // 60} minutes'
                        print(f"⏰ Request {request_id} timed out after {MAX_REQUEST_AGE // 60} minutes")
                        # Clean up from active requests after timeout
                        if req['started']:
                            self.current_concurrent = max(0, self.current_concurrent - 1)
                        # CRITICAL: Remove from active_requests to prevent memory leak
                        print(f"🧹 Cleaning up timed-out request {request_id} from active_requests")
                        # Return timeout result and clean up immediately
                        del self.active_requests[request_id]
                        timeout_minutes = MAX_REQUEST_AGE // 60
                        return {
                            'status': 'completed',
                            'result': None,
                            'error': f'Request timed out after {timeout_minutes} minutes'
                        }
                
                if req['completed']:
                    # Return result and schedule cleanup
                    result = {
                        'status': 'completed',
                        'result': req['result'],
                        'error': req['error']
                    }
                    # Clean up completed requests after a short delay to allow client retrieval
                    import threading
                    def delayed_cleanup():
                        import time
                        time.sleep(DELAYED_CLEANUP)  # Wait for client to retrieve result
                        with self.lock:
                            if request_id in self.active_requests:
                                del self.active_requests[request_id]
                                print(f"🧹 Cleaned up completed request {request_id} from active_requests")
                    threading.Thread(target=delayed_cleanup, daemon=True).start()
                    return result
                elif req['started']:
                    return {
                        'status': 'processing',
                        'message': f'Request is being processed. Active: {self.current_concurrent}/{self.max_concurrent}'
                    }
                else:
                    return {
                        'status': 'queued',
                        'message': f'Request is queued. Position: {self.queue.qsize()}, Active: {self.current_concurrent}/{self.max_concurrent}'
                    }
            else:
                return {'status': 'not_found', 'error': 'Request ID not found'}
    
    def _process_queue(self):
        """Worker thread to process queued requests"""
        while True:
            try:
                # Wait for a request
                request_item = self.queue.get(timeout=1)
                
                # Wait until we have capacity
                while True:
                    with self.lock:
                        if self.current_concurrent < self.max_concurrent:
                            self.current_concurrent += 1
                            request_item['started'] = True
                            request_item['started_at'] = time.time()
                            break
                    time.sleep(0.1)  # Wait a bit before checking again
                
                # Process the request
                try:
                    result = request_item['func'](*request_item['args'], **request_item['kwargs'])
                    request_item['result'] = result
                    # Mark first job as completed globally (for dynamic loading times)
                    global first_job_completed
                    if not first_job_completed:
                        first_job_completed = True
                        
                except Exception as e:
                    request_item['error'] = str(e)
                    print(f"❌ Request {request_item['id']} failed: {e}")
                
                # Mark as completed and free up capacity
                with self.lock:
                    request_item['completed'] = True
                    self.current_concurrent -= 1
                
                self.queue.task_done()
                
            except queue.Empty:
                continue  # No requests to process
            except Exception as e:
                print(f"Queue worker error: {e}")
    
    def _periodic_cleanup(self):
        """Periodically clean up old requests to prevent memory leaks"""
        import time
        while True:
            try:
                time.sleep(CLEANUP_INTERVAL)  # Check periodically
                current_time = time.time()
                cleanup_count = 0
                
                with self.lock:
                    # Find requests older than 15 minutes to clean up
                    to_remove = []
                    for request_id, req in self.active_requests.items():
                        age = current_time - req['created_at']
                        if age > 900:  # 15 minutes
                            to_remove.append(request_id)
                            if req.get('started') and not req.get('completed'):
                                # Free up concurrent slot if needed
                                self.current_concurrent = max(0, self.current_concurrent - 1)
                    
                    # Remove old requests
                    for request_id in to_remove:
                        del self.active_requests[request_id]
                        cleanup_count += 1
                
                if cleanup_count > 0:
                    print(f"🧹 Periodic cleanup removed {cleanup_count} old requests from queue")
                    
            except Exception as e:
                print(f"Periodic cleanup error: {e}")

# Initialize global request queue
request_queue = RequestQueue()  # Allow max 2 concurrent card generations

# Global state tracking for first job completion (for dynamic loading times)
# first_job_completed is now imported from config

def process_card_generation(prompt, width, height, original_card_data):
    """
    Process a card generation request - wrapper function for the queue
    """
    import time
    start_time = time.time()
    
    # Initialize timing variables for safety
    image_generation_time = 0.0
    content_generation_time = 0.0
    cleanup_time = 0.0
    
    try:
        print(f"🎨 Starting queued card generation for: {prompt}")
        
        # Run image and content generation in parallel
        image_start_time = time.time()
        content_start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            print(f"🚀 Submitting parallel tasks: image and content generation")
            # Submit both tasks to run in parallel with card data
            image_future = executor.submit(createCardImage, prompt, width, height, original_card_data)
            content_future = executor.submit(createCardContent, prompt, original_card_data)
            
            # Wait for both to complete and get results
            # Use cold start timeout for first run, warm timeout for subsequent runs
            image_timeout = COLD_START_TIMEOUT if not _models_loaded['image'] else WARM_RUN_TIMEOUT
            content_timeout = COLD_START_TIMEOUT if not _models_loaded['content'] else WARM_RUN_TIMEOUT
            
            print(f"⏳ Waiting for image generation (timeout: {image_timeout}s, {'cold start' if not _models_loaded['image'] else 'warm run'})...")
            try:
                image_data = image_future.result(timeout=image_timeout)
                image_end_time = time.time()
                image_generation_time = image_end_time - image_start_time
                print(f"✅ Image generation completed in {image_generation_time:.2f} seconds")
                # Mark image model as loaded for future warm runs
                _models_loaded['image'] = True
            except concurrent.futures.TimeoutError:
                image_end_time = time.time()
                image_generation_time = image_end_time - image_start_time
                print(f"⏰ Image generation timed out after {image_generation_time:.2f} seconds ({image_timeout}s limit)")
                # Cancel both futures to stop all background processing
                image_future.cancel()
                content_future.cancel()
                print(f"🚫 Cancelled entire request due to image timeout")
                # Fail the entire request immediately on timeout
                raise Exception(f"Request cancelled: Image generation timed out after {image_timeout} seconds. Please try again.")
            except Exception as e:
                image_end_time = time.time()
                image_generation_time = image_end_time - image_start_time
                print(f"❌ Image generation failed after {image_generation_time:.2f} seconds: {e}")
                import traceback
                traceback.print_exc()
                image_data = None
            
            print(f"⏳ Waiting for content generation (timeout: {content_timeout}s, {'cold start' if not _models_loaded['content'] else 'warm run'})...")
            try:
                generated_card_text = content_future.result(timeout=content_timeout)
                content_end_time = time.time()
                content_generation_time = content_end_time - content_start_time
                print(f"✅ Content generation completed in {content_generation_time:.2f} seconds")
                print(f"📝 Generated content preview: {repr(generated_card_text[:100]) if generated_card_text else 'None'}...")
                print(f"🔍 Content generation full result: {repr(generated_card_text)}")
                print(f"🔍 Content type: {type(generated_card_text)}")
                print(f"🔍 Content length: {len(generated_card_text) if generated_card_text else 0}")
                # Mark content model as loaded for future warm runs
                _models_loaded['content'] = True
            except concurrent.futures.TimeoutError:
                content_end_time = time.time()
                content_generation_time = content_end_time - content_start_time
                print(f"⏰ Content generation timed out after {content_generation_time:.2f} seconds ({content_timeout}s limit)")
                # Cancel both futures to stop all background processing
                image_future.cancel()
                content_future.cancel()
                print(f"🚫 Cancelled entire request due to content timeout")
                # Fail the entire request immediately on timeout
                raise Exception(f"Request cancelled: Content generation timed out after {content_timeout} seconds. Please try again.")
            except Exception as e:
                content_end_time = time.time()
                content_generation_time = content_end_time - content_start_time
                print(f"❌ Content generation failed after {content_generation_time:.2f} seconds: {e}")
                import traceback
                traceback.print_exc()
                generated_card_text = None
        
        # Generate complete card image using renderer
        card_image_data = None
        cleanup_start_time = time.time()
        try:
            print("🖼️ Starting cleanup and card rendering...")
            
            # Step 1: Text processing and parsing using centralized function
            text_processing_start = time.time()
            print("  📝 Step 1: Processing text data...")
            
            # Use centralized text processing function
            updated_card_data = process_card_description_text(original_card_data, generated_card_text)
            
            if not generated_card_text:
                print("No card text found, using original description")
                if 'description' not in updated_card_data:
                    updated_card_data['description'] = "Generated card rules text"
            
            text_processing_time = time.time() - text_processing_start
            print(f"   📝 Text processing: {text_processing_time:.2f}s")
            print(f"🔍 Final updated_card_data keys: {list(updated_card_data.keys())}")
            print(f"🔍 Final description: {repr(updated_card_data.get('description', 'NO DESCRIPTION'))}")
            print(f"🔍 Final name: {repr(updated_card_data.get('name', 'NO NAME'))}")
            print(f"🔍 Final flavorText: {repr(updated_card_data.get('flavorText', 'NO FLAVOR'))}")
            
            # Step 2: Stats generation if needed
            stats_generation_start = time.time()
            stats_generated = False
            if (updated_card_data.get('type', '').lower().find('creature') != -1 and 
                (not updated_card_data.get('power') or not updated_card_data.get('toughness'))):
                print("🎯 Creature missing power/toughness - generating stats...")
                generated_stats = generate_creature_stats(updated_card_data)
                if generated_stats:
                    updated_card_data['power'] = generated_stats['power']
                    updated_card_data['toughness'] = generated_stats['toughness']
                    print(f"✅ Generated creature stats: {generated_stats['power']}/{generated_stats['toughness']}")
                    stats_generated = True
            
            # Step 2.5: Vehicle crew cost generation
            vehicle_crew_generated = False
            type_line = updated_card_data.get('typeLine', '').lower()
            if 'vehicle' in type_line and 'artifact' in type_line:
                existing_description = updated_card_data.get('description', '')
                if not existing_description or 'crew' not in existing_description.lower():
                    print("🚗 Vehicle missing crew cost - generating crew ability...")
                    crew_cost = generate_vehicle_crew_cost(updated_card_data)
                    if crew_cost:
                        # Add crew cost to bottom of description (with other active abilities)
                        crew_text = f"Crew {crew_cost}"
                        if existing_description:
                            updated_card_data['description'] = f"{existing_description}\n{crew_text}"
                        else:
                            updated_card_data['description'] = crew_text
                        print(f"✅ Generated vehicle crew cost: Crew {crew_cost}")
                        vehicle_crew_generated = True
            
            stats_generation_time = time.time() - stats_generation_start
            if stats_generated or vehicle_crew_generated:
                generated_items = []
                if stats_generated:
                    generated_items.append("creature P/T")
                if vehicle_crew_generated:
                    generated_items.append("vehicle crew cost")
                print(f"   📊 Stats generation: {stats_generation_time:.2f}s ({', '.join(generated_items)})")
            
            # Step 3: Card image rendering
            rendering_start = time.time()
            card_image_data = card_renderer.generate_card_image(updated_card_data, image_data)
            rendering_time = time.time() - rendering_start
            print(f"   🎨 Card rendering: {rendering_time:.2f}s")
            cleanup_end_time = time.time()
            cleanup_time = cleanup_end_time - cleanup_start_time
            
            if card_image_data:
                print(f"✅ Complete card image generated successfully in {cleanup_time:.2f} seconds")
            else:
                print(f"❌ Failed to generate complete card image after {cleanup_time:.2f} seconds")
        except Exception as e:
            cleanup_end_time = time.time()
            cleanup_time = cleanup_end_time - cleanup_start_time
            print(f"❌ Error generating complete card image after {cleanup_time:.2f} seconds: {e}")
        
        # Build response with detailed timing
        end_time = time.time()
        total_generation_time = end_time - start_time
        
        # Create comprehensive timing breakdown
        print(f"\n🕐 GENERATION TIMING BREAKDOWN:")
        print(f"   🖼️  Image Model: {image_generation_time:.2f}s")
        print(f"   🧠 Content Model: {content_generation_time:.2f}s") 
        print(f"   🧹 Cleanup & Rendering: {cleanup_time:.2f}s")
        print(f"   ⏱️  Total Pipeline: {total_generation_time:.2f}s")
        
        # Calculate model vs cleanup percentage
        model_time = image_generation_time + content_generation_time
        cleanup_percentage = (cleanup_time / total_generation_time) * 100 if total_generation_time > 0 else 0
        model_percentage = (model_time / total_generation_time) * 100 if total_generation_time > 0 else 0
        
        print(f"   📊 Models: {model_percentage:.1f}% | Cleanup: {cleanup_percentage:.1f}%")
        
        if image_data is None and generated_card_text is None:
            # Determine if this was due to timeout or other failure
            timeout_msg = ""
            if (image_generation_time >= (image_timeout - 1)) or (content_generation_time >= (content_timeout - 1)):
                timeout_msg = " (timeout occurred)"
            raise Exception(f'Both image and content generation failed{timeout_msg}. Please try again.')
        elif image_data is None:
            timeout_msg = " (timeout occurred)" if image_generation_time >= (image_timeout - 1) else ""
            print(f"⚠️ Warning: Image generation failed{timeout_msg}, returning content only")
            return {
                'cardData': generated_card_text,
                'imageData': None,
                'card_image': card_image_data,
                'warning': 'Image generation not available',
                'generation_time': total_generation_time
            }
        elif generated_card_text is None:
            timeout_msg = " (timeout occurred)" if content_generation_time >= (content_timeout - 1) else ""
            print(f"⚠️ Warning: Content generation failed{timeout_msg}, returning image only")
            return {
                'cardData': None,
                'imageData': image_data,
                'card_image': card_image_data,
                'warning': 'Content generation failed',
                'generation_time': total_generation_time
            }
        else:
            print("🎉 Both image and content generated successfully!")
            return {
                'cardData': generated_card_text,
                'imageData': image_data,
                'card_image': card_image_data,
                'generation_time': total_generation_time
            }
            
    except Exception as e:
        print(f"❌ Card generation failed: {e}")
        raise e

# Initialize image generation pipeline
image_pipeline = None
image_pipeline_loading = False

def get_image_pipeline():
    """Lazy load the SDXL-Turbo pipeline for image generation"""
    global image_pipeline, image_pipeline_loading
    
    if not DIFFUSERS_AVAILABLE or not TORCH_AVAILABLE:
        print("[ERROR] diffusers or torch not available - image generation disabled")
        image_pipeline = False
        return None
    
    if image_pipeline is not None:
        return image_pipeline
        
    if image_pipeline_loading:
        return None  # Already loading, avoid concurrent loads
    
    if torch is None:
        print("❌ torch is None - cannot load image generation model")
        print(f"🔍 Debug: torch variable = {torch}")
        print(f"🔍 Debug: trying to import torch again...")
        try:
            import torch as torch_test
            print(f"✅ torch re-import successful: {torch_test.__version__}")
            globals()['torch'] = torch_test
        except Exception as e:
            print(f"❌ torch re-import failed: {e}")
            image_pipeline = False  # Mark as failed
            return None
    
    try:
        print(f"🔍 Torch version: {torch.__version__}")
        print(f"🔍 Torch CUDA version: {torch.version.cuda}")
        cuda_available = torch.cuda.is_available()
        print(f"🔍 CUDA available: {cuda_available}")
        if cuda_available:
            print(f"🔍 GPU name: {torch.cuda.get_device_name(0)}")
        else:
            print("🔍 No CUDA GPU detected - will use CPU")


        image_pipeline_loading = True
        import time
        start_time = time.time()
        print("Loading SDXL-Turbo model... (this may take several minutes on first run)")
        
        # Device selection based on USE_CUDA toggle
        if USE_CUDA and torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16  # Use half precision for faster GPU inference
            print("🚀 CUDA GPU mode enabled! Using GPU acceleration")
            if cuda_available:
                print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
                print(f"🎯 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            if not USE_CUDA:
                print("🖥️  CPU mode selected (USE_CUDA = False) - slower but won't stress GPU")
            else:
                print("⚠️  CUDA not available, falling back to CPU mode")
        
        print("🔄 Loading pipeline components...")
        print(f"🎯 Using device: {device}")
        print(f"🎯 Using dtype: {torch_dtype}")
        
        # Load SDXL-Turbo model
        print(f"🎯 Loading SDXL-Turbo model for high-quality image generation")
        print(f"🔍 About to call AutoPipelineForText2Image.from_pretrained...")
        
        try:
            image_pipeline = AutoPipelineForText2Image.from_pretrained(
                IMAGE_MODEL,
                torch_dtype=torch_dtype,
                variant="fp16" if device == "cuda" else None,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True
            )
            print(f"✅ Pipeline loaded successfully!")
            
            # Store SDXL-Turbo generation parameters
            image_pipeline._generation_steps = 1
            image_pipeline._generation_guidance = 0.0
        except Exception as pipeline_error:
            print(f"❌ Pipeline loading failed: {pipeline_error}")
            print(f"🔍 Pipeline error type: {type(pipeline_error)}")
            image_pipeline_loading = False
            image_pipeline = False
            return None
        
        # Move to device after loading
        print(f"🔄 Moving pipeline to device: {device}")
        try:
            image_pipeline.to(device)
            print(f"✅ Pipeline moved to {device} successfully!")
        except Exception as device_error:
            print(f"❌ Failed to move pipeline to {device}: {device_error}")
            image_pipeline_loading = False
            image_pipeline = False
            return None
        
        # Enable GPU memory optimizations for CUDA only
        if device == "cuda" and USE_CUDA:
            print("🔧 Enabling GPU memory optimizations...")
            # Enable memory efficient attention
            try:
                image_pipeline.enable_attention_slicing()
            except Exception as e:
                print(f"⚠️ Attention slicing failed: {e}")
            
            # Enable VAE slicing for memory efficiency
            print("🔧 Enabling VAE slicing...")
            try:
                image_pipeline.vae.enable_slicing()
            except Exception as e:
                print(f"⚠️ VAE slicing failed: {e}")        
            
            try:
                # Enable VAE slicing for lower memory usage
                image_pipeline.enable_vae_slicing()
                print("✅ VAE slicing enabled")
            except Exception as e:
                print(f"⚠️ VAE slicing failed: {e}")
            
            # Compile model for faster inference (PyTorch 2.0+)
            # Disabled due to Triton dependency issues on Windows
            try:
                if hasattr(torch, 'compile'):
                    print("⚠️ torch.compile available but disabled (can cause slowdowns on Windows)")
                    # Skip torch.compile as it can cause massive performance regressions
                    # image_pipeline.unet = torch.compile(image_pipeline.unet, mode="default", fullgraph=False)
                else:
                    print("⚠️ torch.compile not available in this PyTorch version")
            except Exception as e:
                print(f"⚠️ Compilation check failed: {e}")
                
            print("✅ GPU optimizations enabled for faster inference")
        else:
            print("📝 CPU mode - no additional optimizations applied")
        
        elapsed_time = time.time() - start_time
        print(f"🎉 Image generation model loaded successfully on {device}! (took {elapsed_time:.1f} seconds)")
        print(f"🔍 Final pipeline object: {type(image_pipeline)}")
        image_pipeline_loading = False
        return image_pipeline
        
    except Exception as e:
        print(f"❌ Failed to load image generation model: {e}")
        print(f"🔍 Exception type: {type(e)}")
        import traceback
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        image_pipeline_loading = False
        image_pipeline = False
    finally:
        image_pipeline_loading = False

app = Flask(__name__)
CORS(app, 
     origins=["*"],  # Allow all origins for ngrok + S3
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "ngrok-skip-browser-warning", "Accept", "Cache-Control"],
     max_age=86400,  # Cache preflight for 24 hours
     supports_credentials=False)

def add_ngrok_headers(response):
    """Add ngrok-specific headers to response object (Flask-CORS handles CORS headers)"""
    # Add ngrok-specific headers only (avoid duplicates from after_request)
    if 'ngrok-skip-browser-warning' not in response.headers:
        response.headers.add('ngrok-skip-browser-warning', 'any')
    return response

@app.after_request
def after_request(response):
    """Ensure all responses have CORS headers for HTTPS/ngrok compatibility"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,ngrok-skip-browser-warning,Accept,Cache-Control')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Max-Age', '86400')
    response.headers.add('ngrok-skip-browser-warning', 'any')
    return response

def createCardImage(prompt, width=408, height=336, card_data=None):
    """
    Generate card image using SDXL-Turbo with color-aware prompts
    Returns the image as base64 encoded string
    """
    print(f"=== STARTING IMAGE GENERATION ===")
    print(f"Prompt: {prompt}")
    print(f"Dimensions: {width}x{height}")
    
    # Import required modules at function start
    import io
    import base64
    import time
    from PIL import Image, ImageDraw
    import numpy as np
    
    try:
        # Check if image generation is disabled  
        if not ENABLE_IMAGE_GENERATION:
            print("🚫 Image generation disabled (placeholder mode)")
            # Return a simple gray image placeholder
            placeholder_image = Image.new('RGB', (width, height), color=(50, 50, 50))
            # Convert to base64
            buffer = io.BytesIO()
            placeholder_image.save(buffer, format='PNG')
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{image_data}"
        
        pipeline = get_image_pipeline()
        
        if pipeline and pipeline is not False:
            # Generate image using SDXL-Turbo
            print(f"Generating AI image for: {prompt}")
            
            # Build enhanced art prompt using the prompt generator module
            final_prompt = build_enhanced_art_prompt(prompt, card_data)
            
            # Performance timing for image generation
            inference_start = time.time()
            if USE_CUDA and torch.cuda.is_available():
                print(f"🚀 Starting SDXL-Turbo inference on GPU (should take 2-4 seconds)...")
            else:
                print(f"🖥️  Starting SDXL-Turbo inference on CPU (will take 30-60 seconds)...")
            
            # Use model-specific generation parameters
            steps = getattr(pipeline, '_generation_steps', 1)
            guidance = getattr(pipeline, '_generation_guidance', 0.0)
            print(f"🎯 Using {steps} steps, guidance_scale={guidance}")
            
            image = pipeline(
                prompt=final_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,               
                height=height              
            ).images[0]
            
            inference_time = time.time() - inference_start
            print(f"⚡ Image model inference completed in {inference_time:.2f} seconds")
            
            # Debug the generated image
            print(f"🔍 Generated image size: {image.size}")
            print(f"🔍 Generated image mode: {image.mode}")
            
        else:
            # Fallback to placeholder image
            print(f"Using placeholder image for: {prompt}")
            image = Image.new('RGB', (width, height), color='#2c3e50')
        
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_data = buffer.getvalue()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        return img_base64
    
    except Exception as e:
        print(f"Error in createCardImage: {e}")
        print("Falling back to placeholder image")
        try:
            # Fallback to placeholder
            image = Image.new('RGB', (width, height), color='#2c3e50')
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_data = buffer.getvalue()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            return img_base64
        except:
            return None

def calculate_card_power_level(card_data: dict) -> float:
    """
    Calculate comprehensive power level of a card considering multiple factors
    Returns float from 0.0 (weak) to 10.0 (extremely powerful)
    """
    try:
        # Extract basic stats
        power = int(card_data.get('power', 0)) if card_data.get('power', '').isdigit() else 0
        toughness = int(card_data.get('toughness', 0)) if card_data.get('toughness', '').isdigit() else 0
        cmc = card_data.get('cmc', 0)
        mana_cost = card_data.get('manaCost', '')
        
        # Base power level from stats
        stats_total = power + toughness
        base_power = stats_total * 0.5  # Base scaling factor
        
        # CMC efficiency factor (higher stats for lower CMC = higher power level)
        if cmc > 0:
            efficiency = stats_total / cmc
            efficiency_bonus = max(0, efficiency - 1.5) * 2  # Bonus for above-curve stats
        else:
            efficiency_bonus = 0
        
        # Count colored mana pips for commitment penalty/bonus
        colored_pips = 0
        if mana_cost:
            import re
            # Count single colored mana symbols
            colored_pips += len(re.findall(r'\{[WUBRG]\}', mana_cost))
            # Count hybrid mana symbols (count as 1.5 pips each)
            hybrid_matches = re.findall(r'\{[WUBRG]/[WUBRG]\}', mana_cost)
            colored_pips += len(hybrid_matches) * 1.5
        
        # Color commitment factor (more colors = slightly higher power level potential)
        color_bonus = min(colored_pips * 0.3, 2.0)  # Cap at +2.0
        
        # Power vs Toughness distribution factor
        power_focus_bonus = 0
        if power > 0 and toughness > 0:
            total_stats = power + toughness
            power_ratio = power / total_stats
            # Favor aggressive power-heavy creatures slightly
            if power_ratio > 0.6:
                power_focus_bonus = 0.5
            elif power_ratio > 0.75:
                power_focus_bonus = 1.0
        
        # Calculate final power level
        power_level = base_power + efficiency_bonus + color_bonus + power_focus_bonus
        
        # Normalize to 0-10 scale and cap
        power_level = max(0.0, min(power_level, 10.0))
        
        print(f"💪 Power level calculation: P/T {power}/{toughness}, CMC {cmc}, Colored pips: {colored_pips:.1f}")
        print(f"💪 Components: Base {base_power:.1f} + Efficiency {efficiency_bonus:.1f} + Color {color_bonus:.1f} + Power focus {power_focus_bonus:.1f} = {power_level:.2f}")
        
        return power_level
        
    except Exception as e:
        print(f"❌ Error calculating power level: {e}")
        # Fallback: moderate power level
        return 3.0

def generate_vehicle_crew_cost(card_data: dict) -> int:
    """
    Generate appropriate crew cost for vehicles based on comprehensive power level
    Returns crew cost as integer (1-5)
    """
    try:
        # Calculate comprehensive power level
        power_level = calculate_card_power_level(card_data)
        
        # Extract basic stats for logging
        power = int(card_data.get('power', 0)) if card_data.get('power', '').isdigit() else 0
        toughness = int(card_data.get('toughness', 0)) if card_data.get('toughness', '').isdigit() else 0
        cmc = card_data.get('cmc', 0)
        
        print(f"🚗 Vehicle analysis: P/T {power}/{toughness}, CMC {cmc}, Power level: {power_level:.2f}")
        
        # Base crew cost on comprehensive power level
        if power_level <= 2.0:
            crew_cost = 1  # Weak vehicles
        elif power_level <= 3.5:
            crew_cost = 2  # Moderate vehicles
        elif power_level <= 5.0:
            crew_cost = 3  # Strong vehicles
        elif power_level <= 7.0:
            crew_cost = 4  # Very strong vehicles
        else:
            crew_cost = 5  # Extremely powerful vehicles
        
        # Ensure minimum crew 1, maximum crew 5
        crew_cost = max(1, min(crew_cost, 5))
        
        print(f"🚗 Generated crew cost: {crew_cost} (based on power level {power_level:.2f})")
        return crew_cost
        
    except Exception as e:
        print(f"❌ Error generating vehicle crew cost: {e}")
        # Fallback: crew 2 (balanced default)
        return 2

# createCardContent function moved to card_content_generator

@app.route('/api/v1/create_card', methods=['POST', 'OPTIONS'])
def create_card():
    """
    Main endpoint that uses queue but returns synchronously for frontend compatibility
    """
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        print("Handling OPTIONS preflight request")
        response = jsonify({'status': 'ok'})
        return add_ngrok_headers(response)
    
    try:
        # Get the request data
        data = request.get_json()
        
        if not data:
            print("ERROR: No JSON data provided")
            response = jsonify({'error': 'No JSON data provided'})
            return add_ngrok_headers(response), 400
        
        # Extract prompt from request
        prompt = data.get('prompt', '')
        if not prompt:
            response = jsonify({'error': 'No prompt provided'})
            return add_ngrok_headers(response), 400
        
        # Optional parameters - default to Magic card art box aspect ratio
        width = DEFAULT_IMAGE_SIZE[0]
        height = DEFAULT_IMAGE_SIZE[1]
        
        # Extract card data for enhanced prompting
        original_card_data = data.get('cardData', {})
        
        print(f"🔄 Card generation request: {prompt}")
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Add request to queue
        request_queue.add_request(
            request_id, 
            process_card_generation, 
            prompt, 
            width, 
            height, 
            original_card_data
        )
        
        # Wait for completion (synchronous behavior for frontend compatibility)
        print(f"⏳ Request added to queue, waiting for request {request_id} to complete...")
        max_wait_time = MAX_REQUEST_AGE  # Use global timeout configuration
        start_wait = time.time()
        
        while True:
            status_info = request_queue.get_status(request_id)
            
            if status_info['status'] == 'completed':
                if status_info.get('error'):
                    print(f"❌ Request failed: {status_info['error']}")
                    response = jsonify({'error': status_info['error']})
                    return add_ngrok_headers(response), 500
                else:
                    print(f"✅ Request completed successfully, sending response to client")
                    # Return in the format frontend expects
                    response = jsonify(status_info['result'])
                    return add_ngrok_headers(response), 200
            
            elif time.time() - start_wait > max_wait_time:
                print(f"⏰ Request timed out after {max_wait_time} seconds")
                timeout_minutes = max_wait_time // 60
                response = jsonify({'error': f'Request timed out - took longer than {timeout_minutes} minutes'})
                return add_ngrok_headers(response), 504
            
            else:
                # Still processing, wait a bit
                time.sleep(2)
                continue
                
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        response = jsonify({'error': f'Request processing failed: {str(e)}'})
        return add_ngrok_headers(response), 500

# Regenerate text with existing image endpoint
@app.route('/api/v1/regenerate_card_text', methods=['POST', 'OPTIONS'])
def regenerate_card_text():
    """
    Regenerate card text using existing image data - reuses provided image and generates new text
    """
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        print("Handling OPTIONS preflight request for regenerate text endpoint")
        response = jsonify({'status': 'ok'})
        return add_ngrok_headers(response)
    
    try:
        # Get the request data
        data = request.get_json()
        
        if not data:
            print("ERROR: No JSON data provided")
            response = jsonify({'error': 'No JSON data provided'})
            return add_ngrok_headers(response), 400
        
        # Extract prompt from request
        prompt = data.get('prompt', '')
        if not prompt:
            response = jsonify({'error': 'No prompt provided'})
            return add_ngrok_headers(response), 400
        
        # Extract existing image data
        existing_image_data = data.get('imageData', '')
        if not existing_image_data:
            response = jsonify({'error': 'No existing image data provided'})
            return add_ngrok_headers(response), 400
        
        # Remove data URL prefix if present
        if existing_image_data.startswith('data:image/'):
            existing_image_data = existing_image_data.split(',')[1]
        
        # Extract card data for enhanced prompting
        original_card_data = data.get('cardData', {})
        
        print(f"[REGENERATE] Text regeneration request: {prompt}")
        print(f"[REGENERATE] Using existing image data: {len(existing_image_data)} characters")
        
        # Generate new content directly (no queue needed for text regeneration)
        import time
        start_time = time.time()
        
        try:
            print(f"[REGENERATE] Starting text regeneration with existing image...")
            generated_card_text = createCardContent(prompt, original_card_data)
            content_generation_time = time.time() - start_time
            print(f"[REGENERATE] Text regeneration completed in {content_generation_time:.2f} seconds")
            
            # Process the generated text
            if generated_card_text:
                updated_card_data = process_card_description_text(original_card_data, generated_card_text)
                
                # Generate complete card image using existing artwork
                try:
                    print("[REGENERATE] Rendering complete card with new text and existing artwork...")
                    rendering_start = time.time()
                    
                    # Decode the existing image data
                    import base64
                    image_data_bytes = base64.b64decode(existing_image_data)
                    
                    # Generate complete card image using renderer with existing artwork
                    card_image_data = card_renderer.generate_card_image(updated_card_data, existing_image_data)
                    
                    rendering_time = time.time() - rendering_start
                    print(f"[REGENERATE] Card rendering completed in {rendering_time:.2f} seconds")
                    
                    # Return the response with new text and complete card image
                    total_time = time.time() - start_time
                    response_data = {
                        'cardData': json.dumps(updated_card_data),
                        'card_image': card_image_data,  # Complete card with new text
                        'generationTime': total_time,
                        'message': 'Text regenerated successfully with existing artwork'
                    }
                    
                    print(f"[REGENERATE] Returning regeneration response: {len(str(response_data))} bytes")
                    response = jsonify(response_data)
                    return add_ngrok_headers(response)
                    
                except Exception as render_error:
                    print(f"[ERROR] Card rendering failed: {render_error}")
                    # Fall back to just returning the text data
                    response_data = {
                        'cardData': json.dumps(updated_card_data),
                        'generationTime': content_generation_time,
                        'message': 'Text regenerated successfully (card rendering failed)'
                    }
                    response = jsonify(response_data)
                    return add_ngrok_headers(response)
            else:
                print("[ERROR] No content generated during regeneration")
                response = jsonify({'error': 'Failed to regenerate content'})
                return add_ngrok_headers(response), 500
                
        except Exception as e:
            content_generation_time = time.time() - start_time
            print(f"[ERROR] Text regeneration failed after {content_generation_time:.2f} seconds: {e}")
            import traceback
            traceback.print_exc()
            response = jsonify({'error': f'Content regeneration failed: {str(e)}'})
            return add_ngrok_headers(response), 500
        
    except Exception as e:
        print(f"[ERROR] Error processing regenerate text request: {e}")
        response = jsonify({'error': f'Request processing failed: {str(e)}'})
        return add_ngrok_headers(response), 500

# Text-only endpoint for faster generation (no image generation)
@app.route('/api/v1/create_card_text_only', methods=['POST', 'OPTIONS'])
def create_card_text_only():
    """
    Text-only endpoint that skips image generation for significantly faster response
    """
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        print("Handling OPTIONS preflight request for text-only endpoint")
        response = jsonify({'status': 'ok'})
        return add_ngrok_headers(response)
    
    try:
        # Get the request data
        data = request.get_json()
        
        if not data:
            print("ERROR: No JSON data provided")
            response = jsonify({'error': 'No JSON data provided'})
            return add_ngrok_headers(response), 400
        
        # Extract prompt from request
        prompt = data.get('prompt', '')
        if not prompt:
            response = jsonify({'error': 'No prompt provided'})
            return add_ngrok_headers(response), 400
        
        # Extract card data for enhanced prompting
        original_card_data = data.get('cardData', {})
        
        print(f"📝 Text-only generation request: {prompt}")
        
        # Generate content directly (no queue needed for text-only)
        import time
        start_time = time.time()
        
        try:
            print(f"🚀 Starting text-only content generation...")
            generated_card_text = createCardContent(prompt, original_card_data)
            content_generation_time = time.time() - start_time
            print(f"✅ Text generation completed in {content_generation_time:.2f} seconds")
            
            # Process the generated text
            if generated_card_text:
                updated_card_data = process_card_description_text(original_card_data, generated_card_text)
                
                # Return only the text data (no image data)
                response_data = {
                    'cardData': json.dumps(updated_card_data),
                    'generationTime': content_generation_time,
                    'message': 'Text generated successfully'
                }
                
                print(f"📤 Returning text-only response: {len(str(response_data))} bytes")
                response = jsonify(response_data)
                return add_ngrok_headers(response)
            else:
                print("❌ No content generated")
                response = jsonify({'error': 'Failed to generate content'})
                return add_ngrok_headers(response), 500
                
        except Exception as e:
            content_generation_time = time.time() - start_time
            print(f"❌ Text generation failed after {content_generation_time:.2f} seconds: {e}")
            import traceback
            traceback.print_exc()
            response = jsonify({'error': f'Content generation failed: {str(e)}'})
            return add_ngrok_headers(response), 500
        
    except Exception as e:
        print(f"❌ Error processing text-only request: {e}")
        response = jsonify({'error': f'Request processing failed: {str(e)}'})
        return add_ngrok_headers(response), 500

# Async endpoint for clients that want to poll
@app.route('/api/v1/create_card_async', methods=['POST', 'OPTIONS'])
def create_card_async():
    """
    Async endpoint that queues card generation and returns request_id for polling
    """
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        print("Handling OPTIONS preflight request")
        response = jsonify({'status': 'ok'})
        return add_ngrok_headers(response)
    
    try:
        # Get the request data
        data = request.get_json()
        
        if not data:
            print("ERROR: No JSON data provided")
            response = jsonify({'error': 'No JSON data provided'})
            return add_ngrok_headers(response), 400
        
        # Extract prompt from request
        prompt = data.get('prompt', '')
        if not prompt:
            response = jsonify({'error': 'No prompt provided'})
            return add_ngrok_headers(response), 400
        
        # Optional parameters - default to Magic card art box aspect ratio
        width = data.get('width', 408)
        height = data.get('height', 336)
        
        # Extract card data for enhanced prompting
        original_card_data = data.get('cardData', {})
        
        print(f"📨 Async card generation request: {prompt}")
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Add request to queue
        request_queue.add_request(
            request_id, 
            process_card_generation, 
            prompt, 
            width, 
            height, 
            original_card_data
        )
        
        # Return the request ID for polling
        response_data = {
            'request_id': request_id,
            'status': 'queued',
            'message': 'Your card generation request has been queued. Use the request_id to check status.'
        }
        response = jsonify(response_data)
        return add_ngrok_headers(response), 202  # 202 Accepted
        
    except Exception as e:
        print(f"❌ Error processing async request: {e}")
        response = jsonify({'error': f'Request processing failed: {str(e)}'})
        return add_ngrok_headers(response), 500

@app.route('/api/v1/card_status/<request_id>', methods=['GET', 'OPTIONS'])
def get_card_status(request_id):
    """
    Check the status of a card generation request
    """
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        return add_ngrok_headers(response)
    
    try:
        status_info = request_queue.get_status(request_id)
        response = jsonify(status_info)
        return add_ngrok_headers(response), 200
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        response = jsonify({'error': f'Status check failed: {str(e)}'})
        return add_ngrok_headers(response), 500

@app.route('/api/v1/queue_status', methods=['GET', 'OPTIONS'])
def get_queue_status():
    """Get overall queue status"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        return add_ngrok_headers(response)
    
    try:
        with request_queue.lock:
            status_info = {
                'queue_size': request_queue.queue.qsize(),
                'active_requests': request_queue.current_concurrent,
                'max_concurrent': request_queue.max_concurrent,
                'total_active_requests': len(request_queue.active_requests)
            }
        response = jsonify(status_info)
        return add_ngrok_headers(response), 200
    except Exception as e:
        print(f"❌ Error getting queue status: {e}")
        response = jsonify({'error': f'Queue status failed: {str(e)}'})
        return add_ngrok_headers(response), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint - always returns 200 to indicate server is running"""
    global first_job_completed
    response = jsonify({
        'status': 'healthy',
        'models': {
            'image_model': 'ready',
            'content_model': 'ready'
        },
        'first_job_completed': first_job_completed,
        'message': 'Server is running'
    })
    return add_ngrok_headers(response), 200

@app.route('/test-post', methods=['POST', 'OPTIONS'])
def test_post():
    """Simple test POST endpoint to debug connectivity"""
    print(f"=== TEST POST REQUEST: {request.method} ===")
    print(f"Request headers: {dict(request.headers)}")
    
    if request.method == 'OPTIONS':
        print("Handling OPTIONS for test-post")
        response = jsonify({'status': 'options-ok'})
        return add_ngrok_headers(response)
    
    print("Processing POST for test-post")
    try:
        data = request.get_json()
        response = jsonify({
            'status': 'post-success', 
            'received_data': data,
            'message': 'Simple POST test worked!'
        })
        return add_ngrok_headers(response), 200
    except Exception as e:
        print(f"Error in test-post: {e}")
        response = jsonify({'error': str(e), 'status': 'post-failed'})
        return add_ngrok_headers(response), 500

@app.route('/instant', methods=['POST', 'OPTIONS'])
def instant_response():
    """Instant response endpoint to test if timing is the issue"""
    print(f"=== INSTANT RESPONSE: {request.method} ===")
    if request.method == 'OPTIONS':
        return jsonify({'status': 'options-ok'})
    
    # Return immediately without processing
    return jsonify({'status': 'instant-success', 'timestamp': str(request.args)}), 200

if __name__ == '__main__':
    print("[STARTUP] Starting Flask server with intelligent queuing...")
    print("Available endpoints:")
    print("  POST /api/v1/create_card - Generate card (sync, frontend compatible)")
    print("  POST /api/v1/create_card_text_only - Generate text only (fast)")
    print("  POST /api/v1/regenerate_card_text - Regenerate text with existing image")
    print("  POST /api/v1/create_card_async - Queue card generation (async)")
    print("  GET  /api/v1/card_status/<request_id> - Check async request status")
    print("  GET  /api/v1/queue_status - Get overall queue status")
    print("  POST /api/v1/create_card_sync - Generate card (sync, legacy)")
    print("  GET  /health - Health check")
    print("\n[CONFIG] Queue Configuration:")
    print(f"  - Max concurrent requests: {request_queue.max_concurrent}")
    print("  - All endpoints use queue internally to prevent model overload")
    print("  - Frontend-compatible: /api/v1/create_card works synchronously")
    print("\n[USAGE] Usage:")
    print("Frontend: Use /api/v1/create_card (synchronous, queued internally)")
    print("Advanced: Use /api/v1/create_card_async + polling for true async behavior")
    print("\nExample request body:")
    print('{"prompt": "A mystical dragon card", "width": 408, "height": 336}')
    print("\nNote: SDXL-Turbo model will load on first image request")
    
    # Run with HTTP - ngrok will handle HTTPS termination
    app.run(debug=False, host='0.0.0.0', port=5000)
