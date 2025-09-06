import os
# Disable xformers to avoid version conflicts - MUST be set before importing diffusers
os.environ['XFORMERS_DISABLED'] = '1'
os.environ['DISABLE_XFORMERS'] = '1'

# ===== PERFORMANCE TOGGLE =====
# Set to False to force CPU-only mode (slower but won't destroy your GPU)
# Set to True to use CUDA if available (faster but may lag your system)
USE_CUDA = True  # <-- Change this to True when you want GPU mode

# ===== MODEL SELECTION =====
# Choose model based on your hardware capabilities:
# "heavy" - stabilityai/sdxl-turbo (best quality, needs good GPU/lots of RAM)
# "medium" - runwayml/stable-diffusion-v1-5 (balanced quality/performance)
# "light" - CompVis/stable-diffusion-v1-4 (lighter, works better on CPU)
# "placeholder" - disable image generation entirely (for testing/debugging)
MODEL_SIZE = "heavy"  # <-- CUDA enabled! Using SDXL-Turbo for best quality
# ==============================

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
from diffusers import AutoPipelineForText2Image
print(f"🔍 Python executable: {sys.executable}")
print(f"🔍 Python version: {sys.version}")
print(f"🔍 Python path: {sys.path[:3]}...")  # Show first 3 paths
try:
    import torch
    print(f"✅ torch imported successfully: {torch.__version__}")
except ImportError as e:
    print(f"❌ torch import failed: {e}")
    torch = None
import threading
import concurrent.futures
from card_renderer import card_renderer
import queue
import time
import uuid

# Global queuing system for handling concurrent requests
class RequestQueue:
    def __init__(self, max_concurrent=2):
        self.queue = queue.Queue()
        self.active_requests = {}
        self.max_concurrent = max_concurrent
        self.current_concurrent = 0
        self.lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        
        # Start cleanup thread for periodic maintenance
        self.cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self.cleanup_thread.start()
        
        print(f"🚀 Request queue initialized with max {max_concurrent} concurrent requests")
    
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
        print(f"📝 Request {request_id} added to queue. Queue size: {self.queue.qsize()}")
        return request_id
    
    def get_status(self, request_id):
        """Get the status of a request"""
        import time
        with self.lock:
            if request_id in self.active_requests:
                req = self.active_requests[request_id]
                current_time = time.time()
                
                # Check for timeout (10 minutes = 600 seconds)
                if current_time - req['created_at'] > 600:
                    if not req['completed']:
                        req['completed'] = True
                        req['error'] = 'Request timed out after 10 minutes'
                        print(f"⏰ Request {request_id} timed out after 10 minutes")
                        # Clean up from active requests after timeout
                        if req['started']:
                            self.current_concurrent = max(0, self.current_concurrent - 1)
                        # CRITICAL: Remove from active_requests to prevent memory leak
                        print(f"🧹 Cleaning up timed-out request {request_id} from active_requests")
                        # Return timeout result and clean up immediately
                        del self.active_requests[request_id]
                        return {
                            'status': 'completed',
                            'result': None,
                            'error': 'Request timed out after 10 minutes'
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
                        time.sleep(30)  # Wait 30 seconds for client to retrieve result
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
                            print(f"🔄 Starting request {request_item['id']}. Active: {self.current_concurrent}/{self.max_concurrent}")
                            break
                    time.sleep(0.1)  # Wait a bit before checking again
                
                # Process the request
                try:
                    result = request_item['func'](*request_item['args'], **request_item['kwargs'])
                    request_item['result'] = result
                    print(f"✅ Request {request_item['id']} completed successfully")
                except Exception as e:
                    request_item['error'] = str(e)
                    print(f"❌ Request {request_item['id']} failed: {e}")
                
                # Mark as completed and free up capacity
                with self.lock:
                    request_item['completed'] = True
                    self.current_concurrent -= 1
                    print(f"🔓 Request {request_item['id']} finished. Active: {self.current_concurrent}/{self.max_concurrent}")
                
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
                time.sleep(60)  # Check every minute
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
request_queue = RequestQueue(max_concurrent=2)  # Allow max 2 concurrent card generations

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
            print(f"📋 Both futures submitted, waiting for completion...")
            
            # Wait for both to complete and get results
            print(f"⏳ Waiting for image generation (timeout: 600s)...")
            try:
                image_data = image_future.result(timeout=600)  # 10 minute timeout for image
                image_end_time = time.time()
                image_generation_time = image_end_time - image_start_time
                print(f"✅ Image generation completed in {image_generation_time:.2f} seconds")
            except concurrent.futures.TimeoutError:
                image_end_time = time.time()
                image_generation_time = image_end_time - image_start_time
                print(f"⏰ Image generation timed out after {image_generation_time:.2f} seconds (10 minute limit)")
                image_data = None
            except Exception as e:
                image_end_time = time.time()
                image_generation_time = image_end_time - image_start_time
                print(f"❌ Image generation failed after {image_generation_time:.2f} seconds: {e}")
                import traceback
                traceback.print_exc()
                image_data = None
            
            print(f"⏳ Waiting for content generation (timeout: 120s)...")
            try:
                generated_card_text = content_future.result(timeout=120)  # 2 minute timeout for content
                content_end_time = time.time()
                content_generation_time = content_end_time - content_start_time
                print(f"✅ Content generation completed in {content_generation_time:.2f} seconds")
                print(f"📝 Generated content preview: {repr(generated_card_text[:100]) if generated_card_text else 'None'}...")
                print(f"🔍 Content generation full result: {repr(generated_card_text)}")
                print(f"🔍 Content type: {type(generated_card_text)}")
                print(f"🔍 Content length: {len(generated_card_text) if generated_card_text else 0}")
            except concurrent.futures.TimeoutError:
                content_end_time = time.time()
                content_generation_time = content_end_time - content_start_time
                print(f"⏰ Content generation timed out after {content_generation_time:.2f} seconds")
                generated_card_text = None
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
            
            # Step 1: Text processing and parsing
            text_processing_start = time.time()
            print("  📝 Step 1: Processing text data...")
            updated_card_data = original_card_data.copy()
            print(f"🔍 Original card data keys: {list(original_card_data.keys())}")
            print(f"🔍 Original description: {repr(original_card_data.get('description', 'NO DESCRIPTION'))}")
            if generated_card_text:
                print(f"🔧 Processing generated text: {repr(generated_card_text[:100])}...")
                try:
                    # Try to parse structured card data
                    parsed_text = json.loads(generated_card_text)
                    if isinstance(parsed_text, dict):
                        # Update with parsed structured data
                        if 'description' in parsed_text:
                            updated_card_data['description'] = parsed_text['description']
                        if 'name' in parsed_text and parsed_text['name']:
                            updated_card_data['name'] = parsed_text['name']
                        if 'flavorText' in parsed_text:
                            updated_card_data['flavorText'] = parsed_text['flavorText']
                        print(f"Updated card data with parsed structured content")
                    else:
                        # If it's a JSON string, use the string content
                        updated_card_data['description'] = str(parsed_text)
                        print(f"Updated card data with JSON string content")
                except json.JSONDecodeError:
                    # If not JSON, treat as plain description text
                    updated_card_data['description'] = generated_card_text
                    print(f"Updated card data with plain text content")
                
                # Apply text processing and ability reordering to the description
                if 'description' in updated_card_data and updated_card_data['description']:
                    original_text = updated_card_data['description']
                    print(f"🔧 Applying text processing to: {repr(original_text[:100])}...")
                    
                    # Apply the text processing steps that were missing
                    processed_text = original_text
                    
                    # Step 1: Clean up text formatting
                    processed_text = processed_text.replace('\n\n', '\n')  # Double newlines to single
                    processed_text = processed_text.replace(' ~ ', f' {updated_card_data.get("name", "~")} ')  # Replace ~ with card name
                    processed_text = processed_text.replace('~', updated_card_data.get("name", "~"))  # Replace any remaining ~
                    
                    # Step 2: Apply ability reordering
                    processed_text = reorder_abilities_properly(processed_text, updated_card_data)
                    
                    # Step 3: Ensure periods on abilities
                    processed_text = ensure_periods_on_abilities(processed_text)
                    
                    updated_card_data['description'] = processed_text
                    print(f"✅ Text processing complete: {repr(processed_text[:100])}...")
            else:
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
            raise Exception('Both image and content generation failed')
        elif image_data is None:
            print("⚠️ Warning: Image generation failed, returning content only")
            return {
                'cardData': generated_card_text,
                'imageData': None,
                'card_image': card_image_data,
                'warning': 'Image generation not available',
                'generation_time': total_generation_time
            }
        elif generated_card_text is None:
            print("⚠️ Warning: Content generation failed, returning image only")
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

def estimate_tokens(text: str) -> int:
    """
    Rough estimation of CLIP tokens - CLIP tokenizer splits on spaces and punctuation.
    This is a conservative estimate to stay under the 77 token limit.
    """
    # Split on spaces, punctuation, and common word boundaries
    import re
    tokens = re.findall(r'\w+|[^\w\s]', text.lower())
    # Add padding for safety since CLIP tokenization can be complex
    return len(tokens)

def truncate_prompt_smartly(prompt: str, max_tokens: int = 75) -> str:
    """
    Intelligently truncate prompt while preserving the most important elements.
    Priority: subject > style > color palette > lighting
    """
    estimated_tokens = estimate_tokens(prompt)
    
    if estimated_tokens <= max_tokens:
        return prompt
    
    print(f"⚠️  Prompt too long ({estimated_tokens} tokens), truncating...")
    
    # Split prompt into components
    parts = prompt.split(', ')
    
    # Prioritize parts: Subject > Color > Style > Magic context > Lighting
    subject_parts = []  # The card's unique subject matter (HIGHEST priority)
    color_parts = []
    style_parts = []
    magic_parts = []    # Generic Magic context (LOWER priority)  
    lighting_parts = []
    
    for part in parts:
        part_lower = part.lower()
        if 'color palette' in part_lower:
            color_parts.append(part)
        elif any(keyword in part_lower for keyword in ['magic: the gathering', 'card art']):
            magic_parts.append(part)  # Deprioritize generic Magic terms
        elif any(keyword in part_lower for keyword in ['fantasy art', 'style', 'detailed', 'illustration', 'artwork']):
            style_parts.append(part)
        elif any(keyword in part_lower for keyword in ['lighting', 'contrast', 'dramatic']):
            lighting_parts.append(part)
        else:
            subject_parts.append(part)  # The unique subject matter gets highest priority
    
    # Rebuild prompt with subject-first priority order
    final_parts = subject_parts  # Start with the unique subject matter
    
    # Add color palette if space allows (high priority for visual consistency)
    test_prompt = ', '.join(final_parts)
    if color_parts:
        for color_part in color_parts:
            if estimate_tokens(test_prompt + ', ' + color_part) <= max_tokens:
                final_parts.append(color_part)
                test_prompt = ', '.join(final_parts)
                break
    
    # Add style if space allows
    if style_parts:
        for style_part in style_parts:
            if estimate_tokens(test_prompt + ', ' + style_part) <= max_tokens:
                final_parts.append(style_part)
                test_prompt = ', '.join(final_parts)
                break
                
    # Add Magic context only if we have room (lowest priority)
    if magic_parts:
        for magic_part in magic_parts:
            if estimate_tokens(test_prompt + ', ' + magic_part) <= max_tokens:
                final_parts.append(magic_part)
                test_prompt = ', '.join(final_parts)
                break
    
    # Add lighting if space allows
    if lighting_parts:
        for lighting_part in lighting_parts:
            if estimate_tokens(test_prompt + ', ' + lighting_part) <= max_tokens:
                final_parts.append(lighting_part)
                break
    
    final_prompt = ', '.join(final_parts)
    final_tokens = estimate_tokens(final_prompt)
    
    print(f"✂️  Truncated to {final_tokens} tokens: {final_prompt[:100]}...")
    return final_prompt

# Initialize image generation pipeline
image_pipeline = None
image_pipeline_loading = False

def get_image_pipeline():
    """Lazy load the SDXL-Turbo pipeline for image generation"""
    global image_pipeline, image_pipeline_loading
    
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
        
        # Select model based on MODEL_SIZE setting
        model_configs = {
            "heavy": {
                "model_id": "stabilityai/sdxl-turbo",
                "steps": 1,
                "guidance_scale": 0.0,
                "description": "SDXL-Turbo (best quality, heavy)"
            },
            "medium": {
                "model_id": "runwayml/stable-diffusion-v1-5", 
                "steps": 20,
                "guidance_scale": 7.5,
                "description": "SD 1.5 (balanced)"
            },
            "light": {
                "model_id": "CompVis/stable-diffusion-v1-4",
                "steps": 50,
                "guidance_scale": 7.5, 
                "description": "SD 1.4 (light, CPU-friendly)"
            }
        }
        
        # Access global MODEL_SIZE
        global MODEL_SIZE
        
        if MODEL_SIZE == "placeholder":
            print("🚫 Image generation disabled (placeholder mode)")
            return None
            
        if MODEL_SIZE not in model_configs:
            print(f"⚠️  Unknown MODEL_SIZE: {MODEL_SIZE}, falling back to light")
            MODEL_SIZE = "light"
            
        config = model_configs[MODEL_SIZE]
        print(f"🎯 Loading model: {config['description']}")
        print(f"🔍 About to call AutoPipelineForText2Image.from_pretrained...")
        
        try:
            image_pipeline = AutoPipelineForText2Image.from_pretrained(
                config["model_id"],
                torch_dtype=torch_dtype,
                variant="fp16" if device == "cuda" and MODEL_SIZE == "heavy" else None,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True
            )
            print(f"✅ Pipeline loaded successfully!")
            
            # Store generation parameters for this model
            image_pipeline._generation_steps = config["steps"]
            image_pipeline._generation_guidance = config["guidance_scale"]
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
                print("✅ Attention slicing enabled")
            except Exception as e:
                print(f"⚠️ Attention slicing failed: {e}")
            
            # Enable VAE slicing for memory efficiency
            print("🔧 Enabling VAE slicing...")
            try:
                image_pipeline.vae.enable_slicing()
                print("✅ VAE slicing enabled")
            except Exception as e:
                print(f"⚠️ VAE slicing failed: {e}")
            
            # Additional performance optimizations for RTX 3060 Ti
            # Disabled xformers due to dependency conflicts on Windows
            try:
                # Skip xformers to avoid version conflicts
                print("⚠️ xFormers disabled (version conflicts on Windows)")
                # image_pipeline.enable_xformers_memory_efficient_attention()
                # print("✅ xFormers memory efficient attention enabled")
            except Exception as e:
                print(f"⚠️ xFormers not available: {e}")
            
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
        # Check if placeholder mode is enabled  
        global MODEL_SIZE
        if MODEL_SIZE == "placeholder":
            print("🚫 Image generation disabled (placeholder mode)")
            # Return a simple gray image placeholder
            placeholder_image = Image.new('RGB', (width, height), color=(50, 50, 50))
            # Convert to base64
            buffer = io.BytesIO()
            placeholder_image.save(buffer, format='PNG')
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{image_data}"
        
        # Option to disable image generation for testing (re-enabled to show debugging)
        ENABLE_IMAGE_GENERATION = True  # Set to False to disable SDXL
        
        if not ENABLE_IMAGE_GENERATION:
            print("Image generation disabled - using placeholder")
            # Create placeholder image
            image = Image.new('RGB', (width, height), color='#2c3e50')
            draw = ImageDraw.Draw(image)
            draw.text((width//2, height//2), "Artwork\nPlaceholder", 
                     fill='white', anchor='mm')
            
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_data = buffer.getvalue()
            return base64.b64encode(img_data).decode('utf-8')
        
        pipeline = get_image_pipeline()
        
        if pipeline and pipeline is not False:
            # Generate image using SDXL-Turbo
            print(f"Generating AI image for: {prompt}")
            
            # Build enhanced color palette guidance
            color_palette = ""
            colors = card_data.get('colors', []) if card_data else []
            print(f"Debug - Colors received: {colors}")
            
            if colors:
                
                # Handle multicolor combinations first
                if len(colors) >= 5:
                    # Five colors - WUBRG (all colors)
                    color_palette = ", color palette: rainbow prismatic, all five mana colors"
                
                elif len(colors) == 4:
                    # Four color combinations - simplified
                    color_palette = ", color palette: four-color convergence, rich jewel tones"
                
                elif len(colors) == 3:
                    # Three color combinations (Shards and Wedges) - simplified
                    colors_set = set(colors)
                    if colors_set == {'W', 'U', 'G'}:  # Bant
                        color_palette = ", color palette: white marble, blue sapphire, green emerald"
                    elif colors_set == {'U', 'B', 'R'}:  # Grixis
                        color_palette = ", color palette: dark blues, void black, burning red"
                    elif colors_set == {'B', 'R', 'G'}:  # Jund
                        color_palette = ", color palette: shadow black, flame red, wild green"
                    elif colors_set == {'R', 'G', 'W'}:  # Naya
                        color_palette = ", color palette: burning red, emerald green, pure white"
                    elif colors_set == {'G', 'W', 'U'}:  # Same as Bant, reordered
                        color_palette = ", color palette: emerald green, pure white, sapphire blue"
                    elif colors_set == {'W', 'B', 'G'}:  # Abzan
                        color_palette = ", color palette: ivory white, deep black, forest green"
                    elif colors_set == {'U', 'R', 'W'}:  # Jeskai
                        color_palette = ", color palette: sapphire blue, flame red, pure white"
                    elif colors_set == {'B', 'G', 'U'}:  # Sultai
                        color_palette = ", color palette: shadow black, wild green, deep blue"
                    elif colors_set == {'R', 'W', 'B'}:  # Mardu
                        color_palette = ", color palette: burning red, bone white, void black"
                    elif colors_set == {'G', 'U', 'R'}:  # Temur
                        color_palette = ", color palette: emerald green, ocean blue, molten red"
                    else:
                        color_palette = ", color palette: three-color blend, rich jewel tones"
                
                elif len(colors) == 2:
                    # Two color guild combinations - simplified
                    if 'W' in colors and 'U' in colors:
                        color_palette = ", color palette: pristine white, sapphire blue"
                    elif 'W' in colors and 'B' in colors:
                        color_palette = ", color palette: pure white, deep black"
                    elif 'W' in colors and 'R' in colors:
                        color_palette = ", color palette: ivory white, burning red"
                    elif 'W' in colors and 'G' in colors:
                        color_palette = ", color palette: marble white, forest green"
                    elif 'U' in colors and 'B' in colors:
                        color_palette = ", color palette: midnight blue, void black"
                    elif 'U' in colors and 'R' in colors:
                        color_palette = ", color palette: electric blue, molten red"
                    elif 'U' in colors and 'G' in colors:
                        color_palette = ", color palette: ocean blue, living green"
                    elif 'B' in colors and 'R' in colors:
                        color_palette = ", color palette: shadow black, blood red"
                    elif 'B' in colors and 'G' in colors:
                        color_palette = ", color palette: decay black, wild green"
                    elif 'R' in colors and 'G' in colors:
                        color_palette = ", color palette: flame red, primal green"
                
                # Single color palettes - simplified
                elif len(colors) == 1:
                    if 'W' in colors:
                        color_palette = ", color palette: pure white, warm gold"
                    elif 'U' in colors:
                        color_palette = ", color palette: sapphire blue, silver"
                    elif 'B' in colors:
                        color_palette = ", color palette: void black, dark purple"
                    elif 'R' in colors:
                        color_palette = ", color palette: burning red, molten orange"
                    elif 'G' in colors:
                        color_palette = ", color palette: forest green, earth brown"
                
                # Colorless - simplified
                elif 'C' in colors or not colors:
                    color_palette = ", color palette: metallic silver, steel gray"
            
            # Generate type-specific art prompt enhancement
            art_type_context = ""
            card_type = card_data.get('type', '').lower() if card_data else ''
            
            if 'creature' in card_type:
                # Creatures should show the actual creature/being
                # Special handling for blue creatures to diversify away from wizards/mages
                if colors and 'U' in colors and len(colors) == 1:  # Pure blue creatures
                    art_type_context = ", detailed creature portrait, living being, aquatic creature, flying creature, sea monster, elemental being, sphinx, merfolk, bird, octopus, dragon, character focus"
                else:
                    art_type_context = ", detailed creature portrait, living being, character focus"
            elif 'instant' in card_type:
                # Instants should show magical effects in action
                if colors and 'U' in colors and len(colors) == 1:  # Pure blue instants
                    art_type_context = ", water magic, ice effects, wind storm, lightning, teleportation, illusion magic, time distortion, crystal energy, arcane symbols, spell energy"
                else:
                    art_type_context = ", magical effect in progress, spell energy, dynamic action, casting magic"
            elif 'sorcery' in card_type:
                # Sorceries should show powerful magical effects or rituals
                if colors and 'U' in colors and len(colors) == 1:  # Pure blue sorceries  
                    art_type_context = ", tidal wave, storm clouds, ice formation, mystical library, ancient knowledge, arcane research, spell scrolls, crystal formations, time magic"
                else:
                    art_type_context = ", grand magical ritual, powerful spell effect, mystical ceremony, magical transformation"
            elif 'artifact' in card_type:
                # Artifacts should feature the actual artifact/device
                art_type_context = ", detailed artifact object, magical device, ancient relic, crafted item focus"
            elif 'enchantment' in card_type:
                # Enchantments should show magical auras, environments, or ongoing effects
                if colors and 'U' in colors and len(colors) == 1:  # Pure blue enchantments
                    art_type_context = ", shimmering water, floating islands, aurora effects, crystalline structures, frozen landscape, misty atmosphere, magical academy, ancient library, time distortion"
                else:
                    art_type_context = ", magical aura, enchanted environment, mystical atmosphere, ongoing magic effect"
            elif 'land' in card_type:
                # Lands should show landscapes and terrain
                art_type_context = ", landscape view, terrain, natural environment, geographical location"
            elif 'planeswalker' in card_type:
                # Planeswalkers should show the planeswalker character
                if colors and 'U' in colors and len(colors) == 1:  # Pure blue planeswalkers
                    art_type_context = ", powerful planeswalker character, scholar, artificer, elemental master, sea witch, storm caller, ancient being, magical portrait, character focus"
                else:
                    art_type_context = ", powerful planeswalker character, magical being, character focus"
            elif 'battle' in card_type:
                # Battles should show conflict scenes
                art_type_context = ", epic battle scene, conflict, warfare, dramatic confrontation"
            else:
                # Generic fallback
                art_type_context = ", magical fantasy scene"
            
            # Create enhanced art prompt with subject FIRST for better CLIP attention
            # Format: Subject first, then type context, then style, then Magic context, then color palette
            art_prompt = f"{prompt}{art_type_context}{color_palette}, fantasy art, Magic: The Gathering style, detailed illustration, dramatic lighting"
            
            # Apply smart truncation to stay within CLIP's 77 token limit
            final_prompt = truncate_prompt_smartly(art_prompt, max_tokens=75)
            print(f"Debug - Final art prompt ({estimate_tokens(final_prompt)} tokens): {final_prompt}")
            
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
            print(f"⚡ Model inference completed in {inference_time:.2f} seconds")
            
            # Debug the generated image
            print(f"🔍 Generated image size: {image.size}")
            print(f"🔍 Generated image mode: {image.mode}")
            
            # Check if image is actually black
            img_array = np.array(image)
            avg_brightness = np.mean(img_array)
            print(f"🔍 Image average brightness: {avg_brightness:.2f} (0=black, 255=white)")
            
            if avg_brightness < 10:
                print("⚠️  WARNING: Generated image appears to be nearly black!")
                print("🔍 Image min/max values:", np.min(img_array), "/", np.max(img_array))
            else:
                print("✅ Generated image has normal brightness levels")
            
            if inference_time > 10:
                print(f"⚠️ SLOW INFERENCE DETECTED! Expected ~2-4s, got {inference_time:.2f}s")
                print("💡 This suggests GPU optimization issues. Consider:")
                print("   - Updating PyTorch/CUDA drivers")
                print("   - Installing xformers: pip install xformers")
                print("   - Checking GPU memory usage during inference")
            
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

def limit_creature_active_abilities(card_text):
    """
    Ensure creature cards don't have more than 3 active abilities.
    Active abilities are those with activation costs like {T}, {1}, {2}, etc.
    Passive abilities (Flying, Vigilance) and triggered abilities don't count.
    """
    import re
    
    # Pattern to match active abilities: {COST}: effect
    # This matches things like {T}, {1}, {2}, {W}, {U}{R}, etc. followed by a colon
    active_ability_pattern = r'\{[^}]+\}:'
    
    # Find all active abilities
    active_abilities = re.findall(active_ability_pattern, card_text)
    
    if len(active_abilities) <= 3:
        return card_text  # Already within limit
    
    print(f"Warning: Found {len(active_abilities)} active abilities in creature text, limiting to 3")
    
    # Split the text into sentences and abilities
    sentences = [s.strip() for s in card_text.split('.') if s.strip()]
    
    # Rebuild the text, keeping only the first 3 active abilities
    filtered_sentences = []
    active_count = 0
    
    for sentence in sentences:
        sentence_active_abilities = re.findall(active_ability_pattern, sentence)
        
        # If this sentence would push us over the limit, skip it
        if active_count + len(sentence_active_abilities) > 3:
            # If we haven't hit the limit yet, try to keep part of this sentence
            if active_count < 3:
                # Keep passive abilities and triggered abilities from this sentence
                words = sentence.split()
                filtered_words = []
                temp_active_count = active_count
                
                for word in words:
                    if re.search(active_ability_pattern, word):
                        if temp_active_count < 3:
                            filtered_words.append(word)
                            temp_active_count += 1
                        # Skip remaining active abilities in this sentence
                    else:
                        filtered_words.append(word)
                
                if filtered_words:
                    filtered_sentence = ' '.join(filtered_words)
                    if filtered_sentence.strip():
                        filtered_sentences.append(filtered_sentence)
                        active_count = temp_active_count
            break
        else:
            filtered_sentences.append(sentence)
            active_count += len(sentence_active_abilities)
    
    # Rebuild the card text
    result = '. '.join(filtered_sentences)
    if result and not result.endswith('.'):
        result += '.'
    
    print(f"Limited creature abilities from {len(active_abilities)} to {active_count} active abilities")
    return result

def remove_typeline_contamination(abilities_list, card_data):
    """
    Remove any abilities that exactly match the card's typeline
    """
    # Build the complete typeline
    typeline_parts = []
    
    # Add supertype if present
    supertype = (card_data.get('supertype') or '').strip()
    if supertype:
        typeline_parts.append(supertype)
    
    # Add main type
    main_type = (card_data.get('type') or '').strip()
    if main_type:
        typeline_parts.append(main_type)
    
    # Add subtype if present
    subtype = (card_data.get('subtype') or '').strip()
    if subtype:
        typeline_parts.extend(['—', subtype])  # or use '-' depending on what's used
    
    # Create possible typeline variations
    possible_typelines = []
    if typeline_parts:
        # Version with em dash
        typeline_em = ' '.join(typeline_parts)
        possible_typelines.append(typeline_em)
        
        # Version with regular dash 
        typeline_dash = typeline_em.replace('—', '-')
        possible_typelines.append(typeline_dash)
        
        print(f"Checking for typeline contamination: {possible_typelines}")
    
    # Remove any abilities that exactly match a typeline variation
    cleaned_abilities = []
    for ability in abilities_list:
        ability_clean = ability.strip().rstrip('.,!?')
        if not any(ability_clean.lower() == typeline.lower() for typeline in possible_typelines):
            cleaned_abilities.append(ability)
        else:
            print(f"🗑️  Removed typeline contamination: '{ability}'")
    
    return cleaned_abilities

def smart_split_by_periods(text):
    """
    Split text by periods, but respect quoted sections.
    Periods inside quotes should not cause splits.
    """
    if not text:
        return []
    
    parts = []
    current_part = ""
    in_quotes = False
    quote_char = None
    
    i = 0
    while i < len(text):
        char = text[i]
        
        # Handle quote characters
        if char in ['"', "'"]:
            if not in_quotes:
                # Starting a quote
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                # Ending the quote
                in_quotes = False
                quote_char = None
        
        # Handle periods
        elif char == '.' and not in_quotes:
            # Period outside quotes - this is a split point
            current_part += char
            if current_part.strip():
                parts.append(current_part.strip())
            current_part = ""
            i += 1
            continue
        
        current_part += char
        i += 1
    
    # Add any remaining part
    if current_part.strip():
        parts.append(current_part.strip())
    
    return parts

def reorder_abilities_properly(card_text, card_data=None):
    """
    Reorder abilities in proper Magic order based on card type
    """
    import re
    
    # Determine card type for appropriate processing
    card_type = card_data.get('type', '').lower() if card_data else ''
    is_creature = 'creature' in card_type
    is_instant_sorcery = any(t in card_type for t in ['instant', 'sorcery'])
    is_artifact = 'artifact' in card_type
    is_enchantment = 'enchantment' in card_type
    is_planeswalker = 'planeswalker' in card_type
    
    print(f"🎯 Processing {card_type} - creature: {is_creature}, instant/sorcery: {is_instant_sorcery}")
    
    # For instant/sorcery cards, don't apply creature-style ability reordering
    if is_instant_sorcery:
        print(f"📜 Instant/Sorcery detected - using simple text processing")
        # Just clean up the text and ensure proper periods (but not after quotes)
        cleaned_text = card_text.strip()
        if cleaned_text and not cleaned_text.endswith(('.', '!', '?', '"', "'")):
            cleaned_text += '.'
        return cleaned_text
    
    # For planeswalker cards, preserve loyalty abilities order
    if is_planeswalker:
        print(f"🔮 Planeswalker detected - preserving loyalty ability order")
        # Clean up text but don't reorder planeswalker abilities
        lines = [line.strip() for line in card_text.split('\n') if line.strip()]
        cleaned_lines = []
        for line in lines:
            if line and not line.endswith(('.', '!', '?', '"', "'")):
                line += '.'
            cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    
    # For creatures and other permanent types, use ability classification
    # Common Magic keywords (not exhaustive, but covers most used ones)
    keywords = {
        'flying', 'trample', 'vigilance', 'haste', 'reach', 'deathtouch', 'lifelink',
        'first strike', 'double strike', 'hexproof', 'shroud', 'indestructible',
        'flash', 'defender', 'menace', 'prowess', 'skulk', 'madness',
        'ward', 'protection', 'flanking', 'shadow', 'phasing', 'horsemanship',
        'fear', 'intimidate', 'landwalk', 'islandwalk', 'mountainwalk', 'swampwalk',
        'forestwalk', 'plainswalk', 'bushido', 'ninjutsu', 'bloodthirst', 'devour',
        'persist', 'undying', 'cascade', 'storm', 'buyback', 'flashback', 'escape',
        'cycling', 'kicker', 'multikicker', 'suspend', 'vanishing', 'fading',
        'echo', 'morph', 'megamorph', 'bestow', 'enchant', 'equip', 'crew', 'regenerate',
        'reinforce', 'retrace', 'rebound', 'totem armor', 'living weapon',
        'battle cry', 'undaunted', 'surge', 'emerge', 'escalate', 'melee',
        'fabricate', 'improvise', 'revolt', 'explore', 'enrage', 'raid', 'landfall',
        'prowl', 'champion', 'amplify', 'annihilator', 'exalted', 'wither', 'infect',
        'soulbond', 'miracle', 'overload', 'scavenge', 'unleash', 'cipher', 'evolve',
        'extort', 'battalion', 'tribute', 'inspired', 'heroic', 'constellation',
        'ferocious', 'outlast', 'dash', 'exploit', 'renown', 'awaken', 'ingest',
        'devoid', 'cohort', 'delirium', 'transform', 'meld', 'energy', 'aftermath',
        'embalm', 'eternalize', 'afflict', 'rampage', 'changeling', 'convoke', 'delve', 'splice'
    }
    
    # Split text into abilities
    print(f"🔧 PARSING ABILITIES FROM: {repr(card_text)}")
    if '\n' in card_text:
        abilities = [ability.strip() for ability in card_text.split('\n') if ability.strip()]
        print(f"   📋 Split by newlines: {abilities}")
    else:
        # Smart period splitting that respects quotes
        abilities = smart_split_by_periods(card_text)
        print(f"   📋 Split by periods (quote-aware): {abilities}")
    
    keyword_abilities = []
    passive_triggered_abilities = []
    active_abilities = []
    
    for ability in abilities:
        ability_type = classify_ability(ability, keywords)
        
        # Debug individual ability classification
        print(f"   🔍 Classifying: '{ability}' → {ability_type}")
        
        # Ensure ability ends with a period (unless it's a keyword or ends with other punctuation or quotes)
        ability = ability.strip()
        if ability and ability_type != 'keyword' and not ability.endswith(('.', '!', '?', ':', '"', "'")):
            ability += '.'
        
        if ability_type == 'keyword':
            keyword_abilities.append(ability)
        elif ability_type == 'active':
            active_abilities.append(ability)
        else:  # passive/triggered
            passive_triggered_abilities.append(ability)
    
    # Clean up typeline contamination from passive/triggered abilities
    if card_data:
        passive_triggered_abilities = remove_typeline_contamination(passive_triggered_abilities, card_data)
    
    # Combine in proper order: Keywords -> Passive/Triggered -> Active
    # BUT: Keywords should be comma-separated on one line, other abilities use newlines
    final_ability_blocks = []
    
    # Add keywords as a single comma-separated line
    if keyword_abilities:
        keywords_line = ', '.join(keyword_abilities)
        final_ability_blocks.append(keywords_line)
    
    # Add passive/triggered abilities (each on its own line)
    final_ability_blocks.extend(passive_triggered_abilities)
    
    # Add active abilities (each on its own line) 
    final_ability_blocks.extend(active_abilities)
    
    # Use newlines to separate different ability BLOCKS (not individual keywords)
    result = '\n'.join(final_ability_blocks)
    
    # Enhanced debug output for ability reorganization
    print(f"🎯 ABILITY REORGANIZATION DEBUG:")
    print(f"   📝 Input text: {repr(card_text)}")
    print(f"   🔑 Keywords ({len(keyword_abilities)}): {keyword_abilities}")
    print(f"   ⚡ Passive/Triggered ({len(passive_triggered_abilities)}): {passive_triggered_abilities}")
    print(f"   🎯 Active ({len(active_abilities)}): {active_abilities}")
    print(f"   📋 Final ability blocks: {final_ability_blocks}")
    print(f"   📤 Output text: {repr(result)}")
    
    return result

def classify_ability(ability, keywords):
    """
    Classify an ability as keyword, active, or passive/triggered
    """
    import re
    
    clean_ability = ability.rstrip('.,!?').lower()
    
    # Check for active abilities (have activation cost with colon)
    # Patterns: {cost}: effect OR {cost}, additional cost: effect
    # Examples: {T}: Add mana OR {T}, Sacrifice this: Draw a card
    active_patterns = [
        r'\{[^}]+\}:',  # Direct cost with colon: {T}: effect
        r'\{[^}]+\}[^:]*:',  # Cost with additional text then colon: {T}, Sacrifice: effect
    ]
    if any(re.search(pattern, ability) for pattern in active_patterns):
        return 'active'
    
    # Check for triggered abilities first (including "At" triggers)
    # This should be checked before keywords to ensure proper classification
    triggered_words = ['when', 'whenever', 'at the beginning', 'at the end', 'at end of', 'during', 'if']
    if any(trigger in clean_ability for trigger in triggered_words):
        return 'passive_triggered'
    
    # Check for keyword abilities
    # Single keyword or comma-separated keywords
    if ',' in clean_ability and not any(trigger in clean_ability for trigger in ['when', 'whenever', 'at', 'if', 'target', 'choose', 'search', 'draw', 'deal', 'gain', 'lose']):
        # Likely a keyword list like "Trample, haste" or "Deathtouch, menace"
        keywords_in_line = [kw.strip() for kw in clean_ability.split(',')]
        if all(any(keyword in kw for keyword in keywords) for kw in keywords_in_line):
            return 'keyword'
    
    # Single keyword check
    elif any(keyword in clean_ability for keyword in keywords) and len(clean_ability.split()) <= 3:
        # Single keyword or short keyword phrase
        if not any(trigger in clean_ability for trigger in ['when', 'whenever', 'at', 'target', '{', ':']):
            return 'keyword'
    
    # Everything else is passive/triggered
    return 'passive_triggered'

def ensure_periods_on_abilities(card_text):
    """
    Ensure each ability/line ends with a period
    """
    if not card_text:
        return card_text
    
    # Split by newlines to handle each ability separately
    abilities = card_text.split('\n')
    fixed_abilities = []
    
    for ability in abilities:
        ability = ability.strip()
        if ability:  # Skip empty lines
            # Only add period if it doesn't already end with punctuation or quotes
            if not ability.endswith(('.', '!', '?', ':', '"', "'")):
                ability += '.'
        fixed_abilities.append(ability)
    
    return '\n'.join(fixed_abilities)

def format_ability_newlines(card_text):
    """
    Add newlines ONLY before activated abilities that follow this pattern:
    period + space + mana cost + colon
    Example: ". {T}: Do something" becomes ".\n{T}: Do something"
    """
    if not card_text:
        return card_text
    
    # Very specific pattern: period + space + mana cost + colon
    # Only matches: ". {T}:", ". {1}:", ". {2}{G}:", etc.
    pattern = r'\.\s+(\{[^}]+\}:)'
    
    # Replace with period + newline + mana cost
    result = re.sub(pattern, r'.\n\1', card_text)
    
    # Debug output
    matches = re.findall(pattern, card_text)
    if matches:
        print(f"[NEWLINE] Added newlines before activated abilities: {matches}")
    
    return result

def generate_creature_stats(card_data: dict) -> dict:
    """
    Generate power/toughness for creatures based on their mana cost, abilities, and rarity
    Enhanced to include asterisk (*) power/toughness with thematic definitions
    """
    try:
        # Extract mana cost and calculate CMC
        mana_cost = card_data.get('manaCost', '')
        cmc = card_data.get('cmc', 0)
        rarity = card_data.get('rarity', 'common').lower()
        abilities_text = card_data.get('description', '')
        colors = card_data.get('colors', [])
        card_name = card_data.get('name', '')
        
        print(f"🎲 Generating stats for creature with CMC {cmc}, rarity {rarity}, colors {colors}")
        
        # Check if this creature should have asterisk (*) power/toughness
        asterisk_result = should_generate_asterisk_pt(card_data)
        if asterisk_result:
            print(f"⭐ Generating asterisk P/T: {asterisk_result['pattern']}")
            return generate_asterisk_stats(card_data, asterisk_result)
        
        print(f"🎲 Using standard stat generation")
        
        # Base stats calculation from CMC
        if cmc == 0:
            base_total = 2  # 0-cost creatures like 1/1 or 2/0
        elif cmc == 1:
            base_total = 3  # 1-cost creatures like 2/1, 1/2
        elif cmc == 2:
            base_total = 4  # 2-cost creatures like 2/2, 3/1
        elif cmc == 3:
            base_total = 5  # 3-cost creatures like 3/2, 2/3
        elif cmc == 4:
            base_total = 6  # 4-cost creatures like 3/3, 4/2
        elif cmc == 5:
            base_total = 7  # 5-cost creatures like 4/3, 3/4
        elif cmc == 6:
            base_total = 8  # 6-cost creatures like 4/4, 5/3
        else:
            base_total = min(cmc + 2, 12)  # Higher cost creatures, cap at 12
        
        # Adjust for abilities complexity (more abilities = lower stats)
        ability_count = len([line for line in abilities_text.split('\n') if line.strip()])
        if ability_count >= 3:
            base_total -= 1  # Complex creatures get -1 total stats
        elif ability_count >= 5:
            base_total -= 2  # Very complex creatures get -2 total stats
        
        # Adjust for rarity (higher rarity can be slightly more efficient)
        if rarity == 'rare':
            base_total += 1
        elif rarity == 'mythic':
            base_total += 2
        
        # Ensure minimum viable stats
        base_total = max(base_total, 1)
        
        # PERFORMANCE FIX: Skip Ollama call, use fast fallback logic directly
        print(f"🎲 Using fast fallback stat generation (skipping Ollama for performance)")
        
        # Fallback: Simple balanced distribution
        if base_total <= 2:
            power, toughness = 1, max(1, base_total - 1)
        else:
            # Slightly favor toughness for survivability
            power = base_total // 2
            toughness = base_total - power
            if toughness < 1:
                toughness = 1
                power = base_total - 1
        
        print(f"🎲 Fallback generated stats: {power}/{toughness}")
        return {'power': str(power), 'toughness': str(toughness)}
        
    except Exception as e:
        print(f"❌ Error generating creature stats: {e}")
        # Ultimate fallback: 2/2
        return {'power': '2', 'toughness': '2'}

def generate_vehicle_crew_cost(card_data: dict) -> int:
    """
    Generate appropriate crew cost for vehicles based on power/toughness and mana cost
    Returns crew cost as integer (1-6)
    """
    try:
        # Extract power/toughness if available
        power = int(card_data.get('power', 0)) if card_data.get('power', '').isdigit() else 0
        toughness = int(card_data.get('toughness', 0)) if card_data.get('toughness', '').isdigit() else 0
        total_stats = power + toughness
        
        # Extract CMC
        cmc = card_data.get('cmc', 0)
        
        print(f"🚗 Vehicle stats analysis: P/T {power}/{toughness} (total: {total_stats}), CMC: {cmc}")
        
        # Base crew cost on power level and mana cost
        if total_stats <= 3 or cmc <= 2:
            crew_cost = 1  # Small vehicles
        elif total_stats <= 6 or cmc <= 4:
            crew_cost = 2  # Medium vehicles  
        elif total_stats <= 9 or cmc <= 6:
            crew_cost = 3  # Large vehicles
        else:
            crew_cost = 4  # Huge vehicles (cap at 4 for balance)
        
        # Adjust based on power vs toughness ratio (high power = higher crew cost)
        if power > 0 and toughness > 0:
            power_ratio = power / (power + toughness)
            if power_ratio > 0.6:  # High power vehicle
                crew_cost = min(crew_cost + 1, 5)
        
        # Ensure minimum crew 1, maximum crew 5
        crew_cost = max(1, min(crew_cost, 5))
        
        print(f"🚗 Generated crew cost: {crew_cost} (based on stats {total_stats}, CMC {cmc})")
        return crew_cost
        
    except Exception as e:
        print(f"❌ Error generating vehicle crew cost: {e}")
        # Fallback: crew 2 (balanced default)
        return 2

def should_generate_asterisk_pt(card_data: dict) -> dict:
    """
    Determine if a creature should have asterisk (*) power/toughness
    Returns dict with pattern info if yes, None if no
    """
    import random
    
    try:
        cmc = card_data.get('cmc', 0)
        colors = card_data.get('colors', [])
        rarity = card_data.get('rarity', 'common').lower()
        card_name = (card_data.get('name', '') or '').lower()
        
        # Base probability factors
        base_probability = 0.0
        
        # Higher CMC creatures are more likely to have asterisk stats
        if cmc >= 4:
            base_probability += 0.15
        if cmc >= 6:
            base_probability += 0.10
            
        # Higher rarity increases probability
        rarity_bonus = {
            'common': 0.05,
            'uncommon': 0.10, 
            'rare': 0.20,
            'mythic': 0.25
        }
        base_probability += rarity_bonus.get(rarity, 0.05)
        
        # Certain name patterns suggest asterisk stats
        asterisk_name_hints = [
            'avatar', 'incarnation', 'embodiment', 'manifestation',
            'reflection', 'mirror', 'echo', 'shade', 'spirit',
            'essence', 'soul', 'phantom', 'vision', 'dream'
        ]
        
        if any(hint in card_name for hint in asterisk_name_hints):
            base_probability += 0.15
            print(f"⭐ Name '{card_name}' suggests asterisk P/T (+15%)")
        
        # Colors have different tendencies for asterisk mechanics
        color_bonus = {
            'G': 0.12,  # Green loves counting creatures/lands
            'U': 0.10,  # Blue loves counting cards/artifacts  
            'B': 0.08,  # Black loves graveyard counting
            'W': 0.06,  # White moderate (tokens/creatures)
            'R': 0.04   # Red least likely (prefers fixed stats)
        }
        
        for color in colors:
            base_probability += color_bonus.get(color, 0.0)
        
        print(f"⭐ Asterisk P/T probability: {base_probability:.2f} for CMC {cmc}, rarity {rarity}, colors {colors}")
        
        # Roll for asterisk generation
        if random.random() < base_probability:
            # Choose asterisk pattern based on colors and theme
            pattern = choose_asterisk_pattern(card_data)
            return pattern
        
        return None
        
    except Exception as e:
        print(f"❌ Error in should_generate_asterisk_pt: {e}")
        return None

def choose_asterisk_pattern(card_data: dict) -> dict:
    """
    Choose appropriate asterisk pattern based on card colors and theme
    """
    import random
    
    colors = card_data.get('colors', [])
    cmc = card_data.get('cmc', 0)
    card_name = (card_data.get('name', '') or '').lower()
    
    # Color-based pattern preferences
    patterns = []
    
    if 'G' in colors:
        patterns.extend([
            {'type': 'creatures_you_control', 'power': '*', 'toughness': None, 'weight': 3},
            {'type': 'lands_you_control', 'power': '*', 'toughness': None, 'weight': 3},
            {'type': 'creatures_in_graveyard', 'power': None, 'toughness': '*', 'weight': 2},
            {'type': 'forests_you_control', 'power': '*', 'toughness': '*', 'weight': 2}
        ])
    
    if 'U' in colors:
        patterns.extend([
            {'type': 'cards_in_hand', 'power': '*', 'toughness': '*', 'weight': 4},
            {'type': 'artifacts_you_control', 'power': '*', 'toughness': None, 'weight': 3},
            {'type': 'cards_in_library', 'power': None, 'toughness': '*', 'weight': 2},
            {'type': 'instants_in_graveyard', 'power': '*', 'toughness': None, 'weight': 2}
        ])
        
    if 'B' in colors:
        patterns.extend([
            {'type': 'creatures_in_graveyard', 'power': '*', 'toughness': '*', 'weight': 4},
            {'type': 'cards_in_graveyard', 'power': '*', 'toughness': None, 'weight': 3},
            {'type': 'swamps_you_control', 'power': '*', 'toughness': None, 'weight': 2}
        ])
        
    if 'W' in colors:
        patterns.extend([
            {'type': 'creatures_you_control', 'power': None, 'toughness': '*', 'weight': 3},
            {'type': 'plains_you_control', 'power': None, 'toughness': '*', 'weight': 2},
            {'type': 'enchantments_you_control', 'power': '*', 'toughness': '*', 'weight': 2}
        ])
        
    if 'R' in colors:
        patterns.extend([
            {'type': 'mountains_you_control', 'power': '*', 'toughness': None, 'weight': 2},
            {'type': 'cards_in_opponent_hand', 'power': '*', 'toughness': None, 'weight': 3}
        ])
    
    # Fallback patterns if no colors or mixed colors
    if not patterns:
        patterns = [
            {'type': 'cards_in_hand', 'power': '*', 'toughness': '*', 'weight': 2},
            {'type': 'creatures_you_control', 'power': '*', 'toughness': None, 'weight': 2},
            {'type': 'creatures_in_graveyard', 'power': None, 'toughness': '*', 'weight': 2}
        ]
    
    # Weighted random selection
    total_weight = sum(p['weight'] for p in patterns)
    roll = random.random() * total_weight
    current = 0
    
    for pattern in patterns:
        current += pattern['weight']
        if roll <= current:
            return pattern
    
    # Fallback to first pattern
    return patterns[0] if patterns else {'type': 'cards_in_hand', 'power': '*', 'toughness': '*', 'weight': 1}

def generate_asterisk_stats(card_data: dict, pattern: dict) -> dict:
    """
    Generate asterisk power/toughness stats with thematic definitions
    """
    pattern_type = pattern['type']
    power = pattern.get('power')
    toughness = pattern.get('toughness') 
    
    # Map pattern types to rules text definitions
    definitions = {
        'creatures_you_control': "This creature's {stat} is equal to the number of creatures you control.",
        'lands_you_control': "This creature's {stat} is equal to the number of lands you control.", 
        'cards_in_hand': "This creature's {stat} is equal to the number of cards in your hand.",
        'creatures_in_graveyard': "This creature's {stat} is equal to the number of creature cards in your graveyard.",
        'cards_in_graveyard': "This creature's {stat} is equal to the number of cards in your graveyard.",
        'artifacts_you_control': "This creature's {stat} is equal to the number of artifacts you control.",
        'cards_in_library': "This creature's {stat} is equal to the number of cards in your library divided by ten, rounded down.",
        'instants_in_graveyard': "This creature's {stat} is equal to the number of instant cards in your graveyard.",
        'cards_in_opponent_hand': "This creature's {stat} is equal to the number of cards in target opponent's hand.",
        'forests_you_control': "This creature's {stat} is equal to the number of Forests you control.",
        'swamps_you_control': "This creature's {stat} is equal to the number of Swamps you control.",
        'plains_you_control': "This creature's {stat} is equal to the number of Plains you control.",
        'mountains_you_control': "This creature's {stat} is equal to the number of Mountains you control.",
        'enchantments_you_control': "This creature's {stat} is equal to the number of enchantments you control."
    }
    
    # Generate the rules text definition
    definition_template = definitions.get(pattern_type, "This creature's {stat} is equal to the number of cards in your hand.")
    
    rules_parts = []
    if power == '*' and toughness == '*':
        # Both power and toughness are *
        if 'power and toughness are each' in definition_template or 'both' in definition_template:
            rules_text = definition_template.replace('{stat}', 'power and toughness are each')
        else:
            rules_text = definition_template.replace('{stat}', 'power and toughness are each')
        rules_parts.append(rules_text)
        final_power = '*'
        final_toughness = '*'
    elif power == '*':
        # Only power is *
        rules_text = definition_template.replace('{stat}', 'power')
        rules_parts.append(rules_text)
        final_power = '*'
        final_toughness = str(max(1, card_data.get('cmc', 2) - 1))  # Reasonable toughness
    else:
        # Only toughness is *
        rules_text = definition_template.replace('{stat}', 'toughness') 
        rules_parts.append(rules_text)
        final_power = str(max(1, card_data.get('cmc', 2) - 1))  # Reasonable power
        final_toughness = '*'
    
    # Add the definition to the card's description
    existing_description = card_data.get('description', '')
    if existing_description:
        updated_description = '\n'.join(rules_parts) + '\n' + existing_description
    else:
        updated_description = '\n'.join(rules_parts)
    
    card_data['description'] = updated_description
    
    print(f"⭐ Generated asterisk stats: {final_power}/{final_toughness}")
    print(f"⭐ Added definition: {rules_parts[0]}")
    
    return {
        'power': final_power,
        'toughness': final_toughness
    }

def sanitize_planeswalker_abilities(card_text):
    """
    Sanitize planeswalker abilities to ensure proper loyalty format
    """
    import re
    
    # Remove any non-loyalty abilities from planeswalkers
    lines = card_text.split('\n')
    valid_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if it's a proper loyalty ability (+X:, -X:, 0:) or starting loyalty
        loyalty_pattern = r'^[+\-]?\d+:|^Starting loyalty:'
        if re.match(loyalty_pattern, line, re.IGNORECASE) or 'loyalty' in line.lower():
            valid_lines.append(line)
        elif any(keyword in line.lower() for keyword in ['emblem', 'ultimate', 'target', 'draw', 'deal', 'destroy', 'create']):
            # Common planeswalker ability words - likely valid
            valid_lines.append(line)
    
    result = '\n'.join(valid_lines)
    return result if result else card_text

def sanitize_spell_abilities(card_text):
    """
    Sanitize instant/sorcery abilities to focus on spell effects
    """
    import re
    
    # Remove keywords that don't make sense on spells
    invalid_spell_keywords = ['flying', 'trample', 'vigilance', 'haste', 'deathtouch', 'lifelink', 'first strike', 'double strike']
    
    lines = card_text.split('\n')
    valid_lines = []
    
    for line in lines:
        line_lower = line.lower()
        # Skip lines that are just creature keywords
        if any(f"{keyword}," in line_lower or line_lower.strip() == keyword for keyword in invalid_spell_keywords):
            continue
        valid_lines.append(line)
    
    # Limit to 2-3 effects maximum for spells
    if len(valid_lines) > 3:
        valid_lines = valid_lines[:3]
    
    return '\n'.join(valid_lines)

def sanitize_land_abilities(card_text):
    """
    Sanitize land abilities to focus on mana production and utility
    """
    
    # Remove creature-only keywords from lands
    invalid_land_keywords = ['flying', 'trample', 'vigilance', 'haste', 'deathtouch', 'lifelink', 'first strike', 'menace']
    
    lines = card_text.split('\n')
    valid_lines = []
    
    for line in lines:
        line_lower = line.lower()
        # Skip creature keywords
        if any(keyword in line_lower for keyword in invalid_land_keywords):
                continue
        valid_lines.append(line)
    
    return '\n'.join(valid_lines)

def sanitize_permanent_abilities(card_text):
    """
    Sanitize artifact/enchantment abilities for balanced effects
    """
    
    # Artifacts and enchantments can have most abilities, but limit complexity
    lines = card_text.split('\n')
    
    # Limit to 4 abilities maximum
    if len(lines) > 4:
        lines = lines[:4]
    
    return '\n'.join(lines)

def apply_universal_complexity_limits(card_text, card_data):
    """
    Apply complexity limits based on mana cost and rarity for ALL card types
    """
    import re
    
    cmc = card_data.get('cmc', 0)
    rarity = card_data.get('rarity', 'common').lower()
    
    lines = [line.strip() for line in card_text.split('\n') if line.strip()]
    max_abilities = 6  # Global maximum
    
    # Adjust limits based on mana cost and rarity
    if cmc <= 1:
        max_abilities = 2 if rarity in ['rare', 'mythic'] else 1
    elif cmc <= 2:
        max_abilities = 3 if rarity in ['rare', 'mythic'] else 2
    elif cmc <= 3:
        max_abilities = 4 if rarity in ['mythic'] else 3
    elif cmc <= 5:
        max_abilities = 5 if rarity == 'mythic' else 4
    else:
        max_abilities = 6 if rarity == 'mythic' else 5
    
    # Count meaningful abilities (not just keywords on one line)
    ability_count = 0
    keyword_line_count = 0
    
    for line in lines:
        # Check if it's a keyword line (contains only keywords separated by commas)
        if ',' in line and not any(symbol in line for symbol in ['{', '}', ':', 'when', 'whenever', 'target', 'draw', 'deal']):
            keyword_line_count += 1
            ability_count += 1  # Keywords count as 1 ability regardless of how many
        else:
            ability_count += 1
    
    if ability_count > max_abilities:
        print(f"Complexity limit exceeded: {ability_count} > {max_abilities} (CMC:{cmc}, {rarity})")
        # Keep the most important abilities (keywords first, then others)
        limited_lines = lines[:max_abilities]
        result = '\n'.join(limited_lines)
        print(f"Reduced from {len(lines)} to {len(limited_lines)} ability lines")
        return result
    
    return card_text

def validate_rules_text(rules_text: str, card_data: dict) -> bool:
    """
    Validate that rules text doesn't contain type line elements
    Returns True if valid, False if contaminated
    """
    if not card_data or not rules_text:
        return True
        
    # Build type line components to check against
    type_components = []
    
    # Add supertype if present
    supertype = (card_data.get('supertype') or '').strip()
    if supertype:
        type_components.append(supertype)
    
    # Add main type
    main_type = (card_data.get('type') or '').strip()
    if main_type:
        type_components.append(main_type)
    
    # Add subtype if present
    subtype = (card_data.get('subtype') or '').strip()
    if subtype:
        type_components.append(subtype)
    
    # Create full type line variations to check (ONLY full type lines, not individual words)
    type_checks = []
    
    if type_components:
        # Only check for complete type line combinations that shouldn't appear in rules text
        if supertype and main_type and subtype:
            # Full 3-part type line: "Legendary Creature — Beast"
            type_checks.append(f"{supertype} {main_type} — {subtype}")
            type_checks.append(f"{supertype} {main_type} - {subtype}")
            # Also check for the supertype + main type combo: "Legendary Creature"
            type_checks.append(f"{supertype} {main_type}")
        elif main_type and subtype:
            # 2-part type line: "Creature — Beast"  
            type_checks.append(f"{main_type} — {subtype}")
            type_checks.append(f"{main_type} - {subtype}")
        
        # NOTE: We deliberately don't check individual words like "creature", "beast", "legendary"
        # since these are valid in rules text (e.g., "Target creature gains flying")
    
    # Check if any type line elements appear in rules text
    rules_text_lower = rules_text.lower()
    for type_check in type_checks:
        if type_check.lower() in rules_text_lower:
            print(f"⚠️  Rules text contaminated with type line element: '{type_check}'")
            return False
    
    # Check for incorrect mana symbol format (square brackets instead of curly braces)
    import re
    square_bracket_pattern = r'\[([WUBRGCTXYZ0-9]+)\]'
    square_bracket_matches = re.findall(square_bracket_pattern, rules_text, re.IGNORECASE)
    
    if square_bracket_matches:
        print(f"⚠️  Rules text has incorrect mana symbol format: {square_bracket_matches} (should use curly braces {{}} not square brackets [])")
        return False
    
    # Check for card name contamination (only at the beginning of rules text)
    card_name = (card_data.get('name') or '').strip()
    if card_name and len(card_name) > 2:  # Only check meaningful names
        rules_text_lower = rules_text.lower().strip()
        card_name_lower = card_name.lower()
        
        # Check if card name appears at the very beginning (likely a title)
        if rules_text_lower.startswith(card_name_lower):
            # Check what follows the card name at the start
            after_name = rules_text_lower[len(card_name_lower):]
            if after_name.startswith(('\n', ',', ':', ' -', ' —')):
                print(f"⚠️  Rules text starts with card name: '{card_name}' (likely title instead of rules)")
                return False
    
    # Check for X mana cost requirement
    mana_cost = card_data.get('manaCost', '')
    if mana_cost and ('X' in mana_cost.upper() or '{X}' in mana_cost.upper()):
        # Card has X in mana cost, must use X in rules text
        rules_text_upper = rules_text.upper()
        # Enhanced validation: X cards now MUST include explanations
        x_explanation_patterns = [
            r'WHERE\s+X\s+IS\s+THE\s+(AMOUNT\s+OF\s+MANA|NUMBER\s+OF)',  # "where X is the amount of mana" or "where X is the number of"
            r'ENTERS\s+(\w+\s+)*WITH\s+X\s+\+1/\+1\s+COUNTERS?',  # "enters with X +1/+1 counters"
            r'X\s+IS\s+(EQUAL\s+TO|THE\s+(AMOUNT|NUMBER))',  # "X is equal to" or "X is the amount/number"
            r'AMOUNT\s+OF\s+MANA\s+(PAID|SPENT).*CAST',  # "amount of mana paid/spent to cast"
            r'MANA\s+(PAID|SPENT).*FOR\s+X',  # "mana paid/spent for X"
            r'COMES\s+INTO\s+PLAY\s+WITH\s+X',  # "comes into play with X"
            r'X\s+COUNTERS?\s+(ON\s+IT|PLACED)',  # "X counters on it" or "X counters placed"
        ]
        
        x_basic_usage_patterns = [
            r'DEAL\s+X\s+DAMAGE',  # Deal X damage
            r'DRAW\s+X\s+CARDS?',  # Draw X cards
            r'CREATE\s+X\s+.*TOKENS?',  # Create X tokens
            r'X\s+TARGET\s+CREATURES?',  # X target creatures
            r'[+\-]X/[+\-]X',  # Power/toughness modifiers like +X/+X
            r'GAIN\s+X\s+LIFE',  # Gain X life
            r'ADD\s+X\s+MANA',  # Add X mana
            r'DESTROY\s+UP\s+TO\s+X',  # Destroy up to X
            r'RETURN\s+UP\s+TO\s+X',  # Return up to X
        ]
        
        # Check for basic X usage
        has_x_usage = any(re.search(pattern, rules_text_upper) for pattern in x_basic_usage_patterns)
        
        # Check for X explanation
        has_x_explanation = any(re.search(pattern, rules_text_upper) for pattern in x_explanation_patterns)
        
        if not has_x_usage:
            print(f"⚠️  Card has X in mana cost ({mana_cost}) but rules text doesn't use X meaningfully: '{rules_text}'")
            return False
        elif not has_x_explanation:
            print(f"⚠️  Card has X in mana cost ({mana_cost}) but rules text doesn't explain what X represents: '{rules_text}'")
            print(f"    Required: Must include explanation like 'where X is the amount of mana spent' or 'enters with X +1/+1 counters'")
            return False
        else:
            print(f"✅ X mana cost validation passed - found X usage and explanation in rules text")
    
    # Check for duplicate tap symbols (invalid in Magic rules)
    duplicate_tap_patterns = [
        r'\{T\},\s*\{T\}',  # {T}, {T}
        r'Tap,\s*Tap',      # Tap, Tap (case sensitive)
        r'tap,\s*tap',      # tap, tap (lowercase)
    ]
    
    for pattern in duplicate_tap_patterns:
        if re.search(pattern, rules_text, re.IGNORECASE):
            match = re.search(pattern, rules_text, re.IGNORECASE)
            print(f"⚠️  Rules text has invalid duplicate tap symbols: '{match.group()}' (you can't tap twice)")
            return False
    
    # Check for unwanted card elements (mana costs, type lines, titles)
    unwanted_patterns = [
        r'\{\d+\}\{[WUBRG]\}',  # Mana costs like {3}{U}
        r'^[A-Z][a-z]+ [A-Z][a-z]+$',  # Titles like "Lightning Bolt"
        r'(Instant|Sorcery|Creature|Artifact|Enchantment|Land|Planeswalker)\s*[-—]\s*',  # Type lines
        r'^\d+/\d+$',  # Power/toughness like "3/3"
        r'Mana Cost:|Type:|Power/Toughness:',  # Card formatting labels
    ]
    
    for pattern in unwanted_patterns:
        if re.search(pattern, rules_text, re.MULTILINE):
            match = re.search(pattern, rules_text, re.MULTILINE)
            print(f"⚠️  Rules text contains unwanted card elements: '{match.group()}' (should only contain abilities)")
            return False
    
    # Check for Aura enchantment requirements
    card_type = card_data.get('type', '').lower() if card_data else ''
    subtype = (card_data.get('subtype') or '').lower() if card_data else ''
    
    if 'enchantment' in card_type and 'aura' in subtype:
        # Aura enchantments must have "Enchant" as their first ability
        rules_text_upper = rules_text.upper()
        enchant_patterns = [
            r'^ENCHANT\s+CREATURE',  # Most common: "Enchant creature"
            r'^ENCHANT\s+ARTIFACT',  # Less common: "Enchant artifact" 
            r'^ENCHANT\s+LAND',      # Rare: "Enchant land"
            r'^ENCHANT\s+ENCHANTMENT', # Very rare: "Enchant enchantment"
            r'^ENCHANT\s+PERMANENT',   # Flexible: "Enchant permanent"
            r'^ENCHANT\s+PLAYER',      # Rare: "Enchant player"
        ]
        
        has_enchant_ability = any(re.search(pattern, rules_text_upper.strip()) for pattern in enchant_patterns)
        
        if not has_enchant_ability:
            print(f"⚠️  Aura enchantment missing 'Enchant' ability at start of rules text: '{rules_text}'")
            print(f"    Required: Auras must start with 'Enchant creature' (or other valid target)")
            return False
        
        # Also check that it has effects on the enchanted permanent (not always required, but common)
        enchanted_effect_patterns = [
            r'ENCHANTED\s+CREATURE',   # "Enchanted creature gets/has..."
            r'ENCHANTED\s+ARTIFACT',   # "Enchanted artifact..."
            r'ENCHANTED\s+LAND',       # "Enchanted land..."
            r'ENCHANTED\s+PERMANENT',  # "Enchanted permanent..."
        ]
        
        has_enchanted_effects = any(re.search(pattern, rules_text_upper) for pattern in enchanted_effect_patterns)
        
        if has_enchant_ability and not has_enchanted_effects:
            print(f"⚠️  Aura has 'Enchant' but no 'Enchanted [target]' effects: '{rules_text}'")
            print(f"    Recommended: Include effects like 'Enchanted creature gets +1/+1' or similar")
            # Don't fail validation for this, just warn, as some Auras might have other effects
        else:
            print(f"✅ Aura enchantment validation passed - has proper 'Enchant' and effects")
    
    # Check for overuse of "draw cards" effects (encourage variety)
    draw_patterns = [
        r'DRAW\s+(\d+|X|\w+)\s+(CARD|CARDS)',  # "Draw X cards", "Draw three cards", etc.
        r'DRAW\s+A\s+CARD',  # "Draw a card"
    ]
    
    rules_text_upper = rules_text.upper()
    draw_matches = []
    for pattern in draw_patterns:
        matches = re.findall(pattern, rules_text_upper)
        draw_matches.extend(matches)
    
    if len(draw_matches) > 0:
        print(f"⚠️  Card draw effect detected in rules text: '{rules_text}'")
        # Don't fail validation, but add a note about variety
        # This is more of a stylistic preference than a rule violation
        # Check for multiple draw effects or large draw amounts (3+)
        large_draw_match = re.search(r'DRAW\s+(\d+)', rules_text_upper)
        large_draw = large_draw_match and int(large_draw_match.group(1)) >= 3
        
        if len(draw_matches) > 1 or large_draw:
            print(f"    💡 Consider more variety - card draw effects are very common. Try damage, tokens, buffs, removal, etc.")
        else:
            print(f"    ℹ️  Card draw noted - variety is good for Magic diversity")
    
    # Check for asterisk power/toughness definition requirements
    power = card_data.get('power')
    toughness = card_data.get('toughness') 
    has_asterisk_power = power and '*' in str(power)
    has_asterisk_toughness = toughness and '*' in str(toughness)
    
    if has_asterisk_power or has_asterisk_toughness:
        print(f"⭐ Validating asterisk P/T definitions for {power}/{toughness}")
        
        # Common asterisk definition patterns
        asterisk_patterns = [
            r"power.*equal.*to.*the.*number",
            r"toughness.*equal.*to.*the.*number", 
            r"power.*and.*toughness.*are.*each.*equal.*to.*the.*number",
            r"power.*equal.*to.*your.*life",
            r"toughness.*equal.*to.*your.*life"
        ]
        
        has_definition = any(re.search(pattern, rules_text, re.IGNORECASE) for pattern in asterisk_patterns)
        
        if not has_definition:
            asterisk_type = ""
            if has_asterisk_power and has_asterisk_toughness:
                asterisk_type = "both power and toughness (*/*)"
            elif has_asterisk_power:
                asterisk_type = f"power ({power})"
            else:
                asterisk_type = f"toughness ({toughness})"
                
            print(f"❌ ASTERISK VALIDATION FAILED: Card has {asterisk_type} but rules text lacks definition")
            print(f"   Rules text: '{rules_text}'")
            print(f"   Required: Must include definition like 'This creature's power is equal to the number of...'")
            return False
        else:
            print(f"✅ Asterisk P/T validation passed - found definition in rules text")
    
    return True

def createCardContent(prompt, card_data=None):
    """
    Generate card content using Ollama Python client with enhanced context
    Returns the response text from the LLM
    """
    print(f"🧠 createCardContent called with:")
    print(f"   📝 Prompt: {repr(prompt[:100])}...")
    print(f"   🎲 Card data keys: {list(card_data.keys()) if card_data else 'None'}")
    try:
        # Build enhanced prompt based on card properties
        enhanced_prompt = f"Generate ONLY the rules text for a Magic: The Gathering card based on: {prompt}."
        
        if card_data:
            # Analyze mana cost for power level
            cmc = card_data.get('cmc', 0)
            colors = card_data.get('colors', [])
            card_type = (card_data.get('type') or '').lower()
            power = card_data.get('power')
            toughness = card_data.get('toughness')
            rarity = (card_data.get('rarity') or 'common').lower()
            
            # Check for asterisk (*) in power/toughness - CRITICAL VALIDATION
            # Handle cases like "*/2", "3/*", "*/*", "*+1", "2+*", etc.
            has_asterisk_power = power and '*' in str(power)
            has_asterisk_toughness = toughness and '*' in str(toughness)
            
            if has_asterisk_power or has_asterisk_toughness:
                asterisk_guidance = ""
                
                # Enhanced asterisk validation with color-appropriate suggestions
                color_examples = {
                    'G': "creatures you control, lands you control, or Forests you control",
                    'U': "cards in your hand, artifacts you control, or cards in your library", 
                    'B': "creature cards in your graveyard, cards in your graveyard, or Swamps you control",
                    'W': "creatures you control, Plains you control, or enchantments you control",
                    'R': "Mountains you control or cards in target opponent's hand"
                }
                
                # Build color-specific examples
                example_sources = []
                for color in colors:
                    if color in color_examples:
                        example_sources.append(color_examples[color])
                
                if example_sources:
                    examples_text = ", ".join(example_sources[:2])  # Use first 2 color examples
                else:
                    examples_text = "cards in your hand, creatures you control, or cards in your graveyard"
                
                if has_asterisk_power and has_asterisk_toughness:
                    asterisk_guidance = f" CRITICAL ASTERISK RULE: This creature has power {power} and toughness {toughness} - BOTH are asterisks (*). You MUST start your rules text with a definition like '[Card Name]'s power and toughness are each equal to the number of [something you count].' For {colors} colors, appropriate things to count: {examples_text}. REQUIRED FORMAT: Start with the definition, then add any other abilities. Example: 'Patrick Star's power and toughness are each equal to the number of cards in your hand. [Other abilities...]' This asterisk definition is MANDATORY and must be the first line of rules text."
                elif has_asterisk_power:
                    asterisk_guidance = f" MANDATORY ASTERISK RULE: This creature has power {power}. You MUST include an ability that defines what the * equals. For {colors} colors, consider: {examples_text}. Examples: 'This creature's power is equal to the number of cards in your hand' or 'This creature's power is equal to the number of artifacts you control'. The * power MUST be defined in the rules text."
                elif has_asterisk_toughness:
                    asterisk_guidance = f" MANDATORY ASTERISK RULE: This creature has toughness {toughness}. You MUST include an ability that defines what the * equals. For {colors} colors, consider: {examples_text}. Examples: 'This creature's toughness is equal to the number of lands you control' or 'This creature's toughness is equal to your life total'. The * toughness MUST be defined in the rules text."
                
                enhanced_prompt += asterisk_guidance
            
            # Check for regenerate keyword - CRITICAL VALIDATION
            # Regenerate requires specific formatting and rules
            has_regenerate = False
            
            # Check in existing description
            if card_data.get('description'):
                description_text = str(card_data.get('description', '')).lower()
                if 'regenerate' in description_text:
                    has_regenerate = True
            
            # Check in the original prompt
            if 'regenerate' in prompt.lower():
                has_regenerate = True
                
            if has_regenerate:
                regenerate_guidance = " MANDATORY REGENERATE RULE: This card uses regenerate abilities. You MUST use the correct format. Regenerate abilities must be written as '{cost}: Regenerate this creature' or 'Regenerate this creature' (if no cost). Examples: '{1}{G}: Regenerate this creature', '{B}: Regenerate this creature', 'Regenerate this creature'. NEVER write incorrect formats like 'can regenerate', 'has regenerate', or 'regenerates'. Regenerate means: The next time this creature would be destroyed this turn, it isn't. Instead, tap it, remove all damage from it, and remove it from combat. This is the official Magic templating and must be followed exactly."
                enhanced_prompt += regenerate_guidance
            
            supertype = (card_data.get('supertype') or '').lower()
            is_legendary = 'legendary' in supertype
            mana_cost = card_data.get('manaCost', '')
            
            # Check if card has X in mana cost - if so, must use X in rules text
            has_x_cost = 'X' in mana_cost.upper() or '{X}' in mana_cost.upper()
            x_guidance = ""
            if has_x_cost:
                x_guidance = f" CRITICAL X REQUIREMENT: This card has X in its mana cost ({mana_cost}), so the rules text MUST reference X meaningfully AND include an explanation of what X represents. REQUIRED FORMAT: The rules text must contain BOTH an X effect AND an explanation such as 'where X is the amount of mana spent to cast this spell' OR 'enters with X +1/+1 counters on it' OR 'where X is the number of [condition]'. Examples: 'Deal X damage to any target, where X is the amount of mana spent to cast this spell', 'Create X 1/1 token creatures, where X is the amount of mana spent to cast this spell', 'enters with X +1/+1 counters on it', 'X target creatures gain flying until end of turn, where X is the amount of mana spent to cast this spell'."
            
            # Power level guidance based on CMC and rarity
            base_power = ""
            if cmc <= 1:
                base_power = "very simple and low-power"
            elif cmc <= 3:
                base_power = "moderate power"
            elif cmc <= 5:
                base_power = "strong"
            else:
                base_power = "very powerful and game-changing"
            
            # Rarity adjustments with ability guidance - BALANCED POWER LEVELS
            # IMPORTANT: Keywords count as abilities and are valuable!
            if rarity == 'common':
                power_level = f"{base_power}, extremely simple with ONLY 1 total ability - either ONE keyword (Haste, Trample, Deathtouch, Reach, Menace, or Lifelink) OR one simple triggered ability, never both. Keywords are abilities and count toward the limit."
            elif rarity == 'uncommon':
                power_level = f"{base_power}, with modest complexity, maximum 2 total abilities - this can be 1-2 keywords, OR 1 keyword + 1 simple triggered ability, OR 2 simple triggered abilities, OR 1 activated ability. Keywords count as abilities."
            elif rarity == 'rare':
                power_level = f"{base_power}, with unique mechanics, maximum 3 total abilities - this can be 1-2 keywords + 1-2 other abilities, OR 3 non-keyword abilities. All keywords count toward the total ability limit."
            elif rarity == 'mythic':
                power_level = f"{base_power}, with splashy build-around effects, maximum 3-4 total abilities - this includes ALL keywords, triggered abilities, and activated abilities. Keywords are valuable and count toward limits."
            else:
                power_level = base_power
            
            # Enhanced color identity guidance with specific mechanics
            color_guidance = ""
            
            if 'W' in colors:
                color_guidance += " White mechanics: protection from colors, lifegain triggers, exile removal, prevent damage, tap creatures, +1/+1 counters on creatures, enchantment synergies, vigilance, first strike."
            
            if 'U' in colors:
                color_guidance += " Blue mechanics: counter spells, return to hand, tap/untap permanents, scry, flying creatures, mill cards, copy spells, phase out, control magic, card selection, bounce effects, temporary steal effects."
            
            if 'B' in colors:
                color_guidance += " Black mechanics: pay life for effects, destroy creatures, discard from hand, return from graveyard, sacrifice creatures for value, -1/-1 counters, drain life, deathtouch, menace."
            
            if 'R' in colors:
                color_guidance += " Red mechanics: deal damage to creatures/players, haste, can't block, sacrifice for temporary effects, random discard, artifact destruction, first strike, trample, impulse draw."
            
            if 'G' in colors:
                color_guidance += " Green mechanics: ramp (add mana), large creature stats, trample, fight other creatures, destroy artifacts/enchantments, +1/+1 counters, land tutoring, hexproof, reach."
            
            # Add specific multicolor synergies
            if len(colors) > 1:
                if 'W' in colors and 'U' in colors:
                    color_guidance += " Azorius themes: control elements, tax effects, flying creatures, artifact synergies."
                elif 'U' in colors and 'B' in colors:
                    color_guidance += " Dimir themes: mill and graveyard, card selection, unblockable creatures, surveil."
                elif 'B' in colors and 'R' in colors:
                    color_guidance += " Rakdos themes: aggressive damage, sacrifice for value, spectacle costs."
                elif 'R' in colors and 'G' in colors:
                    color_guidance += " Gruul themes: aggressive large creatures, land destruction, riot mechanics."
                elif 'G' in colors and 'W' in colors:
                    color_guidance += " Selesnya themes: token generation, populate, lifegain matters."
                elif 'W' in colors and 'B' in colors:
                    color_guidance += " Orzhov themes: lifegain/lifedrain, sacrifice/recursion, exile effects."
                elif 'U' in colors and 'R' in colors:
                    color_guidance += " Izzet themes: instant/sorcery matters, spell copying, artifact synergies."
                elif 'B' in colors and 'G' in colors:
                    color_guidance += " Golgari themes: graveyard value, creature sacrifice, +1/+1 counters."
                elif 'R' in colors and 'W' in colors:
                    color_guidance += " Boros themes: aggressive creatures, combat tricks, equipment matters."
                elif 'G' in colors and 'U' in colors:
                    color_guidance += " Simic themes: +1/+1 counters, card draw, creature evolution."
            
            # Three-color combinations (Shards and Wedges)
            elif len(colors) == 3:
                sorted_colors = sorted(colors)
                if sorted_colors == ['G', 'U', 'W']:  # Bant
                    color_guidance += " Bant themes: exalted mechanics, control with creatures, artifact interaction, noble and honorable effects."
                elif sorted_colors == ['B', 'U', 'W']:  # Esper
                    color_guidance += " Esper themes: artifact creatures, control magic, evasive threats, combining technology with magic."
                elif sorted_colors == ['B', 'R', 'U']:  # Grixis
                    color_guidance += " Grixis themes: graveyard manipulation, spell copying, creature theft, necromantic power."
                elif sorted_colors == ['G', 'R', 'W']:  # Naya
                    color_guidance += " Naya themes: large creatures, power matters, creature tokens, primal and savage effects."
                elif sorted_colors == ['B', 'G', 'R']:  # Jund
                    color_guidance += " Jund themes: devour mechanics, large threats, resource conversion, predatory nature."
                elif sorted_colors == ['R', 'U', 'W']:  # Jeskai
                    color_guidance += " Jeskai themes: prowess and noncreature spells, tempo plays, martial arts and wisdom."
                elif sorted_colors == ['B', 'G', 'W']:  # Abzan
                    color_guidance += " Abzan themes: +1/+1 counters, toughness matters, endurance and resilience, outlast mechanics."
                elif sorted_colors == ['G', 'R', 'U']:  # Temur
                    color_guidance += " Temur themes: ferocious (power 4+), morph mechanics, savage shamanism and elemental power."
                elif sorted_colors == ['B', 'R', 'W']:  # Mardu
                    color_guidance += " Mardu themes: aggressive creatures, warrior tribal, raid mechanics, honor through combat."
                elif sorted_colors == ['G', 'U', 'B']:  # Sultai
                    color_guidance += " Sultai themes: delve and graveyard, self-mill strategies, ruthless ambition and ancient knowledge."
            
            # Four-color combinations
            elif len(colors) == 4:
                missing_color = set(['W', 'U', 'B', 'R', 'G']) - set(colors)
                if 'W' in missing_color:  # UBRG (No White)
                    color_guidance += " Chaos themes: unpredictable effects, transformation, breaking rules, anti-order mechanics."
                elif 'U' in missing_color:  # WBRG (No Blue)
                    color_guidance += " Aggression themes: direct damage, large creatures, immediate threats, anti-control strategies."
                elif 'B' in missing_color:  # WURG (No Black)
                    color_guidance += " Growth themes: ramp effects, creature enhancement, positive development, anti-death mechanics."
                elif 'R' in missing_color:  # WUBG (No Red)
                    color_guidance += " Control themes: card draw, removal, strategic play, methodical and calculated effects."
                elif 'G' in missing_color:  # WUBR (No Green)
                    color_guidance += " Artifice themes: artifact synergies, constructed beings, technology over nature, precise mechanics."
            
            # Five-color (WUBRG)
            elif len(colors) == 5:
                color_guidance += " WUBRG themes: domain effects, all colors matter, converge mechanics, chromatic unity, powerful legendary effects, mana-intensive abilities that showcase mastery over all five colors of magic."
            
            # TYPE-SPECIFIC CONTEXT PIPELINES
            type_specific_guidance = ""
            
            # INSTANT PIPELINE
            if 'instant' in card_type:
                type_specific_guidance = " INSTANT DESIGN: Generate effects that provide immediate answers, reactions, or advantages. Focus on: counterspells, removal spells, combat tricks, pump spells, bounce effects, damage spells, protection, or temporary buffs. Instants should have immediate impact and be reactive in nature. COHESION FOR INSTANTS: Since instants typically have single focused effects, avoid multiple unrelated abilities. If you include multiple effects, they should be closely related (e.g., 'Deal 3 damage, then scry 1' or 'Counter target spell, draw a card'). Common patterns: 'Counter target spell', 'Destroy target creature', 'Target creature gets +X/+X until end of turn', 'Deal X damage to any target', 'Return target permanent to its owner's hand', 'Target creature gains protection from [color] until end of turn'. Keep effects simple and focused - instants are about timing and immediate utility, not complex interactions."
                
                # Instant-specific rarity scaling
                if rarity == 'common':
                    type_specific_guidance += " Common instant: Simple, focused effect with minimal complexity. Examples: basic counterspell, simple buff, or small damage spell."
                elif rarity == 'uncommon':
                    type_specific_guidance += " Uncommon instant: Moderate complexity, potentially with choices or additional effects. Examples: counterspell with card draw, conditional removal, or larger effect."
                elif rarity in ['rare', 'mythic']:
                    type_specific_guidance += " Rare/Mythic instant: Powerful unique effects, potentially game-changing. Examples: powerful counterspells with additional effects, mass effects, or unique utility."
            
            # SORCERY PIPELINE  
            elif 'sorcery' in card_type:
                type_specific_guidance = " SORCERY DESIGN: Generate proactive effects that provide significant impact on your turn. Focus on: tutoring, mass effects, creature tokens, permanent solutions, board development, reanimation, or transformation effects. Sorceries should be powerful but require planning since they're sorcery speed. COHESION FOR SORCERIES: Sorceries can have multiple effects, but they must support a unified strategy. Good themes: token creation + token buffs, reanimation + graveyard filling, ramp + expensive effects, or mass removal + card advantage. Avoid combining unrelated effects like 'Create tokens + Counter next spell + Gain life'. Common patterns: 'Search your library for...', 'Destroy all creatures', 'Create X creature tokens', 'Return target card from graveyard to hand', 'Transform target creature', 'Put a creature from your graveyard onto the battlefield'. Sorceries can be more complex than instants since timing isn't critical."
                
                # Sorcery-specific rarity scaling
                if rarity == 'common':
                    type_specific_guidance += " Common sorcery: Straightforward effects like simple creature tokens, basic removal, or minor utility effects."
                elif rarity == 'uncommon':
                    type_specific_guidance += " Uncommon sorcery: Moderate complexity with choices or multiple effects. Can affect multiple targets or have additional benefits."
                elif rarity in ['rare', 'mythic']:
                    type_specific_guidance += " Rare/Mythic sorcery: Powerful unique effects that can significantly impact the game state. Mass effects, powerful tutoring, or unique mechanics."
            
            # ARTIFACT PIPELINE
            elif 'artifact' in card_type:
                type_specific_guidance = " ARTIFACT DESIGN: Generate effects that provide ongoing utility, activated abilities, or passive benefits. Artifacts are colorless and should feel mechanical/technological. Focus on: activated abilities with costs ({T}:, {1}:, etc.), static effects that modify the game, or utility functions. COHESION FOR ARTIFACTS: Artifacts should have a clear purpose or theme. Good themes: mana production + mana sinks, sacrifice artifacts + artifact recursion, +1/+1 counters + counter synergies, or card selection + card advantage. Avoid random combinations like 'Tap for mana + Flying creatures + Graveyard removal'. Common patterns: '{T}: Add one mana of any color', '{2}, {T}: Card selection effect', 'Creatures you control get +1/+1', '{1}, Sacrifice ~: Deal 2 damage to any target'. Artifacts often have multiple modes of use or ongoing value."
                
                # Artifact-specific types - check both subtype and full typeline
                full_type = (card_data.get('typeLine') or '').lower()
                subtype = (card_data.get('subtype') or '').lower()
                
                if 'equipment' in subtype or 'equipment' in full_type:
                    type_specific_guidance += " EQUIPMENT: MANDATORY - All Equipment MUST have an 'Equip {cost}' ability (e.g., 'Equip {1}', 'Equip {2}', 'Equip {3}', etc.). Focus on 'Equipped creature gets/has...' effects that enhance creatures with stats, keywords, or abilities. The equip cost is essential and required for all Equipment."
                elif 'vehicle' in subtype or 'vehicle' in full_type:
                    type_specific_guidance += " VEHICLE MANDATORY RULES: ALL Vehicles MUST have 'Crew X (Tap any number of creatures you control with total power X or greater: This Vehicle becomes an artifact creature until end of turn.)' ability. CRITICAL FORMAT: Crew is a STANDALONE ability - just 'Crew X', NEVER '{T}, Crew X' or any other cost additions. Examples: 'Crew 2', 'Crew 1', 'Crew 3' (standalone abilities). Common crew costs: Crew 1 (small vehicles), Crew 2 (medium vehicles), Crew 3 (large vehicles), Crew 4+ (huge vehicles). Vehicles start as non-creature artifacts and only become creatures when crewed. They should have strong creature stats (power/toughness) to justify the crew cost. Vehicle abilities should focus on: combat abilities (flying, trample, vigilance), triggered abilities when attacking, or static abilities while crewed. Balance: Higher crew cost = better stats/abilities. NEVER add additional costs to crew abilities - crew activates by tapping other creatures, not the vehicle itself."
                elif 'food' in subtype or 'food' in full_type:
                    type_specific_guidance += " FOOD TOKEN: MANDATORY - All Food tokens MUST have the ability '{2}, {T}, Sacrifice this artifact: You gain 2 life.' This is the defining characteristic of Food tokens as specified by the user. You may add one additional minor ability, but this exact sacrifice ability is required."
                else:
                    type_specific_guidance += " Generic artifact: Utility effects, activated abilities, or static benefits available to all colors."
            
            # ENCHANTMENT PIPELINE
            elif 'enchantment' in card_type:
                type_specific_guidance = " ENCHANTMENT DESIGN: Generate ongoing effects that modify game rules or provide continuous benefits. Enchantments represent magical effects that persist. Focus on: static effects ('Creatures you control have...'), triggered abilities ('Whenever/When...'), or activated abilities that represent magical powers. COHESION FOR ENCHANTMENTS: All abilities should support a unified magical theme. Good themes: creature buffs + creature synergies, graveyard effects + death triggers, mana enhancement + expensive activated abilities, or tribal effects + creature type matters. Avoid random combinations like 'Creature buffs + Land destruction + Card draw + Life gain'. Common patterns: 'Creatures you control get +1/+1', 'Whenever a creature enters the battlefield, ...', '{T}: Target creature gains flying until end of turn', 'At the beginning of your upkeep, ...' Enchantments should feel magical and provide long-term value."
                
                # Enchantment-specific types
                if 'aura' in (card_data.get('subtype') or '').lower():
                    type_specific_guidance += " AURA ENCHANTMENT: MANDATORY - ALL Auras MUST have 'Enchant creature' as their first ability (unless specifically targeting something else like artifacts or lands, but 95% should enchant creatures). REQUIRED FORMAT: Start with 'Enchant creature' followed by effects on 'Enchanted creature gets/has/gains...' Examples: 'Enchant creature. Enchanted creature gets +2/+2', 'Enchant creature. Enchanted creature has flying and vigilance', 'Enchant creature. Enchanted creature gets +1/+1 for each artifact you control'. Auras provide ongoing benefits to the creature they're attached to. Focus on stat boosts, keyword abilities, or special powers for the enchanted creature. Common patterns: +X/+X boosts, keyword abilities (flying, trample, lifelink, etc.), protection abilities, tap/untap effects, or triggered abilities that benefit the enchanted creature."
                elif 'saga' in (card_data.get('subtype') or '').lower():
                    type_specific_guidance += " SAGA: Must have chapter abilities (I, II, III) that tell a story progression. Each chapter should be a triggered ability that activates in sequence."
                else:
                    type_specific_guidance += " Generic enchantment: Ongoing magical effects that modify the game state or provide continuous benefits."
            
            # LAND PIPELINE
            elif 'land' in card_type:
                type_specific_guidance = " LAND DESIGN: Generate mana-producing abilities and/or utility effects. Lands are the foundation of Magic's resource system. Focus on: mana generation ('{T}: Add {color}'), activated abilities with costs, or utility functions. Most lands should produce mana as their primary function. Common patterns: '{T}: Add {W}', '{T}: Add one mana of any color', '{1}, {T}: Draw a card', '{T}: Target creature gets +1/+0 until end of turn'. Utility lands should have higher activation costs to balance their additional effects."
                
                # Land-specific rarity scaling
                if rarity == 'common':
                    type_specific_guidance += " Common land: Simple mana production, possibly with basic utility. Examples: basic lands, simple dual lands, or lands with minor activated abilities."
                elif rarity == 'uncommon':
                    type_specific_guidance += " Uncommon land: Dual mana production or useful activated abilities. Balance mana fixing with utility effects."
                elif rarity in ['rare', 'mythic']:
                    type_specific_guidance += " Rare/Mythic land: Powerful utility effects or unique mana abilities. Can have complex activated abilities or game-changing effects."
            
            # PLANESWALKER PIPELINE
            elif 'planeswalker' in card_type:
                type_specific_guidance = " PLANESWALKER DESIGN: Generate loyalty abilities that represent a powerful ally. Planeswalkers have starting loyalty and 2-4 abilities with loyalty costs. Format as '+X: [effect]', '-X: [effect]', and optional ultimate '-X: [powerful effect]'. First ability should be positive or neutral loyalty, middle ability(ies) should cost loyalty for stronger effects, ultimate should be game-changing but expensive. Common patterns: '+1: Draw a card', '-2: Deal 3 damage to any target', '-7: You get an emblem with...'. Each ability should feel distinct and flavorful to the character."
                
                # Planeswalker complexity by rarity
                if rarity in ['rare', 'mythic']:
                    type_specific_guidance += " Rare/Mythic planeswalker: 3-4 abilities including a powerful ultimate. Starting loyalty 3-5. Abilities should synergize and tell a story."
                else:
                    type_specific_guidance += " Uncommon planeswalker: 2-3 abilities, simpler effects. Starting loyalty 2-4. Focus on utility rather than game-ending effects."
            
            # BATTLE PIPELINE
            elif 'battle' in card_type:
                type_specific_guidance = " BATTLE DESIGN: Generate effects that trigger when the battle enters or is defeated. Battles start with defense counters and have effects when they transform or are defeated. Focus on: enter-the-battlefield effects, static effects while on battlefield, and powerful 'when this battle is defeated' triggers. Common patterns: 'When ~ enters the battlefield, ...', 'Whenever ~ loses a defense counter, ...', 'When ~ is defeated, ...'. Battles should feel like epic conflicts with meaningful rewards for defeating them."
            
            # Enhanced creature-specific guidance with stats balancing
            creature_guidance = ""
            if 'creature' in card_type and power and toughness:
                try:
                    p = int(power) if power.isdigit() else 0
                    t = int(toughness) if toughness.isdigit() else 0
                    stat_total = p + t
                    
                    # More aggressive stat-based ability limiting
                    if stat_total >= 10:  # Large creatures like 6/6, 5/5, etc.
                        creature_guidance = " This creature has very large stats, so limit to AT MOST 0-2 simple abilities (prefer keywords like Trample, Vigilance). Avoid complex activated abilities."
                    elif stat_total >= 7:  # Medium-large creatures like 4/4, 3/4, etc.
                        creature_guidance = " This creature has large stats for its cost, so abilities should be minimal - prefer 0-2 keywords or one simple triggered ability. Avoid multiple activated abilities."
                    elif stat_total >= 5:  # Average creatures
                        if stat_total > cmc * 2.2:  # Above-rate stats
                            creature_guidance = " This creature has above-average stats, so limit abilities to 0-2 simple ones (mostly keywords)."
                        else:
                            creature_guidance = " This creature has moderate stats, so it can have 2-3 balanced abilities."
                    elif stat_total < cmc * 1.5:  # Below-rate stats
                        creature_guidance = " This creature has low stats for its cost, so it should have multiple powerful abilities (2-4 depending on rarity) to compensate."
                    else:
                        creature_guidance = " This creature has balanced stats, so it can have moderate utility abilities."
                        
                    # Special case for defensive creatures (high toughness, low power)
                    if t >= 5 or (t > p and t >= 3):
                        creature_guidance += " This is a defensive creature - consider abilities like Defender, Wall synergies, or activated abilities that don't require attacking."
                        
                except:
                    pass
            
            # Legendary creature limitation with name/subtype awareness
            legendary_guidance = ""
            if is_legendary and 'creature' in card_type:
                card_name = card_data.get('name', '')
                subtype = card_data.get('subtype', '')
                
                legendary_guidance = " LEGENDARY CONSTRAINT: This is a legendary creature - structure abilities as follows: "
                
                # Power level determines ability structure - FOLLOW SAME RARITY LIMITS AS NON-LEGENDARY
                # Legendary status does not grant extra abilities, just unique flavor
                if rarity == 'mythic':
                    legendary_guidance += "Mythic legendary: Maximum 3-4 total abilities including keywords, triggered, and activated abilities. Make abilities feel unique and build-around worthy."
                elif rarity == 'rare':
                    legendary_guidance += "Rare legendary: Maximum 3 total abilities including keywords. Focus on unique mechanics that feel special."
                elif rarity == 'uncommon':
                    legendary_guidance += "Uncommon legendary: Maximum 2 total abilities including keywords. Simple but memorable effects."
                else:  # common
                    legendary_guidance += "Common legendary: Maximum 1 total ability - either ONE keyword OR one simple triggered ability. Being legendary doesn't grant extra complexity."
                
                # Add name and subtype flavor guidance
                if card_name:
                    legendary_guidance += f" IMPORTANT: This creature is named '{card_name}' - design abilities that reflect this specific character's identity, personality, and lore. Make the abilities feel unique to this individual."
                
                if subtype:
                    # Add subtype-specific ability suggestions
                    subtype_lower = subtype.lower()
                    
                    if 'dragon' in subtype_lower:
                        legendary_guidance += " As a Dragon, consider abilities like flying, dealing damage, treasure generation, or breath weapon effects."
                    elif 'angel' in subtype_lower:
                        legendary_guidance += " As an Angel, consider abilities like flying OR vigilance (not both), lifegain, protection effects, flash, hexproof, or helping other creatures."
                    elif 'demon' in subtype_lower:
                        legendary_guidance += " As a Demon, consider abilities like flying, menace, sacrifice effects, life drain, or punishing opponents."
                    elif 'beast' in subtype_lower:
                        legendary_guidance += " As a Beast, consider abilities like trample, fighting other creatures, +1/+1 counters, or natural/primal effects."
                    elif 'wizard' in subtype_lower or 'mage' in subtype_lower:
                        legendary_guidance += " As a Wizard/Mage, consider abilities related to spells, card draw, instant/sorcery synergies, or magical effects."
                    elif 'warrior' in subtype_lower or 'soldier' in subtype_lower:
                        legendary_guidance += " As a Warrior/Soldier, consider abilities like first strike, vigilance, combat bonuses, or military tactics."
                    elif 'rogue' in subtype_lower or 'assassin' in subtype_lower:
                        legendary_guidance += " As a Rogue/Assassin, consider abilities like deathtouch, unblockable, card advantage through sneaky means, or removal effects."
                    elif 'knight' in subtype_lower:
                        legendary_guidance += " As a Knight, consider abilities like first strike, vigilance, protection effects, or honor-based abilities."
                    elif 'spirit' in subtype_lower:
                        legendary_guidance += " As a Spirit, consider abilities like flying, phasing, graveyard interactions, or ethereal effects."
                    elif 'elemental' in subtype_lower:
                        legendary_guidance += " As an Elemental, consider abilities related to basic lands, elemental forces, or effects that match your colors (fire=damage, water=card draw, etc.)."
                    elif 'vampire' in subtype_lower:
                        legendary_guidance += " As a Vampire, consider abilities like lifelink, flying, life drain effects, or graveyard recursion."
                    elif 'zombie' in subtype_lower:
                        legendary_guidance += " As a Zombie, consider abilities like deathtouch, graveyard recursion, sacrifice synergies, or undeath effects."
                    elif 'elf' in subtype_lower:
                        legendary_guidance += " As an Elf, consider abilities like mana generation, creature synergies, forest/nature effects, or tribal bonuses."
                    elif 'goblin' in subtype_lower:
                        legendary_guidance += " As a Goblin, consider abilities like haste, direct damage, artifact interactions, or chaotic/random effects."
                    elif 'human' in subtype_lower:
                        legendary_guidance += " As a Human, consider versatile abilities that could represent leadership, innovation, adaptability, or cooperation with other creatures."
                    
                    legendary_guidance += f" The subtype '{subtype}' should strongly influence the flavor and mechanics of the abilities."
            
            # Add card name inspiration for creative abilities
            name_inspiration = ""
            card_name = (card_data.get('name') or '').strip()
            if card_name and len(card_name) > 2:
                name_inspiration += f" CARD NAME INSPIRATION: This card is named '{card_name}' - use this name as creative inspiration for unique abilities. Extract thematic concepts from the name: if it mentions elements (fire, ice, storm), create elemental effects; if it mentions creatures (dragon, angel, demon), incorporate those creature themes; if it mentions objects (sword, tome, crown), design abilities that reflect those items; if it mentions actions (strike, whisper, shatter), create abilities based on those verbs; if it mentions places (tower, grove, sanctum), include location-based effects. Make the abilities feel specifically tied to this card's identity, not generic effects."
            
            enhanced_prompt += f" The card costs {cmc} mana and should be {power_level}.{color_guidance}{type_specific_guidance}{creature_guidance}{legendary_guidance}{name_inspiration}{x_guidance}"
        
        # Add CMC guidance that respects rarity limits (rarity limits take precedence)
        if card_data and 'creature' in card_data.get('type', '').lower():
            # CMC provides flavor guidance but CANNOT exceed rarity ability limits
            if cmc <= 1:
                enhanced_prompt += f" CMC GUIDANCE: 1 mana creatures prefer efficient, simple effects within your {rarity} rarity limit. Favor keywords like Haste, Deathtouch, Menace, Reach, or Lifelink. Avoid activated abilities on cheap creatures."
            elif cmc == 2:
                enhanced_prompt += f" CMC GUIDANCE: 2 mana creatures work well with keywords or simple triggers within your {rarity} rarity limit. Examples: single keywords or 'When this enters the battlefield' effects."
            elif cmc == 3:
                enhanced_prompt += f" CMC GUIDANCE: 3 mana creatures can support moderate complexity within your {rarity} rarity limit. Consider activated abilities like '{{T}}: Add mana' or utility effects."
            elif cmc <= 5:
                enhanced_prompt += f" CMC GUIDANCE: 4-5 mana creatures justify more abilities within your {rarity} rarity limit. Can support activated abilities and synergistic effects."
            else:
                enhanced_prompt += f" CMC GUIDANCE: High-cost creatures (6+ mana) should feel impactful within your {rarity} rarity limit. Focus on game-changing effects appropriate for the mana investment."
            
            enhanced_prompt += " REMEMBER: Activated abilities (costs like {T}:, {1}:) are the most complex. Keyword abilities (Haste, Trample, Deathtouch, Menace, Lifelink, Reach, etc.) and triggered abilities (When/Whenever) are simpler. Lower mana cost = fewer and simpler abilities. VARIETY: Avoid overusing 'Flying' and 'Vigilance' together - consider diverse keyword combinations. CREATURE COHESION: All abilities must work together thematically. Good creature themes include: aggressive (Haste + Trample + attack benefits), defensive (Vigilance + blocking rewards), graveyard-focused (death triggers + graveyard recursion), token-maker (creates tokens + sacrifice outlets), tribal (creature type synergies), or utility (mana abilities + activated effects). AVOID mixing unrelated mechanics like 'Flying + Graveyard recursion + Mana production + Life gain' - pick 1-2 related themes. IMPORTANT: Multiple keywords should be comma-separated on one line (like 'Flying, trample'), not on separate lines."
        
        # Type-specific formatting instructions
        if 'planeswalker' in card_type:
            enhanced_prompt += " PLANESWALKER FORMATTING: Generate 2-4 loyalty abilities in the format '+X: [effect]', '0: [effect]', or '-X: [effect]'. List each ability on its own line. Include starting loyalty as the first line like 'Starting loyalty: 3'. Example format: 'Starting loyalty: 3\\n+1: Draw a card\\n-2: Deal 3 damage to any target\\n-7: You get an emblem with \"Creatures you control have flying\"'."
        elif 'instant' in card_type or 'sorcery' in card_type:
            enhanced_prompt += " INSTANT/SORCERY FORMATTING: Generate spell effects that happen when cast. Keep effects concise and focused. Example formats: 'Counter target spell', 'Destroy target creature', 'Draw three cards', 'Create two 1/1 creature tokens', 'Deal 4 damage to any target'. Use standard Magic spell language."
        elif 'land' in card_type:
            enhanced_prompt += " LAND FORMATTING: Focus on mana abilities and utility effects. Format activated abilities with proper costs. Example formats: '{T}: Add {W}', '{T}: Add one mana of any color', '{1}, {T}: Draw a card', '{T}: Target creature gets +1/+0 until end of turn'."
        else:
            enhanced_prompt += " FORMATTING: Keywords should be comma-separated on ONE line (like 'Trample, menace'), while other abilities use separate lines. For activated abilities use format '{cost}: {effect}'. For triggered abilities use 'When/Whenever/At' format. Example: 'Trample, menace\\n{T}: Add one mana of any color\\n{2}: Target creature gains first strike until end of turn'."
        
        enhanced_prompt += " CREATIVITY AND UNIQUENESS REQUIREMENTS: 1) AVOID OVERUSED GENERIC ABILITIES: Never use these repetitive effects: 'Draw 3 cards', 'Draw a card', '{T}: Add one mana of any color', 'Tap: Create 1 mana', 'When this enters the battlefield, draw a card', 'Sacrifice this: Draw a card'. These are boring and overused. 2) FOCUS ON THE CARD'S IDENTITY: Use the card's name, type, and power level as inspiration. If the card is named 'Sword of Fire', create fire-themed combat abilities. If it's called 'Ancient Tome', focus on knowledge/library effects, not generic card draw. If it's a 'Dragon Engine', combine draconic and mechanical themes. 3) PRIORITIZE UNIQUE MECHANICS: Instead of generic effects, create interesting abilities like: temporary creature theft, conditional countering, combat phase manipulation, alternate win conditions, unique token creation, innovative triggered conditions, creative activated abilities that aren't just mana generation, spell copying with twists, unique protection effects, interesting sacrifice effects, creative pump effects, unique evasion beyond flying. 4) MAKE IT MEMORABLE: Every ability should feel distinctive and tied to the card's concept. VARIETY REQUIREMENT: Avoid overused effects like 'draw cards' and 'add mana' - instead prioritize diverse, creative effects that match the card's colors and type. Explore unique mechanics, interesting interactions, and varied effect types. COHESION REQUIREMENT: All abilities on a single card must work together thematically and mechanically. Do NOT combine random unrelated abilities. Instead, create cards with unified themes such as: sacrifice synergies (sacrifice creatures → get benefits), +1/+1 counter themes (place counters → counter-based benefits), graveyard strategies (mill → graveyard value), tribal synergies (creature types matter), or mana ramp strategies (produce mana → expensive effects). Each ability should support or enhance the others. Example of GOOD cohesion: 'When this enters, create two 1/1 tokens' + '{T}, Sacrifice a creature: Draw a card' (token generation supports sacrifice). Example of BAD cohesion: 'Flying' + '{T}: Add mana' + 'Whenever a creature dies, gain 2 life' + 'Discard a card: Deal 1 damage' (random unrelated abilities). CRITICAL OUTPUT FORMAT: Generate ONLY a single block of rules text - nothing else. Do NOT include: card names, mana costs (like {3}{U}{U}), type lines (like 'Instant' or 'Creature - Human'), power/toughness numbers, flavor text, card titles, set symbols, or any other card elements. Your response should contain ONLY the text that would appear inside the rules text box of the card. IMPORTANT: Use {T} for tap symbol, never write 'Tap:'. Use standard Magic card formatting without surrounding quotes. NEVER include ability type labels like 'Triggered Ability:', 'Passive Ability:', 'Active Ability:', 'Keywords:', etc. Just write the abilities directly. Example of CORRECT output: 'Counter target spell' or 'Flying\\nWhenever this creature attacks, gain 2 life' or 'Flying\\n{T}: Add one mana of any color'. Example of INCORRECT output: 'Lightning Bolt {2}{R}\\nInstant\\nDeal 3 damage to any target' or 'Keywords: Flying\\nTriggered Ability: When this enters, draw a card'. Generate ONLY the raw abilities or effects as they would appear on an actual Magic card - no labels, no categories, no other formatting."
        
        # Generate and validate rules text (retry if contaminated)
        max_attempts = 3
        card_text = ""
        
        for attempt in range(max_attempts):
            response = ollama.generate(
                model='mistral:latest',
                prompt=enhanced_prompt
            )
            
            # Clean up the response
            card_text = response['response'].strip()
            print(f"📜 Original model output: {repr(card_text)}")
            
            # Validate the response doesn't contain type line elements
            if validate_rules_text(card_text, card_data):
                print(f"✅ Rules text validation passed on attempt {attempt + 1}")
                break
            else:
                print(f"❌ Rules text validation failed on attempt {attempt + 1}, regenerating...")
                if attempt < max_attempts - 1:
                    # Add additional constraint for retry
                    enhanced_prompt += f" CRITICAL: Do NOT include type line elements like '{card_data.get('type', '')}' or '{card_data.get('subtype', '')}' in the rules text. Generate ONLY the abilities text."
                else:
                    print(f"⚠️  Max validation attempts reached, using last generated text")
                    break
        
        # Remove surrounding quotes if present
        if (card_text.startswith('"') and card_text.endswith('"')) or \
           (card_text.startswith("'") and card_text.endswith("'")):
            card_text = card_text[1:-1].strip()
        
        # Fix common formatting issues
        # Replace literal \\n with actual newlines
        card_text = card_text.replace('\\n', '\n')
        
        # Replace "Tap:" with "{T}:" for tap symbols
        card_text = card_text.replace('Tap:', '{T}:')
        
        # Remove any ability type labels that might have slipped through
        import re
        ability_label_patterns = [
            r'Triggered Ability:\s*',
            r'Passive Ability:\s*',
            r'Active Ability:\s*',
            r'Keyword Ability:\s*',
            r'Keywords:\s*',
            r'Abilities:\s*',
            r'Static Ability:\s*',
            r'Activated Ability:\s*',
            r'Ability:\s*'
        ]
        
        for pattern in ability_label_patterns:
            card_text = re.sub(pattern, '', card_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up any extra whitespace or newlines created by label removal
        card_text = re.sub(r'\n\s*\n', '\n', card_text).strip()
        
        # Convert double newlines to single newlines
        card_text = card_text.replace('\n\n', '\n')
        
        # Replace ~ symbol with actual card name (simple global replacement)
        if card_data and card_data.get('name'):
            card_name = (card_data.get('name') or '').strip()
            if card_name:  # Only replace if we have a valid card name
                card_text = card_text.replace('~', card_name)
        
        # Ensure each ability ends with a period
        card_text = ensure_periods_on_abilities(card_text)
        
        # Add newlines before ability costs that come after periods (not in quotes)
        card_text = format_ability_newlines(card_text)
        
        # Limit to 3-4 sentences by splitting on periods and taking first 4
        sentences = [s.strip() for s in card_text.split('.') if s.strip()]
        if len(sentences) > 4:
            card_text = '. '.join(sentences[:4]) + '.'
        
        # UNIFIED SANITATION PIPELINE - Applied to ALL card types
        if card_data:
            card_type = card_data.get('type', '').lower()
            
            # Step 1: Type-specific ability limits
            if 'creature' in card_type:
                card_text = limit_creature_active_abilities(card_text)
            elif 'planeswalker' in card_type:
                # Planeswalkers should have proper loyalty ability format
                card_text = sanitize_planeswalker_abilities(card_text)
            elif 'instant' in card_type or 'sorcery' in card_type:
                # Spells should have single cohesive effects
                card_text = sanitize_spell_abilities(card_text)
            elif 'land' in card_type:
                # Lands should focus on mana abilities
                card_text = sanitize_land_abilities(card_text)
            elif 'artifact' in card_type or 'enchantment' in card_type:
                # Artifacts/enchantments get general permanent sanitation
                card_text = sanitize_permanent_abilities(card_text)
            
            # Step 2: Universal ability reordering (ALL card types benefit from proper ordering)
            card_text = reorder_abilities_properly(card_text, card_data)
            
            # Step 3: Universal complexity limits (prevent overpowered cards)
            card_text = apply_universal_complexity_limits(card_text, card_data)
        
        # Final validation after all processing (check for post-processing issues)
        if not validate_rules_text(card_text, card_data):
            print(f"⚠️  Final validation failed after text processing - issues introduced during formatting")
            print(f"⚠️  Processed text: {repr(card_text)}")
            # For now, return the text anyway, but log the issue
            # TODO: Could implement full regeneration loop here if needed
        
        print(f"🧠 createCardContent returning:")
        print(f"   📝 Result: {repr(card_text)}")
        print(f"   📏 Length: {len(card_text) if card_text else 0}")
        return card_text
        
    except Exception as e:
        print(f"❌ Error in createCardContent: {e}")
        print("Make sure Mistral model is installed: 'ollama pull mistral:latest'")
        import traceback
        traceback.print_exc()
        return None

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
        width = data.get('width', 408)
        height = data.get('height', 336)
        
        # Extract card data for enhanced prompting
        original_card_data = data.get('cardData', {})
        
        print(f"🔄 Synchronous card generation request (queued)")
        print(f"Prompt: {prompt}")
        
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
        print(f"⏳ Waiting for request {request_id} to complete...")
        max_wait_time = 300  # 5 minutes max wait
        start_wait = time.time()
        
        while True:
            status_info = request_queue.get_status(request_id)
            
            if status_info['status'] == 'completed':
                if status_info.get('error'):
                    print(f"❌ Request failed: {status_info['error']}")
                    response = jsonify({'error': status_info['error']})
                    return add_ngrok_headers(response), 500
                else:
                    print(f"✅ Request completed successfully")
                    # Return in the format frontend expects
                    response = jsonify(status_info['result'])
                    return add_ngrok_headers(response), 200
            
            elif time.time() - start_wait > max_wait_time:
                print(f"⏰ Request timed out after {max_wait_time} seconds")
                response = jsonify({'error': 'Request timed out - took longer than 5 minutes'})
                return add_ngrok_headers(response), 504
            
            else:
                # Still processing, wait a bit
                time.sleep(2)
                continue
                
    except Exception as e:
        print(f"❌ Error processing request: {e}")
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
        
        print(f"📨 Async card generation request received")
        print(f"Prompt: {prompt}")
        
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

# Legacy endpoint for backwards compatibility - uses queue but waits for completion
@app.route('/api/v1/create_card_sync', methods=['POST', 'OPTIONS'])
def create_card_sync():
    """
    Legacy synchronous endpoint - uses queue but waits for completion
    For clients that expect immediate response
    """
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
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
        
        # Optional parameters
        width = data.get('width', 408)
        height = data.get('height', 336)
        
        # Extract card data for enhanced prompting
        original_card_data = data.get('cardData', {})
        
        print(f"🔄 Synchronous card generation request")
        print(f"Prompt: {prompt}")
        
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
        
        # Poll for completion (synchronous behavior)
        print(f"⏳ Waiting for request {request_id} to complete...")
        max_wait_time = 300  # 5 minutes max wait
        start_wait = time.time()
        
        while True:
            status_info = request_queue.get_status(request_id)
            
            if status_info['status'] == 'completed':
                if status_info.get('error'):
                    print(f"❌ Request failed: {status_info['error']}")
                    response = jsonify({'error': status_info['error']})
                    return add_ngrok_headers(response), 500
                else:
                    print(f"✅ Request completed successfully")
                    response = jsonify(status_info['result'])
                    return add_ngrok_headers(response), 200
            
            elif time.time() - start_wait > max_wait_time:
                print(f"⏰ Request timed out after {max_wait_time} seconds")
                response = jsonify({'error': 'Request timed out - took longer than 5 minutes'})
                return add_ngrok_headers(response), 504
            
            else:
                # Still processing, wait a bit
                time.sleep(2)
                continue
                
    except Exception as e:
        print(f"❌ Error processing synchronous request: {e}")
        response = jsonify({'error': f'Synchronous request failed: {str(e)}'})
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
    response = jsonify({
        'status': 'healthy',
        'models': {
            'image_model': 'ready',
            'content_model': 'ready'
        },
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
    print("🚀 Starting Flask server with intelligent queuing...")
    print("Available endpoints:")
    print("  POST /api/v1/create_card - Generate card (sync, frontend compatible)")
    print("  POST /api/v1/create_card_async - Queue card generation (async)")
    print("  GET  /api/v1/card_status/<request_id> - Check async request status")
    print("  GET  /api/v1/queue_status - Get overall queue status")
    print("  POST /api/v1/create_card_sync - Generate card (sync, legacy)")
    print("  GET  /health - Health check")
    print("\n📋 Queue Configuration:")
    print(f"  - Max concurrent requests: {request_queue.max_concurrent}")
    print("  - All endpoints use queue internally to prevent model overload")
    print("  - Frontend-compatible: /api/v1/create_card works synchronously")
    print("\n💡 Usage:")
    print("Frontend: Use /api/v1/create_card (synchronous, queued internally)")
    print("Advanced: Use /api/v1/create_card_async + polling for true async behavior")
    print("\nExample request body:")
    print('{"prompt": "A mystical dragon card", "width": 408, "height": 336}')
    print("\nNote: SDXL-Turbo model will load on first image request")
    
    # Run with HTTP - ngrok will handle HTTPS termination
    app.run(debug=False, host='0.0.0.0', port=5000)
