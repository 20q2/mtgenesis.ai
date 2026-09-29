import os
import sys
# Logs use emoji; Windows defaults to cp1252 when output is redirected, which crashes print()
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ===== CONFIGURATION =====
# All toggles (USE_CUDA, MODEL_SIZE, image model settings, ADMIN_PIN, DATA_DIR,
# timeouts) live in config.py.
from config import *

# Track model loading state for cold vs warm timeout detection
_models_loaded = {
    'image': False,
    'content': False
}
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
import image_generation
import rules_text

# The ollama module-level client has no timeout: a stalled Ollama (hung model load,
# driver hiccup) would block the single AI Night text worker forever. With a timeout
# the generate call raises, createCardContent returns None and the card fails.
OLLAMA_TIMEOUT_SECONDS = 120
ollama_client = ollama.Client(timeout=OLLAMA_TIMEOUT_SECONDS)
print(f"🔍 Python executable: {sys.executable}")
print(f"🔍 Python version: {sys.version}")
print(f"🔍 Python path: {sys.path[:3]}...")  # Show first 3 paths
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
request_queue = RequestQueue(max_concurrent=2)  # Allow max 2 concurrent card generations

# Global state tracking for first job completion (for dynamic loading times)
first_job_completed = False

def finalize_card(card_params, generated_text, art_b64, force_name=None):
    """
    Post-process generated rules text and render the complete card.

    Shared by the legacy process_card_generation route and the AI Night
    GenerationQueue, which uses it as its RenderFn.

    Steps:
    - parse the LLM text (JSON with name/description/flavorText, or plain rules text)
    - apply force_name, so a set's commander name overrides any name the LLM chose
    - replace ~ with the card name, fix bullet points and periods
    - generate missing creature P/T and Vehicle crew cost
    - render with card_renderer.generate_card_image

    Returns (final card dict, rendered card as raw base64 PNG or None).
    card_params is not mutated. Rendering errors propagate to the caller.
    """
    import time

    # Step 1: Text processing and parsing
    text_processing_start = time.time()
    print("  📝 Step 1: Processing text data...")
    updated_card_data = card_params.copy()
    print(f"🔍 Original card data keys: {list(card_params.keys())}")
    print(f"🔍 Original description: {repr(card_params.get('description', 'NO DESCRIPTION'))}")
    if generated_text:
        try:
            # Try to parse structured card data
            parsed_text = json.loads(generated_text)
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
            updated_card_data['description'] = generated_text
            print(f"Updated card data with plain text content")

    # A forced name (a set's commander name) wins over any LLM-chosen name, and is
    # applied before ~ replacement so the rules text and the render both use it
    if force_name:
        updated_card_data['name'] = force_name

    if generated_text:
        # Apply text processing and ability reordering to the description
        if 'description' in updated_card_data and updated_card_data['description']:
            original_text = updated_card_data['description']

            # Apply the text processing steps that were missing
            processed_text = original_text
            print(f"🔍 Step 0 - Original: {repr(processed_text)}")

            # Step 1: Clean up text formatting
            processed_text = processed_text.replace('\n\n', '\n')  # Double newlines to single
            processed_text = processed_text.replace(' ~ ', f' {updated_card_data.get("name", "~")} ')  # Replace ~ with card name
            processed_text = processed_text.replace('~', updated_card_data.get("name", "~"))  # Replace any remaining ~
            print(f"🔍 Step 1 - After cleanup: {repr(processed_text)}")

            # Step 1.5: Fix markdown bullet points (convert "* item" to "item")
            processed_text = fix_markdown_bullet_points(processed_text)
            print(f"🔍 Step 1.5 - After bullet fix: {repr(processed_text)}")

            # Step 3: Ensure periods on abilities
            processed_text = ensure_periods_on_abilities(processed_text)
            print(f"🔍 Step 3 - After period fix: {repr(processed_text)}")

            updated_card_data['description'] = processed_text
            print(f"🔧 Content model parsed output: {repr(processed_text)}")
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
    subtype_and_type_line = f"{updated_card_data.get('subtype') or ''} {updated_card_data.get('typeLine') or ''}"
    is_vehicle = 'vehicle' in subtype_and_type_line.lower()
    if (('creature' in (updated_card_data.get('type') or '').lower() or is_vehicle) and
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
    if is_vehicle:
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
    card_image_data = card_renderer.generate_card_image(updated_card_data, art_b64)
    rendering_time = time.time() - rendering_start
    print(f"   🎨 Card rendering: {rendering_time:.2f}s")

    return updated_card_data, card_image_data

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
            _, card_image_data = finalize_card(original_card_data, generated_card_text, image_data)
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

app = Flask(__name__)
# Bounds every request body (413 beyond it): AI Night JSON bodies are a few KB, and an
# unbounded prompt on a locked set would be re-downloaded by every voter on every poll.
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024
CORS(app, 
     origins=["*"],  # Allow all origins for ngrok + S3
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "ngrok-skip-browser-warning", "Accept", "Cache-Control",
                    "X-User-Id", "X-Admin-Pin"],
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
    """Ensure all responses have CORS headers for HTTPS/ngrok compatibility.

    Assign (not .add) so each header stays single-valued: on exception-handled
    responses Flask-CORS has already set Access-Control-Allow-Origin, and a
    second value makes browsers reject the response.
    """
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,ngrok-skip-browser-warning,Accept,Cache-Control,X-User-Id,X-Admin-Pin'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    response.headers['Access-Control-Max-Age'] = '86400'
    response.headers['ngrok-skip-browser-warning'] = 'any'
    return response

def createCardImage(prompt, width=408, height=336, card_data=None):
    """
    Generate card art. Thin wrapper over image_generation.generate_art; width and
    height are ignored (the art is always config.ART_BOX_SIZE).
    Returns raw base64 PNG (no data: prefix); raises if the image model fails.
    """
    return image_generation.generate_art(prompt, card_data)

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
            # Keyword lines ("Flying, trample", "Equip {2}", "Crew 3") take no period;
            # everything else ends with one unless it already ends in punctuation/quotes
            if rules_text.is_keyword_line(ability):
                ability = ability.rstrip('.')
            elif not ability.endswith(('.', '!', '?', ':', '"', "'")):
                ability += '.'
        fixed_abilities.append(ability)
    
    return '\n'.join(fixed_abilities)

def fix_markdown_bullet_points(card_text):
    """
    Convert markdown-style bullet points (* item) to proper Magic card formatting.
    Magic cards don't use bullet points - they use proper sentence structure.
    """
    if not card_text:
        return card_text
    
    # Split into lines and process each one
    lines = card_text.split('\n')
    fixed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            fixed_lines.append('')
            continue
            
        # Check if line starts with "* " (markdown bullet point)
        if line.startswith('* '):
            # Remove the "* " and treat as a regular ability
            fixed_line = line[2:].strip()
            # Ensure it starts with a capital letter
            if fixed_line and fixed_line[0].islower():
                fixed_line = fixed_line[0].upper() + fixed_line[1:]
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


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
        
        # Variable (*) power/toughness only comes from the request itself: the rules text
        # is written before stats, so a * rolled here would never be defined on the card.
        
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
        if ability_count >= 5:
            base_total -= 2  # Very complex creatures get -2 total stats
        elif ability_count >= 3:
            base_total -= 1  # Complex creatures get -1 total stats
        
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

def createCardContent(prompt, card_data=None):
    """
    Rules text for a card, generated by config.TEXT_MODEL and cleaned into legal
    templating by rules_text.py. Returns None when the model call fails (Ollama down,
    timeout, model not pulled), so callers can mark the card failed.
    """
    try:
        return rules_text.generate_rules_text(
            prompt, card_data, ollama_client, TEXT_MODEL,
            attempts=TEXT_ATTEMPTS, think=TEXT_THINK,
            keep_alive="30m",  # stay resident between cards on AI Night
        )
    except Exception as e:
        print(f"❌ Error in createCardContent: {e}")
        print(f"Make sure the text model is installed: 'ollama pull {TEXT_MODEL}'")
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
        print(f"⏳ Waiting for request {request_id} to complete...")
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
                    print(f"✅ Request completed successfully")
                    # Return in the format frontend expects
                    response = jsonify(status_info['result'])
                    return add_ngrok_headers(response), 200
            
            elif time.time() - start_wait > max_wait_time:
                print(f"⏰ Request timed out after {max_wait_time} seconds")
                response = jsonify({'error': 'Request timed out - took longer than 10 minutes'})
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
        
        print(f"🔄 Sync card generation request: {prompt}")
        
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
                    print(f"✅ Request completed successfully")
                    response = jsonify(status_info['result'])
                    return add_ngrok_headers(response), 200
            
            elif time.time() - start_wait > max_wait_time:
                print(f"⏰ Request timed out after {max_wait_time} seconds")
                response = jsonify({'error': 'Request timed out - took longer than 10 minutes'})
                return add_ngrok_headers(response), 504
            
            else:
                # Still processing, wait a bit
                time.sleep(2)
                continue
                
    except Exception as e:
        print(f"❌ Error processing synchronous request: {e}")
        response = jsonify({'error': f'Synchronous request failed: {str(e)}'})
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

def warn_about_admin_pin(pin):
    """Print a loud startup banner when ADMIN_PIN is empty or still the default."""
    from api_routes import admin_pin_warning

    warning = admin_pin_warning(pin)
    if warning:
        bar = "!" * 78
        print(f"\n{bar}\n!!! WARNING: {warning}\n{bar}\n")


def init_ai_night(app):
    """
    Set up AI Night: data folders, SQLite storage, the two-stage generation queue
    and the /api/v1 blueprint (spec §3-§5). Returns the GenerationQueue.

    Called only from the __main__ block, so importing app.py (e.g. in tests)
    opens no database and starts no worker threads.
    """
    from pathlib import Path

    import image_generation
    from api_routes import create_api_blueprint
    from generation_queue import GenerationQueue
    from storage import Storage

    data_dir = Path(DATA_DIR)
    (data_dir / "art").mkdir(parents=True, exist_ok=True)
    (data_dir / "cards").mkdir(parents=True, exist_ok=True)

    warn_about_admin_pin(ADMIN_PIN)
    storage = Storage(data_dir / "mtgenesis.db")
    gen_queue = GenerationQueue(storage, data_dir, createCardContent,
                                image_generation.generate_art, finalize_card)
    gen_queue.recover_on_startup()
    app.register_blueprint(create_api_blueprint(storage, gen_queue, data_dir, ADMIN_PIN),
                           url_prefix="/api/v1")
    print(f"🌙 AI Night ready: data in {data_dir}")
    return gen_queue

if __name__ == '__main__':
    print("🚀 Starting Flask server with intelligent queuing...")
    gen_queue = init_ai_night(app)
    print("Available endpoints:")
    print("  POST /api/v1/create_card - Generate card (sync, frontend compatible)")
    print("  POST /api/v1/create_card_async - Queue card generation (async)")
    print("  GET  /api/v1/card_status/<request_id> - Check async request status")
    print("  POST /api/v1/create_card_sync - Generate card (sync, legacy)")
    print("  GET  /health - Health check")
    print("\n🌙 AI Night endpoints (X-User-Id header on user routes, X-Admin-Pin on admin routes):")
    print("  POST /api/v1/users/login - Log in or register by username")
    print("  GET  /api/v1/me/cards - My cards, newest first")
    print("  GET  /api/v1/me/sets/current - My current commander set")
    print("  POST /api/v1/generations - Queue 1 free-play card or a 3-card commander set")
    print("  GET  /api/v1/cards/<id> - Card status, queue position and ETA")
    print("  POST /api/v1/cards/<id>/reroll - Reroll a set card")
    print("  POST /api/v1/sets/<id>/lock | /unlock - Lock a set into the open event, or unlock it")
    print("  GET  /api/v1/events/current | /events | /events/<id> - Events and their locked sets")
    print("  POST /api/v1/votes - Vote for a card in a locked set")
    print("  GET  /api/v1/queue_status - Generation queue status and ETA")
    print("  GET  /api/v1/media/cards/<id>.png | /media/art/<id>.png - Rendered card and artwork")
    print("  POST /api/v1/admin/events | /admin/events/<id>/close - Host event controls")
    print("\n📋 Queue Configuration:")
    print(f"  - Max concurrent requests: {request_queue.max_concurrent}")
    print("  - All endpoints use queue internally to prevent model overload")
    print("  - Frontend-compatible: /api/v1/create_card works synchronously")
    print("\n💡 Usage:")
    print("Frontend: Use /api/v1/create_card (synchronous, queued internally)")
    print("Advanced: Use /api/v1/create_card_async + polling for true async behavior")
    print("\nExample request body:")
    print('{"prompt": "A mystical dragon card", "width": 408, "height": 336}')
    print("\nNote: the image model loads on first image request")
    
    # Run with HTTP - ngrok will handle HTTPS termination
    app.run(debug=False, host='0.0.0.0', port=5000)
