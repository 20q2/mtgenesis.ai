from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import requests
import base64
import io
from PIL import Image
import tempfile
import ollama
from diffusers import AutoPipelineForText2Image
import torch
import threading
import concurrent.futures

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
        
    try:
        image_pipeline_loading = True
        print("Loading SDXL-Turbo model...")
        
        # Check if CUDA is available, otherwise use CPU
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
            variant = "fp16"
        else:
            device = "cpu"
            torch_dtype = torch.float32
            variant = None
        
        image_pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch_dtype,
            variant=variant
        )
        image_pipeline.to(device)
        print("Image generation model loaded successfully!")
        return image_pipeline
        
    except Exception as e:
        print(f"Failed to load image generation model: {e}")
        print("Will use placeholder images")
        image_pipeline = False  # Mark as failed
        return None
    finally:
        image_pipeline_loading = False

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def createCardImage(prompt, width=384, height=288, card_data=None):
    """
    Generate card image using SDXL-Turbo with color-aware prompts
    Returns the image as base64 encoded string
    """
    try:
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
                    color_palette = ", color palette: prismatic rainbow spectrum, all five magical energies unified, radiant golden light with white marble, deep ocean blues, shadow blacks, molten reds, and emerald greens in perfect harmony"
                
                elif len(colors) == 4:
                    # Four color combinations
                    colors_set = set(colors)
                    if colors_set == {'W', 'U', 'B', 'R'}:  # Missing Green
                        color_palette = ", color palette: urban cityscape colors, marble whites, steel blues, shadow blacks, and burning reds, artificial power and control"
                    elif colors_set == {'W', 'U', 'B', 'G'}:  # Missing Red
                        color_palette = ", color palette: natural order palette, pure whites, ocean blues, deep blacks, and forest greens, growth through structure"
                    elif colors_set == {'W', 'U', 'R', 'G'}:  # Missing Black
                        color_palette = ", color palette: elemental harmony, ivory whites, sapphire blues, flame reds, and emerald greens, pure magical forces"
                    elif colors_set == {'W', 'B', 'R', 'G'}:  # Missing Blue
                        color_palette = ", color palette: primal conflict, bone whites, void blacks, blood reds, and wild greens, instinct over intellect"
                    elif colors_set == {'U', 'B', 'R', 'G'}:  # Missing White
                        color_palette = ", color palette: chaotic spectrum, midnight blues, shadow blacks, molten reds, and living greens, raw untamed power"
                    else:
                        color_palette = ", color palette: four-color magical convergence, rich jewel tones with golden highlights, complex magical energies"
                
                elif len(colors) == 3:
                    # Three color combinations (Shards and Wedges)
                    colors_set = set(colors)
                    if colors_set == {'W', 'U', 'G'}:  # Bant
                        color_palette = ", color palette: noble harmony, pristine whites, clear blues, and vibrant greens, order and growth in balance"
                    elif colors_set == {'U', 'B', 'R'}:  # Grixis
                        color_palette = ", color palette: dark sorcery, deep blues, void blacks, and burning reds, undeath and arcane power"
                    elif colors_set == {'B', 'R', 'G'}:  # Jund
                        color_palette = ", color palette: savage wilderness, shadow blacks, flame reds, and wild greens, primal hunger and survival"
                    elif colors_set == {'R', 'G', 'W'}:  # Naya
                        color_palette = ", color palette: fierce nature, burning reds, emerald greens, and pure whites, wild majesty and power"
                    elif colors_set == {'G', 'W', 'U'}:  # Same as Bant, reordered
                        color_palette = ", color palette: noble harmony, emerald greens, pristine whites, and sapphire blues, civilization meets nature"
                    elif colors_set == {'W', 'B', 'G'}:  # Abzan
                        color_palette = ", color palette: enduring legacy, ivory whites, deep blacks, and forest greens, death feeds life"
                    elif colors_set == {'U', 'R', 'W'}:  # Jeskai
                        color_palette = ", color palette: enlightened mastery, sapphire blues, flame reds, and pure whites, wisdom through conflict"
                    elif colors_set == {'B', 'G', 'U'}:  # Sultai
                        color_palette = ", color palette: decadent corruption, shadow blacks, wild greens, and deep blues, luxury through exploitation"
                    elif colors_set == {'R', 'W', 'B'}:  # Mardu
                        color_palette = ", color palette: warrior's pride, burning reds, bone whites, and void blacks, honor through battle"
                    elif colors_set == {'G', 'U', 'R'}:  # Temur
                        color_palette = ", color palette: frontier survival, emerald greens, ocean blues, and molten reds, adaptation and cunning"
                    else:
                        color_palette = ", color palette: three-color magical fusion, rich jewel tones with metallic accents, complex arcane energies"
                
                elif len(colors) == 2:
                    # Two color guild combinations
                    if 'W' in colors and 'U' in colors:
                        color_palette = ", color palette: pristine whites and sapphire blues, order meets knowledge, divine scholars"
                    elif 'W' in colors and 'B' in colors:
                        color_palette = ", color palette: pure whites contrasting deep blacks, light vs shadow, moral conflict"
                    elif 'W' in colors and 'R' in colors:
                        color_palette = ", color palette: ivory whites with burning reds, righteous fury, controlled passion"
                    elif 'W' in colors and 'G' in colors:
                        color_palette = ", color palette: marble whites with forest greens, civilized nature, structured growth"
                    elif 'U' in colors and 'B' in colors:
                        color_palette = ", color palette: midnight blues and void blacks, forbidden knowledge, dark secrets"
                    elif 'U' in colors and 'R' in colors:
                        color_palette = ", color palette: electric blues with molten reds, creative chaos, innovative destruction"
                    elif 'U' in colors and 'G' in colors:
                        color_palette = ", color palette: ocean blues with living greens, natural evolution, adaptive wisdom"
                    elif 'B' in colors and 'R' in colors:
                        color_palette = ", color palette: shadow blacks with blood reds, ruthless ambition, destructive power"
                    elif 'B' in colors and 'G' in colors:
                        color_palette = ", color palette: decay blacks with wild greens, death and rebirth, savage growth"
                    elif 'R' in colors and 'G' in colors:
                        color_palette = ", color palette: flame reds with primal greens, untamed fury, wild instincts"
                
                # Single color palettes
                elif len(colors) == 1:
                    if 'W' in colors:
                        color_palette = ", color palette: pure whites, warm golds, ivory marble, celestial radiance, divine light"
                    elif 'U' in colors:
                        color_palette = ", color palette: deep sapphire blues, silver accents, twilight purples, crystalline ice tones"
                    elif 'B' in colors:
                        color_palette = ", color palette: void blacks, dark purples, bone whites, sickly greens, shadow tones"
                    elif 'R' in colors:
                        color_palette = ", color palette: burning reds, molten oranges, lightning yellows, volcanic heat"
                    elif 'G' in colors:
                        color_palette = ", color palette: deep forest greens, rich earth browns, bark textures, vibrant life"
                
                # Colorless
                elif 'C' in colors or not colors:
                    color_palette = ", color palette: metallic silvers, steel grays, copper accents, ancient stone textures"
            
            # Create enhanced art prompt
            art_prompt = f"Magic: The Gathering card art, {prompt}, fantasy art style, detailed illustration, game card artwork{color_palette}, high contrast, dramatic lighting"
            print(f"Debug - Final art prompt: {art_prompt}")
            
            image = pipeline(
                prompt=art_prompt,
                num_inference_steps=1,
                guidance_scale=0.0,
                width=width,
                height=height
            ).images[0]
            
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

def createCardContent(prompt, card_data=None):
    """
    Generate card content using Ollama Python client with enhanced context
    Returns the response text from the LLM
    """
    try:
        # Build enhanced prompt based on card properties
        enhanced_prompt = f"Generate ONLY the rules text for a Magic: The Gathering card based on: {prompt}."
        
        if card_data:
            # Analyze mana cost for power level
            cmc = card_data.get('cmc', 0)
            colors = card_data.get('colors', [])
            card_type = card_data.get('type', '').lower()
            power = card_data.get('power')
            toughness = card_data.get('toughness')
            rarity = card_data.get('rarity', 'common').lower()
            
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
            
            # Rarity adjustments
            if rarity == 'common':
                power_level = f"{base_power}, simple and straightforward"
            elif rarity == 'uncommon':
                power_level = f"{base_power}, with interesting utility or synergy"
            elif rarity == 'rare':
                power_level = f"{base_power}, with unique or complex abilities"
            elif rarity == 'mythic':
                power_level = f"{base_power}, with splashy, memorable, and potentially build-around effects"
            else:
                power_level = base_power
            
            # Enhanced color identity guidance with specific mechanics
            color_guidance = ""
            
            if 'W' in colors:
                color_guidance += " White mechanics: protection from colors, lifegain triggers, exile removal, prevent damage, tap creatures, +1/+1 counters on creatures, enchantment synergies, vigilance, first strike."
            
            if 'U' in colors:
                color_guidance += " Blue mechanics: draw cards, counter spells, return to hand, tap/untap permanents, scry, flying creatures, mill cards, copy spells, phase out, control magic."
            
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
            
            # Creature-specific guidance
            creature_guidance = ""
            if 'creature' in card_type and power and toughness:
                try:
                    p = int(power) if power.isdigit() else 0
                    t = int(toughness) if toughness.isdigit() else 0
                    stat_total = p + t
                    
                    if stat_total > cmc * 2.5:
                        creature_guidance = " This creature has high stats for its cost, so abilities should be minimal or have drawbacks."
                    elif stat_total < cmc * 1.5:
                        creature_guidance = " This creature has low stats for its cost, so it should have powerful or multiple abilities."
                    else:
                        creature_guidance = " This creature has balanced stats, so it can have moderate utility abilities."
                except:
                    pass
            
            enhanced_prompt += f" The card costs {cmc} mana and should be {power_level}.{color_guidance}{creature_guidance}"
        
        enhanced_prompt += " Keep the rules text to 3-4 sentences maximum. Do not include card name, mana cost, type line, power/toughness, flavor text, or any other card elements. Return only the ability text that would appear in the text box. Examples: 'Flying, vigilance' or 'When this enters the battlefield, draw a card' or 'Tap: Add one mana of any color'. Be concise and avoid overly complex abilities. IMPORTANT: Do not put quotes around the entire response. Only use quotes when describing granted abilities, such as 'Target creature gains \"Flying\" until end of turn' or 'Creatures you control have \"Tap: Add one mana of any color\"'. Use standard Magic card formatting without surrounding quotes."
        
        response = ollama.generate(
            model='mistral:latest',
            prompt=enhanced_prompt
        )
        
        # Clean up the response
        card_text = response['response'].strip()
        
        # Remove surrounding quotes if present
        if (card_text.startswith('"') and card_text.endswith('"')) or \
           (card_text.startswith("'") and card_text.endswith("'")):
            card_text = card_text[1:-1].strip()
        
        # Limit to 3-4 sentences by splitting on periods and taking first 4
        sentences = [s.strip() for s in card_text.split('.') if s.strip()]
        if len(sentences) > 4:
            card_text = '. '.join(sentences[:4]) + '.'
        
        return card_text
        
    except Exception as e:
        print(f"Error in createCardContent: {e}")
        print("Make sure Mistral model is installed: 'ollama pull mistral:latest'")
        return None

@app.route('/api/v1/create_card', methods=['POST'])
def create_card():
    """
    Main endpoint that accepts POST requests and generates both image and content
    """
    try:
        # Get the request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract prompt from request
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Optional parameters
        width = data.get('width', 384)
        height = data.get('height', 288)
        
        # Extract card data for enhanced prompting
        card_data = data.get('cardData', {})
        
        # Run image and content generation in parallel
        print(f"Starting parallel generation for prompt: {prompt}")
        print(f"Card data received: {card_data}")
        print(f"Colors in card data: {card_data.get('colors', 'No colors found')}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks to run in parallel with card data
            image_future = executor.submit(createCardImage, prompt, width, height, card_data)
            content_future = executor.submit(createCardContent, prompt, card_data)
            
            # Wait for both to complete and get results
            try:
                image_data = image_future.result(timeout=60)  # 60 second timeout for image
                print("Image generation completed")
            except concurrent.futures.TimeoutError:
                print("Image generation timed out")
                image_data = None
            except Exception as e:
                print(f"Image generation failed: {e}")
                image_data = None
            
            try:
                card_data = content_future.result(timeout=30)  # 30 second timeout for content
                print("Content generation completed")
            except concurrent.futures.TimeoutError:
                print("Content generation timed out")
                card_data = None
            except Exception as e:
                print(f"Content generation failed: {e}")
                card_data = None
        
        # Check if both operations were successful
        if image_data is None and card_data is None:
            return jsonify({'error': 'Both image and content generation failed'}), 500
        elif image_data is None:
            print("Warning: Image generation failed, returning content only")
            # Return content only if image generation failed
            response = {
                'cardData': card_data,
                'imageData': None,
                'warning': 'Image generation not available'
            }
            return jsonify(response), 200
        elif card_data is None:
            print("Warning: Content generation failed, returning image only")
            # Return image only if content generation failed
            response = {
                'cardData': None,
                'imageData': image_data,
                'warning': 'Content generation failed'
            }
            return jsonify(response), 200
        
        # Return the response
        response = {
            'cardData': card_data,
            'imageData': image_data
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error in create_card endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Available endpoints:")
    print("  POST /api/v1/create_card - Generate card image and content")
    print("  GET /health - Health check")
    print("\nExample request body:")
    print('{"prompt": "A mystical dragon card", "width": 384, "height": 288}')
    print("\nNote: SDXL-Turbo model will load on first image request")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
