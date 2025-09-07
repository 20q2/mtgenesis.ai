from PIL import Image, ImageDraw, ImageFont
import os
import base64
import io
import re
import numpy as np
from typing import Dict, List, Optional, Tuple
try:
    from cairosvg import svg2png
    SVG_SUPPORT = True
    SVG_METHOD = 'cairosvg'
    print("SVG support enabled with CairoSVG - mana symbols will render as images")
except (ImportError, OSError) as e:
    try:
        from wand.image import Image as WandImage
        SVG_SUPPORT = True
        SVG_METHOD = 'wand'
        print("SVG support enabled with Wand - mana symbols will render as images")
    except ImportError as e2:
        print(f"SVG support not available (CairoSVG: {e}, Wand: {e2}), mana symbols will use PNG fallback")
        SVG_SUPPORT = False
        SVG_METHOD = None

class MagicCardRenderer:
    """
    Simplified Magic card renderer focusing on essential components:
    - Normal vs Legendary frames
    - Color-based frame selection
    - Power/Toughness boxes for creatures
    """
    
    def __init__(self):
        self.assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        self.fonts_dir = os.path.join(self.assets_dir, 'fonts')
        self.m15_dir = os.path.join(self.assets_dir, 'm15')  # Modern frame style
        self.mana_symbols_dir = os.path.join(self.assets_dir, 'manaSymbols')
        
        # Frame cache to avoid loading images repeatedly
        self._image_cache: Dict[str, Image.Image] = {}
        
        # Card dimensions (proper Magic card aspect ratio based on 744x1039)
        self.card_width = 496   # Keep current width
        self.card_height = 693  # Adjusted to match real Magic card proportions (496 / 0.716)
        
        # Text positioning for M15 frame layout (adjusted for proper aspect ratio)
        self.name_pos = (44, 42)      # Card name position
        self.mana_cost_pos = (458, 50)  # Mana cost position
        self.type_pos = (44, 400)     # Type line moved up 2px more (402 - 2 = 400)
        self.text_pos = (40, 440)     # Rules text moved up 8px (446 - 8 = 438)  
        self.text_width = 385         # Keep same text width
        self.pt_pos = (416, 643)      # P/T moved left 4px more (420 - 4 = 416)
        self.art_pos = (38, 78)       # Art position
        self.art_size = (420, 307)    # Art box made taller to fill space properly (1px shorter)
        
        # Load fonts
        self.load_fonts()
        
        # Set symbol/logo positioning (right side of type line, properly aligned)
        self.logo_pos = (self.card_width - 63, self.type_pos[1] - 2)  # Moved 4px more left (59 + 4 = 63)
        self.logo_size = 25  # Made bigger (15% increase: 22 * 1.15 = 25.3, rounded to 25)
    
    def _load_image_cached(self, image_path: str) -> Image.Image:
        """
        Load an image with caching to avoid repeated disk reads.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        if image_path not in self._image_cache:
            self._image_cache[image_path] = Image.open(image_path)
        return self._image_cache[image_path].copy()  # Return copy to avoid modifying cached image
    
    def clear_image_cache(self):
        """Clear the image cache to free up memory if needed."""
        self._image_cache.clear()
        print("Image cache cleared")
    
    def get_cache_stats(self) -> dict:
        """Get statistics about the current cache state."""
        return {
            'cached_images': len(self._image_cache),
            'cache_keys': list(self._image_cache.keys()) if len(self._image_cache) < 20 else f"{len(self._image_cache)} images cached"
        }
    
    def add_rounded_corners(self, image: Image.Image, radius: int = 20) -> Image.Image:
        """
        Add rounded corners to an image
        
        Args:
            image: PIL Image to add rounded corners to
            radius: Corner radius in pixels
            
        Returns:
            Image with rounded corners
        """
        # Create a mask for rounded corners
        mask = Image.new('L', image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        
        # Draw rounded rectangle on the mask
        mask_draw.rounded_rectangle(
            [(0, 0), image.size], 
            radius=radius, 
            fill=255
        )
        
        # Apply the mask to the image
        output = Image.new('RGBA', image.size, (0, 0, 0, 0))
        output.paste(image, (0, 0))
        output.putalpha(mask)
        
        return output
    
    def load_fonts(self):
        """Load the Beleren fonts for card text"""
        try:
            self.beleren_bold_path = os.path.join(self.fonts_dir, 'Beleren2016-Bold.ttf')
            beleren_smallcaps_path = os.path.join(self.fonts_dir, 'Beleren2016SmallCaps-Bold.ttf')
            self.elegant_garamond_path = os.path.join(self.assets_dir, 'Elegant Garamond Regular.otf')
            
            # Different font sizes for different text elements
            self.title_font = ImageFont.truetype(self.beleren_bold_path, 22)  # Increased by 1 more (21 + 1 = 22)
            self.type_font = ImageFont.truetype(self.beleren_bold_path, 20)  # Increased by 1 more (19 + 1 = 20)
            
            # Check if Elegant Garamond is available for rules text
            try:
                # Test load to see if Elegant Garamond is available
                ImageFont.truetype(self.elegant_garamond_path, 27)
                self.use_elegant_garamond = True
                print("Elegant Garamond available for dynamic rules text sizing")
            except Exception as font_e:
                print(f"Could not load Elegant Garamond font: {font_e}")
                print("Will use Beleren for dynamic rules text sizing")
                self.use_elegant_garamond = False
            
            # Default rules text font (will be dynamically sized)
            self.text_font = self.get_text_font(27)  # Start with 27px default
            
            self.mana_font = ImageFont.truetype(self.beleren_bold_path, 16)
            self.pt_font = ImageFont.truetype(self.beleren_bold_path, 24)
            
            # Available rules box height (from text_pos to roughly PT area)
            # Text starts at 440px, P/T at 643px, giving ~200px usable space
            # Extended to 220px to allow larger font sizes and better text utilization
            self.available_text_height = 220  # Extended from 190px for better text fitting
            
            print("Fonts loaded successfully with dynamic sizing capability")
        except Exception as e:
            print(f"Error loading fonts: {e}")
            # Fallback to default font
            self.title_font = ImageFont.load_default()
            self.type_font = ImageFont.load_default()
            self.text_font = ImageFont.load_default()
            self.mana_font = ImageFont.load_default()
            self.pt_font = ImageFont.load_default()
    
    def get_text_font(self, size: int):
        """Get rules text font at specified size"""
        if self.use_elegant_garamond:
            return ImageFont.truetype(self.elegant_garamond_path, size)
        else:
            return ImageFont.truetype(self.beleren_bold_path, size)
    
    def calculate_optimal_font_size(self, text: str, max_width: int, max_height: int) -> int:
        """
        Calculate the optimal font size for text to fit within given constraints.
        Uses progressive font size reduction from 27px down to minimum readable size.
        """
        # Font size tiers (MTG standard 27px down to minimum readable)
        # Added intermediate sizes for better readability: 19, 16, 14, 13
        font_sizes = [27, 24, 21, 19, 18, 16, 15, 14, 12]  # Progressive reduction with more granular steps
        
        for font_size in font_sizes:
            if self._text_fits_in_bounds(text, max_width, max_height, font_size):
                print(f"📏 Optimal font size for rules text: {font_size}px")
                return font_size
        
        # If even minimum doesn't fit, use smallest anyway and let it overflow
        print(f"⚠️  Text too large even at minimum size, using 12px")
        return 12
    
    def _text_fits_in_bounds(self, text: str, max_width: int, max_height: int, font_size: int) -> bool:
        """Check if text fits within bounds at given font size"""
        try:
            font = self.get_text_font(font_size)
            
            # Create a temporary image for text measurement
            temp_img = Image.new('RGB', (max_width + 100, max_height + 100), 'white')
            draw = ImageDraw.Draw(temp_img)
            
            # Simulate the same text wrapping logic as draw_text_with_mana_symbols
            # but just measure height instead of actually drawing
            
            paragraphs = text.split('\n')
            line_height = int(font_size * 1.2)  # 1.2x line spacing
            ability_separation = int(font_size * 1.45)  # Extra space between abilities (increased from 1.35x)
            
            total_height = 0
            
            for i, paragraph in enumerate(paragraphs):
                if not paragraph.strip():
                    continue
                    
                # Count wrapped lines for this paragraph
                words = paragraph.split(' ')
                current_line_words = []
                
                for word in words:
                    # Clean word of mana symbols for width calculation (approximate)
                    clean_word = word.replace('{', '').replace('}', '').replace('/', '')
                    test_line = ' '.join(current_line_words + [clean_word])
                    
                    bbox = draw.textbbox((0, 0), test_line, font=font)
                    if bbox[2] <= max_width:
                        current_line_words.append(clean_word)
                    else:
                        if current_line_words:
                            total_height += line_height
                        current_line_words = [clean_word]
                
                # Add final line of paragraph
                if current_line_words:
                    total_height += line_height
                
                # Add ability separation (except for last paragraph)
                if i < len(paragraphs) - 1:
                    total_height += (ability_separation - line_height)
            
            fits = total_height <= max_height
            if not fits:
                print(f"   Font size {font_size}px: {total_height}px height > {max_height}px limit")
            
            return fits
            
        except Exception as e:
            print(f"Error measuring text at font size {font_size}: {e}")
            return False
    
    def sort_colors_for_pipes(self, colors: List[str]) -> List[str]:
        """Sort colors for pipe display - ensure proper visual ordering for common combinations"""
        if len(colors) != 2:
            return colors
        
        # Common two-color guild pairings with preferred left-right visual ordering
        guild_orders = {
            # Allied colors (adjacent on color pie) - traditional left-right ordering
            ('W', 'U'): ['W', 'U'],  # Azorius: White left, Blue right
            ('U', 'B'): ['U', 'B'],  # Dimir: Blue left, Black right  
            ('B', 'R'): ['B', 'R'],  # Rakdos: Black left, Red right
            ('R', 'G'): ['R', 'G'],  # Gruul: Red left, Green right
            ('G', 'W'): ['G', 'W'],  # Selesnya: Green left, White right
            
            # Enemy colors (opposite on color pie) - visual preference ordering
            ('W', 'B'): ['W', 'B'],  # Orzhov: White left, Black right
            ('U', 'R'): ['U', 'R'],  # Izzet: Blue left, Red right
            ('B', 'G'): ['B', 'G'],  # Golgari: Black left, Green right  
            ('R', 'W'): ['R', 'W'],  # Boros: Red left, White right
            ('G', 'U'): ['G', 'U'],  # Simic: Green left, Blue right
        }
        
        # Try both orders to find match
        colors_set = set(colors)
        for (c1, c2), preferred_order in guild_orders.items():
            if colors_set == {c1, c2}:
                return preferred_order
        
        # Fallback to WUBRG order if no specific guild match
        wubrg_order = ['W', 'U', 'B', 'R', 'G']
        result = [color for color in wubrg_order if color in colors]
        return result
    
    def extract_pinlines_from_frame(self, frame_color: str) -> Image.Image:
        """
        Extract pinlines from a specific color frame using the pinline mask.
        Returns just the pinlines from that frame.
        """
        try:
            # Load the color frame and pinline mask
            frame_path = os.path.join(self.m15_dir, f'm15Frame{frame_color}.png')
            mask_path = os.path.join(self.m15_dir, 'm15MaskPinline.png')
            
            if not os.path.exists(frame_path) or not os.path.exists(mask_path):
                print(f"Warning: Frame or mask not found for pinline extraction: {frame_color}")
                return None
            
            frame = self._load_image_cached(frame_path)
            mask = self._load_image_cached(mask_path)
            
            # Ensure mask is same size as frame
            if mask.size != frame.size:
                mask = mask.resize(frame.size, Image.Resampling.LANCZOS)
            
            
            # Create a proper mask - the pinline areas are red (255, 64, 0), transparent areas are black
            # Convert to RGBA first to handle transparency properly
            if mask.mode != 'RGBA':
                mask = mask.convert('RGBA')
            
            # Create a grayscale mask where red areas become white (255) and everything else becomes black (0)
            # OPTIMIZED VERSION using numpy for 100x+ speed improvement
            
            # Convert to numpy array for fast pixel operations
            mask_array = np.array(mask)
            grayscale_array = np.zeros((mask.height, mask.width), dtype=np.uint8)
            
            # Extract RGB channels
            r_channel = mask_array[:, :, 0]
            g_channel = mask_array[:, :, 1] 
            b_channel = mask_array[:, :, 2]
            
            # Check if this is a red pinline area (approximately red: 255, green: 64, blue: 0)
            pinline_condition = (r_channel > 200) & (g_channel < 100) & (b_channel < 50)
            grayscale_array[pinline_condition] = 255  # White = pinline area
            pinline_pixels = np.sum(pinline_condition)
            
            # Convert back to PIL Image
            grayscale_mask = Image.fromarray(grayscale_array, 'L')
            
            mask = grayscale_mask
            
            # Create transparent image for the pinlines
            pinlines = Image.new('RGBA', frame.size, (0, 0, 0, 0))
            
            # Use the mask to extract only the pinline areas from the frame
            # The mask white areas = pinlines, black areas = everything else
            frame_rgba = frame.convert('RGBA')
            
            # Apply the mask to extract pinlines with enhanced opacity
            # OPTIMIZED VERSION using numpy arrays for 100x+ speed improvement
            frame_array = np.array(frame_rgba)
            mask_array = np.array(mask)
            
            # Create enhanced frame array
            enhanced_array = np.zeros_like(frame_array)
            
            # White areas in mask = pinlines
            pinline_mask = mask_array > 128
            
            # Copy RGB channels for pinline areas
            enhanced_array[pinline_mask] = frame_array[pinline_mask]
            # Set alpha to full opacity for pinlines
            if frame_array.shape[2] == 4:  # RGBA
                enhanced_array[pinline_mask, 3] = 255
            
            # Convert back to PIL Image
            enhanced_frame = Image.fromarray(enhanced_array, 'RGBA')
            
            pinlines = enhanced_frame
            
            print(f"Extracted pinlines from {frame_color} frame")
            return pinlines
            
        except Exception as e:
            print(f"Error extracting pinlines from {frame_color} frame: {e}")
            return None

    def blend_two_color_pinlines(self, colors: List[str]) -> Image.Image:
        """
        Create gradient-blended pinlines for two-color cards.
        Blends left-to-right in WUBRG order with smooth transition.
        """
        try:
            # Sort colors for proper pipe display
            sorted_colors = self.sort_colors_for_pipes(colors)
            if len(sorted_colors) != 2:
                raise ValueError(f"Expected exactly 2 colors, got {len(sorted_colors)}")
            
            left_color, right_color = sorted_colors
            
            # Extract pinlines from both color frames
            left_pinlines = self.extract_pinlines_from_frame(left_color)
            right_pinlines = self.extract_pinlines_from_frame(right_color)
            
            if left_pinlines is None or right_pinlines is None:
                print(f"Warning: Could not extract pinlines for {left_color}/{right_color}")
                return None
            
            # Ensure both pinline images are the same size
            if left_pinlines.size != right_pinlines.size:
                target_size = (max(left_pinlines.width, right_pinlines.width),
                             max(left_pinlines.height, right_pinlines.height))
                left_pinlines = left_pinlines.resize(target_size, Image.Resampling.LANCZOS)
                right_pinlines = right_pinlines.resize(target_size, Image.Resampling.LANCZOS)
            
            # Create the same gradient mask we use for crowns
            # OPTIMIZED VERSION using numpy for 100x+ speed improvement
            width, height = left_pinlines.size
            
            # Create gradient using numpy arrays
            x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
            progress = x_coords / width  # 0.0 (left) to 1.0 (right)
            
            # Smooth S-curve transition (same as crown blending)
            mask_values = np.zeros_like(progress, dtype=np.uint8)
            
            # Full left pinlines
            left_mask = progress <= 0.3
            mask_values[left_mask] = 255
            
            # Full right pinlines  
            right_mask = progress >= 0.7
            mask_values[right_mask] = 0
            
            # Smooth transition zone (30% to 70%)
            transition_mask = (progress > 0.3) & (progress < 0.7)
            transition_progress = (progress[transition_mask] - 0.3) / 0.4
            smooth_progress = 3 * transition_progress**2 - 2 * transition_progress**3
            mask_values[transition_mask] = (255 * (1.0 - smooth_progress)).astype(np.uint8)
            
            # Convert to PIL Image
            gradient_mask = Image.fromarray(mask_values, 'L')
            
            # Create blended pinlines with enhanced opacity
            blended_pinlines = Image.composite(left_pinlines, right_pinlines, gradient_mask)
            
            # Enhance the overall opacity of the blended pinlines
            # OPTIMIZED VERSION using numpy arrays for 100x+ speed improvement
            blended_array = np.array(blended_pinlines)
            enhanced_array = blended_array.copy()
            
            # Only enhance non-transparent pixels
            if blended_array.shape[2] == 4:  # RGBA
                alpha_channel = blended_array[:, :, 3]
                non_transparent = alpha_channel > 0
                # Make all blended pinlines fully opaque
                enhanced_array[non_transparent, 3] = 255
            
            # Convert back to PIL Image
            enhanced_blended = Image.fromarray(enhanced_array, 'RGBA')
            
            blended_pinlines = enhanced_blended
            
            print(f"Created gradient pinlines: {left_color} (left) → {right_color} (right)")
            return blended_pinlines
            
        except Exception as e:
            print(f"Error creating gradient pinlines: {e}")
            return None

    def blend_two_color_crown(self, colors: List[str]) -> Image.Image:
        """
        Create a gradient-blended crown for two-color legendary cards.
        Blends left-to-right in WUBRG order with smooth transition at 50% mark.
        """
        try:
            # Sort colors for proper pipe display
            sorted_colors = self.sort_colors_for_pipes(colors)
            if len(sorted_colors) != 2:
                raise ValueError(f"Expected exactly 2 colors, got {len(sorted_colors)}")
            
            left_color, right_color = sorted_colors
            
            # Load both crown assets
            left_crown_path = os.path.join(self.m15_dir, f'm15Crown{left_color}.png')
            right_crown_path = os.path.join(self.m15_dir, f'm15Crown{right_color}.png')
            
            if not os.path.exists(left_crown_path) or not os.path.exists(right_crown_path):
                print(f"Warning: Crown assets not found for {left_color}/{right_color}, falling back to multicolor")
                # Fallback to multicolor crown
                fallback_path = os.path.join(self.m15_dir, 'm15CrownM.png')
                if os.path.exists(fallback_path):
                    return self._load_image_cached(fallback_path)
                else:
                    return None
            
            left_crown = self._load_image_cached(left_crown_path)
            right_crown = self._load_image_cached(right_crown_path)
            
            # Ensure both crowns are the same size
            if left_crown.size != right_crown.size:
                # Resize to match the larger crown
                target_size = (max(left_crown.width, right_crown.width), 
                             max(left_crown.height, right_crown.height))
                left_crown = left_crown.resize(target_size, Image.Resampling.LANCZOS)
                right_crown = right_crown.resize(target_size, Image.Resampling.LANCZOS)
            
            # Create gradient mask for blending
            width, height = left_crown.size
            
            # Create horizontal gradient mask (left=255/white, right=0/black)
            # OPTIMIZED VERSION using numpy for 100x+ speed improvement
            
            # Create gradient using numpy arrays
            x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
            progress = x_coords / width  # 0.0 (left) to 1.0 (right)
            
            # Initialize mask values array
            mask_values = np.zeros_like(progress, dtype=np.uint8)
            
            # Full left crown
            left_mask = progress <= 0.3
            mask_values[left_mask] = 255
            
            # Full right crown
            right_mask = progress >= 0.7
            mask_values[right_mask] = 0
            
            # Smooth transition zone (30% to 70%)
            transition_mask = (progress > 0.3) & (progress < 0.7)
            transition_progress = (progress[transition_mask] - 0.3) / 0.4  # 0.0 to 1.0 in transition zone
            # Apply smooth curve
            smooth_progress = 3 * transition_progress**2 - 2 * transition_progress**3
            mask_values[transition_mask] = (255 * (1.0 - smooth_progress)).astype(np.uint8)
            
            # Convert to PIL Image
            gradient_mask = Image.fromarray(mask_values, 'L')
            
            # Create blended crown
            blended_crown = Image.composite(left_crown, right_crown, gradient_mask)
            
            print(f"Created gradient crown: {left_color} (left) → {right_color} (right)")
            return blended_crown
            
        except Exception as e:
            print(f"Error creating gradient crown: {e}")
            # Fallback to multicolor crown
            fallback_path = os.path.join(self.m15_dir, 'm15CrownM.png')
            if os.path.exists(fallback_path):
                return self._load_image_cached(fallback_path)
            else:
                return None

    def get_color_code(self, colors: List[str], card_type: str = '') -> str:
        """Get single character code for frame colors"""
        # FIRST check if this is an artifact (based on typeline, not colors)
        if 'artifact' in card_type.lower():
            return 'Artifact'  # Use special artifact frame
        elif not colors or colors == ['C']:
            return 'A'  # Regular colorless
        elif len(colors) == 1:
            return colors[0]
        else:
            return 'M'  # Multicolor
    
    def get_color_specific_mask(self, mana_colors: List[str]) -> str:
        return 'm15MaskBorderSliver_red.png'
    
    def load_base_frame(self, colors: List[str], is_legendary: bool = False, card_type: str = '', mana_cost: str = '') -> Image.Image:
        """
        Load the base card frame based on colors and card type
        Special handling for artifacts with colored mana costs
        """
        color_code = self.get_color_code(colors, card_type)
        
        # Handle artifact frames specially
        if color_code == 'Artifact':
            # Check if artifact has colored mana symbols
            mana_colors = self.extract_colors_from_mana_cost(mana_cost) if mana_cost else []
            
            print(f"🔍 ARTIFACT DEBUG: color_code='{color_code}', mana_cost='{mana_cost}', mana_colors={mana_colors}")
            
            if mana_colors:
                print(f"🎨 Artifact with colored mana cost detected. Mana colors: {mana_colors}")
                # Create layered frame: [colored frame] -> [masked artifact frame] -> [pipes if needed]
                return self.create_colored_artifact_frame(mana_colors, is_legendary)
            else:
                print(f"🎨 Pure colorless artifact - using standard M15A frame")
                # Pure artifact - use standard M15A frame
                frame_file = 'm15FrameA.png'
                frame_path = os.path.join(self.m15_dir, frame_file)
        else:
            # M15 frame files use single letter codes
            frame_file = f'm15Frame{color_code}.png'
            frame_path = os.path.join(self.m15_dir, frame_file)
        
        try:
            frame = self._load_image_cached(frame_path)
            frame = frame.resize((self.card_width, self.card_height), Image.Resampling.LANCZOS)
            
            # Special handling for two-color cards - add gradient pinlines to gold frame
            if len(colors) == 2 and all(c in ['W', 'U', 'B', 'R', 'G'] for c in colors) and color_code == 'M':
                print(f"Adding gradient pinlines to gold frame for two-color card: {colors}")
                gradient_pinlines = self.blend_two_color_pinlines(colors)
                if gradient_pinlines is not None:
                    # Resize gradient pinlines to match frame
                    gradient_pinlines = gradient_pinlines.resize((self.card_width, self.card_height), Image.Resampling.LANCZOS)
                    
                    # Simply paste the gradient pinlines directly on top of the gold frame
                    frame.paste(gradient_pinlines, (0, 0), gradient_pinlines)
                    print("Applied gradient pinlines to gold frame")
                else:
                    print("Failed to create gradient pinlines, using standard gold frame")
            
            # Add legendary crown if needed
            if is_legendary:
                frame = self.add_legendary_crown(frame, colors, card_type)
            
            return frame
        except Exception as e:
            print(f"Error loading frame {frame_file}: {e}")
            # Fallback to basic frame
            return self.create_fallback_frame()
    
    def create_colored_artifact_frame(self, mana_colors: List[str], is_legendary: bool = False) -> Image.Image:
        """
        Create layered artifact frame with colored mana cost
        New approach: [colored frame base] -> [MASKED artifact frame overlay] -> [crown/text]
        """
        try:
            print(f"🎨 Creating colored artifact frame with mana colors: {mana_colors}")
            
            # Step 1: Load the colored base frame based on mana colors
            color_code = self.get_color_code(mana_colors, '')
            colored_frame_file = f'm15Frame{color_code}.png'
            colored_frame_path = os.path.join(self.m15_dir, colored_frame_file)
            
            colored_frame = self._load_image_cached(colored_frame_path)
            colored_frame = colored_frame.resize((self.card_width, self.card_height), Image.Resampling.LANCZOS)
            print(f"✅ Loaded colored base frame: {colored_frame_file}")
            
            # Step 2: Load the artifact frame
            artifact_frame_file = 'm15FrameA.png'
            artifact_frame_path = os.path.join(self.m15_dir, artifact_frame_file)
            artifact_frame = self._load_image_cached(artifact_frame_path)
            artifact_frame = artifact_frame.resize((self.card_width, self.card_height), Image.Resampling.LANCZOS)
            print(f"✅ Loaded artifact frame for masking: {artifact_frame_file}")
            
            # Step 3: Load the color-specific mask for artifact overlay
            # Use color-specific masks that have better masking properties (RGB tinted instead of black)
            mask_file = self.get_color_specific_mask(mana_colors)
            print(f"🎭 Selected color-specific mask: {mask_file}")
            
            # Load the color-specific mask (always from m15 directory)
            mask_path = os.path.join(self.m15_dir, mask_file)        
            
            sliver_mask = self._load_image_cached(mask_path)
            sliver_mask = sliver_mask.resize((self.card_width, self.card_height), Image.Resampling.LANCZOS)
            actual_mask_file = os.path.basename(mask_path)
            print(f"✅ Loaded color-specific mask: {actual_mask_file}")
            mask_file_used = actual_mask_file
            
            if sliver_mask is None:
                print(f"❌ No artifact mask found, using unmasked overlay")
                # Fallback: no masking, just overlay the artifact frame directly
                result_frame = colored_frame.copy()
                if result_frame.mode != 'RGBA':
                    result_frame = result_frame.convert('RGBA')
                if artifact_frame.mode != 'RGBA':
                    artifact_frame = artifact_frame.convert('RGBA')
                # Blend artifact frame on top with some opacity
                blended = Image.blend(result_frame.convert('RGBA'), artifact_frame.convert('RGBA'), alpha=0.7)
                result_frame = blended
            else:
                # Step 4: Process the mask properly
                print(f"🎭 Processing mask: {mask_file_used}")
                
                # Start with colored frame as base
                result_frame = colored_frame.copy()
                if result_frame.mode != 'RGBA':
                    result_frame = result_frame.convert('RGBA')
                
                if artifact_frame.mode != 'RGBA':
                    artifact_frame = artifact_frame.convert('RGBA')
                
                # Convert mask for processing
                if mask_file_used == 'm15MaskBorderSliver_red.png':
                    # Color-based mask - extract based on red areas
                    print("🔍 Using color-based mask processing")
                    if sliver_mask.mode not in ['RGB', 'RGBA']:
                        sliver_mask = sliver_mask.convert('RGB')
                    
                    # Create binary mask from color analysis
                    target_red, target_green, target_blue = 255, 60, 0
                    tolerance = 30
                    
                    mask_data = []
                    pixel_data = list(sliver_mask.getdata())
                    keep_count = 0
                    
                    for pixel in pixel_data:
                        r, g, b = pixel[:3]
                        
                        # Check if pixel matches target color (areas to show artifact)
                        if (abs(r - target_red) <= tolerance and 
                            abs(g - target_green) <= tolerance and 
                            abs(b - target_blue) <= tolerance):
                            mask_data.append(255)  # White = show artifact frame
                            keep_count += 1
                        else:
                            mask_data.append(0)    # Black = show colored frame
                    
                    # Create final grayscale mask
                    final_mask = Image.new('L', sliver_mask.size)
                    final_mask.putdata(mask_data)
                    
                    total_pixels = len(pixel_data)
                    print(f"🎨 Color-based masking: {keep_count}/{total_pixels} pixels for artifact overlay ({keep_count/total_pixels*100:.1f}%)")
                    
                    if keep_count == 0:
                        print("⚠️  WARNING: No mask pixels found! Mask will be completely black - no artifact overlay will show")
                        # Sample a few pixels to see what colors we're actually getting
                        sample_pixels = pixel_data[:10]
                        print(f"🔍 Sample pixel colors: {sample_pixels}")
                        
                        # Try alternative color matching - maybe the red mask uses different values
                        print("🔄 Trying broader color detection...")
                        alt_keep_count = 0
                        alt_mask_data = []
                        
                        for pixel in pixel_data:
                            r, g, b = pixel[:3]
                            # Look for pixels that are more red than other colors
                            if r > 100 and r > g and r > b:
                                alt_mask_data.append(255)  # White = show artifact frame
                                alt_keep_count += 1
                            else:
                                alt_mask_data.append(0)    # Black = show colored frame
                        
                        if alt_keep_count > 0:
                            print(f"🎯 Alternative detection found {alt_keep_count} pixels - using alternative mask")
                            mask_data = alt_mask_data
                            keep_count = alt_keep_count
                    
                else:
                    # Standard mask - convert to grayscale
                    print("⚪ Using standard mask processing")
                    if sliver_mask.mode == 'RGBA':
                        # Use alpha channel as mask
                        final_mask = sliver_mask.split()[-1]  # Get alpha channel
                        print("Using alpha channel as mask")
                    elif sliver_mask.mode in ['RGB', 'L']:
                        # Convert to grayscale
                        final_mask = sliver_mask.convert('L')
                        print("Converted to grayscale mask")
                    else:
                        final_mask = sliver_mask.convert('L')
                
                # Step 5: Apply the masked artifact frame overlay
                try:
                    print("✅ Applying masked artifact frame overlay")
                    # Use the mask to composite artifact frame over colored frame
                    result_frame.paste(artifact_frame, (0, 0), final_mask)
                    print("✅ Successfully applied masked artifact overlay")
                    
                except Exception as mask_error:
                    print(f"❌ Error applying mask: {mask_error}")
                    # Fallback: blend without mask
                    result_frame = Image.blend(result_frame, artifact_frame, alpha=0.5)
                    print("🔄 Applied fallback blend instead")
            
            # Step 6: Add gradient pinlines for multicolor if needed  
            if len(mana_colors) == 2:
                print(f"Adding gradient pinlines for two-color artifact: {mana_colors}")
                gradient_pinlines = self.blend_two_color_pinlines(mana_colors)
                if gradient_pinlines is not None:
                    gradient_pinlines = gradient_pinlines.resize((self.card_width, self.card_height), Image.Resampling.LANCZOS)
                    result_frame.paste(gradient_pinlines, (0, 0), gradient_pinlines)
                    print("✅ Applied gradient pinlines to colored artifact")
            
            # Step 7: Add legendary crown if needed
            if is_legendary:
                result_frame = self.add_legendary_crown(result_frame, mana_colors, 'artifact')
            
            print("🎨 Colored artifact frame creation complete")
            return result_frame
            
        except Exception as e:
            print(f"❌ Error creating colored artifact frame: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to standard artifact frame")
            # Fallback to standard artifact frame
            try:
                artifact_frame_file = 'm15FrameA.png'
                artifact_frame_path = os.path.join(self.m15_dir, artifact_frame_file)
                fallback_frame = self._load_image_cached(artifact_frame_path)
                return fallback_frame.resize((self.card_width, self.card_height), Image.Resampling.LANCZOS)
            except:
                return self.create_fallback_frame()
    
    def add_legendary_crown(self, frame: Image.Image, colors: List[str], card_type: str = '') -> Image.Image:
        """Add legendary crown to frame, scaled to fit card width"""
        try:
            # Special handling for two-color cards - create gradient blend
            if len(colors) == 2 and all(c in ['W', 'U', 'B', 'R', 'G'] for c in colors):
                crown = self.blend_two_color_crown(colors)
                if crown is None:
                    print("Gradient crown creation failed, falling back to multicolor")
                    crown_path = os.path.join(self.m15_dir, 'm15CrownM.png')
                    crown = self._load_image_cached(crown_path) if os.path.exists(crown_path) else None
            else:
                # Use standard single crown logic for 1, 3+, or special color combinations
                color_code = self.get_color_code(colors, card_type)
                
                # Handle artifact crown specially
                if color_code == 'Artifact':
                    # For artifacts, use the regular colorless crown since there may not be artifact-specific crowns
                    crown_file = 'm15CrownA.png'
                else:
                    crown_file = f'm15Crown{color_code}.png'
                    
                crown_path = os.path.join(self.m15_dir, crown_file)
                
                if os.path.exists(crown_path):
                    crown = self._load_image_cached(crown_path)
                else:
                    print(f"Crown file not found: {crown_file}")
                    return frame
            
            if crown is None:
                print("No crown loaded")
                return frame
            
            # Scale crown if it's wider than the card
            if crown.width > self.card_width:
                # Scale down to fit card width while maintaining aspect ratio
                scale_factor = self.card_width / crown.width
                new_width = self.card_width - 26  # Was 27, now 26 (1px wider)
                new_height = int(crown.height * scale_factor) - 10  # Was -14, now -10 (taller)
                crown = crown.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Center crown horizontally at the top of the card
            crown_x = (self.card_width - crown.width) // 2
            crown_y = 14  # Position at top (lowered 1px)
            
            # Add black rectangle behind crown to cover the border
            draw = ImageDraw.Draw(frame)
            # Make rectangle slightly larger than crown to ensure full coverage
            rect_x1 = crown_x - 5
            rect_y1 = 0  # Start from very top
            rect_x2 = crown_x + crown.width + 5
            rect_y2 = crown_y + 15
            
            draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill='black')
            
            # Paste crown over frame (assumes crown has transparency)
            frame.paste(crown, (crown_x, crown_y), crown if crown.mode == 'RGBA' else None)
            
            # Log appropriate message based on crown type
            if len(colors) == 2 and all(c in ['W', 'U', 'B', 'R', 'G'] for c in colors):
                sorted_colors = self.sort_colors_for_pipes(colors)
            else:
                color_code = self.get_color_code(colors, card_type)
                crown_type = "artifact" if color_code == 'Artifact' else color_code
                
        except Exception as e:
            print(f"Error adding legendary crown: {e}")
        
        return frame
    
    def load_pt_box(self, colors: List[str], card_type: str = '', mana_cost: str = '', subtype: str = '') -> Optional[Image.Image]:
        """Load power/toughness box for creatures and vehicles - use color-specific P/T box matching card frame"""
        try:
            # Vehicles should use colored P/T boxes like creatures, even if they're artifacts
            # Check both main type and subtype for Vehicle (e.g., "Artifact - Vehicle")
            if 'vehicle' in card_type.lower() or 'vehicle' in subtype.lower():
                print(f"Vehicle P/T: using colored frame P/T box for colors: {colors}")
                # For vehicles, always use colored P/T box based on frame colors
                color_code = self.get_color_code(colors, card_type)
                pt_file = f'm15PT{color_code}.png'
                pt_path = os.path.join(self.m15_dir, pt_file)
            elif 'artifact' in card_type.lower():
                # Special logic for non-vehicle artifacts
                # Check if artifact has colored mana symbols
                mana_colors = self.extract_colors_from_mana_cost(mana_cost) if mana_cost else []
                
                if mana_colors:
                    print(f"🎨 Colored artifact P/T: using colored frame P/T box for mana colors: {mana_colors}")
                    # For colored artifacts, use the colored frame's P/T box (not artifact P/T)
                    color_code = self.get_color_code(mana_colors, '')
                    pt_file = f'm15PT{color_code}.png'
                    pt_path = os.path.join(self.m15_dir, pt_file)
                else:
                    print(f"🎨 Pure artifact P/T: using artifact P/T box")
                    # Pure artifact - use artifact-specific P/T box
                    pt_file = 'inventionPT.png'
                    pt_path = os.path.join(self.m15_dir, 'invention', pt_file)
            else:
                # Normal non-artifact cards - use color-specific P/T box
                color_code = self.get_color_code(colors, card_type)
                pt_file = f'm15PT{color_code}.png'
                pt_path = os.path.join(self.m15_dir, pt_file)
            
            if os.path.exists(pt_path):
                pt_box = self._load_image_cached(pt_path)
                # Make P/T box 3x smaller
                original_size = pt_box.size
                new_size = (original_size[0] // 3, original_size[1] // 3)
                pt_box = pt_box.resize(new_size, Image.Resampling.LANCZOS)
                print(f"Loaded P/T box: {pt_file} (original: {original_size}, resized to: {new_size})")
                return pt_box
            else:
                print(f"P/T box not found: {pt_file}")
                # Fallback to multicolor P/T box
                fallback_file = 'm15PTM.png'
                fallback_path = os.path.join(self.m15_dir, fallback_file)
                
                if os.path.exists(fallback_path):
                    pt_box = self._load_image_cached(fallback_path)
                    # Make fallback P/T box 3x smaller too
                    original_size = pt_box.size
                    new_size = (original_size[0] // 3, original_size[1] // 3)
                    pt_box = pt_box.resize(new_size, Image.Resampling.LANCZOS)
                    print(f"Using fallback multicolor P/T box: {fallback_file} (original: {original_size}, resized to: {new_size})")
                    return pt_box
                else:
                    print(f"Fallback P/T box also not found: {fallback_file}")
                    return None
                
        except Exception as e:
            print(f"Error loading P/T box: {e}")
            return None
    
    def create_fallback_frame(self) -> Image.Image:
        """Create a simple fallback frame"""
        frame = Image.new('RGB', (self.card_width, self.card_height), color='#D3C7B8')
        draw = ImageDraw.Draw(frame)
        
        # Draw simple border
        border_color = '#8B4513'
        border_width = 10
        draw.rectangle([0, 0, self.card_width-1, self.card_height-1], 
                      outline=border_color, width=border_width)
        
        # Art box
        draw.rectangle([self.art_pos[0]-2, self.art_pos[1]-2, 
                       self.art_pos[0] + self.art_size[0]+2, 
                       self.art_pos[1] + self.art_size[1]+2], 
                      outline='black', width=2)
        
        return frame
    
    def process_artwork(self, artwork_base64: str) -> Image.Image:
        """
        Process the AI-generated artwork to fit the card's art box
        Apply custom scaling: +8px width (4px each side), -24px height (squished vertically)
        Then scale to fit art box without black bars
        """
        try:
            # Decode base64 artwork
            artwork_data = base64.b64decode(artwork_base64)
            artwork = Image.open(io.BytesIO(artwork_data))
            
            # Calculate modified dimensions for stretching/squishing effect
            # Generated image: 408x336 -> stretched/squished: 416x312
            stretch_width = artwork.width + 8   # Add 8px width (4px each side effect)
            squish_height = artwork.height - 24  # Subtract 24px height (squish effect)
            
            # Apply the stretch/squish transformation
            artwork = artwork.resize((stretch_width, squish_height), Image.Resampling.LANCZOS)
            
            # Now scale the stretched/squished image to fit the art box exactly
            # This preserves the stretch/squish effect but ensures it fills the entire art box
            final_artwork = artwork.resize(self.art_size, Image.Resampling.LANCZOS)
            
            print(f"Artwork processed: original -> stretched/squished to {stretch_width}x{squish_height} -> scaled to fit {self.art_size}")
            return final_artwork
            
        except Exception as e:
            print(f"Error processing artwork: {e}")
            # Create placeholder artwork
            placeholder = Image.new('RGB', self.art_size, color='#2C3E50')
            draw = ImageDraw.Draw(placeholder)
            text_pos = (self.art_size[0] // 2, self.art_size[1] // 2)
            draw.text(text_pos, "Artwork", fill='white', anchor='mm')
            return placeholder
    
    def parse_mana_symbols(self, mana_cost: str) -> List[str]:
        """Parse mana cost string into individual symbols"""
        # Match mana symbols in {X} format
        symbols = re.findall(r'\{([^}]+)\}', mana_cost)
        return symbols
    
    def extract_colors_from_mana_cost(self, mana_cost: str) -> List[str]:
        """Extract color symbols from mana cost string"""
        if not mana_cost:
            return []
        
        symbols = self.parse_mana_symbols(mana_cost)
        colors = []
        
        for symbol in symbols:
            # Check for basic color symbols
            if symbol.upper() in ['W', 'U', 'B', 'R', 'G']:
                if symbol.upper() not in colors:
                    colors.append(symbol.upper())
            # Check for hybrid symbols like 'W/U', '2/W', etc.
            elif '/' in symbol:
                parts = symbol.split('/')
                for part in parts:
                    if part.upper() in ['W', 'U', 'B', 'R', 'G']:
                        if part.upper() not in colors:
                            colors.append(part.upper())
        
        return colors
    
    def load_mana_symbol(self, symbol: str, size: int = 20) -> Optional[Image.Image]:
        """Load and render a mana symbol from PNG, SVG, or fallback sources"""
        symbol_lower = symbol.lower()
        
        # Try direct PNG files first (your new PNG files)
        direct_png_paths = [
            os.path.join(self.mana_symbols_dir, f'{symbol_lower}.png'),
            os.path.join(self.mana_symbols_dir, f'{symbol}.png'),  # Try uppercase too
        ]
        
        for png_path in direct_png_paths:
            if os.path.exists(png_path):
                try:
                    symbol_image = self._load_image_cached(png_path)
                    # Resize to desired size
                    symbol_image = symbol_image.resize((size, size), Image.Resampling.LANCZOS)
                    # Ensure RGBA mode for transparency
                    if symbol_image.mode != 'RGBA':
                        symbol_image = symbol_image.convert('RGBA')
                    return symbol_image
                except Exception as e:
                    print(f"Error loading direct PNG mana symbol {png_path}: {e}")
        
        # Try SVG if PNG not found and SVG support is available
        if SVG_SUPPORT:
            svg_path = os.path.join(self.mana_symbols_dir, f'{symbol_lower}.svg')
            if os.path.exists(svg_path):
                try:
                    if SVG_METHOD == 'cairosvg':
                        # Convert SVG to PNG data using CairoSVG
                        png_data = svg2png(url=svg_path, output_width=size, output_height=size)
                        
                        # Load PNG data as PIL Image
                        symbol_image = Image.open(io.BytesIO(png_data))
                        
                    elif SVG_METHOD == 'wand':
                        # Convert SVG to PNG using Wand
                        with WandImage(filename=svg_path, resolution=150) as wand_img:
                            wand_img.format = 'png'
                            wand_img.resize(size, size)
                            
                            # Convert to PIL Image
                            png_blob = wand_img.make_blob()
                            symbol_image = Image.open(io.BytesIO(png_blob))
                    
                    # Ensure RGBA mode for transparency
                    if symbol_image.mode != 'RGBA':
                        symbol_image = symbol_image.convert('RGBA')
                        
                    print(f"Loaded mana symbol from SVG using {SVG_METHOD}: {symbol}")
                    return symbol_image
                except Exception as e:
                    print(f"Error loading SVG mana symbol {symbol} with {SVG_METHOD}: {e}")
        
        # Final fallback to PNG from m21 folder
        m21_png_paths = [
            os.path.join(self.mana_symbols_dir, 'm21', f'm21{symbol_lower}.png'),
            os.path.join(self.mana_symbols_dir, 'm21', f'm21{symbol}.png'),  # Try uppercase too
        ]
        
        for png_path in m21_png_paths:
            if os.path.exists(png_path):
                try:
                    symbol_image = self._load_image_cached(png_path)
                    # Resize to desired size
                    symbol_image = symbol_image.resize((size, size), Image.Resampling.LANCZOS)
                    # Ensure RGBA mode for transparency
                    if symbol_image.mode != 'RGBA':
                        symbol_image = symbol_image.convert('RGBA')
                    return symbol_image
                except Exception as e:
                    print(f"Error loading m21 PNG mana symbol {png_path}: {e}")
        
        print(f"Mana symbol not found in any format: {symbol}")
        return None
    
    def load_site_logo(self, rarity: str = 'common') -> Optional[Image.Image]:
        """Load the MTGenesis.AI site logo for use as set symbol based on rarity"""
        # Map rarity to logo filename
        rarity_logos = {
            'common': 'site_logo_common.png',
            'uncommon': 'site_logo_uncommon.png', 
            'rare': 'site_logo_rare.png',
            'mythic': 'site_logo_mythic.png'
        }
        
        # Get the appropriate logo filename, default to common if rarity not found
        logo_filename = rarity_logos.get(rarity.lower(), 'site_logo_common.png')
        logo_path = os.path.join(self.assets_dir, logo_filename)
        
        print(f"Looking for rarity-specific logo: {logo_path}")
        if os.path.exists(logo_path):
            try:
                logo_image = self._load_image_cached(logo_path)
                # Resize to set symbol size
                logo_image = logo_image.resize((self.logo_size, self.logo_size), Image.Resampling.LANCZOS)
                # Ensure RGBA mode for transparency
                if logo_image.mode != 'RGBA':
                    logo_image = logo_image.convert('RGBA')
                return logo_image
            except Exception as e:
                print(f"Error loading rarity logo {logo_path}: {e}")
        
        # Fallback to generic site_logo.png if rarity-specific not found
        fallback_path = os.path.join(self.assets_dir, 'site_logo.png')
        print(f"Fallback to generic logo: {fallback_path}")
        if os.path.exists(fallback_path):
            try:
                logo_image = self._load_image_cached(fallback_path)
                logo_image = logo_image.resize((self.logo_size, self.logo_size), Image.Resampling.LANCZOS)
                if logo_image.mode != 'RGBA':
                    logo_image = logo_image.convert('RGBA')
                return logo_image
            except Exception as e:
                print(f"Error loading fallback logo {fallback_path}: {e}")
        
        print("No site logo found - skipping logo placement")
        return None
    
    def render_mana_cost(self, card_image: Image.Image, mana_cost: str, position: Tuple[int, int]):
        """
        Render mana cost symbols directly onto the card image
        """
        if not mana_cost:
            print("No mana cost provided")
            return
            
        # Parse mana symbols
        symbols = self.parse_mana_symbols(mana_cost)
        
        if not symbols:
            # Fallback to text if no symbols found
            draw = ImageDraw.Draw(card_image)
            draw.text(position, mana_cost, fill='black', font=self.mana_font, anchor='ra')
            return
        
        # Calculate positioning for right-aligned symbols
        symbol_size = 22  # Increased size by 2px (20 + 2 = 22)
        symbol_spacing = 3  # Slightly more spacing
        total_width = len(symbols) * symbol_size + (len(symbols) - 1) * symbol_spacing
        
        # Start from right edge and work backwards
        current_x = position[0] - total_width
        current_y = position[1] - symbol_size // 2  # Center vertically
        
        symbols_rendered = 0
        for i, symbol in enumerate(symbols):
            
            # Try to load symbol image (SVG or PNG fallback)
            symbol_image = self.load_mana_symbol(symbol, symbol_size)
            if symbol_image:
                # Create hard drop shadow (black, offset 1px left and 1px down)
                shadow_x = current_x - 1
                shadow_y = current_y + 1
                
                # Create black shadow version of the symbol
                shadow_symbol = Image.new('RGBA', symbol_image.size, (0, 0, 0, 0))
                for y in range(symbol_image.height):
                    for x in range(symbol_image.width):
                        pixel = symbol_image.getpixel((x, y))
                        if len(pixel) == 4 and pixel[3] > 0:  # Has alpha and not transparent
                            shadow_symbol.putpixel((x, y), (0, 0, 0, pixel[3]))  # Black shadow with original alpha
                
                # Paste shadow first (behind the main symbol)
                if shadow_x >= 0 and shadow_y >= 0:  # Only draw shadow if it's within bounds
                    card_image.paste(shadow_symbol, (shadow_x, shadow_y), shadow_symbol)
                
                # Paste the colorful symbol image on top
                paste_pos = (current_x, current_y)
                card_image.paste(symbol_image, paste_pos, symbol_image)
                symbols_rendered += 1
                current_x += symbol_size + symbol_spacing
                continue
            
            # Fallback: draw symbol text if image loading failed
            draw = ImageDraw.Draw(card_image)
            symbol_text = f"{{{symbol}}}"
            text_pos = (current_x + symbol_size//2, position[1])
            print(f"Drawing text '{symbol_text}' at {text_pos}")
            draw.text(text_pos, symbol_text, fill='black', font=self.mana_font, anchor='ma')
            current_x += symbol_size + symbol_spacing
        
        print(f"Mana cost rendering complete: {symbols_rendered}/{len(symbols)} symbols rendered as images")
    
    def draw_wrapped_text(self, draw: ImageDraw.Draw, text: str, position: Tuple[int, int], 
                         max_width: int, font: ImageFont.ImageFont, fill='black') -> int:
        """
        Draw text with word wrapping within specified width, respecting explicit newlines
        Returns the height used by the text
        """
        # First split by explicit newlines to preserve paragraph structure
        paragraphs = text.split('\n')
        all_lines = []
        
        for paragraph in paragraphs:
            if paragraph.strip() == '':
                # Empty line - add spacing
                all_lines.append('')
                continue
                
            # Word wrap each paragraph
            words = paragraph.split(' ')
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                bbox = draw.textbbox((0, 0), test_line, font=font)
                if bbox[2] <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        all_lines.append(' '.join(current_line))
                    current_line = [word]
            
            if current_line:
                all_lines.append(' '.join(current_line))
        
        # Draw the lines
        line_height = 18
        y_offset = 0
        for line in all_lines:
            if line == '':
                # Empty line - add half line spacing for paragraph breaks
                y_offset += line_height // 2
            else:
                draw.text((position[0], position[1] + y_offset), line, fill=fill, font=font)
                y_offset += line_height
        
        return y_offset
    
    def parse_text_segments(self, text: str) -> List[dict]:
        """
        Parse text into segments of alternating text and mana symbols
        Returns list of {'type': 'text'|'symbol', 'content': str}
        """
        import re
        segments = []
        current_pos = 0
        
        # Find all mana symbol patterns {X}
        symbol_pattern = r'\{([^}]+)\}'
        
        for match in re.finditer(symbol_pattern, text):
            # Add text segment before the symbol (if any)
            if match.start() > current_pos:
                text_content = text[current_pos:match.start()]
                if text_content:
                    segments.append({'type': 'text', 'content': text_content})
            
            # Add the mana symbol segment
            symbol_content = match.group(1)  # Content inside {}
            segments.append({'type': 'symbol', 'content': symbol_content})
            
            current_pos = match.end()
        
        # Add remaining text after last symbol (if any)
        if current_pos < len(text):
            remaining_text = text[current_pos:]
            if remaining_text:
                segments.append({'type': 'text', 'content': remaining_text})
        
        return segments
    
    def draw_text_with_mana_symbols(self, card_image: Image.Image, text: str, position: Tuple[int, int], 
                                   max_width: int, font: ImageFont.ImageFont = None, fill='black') -> int:
        """
        Draw text with embedded mana symbols, handling word wrapping and explicit newlines.
        Uses dynamic font sizing to fit text within available space.
        Returns the height used by the text
        """
        draw = ImageDraw.Draw(card_image)
        
        # Calculate optimal font size if no font provided
        if font is None:
            optimal_font_size = self.calculate_optimal_font_size(text, max_width, self.available_text_height)
            font = self.get_text_font(optimal_font_size)
            font_size = optimal_font_size
            print(f"🎨 Using dynamic font size: {optimal_font_size}px for rules text")
        else:
            # Use provided font - assume current size (17px fallback)
            font_size = 17
        
        # First handle explicit newlines by splitting text into paragraphs
        paragraphs = text.split('\n')
        base_line_height = int(font_size * 1.2)  # 1.2x font size for line height
        wrapped_line_height = int(base_line_height * 0.85)  # 0.85x for wrapped lines within same ability (tighter)
        ability_separation_height = int(base_line_height * 1.3)  # 1.3x for separating abilities (increased from 1.15x)
        symbol_size = max(12, int(font_size * 0.8))  # Proportional to font size (80% of font height, min 12px)
        current_x = position[0]
        current_y = position[1]
        line_start_x = position[0]
        
        # Track if we're at the start of a line to avoid extra spaces
        at_line_start = True
        
        # Process each paragraph separately 
        for para_index, paragraph in enumerate(paragraphs):
            if para_index > 0:
                # Explicit newline (between abilities): use larger spacing
                current_y += ability_separation_height
                current_x = line_start_x
                at_line_start = True
            
            if paragraph.strip() == '':
                # Empty paragraph - just add line spacing
                continue
            
            # Parse this paragraph into segments  
            segments = self.parse_text_segments(paragraph)
            
            for segment in segments:
                if segment['type'] == 'text':
                    # Handle text segment - may need word wrapping
                    words = segment['content'].split(' ')
                    
                    for i, word in enumerate(words):
                        if word == '':  # Skip empty words from split
                            continue
                            
                        # Add space before word only if not at line start
                        word_with_space = (' ' + word) if not at_line_start else word
                        
                        # Check if this word fits on current line
                        word_bbox = draw.textbbox((0, 0), word_with_space, font=font)
                        word_width = word_bbox[2]
                        
                        if current_x + word_width > position[0] + max_width and not at_line_start:
                            # Word doesn't fit, move to next line (within same ability)
                            current_y += wrapped_line_height
                            current_x = line_start_x
                            at_line_start = True
                            word_with_space = word  # No leading space on new line
                            word_bbox = draw.textbbox((0, 0), word_with_space, font=font)
                            word_width = word_bbox[2]
                        
                        # Draw the word
                        draw.text((current_x, current_y), word_with_space, fill=fill, font=font)
                        current_x += word_width
                        at_line_start = False
                        
                elif segment['type'] == 'symbol':
                    # Handle mana symbol segment - use exact same logic as mana cost rendering
                    print(f"[ACTIVE ABILITY] Processing rules text symbol: '{segment['content']}'")
                    symbol_image = self.load_mana_symbol(segment['content'], symbol_size)
                    
                    if symbol_image:
                        # Check if symbol fits on current line
                        if current_x + symbol_size > position[0] + max_width and not at_line_start:
                            # Symbol doesn't fit, move to next line (within same ability)
                            current_y += wrapped_line_height
                            current_x = line_start_x
                            at_line_start = True
                        
                        # Position symbol to align with text baseline - consistent positioning
                        if segment['content'].upper() == 'T':
                            # Special positioning for tap symbol: adjusted left 1px more
                            symbol_x = current_x + 6  # Move right 6px for {T} symbol (was 7px, now 1px less)
                            symbol_y = current_y - symbol_size // 2 + 10  # Move down for better alignment with text
                        else:
                            # Positioning for other mana symbols - match tap symbol alignment
                            symbol_x = current_x + 6  # Move right 6px (same as tap symbol, 1px left from before)
                            symbol_y = current_y - symbol_size // 2 + 10  # Same vertical alignment as tap symbol
                        
                        # Create hard drop shadow (black, offset 1px left and 1px down from symbol position)
                        shadow_x = symbol_x - 1
                        shadow_y = symbol_y + 1
                        
                        # Create black shadow version of the symbol
                        shadow_symbol = Image.new('RGBA', symbol_image.size, (0, 0, 0, 0))
                        for y in range(symbol_image.height):
                            for x in range(symbol_image.width):
                                pixel = symbol_image.getpixel((x, y))
                                if len(pixel) == 4 and pixel[3] > 0:  # Has alpha and not transparent
                                    shadow_symbol.putpixel((x, y), (0, 0, 0, pixel[3]))  # Black shadow with original alpha
                        
                        # Paste shadow first (behind the main symbol)
                        if shadow_x >= 0 and shadow_y >= 0:  # Only draw shadow if it's within bounds
                            card_image.paste(shadow_symbol, (shadow_x, shadow_y), shadow_symbol)
                        
                        # Paste colorful symbol on top
                        paste_pos = (symbol_x, symbol_y)
                        card_image.paste(symbol_image, paste_pos, symbol_image)
                        current_x += symbol_size + 3  # Use same spacing as mana cost (3px)
                        at_line_start = False
                    else:
                        # Fallback: render as text if symbol not found (same as mana cost)
                        print(f"Symbol '{segment['content']}' not found, using text fallback")
                        fallback_text = f"{{{segment['content']}}}"
                        fallback_bbox = draw.textbbox((0, 0), fallback_text, font=font)
                        fallback_width = fallback_bbox[2]
                        
                        if current_x + fallback_width > position[0] + max_width and not at_line_start:
                            current_y += wrapped_line_height
                            current_x = line_start_x
                            at_line_start = True
                        
                        draw.text((current_x, current_y), fallback_text, fill=fill, font=font)
                        current_x += fallback_width
                        at_line_start = False
        
        # Return total height used
        return current_y - position[1] + base_line_height
    
    def generate_card_image(self, card_data: Dict, artwork_base64: Optional[str] = None) -> str:
        """
        Generate a complete Magic card image using the simplified approach
        
        Args:
            card_data: Dictionary containing card information
            artwork_base64: Base64 encoded artwork image
            
        Returns:
            Base64 encoded complete card image
        """
        try:
            import time
            render_start = time.time()
            print("🎨 Starting card image rendering...")
            
            # Validate card_data type
            
            # Ensure card_data is a dictionary
            if isinstance(card_data, str):
                print("ERROR: card_data is a string, not a dictionary!")
                return None
            elif card_data is None:
                print("ERROR: card_data is None!")
                return None
                
            # Extract card data
            name = card_data.get('name', 'Card Name')
            # Try both possible mana cost field names
            mana_cost = card_data.get('manaCost', card_data.get('mana_cost', ''))
            supertype = card_data.get('supertype', '')
            card_type = card_data.get('type', 'Creature')
            subtype = card_data.get('subtype', '')
            colors = card_data.get('colors', [])
            # Try both possible description field names
            text = card_data.get('description', card_data.get('text', ''))
            power = card_data.get('power')
            toughness = card_data.get('toughness')
            rarity = card_data.get('rarity', 'common')
            
            # Card data loaded successfully
            
            # Log artifact detection once at start of rendering
            if 'artifact' in card_type.lower():
                print(f"🔍 ARTIFACT DETECTED from typeline: '{card_type}'")
            
            # Check if legendary
            is_legendary = 'Legendary' in supertype if supertype else False
            is_creature = 'Creature' in card_type
            # Vehicles also need P/T boxes since they become creatures when crewed
            # Check both main type and subtype for Vehicle (e.g., "Artifact - Vehicle")
            is_vehicle = 'Vehicle' in card_type or ('Vehicle' in subtype if subtype else False)
            needs_pt_box = is_creature or is_vehicle
            
            print(f"Is legendary: {is_legendary}, Is creature: {is_creature}, Is vehicle: {is_vehicle}, Needs P/T: {needs_pt_box}")
            
            # Step 1: Load base frame
            frame_start = time.time()
            card_image = self.load_base_frame(colors, is_legendary, card_type, mana_cost)
            frame_time = time.time() - frame_start
            print(f"   🖼️ Frame loading: {frame_time:.3f}s")
            
            # Step 2: Process and overlay artwork
            artwork_start = time.time()
            if artwork_base64:
                artwork = self.process_artwork(artwork_base64)
                card_image.paste(artwork, self.art_pos)
                artwork_time = time.time() - artwork_start
                print(f"   🎨 Artwork processing: {artwork_time:.3f}s")
            else:
                artwork_time = time.time() - artwork_start
                print(f"   🎨 Artwork processing (skip): {artwork_time:.3f}s")
            
            # Step 3: Load and position P/T box for creatures and vehicles
            pt_box_start = time.time()
            if needs_pt_box and (power is not None and toughness is not None):
                pt_box = self.load_pt_box(colors, card_type, mana_cost, subtype)
                if pt_box:
                    # Position P/T box in bottom right corner (moved 11px left and 14px up from original)
                    pt_box_pos = (self.card_width - pt_box.width - 26,  # 15 + 8 + 2 + 1 = 26 (11px more left)
                                 self.card_height - pt_box.height - 29)  # 15 + 8 + 8 - 2 = 29 (14px more up)
                    card_image.paste(pt_box, pt_box_pos, pt_box if pt_box.mode == 'RGBA' else None)
                    print(f"Added P/T box at position {pt_box_pos}")
                else:
                    print("No P/T box found, will draw text only")
            pt_box_time = time.time() - pt_box_start
            print(f"   📦 P/T box loading: {pt_box_time:.3f}s")
            
            # Step 4: Initialize text drawing
            text_setup_start = time.time()
            draw = ImageDraw.Draw(card_image)
            text_setup_time = time.time() - text_setup_start
            print(f"   📝 Text setup: {text_setup_time:.3f}s")
            
            # Step 5: Draw card name
            name_start = time.time()
            draw.text(self.name_pos, name, fill='black', font=self.title_font)
            name_time = time.time() - name_start
            print(f"   📛 Card name: {name_time:.3f}s")
            
            # Step 6: Render mana cost symbols
            mana_cost_start = time.time()
            if mana_cost:
                self.render_mana_cost(card_image, mana_cost, self.mana_cost_pos)
            mana_cost_time = time.time() - mana_cost_start
            print(f"   💎 Mana cost symbols: {mana_cost_time:.3f}s")
            
            # Step 7: Draw type line with proper Magic formatting (em dash between type and subtype)
            type_line_start = time.time()
            type_parts = []
            if supertype:
                type_parts.append(supertype)
            if card_type:
                type_parts.append(card_type)
            
            # Join supertype and type with spaces
            main_type = ' '.join(type_parts)
            
            # Add subtype with em dash if it exists
            if subtype:
                full_type = f"{main_type} — {subtype}"
            else:
                full_type = main_type
                
            draw.text(self.type_pos, full_type, fill='black', font=self.type_font)
            print(f"Type line: '{full_type}'")
            type_line_time = time.time() - type_line_start
            print(f"   📋 Type line: {type_line_time:.3f}s")
            
            # Step 8: Add site logo as set symbol based on rarity
            logo_start = time.time()
            site_logo = self.load_site_logo(rarity)
            if site_logo:
                # Paste without mask for solid, opaque appearance
                # Only use mask if the logo has transparent areas that need to be preserved
                if site_logo.mode == 'RGBA':
                    # Check if the logo actually has transparency
                    has_transparency = any(pixel[3] < 255 for pixel in site_logo.getdata() if len(pixel) == 4)
                    if has_transparency:
                        # Logo has transparency, use it as mask
                        card_image.paste(site_logo, self.logo_pos, site_logo)
                        print(f"[SET SYMBOL] Added {rarity} rarity logo with transparency at position {self.logo_pos}")
                    else:
                        # Logo is fully opaque, paste without mask for solid appearance
                        card_image.paste(site_logo, self.logo_pos)
                        print(f"[SET SYMBOL] Added {rarity} rarity logo (solid) at position {self.logo_pos}")
                else:
                    # Logo is not RGBA, paste directly
                    card_image.paste(site_logo, self.logo_pos)
                    print(f"[SET SYMBOL] Added {rarity} rarity logo (direct) at position {self.logo_pos}")
            logo_time = time.time() - logo_start
            print(f"   🏷️ Set symbol: {logo_time:.3f}s")
            
            # Step 9: Draw card text with mana symbols
            rules_text_start = time.time()
            if text:
                print(f"Drawing card text with mana symbols at position {self.text_pos}: '{text}'")
                self.draw_text_with_mana_symbols(card_image, text, self.text_pos, 
                                                self.text_width)
            else:
                print("No card text to draw")
            rules_text_time = time.time() - rules_text_start
            print(f"   📜 Rules text: {rules_text_time:.3f}s")
            
            # Step 10: Draw power/toughness text (position it in the P/T box if it exists)
            pt_text_start = time.time()
            if needs_pt_box and (power is not None and toughness is not None):
                pt_text = f"{power}/{toughness}"
                
                # If we have a P/T box, center text in it
                pt_box = self.load_pt_box(colors, card_type, mana_cost, subtype)
                if pt_box:
                    pt_box_pos = (self.card_width - pt_box.width - 26,  # 15 + 8 + 2 + 1 = 26 (11px more left)
                                 self.card_height - pt_box.height - 29)  # 15 + 8 + 8 - 2 = 29 (14px more up)
                    pt_text_pos = (pt_box_pos[0] + pt_box.width // 2 + 4,
                                  pt_box_pos[1] + pt_box.height // 2)
                    draw.text(pt_text_pos, pt_text, fill='black', font=self.pt_font, anchor='mm')
                else:
                    # Fallback to original position
                    draw.text(self.pt_pos, pt_text, fill='black', font=self.pt_font, anchor='rb')
            pt_text_time = time.time() - pt_text_start
            print(f"   ⚔️ P/T text: {pt_text_time:.3f}s")
            
            # Step 11: Add rounded corners to the final card image
            corners_start = time.time()
            card_image_rounded = self.add_rounded_corners(card_image, radius=18)
            corners_time = time.time() - corners_start
            print(f"   🔲 Rounded corners: {corners_time:.3f}s")
            
            # Step 12: Convert to base64
            conversion_start = time.time()
            buffer = io.BytesIO()
            card_image_rounded.save(buffer, format='PNG', quality=95)
            img_data = buffer.getvalue()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            conversion_time = time.time() - conversion_start
            print(f"   💾 Base64 conversion: {conversion_time:.3f}s")
            
            # Final timing summary
            total_render_time = time.time() - render_start
            print(f"🎨 Total card rendering time: {total_render_time:.3f}s")
            print(f"Successfully generated card image for {name}")
            return img_base64
            
        except Exception as e:
            print(f"Error generating card image: {e}")
            import traceback
            traceback.print_exc()
            return None

# Global renderer instance
card_renderer = MagicCardRenderer()