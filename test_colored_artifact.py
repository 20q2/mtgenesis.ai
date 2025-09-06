#!/usr/bin/env python3
"""
Test script for colored artifact card rendering
"""

import sys
import os
sys.path.append('proxy-server')

from card_renderer import MagicCardRenderer

def test_colored_artifact():
    """Test colored artifact rendering with the new masked approach"""
    
    renderer = MagicCardRenderer()
    
    # Test data for a colored artifact creature
    test_card_data = {
        'name': 'Colored Artifact Test',
        'manaCost': '{2}{R}{W}',  # Red/White mana cost
        'type': 'Artifact Creature',
        'subtype': 'Golem',
        'colors': [],  # Artifacts have no color identity, but mana cost has colors
        'description': 'Vigilance, haste\nWhen Colored Artifact Test enters the battlefield, deal 2 damage to any target.',
        'power': '3',
        'toughness': '2',
        'rarity': 'rare'
    }
    
    print("Testing colored artifact rendering...")
    print(f"Card: {test_card_data['name']}")
    print(f"Mana Cost: {test_card_data['manaCost']}")
    print(f"Type: {test_card_data['type']}")
    print(f"P/T: {test_card_data['power']}/{test_card_data['toughness']}")
    print()
    
    try:
        # Generate the card image
        result = renderer.generate_card_image(test_card_data)
        
        if result:
            print("SUCCESS: Colored artifact rendering test PASSED!")
            print("   Card image generated successfully")
            
            # Save the result to file for inspection
            import base64
            with open('test_colored_artifact_output.png', 'wb') as f:
                f.write(base64.b64decode(result))
            print("   Saved test result to: test_colored_artifact_output.png")
            
        else:
            print("ERROR: Colored artifact rendering test FAILED!")
            print("   No image data returned")
            
    except Exception as e:
        print(f"ERROR: Colored artifact rendering test ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_colored_artifact()