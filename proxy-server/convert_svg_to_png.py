#!/usr/bin/env python3
"""
SVG to PNG Converter for Mana Symbols
Converts all SVG files in the project to PNG format while preserving directory structure.
"""

import os
import glob
from pathlib import Path
import argparse
from typing import List, Tuple

try:
    import cairosvg
except ImportError:
    print("Error: cairosvg is not installed. Install it with: pip install cairosvg")
    exit(1)


def find_svg_files(root_dir: str) -> List[str]:
    """Find all SVG files recursively in the given directory."""
    svg_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.svg'):
                svg_files.append(os.path.join(root, file))
    return svg_files


def convert_svg_to_png(svg_path: str, output_size: int = 128, quality: int = 95) -> Tuple[bool, str]:
    """
    Convert a single SVG file to PNG.
    
    Args:
        svg_path: Path to the SVG file
        output_size: Size of the output PNG (width and height in pixels)
        quality: PNG quality (not directly applicable to PNG, but affects compression)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Generate output path (same location, .png extension)
        png_path = os.path.splitext(svg_path)[0] + '.png'
        
        # Convert SVG to PNG
        cairosvg.svg2png(
            url=svg_path,
            write_to=png_path,
            output_width=output_size,
            output_height=output_size
        )
        
        return True, f"Successfully converted: {os.path.basename(svg_path)} -> {os.path.basename(png_path)}"
        
    except Exception as e:
        return False, f"Failed to convert {os.path.basename(svg_path)}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Convert SVG mana symbols to PNG format')
    parser.add_argument('--size', type=int, default=128, 
                      help='Output PNG size in pixels (default: 128)')
    parser.add_argument('--root-dir', type=str, default='.',
                      help='Root directory to search for SVG files (default: current directory)')
    parser.add_argument('--dry-run', action='store_true',
                      help='Show what would be converted without actually converting')
    
    args = parser.parse_args()
    
    # Find all SVG files
    print(f"Searching for SVG files in: {os.path.abspath(args.root_dir)}")
    svg_files = find_svg_files(args.root_dir)
    
    if not svg_files:
        print("No SVG files found!")
        return
    
    print(f"Found {len(svg_files)} SVG files")
    
    if args.dry_run:
        print("\nDRY RUN - Files that would be converted:")
        for svg_file in svg_files:
            png_file = os.path.splitext(svg_file)[0] + '.png'
            print(f"  {svg_file} -> {png_file}")
        return
    
    # Convert files
    print(f"\nConverting SVG files to PNG (size: {args.size}x{args.size} pixels)...")
    success_count = 0
    error_count = 0
    
    for svg_file in svg_files:
        success, message = convert_svg_to_png(svg_file, args.size)
        if success:
            success_count += 1
            print(f"✓ {message}")
        else:
            error_count += 1
            print(f"✗ {message}")
    
    # Summary
    print(f"\n--- Conversion Summary ---")
    print(f"Successfully converted: {success_count} files")
    print(f"Errors: {error_count} files")
    print(f"Total processed: {len(svg_files)} files")
    
    if success_count > 0:
        print(f"\nPNG files have been created alongside their corresponding SVG files.")


if __name__ == "__main__":
    main()
