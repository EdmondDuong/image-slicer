#!/usr/bin/env python
"""
Image Slicer Command Line Interface

A unified command-line interface for image slicing operations with support for both
individual image processing and batch operations. Provides full access to all features
of the image slicing library through an easy-to-use command line tool.

Features:
    - Single image processing with customizable settings
    - Batch processing of multiple images
    - Multiple slicing modes (pixel_overlap, image_count)
    - Custom slice dimensions and overlap control
    - Multi-threading support
    - Progress tracking and detailed statistics
    - Verbose output option for debugging

Usage Examples:
    1. Single Image - Basic:
       python image_slicer_cli.py single input.jpg -o output_dir -s 1000,100

    2. Single Image - With Overlap:
       python image_slicer_cli.py single input.jpg -s 400,400 --overlap 50,50

    3. Single Image - Image Count Mode:
       python image_slicer_cli.py single input.jpg --mode image_count --horizontal-images 5

    4. Batch Processing - Basic:
       python image_slicer_cli.py batch "*.jpg" -o batch_output

    5. Batch Processing - Advanced:
       python image_slicer_cli.py batch "dataset/*.png" -s 500,500 --overlap 100,100 --threads 8

Exit Codes:
    0: Success
    1: Invalid arguments or configuration
    2: Processing error

Author: Edmond Duong
Version: 1.0.0
"""

import os
import sys
import glob
import argparse
import time
from image_slicer import slice_image
from batch_process import batch_process

def format_time(seconds):
    """Format time in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {int(seconds)} sec"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)} hr {int(minutes)} min"

def main():
    # Create main parser
    parser = argparse.ArgumentParser(
        description="Image Slicer - Slice images into smaller segments with optional overlap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - slice a single image with default settings
  python image_slicer_cli.py single example.jpg
  
  # Slice an image with custom slice size and overlap
  python image_slicer_cli.py single example.jpg -s 300,200 --overlap 50,50
  
  # Use image_count mode to create exactly 5 horizontal slices
  python image_slicer_cli.py single example.jpg --mode image_count --horizontal-images 5
  
  # Batch process all JPG files in a directory
  python image_slicer_cli.py batch "images/*.jpg" -o output_folder
  
  # Batch process with detailed output
  python image_slicer_cli.py batch "images/*.png" --verbose
  """
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Single image slicing command
    single_parser = subparsers.add_parser("single", help="Slice a single image")
    single_parser.add_argument("image_path", help="Path to the input image file")
    single_parser.add_argument("-o", "--output", help="Output directory (optional)")
    single_parser.add_argument("-s", "--size", type=str, default="400,400",
                          help="Slice size as width,height (default: 400,400)")
    single_parser.add_argument("--overlap", type=str, default="0,0",
                          help="Overlap as horizontal,vertical pixels (default: 0,0)")
    single_parser.add_argument("--mode", choices=["pixel_overlap", "image_count"], default="pixel_overlap",
                          help="Slicing mode (default: pixel_overlap)")
    single_parser.add_argument("--horizontal-images", type=int,
                          help="Number of horizontal images in image_count mode")
    single_parser.add_argument("--threads", type=int,
                          help="Maximum number of threads (default: CPU count)")
    single_parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")
    
    # Batch processing command
    batch_parser = subparsers.add_parser("batch", help="Batch process multiple images")
    batch_parser.add_argument("input_pattern", help="Glob pattern for input files (e.g., '*.jpg' or 'folder/*.png')")
    batch_parser.add_argument("-o", "--output", help="Output directory (optional)")
    batch_parser.add_argument("-s", "--size", type=str, default="400,400",
                         help="Slice size as width,height (default: 400,400)")
    batch_parser.add_argument("--overlap", type=str, default="0,0",
                         help="Overlap as horizontal,vertical pixels (default: 0,0)")
    batch_parser.add_argument("--mode", choices=["pixel_overlap", "image_count"], default="pixel_overlap",
                         help="Slicing mode (default: pixel_overlap)")
    batch_parser.add_argument("--horizontal-images", type=int,
                         help="Number of horizontal images in image_count mode")
    batch_parser.add_argument("--threads", type=int,
                         help="Maximum number of threads (default: CPU count)")
    batch_parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Disable debug output unless verbose mode is on
    if not hasattr(args, "verbose") or not args.verbose:
        # Replace print with a no-op function for debug statements
        import builtins
        real_print = builtins.print
        
        def filtered_print(*args, **kwargs):
            # Only allow prints that don't start with DEBUG:
            if args and isinstance(args[0], str) and args[0].startswith("DEBUG:"):
                return
            return real_print(*args, **kwargs)
        
        builtins.print = filtered_print
    
    if args.command is None:
        parser.print_help()
        return
    
    # Parse slice size for both commands
    try:
        slice_size = tuple(map(int, args.size.split(",")))
        if len(slice_size) != 2:
            raise ValueError("Slice size must have exactly two components")
    except ValueError as e:
        print(f"Invalid slice size format: {e}")
        return
    
    # Parse overlap for both commands
    try:
        overlap = tuple(map(int, args.overlap.split(",")))
        if len(overlap) != 2:
            raise ValueError("Overlap must have exactly two components")
    except ValueError as e:
        print(f"Invalid overlap format: {e}")
        return
    
    start_time = time.time()
    
    # Execute the appropriate command
    if args.command == "single":
        if not os.path.isfile(args.image_path):
            print(f"Error: File not found: {args.image_path}")
            return
            
        print(f"Slicing image: {args.image_path}")
        slice_image(
            args.image_path,
            args.output,
            slice_size=slice_size,
            overlap=overlap,
            mode=args.mode,
            horizontal_images=args.horizontal_images,
            max_threads=args.threads
        )
    
    elif args.command == "batch":
        # Before running batch, let's see how many images we're processing
        image_files = glob.glob(args.input_pattern)
        if not image_files:
            print(f"No files found matching pattern: {args.input_pattern}")
            return
            
        print(f"Found {len(image_files)} images to process")
        
        # Run batch processing
        batch_process(
            args.input_pattern,
            args.output,
            slice_size=slice_size,
            overlap=overlap,
            mode=args.mode,
            horizontal_images=args.horizontal_images,
            max_threads=args.threads
        )
    
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {format_time(elapsed_time)}")

if __name__ == "__main__":
    main()
