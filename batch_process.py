"""
Batch Processing Module for Image Slicer

A high-performance batch processing tool for slicing multiple images using consistent settings.
Designed for processing large datasets with memory optimization and detailed progress tracking.

Features:
- Efficient batch processing of multiple images
- Memory usage monitoring and optimization
- Detailed statistics and progress tracking
- Robust error handling and logging
- Multi-threaded processing support
- Configurable slice settings for all images

Example Usage:
    # Basic usage
    result = batch_process("*.jpg", "output_dir", slice_size=(1000, 100))
    
    # Advanced usage with custom settings
    result = batch_process(
        "dataset/*.png",
        "output_dir",
        slice_size=(400, 400),
        overlap=(50, 50),
        mode="image_count",
        horizontal_images=5,
        max_threads=8
    )

Return Value Structure:
    {
        'summary': {
            'total_images_processed': int,
            'total_slices_created': int,
            'average_processing_time': float,
            'total_processing_time': float,
            'success_rate': float
        },
        'details': [
            {
                'input_file': str,
                'total_slices': int,
                'processing_time': float,
                'original_size': tuple,
                'slice_size': tuple,
                'mode': str
            },
            ...
        ]
    }
"""

import os
import sys
import glob
import argparse
import logging
import psutil
import time
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
from image_slicer import slice_image, ImageSlicerError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchProcessingError(Exception):
    """Raised when there's an error during batch processing."""
    pass

def check_memory_usage(image_size: Tuple[int, int]) -> bool:
    """
    Check if processing the next image might cause memory issues.
    Returns True if it's safe to proceed, False if memory is low.
    """
    mem = psutil.virtual_memory()
    estimated_memory = (image_size[0] * image_size[1] * 3) / (1024 * 1024)  # Rough estimate in MB
    return mem.available > (estimated_memory * 3)  # Ensure we have 3x the estimated memory needed

def batch_process(input_pattern: str, output_dir: Optional[str] = None, slice_size: Tuple[int, int] = (400, 400),
                 overlap=(0, 0), mode="pixel_overlap", horizontal_images=None,
                 max_threads=None):
    """
    Process multiple images matching the input pattern.
    
    Args:
        input_pattern (str): Glob pattern to match input files
        output_dir (str): Directory for outputs (if None, subdirectories are created for each image)
        slice_size, overlap, mode, horizontal_images, max_threads: passed to slice_image
        
    Returns:
        dict: Statistics about the batch processing operation
        
    Raises:
        BatchProcessingError: If there are issues with input/output paths or processing
    """
    # Validate output directory if specified
    if output_dir is not None:
        try:
            # Create all parent directories if they don't exist
            parent_dir = os.path.dirname(os.path.abspath(output_dir))
            if not os.path.exists(parent_dir):
                try:
                    os.makedirs(parent_dir)
                except (OSError, PermissionError) as e:
                    raise BatchProcessingError(f"Cannot create output directory {output_dir}: {e}")
            elif not os.path.isdir(parent_dir):
                raise BatchProcessingError(f"Invalid output directory path: {output_dir}")
            elif not os.access(parent_dir, os.W_OK):
                raise BatchProcessingError(f"No write permission for output directory: {output_dir}")
        except Exception as e:
            raise BatchProcessingError(f"Invalid output directory path: {output_dir} - {str(e)}")
    # Validate output directory path format first
    if output_dir is not None:
        try:
            # Check if path contains invalid characters (excluding path separators)
            if any(c in '*?"<>|' for c in output_dir):
                raise BatchProcessingError(f"Invalid characters in output directory path: {output_dir}")
            # Try to get the absolute path to validate format
            output_dir = os.path.abspath(output_dir)
        except Exception as e:
            raise BatchProcessingError(f"Invalid output directory path: {output_dir} - {str(e)}")

    # Find all files matching the pattern
    image_files = glob.glob(input_pattern)
    if not image_files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        raise BatchProcessingError(f"No files found matching pattern: {input_pattern}")

    # Initialize statistics
    processed_count = 0
    total_slices = 0
    total_processing_time = 0.0
    stats_list = []
    start_time = time.time()

    logger.info(f"Found {len(image_files)} images to process")
    
    for image_file in image_files:
        # Check available memory before processing
        with Image.open(image_file) as img:
            if not check_memory_usage(img.size):
                logger.warning(f"Low memory detected. Waiting before processing {image_file}")
                while not check_memory_usage(img.size):
                    time.sleep(1)
        try:
            # Create individual output directory if needed
            if output_dir is None:
                base_name = os.path.splitext(os.path.basename(image_file))[0]
                img_output_dir = f"{base_name}_slices"
            else:
                # Check if output directory path is valid
                output_parent = os.path.dirname(output_dir)
                if output_parent and not os.path.exists(output_parent):
                    raise BatchProcessingError(f"Invalid output directory path: {output_dir}")
                
                # Use subdirectory with image name inside the given output directory
                base_name = os.path.splitext(os.path.basename(image_file))[0]
                img_output_dir = os.path.join(output_dir, base_name)
            
            logger.info(f"\nProcessing: {image_file}")
            slice_stats = slice_image(
                image_file,
                img_output_dir,
                slice_size=slice_size,
                overlap=overlap,
                mode=mode,
                horizontal_images=horizontal_images,
                max_threads=max_threads
            )
            processed_count += 1
            total_slices += slice_stats['total_slices']
            total_processing_time += slice_stats['processing_time']
            slice_stats['input_file'] = image_file
            stats_list.append(slice_stats)
            logger.info(f"Successfully processed {image_file}")
        except ImageSlicerError as e:
            logger.error(f"Error processing {image_file}: {e}")
            raise BatchProcessingError(f"Failed to process {image_file}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error processing {image_file}: {e}")
            raise BatchProcessingError(f"Failed to process {image_file}: {str(e)}")
    
    # Calculate final statistics
    total_time = time.time() - start_time
    avg_processing_time = total_processing_time / len(image_files) if image_files else 0
    summary_stats = {
        'total_images_processed': processed_count,
        'total_images_attempted': len(image_files),
        'total_slices_created': total_slices,
        'average_processing_time': avg_processing_time,
        'total_processing_time': total_processing_time,
        'success_rate': processed_count / len(image_files) if image_files else 0
    }
    
    logger.info("\nBatch processing complete!")
    logger.info(f"Processed {processed_count} out of {len(image_files)} images")
    logger.info(f"Total processing time: {total_processing_time:.2f}s")
    logger.info(f"Average time per image: {avg_processing_time:.2f}s")
    logger.info(f"Created a total of {total_slices} slices")
    
    return {
        'summary': summary_stats,
        'details': stats_list
    }

if __name__ == "__main__":
    # Create command line argument parser
    parser = argparse.ArgumentParser(description="Batch process images with the image slicer tool")
    
    parser.add_argument("input_pattern", help="Glob pattern for input files (e.g., '*.jpg' or 'folder/*.png')")
    parser.add_argument("-o", "--output", help="Output directory (optional)")
    parser.add_argument("-s", "--size", type=str, default="400,400",
                        help="Slice size as width,height (default: 400,400)")
    parser.add_argument("--overlap", type=str, default="0,0",
                        help="Overlap as horizontal,vertical pixels (default: 0,0)")
    parser.add_argument("--mode", choices=["pixel_overlap", "image_count"], default="pixel_overlap",
                        help="Slicing mode (default: pixel_overlap)")
    parser.add_argument("--horizontal-images", type=int,
                        help="Number of horizontal images in image_count mode")
    parser.add_argument("--threads", type=int,
                        help="Maximum number of threads (default: CPU count)")
    
    args = parser.parse_args()
    
    # Parse slice size
    try:
        slice_size = tuple(map(int, args.size.split(",")))
        if len(slice_size) != 2:
            raise ValueError("Slice size must have exactly two components")
    except ValueError as e:
        print(f"Invalid slice size format: {e}")
        sys.exit(1)
    
    # Parse overlap
    try:
        overlap = tuple(map(int, args.overlap.split(",")))
        if len(overlap) != 2:
            raise ValueError("Overlap must have exactly two components")
    except ValueError as e:
        print(f"Invalid overlap format: {e}")
        sys.exit(1)
    
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
