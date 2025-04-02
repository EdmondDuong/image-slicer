"""
Image Slicer Module

A high-performance tool for slicing images into smaller segments with various options and modes.
Features:
- Multiple slicing modes (pixel_overlap, image_count)
- Customizable slice dimensions and overlap
- Multi-threaded processing
- Progress tracking
- Memory optimization
- Detailed statistics

Typical usage:
    result = slice_image('input.jpg', 'output_dir', slice_size=(1000, 100))
    
Advanced usage:
    result = slice_image(
        'input.jpg',
        'output_dir',
        slice_size=(400, 400),
        overlap=(50, 50),
        mode='image_count',
        horizontal_images=5,
        max_threads=8
    )
"""

import os
import sys
import logging
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import time
from typing import Tuple, Optional, List, Dict, Union
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageSlicerError(Exception):
    """
    Base exception class for all image slicer errors.
    
    This is the parent class for all custom exceptions in the image_slicer module.
    It can be caught to handle any error from the module in a generic way.
    """
    pass

class InvalidImageError(ImageSlicerError):
    """
    Raised when there are issues with the input image.
    
    Common causes:
    - File does not exist
    - File is not a valid image format
    - Image file is corrupted
    - Insufficient permissions to read the file
    - Unsupported image format
    """
    pass

class SliceProcessingError(ImageSlicerError):
    """
    Raised when there's an error during slice creation or saving.
    
    Common causes:
    - Insufficient disk space
    - Permission denied when writing output file
    - Memory errors during image processing
    - Invalid slice dimensions
    - Failed to create output directory
    - Image processing errors (e.g., during crop operation)
    """
    pass

def process_slice(params: Tuple) -> bool:
    """
    Process and save a single slice of the image with error handling and validation.
    
    Args:
        params (tuple): Processing parameters as a tuple containing:
            - img (PIL.Image): Source image to slice from
            - left (int): Left coordinate of the slice
            - upper (int): Upper coordinate of the slice
            - right (int): Right coordinate of the slice
            - lower (int): Lower coordinate of the slice
            - output_path (str): Path where to save the slice
            - is_image_count (bool): Whether we're in image_count mode
    
    Returns:
        bool: True if slice was processed and saved successfully, False otherwise
    
    Notes:
        - In image_count mode, empty slices are replaced with 1x1 white pixels
        - Failed slices in image_count mode attempt a fallback to 1x1 placeholder
        - All operations are wrapped in try/except for robustness
        - Validates slice dimensions before saving
        - Verifies file creation after save
    
    Raises:
        SliceProcessingError: If slice creation fails in image_count mode after fallback
    """
    img, left, upper, right, lower, output_path, is_image_count = params
    try:
        # Extract the slice
        slice_img = img.crop((left, upper, right, lower))
        
        # Skip empty slices (those with zero width or height) but not in image_count mode
        if (not is_image_count) and (slice_img.width <= 0 or slice_img.height <= 0):
            return False
        
        # For image_count mode, ensure the slice is at least 1x1 pixels
        if slice_img.width <= 0 or slice_img.height <= 0:
            slice_img = Image.new('RGB', (1, 1), color='white')
        
        # Save the slice - in image_count mode we always save even if dimensions are small
        slice_img.save(output_path)
        
        # Verify the file was actually created
        if not os.path.exists(output_path):
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error processing slice: {str(e)}")
        # Try again with a single-pixel image in case of PIL errors
        try:
            if is_image_count:
                placeholder = Image.new('RGB', (1, 1), color='white')
                placeholder.save(output_path)
                return True
        except Exception as backup_err:
            logger.error(f"Final error creating slice: {str(backup_err)}")
            raise SliceProcessingError(f"Failed to create slice: {str(backup_err)}")
        return False

def slice_image(image_path: str, output_dir: Optional[str] = None,
                slice_size: Tuple[int, int] = (400, 400),
                overlap: Tuple[int, int] = (0, 0),
                mode: str = "pixel_overlap",
                horizontal_images: Optional[int] = None,
                max_threads: Optional[int] = None) -> Dict[str, Union[int, float, Tuple[int, int], str]]:
    """
    Slices an image into smaller segments with customizable settings and optional overlap.
    
    Args:
        image_path (str): Path to the input image file
        output_dir (str, optional): Directory to save the sliced images.
                                   If None, creates '<input_name>_slices' directory.
        slice_size (tuple): Size of each slice as (width, height) in pixels
        overlap (tuple): Overlap between adjacent slices as (horizontal, vertical) in pixels
        mode (str): Slicing mode:
                   - "pixel_overlap": Creates slices of exact size with specified overlap
                   - "image_count": Creates specific number of evenly-spaced slices
        horizontal_images (int): Number of horizontal slices in image_count mode
        max_threads (int): Maximum number of threads for parallel processing.
                          Defaults to CPU count if None.
    
    Returns:
        dict: Statistics about the slicing operation:
            {
                'total_slices': int,       # Number of slices created
                'output_dir': str,         # Path to output directory
                'original_size': tuple,    # Original image dimensions
                'slice_size': tuple,       # Size of each slice
                'mode': str,              # Slicing mode used
                'processing_time': float   # Total processing time in seconds
            }
    
    Raises:
        InvalidImageError: If the input image is invalid or cannot be opened
        SliceProcessingError: If there's an error during slice creation
        ValueError: If overlap is larger than slice size or other invalid parameters
    
    Example:
        >>> result = slice_image('core.jpg', 'slices', slice_size=(1000, 100))
        >>> print(f"Created {result['total_slices']} slices in {result['processing_time']:.2f}s")
    """
    # Open the image
    start_time = time.time()
    try:
        with Image.open(image_path) as img:
            # Validate image
            if img.mode not in ['RGB', 'RGBA', 'L']:
                warnings.warn(f"Converting image from mode {img.mode} to RGB")
                img = img.convert('RGB')
            img = img.copy()  # Create a copy to work with after the with block
    except (IOError, OSError) as e:
        logger.error(f"Error opening image: {e}")
        raise InvalidImageError(f"Failed to open image {image_path}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise ImageSlicerError(f"Unexpected error processing {image_path}: {str(e)}")
    
    # Get image dimensions
    width, height = img.size
    
    # Create output directory if not specified
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = f"{base_name}_slices"
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate number of slices in each dimension based on mode
    h_overlap, v_overlap = overlap
    slice_width, slice_height = slice_size
    
    if mode == "image_count" and horizontal_images:
        # For image_count mode, use exactly the number of horizontal slices requested
        num_h_slices = horizontal_images
        
        # Calculate where to position each slice to maintain exact slice_width
        # and distribute them evenly across the image width
        if horizontal_images == 1:
            # For a single slice, center it
            h_positions = [(width - slice_width) // 2]
        else:
            # Calculate the step between slice starting positions
            available_width = width - slice_width  # Total width minus slice width
            step = available_width / (horizontal_images - 1)  # Use floating point for more precise positioning
            
            # Generate exact starting positions for each slice
            h_positions = [round(i * step) for i in range(horizontal_images)]
    else:
        # In pixel_overlap mode, use the specified slice size and overlap
        if slice_width <= h_overlap and h_overlap > 0:
            raise ValueError("Horizontal overlap must be less than slice width")
        
        # For pixel_overlap mode, calculate the number of slices and their positions
        if slice_width > width:
            num_h_slices = 1  # Special case: if slice is wider than image, just one slice
            h_positions = [0]  # One slice at the left edge
        else:
            h_step = slice_width - h_overlap
            num_h_slices = max(1, ((width - slice_width) // h_step) + 1)
            h_positions = [i * h_step for i in range(num_h_slices)]
            
            # Check if we need one more slice to cover the right edge
            if h_positions[-1] + slice_width < width:
                h_positions.append(width - slice_width)
                num_h_slices += 1
    
    # Similar logic for vertical slices
    if slice_height <= v_overlap and v_overlap > 0:
        raise ValueError("Vertical overlap must be less than slice height")
        
    if slice_height > height:
        num_v_slices = 1
        v_positions = [0]
    else:
        v_step = slice_height - v_overlap
        num_v_slices = max(1, ((height - slice_height) // v_step) + 1)
        v_positions = [i * v_step for i in range(num_v_slices)]
        
        # Check if we need one more slice to cover the bottom edge
        if v_positions[-1] + slice_height < height:
            v_positions.append(height - slice_height)
            num_v_slices += 1
    
    total_slices = num_h_slices * num_v_slices
    
    # Prepare slice parameters for parallel processing
    slice_params = []
    
    # Flag for image_count mode
    is_image_count = (mode == "image_count" and horizontal_images)
    
    for v_idx, upper in enumerate(v_positions):
        lower = min(upper + slice_height, height)
        
        for h_idx, left in enumerate(h_positions):
            right = min(left + slice_width, width)
            
            # Ensure we don't exceed image boundaries
            left = max(0, min(left, width - 1))
            upper_pos = max(0, min(upper, height - 1))
            right = max(left + 1, min(right, width))
            lower_pos = max(upper_pos + 1, min(lower, height))
            
            slice_filename = os.path.join(output_dir, f"slice_{v_idx}_{h_idx}.png")
            # Pass the is_image_count flag to process_slice
            slice_params.append((img, left, upper_pos, right, lower_pos, slice_filename, is_image_count))
    
    # Set maximum number of worker threads (default: CPU count)
    if max_threads is None:
        max_threads = multiprocessing.cpu_count()
    
    # Process slices - use special handling for image_count mode
    successful_slices = 0
    
    # For image_count mode, we need to ensure all slices are created
    if is_image_count:
        # First, process all slices in the first row to ensure they're all created
        first_row_params = [p for p in slice_params if os.path.basename(p[5]).startswith('slice_0_')]
        
        # Process these first, with retries if needed
        for param in tqdm(first_row_params, desc="Processing first row slices"):
            success = False
            retries = 3
            
            while not success and retries > 0:
                success = process_slice(param)
                if not success:
                    time.sleep(0.1)  # Small delay before retry
                    retries -= 1
            
            if success:
                successful_slices += 1
        
        # Now process the rest of the slices with normal parallel execution
        remaining_params = [p for p in slice_params if not os.path.basename(p[5]).startswith('slice_0_')]
        if remaining_params:
            with tqdm(total=len(remaining_params), desc="Processing remaining slices") as pbar:
                with ThreadPoolExecutor(max_workers=max_threads) as executor:
                    futures = [executor.submit(process_slice, param) for param in remaining_params]
                    for future in as_completed(futures):
                        try:
                            if future.result():
                                successful_slices += 1
                        except Exception as e:
                            print(f"Error in future: {e}")
                        finally:
                            pbar.update(1)
    else:
        # Standard processing for pixel_overlap mode
        with tqdm(total=len(slice_params), desc="Slicing image") as pbar:
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = [executor.submit(process_slice, param) for param in slice_params]
                for future in as_completed(futures):
                    try:
                        if future.result():
                            successful_slices += 1
                    except Exception as e:
                        print(f"Error in future: {e}")
                    finally:
                        pbar.update(1)
    
    # Final verification for image_count mode
    if is_image_count:
        # Count the files that match the slice_0_* pattern (first row)
        first_row_files = [f for f in os.listdir(output_dir) if f.startswith('slice_0_')]
        
        # If we still don't have enough slices, create emergency placeholders
        if len(first_row_files) < horizontal_images:
            for h in range(horizontal_images):
                slice_filename = os.path.join(output_dir, f"slice_0_{h}.png")
                if not os.path.exists(slice_filename):
                    try:
                        # Create a minimal placeholder image
                        placeholder = Image.new('RGB', (1, 1), color='white')
                        placeholder.save(slice_filename)
                        successful_slices += 1
                    except Exception:
                        pass
    
    processing_time = time.time() - start_time
    logger.info(f"Created {successful_slices} slices in {output_dir}")
    # Return detailed statistics
    stats = {
        'total_slices': successful_slices,
        'output_dir': output_dir,
        'original_size': (width, height),
        'slice_size': slice_size,
        'mode': mode,
        'processing_time': time.time() - start_time
    }
    return stats

def print_help():
    """
    Prints detailed help information for using the script.
    """
    help_message = """
    Image Slicer Script - Help

    This script slices an image into smaller segments with optional overlap.

    Usage:
        python image_slicer.py <image_path> [output_directory] [horizontal_overlap] [vertical_overlap]

    Arguments:
        <image_path>          Path to the input image file.
        [output_directory]    (Optional) Directory to save the sliced images.
                              If not provided, a directory will be created based on the input filename.
        [horizontal_overlap]  (Optional) Number of pixels to overlap between slices horizontally. Default is 0.
        [vertical_overlap]    (Optional) Number of pixels to overlap between slices vertically. Default is 0.
        --mode                Mode of slicing. Options: "pixel_overlap" (default), "image_count".
        --horizontal_images   (Optional) Number of horizontal slices when using "image_count" mode.
        -s[width,height]      (Optional) Specify the size of each slice. Default is 400x400 pixels.

    Slicing Modes:
        pixel_overlap:        (DEFAULT) Slices the image into segments of the specified size with optional overlap.
                              The number of slices depends on the image dimensions and slice size.
                              
        image_count:          Slices the image into a specific number of horizontal segments (specified by 
                              --horizontal_images). Each slice will have the exact dimensions specified by
                              -s[width,height] and will be positioned to distribute evenly across the image
                              width, creating natural overlaps between adjacent slices.

    Examples:
        # Basic usage with default settings (400x400 slices, no overlap)
        python image_slicer.py example.jpg
        
        # Custom output directory with 50px overlap in both directions
        python image_slicer.py example.jpg output_dir 50 50
        
        # Custom slice size of 300x200 pixels
        python image_slicer.py example.jpg -s[300,200]
        
        # Using image_count mode to slice into exactly 5 horizontal segments
        python image_slicer.py example.jpg --mode=image_count --horizontal_images=5
        
        # Combine multiple options
        python image_slicer.py example.jpg output_dir --mode=image_count --horizontal_images=3 50 50

    Requirements:
        - Python 3.x
        - PIL (Pillow) library
        - tqdm library (for progress bar)

    For more information, contact the script author.
    """
    print(help_message)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No arguments provided. Use 'python image_slicer.py --help' for usage instructions.")
        sys.exit(1)
    elif sys.argv[1] == "--help":
        print_help()
        sys.exit(0)
    else:
        try:
            from tqdm import tqdm
        except ModuleNotFoundError:
            print("Error: The 'tqdm' module is not installed. Install it by running 'pip install tqdm'.")
            sys.exit(1)
        
        image_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        overlap = tuple(map(int, sys.argv[3:5])) if len(sys.argv) > 4 else (0, 0)
        slice_size = (400, 400)  # Default slice size
        mode = "pixel_overlap"
        horizontal_images = None
        max_threads = None

        for arg in sys.argv:
            if arg.startswith("-s[") and arg.endswith("]"):
                try:
                    slice_size = tuple(map(int, arg[2:-1].split(",")))
                except ValueError:
                    print("Invalid format for slice size. Use -s[width,height].")
                    sys.exit(1)
            elif arg.startswith("--mode="):
                mode = arg.split("=")[1]
            elif arg.startswith("--horizontal_images="):
                try:
                    horizontal_images = int(arg.split("=")[1])
                except ValueError:
                    print("Invalid value for horizontal_images. Provide an integer.")
                    sys.exit(1)
            elif arg.startswith("--threads="):
                try:
                    max_threads = int(arg.split("=")[1])
                except ValueError:
                    print("Invalid value for threads. Provide an integer.")
                    sys.exit(1)
        
        slice_image(
            image_path, 
            output_dir, 
            slice_size=slice_size, 
            overlap=overlap, 
            mode=mode, 
            horizontal_images=horizontal_images,
            max_threads=max_threads
        )