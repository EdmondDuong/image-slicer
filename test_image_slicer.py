"""
Test script for the image_slicer.py tool.
This script will test various functionalities of the image slicer
by generating test images and running the slicer with different parameters.
"""

import os
import sys
import shutil
import time
from PIL import Image, ImageDraw
import unittest
import logging
from image_slicer import (
    slice_image,
    ImageSlicerError,
    InvalidImageError,
    SliceProcessingError
)

# Disable logging during tests
logging.getLogger('image_slicer').setLevel(logging.ERROR)

# Directory for test outputs
TEST_DIR = "image_slicer_tests"

def create_test_image(width, height, filename="test_image.png"):
    """Create a test image with a checkerboard pattern for easy visual verification."""
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a border
    draw.rectangle([(0, 0), (width-1, height-1)], outline='black')
    
    # Draw horizontal and vertical lines every 50 pixels
    for i in range(50, width, 50):
        draw.line([(i, 0), (i, height-1)], fill='black')
    for i in range(50, height, 50):
        draw.line([(0, i), (width-1, i)], fill='black')
    
    # Draw some filled shapes to make the test image more distinct
    draw.rectangle([(width//4, height//4), (3*width//4, 3*height//4)], outline='red', width=3)
    draw.ellipse([(width//3, height//3), (2*width//3, 2*height//3)], outline='blue', width=3)
    
    # Add markers at corners and center for easier verification
    for x, y in [(10, 10), (width-10, 10), (10, height-10), (width-10, height-10), (width//2, height//2)]:
        draw.rectangle([(x-5, y-5), (x+5, y+5)], fill='green')
    
    # Save the image
    img.save(filename)
    return filename

class ImageSlicerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create the test directory and generate test images."""
        if not os.path.exists(TEST_DIR):
            os.makedirs(TEST_DIR)
        
        # Create test images of different sizes
        cls.small_image = os.path.join(TEST_DIR, "small.png")
        cls.medium_image = os.path.join(TEST_DIR, "medium.png")
        cls.large_image = os.path.join(TEST_DIR, "large.png")
        
        create_test_image(200, 200, cls.small_image)
        create_test_image(800, 600, cls.medium_image)
        create_test_image(1500, 1000, cls.large_image)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        # Uncomment the following line to remove test files when done
        # shutil.rmtree(TEST_DIR)
        pass
    
    def test_basic_slicing(self):
        """Test basic slicing with default parameters."""
        output_dir = os.path.join(TEST_DIR, "basic_test")
        result = slice_image(self.medium_image, output_dir)
        self.assertGreater(result['total_slices'], 0, "Should have created at least one slice")
        self.assertTrue(os.path.exists(output_dir), "Output directory should exist")
        self.assertIn('processing_time', result, "Should include processing time")
    
    def test_custom_slice_size(self):
        """Test slicing with custom slice sizes."""
        output_dir = os.path.join(TEST_DIR, "custom_size_test")
        slice_size = (300, 200)
        result = slice_image(self.medium_image, output_dir, slice_size=slice_size)
        self.assertGreater(result['total_slices'], 0, "Should have created at least one slice")
        self.assertEqual(result['slice_size'], slice_size, "Should record correct slice size")
        
        # Verify slice dimensions (except possibly edge slices)
        first_slice = os.path.join(output_dir, "slice_0_0.png")
        if os.path.exists(first_slice):
            with Image.open(first_slice) as img:
                self.assertLessEqual(img.width, slice_size[0], "Slice width should be at most the specified width")
                self.assertLessEqual(img.height, slice_size[1], "Slice height should be at most the specified height")
    
    def test_overlap(self):
        """Test slicing with overlap."""
        output_dir = os.path.join(TEST_DIR, "overlap_test")
        overlap = (50, 50)
        result = slice_image(self.medium_image, output_dir, overlap=overlap)
        self.assertGreater(result['total_slices'], 0, "Should have created at least one slice")
        self.assertIn('processing_time', result, "Should include processing time")
    
    def test_image_count_mode(self):
        """Test image_count mode with fixed slice sizes."""
        output_dir = os.path.join(TEST_DIR, "image_count_test")
        horizontal_images = 5
        slice_width = 200
        slice_height = 150
        
        # Slice the image with image_count mode and fixed slice dimensions
        result = slice_image(
            self.large_image, 
            output_dir,
            slice_size=(slice_width, slice_height),
            mode="image_count",
            horizontal_images=horizontal_images
        )
        
        # Get slices from the first row
        slice_files = [f for f in os.listdir(output_dir) if f.startswith('slice_0_')]
        
        # Test 1: Verify we have the exact number of slices requested
        self.assertEqual(len(slice_files), horizontal_images, 
                        f"Should have created exactly {horizontal_images} horizontal slices")
        
        # Test 2: Verify each slice has the exact dimensions specified
        for i in range(horizontal_images):
            slice_path = os.path.join(output_dir, f"slice_0_{i}.png")
            self.assertTrue(os.path.exists(slice_path), f"Slice {i} should exist")
            
            with Image.open(slice_path) as img:
                self.assertEqual(img.width, slice_width, f"Slice {i} should have width {slice_width}")
                self.assertEqual(img.height, slice_height, f"Slice {i} should have height {slice_height}")
        
        # Test 3: Verify the slices are properly distributed
        # Calculate expected positions for a 1500px wide image with 5 slices of 200px each
        img_width = 1500  # Width of large test image
        expected_positions = []
        
        if horizontal_images > 1:
            # Calculate step between slice positions
            step = (img_width - slice_width) / (horizontal_images - 1)
            expected_positions = [round(i * step) for i in range(horizontal_images)]
        
        # Verify that slices have the expected positions
        for i, expected_left in enumerate(expected_positions):
            # Extract actual position from slice metadata
            slice_path = os.path.join(output_dir, f"slice_0_{i}.png")
            with Image.open(slice_path) as img:
                # Open the original image to extract the slice at the expected position
                with Image.open(self.large_image) as orig_img:
                    expected_slice = orig_img.crop((expected_left, 0, expected_left + slice_width, slice_height))
                    
                    # Compare a few pixels from the actual slice with the expected slice
                    # We'll check pixels at corners and center
                    pixels_to_check = [(0, 0), (slice_width-1, 0), (0, slice_height-1), 
                                      (slice_width-1, slice_height-1), (slice_width//2, slice_height//2)]
                    
                    for x, y in pixels_to_check:
                        if 0 <= x < img.width and 0 <= y < img.height and \
                           0 <= x < expected_slice.width and 0 <= y < expected_slice.height:
                            self.assertEqual(img.getpixel((x, y)), expected_slice.getpixel((x, y)), 
                                          f"Pixel mismatch at ({x}, {y}) in slice {i}")

    def test_image_count_specific_dimensions(self):
        """Test image_count mode with specific image dimensions and slice count."""
        output_dir = os.path.join(TEST_DIR, "image_count_dimensions_test")
        
        # Create a test image with a known width for predictable calculations
        test_image = os.path.join(TEST_DIR, "test_600px.png")
        create_test_image(600, 400, test_image)
        
        # Parameters for the test
        horizontal_images = 5
        slice_width = 200
        slice_height = 150
        
        # Expected positions for slices in a 600px wide image with 5 slices of 200px width
        # Step should be (600-200)/(5-1) = 100px
        expected_positions = [0, 100, 200, 300, 400]
        
        result = slice_image(
            test_image, 
            output_dir,
            slice_size=(slice_width, slice_height),
            mode="image_count",
            horizontal_images=horizontal_images
        )
        
        # Get slices from the first row
        slice_files = [f for f in os.listdir(output_dir) if f.startswith('slice_0_')]
        
        # Verify we have the exact number of slices
        self.assertEqual(len(slice_files), horizontal_images, 
                        f"Should have created exactly {horizontal_images} horizontal slices")
        
        # Verify each slice has the correct dimensions
        for i in range(horizontal_images):
            slice_path = os.path.join(output_dir, f"slice_0_{i}.png")
            with Image.open(slice_path) as img:
                self.assertEqual(img.width, slice_width, f"Slice {i} should have width {slice_width}")
                self.assertEqual(img.height, slice_height, f"Slice {i} should have height {slice_height}")
                
                # Create reference slice from the original image at expected position
                with Image.open(test_image) as orig_img:
                    expected_slice = orig_img.crop((expected_positions[i], 0,
                                                 expected_positions[i] + slice_width, slice_height))
                    
                    # Create a copy of the expected slice since we'll use it after orig_img is closed
                    expected_slice_copy = expected_slice.copy()
                    
                    # Verify a sample of pixels match
                    center_x, center_y = slice_width // 2, slice_height // 2
                    self.assertEqual(img.getpixel((center_x, center_y)),
                                    expected_slice_copy.getpixel((center_x, center_y)),
                                    f"Center pixel mismatch in slice {i}")
    
    def test_multithreading(self):
        """Test multithreaded processing."""
        output_dir = os.path.join(TEST_DIR, "multithreading_test")
        
        # Time with single thread
        start_time = time.time()
        slice_image(self.large_image, output_dir + "_single", max_threads=1)
        single_thread_time = time.time() - start_time
        
        # Time with multiple threads
        start_time = time.time()
        slice_image(self.large_image, output_dir + "_multi", max_threads=None)  # Use CPU count
        multi_thread_time = time.time() - start_time
        
        # Multi-threading should generally be faster, but this is not guaranteed
        # So we just print the results rather than asserting
        print(f"Single thread time: {single_thread_time:.2f}s")
        print(f"Multi thread time: {multi_thread_time:.2f}s")
        print(f"Speedup: {single_thread_time / multi_thread_time:.2f}x")
    
    def test_edge_cases(self):
        """Test edge cases: small images, large slices."""
        # Test with slice size larger than image
        output_dir = os.path.join(TEST_DIR, "edge_case_large_slice")
        result = slice_image(self.small_image, output_dir, slice_size=(500, 500))
        self.assertEqual(result['total_slices'], 1, "Should create exactly one slice for oversized slice dimensions")
        self.assertEqual(result['slice_size'], (500, 500), "Should record correct slice size")
        
        # Test with very small slices
        output_dir = os.path.join(TEST_DIR, "edge_case_small_slice")
        result = slice_image(self.small_image, output_dir, slice_size=(10, 10))
        self.assertGreater(result['total_slices'], 10, "Should create many small slices")
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with non-existent image
        output_dir = os.path.join(TEST_DIR, "error_handling")
        with self.assertRaises(InvalidImageError):
            slice_image("nonexistent_image.jpg", output_dir)
        
        # Test with invalid overlap
        with self.assertRaises(ValueError):
            slice_image(self.medium_image, output_dir, slice_size=(100, 100), overlap=(200, 0))
        
        # Test with corrupted/invalid image
        invalid_image = os.path.join(TEST_DIR, "invalid.png")
        with open(invalid_image, 'w') as f:
            f.write("Not a valid image")
        with self.assertRaises(InvalidImageError):
            slice_image(invalid_image, output_dir)

def run_tests():
    """Run the test cases."""
    print(f"Running image slicer tests... Test outputs will be in {os.path.abspath(TEST_DIR)}")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    run_tests()
