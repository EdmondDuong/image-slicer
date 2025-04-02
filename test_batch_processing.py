"""
Test script for batch image processing.
This script validates that the batch processing feature works correctly with
different settings and input patterns.
"""

import os
import sys
import shutil
import unittest
from PIL import Image, ImageDraw
import logging
from batch_process import batch_process, BatchProcessingError
from image_slicer import ImageSlicerError

# Disable logging during tests
logging.getLogger('batch_process').setLevel(logging.ERROR)
logging.getLogger('image_slicer').setLevel(logging.ERROR)

# Directory for test outputs
TEST_DIR = "batch_process_tests"

def create_test_images(count=5, width=400, height=300, base_name="test_image"):
    """Create multiple test images for batch processing testing."""
    images = []
    for i in range(count):
        filename = os.path.join(TEST_DIR, f"{base_name}_{i+1}.png")
        # Create image with different colors for easier identification
        color = (i * 50 % 255, (i * 30 + 100) % 255, (i * 70 + 50) % 255)
        with Image.new('RGB', (width, height), color=color) as img:
            draw = ImageDraw.Draw(img)
            
            # Add some identifying text and patterns
            draw.text((10, 10), f"Test Image {i+1}", fill="white")
            draw.rectangle([(20, 20), (width-20, height-20)], outline="white")
            
            # Draw different patterns for each image
            if i % 3 == 0:
                for x in range(0, width, 50):
                    draw.line([(x, 0), (x, height)], fill="white")
            elif i % 3 == 1:
                for y in range(0, height, 50):
                    draw.line([(0, y), (width, y)], fill="white")
            else:
                for j in range(0, min(width, height), 50):
                    draw.rectangle([(j, j), (j+20, j+20)], fill="white")
            
            img.save(filename)
            images.append(filename)
    
    return images

class BatchProcessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create test directory and sample images."""
        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)
        os.makedirs(TEST_DIR)
        
        # Create different types of test images
        cls.test_images = create_test_images(count=5)
        cls.small_images = create_test_images(count=3, width=150, height=100, base_name="small")
        cls.large_images = create_test_images(count=2, width=1200, height=800, base_name="large")

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        # Uncomment to remove the test directory when done
        # shutil.rmtree(TEST_DIR)
        pass

    def test_basic_batch(self):
        """Test basic batch processing with default settings."""
        output_dir = os.path.join(TEST_DIR, "basic_batch")
        result = batch_process(os.path.join(TEST_DIR, "test_image_*.png"), output_dir)
        
        # Check summary statistics
        self.assertEqual(result['summary']['total_images_processed'], 5, "Should have processed 5 images")
        self.assertGreater(result['summary']['total_slices_created'], 0, "Should have created slices")
        self.assertGreater(result['summary']['total_processing_time'], 0, "Processing time should be recorded")
        
        # Check that the output directories exist
        for i in range(1, 6):
            expected_dir = os.path.join(output_dir, f"test_image_{i}")
            self.assertTrue(os.path.isdir(expected_dir), f"Output directory {expected_dir} should exist")
            # Check that slices were created
            slices = [f for f in os.listdir(expected_dir) if f.startswith("slice_")]
            self.assertTrue(len(slices) > 0, f"Should have created at least one slice in {expected_dir}")

    def test_mixed_sizes(self):
        """Test batch processing images of different sizes."""
        output_dir = os.path.join(TEST_DIR, "mixed_sizes")
        # Process all images (different sizes)
        result = batch_process(os.path.join(TEST_DIR, "*.png"), output_dir)
        self.assertEqual(result['summary']['total_images_processed'], 10, "Should have processed 10 images")
        
        # Verify statistics for each image
        for stat in result['details']:
            self.assertIn('original_size', stat, "Should include original image size")
            self.assertIn('processing_time', stat, "Should include processing time")

    def test_custom_settings(self):
        """Test batch processing with custom slice settings."""
        output_dir = os.path.join(TEST_DIR, "custom_settings")
        result = batch_process(
            os.path.join(TEST_DIR, "large_*.png"), 
            output_dir,
            slice_size=(300, 200),
            overlap=(50, 50)
        )
        self.assertEqual(result['summary']['total_images_processed'], 2, "Should have processed 2 large images")
        
        # Verify that more slices were created with smaller size and overlap
        total_slices = result['summary']['total_slices_created']
        self.assertGreater(total_slices, 8, "Should create many slices with smaller size and overlap")
        
        # Verify custom settings were applied
        for stat in result['details']:
            self.assertEqual(stat['slice_size'], (300, 200), "Should use custom slice size")

    def test_image_count_mode(self):
        """Test batch processing with image_count mode and verify slice dimensions."""
        output_dir = os.path.join(TEST_DIR, "image_count")
        horizontal_images = 4
        slice_width = 300
        slice_height = 200
        
        result = batch_process(
            os.path.join(TEST_DIR, "large_*.png"), 
            output_dir,
            mode="image_count",
            horizontal_images=horizontal_images,
            slice_size=(slice_width, slice_height)
        )
        self.assertEqual(result['summary']['total_images_processed'], 2, "Should have processed 2 large images")
        
        # Verify image count mode settings
        for stat in result['details']:
            self.assertEqual(stat['mode'], 'image_count', "Should use image_count mode")
        # For each processed image, verify:
        for i in range(1, 3):
            img_output_dir = os.path.join(output_dir, f"large_{i}")
            
            # 1. Check that we have exactly the requested number of horizontal slices in first row
            first_row_slices = [f for f in os.listdir(img_output_dir) if f.startswith('slice_0_')]
            self.assertEqual(len(first_row_slices), horizontal_images, 
                         f"Should have created exactly {horizontal_images} horizontal slices in first row")
            
            # 2. Check that each slice has the exact dimensions specified
            for j in range(horizontal_images):
                slice_path = os.path.join(img_output_dir, f"slice_0_{j}.png")
                self.assertTrue(os.path.exists(slice_path), f"Slice {j} should exist for image {i}")
                
                with Image.open(slice_path) as img:
                    self.assertEqual(img.width, slice_width, 
                                    f"Slice {j} for image {i} should have width {slice_width}")
                    self.assertEqual(img.height, slice_height, 
                                    f"Slice {j} for image {i} should have height {slice_height}")
            
            # 3. Verify spacing between slices (for images with width 1200px)
            # For 4 slices of width 300px, step should be (1200-300)/(4-1) = 300px
            expected_positions = [0, 300, 600, 900]
            original_image = os.path.join(TEST_DIR, f"large_{i}.png")
            
            if os.path.exists(original_image):
                with Image.open(original_image) as orig_img:
                    for j, expected_left in enumerate(expected_positions):
                        slice_path = os.path.join(img_output_dir, f"slice_0_{j}.png")
                        with Image.open(slice_path) as slice_img:
                            # Check a key pixel location (center of slice)
                            center_x, center_y = slice_width // 2, slice_height // 2
                            expected_slice = orig_img.crop((expected_left, 0,
                                                          expected_left + slice_width, slice_height))
                            expected_slice_copy = expected_slice.copy()
                            
                            # Check if the center pixel matches
                            if center_x < slice_img.width and center_y < slice_img.height and \
                                center_x < expected_slice_copy.width and center_y < expected_slice_copy.height:
                                self.assertEqual(slice_img.getpixel((center_x, center_y)),
                                                expected_slice_copy.getpixel((center_x, center_y)),
                                                f"Center pixel mismatch in slice {j} for image {i}")
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with nonexistent pattern first
        with self.assertRaises(BatchProcessingError):
            batch_process("nonexistent_pattern_*.png", None)

        # Test with invalid directory (a completely invalid path format)
        with self.assertRaises(BatchProcessingError):
            batch_process(os.path.join(TEST_DIR, "test_image_*.png"), "\\\\invalid\\:*?dir")

def run_tests():
    """Run the test cases."""
    print(f"Running batch processing tests... Test outputs will be in {os.path.abspath(TEST_DIR)}")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    run_tests()
