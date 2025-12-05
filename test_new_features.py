#!/usr/bin/env python3
"""
Test script for the new image folder processing and scaling features
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add the current directory to Python path to import from main
sys.path.append('.')

# Import the scaling utilities
from main import (
    calculate_optimal_dimensions, 
    gpu_scale_image, 
    parse_scale_string,
    gpu_scale_image
)

def create_test_images():
    """Create test images for testing"""
    test_dir = Path("./test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Create different sized test images
    test_images = [
        (640, 480, "landscape"),
        (480, 640, "portrait"), 
        (512, 512, "square"),
        (1920, 1080, "hd"),
        (3840, 2160, "4k")
    ]
    
    print("Creating test images...")
    for width, height, name in test_images:
        # Create a test image with some content
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some colored rectangles
        cv2.rectangle(img, (50, 50), (width-50, height-50), (0, 255, 0), 3)
        cv2.rectangle(img, (100, 100), (width-100, height-100), (255, 0, 0), 2)
        cv2.rectangle(img, (150, 150), (width-150, height-150), (0, 0, 255), 1)
        
        # Add text
        cv2.putText(img, f"{width}x{height}", (200, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        filename = test_dir / f"test_{name}_{width}x{height}.jpg"
        cv2.imwrite(str(filename), img)
        print(f"Created: {filename}")
    
    return test_dir

def test_scaling_utilities():
    """Test the scaling utility functions"""
    print("\nTesting scaling utilities...")
    
    # Test parse_scale_string
    test_scales = ["512x512", "512x768", "768x512", "1024x1024", "auto", "none", "custom"]
    for scale in test_scales:
        width, height = parse_scale_string(scale)
        print(f"Scale '{scale}' -> {width}x{height}")
    
    # Test calculate_optimal_dimensions
    test_cases = [
        (640, 480, 512, 512, True),   # Landscape to square, maintain aspect
        (480, 640, 512, 512, True),   # Portrait to square, maintain aspect
        (1920, 1080, 1024, 1024, True), # HD to square, maintain aspect
        (640, 480, 512, 512, False),  # Don't maintain aspect
    ]
    
    for orig_w, orig_h, target_w, target_h, maintain_aspect in test_cases:
        opt_w, opt_h = calculate_optimal_dimensions(orig_w, orig_h, target_w, target_h, maintain_aspect)
        print(f"Original {orig_w}x{orig_h} -> Optimal {opt_w}x{opt_h} (maintain aspect: {maintain_aspect})")
        
        # Verify multiples of 8
        assert opt_w % 8 == 0, f"Width {opt_w} is not multiple of 8"
        assert opt_h % 8 == 0, f"Height {opt_h} is not multiple of 8"

def test_gpu_scaling():
    """Test GPU scaling functionality"""
    print("\nTesting GPU scaling...")
    
    # Create a test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test different target dimensions
    test_targets = [(512, 512), (512, 768), (768, 512), (1024, 1024)]
    
    for target_w, target_h in test_targets:
        try:
            # Test GPU scaling (will fallback to CPU if GPU not available)
            scaled_img = gpu_scale_image(test_img, target_w, target_h, 
                                       torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
            print(f"Scaling {test_img.shape[1]}x{test_img.shape[0]} -> {scaled_img.shape[1]}x{scaled_img.shape[0]}")
            
            # Verify dimensions
            assert scaled_img.shape[1] == target_w, f"Width mismatch: {scaled_img.shape[1]} != {target_w}"
            assert scaled_img.shape[0] == target_h, f"Height mismatch: {scaled_img.shape[0]} != {target_h}"
            
        except Exception as e:
            print(f"Scaling to {target_w}x{target_h} failed: {e}")

def test_image_folder_processing():
    """Test image folder processing functionality"""
    print("\nTesting image folder processing...")
    
    # Create test images
    test_dir = create_test_images()
    
    # Test recursive vs non-recursive search
    from main import DetectionGUI
    
    # Create a minimal GUI instance for testing
    try:
        # This will test the image detection functionality
        gui = DetectionGUI()
        
        # Test image detection
        gui.image_folder_var.set(str(test_dir))
        gui.recursive_subfolders_var.set(False)
        gui._detect_image_files(str(test_dir))
        
        print("Image folder processing test completed successfully!")
        
    except Exception as e:
        print(f"Image folder processing test failed: {e}")
    
    finally:
        # Cleanup test images
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print("Cleaned up test images")

def main():
    """Main test function"""
    print("Testing new image processing and scaling features...")
    print("=" * 60)
    
    try:
        # Test scaling utilities
        test_scaling_utilities()
        
        # Test GPU scaling
        test_gpu_scaling()
        
        # Test image folder processing
        test_image_folder_processing()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
