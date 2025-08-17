#!/bin/bash

# Quick Start Script for New Image Processing and Scaling Features
# This script demonstrates the new capabilities of the YOLO detection system

echo "=========================================="
echo "YOLO Detection System - New Features Demo"
echo "=========================================="
echo ""

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Virtual environment not found. Please run:"
    echo "python -m venv .venv"
    echo "source .venv/bin/activate"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Check if required packages are installed
echo "Checking required packages..."
python -c "import torch, cv2, ultralytics, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install ultralytics opencv-python numpy pillow matplotlib seaborn
fi

echo ""
echo "Available features:"
echo "1. Image folder processing with recursive subfolder support"
echo "2. Advanced GPU-accelerated scaling (512x512, 512x768, 768x512, 1024x1024)"
echo "3. Custom dimension scaling with aspect ratio preservation"
echo "4. Comprehensive validation suite for all features"
echo ""

# Create test image folder if it doesn't exist
if [ ! -d "demo_images" ]; then
    echo "Creating demo image folder..."
    mkdir -p demo_images/subfolder1
    mkdir -p demo_images/subfolder2
    
    # Create some test images using Python
    python3 -c "
import cv2
import numpy as np
import os

# Create test images
for i in range(5):
    # Create different sized images
    sizes = [(640, 480), (480, 640), (512, 512), (1920, 1080), (3840, 2160)]
    w, h = sizes[i % len(sizes)]
    
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (w-50, h-50), (0, 255, 0), 3)
    cv2.rectangle(img, (100, 100), (w-100, h-100), (255, 0, 0), 2)
    cv2.putText(img, f'Demo {i+1} - {w}x{h}', (150, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    filename = f'demo_images/demo_{i+1}_{w}x{h}.jpg'
    cv2.imwrite(filename, img)
    print(f'Created: {filename}')

# Create subfolder images
for i in range(3):
    w, h = 800, 600
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (w-50, h-50), (0, 255, 255), 3)
    cv2.putText(img, f'Subfolder {i+1}', (150, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    filename = f'demo_images/subfolder1/sub_{i+1}_{w}x{h}.jpg'
    cv2.imwrite(filename, img)
    print(f'Created: {filename}')

for i in range(2):
    w, h = 1024, 768
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (w-50, h-50), (255, 255, 0), 3)
    cv2.putText(img, f'Subfolder2 {i+1}', (150, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    filename = f'demo_images/subfolder2/sub2_{i+1}_{w}x{h}.jpg'
    cv2.imwrite(filename, img)
    print(f'Created: {filename}')

print('Demo images created successfully!')
"
fi

echo ""
echo "Demo setup complete!"
echo ""
echo "To test the new features:"
echo ""
echo "1. Launch the GUI with image folder support:"
echo "   python main_v3.py"
echo ""
echo "2. In the GUI:"
echo "   - Select 'image_folder' as input type"
echo "   - Browse to 'demo_images' folder"
echo "   - Enable 'Process Subfolders Recursively' for full demo"
echo "   - Configure scaling options (e.g., 512x512 preprocessing)"
echo "   - Enable GPU scaling for best performance"
echo "   - Start detection"
echo ""
echo "3. Test the validation suite:"
echo "   python validation_suite_v3.py --model yolo11s.pt --image-folder demo_images --output ./validation_results"
echo ""
echo "4. Run the feature test script:"
echo "   python test_new_features.py"
echo ""
echo "5. Check the output folders:"
echo "   - ./output/ - Detection results"
echo "   - ./validation_results/ - Validation reports and scaling tests"
echo ""

# Check if CUDA is available
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️  CUDA not available - GPU scaling will fallback to CPU')
"

echo ""
echo "Demo images are in the 'demo_images' folder."
echo "You can add your own images to test with different formats and sizes."
echo ""
echo "Happy testing! 🚀"
