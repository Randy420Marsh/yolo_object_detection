#!/bin/bash

# YOLO Detection Validation Suite Test Runner
# This script runs a quick validation test to ensure everything is working

echo "=========================================="
echo "YOLO Detection Validation Suite v3"
echo "=========================================="

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Activating virtual environment..."
    source ../yolo_image_sorter/venv/bin/activate
fi

# Set default values
MODEL_PATH="yolo11s_best_v11.pt"
VIDEO_PATH="video-3840x2160.mp4"
OUTPUT_DIR="./validation_results"
DURATION=10

# Check if files exist
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

if [[ ! -f "$VIDEO_PATH" ]]; then
    echo "Error: Video file not found: $VIDEO_PATH"
    exit 1
fi

echo "Model: $MODEL_PATH"
echo "Video: $VIDEO_PATH"
echo "Output: $OUTPUT_DIR"
echo "Duration: ${DURATION}s per test configuration"
echo ""

# Run validation suite
echo "Starting validation suite..."
python validation_suite_v3.py \
    --model "$MODEL_PATH" \
    --video "$VIDEO_PATH" \
    --output "$OUTPUT_DIR" \
    --duration "$DURATION"

if [[ $? -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo "Validation completed successfully!"
    echo "Check results in: $OUTPUT_DIR"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Validation failed!"
    echo "Check logs for errors"
    echo "=========================================="
    exit 1
fi


