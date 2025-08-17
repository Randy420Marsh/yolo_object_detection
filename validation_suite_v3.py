#!/usr/bin/env python3
"""
YOLO Detection Validation Suite v3
Comprehensive testing and benchmarking for the YOLO detection system
"""

import numpy as np
import cv2
from ultralytics import YOLO
from sort import sort
import torch
import time
import os
import json
import argparse
from pathlib import Path
from threading import Thread, Lock
import queue
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import gc
import glob
import math

# Import scaling utilities from main script
def calculate_optimal_dimensions(width, height, target_width, target_height, maintain_aspect=True):
    """Calculate optimal dimensions maintaining aspect ratio and ensuring multiples of 8"""
    if not maintain_aspect:
        # Round to nearest multiple of 8
        optimal_w = round(target_width / 8) * 8
        optimal_h = round(target_height / 8) * 8
        return optimal_w, optimal_h
    
    # Calculate aspect ratio
    aspect_ratio = width / height
    target_aspect = target_width / target_height
    
    if aspect_ratio > target_aspect:
        # Image is wider than target - fit to width
        optimal_w = round(target_width / 8) * 8
        optimal_h = round(optimal_w / aspect_ratio / 8) * 8
    else:
        # Image is taller than target - fit to height
        optimal_h = round(target_height / 8) * 8
        optimal_w = round(optimal_h * aspect_ratio / 8) * 8
    
    return optimal_w, optimal_h

def gpu_scale_image(image, target_width, target_height, device, interpolation='lanczos'):
    """Scale image using GPU with high-quality interpolation"""
    try:
        if device.type != 'cuda':
            # Fallback to CPU scaling
            return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to GPU tensor
        if len(image.shape) == 3:
            # Color image: HWC -> CHW -> BCHW
            image_tensor = torch.from_numpy(image).to(device, dtype=torch.float32, non_blocking=True)
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            # Grayscale image: HW -> BHW
            image_tensor = torch.from_numpy(image).to(device, dtype=torch.float32, non_blocking=True).unsqueeze(0).unsqueeze(0)
        
        # GPU scaling with high-quality interpolation
        if interpolation == 'lanczos':
            # Use bicubic as closest to Lanczos in PyTorch
            scaled_tensor = torch.nn.functional.interpolate(
                image_tensor, 
                size=(target_height, target_width), 
                mode='bicubic', 
                align_corners=False,
                antialias=True
            )
        else:
            scaled_tensor = torch.nn.functional.interpolate(
                image_tensor, 
                size=(target_height, target_width), 
                mode='bilinear', 
                align_corners=False,
                antialias=True
            )
        
        # Convert back to CPU numpy array
        if len(image.shape) == 3:
            # Color image: BCHW -> CHW -> HWC
            result = scaled_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            # Grayscale image: BHW -> HW
            result = scaled_tensor.squeeze(0).squeeze(0).cpu().numpy()
        
        # Ensure correct data type
        if image.dtype == np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
        
    except Exception as e:
        # Fallback to CPU scaling
        print(f"GPU scaling failed: {e}, falling back to CPU")
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

def parse_scale_string(scale_str, default_width=512, default_height=512):
    """Parse scale string like '512x512' or 'auto' to get dimensions"""
    if scale_str == "auto":
        return default_width, default_height
    elif scale_str == "none" or scale_str == "original":
        return None, None
    elif "x" in scale_str:
        try:
            width, height = scale_str.split("x")
            return int(width), int(height)
        except:
            return default_width, default_height
    else:
        return default_width, default_height

class ValidationSuite:
    def __init__(self, model_path, video_path=None, image_folder=None, output_dir="./validation_results"):
        self.model_path = model_path
        self.video_path = video_path
        self.image_folder = image_folder
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "saved_images").mkdir(exist_ok=True)
        (self.output_dir / "scaling_tests").mkdir(exist_ok=True)
        
        # Initialize components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tracker = None
        self.cap = None
        
        # Performance metrics
        self.performance_data = {
            'fps_history': [],
            'inference_times': [],
            'detection_counts': [],
            'save_times': [],
            'memory_usage': [],
            'gpu_utilization': [],
            'performance_benchmarks': {},
            'scaling_performance': {}
        }
        
        # Validation metrics
        self.validation_results = {
            'class_detection_accuracy': {},
            'bounding_box_consistency': {},
            'file_naming_validation': {},
            'scaling_validation': {},
            'error_summary': []
        }
        
        # Test configurations
        self.test_configs = [
            {'save_cropped': True, 'save_frames': False, 'name': 'crops_only'},
            {'save_cropped': False, 'save_frames': True, 'name': 'frames_only'},
            {'save_cropped': True, 'save_frames': True, 'name': 'both_enabled'},
            {'save_cropped': False, 'save_frames': False, 'name': 'detection_only'}
        ]
        
        # Scaling test configurations
        self.scaling_test_configs = [
            {'preprocess': '512x512', 'postprocess': '512x512', 'name': '512x512_both'},
            {'preprocess': '512x768', 'postprocess': '512x768', 'name': '512x768_both'},
            {'preprocess': '768x512', 'postprocess': '768x512', 'name': '768x512_both'},
            {'preprocess': '1024x1024', 'postprocess': '1024x1024', 'name': '1024x1024_both'},
            {'preprocess': 'none', 'postprocess': '512x512', 'name': 'postprocess_only'},
            {'preprocess': '512x512', 'postprocess': 'original', 'name': 'preprocess_only'},
            {'preprocess': 'custom', 'postprocess': 'custom', 'name': 'custom_scaling'}
        ]
        
        # Initialize logging
        self.log_file = self.output_dir / "logs" / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self._log(f"Validation Suite initialized for {model_path}")
        if video_path:
            self._log(f"Video source: {video_path}")
        if image_folder:
            self._log(f"Image folder source: {image_folder}")
        self._log(f"Device: {self.device}")
        
    def _log(self, message, level="INFO"):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry + "\n")
            
    def _initialize_model(self):
        """Initialize YOLO model with optimizations"""
        try:
            self._log("Loading YOLO model...")
            self.model = YOLO(self.model_path)
            
            if self.device.type == 'cuda':
                self.model.to(self.device)
                # Enable GPU optimizations
                try:
                    self.model.fuse()
                    if hasattr(self.model, 'half'):
                        self.model.half()
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                    torch.cuda.set_per_process_memory_fraction(0.9)
                    
                    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                        torch.backends.cuda.enable_flash_sdp(True)
                    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                        torch.backends.cuda.enable_mem_efficient_sdp(True)
                    if hasattr(torch.backends.cuda, 'enable_math_sdp'):
                        torch.backends.cuda.enable_math_sdp(True)
                        
                    self._log("GPU optimizations enabled successfully")
                except Exception as e:
                    self._log(f"Some GPU optimizations failed: {e}", "WARNING")
                    
            # Get class names
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = list(self.model.names.values())
            else:
                self.class_names = [f"Class_{i}" for i in range(1000)]
                
            self._log(f"Model loaded with {len(self.class_names)} classes")
            
        except Exception as e:
            self._log(f"Failed to load model: {e}", "ERROR")
            raise
            
    def _initialize_tracker(self):
        """Initialize SORT tracker"""
        self.tracker = sort()
        self._log("SORT tracker initialized")
        
    def _initialize_video(self):
        """Initialize video capture or image folder"""
        if self.image_folder:
            self._initialize_image_folder()
        elif self.video_path:
            self._initialize_video_capture()
        else:
            raise ValueError("No video path or image folder specified")
            
    def _initialize_video_capture(self):
        """Initialize video capture"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self._log(f"Video: {self.frame_width}x{self.frame_height} @ {self.fps:.2f} FPS, {self.total_frames} frames")
        
    def _initialize_image_folder(self):
        """Initialize image folder processing"""
        try:
            folder_path = Path(self.image_folder)
            if not folder_path.exists() or not folder_path.is_dir():
                raise ValueError(f"Image folder {self.image_folder} does not exist")
            
            # Define supported image extensions
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
            
            # Collect image files
            self.image_files = []
            for ext in image_extensions:
                self.image_files.extend(folder_path.glob(ext))
            
            # Sort files for consistent processing order
            self.image_files.sort()
            
            if not self.image_files:
                raise ValueError(f"No image files found in {self.image_folder}")
            
            # Get dimensions from first image
            first_image = cv2.imread(str(self.image_files[0]))
            if first_image is not None:
                self.frame_height, self.frame_width = first_image.shape[:2]
                self._log(f"Image dimensions: {self.frame_width}x{self.frame_height}")
            else:
                raise ValueError(f"Failed to read first image: {self.image_files[0]}")
            
            # Set video-like properties for compatibility
            self.total_frames = len(self.image_files)
            self.fps = 1.0  # Process 1 image per second by default
            self.current_image_index = 0
            
            self._log(f"Image Folder: {len(self.image_files)} images found")
            
        except Exception as e:
            self._log(f"Error initializing image folder: {e}", "ERROR")
            raise
            
    def _get_frame(self):
        """Get frame from video or image folder"""
        if self.image_folder:
            return self._get_image_frame()
        else:
            return self.cap.read()
            
    def _get_image_frame(self):
        """Get next image frame for processing"""
        if self.current_image_index >= len(self.image_files):
            return False, None
        
        try:
            image_path = self.image_files[self.current_image_index]
            frame = cv2.imread(str(image_path))
            
            if frame is not None:
                # Update frame dimensions
                self.frame_height, self.frame_width = frame.shape[:2]
                self.current_image_index += 1
                return True, frame
            else:
                self._log(f"Failed to read image: {image_path}", "ERROR")
                self.current_image_index += 1
                return False, None
                
        except Exception as e:
            self._log(f"Error reading image {self.current_image_index}: {e}", "ERROR")
            self.current_image_index += 1
            return False, None
            
    def _get_memory_usage(self):
        """Get current memory usage"""
        if self.device.type == 'cuda':
            return {
                'gpu_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'gpu_cached': torch.cuda.memory_reserved() / 1024**3,     # GB
                'gpu_utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            }
        else:
            import psutil
            process = psutil.Process()
            return {
                'cpu_memory': process.memory_info().rss / 1024**3,  # GB
                'cpu_percent': process.cpu_percent()
            }
            
    def _run_performance_test(self, test_config, duration=15):
        """Run performance test for specified duration"""
        self._log(f"Starting performance test: {test_config['name']} for {duration} seconds")
        
        # Reset metrics
        start_time = time.time()
        frame_count = 0
        detection_count = 0
        inference_times = []
        save_times = []
        
        # Create output directories for this test
        test_output_dir = self.output_dir / "saved_images" / test_config['name']
        test_output_dir.mkdir(exist_ok=True)
        
        cropped_dir = test_output_dir / "cropped" if test_config['save_cropped'] else None
        frames_dir = test_output_dir / "frames" if test_config['save_frames'] else None
        
        if cropped_dir:
            cropped_dir.mkdir(exist_ok=True)
        if frames_dir:
            frames_dir.mkdir(exist_ok=True)
            
        # Reset video to beginning
        if self.image_folder:
            self.current_image_index = 0
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while time.time() - start_time < duration:
            ret, frame = self._get_frame()
            if not ret:
                if self.image_folder:
                    # End of images
                    break
                else:
                    # Loop video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
            frame_count += 1
            
            # Performance test with different input scales
            for scale_factor in [1.0]:
                # Scale frame
                if scale_factor != 1.0:
                    scaled_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
                else:
                    scaled_frame = frame
                    
                # YOLO inference timing
                inference_start = time.time()
                try:
                    results = self.model(scaled_frame, stream=True, device=self.device, verbose=False)
                    
                    # Process results
                    detections = []
                    for res in results:
                        if res.boxes is not None and len(res.boxes) > 0:
                            boxes = res.boxes.xyxy.cpu().numpy()
                            confs = res.boxes.conf.cpu().numpy()
                            classes = res.boxes.cls.cpu().numpy()
                            
                            for box, conf, cls in zip(boxes, confs, classes):
                                if conf > 0.3:  # Confidence threshold
                                    detections.append({
                                        'box': box,
                                        'conf': conf,
                                        'class': int(cls)
                                    })
                                    
                except Exception as e:
                    self._log(f"Inference error: {e}", "ERROR")
                    continue
                    
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                # Save operations timing
                if detections and (test_config['save_cropped'] or test_config['save_frames']):
                    save_start = time.time()
                    
                    if test_config['save_cropped']:
                        for det in detections:
                            self._save_test_crop(scaled_frame, det, cropped_dir)
                            
                    if test_config['save_frames']:
                        self._save_test_frame(scaled_frame, frames_dir, frame_count)
                        
                    save_time = time.time() - save_start
                    save_times.append(save_time)
                    
                detection_count += len(detections)
                
                # Memory monitoring
                memory_info = self._get_memory_usage()
                self.performance_data['memory_usage'].append(memory_info)
                
        # Calculate performance metrics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        avg_inference = np.mean(inference_times) * 1000 if inference_times else 0
        avg_save_time = np.mean(save_times) * 1000 if save_times else 0
        
        # Store results
        test_results = {
            'name': test_config['name'],
            'duration': duration,
            'frames_processed': frame_count,
            'detections': detection_count,
            'avg_fps': avg_fps,
            'avg_inference_ms': avg_inference,
            'avg_save_time_ms': avg_save_time,
            'memory_usage': memory_info
        }
        
        self.performance_data['performance_benchmarks'][test_config['name']] = test_results
        
        self._log(f"Test {test_config['name']} completed: {avg_fps:.1f} FPS, {avg_inference:.1f}ms inference")
        
        return test_results
        
    def _run_scaling_test(self, scaling_config, duration=10):
        """Run scaling performance test"""
        self._log(f"Starting scaling test: {scaling_config['name']} for {duration} seconds")
        
        # Reset to beginning
        if self.image_folder:
            self.current_image_index = 0
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        start_time = time.time()
        frame_count = 0
        scaling_times = []
        
        # Parse scaling parameters
        preprocess_dims = parse_scale_string(scaling_config['preprocess'])
        postprocess_dims = parse_scale_string(scaling_config['postprocess'])
        
        while time.time() - start_time < duration:
            ret, frame = self._get_frame()
            if not ret:
                if self.image_folder:
                    break
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                    
            frame_count += 1
            
            # Test preprocessing scaling
            if preprocess_dims[0] is not None:
                preprocess_start = time.time()
                try:
                    if self.device.type == 'cuda':
                        scaled_frame = gpu_scale_image(frame, preprocess_dims[0], preprocess_dims[1], self.device, 'lanczos')
                    else:
                        scaled_frame = cv2.resize(frame, (preprocess_dims[0], preprocess_dims[1]), interpolation=cv2.INTER_LANCZOS4)
                    preprocess_time = time.time() - preprocess_start
                    scaling_times.append(preprocess_time)
                except Exception as e:
                    self._log(f"Preprocessing scaling failed: {e}", "ERROR")
                    continue
            else:
                scaled_frame = frame
                
            # Test postprocessing scaling
            if postprocess_dims[0] is not None:
                postprocess_start = time.time()
                try:
                    if self.device.type == 'cuda':
                        final_frame = gpu_scale_image(scaled_frame, postprocess_dims[0], postprocess_dims[1], self.device, 'lanczos')
                    else:
                        final_frame = cv2.resize(scaled_frame, (postprocess_dims[0], postprocess_dims[1]), interpolation=cv2.INTER_LANCZOS4)
                    postprocess_time = time.time() - postprocess_start
                    scaling_times.append(postprocess_time)
                except Exception as e:
                    self._log(f"Postprocessing scaling failed: {e}", "ERROR")
                    continue
            else:
                final_frame = scaled_frame
                
            # Save sample scaled image
            if frame_count <= 5:  # Save first 5 frames
                scaling_output_dir = self.output_dir / "scaling_tests" / scaling_config['name']
                scaling_output_dir.mkdir(exist_ok=True)
                
                timestamp = int(time.time() * 1000)
                filename = f"scaled_frame_{frame_count}_{timestamp}.jpg"
                filepath = scaling_output_dir / filename
                cv2.imwrite(str(filepath), final_frame)
                
        # Calculate scaling performance metrics
        total_time = time.time() - start_time
        avg_scaling_time = np.mean(scaling_times) * 1000 if scaling_times else 0
        scaling_fps = frame_count / total_time
        
        # Store scaling test results
        scaling_results = {
            'name': scaling_config['name'],
            'duration': duration,
            'frames_processed': frame_count,
            'avg_scaling_time_ms': avg_scaling_time,
            'scaling_fps': scaling_fps,
            'preprocess_dims': preprocess_dims,
            'postprocess_dims': postprocess_dims,
            'device': str(self.device)
        }
        
        self.performance_data['scaling_performance'][scaling_config['name']] = scaling_results
        
        self._log(f"Scaling test {scaling_config['name']} completed: {scaling_fps:.1f} FPS, {avg_scaling_time:.1f}ms avg scaling")
        
        return scaling_results
        
    def _save_test_crop(self, frame, detection, output_dir):
        """Save test crop for validation"""
        try:
            box = detection['box']
            class_id = detection['class']
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
            
            # Crop object
            x1, y1, x2, y2 = map(int, box)
            cropped = frame[y1:y2, x1:x2]
            
            if cropped.size > 0:
                # Save with timestamp and class info
                timestamp = int(time.time() * 1000)
                filename = f"{class_name}_{timestamp}_{class_id}.jpg"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), cropped)
                
        except Exception as e:
            self._log(f"Failed to save test crop: {e}", "ERROR")
            
    def _save_test_frame(self, frame, output_dir, frame_num):
        """Save test frame for validation"""
        try:
            timestamp = int(time.time() * 1000)
            filename = f"frame_{frame_num}_{timestamp}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame)
            
        except Exception as e:
            self._log(f"Failed to save test frame: {e}", "ERROR")
            
    def _validate_scaling_quality(self):
        """Validate scaling quality and performance"""
        self._log("Validating scaling quality...")
        
        scaling_validation = {}
        scaling_tests_dir = self.output_dir / "scaling_tests"
        
        for test_dir in scaling_tests_dir.iterdir():
            if not test_dir.is_dir():
                continue
                
            test_name = test_dir.name
            scaling_validation[test_name] = {
                'total_images': 0,
                'avg_file_size': 0,
                'dimensions_consistent': True,
                'errors': []
            }
            
            image_files = list(test_dir.glob("*.jpg"))
            if not image_files:
                continue
                
            scaling_validation[test_name]['total_images'] = len(image_files)
            
            # Check dimensions consistency
            expected_dims = None
            total_size = 0
            
            for img_file in image_files:
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        h, w = img.shape[:2]
                        current_dims = (w, h)
                        
                        if expected_dims is None:
                            expected_dims = current_dims
                        elif current_dims != expected_dims:
                            scaling_validation[test_name]['dimensions_consistent'] = False
                            scaling_validation[test_name]['errors'].append(f"Dimension mismatch: {current_dims} vs {expected_dims}")
                            
                        total_size += img_file.stat().st_size
                        
                except Exception as e:
                    scaling_validation[test_name]['errors'].append(f"Failed to read {img_file.name}: {e}")
                    
            if image_files:
                scaling_validation[test_name]['avg_file_size'] = total_size / len(image_files)
                
        self.validation_results['scaling_validation'] = scaling_validation
        
        # Log validation results
        for test_name, results in scaling_validation.items():
            if results['errors']:
                self._log(f"Scaling test {test_name}: {len(results['errors'])} errors", "WARNING")
            else:
                self._log(f"Scaling test {test_name}: All validations passed", "SUCCESS")
                
    def _validate_class_detection_accuracy(self):
        """Validate that detected classes match saved filenames"""
        self._log("Validating class detection accuracy...")
        
        class_validation = {}
        saved_images_dir = self.output_dir / "saved_images"
        
        for test_dir in saved_images_dir.iterdir():
            if not test_dir.is_dir():
                continue
                
            test_name = test_dir.name
            class_validation[test_name] = {
                'total_detections': 0,
                'class_matches': 0,
                'class_mismatches': 0,
                'errors': []
            }
            
            for img_file in test_dir.rglob("*.jpg"):
                if "cropped" in str(img_file.parent):
                    filename = img_file.name
                    parts = filename.replace('.jpg', '').split('_')
                    
                    if len(parts) >= 3:
                        class_name = parts[0]
                        class_id = int(parts[-1])
                        
                        # Validate class name matches class ID
                        if 0 <= class_id < len(self.class_names):
                            expected_class = self.class_names[class_id]
                            if class_name == expected_class:
                                class_validation[test_name]['class_matches'] += 1
                            else:
                                class_validation[test_name]['class_mismatches'] += 1
                                class_validation[test_name]['errors'].append(
                                    f"Class mismatch: {class_name} vs {expected_class} (ID: {class_id})"
                                )
                        else:
                            class_validation[test_name]['errors'].append(f"Invalid class ID: {class_id}")
                            
                        class_validation[test_name]['total_detections'] += 1
                        
        self.validation_results['class_detection_accuracy'] = class_validation
        
        # Log validation results
        for test_name, results in class_validation.items():
            if results['total_detections'] > 0:
                accuracy = (results['class_matches'] / results['total_detections']) * 100
                self._log(f"Test {test_name}: {accuracy:.1f}% accuracy ({results['class_matches']}/{results['total_detections']})")
                
    def _generate_performance_plots(self):
        """Generate performance visualization plots"""
        self._log("Generating performance plots...")
        
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Performance comparison plot
        if self.performance_data['performance_benchmarks']:
            plt.figure(figsize=(12, 8))
            
            test_names = list(self.performance_data['performance_benchmarks'].keys())
            fps_values = [self.performance_data['performance_benchmarks'][name]['avg_fps'] for name in test_names]
            inference_times = [self.performance_data['performance_benchmarks'][name]['avg_inference_ms'] for name in test_names]
            
            # FPS subplot
            plt.subplot(2, 1, 1)
            bars1 = plt.bar(test_names, fps_values, color='skyblue')
            plt.title('Performance Comparison - FPS')
            plt.ylabel('FPS')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars1, fps_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.1f}', ha='center', va='bottom')
            
            # Inference time subplot
            plt.subplot(2, 1, 2)
            bars2 = plt.bar(test_names, inference_times, color='lightcoral')
            plt.title('Performance Comparison - Inference Time')
            plt.ylabel('Inference Time (ms)')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars2, inference_times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        # Scaling performance plot
        if self.performance_data['scaling_performance']:
            plt.figure(figsize=(14, 10))
            
            scaling_names = list(self.performance_data['scaling_performance'].keys())
            scaling_fps = [self.performance_data['scaling_performance'][name]['scaling_fps'] for name in scaling_names]
            scaling_times = [self.performance_data['scaling_performance'][name]['avg_scaling_time_ms'] for name in scaling_names]
            
            # Scaling FPS subplot
            plt.subplot(2, 1, 1)
            bars1 = plt.bar(scaling_names, scaling_fps, color='lightgreen')
            plt.title('Scaling Performance - FPS')
            plt.ylabel('Scaling FPS')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars1, scaling_fps):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.1f}', ha='center', va='bottom')
            
            # Scaling time subplot
            plt.subplot(2, 1, 2)
            bars2 = plt.bar(scaling_names, scaling_times, color='orange')
            plt.title('Scaling Performance - Scaling Time')
            plt.ylabel('Scaling Time (ms)')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars2, scaling_times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'scaling_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        self._log(f"Performance plots saved to: {plots_dir}")
        
    def _generate_performance_report(self):
        """Generate performance report"""
        performance_report_file = self.output_dir / "performance_report.txt"
        
        with open(performance_report_file, 'w') as f:
            f.write("YOLO Detection Performance Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n\n")
            
            f.write("Performance Test Results:\n")
            f.write("-" * 30 + "\n")
            
            for test_name, results in self.performance_data['performance_benchmarks'].items():
                f.write(f"\nTest: {test_name}\n")
                f.write(f"  Duration: {results['duration']}s\n")
                f.write(f"  Frames Processed: {results['frames_processed']}\n")
                f.write(f"  Avg FPS: {results['avg_fps']:.2f}\n")
                f.write(f"  Avg Inference Time: {results['avg_inference_ms']:.2f}ms\n")
                f.write(f"  Avg Save Time: {results['avg_save_time_ms']:.2f}ms\n")
                f.write(f"  Memory Usage: {results['memory_usage']}\n")
                
        self._log(f"Performance report saved to: {performance_report_file}")
        
    def _generate_scaling_report(self):
        """Generate scaling performance report"""
        scaling_report_file = self.output_dir / "scaling_performance_report.txt"
        
        with open(scaling_report_file, 'w') as f:
            f.write("YOLO Detection Scaling Performance Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n\n")
            
            f.write("Scaling Test Results:\n")
            f.write("-" * 30 + "\n")
            
            for test_name, results in self.performance_data['scaling_performance'].items():
                f.write(f"\nTest: {test_name}\n")
                f.write(f"  Duration: {results['duration']}s\n")
                f.write(f"  Frames Processed: {results['frames_processed']}\n")
                f.write(f"  Scaling FPS: {results['scaling_fps']:.2f}\n")
                f.write(f"  Avg Scaling Time: {results['avg_scaling_time_ms']:.2f}ms\n")
                f.write(f"  Preprocess Dims: {results['preprocess_dims']}\n")
                f.write(f"  Postprocess Dims: {results['postprocess_dims']}\n")
                
        self._log(f"Scaling report saved to: {scaling_report_file}")
        
    def _generate_validation_report(self):
        """Generate comprehensive validation report"""
        validation_report_file = self.output_dir / "validation_report.txt"
        
        with open(validation_report_file, 'w') as f:
            f.write("YOLO Detection Validation Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n\n")
            
            # File naming validation
            f.write("File Naming Validation:\n")
            f.write("-" * 25 + "\n")
            naming_results = self.validation_results['file_naming_validation']
            f.write(f"Total Files: {naming_results['total_files']}\n")
            f.write(f"Errors: {len(naming_results['errors'])}\n")
            f.write(f"Error Rate: {naming_results['error_rate']:.2f}%\n\n")
            
            # Scaling validation
            f.write("Scaling Validation:\n")
            f.write("-" * 20 + "\n")
            scaling_results = self.validation_results['scaling_validation']
            for test_name, results in scaling_results.items():
                f.write(f"{test_name}:\n")
                f.write(f"  Images: {results['total_images']}\n")
                f.write(f"  Avg File Size: {results['avg_file_size']:.0f} bytes\n")
                f.write(f"  Dimensions Consistent: {results['dimensions_consistent']}\n")
                if results['errors']:
                    f.write(f"  Errors: {len(results['errors'])}\n")
                f.write("\n")
                
            # Class detection accuracy
            f.write("Class Detection Accuracy:\n")
            f.write("-" * 25 + "\n")
            class_results = self.validation_results['class_detection_accuracy']
            for test_name, results in class_results.items():
                f.write(f"{test_name}:\n")
                f.write(f"  Total Detections: {results['total_detections']}\n")
                f.write(f"  Class Matches: {results['class_matches']}\n")
                f.write(f"  Class Mismatches: {results['class_mismatches']}\n")
                if results['errors']:
                    f.write(f"  Errors: {len(results['errors'])}\n")
                f.write("\n")
                
        self._log(f"Validation report saved to: {validation_report_file}")
        
    def _validate_crop_filename(self, filename):
        """Validate crop filename format"""
        parts = filename.replace('.jpg', '').split('_')
        if len(parts) < 3:
            return False
            
        # Check if second part is timestamp (numeric)
        try:
            int(parts[-2])
            int(parts[-1])
            return True
        except ValueError:
            return False
            
    def _validate_frame_filename(self, filename):
        """Validate frame filename format"""
        parts = filename.replace('.jpg', '').split('_')
        if len(parts) < 3:
            return False
            
        # Check if parts are numeric
        try:
            int(parts[1])  # frame number
            int(parts[2])  # timestamp
            return True
        except ValueError:
            return False
            
    def _validate_file_naming(self):
        """Validate that saved files follow naming conventions"""
        self._log("Validating file naming conventions...")
        
        naming_errors = []
        saved_images_dir = self.output_dir / "saved_images"
        
        for test_dir in saved_images_dir.iterdir():
            if not test_dir.is_dir():
                continue
                
            for img_file in test_dir.rglob("*.jpg"):
                filename = img_file.name
                
                # Validate crop naming: ClassName_timestamp_classid.jpg
                if "cropped" in str(img_file.parent):
                    if not self._validate_crop_filename(filename):
                        naming_errors.append({
                            'file': str(img_file),
                            'error': 'Invalid crop filename format',
                            'expected': 'ClassName_timestamp_classid.jpg'
                        })
                        
                # Validate frame naming: frame_framenum_timestamp.jpg
                elif "frames" in str(img_file.parent):
                    if not self._validate_frame_filename(filename):
                        naming_errors.append({
                            'file': str(img_file),
                            'error': 'Invalid frame filename format',
                            'expected': 'frame_framenum_timestamp.jpg'
                        })
                        
        self.validation_results['file_naming_validation'] = {
            'total_files': len([f for f in saved_images_dir.rglob("*.jpg")]),
            'errors': naming_errors,
            'error_rate': len(naming_errors) / max(1, len([f for f in saved_images_dir.rglob("*.jpg")])) * 100
        }
        
        if naming_errors:
            self._log(f"Found {len(naming_errors)} naming convention violations", "WARNING")
        else:
            self._log("All files follow naming conventions", "SUCCESS")
            
    def _validate_class_detection_accuracy(self):
        """Validate that detected classes match saved filenames"""
        self._log("Validating class detection accuracy...")
        
        class_validation = {}
        saved_images_dir = self.output_dir / "saved_images"
        
        for test_dir in saved_images_dir.iterdir():
            if not test_dir.is_dir():
                continue
                
            test_name = test_dir.name
            class_validation[test_name] = {
                'total_detections': 0,
                'class_matches': 0,
                'class_mismatches': 0,
                'errors': []
            }
            
            for img_file in test_dir.rglob("*.jpg"):
                if "cropped" in str(img_file.parent):
                    filename = img_file.name
                    parts = filename.replace('.jpg', '').split('_')
                    
                    if len(parts) >= 3:
                        class_name = parts[0]
                        class_id = int(parts[-1])
                        
                        # Validate class name matches class ID
                        if 0 <= class_id < len(self.class_names):
                            expected_class = self.class_names[class_id]
                            if class_name == expected_class:
                                class_validation[test_name]['class_matches'] += 1
                            else:
                                class_validation[test_name]['class_mismatches'] += 1
                                class_validation[test_name]['errors'].append(
                                    f"Class mismatch: {class_name} vs {expected_class} (ID: {class_id})"
                                )
                        else:
                            class_validation[test_name]['errors'].append(f"Invalid class ID: {class_id}")
                            
                        class_validation[test_name]['total_detections'] += 1
                        
        self.validation_results['class_detection_accuracy'] = class_validation
        
        # Log validation results
        for test_name, results in class_validation.items():
            if results['total_detections'] > 0:
                accuracy = (results['class_matches'] / results['total_detections']) * 100
                self._log(f"Test {test_name}: {accuracy:.1f}% accuracy ({results['class_matches']}/{results['total_detections']})")
                
    def _generate_performance_plots(self):
        """Generate performance visualization plots"""
        self._log("Generating performance plots...")
        
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Performance comparison plot
        if self.performance_data['performance_benchmarks']:
            plt.figure(figsize=(12, 8))
            
            test_names = list(self.performance_data['performance_benchmarks'].keys())
            fps_values = [self.performance_data['performance_benchmarks'][name]['avg_fps'] for name in test_names]
            inference_times = [self.performance_data['performance_benchmarks'][name]['avg_inference_ms'] for name in test_names]
            
            # FPS subplot
            plt.subplot(2, 1, 1)
            bars1 = plt.bar(test_names, fps_values, color='skyblue')
            plt.title('Performance Comparison - FPS')
            plt.ylabel('FPS')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars1, fps_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.1f}', ha='center', va='bottom')
            
            # Inference time subplot
            plt.subplot(2, 1, 2)
            bars2 = plt.bar(test_names, inference_times, color='lightcoral')
            plt.title('Performance Comparison - Inference Time')
            plt.ylabel('Inference Time (ms)')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars2, inference_times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        # Scaling performance plot
        if self.performance_data['scaling_performance']:
            plt.figure(figsize=(14, 10))
            
            scaling_names = list(self.performance_data['scaling_performance'].keys())
            scaling_fps = [self.performance_data['scaling_performance'][name]['scaling_fps'] for name in scaling_names]
            scaling_times = [self.performance_data['scaling_performance'][name]['avg_scaling_time_ms'] for name in scaling_names]
            
            # Scaling FPS subplot
            plt.subplot(2, 1, 1)
            bars1 = plt.bar(scaling_names, scaling_fps, color='lightgreen')
            plt.title('Scaling Performance - FPS')
            plt.ylabel('Scaling FPS')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars1, scaling_fps):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.1f}', ha='center', va='bottom')
            
            # Scaling time subplot
            plt.subplot(2, 1, 2)
            bars2 = plt.bar(scaling_names, scaling_times, color='orange')
            plt.title('Scaling Performance - Scaling Time')
            plt.ylabel('Scaling Time (ms)')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars2, scaling_times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'scaling_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        self._log(f"Performance plots saved to: {plots_dir}")
        
    def run_validation(self):
        """Run complete validation suite"""
        try:
            self._log("Starting comprehensive validation suite...")
            
            # Initialize components
            self._initialize_model()
            self._initialize_tracker()
            self._initialize_video()
            
            # Run performance tests
            self._log("Running performance tests...")
            for test_config in self.test_configs:
                self._run_performance_test(test_config)
                
            # Run scaling tests
            self._log("Running scaling performance tests...")
            for scaling_config in self.scaling_test_configs:
                self._run_scaling_test(scaling_config)
                
            # Run validation checks
            self._log("Running validation checks...")
            self._validate_file_naming()
            self._validate_class_detection_accuracy()
            self._validate_scaling_quality()
            
            # Generate reports and plots
            self._log("Generating validation reports...")
            self._generate_performance_report()
            self._generate_scaling_report()
            self._generate_validation_report()
            self._generate_performance_plots()
            
            self._log("Validation suite completed successfully!")
            
        except Exception as e:
            self._log(f"Validation suite failed: {e}", "ERROR")
            raise
            
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="YOLO Detection Validation Suite v3")
    parser.add_argument("--model", required=True, help="Path to YOLO model file")
    parser.add_argument("--video", help="Path to video file for testing")
    parser.add_argument("--image-folder", help="Path to image folder for testing")
    parser.add_argument("--output", default="./validation_results", help="Output directory for results")
    parser.add_argument("--duration", type=int, default=15, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.video and not args.image_folder:
        print("Error: Must specify either --video or --image-folder")
        return 1
        
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
        
    try:
        # Initialize validation suite
        suite = ValidationSuite(
            model_path=args.model,
            video_path=args.video,
            image_folder=args.image_folder,
            output_dir=args.output
        )
        
        # Run validation
        suite.run_validation()
        
        # Cleanup
        suite.cleanup()
        
        print(f"\nValidation completed successfully! Results saved to: {args.output}")
        return 0
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
