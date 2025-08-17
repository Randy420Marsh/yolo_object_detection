import numpy as np
import cv2
from ultralytics import YOLO
from sort import sort
import torch
import time
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from threading import Thread, Lock, Event
import queue
import gc
import subprocess
import platform
import mss
import mss.tools

# Fix Qt threading issues with OpenCV
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.*=false'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

# Import PIL for image processing in tkinter
try:
    from PIL import Image, ImageTk
except ImportError:
    print("Warning: PIL/Pillow not found. Install with: pip install Pillow")
    Image = None
    ImageTk = None

# Add new imports for image processing and GPU scaling
import glob
from pathlib import Path
import math

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

class DetectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLO Detection Settings v3")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)
        
        # Apply dark theme
        self._apply_dark_theme()
        
        # Video player instance
        self.player = None
        self.processing_thread = None
        self.stop_event = Event()
        
        # Class selection variables
        self.class_checkboxes = {}
        self.selected_classes = set()
        self.model_classes_loaded = False
        
        # Performance monitoring
        self.performance_data = {
            'fps_history': [],
            'inference_times': [],
            'detection_counts': [],
            'save_times': []
        }
        
        self._create_gui()
        self._load_initial_classes()
        self._initialize_preview_capture()
        
    def _apply_dark_theme(self):
        """Apply dark theme to the GUI"""
        style = ttk.Style()
        
        # Configure dark theme colors
        style.theme_use('clam')
        style.configure('.', 
                      background='#2b2b2b', 
                      foreground='#ffffff',
                      fieldbackground='#3c3c3c',
                      troughcolor='#3c3c3c',
                      selectbackground='#404040',
                      selectforeground='#ffffff')
        
        # Configure specific widget styles
        style.configure('TLabel', background='#2b2b2b', foreground='#ffffff')
        style.configure('TButton', background='#404040', foreground='#ffffff')
        style.configure('TEntry', fieldbackground='#3c3c3c', foreground='#ffffff')
        style.configure('TCheckbutton', background='#2b2b2b', foreground='#ffffff')
        style.configure('TCombobox', fieldbackground='#3c3c3c', foreground='#ffffff')
        style.configure('TFrame', background='#2b2b2b')
        style.configure('Horizontal.TScale', background='#2b2b2b', troughcolor='#3c3c3c')
        
        # Configure the root window
        self.root.configure(bg='#2b2b2b')
        
    def _create_gui(self):
        """Create the GUI elements with improved two-column layout"""
        # Main frame with better spacing
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for responsive two-column layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)  # Left column (controls)
        main_frame.columnconfigure(1, weight=1)  # Right column (preview)
        
        # Bind window resize event for preview scaling
        self.root.bind('<Configure>', self._on_window_resize)
        
        # Title spanning both columns
        title_label = ttk.Label(main_frame, text="YOLO Detection Settings v3", font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Create left and right columns
        self._create_left_column(main_frame)
        self._create_right_column(main_frame)
        
    def _create_left_column(self, main_frame):
        """Create the left column with all control elements"""
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 20))
        left_frame.columnconfigure(1, weight=1)
        
        row = 0
        
        # Input Source Section
        ttk.Label(left_frame, text="Input Source:", font=("Arial", 12, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        
        # Input type selection
        ttk.Label(left_frame, text="Input Type:").grid(row=row, column=0, sticky=tk.W)
        self.input_type_var = tk.StringVar(value="video_file")
        input_type_combo = ttk.Combobox(left_frame, textvariable=self.input_type_var, 
                                       values=["video_file", "image_folder", "display_capture", "rtmp_stream"], 
                                       state="readonly", width=20)
        input_type_combo.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        input_type_combo.bind('<<ComboboxSelected>>', self._on_input_type_change)
        row += 1
        
        # Video file selection (initially visible)
        ttk.Label(left_frame, text="Video File:").grid(row=row, column=0, sticky=tk.W)
        self.video_path_var = tk.StringVar(value="video-3840x2160.mp4")
        video_entry = ttk.Entry(left_frame, textvariable=self.video_path_var, width=40)
        video_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        ttk.Button(left_frame, text="Browse", command=self._browse_video).grid(row=row, column=2, padx=(10, 0))
        row += 1
        
        # Image folder selection (initially hidden)
        self.image_folder_frame = ttk.Frame(left_frame)
        ttk.Label(self.image_folder_frame, text="Image Folder:").grid(row=0, column=0, sticky=tk.W)
        self.image_folder_var = tk.StringVar(value="./images")
        image_folder_entry = ttk.Entry(self.image_folder_frame, textvariable=self.image_folder_var, width=40)
        image_folder_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        ttk.Button(self.image_folder_frame, text="Browse", command=self._browse_image_folder).grid(row=0, column=2, padx=(10, 0))
        row += 1
        
        # Recursive subfolder processing checkbox
        self.recursive_subfolders_var = tk.BooleanVar(value=False)
        recursive_cb = ttk.Checkbutton(self.image_folder_frame, text="Process Subfolders Recursively", 
                                     variable=self.recursive_subfolders_var)
        recursive_cb.grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=(10, 0))
        row += 1
        
        # Display capture options (initially hidden)
        self.display_frame = ttk.Frame(left_frame)
        ttk.Label(self.display_frame, text="Display:").grid(row=0, column=0, sticky=tk.W)
        self.display_var = tk.StringVar(value=":0.0")
        display_entry = ttk.Entry(self.display_frame, textvariable=self.display_var, width=20)
        display_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        ttk.Button(self.display_frame, text="Refresh", command=self._refresh_displays).grid(row=0, column=2, padx=(10, 0))
        ttk.Label(self.display_frame, text="(e.g., :0.0, :1.0, or monitor name)").grid(row=0, column=3, sticky=tk.W, padx=(10, 0))
        row += 1
        
        # RTMP stream options (initially hidden)
        self.rtmp_frame = ttk.Frame(left_frame)
        ttk.Label(self.rtmp_frame, text="RTMP URL:").grid(row=0, column=0, sticky=tk.W)
        self.rtmp_url_var = tk.StringVar(value="rtmp://localhost/live/stream")
        rtmp_entry = ttk.Entry(self.rtmp_frame, textvariable=self.rtmp_url_var, width=40)
        rtmp_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        ttk.Button(self.rtmp_frame, text="Test", command=self._test_rtmp).grid(row=0, column=2, padx=(10, 0))
        ttk.Label(self.rtmp_frame, text="(e.g., rtmp://server/app/stream)").grid(row=0, column=3, sticky=tk.W, padx=(10, 0))
        row += 1
        
        # Model file selection
        ttk.Label(left_frame, text="Model File:").grid(row=row, column=0, sticky=tk.W)
        self.model_path_var = tk.StringVar(value="yolo11s_best_v11.pt")
        model_entry = ttk.Entry(left_frame, textvariable=self.model_path_var, width=40)
        model_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        ttk.Button(left_frame, text="Browse", command=self._browse_model).grid(row=row, column=2, padx=(10, 0))
        row += 1
        
        # Output Directory Section
        ttk.Label(left_frame, text="Output Settings:", font=("Arial", 12, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(20, 5))
        row += 1
        
        # Output directory selection
        ttk.Label(left_frame, text="Output Dir:").grid(row=row, column=0, sticky=tk.W)
        self.output_dir_var = tk.StringVar(value="./output")
        output_entry = ttk.Entry(left_frame, textvariable=self.output_dir_var, width=40)
        output_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        ttk.Button(left_frame, text="Browse", command=self._browse_output).grid(row=row, column=2, padx=(10, 0))
        row += 1
        
        # Detection Settings Section
        ttk.Label(left_frame, text="Detection Settings:", font=("Arial", 12, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(20, 5))
        row += 1
        
        # Detection FPS
        ttk.Label(left_frame, text="Detection FPS:").grid(row=row, column=0, sticky=tk.W)
        self.detection_fps_var = tk.StringVar(value="5.0")
        fps_entry = ttk.Entry(left_frame, textvariable=self.detection_fps_var, width=10)
        fps_entry.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        ttk.Label(left_frame, text="(0.1 = 1 frame every 10 seconds)").grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1
        
        # Confidence threshold
        ttk.Label(left_frame, text="Confidence:").grid(row=row, column=0, sticky=tk.W)
        self.confidence_var = tk.StringVar(value="0.3")
        conf_entry = ttk.Entry(left_frame, textvariable=self.confidence_var, width=10)
        conf_entry.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        ttk.Label(left_frame, text="(0.0 - 1.0)").grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1
        
        # Crop mode selection
        ttk.Label(left_frame, text="Crop Mode:").grid(row=row, column=0, sticky=tk.W)
        self.crop_mode_var = tk.StringVar(value="square")
        crop_combo = ttk.Combobox(left_frame, textvariable=self.crop_mode_var, values=["square", "portrait", "landscape"], state="readonly", width=15)
        crop_combo.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        row += 1
        
        # Performance Settings
        ttk.Label(left_frame, text="Performance Settings:", font=("Arial", 12, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(20, 5))
        row += 1
        
        # Input scale factor
        ttk.Label(left_frame, text="Input Scale:").grid(row=row, column=0, sticky=tk.W)
        self.input_scale_var = tk.DoubleVar(value=0.5)
        input_scale_scale = ttk.Scale(left_frame, from_=0.1, to=1.0, variable=self.input_scale_var, orient='horizontal', length=200)
        input_scale_scale.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        ttk.Label(left_frame, text="(0.1 = 10% size, 1.0 = full size)").grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1
        
        # Display scale factor
        ttk.Label(left_frame, text="Display Scale:").grid(row=row, column=0, sticky=tk.W)
        self.display_scale_var = tk.DoubleVar(value=0.5)
        display_scale_scale = ttk.Scale(left_frame, from_=0.1, to=1.0, variable=self.display_scale_var, orient='horizontal', length=200)
        display_scale_scale.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        ttk.Label(left_frame, text="(0.1 = 10% size, 1.0 = full size)").grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1
        
        # Advanced Scaling Settings
        ttk.Label(left_frame, text="Advanced Scaling Settings:", font=("Arial", 12, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(20, 5))
        row += 1
        
        # Preprocessing scaling options
        ttk.Label(left_frame, text="Preprocess Scaling:").grid(row=row, column=0, sticky=tk.W)
        self.preprocess_scale_var = tk.StringVar(value="auto")
        preprocess_combo = ttk.Combobox(left_frame, textvariable=self.preprocess_scale_var, 
                                       values=["auto", "512x512", "512x768", "768x512", "1024x1024", "custom", "none"], 
                                       state="readonly", width=15)
        preprocess_combo.bind('<<ComboboxSelected>>', self._on_preprocess_scale_change)
        preprocess_combo.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        row += 1
        
        # Custom preprocessing dimensions
        self.custom_preprocess_frame = ttk.Frame(left_frame)
        ttk.Label(self.custom_preprocess_frame, text="Custom Width:").grid(row=0, column=0, sticky=tk.W)
        self.custom_width_var = tk.StringVar(value="512")
        custom_width_entry = ttk.Entry(self.custom_preprocess_frame, textvariable=self.custom_width_var, width=8)
        custom_width_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        ttk.Label(self.custom_preprocess_frame, text="Height:").grid(row=0, column=2, sticky=tk.W, padx=(10, 0))
        self.custom_height_var = tk.StringVar(value="512")
        custom_height_entry = ttk.Entry(self.custom_preprocess_frame, textvariable=self.custom_height_var, width=8)
        custom_height_entry.grid(row=0, column=3, sticky=tk.W, padx=(5, 0))
        self.custom_preprocess_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=(10, 0))
        self.custom_preprocess_frame.grid_remove()  # Initially hidden
        row += 1
        
        # Postprocessing scaling options
        ttk.Label(left_frame, text="Postprocess Scaling:").grid(row=row, column=0, sticky=tk.W)
        self.postprocess_scale_var = tk.StringVar(value="auto")
        postprocess_combo = ttk.Combobox(left_frame, textvariable=self.postprocess_scale_var, 
                                        values=["auto", "512x512", "512x768", "768x512", "1024x1024", "custom", "original"], 
                                        state="readonly", width=15)
        postprocess_combo.bind('<<ComboboxSelected>>', self._on_postprocess_scale_change)
        postprocess_combo.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        row += 1
        
        # Custom postprocessing dimensions
        self.custom_postprocess_frame = ttk.Frame(left_frame)
        ttk.Label(self.custom_postprocess_frame, text="Custom Width:").grid(row=0, column=0, sticky=tk.W)
        self.custom_post_width_var = tk.StringVar(value="512")
        custom_post_width_entry = ttk.Entry(self.custom_postprocess_frame, textvariable=self.custom_post_width_var, width=8)
        custom_post_width_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        ttk.Label(self.custom_postprocess_frame, text="Height:").grid(row=0, column=2, sticky=tk.W, padx=(10, 0))
        self.custom_post_height_var = tk.StringVar(value="512")
        custom_post_height_entry = ttk.Entry(self.custom_postprocess_frame, textvariable=self.custom_post_height_var, width=8)
        custom_post_height_entry.grid(row=0, column=3, sticky=tk.W, padx=(5, 0))
        self.custom_postprocess_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=(10, 0))
        self.custom_postprocess_frame.grid_remove()  # Initially hidden
        row += 1
        
        # GPU scaling options
        ttk.Label(left_frame, text="GPU Scaling:").grid(row=row, column=0, sticky=tk.W)
        self.gpu_scaling_var = tk.BooleanVar(value=True)
        gpu_scaling_cb = ttk.Checkbutton(left_frame, text="Use NVIDIA GPU for Scaling (Lanczos)", 
                                        variable=self.gpu_scaling_var)
        gpu_scaling_cb.grid(row=row, column=1, columnspan=2, sticky=tk.W, padx=(10, 0))
        row += 1
        
        # Maintain aspect ratio checkbox
        self.maintain_aspect_var = tk.BooleanVar(value=True)
        maintain_aspect_cb = ttk.Checkbutton(left_frame, text="Maintain Aspect Ratio", 
                                           variable=self.maintain_aspect_var)
        maintain_aspect_cb.grid(row=row, column=1, columnspan=2, sticky=tk.W, padx=(10, 0))
        row += 1
        
        # Class Selection Section
        ttk.Label(left_frame, text="Class Selection:", font=("Arial", 12, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(20, 5))
        row += 1
        
        # Class selection frame
        self.class_frame = ttk.Frame(left_frame)
        self.class_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        # Load Classes Button
        ttk.Button(left_frame, text="Load Model Classes", command=self._load_model_classes).grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        # Sample Detection Button (for identifying objects)
        ttk.Button(left_frame, text="Sample Detection (3s)", command=self._sample_detection, 
                  style="Accent.TButton").grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        # Object Tracking Section
        ttk.Label(left_frame, text="Object Tracking:", font=("Arial", 12, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(20, 5))
        row += 1
        
        # Track specific objects checkbox
        self.track_specific_objects_var = tk.BooleanVar(value=False)
        track_objects_cb = ttk.Checkbutton(left_frame, text="Track Only Selected Objects", 
                                         variable=self.track_specific_objects_var,
                                         command=self._on_track_objects_change)
        track_objects_cb.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 0))
        row += 1
        
        # Object selection frame (initially hidden)
        self.object_selection_frame = ttk.Frame(left_frame)
        self.object_selection_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        self.object_selection_frame.grid_remove()  # Initially hidden
        row += 1
        
        # Object selection label
        self.object_selection_label = ttk.Label(self.object_selection_frame, text="Select objects to track:")
        self.object_selection_label.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        
        # Object checkboxes frame
        self.object_checkboxes_frame = ttk.Frame(self.object_selection_frame)
        self.object_checkboxes_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Save options
        ttk.Label(left_frame, text="Save Options:", font=("Arial", 12, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(20, 5))
        row += 1
        
        # Save cropped objects checkbox
        self.save_cropped_var = tk.BooleanVar(value=True)
        save_cropped_cb = ttk.Checkbutton(left_frame, text="Save Cropped Objects", variable=self.save_cropped_var)
        save_cropped_cb.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 0))
        row += 1
        
        # Save full frames checkbox
        self.save_frames_var = tk.BooleanVar(value=True)
        save_frames_cb = ttk.Checkbutton(left_frame, text="Save Full Frames", variable=self.save_frames_var)
        save_frames_cb.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 0))
        row += 1
        
        # Processing controls
        ttk.Label(left_frame, text="Processing:", font=("Arial", 12, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(20, 5))
        row += 1
        
        # Start/Stop button
        self.start_button = ttk.Button(left_frame, text="Start Detection", command=self._start_detection, style="Accent.TButton")
        self.start_button.grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        progress_label = ttk.Label(left_frame, textvariable=self.progress_var)
        progress_label.grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        # Performance monitoring
        ttk.Label(left_frame, text="Performance Monitor:", font=("Arial", 12, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(20, 5))
        row += 1
        
        # Performance labels
        self.fps_label = ttk.Label(left_frame, text="Current FPS: --")
        self.fps_label.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 0))
        row += 1
        
        self.inference_label = ttk.Label(left_frame, text="Avg Inference: -- ms")
        self.inference_label.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=(10, 0))
        row += 1
        
        # Status display
        ttk.Label(left_frame, text="Status Log:", font=("Arial", 12, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(20, 5))
        row += 1
        
        # Status display
        self.status_text = tk.Text(left_frame, height=8, width=60, bg='#3c3c3c', fg='#ffffff', insertbackground='#ffffff')
        self.status_text.grid(row=row, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        # Scrollbar for status text
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        scrollbar.grid(row=row, column=3, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        # Configure status text
        self.status_text.tag_configure("info", foreground="lightblue")
        self.status_text.tag_configure("success", foreground="lightgreen")
        self.status_text.tag_configure("error", foreground="lightcoral")
        self.status_text.tag_configure("warning", foreground="yellow")
        
        # Add initial status
        self._log_status("GUI initialized. Loading classes from model...", "info")
        
    def _create_right_column(self, main_frame):
        """Create the right column with video preview and controls"""
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(20, 0))
        right_frame.columnconfigure(0, weight=1)
        
        # Video Preview Section
        ttk.Label(right_frame, text="Video Preview:", font=("Arial", 14, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 15))
        
        # Video preview canvas
        self.preview_canvas = tk.Canvas(right_frame, bg='black', width=640, height=480)
        self.preview_canvas.grid(row=1, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Preview controls
        preview_controls_frame = ttk.Frame(right_frame)
        preview_controls_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Preview play/pause button
        self.preview_play_var = tk.BooleanVar(value=False)
        self.preview_play_button = ttk.Button(preview_controls_frame, text="▶ Play Preview", 
                                             command=self._toggle_preview_play)
        self.preview_play_button.grid(row=0, column=0, padx=(0, 10))
        
        # Preview refresh button
        ttk.Button(preview_controls_frame, text="🔄 Refresh", 
                  command=self._refresh_preview).grid(row=0, column=1, padx=(0, 10))
        
        # Preview status label
        self.preview_status_var = tk.StringVar(value="Preview: Ready")
        preview_status_label = ttk.Label(preview_controls_frame, textvariable=self.preview_status_var)
        preview_status_label.grid(row=0, column=2, padx=(10, 0))
        
        # Preview info
        ttk.Label(right_frame, text="Preview Info:", font=("Arial", 12, "bold")).grid(row=3, column=0, sticky=tk.W, pady=(20, 5))
        
        # Preview info display
        self.preview_info_text = tk.Text(right_frame, height=6, width=50, bg='#3c3c3c', fg='#ffffff', 
                                        insertbackground='#ffffff', state=tk.DISABLED)
        self.preview_info_text.grid(row=4, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Configure preview info text
        self.preview_info_text.tag_configure("info", foreground="lightblue")
        self.preview_info_text.tag_configure("success", foreground="lightgreen")
        self.preview_info_text.tag_configure("error", foreground="lightcoral")
        self.preview_info_text.tag_configure("warning", foreground="yellow")
        
    def _load_initial_classes(self):
        """Load initial classes from model or fallback"""
        try:
            model_path = self.model_path_var.get()
            if os.path.exists(model_path):
                self._load_model_classes()
            else:
                self._create_fallback_classes()
                self._log_status("Model not found, using fallback classes. Load model to get actual classes.", "warning")
        except Exception as e:
            self._create_fallback_classes()
            self._log_status(f"Error loading model: {e}. Using fallback classes.", "error")
            
    def _create_fallback_classes(self):
        """Create fallback class selection"""
        fallback_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite"
        ]
        self._create_class_checkboxes(fallback_classes)
        
        # Set default selection
        if "person" in self.class_checkboxes:
            self.class_checkboxes["person"].set(True)
            self.selected_classes.add("person")
            
    def _create_class_checkboxes(self, class_names):
        """Create checkboxes for class selection"""
        # Clear existing checkboxes
        for widget in self.class_frame.winfo_children():
            widget.destroy()
            
        # Create new checkboxes
        self.class_checkboxes = {}
        
        # Create a frame for the checkboxes with scrollbar
        canvas = tk.Canvas(self.class_frame, bg='#2b2b2b', height=150)
        scrollbar = ttk.Scrollbar(self.class_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create checkboxes in a grid layout
        cols = 3
        for i, class_name in enumerate(class_names):
            row = i // cols
            col = i % cols
            
            var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(scrollable_frame, text=class_name, variable=var, 
                               command=lambda n=class_name, v=var: self._on_class_selection(n, v))
            cb.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            
            self.class_checkboxes[class_name] = var
            
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Preserve existing selections when updating
        for class_name, var in self.class_checkboxes.items():
            if class_name in self.selected_classes:
                var.set(True)
                
    def _on_class_selection(self, class_name, var):
        """Handle class selection changes"""
        if var.get():
            self.selected_classes.add(class_name)
        else:
            self.selected_classes.discard(class_name)
        
        # Update filter classes string
        filter_str = ",".join(sorted(self.selected_classes))
        self._log_status(f"Selected classes: {filter_str}", "info")
        
    def _on_input_type_change(self, event=None):
        """Handle input type selection change"""
        input_type = self.input_type_var.get()
        
        # Hide all input option frames
        self.display_frame.grid_remove()
        self.rtmp_frame.grid_remove()
        self.image_folder_frame.grid_remove()
        
        # Show appropriate frame based on selection
        if input_type == "display_capture":
            self.display_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        elif input_type == "rtmp_stream":
            self.rtmp_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        elif input_type == "image_folder":
            self.image_folder_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
            # Auto-detect images if folder is already set
            if self.image_folder_var.get() and os.path.exists(self.image_folder_var.get()):
                self._detect_image_files(self.image_folder_var.get())
            
        # Refresh preview for new input type
        self._refresh_preview()
            
    def _browse_video(self):
        """Browse for video file"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.video_path_var.set(filename)
            
    def _browse_image_folder(self):
        """Browse for image folder"""
        directory = filedialog.askdirectory(title="Select Image Folder")
        if directory:
            self.image_folder_var.set(directory)
            # Auto-detect image files
            self._detect_image_files(directory)
            
    def _detect_image_files(self, folder_path):
        """Detect and count image files in the selected folder"""
        try:
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
            image_files = []
            
            if self.recursive_subfolders_var.get():
                # Recursive search
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
            else:
                # Non-recursive search
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            
            if image_files:
                self._log_status(f"Found {len(image_files)} image files in {folder_path}", "success")
                # Show first few image names
                sample_names = [os.path.basename(f) for f in image_files[:5]]
                if len(image_files) > 5:
                    sample_names.append(f"... and {len(image_files) - 5} more")
                self._log_status(f"Sample files: {', '.join(sample_names)}", "info")
            else:
                self._log_status(f"No image files found in {folder_path}", "warning")
                
        except Exception as e:
            self._log_status(f"Error detecting image files: {e}", "error")
            
    def _browse_model(self):
        """Browse for model file"""
        filename = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("Model files", "*.pt *.pth"), ("All files", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
            # Auto-load classes from new model
            self._load_model_classes()
            
    def _browse_output(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)
            
    def _refresh_displays(self):
        """Refresh available display sources using mss"""
        try:
            # Initialize mss to get monitor information
            with mss.mss() as sct:
                monitors = sct.monitors
                
                if len(monitors) > 1:  # More than just the "all monitors" entry
                    display_options = []
                    
                    # Add monitor indices
                    for i in range(1, len(monitors)):  # Skip index 0 (all monitors)
                        monitor = monitors[i]
                        display_options.append(str(i))
                        display_options.append(f":0.{i-1}")  # X11 format
                    
                    # Set default to first monitor
                    if display_options:
                        self.display_var.set(display_options[0])
                        self._log_status(f"Available displays: {', '.join(display_options)}", "info")
                        self._log_status(f"Monitor 1: {monitors[1]['width']}x{monitors[1]['height']}", "info")
                        if len(monitors) > 2:
                            self._log_status(f"Monitor 2: {monitors[2]['width']}x{monitors[2]['height']}", "info")
                    else:
                        self._log_status("No monitors detected", "warning")
                else:
                    self._log_status("Only one monitor detected", "info")
                    
        except Exception as e:
            self._log_status(f"Error detecting displays: {e}", "error")
            # Fallback to default
            self.display_var.set("1")
            
    def _test_rtmp(self):
        """Test RTMP stream connection"""
        rtmp_url = self.rtmp_url_var.get().strip()
        if not rtmp_url:
            messagebox.showerror("Error", "Please enter RTMP URL first!")
            return
            
        self._log_status(f"Testing RTMP connection to: {rtmp_url}", "info")
        
        try:
            # Try to open RTMP stream
            cap = cv2.VideoCapture(rtmp_url)
            if cap.isOpened():
                # Get stream info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                self._log_status(f"RTMP connection successful! Stream: {width}x{height} @ {fps:.2f} FPS", "success")
                cap.release()
            else:
                self._log_status("RTMP connection failed - stream not accessible", "error")
        except Exception as e:
            self._log_status(f"RTMP test failed: {e}", "error")
            
    def _log_status(self, message, tag="info"):
        """Add message to status display"""
        timestamp = time.strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n", tag)
        self.status_text.see(tk.END)
        self.root.update_idletasks()
        
    def _load_model_classes(self):
        """Load classes from the selected model"""
        try:
            if not os.path.exists(self.model_path_var.get()):
                messagebox.showerror("Error", "Model file not found!")
                return
                
            self._log_status("Loading model to get class names...", "info")
            
            # Load model in a separate thread to avoid blocking GUI
            load_thread = Thread(target=self._load_model_classes_thread)
            load_thread.daemon = True
            load_thread.start()
            
        except Exception as e:
            self._log_status(f"Error: {str(e)}", "error")
            
    def _load_model_classes_thread(self):
        """Load model classes in separate thread"""
        try:
            model = YOLO(self.model_path_var.get())
            class_names = list(model.names.values()) if hasattr(model, 'names') and model.names else []
            
            if not class_names:
                # Fallback to generic names
                class_names = [f"Class_{i}" for i in range(1000)]
                
            # Update GUI in main thread
            self.root.after(0, lambda: self._create_class_checkboxes(class_names))
            self.root.after(0, lambda: self._log_status(f"Model loaded with {len(class_names)} classes", "success"))
            self.model_classes_loaded = True
            
        except Exception as e:
            self.root.after(0, lambda: self._log_status(f"Error loading model: {str(e)}", "error"))
            
    def _on_track_objects_change(self):
        """Handle track specific objects checkbox change"""
        if self.track_specific_objects_var.get():
            self.object_selection_frame.grid()
            self._log_status("Object tracking enabled - select objects to track", "info")
        else:
            self.object_selection_frame.grid_remove()
            self._log_status("Object tracking disabled", "info")
            
    def _sample_detection(self):
        """Run a 3-second sample detection to identify objects"""
        if not os.path.exists(self.model_path_var.get()):
            messagebox.showerror("Error", "Model file not found!")
            return
            
        if not self.selected_classes:
            messagebox.showerror("Error", "Please select at least one class to detect!")
            return
            
        self._log_status("Starting 3-second sample detection...", "info")
        
        # Create output directory
        os.makedirs(self.output_dir_var.get(), exist_ok=True)
        
        # Start sample detection thread
        sample_thread = Thread(target=self._run_sample_detection)
        sample_thread.daemon = True
        sample_thread.start()
        
    def _run_sample_detection(self):
        """Run sample detection for 3 seconds to identify objects"""
        try:
            # Initialize model
            model = YOLO(self.model_path_var.get())
            model.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Initialize video source
            input_type = self.input_type_var.get()
            if input_type == "video_file":
                cap = cv2.VideoCapture(self.video_path_var.get())
            elif input_type == "display_capture":
                # Use mss for display capture
                with mss.mss() as sct:
                    monitors = sct.monitors
                    target_monitor = 1  # Default to primary monitor
                    if self.display_var.get().isdigit():
                        target_monitor = int(self.display_var.get())
                    elif self.display_var.get().startswith(":0."):
                        target_monitor = int(self.display_var.get().split(".")[1]) + 1
            elif input_type == "rtmp_stream":
                cap = cv2.VideoCapture(self.rtmp_url_var.get())
            else:
                self.root.after(0, lambda: self._log_status("Unknown input type", "error"))
                return
                
            # Run detection for 3 seconds
            start_time = time.time()
            detected_objects = set()
            
            while time.time() - start_time < 3.0:
                # Get frame
                if input_type == "display_capture":
                    screenshot = sct.grab(monitors[target_monitor])
                    frame = np.array(screenshot)
                    if frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                # Scale frame for detection
                input_scale = self.input_scale_var.get()
                if input_scale != 1.0:
                    h, w = frame.shape[:2]
                    new_h, new_w = int(h * input_scale), int(w * input_scale)
                    frame_scaled = cv2.resize(frame, (new_w, new_h))
                else:
                    frame_scaled = frame
                    
                # Run detection
                results = model(frame_scaled, verbose=False)
                
                # Process results
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            confidence = float(box.conf[0])
                            if confidence >= float(self.confidence_var.get()):
                                class_id = int(box.cls[0])
                                class_name = model.names[class_id]
                                detected_objects.add(class_name)
                                
                # Small delay
                time.sleep(0.01)
                
            # Cleanup
            if input_type != "display_capture":
                cap.release()
                
            # Update GUI with detected objects
            self.root.after(0, lambda: self._create_object_checkboxes(list(detected_objects)))
            self.root.after(0, lambda: self._log_status(f"Sample detection complete. Found {len(detected_objects)} object types: {', '.join(detected_objects)}", "success"))
            
        except Exception as e:
            self.root.after(0, lambda: self._log_status(f"Sample detection error: {str(e)}", "error"))
            
    def _create_object_checkboxes(self, object_names):
        """Create checkboxes for detected objects"""
        # Clear existing checkboxes
        for widget in self.object_checkboxes_frame.winfo_children():
            widget.destroy()
            
        # Create new checkboxes
        self.object_checkboxes = {}
        for i, obj_name in enumerate(object_names):
            var = tk.BooleanVar(value=True)  # Default to selected
            self.object_checkboxes[obj_name] = var
            
            cb = ttk.Checkbutton(self.object_checkboxes_frame, text=obj_name, variable=var)
            cb.grid(row=i//3, column=i%3, sticky=tk.W, padx=(0, 20), pady=2)
            
        # Update object selection label
        self.object_selection_label.config(text=f"Select objects to track ({len(object_names)} found):")
        
    def _toggle_preview_play(self):
        """Toggle preview play/pause"""
        if self.preview_play_var.get():
            self.preview_play_var.set(False)
            self.preview_play_button.config(text="▶ Play Preview")
            self.preview_status_var.set("Preview: Paused")
        else:
            self.preview_play_var.set(True)
            self.preview_play_button.config(text="⏸ Pause Preview")
            self.preview_status_var.set("Preview: Playing")
            self._start_preview_update()
            
    def _start_preview_update(self):
        """Start preview update loop"""
        if self.preview_play_var.get():
            self._update_preview()
            # Schedule next update
            self.root.after(50, self._start_preview_update)  # 20 FPS preview
            
    def _update_preview(self):
        """Update the preview canvas with current frame"""
        try:
            if not self.preview_play_var.get():
                return
                
            # Get current frame based on input type
            input_type = self.input_type_var.get()
            frame = None
            
            if input_type == "video_file":
                if hasattr(self, 'preview_cap') and self.preview_cap:
                    ret, frame = self.preview_cap.read()
                    if not ret:
                        # Reset to beginning
                        self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.preview_cap.read()
            elif input_type == "display_capture":
                # Use mss for display capture
                try:
                    with mss.mss() as sct:
                        monitors = sct.monitors
                        target_monitor = 1  # Default to primary monitor
                        if self.display_var.get().isdigit():
                            target_monitor = int(self.display_var.get())
                        elif self.display_var.get().startswith(":0."):
                            target_monitor = int(self.display_var.get().split(".")[1]) + 1
                            
                        if target_monitor < len(monitors):
                            screenshot = sct.grab(monitors[target_monitor])
                            frame = np.array(screenshot)
                            if frame.shape[2] == 4:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                except Exception as e:
                    self.preview_status_var.set(f"Preview Error: {str(e)}")
                    return
            elif input_type == "rtmp_stream":
                if hasattr(self, 'preview_cap') and self.preview_cap:
                    ret, frame = self.preview_cap.read()
                    if not ret:
                        # Try to reconnect
                        self._initialize_preview_capture()
                        return
                        
            if frame is not None:
                # Resize frame to fit canvas while maintaining aspect ratio
                canvas_width = self.preview_canvas.winfo_width()
                canvas_height = self.preview_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:  # Ensure canvas is properly sized
                    # Calculate aspect ratio preserving dimensions
                    frame_h, frame_w = frame.shape[:2]
                    canvas_aspect = canvas_width / canvas_height
                    frame_aspect = frame_w / frame_h
                    
                    if frame_aspect > canvas_aspect:
                        # Frame is wider than canvas
                        new_width = canvas_width
                        new_height = int(canvas_width / frame_aspect)
                    else:
                        # Frame is taller than canvas
                        new_height = canvas_height
                        new_width = int(canvas_height * frame_aspect)
                    
                    # Resize frame
                    frame_resized = cv2.resize(frame, (new_width, new_height))
                    
                    # Convert to PhotoImage for tkinter
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_photo = ImageTk.PhotoImage(frame_pil)
                    
                    # Clear canvas and display new frame
                    self.preview_canvas.delete("all")
                    self.preview_canvas.create_image(canvas_width//2, canvas_height//2, image=frame_photo, anchor=tk.CENTER)
                    
                    # Keep reference to prevent garbage collection
                    self.current_preview_image = frame_photo
                    
        except Exception as e:
            self.preview_status_var.set(f"Preview Error: {str(e)}")
            
    def _refresh_preview(self):
        """Refresh the preview capture"""
        try:
            input_type = self.input_type_var.get()
            
            # Clean up existing capture
            if hasattr(self, 'preview_cap') and self.preview_cap:
                self.preview_cap.release()
                
            # Initialize new capture
            if input_type == "video_file":
                video_path = self.video_path_var.get()
                if os.path.exists(video_path):
                    self.preview_cap = cv2.VideoCapture(video_path)
                    if self.preview_cap.isOpened():
                        self.preview_status_var.set("Preview: Video loaded")
                    else:
                        self.preview_status_var.set("Preview: Failed to load video")
                else:
                    self.preview_status_var.set("Preview: Video file not found")
            elif input_type == "rtmp_stream":
                rtmp_url = self.rtmp_url_var.get()
                if rtmp_url.strip():
                    self.preview_cap = cv2.VideoCapture(rtmp_url)
                    if self.preview_cap.isOpened():
                        self.preview_status_var.set("Preview: RTMP connected")
                    else:
                        self.preview_status_var.set("Preview: RTMP failed")
                else:
                    self.preview_status_var.set("Preview: RTMP URL required")
            else:
                self.preview_status_var.set("Preview: Display capture active")
                
        except Exception as e:
            self.preview_status_var.set(f"Preview Error: {str(e)}")
            
    def _initialize_preview_capture(self):
        """Initialize preview capture based on input type"""
        self._refresh_preview()
        
    def _on_window_resize(self, event):
        """Handle window resize events for preview scaling"""
        # Only handle main window resize, not child widgets
        if event.widget == self.root:
            # Update preview canvas size if needed
            if hasattr(self, 'preview_canvas'):
                # Trigger preview update to recalculate scaling
                if self.preview_play_var.get():
                    self.root.after(100, self._update_preview)
        
    def _get_selected_track_objects(self):
        """Get list of selected objects to track"""
        if not self.track_specific_objects_var.get():
            return []
        return [obj_name for obj_name, var in self.object_checkboxes.items() if var.get()]
        
    def _start_detection(self):
        """Start the detection process"""
        # If already running, stop first
        if self.processing_thread and self.processing_thread.is_alive():
            self._stop_detection()
            return
            
        # Validate inputs based on input type
        input_type = self.input_type_var.get()
        
        if input_type == "video_file":
            if not os.path.exists(self.video_path_var.get()):
                messagebox.showerror("Error", "Video file not found!")
                return
        elif input_type == "image_folder":
            if not os.path.exists(self.image_folder_var.get()):
                messagebox.showerror("Error", "Image folder not found!")
                return
            # Check if folder contains images
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
            image_files = []
            if self.recursive_subfolders_var.get():
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(self.image_folder_var.get(), '**', ext), recursive=True))
            else:
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(self.image_folder_var.get(), ext)))
            if not image_files:
                messagebox.showerror("Error", "No image files found in the selected folder!")
                return
        elif input_type == "display_capture":
            if not self.display_var.get().strip():
                messagebox.showerror("Error", "Please specify display source!")
                return
        elif input_type == "rtmp_stream":
            if not self.rtmp_url_var.get().strip():
                messagebox.showerror("Error", "Please specify RTMP URL!")
                return
            
        if not os.path.exists(self.model_path_var.get()):
            messagebox.showerror("Error", "Model file not found!")
            return
            
        try:
            float(self.detection_fps_var.get())
            float(self.confidence_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid FPS or confidence values!")
            return
            
        # Check if at least one class is selected
        if not self.selected_classes:
            messagebox.showerror("Error", "Please select at least one class to detect!")
            return
            
        # Create output directory
        os.makedirs(self.output_dir_var.get(), exist_ok=True)
        
        # Update GUI
        self.start_button.config(text="Stop Detection")
        self.progress_var.set("Starting detection...")
        self._log_status("Starting detection process...", "info")
        
        # Start processing thread
        self.stop_event.clear()
        self.processing_thread = Thread(target=self._run_detection_with_classes)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def _stop_detection(self):
        """Stop the detection process"""
        self._log_status("Stopping detection process...", "info")
        
        # Set stop event
        self.stop_event.set()
        
        # Wait for thread to finish with timeout
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=3.0)
            
            # Force kill if still alive
            if self.processing_thread.is_alive():
                self._log_status("Force stopping detection...", "error")
                
        # Reset video player
        if self.player:
            try:
                self.player.stop_video()
            except:
                pass
            self.player = None
            
        # Reset processing thread
        self.processing_thread = None
        
        # Reset GUI
        self.start_button.config(text="Start Detection")
        self.progress_var.set("Ready")
        self._log_status("Detection stopped. Ready to start again.", "success")
        
    def _run_detection_with_classes(self):
        """Run detection with selected classes"""
        try:
            # Create video player with current settings
            video_source = self.video_path_var.get()
            if self.input_type_var.get() == "image_folder":
                video_source = self.image_folder_var.get()
            elif self.input_type_var.get() == "display_capture":
                video_source = self.display_var.get()
            elif self.input_type_var.get() == "rtmp_stream":
                video_source = self.rtmp_url_var.get()
                
            self.player = VideoPlayer(
                video_source=video_source,
                input_type=self.input_type_var.get(),
                model_path=self.model_path_var.get(),
                confidence_threshold=float(self.confidence_var.get()),
                output_dir=self.output_dir_var.get(),
                detection_fps=float(self.detection_fps_var.get()),
                filter_classes=list(self.selected_classes),
                crop_mode=self.crop_mode_var.get(),
                save_cropped=self.save_cropped_var.get(),
                save_frames=self.save_frames_var.get(),
                input_scale_factor=self.input_scale_var.get(),
                display_scale_factor=self.display_scale_var.get(),
                status_callback=self._log_status,
                stop_event=self.stop_event,
                performance_callback=self._update_performance,
                display_source=self.display_var.get(),
                rtmp_url=self.rtmp_url_var.get(),
                track_specific_objects=self.track_specific_objects_var.get(),
                track_object_classes=self._get_selected_track_objects(),
                preprocess_scale=self.preprocess_scale_var.get(),
                postprocess_scale=self.postprocess_scale_var.get(),
                custom_preprocess_dims=(int(self.custom_width_var.get()), int(self.custom_height_var.get())),
                custom_postprocess_dims=(int(self.custom_post_width_var.get()), int(self.custom_post_height_var.get())),
                use_gpu_scaling=self.gpu_scaling_var.get(),
                maintain_aspect_ratio=self.maintain_aspect_var.get(),
                recursive_subfolders=self.recursive_subfolders_var.get()
            )
            
            # Run detection
            self.player.run()
            
        except Exception as e:
            self._log_status(f"Error during detection: {str(e)}", "error")
            # Schedule GUI update in main thread
            self.root.after(0, self._stop_detection)
            
    def _update_performance(self, fps, inference_time):
        """Update performance display"""
        self.root.after(0, lambda: self.fps_label.config(text=f"Current FPS: {fps:.1f}"))
        self.root.after(0, lambda: self.inference_label.config(text=f"Avg Inference: {inference_time:.1f} ms"))
        
        # Store performance data
        self.performance_data['fps_history'].append(fps)
        self.performance_data['inference_times'].append(inference_time)
        
        # Keep only last 100 entries
        if len(self.performance_data['fps_history']) > 100:
            self.performance_data['fps_history'].pop(0)
            self.performance_data['inference_times'].pop(0)
            
    def _on_postprocess_scale_change(self, event=None):
        """Handle postprocess scale selection change"""
        scale_type = self.postprocess_scale_var.get()
        if scale_type == "custom":
            self.custom_postprocess_frame.grid()
        else:
            self.custom_postprocess_frame.grid_remove()
            
        # Also handle preprocessing scale if it's set to custom
        preprocess_scale = self.preprocess_scale_var.get()
        if preprocess_scale == "custom":
            self.custom_preprocess_frame.grid()
        else:
            self.custom_preprocess_frame.grid_remove()
            
    def _on_preprocess_scale_change(self, event=None):
        """Handle preprocess scale selection change"""
        scale_type = self.preprocess_scale_var.get()
        if scale_type == "custom":
            self.custom_preprocess_frame.grid()
        else:
            self.custom_preprocess_frame.grid_remove()
            
    def _cleanup_gui(self):
        """Clean up GUI resources to prevent segmentation faults"""
        try:
            # Stop preview if running
            if hasattr(self, 'preview_play_var'):
                self.preview_play_var.set(False)
                
            # Close preview capture
            if hasattr(self, 'preview_cap') and self.preview_cap:
                try:
                    self.preview_cap.release()
                    self.preview_cap = None
                except:
                    pass
                    
            # Stop any running detection
            if self.processing_thread and self.processing_thread.is_alive():
                self.stop_event.set()
                self.processing_thread.join(timeout=2.0)
                
            # Clean up video player
            if self.player:
                try:
                    self.player.is_playing = False
                    self.player.is_paused = False
                except:
                    pass
                    
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"GUI cleanup error: {e}")
            
        # Ensure OpenCV windows are closed
        try:
            cv2.destroyAllWindows()
        except:
            pass
            
    def run(self):
        """Start the GUI main loop"""
        try:
            self.root.mainloop()
        finally:
            self._cleanup_gui()
            
class VideoPlayer:
    def __init__(self, video_source, input_type, model_path, confidence_threshold=0.3, 
                 output_dir="./output", detection_fps=1.0, filter_classes=None, 
                 crop_mode="square", save_cropped=True, save_frames=True,
                 input_scale_factor=0.5, display_scale_factor=0.5,
                 status_callback=None, stop_event=None, performance_callback=None,
                 display_source=":0.0", rtmp_url="rtmp://localhost/live/stream",
                 track_specific_objects=False, track_object_classes=None,
                 preprocess_scale="auto", postprocess_scale="auto",
                 custom_preprocess_dims=(512, 512), custom_postprocess_dims=(512, 512),
                 use_gpu_scaling=True, maintain_aspect_ratio=True,
                 recursive_subfolders=False):
        self.video_source = video_source
        self.input_type = input_type
        self.display_source = display_source
        self.rtmp_url = rtmp_url
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.output_dir = output_dir
        self.detection_fps = detection_fps
        self.filter_classes = filter_classes
        self.crop_mode = crop_mode
        self.save_cropped = save_cropped
        self.save_frames = save_frames
        self.input_scale_factor = input_scale_factor
        self.display_scale_factor = display_scale_factor
        self.status_callback = status_callback
        self.stop_event = stop_event
        self.performance_callback = performance_callback
        
        # New scaling parameters
        self.preprocess_scale = preprocess_scale
        self.postprocess_scale = postprocess_scale
        self.custom_preprocess_dims = custom_preprocess_dims
        self.custom_postprocess_dims = custom_postprocess_dims
        self.use_gpu_scaling = use_gpu_scaling
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.recursive_subfolders = recursive_subfolders
        
        # Image folder processing
        self.image_files = []
        self.current_image_index = 0
        self.image_processing_mode = input_type == "image_folder"
        
        # Object tracking
        self.track_specific_objects = track_specific_objects
        self.track_object_classes = track_object_classes or []
        self.tracked_objects = {}  # Store object IDs for tracking
        
        # Video control variables
        self.is_playing = True
        self.is_paused = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0
        self.seek_position = 0
        
        # Performance optimization
        self.input_interpolation = cv2.INTER_AREA
        self.display_interpolation = cv2.INTER_LINEAR
        
        # Output directories
        self.cropped_dir = os.path.join(output_dir, "cropped")
        self.full_dir = os.path.join(output_dir, "full")
        self._create_output_directories()
        
        # Threading and queues
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.lock = Lock()
        
        # Initialize components
        self._initialize_video()
        self._initialize_model()
        self._initialize_tracker()
        
        # Performance metrics
        self.frame_times = []
        self.fps_update_interval = 1.0
        self.last_fps_update = time.time()
        self.current_fps = 0
        self.avg_inference_time = 0
        
        # Detection statistics
        self.detection_count = 0
        self.saved_crops = 0
        self.saved_frames = 0
        
        # Detection rate control
        self.last_detection_time = 0
        
        # Parse filter classes
        if filter_classes:
            self.filter_classes_list = [name.strip() for name in filter_classes]
        else:
            self.filter_classes_list = []
            
        if self.status_callback:
            self.status_callback(f"Initialized with {len(self.filter_classes_list)} filter classes", "info")
        
    def _create_output_directories(self):
        """Create output directories for saving"""
        os.makedirs(self.cropped_dir, exist_ok=True)
        os.makedirs(self.full_dir, exist_ok=True)
        if self.status_callback:
            self.status_callback(f"Output directories: {self.cropped_dir}, {self.full_dir}", "info")
        
    def _initialize_video(self):
        """Initialize video capture based on input type"""
        if self.input_type == "video_file":
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                raise ValueError(f"Error: Could not open video file {self.video_source}")
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if self.status_callback:
                self.status_callback(f"Video: {self.frame_width}x{self.frame_height} @ {self.fps:.2f} FPS, {self.total_frames} frames", "info")
                
        elif self.input_type == "display_capture":
            if not self._initialize_display_capture():
                raise ValueError(f"Error: Could not initialize display capture for {self.display_source}")
            
            # For display capture, we don't have total frames
            self.total_frames = float('inf')  # Infinite frames
            self.fps = 30.0  # Default display refresh rate
            
            # Get frame dimensions from the first capture
            test_frame = self._capture_display_frame()
            if test_frame is not None:
                self.frame_width = test_frame.shape[1]
                self.frame_height = test_frame.shape[0]
                if self.status_callback:
                    self.status_callback(f"Display Capture: {self.frame_width}x{self.frame_height} @ {self.fps:.2f} FPS", "success")
            else:
                raise ValueError("Failed to capture initial display frame")
                
        elif self.input_type == "rtmp_stream":
            self.cap = cv2.VideoCapture(self.rtmp_url)
            if not self.cap.isOpened():
                raise ValueError(f"Error: Could not connect to RTMP stream {self.rtmp_url}")
            
            # For RTMP streams, we don't have total frames
            self.total_frames = float('inf')  # Infinite frames
            self.fps = 30.0  # Default stream FPS
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if self.status_callback:
                self.status_callback(f"RTMP Stream: {self.frame_width}x{self.frame_height} @ {self.fps:.2f} FPS", "info")
                
        elif self.input_type == "image_folder":
            if not self._initialize_image_folder():
                raise ValueError(f"Error: Could not initialize image folder {self.video_source}")
            
            # For image folders, we don't have total frames in the traditional sense
            self.total_frames = len(self.image_files)
            self.fps = 1.0  # Process 1 image per second by default
            self.frame_width = 0  # Will be set from first image
            self.frame_height = 0
            
            if self.status_callback:
                self.status_callback(f"Image Folder: {len(self.image_files)} images found", "success")
                
        else:
            raise ValueError(f"Unknown input type: {self.input_type}")
            
    def _initialize_display_capture(self):
        """Initialize display capture using mss library for reliable cross-platform capture"""
        try:
            # Initialize mss for screen capture
            self.mss_instance = mss.mss()
            
            # Get available monitors
            monitors = self.mss_instance.monitors
            
            if self.status_callback:
                self.status_callback(f"Available monitors: {len(monitors)}", "info")
                for i, monitor in enumerate(monitors):
                    if i == 0:  # Skip the "all monitors" entry
                        continue
                    self.status_callback(f"Monitor {i}: {monitor['width']}x{monitor['height']} at ({monitor['left']}, {monitor['top']})", "info")
            
            # Parse display source to determine which monitor to capture
            monitor_index = self._parse_display_source(self.display_source)
            
            if monitor_index is not None and monitor_index < len(monitors):
                self.target_monitor = monitor_index
                if self.status_callback:
                    self.status_callback(f"Selected monitor {monitor_index} for capture", "success")
                return True  # Successfully initialized
            else:
                # Default to primary monitor (usually index 1)
                self.target_monitor = 1
                if self.status_callback:
                    self.status_callback(f"Using default monitor {self.target_monitor}", "info")
                return True
                
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Failed to initialize mss display capture: {e}", "error")
            return False
            
    def _parse_display_source(self, display_source):
        """Parse display source string to determine monitor index"""
        try:
            # Handle different display source formats
            if display_source.startswith(":0."):
                # X11 format :0.0, :0.1, etc.
                monitor_num = int(display_source.split(".")[1])
                return monitor_num + 1  # mss uses 1-based indexing for monitors
            elif display_source.isdigit():
                # Direct monitor number
                return int(display_source)
            elif "DP-" in display_source or "HDMI-" in display_source:
                # Monitor name format - try to match with available monitors
                # For now, return None to use default
                return None
            else:
                return None
        except:
            return None
            
    def _capture_display_frame(self):
        """Capture a frame from the selected display using mss with improved quality"""
        try:
            if hasattr(self, 'mss_instance') and hasattr(self, 'target_monitor'):
                # Capture the selected monitor with optimized settings
                monitor = self.mss_instance.monitors[self.target_monitor]
                
                # Use optimized capture settings to reduce artifacts
                screenshot = self.mss_instance.grab(monitor)
                
                # Convert to numpy array (BGR format for OpenCV)
                frame = np.array(screenshot)
                
                # Convert from BGRA to BGR if needed
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Apply slight Gaussian blur to reduce capture artifacts
                frame = cv2.GaussianBlur(frame, (3, 3), 0.5)
                
                # Apply bilateral filter to preserve edges while reducing noise
                frame = cv2.bilateralFilter(frame, 5, 50, 50)
                
                return frame
            else:
                return None
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Display capture error: {e}", "error")
            return None
        
    def _initialize_model(self):
        """Initialize YOLO model with device optimization and get class names"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.status_callback:
            self.status_callback(f"Using device: {self.device}", "info")
        
        # Load model with optimizations
        self.model = YOLO(self.model_path)
        if self.device.type == 'cuda':
            self.model.to(self.device)
            # Enable all GPU optimizations for RTX 2070S
            try:
                # Enable TensorRT for better performance
                self.model.fuse()
                # Enable half precision for faster inference
                if hasattr(self.model, 'half'):
                    self.model.half()
                # Enable CUDA optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                # Set CUDA memory allocation strategy
                torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
                # Enable memory efficient attention if available
                if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                    torch.backends.cuda.enable_flash_sdp(True)
                if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                if hasattr(torch.backends.cuda, 'enable_math_sdp'):
                    torch.backends.cuda.enable_math_sdp(True)
                if self.status_callback:
                    self.status_callback("GPU optimizations enabled: TensorRT, FP16, cuDNN benchmark, Flash Attention", "success")
            except Exception as e:
                if self.status_callback:
                    self.status_callback(f"Some GPU optimizations failed: {e}", "error")
        
        # Get class names from the model dynamically
        try:
            if hasattr(self.model, 'names') and self.model.names:
                self.CLASS_NAMES = list(self.model.names.values())
                if self.status_callback:
                    self.status_callback(f"Loaded {len(self.CLASS_NAMES)} class names from model", "success")
            else:
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                    self.CLASS_NAMES = list(self.model.model.names.values())
                    if self.status_callback:
                        self.status_callback(f"Loaded {len(self.CLASS_NAMES)} class names from model config", "success")
                else:
                    self.CLASS_NAMES = [f"Class_{i}" for i in range(1000)]
                    if self.status_callback:
                        self.status_callback("Warning: Using generic class names", "error")
                        
        except Exception as e:
            self.CLASS_NAMES = [f"Class_{i}" for i in range(1000)]
            if self.status_callback:
                self.status_callback(f"Error loading class names: {e}", "error")
        
    def _initialize_tracker(self):
        """Initialize SORT tracker"""
        self.tracker = sort()
        
    def _should_process_detection(self):
        """Check if we should process detection based on FPS limit"""
        current_time = time.time()
        if current_time - self.last_detection_time >= (1.0 / self.detection_fps):
            self.last_detection_time = current_time
            return True
        return False
        
    def _crop_and_save_object(self, frame, bbox, class_name, track_id):
        """Crop detected object and save with smart expansion and actual cropping (no padding)"""
        if not self.save_cropped:
            return
            
        xmin, ymin, xmax, ymax = bbox
        frame_h, frame_w = frame.shape[:2]
        
        # Calculate object dimensions
        obj_width = xmax - xmin
        obj_height = ymax - ymin
        
        # Determine target dimensions based on mode
        if self.crop_mode == "square":
            target_width = target_height = 512
        elif self.crop_mode == "portrait":
            target_width, target_height = 512, 768
        else:  # landscape
            target_width, target_height = 768, 512
            
        # Calculate crop area to center the object
        expansion_factor = 1.5  # Expand by 50%
        expanded_size = int(max(obj_width, obj_height) * expansion_factor)
        
        # Calculate crop coordinates (centered on object)
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        
        crop_x1 = max(0, center_x - expanded_size // 2)
        crop_y1 = max(0, center_y - expanded_size // 2)
        crop_x2 = min(frame_w, crop_x1 + expanded_size)
        crop_y2 = min(frame_h, crop_y1 + expanded_size)
        
        # Adjust if we hit frame boundaries
        if crop_x2 - crop_x1 < expanded_size:
            if crop_x1 == 0:
                crop_x2 = min(frame_w, expanded_size)
            else:
                crop_x1 = max(0, frame_w - expanded_size)
        if crop_y2 - crop_y1 < expanded_size:
            if crop_y1 == 0:
                crop_y2 = min(frame_h, expanded_size)
            else:
                crop_y1 = max(0, frame_h - expanded_size)
                
        # Crop the object
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_h, crop_w = cropped.shape[:2]
        
        # Calculate aspect ratios
        crop_aspect = crop_w / crop_h
        target_aspect = target_width / target_height
        
        # Crop to exact target dimensions while maintaining aspect ratio
        if crop_aspect > target_aspect:
            # Crop is wider than target - crop horizontally to match target aspect
            new_height = crop_h
            new_width = int(crop_h * target_aspect)
            # Center the horizontal crop
            x_start = (crop_w - new_width) // 2
            x_end = x_start + new_width
            cropped = cropped[:, x_start:x_end]
        else:
            # Crop is taller than target - crop vertically to match target aspect
            new_width = crop_w
            new_height = int(crop_w / target_aspect)
            # Center the vertical crop
            y_start = (crop_h - new_height) // 2
            y_end = y_start + new_height
            cropped = cropped[y_start:y_end, :]
        
        # Resize to exact target dimensions
        final_image = cv2.resize(cropped, (target_width, target_height))
        
        # Save cropped image
        timestamp = int(time.time() * 1000)
        filename = f"{self.cropped_dir}/{class_name}_{track_id}_{timestamp}.jpg"
        cv2.imwrite(filename, final_image)
        self.saved_crops += 1
        
        if self.status_callback:
            self.status_callback(f"Saved crop: {filename}", "success")
        
    def _save_frame_if_detected(self, frame, detected_classes):
        """Save full frame if any of the filtered classes are detected"""
        if not self.save_frames:
            return
            
        # Check if any filtered class was detected
        should_save = False
        if self.filter_classes_list:
            # If we have filter classes, check if any detected class matches
            should_save = any(cls in detected_classes for cls in self.filter_classes_list)
        else:
            # If no filter classes, save frame if any class was detected
            should_save = len(detected_classes) > 0
        
        if should_save:
            timestamp = int(time.time() * 1000)
            filename = f"{self.full_dir}/frame_{self.current_frame}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            self.saved_frames += 1
            
            if self.status_callback:
                self.status_callback(f"Saved frame: {filename}", "success")
        
    def _update_fps(self, frame_time):
        """Update FPS calculation using frame processing times"""
        current_time = time.time()
        self.frame_times.append(frame_time)
        
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        if current_time - self.last_fps_update >= self.fps_update_interval:
            if self.frame_times:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                self.last_fps_update = current_time
                self.frame_times.clear()
                
                # Call performance callback
                if self.performance_callback:
                    self.performance_callback(self.current_fps, self.avg_inference_time * 1000)
        
    def _process_frame(self, frame):
        """Process a single frame with YOLO detection and tracking"""
        # Check if we should stop
        if self.stop_event and self.stop_event.is_set():
            return frame
            
        start_time = time.time()
        
        # Skip processing if paused or detection rate limit reached
        if self.is_paused or not self._should_process_detection():
            return frame
            
        # Ensure frame is a proper numpy array
        if not isinstance(frame, np.ndarray):
            if self.status_callback:
                self.status_callback("Warning: Invalid frame format, skipping processing", "error")
            return frame
            
        # Get original frame dimensions
        original_h, original_w = frame.shape[:2]
        
        # Input scaling for YOLO processing
        input_w = int(original_w * self.input_scale_factor)
        input_h = int(original_h * self.input_scale_factor)
        
        # Convert frame to GPU tensor for processing
        if self.device.type == 'cuda':
            try:
                # Move frame to GPU and convert to tensor
                frame_tensor = torch.from_numpy(frame).to(self.device, dtype=torch.float32, non_blocking=True)
                
                if self.input_scale_factor != 1.0:
                    # GPU-based scaling with high-quality interpolation
                    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # BHWC -> BCHW
                    frame_for_yolo = torch.nn.functional.interpolate(
                        frame_tensor, 
                        size=(input_h, input_w), 
                        mode='bicubic',  # High-quality interpolation
                        align_corners=False
                    )
                    frame_for_yolo = frame_for_yolo.squeeze(0).permute(1, 2, 0)  # BCHW -> HWC
                    frame_for_yolo = frame_for_yolo.cpu().numpy().astype(np.uint8)
                else:
                    frame_for_yolo = frame
                    input_w, input_h = original_w, original_h
            except Exception as e:
                if self.status_callback:
                    self.status_callback(f"GPU scaling failed, falling back to CPU: {e}", "error")
                # Fallback to CPU scaling
                if self.input_scale_factor != 1.0:
                    frame_for_yolo = cv2.resize(frame, (input_w, input_h), 
                                              interpolation=self.input_interpolation)
                else:
                    frame_for_yolo = frame
                    input_w, input_h = original_w, original_h
        else:
            # CPU fallback
            if self.input_scale_factor != 1.0:
                frame_for_yolo = cv2.resize(frame, (input_w, input_h), 
                                          interpolation=self.input_interpolation)
            else:
                frame_for_yolo = frame
                input_w, input_h = original_w, original_h
            
        # YOLO inference
        try:
            results = self.model(frame_for_yolo, stream=True, device=self.device, verbose=False)
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"YOLO inference failed: {e}", "error")
            return frame
        
        detected_classes = []
        filtered_tracks = []  # Store only filtered tracks for drawing
        
        for res in results:
            # Filter detections by confidence
            filtered_indices = (res.boxes.conf > self.confidence_threshold).nonzero(as_tuple=True)[0]
            
            if filtered_indices.numel() == 0:
                tracks = self.tracker.update(np.empty((0, 5)), class_ids=np.empty((0,)))
                tracks = tracks.astype(int)
                continue
                
            # Get filtered detections
            gpu_filtered_boxes_scaled = res.boxes.xyxy[filtered_indices]
            gpu_filtered_scores = res.boxes.conf[filtered_indices]
            gpu_filtered_classes = res.boxes.cls[filtered_indices]
            
            # Scale bounding boxes back to original frame size
            scale_x_to_original = original_w / input_w
            scale_y_to_original = original_h / input_h
            scale_tensor_to_original = torch.tensor([scale_x_to_original, scale_y_to_original, 
                                                   scale_x_to_original, scale_y_to_original], 
                                                  device=self.device, dtype=torch.float32)
            
            gpu_filtered_boxes_original_scale = gpu_filtered_boxes_scaled * scale_tensor_to_original
            
            # Combine boxes and scores
            gpu_boxes_with_scores = torch.cat((gpu_filtered_boxes_original_scale, 
                                             gpu_filtered_scores.unsqueeze(1)), dim=1)
            
            # Move to CPU for tracker
            boxes_for_tracker = gpu_boxes_with_scores.cpu().numpy()
            class_ids_for_tracker = gpu_filtered_classes.cpu().numpy()
            
            # Update tracker
            tracks = self.tracker.update(boxes_for_tracker, class_ids=class_ids_for_tracker)
            tracks = tracks.astype(int)
            
            # Process tracking results and filter by selected classes
            for xmin, ymin, xmax, ymax, track_id, class_id in tracks:
                xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(original_w, xmax), min(original_h, ymax)
                
                # Handle class names dynamically
                if 0 <= class_id < len(self.CLASS_NAMES):
                    class_name = self.CLASS_NAMES[class_id]
                else:
                    class_name = f"Class_{class_id}" if class_id >= 0 else "Unknown"
                    
                # Check if this class should be processed (filtering)
                if not self.filter_classes_list or class_name in self.filter_classes_list:
                    # Object tracking logic
                    if self.track_specific_objects and self.track_object_classes:
                        # Only track specific objects if tracking is enabled
                        if class_name in self.track_object_classes:
                            # Check if this is a new object or existing tracked object
                            if track_id not in self.tracked_objects:
                                self.tracked_objects[track_id] = {
                                    'class': class_name,
                                    'first_seen': time.time(),
                                    'last_seen': time.time(),
                                    'frames_count': 1
                                }
                                if self.status_callback:
                                    self.status_callback(f"Started tracking {class_name} (ID: {track_id})", "info")
                            else:
                                # Update existing tracked object
                                self.tracked_objects[track_id]['last_seen'] = time.time()
                                self.tracked_objects[track_id]['frames_count'] += 1
                    else:
                        # Normal detection without specific tracking
                        pass
                    
                    detected_classes.append(class_name)
                    self.detection_count += 1
                    
                    # Store filtered track for drawing
                    filtered_tracks.append((xmin, ymin, xmax, ymax, track_id, class_name))
                    
                    # Save cropped object if enabled
                    if self.save_cropped:
                        try:
                            self._crop_and_save_object(frame, (xmin, ymin, xmax, ymax), class_name, track_id)
                        except Exception as e:
                            if self.status_callback:
                                self.status_callback(f"Failed to save crop: {e}", "error")
        
        # Draw ONLY the filtered bounding boxes (no other classes)
        for xmin, ymin, xmax, ymax, track_id, class_name in filtered_tracks:
            try:
                # Draw bounding box and label for selected classes only
                display_text = f"{class_name}: {track_id}"
                cv2.putText(frame, display_text, (xmin, ymin - 10), 
                           cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            except Exception as e:
                if self.status_callback:
                    self.status_callback(f"Failed to draw bounding box: {e}", "error")
        
        # Save frame if any filtered classes were detected
        if detected_classes:
            try:
                self._save_frame_if_detected(frame, detected_classes)
            except Exception as e:
                if self.status_callback:
                    self.status_callback(f"Failed to save frame: {e}", "error")
        
        # Calculate inference time
        inference_time = time.time() - start_time
        self.avg_inference_time = 0.9 * self.avg_inference_time + 0.1 * inference_time
        
        # Update FPS calculation
        self._update_fps(inference_time)
        
        return frame
        
    def _draw_ui_overlay(self, frame):
        """Draw UI overlay with controls and information"""
        # Ensure frame is a proper numpy array with correct layout
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame, dtype=np.uint8)
        
        # Create a copy for overlay to avoid modifying original
        overlay = frame.copy()
        
        # Ensure overlay has the correct shape and type
        if len(overlay.shape) != 3 or overlay.shape[2] != 3:
            # Convert to proper 3-channel format if needed
            if len(overlay.shape) == 2:
                overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
            elif overlay.shape[2] == 4:
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2BGR)
        
        # Draw semi-transparent overlay
        cv2.rectangle(overlay, (0, 0), (500, 400), (0, 0, 0), -1)
        
        # Use proper alpha blending
        alpha = 0.7
        beta = 0.3
        gamma = 0
        
        # Ensure both arrays have the same shape and type
        if frame.shape != overlay.shape:
            frame = cv2.resize(frame, (overlay.shape[1], overlay.shape[0]))
        
        # Convert to float32 for blending
        frame_float = frame.astype(np.float32)
        overlay_float = overlay.astype(np.float32)
        
        # Perform blending
        blended = cv2.addWeighted(frame_float, beta, overlay_float, alpha, gamma)
        
        # Convert back to uint8
        result = blended.astype(np.uint8)
        
        # Draw information
        y_offset = 30
        if self.total_frames != float('inf'):
            cv2.putText(result, f"Frame: {self.current_frame}/{self.total_frames}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(result, f"Frame: {self.current_frame} (LIVE)", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # Show current timestamp
        current_time = self.current_frame / self.fps if self.fps > 0 else 0
        minutes = int(current_time // 60)
        seconds = int(current_time % 60)
        cv2.putText(result, f"Time: {minutes:02d}:{seconds:02d}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(result, f"FPS: {self.current_fps:.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(result, f"Inference: {self.avg_inference_time*1000:.1f}ms", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(result, f"Detections: {self.detection_count}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(result, f"Saved crops: {self.saved_crops}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(result, f"Saved frames: {self.saved_frames}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # Display tracking information if enabled
        if self.track_specific_objects and self.tracked_objects:
            cv2.putText(result, f"Tracking {len(self.tracked_objects)} objects:", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 20
            
            # Show first few tracked objects
            for i, (track_id, obj_info) in enumerate(list(self.tracked_objects.items())[:3]):
                obj_text = f"{obj_info['class']} (ID: {track_id}) - {obj_info['frames_count']} frames"
                cv2.putText(result, obj_text, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                y_offset += 15
                
            if len(self.tracked_objects) > 3:
                cv2.putText(result, f"... and {len(self.tracked_objects) - 3} more", 
                           (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                y_offset += 15
        y_offset += 10
        
        # Draw controls
        cv2.putText(result, "SPACE: Pause/Play | Q: Quit | S: Stop", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        cv2.putText(result, "Left/Right: Seek 10s | Up/Down: Seek 1s", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        # Draw seekbar (only for video files with finite frames)
        if self.total_frames != float('inf'):
            seekbar_width = 400
            seekbar_height = 20
            seekbar_x = 10
            seekbar_y = y_offset + 10
            
            # Background bar
            cv2.rectangle(result, (seekbar_x, seekbar_y), 
                         (seekbar_x + seekbar_width, seekbar_y + seekbar_height), 
                         (100, 100, 100), -1)
            
            # Progress bar
            progress_fill = int((self.current_frame / self.total_frames) * seekbar_width)
            cv2.rectangle(result, (seekbar_x, seekbar_y), 
                         (seekbar_x + progress_fill, seekbar_y + seekbar_height), 
                         (0, 255, 0), -1)
            
            # Seekbar text
            cv2.putText(result, "SEEKBAR - Click to seek", 
                       (seekbar_x, seekbar_y + seekbar_height + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            # For live sources, show status instead of seekbar
            cv2.putText(result, "LIVE SOURCE - No seeking available", 
                       (10, y_offset + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
        
    def run(self):
        """Main video processing loop"""
        cv2.namedWindow("YOLO Detection v3", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Detection v3", 1280, 720)
        
        if self.status_callback:
            self.status_callback("Detection window opened. Press SPACE to pause, Q to quit.", "info")
            self.status_callback("Advanced seeking: Click on seekbar or use arrow keys", "info")
        
        # Set mouse callback for seeking
        cv2.setMouseCallback("YOLO Detection v3", self._mouse_callback)
        
        while self.is_playing:
            # Check if we should stop
            if self.stop_event and self.stop_event.is_set():
                break
                
            if self.is_paused:
                # Show paused frame without processing
                if hasattr(self, 'current_display_frame') and self.current_display_frame is not None:
                    cv2.putText(self.current_display_frame, "PAUSED", 
                               (self.frame_width//2 - 100, self.frame_height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    cv2.imshow("YOLO Detection v3", self.current_display_frame)
                key = cv2.waitKey(100) & 0xFF
                if not self._handle_key(key):
                    break
                continue
                
            # Get frame based on input type
            if self.input_type == "display_capture":
                frame = self._capture_display_frame()
                if frame is None:
                    if self.status_callback:
                        self.status_callback("Failed to capture display frame", "error")
                    continue
                ret = True
            elif self.input_type == "image_folder":
                frame, ret = self._get_next_image_frame()
                if frame is None:
                    if ret:
                        # Continue to next image
                        continue
                    else:
                        # End of images
                        if self.status_callback:
                            self.status_callback("All images processed", "info")
                        break
            else:
                ret, frame = self.cap.read()
                
            if not ret or frame is None:
                if self.input_type == "video_file":
                    if self.status_callback:
                        self.status_callback("End of video reached", "info")
                    break
                else:
                    # For live sources, continue trying
                    time.sleep(0.01)  # Small delay to prevent CPU spinning
                    continue
                
            # Process frame (only if not paused and detection rate allows)
            processed_frame = self._process_frame(frame)
            
            # Check if we should stop after processing
            if self.stop_event and self.stop_event.is_set():
                break
                
            # Apply postprocessing scaling if enabled
            if self.postprocess_scale != "original":
                processed_frame = self._apply_postprocessing_scaling(processed_frame)
                
            # Scale for display
            display_w = int(self.frame_width * self.display_scale_factor)
            display_h = int(self.frame_height * self.display_scale_factor)
            
            # GPU-based display scaling for better quality
            if self.device.type == 'cuda' and self.display_scale_factor != 1.0:
                try:
                    # Convert to GPU tensor for scaling
                    frame_tensor = torch.from_numpy(processed_frame).to(self.device, dtype=torch.float32, non_blocking=True)
                    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # BHWC -> BCHW
                    
                    # GPU scaling with high-quality interpolation
                    display_frame_tensor = torch.nn.functional.interpolate(
                        frame_tensor, 
                        size=(display_h, display_w), 
                        mode='bicubic',  # High-quality interpolation
                        align_corners=False
                    )
                    
                    # Convert back to CPU numpy array
                    display_frame = display_frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                except Exception as e:
                    # Fallback to CPU scaling
                    display_frame = cv2.resize(processed_frame, (display_w, display_h), 
                                             interpolation=self.display_interpolation)
            else:
                # CPU fallback or no scaling
                display_frame = cv2.resize(processed_frame, (display_w, display_h), 
                                         interpolation=self.display_interpolation)
            
            # Store current display frame for pause functionality
            self.current_display_frame = display_frame.copy()
            
            # Draw UI overlay
            display_frame = self._draw_ui_overlay(display_frame)
            
            # Show frame
            cv2.imshow("YOLO Detection v3", display_frame)
            
            # Update frame counter
            self.current_frame += 1
            
            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_key(key):
                break
                
        # Cleanup
        try:
            # Stop any ongoing operations first
            self.is_playing = False
            self.is_paused = False
            
            # Close video capture safely
            if hasattr(self, 'cap') and self.cap:
                try:
                    self.cap.release()
                    self.cap = None
                except:
                    pass
                
            # Close mss instance safely
            if hasattr(self, 'mss_instance'):
                try:
                    self.mss_instance.close()
                    self.mss_instance = None
                except:
                    pass
                
            # Close OpenCV windows safely to prevent segmentation faults
            try:
                cv2.destroyAllWindows()
                # Additional cleanup for specific windows
                cv2.destroyWindow("YOLO Detection v3")
            except:
                pass
                
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
                
            # Longer delay to ensure complete cleanup
            time.sleep(0.5)
            
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Cleanup error: {e}", "warning")
            # Force cleanup even if there's an error
            try:
                cv2.destroyAllWindows()
                gc.collect()
            except:
                pass
        
        # Final statistics
        if self.status_callback:
            self.status_callback(f"Detection completed. Total: {self.detection_count}, Crops: {self.saved_crops}, Frames: {self.saved_frames}", "success")
        
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for seeking"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Only handle seeking for video files with finite frames
            if self.total_frames != float('inf'):
                # Calculate seek position based on x coordinate
                window_width = 1280  # Default window width
                seekbar_x = 10  # Seekbar x position
                seekbar_width = 400  # Seekbar width
                
                # Check if click is within seekbar area
                if seekbar_x <= x <= seekbar_x + seekbar_width:
                    seek_percentage = (x - seekbar_x) / seekbar_width
                    target_frame = int(seek_percentage * self.total_frames)
                    self.seek_to_frame(target_frame)
                    
                    if self.status_callback:
                        self.status_callback(f"Seeked to frame {target_frame} ({seek_percentage*100:.1f}%)", "info")
            else:
                # For live sources, show info message
                if self.status_callback:
                    self.status_callback("Seeking not available for live sources", "info")
        
    def _handle_key(self, key):
        """Handle keyboard input"""
        if key == ord('q'):
            return False  # Quit
        elif key == ord(' '):
            self.toggle_pause()
        elif key == ord('s'):
            self.stop_video()
            return False
        elif key == ord('a') or key == 81:  # Left arrow
            if self.total_frames != float('inf'):
                self.seek_to_frame(max(0, self.current_frame - int(self.fps * 10)))
            else:
                if self.status_callback:
                    self.status_callback("Seeking not available for live sources", "info")
        elif key == ord('d') or key == 83:  # Right arrow
            if self.total_frames != float('inf'):
                self.seek_to_frame(min(self.total_frames - 1, self.current_frame + int(self.fps * 10)))
            else:
                if self.status_callback:
                    self.status_callback("Seeking not available for live sources", "info")
        elif key == ord('w') or key == 82:  # Up arrow
            if self.total_frames != float('inf'):
                self.seek_to_frame(max(0, self.current_frame - int(self.fps)))
            else:
                if self.status_callback:
                    self.status_callback("Seeking not available for live sources", "info")
        elif key == ord('s') or key == 84:  # Down arrow
            if self.total_frames != float('inf'):
                self.seek_to_frame(min(self.total_frames - 1, self.current_frame + int(self.fps)))
            else:
                if self.status_callback:
                    self.status_callback("Seeking not available for live sources", "info")
            
        return True
        
    def seek_to_frame(self, frame_number):
        """Seek to a specific frame number"""
        with self.lock:
            # Only allow seeking for video files with finite frames
            if self.total_frames != float('inf') and 0 <= frame_number < self.total_frames:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                self.current_frame = frame_number
                self.seek_position = frame_number / self.total_frames
            elif self.total_frames == float('inf'):
                if self.status_callback:
                    self.status_callback("Seeking not available for live sources", "warning")
                
    def toggle_pause(self):
        """Toggle pause/play state"""
        self.is_paused = not self.is_paused
        
    def stop_video(self):
        """Stop video playback"""
        self.is_playing = False
        # Reset state for potential restart
        self.current_frame = 0
        self.detection_count = 0
        self.saved_crops = 0
        self.saved_frames = 0
        self.frame_times.clear()
        self.avg_inference_time = 0
        self.current_fps = 0
        self.last_detection_time = 0

    def _initialize_image_folder(self):
        """Initialize image folder processing"""
        try:
            folder_path = Path(self.video_source)
            if not folder_path.exists() or not folder_path.is_dir():
                if self.status_callback:
                    self.status_callback(f"Image folder {self.video_source} does not exist", "error")
                return False
            
            # Define supported image extensions
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
            
            # Collect image files
            if self.recursive_subfolders:
                # Recursive search
                for ext in image_extensions:
                    self.image_files.extend(folder_path.glob(f"**/{ext}"))
            else:
                # Non-recursive search
                for ext in image_extensions:
                    self.image_files.extend(folder_path.glob(ext))
            
            # Sort files for consistent processing order
            self.image_files.sort()
            
            if not self.image_files:
                if self.status_callback:
                    self.status_callback(f"No image files found in {self.video_source}", "error")
                return False
            
            # Get dimensions from first image
            first_image = cv2.imread(str(self.image_files[0]))
            if first_image is not None:
                self.frame_height, self.frame_width = first_image.shape[:2]
                if self.status_callback:
                    self.status_callback(f"Image dimensions: {self.frame_width}x{self.frame_height}", "info")
            else:
                if self.status_callback:
                    self.status_callback(f"Failed to read first image: {self.image_files[0]}", "error")
                return False
            
            if self.status_callback:
                self.status_callback(f"Successfully initialized image folder with {len(self.image_files)} images", "success")
            
            return True
            
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Error initializing image folder: {e}", "error")
            return False

    def _get_next_image_frame(self):
        """Get the next image frame for processing"""
        if self.current_image_index >= len(self.image_files):
            return None, False
        
        try:
            image_path = self.image_files[self.current_image_index]
            frame = cv2.imread(str(image_path))
            
            if frame is not None:
                # Update frame dimensions
                self.frame_height, self.frame_width = frame.shape[:2]
                
                # Apply preprocessing scaling if enabled
                if self.preprocess_scale != "none":
                    frame = self._apply_preprocessing_scaling(frame)
                
                self.current_image_index += 1
                return frame, True
            else:
                if self.status_callback:
                    self.status_callback(f"Failed to read image: {image_path}", "error")
                self.current_image_index += 1
                return None, True  # Continue to next image
                
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Error reading image {self.current_image_index}: {e}", "error")
            self.current_image_index += 1
            return None, True  # Continue to next image
    
    def _apply_preprocessing_scaling(self, frame):
        """Apply preprocessing scaling to the frame"""
        try:
            # Parse scaling parameters
            if self.preprocess_scale == "custom":
                target_width, target_height = self.custom_preprocess_dims
            else:
                target_width, target_height = parse_scale_string(self.preprocess_scale)
            
            if target_width is None or target_height is None:
                return frame  # No scaling
            
            # Calculate optimal dimensions
            optimal_width, optimal_height = calculate_optimal_dimensions(
                frame.shape[1], frame.shape[0], target_width, target_height, self.maintain_aspect_ratio
            )
            
            # Apply scaling
            if self.use_gpu_scaling:
                return gpu_scale_image(frame, optimal_width, optimal_height, self.device, 'lanczos')
            else:
                return cv2.resize(frame, (optimal_width, optimal_height), interpolation=cv2.INTER_LANCZOS4)
                
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Preprocessing scaling failed: {e}", "error")
            return frame
    
    def _apply_postprocessing_scaling(self, frame):
        """Apply postprocessing scaling to the frame"""
        try:
            # Parse scaling parameters
            if self.postprocess_scale == "custom":
                target_width, target_height = self.custom_postprocess_dims
            elif self.postprocess_scale == "original":
                return frame  # No scaling
            else:
                target_width, target_height = parse_scale_string(self.postprocess_scale)
            
            if target_width is None or target_height is None:
                return frame  # No scaling
            
            # Calculate optimal dimensions
            optimal_width, optimal_height = calculate_optimal_dimensions(
                frame.shape[1], frame.shape[0], target_width, target_height, self.maintain_aspect_ratio
            )
            
            # Apply scaling
            if self.use_gpu_scaling:
                return gpu_scale_image(frame, optimal_width, optimal_height, self.device, 'lanczos')
            else:
                return cv2.resize(frame, (optimal_width, optimal_height), interpolation=cv2.INTER_LANCZOS4)
                
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Postprocessing scaling failed: {e}", "error")
            return frame

def main():
    """Main function - launch GUI"""
    try:
        gui = DetectionGUI()
        gui.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
