#!/usr/bin/env python3
"""
OCR Configuration - Adjust GPU and Processing Settings
Edit these values to optimize for your system
"""

# ===========================
# GPU SETTINGS
# ===========================

# Number of parallel workers
# Lower = less memory usage, slower processing
# Higher = more memory usage, faster processing
# For RTX 3050 6GB: 4 is optimal (was causing OOM at 8)
# Try 2 for very large PDFs, 4-6 for normal use
MAX_WORKERS = 4

# GPU Memory Management
# Automatically cleanup GPU memory after each image
GPU_MEMORY_CLEANUP = True

# Use float32 for GPU operations (more accurate, more memory)
# or uint8 (less memory, slightly less accurate)
GPU_DATA_TYPE = "uint8"  # Options: "float32", "uint8"

# ===========================
# OCR SETTINGS
# ===========================

# DPI for PDF conversion (higher = better quality, slower)
# 300 = standard
# 400 = high quality (recommended)
# 600 = very high quality (slow, memory intensive)
DPI_SETTING = 400

# Confidence threshold for word detection
# Words below this confidence are filtered out
# Range: 0-100
CONFIDENCE_THRESHOLD = 40

# Language for OCR
# "ara" = Arabic only
# "ara+eng" = Arabic and English
OCR_LANGUAGE = "ara"

# ===========================
# PREPROCESSING SETTINGS
# ===========================

# Contrast enhancement factor (1.0 = no change)
CONTRAST_ENHANCEMENT = 1.5

# Sharpness enhancement factor (1.0 = no change)
SHARPNESS_ENHANCEMENT = 1.2

# ===========================
# PERFORMANCE TUNING
# ===========================

# Enable parallel processing
PARALLEL_PROCESSING = True

# Enable GPU acceleration (if available)
USE_GPU = True

# Batch processing size (pages per batch)
BATCH_SIZE = 4

# ===========================
# MEMORY SETTINGS
# ===========================

# Max GPU memory to use (in MB)
# RTX 3050 has 6144 MB total
# Leave ~1GB for system/display = 5000 MB safe limit
MAX_GPU_MEMORY = 5000

# Cache cleared after processing
AUTO_CLEAR_GPU_CACHE = True

# ===========================
# LOGGING
# ===========================

# Verbose output
VERBOSE = True

# Log GPU memory usage
LOG_MEMORY = True

# ===========================
# RECOMMENDED CONFIGURATIONS
# ===========================

"""
FOR RTX 3050 6GB:

Fast Processing (12 PDFs with 12+ pages each):
  MAX_WORKERS = 2
  DPI_SETTING = 300
  GPU memory efficient

Balanced Processing (40 PDFs with 10-15 pages each):
  MAX_WORKERS = 4         ← CURRENT (Optimal)
  DPI_SETTING = 400
  Good quality, reasonable speed

High Quality (Fewer PDFs, maximum quality):
  MAX_WORKERS = 2
  DPI_SETTING = 600
  Slower, best results

GPU Memory Optimization:
  GPU_MEMORY_CLEANUP = True        (Enabled)
  GPU_DATA_TYPE = "uint8"          (Memory efficient)
  AUTO_CLEAR_GPU_CACHE = True      (Enabled)
"""

if __name__ == "__main__":
    print("OCR Configuration Settings")
    print("=" * 50)
    print(f"GPU Workers: {MAX_WORKERS}")
    print(f"DPI: {DPI_SETTING}")
    print(f"GPU Memory Cleanup: {GPU_MEMORY_CLEANUP}")
    print(f"GPU Data Type: {GPU_DATA_TYPE}")
    print(f"Parallel Processing: {PARALLEL_PROCESSING}")
