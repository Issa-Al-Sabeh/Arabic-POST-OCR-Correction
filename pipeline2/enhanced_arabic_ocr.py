#!/usr/bin/env python3
import os
import json
from pathlib import Path
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from functools import partial
import time

import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Try to import GPU acceleration libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[GPU] GPU acceleration available with CuPy")
    print(f"   GPU Threads: {cp.cuda.Device().attributes['MaxThreadsPerBlock']}")
except ImportError:
    GPU_AVAILABLE = False
    print("[CPU] GPU acceleration not available, using CPU only")

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
print("[OK] Tesseract configured for Arabic OCR")

# ===========================
# SETTINGS
# ===========================
PDF_INPUT_DIR = r"C:\Users\ali-d\Desktop\al anbaa"  # Folder containing PDF files
JSON_OUTPUT_DIR = "arabic_ocr"  # Output folder inside pipeline2

# Performance settings
MAX_WORKERS = min(3, multiprocessing.cpu_count())  # Reduced to 2 for Tesseract stability
DPI_SETTING = 400  # Lower DPI for faster processing (was 600)
PARALLEL_PROCESSING = True  # Enable parallel processing

# Path to tesseract.exe (change if different)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Note: Output directory will be created dynamically in main()


def optimized_arabic_preprocessing(img):
    """
    Optimized preprocessing for Arabic OCR based on testing results
    Simple approach that preserves Arabic text readability
    """
    # Convert to PIL if needed
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    
    # Enhanced contrast - this gave best results in testing
    enhancer = ImageEnhance.Contrast(img)
    enhanced = enhancer.enhance(1.5)
    
    # Optional: slight sharpening to improve text clarity
    enhancer_sharp = ImageEnhance.Sharpness(enhanced)
    result = enhancer_sharp.enhance(1.2)
    
    return result


def gpu_accelerated_preprocessing(img):
    """
    GPU-accelerated preprocessing for faster image processing
    Memory-optimized version for limited VRAM (6GB RTX 3050)
    """
    if GPU_AVAILABLE:
        try:
            # Convert PIL to numpy for GPU operations
            img_array = np.array(img)
            
            # Use uint8 directly to save memory (no float32 conversion)
            if len(img_array.shape) == 3:
                # For RGB images, use uint8 on GPU to save memory
                img_gpu = cp.asarray(img_array, dtype=cp.uint8)
                
                # Memory-efficient contrast enhancement
                # Convert to float only for calculation
                img_float = img_gpu.astype(cp.float32)
                mean_val = cp.mean(img_float)
                
                # Apply contrast
                contrast_enhanced = cp.clip((img_float - mean_val) * 1.5 + mean_val, 0, 255)
                result_gpu = contrast_enhanced.astype(cp.uint8)
                
                # Free intermediate GPU memory
                del img_float, img_gpu
                cp.get_default_memory_pool().free_all_blocks()
                
            else:
                # For grayscale
                img_gpu = cp.asarray(img_array, dtype=cp.uint8)
                img_float = img_gpu.astype(cp.float32)
                mean_val = cp.mean(img_float)
                contrast_enhanced = cp.clip((img_float - mean_val) * 1.5 + mean_val, 0, 255)
                result_gpu = contrast_enhanced.astype(cp.uint8)
                
                del img_float, img_gpu
                cp.get_default_memory_pool().free_all_blocks()
            
            # Transfer back to CPU
            result_array = cp.asnumpy(result_gpu)
            
            # Free GPU memory
            del result_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            # Convert back to PIL
            if len(result_array.shape) == 3:
                processed_img = Image.fromarray(result_array, mode='RGB')
            else:
                processed_img = Image.fromarray(result_array, mode='L')
            
            # Apply sharpening using PIL
            enhancer = ImageEnhance.Sharpness(processed_img)
            processed_img = enhancer.enhance(1.2)
            
            return processed_img
            
        except Exception as e:
            print(f"  ⚠️ GPU processing failed, falling back to CPU: {e}")
            # Clear GPU memory on failure
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
            return optimized_arabic_preprocessing(img)
    else:
        return optimized_arabic_preprocessing(img)


def fast_preprocessing(img):
    """
    Faster preprocessing with reduced operations for speed
    """
    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Simple adaptive threshold (faster than bilateral filter)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Minimal morphological operations
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to PIL
    processed_img = Image.fromarray(cleaned)
    
    # Quick contrast enhancement
    enhancer = ImageEnhance.Contrast(processed_img)
    processed_img = enhancer.enhance(1.2)
    
    return processed_img


def advanced_arabic_preprocessing(img):
    """
    Advanced preprocessing specifically for Arabic OCR (original function)
    """
    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive threshold to handle varying lighting
    adaptive_thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological operations to clean up text
    # Use a smaller kernel for Arabic text
    kernel = np.ones((2, 2), np.uint8)
    
    # Opening (erosion followed by dilation) to remove noise
    opened = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Closing (dilation followed by erosion) to close gaps in text
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Convert back to PIL
    processed_img = Image.fromarray(closed)
    
    # Additional PIL enhancements
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(processed_img)
    processed_img = enhancer.enhance(1.5)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(processed_img)
    processed_img = enhancer.enhance(2.0)
    
    return processed_img


def process_page_parallel(args):
    """
    Process a single page in parallel
    """
    page_num, img, use_gpu = args
    
    # Choose preprocessing method - ALWAYS USE GPU if available for maximum performance
    if GPU_AVAILABLE:
        try:
            # print(f"  [GPU] Page {page_num}: Using GPU acceleration")
            processed_img = gpu_accelerated_preprocessing(img)
        except Exception as e:
            print(f"  ⚠️ Page {page_num}: GPU failed, using CPU: {e}")
            processed_img = optimized_arabic_preprocessing(img)
    else:
        # print(f"  [CPU] Page {page_num}: GPU not available, using CPU")
        processed_img = optimized_arabic_preprocessing(img)  # Use optimized preprocessing
    
    # Optimized OCR configuration based on testing results
    # Enhanced contrast approach worked best
    arabic_config = r'--oem 3 --psm 3 -c preserve_interword_spaces=1 -c textord_really_old_xheight=1'
    
    try:
        # Get detailed OCR data with mixed languages for best results
        data = pytesseract.image_to_data(
            processed_img, 
            lang="ara",  # Mixed languages worked better in testing
            output_type=pytesseract.Output.DICT,
            config=arabic_config
        )
        
        # Get plain text
        plain_text = pytesseract.image_to_string(
            processed_img, 
            lang="ara",  # Mixed languages for plain text too
            config=arabic_config
        )
        
        # Process the detailed data with quality filtering
        words = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 20:  # Lower threshold to capture more words
                text_content = data['text'][i].strip()
                if text_content and len(text_content) > 0:
                    word_info = {
                        "text": text_content,
                        "confidence": int(data['conf'][i]),
                        "bbox": {
                            "x": int(data['left'][i]),
                            "y": int(data['top'][i]),
                            "width": int(data['width'][i]),
                            "height": int(data['height'][i])
                        },
                        "level": int(data['level'][i]),
                        "block_num": int(data['block_num'][i]),
                        "line_num": int(data['line_num'][i]),
                        "word_num": int(data['word_num'][i])
                    }
                    words.append(word_info)
        
        # Calculate average confidence
        confidences = [w['confidence'] for w in words]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        page_data = {
            "page_number": page_num,
            "text": plain_text.strip(),
            "word_count": len([w for w in words if w['text'].strip()]),
            "words": words,
            "average_confidence": avg_confidence
        }
        
        return page_data
        
    except Exception as e:
        print(f"  ⚠️ Page {page_num}: OCR error (returning empty): {str(e)[:80]}")
        # Return empty page data instead of None so processing continues
        return {
            "page_number": page_num,
            "text": "",
            "word_count": 0,
            "words": [],
            "average_confidence": 0,
            "error": str(e)[:100]
        }


def ocr_pdf_to_json_enhanced_fast(pdf_path: Path, json_path: Path):
    """
    Fast enhanced OCR with GPU acceleration and parallel processing
    """
    if GPU_AVAILABLE:
        print(f"[GPU] Fast Enhanced OCR (GPU ACCELERATED): {pdf_path.name}")
    else:
        print(f"[CPU] Fast Enhanced OCR (CPU): {pdf_path.name}")
    start_time = time.time()

    # Convert PDF with optimized DPI for speed
    pages = convert_from_path(
        pdf_path,
        dpi=DPI_SETTING,
        fmt='png'
    )

    ocr_result = {
        "document": pdf_path.name,
        "total_pages": len(pages),
        "pages": []
    }

    if PARALLEL_PROCESSING and len(pages) > 1:
        gpu_status = "GPU" if GPU_AVAILABLE else "CPU"
        print(f"  [PARALLEL] Processing {len(pages)} pages in parallel ({MAX_WORKERS} workers, {gpu_status})...")
        
        # Prepare arguments for parallel processing
        page_args = [(page_num, img, GPU_AVAILABLE) for page_num, img in enumerate(pages, start=1)]
        
        # Process pages in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(process_page_parallel, page_args))
        
        # Filter out completely failed pages (None) but keep error pages
        valid_results = [r for r in results if r is not None]
        valid_results.sort(key=lambda x: x['page_number'])
        ocr_result["pages"] = valid_results
        
        # Report errors
        error_pages = [r for r in valid_results if 'error' in r]
        if error_pages:
            print(f"  ⚠️ {len(error_pages)} pages had errors but data was recovered")
        
    else:
        # Sequential processing for single page or when parallel is disabled
        for page_num, img in enumerate(pages, start=1):
            print(f"  - Page {page_num}...", end=" ", flush=True)
            
            page_data = process_page_parallel((page_num, img, GPU_AVAILABLE))
            if page_data:
                ocr_result["pages"].append(page_data)
                print(f"done ({page_data['word_count']} words)")
            else:
                print("failed")

    # Update total pages
    ocr_result["total_pages"] = len(ocr_result["pages"])

    # Save as JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ocr_result, f, ensure_ascii=False, indent=2)

    total_words = sum(page['word_count'] for page in ocr_result['pages'])
    avg_conf = sum(page.get('average_confidence', 0) for page in ocr_result['pages']) / len(ocr_result['pages']) if ocr_result['pages'] else 0
    processing_time = time.time() - start_time
    
    print(f"  [OK] Fast Enhanced OCR JSON: {json_path.name}")
    print(f"  [STATS] Total words: {total_words}, Average confidence: {avg_conf:.1f}%")
    print(f"  [TIME] Processing time: {processing_time:.2f} seconds\n")


def main():
    """
    Process all PDF files in the input directory and convert them to JSON using fast enhanced OCR
    """
    start_time = time.time()
    
    print(f"[START] Fast Enhanced Arabic OCR Pipeline")
    print(f"[CPU] CPU Cores: {multiprocessing.cpu_count()}, Using: {MAX_WORKERS} workers")
    print(f"[GPU] GPU Available: {'Yes' if GPU_AVAILABLE else 'No'}")
    print(f"[PARALLEL] Parallel Processing: {'Enabled' if PARALLEL_PROCESSING else 'Disabled'}")
    print(f"[DPI] DPI Setting: {DPI_SETTING} (optimized for speed)")
    print("="*60)
    
    # Setup directories
    pdf_input_path = Path(PDF_INPUT_DIR)
    
    # Create output directory inside pipeline2
    current_dir = Path(__file__).parent  # pipeline2 directory
    json_output_path = current_dir / JSON_OUTPUT_DIR
    json_output_path.mkdir(exist_ok=True)
    
    # Find all PDF files in input directory
    pdf_files = list(pdf_input_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"[ERROR] No PDF files found in {pdf_input_path}")
        print(f"[INFO] Make sure PDF files are in: {pdf_input_path.resolve()}")
        return
    
    print(f"[SCAN] Found {len(pdf_files)} PDF files in {pdf_input_path}")
    print(f"[OUTPUT] Output directory: {json_output_path.resolve()}")
    print(f"{'='*60}")
    
    # Process each PDF file
    successful_conversions = 0
    total_words = 0
    total_pages = 0
    
    for pdf_file in pdf_files:
        print(f"\n[PDF] Processing: {pdf_file.name}")
        
        # Generate output JSON filename (keep same name as PDF)
        json_filename = pdf_file.stem + ".json"
        json_file_path = json_output_path / json_filename
        
        # Check if this file has already been processed
        if json_file_path.exists():
            print(f"[SKIP] File already processed: {json_filename}")
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                    total_pages += result_data['total_pages']
                    total_words += sum(page['word_count'] for page in result_data['pages'])
                    successful_conversions += 1
            except Exception as e:
                print(f"[WARNING] Could not read existing file: {e}")
            continue
        
        try:
            # Convert PDF to JSON using fast enhanced OCR
            ocr_pdf_to_json_enhanced_fast(pdf_file, json_file_path)
            
            if json_file_path.exists():
                successful_conversions += 1
                
                # Read the generated JSON to get statistics
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                    total_pages += result_data['total_pages']
                    total_words += sum(page['word_count'] for page in result_data['pages'])
        
        except Exception as e:
            print(f"  [ERROR] Error processing {pdf_file.name}: {e}")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"[DONE] Fast Enhanced OCR Processing Complete!")
    print(f"[RESULT] Successfully processed: {successful_conversions}/{len(pdf_files)} PDFs")
    print(f"[STATS] Total pages processed: {total_pages}")
    print(f"[STATS] Total words extracted: {total_words}")
    print(f"[TIME] Total processing time: {total_time:.2f} seconds")
    if total_pages > 0:
        print(f"[AVG] Average time per page: {total_time/total_pages:.2f} seconds")
    print(f"[OUTPUT] JSON files saved in: {json_output_path.resolve()}")
    
    # Performance tips
    if not GPU_AVAILABLE:
        print(f"\n[INFO] Performance Tip: Install CuPy for GPU acceleration:")
        print(f"   pip install cupy-cuda11x  # For CUDA 11.x")
        print(f"   pip install cupy-cuda12x  # For CUDA 12.x")


if __name__ == "__main__":
    main()