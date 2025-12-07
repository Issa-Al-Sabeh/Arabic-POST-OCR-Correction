#!/usr/bin/env python3
"""
Check if pages are aligned BEFORE chunking
"""

import json
from pathlib import Path
from difflib import SequenceMatcher

def check_page_similarity(ocr_text, gt_text):
    """Calculate similarity between two texts"""
    ocr_norm = ocr_text.lower().strip()[:500]  # First 500 chars
    gt_norm = gt_text.lower().strip()[:500]
    
    similarity = SequenceMatcher(None, ocr_norm, gt_norm).ratio()
    return similarity

# Load label data
with open('labeled_data/label_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Checking {len(data)} pairs...\n")

# Check first 3 pairs
for i, pair in enumerate(data[:3]):
    pair_id = pair['pair_id']
    print(f"{'='*80}")
    print(f"Pair: {pair_id}")
    print(f"{'='*80}")
    
    ocr_pages = pair['ocr_data']['pages']
    gt_pages = pair['ground_truth_data']['pages']
    
    print(f"OCR pages: {len(ocr_pages)} | GT pages: {len(gt_pages)}")
    
    # Check first 3 pages
    for page_idx in range(min(3, len(ocr_pages), len(gt_pages))):
        ocr_text = ocr_pages[page_idx].get('text', '')
        gt_text = gt_pages[page_idx].get('text', '')
        
        similarity = check_page_similarity(ocr_text, gt_text)
        
        print(f"\n  Page {page_idx + 1}:")
        print(f"    Similarity: {similarity:.2%}")
        print(f"    OCR preview: {ocr_text[:100]}")
        print(f"    GT preview:  {gt_text[:100]}")
        
        if similarity < 0.50:
            print(f"    ⚠️ WARNING: Low similarity!")
    
    print()
