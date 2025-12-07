#!/usr/bin/env python3
"""
Analyze OCR vs Ground Truth structure differences
"""

import json
from pathlib import Path

# Load one pair to analyze
ocr_file = Path("arabic_ocr/03-03-2025.json")
gt_file = Path("xml_converted_json/03-03-2025.json")

print("="*80)
print("STRUCTURE ANALYSIS: OCR vs Ground Truth")
print("="*80)

# Load OCR
with open(ocr_file, 'r', encoding='utf-8') as f:
    ocr_data = json.load(f)

# Load GT
with open(gt_file, 'r', encoding='utf-8') as f:
    gt_data = json.load(f)

print(f"\n📄 File: 03-03-2025.json")
print(f"\n{'─'*80}")
print(f"BASIC STRUCTURE:")
print(f"{'─'*80}")
print(f"OCR Pages: {len(ocr_data['pages'])}")
print(f"GT Pages:  {len(gt_data['pages'])}")

print(f"\n{'─'*80}")
print(f"PAGE 1 COMPARISON:")
print(f"{'─'*80}")

ocr_page1 = ocr_data['pages'][0]
gt_page1 = gt_data['pages'][0]

print(f"\nOCR Page 1:")
print(f"  Word count: {ocr_page1.get('word_count', 'N/A')}")
print(f"  Text length: {len(ocr_page1['text'])} characters")
print(f"  Text preview (first 300 chars):")
print(f"  {ocr_page1['text'][:300]}")

print(f"\nGround Truth Page 1:")
print(f"  Word count: {gt_page1.get('word_count', 'N/A')}")
print(f"  Text length: {len(gt_page1['text'])} characters")
print(f"  Text preview (first 300 chars):")
print(f"  {gt_page1['text'][:300]}")

# Check if texts match
from difflib import SequenceMatcher
similarity = SequenceMatcher(None, ocr_page1['text'][:500], gt_page1['text'][:500]).ratio()

print(f"\n📊 Similarity (first 500 chars): {similarity:.2%}")

# Check all pages
print(f"\n{'─'*80}")
print(f"ALL PAGES SIMILARITY:")
print(f"{'─'*80}")

for i in range(min(5, len(ocr_data['pages']), len(gt_data['pages']))):
    ocr_text = ocr_data['pages'][i]['text'][:500]
    gt_text = gt_data['pages'][i]['text'][:500]
    sim = SequenceMatcher(None, ocr_text.lower(), gt_text.lower()).ratio()
    
    print(f"Page {i+1}: {sim:.2%} similarity | OCR: {len(ocr_data['pages'][i]['text'])} chars | GT: {len(gt_data['pages'][i]['text'])} chars")
    
    if sim < 0.3:
        print(f"  ⚠️  Very different content!")
        print(f"  OCR starts: {ocr_text[:80]}")
        print(f"  GT starts:  {gt_text[:80]}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}\n")
