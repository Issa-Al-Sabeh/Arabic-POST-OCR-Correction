#!/usr/bin/env python3
"""
Calculate dataset size WITHOUT chunking (full pages only)
"""

import json
from pathlib import Path

# Load label data
with open('labeled_data/label_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("="*80)
print("DATASET SIZE ANALYSIS: Full Pages (No Chunking)")
print("="*80)

total_pages = 0
total_pairs = len(data)
page_lengths = []

# Count all pages across all pairs
for pair in data:
    ocr_pages = pair['ocr_data']['pages']
    gt_pages = pair['ground_truth_data']['pages']
    
    num_pages = min(len(ocr_pages), len(gt_pages))
    total_pages += num_pages
    
    # Collect page text lengths for analysis
    for i in range(num_pages):
        ocr_len = len(ocr_pages[i].get('text', ''))
        gt_len = len(gt_pages[i].get('text', ''))
        page_lengths.append((ocr_len, gt_len))

print(f"\n📊 Dataset Statistics:")
print(f"{'─'*80}")
print(f"Total file pairs: {total_pairs}")
print(f"Total pages: {total_pages}")
print(f"\n💡 Training examples (without chunking): {total_pages}")

# Calculate 80/10/10 split
train_size = int(total_pages * 0.8)
val_size = int(total_pages * 0.1)
test_size = total_pages - train_size - val_size

print(f"\n📈 Train/Val/Test Split (80/10/10):")
print(f"{'─'*80}")
print(f"Training:   {train_size} pages")
print(f"Validation: {val_size} pages")
print(f"Test:       {test_size} pages")

# Analyze page lengths
if page_lengths:
    avg_ocr_len = sum(p[0] for p in page_lengths) / len(page_lengths)
    avg_gt_len = sum(p[1] for p in page_lengths) / len(page_lengths)
    max_ocr_len = max(p[0] for p in page_lengths)
    max_gt_len = max(p[1] for p in page_lengths)
    
    print(f"\n📏 Page Length Statistics:")
    print(f"{'─'*80}")
    print(f"Average OCR page length: {avg_ocr_len:.0f} characters")
    print(f"Average GT page length:  {avg_gt_len:.0f} characters")
    print(f"Maximum OCR page length: {max_ocr_len} characters")
    print(f"Maximum GT page length:  {max_gt_len} characters")
    
    # Check if pages are too long for T5
    print(f"\n⚠️  T5 Model Token Limits:")
    print(f"{'─'*80}")
    print(f"T5-base max input tokens: 512 tokens (~2000 chars)")
    print(f"T5-large max input tokens: 512 tokens (~2000 chars)")
    print(f"AraFix (mT5-based) typical: 512 tokens (~2000 chars)")
    
    # Estimate how many pages exceed limit
    long_pages = sum(1 for p in page_lengths if p[0] > 2000 or p[1] > 2000)
    very_long_pages = sum(1 for p in page_lengths if p[0] > 4000 or p[1] > 4000)
    
    print(f"\n📊 Pages by Length:")
    print(f"{'─'*80}")
    print(f"Pages > 2000 chars (may need truncation): {long_pages}/{total_pages} ({long_pages/total_pages*100:.1f}%)")
    print(f"Pages > 4000 chars (definitely too long):  {very_long_pages}/{total_pages} ({very_long_pages/total_pages*100:.1f}%)")
    print(f"Pages ≤ 2000 chars (perfect fit):          {total_pages-long_pages}/{total_pages} ({(total_pages-long_pages)/total_pages*100:.1f}%)")

print(f"\n{'='*80}")
print("RECOMMENDATION:")
print(f"{'='*80}")

if total_pages < 500:
    print("\n❌ NOT ENOUGH DATA!")
    print(f"   You have {total_pages} pages, but need at least 500-1000 for decent training.")
    print("   RECOMMENDATION: Keep chunking OR process more PDFs.")
elif total_pages < 1000:
    print("\n⚠️  MARGINAL - May work but not ideal")
    print(f"   You have {total_pages} pages. This might work but more data is better.")
    print("   RECOMMENDATION: Try it, but consider processing more PDFs if results aren't good.")
else:
    print(f"\n✅ SUFFICIENT DATA!")
    print(f"   You have {total_pages} pages - enough for training without chunking.")
    
    if long_pages > total_pages * 0.3:
        print(f"\n⚠️  However, {long_pages/total_pages*100:.0f}% of pages are quite long.")
        print("   You may need to:")
        print("   1. Truncate long pages to fit model's token limit, OR")
        print("   2. Use smart chunking with proper alignment")
    else:
        print(f"\n✅ Most pages ({(total_pages-long_pages)/total_pages*100:.0f}%) fit model limits well!")
        print("   You can proceed with full-page training.")

print(f"\n{'='*80}\n")
