#!/usr/bin/env python3
"""
Dataset Alignment Verifier
Checks if OCR and ground truth text pairs are properly aligned
"""

import json
from pathlib import Path
import random
from difflib import SequenceMatcher
import re

class AlignmentVerifier:
    """
    Verifies alignment quality of training data
    """
    
    def __init__(self, dataset_file="AraFix_dataset/train_data.jsonl"):
        self.dataset_file = Path(dataset_file)
        
    def load_samples(self, num_samples=10):
        """
        Load random samples from the dataset
        """
        samples = []
        
        with open(self.dataset_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        # Get random samples
        if len(all_lines) > num_samples:
            selected_lines = random.sample(all_lines, num_samples)
        else:
            selected_lines = all_lines
        
        for line in selected_lines:
            try:
                data = json.loads(line)
                samples.append(data)
            except:
                continue
        
        return samples
    
    def calculate_similarity(self, text1, text2):
        """
        Calculate similarity ratio between two texts
        """
        # Normalize texts for comparison
        text1_norm = re.sub(r'\s+', ' ', text1.lower()).strip()
        text2_norm = re.sub(r'\s+', ' ', text2.lower()).strip()
        
        # Calculate similarity
        similarity = SequenceMatcher(None, text1_norm, text2_norm).ratio()
        
        return similarity
    
    def count_words(self, text):
        """
        Count words in Arabic text
        """
        words = text.split()
        return len(words)
    
    def analyze_sample(self, sample):
        """
        Analyze a single training sample
        """
        # Extract OCR text (remove "correct: " prefix)
        ocr_text = sample['input'].replace('correct: ', '')
        gt_text = sample['target']
        
        # Calculate metrics
        similarity = self.calculate_similarity(ocr_text, gt_text)
        ocr_words = self.count_words(ocr_text)
        gt_words = self.count_words(gt_text)
        word_diff = abs(ocr_words - gt_words)
        word_diff_percent = (word_diff / max(ocr_words, gt_words) * 100) if max(ocr_words, gt_words) > 0 else 0
        
        analysis = {
            'source': sample.get('source', 'unknown'),
            'similarity': similarity,
            'ocr_words': ocr_words,
            'gt_words': gt_words,
            'word_difference': word_diff,
            'word_diff_percent': word_diff_percent,
            'ocr_preview': ocr_text[:100],
            'gt_preview': gt_text[:100],
            'ocr_length': len(ocr_text),
            'gt_length': len(gt_text)
        }
        
        return analysis
    
    def check_alignment_quality(self, samples):
        """
        Check overall alignment quality
        """
        print("\n" + "="*80)
        print("DATASET ALIGNMENT VERIFICATION REPORT")
        print("="*80)
        
        total_samples = len(samples)
        print(f"\nTotal samples analyzed: {total_samples}")
        
        # Analyze each sample
        results = []
        for i, sample in enumerate(samples, 1):
            analysis = self.analyze_sample(sample)
            results.append(analysis)
            
            print(f"\n{'─'*80}")
            print(f"Sample #{i}: {analysis['source']}")
            print(f"{'─'*80}")
            print(f"Similarity Score: {analysis['similarity']:.2%}")
            print(f"Word Count - OCR: {analysis['ocr_words']} | Ground Truth: {analysis['gt_words']}")
            print(f"Word Difference: {analysis['word_difference']} words ({analysis['word_diff_percent']:.1f}%)")
            print(f"Character Count - OCR: {analysis['ocr_length']} | Ground Truth: {analysis['gt_length']}")
            
            print(f"\nOCR Text Preview:")
            print(f"  {analysis['ocr_preview']}...")
            print(f"\nGround Truth Preview:")
            print(f"  {analysis['gt_preview']}...")
            
            # Quality assessment
            if analysis['similarity'] > 0.85:
                status = "✅ EXCELLENT - Well aligned"
            elif analysis['similarity'] > 0.70:
                status = "✅ GOOD - Acceptable alignment"
            elif analysis['similarity'] > 0.50:
                status = "⚠️  MODERATE - Review recommended"
            else:
                status = "❌ POOR - Misalignment detected"
            
            print(f"\nAlignment Status: {status}")
        
        # Overall statistics
        print(f"\n{'='*80}")
        print("OVERALL STATISTICS")
        print(f"{'='*80}")
        
        avg_similarity = sum(r['similarity'] for r in results) / len(results)
        avg_word_diff = sum(r['word_diff_percent'] for r in results) / len(results)
        
        print(f"\nAverage Similarity: {avg_similarity:.2%}")
        print(f"Average Word Difference: {avg_word_diff:.1f}%")
        
        # Quality distribution
        excellent = sum(1 for r in results if r['similarity'] > 0.85)
        good = sum(1 for r in results if 0.70 < r['similarity'] <= 0.85)
        moderate = sum(1 for r in results if 0.50 < r['similarity'] <= 0.70)
        poor = sum(1 for r in results if r['similarity'] <= 0.50)
        
        print(f"\nQuality Distribution:")
        print(f"  ✅ Excellent (>85%): {excellent}/{total_samples} ({excellent/total_samples*100:.1f}%)")
        print(f"  ✅ Good (70-85%):    {good}/{total_samples} ({good/total_samples*100:.1f}%)")
        print(f"  ⚠️  Moderate (50-70%): {moderate}/{total_samples} ({moderate/total_samples*100:.1f}%)")
        print(f"  ❌ Poor (<50%):      {poor}/{total_samples} ({poor/total_samples*100:.1f}%)")
        
        # Overall recommendation
        print(f"\n{'='*80}")
        print("RECOMMENDATION")
        print(f"{'='*80}")
        
        if avg_similarity > 0.75:
            print("\n✅ Dataset quality is GOOD for fine-tuning!")
            print("   The OCR and ground truth pairs are well-aligned.")
            print("   You can proceed with training confidently.")
        elif avg_similarity > 0.60:
            print("\n⚠️  Dataset quality is ACCEPTABLE but could be improved.")
            print("   Consider reviewing samples with low similarity scores.")
            print("   Training should still work but may need more epochs.")
        else:
            print("\n❌ Dataset quality needs IMPROVEMENT!")
            print("   Many pairs appear misaligned.")
            print("   Review the data pairing process before training.")
        
        print(f"\n{'='*80}\n")
        
        return results

def main():
    """
    Main function to verify dataset alignment
    """
    print("\n🔍 Starting Dataset Alignment Verification...\n")
    
    verifier = AlignmentVerifier("AraFix_dataset/train_data.jsonl")
    
    # Load and analyze samples
    print("Loading random samples from dataset...")
    samples = verifier.load_samples(num_samples=10)
    
    if not samples:
        print("❌ Error: Could not load samples from dataset!")
        return
    
    # Check alignment quality
    results = verifier.check_alignment_quality(samples)
    
    print("✅ Verification complete!")

if __name__ == "__main__":
    main()
