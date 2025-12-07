#!/usr/bin/env python3
"""
Evaluate fine-tuned AraFix model on test set
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from difflib import SequenceMatcher
import time

# Configuration
MODEL_PATH = "AraFix-Finetuned"
TEST_FILE = "AraFix_dataset_aligned/test_data.jsonl"

print("\n" + "="*80)
print("ARAFIX MODEL EVALUATION ON TEST SET")
print("="*80)

# Load model and tokenizer
print(f"\n📦 Loading model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()  # Set to evaluation mode
print(f"✅ Model loaded on: {device}")

# Load test data
print(f"\n📊 Loading test data from: {TEST_FILE}")
test_data = []
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        test_data.append(json.loads(line))

print(f"✅ Loaded {len(test_data)} test examples")

# Evaluation metrics
def calculate_metrics(predictions, references):
    """Calculate various evaluation metrics"""
    
    # Character-level accuracy
    char_correct = 0
    char_total = 0
    
    # Word-level accuracy
    word_correct = 0
    word_total = 0
    
    # Similarity scores
    similarities = []
    
    for pred, ref in zip(predictions, references):
        # Character level
        for p_char, r_char in zip(pred, ref):
            char_total += 1
            if p_char == r_char:
                char_correct += 1
        
        # Word level
        pred_words = pred.split()
        ref_words = ref.split()
        word_total += len(ref_words)
        
        # Count matching words
        for pred_word, ref_word in zip(pred_words, ref_words):
            if pred_word == ref_word:
                word_correct += 1
        
        # Similarity
        similarity = SequenceMatcher(None, pred, ref).ratio()
        similarities.append(similarity)
    
    return {
        'char_accuracy': (char_correct / char_total * 100) if char_total > 0 else 0,
        'word_accuracy': (word_correct / word_total * 100) if word_total > 0 else 0,
        'avg_similarity': sum(similarities) / len(similarities) * 100 if similarities else 0
    }

# Run evaluation
print(f"\n🔄 Running evaluation on {len(test_data)} examples...")
print("This may take a few minutes...\n")

predictions = []
references = []
start_time = time.time()

for example in tqdm(test_data, desc="Evaluating"):
    # Get input and reference
    input_text = example['input']
    reference = example['target']
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
    
    # Decode
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    predictions.append(prediction)
    references.append(reference)

eval_time = time.time() - start_time

# Calculate metrics
print(f"\n📈 Calculating metrics...")
metrics = calculate_metrics(predictions, references)

# Print results
print(f"\n{'='*80}")
print("EVALUATION RESULTS")
print(f"{'='*80}")
print(f"\n📊 Dataset Statistics:")
print(f"   Total examples: {len(test_data)}")
print(f"   Evaluation time: {eval_time:.1f} seconds")
print(f"   Speed: {len(test_data)/eval_time:.2f} examples/second")

print(f"\n🎯 Performance Metrics:")
print(f"   Character Accuracy: {metrics['char_accuracy']:.2f}%")
print(f"   Word Accuracy: {metrics['word_accuracy']:.2f}%")
print(f"   Average Similarity: {metrics['avg_similarity']:.2f}%")

# Show sample predictions
print(f"\n{'='*80}")
print("SAMPLE PREDICTIONS (First 5)")
print(f"{'='*80}\n")

for i in range(min(5, len(test_data))):
    ocr_text = test_data[i]['input'].replace('correct: ', '')
    print(f"Example {i+1}:")
    print(f"  OCR Input:    {ocr_text[:100]}...")
    print(f"  Ground Truth: {references[i][:100]}...")
    print(f"  Predicted:    {predictions[i][:100]}...")
    
    # Calculate similarity for this example
    sim = SequenceMatcher(None, predictions[i], references[i]).ratio()
    print(f"  Similarity:   {sim*100:.1f}%")
    print()

# Analyze error distribution
print(f"{'='*80}")
print("ERROR ANALYSIS")
print(f"{'='*80}\n")

# Group by similarity ranges
excellent = sum(1 for s in [SequenceMatcher(None, p, r).ratio() for p, r in zip(predictions, references)] if s >= 0.9)
good = sum(1 for s in [SequenceMatcher(None, p, r).ratio() for p, r in zip(predictions, references)] if 0.7 <= s < 0.9)
fair = sum(1 for s in [SequenceMatcher(None, p, r).ratio() for p, r in zip(predictions, references)] if 0.5 <= s < 0.7)
poor = sum(1 for s in [SequenceMatcher(None, p, r).ratio() for p, r in zip(predictions, references)] if s < 0.5)

total = len(predictions)

print(f"Similarity Distribution:")
print(f"  Excellent (≥90%):  {excellent}/{total} ({excellent/total*100:.1f}%)")
print(f"  Good (70-89%):     {good}/{total} ({good/total*100:.1f}%)")
print(f"  Fair (50-69%):     {fair}/{total} ({fair/total*100:.1f}%)")
print(f"  Poor (<50%):       {poor}/{total} ({poor/total*100:.1f}%)")

# Overall assessment
print(f"\n{'='*80}")
print("OVERALL ASSESSMENT")
print(f"{'='*80}\n")

avg_sim = metrics['avg_similarity']
if avg_sim >= 85:
    quality = "EXCELLENT ✅"
    comment = "Model performs very well on OCR correction!"
elif avg_sim >= 75:
    quality = "GOOD ✅"
    comment = "Model shows strong OCR correction capabilities."
elif avg_sim >= 65:
    quality = "ACCEPTABLE ⚠️"
    comment = "Model provides useful corrections but has room for improvement."
else:
    quality = "NEEDS IMPROVEMENT ❌"
    comment = "Model may need more training or data."

print(f"Quality Rating: {quality}")
print(f"{comment}")
print(f"\n{'='*80}\n")

# Save detailed results
results = {
    'metrics': metrics,
    'total_examples': len(test_data),
    'eval_time': eval_time,
    'distribution': {
        'excellent': excellent,
        'good': good,
        'fair': fair,
        'poor': poor
    }
}

with open('AraFix-Finetuned/evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"💾 Detailed results saved to: AraFix-Finetuned/evaluation_results.json")
