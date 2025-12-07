#!/usr/bin/env python3
"""
Test the fine-tuned AraFix model with sample OCR text
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned model
MODEL_PATH = "AraFix-Finetuned"

print("\n" + "="*80)
print("TESTING FINE-TUNED ARAFIX MODEL")
print("="*80)

print(f"\n📦 Loading model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"✅ Model loaded on: {device}")

# Test samples with OCR errors
test_samples = [
    # Sample 1: Missing spaces and character errors
    "التعليم العالي» ند عو المسحوبةجنسياتهم إلى مراجحتها لتسلّم 'براءةذمة'",
    
    # Sample 2: Common OCR mistakes
    "أعلثت وزازة التعلية العالي أنه ينعين على الذين سحيت متهم الجنسيةمراجعة",
    
    # Sample 3: Merged words
    "مينى الوزارة في صالة المراجعين (برج السنابل - 013), ابتداء من اليومالاثنين",
    
    # Sample 4: Number recognition errors
    "3 الجاري من الساعة 10 صباحا حت الماعة 12 ظهراء وذلك لتسلم شهادة براءةذمة",
    
    # Sample 5: Mixed errors
    "صاحب السمق الأمير الشبع ممعل امد مستقبلا نائب رئيس مجلس إ"
]

print(f"\n{'='*80}")
print("TESTING OCR CORRECTION")
print(f"{'='*80}\n")

for i, ocr_text in enumerate(test_samples, 1):
    print(f"{'─'*80}")
    print(f"Test {i}:")
    print(f"{'─'*80}")
    print(f"OCR Input:  {ocr_text}")
    
    # Prepare input (add prefix as used in training)
    input_text = f"correct: {ocr_text}"
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)
    
    # Generate correction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=5,  # Use beam search for better quality
            early_stopping=True
        )
    
    # Decode
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Corrected:  {corrected_text}")
    print()

print(f"{'='*80}")
print("✅ TESTING COMPLETE!")
print(f"{'='*80}\n")

# Interactive mode
print("💡 Want to test your own text? (Press Ctrl+C to exit)")
print()

try:
    while True:
        user_input = input("Enter OCR text to correct: ").strip()
        
        if not user_input:
            continue
        
        # Prepare and tokenize
        input_text = f"correct: {user_input}"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
        
        corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"→ Corrected: {corrected}\n")

except KeyboardInterrupt:
    print("\n\n👋 Goodbye!")
