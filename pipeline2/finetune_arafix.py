#!/usr/bin/env python3
"""
Fine-tune AraFix model on Arabic OCR correction dataset
Optimized for RTX 3050 6GB GPU
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import os

# Configuration
MODEL_NAME = "../AraFix-V3.0"  # Your base model folder (relative to pipeline2)
TRAIN_FILE = "AraFix_dataset_aligned/train_data.jsonl"
VAL_FILE = "AraFix_dataset_aligned/val_data.jsonl"
OUTPUT_DIR = "AraFix-Finetuned"

# GPU-optimized settings for RTX 3050 6GB
BATCH_SIZE = 4  # Small batch for 6GB VRAM
GRADIENT_ACCUMULATION = 4  # Effective batch size = 4 * 4 = 16
MAX_LENGTH = 512  # Standard for T5/mT5
LEARNING_RATE = 3e-5
EPOCHS = 3

print("\n" + "="*80)
print("ARAFIX MODEL FINE-TUNING")
print("="*80)
print(f"\n📋 Configuration:")
print(f"   Base model: {MODEL_NAME}")
print(f"   Training data: {TRAIN_FILE}")
print(f"   Validation data: {VAL_FILE}")
print(f"   Output directory: {OUTPUT_DIR}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Gradient accumulation: {GRADIENT_ACCUMULATION}")
print(f"   Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Epochs: {EPOCHS}")

# Check CUDA
print(f"\n🔧 GPU Status:")
if torch.cuda.is_available():
    print(f"   ✅ CUDA available")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print(f"   ⚠️  CUDA not available - will use CPU (much slower!)")

# Load tokenizer and model
print(f"\n📦 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print(f"   ✅ Model loaded: {model.config.model_type}")

# Load dataset
def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

print(f"\n📊 Loading datasets...")
train_data = load_jsonl(TRAIN_FILE)
val_data = load_jsonl(VAL_FILE)

print(f"   Training examples: {len(train_data)}")
print(f"   Validation examples: {len(val_data)}")

# Convert to Hugging Face Dataset
def prepare_dataset(data):
    """Convert data to format needed for training"""
    inputs = [item['input'] for item in data]
    targets = [item['target'] for item in data]
    return Dataset.from_dict({'input': inputs, 'target': targets})

train_dataset = prepare_dataset(train_data)
val_dataset = prepare_dataset(val_data)

# Tokenization function
def preprocess_function(examples):
    """Tokenize inputs and targets"""
    model_inputs = tokenizer(
        examples['input'],
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False  # Handled by data collator
    )
    
    # Tokenize targets
    labels = tokenizer(
        examples['target'],
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False
    )
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

print(f"\n🔄 Tokenizing datasets...")
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train"
)

tokenized_val = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names,
    desc="Tokenizing validation"
)

print(f"   ✅ Tokenization complete")

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Training settings
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_steps=500,
    weight_decay=0.01,
    
    # Memory optimization for 6GB GPU
    fp16=True,  # Mixed precision training
    gradient_checkpointing=True,  # Save memory
    optim="adafactor",  # Memory-efficient optimizer
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,  # Keep only 2 best checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # Logging
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=100,
    report_to="none",  # Disable wandb/tensorboard
    
    # Generation settings for evaluation
    predict_with_generate=True,
    generation_max_length=MAX_LENGTH,
    
    # Other
    remove_unused_columns=True,
    push_to_hub=False,
)

# Create trainer
print(f"\n🎯 Initializing trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print(f"   ✅ Trainer ready")

# Start training
print(f"\n{'='*80}")
print("🚀 STARTING TRAINING")
print(f"{'='*80}\n")

try:
    trainer.train()
    
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETED!")
    print(f"{'='*80}")
    
    # Save final model
    print(f"\n💾 Saving final model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"   ✅ Model saved successfully")
    
    # Evaluate on validation set
    print(f"\n📊 Final evaluation...")
    metrics = trainer.evaluate()
    
    print(f"\n📈 Final Metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    print(f"\n{'='*80}")
    print(f"✅ FINE-TUNING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📂 Model saved to: {OUTPUT_DIR}/")
    print(f"🎯 You can now use this model for OCR correction!")
    print(f"\n{'='*80}\n")

except KeyboardInterrupt:
    print(f"\n\n⚠️  Training interrupted by user")
    print(f"💾 Saving current state...")
    trainer.save_model(f"{OUTPUT_DIR}/interrupted")
    print(f"   Saved to: {OUTPUT_DIR}/interrupted")

except Exception as e:
    print(f"\n\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
