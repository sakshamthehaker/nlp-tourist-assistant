"""
train_nmt.py
Fine-tunes a Sequence-to-Sequence Neural Machine Translation (NMT) Model (English -> Hindi)
on the IIT Bombay English-Hindi Parallel Corpus using Hugging Face Transformers.

Usage:
    python train_nmt.py --num_samples 10000 --epochs 3 --batch_size 16

Requirements:
    pip install torch transformers datasets evaluate accelerate sacrebleu sentencepiece
"""

import os
import sys

# Resolve Windows OpenMP duplicate runtime conflict (Intel MKL & PyTorch)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

# Verify dependencies before importing heavy libraries
def verify_prerequisites():
    missing = []
    for pkg in ["torch", "transformers", "datasets", "evaluate", "accelerate", "sentencepiece"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("\n" + "=" * 70)
        print("⚠️  MISSING DEPENDENCIES FOR DEEP LEARNING MODEL TRAINING:")
        print(f"   Missing packages: {', '.join(missing)}")
        print("\n   Please install them via pip:")
        print(f"   pip install {' '.join(missing)}")
        print("\n💡 NOTE FOR ACADEMIC PROJECTS:")
        print("   If you do not have a dedicated GPU locally, you can run this script")
        print("   in a free Google Colab (T4 GPU) notebook in ~10 minutes.")
        print("   See 'train_nmt_colab.ipynb' in this directory.")
        print("=" * 70 + "\n")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune NMT Model for English to Hindi")
    parser.add_argument("--model_name", type=str, default="Helsinki-NLP/opus-mt-en-hi",
                        help="Base pre-trained model (e.g., Helsinki-NLP/opus-mt-en-hi or google/mt5-small)")
    parser.add_argument("--num_samples", type=int, default=15000,
                        help="Number of parallel sentence pairs to train on (default: 15,000 for fast convergence)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--output_dir", type=str, default="models/custom_en_hi_model",
                        help="Directory to save the fine-tuned model and tokenizer")
    return parser.parse_args()


def main():
    verify_prerequisites()

    import torch
    import numpy as np
    from datasets import load_dataset
    import evaluate
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer
    )

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("🚀 STARTING NEURAL MACHINE TRANSLATION (NMT) FINE-TUNING")
    print(f"   Base Model:        {args.model_name}")
    print(f"   Dataset:           IIT Bombay English-Hindi Parallel Corpus (cfilt/iitb-english-hindi)")
    print(f"   Sample Size:       {args.num_samples:,} sentence pairs")
    print(f"   Epochs:            {args.epochs}")
    print(f"   Batch Size:        {args.batch_size}")
    print(f"   Learning Rate:     {args.lr}")
    print(f"   Compute Device:    {device.upper()} {'⚡' if device == 'cuda' else '(CPU training)'}")
    print(f"   Output Directory:  {args.output_dir}")
    print("=" * 70)

    # 1. Load Pretrained Tokenizer and Model
    print("\n[Step 1/5] Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # 2. Load IIT Bombay Dataset
    print("\n[Step 2/5] Fetching IIT Bombay English-Hindi dataset...")
    raw_dataset = load_dataset("cfilt/iitb-english-hindi")
    print(f"   Full train corpus contains {len(raw_dataset['train']):,} pairs.")

    # Slice a manageable, high-quality subset for student fine-tuning
    train_slice = raw_dataset["train"].shuffle(seed=42).select(range(min(args.num_samples, len(raw_dataset["train"]))))
    val_slice = raw_dataset["validation"].shuffle(seed=42).select(range(min(1000, len(raw_dataset["validation"]))))

    print(f"   Training slice:   {len(train_slice):,} pairs")
    print(f"   Validation slice: {len(val_slice):,} pairs")

    # 3. Tokenization & Preprocessing Function
    print("\n[Step 3/5] Tokenizing sentence pairs...")
    def preprocess_function(examples):
        inputs = [ex["en"] if ex["en"] else "" for ex in examples["translation"]]
        targets = [ex["hi"] if ex["hi"] else "" for ex in examples["translation"]]

        model_inputs = tokenizer(inputs, max_length=args.max_length, truncation=True)
        labels = tokenizer(text_target=targets, max_length=args.max_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_slice.map(preprocess_function, batched=True, remove_columns=["translation"])
    tokenized_val = val_slice.map(preprocess_function, batched=True, remove_columns=["translation"])

    # 4. Metric Computation (SacreBLEU)
    bleu_metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we cannot decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": round(result["score"], 2)}

    # 5. Training Setup
    print("\n[Step 4/5] Initializing Seq2Seq Trainer...")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=(device == "cuda"),
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to="none"
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_val,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics
    }

    # Support all versions of transformers (processing_class in >=4.45/5.0 vs tokenizer in older)
    try:
        trainer = Seq2SeqTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        try:
            trainer = Seq2SeqTrainer(tokenizer=tokenizer, **trainer_kwargs)
        except TypeError:
            trainer = Seq2SeqTrainer(**trainer_kwargs)

    # 6. Execute Training
    print("\n[Step 5/5] Training model...")
    train_result = trainer.train()
    print("\nTraining completed successfully!")
    print(f"Global steps: {train_result.global_step} | Training loss: {train_result.training_loss:.4f}")

    # 7. Save Model & Tokenizer
    print(f"\nSaving fine-tuned model and tokenizer to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 8. Sample Inference Test on Tourist Phrases
    print("\n" + "=" * 70)
    print("🎯 TESTING FINE-TUNED MODEL ON REPRESENTATIVE TOURIST PHRASES:")
    print("=" * 70)
    test_phrases = [
        "How much does this cost?",
        "Can you please give me a discount?",
        "Where is the nearest railway station?",
        "Please make the food non-spicy.",
        "I need help immediately, call the police."
    ]

    model.eval()
    for phrase in test_phrases:
        inputs = tokenizer(phrase, return_tensors="pt", max_length=args.max_length, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=args.max_length, num_beams=4, early_stopping=True)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"EN: '{phrase}'")
        print(f"HI:  {translated}\n")

    print(f"Done! Your fine-tuned model is ready at: '{args.output_dir}'.")


if __name__ == "__main__":
    main()
