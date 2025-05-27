# scripts/fine_tune_model.py
import os
from pathlib import Path
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
import torch


def fine_tune():
    project_root = Path(__file__).parent.parent
    train_file_path = project_root / "data" / "processed_texts" / "domain_the_bible" / "training_corpus.txt"  # UPDATED
    # For Phase 1, we'll use the same file for validation for simplicity,
    # but ideally, you should have a separate validation file.
    validation_file_path = train_file_path

    model_output_dir = project_root / "trained_models" / "the_bible_gpt2_small"  # UPDATED
    model_output_dir.mkdir(parents=True, exist_ok=True)

    base_model_name = "gpt2"  # Using the smallest GPT-2 for faster training

    # 1. Load Tokenizer and Model
    print(f"Loading tokenizer and model: {base_model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
    model = GPT2LMHeadModel.from_pretrained(base_model_name)

    # Add a padding token if it doesn't exist (GPT-2 usually doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings to include new token

    # 2. Prepare Dataset
    print(f"Loading and preparing dataset from: {train_file_path}")
    if not train_file_path.exists():
        print(f"Training file not found: {train_file_path}")
        print("Please run the 'extract_text_from_pdfs.py' script first.")
        return

    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=str(train_file_path),
        block_size=128  # Adjust based on your VRAM and typical sentence length
    )

    eval_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=str(validation_file_path),  # Use your validation file here
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We are doing Causal LM (next word prediction), not Masked LM
    )

    # 3. Define Training Arguments
    print("Defining training arguments...")
    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
        overwrite_output_dir=True,
        num_train_epochs=1,  # Start with 1 epoch for a quick test, increase for better results
        per_device_train_batch_size=2,  # Adjust based on VRAM (e.g., 2, 4, 8)
        per_device_eval_batch_size=2,  # Adjust based on VRAM
        save_steps=500,  # Save checkpoint every X steps
        save_total_limit=2,  # Only keep the last 2 checkpoints
        logging_steps=100,  # Log training progress every X steps
        # evaluation_strategy="steps",  # Evaluate during training
        eval_steps=200,  # Evaluate every X steps
        fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA is available
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # 5. Start Fine-tuning
    print("Starting fine-tuning...")
    trainer.train()

    # 6. Save the final model and tokenizer
    print(f"Saving model and tokenizer to {model_output_dir}")
    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))

    print("Fine-tuning complete!")
    print(f"Model saved to: {model_output_dir}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Training will run on CPU and will be very slow.")
    else:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    fine_tune()