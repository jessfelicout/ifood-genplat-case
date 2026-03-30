import argparse
from pyexpat import model
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # Load dataset (expects JSONL with a "text" field)
    dataset = load_dataset("json", data_files={"train": args.dataset})
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)


    # Ensure pad_token is defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenization function
    def tokenize(batch):
        tokens = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
        # Add labels (same as input_ids for causal LM training)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    # Apply tokenization to dataset
    tokenized_dataset = dataset.map(tokenize, batched=True)

    # Load pre-trained model
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # Training configuration
    training_args = TrainingArguments(
        output_dir="./output",              # Directory to save checkpoints
        overwrite_output_dir=True,          # Overwrite old outputs
        num_train_epochs=args.epochs,       # Number of epochs
        per_device_train_batch_size=args.batch_size,  # Batch size per device
        save_steps=500,                     # Save checkpoint every 500 steps
        save_total_limit=2,                 # Keep only last 2 checkpoints
        logging_dir="./logs",               # Directory for logs
        logging_steps=100,                  # Log every 100 steps
    )

    # Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
    )

    # Start training
    trainer.train()

    # Save model and tokenizer after training
    tokenizer.save_pretrained("./output")
    model.save_pretrained("./output")
    trainer.save_model("./output")

if __name__ == "__main__":
    main()