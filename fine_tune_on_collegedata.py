import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Fine-tune GPT-2 model on custom data
def fine_tune_gpt2_model(text_file, model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Prepare the dataset and collator
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=text_file,
        block_size=128
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./gpt2_finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Train the model
    trainer.train()
    trainer.save_model("./gpt2_finetuned")

# Load and fine-tune GPT-2 on the college and research dataset
fine_tune_gpt2_model("dataset/research_data.txt")
