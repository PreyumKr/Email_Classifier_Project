from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification

# Load dataset
raw_datasets = load_dataset("json", data_files="dataset/emailfinetune.json")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Apply tokenization
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


# Inspect unique labels
unique_labels = set()
for example in tokenized_datasets["train"]:
    unique_labels.add(example["label"])
print("Unique labels in the dataset:", unique_labels)

# Update label mapping to include all unique labels
label_map = {"general": 0, "sensitive": 1, "research": 2}

# Convert labels to numerical values
def convert_labels(examples):
    examples["label"] = [label_map[label] for label in examples["label"]]
    return examples

# Apply the label conversion to the dataset
tokenized_datasets = tokenized_datasets.map(convert_labels, batched=True)

# Split dataset into train and test sets
train_test_split = tokenized_datasets["train"].train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Initialize model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_map))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=100,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()

# Save the model
model.save_pretrained("./email_classification_model_bert100")