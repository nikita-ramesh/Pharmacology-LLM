import base64
import getpass
import codecs
import requests
import psycopg2
import pandas as pd
import torch
# from evaluate_model import evaluate_query  # Import evaluation function
# from evaluate_model import evaluate_dataset  # Import dataset-level evaluation if needed
import json
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments


# Password generation function
def pwd():
    return getpass.getpass(prompt="Enter the database password: ")

# Database connection details
db_config = {
    'host': 'localhost',
    'database': 'guide_to_pharmacology',
    'user': 'postgres',
    'password': pwd(),
}

# Function to connect to the PostgreSQL database
def connect_to_db():
    try:
        conn = psycopg2.connect(**db_config)
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

# Load the training dataset
def load_training_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Prepare data for model
def prepare_data(data):
    inputs = data['Natural Language Query'].tolist()
    targets = data['SQL'].tolist()
    return inputs, targets

# Fine-tune the model
def fine_tune_model(train_inputs, train_targets, val_inputs, val_targets):
    model_name = "t5-small"  # Choose a T5 model variant
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Tokenisation
    train_encodings = tokenizer(train_inputs, truncation=True, padding=True, max_length=512)
    train_labels = tokenizer(train_targets, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_inputs, truncation=True, padding=True, max_length=512)
    val_labels = tokenizer(val_targets, truncation=True, padding=True, max_length=512)

    # Trainer-specific data structure
    train_dataset = Dataset(train_encodings, train_labels)
    val_dataset = Dataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir="./results",          # Directory for checkpoints
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    return model, tokenizer

# Custom Dataset for Trainer
class Dataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item

    def __len__(self):
        return len(self.labels['input_ids'])

# Evaluate Model
# def evaluate_model(model, tokenizer, inputs, targets):
#     correct = 0
#     for i, input_text in enumerate(inputs):
#         input_ids = tokenizer.encode(input_text, return_tensors="pt")
#         output = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
#         generated_sql = tokenizer.decode(output[0], skip_special_tokens=True)
#         if generated_sql.strip().lower() == targets[i].strip().lower():
#             correct += 1
#     accuracy = correct / len(inputs)
#     print(f"Accuracy: {accuracy * 100:.2f}%")
#     return accuracy

# Main function
def main():
    # Load training data
    filepath = "Training/all_queries_categorised_train.csv"  # Replace with your dataset path
    data = load_training_data(filepath)
    print("loaded training data")

    # Split into training and validation
    inputs, targets = prepare_data(data)
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs, targets, test_size=0.2)
    print("split training and test set")

    # Fine-tune the model
    model, tokenizer = fine_tune_model(train_inputs, train_targets, val_inputs, val_targets)
    print("fine tuned model")

    # Evaluate the model
    # evaluate_model(model, tokenizer, val_inputs, val_targets)

if __name__ == "__main__":
    main()