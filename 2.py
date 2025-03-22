from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Define the local path where the model will be saved
MODEL_DIR = "./models/bart-large-cnn"

# Create the directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Download the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Save the tokenizer and model locally
tokenizer.save_pretrained(MODEL_DIR)
model.save_pretrained(MODEL_DIR)

print(f"Model and tokenizer saved to {MODEL_DIR}")
