from flask import Flask, jsonify, request, render_template
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask_cors import CORS
import threading
import os

# Disable GPU for compatibility
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

CSV_DIR = os.path.join(os.getcwd(), "movie")  
MODEL_DIR = "./models/bart-large-cnn"          


tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
# Global variable to track progress
progress = 0

# Configuration constants
MAX_REVIEWS = 100  # Maximum number of reviews to process
SUMMARY_MAX_LENGTH = 200
SUMMARY_MIN_LENGTH = 100


def summarize_reviews(data):
    """
    Summarize the first MAX_REVIEWS reviews as a single block.
    Dynamically updates progress during processing.
    """
    global progress

    # Reset progress
    progress = 0

    # Combine the first MAX_REVIEWS reviews into a single text
    reviews_to_summarize = " ".join(data['review'][:MAX_REVIEWS])
    progress = 50  # Set progress after combining reviews

    # Tokenize the combined reviews using the tokenizer
    inputs = tokenizer(reviews_to_summarize, return_tensors="pt", truncation=True, padding=True, max_length=1024)

    # Ensure that the input is within the max token length
    input_ids = inputs['input_ids']

    # Summarize the combined reviews
    summary_ids = model.generate(input_ids, max_length=SUMMARY_MAX_LENGTH, min_length=SUMMARY_MIN_LENGTH, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the summarized text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Set progress to 100% as the task is completed
    progress = 100
    return summary


@app.route('/list_files', methods=['GET'])
def list_files():
    """
    List all CSV files in the directory.
    Optionally filter based on a 'keyword' query parameter.
    """
    keyword = request.args.get('keyword', '').lower()
    try:
        # Get all CSV files in the directory
        files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
        
        # Filter by keyword if provided
        if keyword:
            files = [f for f in files if keyword in f.lower()]
        
        # Return the list of files with their full paths
        return jsonify({'files': [{'name': f, 'path': os.path.join(CSV_DIR, f)} for f in files]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles file uploads and starts summarization.
    """
    global progress
    progress = 0  # Reset progress for a new upload

    file_path = request.form.get('filePath')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "Invalid file path"}), 400

    try:
        # Load CSV file
        data = pd.read_csv(file_path)

        if 'review' not in data.columns:
            return jsonify({"error": "'review' column not found in file"}), 400

        if data.empty:
            return jsonify({"error": "File is empty"}), 400

        # Ensure reviews are limited to MAX_REVIEWS
        data = data.head(MAX_REVIEWS)

        # Start summarization in a separate thread
        summary_result = None

        def summarize_task():
            nonlocal summary_result
            summary_result = summarize_reviews(data)

        thread = threading.Thread(target=summarize_task)
        thread.start()
        thread.join()  # Wait for the thread to finish

        return jsonify({"summary": summary_result})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/progress', methods=['GET'])
def get_progress():
    """
    Returns the progress of the summarization process.
    """
    global progress
    return jsonify({"progress": progress})


@app.route('/')
def home():
    """
    Renders the first page (index.html).
    """
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)






