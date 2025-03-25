from flask import Flask, jsonify, request, render_template
import pandas as pd
from flask_cors import CORS
import os
import requests
import logging

# Disable GPU (if needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Configuration
CSV_DIR = os.path.join(os.getcwd(), "movie")  # Directory containing CSV files
OLLAMA_URL = "http://localhost:11434"  # Ollama API URL
MODEL_NAME = "gemma3:4b"  # Ollama Model

# Global progress tracker
progress = 0

# Constants
MAX_REVIEWS = 100  # Max reviews to summarize
SUMMARY_MAX_LENGTH = 200
SUMMARY_MIN_LENGTH = 100

# Logging setup
logging.basicConfig(level=logging.DEBUG)


def summarize_reviews(data):
    """Summarizes the reviews using Ollama API."""
    global progress
    progress = 0  # Reset progress

    # Combine reviews into a single text block
    reviews_to_summarize = " ".join(data['review'][:MAX_REVIEWS])[:5000]  # Limit characters
    progress = 50  # Halfway through processing

    # Define the prompt for Ollama
    prompt = f"Summarize the following movie reviews:\n\n{reviews_to_summarize}"

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=30
        )
        response.raise_for_status()

        # Log and print raw response for debugging
        logging.debug(f"Raw Ollama Response: {response.text}")
        print(f"Raw Ollama Response: {response.text}")

        try:
            # Attempt JSON parsing
            result = response.json()
            summary = result.get("response", "").strip()

            if not summary:
                raise ValueError("Empty response from Ollama")

            progress = 100  # Mark completion
            return summary

        except ValueError as e:
            logging.error(f"Error parsing JSON response: {e}")
            return f"Error parsing JSON: {response.text[:500]}"

    except requests.exceptions.RequestException as e:
        logging.error(f"Error communicating with Ollama: {e}")
        return "Error communicating with Ollama."


@app.route('/list_files', methods=['GET'])
def list_files():
    """Lists CSV files in the movie directory."""
    keyword = request.args.get('keyword', '').lower()  # Optional keyword filter

    try:
        if not os.path.exists(CSV_DIR):
            return jsonify({'error': f"Directory '{CSV_DIR}' does not exist"}), 400

        files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]

        # Filter by keyword (if provided)
        if keyword:
            files = [f for f in files if keyword in f.lower()]

        return jsonify({'files': [{'name': f, 'path': os.path.join(CSV_DIR, f)} for f in files]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload and summarizes reviews."""
    global progress
    progress = 0  # Reset progress

    file_path = request.form.get('filePath')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "Invalid file path"}), 400

    try:
        # Load CSV data
        data = pd.read_csv(file_path)

        if 'review' not in data.columns:
            return jsonify({"error": "'review' column not found in file"}), 400

        if data.empty:
            return jsonify({"error": "File is empty"}), 400

        # Limit reviews for processing
        data = data.head(MAX_REVIEWS)

        # Generate summary
        summary = summarize_reviews(data)

        return jsonify({"summary": summary})

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/progress', methods=['GET'])
def get_progress():
    """Returns the summarization progress."""
    global progress
    return jsonify({"progress": progress})


@app.route('/')
def home():
    """Renders the web page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
