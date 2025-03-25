# NN Mini Project

## Overview
This project is a Flask application that summarizes movie reviews using the GEMMA3 model. It allows users to upload CSV files containing reviews, which are then processed and summarized.

## Project Structure
```
NN-mini
├── models
│   └── gemma3
├── movie
├── templates
│   └── index.html
├── 1.py
└── README.md
```

## Requirements
- Python 3.x
- Flask
- Pandas
- Transformers
- Flask-CORS

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd NN-mini
   ```

2. Install the required packages:
   ```
   pip install flask pandas transformers flask-cors
   ```

3. Download the GEMMA3 model and place it in the `models/gemma3` directory.

## Usage
1. Start the Flask application:
   ```
   python 1.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/` to access the application.

3. Use the provided interface to upload a CSV file containing movie reviews. The CSV file should have a column named `review`.

4. After uploading, the application will process the reviews and provide a summarized output.

## Endpoints
- **GET /list_files**: Lists all CSV files in the `movie` directory. You can filter the results using a `keyword` query parameter.
- **POST /upload**: Uploads a CSV file and starts the summarization process.
- **GET /progress**: Returns the progress of the summarization process.
- **GET /**: Renders the home page.

## Notes
- Ensure that the CSV files are formatted correctly with a `review` column.
- The summarization process runs in a separate thread to keep the application responsive.