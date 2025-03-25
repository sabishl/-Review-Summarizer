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

- **GET /list_files**: Lists all CSV files in the `movie` directory. You can filter the results using a `keyword` query parameter.
- **POST /upload**: Uploads a CSV file and starts the summarization process.
- **GET /progress**: Returns the progress of the summarization process.
- **GET /**: Renders the home page.

## Notes
- Ensure that the CSV files are formatted correctly with a `review` column.
- The summarization process runs in a separate thread to keep the application responsive.
