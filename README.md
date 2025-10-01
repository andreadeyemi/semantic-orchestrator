# semantic-orchestrator

A FastAPI-based semantic search application that allows you to ingest CSV data and perform semantic searches using TF-IDF vectorization and pandas.

## Features

- **CSV Ingestion**: Load CSV files and build a searchable index
- **Semantic Search**: Ask questions and get relevant results based on semantic similarity
- **Pandas Integration**: Results are returned as pandas-compatible dictionaries
- **RESTful API**: Simple HTTP endpoints for ingestion and querying

## Installation

1. Clone the repository:
```bash
git clone https://github.com/andreadeyemi/semantic-orchestrator.git
cd semantic-orchestrator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

Run the FastAPI application:
```bash
python app.py
```

The server will start on `http://localhost:8000`

### API Endpoints

#### 1. Root Endpoint
```bash
GET /
```

Returns information about available endpoints.

#### 2. Ingest CSV Data
```bash
POST /ingest
Content-Type: application/json

{
  "csv_path": "/path/to/your/data.csv"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "/path/to/data.csv"}'
```

**Response:**
```json
{
  "message": "CSV data successfully ingested and indexed",
  "num_rows": 10,
  "columns": ["id", "product", "category", "price", "description"]
}
```

#### 3. Ask Questions
```bash
POST /ask
Content-Type: application/json

{
  "question": "your question here",
  "top_k": 5  // optional, default is 5
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "laptop for work", "top_k": 3}'
```

**Response:**
```json
{
  "question": "laptop for work",
  "results": [
    {
      "id": 2,
      "product": "MacBook Pro",
      "category": "Electronics",
      "price": 2499,
      "description": "High-performance laptop for professionals"
    }
  ],
  "scores": [0.403]
}
```

## How It Works

1. **Ingestion**: The `/ingest` endpoint reads a CSV file, converts each row to a text representation, and creates TF-IDF vectors for semantic indexing.

2. **Querying**: The `/ask` endpoint vectorizes your question using the same TF-IDF model, calculates cosine similarity with all indexed documents, and returns the top-k most similar results with their similarity scores.

3. **Results**: Results are returned as pandas-compatible dictionaries with similarity scores, making it easy to further process the data with pandas.

## Example CSV Format

```csv
id,product,category,price,description
1,iPhone 14,Electronics,999,Latest Apple smartphone with advanced camera
2,MacBook Pro,Electronics,2499,High-performance laptop for professionals
```

## License

MIT License - see LICENSE file for details
