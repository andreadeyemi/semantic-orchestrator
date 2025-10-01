# Semantic Orchestrator

A Python application for ingesting CSV files, building semantic indexes with embeddings, and performing natural language queries using Pandas and SQL-like operations.

## Features

- **CSV Ingestion**: Load and index CSV files with semantic embeddings
- **Semantic Search**: Find relevant data using natural language queries and sentence transformers
- **Natural Language Queries (NLQ)**: Convert natural language to Pandas/SQL operations
- **FastAPI Backend**: REST API with `/ingest` and `/ask` endpoints
- **CLI Interface**: Command-line tool for indexing and querying data
- **Modular Architecture**: Clean separation of concerns with engine, NLQ, and utils modules

## Architecture

```
semantic-orchestrator/
├── src/semantic_orchestrator/
│   ├── __init__.py          # Package initialization
│   ├── engine.py            # Semantic indexing engine with sentence transformers
│   ├── nlq.py              # Natural language query processor
│   └── utils.py            # Utility functions
├── main.py                  # FastAPI application
├── cli.py                   # Command-line interface
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/andreadeyemi/semantic-orchestrator.git
cd semantic-orchestrator
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### FastAPI Backend

Start the FastAPI server:

```bash
python main.py
# or
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

#### API Endpoints

**1. Ingest CSV** (`POST /ingest`)
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data.csv"
```

**2. Ask Questions** (`POST /ask`)
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the highest sales?", "top_k": 5}'
```

**3. Get Data Info** (`GET /info`)
```bash
curl "http://localhost:8000/info"
```

### Command-Line Interface

The CLI provides two main commands: `index` and `query`.

#### Index Command

Index a CSV file and optionally enter interactive mode:

```bash
# Basic indexing
python cli.py index data.csv

# Interactive mode
python cli.py index data.csv --interactive

# Custom model and top-k results
python cli.py index data.csv -i --model all-MiniLM-L6-v2 --top-k 10
```

#### Query Command

Execute a one-time query:

```bash
# Basic query
python cli.py query data.csv "What are the top sales?"

# With JSON output
python cli.py query data.csv "Show average prices" --json

# Custom parameters
python cli.py query data.csv "Find products under $50" --top-k 10 --model all-MiniLM-L6-v2
```

### Python API

Use the semantic orchestrator programmatically:

```python
from src.semantic_orchestrator.engine import SemanticEngine
from src.semantic_orchestrator.nlq import NLQProcessor

# Initialize engine
engine = SemanticEngine()

# Ingest CSV
stats = engine.ingest_csv("data.csv")
print(f"Indexed {stats['rows_ingested']} rows")

# Semantic search
results = engine.search("high value customers", top_k=5)
for result in results:
    print(f"Score: {result['similarity_score']}")
    print(f"Data: {result['data']}")

# Natural language query
nlq = NLQProcessor(engine.get_dataframe(), engine.get_column_info())
nlq_results = nlq.execute_query("What is the average price?")
print(nlq_results['pandas_code'])  # Shows equivalent Pandas code
print(nlq_results['results'])
```

## How It Works

### Semantic Engine

The `SemanticEngine` class uses Sentence Transformers to create embeddings for each row in your CSV:

1. **Ingestion**: Converts each row to a text representation
2. **Embedding**: Creates vector embeddings using `all-MiniLM-L6-v2` model
3. **Search**: Compares query embeddings with row embeddings using cosine similarity

### NLQ Processor

The `NLQProcessor` interprets natural language queries:

- **Intent Detection**: Identifies aggregation, search, or filter operations
- **Column Matching**: Maps natural language to column names
- **Code Generation**: Produces equivalent Pandas code
- **Execution**: Runs the query and returns results

Supported query patterns:
- Aggregations: "average", "sum", "count", "max", "min"
- Sorting: "sort by", "order by", "highest", "lowest"
- Limiting: "top 5", "first 10", "limit 3"

## Dependencies

- **fastapi**: Web framework for the API
- **uvicorn**: ASGI server
- **sentence-transformers**: Semantic embeddings
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning utilities
- **pydantic**: Data validation

## Examples

### Example 1: Sales Data

```python
# Index sales data
python cli.py index sales.csv --interactive

Query> What are the highest sales?
Query> Show average revenue by region
Query> Find products with price above $100
```

### Example 2: Customer Data

```python
# Query customer data
python cli.py query customers.csv "Find customers in California"
python cli.py query customers.csv "What is the average age?" --json
```

### Example 3: API Usage

```python
import requests

# Ingest data
with open('data.csv', 'rb') as f:
    response = requests.post('http://localhost:8000/ingest', files={'file': f})
    print(response.json())

# Ask question
response = requests.post('http://localhost:8000/ask', 
    json={'query': 'Show top performers', 'top_k': 5})
print(response.json())
```

## Development

### Project Structure

- `src/semantic_orchestrator/engine.py`: Core semantic indexing engine
- `src/semantic_orchestrator/nlq.py`: Natural language query processing
- `src/semantic_orchestrator/utils.py`: Utility functions for data processing
- `main.py`: FastAPI application with REST endpoints
- `cli.py`: Command-line interface

### Testing

Run the CLI with sample data:

```bash
# Create a sample CSV
echo "name,price,category\nProduct A,29.99,Electronics\nProduct B,49.99,Books\nProduct C,19.99,Electronics" > sample.csv

# Index and query
python cli.py index sample.csv -i
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

André Adeyemi