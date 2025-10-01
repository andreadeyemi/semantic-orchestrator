# Quick Start Guide

## Installation

```bash
git clone https://github.com/andreadeyemi/semantic-orchestrator.git
cd semantic-orchestrator
pip install -r requirements.txt
```

## Quick Examples

### 1. CLI - Interactive Mode

```bash
python cli.py index sample.csv --interactive
```

Then try queries like:
- `books`
- `electronics products`
- `What is the average price?`
- `Show highest rating`
- `exit` (to quit)

### 2. CLI - One-time Query

```bash
python cli.py query sample.csv "show expensive items"
```

### 3. FastAPI Server

**Start the server:**
```bash
python main.py
```

**Ingest CSV:**
```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@sample.csv"
```

**Ask a question:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "high value items", "top_k": 5}'
```

**Get data info:**
```bash
curl "http://localhost:8000/info"
```

### 4. Python API

```python
from src.semantic_orchestrator.engine import SemanticEngine
from src.semantic_orchestrator.nlq import NLQProcessor

# Initialize and load data
engine = SemanticEngine()
engine.ingest_csv("sample.csv")

# Semantic search
results = engine.search("electronics", top_k=3)
for r in results:
    print(f"Score: {r['similarity_score']:.3f}, Data: {r['data']}")

# Natural language query
nlq = NLQProcessor(engine.get_dataframe(), engine.get_column_info())
result = nlq.execute_query("What is the average price?")
print(result['pandas_code'])  # Shows: df[['price']].mean()
print(result['results'])
```

## Supported Query Types

### Semantic Search Queries
- Product/category names: `"books"`, `"electronics"`, `"furniture"`
- Descriptive terms: `"affordable items"`, `"high-end products"`
- Any text that might appear in your data

### Natural Language Queries

**Aggregations:**
- `"What is the average price?"`
- `"Show the total stock"`
- `"Count the products"`
- `"Find the maximum rating"`
- `"Get the minimum price"`

**Filtering with columns:**
- `"Show me the price column"`
- `"Display rating and stock"`

**Sorting:**
- `"Sort by price ascending"`
- `"Order by rating descending"`

**Limiting:**
- `"Show top 5 products"`
- `"Get first 10 items"`
- `"Limit to 3 results"`

## Sample CSV Format

```csv
name,price,category,rating,stock
Laptop Pro,1299.99,Electronics,4.5,15
Wireless Mouse,29.99,Electronics,4.2,50
Python Book,49.99,Books,4.8,30
```

## Troubleshooting

**Issue: Slow model loading**
- The system uses TF-IDF fallback if Sentence Transformers can't download
- This is normal in offline/restricted environments
- TF-IDF still provides good keyword-based similarity

**Issue: "No data ingested" error**
- Make sure to call `/ingest` endpoint or `ingest_csv()` first
- Check that your CSV file exists and is readable

**Issue: Low similarity scores**
- TF-IDF is keyword-based, try queries with exact words from your data
- For better semantic understanding, use environment with internet access

## Advanced Usage

### Custom Model Selection
```bash
python cli.py index data.csv --model all-mpnet-base-v2
```

### JSON Output
```bash
python cli.py query data.csv "show data" --json > results.json
```

### Programmatic DataFrame Ingestion
```python
import pandas as pd

df = pd.read_csv("data.csv")
# ... process dataframe ...
engine.ingest_dataframe(df)
```

## API Documentation

When the server is running, visit:
- Interactive docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`
