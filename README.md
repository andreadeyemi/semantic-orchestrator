# semantic-orchestrator

A semantic search orchestrator that builds searchable indexes from CSV data and enables natural language querying.

## Features

- **Ingest CSV data**: Build semantic search indexes from CSV files
- **Natural Language Queries**: Query your data using natural language questions
- **Semantic Search**: Powered by sentence transformers and FAISS for efficient similarity search

## Installation

```bash
pip install -e .
```

## Usage

### Ingest Command

Build a semantic search index from a CSV file:

```bash
semantic-orchestrator ingest data.csv
```

Options:
- `--index-path, -i`: Path to save the index (default: `semantic_index`)
- `--text-column, -c`: Specific column to index (default: all columns concatenated)
- `--model, -m`: Sentence transformer model to use (default: `all-MiniLM-L6-v2`)

Example:
```bash
semantic-orchestrator ingest products.csv --index-path my_index --text-column description
```

### Ask Command

Query the index with a natural language question:

```bash
semantic-orchestrator ask "What are the best laptops?"
```

Options:
- `--index-path, -i`: Path to the index (default: `semantic_index`)
- `--top-k, -k`: Number of results to return (default: 5)
- `--model, -m`: Sentence transformer model to use (default: `all-MiniLM-L6-v2`)

Example:
```bash
semantic-orchestrator ask "affordable smartphones" --top-k 10 --index-path my_index
```

## License

MIT