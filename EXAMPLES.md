# CLI Implementation Examples

## Commands Overview

The semantic-orchestrator CLI has been successfully implemented with the following commands:

### 1. `ingest` Command
Builds a semantic search index from CSV files.

**Syntax:**
```bash
semantic-orchestrator ingest CSV_FILE [OPTIONS]
```

**Options:**
- `-i, --index-path TEXT`: Path to save the index (default: semantic_index)
- `-c, --text-column TEXT`: Column name to index (default: all columns concatenated)
- `-m, --model TEXT`: Sentence transformer model to use (default: all-MiniLM-L6-v2)

**Example Usage:**
```bash
# Index all columns from a products CSV
semantic-orchestrator ingest products.csv

# Index only the description column
semantic-orchestrator ingest products.csv --text-column description

# Save to a custom index path
semantic-orchestrator ingest products.csv --index-path my_index
```

### 2. `ask` Command
Queries the semantic search index with natural language questions.

**Syntax:**
```bash
semantic-orchestrator ask QUERY [OPTIONS]
```

**Options:**
- `-i, --index-path TEXT`: Path to the index (default: semantic_index)
- `-k, --top-k INTEGER`: Number of results to return (default: 5)
- `-m, --model TEXT`: Sentence transformer model to use (default: all-MiniLM-L6-v2)

**Example Usage:**
```bash
# Simple query
semantic-orchestrator ask "find laptops"

# Get more results
semantic-orchestrator ask "affordable office furniture" --top-k 10

# Use custom index
semantic-orchestrator ask "wireless devices" --index-path my_index
```

## Implementation Details

### Components

1. **cli.py**: Command-line interface using Click framework
   - Defines `ingest` and `ask` commands
   - Handles command-line arguments and options
   - Provides user-friendly error messages

2. **indexer.py**: Semantic indexing engine
   - `SemanticIndex` class for building and querying indexes
   - Uses sentence-transformers for text embeddings
   - Uses FAISS for efficient similarity search
   - Supports saving/loading indexes to/from disk

3. **__init__.py**: Package initialization
   - Defines package version

### Dependencies

- **click**: Command-line interface framework
- **pandas**: CSV data handling
- **sentence-transformers**: Text embedding generation
- **faiss-cpu**: Vector similarity search
- **numpy**: Numerical operations

### Installation

```bash
# Install the package
pip install -e .

# The CLI becomes available as:
semantic-orchestrator
```

## Testing

The CLI has been verified to work correctly:

```bash
$ semantic-orchestrator --help
Usage: semantic-orchestrator [OPTIONS] COMMAND [ARGS]...

  Semantic Orchestrator - Build and query semantic search indexes from CSV
  data.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  ask     Query the semantic search index with a natural language question.
  ingest  Build a semantic search index from a CSV file.

$ semantic-orchestrator --version
semantic-orchestrator, version 0.1.0
```

## Notes

- The first time you run the commands, sentence-transformers will download the specified model (default: all-MiniLM-L6-v2)
- Index files are saved with `.faiss` and `.pkl` extensions
- Indexes can be reused across sessions by specifying the same `--index-path`
