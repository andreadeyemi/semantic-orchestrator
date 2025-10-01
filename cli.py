#!/usr/bin/env python
"""Command-line interface for semantic orchestrator."""

import argparse
import sys
import json
from pathlib import Path

from src.semantic_orchestrator.engine import SemanticEngine
from src.semantic_orchestrator.nlq import NLQProcessor


def index_command(args):
    """Index a CSV file."""
    csv_path = Path(args.csv)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    print(f"Indexing CSV file: {csv_path}")
    print("Initializing semantic engine...")
    
    engine = SemanticEngine(model_name=args.model)
    
    print("Loading and indexing data...")
    stats = engine.ingest_csv(str(csv_path))
    
    print("\nIndexing complete!")
    print(f"Rows indexed: {stats['rows_ingested']}")
    print(f"Columns: {', '.join(stats['columns'])}")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    
    if args.interactive:
        print("\n" + "="*50)
        print("Entering interactive query mode...")
        print("Type 'exit' or 'quit' to exit")
        print("="*50 + "\n")
        
        nlq_processor = NLQProcessor(engine.get_dataframe(), engine.get_column_info())
        
        while True:
            try:
                query = input("Query> ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Perform semantic search
                print("\n--- Semantic Search Results ---")
                results = engine.search(query, top_k=args.top_k)
                
                for i, result in enumerate(results, 1):
                    print(f"\nResult {i} (similarity: {result['similarity_score']:.4f}):")
                    for key, value in result['data'].items():
                        print(f"  {key}: {value}")
                
                # Perform NLQ processing
                print("\n--- NLQ Processing Results ---")
                nlq_result = nlq_processor.execute_query(query)
                
                if nlq_result['success']:
                    print(f"Intent: {nlq_result['parsed_intent']['intent']}")
                    print(f"Pandas code: {nlq_result['pandas_code']}")
                    print(f"Result count: {nlq_result['result_count']}")
                    
                    if nlq_result['results']:
                        print("\nResults:")
                        for i, record in enumerate(nlq_result['results'][:5], 1):
                            print(f"  {i}. {record}")
                else:
                    print(f"Error: {nlq_result.get('error', 'Unknown error')}")
                
                print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def query_command(args):
    """Query indexed data."""
    csv_path = Path(args.csv)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    print(f"Loading CSV file: {csv_path}")
    
    engine = SemanticEngine(model_name=args.model)
    engine.ingest_csv(str(csv_path))
    
    print(f"Executing query: {args.query}\n")
    
    # Semantic search
    print("--- Semantic Search Results ---")
    results = engine.search(args.query, top_k=args.top_k)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (similarity: {result['similarity_score']:.4f}):")
        for key, value in result['data'].items():
            print(f"  {key}: {value}")
    
    # NLQ processing
    print("\n--- NLQ Processing Results ---")
    nlq_processor = NLQProcessor(engine.get_dataframe(), engine.get_column_info())
    nlq_result = nlq_processor.execute_query(args.query)
    
    if nlq_result['success']:
        print(f"Intent: {nlq_result['parsed_intent']['intent']}")
        print(f"Pandas code: {nlq_result['pandas_code']}")
        print(f"Result count: {nlq_result['result_count']}")
        
        if nlq_result['results']:
            print("\nResults:")
            for i, record in enumerate(nlq_result['results'][:10], 1):
                print(f"  {i}. {record}")
    else:
        print(f"Error: {nlq_result.get('error', 'Unknown error')}")
    
    if args.json:
        output = {
            "semantic_results": results,
            "nlq_results": nlq_result
        }
        print("\n--- JSON Output ---")
        print(json.dumps(output, indent=2))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic Orchestrator CLI - Index and query CSV data with natural language"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index a CSV file')
    index_parser.add_argument('csv', type=str, help='Path to CSV file')
    index_parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                             help='Sentence transformer model name')
    index_parser.add_argument('--interactive', '-i', action='store_true',
                             help='Enter interactive query mode after indexing')
    index_parser.add_argument('--top-k', type=int, default=5,
                             help='Number of top results to return (default: 5)')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query a CSV file')
    query_parser.add_argument('csv', type=str, help='Path to CSV file')
    query_parser.add_argument('query', type=str, help='Natural language query')
    query_parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                             help='Sentence transformer model name')
    query_parser.add_argument('--top-k', type=int, default=5,
                             help='Number of top results to return (default: 5)')
    query_parser.add_argument('--json', action='store_true',
                             help='Output results as JSON')
    
    args = parser.parse_args()
    
    if args.command == 'index':
        index_command(args)
    elif args.command == 'query':
        query_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
