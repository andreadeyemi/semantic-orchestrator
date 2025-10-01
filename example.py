#!/usr/bin/env python
"""
Example usage of the Semantic Orchestrator.

This script demonstrates how to use the semantic orchestrator
for CSV ingestion and natural language querying.
"""

from src.semantic_orchestrator.engine import SemanticEngine
from src.semantic_orchestrator.nlq import NLQProcessor
import pandas as pd


def main():
    print("=" * 60)
    print("Semantic Orchestrator - Example Usage")
    print("=" * 60)
    
    # Create sample data
    data = {
        'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
        'price': [1200, 25, 80, 350, 150],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics'],
        'rating': [4.5, 4.0, 4.3, 4.7, 4.2],
        'stock': [10, 50, 30, 15, 25]
    }
    df = pd.DataFrame(data)
    
    print("\n1. Sample Data:")
    print(df.to_string(index=False))
    
    # Initialize the engine
    print("\n2. Initializing Semantic Engine...")
    engine = SemanticEngine()
    
    # Ingest the data
    print("\n3. Ingesting data...")
    stats = engine.ingest_dataframe(df)
    print(f"   - Rows ingested: {stats['rows_ingested']}")
    print(f"   - Columns: {', '.join(stats['columns'])}")
    print(f"   - Embedding method: {stats['embedding_method']}")
    print(f"   - Embedding dimension: {stats['embedding_dimension']}")
    
    # Semantic search examples
    print("\n4. Semantic Search Examples:")
    
    queries = [
        "high-end products",
        "affordable electronics",
        "best rated items"
    ]
    
    for query in queries:
        print(f"\n   Query: '{query}'")
        results = engine.search(query, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result['data']['product']} (score: {result['similarity_score']:.3f})")
    
    # NLQ examples
    print("\n5. Natural Language Query Examples:")
    
    nlq = NLQProcessor(engine.get_dataframe(), engine.get_column_info())
    
    nlq_queries = [
        "What is the average price?",
        "Show the highest rating",
        "Count total products"
    ]
    
    for query in nlq_queries:
        print(f"\n   Query: '{query}'")
        result = nlq.execute_query(query)
        if result['success']:
            print(f"   Intent: {result['parsed_intent']['intent']}")
            print(f"   Pandas code: {result['pandas_code']}")
            if result['results']:
                print(f"   Result: {result['results'][0]}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
