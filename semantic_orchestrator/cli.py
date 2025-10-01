"""Command-line interface for semantic orchestrator."""

import click
import os
from pathlib import Path
from .indexer import SemanticIndex


DEFAULT_INDEX_PATH = "semantic_index"


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Semantic Orchestrator - Build and query semantic search indexes from CSV data."""
    pass


@cli.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.option('--index-path', '-i', default=DEFAULT_INDEX_PATH, 
              help='Path to save the index (default: semantic_index)')
@click.option('--text-column', '-c', default=None,
              help='Column name to index. If not specified, all columns are used.')
@click.option('--model', '-m', default='all-MiniLM-L6-v2',
              help='Sentence transformer model to use (default: all-MiniLM-L6-v2)')
def ingest(csv_file, index_path, text_column, model):
    """Build a semantic search index from a CSV file.
    
    CSV_FILE: Path to the CSV file to index
    """
    click.echo(f"Building index from {csv_file}...")
    
    try:
        # Create index
        index = SemanticIndex(model_name=model)
        
        # Build from CSV
        num_docs = index.build_from_csv(csv_file, text_column=text_column)
        
        # Save index
        index.save(index_path)
        
        click.echo(f"✓ Successfully indexed {num_docs} documents")
        click.echo(f"✓ Index saved to {index_path}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('query')
@click.option('--index-path', '-i', default=DEFAULT_INDEX_PATH,
              help='Path to the index (default: semantic_index)')
@click.option('--top-k', '-k', default=5, type=int,
              help='Number of results to return (default: 5)')
@click.option('--model', '-m', default='all-MiniLM-L6-v2',
              help='Sentence transformer model to use (default: all-MiniLM-L6-v2)')
def ask(query, index_path, top_k, model):
    """Query the semantic search index with a natural language question.
    
    QUERY: Natural language query to search for
    """
    try:
        # Check if index exists
        if not os.path.exists(f"{index_path}.faiss"):
            click.echo(f"✗ Error: Index not found at {index_path}", err=True)
            click.echo("Run 'ingest' command first to build an index", err=True)
            raise click.Abort()
        
        # Load index
        click.echo(f"Loading index from {index_path}...")
        index = SemanticIndex(model_name=model)
        index.load(index_path)
        
        # Search
        click.echo(f"\nSearching for: '{query}'\n")
        results = index.search(query, top_k=top_k)
        
        # Display results
        if not results:
            click.echo("No results found.")
        else:
            for result in results:
                click.echo(f"[{result['rank']}] Score: {result['score']:.4f}")
                click.echo(f"    Text: {result['text'][:200]}{'...' if len(result['text']) > 200 else ''}")
                click.echo(f"    Metadata: {result['metadata']}")
                click.echo()
                
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    cli()
