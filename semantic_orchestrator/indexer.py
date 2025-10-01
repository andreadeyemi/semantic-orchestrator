"""Index builder for semantic search on CSV data."""

import os
import pickle
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class SemanticIndex:
    """Build and manage a semantic search index from CSV data."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the semantic index with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.metadata = []
        
    def build_from_csv(self, csv_path: str, text_column: str = None) -> int:
        """Build semantic index from a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            text_column: Name of the column to index. If None, all columns are concatenated
            
        Returns:
            Number of documents indexed
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        # Prepare text for indexing
        if text_column:
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in CSV")
            texts = df[text_column].astype(str).tolist()
        else:
            # Concatenate all columns
            texts = df.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist()
        
        # Store documents and metadata
        self.documents = texts
        self.metadata = df.to_dict('records')
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        return len(texts)
    
    def save(self, index_path: str):
        """Save the index to disk.
        
        Args:
            index_path: Path to save the index
        """
        os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else '.', exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{index_path}.faiss")
        
        # Save documents and metadata
        with open(f"{index_path}.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
    
    def load(self, index_path: str):
        """Load the index from disk.
        
        Args:
            index_path: Path to load the index from
        """
        # Load FAISS index
        self.index = faiss.read_index(f"{index_path}.faiss")
        
        # Load documents and metadata
        with open(f"{index_path}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the index with a natural language query.
        
        Args:
            query: Natural language query
            top_k: Number of top results to return
            
        Returns:
            List of search results with metadata and scores
        """
        if self.index is None:
            raise ValueError("Index not built or loaded")
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Prepare results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    'rank': i + 1,
                    'score': float(dist),
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx]
                })
        
        return results
