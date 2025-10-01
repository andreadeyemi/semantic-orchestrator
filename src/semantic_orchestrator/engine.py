"""Semantic indexing engine using sentence transformers."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .utils import dataframe_to_text_chunks, get_column_info, cosine_similarity

# Try to import sentence transformers, fall back to TF-IDF if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import sklearn for TF-IDF fallback
from sklearn.feature_extraction.text import TfidfVectorizer


class SemanticEngine:
    """Semantic indexing engine for CSV data."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the semantic engine.
        
        Args:
            model_name: Name of the sentence transformer model to use (if available)
        """
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self.use_transformers = False
        
        # Try to initialize sentence transformers
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self.use_transformers = True
                print(f"Using Sentence Transformers model: {model_name}")
            except Exception as e:
                print(f"Could not load Sentence Transformers model: {e}")
                print("Falling back to TF-IDF vectorization")
                self.vectorizer = TfidfVectorizer(max_features=384)
        else:
            print("Sentence Transformers not available, using TF-IDF vectorization")
            self.vectorizer = TfidfVectorizer(max_features=384)
        
        self.dataframe: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.text_chunks: Optional[List[str]] = None
        self.column_info: Optional[Dict[str, Any]] = None
        
    def ingest_csv(self, file_path: str) -> Dict[str, Any]:
        """Ingest a CSV file and create embeddings.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary with ingestion statistics
        """
        # Load the CSV
        self.dataframe = pd.read_csv(file_path)
        
        # Get column information
        self.column_info = get_column_info(self.dataframe)
        
        # Convert to text chunks
        self.text_chunks = dataframe_to_text_chunks(self.dataframe)
        
        # Create embeddings
        if self.use_transformers and self.model:
            self.embeddings = self.model.encode(self.text_chunks, show_progress_bar=True)
        else:
            # Use TF-IDF
            self.embeddings = self.vectorizer.fit_transform(self.text_chunks).toarray()
        
        return {
            "rows_ingested": len(self.dataframe),
            "columns": list(self.dataframe.columns),
            "embedding_dimension": self.embeddings.shape[1],
            "embedding_method": "sentence-transformers" if self.use_transformers else "tfidf"
        }
    
    def ingest_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Ingest a pandas DataFrame and create embeddings.
        
        Args:
            df: DataFrame to ingest
            
        Returns:
            Dictionary with ingestion statistics
        """
        self.dataframe = df
        
        # Get column information
        self.column_info = get_column_info(self.dataframe)
        
        # Convert to text chunks
        self.text_chunks = dataframe_to_text_chunks(self.dataframe)
        
        # Create embeddings
        if self.use_transformers and self.model:
            self.embeddings = self.model.encode(self.text_chunks, show_progress_bar=True)
        else:
            # Use TF-IDF
            self.embeddings = self.vectorizer.fit_transform(self.text_chunks).toarray()
        
        return {
            "rows_ingested": len(self.dataframe),
            "columns": list(self.dataframe.columns),
            "embedding_dimension": self.embeddings.shape[1],
            "embedding_method": "sentence-transformers" if self.use_transformers else "tfidf"
        }
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant rows using semantic similarity.
        
        Args:
            query: Natural language query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing row data and similarity scores
        """
        if self.embeddings is None or self.dataframe is None:
            raise ValueError("No data ingested. Please ingest a CSV first.")
        
        # Encode the query
        if self.use_transformers and self.model:
            query_embedding = self.model.encode([query])[0]
        else:
            # Use TF-IDF
            query_embedding = self.vectorizer.transform([query]).toarray()[0]
        
        # Calculate similarities
        similarities = []
        for idx, emb in enumerate(self.embeddings):
            sim = cosine_similarity(query_embedding, emb)
            similarities.append((idx, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k results
        results = []
        for idx, score in similarities[:top_k]:
            row_data = self.dataframe.iloc[idx].to_dict()
            results.append({
                "row_index": int(idx),
                "similarity_score": float(score),
                "data": row_data
            })
        
        return results
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get the ingested DataFrame.
        
        Returns:
            The ingested DataFrame
        """
        if self.dataframe is None:
            raise ValueError("No data ingested.")
        return self.dataframe
    
    def get_column_info(self) -> Dict[str, Any]:
        """Get column information.
        
        Returns:
            Dictionary with column metadata
        """
        if self.column_info is None:
            raise ValueError("No data ingested.")
        return self.column_info
