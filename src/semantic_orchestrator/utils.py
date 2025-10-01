"""Utility functions for semantic orchestrator."""

import pandas as pd
from typing import List, Dict, Any
import numpy as np


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the CSV data
    """
    return pd.read_csv(file_path)


def dataframe_to_text_chunks(df: pd.DataFrame) -> List[str]:
    """Convert DataFrame rows to text chunks for embedding.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        List of text representations of each row
    """
    chunks = []
    for idx, row in df.iterrows():
        # Create a text representation of each row
        text_parts = []
        for col in df.columns:
            value = row[col]
            if pd.notna(value):
                text_parts.append(f"{col}: {value}")
        chunks.append(" | ".join(text_parts))
    return chunks


def get_column_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get metadata about DataFrame columns.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with column names, types, and sample values
    """
    info = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": df.shape,
        "sample_values": {}
    }
    
    for col in df.columns:
        # Get a few non-null sample values
        samples = df[col].dropna().head(3).tolist()
        info["sample_values"][col] = samples
    
    return info


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)
