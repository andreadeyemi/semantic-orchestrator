from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Semantic Orchestrator")

# Global state for storing the index and data
class DataStore:
    def __init__(self):
        self.dataframe: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.text_data: Optional[List[str]] = None
    
    def reset(self):
        self.dataframe = None
        self.embeddings = None
        self.vectorizer = None
        self.text_data = None

data_store = DataStore()

class IngestRequest(BaseModel):
    csv_path: str

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

class IngestResponse(BaseModel):
    message: str
    num_rows: int
    columns: List[str]

class AskResponse(BaseModel):
    question: str
    results: List[Dict]
    scores: List[float]

@app.get("/")
async def root():
    return {
        "message": "Semantic Orchestrator API",
        "endpoints": {
            "/ingest": "POST - Ingest CSV data and build index",
            "/ask": "POST - Ask questions and get semantic search results"
        }
    }

@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Ingest a CSV file and build a semantic search index using TF-IDF vectorization.
    
    Args:
        request: IngestRequest containing the path to the CSV file
    
    Returns:
        IngestResponse with status information
    """
    # Check if file exists
    if not os.path.exists(request.csv_path):
        raise HTTPException(status_code=404, detail=f"CSV file not found: {request.csv_path}")
    
    try:
        # Load CSV data
        df = pd.read_csv(request.csv_path)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Create text representation of each row by concatenating all columns
        # This creates a searchable text from all column values
        text_data = []
        for _, row in df.iterrows():
            row_text = " ".join([f"{col}: {str(val)}" for col, val in row.items()])
            text_data.append(row_text)
        
        # Generate TF-IDF vectors
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(text_data)
        
        # Store in global state
        data_store.dataframe = df
        data_store.embeddings = embeddings
        data_store.vectorizer = vectorizer
        data_store.text_data = text_data
        
        return IngestResponse(
            message="CSV data successfully ingested and indexed",
            num_rows=len(df),
            columns=df.columns.tolist()
        )
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Ask a question and get semantically similar results from the indexed data.
    
    Args:
        request: AskRequest containing the question and optional top_k parameter
    
    Returns:
        AskResponse with matching results and similarity scores
    """
    # Check if index exists
    if data_store.embeddings is None or data_store.dataframe is None or data_store.vectorizer is None:
        raise HTTPException(
            status_code=400, 
            detail="No data has been ingested. Please use /ingest endpoint first."
        )
    
    try:
        # Vectorize the question
        question_vector = data_store.vectorizer.transform([request.question])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(question_vector, data_store.embeddings)[0]
        
        # Get top k results
        k = min(request.top_k, len(data_store.dataframe))
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Get the matching rows and scores
        results = []
        scores = []
        for idx in top_indices:
            row_dict = data_store.dataframe.iloc[idx].to_dict()
            results.append(row_dict)
            scores.append(float(similarities[idx]))
        
        return AskResponse(
            question=request.question,
            results=results,
            scores=scores
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
