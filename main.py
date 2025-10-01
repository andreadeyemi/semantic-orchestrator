"""FastAPI backend for semantic orchestrator."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pandas as pd
import io
import os

from src.semantic_orchestrator.engine import SemanticEngine
from src.semantic_orchestrator.nlq import NLQProcessor


app = FastAPI(
    title="Semantic Orchestrator API",
    description="CSV ingestion and natural language querying with semantic search",
    version="0.1.0"
)

# Global engine instance
engine: Optional[SemanticEngine] = None


class AskRequest(BaseModel):
    """Request model for ask endpoint."""
    query: str
    top_k: Optional[int] = 5


@app.on_event("startup")
async def startup_event():
    """Initialize the semantic engine on startup."""
    global engine
    engine = SemanticEngine()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Semantic Orchestrator API",
        "version": "0.1.0",
        "endpoints": {
            "/ingest": "POST - Ingest a CSV file",
            "/ask": "POST - Ask a natural language question",
            "/info": "GET - Get information about ingested data"
        }
    }


@app.post("/ingest")
async def ingest_csv(file: UploadFile = File(...)):
    """Ingest a CSV file and create semantic index.
    
    Args:
        file: CSV file to ingest
        
    Returns:
        Ingestion statistics
    """
    global engine
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read the uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Ingest the dataframe
        stats = engine.ingest_dataframe(df)
        
        return {
            "success": True,
            "message": "CSV ingested successfully",
            "statistics": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting CSV: {str(e)}")


@app.post("/ask")
async def ask_question(request: AskRequest):
    """Ask a natural language question about the data.
    
    Args:
        request: Ask request containing the query
        
    Returns:
        Query results with semantic search and NLQ processing
    """
    global engine
    
    if engine is None or engine.dataframe is None:
        raise HTTPException(
            status_code=400, 
            detail="No data ingested. Please ingest a CSV first using /ingest endpoint"
        )
    
    try:
        # First, do semantic search to find relevant rows
        semantic_results = engine.search(request.query, top_k=request.top_k)
        
        # Then, try to process as NLQ
        nlq_processor = NLQProcessor(engine.get_dataframe(), engine.get_column_info())
        nlq_results = nlq_processor.execute_query(request.query)
        
        return {
            "success": True,
            "query": request.query,
            "semantic_search": {
                "top_k": request.top_k,
                "results": semantic_results
            },
            "nlq_processing": nlq_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/info")
async def get_info():
    """Get information about the ingested data.
    
    Returns:
        Information about columns, shape, and sample data
    """
    global engine
    
    if engine is None or engine.dataframe is None:
        raise HTTPException(
            status_code=400,
            detail="No data ingested. Please ingest a CSV first using /ingest endpoint"
        )
    
    try:
        column_info = engine.get_column_info()
        
        return {
            "success": True,
            "data_info": column_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting info: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
