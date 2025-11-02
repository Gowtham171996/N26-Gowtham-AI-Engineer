import uvicorn
import time
import os
import sys
import shutil
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse 
from fastapi.concurrency import asynccontextmanager
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# NEW: Import Gemini LLM and the base LLM type for direct generation calls
# NOTE: Ensure 'llama-index-google-genai' is installed
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import LLM

# Module level imports
from app.ingest import _calculate_and_persist_health_score, ingest_documents
# We keep the imports for model-agnostic retrieval components
from app.rag import get_embedding, retrieve_chunks 
from app.config import RAGSystemInitializer
from app.reranker import RerankerModel # Feature preserved

# --- Qdrant Integration ---
from qdrant_client.http.models import CollectionInfo
from qdrant_client.http.exceptions import UnexpectedResponse 
from qdrant_client import QdrantClient # Need to import the client class for typing

# --- LlamaIndex/Model Configuration ---
from llama_index.core import Settings


# --- Data Models for API ---
class QueryRequest(BaseModel):
    query: str

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]


# --- RAG Helper Function for Generation (NEW: Gemini Specific) ---
def generate_gemini_response(prompt: str) -> str:
    """
    Generates a response from the configured LLM (Gemini 1.5 Flash).
    Relies on Settings.llm being set during the application lifespan/ingest process.
    """
    llm: Optional[LLM] = Settings.llm

    if not llm:
        # This case should ideally not happen if startup succeeded
        raise ValueError("LLM not initialized. Check application startup and ingest logs.")

    try:
        # LlamaIndex's complete method uses the LLM instance from Settings.llm
        response = llm.complete(prompt)
        return str(response)
    except Exception as e:
        print(f"ERROR: Gemini LLM generation failed: {e}")
        # Re-raise as a ValueError to be caught by the endpoint's try-except block
        raise ValueError(f"LLM Generation Error: {e}")


# --- APPLICATION LIFESPAN CONTEXT MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager that runs startup and shutdown logic. All initialization 
    is moved here and stored in the app.state for access by endpoints.
    """
    app.state.index = None
    app.state.config = {}
    app.state.reranker = None
    app.state.is_indexing = False
    qdrant_client: Optional[QdrantClient] = None
    
    print("--- FastAPI Startup: RAG System Initialization ---")
    
    try:
        # Load configuration, initialize Qdrant client, and set up LlamaIndex models
        rAGSystemInitializer = RAGSystemInitializer()
        app.state.config = rAGSystemInitializer.config
        app.state.vector_store = rAGSystemInitializer.vector_store
        app.state.rag_prompt_template = rAGSystemInitializer.load_rag_prompt_template()
        qdrant_client = rAGSystemInitializer.qdrant_client
        COLLECTION_NAME = rAGSystemInitializer.config["QDRANT_COLLECTION_NAME"]
        
        # Load the Reranker model (Feature preserved)
        app.state.reranker = RerankerModel()
        
        # NOTE: ingest_documents() is called once to ensure all global Settings 
        # (LLM/Embedding Model: Gemini) are configured and the index is loaded/updated.
        
        app.state.is_indexing = True
        app.state.qdrant_client = qdrant_client

                # 1. Check if the collection exists and get its status
        info: CollectionInfo = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        vector_size = info.points_count if info.points_count is not None else 0
        
        if vector_size == 0:
            # Call ingest_documents once to set global Settings and load the index
            ingest_documents() 
            
            # Now that Settings are configured globally, load the index using the configured components
            from app.ingest import build_or_update_index # Import locally to avoid circular dependencies
            app.state.index = build_or_update_index(
                app.state.config, 
                app.state.vector_store, 
                Settings.embed_model
            )
        
        app.state.health_score = _calculate_and_persist_health_score(rAGSystemInitializer.config, rAGSystemInitializer.vector_store, Settings.embed_model)
        app.state.is_indexing = False
        print("--- RAG System Ready ---")

    except Exception as e:
        print(f"FATAL STARTUP ERROR: {e}")
        
    yield
    
    # --- Shutdown logic ---
    print("--- FastAPI Shutdown ---")


# --- FASTAPI APP SETUP ---
app = FastAPI(lifespan=lifespan, 
              title="N26 RAG Backend", 
              description="RAG System using Qdrant, LlamaIndex, and Gemini Models.")

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



@app.get("/", response_class=FileResponse)
async def root():
    """Serves the interactive RAG dashboard (index.html) at the root URL."""
    return FileResponse("index.html")

@app.get("/status")
async def get_current_status(request: Request):
    """
    API endpoint to provide current system status for the dashboard.
    Dependencies accessed via request.app.state.
    """
    client: QdrantClient = request.app.state.qdrant_client
    config = request.app.state.config
    COLLECTION_NAME = config["QDRANT_COLLECTION_NAME"]
    
    qdrant_status = "UNAVAILABLE (Check Qdrant Host)"
    vector_size = 0
    
    try:
        # 1. Check if the collection exists and get its status
        info: CollectionInfo = client.get_collection(collection_name=COLLECTION_NAME)
        vector_size = info.points_count if info.points_count is not None else 0
        
        if vector_size == 0:
            qdrant_status = "EMPTY (Requires Ingestion)"
        else:
            qdrant_status = "READY"
        
    except UnexpectedResponse as e:
        if "Not found" in str(e):
             qdrant_status = "EMPTY (Collection Missing)" 
             vector_size = 0
        else:
            qdrant_status = "UNAVAILABLE (Qdrant Error)"
            print(f"ERROR: Unexpected Qdrant response: {e}")
    except Exception as e:
        print(f"ERROR: Failed to connect to Qdrant or generic error: {e}")
        # qdrant_status remains "UNAVAILABLE (Check Qdrant Host)"

    return {
        "qdrant_status": qdrant_status,
        "vector_size": vector_size,
        "llm_model": config['LLM_MODEL'],
        "embed_model": config['EMBEDDING_MODEL'],
        "health_score": 
        {
            "score": request.app.state.health_score["score"],
            "timestamp": request.app.state.health_score["timestamp"],
            "sample_size": request.app.state.health_score["sample_size"],
            "query": request.app.state.health_score["query"]
        }
    }


# --- MANUAL INGESTION ENDPOINT ---
@app.post("/ingest")
async def ingest_trigger():
    """
    Manually triggers the ingestion and indexing pipeline.
    """
    if app.state.is_indexing:
        raise HTTPException(status_code=429, detail="Indexing is already in progress.")

    try:
        app.state.is_indexing = True
        ingest_documents() # This updates the global Settings and rebuilds/updates the index
        
        # Reload the index after ingestion to ensure the app uses the latest version
        from app.ingest import build_or_update_index
        app.state.index = build_or_update_index(
            app.state.config, 
            app.state.vector_store, 
            Settings.embed_model
        )
        
        app.state.is_indexing = False
        
        # Recalculate health score after successful ingestion
        _calculate_and_persist_health_score(
            app.state.config, 
            app.state.vector_store, 
            Settings.embed_model
        )
        
        return {"message": "Ingestion completed successfully. Index reloaded."}
    except Exception as e:
        app.state.is_indexing = False
        print(f"INGESTION ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

@app.delete("/ingest/delete")
async def delete_ingestion_data(request: Request):
    """
    Permanently deletes the Qdrant collection and the local LlamaIndex data directory.
    """
    client: QdrantClient = request.app.state.qdrant_client
    config = request.app.state.config
    COLLECTION_NAME = config["QDRANT_COLLECTION_NAME"]

    qdrant_deleted = False
    local_data_deleted = False

    print("INFO: Starting data deletion process (Qdrant collection and local index)...")

    # 1. Delete Qdrant Collection
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        qdrant_deleted = True
        print(f"SUCCESS: Qdrant collection '{COLLECTION_NAME}' deleted.")
    except UnexpectedResponse as e:
        if "Not found" in str(e):
            qdrant_deleted = True 
            print(f"INFO: Qdrant collection '{COLLECTION_NAME}' not found, treating as deleted.")
        else:
            print(f"ERROR: Failed to delete Qdrant collection (Unexpected Response): {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete Qdrant collection: {e}")
    except Exception as e:
        print(f"ERROR: Failed to delete Qdrant collection (Generic Error): {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete Qdrant collection: {e}")

    # 2. Delete Local LlamaIndex 'qdrant_storage' folder
    persistentPath = config["PERSIST_DIR"]
    if os.path.exists(persistentPath):
        try:
            shutil.rmtree(persistentPath)
            local_data_deleted = True
            print(f"SUCCESS: Local directory '{persistentPath}' deleted.")
        except Exception as e:
            print(f"ERROR: Failed to delete local data directory: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete local data directory: {e}")
    else:
        local_data_deleted = True
        print(f"INFO: Local directory '{persistentPath}' not found, treating as deleted.")

    return {
        "message": f"Data deletion successful. Qdrant collection: {'Deleted' if qdrant_deleted else 'Failed'}, Local Index: {'Deleted' if local_data_deleted else 'Failed'}.",
        "vector_store_size": 0
    }

# --- MAIN RAG ENDPOINT ---
@app.post("/query/rag", response_model=RAGResponse)
async def rag_query(request: QueryRequest):
    """
    Performs the RAG process: Retrieval, Reranking, and Gemini LLM Generation.
    """
    query = request.query

    try:
        # 1. Query Embedding
        query_vector = get_embedding(query)
        print("INFO: Query embedding generated.")

        # 1. Retrieval (uses Settings.embed_model, now Gemini Embedding)
        retrieved_chunks = retrieve_chunks(query_vector)
        
        if not retrieved_chunks:
            # Fallback for empty retrieval
            rag_prompt = f"No relevant information was found in the knowledge base for the query: '{query}'. Please answer only with information you know generally about the topic, or state clearly that no specific context was found."
            
        else:
            # 2. Reranking (Feature preserved)
            RERANKER_MODEL = app.state.reranker
            
            # 3. Rerank the Retrieved Chunks using LLM
            if RERANKER_MODEL and len(retrieved_chunks) > 0:
                print(f"INFO: Starting fast cross-encoder reranking for {len(retrieved_chunks)} chunks.")
                
                # Prepare data for the Cross-Encoder: list of [query, chunk_text] pairs
                query_chunk_pairs = [[query, c['text']] for c in retrieved_chunks]
                
                # 3. Reranking Step: Get new relevance scores using the cross-encoder
                rerank_scores = RERANKER_MODEL.predict(query_chunk_pairs)
                
                # Pair the new scores with the original chunks
                scored_chunks = list(zip(rerank_scores, retrieved_chunks))
                
                # Sort the chunks based on the new rerank_score (highest first)
                scored_chunks.sort(key=lambda x: x[0], reverse=True)
                
                # Get the top N chunks after reranking
                TOP_N_RERANKED = app.state.config["TOP_N_RERANKED"]
                
                # Use the reordered chunks for the context (discarding the temporary score)
                final_context_chunks = [c[1] for c in scored_chunks[:TOP_N_RERANKED]]
                
                print(f"INFO: Reranking complete. Selected Top {len(final_context_chunks)} chunks for context.")
            else:
                final_context_chunks = retrieved_chunks # Use all retrieved if no reranker/no chunks
            
            # 3. Context Formatting (Feature preserved)
            context_text = "\n\n---\n\n".join([c['text'] for c in final_context_chunks])

            # 4. Prompt Engineering
            rag_prompt = app.state.rag_prompt_template.format(context=context_text, query=query)



        # 5. Generation (LLM Call - using the new Gemini function)
        # This replaces generate_ollama_response
        llm_answer = generate_gemini_response(rag_prompt)

        # 6. Format Response (Sources feature preserved)
        sources_list = [
            {"source": c['source'], "similarity_score": f"{c['score']:.4f}"} 
            for c in retrieved_chunks
        ]
        
        return RAGResponse(
            answer=llm_answer,
            sources=sources_list
        )

    except ValueError as e:
        # Catches LLM generation errors
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Passes through existing HTTPExceptions
        raise
    except Exception as e:
        print(f"FATAL RAG ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during RAG process: {e}")


# --- Run the application ---
if __name__ == "__main__":
    # Temporarily instantiate RAGSystemInitializer to get host/port for uvicorn only
    try:
        temp_config_loader = RAGSystemInitializer()
        temp_config = temp_config_loader.config # config is loaded in __init__
    except Exception as e:
        print(f"FATAL: Could not load initial config for uvicorn: {e}")
        sys.exit(1)

    api_host = temp_config.get("API_HOST", "0.0.0.0")
    api_port = temp_config.get("API_PORT", 8000)
    
    uvicorn.run("api:app", host=api_host, port=int(api_port), reload=True)
