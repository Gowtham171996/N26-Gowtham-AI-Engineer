import uvicorn
import time
import os
import sys
import shutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse 
from fastapi.concurrency import asynccontextmanager
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Module level imports
from app.ingest import ingest_documents
from app.rag import generate_ollama_response, get_embedding, retrieve_chunks
from app.config import RAGSystemInitializer
from app.reranker import RerankerModel # Assuming this provides a model object or similar utility

# --- Qdrant Integration ---
from qdrant_client.http.models import CollectionInfo
from qdrant_client.http.exceptions import UnexpectedResponse 
from qdrant_client import QdrantClient # Need to import the client class for typing

# --- LlamaIndex/Ollama Configuration ---
from llama_index.core import Settings


# --- Data Models for API ---
class QueryRequest(BaseModel):
    query: str

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]


# --- APPLICATION LIFESPAN CONTEXT MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager that runs startup and shutdown logic. All initialization 
    is moved here and stored in app.state.
    """
    print("\n--- Running FastAPI Startup Lifespan ---")
    
    # Dependencies initialized inside the try block
    qdrant_client: Optional[QdrantClient] = None
    initializer: Optional[RAGSystemInitializer] = None
    
    try:
        # 1. Instantiate the Initializer, which loads config and sets LlamaIndex settings
        initializer = RAGSystemInitializer()
        
        # 2. Extract necessary components from the initialized object
        qdrant_client = initializer.qdrant_client
        config = initializer.config
        rag_prompt_template = initializer.load_rag_prompt_template()
        reranker_model = RerankerModel() 

        # 3. Store initialized components in the app state for dependency injection
        app.state.config = config
        app.state.qdrant_client = qdrant_client
        app.state.rag_prompt_template = rag_prompt_template
        app.state.reranker_model = reranker_model
        
        # 4. Perform Initial Collection Check & Ingestion (logic moved from @app.on_event)
        COLLECTION_NAME = config["QDRANT_COLLECTION_NAME"]
        should_ingest = False
        
        try:
            # Check if the collection exists and get its status
            info: CollectionInfo = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
            count = info.points_count if info.points_count is not None else 0
            
            if count == 0:
                print(f"WARNING: Qdrant collection '{COLLECTION_NAME}' exists but is EMPTY (0 points). Triggering ingestion.")
                should_ingest = True
            else:
                print(f"SUCCESS: Qdrant collection '{COLLECTION_NAME}' found with {count} points.")
                
        except UnexpectedResponse as e:
            # Catches 404 Not Found error from Qdrant when collection is missing.
            if "Not found" in str(e):
                print(f"WARNING: Qdrant collection '{COLLECTION_NAME}' not found. Triggering ingestion.")
                should_ingest = True
            else:
                print(f"WARNING: Qdrant check failed (Unexpected Response: {e}). Trying ingestion.")
                should_ingest = True 

        except Exception as e:
            print(f"WARNING: Qdrant connection check failed (connection issue: {e}). Trying ingestion.")
            should_ingest = True
                
        if should_ingest:
            # Call the imported ingestion function
            ingest_documents() 
            print(f"INFO: Initial ingestion finished.")

        print("INFO: All RAG components successfully initialized and stored in app state.")
        
    except Exception as e:
        print(f"CRITICAL STARTUP FAILURE during lifespan: {e}")
        # Re-raise the exception to prevent the server from starting if initialization failed
        sys.exit(1)

    # Yield control to the application to handle requests
    yield
    
    # --- Shutdown Logic (runs when the application is shutting down) ---
    print("\n--- Running FastAPI Shutdown Lifespan ---")
    # Add any necessary cleanup here (e.g., closing client connections if necessary)
    print("INFO: Shutdown sequence completed.")


# --- Fast API Setup: Pass the lifespan function ---
app = FastAPI(
    title="Ollama Qdrant RAG API",
    description="A RAG Backend using Ollama for embeddings/LLM and Qdrant for vector storage.",
    lifespan=lifespan # Use the modern lifespan event handler
)

# CORS Middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- API Endpoints ---

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
        "llm_model": config['OLLAMA_LLM_MODEL'],
        "embed_model": config['OLLAMA_EMBEDDING_MODEL']
    }

@app.post("/ingest")
async def trigger_ingestion(request: Request):
    """
    Triggers the document ingestion process, reloading the Qdrant vector store.
    """
    print("INFO: Ingestion API endpoint hit. Starting document ingestion...")
    
    client: QdrantClient = request.app.state.qdrant_client
    config = request.app.state.config
    COLLECTION_NAME = config["QDRANT_COLLECTION_NAME"]

    MAX_RETRIES = 5
    RETRY_DELAY = 1.0 # seconds
    
    try:
        # 1. The ingestion function handles collection recreation and upload.
        ingest_documents()
        
        print("INFO: Ingestion reported complete. Verifying collection status...")
        
        # 2. Polling loop to wait for Qdrant collection count to be updated/visible
        count = 0
        for attempt in range(MAX_RETRIES):
            try:
                info: CollectionInfo = client.get_collection(collection_name=COLLECTION_NAME)
                count = info.points_count if info.points_count is not None else 0
                
                if count > 0:
                    print(f"SUCCESS: Collection verified after {attempt+1} attempts with {count} points.")
                    break
                else:
                    print(f"WARNING: Collection found but count is 0 on attempt {attempt+1}. Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
            
            except UnexpectedResponse as e:
                if "Not found" in str(e):
                    print(f"WARNING: Collection not yet visible on attempt {attempt+1}. Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise 

            except Exception as e:
                print(f"ERROR during collection verification: {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
        
        if count == 0:
             error_msg = "Ingestion completed but collection count is 0 after retries. Check Qdrant logs."
             print(f"FATAL: {error_msg}")
             raise HTTPException(status_code=500, detail=error_msg)
        
        return {"message": f"Ingestion completed successfully! Collection now verified with {count} points.", "vector_store_size": count}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"FATAL INGESTION ERROR: {e}")
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

@app.post("/query/rag", response_model=RAGResponse)
async def handle_rag_query(request: Request, api_request: QueryRequest):
    """The main RAG endpoint."""
    
    client: QdrantClient = request.app.state.qdrant_client
    config = request.app.state.config
    COLLECTION_NAME = config["QDRANT_COLLECTION_NAME"]
    RERANKER_MODEL = request.app.state.reranker_model
    RAG_PROMPT_TEMPLATE = request.app.state.rag_prompt_template

    # Health check for Qdrant *before* running RAG
    try:
        info: CollectionInfo = client.get_collection(collection_name=COLLECTION_NAME)
        if info.points_count is None or info.points_count == 0:
             raise HTTPException(status_code=503, detail="Qdrant collection is empty. Please run ingestion or check server logs.")
    except Exception:
        raise HTTPException(status_code=503, detail="Qdrant connection failed or collection is missing. The vector database is unavailable.")

    try:
        query = api_request.query
        print(f"\n--- API Query Received: '{query}' ---")
        
        # 1. Query Embedding
        query_vector = get_embedding(query)
        print("INFO: Query embedding generated.")

        # 2. Retrieval (Queries Qdrant)
        retrieved_chunks = retrieve_chunks(query_vector)
        print(f"INFO: Retrieved {len(retrieved_chunks)} relevant chunks from Qdrant.")

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
            TOP_N_RERANKED = config["TOP_N_RERANKED"]
            
            # Use the reordered chunks for the context (discarding the temporary score)
            final_context_chunks = [c[1] for c in scored_chunks[:TOP_N_RERANKED]]
            
            print(f"INFO: Reranking complete. Selected Top {len(final_context_chunks)} chunks for context.")
        else:
             final_context_chunks = retrieved_chunks # Use all retrieved if no reranker/no chunks
        
        # 4. Prompt Construction
        # Use the *reranked/selected* chunks for the context
        context = "\n---\n".join([c['text'] for c in final_context_chunks])
        
        if not RAG_PROMPT_TEMPLATE:
             raise RuntimeError("RAG Prompt Template failed to load during startup.")

        rag_prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=query)

        # 5. Generation (LLM Call)
        llm_answer = generate_ollama_response(rag_prompt)

        # 6. Format Response (Sources should include *all* retrieved chunks for transparency)
        sources_list = [
            {"source": c['source'], "similarity_score": f"{c['score']:.4f}"} 
            for c in retrieved_chunks
        ]
        
        return RAGResponse(
            answer=llm_answer,
            sources=sources_list
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
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

    print(f"Starting API server on {api_host}:{api_port}")
    # The application itself is started by referencing the app object created above
    uvicorn.run(app, host=api_host, port=api_port)
