import uvicorn
import time
import os
import sys
import shutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse 
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# --- Import Ingestion Logic & Config Loader ---
from ingest import ingest_documents, load_config

# --- Qdrant Integration ---
from qdrant_client import QdrantClient
from qdrant_client.http.models import CollectionInfo
from qdrant_client.http.exceptions import UnexpectedResponse # Added for explicit error handling

# --- LlamaIndex/Ollama Configuration ---
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# --- Global State & Configuration Setup (STRICTLY FROM CONFIG) ---

try:
    # Load configuration from config.yaml (SINGLE SOURCE OF TRUTH)
    config = load_config()
    print("INFO: Configuration loaded successfully from config.yaml.")
        
    # Apply LlamaIndex Settings
    Settings.llm = Ollama(
        model=config["OLLAMA_LLM_MODEL"], 
        base_url=config["OLLAMA_BASE_URL"], 
        request_timeout=config["OLLAMA_TIMEOUT"]
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=config["OLLAMA_EMBEDDING_MODEL"], 
        base_url=config["OLLAMA_BASE_URL"]
    )

    # Initialize Qdrant Client globally
    qdrant_client = QdrantClient(
        host=config["QDRANT_HOST"], 
        port=config["QDRANT_PORT"]
    )

    COLLECTION_NAME = config["COLLECTION_NAME"]

except Exception as e:
    # If config loading or initialization fails, terminate the script gracefully.
    print(f"FATAL: Application initialization failed due to configuration error: {e}")
    print("Please ensure 'config.yaml' exists and is correctly formatted, and that 'load_config' is functional in 'ingest.py'.")
    sys.exit(1)


# --- Data Models for API ---
class QueryRequest(BaseModel):
    query: str

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]

# --- Utility Functions (Embedding for Query) ---

def get_embedding(text: str) -> List[float]:
    """Generates the embedding for the given query text using the configured Ollama model."""
    try:
        # Relies on global Settings.embed_model initialized above
        return Settings.embed_model.get_text_embedding(text)
    except Exception as e:
        print(f"ERROR: Failed to get embedding for text: {e}")
        raise HTTPException(status_code=500, detail="Embedding generation failed. Check Ollama server.")

# --- Retrieval Utilities (Queries Qdrant) ---

def retrieve_chunks(query_vector: List[float]) -> List[Dict[str, Any]]:
    """Retrieves the top K most relevant chunks by querying Qdrant."""
    
    try:
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=config['TOP_K_CHUNKS'],
            with_payload=True,
        )
    except Exception as e:
        print(f"FATAL: Qdrant search failed: {e}")
        raise ValueError(f"Could could not perform search on Qdrant. Is the collection '{COLLECTION_NAME}' created?")


    retrieved_chunks = []
    
    for i, hit in enumerate(search_result):
        # Debugging: Log the keys of the payload for the first hit
        if i == 0:
            print(f"DEBUG: Keys found in first retrieved payload: {list(hit.payload.keys())}")
            
        try:
            chunk = {
                "score": hit.score,
                "text": hit.payload['_node_content'],  # CORRECTED: Use LlamaIndex node content key
                "source": hit.payload['file_name'] # CORRECTED: Use a reliable file name key for attribution
            }
            retrieved_chunks.append(chunk)
        except KeyError as e:
            # Raise a specific error indicating payload structure issue
            print(f"FATAL: Missing key {e} in Qdrant payload! Actual keys: {list(hit.payload.keys())}")
            raise KeyError(f"Missing expected key '{e}' in Qdrant payload. Keys found: {list(hit.payload.keys())}. Please check that your ingestion process is using '_node_content' for text and 'file_name' for source.")
    
    return retrieved_chunks


def generate_ollama_response(rag_prompt: str) -> str:
    """
    Generates the final answer using the globally configured Ollama LLM via LlamaIndex binding.
    """
    print(f"INFO: Calling Ollama LLM for generation with prompt length {len(rag_prompt)}")
    
    try:
        response = Settings.llm.complete(rag_prompt)
        return response.text
    except Exception as e:
        print(f"ERROR: Ollama LLM generation failed: {e}")
        return "The LLM service is currently unavailable or returned an error."


# --- FastAPI Application Setup ---
app = FastAPI(
    title="Ollama Qdrant RAG API",
    description="A RAG Backend using Ollama for embeddings/LLM and Qdrant for vector storage."
)

# CORS Middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global list kept for health check status reporting
vector_store: List[Any] = []


@app.on_event("startup")
async def check_qdrant_collection():
    """Checks the existence and size of the Qdrant collection on API startup.
    Runs ingestion only if the collection is missing or empty (0 points).
    """
    global vector_store 
    vector_store = [] # Assume empty until confirmed
    
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
            # Update the global list size for the health check endpoint
            vector_store = [{} for _ in range(count)] 
            return # Exit if collection is ready
            
    except UnexpectedResponse as e:
        # This typically catches 404 Not Found error from Qdrant when collection is missing.
        if "Not found" in str(e):
            print(f"WARNING: Qdrant collection '{COLLECTION_NAME}' not found. Triggering ingestion.")
            should_ingest = True
        else:
            print(f"WARNING: Qdrant check failed (Unexpected Response: {e}). Triggering ingestion if possible.")
            should_ingest = True # Try to ingest anyway

    except Exception as e:
        # This block catches connection issues.
        print(f"WARNING: Qdrant connection check failed (connection issue: {e}). Triggering ingestion.")
        should_ingest = True
            
    if should_ingest:
        # Call the imported ingestion function
        ingest_documents() 
        
        # After ingestion, re-check the count for the health endpoint
        # This block remains but is isolated to startup only.
        try:
            info: CollectionInfo = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
            count = info.points_count if info.points_count is not None else 0
            vector_store = [{} for _ in range(count)] 
            print(f"INFO: Ingestion finished. Collection now has {count} points.")
        except Exception as e:
            print(f"ERROR: Failed to verify collection count after ingestion: {e}")
            # vector_store remains [] if re-check fails

# --- API Endpoints ---

@app.get("/", response_class=FileResponse)
async def root():
    """Serves the interactive RAG dashboard (index.html) at the root URL."""
    return FileResponse("index.html")

@app.get("/status")
async def get_current_status():
    """
    API endpoint to provide current system status for the dashboard.
    Distinguishes between 'Collection Missing' and 'Host Unavailable'.
    """
    qdrant_status = "UNAVAILABLE (Check Qdrant Host)"
    vector_size = 0
    
    try:
        # 1. Check if the collection exists and get its status
        info: CollectionInfo = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        vector_size = info.points_count if info.points_count is not None else 0
        
        if vector_size == 0:
            qdrant_status = "EMPTY (Requires Ingestion)"
        else:
            qdrant_status = "READY"
            
    except UnexpectedResponse as e:
        # This typically catches 404 Not Found error from Qdrant when collection is missing.
        if "Not found" in str(e):
             # NEW STATUS: Collection is missing, which means it's empty from a data perspective
             qdrant_status = "EMPTY (Collection Missing)" 
             vector_size = 0
        else:
            # Handle other unexpected errors as potential connection issues
            qdrant_status = "UNAVAILABLE (Qdrant Error)"
            print(f"ERROR: Unexpected Qdrant response: {e}")
    except Exception as e:
        # Catch connection failures or other generic errors
        print(f"ERROR: Failed to connect to Qdrant or generic error: {e}")
        # qdrant_status remains "UNAVAILABLE (Check Qdrant Host)"

    return {
        "qdrant_status": qdrant_status,
        "vector_size": vector_size,
        "llm_model": config['OLLAMA_LLM_MODEL'],
        "embed_model": config['OLLAMA_EMBEDDING_MODEL']
    }

@app.post("/ingest")
async def trigger_ingestion():
    """
    Triggers the document ingestion process, reloading the Qdrant vector store.
    Includes a retry loop to ensure the collection count is available after upload.
    """
    print("INFO: Ingestion API endpoint hit. Starting document ingestion...")
    
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
                # Check for the collection size using the main Qdrant client
                info: CollectionInfo = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
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
                    # Reraise unexpected errors
                    raise 

            except Exception as e:
                # Catch connection errors during polling
                print(f"ERROR during collection verification: {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)

        # 3. Final status update
        global vector_store 
        vector_store = [{} for _ in range(count)] # Update global state
        
        if count == 0:
             # If we tried all retries and count is still 0
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
async def delete_ingestion_data():
    """
    Permanently deletes the Qdrant collection and the local LlamaIndex data directory.
    This effectively resets the entire RAG knowledge base.
    """
    qdrant_deleted = False
    local_data_deleted = False

    print("INFO: Starting data deletion process (Qdrant collection and local index)...")

    # 1. Delete Qdrant Collection
    try:
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        qdrant_deleted = True
        print(f"SUCCESS: Qdrant collection '{COLLECTION_NAME}' deleted.")
    except UnexpectedResponse as e:
        # Qdrant returns 404 if the collection doesn't exist, which is fine for a delete operation
        if "Not found" in str(e):
            qdrant_deleted = True # Treat as successful if it wasn't there
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

    # 3. Update global status (set vector count to 0)
    global vector_store
    vector_store = []
    
    return {
        "message": f"Data deletion successful. Qdrant collection: {'Deleted' if qdrant_deleted else 'Failed'}, Local Index: {'Deleted' if local_data_deleted else 'Failed'}.",
        "vector_store_size": 0
    }


@app.post("/query", response_model=RAGResponse)
async def handle_rag_query(request: QueryRequest):
    """The main RAG endpoint."""
    # Check for collection health based on successful check in startup
    try:
        info: CollectionInfo = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        if info.points_count is None or info.points_count == 0:
             raise HTTPException(status_code=503, detail="Qdrant collection is empty. Please run ingestion or check server logs.")
    except Exception:
        raise HTTPException(status_code=503, detail="Qdrant connection failed or collection is missing. The vector database is unavailable.")

    try:
        query = request.query
        print(f"\n--- API Query Received: '{query}' ---")
        
        # 1. Query Embedding
        query_vector = get_embedding(query)
        print("INFO: Query embedding generated.")

        # 2. Retrieval (Queries Qdrant)
        retrieved_chunks = retrieve_chunks(query_vector)
        print(f"INFO: Retrieved {len(retrieved_chunks)} relevant chunks from Qdrant.")

        # 3. Prompt Construction
        context = "\n---\n".join([c['text'] for c in retrieved_chunks])
        rag_prompt = (
            "You are a helpful and accurate assistant. Use the following "
            "context to answer the user's query. If the context does not "
            "contain the answer, state that you cannot find the answer in the "
            "provided documents.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"USER QUERY: {query}"
        )

        # 4. Generation (LLM Call)
        llm_answer = generate_ollama_response(rag_prompt)

        # 5. Format Response
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
        # Return the specific error message to the client for better debugging
        raise HTTPException(status_code=500, detail=f"Internal server error during RAG process: {e}")


# --- Run the application ---
if __name__ == "__main__":
    # Start the API server. Ingestion is now triggered via the /ingest API endpoint.
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
