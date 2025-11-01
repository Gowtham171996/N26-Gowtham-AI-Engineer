import uvicorn
import yaml
import json
import time
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse # <-- Added for serving HTML
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# --- Import Ingestion Logic ---
from ingest import ingest_documents


# --- Qdrant Integration ---
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import CollectionInfo

# --- LlamaIndex/Ollama Configuration ---
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# --- Global State & Configuration Setup (STRICTLY FROM CONFIG) ---

# Load configuration from config.yaml (SINGLE SOURCE OF TRUTH)
try:
    with open("config.yaml", 'r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    print("INFO: Configuration loaded successfully from config.yaml.")
except FileNotFoundError:
    print("FATAL: config.yaml not found. The application requires this file and cannot proceed.")
    sys.exit(1)
except Exception as e:
    print(f"FATAL: Failed to load config.yaml: {e}. Cannot proceed.")
    sys.exit(1)
    
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
        raise ValueError(f"Could not perform search on Qdrant. Is the collection '{COLLECTION_NAME}' created?")


    retrieved_chunks = [
        {
            "score": hit.score,
            "text": hit.payload['text'],
            "source": hit.payload['source']
        }
        for hit in search_result
    ]
    
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

# --- HTML Dashboard Utility ---

def get_dashboard_html(qdrant_status: str, llm_model: str, embed_model: str, vector_size: int) -> str:
    """Generates the HTML content for the interactive RAG dashboard."""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Ollama Qdrant RAG Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {{ font-family: 'Inter', sans-serif; }}
        </style>
    </head>
    <body class="bg-gray-100 min-h-screen flex items-center justify-center p-4">
        <div class="bg-white p-8 rounded-xl shadow-2xl max-w-lg w-full">
            <h1 class="text-3xl font-extrabold text-indigo-700 mb-6 border-b pb-2">RAG API Status</h1>
            
            <div class="space-y-3 mb-8">
                <p class="text-gray-700 font-semibold">Service Status: <span class="font-bold text-green-600">ONLINE</span></p>
                <p class="text-gray-700 font-semibold">Qdrant Store: <span class="font-bold text-blue-600">{qdrant_status}</span></p>
                <p class="text-gray-700">Vector Count: <span id="vector-count" class="font-mono text-sm bg-gray-200 px-2 py-0.5 rounded">{vector_size}</span> points</p>
                <p class="text-gray-700">LLM Model: <span class="font-mono text-sm bg-gray-200 px-2 py-0.5 rounded">{llm_model}</span></p>
                <p class="text-gray-700">Embedding Model: <span class="font-mono text-sm bg-gray-200 px-2 py-0.5 rounded">{embed_model}</span></p>
            </div>

            <button id="ingest-button" 
                    class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-lg transition duration-150 ease-in-out shadow-lg transform hover:scale-[1.01]">
                🚀 Trigger Document Ingestion (POST /ingest)
            </button>
            <p id="ingest-message" class="mt-4 text-center text-sm text-gray-500 hidden"></p>

            <script>
                document.getElementById('ingest-button').addEventListener('click', async () => {{
                    const button = document.getElementById('ingest-button');
                    const message = document.getElementById('ingest-message');
                    const vectorCountSpan = document.getElementById('vector-count');
                    const originalText = button.textContent;
                    
                    button.disabled = true;
                    button.textContent = 'Ingesting... Please wait.';
                    button.classList.remove('bg-indigo-600', 'hover:bg-indigo-700', 'bg-yellow-500');
                    button.classList.add('bg-yellow-500');
                    message.textContent = 'Ingestion in progress. Check server logs for details.';
                    message.classList.remove('hidden', 'text-green-600', 'text-red-600');
                    message.classList.add('text-yellow-600');

                    try {{
                        const response = await fetch('/ingest', {{ method: 'POST' }});
                        const data = await response.json();
                        
                        if (response.ok) {{
                            message.textContent = data.message;
                            message.classList.remove('text-yellow-600', 'text-red-600');
                            message.classList.add('text-green-600');
                            vectorCountSpan.textContent = data.vector_store_size;
                        }} else {{
                            message.textContent = 'Error: ' + (data.detail || 'Ingestion failed.');
                            message.classList.remove('text-yellow-600', 'text-green-600');
                            message.classList.add('text-red-600');
                        }}
                    }} catch (error) {{
                        message.textContent = 'Network Error: Could not reach the API.';
                        message.classList.remove('text-yellow-600', 'text-green-600');
                        message.classList.add('text-red-600');
                    }} finally {{
                        button.disabled = false;
                        button.textContent = originalText;
                        button.classList.remove('bg-yellow-500');
                        button.classList.add('bg-indigo-600', 'hover:bg-indigo-700');
                    }}
                }});
            </script>
        </div>
    </body>
    </html>
    """
    return html_content


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
            
    except Exception as e:
        # This block catches errors like 'Collection not found' or connection issues.
        print(f"WARNING: Qdrant collection check failed (likely not found or Qdrant connection issue: {e}). Triggering ingestion.")
        should_ingest = True
            
    if should_ingest:
        # Call the imported ingestion function
        ingest_documents() 
        
        # After ingestion, re-check the count for the health endpoint
        try:
            info: CollectionInfo = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
            count = info.points_count if info.points_count is not None else 0
            vector_store = [{} for _ in range(count)] 
            print(f"INFO: Ingestion finished. Collection now has {count} points.")
        except Exception as e:
            print(f"ERROR: Failed to verify collection count after ingestion: {e}")
            # vector_store remains [] if re-check fails

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serves the interactive RAG dashboard and status at the root URL."""
    
    # 1. Check Qdrant status for HTML report
    qdrant_status = "OK"
    vector_size = 0
    
    try:
        info: CollectionInfo = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        vector_size = info.points_count if info.points_count is not None else 0
        if vector_size == 0:
            qdrant_status = "EMPTY (Requires Ingestion)"
        else:
            qdrant_status = "READY"
    except Exception:
        qdrant_status = "UNAVAILABLE (Check Qdrant Host)"

    # 2. Generate and return HTML
    html_content = get_dashboard_html(
        qdrant_status=qdrant_status,
        llm_model=config['OLLAMA_LLM_MODEL'],
        embed_model=config['OLLAMA_EMBEDDING_MODEL'],
        vector_size=vector_size
    )
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/healthcheck")
async def health_check():
    """Minimal endpoint for automated health checks."""
    
    qdrant_ok = False
    try:
        qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        qdrant_ok = True
    except Exception:
        pass 

    # A simple, fast status response
    return {
        "status": "ok", 
        "qdrant_status": "ready" if qdrant_ok else "unavailable",
        "timestamp": time.time()
    }


@app.post("/ingest")
async def trigger_ingestion():
    """Triggers the document ingestion process, reloading the Qdrant vector store."""
    print("INFO: Ingestion API endpoint hit. Starting document ingestion...")
    try:
        # The ingestion function handles collection recreation and upload.
        ingest_documents()
        
        # After ingestion, immediately re-run the startup check logic 
        # to update the global vector_store count for the health check.
        await check_qdrant_collection() 
        
        return {"message": f"Ingestion completed successfully! {len(vector_store)} points uploaded to Qdrant.", "vector_store_size": len(vector_store)}
    except Exception as e:
        print(f"FATAL INGESTION ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


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
        raise HTTPException(status_code=500, detail="Internal server error during RAG process.")


# --- Run the application ---
if __name__ == "__main__":
    # Start the API server. Ingestion is now triggered via the /ingest API endpoint.
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
