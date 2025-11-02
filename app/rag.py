import sys
import qdrant_client
from typing import List, Dict, Any
from fastapi import HTTPException

from app.reranker import RerankerModel
from app.config import RAGSystemInitializer
rAGSystemInitializer = RAGSystemInitializer()

# Load configuration from config.yaml 
config = rAGSystemInitializer.config
print("INFO: Configuration loaded successfully from config.yaml.")

# Load the RAG prompt template after configuration
RAG_PROMPT_TEMPLATE: str = rAGSystemInitializer.load_rag_prompt_template()
print("INFO: RAG Prompt template loaded from system_prompt.txt.")
    
# Apply LlamaIndex Settings
Settings = rAGSystemInitializer.Settings

# Initialize Qdrant Client globally
qdrant_client = rAGSystemInitializer.qdrant_client
RERANKER_MODEL = RerankerModel()
COLLECTION_NAME = config["QDRANT_COLLECTION_NAME"]


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
