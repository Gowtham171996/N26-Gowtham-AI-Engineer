from sentence_transformers import CrossEncoder

from app.config import RAGSystemInitializer
rAGSystemInitializer = RAGSystemInitializer()

def RerankerModel() -> CrossEncoder:
    try:
        config = rAGSystemInitializer.config
        print("INFO: Reranker: Configuration loaded successfully from config.yaml.")
        
        # Use a small, distilled cross-encoder for fast reranking
        # A widely used, efficient model is 'cross-encoder/ms-marco-MiniLM-L-6-v2' (~20M params)
        RERANKER_MODEL = CrossEncoder(
            model_name=config["RERANKER_MODEL"], 
            max_length=config["CHUNK_SIZE"] # Set a reasonable max length for the concatenated query+chunk
         
        )
        print("INFO: Cross-Encoder Reranker 'ms-marco-MiniLM-L-6-v2' loaded successfully.")
        
    except Exception as e:
        # Reranking is often optional, but if it fails to load, log it.
        print(f"WARNING: Failed to load Cross-Encoder Reranker: {e}")
        RERANKER_MODEL = None # Set to None to allow the system to fall back to no reranking
    
    return RERANKER_MODEL