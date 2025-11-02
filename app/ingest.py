import os
import sys
import json
import numpy as np
import shutil 
from datetime import datetime
from typing import Dict, Any, Optional

# Qdrant and LlamaIndex Dependencies
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.indices.base import BaseIndex 
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter 

# NEW: Import Gemini specific models
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# Module level imports
from app.config import RAGSystemInitializer


# --- Helper function for Cosine Similarity ---
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two numpy arrays."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# --- 1. HEALTH SCORE CALCULATION AND PERSISTENCE ---
def _calculate_and_persist_health_score(
    config: Dict[str, Any],
    vector_store: QdrantVectorStore,
    embed_model: BaseEmbedding
):
    """
    Calculates the Index Coherence Score (ICS) by comparing a reference vector
    to a sample of vectors from the Qdrant collection, and saves the score.
    """
    print("\n--- Calculating Index Coherence Score (ICS) ---")
    
    persist_dir = config["PERSIST_DIR"]
    # --- Health Score Constants ---
    HEALTH_FILE =  config["HEALTH_FILE"]
    COHERENCE_QUERY = config["COHERENCE_QUERY"]
    SAMPLE_SIZE = config["SAMPLE_SIZE"] # Number of vectors to sample from Qdrant for health check
    health_filepath = os.path.join(persist_dir, HEALTH_FILE)
    
    try:
        # 1. Generate Reference Vector
        ref_vector = embed_model.get_text_embedding(COHERENCE_QUERY)
        ref_vector_np = np.array(ref_vector)
        
        # 2. Get Qdrant Client (Using the client from the VectorStore)
        qdrant_client = vector_store.client
        collection_name = config["QDRANT_COLLECTION_NAME"]
        
        # 3. Get total count and sample size
        info = qdrant_client.get_collection(collection_name=collection_name)
        total_count = info.points_count or 0
        
        if total_count < SAMPLE_SIZE:
            sample_count = total_count
        else:
            sample_count = SAMPLE_SIZE

        if sample_count == 0:
            print("WARNING: Collection is empty. ICS calculation skipped.")
            return

        # 4. Sample vectors from Qdrant using scroll
        # NOTE: scroll returns a tuple (points, next_offset). We only need the points.
        points_batch, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=sample_count, # Limit the number of points to fetch
            with_vectors=True, # Crucial: Request the vector data
        )
        
        # Ensure points_batch is a list of points
        sampled_vectors = [np.array(p.vector) for p in points_batch if p.vector is not None]
        
        if not sampled_vectors:
             print("WARNING: Failed to retrieve sampled vectors. ICS calculation skipped.")
             return

        # 5. Calculate Average Cosine Similarity (ICS)
        similarities = [cosine_similarity(ref_vector_np, vec) for vec in sampled_vectors]
        avg_similarity = float(np.mean(similarities))
        
        # 6. Persist the score
        score_data = {
            "score": round(avg_similarity, 4),
            "timestamp":  datetime.now().isoformat(),
            "sample_size": len(sampled_vectors),
            "query": COHERENCE_QUERY
        }
        
        with open(health_filepath, 'w') as f:
            json.dump(score_data, f, indent=4)
            
        print(f"ICS calculated and persisted: {score_data['score']:.4f} (Sampled {len(sampled_vectors)} points)")
        
        return score_data
    except Exception as e:
        print(f"ERROR: Failed to calculate or persist Index Coherence Score: {e}")


# --- 2. FULL INDEX BUILDING HELPER ---
def _build_full_index(storage_context: StorageContext, config: Dict[str, Any]) -> BaseIndex:
    """
    Private helper function to perform a complete, clean index build.
    """
    data_dir = config["DATA_DIR"]
    persist_dir = config["PERSIST_DIR"]

    # Load documents 
    documents = SimpleDirectoryReader(data_dir).load_data()
    
    if not documents:
        print(f"WARNING: No documents found in '{data_dir}'. Indexing skipped.")
        return VectorStoreIndex([], storage_context=storage_context) 

    print(f"Successfully loaded {len(documents)} source documents for full index build.")

    # Define the single-layer chunking strategy (Paragraph Splitter)
    SIMPLE_CHUNK_SIZE = config["CHUNK_SIZE"]
    SIMPLE_CHUNK_OVERLAP = config["CHUNK_OVERLAP"]
    splitter = SentenceSplitter(
        chunk_size=SIMPLE_CHUNK_SIZE, 
        chunk_overlap=SIMPLE_CHUNK_OVERLAP,
        paragraph_separator="\n\n"
    )

    # Create nodes
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    
    # Create the VectorStoreIndex using the nodes
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True
    )
    print(f"Full Indexing complete. Vectors stored in Qdrant. Total searchable nodes: {len(nodes)}")
    
    # Persist the LlamaIndex metadata (DocStore, IndexStore)
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"LlamaIndex metadata persisted to {persist_dir}.")
    
    return index


# --- 3. CORE INDEX BUILDING/UPDATING LOGIC (INCREMENTAL) ---
def build_or_update_index(
    config: Dict[str, Any],
    vector_store: QdrantVectorStore,
    embed_model: BaseEmbedding
) -> BaseIndex:
    """
    Handles loading an existing index or building a new one. 
    Crucially performs incremental updates using refresh_ref_docs().
    """
    data_dir = config["DATA_DIR"]
    persist_dir = config["PERSIST_DIR"]
    
    # 1. Setup Environment Check
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        sys.exit("Please place source documents (PDFs, TXT) in the './data' directory and run again.")

    # 2. Configure Node Parser (Chunker)
    SIMPLE_CHUNK_SIZE = config["CHUNK_SIZE"]
    SIMPLE_CHUNK_OVERLAP = config["CHUNK_OVERLAP"]
    splitter = SentenceSplitter(
        chunk_size=SIMPLE_CHUNK_SIZE, 
        chunk_overlap=SIMPLE_CHUNK_OVERLAP,
        paragraph_separator="\n\n"
    )
    
    index: Optional[BaseIndex] = None
    
    # 3. Check for existing LlamaIndex metadata persistence
    if os.path.exists(persist_dir):
        print(f"\n--- 3. Loading Existing Index (Metadata) ---")
        try:
            # Load storage context from persistence directory, linking it to the vector store
            storage_context_loaded = StorageContext.from_defaults(
                persist_dir=persist_dir, 
                vector_store=vector_store
            )
            # Load the index structure
            index = load_index_from_storage(storage_context=storage_context_loaded)
            print("Index metadata loaded successfully.")
            
            # --- INCREMENTAL UPDATE STEP ---
            print("\n--- 4. Checking for New/Modified Documents (Incremental Update) ---")
            
            documents = SimpleDirectoryReader(data_dir).load_data()
            print(f"Loaded {len(documents)} documents for comparison.")
            
            # For incremental updates, we must ensure the index uses the current splitter settings
            index.index_struct.text_splitter = splitter

            # Refresh will only process documents whose content hash has changed or which are entirely new files.
            update_results = index.refresh_ref_docs(documents)
            
            # Calculate how many documents were actually updated or newly added
            num_refreshed = sum([r.is_updated for r in update_results])
            num_deleted = sum([r.is_deleted for r in update_results])

            print(f"Incremental Indexing Complete:")
            print(f"  - {num_refreshed} documents updated or newly indexed.")
            print(f"  - {num_deleted} documents deleted from index.")
            
            # Persist the updated LlamaIndex metadata
            index.storage_context.persist(persist_dir=persist_dir)

        except Exception as e:
            # Fallback for corrupt or incomplete metadata: delete and rebuild
            print(f"WARNING: Error loading index metadata: {e}. Falling back to full rebuild...")
            
            # Clean up corrupted local metadata before rebuilding
            if os.path.exists(persist_dir):
                 # Now safe to use as `import shutil` was added
                 shutil.rmtree(persist_dir) 
                 print(f"Cleaned up corrupted metadata directory: {persist_dir}")
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = _build_full_index(storage_context, config)
            
    else:
        # --- FULL BUILD STEP (First Run) ---
        print(f"\n--- 3. Building Full Index (First Run) ---")
        print("LlamaIndex metadata storage not found. Performing full build...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = _build_full_index(storage_context, config)
    
    return index


# --- 4. MAIN EXECUTION ---
def ingest_documents():
    """Orchestrates the entire ingestion and indexing pipeline."""
    rAGSystemInitializer = RAGSystemInitializer()

    # 1. Load Configuration and Clients
    config = rAGSystemInitializer.config
    vector_store = rAGSystemInitializer.vector_store
    
    # --- MODEL CONFIGURATION FIX (Updated Gemini Setup) ---
    print("\n--- Configuring Gemini Models ---")
    
    # 1. Configure Embedding Model: Gemini Embedding (text-embedding-004 is current alias)
    gemini_embed_model = GeminiEmbedding(
        model_name= config["EMBEDDING_MODEL"]
    )

    Settings.embed_model = gemini_embed_model
    print(f"Embedding Model set to: {Settings.embed_model.model_name}")
    
    # 2. Configure LLM: Gemini 2.5 Flash
    # FIX: Changing from 'gemini-1.5-flash' to 'gemini-2.5-flash' to resolve the 404 error, 
    # as 2.5 Flash is the latest stable and universally available alias.
    gemini_llm = Gemini(
        model=  config["LLM_MODEL"] ,#"gemini-2.5-flash",
        temperature=0.1 # Example: setting a low temperature for factual RAG
    )
    Settings.llm = gemini_llm
    print(f"LLM set to: {Settings.llm.model}")
    # --- END MODEL CONFIGURATION FIX ---
    
    # Use the embed_model from LlamaIndex Settings (which we just set)
    embed_model = Settings.embed_model 

    # 2. Conditional Index Building/Loading (performs the incremental update)
    index = build_or_update_index(config, vector_store, embed_model)
    
    # 3. Calculate and persist the health score
    _calculate_and_persist_health_score(config, vector_store, embed_model)
    
    print("\nSUCCESS: Indexing pipeline execution complete.")

if __name__ == "__main__":
    ingest_documents()
