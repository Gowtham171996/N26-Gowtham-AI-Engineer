import os
import sys

# Qdrant and LlamaIndex Dependencies
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage 
)
from llama_index.core.indices.base import BaseIndex # Used for typing
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Simple indexing imports
from llama_index.core.node_parser import SentenceSplitter 
from typing import Dict, Any

# Module level imports
from app.config import RAGSystemInitializer


# --- 1. CORE INDEX BUILDING/UPDATING LOGIC ---
def build_or_update_index(
    storage_context: StorageContext, 
    config: Dict[str, Any],
    vector_store: QdrantVectorStore,
) -> BaseIndex:
    """
    Loads documents, determines if a full index needs to be built (if no metadata exists),
    or updates an existing index incrementally (for new/modified files).
    """
    data_dir = config["DATA_DIR"]
    persist_dir = config["PERSIST_DIR"]
    
    # 1. Setup Environment
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        sys.exit("Please place source documents (PDFs, TXT) in the './data' directory and run again.")

    # 2. Configure Node Parser (Chunker)
    SIMPLE_CHUNK_SIZE = config["CHUNK_SIZE"]
    SIMPLE_CHUNK_OVERLAP = config["CHUNK_OVERLAP"]
    splitter = SentenceSplitter(chunk_size=SIMPLE_CHUNK_SIZE, chunk_overlap=SIMPLE_CHUNK_OVERLAP) 
    
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
            index: BaseIndex = load_index_from_storage(storage_context=storage_context_loaded)
            print("Index metadata loaded successfully.")
            
            # --- INCREMENTAL UPDATE STEP ---
            print("\n--- 4. Checking for New/Modified Documents (Incremental Update) ---")
            
            # This is the key change: we compare the documents in the folder 
            # against the index's document store metadata.
            documents = SimpleDirectoryReader(data_dir).load_data()
            
            # Refresh will only process documents whose content hash has changed 
            # or which are entirely new files.
            print(f"Loaded {len(documents)} documents for comparison.")
            
            # Set the new parser before refresh
            index.index_struct.text_splitter = splitter

            update_results = index.refresh_ref_docs(documents)
            
            # Calculate how many documents were actually updated or newly added
            num_refreshed = sum([r.is_updated for r in update_results])
            num_deleted = sum([r.is_deleted for r in update_results])

            print(f"Incremental Indexing Complete:")
            print(f"  - {num_refreshed} documents updated or newly indexed.")
            print(f"  - {num_deleted} documents deleted from index.")
            
            # Persist the updated LlamaIndex metadata
            index.storage_context.persist(persist_dir=persist_dir)
            
            return index

        except Exception as e:
            # Fallback for corrupt or incomplete metadata: delete and rebuild
            print(f"WARNING: Error loading index metadata: {e}. Falling back to full rebuild...")
            
            # Clean up corrupted local metadata before rebuilding
            if os.path.exists(persist_dir):
                 import shutil
                 shutil.rmtree(persist_dir)
                 print(f"Cleaned up corrupted metadata directory: {persist_dir}")
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            return _build_full_index(storage_context, config)
            
    else:
        # --- FULL BUILD STEP (First Run) ---
        print(f"\n--- 3. Building Full Index (First Run) ---")
        print("LlamaIndex metadata storage not found. Performing full build...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return _build_full_index(storage_context, config)


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

    # Define the single-layer chunking strategy (Simple SentenceSplitter)
    SIMPLE_CHUNK_SIZE = config["CHUNK_SIZE"]
    SIMPLE_CHUNK_OVERLAP = config["CHUNK_OVERLAP"]
    splitter = SentenceSplitter(chunk_size=SIMPLE_CHUNK_SIZE, chunk_overlap=SIMPLE_CHUNK_OVERLAP) 

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


# --- 5. MAIN EXECUTION ---
def ingest_documents():
    """Orchestrates the entire ingestion and indexing pipeline."""
    rAGSystemInitializer = RAGSystemInitializer()

    # 1. Load Configuration
    config = rAGSystemInitializer.config
    
    # 2. Setup Clients & Settings (returns only the QdrantVectorStore)
    vector_store = rAGSystemInitializer.vector_store
    
    # 3. Conditional Index Building/Loading
    build_or_update_index(
        storage_context=StorageContext.from_defaults(vector_store=vector_store), # Initial context for fresh builds
        config=config, 
        vector_store=vector_store
    )
    
    print("\nSUCCESS: Indexing pipeline execution complete.")

if __name__ == "__main__":
    ingest_documents()
