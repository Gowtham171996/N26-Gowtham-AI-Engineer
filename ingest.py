import yaml
import os
import sys

# Qdrant and LlamaIndex Dependencies
from qdrant_client import QdrantClient
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage 
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Simple indexing imports
from llama_index.core.node_parser import SentenceSplitter 
from typing import Dict, Any

# --- Module-level Constant for Config File ---
CONFIG_FILE_NAME = "config.yaml"

# --- 1. CONFIGURATION LOADING ---
def load_config() -> Dict[str, Any]:
    """Loads configuration from config.yaml."""
    try:
        # FIX: Directly use the constant CONFIG_FILE_NAME instead of reading from an uninitialized 'config' variable
        with open(CONFIG_FILE_NAME, 'r') as f: 
            config = yaml.safe_load(f)
        print("INFO: Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"FATAL: {CONFIG_FILE_NAME} not found. Cannot proceed.")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL: Error loading {CONFIG_FILE_NAME}: {e}")
        sys.exit(1)

# --- 2. INITIALIZE CLIENTS & SETTINGS ---
def initialize_clients_and_settings(config: Dict[str, Any]) -> QdrantVectorStore:
    """Initializes LlamaIndex settings, Qdrant client, and returns the VectorStore instance."""
    print("\n--- 2. Initializing Clients and Settings ---")

    # Set LlamaIndex Settings
    Settings.llm = Ollama(
        model=config["OLLAMA_LLM_MODEL"], 
        base_url=config["OLLAMA_BASE_URL"], 
        request_timeout=config["OLLAMA_TIMEOUT"]
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=config["OLLAMA_EMBEDDING_MODEL"], 
        base_url=config["OLLAMA_BASE_URL"]
    )
    Settings.num_workers = config["OLLAMA_WORKERS"] 
    print(f"LlamaIndex Settings configured. LLM: {config['OLLAMA_LLM_MODEL']}")

    # Initialize Qdrant Client and Vector Store 
    try:
        qdrant_client = QdrantClient(host=config["QDRANT_HOST"], port=config["QDRANT_PORT"])
        qdrant_client.get_collections() # Check connection
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=config["COLLECTION_NAME"]
        )
        print(f"Qdrant connection successful. Collection: {config['COLLECTION_NAME']}")
        return vector_store
    except Exception as e:
        print(f"FATAL: Could not connect to Qdrant at {config['QDRANT_HOST']}:{config['QDRANT_PORT']}. Is the service running?")
        print(f"Error: {e}")
        sys.exit(1)


# --- 3. INDEXING (INGESTION) - SIMPLE ARCHITECTURE (Paragraph/Sentence Chunking) ---
def build_simple_index(storage_context: StorageContext, config: Dict[str, Any]) -> VectorStoreIndex:
    """
    Loads documents using SimpleDirectoryReader defaults, performs single-layer 
    SentenceSplitter chunking, builds the VectorStoreIndex, and persists metadata.
    """
    data_dir = config["DATA_DIR"]
    persist_dir = config["PERSIST_DIR"]
    
    print("\n--- 3. Building Simple Index (Single Layer Chunking) ---")
    
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")
        sys.exit("Please place source documents (PDFs, TXT) in the './data' directory and run again.")

    # 1. Load documents - Uses SimpleDirectoryReader's default internal loaders.
    documents = SimpleDirectoryReader(data_dir).load_data()
    
    if not documents:
        print(f"WARNING: No documents found in '{data_dir}'. Indexing skipped.")
        # Return a dummy index if no documents are found, to prevent crashes later
        return VectorStoreIndex([], storage_context=storage_context) 

    print(f"Successfully loaded {len(documents)} source documents.")

    # 2. Define the single-layer chunking strategy (Simple SentenceSplitter)
    SIMPLE_CHUNK_SIZE = config["CHUNK_SIZE"]
    SIMPLE_CHUNK_OVERLAP = config["CHUNK_OVERLAP"]
    splitter = SentenceSplitter(chunk_size=SIMPLE_CHUNK_SIZE, chunk_overlap=SIMPLE_CHUNK_OVERLAP) 

    # 3. Create nodes
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    
    # 4. Create the VectorStoreIndex using the nodes
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True
    )
    print(f"Indexing complete. Vectors stored in Qdrant. Total searchable nodes: {len(nodes)}")
    
    # 5. Persist the LlamaIndex metadata (DocStore, IndexStore)
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"LlamaIndex metadata persisted to {persist_dir}.")
    
    return index

# --- 4. INDEX LOADING/BUILDING FUNCTION ---
def get_or_build_index(config: Dict[str, Any], vector_store: QdrantVectorStore) -> VectorStoreIndex:
    """
    Checks for and loads a persistent LlamaIndex (metadata), or rebuilds the 
    index if metadata is missing.
    """
    index = None
    STORAGE_DIR = config["PERSIST_DIR"]

    if os.path.exists(STORAGE_DIR):
        print(f"\n--- 4. Loading Index ---")
        print(f"Attempting to load LlamaIndex metadata from: {STORAGE_DIR}")
        try:
            # Load storage context from persistence directory, linking it to the vector store
            storage_context = StorageContext.from_defaults(
                persist_dir=STORAGE_DIR, 
                vector_store=vector_store
            )
            # Load the index 
            index = load_index_from_storage(storage_context=storage_context)
            print("Index loaded successfully.")
        except Exception as e:
            # This is the fallback path if the metadata is corrupt or incomplete
            print(f"Error loading index metadata: {e}. Rebuilding index...")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = build_simple_index(storage_context, config)
            
    else:
        print(f"\n--- 4. Building Index ---")
        print("LlamaIndex metadata storage not found. Building index...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = build_simple_index(storage_context, config)     
    return index


# --- 5. MAIN EXECUTION ---
def ingest_documents():
    """Orchestrates the entire ingestion and indexing pipeline."""
    config = load_config()
    
    # 2. Setup Clients & Settings (returns only the VectorStore)
    vector_store = initialize_clients_and_settings(config)
    
    # 3. Conditional Index Building/Loading
    get_or_build_index(config, vector_store)
    
    print("\nSUCCESS: Indexing complete.")

if __name__ == "__main__":
    ingest_documents()
