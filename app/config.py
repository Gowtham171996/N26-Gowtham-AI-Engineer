from typing import Dict, Any
import yaml
import sys

# --- LlamaIndex/Ollama Configuration ---
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore


# --- Qdrant Integration ---
from qdrant_client import QdrantClient

# --- RAG SYSTEM INITIALIZER CLASS ---
@staticmethod
class RAGSystemInitializer:
    """
    A class to handle all configuration loading and initialization of 
    core RAG components (LlamaIndex Settings, Ollama clients, Qdrant Vector Store).
    
    This provides a clean, single-point interface for setting up the RAG system.
    """
    CONFIG_FILE_NAME = "config.yaml"
    
    def __init__(self):
        # Configuration is loaded immediately upon instantiation
        self.config: Dict[str, Any] = self._load_config()
        self.qdrant_client: QdrantClient = None
        self.vector_store: QdrantVectorStore = self.initialize_system()
        self.Settings = Settings
        
    def _load_config(self) -> Dict[str, Any]:
        """Private method to load configuration from config.yaml."""
        print("\n--- 1. Loading Configuration ---")
        try:
            with open(self.CONFIG_FILE_NAME, 'r') as f: 
                config = yaml.safe_load(f)
            print(f"INFO: Configuration loaded successfully from {self.CONFIG_FILE_NAME}.")
            return config
        except FileNotFoundError:
            print(f"FATAL: {self.CONFIG_FILE_NAME} not found. Cannot proceed.")
            sys.exit(1)
        except Exception as e:
            print(f"FATAL: Error loading {self.CONFIG_FILE_NAME}: {e}")
            sys.exit(1)

    def initialize_system(self) -> QdrantVectorStore:
        """
        Initializes LlamaIndex settings and Qdrant client. 
        
        Returns:
            A tuple containing the loaded configuration (Dict) and the 
            initialized QdrantVectorStore instance.
        """
        print("\n--- 2. Initializing Clients and Settings ---")

        # Set LlamaIndex Settings
        Settings.llm = Ollama(
            model=self.config["OLLAMA_LLM_MODEL"], 
            base_url=self.config["OLLAMA_BASE_URL"], 
            request_timeout=self.config["OLLAMA_TIMEOUT"]
        )
        Settings.embed_model = OllamaEmbedding(
            model_name=self.config["OLLAMA_EMBEDDING_MODEL"], 
            base_url=self.config["OLLAMA_BASE_URL"]
        )
        Settings.num_workers = self.config["OLLAMA_WORKERS"] 
        print(f"LlamaIndex Settings configured. LLM: {self.config['OLLAMA_LLM_MODEL']}")

        # Initialize Qdrant Client and Vector Store 
        try:
            # Check connection first
            self.qdrant_client = QdrantClient(host=self.config["QDRANT_HOST"], port=self.config["QDRANT_PORT"])
            self.qdrant_client.get_collections() 
            
            # Create the vector store instance
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.config["QDRANT_COLLECTION_NAME"]
            )
            print(f"Qdrant connection successful. Collection: {self.config['QDRANT_COLLECTION_NAME']}")
            
            # Return both for use in the API/Ingestion scripts
            return self.vector_store
            
        except Exception as e:
            print(f"FATAL: Could not connect to Qdrant at {self.config['QDRANT_HOST']}:{self.config['QDRANT_PORT']}. Is the service running?")
            print(f"Error: {e}")
            sys.exit(1)
        
    def load_rag_prompt_template(self,filepath: str = "system_prompt.txt") -> str:
        """Reads the RAG prompt template from the specified file."""
        try:
            filepath = self.config["RAG_PROMPT_TEMPLATE"]
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"FATAL ERROR: Prompt file '{filepath}' not found. Using hardcoded default.")
            # Fallback to a hardcoded string if file is missing, for robustness
            return (
                "You are a helpful and accurate assistant. Use the following CONTEXT to answer "
                "the USER QUERY. If the context does not contain the answer, state that you "
                "cannot find the answer in the provided documents.\n\n"
                "CONTEXT:\n{context}\n\nUSER QUERY: {query}"
            )
