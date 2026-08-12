"""
Vector Store Factory - Unified interface for multiple vector store providers

Supports:
- ChromaDB (persistent or in-memory)
- Milvus (scalable, production-grade)
- Qdrant (vector search engine)
- FAISS (Facebook similarity search)
- PGVector (PostgreSQL + pgvector, HNSW indexes)
"""

import os
import faiss
from typing import Optional, Literal
from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore
from dotenv import load_dotenv

from .pgvector_store import (
    PGDistanceMetric,
    check_pgvector_available,
    collection_name_for_metric,
    create_pgvector_store,
    get_pgvector_connection,
    normalize_distance_metric,
)

load_dotenv("../.env", override=True)

VectorStoreProviderName = Literal[
    "chroma", "faiss", "milvus", "qdrant", "pgvector"
]


class VectorStoreProvider(ABC):
    """Abstract base class for vector store providers"""
    
    @abstractmethod
    def create_vector_store(
        self,
        texts: list[str],
        embeddings: Embeddings,
        metadatas: Optional[list[dict]] = None,
        **kwargs
    ) -> VectorStore:
        """Create and return a vector store instance"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate that required configurations are set"""
        pass


class ChromaDBProvider(VectorStoreProvider):
    """ChromaDB Vector Store Provider"""
    
    def validate_config(self) -> bool:
        """ChromaDB doesn't require external configuration"""
        return True
    
    def create_vector_store(
        self,
        texts: list[str],
        embeddings: Embeddings,
        metadatas: Optional[list[dict]] = None,
        collection_name: str = "tool_descriptors",
        persist_directory: str = "./chroma_db",
        **kwargs
    ) -> VectorStore:
        """Create ChromaDB vector store"""
        try:
            vectorstore = Chroma.from_texts(
                ids=[str(i) for i in range(len(texts))],
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas or [],
                persist_directory=persist_directory,
                collection_name=collection_name
            )
            return vectorstore
        except ImportError:
            raise ImportError("chromadb is not installed. Install with: pip install chromadb")


class FAISSProvider(VectorStoreProvider):
    """FAISS Vector Store Provider"""
    
    def validate_config(self) -> bool:
        """FAISS doesn't require external configuration"""
        return True
    
    def create_vector_store(
        self,
        texts: list[str],
        embeddings: Embeddings,
        metadatas: Optional[list[dict]] = None,
        **kwargs
    ) -> VectorStore:
        """Create FAISS vector store"""
        try:
            if not texts:
                raise ValueError("FAISS requires at least one text to determine embedding dimension")
            
            # Get embedding dimension
            embedding_dim = len(embeddings.embed_query("test"))
            
            # Create FAISS index
            index = faiss.IndexFlatL2(embedding_dim)
            
            # Create vector store
            vectorstore = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                distance_strategy=DistanceStrategy.COSINE
            )
            
            # Add texts
            vectorstore.add_texts(texts=texts, ids=[str(i) for i in range(len(texts))], metadatas=metadatas or [])
            
            return vectorstore
        except ImportError as e:
            raise ImportError(f"FAISS dependencies not installed: {e}")


class MilvusProvider(VectorStoreProvider):
    """Milvus Vector Store Provider"""
    
    def validate_config(self) -> bool:
        """Check if Milvus is configured"""
        # Milvus can run locally or via external connection
        return True
    
    def create_vector_store(
        self,
        texts: list[str],
        embeddings: Embeddings,
        metadatas: Optional[list[dict]] = None,
        collection_name: str = "tool_descriptors",
        connection_args: Optional[dict] = None,
        **kwargs
    ) -> VectorStore:
        """Create Milvus vector store"""
        try:
            from langchain_milvus import Milvus
            
            # Default connection (local SQLite backend)
            if connection_args is None:
                connection_args = {"uri": "./milvus.db"}
            
            vectorstore = Milvus(
                embedding_function=embeddings,
                collection_name=collection_name,
                connection_args=connection_args,
                index_params={"index_type": "FLAT", "metric_type": "L2"},
            )
            
            # Add texts
            if texts:
                vectorstore.add_texts(
                    texts=texts, 
                    ids=[str(i) for i in range(len(texts))],
                    metadatas=metadatas or []
                )
            
            return vectorstore
        except ImportError:
            raise ImportError("langchain-milvus is not installed. Install with: pip install langchain-milvus")


class QdrantProvider(VectorStoreProvider):
    """Qdrant Vector Store Provider"""
    
    def validate_config(self) -> bool:
        """Check if Qdrant is configured"""
        # Qdrant can run in-memory, locally, or via remote connection
        return True
    
    def create_vector_store(
        self,
        texts: list[str],
        embeddings: Embeddings,
        metadatas: Optional[list[dict]] = None,
        collection_name: str = "tool_descriptors",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[str] = None,
        in_memory: bool = True,
        **kwargs
    ) -> VectorStore:
        """Create Qdrant vector store"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            from langchain_qdrant import QdrantVectorStore
            import uuid
            
            # Determine connection type
            if url:
                # Remote Qdrant
                client = QdrantClient(url=url, api_key=api_key)
            elif path:
                # Local Qdrant with persistent storage
                client = QdrantClient(path=path)
            else:
                # In-memory Qdrant (default)
                client = QdrantClient(":memory:")
            
            # Get embedding dimension
            if texts:
                embedding_dim = len(embeddings.embed_query("test"))
            else:
                embedding_dim = 384  # Default dimension, adjust as needed
            
            # Create collection if it doesn't exist
            try:
                client.get_collection(collection_name)
            except Exception:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE
                    )
                )
            
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings
            )
            
            # Add texts
            if texts:
                vectorstore.add_texts(
                    texts=texts, 
                    ids=[str(uuid.uuid4()) for i in range(len(texts))], 
                    metadatas=metadatas or []
                )
            
            return vectorstore
        except ImportError:
            raise ImportError("qdrant-client is not installed. Install with: pip install langchain-qdrant qdrant-client")


class PGVectorProvider(VectorStoreProvider):
    """PostgreSQL pgvector store with configurable distance metrics and HNSW indexes."""

    def validate_config(self) -> bool:
        return check_pgvector_available()

    def create_vector_store(
        self,
        texts: list[str],
        embeddings: Embeddings,
        metadatas: Optional[list[dict]] = None,
        collection_name: str = "tool_descriptors",
        connection: Optional[str] = None,
        distance_metric: str = "cosine",
        use_hnsw_index: bool = True,
        pre_delete_collection: bool = False,
        **kwargs,
    ) -> VectorStore:
        try:
            metric: PGDistanceMetric = normalize_distance_metric(distance_metric)
            conn = connection or get_pgvector_connection()
            resolved_collection = collection_name_for_metric(collection_name, metric)

            return create_pgvector_store(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                connection=conn,
                collection_name=resolved_collection,
                distance_metric=metric,
                use_hnsw_index=use_hnsw_index,
                pre_delete_collection=pre_delete_collection,
            )
        except ImportError as e:
            raise ImportError(
                "PGVector dependencies not installed. "
                "Install with: pip install langchain-postgres psycopg[binary] pgvector"
            ) from e


# Provider registry
PROVIDERS = {
    "chroma": ChromaDBProvider,
    "faiss": FAISSProvider,
    "milvus": MilvusProvider,
    "qdrant": QdrantProvider,
    "pgvector": PGVectorProvider,
}

# Default configuration per provider
DEFAULT_CONFIG = {
    "chroma": {
        "persist_directory": "./chroma_db",
        "collection_name": "tool_descriptors",
    },
    "faiss": {},
    "milvus": {
        "collection_name": "tool_descriptors",
        "connection_args": {"uri": "./milvus.db"},
    },
    "qdrant": {
        "collection_name": "tool_descriptors",
        "in_memory": True,
    },
    "pgvector": {
        "collection_name": "tool_descriptors",
        "distance_metric": "cosine",
        "use_hnsw_index": True,
    },
}


class VectorStoreFactory:
    """Factory for creating vector store instances from different providers"""
    
    _default_provider: str = "chroma"
    
    @classmethod
    def set_default_provider(
        cls,
        provider: VectorStoreProviderName,
    ):
        """Set the default vector store provider"""
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDERS.keys())}")
        cls._default_provider = provider
    
    @classmethod
    def get_default_provider(cls) -> str:
        """Get the current default provider"""
        return cls._default_provider
    
    @classmethod
    def create(
        cls,
        texts: list[str],
        embeddings: Embeddings,
        metadatas: Optional[list[dict]] = None,
        provider: Optional[VectorStoreProviderName] = None,
        **kwargs
    ) -> VectorStore:
        """
        Create a vector store instance
        
        Args:
            texts: List of text documents to add to the store
            embeddings: Embeddings instance for vectorizing texts
            metadatas: Optional list of metadata dicts (one per text)
            provider: Vector store provider ("chroma", "faiss", "milvus", "qdrant", "pgvector")
                     Uses default if not specified
            **kwargs: Additional provider-specific arguments
                - ChromaDB: persist_directory, collection_name
                - FAISS: (none required)
                - Milvus: collection_name, connection_args
                - Qdrant: collection_name, url, api_key, path, in_memory
                - PGVector: connection, collection_name, distance_metric (l2|cosine|inner_product|l1),
                  use_hnsw_index, pre_delete_collection
        
        Returns:
            A VectorStore instance ready to use
            
        Examples:
            # Using default provider (ChromaDB)
            store = VectorStoreFactory.create(texts, embeddings)
            
            # Using specific provider
            store = VectorStoreFactory.create(
                texts, embeddings, provider="qdrant"
            )
            
            # With custom configuration
            store = VectorStoreFactory.create(
                texts, embeddings,
                provider="chroma",
                persist_directory="./my_db",
                collection_name="my_collection"
            )
        """
        if provider is None:
            provider = cls._default_provider
        
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDERS.keys())}")
        
        provider_class = PROVIDERS[provider]
        provider_instance = provider_class()
        
        # Validate configuration
        if not provider_instance.validate_config():
            raise ValueError(
                f"Required configuration for {provider} not found. "
                f"Please set the appropriate environment variables or configuration."
            )
        
        # Merge default config with provided kwargs
        config = {**DEFAULT_CONFIG.get(provider, {}), **kwargs}
        
        return provider_instance.create_vector_store(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            **config
        )
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all available providers"""
        return list(PROVIDERS.keys())
    
    @classmethod
    def get_available_providers(cls) -> dict[str, bool]:
        """Get available providers and their configuration status"""
        available = {}
        for name, provider_class in PROVIDERS.items():
            provider_instance = provider_class()
            available[name] = provider_instance.validate_config()
        return available


def create_vector_store(
    texts: list[str],
    embeddings: Embeddings,
    metadatas: Optional[list[dict]] = None,
    provider: Optional[str] = None,
    **kwargs
) -> VectorStore:
    """
    Convenience function to create a vector store
    
    Args:
        texts: List of text documents
        embeddings: Embeddings instance
        metadatas: Optional metadata list
        provider: Vector store provider name
        **kwargs: Additional configuration
        
    Returns:
        A VectorStore instance
    """
    return VectorStoreFactory.create(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        provider=provider,
        **kwargs
    )
