"""
Embedding Factory - Unified interface for different embedding models
Supports: HuggingFace, Ollama, OpenAI, and other LangChain embedding providers
"""

from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper for SentenceTransformer to conform to LangChain Embeddings interface"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize SentenceTransformer embeddings
        
        Args:
            model_name: Model identifier from HuggingFace Hub
        """
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query"""
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

    # def similarity_search(self, query: str, texts: list[str], k: int = 3) -> list[str]:
    #     """Calculate the similarity between a query and a text"""
    #     query_embedding = self.embed_query(query)
    #     text_embeddings = self.embed_documents(texts)
    #     similarities = cosine_similarity([query_embedding], text_embeddings)[0]
    #     return similarities.argsort()[::-1][:k].tolist()



class EmbeddingFactory:
    """Factory for creating embedding model instances"""
    
    _PROVIDERS = {
        "huggingface": "HuggingFaceEmbeddings",
        "ollama": "OllamaEmbeddings",
        "openai": "OpenAIEmbeddings",
        "sentence_transformers": "SentenceTransformerEmbeddings"
    }
    
    @staticmethod
    def list_providers() -> list[str]:
        """List all available embedding providers"""
        return list(EmbeddingFactory._PROVIDERS.keys())
    
    @staticmethod
    def create_embedding(
        provider: str = "openai",
        model_name: Optional[str] = None,
        **kwargs
    ) -> Embeddings:
        """
        Create an embedding model instance
        
        Args:
            provider: Embedding provider ("huggingface", "ollama", "openai", "sentence_transformers")
            model_name: Model name/identifier for the provider
            **kwargs: Additional arguments to pass to the embedding provider
        
        Returns:
            Embeddings: Instance of the selected embedding provider
        
        Raises:
            ValueError: If provider is not supported
        
        Examples:
            # HuggingFace embeddings
            embeddings = EmbeddingFactory.create_embedding(
                provider="huggingface",
                model_name="all-MiniLM-L12-v2"
            )
            
            # SentenceTransformer embeddings
            embeddings = EmbeddingFactory.create_embedding(
                provider="sentence_transformers",
                model_name="all-MiniLM-L6-v2"
            )
            
            # Ollama embeddings
            embeddings = EmbeddingFactory.create_embedding(
                provider="ollama",
                model_name="llama3"
            )
            
            # OpenAI embeddings
            embeddings = EmbeddingFactory.create_embedding(
                provider="openai",
                model_name="text-embedding-3-small"
            )
        """
        provider = provider.lower()
        
        if provider not in EmbeddingFactory._PROVIDERS:
            available = ", ".join(EmbeddingFactory.list_providers())
            raise ValueError(
                f"Unknown embedding provider: {provider}. "
                f"Available providers: {available}"
            )
        
        try:
            if provider == "huggingface":
                model = model_name or "all-MiniLM-L6-v2"
                return HuggingFaceEmbeddings(model_name=model, **kwargs)
            
            elif provider == "sentence_transformers":
                model = model_name or "all-MiniLM-L6-v2"
                return SentenceTransformerEmbeddings(model_name=model)
            
            elif provider == "ollama":
                model = model_name or "llama3"
                return OllamaEmbeddings(model=model, **kwargs)
            
            elif provider == "openai":
                model = model_name or "text-embedding-3-small"
                return OpenAIEmbeddings(model=model, **kwargs)
        
        except Exception as e:
            raise RuntimeError(
                f"Failed to create {provider} embedding: {str(e)}"
            ) from e


def create_embedding(
    provider: str = "openai",
    model_name: Optional[str] = None,
    **kwargs
) -> Embeddings:
    """
    Convenience function to create embeddings using the factory
    
    Args:
        provider: Embedding provider ("huggingface", "ollama", "openai", "sentence_transformers")
        model_name: Model name/identifier for the provider
        **kwargs: Additional arguments to pass to the embedding provider
    
    Returns:
        Embeddings: Instance of the selected embedding provider
    """
    return EmbeddingFactory.create_embedding(provider, model_name, **kwargs)
