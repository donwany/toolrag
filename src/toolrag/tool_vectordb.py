
from .embed_factory import EmbeddingFactory, create_embedding
from .vector_store_factory import VectorStoreFactory, create_vector_store
from loguru import logger

# ============================================================================
# VECTOR DATABASE SETUP
# ============================================================================

class ToolVectorDB:
    """Vector database for tool retrieval"""
    
    def __init__(
        self,
        tool_descriptors: list[dict],
        vector_store_provider: str = "chroma",
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        pg_distance_metric: str = "cosine",
        pgvector_connection: str | None = None,
        pgvector_collection_name: str = "tool_descriptors",
        pgvector_use_hnsw: bool = True,
        pgvector_pre_delete_collection: bool = False,
    ):
        # Create embeddings using the factory
        self.embeddings = create_embedding(
            provider=embedding_provider,
            model_name=embedding_model
        )
        self.tool_descriptors = {desc['tool_id']: desc for desc in tool_descriptors}
        self.vector_store_provider = vector_store_provider
        self.pg_distance_metric = pg_distance_metric
        
        # Convert tool descriptors to text for embedding
        texts = []
        metadatas = []
        
        for tool_desc in tool_descriptors:
            # Combine all relevant text for embedding
            text = f"""
            Tool: {tool_desc['tool_id']}
            Inputs: {tool_desc['inputs']}
            Description: {tool_desc['description']}
            When to use: {tool_desc['when_to_use']}
            Examples: {', '.join(tool_desc['examples'])}
            """
            
            # text = f"""
            # Tool: {tool_desc['tool_id']}
            # """
            
            texts.append(text)
            # Store only simple metadata (strings) - use tool_id to lookup full descriptor
            metadatas.append({"tool_id": tool_desc['tool_id']})
        
        # Create vector store using factory
        if vector_store_provider.lower() == "chroma":
            self.vectorstore = create_vector_store(
                texts=texts,
                embeddings=self.embeddings,
                metadatas=metadatas,
                provider="chroma",
                persist_directory="./tools_chroma_db",
                collection_name="tool_descriptors"
            )
        elif vector_store_provider.lower() == "faiss":
            self.vectorstore = create_vector_store(
                texts=texts,
                embeddings=self.embeddings,
                metadatas=metadatas,
                provider="faiss"
            )
        elif vector_store_provider.lower() == "milvus":
            self.vectorstore = create_vector_store(
                texts=texts,
                embeddings=self.embeddings,
                metadatas=metadatas,
                provider="milvus",
                collection_name="tool_descriptors",
                connection_args={"uri": "./milvus_tools.db"}
            )
        elif vector_store_provider.lower() == "qdrant":
            self.vectorstore = create_vector_store(
                texts=texts,
                embeddings=self.embeddings,
                metadatas=metadatas,
                provider="qdrant",
                collection_name="tool_descriptors",
                in_memory=True
            )
        elif vector_store_provider.lower() in ("pgvector", "postgres", "postgresql"):
            pg_kwargs: dict = {
                "collection_name": pgvector_collection_name,
                "distance_metric": pg_distance_metric,
                "use_hnsw_index": pgvector_use_hnsw,
                "pre_delete_collection": pgvector_pre_delete_collection,
            }
            if pgvector_connection:
                pg_kwargs["connection"] = pgvector_connection
            self.vectorstore = create_vector_store(
                texts=texts,
                embeddings=self.embeddings,
                metadatas=metadatas,
                provider="pgvector",
                **pg_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown vector store provider: {vector_store_provider}. "
                f"Available: {VectorStoreFactory.list_providers()}"
            )
    
    def similarity_search(self, query: str, k: int = 3) -> list[dict]:
        """Search for relevant tools based on query"""
        results = self.vectorstore.similarity_search(query, k=k)
        # Retrieve full tool descriptors using tool_id from metadata
        retrieved_tools = []
        for doc in results:
            tool_id = doc.metadata.get('tool_id')
            if tool_id in self.tool_descriptors:
                retrieved_tools.append(self.tool_descriptors[tool_id])
        return retrieved_tools

    
    def similarity_search_with_score(self, query: str, k: int = 3) -> list[dict]:
        """Search for relevant tools based on query"""

        results_with_score = self.vectorstore.similarity_search_with_score(query, k=k)
        # logger.info("Retrieved tool_ids={}", [doc.metadata.get("tool_id") for doc, _ in results_with_score],)
        # logger.info("Retrieved tool_scores={}", [float(score) for _, score in results_with_score])
        # logger.info("Retrieved tool_similarities={}", [1 - float(score) for _, score in results_with_score])
    
        retrieved_tools = []

        for doc, score in results_with_score:
            tool_id = doc.metadata.get("tool_id")

            if tool_id in self.tool_descriptors:
                tool = self.tool_descriptors[tool_id].copy()
                tool["score"] = float(score)
                similarity = 1 - score
                tool["similarity"] = similarity # Higher is better
                retrieved_tools.append(tool)

        return retrieved_tools
