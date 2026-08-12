# Vector Store Configuration Guide

This guide explains how to use different vector store providers with the RAG Tool agent.

## Supported Vector Stores

- **ChromaDB** (default) - Lightweight, persistent, easy to use
- **FAISS** - Fast similarity search, in-memory
- **Milvus** - Scalable, production-grade vector database
- **Qdrant** - Modern vector search engine, flexible deployment

## Environment Variables

### ChromaDB
No special configuration needed. By default stores data in `./chroma_db/`

```bash
# Optional: customize storage location
CHROMA_PERSIST_DIRECTORY=./my_chroma_db
```

### FAISS
No configuration needed. Stores embeddings in-memory (not persistent).

```bash
# Optional: save to disk after creation
FAISS_INDEX_PATH=./faiss_index.index
```

### Milvus
Can run as local SQLite backend or connect to a server.

```bash
# Local (default)
MILVUS_URI=./milvus.db

# Or remote server
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### Qdrant
Can run in-memory, locally, or connect to remote server.

```bash
# In-memory (default, temporary)
# No configuration needed

# Local persistent
QDRANT_PATH=./qdrant_storage

# Remote server
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-api-key
```

## Basic Usage

### Using Default Provider (ChromaDB)

```python
from agent_with_vector_tools import run_agent

# Uses ChromaDB by default
result = run_agent("Your question")
```

### Using a Specific Provider

```python
from agent_with_vector_tools import run_agent

# Use FAISS
result = run_agent("Your question", vector_store_provider="faiss")

# Use Milvus
result = run_agent("Your question", vector_store_provider="milvus")

# Use Qdrant
result = run_agent("Your question", vector_store_provider="qdrant")
```

## Provider Comparison

| Feature | ChromaDB | FAISS | Milvus | Qdrant |
|---------|----------|-------|--------|--------|
| **Persistence** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes |
| **Scalability** | Medium | High | Very High | High |
| **Setup** | ✅ Easy | ✅ Easy | 🔧 Medium | 🔧 Medium |
| **Memory Usage** | Low | Medium | Low | Low |
| **Production Ready** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Remote Server** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes |
| **Cost** | Free | Free | Free | Free |

## Detailed Setup for Each Provider

### ChromaDB (Recommended for Getting Started)

**Pros:**
- Zero configuration
- Persistent storage by default
- Great for prototyping
- Good performance for smaller datasets

**Cons:**
- Not ideal for very large-scale deployments

**Setup:**
```bash
# Just install
pip install chromadb

# Use it
from agent_with_vector_tools import run_agent
result = run_agent("Your question", vector_store_provider="chromadb")
```

**Configuration:**
```python
from agent_with_vector_tools import run_agent

# Custom collection name and storage path
result = run_agent(
    "Your question",
    vector_store_provider="chromadb",
    # These are automatically handled internally
)
```

### FAISS (For In-Memory, High-Speed Search)

**Pros:**
- Fastest similarity search
- Very memory efficient
- Excellent for batch operations
- Good for development/testing

**Cons:**
- Not persistent (data lost on restart)
- Single machine only
- Not designed for remote access

**Setup:**
```bash
# Install dependencies
pip install faiss-cpu
# or for GPU:
# pip install faiss-gpu

# Use it
from agent_with_vector_tools import run_agent
result = run_agent("Your question", vector_store_provider="faiss")
```

### Milvus (For Large-Scale Production)

**Pros:**
- Highly scalable
- Distributed architecture
- Production-grade reliability
- Multiple deployment options

**Cons:**
- More complex setup
- Higher resource requirements
- Requires additional services

**Setup:**

**Option 1: Local (SQLite backend)**
```bash
pip install langchain-milvus

# Use it
result = run_agent("Your question", vector_store_provider="milvus")
```

**Option 2: Docker (Recommended for Production)**
```bash
# Start Milvus in Docker
docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest

# Update connection in code
# Set MILVUS_HOST and MILVUS_PORT environment variables
```

**Option 3: Milvus Cloud**
```bash
# Use Milvus managed service
# See https://cloud.milvus.io
```

### Qdrant (For Modern, Flexible Setup)

**Pros:**
- Modern vector search engine
- Flexible deployment options
- REST API for easy integration
- Good documentation

**Cons:**
- Relatively new (ecosystem still growing)
- Medium resource requirements

**Setup:**

**Option 1: In-Memory (Development)**
```bash
pip install qdrant-client

# Use it (temporary, data not persisted)
result = run_agent("Your question", vector_store_provider="qdrant")
```

**Option 2: Local Persistent**
```bash
pip install qdrant-client

# Set environment
export QDRANT_PATH=./qdrant_storage

# Use it
result = run_agent("Your question", vector_store_provider="qdrant")
```

**Option 3: Docker (Recommended)**
```bash
# Start Qdrant server
docker run -p 6333:6333 qdrant/qdrant

# Configure in code
export QDRANT_URL=http://localhost:6333

# Use it
result = run_agent("Your question", vector_store_provider="qdrant")
```

**Option 4: Qdrant Cloud**
```bash
# Use managed Qdrant
# See https://qdrant.io/cloud/
```

## Advanced Usage

### Switching Between Providers

```python
from agent_with_vector_tools import run_agent
from vector_store_factory import VectorStoreFactory

# List available providers
providers = VectorStoreFactory.list_providers()
print(providers)  # ['chromadb', 'faiss', 'milvus', 'qdrant']

# Check configuration status
status = VectorStoreFactory.get_available_providers()
print(status)
# {
#     'chromadb': True,
#     'faiss': True,
#     'milvus': True,
#     'qdrant': True
# }

# Run with different providers
for provider in ['chromadb', 'faiss', 'qdrant']:
    print(f"Using {provider}...")
    result = run_agent("Your question", vector_store_provider=provider)
```

### Using Vector Store Directly

```python
from vector_store_factory import create_vector_store
from langchain_openai import OpenAIEmbeddings

texts = ["Document 1", "Document 2", "Document 3"]
embeddings = OpenAIEmbeddings()

# Create store with FAISS
store = create_vector_store(
    texts=texts,
    embeddings=embeddings,
    provider="faiss"
)

# Search
results = store.similarity_search("query", k=3)
```

## Performance Comparison

### Query Speed (for 1000 documents)
```
FAISS:     ~1-5ms
Qdrant:    ~5-10ms
ChromaDB:  ~10-20ms
Milvus:    ~10-30ms (depends on network)
```

### Memory Usage
```
FAISS:     Highest (all in RAM)
Qdrant:    Medium-Low
ChromaDB:  Low
Milvus:    Low (server handles)
```

### Setup Complexity
```
1. ChromaDB (easiest)
2. FAISS
3. Qdrant
4. Milvus (most complex)
```

## Choosing a Provider

**For Development:**
```python
# Use ChromaDB - zero config, persistent by default
result = run_agent("question", vector_store_provider="chromadb")
```

**For Testing/Prototyping:**
```python
# Use FAISS - fast, in-memory, no persistence needed
result = run_agent("question", vector_store_provider="faiss")
```

**For Small Production:**
```python
# Use ChromaDB or Qdrant - both can scale reasonably
result = run_agent("question", vector_store_provider="chromadb")
```

**For Large Production:**
```python
# Use Milvus or Qdrant Cloud
result = run_agent("question", vector_store_provider="milvus")
```

**For Maximum Flexibility:**
```python
# Use Qdrant - works locally and remotely, easy to scale
result = run_agent("question", vector_store_provider="qdrant")
```

## Troubleshooting

### ChromaDB Issues

**"Chroma collection not found"**
```python
# Chroma creates the collection automatically
# If you get this error, check the persist_directory
```

### FAISS Issues

**"Index dimension mismatch"**
```python
# FAISS requires all embeddings to have the same dimension
# Make sure you're using the same embedding model
```

### Milvus Issues

**"Connection refused"**
```bash
# Start Milvus server first
docker run -d -p 19530:19530 milvusdb/milvus:latest

# Or set correct connection parameters
export MILVUS_URI=./milvus.db
```

### Qdrant Issues

**"Collection not found"**
```python
# Qdrant creates the collection automatically on first insert
# Make sure you're using consistent collection names
```

**"Cannot connect to Qdrant"**
```bash
# If using Docker, make sure it's running
docker ps | grep qdrant

# Start if needed
docker run -p 6333:6333 qdrant/qdrant
```

## Migration Between Providers

To migrate data from one vector store to another:

```python
from agent_with_vector_tools import ToolVectorDB, TOOL_DESCRIPTORS
from vector_store_factory import create_vector_store
from langchain_openai import OpenAIEmbeddings

# Read from old store
old_db = ToolVectorDB(TOOL_DESCRIPTORS, vector_store_provider="chromadb")

# Create new store and add data
embeddings = OpenAIEmbeddings()

# This will be automated in the new store
new_store = create_vector_store(
    texts=[...],
    embeddings=embeddings,
    provider="milvus"
)
```

## Best Practices

1. **Use Persistent Storage in Production**
   - ChromaDB, Milvus, or Qdrant with persistent storage
   - Avoid FAISS for production (in-memory only)

2. **Monitor Vector Store Performance**
   - Track query latency
   - Monitor memory/disk usage
   - Watch for dimension mismatches

3. **Backup Your Data**
   - Export embeddings regularly
   - Keep source documents backed up
   - Test recovery procedures

4. **Use Appropriate Batch Sizes**
   - FAISS: Can handle large batches
   - Qdrant/Milvus: Start with 100-1000 documents
   - ChromaDB: Works well with any size

5. **Index Optimization**
   - Milvus: Use HNSW for better performance
   - Qdrant: Tuning options available
   - FAISS: Quantization for memory efficiency

## Resources

- **ChromaDB**: https://docs.trychroma.com/
- **FAISS**: https://github.com/facebookresearch/faiss
- **Milvus**: https://milvus.io/docs
- **Qdrant**: https://qdrant.tech/documentation/
