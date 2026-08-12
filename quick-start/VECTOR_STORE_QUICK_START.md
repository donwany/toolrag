# Vector Store Support - Quick Reference

Your RAG agent now supports **4 major vector store providers**!

## Quick Setup

### Default (ChromaDB)
```python
from agent_with_vector_tools import run_agent

result = run_agent("Your question")  # Uses ChromaDB automatically
```

### Switch to Different Provider
```python
from agent_with_vector_tools import run_agent

# FAISS (in-memory, fastest)
result = run_agent("Your question", vector_store_provider="faiss")

# Milvus (scalable)
result = run_agent("Your question", vector_store_provider="milvus")

# Qdrant (flexible)
result = run_agent("Your question", vector_store_provider="qdrant")
```

## Provider Overview

| Provider | Best For | Setup | Persistence |
|----------|----------|-------|-------------|
| **ChromaDB** | Getting started | ✅ Zero config | ✅ Yes |
| **FAISS** | Speed, testing | ✅ Simple | ❌ No |
| **Milvus** | Scale, production | 🔧 Moderate | ✅ Yes |
| **Qdrant** | Flexibility | 🔧 Moderate | ✅ Yes |

## Installation

```bash
# ChromaDB (default, already included)
pip install chromadb

# FAISS
pip install faiss-cpu

# Milvus
pip install langchain-milvus

# Qdrant
pip install qdrant-client
```

## Usage Examples

### Using ChromaDB (Recommended for Development)
```python
from agent_with_vector_tools import run_agent

# All data persists automatically
result1 = run_agent("First question", vector_store_provider="chromadb")
result2 = run_agent("Second question", vector_store_provider="chromadb")
# Both queries can use cached embeddings
```

### Using FAISS (Development/Testing)
```python
from agent_with_vector_tools import run_agent

# Fast similarity search, data lost on exit
result = run_agent("Your question", vector_store_provider="faiss")
```

### Using Milvus (Production)
```bash
# Start Milvus in Docker first
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest
```

```python
from agent_with_vector_tools import run_agent

# Connect to Milvus
result = run_agent("Your question", vector_store_provider="milvus")
```

### Using Qdrant (Production)
```bash
# Start Qdrant in Docker
docker run -p 6333:6333 qdrant/qdrant
```

```python
from agent_with_vector_tools import run_agent

# Connect to Qdrant
result = run_agent("Your question", vector_store_provider="qdrant")
```

## Check Available Providers

```python
from vector_store_factory import VectorStoreFactory

# List all providers
providers = VectorStoreFactory.list_providers()
print(providers)  # ['chromadb', 'faiss', 'milvus', 'qdrant']

# Check which are configured
status = VectorStoreFactory.get_available_providers()
for provider, available in status.items():
    print(f"{provider}: {'✓' if available else '✗'}")
```

## Performance Characteristics

### Speed (Query Latency for 1000 docs)
```
🏆 FAISS:    1-5ms
🥈 Qdrant:   5-10ms
🥉 ChromaDB: 10-20ms
   Milvus:   10-30ms
```

### Storage/Memory
```
Lowest:  ChromaDB, Qdrant, Milvus (disk-based)
Highest: FAISS (in-memory only)
```

### Scalability
```
Development:  ChromaDB, FAISS
Small Scale:  Qdrant, ChromaDB
Large Scale:  Milvus, Qdrant
```

## Choosing a Provider

### "I'm just testing"
→ Use **FAISS** (fastest, no setup)

### "I'm building a prototype"
→ Use **ChromaDB** (persistent, zero config)

### "I need something for production"
→ Use **Qdrant** (balanced, flexible, scalable)

### "I need massive scale"
→ Use **Milvus** (distributed, designed for scale)

## Common Tasks

### Run Agent with Different Stores
```python
from agent_with_vector_tools import run_agent

stores = ["chromadb", "faiss", "qdrant", "milvus"]
for store in stores:
    try:
        print(f"\nTesting {store}...")
        result = run_agent(
            "Will it rain tomorrow?",
            vector_store_provider=store
        )
        print("✓ Success")
    except Exception as e:
        print(f"✗ Failed: {e}")
```

### Create Custom Vector Store
```python
from vector_store_factory import create_vector_store
from langchain_openai import OpenAIEmbeddings

texts = ["Document 1", "Document 2", "Document 3"]
embeddings = OpenAIEmbeddings()

# Create with specific provider
store = create_vector_store(
    texts=texts,
    embeddings=embeddings,
    provider="chromadb",  # or "faiss", "milvus", "qdrant"
    persist_directory="./my_store"  # ChromaDB only
)

# Use it
results = store.similarity_search("query", k=3)
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'chromadb'"
```bash
pip install chromadb
```

### "ModuleNotFoundError: No module named 'faiss'"
```bash
pip install faiss-cpu
# or for GPU: pip install faiss-gpu
```

### "Failed to connect to Milvus"
```bash
# Start Milvus first
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest
```

### "Failed to connect to Qdrant"
```bash
# Start Qdrant first
docker run -p 6333:6333 qdrant/qdrant
```

## Files

- **Implementation**: [vector_store_factory.py](vector_store_factory.py)
- **Detailed Docs**: [VECTOR_STORE_SETUP.md](VECTOR_STORE_SETUP.md)
- **Agent Code**: [agent_with_vector_tools.py](agent_with_vector_tools.py)

## Next Steps

1. **Try default (ChromaDB)**
   ```python
   python -c "from agent_with_vector_tools import run_agent; run_agent('Test question')"
   ```

2. **Try different providers**
   ```python
   # Test each provider
   run_agent("Test", vector_store_provider="faiss")
   run_agent("Test", vector_store_provider="qdrant")
   ```

3. **Read full docs** at [VECTOR_STORE_SETUP.md](VECTOR_STORE_SETUP.md)

4. **Deploy with your choice**
   ```python
   # Production setup
   result = run_agent(
       "Your question",
       vector_store_provider="milvus"  # or your choice
   )
   ```
