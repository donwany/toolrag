# Quick Reference: Multi-LLM Support

## What's New?

Your RAG agent now supports **4 different LLM providers**:
- ✅ **OpenAI** (GPT-4, GPT-4o, etc.)
- ✅ **Anthropic** (Claude)
- ✅ **Google Gemini**
- ✅ **Ollama** (local models)

## Files Added/Modified

| File | Purpose |
|------|---------|
| `llm_factory.py` | LLM provider factory - the core abstraction |
| `LLM_SETUP.md` | Detailed configuration guide |
| `multi_llm_examples.py` | Examples for different providers |
| `.env.example` | Template for API keys |
| `agent_with_vector_tools.py` | Updated to use LLMFactory |
| `pyproject.toml` | Added langchain-anthropic and langchain-google-genai |

## Setup (5 minutes)

### 1. Copy environment template
```bash
cp .env.example .env
```

### 2. Add your API key (choose ONE or more)
```bash
# Option A: OpenAI (recommended to start)
echo "OPENAI_API_KEY=sk-..." >> .env

# Option B: Anthropic
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# Option C: Google Gemini
echo "GOOGLE_API_KEY=AIzaSy..." >> .env

# Option D: Ollama (local, free, no key needed)
# Just install and run: ollama serve
```

### 3. Run the agent (it will use your default provider)
```python
from agent_with_vector_tools import run_agent
result = run_agent("Your question here")
```

## Usage Examples

### Use Default Provider
```python
from llm_factory import get_llm

llm = get_llm()  # Uses OpenAI by default
response = llm.invoke("Hello!")
```

### Switch Providers
```python
from llm_factory import LLMFactory
from agent_with_vector_tools import run_agent

# Use Anthropic for all agent operations
LLMFactory.set_default_provider("anthropic")
result = run_agent("Your question")

# Switch to Ollama (local)
LLMFactory.set_default_provider("ollama")
result = run_agent("Another question")
```

### Use Specific Model
```python
from llm_factory import get_llm

# OpenAI GPT-4 Turbo
llm = get_llm(provider="openai", model="gpt-4-turbo")

# Claude 3 Opus
llm = get_llm(provider="anthropic", model="claude-3-opus-20240229")

# Ollama Mistral
llm = get_llm(provider="ollama", model="mistral")
```

### Check Available Providers
```python
from llm_factory import LLMFactory

# See which providers are configured
available = LLMFactory.get_available_providers()
for provider, is_available in available.items():
    status = "✓" if is_available else "✗"
    print(f"{status} {provider}")
```

## Command Line Examples

```bash
# Check available providers
python multi_llm_examples.py --check

# Run with OpenAI
python multi_llm_examples.py --provider openai

# Run with Anthropic
python multi_llm_examples.py --provider anthropic

# Run with Ollama (local)
python multi_llm_examples.py --provider ollama

# Test different models
python multi_llm_examples.py --models

# See temperature effects
python multi_llm_examples.py --temperature
```

## Architecture

```
┌─────────────────────────────────┐
│   Your Code                     │
│   run_agent() / get_llm()       │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   LLMFactory (llm_factory.py)   │
│   - Unified interface           │
│   - Provider selection          │
│   - Model configuration         │
└────────────┬────────────────────┘
             │
      ┌──────┼──────┬──────┬────────┐
      ▼      ▼      ▼      ▼        ▼
   OpenAI Anthropic Gemini Ollama  (Other)
```

## Common Tasks

### Switch Default Provider for All Scripts
```bash
# In .env file
LLM_PROVIDER=anthropic
```

```python
import os
from llm_factory import LLMFactory

provider = os.getenv("LLM_PROVIDER", "openai")
LLMFactory.set_default_provider(provider)
```

### Use Different Providers in Same Script
```python
from llm_factory import get_llm

# Budget reasoning model
reasoning_llm = get_llm(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0
)

# Powerful response model
response_llm = get_llm(
    provider="anthropic",
    model="claude-3-opus-20240229",
    temperature=0.7
)
```

### Cost Optimization
```python
from llm_factory import get_llm

# For development: Use Ollama (free, local)
llm = get_llm(provider="ollama", model="mistral")

# For production: Use cost-effective model
llm = get_llm(provider="openai", model="gpt-4o-mini")
```

## Troubleshooting

### "API key not found" Error
```bash
# Check your .env file
cat .env | grep API_KEY

# Or set directly
export OPENAI_API_KEY="sk-..."
```

### Ollama Not Working
```bash
# Install Ollama from https://ollama.ai
# Start Ollama in one terminal
ollama serve

# In another terminal, pull a model
ollama pull mistral

# Test it
python multi_llm_examples.py --provider ollama
```

### Check Provider Status
```python
from llm_factory import LLMFactory

available = LLMFactory.get_available_providers()
print(available)
# {'openai': True, 'anthropic': False, 'gemini': False, 'ollama': True}
```

## Key Features

✅ **Unified Interface** - Same code works with any provider
✅ **Easy Switching** - Change providers without changing code
✅ **Temperature Control** - Set temperature per LLM instance
✅ **Multiple Models** - Use different models from same provider
✅ **Fallback Support** - Try different providers as fallback
✅ **Cost Optimization** - Use cheap models for development
✅ **Local Option** - Ollama for privacy and offline use

## More Information

- **Detailed Setup Guide**: [LLM_SETUP.md](LLM_SETUP.md)
- **Full Examples**: [multi_llm_examples.py](multi_llm_examples.py)
- **API Configuration**: [.env.example](.env.example)

## API Key Links

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com
- **Google Gemini**: https://ai.google.dev/
- **Ollama**: https://ollama.ai (no API key needed)
