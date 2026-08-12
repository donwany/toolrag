# LLM Provider Configuration Guide

This guide explains how to use different LLM providers with the RAG Tool agent.

## Supported Providers

- **OpenAI** - GPT-4, GPT-4o, GPT-3.5-turbo, etc.
- **Anthropic** - Claude 3.5 Sonnet, Claude 3 Opus, etc.
- **Google Gemini** - Gemini 2.0, Gemini Pro, etc.
- **Ollama** - Local models (llama2, mistral, neural-chat, etc.)

## Environment Variables

Set the appropriate API keys in your `.env` file:

### OpenAI
```bash
OPENAI_API_KEY=sk-...
```

### Anthropic
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

### Google Gemini
```bash
GOOGLE_API_KEY=AIzaSy...
```

### Ollama (Local)
```bash
OLLAMA_BASE_URL=http://localhost:11434  # Optional, defaults to localhost:11434
```

## Basic Usage

### Using Default Provider (OpenAI)

```python
from llm_factory import get_llm

# Uses OpenAI with default model (gpt-4o-mini)
llm = get_llm()
response = llm.invoke("Hello!")
```

### Using a Specific Provider

```python
from llm_factory import get_llm

# Use Anthropic Claude
llm = get_llm(provider="anthropic")

# Use Google Gemini
llm = get_llm(provider="gemini")

# Use Ollama (local)
llm = get_llm(provider="ollama")
```

### Using a Specific Model

```python
from llm_factory import get_llm

# OpenAI GPT-4 Turbo
llm = get_llm(provider="openai", model="gpt-4-turbo")

# Anthropic Claude 3 Opus
llm = get_llm(provider="anthropic", model="claude-3-opus-20240229")

# Gemini Pro
llm = get_llm(provider="gemini", model="gemini-pro")

# Ollama Mistral
llm = get_llm(provider="ollama", model="mistral")
```

### Setting Default Provider

```python
from llm_factory import LLMFactory

# Set Anthropic as default for all subsequent calls
LLMFactory.set_default_provider("anthropic")

# Now get_llm() uses Anthropic by default
llm = get_llm()  # Uses Anthropic Claude
```

## Using in the Agent

The agent automatically uses the configured LLM throughout all its reasoning, tool selection, and response generation steps.

### Configure Before Running Agent

```python
from llm_factory import LLMFactory
from agent_with_vector_tools import run_agent

# Set your preferred provider before running the agent
LLMFactory.set_default_provider("anthropic")

# Now run the agent - it will use Anthropic for all LLM calls
result = run_agent("Will it rain in Accra this weekend?")
```

### Run Agent with Different Providers

```python
from agent_with_vector_tools import run_agent
from llm_factory import LLMFactory

# Run with OpenAI
LLMFactory.set_default_provider("openai")
result1 = run_agent("Query 1")

# Run with Anthropic
LLMFactory.set_default_provider("anthropic")
result2 = run_agent("Query 2")

# Run with Ollama (local)
LLMFactory.set_default_provider("ollama")
result3 = run_agent("Query 3")
```

## Checking Available Providers

```python
from llm_factory import LLMFactory

# List all available providers
providers = LLMFactory.list_providers()
print(providers)  # ['openai', 'anthropic', 'gemini', 'ollama']

# Check which providers have API keys configured
available = LLMFactory.get_available_providers()
print(available)
# {
#     'openai': True,          # API key is set
#     'anthropic': False,      # API key is missing
#     'gemini': True,
#     'ollama': True           # No API key needed
# }
```

## Default Models per Provider

| Provider | Default Model | Model ID |
|----------|---------------|----------|
| OpenAI | GPT-4o Mini | `gpt-4o-mini` |
| Anthropic | Claude 3.5 Sonnet | `claude-3-5-sonnet-20241022` |
| Gemini | Gemini 2.0 Flash | `gemini-2.0-flash` |
| Ollama | Llama2 | `llama2` |

## Cost Considerations

- **OpenAI**: Pay-as-you-go, typically $0.02-0.30 per 1K tokens
- **Anthropic**: Pricing similar to OpenAI, generally competitive
- **Gemini**: Free tier available with rate limits; paid tier available
- **Ollama**: Free (runs locally on your machine)

### Cost-Effective Setup

For development and testing with minimal costs, consider:

1. **Ollama (Recommended for Development)**
   - Free, runs locally
   - No API keys needed
   - Works offline
   - Good models: `mistral`, `neural-chat`, `dolphin-mixtral`

2. **OpenAI gpt-4o-mini**
   - Cheap for production ($0.15 per 1M input tokens)
   - Fast and reliable

## Model Selection Guide

### For Complex Reasoning
- **Best**: Claude 3.5 Sonnet or GPT-4 Turbo
- **Budget**: Gemini Pro, GPT-4o-mini

### For Speed
- **Best**: Gemini 2.0 Flash, GPT-4o-mini
- **Local**: Mistral via Ollama

### For Accuracy
- **Best**: Claude 3.5 Sonnet, GPT-4 Turbo
- **Good**: Gemini Pro, GPT-4o

### For Cost (Production)
- **Best**: Ollama (free, local)
- **Next**: OpenAI gpt-4o-mini (~$0.15/1M tokens)
- **Premium**: Anthropic Claude (similar pricing to OpenAI)

## Troubleshooting

### "API key not found" Error

Make sure your `.env` file has the correct API key:

```bash
# For OpenAI
export OPENAI_API_KEY=sk-...

# Or add to .env file
echo "OPENAI_API_KEY=sk-..." >> .env
```

### Ollama Connection Error

If using Ollama locally, make sure it's running:

```bash
# Start Ollama
ollama serve

# In another terminal, pull a model
ollama pull mistral

# Or use the default
ollama pull llama2
```

### Rate Limiting

If you hit rate limits, consider:
1. Using a different provider
2. Reducing request frequency
3. Upgrading your API plan
4. Switching to Ollama (local, unlimited)

## Examples

See `example_usage.py` for complete examples of using different providers.

```python
# Run with different providers
python example_usage.py --provider openai
python example_usage.py --provider anthropic
python example_usage.py --provider gemini
python example_usage.py --provider ollama
```

## Advanced Configuration

### Custom Temperature Settings

```python
from llm_factory import get_llm

# Deterministic (good for tool selection)
llm_cold = get_llm(temperature=0.0)

# Balanced
llm_balanced = get_llm(temperature=0.7)

# Creative (good for response generation)
llm_creative = get_llm(temperature=0.9)
```

### Using Different Models in Same Workflow

```python
from llm_factory import get_llm

# Use cheap model for reasoning
reasoning_llm = get_llm(provider="openai", model="gpt-4o-mini", temperature=0)

# Use powerful model for response generation
response_llm = get_llm(provider="anthropic", model="claude-3-opus-20240229", temperature=0.7)
```

## Switching Providers in Production

To switch LLM providers without code changes:

```python
import os
from llm_factory import LLMFactory

# Read from environment variable
provider = os.getenv("LLM_PROVIDER", "openai")
LLMFactory.set_default_provider(provider)

# Start agent
from agent_with_vector_tools import run_agent
result = run_agent("Your query here")
```

Then set the environment variable:

```bash
# Use OpenAI
export LLM_PROVIDER=openai

# Use Anthropic
export LLM_PROVIDER=anthropic

# Use Ollama
export LLM_PROVIDER=ollama
```
