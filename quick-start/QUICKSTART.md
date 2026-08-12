# Quick Start Guide

## Try it Now (No Dependencies)

The **standalone_demo.py** runs immediately without any API keys or installations:

```bash
python standalone_demo.py
```

This demonstrates the complete flow from the screenshots:
1. User query → "Will it rain in Accra this weekend?"
2. Vector search → Find relevant tools
3. LLM reasoning → Select best tool
4. MCP tool call → Execute with parameters
5. Response → Natural language output

## Files Overview

### Ready to Run
- **`standalone_demo.py`** - Complete demo, no dependencies required ✅
  - Simulates the entire flow
  - Shows step-by-step execution
  - No API keys needed

### Production Implementation
- **`agent_with_vector_tools.py`** - Full LangGraph implementation
  - Requires: OpenAI API key, LangGraph, ChromaDB
  - Real embeddings and vector search
  - Actual LLM reasoning
  - Production-ready

### Examples and Docs
- **`example_usage.py`** - Multiple use case examples
- **`demo_agent.py`** - Demo version with mock LLM
- **`README.md`** - Comprehensive documentation
- **`ARCHITECTURE.md`** - System architecture diagrams
- **`requirements.txt`** - Python dependencies

## Installation (for Full Version)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key
export OPENAI_API_KEY='your-key-here'

# 4. Run
python agent_with_vector_tools.py
```

## What You'll See

When you run `standalone_demo.py`, you'll see output like this:

```
██████████████████████████████████████████████████████████████████
  LANGGRAPH-STYLE AGENT WITH VECTOR DB TOOL RETRIEVAL
██████████████████████████████████████████████████████████████████

======================================================================
STEP 1: USER ASKS A QUESTION
======================================================================

💬 User: "Will it rain in Accra this weekend?"

======================================================================
STEP 2: AGENT SEARCHES TOOL VECTORS
======================================================================

🔍 Python Code:
   tools = vector_db.similarity_search(query_embedding, k=3)

📋 Retrieved Tools (top 3):
   1. weather.get_forecast
      └─ Get 7-day weather forecast for a city
   2. calendar.get_events
      └─ Retrieve calendar events for a date range
   3. stock.get_price
      └─ Get current stock price and market data

======================================================================
STEP 3: AGENT REASONS
======================================================================

💭 Agent's Internal Reasoning:
   • "This looks like a weather forecast request"
   • "Tool: weather.get_forecast matches intent"

✅ Selected Tool: weather.get_forecast

======================================================================
STEP 4: AGENT CALLS MCP TOOL
======================================================================

📤 MCP Tool Call (JSON):
   {
   "name": "weather.get_forecast",
   "arguments": {
      "city": "Accra",
      "units": "metric"
   }
}

======================================================================
STEP 5: TOOL EXECUTES → RESPONSE RETURNED
======================================================================

📥 Tool Response:
   {
   "city": "Accra",
   "units": "metric",
   "forecast": [...]
   }

======================================================================
STEP 6: GENERATE FINAL RESPONSE
======================================================================

💬 Final Response to User:
   Here's the weekend forecast for Accra:
   • Saturday: Partly Cloudy, 28°C (20% chance of rain)
   • Sunday: Light Rain, 26°C (60% chance of rain)
   
   Yes, there's a chance of rain this weekend!
```

## Key Concepts

### What Goes in Vector DB
```json
[
   {
   "tool_id": "get_forecast",
   "description": "Get 7-day weather forecast for a city",
   "inputs": {
      "city": "string",
      "units": "metric | imperial"
   },
   "when_to_use": "User asks about future weather conditions",
   "examples": [
      "What will the weather be in Accra next week?",
      "Is it going to rain tomorrow in NYC?"
   ]
   },
   {
      "tool_id": "email_send",
      "description": "Send an email to a recipient",
      "inputs": {
         "to": "email address",
         "subject": "string",
         "body": "string"
      },
      "when_to_use": "User wants to send an email",
      "examples": ["Send an email to john@example.com"]
   }
]

```

**Important**: Only descriptors go in the vector DB, NOT executable code!

### The Flow

```
User Query
    ↓
Vector Search (semantic similarity)
    ↓
LLM Reasoning (select best tool)
    ↓
Parameter Extraction (LLM)
    ↓
MCP Tool Execution
    ↓
Natural Language Response
```

## Customization

### Add Your Own Tool

1. **Define the descriptor**:
```python
my_tool = {
    "tool_id": "email.send",
    "description": "Send an email to a recipient",
    "inputs": {
        "to": "email address",
        "subject": "string",
        "body": "string"
    },
    "when_to_use": "User wants to send an email",
    "examples": ["Send an email to john@example.com"]
}

TOOL_DESCRIPTORS.append(my_tool)
```

2. **Implement the function**:
```python
@staticmethod
def email_send(to: str, subject: str, body: str) -> Dict[str, Any]:
    # Your implementation
    return {"status": "sent", "to": to}
```

3. **Register it**:
```python
MCPToolRegistry.email_send = email_send
```

## Next Steps

1. Run `standalone_demo.py` to see it in action
2. Read `README.md` for detailed documentation
3. Check `ARCHITECTURE.md` for system design
4. Try `agent_with_vector_tools.py` for production version
5. Experiment with adding your own tools

## Troubleshooting

**Q: Import errors when running agent_with_vector_tools.py?**
A: Run `pip install -r requirements.txt` first

**Q: OpenAI API errors?**
A: Set your API key: `export OPENAI_API_KEY='your-key'`

**Q: Want to test without API keys?**
A: Use `standalone_demo.py` instead - works immediately!

## Support

For issues or questions:
1. Check the README.md
2. Review the code comments
3. See ARCHITECTURE.md for design details
