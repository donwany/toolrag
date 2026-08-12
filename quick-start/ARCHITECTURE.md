# Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION                                 │
│                                                                          │
│  User: "Will it rain in Accra this weekend?"                            │
└─────────────────────────┬────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH AGENT ORCHESTRATOR                          │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ STATE: {                                                        │    │
│  │   user_query: str                                              │    │
│  │   retrieved_tools: List[Dict]                                  │    │
│  │   selected_tool: Dict                                          │    │
│  │   tool_result: Any                                             │    │
│  │ }                                                               │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │  Retrieve   │→ │ Reasoning│→ │   Tool   │→ │ Response │            │
│  │   Tools     │  │          │  │ Execution│  │Generation│            │
│  └─────────────┘  └──────────┘  └──────────┘  └──────────┘            │
└───────┬───────────────┬──────────────┬────────────────────────────────┘
        │               │              │
        ▼               ▼              ▼
┌───────────────┐ ┌──────────┐ ┌─────────────────┐
│  VECTOR DB    │ │   LLM    │ │  MCP TOOLS      │
│  (ChromaDB)   │ │  GPT-4   │ │  (Executable)   │
│               │ │          │ │                 │
│ Tool          │ │ Reasons  │ │ weather_func()  │
│ Descriptors   │ │ Selects  │ │ calendar_func() │
│ (Embedded)    │ │ Extracts │ │ stock_func()    │
└───────────────┘ └──────────┘ └─────────────────┘
```

```
+--------+            +---------+                 +---------+
|  User  |            |  Client |                 |  Server |
+--------+            +---------+                 +---------+

                             ┌────────────────────────────────┐
                             │ Server initiates processing     │
                             └───────────────┬────────────────┘
                                             │
                                             │ retrieve_tools
                                             ▼
                             ┌────────────────────────────────┐
                             │ reasoning (with confidence)     │
                             └───────────────┬────────────────┘
                                             │
                                             │ tool_execution
                                             ▼
                             ┌────────────────────────────────┐
                             │ tool_execution (track success)  │
                             └───────────────┬────────────────┘
                                             │
                                             │ validation
                                             ▼
                             ┌────────────────────────────────┐
                             │ validation (router)             │
                             └───────┬───────────────┬────────┘
                                     │               │
                                     │ success       │ needs info
                                     │               │
                                     ▼               ▼
                    ┌────────────────────────┐   ┌─────────────────────┐
                    │ response_generation    │   │ elicitation/create   │
                    │ (success response)     │   └─────────┬───────────┘
                    └──────────┬─────────────┘             │
                               │                            │
                               │ return response            │ present UI
                               ▼                            ▼
+--------+            +---------+                 +---------+
|  User  |◀────────── |  Client | ◀───────────── |  Server |
+--------+            +---------+                 +---------+
      ▲                     │
      │ provide info        │ collect input
      │                     ▼
      │           ┌────────────────────────┐
      │           │ user_feedback (prompt) │
      │           └──────────┬─────────────┘
      │                      │
      │                      │ continue processing
      │                      ▼
                             ┌────────────────────────────────┐
                             │ tool_execution (with feedback)  │
                             └───────────────┬────────────────┘
                                             │
                                             ▼
                             ┌────────────────────────────────┐
                             │ response_generation             │
                             │ (success or fallback)           │
                             └────────────────────────────────┘


```


```
retrieve_tools → reasoning → tool_execution → validation ─┐
                                                           ├→ [success?]
                                                           │
                                        [retry & attempts remaining]
                                                 ↓
                                        refine query
                                        [loop back] ←──┘
                                                   OR
                                        [proceed to response]
                                                   ↓
                                        response_generation → END



Key Changes:

Updated AgentState - Added 4 new fields:

attempt_count: Tracks current attempt (1-indexed)
max_attempts: Maximum retries (default: 3)
refined_query: Stores refined query for retry
tool_execution_success: Boolean flag for execution status
Enhanced retrieve_tools_node - Now checks if it's a retry and uses refined_query if available

Updated tool_execution_node - Returns tool_execution_success flag instead of assuming success

New validation_node - Core retry logic:

If tool executed successfully → proceeds to response generation
If failed AND attempts < max_attempts → generates refined query and sets up retry
If max attempts exceeded → generates fallback response
Updated response_generation_node - Detects execution success and generates appropriate response (success vs. fallback error message)

Added should_retry() - Conditional edge function that routes either back to retrieve_tools (retry) or forward to response_generation (proceed)
```

## Detailed Flow

### Step 1: User Input
```
User Query
    │
    ├─> "Will it rain in Accra this weekend?"
    │
    └─> Sent to Agent
```

### Step 2: Vector Search
```
Agent
    │
    ├─> Embed query using OpenAI embeddings
    │
    ├─> vector_db.similarity_search(query_embedding, k=3)
    │
    └─> Returns: [weather.get_forecast, calendar.get_events, stock.get_price]
```

### Step 3: LLM Reasoning
```
LLM (GPT-4)
    │
    ├─> Input: User query + Retrieved tool descriptors
    │
    ├─> Analysis:
    │   • "This looks like a weather forecast request"
    │   • "Tool: weather.get_forecast matches intent"
    │
    └─> Output: {selected_tool_id: "weather.get_forecast"}
```

### Step 4: Tool Call Preparation
```
LLM (GPT-4)
    │
    ├─> Extract parameters from query
    │
    └─> Output: {
          "city": "Accra",
          "units": "metric"
        }
```

### Step 5: MCP Tool Execution
```
MCP Tool Call
    │
    ├─> {
    │     "name": "weather.get_forecast",
    │     "arguments": {
    │       "city": "Accra",
    │       "units": "metric"
    │     }
    │   }
    │
    ├─> Execute actual tool function
    │
    └─> Response: {
          "city": "Accra",
          "forecast": [
            {"day": "Sat", "temp": 28, "condition": "Partly Cloudy"},
            {"day": "Sun", "temp": 26, "condition": "Light Rain"}
          ]
        }
```

### Step 6: Response Generation
```
LLM (GPT-4)
    │
    ├─> Input: Tool result
    │
    ├─> Generate natural language response
    │
    └─> Output: "Here's the weekend forecast for Accra:
                • Saturday: Partly Cloudy, 28°C
                • Sunday: Light Rain, 26°C
                Yes, there's a chance of rain!"
```

## Data Flow Diagram

```
┌─────────────┐
│    User     │
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Query Embedding                    │
│  "Will it rain..." → [0.12, 0.45...]│
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Vector DB Search                   │
│  ┌───────────────────────────────┐  │
│  │ Tool 1: weather.get_forecast  │  │
│  │ Embedding: [0.15, 0.43, ...]  │  │
│  │ Similarity: 0.92              │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ Tool 2: calendar.get_events   │  │
│  │ Embedding: [0.35, 0.12, ...]  │  │
│  │ Similarity: 0.23              │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ Tool 3: stock.get_price       │  │
│  │ Embedding: [0.55, 0.67, ...]  │  │
│  │ Similarity: 0.18              │  │
│  └───────────────────────────────┘  │
└──────┬──────────────────────────────┘
       │ Top 3 tools
       ▼
┌─────────────────────────────────────┐
│  LLM Reasoning                      │
│  Analyzes: query + tool descriptors │
│  Selects: weather.get_forecast      │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Parameter Extraction               │
│  Extract: city="Accra", units="..."│
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  MCP Tool Execution                 │
│  Call: weather.get_forecast(...)    │
│  Result: {forecast: [...]}          │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Response Generation                │
│  Format result as natural language  │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────┐
│  Response   │
│  to User    │
└─────────────┘
```

## Key Components

### 1. Vector Database (ChromaDB)
- **Purpose**: Store tool descriptors, not executable code
- **Data Format**: JSON tool descriptors
- **Search Method**: Cosine similarity on embeddings
- **Index Size**: O(n) where n = number of tools

### 2. LangGraph State Machine
- **State**: Tracks query, retrieved tools, selected tool, results
- **Nodes**: Retrieve, Reason, Execute, Generate
- **Edges**: Sequential flow with no cycles
- **Persistence**: State passed between nodes

### 3. Language Model (GPT-4)
- **Role 1**: Tool selection reasoning
- **Role 2**: Parameter extraction
- **Role 3**: Response generation
- **Temperature**: 0 for reasoning, 0.7 for generation

### 4. MCP Tool Registry
- **Pattern**: Registry pattern for tool lookup
- **Tools**: Actual executable functions
- **Interface**: Standardized input/output format
- **Error Handling**: Try-catch with fallbacks

## Scalability

### Adding New Tools
1. Create tool descriptor (JSON)
2. Implement tool function
3. Add to registry
4. Re-index vector DB

### Performance Optimization
- Cache embeddings
- Batch vector searches
- Use faster embeddings model
- Implement tool result caching

### Monitoring
- Track retrieval accuracy
- Log tool selection decisions
- Monitor execution times
- Alert on failures
