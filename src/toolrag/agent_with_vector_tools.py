"""
End-to-end LangGraph Agent with Vector DB Tool Retrieval
Implements the flow shown in the screenshots:
1. User asks a question
2. Agent searches tool vectors
3. Agent reasons about which tool to use
4. Agent calls MCP tool
5. Tool executes and returns response
"""

from loguru import logger

try:
    from toolrag.logging import setup_logging
except Exception:
    def setup_logging() -> None:  # type: ignore[no-redef]
        return None

import json
import operator
import os
from typing import Annotated, Sequence, TypedDict

import faiss
from dotenv import load_dotenv
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, StateGraph

from embed_factory import EmbeddingFactory, create_embedding
from llm_factory import LLMFactory, get_llm
from tool_descriptors import (TOOL_DESCRIPTORS, calculator_compute,
                              calendar_get_events, stock_get_price,
                              weather_get_forecast)
from tool_vectordb import ToolVectorDB
from vector_store_factory import VectorStoreFactory, create_vector_store

# Load environment variables from .env file
load_dotenv("../.env", override=True) 

# Configure logging for script runs
setup_logging()
logger.debug("Logging initialized for 01_agent_with_vector_tools")

# ============================================================================
# EMBEDDING MODEL
# ============================================================================
# Create embeddings using the factory
# Options: "huggingface", "ollama", "openai"


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """The state of our agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    retrieved_tools: list[dict]
    selected_tool: dict | None
    tool_result: str | None
    attempt_count: int
    max_attempts: int
    refined_query: str
    tool_execution_success: bool
    tool_confidence: float
    confidence_threshold: float
    failed_tools: list[str]
    user_feedback: str | None
    requires_user_feedback: bool
    tried_tools_count: int


# Map tool IDs to actual tool functions
TOOL_REGISTRY = {
    "weather.get_forecast": weather_get_forecast,
    "calendar.get_events": calendar_get_events,
    "stock.get_price": stock_get_price,
    "calculator.compute": calculator_compute
}


# ============================================================================
# AGENT NODES
# ============================================================================

def retrieve_tools_node(
    state: AgentState,
    vector_store_provider: str = "chromadb",
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small"
) -> AgentState:
    """
    Step 2: Agent searches tool vectors
    Retrieves relevant tools from vector DB based on user query
    On retry, uses refined_query for better matching
    
    Args:
        state: Current agent state
        vector_store_provider: Vector store to use
        embedding_provider: Embedding model provider
        embedding_model: Specific embedding model
    """
    print("\n=== STEP 2: RETRIEVING TOOLS FROM VECTOR DB ===")
    print(f"Using vector store: {vector_store_provider}")
    print(f"Using embeddings: {embedding_provider}/{embedding_model}")
    
    user_query = state["user_query"]
    attempt = state["attempt_count"]
    
    # Use refined query if this is a retry attempt
    search_query = state["refined_query"] if state["refined_query"] else user_query
    
    if attempt > 1:
        print(f"Attempt {attempt}: Using refined query")
        print(f"Original: {user_query}")
        print(f"Refined: {search_query}")
    
    # Initialize vector DB with specified providers
    tool_db = ToolVectorDB(
        TOOL_DESCRIPTORS,
        vector_store_provider=vector_store_provider,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model
    )
    
    # Search for relevant tools
    retrieved_tools = tool_db.similarity_search(search_query, k=3)
    
    print(f"Query: {search_query}")
    print(f"Retrieved {len(retrieved_tools)} tools:")
    for i, tool in enumerate(retrieved_tools, 1):
        print(f"  {i}. {tool['tool_id']}")
    
    return {
        **state,
        "retrieved_tools": retrieved_tools
    }


def reasoning_node(state: AgentState) -> AgentState:
    """
    Step 3: Agent reasons about which tool to use
    Uses LLM to select the most appropriate tool with confidence scoring
    """
    print("\n=== STEP 3: REASONING ABOUT TOOL SELECTION ===")
    
    llm = get_llm(provider="ollama", model="gpt-oss:20b", temperature=0)
    
    user_query = state["user_query"]
    retrieved_tools = state["retrieved_tools"]
    failed_tools = state["failed_tools"]
    
    # Filter out failed tools from previous attempts
    available_tools = [t for t in retrieved_tools if t["tool_id"] not in failed_tools]
    
    if not available_tools:
        print("No available tools left to try")
        return {
            **state,
            "selected_tool": None,
            "tool_confidence": 0.0
        }
    
    # Create reasoning prompt with confidence scoring
    tools_text = "\n\n".join([
        f"Tool {i+1}: {tool['tool_id']}\n"
        f"Description: {tool['description']}\n"
        f"When to use: {tool['when_to_use']}"
        for i, tool in enumerate(available_tools)
    ])
    
    # Create a list of valid tool IDs for the LLM
    valid_tool_ids = [tool["tool_id"] for tool in available_tools]
    
    reasoning_prompt = f"""
You are analyzing a user query to select the most appropriate tool.

User Query: "{user_query}"

Available Tools:
{tools_text}

IMPORTANT: You must select ONE of these exact tool IDs: {valid_tool_ids}

Analyze the query and determine which tool best matches the user's intent.
Provide a confidence score (0.0-1.0) based on how well the tool matches the query.
1.0 = perfect match, 0.5 = moderate match, 0.0 = no match

Respond with ONLY a JSON object:
{{
    "reasoning": "your reasoning here",
    "selected_tool_id": "MUST BE ONE OF: {', '.join(valid_tool_ids)}",
    "confidence": 0.85
}}
"""
    
    messages = [SystemMessage(content=reasoning_prompt)]
    response = llm.invoke(messages)
    
    # Parse response
    try:
        reasoning_result = json.loads(response.content)
        selected_tool_id = reasoning_result["selected_tool_id"]
        confidence = reasoning_result.get("confidence", 0.5)
        
        # Find the full tool descriptor
        selected_tool = next(
            (tool for tool in available_tools if tool["tool_id"] == selected_tool_id),
            None
        )
        
        # If tool_id wasn't found, use first available tool
        if selected_tool is None:
            print(f"Warning: Invalid tool_id '{selected_tool_id}' returned, using first available tool")
            selected_tool = available_tools[0] if available_tools else None
            confidence = 0.3
        
        print(f"Reasoning: {reasoning_result['reasoning']}")
        print(f"Selected Tool: {selected_tool['tool_id'] if selected_tool else 'None'}")
        print(f"Confidence: {confidence:.2f}")
        
    except json.JSONDecodeError:
        print("Error parsing LLM response, using first tool with low confidence")
        selected_tool = available_tools[0] if available_tools else None
        confidence = 0.3
    
    # Add reasoning message to conversation
    messages_update = [
        AIMessage(content=f"Reasoning: {reasoning_result.get('reasoning', 'Selected based on similarity')}")
    ]
    
    return {
        **state,
        "messages": messages_update,
        "selected_tool": selected_tool,
        "tool_confidence": confidence
    }


def tool_execution_node(state: AgentState) -> AgentState:
    """
    Step 4 & 5: Agent calls MCP tool and receives response
    Executes the selected tool with extracted parameters
    Tracks success/failure for validation
    """
    print("\n=== STEP 4-5: EXECUTING TOOL ===")
    
    llm = get_llm(temperature=0)
    
    user_query = state["user_query"]
    selected_tool = state["selected_tool"]
    
    if not selected_tool:
        return {
            **state,
            "tool_result": "No tool selected",
            "tool_execution_success": False
        }
    
    # Extract parameters using LLM
    param_extraction_prompt = f"""
Extract the parameters for this tool call from the user query.

User Query: "{user_query}"

Tool: {selected_tool['tool_id']}
Required Inputs: {json.dumps(selected_tool['inputs'], indent=2)}

Respond with ONLY a JSON object containing the parameter values:
"""
    
    messages = [SystemMessage(content=param_extraction_prompt)]
    param_response = llm.invoke(messages)
    
    try:
        params = json.loads(param_response.content)
        print(f"Extracted parameters: {params}")
    except json.JSONDecodeError:
        params = {}
        print("Could not extract parameters, using empty dict")
    
    # Execute the tool
    tool_function = TOOL_REGISTRY.get(selected_tool["tool_id"])
    execution_success = False
    
    if tool_function:
        print(f"Calling tool: {selected_tool['tool_id']}")
        print(f"Arguments: {json.dumps(params, indent=2)}")
        
        try:
            result = tool_function.invoke(params)
            tool_result = json.dumps(result, indent=2)
            print(f"Tool Result: {tool_result}")
            execution_success = True
        except Exception as e:
            tool_result = f"Error executing tool: {str(e)}"
            print(tool_result)
            execution_success = False
    else:
        tool_result = f"Tool {selected_tool['tool_id']} not found in registry"
        print(tool_result)
        execution_success = False
    
    return {
        **state,
        "tool_result": tool_result,
        "tool_execution_success": execution_success
    }


def validation_node(state: AgentState) -> AgentState:
    """
    Step 5.5: Validate tool execution result
    Handles three scenarios:
    1. If execution succeeded → proceed to response generation
    2. If failed but confidence is low → ask for user feedback
    3. If failed with high confidence → try next tool or refine query
    """
    print("\n=== STEP 5.5: VALIDATING TOOL EXECUTION ===")
    
    execution_success = state["tool_execution_success"]
    attempt = state["attempt_count"]
    max_attempts = state["max_attempts"]
    confidence = state["tool_confidence"]
    confidence_threshold = state["confidence_threshold"]
    tried_tools = state["tried_tools_count"]
    retrieved_tools = state["retrieved_tools"]
    failed_tools = state["failed_tools"]
    
    print(f"Execution successful: {execution_success}")
    print(f"Confidence: {confidence:.2f} (threshold: {confidence_threshold})")
    print(f"Attempt: {attempt}/{max_attempts}")
    print(f"Tools tried: {tried_tools}/{len(retrieved_tools)}")
    
    if execution_success:
        print("✓ Tool executed successfully, proceeding to response generation")
        return state
    
    # If confidence is below threshold, ask for user feedback
    elif confidence < confidence_threshold:
        print(f"✗ Low confidence ({confidence:.2f}) - requesting user clarification")
        
        selected_tool = state["selected_tool"]
        return {
            **state,
            "requires_user_feedback": True,
            "user_feedback": None
        }
    
    # Try next tool if available
    elif tried_tools < len(retrieved_tools):
        print(f"✗ Tool execution failed, trying next tool ({tried_tools + 1}/{len(retrieved_tools)})...")
        
        if state["selected_tool"]:
            failed_tools.append(state["selected_tool"]["tool_id"])
        
        return {
            **state,
            "tried_tools_count": tried_tools + 1,
            "failed_tools": failed_tools,
            "tool_execution_success": False,
            "selected_tool": None,
            "tool_result": None
        }
    
    # Max tools tried, now refine query with LLM if attempts remaining
    elif attempt < max_attempts:
        print(f"✗ All tools tried, refining query (attempt {attempt + 1}/{max_attempts})...")
        
        llm = get_llm(temperature=0.3)
        
        refinement_prompt = f"""
Multiple tool attempts have failed. Help refine the user's query for better matching.

Original Query: "{state['user_query']}"
Failed Tools: {', '.join(failed_tools)}
Last Error: {state['tool_result']}

Provide a refined version of the query that describes the intent differently.
Respond with ONLY the refined query text, nothing else.
"""
        
        messages = [SystemMessage(content=refinement_prompt)]
        refinement_response = llm.invoke(messages)
        refined_query = refinement_response.content.strip()
        
        print(f"Refined query: {refined_query}")
        
        return {
            **state,
            "attempt_count": attempt + 1,
            "refined_query": refined_query,
            "tool_execution_success": False,
            "selected_tool": None,
            "tool_result": None,
            "tried_tools_count": 0,
            "failed_tools": []
        }
    
    else:
        print(f"✗ Max attempts ({max_attempts}) and tools exhausted, proceeding to fallback response")
        return state


def user_feedback_node(state: AgentState) -> AgentState:
    """
    Step 5.7: Request user feedback when tool selection confidence is low
    In a real implementation, this would prompt the user for clarification
    """
    print("\n=== STEP 5.7: REQUESTING USER FEEDBACK ===")
    
    user_query = state["user_query"]
    selected_tool = state["selected_tool"]
    confidence = state["tool_confidence"]
    
    print(f"Low confidence tool selection detected ({confidence:.2f})")
    print(f"Selected Tool: {selected_tool['tool_id'] if selected_tool else 'None'}")
    
    # In a real system, this would show a UI asking the user to clarify
    feedback_prompt = f"""
The system is unsure about the best tool for your query. Please clarify:

Your Query: "{user_query}"
Suggested Tool: {selected_tool['tool_id'] if selected_tool else 'None'}
Confidence: {confidence:.2f}

Options:
1. Proceed with the suggested tool
2. Describe what you want more clearly
3. Choose a different tool

For this demo, we'll proceed with the suggested tool anyway.
"""
    
    print(feedback_prompt)
    
    # For demo purposes, we automatically proceed
    # In a real system, user_feedback would be populated from user input
    user_feedback = state.get("user_feedback") or "proceed"
    
    print(f"Using feedback: {user_feedback}")
    
    return {
        **state,
        "requires_user_feedback": False,
        "user_feedback": user_feedback
    }


def response_generation_node(state: AgentState) -> AgentState:
    """
    Generate final response to user based on tool result or feedback status
    """
    print("\n=== GENERATING FINAL RESPONSE ===")
    
    llm = get_llm(temperature=0.7)
    
    user_query = state["user_query"]
    tool_result = state["tool_result"]
    execution_success = state["tool_execution_success"]
    user_feedback = state.get("user_feedback")
    
    if execution_success:
        response_prompt = f"""
Generate a natural, helpful response to the user based on the tool execution result.

User Query: "{user_query}"

Tool Result:
{tool_result}

Provide a clear, conversational response that directly answers the user's question.
"""
    elif user_feedback:
        response_prompt = f"""
The system had low confidence in the tool selection and requested user feedback.
The user provided feedback, and we're proceeding.

User Query: "{user_query}"
User Feedback: {user_feedback}

Generate a helpful response acknowledging their clarification and explaining next steps.
"""
    else:
        response_prompt = f"""
The agent attempted multiple times but could not find and execute the appropriate tool.
Generate a helpful response explaining the situation and what went wrong.

User Query: "{user_query}"
Attempts: {state['attempt_count']}/{state['max_attempts']}
Tools Tried: {state['tried_tools_count']}
Last Error: {tool_result}

Provide a helpful response that apologizes for the issue and suggests alternatives if possible.
"""
    
    messages = [SystemMessage(content=response_prompt)]
    response = llm.invoke(messages)
    
    final_message = AIMessage(content=response.content)
    
    print(f"\nFinal Response: {response.content}")
    
    return {
        **state,
        "messages": [final_message]
    }


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_agent_graph(
    vector_store_provider: str = "chromadb",
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small"
):
    """Create the LangGraph workflow with multi-tool, confidence-based, and user feedback features
    
    Args:
        vector_store_provider: Vector store to use ("chromadb", "faiss", "milvus", "qdrant")
        embedding_provider: Embedding provider ("huggingface", "ollama", "openai")
        embedding_model: Specific embedding model name
    """
    
    workflow = StateGraph(AgentState)
    
    # Create a wrapper for retrieve_tools_node with the vector store and embedding providers
    def retrieve_tools_wrapper(state: AgentState) -> AgentState:
        return retrieve_tools_node(
            state,
            vector_store_provider=vector_store_provider,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model
        )
    
    # Add nodes
    workflow.add_node("retrieve_tools", retrieve_tools_wrapper)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("user_feedback", user_feedback_node)
    workflow.add_node("response_generation", response_generation_node)
    
    # Define edges (flow with complex routing)
    workflow.set_entry_point("retrieve_tools")
    workflow.add_edge("retrieve_tools", "reasoning")
    workflow.add_edge("reasoning", "tool_execution")
    workflow.add_edge("tool_execution", "validation")
    
    # Conditional edges from validation
    workflow.add_conditional_edges(
        "validation",
        validation_router,
        {
            "success": "response_generation",
            "user_feedback": "user_feedback",
            "retry_tool": "reasoning",
            "refine_query": "retrieve_tools",
            "fallback": "response_generation"
        }
    )
    
    # Edge from user feedback
    workflow.add_edge("user_feedback", "tool_execution")
    
    # Final edge
    workflow.add_edge("response_generation", END)

    return workflow.compile()


def validation_router(state: AgentState) -> str:
    """
    Router to determine the next step after validation:
    - "success": Tool executed successfully
    - "user_feedback": Low confidence, request clarification
    - "retry_tool": Try next tool in retrieved list
    - "refine_query": All tools tried, refine query and restart
    - "fallback": Max attempts reached, generate fallback response
    """
    execution_success = state["tool_execution_success"]
    confidence = state["tool_confidence"]
    confidence_threshold = state["confidence_threshold"]
    tried_tools = state["tried_tools_count"]
    retrieved_tools = state["retrieved_tools"]
    failed_tools = state["failed_tools"]
    attempt = state["attempt_count"]
    max_attempts = state["max_attempts"]
    
    if execution_success:
        return "success"
    elif confidence < confidence_threshold:
        return "user_feedback"
    elif tried_tools < len(retrieved_tools):
        return "retry_tool"
    elif attempt < max_attempts:
        return "refine_query"
    else:
        return "fallback"


def should_retry(state: AgentState) -> str:
    """Legacy function - kept for backward compatibility"""
    router = validation_router(state)
    return "retry" if router in ["retry_tool", "refine_query"] else "proceed"


# ============================================================================
# GRAPH VISUALIZATION UTILITY
# ============================================================================
def write_graph_to_file(graph: StateGraph, filename: str=None):
    """Write the graph structure to a file (for visualization/debugging)"""
    import uuid
    
    if filename is None:
        filename = f"agent_graph_{uuid.uuid4().hex}.png"
    # Generate the image data
    image_data = graph.get_graph().draw_mermaid_png()

    # Save the image data to a file
    with open(filename, mode="wb") as f:
        f.write(image_data)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_agent(
    user_query: str,
    vector_store_provider: str = "chromadb",
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small"
):
    """Run the agent with a user query
    
    Args:
        user_query: The user's question
        vector_store_provider: Vector store to use ("chromadb", "faiss", "milvus", "qdrant")
        embedding_provider: Embedding provider ("huggingface", "ollama", "openai")
        embedding_model: Specific embedding model name
    """
    
    print("=" * 80)
    print("AGENT FLOW (STEP-BY-STEP)")
    print("=" * 80)
    print(f"\n=== STEP 1: USER ASKS A QUESTION ===")
    print(f'"{user_query}"')
    
    # Initialize state
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "retrieved_tools": [],
        "selected_tool": None,
        "tool_result": None,
        "attempt_count": 1,
        "max_attempts": 3,
        "refined_query": "",
        "tool_execution_success": False,
        "tool_confidence": 0.0,
        "confidence_threshold": 0.6,  # Require 60% confidence, otherwise ask for feedback
        "failed_tools": [],
        "user_feedback": None,
        "requires_user_feedback": False,
        "tried_tools_count": 0
    }
    
    # Create graph with providers
    graph = create_agent_graph(
        vector_store_provider=vector_store_provider,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model
    )
    
    if graph is None:
        raise ValueError("Graph creation failed.")
    
    # Visualize graph
    # write_graph_to_file(graph)
    
    final_state = graph.invoke(initial_state)
    
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)
    
    return final_state


if __name__ == "__main__":

    query = " Will it rain in Accra this weekend?" 
    # query = "whats on my calendar this week?"  
    # query = "What's the current price of Apple stock?"
    # query = "What's 15% of 250?"
    
    result = run_agent(user_query=query)
    
    print("\n\nFinal Messages:")
    for msg in result["messages"]:
        print(f"\n{msg.__class__.__name__}: {msg.content}")