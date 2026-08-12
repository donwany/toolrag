"""
End-to-end LangGraph Agent with Vector DB Tool Retrieval
Implements the flow shown in the screenshots:
1. User asks a question
2. Agent searches tool vectors
3. Agent reasons about which tool to use
4. Agent calls MCP tool
5. Tool executes and returns response
"""
from toolrag import __version__
import asyncio
from typing import Any
import os
import json
import operator
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage, SystemMessage)
from langchain_core.tools import tool
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from langchain_mcp_adapters.client import MultiServerMCPClient
from loguru import logger
from toolrag.utils import create_parser
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

try:
    # When running as installed package: python -m tv ...
    from logging_factory import setup_logging
    from console_factory import panel, tools_table, pretty_json, scores_table
except Exception:
    # Fallback for direct file execution from within src/tv/
    def setup_logging() -> None:  # type: ignore[no-redef]
        return None

    def panel(title: str, body: str, **_: Any) -> None:  # type: ignore[no-redef]
        # Ensure user-visible output even without Rich installed/importable.
        print(f"\n[{title}]\n{body}\n")

    def tools_table(tools: list[dict], **_: Any) -> None:  # type: ignore[no-redef]
        print("\n[Retrieved tools]")
        for i, tool in enumerate(tools, 1):
            print(f"  {i}. {tool.get('tool_id')} - {tool.get('description', '')}")
        print()

    def pretty_json(data: Any, **_: Any) -> None:  # type: ignore[no-redef]
        try:
            if isinstance(data, str):
                print(data)
            else:
                print(json.dumps(data, indent=2, default=str))
        except Exception:
            print(data)

from embed_factory import EmbeddingFactory, create_embedding
from llm_factory import LLMFactory, get_llm
from tool_descriptors import TOOL_DESCRIPTORS
from tool_vectordb import ToolVectorDB
from vector_store_factory import VectorStoreFactory, create_vector_store

# Load environment variables from .env file
load_dotenv("../.env", override=True) 

# ============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================================
args = create_parser()
query = args.query
vector_store_provider = args.vector_store_provider
embedding_provider = args.embedding_provider
embedding_model = args.embedding_model
llm_provider = args.llm_provider
# temperature = args.temperature
num_tools = args.num_tools
confidence_threshold = args.confidence_threshold
max_attempts = args.max_attempts
retrieved_tools = args.retrieved_tools
failed_tools = args.failed_tools
selected_tool = args.selected_tool
tool_result = args.tool_result
tool_execution_success = args.tool_execution_success
tool_confidence = args.tool_confidence
refined_query = args.refined_query
requires_user_feedback = args.requires_user_feedback
tried_tools_count = args.tried_tools_count
mcp_tool_map = args.mcp_tool_map
validation_result = args.validation_result
user_feedback = args.user_feedback
messages = args.messages
attempt_count = args.attempt_count
r_temperature = args.r_temperature
g_temperature = args.g_temperature

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
    validation_result: str  # "success" | "feedback" | "retry_tool" | "retry_query" | "failed"
    # MCP tool name -> LangChain tool instance (from MultiServerMCPClient)
    mcp_tool_map: dict[str, Any]


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
    logger.info("STEP 2: Retrieving tools from vector DB")
    logger.info(f"vector_store_provider={vector_store_provider}, embeddings={embedding_provider}/{embedding_model}")
    
    user_query = state["user_query"]
    attempt = state["attempt_count"]
    
    # Use refined query if this is a retry attempt
    search_query = state["refined_query"] if state["refined_query"] else user_query
    
    if attempt > 1:
        logger.warning("Retry attempt {} using refined query", attempt)
        logger.debug("original_query={!r}", user_query)
        logger.debug("refined_query={!r}", search_query)
    
    # Initialize vector DB with specified providers
    tool_db = ToolVectorDB(
        tool_descriptors=TOOL_DESCRIPTORS,
        vector_store_provider=vector_store_provider,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model
    )
    
    # Search for relevant tools
    retrieved_tools = tool_db.similarity_search_with_score(search_query, k=num_tools)
    
    logger.info("Retrieved {} tools for query={!r}", len(retrieved_tools), search_query)
    logger.info("Retrieved_tool_ids={}", [t.get("tool_id") for t in retrieved_tools])
    # User-facing (pretty) output when available
    try:
        tools_table(retrieved_tools, title="Retrieved tools (top-k)")
    except Exception:
        pass
    
    try:
        scores_table(retrieved_tools, title="Retrieved tools with scores")
    except Exception:
        pass
    
    return {
        **state,
        "retrieved_tools": retrieved_tools
    }


def reasoning_node(state: AgentState) -> AgentState:
    """
    Step 3: Agent reasons about which tool to use
    Uses LLM to select the most appropriate tool with confidence scoring
    """
    logger.info("STEP 3: Reasoning about tool selection")

    llm = get_llm(provider=llm_provider, temperature=r_temperature)
    
    user_query = state["user_query"]
    retrieved_tools = state["retrieved_tools"]
    failed_tools = state["failed_tools"]
    
    # Filter out failed tools from previous attempts
    available_tools = [t for t in retrieved_tools if t["tool_id"] not in failed_tools]
    
    if not available_tools:
        logger.error("No available tools left to try (all retrieved tools previously failed)")
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
1.0 = perfect match, 0.5 = moderate match, 0.0 = no match.

If none of the tools does not correspond to any of the available tools with confidence = 0.0.

Do not try answering the query, just retry.

Respond with ONLY a JSON object:
{{
    "reasoning": "your reasoning here",
    "selected_tool_id": "MUST BE ONE OF: {', '.join(valid_tool_ids)}",
    "confidence": 0.85
}}
"""

    # prompt = tool_selection_prompt.format_messages(
    #     user_query=user_query,
    #     tools_text=tools_text,
    #     valid_tool_ids=", ".join(valid_tool_ids),
    # )

    messages = [SystemMessage(content=reasoning_prompt)]
    response = llm.invoke(messages)
    
    # Parse response
    try:
        reasoning_result = json.loads(response.content)
        selected_tool_id = reasoning_result["selected_tool_id"]
        raw_confidence = reasoning_result.get("confidence", 0.5)
        confidence = float(raw_confidence) if raw_confidence is not None else 0.5
        
        # Find the full tool descriptor
        selected_tool = next(
            (tool for tool in available_tools if tool["tool_id"] == selected_tool_id),
            None
        )
        
        # If tool_id wasn't found, use first available tool
        if selected_tool is None:
            logger.warning("Invalid tool_id '{}' returned by LLM; using first available tool", selected_tool_id)
            selected_tool = available_tools[0] if available_tools else None
            confidence = 0.3
        
        logger.info("Selected tool={} confidence={:.2f}", selected_tool["tool_id"] if selected_tool else None, confidence)
        logger.info("selection_reasoning={}", reasoning_result.get("reasoning"))
        
    except json.JSONDecodeError:
        logger.warning("Could not parse LLM JSON; defaulting to first tool with low confidence")
        selected_tool = available_tools[0] if available_tools else None
        confidence = 0.3
    
    return {
        **state,
        "selected_tool": selected_tool,
        "tool_confidence": confidence
    }
    
    
async def _load_mcp_tool_map() -> dict[str, Any]:
    """
    Load MCP tools via MultiServerMCPClient and return a name->tool map.

    We keep this separate so the (mostly sync) LangGraph workflow can still run,
    while MCP tool loading happens once up front.
    """
    client = MultiServerMCPClient(
        {
            "tools-server": {
                "url": os.getenv("TOOLS_MCP_SERVER_URL", "http://127.0.0.1:8000/mcp"),
                "transport": "http",
            },
            "arxiv-research": {
                "url": os.getenv("SEARCH_MCP_SERVER_URL", "http://127.0.0.1:8001/mcp"),
                "transport": "http"
		    },
        }
    )
    tools = await client.get_tools()
    return {t.name: t for t in tools}


def tool_execution_node(state: AgentState) -> AgentState:
    """
    Step 4 & 5: Agent calls MCP tool via HTTP endpoint and receives response
    Executes the selected tool with extracted parameters
    Tracks success/failure for validation
    """
    logger.info("STEP 4-5: Executing selected tool")
    
    llm = get_llm(provider=llm_provider, temperature=r_temperature)
    
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
Example: {{"location": "Accra", "days": 3}}
"""

    # prompt = param_extraction_prompt.format_messages(
    # user_query=user_query,
    # tool_id=selected_tool["tool_id"],
    # tool_inputs=json.dumps(selected_tool["inputs"], indent=2),
    # )


    messages = [SystemMessage(content=param_extraction_prompt)]
    param_response = llm.invoke(messages)
    
    try:
        params = json.loads(param_response.content)
        if not isinstance(params, dict):
            params = {}
        logger.debug("extracted_params={}", params)
    except json.JSONDecodeError:
        params = {}
        logger.warning("Could not extract parameters (non-JSON); using empty dict")

    # Fill missing required inputs from tool descriptor with sensible defaults
    # so MCP tools (e.g. weekday_name(date)) don't get empty params and raise.
    from datetime import datetime as _dt
    today_str = _dt.now().strftime("%Y-%m-%d")
    for key in selected_tool.get("inputs", {}):
        if key not in params or params[key] is None or params[key] == "":
            if key in ("date", "start_date"):
                params[key] = today_str
            elif key == "end_date":
                # default end_date = 7 days from today
                from datetime import timedelta
                end = (_dt.now() + timedelta(days=7)).strftime("%Y-%m-%d")
                params[key] = end
            elif key in ("city", "location", "text", "expression", "symbol", "exchange"):
                params[key] = params.get(key) or user_query.split()[-1] if user_query else ""
            else:
                params[key] = params.get(key) or ""

    # Call MCP tool via HTTP endpoint
    logger.info("Calling MCP tool {}", selected_tool["tool_id"])
    logger.debug("tool_args={}", params)
    
    try:
        tool_id = selected_tool["tool_id"]
        mcp_tool = state.get("mcp_tool_map", {}).get(tool_id)

        if not mcp_tool:
            tool_result = (
                f"Error: No MCP tool found matching '{tool_id}'. "
                f"Available MCP tools: {sorted(state.get('mcp_tool_map', {}).keys())}"
            )
            execution_success = False
        else:
            # MCP tools are typically async-only; use ainvoke from sync graph.
            tool_result_raw = asyncio.run(mcp_tool.ainvoke(params))
            tool_result = (
                tool_result_raw
                if isinstance(tool_result_raw, str)
                else json.dumps(tool_result_raw, indent=2, default=str)
            )
            execution_success = True
        
        if execution_success:
            logger.info("Tool executed successfully")
            logger.debug("tool_result={}", tool_result)
            try:
                pretty_json(tool_result, title="Tool result")
            except Exception:
                pass
        else:
            logger.error("Tool execution failed: {}", tool_result)
    
    except Exception as e:
        tool_result = f"Error executing tool: {str(e)}"
        execution_success = False
        logger.exception("Tool execution threw exception")
    
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
    logger.info("STEP 5.5: Validating tool execution")
    
    execution_success = state["tool_execution_success"]
    attempt = state["attempt_count"]
    max_attempts = state["max_attempts"]
    confidence = float(state["tool_confidence"]) if state.get("tool_confidence") is not None else 0.0
    confidence_threshold = float(state["confidence_threshold"]) if state.get("confidence_threshold") is not None else 0.6
    tried_tools = state["tried_tools_count"]
    retrieved_tools = state["retrieved_tools"]
    failed_tools = state["failed_tools"]
    selected_tool = state["selected_tool"]
    
    logger.info(
        "execution_success={} confidence={:.2f} threshold={:.2f} attempt={}/{} tried_tools={}/{}",
        execution_success,
        confidence,
        confidence_threshold,
        attempt,
        max_attempts,
        tried_tools,
        len(retrieved_tools),
    )
    
    # Initialize validation result
    validation_result = "failed"

    # Check low confidence first: request user feedback whenever below threshold
    # (regardless of tool success), so feedback is actually triggered.
    if confidence < confidence_threshold:
        logger.warning(
            "Low confidence selection ({:.2f} < {:.2f}) → requesting user feedback",
            confidence,
            confidence_threshold,
        )
        validation_result = "feedback"
        return {
            **state,
            "requires_user_feedback": True,
            "validation_result": validation_result,
            "failed_tools": failed_tools + [selected_tool["tool_id"]] if selected_tool else failed_tools
        }

    if execution_success:
        logger.info("Validation success → response generation")
        validation_result = "success"

    elif tried_tools < len(retrieved_tools) - 1:
        logger.warning("Retrying with next available tool")
        validation_result = "retry_tool"
        return {
            **state,
            "tried_tools_count": tried_tools + 1,
            "validation_result": validation_result,
            "failed_tools": failed_tools + [selected_tool["tool_id"]] if selected_tool else failed_tools,
            "tool_execution_success": False
        }
    
    elif attempt < max_attempts:
        logger.warning("Retrying with refined query")
        validation_result = "retry_query"
        return {
            **state,
            "attempt_count": attempt + 1,
            "validation_result": validation_result,
            "refined_query": f"Alternative interpretation: {state['user_query']}",
            "tried_tools_count": 0,
            "failed_tools": []
        }
    
    else:
        logger.error("Max attempts reached; unable to complete request")
        validation_result = "failed"
    
    return {
        **state,
        "validation_result": validation_result,
        "requires_user_feedback": False
    }


def validation_router(state: AgentState) -> str:
    """
    Route based on validation results
    
    Returns:
        "success": proceed to response generation
        "feedback": request user feedback
        "retry_tool": try next tool
        "retry_query": refine and retry query
        "failed": max attempts reached
    """
    validation_result = state.get("validation_result", "failed")
    logger.debug("validation_router: routing to '{}' (validation_result={})", validation_result, validation_result)
    
    # Ensure "feedback" is properly routed
    if validation_result == "feedback":
        logger.info("validation_router: routing to user_feedback node")
        return "feedback"
    
    return validation_result


def user_feedback_node(state: AgentState) -> AgentState:
    """
    Step 5.7: Request user feedback when tool selection confidence is low
    Prompts the user via terminal to clarify or confirm tool selection
    """
    logger.info("=" * 80)
    logger.info("STEP 5.7: USER_FEEDBACK_NODE TRIGGERED")
    logger.info("=" * 80)
    logger.info("Requesting user feedback (low confidence)")
    
    user_query = state["user_query"]
    selected_tool = state["selected_tool"]
    confidence = state["tool_confidence"]
    retrieved_tools = state["retrieved_tools"]
    
    try:
        panel(
            "Low confidence tool selection",
            f"confidence={confidence:.2f}\nquery={user_query}\nselected={selected_tool['tool_id'] if selected_tool else 'None'}",
            style="yellow",
        )
    except Exception:
        logger.warning("Low confidence tool selection confidence={:.2f}", confidence)
        logger.info("Original query: {}", user_query)
        logger.info("Selected tool: {}", selected_tool["tool_id"] if selected_tool else None)
    
    if selected_tool:
        logger.debug("tool_description={}", selected_tool.get("description"))
        logger.debug("tool_when_to_use={}", selected_tool.get("when_to_use"))
    
    try:
        tools_table(retrieved_tools, title="Available tools")
    except Exception:
        logger.info("Available tools: {}", [t.get("tool_id") for t in retrieved_tools])
    
    logger.info("Awaiting user feedback via stdin")
    
    # Get user feedback from terminal
    while True:
        user_input = input("\nYour feedback: ").strip().lower()
        
        if not user_input:
            logger.warning("Empty feedback; prompting again")
            continue
        
        if user_input == "confirm":
            feedback = "User confirmed the selected tool"
            logger.info(feedback)
            break
        
        elif user_input.startswith("select "):
            try:
                tool_num = int(user_input.split()[1]) - 1
                if 0 <= tool_num < len(retrieved_tools):
                    selected_tool = retrieved_tools[tool_num]
                    feedback = f"User selected: {selected_tool['tool_id']}"
                    logger.info(feedback)
                    break
                else:
                    logger.warning("Invalid selection index; must be 1..{}", len(retrieved_tools))
            except (ValueError, IndexError):
                logger.warning("Invalid format. Use: select <number>")
        
        elif user_input.startswith("clarify:"):
            clarification = user_input.replace("clarify:", "").strip()
            if clarification:
                feedback = f"User clarification: {clarification}"
                logger.info(feedback)
                break
            else:
                logger.warning("Clarify provided without content")
        
        elif user_input.startswith("new:"):
            new_query = user_input.replace("new:", "").strip()
            if new_query:
                feedback = f"User provided new query: {new_query}"
                logger.info(feedback)
                # Update the user query for the next attempt
                return {
                    **state,
                    "user_query": new_query,
                    "user_feedback": feedback,
                    "requires_user_feedback": False,
                    "attempt_count": state["attempt_count"] + 1,
                    "tried_tools_count": 0,
                    "failed_tools": [],
                    "refined_query": ""
                }
            else:
                logger.warning("New provided without content")
        
        else:
            logger.warning("Invalid input; expecting confirm/select/clarify/new")
            continue
    
    return {
        **state,
        "selected_tool": selected_tool,
        "user_feedback": feedback,
        "requires_user_feedback": False
    }


def response_generation_node(state: AgentState) -> AgentState:
    """
    Generate final response to user based on tool result or feedback status
    """
    logger.info("STEP 6: Generating final response")
    
    llm = get_llm(provider=llm_provider, temperature=g_temperature)

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
    
    # logger.info("Final response generated")
    # try:
    #     panel("Final response", response.content, style="green")
    # except Exception:
    #     pass
    
    return {
        **state,
        "messages": [AIMessage(content=response.content)]
    }


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_agent_graph(
    vector_store_provider: str = "chromadb",
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small"
):
    """
    Create the LangGraph agent graph with all nodes and edges
    
    Args:
        vector_store_provider: Vector store to use
        embedding_provider: Embedding provider
        embedding_model: Embedding model name
        
    Returns:
        Compiled graph ready for execution
    """
    
    workflow = StateGraph(AgentState)
    
    # Add nodes with partial application of parameters
    workflow.add_node(
        "retrieve_tools",
        lambda state: retrieve_tools_node(
            state,
            vector_store_provider=vector_store_provider,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model
        )
    )
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("user_feedback", user_feedback_node)
    workflow.add_node("response_generation", response_generation_node)
    
    # Set entry point
    workflow.set_entry_point("retrieve_tools")
    
    # Define edges
    workflow.add_edge("retrieve_tools", "reasoning")
    workflow.add_edge("reasoning", "tool_execution")
    workflow.add_edge("tool_execution", "validation")
    
    # Validation routing
    workflow.add_conditional_edges(
        "validation",
        validation_router,
        {
            "success": "response_generation",
            "feedback": "user_feedback",
            "retry_tool": "reasoning",
            "retry_query": "retrieve_tools",
            "failed": "response_generation"
        }
    )
    
    # User feedback flows back to tool execution
    workflow.add_edge("user_feedback", "tool_execution")
    
    # Response generation ends
    workflow.add_edge("response_generation", END)
    
    return workflow.compile()


def should_retry(state: AgentState) -> str:
    """Determine if we should retry the agent execution"""
    if state["tool_execution_success"]:
        return "success"
    elif state["attempt_count"] < state["max_attempts"]:
        return "retry"
    else:
        return "failed"


# ============================================================================
# GRAPH VISUALIZATION UTILITY
# ============================================================================
def write_graph_to_file(graph):
    """
    Write graph visualization to PNG file
    
    Args:
        graph: Compiled LangGraph graph
        filename: Output filename (default: agent_graph.png)
    """
    import uuid
    filename = f"../../assets/agent_graph_{uuid.uuid4().hex}.png"
    
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        logger.info("Graph visualization saved to {}", filename)
    except Exception as e:
        logger.warning("Could not save graph visualization: {}", e)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_agent(
    user_query: str,
    vector_store_provider: str = "chromadb",
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small"
):
    """
    Run the agent with a user query
    
    Args:
        user_query: The user's question
        vector_store_provider: Vector store to use ("chromadb", "faiss", "milvus", "qdrant")
        embedding_provider: Embedding provider ("huggingface", "ollama", "openai")
        embedding_model: Specific embedding model name
        
    Returns:
        Final agent state with results
    """
    try:
        panel("User query", user_query, style="cyan")
    except Exception:
        logger.info("User query: {}", user_query)
    
    # Initialize state
    mcp_tool_map = asyncio.run(_load_mcp_tool_map())
    logger.info(f"Loaded {len(mcp_tool_map)} Tools MCP and Search MCP server running on: ✅ {os.getenv("TOOLS_MCP_SERVER_URL")}, {os.getenv("SEARCH_MCP_SERVER_URL")}")

    initial_state: AgentState = {
        "messages": messages,
        "user_query": user_query,
        "retrieved_tools": retrieved_tools,
        "selected_tool": selected_tool,
        "tool_result": tool_result,
        "attempt_count": attempt_count,
        "max_attempts": max_attempts,
        "refined_query": refined_query,
        "tool_execution_success": tool_execution_success,
        "tool_confidence": tool_confidence,
        "confidence_threshold": confidence_threshold,
        "failed_tools": failed_tools,
        "user_feedback": user_feedback,
        "requires_user_feedback": requires_user_feedback,
        "tried_tools_count": tried_tools_count,
        "validation_result": validation_result,  # Initialize from args
        "mcp_tool_map": mcp_tool_map,
    }
    
    # Create graph with providers
    graph = create_agent_graph(
        vector_store_provider=vector_store_provider,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model
    )
    
    if graph is None:
        raise ValueError("Graph creation failed.")
    
    # write_graph_to_file(graph)
    # logger.info("Graph created successfully")
    
    final_state = graph.invoke(initial_state, config={"recursion_limit": 100})
    logger.info("EXECUTION COMPLETE")
    
    return final_state


if __name__ == "__main__":
    
    setup_logging()
    logger.info("AGENT FLOW START")
    logger.info(f"Running agent with query={query} ")
    logger.info(f"LLM provider={llm_provider}")
    logger.info(f"Vector store provider={vector_store_provider}")
    logger.info(f"Embedding provider={embedding_provider}")
    logger.info(f"Embedding model={embedding_model}")
    
    result = run_agent(
        user_query=query, 
        vector_store_provider=vector_store_provider,
        embedding_provider=embedding_provider, 
        embedding_model=embedding_model
    )
    
    logger.info("Final Response:")
    if result["messages"]:
        try:
            panel("Final response", result["messages"][-1].content, style="green")
        except Exception:
            logger.info(result["messages"][-1].content)








