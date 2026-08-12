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
import time
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

import toolrag.langgraph_setup  # noqa: F401  # before langgraph (Reviver deprecation)

from langgraph.graph import END, StateGraph
from langchain_mcp_adapters.client import MultiServerMCPClient
from loguru import logger
from toolrag.utils import create_parser, write_graph_to_file
import warnings
from toolrag.embed_factory import EmbeddingFactory, create_embedding
# from .embed_factory import EmbeddingFactory, create_embedding
from toolrag.llm_factory import LLMFactory, get_llm
# from tool_descriptors import TOOL_DESCRIPTORS
from toolrag.tool_vectordb import ToolVectorDB
from toolrag.vector_store_factory import VectorStoreFactory, create_vector_store

# load tools descriptors data
from tools_data.full_tool_descriptors import FULL_TOOL_DESCRIPTORS
from tools_data.inputs_descriptors import INPUT_DESCRIPTORS
from tools_data.tool_id_descriptors import TOOL_ID_DESCRIPTORS

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

# Load environment variables from .env file
load_dotenv("../.env", override=True) 

try:
    # When running as installed package: python -m tv ...
    from toolrag.logging_factory import setup_logging
    from toolrag.console_factory import panel, tools_table, pretty_json, scores_table
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


# ============================================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================================
args = create_parser()
query = args.query
vector_store_provider = args.vector_store_provider
pg_distance_metric = args.pg_distance_metric
pgvector_connection = getattr(args, "pgvector_connection", None)
pgvector_collection_name = getattr(args, "pgvector_collection_name", "tool_descriptors")
pgvector_use_hnsw = getattr(args, "pgvector_use_hnsw", True)
pgvector_pre_delete_collection = getattr(args, "pgvector_pre_delete_collection", False)
embedding_provider = args.embedding_provider
embedding_model = args.embedding_model
llm_provider = args.llm_provider
# temperature = args.temperature
num_tools = args.num_tools
confidence_threshold = args.confidence_threshold
max_attempts = args.max_attempts
retrieved_tools = args.retrieved_tools
failed_tools = args.failed_tools
selected_tools = args.selected_tools
tool_results = args.tool_results
tool_execution_success = args.tool_execution_success
tool_confidences = args.tool_confidences
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
    selected_tools: list[dict]
    tool_results: list[str] | None
    attempt_count: int
    max_attempts: int
    refined_query: str
    tool_execution_success: bool
    tool_confidences: list[float]
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
    embedding_model: str = "text-embedding-3-small",
    pg_distance_metric: str = "cosine",
    pgvector_connection: str | None = None,
    pgvector_collection_name: str = "tool_descriptors",
    pgvector_use_hnsw: bool = True,
    pgvector_pre_delete_collection: bool = False,
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
        tool_descriptors=FULL_TOOL_DESCRIPTORS,
        vector_store_provider=vector_store_provider,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        pg_distance_metric=pg_distance_metric,
        pgvector_connection=pgvector_connection,
        pgvector_collection_name=pgvector_collection_name,
        pgvector_use_hnsw=pgvector_use_hnsw,
        pgvector_pre_delete_collection=pgvector_pre_delete_collection,
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
            "selected_tools": [],
            "tool_confidences": []
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
You are analyzing a user query to select the most appropriate tool or tools.

User Query: "{user_query}"

Available Tools:
{tools_text}

IMPORTANT: You must select ONE or MORE of these exact tool IDs based on the user's query: {valid_tool_ids}

Analyze the query and determine which tools best match the user's intent.
For each selected tool, provide a confidence score (0.0-1.0) based on how well the tool matches the query.
1.0 = perfect match, 0.5 = moderate match, 0.0 = no match.

If none of the tools are a good match, return an empty list for "selections".

Respond with ONLY a JSON object:
{{
    "reasoning": "your reasoning here",
    "selections": [
        {{
            "tool_id": "MUST BE ONE OF: {', '.join(valid_tool_ids)}",
            "confidence": 0.85
        }},
        {{
            "tool_id": "ANOTHER TOOL ID or empty list",
            "confidence": 0.7
        }}
    ]
}}
"""

    messages = [SystemMessage(content=reasoning_prompt)]
    response = llm.invoke(messages)
    
    # Parse response
    try:
        reasoning_result = json.loads(response.content)
        selections = reasoning_result.get("selections", [])
        
        selected_tools = []
        confidences = []

        if not selections:
            logger.warning("LLM returned no tool selections.")
            return {
                **state,
                "selected_tools": [],
                "tool_confidences": [],
            }

        for selection in selections:
            selected_tool_id = selection.get("tool_id")
            raw_confidence = selection.get("confidence", 0.5)
            confidence = float(raw_confidence) if raw_confidence is not None else 0.5

            # Find the full tool descriptor
            selected_tool = next(
                (tool for tool in available_tools if tool["tool_id"] == selected_tool_id),
                None
            )

            if selected_tool:
                selected_tools.append(selected_tool)
                confidences.append(confidence)
                logger.info("Selected tool={} confidence={:.2f}", selected_tool["tool_id"], confidence)
            else:
                logger.warning("Invalid tool_id '{}' returned by LLM; skipping.", selected_tool_id)

        logger.info("selection_reasoning={}", reasoning_result.get("reasoning"))

    except (json.JSONDecodeError, AttributeError):
        logger.warning("Could not parse LLM JSON; returning no tools.")
        selected_tools = []
        confidences = []

    return {
        **state,
        "selected_tools": selected_tools,
        "tool_confidences": confidences,
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
    Executes the selected tools with extracted parameters
    Tracks success/failure for validation
    """
    logger.info("STEP 4-5: Executing selected tools")
    
    llm = get_llm(provider=llm_provider, temperature=r_temperature)
    user_query = state["user_query"]
    selected_tools = state.get("selected_tools", [])
    
    if not selected_tools:
        return {
            **state,
            "tool_results": ["No tools selected"],
            "tool_execution_success": False,
        }

    tool_results = []
    overall_execution_success = True

    for selected_tool in selected_tools:
        # Extract parameters using LLM
        param_extraction_prompt = f"""
Extract the parameters for this tool call from the user query.

User Query: "{user_query}"

Tool: {selected_tool['tool_id']}
Required Inputs: {json.dumps(selected_tool['inputs'], indent=2)}

Respond with ONLY a JSON object containing the parameter values:
Example: {{"location": "Accra", "days": 3}}
"""
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

        # Fill missing required inputs
        from datetime import datetime as _dt
        today_str = _dt.now().strftime("%Y-%m-%d")
        for key in selected_tool.get("inputs", {}):
            if key not in params or params[key] is None or params[key] == "":
                if key in ("date", "start_date"):
                    params[key] = today_str
                elif key == "end_date":
                    from datetime import timedelta
                    end = (_dt.now() + timedelta(days=7)).strftime("%Y-%m-%d")
                    params[key] = end
                elif key in ("city", "location", "text", "expression", "symbol", "exchange"):
                    params[key] = params.get(key) or user_query.split()[-1] if user_query else ""
                else:
                    params[key] = params.get(key) or ""

        # Call MCP tool
        logger.info("Calling MCP tool {}", selected_tool["tool_id"])
        logger.debug("tool_args={}", params)
        
        try:
            tool_id = selected_tool["tool_id"]
            mcp_tool = state.get("mcp_tool_map", {}).get(tool_id)

            if not mcp_tool:
                tool_result = f"Error: No MCP tool found matching '{tool_id}'."
                execution_success = False
            else:
                tool_result_raw = asyncio.run(mcp_tool.ainvoke(params))
                tool_result = (
                    tool_result_raw
                    if isinstance(tool_result_raw, str)
                    else json.dumps(tool_result_raw, indent=2, default=str)
                )
                execution_success = True
            
            if execution_success:
                logger.info(f"Tool {tool_id} executed successfully.")
                logger.debug("tool_result={}", tool_result)
                tool_results.append(tool_result)
                try:
                    pretty_json(tool_result, title=f"Tool result for {tool_id}")
                except Exception:
                    pass
            else:
                logger.error(f"Tool {tool_id} execution failed: {tool_result}")
                tool_results.append(tool_result)
                overall_execution_success = False
        
        except Exception as e:
            tool_result = f"Error executing tool {selected_tool['tool_id']}: {str(e)}"
            tool_results.append(tool_result)
            overall_execution_success = False
            logger.exception(f"Tool {selected_tool['tool_id']} execution threw exception")
    
    return {
        **state,
        "tool_results": tool_results,
        "tool_execution_success": overall_execution_success,
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
    confidences = state.get("tool_confidences", [])
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    confidence_threshold = float(state["confidence_threshold"]) if state.get("confidence_threshold") is not None else 0.6
    
    logger.info(
        "execution_success={} avg_confidence={:.2f} threshold={:.2f} attempt={}/{}",
        execution_success,
        avg_confidence,
        confidence_threshold,
        attempt,
        max_attempts,
    )
    
    # Initialize validation result
    validation_result = "failed"

    if avg_confidence < confidence_threshold:
        logger.warning(
            "Low average confidence ({:.2f} < {:.2f}) → requesting user feedback",
            avg_confidence,
            confidence_threshold,
        )
        validation_result = "feedback"
        return {
            **state,
            "requires_user_feedback": True,
            "validation_result": validation_result,
        }

    if execution_success:
        logger.info("Validation success → response generation")
        validation_result = "success"
    
    elif attempt < max_attempts:
        logger.warning("Execution failed. Retrying with refined query")
        validation_result = "retry_query"
        return {
            **state,
            "attempt_count": attempt + 1,
            "validation_result": validation_result,
            "refined_query": f"Alternative interpretation: {state['user_query']}",
            "tried_tools_count": 0,
            "failed_tools": [],
            "selected_tools": [],
            "tool_confidences": [],
        }
    
    else:
        logger.error("Max attempts reached; unable to complete request")
        validation_result = "failed"
    
    return {
        **state,
        "validation_result": validation_result,
        "requires_user_feedback": False,
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
    selected_tools = state.get("selected_tools", [])
    confidences = state.get("tool_confidences", [])
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    selected_tool_ids = [tool['tool_id'] for tool in selected_tools]

    try:
        panel(
            "Low confidence tool selection",
            f"average_confidence={avg_confidence:.2f}\nquery={user_query}\nselected_tools={', '.join(selected_tool_ids)}",
            style="yellow",
        )
    except Exception:
        logger.warning("Low confidence tool selection avg_confidence={:.2f}", avg_confidence)
        logger.info("Original query: {}", user_query)
        logger.info("Selected tools: {}", selected_tool_ids)
    
    logger.info("Awaiting user feedback via stdin: expecting confirm/new:")
    
    # Get user feedback from terminal
    while True:
        user_input = input("\nYour feedback (confirm/new:<new query>): ").strip().lower()
        
        if not user_input:
            logger.warning("Empty feedback; prompting again")
            continue
        
        if user_input == "confirm":
            feedback = "User confirmed the selected tools"
            logger.info(feedback)
            # We are not changing the selected_tools, so we just proceed.
            # The state already has the selected_tools.
            return {
                **state,
                "user_feedback": feedback,
                "requires_user_feedback": False,
            }
        
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
                    "refined_query": "",
                    "selected_tools": [],
                    "tool_confidences": [],
                }
            else:
                logger.warning("New provided without content")
        
        else:
            logger.warning("Invalid input; expecting confirm or new:<new query>")
            continue
    
    return state


def response_generation_node(state: AgentState) -> AgentState:
    """
    Generate final response to user based on tool result or feedback status
    """
    logger.info("STEP 6: Generating final response")
    
    llm = get_llm(provider=llm_provider, temperature=g_temperature)

    user_query = state["user_query"]
    tool_results = state.get("tool_results", [])
    execution_success = state["tool_execution_success"]
    user_feedback = state.get("user_feedback")
    
    # Join multiple tool results into a single string
    final_tool_result = "\n\n".join(tool_results)

    if execution_success:
        response_prompt = f"""
Generate a natural, helpful response to the user based on the tool execution results.

User Query: "{user_query}"

Tool Results:
{final_tool_result}

Provide a clear, conversational response that directly answers the user's question based on the combined results.
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
Last Error: {final_tool_result}

Provide a helpful response that apologizes for the issue and suggests alternatives if possible.
"""
    
    messages = [SystemMessage(content=response_prompt)]
    response = llm.invoke(messages)
    
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
    embedding_model: str = "text-embedding-3-small",
    pg_distance_metric: str = "cosine",
    pgvector_connection: str | None = None,
    pgvector_collection_name: str = "tool_descriptors",
    pgvector_use_hnsw: bool = True,
    pgvector_pre_delete_collection: bool = False,
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
            embedding_model=embedding_model,
            pg_distance_metric=pg_distance_metric,
            pgvector_connection=pgvector_connection,
            pgvector_collection_name=pgvector_collection_name,
            pgvector_use_hnsw=pgvector_use_hnsw,
            pgvector_pre_delete_collection=pgvector_pre_delete_collection,
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
# MAIN EXECUTION
# ============================================================================

def run_agent(
    user_query: str,
    vector_store_provider: str = "chromadb",
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
    pg_distance_metric: str = "cosine",
    pgvector_connection: str | None = None,
    pgvector_collection_name: str = "tool_descriptors",
    pgvector_use_hnsw: bool = True,
    pgvector_pre_delete_collection: bool = False,
):
    """
    Run the agent with a user query
    
    Args:
        user_query: The user's question
        vector_store_provider: Vector store to use ("chromadb", "faiss", "milvus", "qdrant", "pgvector")
        embedding_provider: Embedding provider ("huggingface", "ollama", "openai")
        embedding_model: Specific embedding model name
        pg_distance_metric: pgvector distance (l2, cosine, inner_product, l1)
        
    Returns:
        Final agent state with results
    """
    flow_start = time.perf_counter()

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
        "selected_tools": selected_tools,
        "tool_results": tool_results,
        "attempt_count": attempt_count,
        "max_attempts": max_attempts,
        "refined_query": refined_query,
        "tool_execution_success": tool_execution_success,
        "tool_confidences": tool_confidences,
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
        embedding_model=embedding_model,
        pg_distance_metric=pg_distance_metric,
        pgvector_connection=pgvector_connection,
        pgvector_collection_name=pgvector_collection_name,
        pgvector_use_hnsw=pgvector_use_hnsw,
        pgvector_pre_delete_collection=pgvector_pre_delete_collection,
    )
    
    if graph is None:
        raise ValueError("Graph creation failed.")
    
    # write_graph_to_file(graph)
    # logger.info("Graph created successfully")
    
    final_state = graph.invoke(initial_state, config={"recursion_limit": 100})
    flow_elapsed = time.perf_counter() - flow_start
    logger.info("EXECUTION COMPLETE")
    logger.info("⏱️ Total flow time: {:.3f} seconds", flow_elapsed)
    # print(f"\n⏱️  Total flow time: {flow_elapsed:.3f} seconds")
    
    return final_state


def main():
    
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
        embedding_model=embedding_model,
        pg_distance_metric=pg_distance_metric,
        pgvector_connection=pgvector_connection,
        pgvector_collection_name=pgvector_collection_name,
        pgvector_use_hnsw=pgvector_use_hnsw,
        pgvector_pre_delete_collection=pgvector_pre_delete_collection,
    )
    
    logger.info("Final Response ✅")
    if result["messages"]:
        try:
            panel("Final response", result["messages"][-1].content, style="green")
        except Exception:
            logger.info(result["messages"][-1].content)


if __name__ == "__main__":
    main()








