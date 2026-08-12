from toolrag import __version__
import argparse
import sys
from argparse import Namespace

from .logging_factory import logger
from .pgvector_store import PGVECTOR_OPERATORS, get_pgvector_connection
from .vector_store_factory import VectorStoreFactory
from pathlib import Path

VECTOR_STORE_CHOICES = [
    "chroma",
    "chromadb",
    "faiss",
    "milvus",
    "qdrant",
    "pgvector",
    "postgres",
    "postgresql",
]

PG_DISTANCE_METRIC_CHOICES = list(PGVECTOR_OPERATORS.keys())

EPIGLOG = """
Examples:
=====================================================
# Qdrant (default local / in-memory)
python -m toolrag.cli \\
    --query "What is the weather in Accra?" \\
    --vector_store_provider qdrant \\
    --embedding_provider ollama \\
    --embedding_model nomic-embed-text

# PGVector + L2 (requires: docker compose -f docker-compose.pgvector.yml up -d)
python -m toolrag.cli \\
    --query "Will it rain in Accra this weekend?" \\
    --vector_store_provider pgvector \\
    --pg_distance_metric l2 \\
    --embedding_provider ollama \\
    --embedding_model nomic-embed-text \\
    --num_tools 3

# PGVector ablation — cosine / inner_product / l1
python -m toolrag.cli ... --vector_store_provider pgvector --pg_distance_metric cosine
python -m toolrag.cli ... --vector_store_provider pgvector --pg_distance_metric inner_product
python -m toolrag.cli ... --vector_store_provider pgvector --pg_distance_metric l1

# PGVector with explicit connection (or set PGVECTOR_CONNECTION in .env)
python -m toolrag.cli ... \\
    --vector_store_provider pgvector \\
    --pgvector_connection "postgresql+psycopg://postgres:postgres@localhost:54320/tool_rag_db"
"""


def normalize_vector_store_provider(provider: str) -> str:
    """Map CLI aliases to providers used by ToolVectorDB / VectorStoreFactory."""
    key = provider.strip().lower()
    aliases = {
        "chromadb": "chroma",
        "postgres": "pgvector",
        "postgresql": "pgvector",
    }
    return aliases.get(key, key)


def is_pgvector_provider(provider: str) -> bool:
    return normalize_vector_store_provider(provider) == "pgvector"


def validate_cli_args(args: Namespace) -> None:
    """Validate vector-store and pgvector-specific CLI combinations."""
    provider = args.vector_store_provider
    available = set(VectorStoreFactory.list_providers())

    if provider not in available:
        raise SystemExit(
            f"Unknown vector store provider: {provider!r}. "
            f"Choose from: {', '.join(sorted(available))}"
        )

    pg_metrics_set = getattr(args, "pg_distance_metric_explicit", False)
    if pg_metrics_set and not is_pgvector_provider(provider):
        logger.warning(
            "--pg_distance_metric is ignored unless --vector_store_provider is "
            "pgvector (also: postgres, postgresql)"
        )

    if is_pgvector_provider(provider):
        if args.pgvector_connection is None:
            args.pgvector_connection = get_pgvector_connection()
        if not args.pgvector_connection.startswith("postgresql"):
            raise SystemExit(
                "--pgvector_connection must be a postgresql+psycopg:// URL "
                "(see pgvector_exec.md)"
            )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser (call parse_args separately if needed)."""
    parser = argparse.ArgumentParser(
        prog="toolrag",
        description="Semantic Vector-Based Tool Retrieval for Agentic Systems",
        epilog=EPIGLOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query", "-q", type=str, required=True, help="The user's query")
    parser.add_argument(
        "--llm_provider",
        type=str,
        choices=["openai", "anthropic", "gemini", "ollama"],
        required=False,
        default="ollama",
        help="LLM provider",
    )
    parser.add_argument(
        "--r_temperature",
        type=float,
        required=False,
        default=0.0,
        help="Temperature for the reasoning LLM",
    )
    parser.add_argument(
        "--g_temperature",
        type=float,
        required=False,
        default=0.7,
        help="Temperature for the generation LLM",
    )
    parser.add_argument(
        "--num_tools",
        type=int,
        required=False,
        default=3,
        help="Top k tools to retrieve",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        required=False,
        default=0.6,
        help="Confidence threshold for tool selection",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        required=False,
        default=3,
        help="Max attempts for tool selection",
    )
    parser.add_argument(
        "--retrieved_tools",
        type=list,
        required=False,
        default=[],
        help="Retrieved tools",
    )
    parser.add_argument(
        "--selected_tools", type=list, required=False, default=[], help="Selected tools"
    )
    parser.add_argument(
        "--failed_tools", type=list, required=False, default=[], help="Failed tools"
    )
    parser.add_argument(
        "--tool_results", type=list, required=False, default=[], help="Tool results"
    )
    parser.add_argument(
        "--tool_execution_success",
        type=bool,
        required=False,
        default=False,
        help="Tool execution success",
    )
    parser.add_argument(
        "--tool_confidences",
        type=list,
        required=False,
        default=[],
        help="Tool confidences",
    )
    parser.add_argument(
        "--refined_query", type=str, required=False, default="", help="Refined query"
    )
    parser.add_argument(
        "--requires_user_feedback",
        type=bool,
        required=False,
        default=False,
        help="Requires user feedback",
    )
    parser.add_argument(
        "--user_feedback", type=str, required=False, default=None, help="User feedback"
    )
    parser.add_argument(
        "--tried_tools_count",
        type=int,
        required=False,
        default=0,
        help="Tried tools count",
    )
    parser.add_argument(
        "--messages", type=list, required=False, default=[], help="Messages"
    )
    parser.add_argument(
        "--attempt_count", type=int, required=False, default=1, help="Attempt count"
    )
    parser.add_argument(
        "--mcp_tool_map", type=dict, required=False, default={}, help="MCP tool map"
    )
    parser.add_argument(
        "--validation_result",
        type=str,
        required=False,
        default="failed",
        help="Validation result",
    )
    parser.add_argument(
        "--vector_store_provider",
        type=str,
        choices=VECTOR_STORE_CHOICES,
        required=False,
        default="chroma",
        help=(
            "Vector store backend: chroma, faiss, milvus, qdrant, or pgvector "
            "(aliases: chromadb, postgres, postgresql)"
        ),
    )

    pg_group = parser.add_argument_group(
        "pgvector",
        "PostgreSQL + pgvector options (only when --vector_store_provider is "
        "pgvector, postgres, or postgresql). See pgvector_exec.md.",
    )
    pg_group.add_argument(
        "--pg_distance_metric",
        type=str,
        choices=PG_DISTANCE_METRIC_CHOICES,
        required=False,
        default="cosine",
        help=(
            "Distance metric: l2 (<->), cosine (<=>), inner_product (<#>), l1 (<+>)"
        ),
    )
    pg_group.add_argument(
        "--pgvector_connection",
        type=str,
        default=None,
        metavar="URL",
        help=(
            "PostgreSQL connection string (psycopg3). "
            f"Default: env PGVECTOR_CONNECTION or {get_pgvector_connection()}"
        ),
    )
    pg_group.add_argument(
        "--pgvector_collection_name",
        type=str,
        default="tool_descriptors",
        help="Base collection name; stored as {name}_{metric} per distance metric",
    )
    pg_group.add_argument(
        "--pgvector-use-hnsw",
        dest="pgvector_use_hnsw",
        action="store_true",
        default=True,
        help="Create HNSW index after ingest (default: enabled)",
    )
    pg_group.add_argument(
        "--no-pgvector-hnsw",
        dest="pgvector_use_hnsw",
        action="store_false",
        help="Skip HNSW index creation",
    )
    pg_group.add_argument(
        "--pgvector-pre-delete-collection",
        dest="pgvector_pre_delete_collection",
        action="store_true",
        default=False,
        help="Drop existing collection before re-indexing tools",
    )
    parser.add_argument(
        "--embedding_provider",
        type=str,
        choices=["ollama", "openai", "huggingface"],
        required=False,
        default="ollama",
        help="Embedding provider",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        choices=[
            "nomic-embed-text",
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
            "mahonzhan/all-MiniLM-L6-v2:latest",
            "qwen3-embedding:8b",
            "bge-m3:latest",
        ],
        required=False,
        default="nomic-embed-text",
        help="Embedding model",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the version number and exit",
    )
    return parser


def create_parser() -> Namespace:
    """Parse CLI arguments and apply vector-store / pgvector normalization."""
    parser = build_parser()
    args = parser.parse_args()

    # Detect whether user explicitly passed --pg_distance_metric on the command line
    args.pg_distance_metric_explicit = any(
        arg == "--pg_distance_metric" or arg.startswith("--pg_distance_metric=")
        for arg in sys.argv[1:]
    )

    args.vector_store_provider = normalize_vector_store_provider(
        args.vector_store_provider
    )
    validate_cli_args(args)
    return args


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
    # Get project root (adjust levels if needed)
    # BASE_DIR = Path(__file__).resolve().parent

    # Navigate to assets folder (relative to script)
    # assets_dir = BASE_DIR / "assets"
    # assets_dir = Path(__file__).resolve().parents[2] / "assets"
    # Ensure directory exists
    # assets_dir.mkdir(parents=True, exist_ok=True)
    # filename = assets_dir / f"agent_graph_{uuid.uuid4().hex}.png"
    
    filename = f"../../assets/agent_graph_{uuid.uuid4().hex}.png"
    
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        logger.info("Graph visualization saved to {}", filename)
    except Exception as e:
        logger.warning("Could not save graph visualization: {}", e)