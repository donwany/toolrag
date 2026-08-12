"""PGVector helpers: distance metrics, HNSW indexes, and L1 vector store support."""

from __future__ import annotations

import os
import re
from typing import Literal, Optional

from loguru import logger
import sqlalchemy
from langchain_core.embeddings import Embeddings
from langchain_postgres.vectorstores import PGVector
from langchain_postgres.vectorstores import DistanceStrategy as PGDistanceStrategy

PGDistanceMetric = Literal["l2", "cosine", "inner_product", "l1"]

# pgvector operators: <-> L2, <#> inner product, <=> cosine, <+> L1
PGVECTOR_OPERATORS: dict[PGDistanceMetric, str] = {
    "l2": "<->",
    "inner_product": "<#>",
    "cosine": "<=>",
    "l1": "<+>",
}

HNSW_OPS: dict[PGDistanceMetric, str] = {
    "l2": "vector_l2_ops",
    "inner_product": "vector_ip_ops",
    "cosine": "vector_cosine_ops",
    "l1": "vector_l1_ops",
}

LANGCHAIN_DISTANCE: dict[PGDistanceMetric, PGDistanceStrategy] = {
    "l2": PGDistanceStrategy.EUCLIDEAN,
    "cosine": PGDistanceStrategy.COSINE,
    "inner_product": PGDistanceStrategy.MAX_INNER_PRODUCT,
}

DEFAULT_PGVECTOR_CONNECTION = (
    "postgresql+psycopg://postgres:postgres@localhost:54320/tool_rag_db"
)

# Matches Ollama nomic-embed-text and many production embedding models.
DEFAULT_EMBEDDING_DIMENSION = 768

DEFAULT_PGVECTOR_TEST_CONNECTION = (
    "postgresql+psycopg://postgres:postgres@localhost:54320/tool_rag_db_test"
)

EMBEDDING_TABLE = "langchain_pg_embedding"
EMBEDDING_COLUMN = "embedding"


class PGVectorL1(PGVector):
    """PGVector with L1 (taxicab) distance — not supported by langchain-postgres natively."""

    @property
    def distance_strategy(self):
        return self.EmbeddingStore.embedding.l1_distance


def get_pgvector_connection() -> str:
    return os.getenv("PGVECTOR_CONNECTION", DEFAULT_PGVECTOR_CONNECTION)


def normalize_distance_metric(
    distance_metric: str,
) -> PGDistanceMetric:
    normalized = distance_metric.strip().lower().replace("-", "_")
    aliases = {
        "euclidean": "l2",
        "l2": "l2",
        "cosine": "cosine",
        "inner": "inner_product",
        "inner_product": "inner_product",
        "ip": "inner_product",
        "max_inner_product": "inner_product",
        "l1": "l1",
        "manhattan": "l1",
        "taxicab": "l1",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unknown pgvector distance metric: {distance_metric}. "
            f"Choose from: {list(PGVECTOR_OPERATORS)}"
        )
    return aliases[normalized]  # type: ignore[return-value]


def collection_name_for_metric(
    base_name: str,
    distance_metric: PGDistanceMetric,
) -> str:
    """Isolate ablation runs per distance metric."""
    return f"{base_name}_{distance_metric}"


def create_pgvector_store(
    *,
    texts: list[str],
    embeddings: Embeddings,
    metadatas: Optional[list[dict]],
    connection: str,
    collection_name: str,
    distance_metric: PGDistanceMetric,
    use_hnsw_index: bool = True,
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 64,
    pre_delete_collection: bool = False,
) -> PGVector:
    """Create a PGVector store, load documents, and optionally build an HNSW index."""
    store_cls = PGVectorL1 if distance_metric == "l1" else PGVector
    distance_strategy = LANGCHAIN_DISTANCE.get(distance_metric)

    init_kwargs: dict = {
        "embeddings": embeddings,
        "collection_name": collection_name,
        "connection": connection,
        "use_jsonb": True,
        "pre_delete_collection": pre_delete_collection,
    }
    if distance_strategy is not None:
        init_kwargs["distance_strategy"] = distance_strategy

    dimension: Optional[int] = None
    if texts:
        dimension = embedding_dimension(embeddings, texts)
        # Migrate existing DB only; fresh DBs get tables from PGVector on init.
        sync_embedding_column_dimension(connection, dimension)

    vectorstore = store_cls(**init_kwargs)

    if texts:
        ids = [f"{collection_name}_{i}" for i in range(len(texts))]
        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas or [],
            ids=ids,
        )
        # LangChain may leave `vector` without a fixed size until we set it for HNSW.
        if get_embedding_column_dimension(connection) != dimension:
            ensure_embedding_dimensions(connection, dimension)

    if use_hnsw_index and texts and dimension is not None:
        create_hnsw_index(
            connection=connection,
            distance_metric=distance_metric,
            index_name=f"ix_{_safe_identifier(collection_name)}_hnsw",
            m=hnsw_m,
            ef_construction=hnsw_ef_construction,
        )

    return vectorstore


def embedding_dimension(embeddings: Embeddings, texts: list[str]) -> int:
    """Infer vector size from the configured embedding model."""
    sample = texts[0] if texts else "dimension_probe"
    return len(embeddings.embed_query(sample))


def get_embedding_column_dimension(connection: str) -> Optional[int]:
    """Return fixed pgvector dimension for the embedding column, or None if unset."""
    engine = _engine_from_connection(connection)
    with engine.connect() as conn:
        row = conn.execute(
            sqlalchemy.text(
                """
                SELECT a.atttypmod
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE c.relname = :table
                  AND a.attname = :column
                  AND NOT a.attisdropped
                  AND n.nspname = 'public'
                """
            ),
            {"table": EMBEDDING_TABLE, "column": EMBEDDING_COLUMN},
        ).fetchone()
    engine.dispose()
    if not row or row[0] is None or int(row[0]) < 0:
        return None
    return int(row[0])


def ensure_postgres_database(connection: str) -> None:
    """Create the target database if it does not exist (for isolated test DBs)."""
    from sqlalchemy.engine import make_url

    url = make_url(connection)
    db_name = url.database
    if not db_name:
        return

    admin_url = url.set(database="postgres")
    engine = sqlalchemy.create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        exists = conn.execute(
            sqlalchemy.text("SELECT 1 FROM pg_database WHERE datname = :db"),
            {"db": db_name},
        ).scalar()
        if not exists:
            conn.execute(sqlalchemy.text(f'CREATE DATABASE "{db_name}"'))
    engine.dispose()


def _drop_embedding_hnsw_indexes(connection: str) -> None:
    """Remove HNSW indexes so the embedding column type can be changed."""
    engine = _engine_from_connection(connection)
    with engine.connect() as conn:
        rows = conn.execute(
            sqlalchemy.text(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE tablename = :table
                  AND indexdef ILIKE '%using hnsw%'
                """
            ),
            {"table": EMBEDDING_TABLE},
        ).fetchall()
        for (index_name,) in rows:
            safe_name = _safe_identifier(index_name)
            conn.execute(sqlalchemy.text(f'DROP INDEX IF EXISTS "{safe_name}"'))
        conn.commit()
    engine.dispose()


def _embedding_table_exists(connection: str) -> bool:
    engine = _engine_from_connection(connection)
    with engine.connect() as conn:
        exists = conn.execute(
            sqlalchemy.text(
                """
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = :table
                """
            ),
            {"table": EMBEDDING_TABLE},
        ).scalar()
    engine.dispose()
    return bool(exists)


def _embedding_table_has_rows(connection: str) -> bool:
    engine = _engine_from_connection(connection)
    with engine.connect() as conn:
        count = conn.execute(
            sqlalchemy.text(f"SELECT COUNT(*) FROM {EMBEDDING_TABLE}")
        ).scalar()
    engine.dispose()
    return bool(count)


def sync_embedding_column_dimension(connection: str, dimension: int) -> None:
    """
    Align the embedding column with the active embedding model.

    If a previous run used a different dimension (e.g. pytest mocks), drop HNSW
    indexes and clear incompatible rows before altering the column type.
    """
    if not _embedding_table_exists(connection):
        return

    current = get_embedding_column_dimension(connection)
    if current == dimension:
        return

    if current is not None and current != dimension:
        logger.info(
            "Migrating PGVector embedding column from vector({}) to vector({})",
            current,
            dimension,
        )
        _drop_embedding_hnsw_indexes(connection)
        if _embedding_table_has_rows(connection):
            engine = _engine_from_connection(connection)
            with engine.connect() as conn:
                conn.execute(
                    sqlalchemy.text(
                        "TRUNCATE langchain_pg_embedding, langchain_pg_collection CASCADE"
                    )
                )
                conn.commit()
            engine.dispose()

    ensure_embedding_dimensions(connection, dimension)


def ensure_embedding_dimensions(connection: str, dimension: int) -> None:
    """Set a fixed dimension on the embedding column so HNSW indexes can be built."""
    engine = _engine_from_connection(connection)
    ddl = sqlalchemy.text(
        f"""
        ALTER TABLE {EMBEDDING_TABLE}
        ALTER COLUMN {EMBEDDING_COLUMN} TYPE vector({int(dimension)})
        """
    )
    with engine.connect() as conn:
        conn.execute(ddl)
        conn.commit()
    engine.dispose()


def create_hnsw_index(
    *,
    connection: str,
    distance_metric: PGDistanceMetric,
    index_name: str = "ix_langchain_pg_embedding_hnsw",
    m: int = 16,
    ef_construction: int = 64,
) -> None:
    """
    Create an HNSW index on the langchain embedding table.

    Example SQL (cosine):
        CREATE INDEX ON langchain_pg_embedding
        USING hnsw (embedding vector_cosine_ops);
    """
    ops = HNSW_OPS[distance_metric]
    safe_index = _safe_identifier(index_name)
    engine = _engine_from_connection(connection)
    # HNSW index options must be literals (PostgreSQL does not accept bind params here).
    ddl = sqlalchemy.text(
        f"""
        CREATE INDEX IF NOT EXISTS {safe_index}
        ON {EMBEDDING_TABLE}
        USING hnsw ({EMBEDDING_COLUMN} {ops})
        WITH (m = {int(m)}, ef_construction = {int(ef_construction)})
        """
    )
    with engine.connect() as conn:
        conn.execute(ddl)
        conn.commit()
    engine.dispose()


def _engine_from_connection(connection: str) -> sqlalchemy.engine.Engine:
    if isinstance(connection, sqlalchemy.engine.Engine):
        return connection
    return sqlalchemy.create_engine(connection)


def _safe_identifier(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)[:48]


def ensure_vector_extension(connection: Optional[str] = None) -> None:
    """Create the pgvector extension if it is not already installed."""
    conn_str = connection or get_pgvector_connection()
    engine = _engine_from_connection(conn_str)
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    engine.dispose()


def check_pgvector_available(connection: Optional[str] = None) -> bool:
    """Return True if PostgreSQL is reachable and pgvector can be used."""
    conn_str = connection or get_pgvector_connection()
    try:
        engine = _engine_from_connection(conn_str)
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        engine.dispose()
        ensure_vector_extension(conn_str)
        return True
    except Exception:
        return False
