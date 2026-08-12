"""Shared fixtures for PGVector integration tests."""

import os

import pytest

from toolrag.pgvector_store import (
    DEFAULT_PGVECTOR_CONNECTION,
    DEFAULT_PGVECTOR_TEST_CONNECTION,
    check_pgvector_available,
    ensure_postgres_database,
)
from toolrag.tool_descriptors import TOOL_DESCRIPTORS


@pytest.fixture(scope="session", autouse=True)
def isolate_pgvector_test_database():
    """
    Point PGVector at a separate database so pytest never pins tool_rag_db to
    vector(32) and breaks production runs with 768-dim models (e.g. nomic-embed-text).
    """
    test_conn = os.getenv("PGVECTOR_TEST_CONNECTION", DEFAULT_PGVECTOR_TEST_CONNECTION)
    ensure_postgres_database(test_conn)

    previous = os.environ.get("PGVECTOR_CONNECTION")
    os.environ["PGVECTOR_CONNECTION"] = test_conn
    yield test_conn

    if previous is None:
        os.environ.pop("PGVECTOR_CONNECTION", None)
    else:
        os.environ["PGVECTOR_CONNECTION"] = previous


@pytest.fixture(scope="session")
def pg_connection(isolate_pgvector_test_database: str) -> str:
    return isolate_pgvector_test_database


@pytest.fixture(scope="session")
def pgvector_available(pg_connection: str) -> str:
    if not check_pgvector_available(pg_connection):
        pytest.skip(
            "PostgreSQL with pgvector not reachable. "
            "Start with: docker compose -f docker-compose.pgvector.yml up -d"
        )
    return pg_connection


@pytest.fixture
def sample_tool_descriptors() -> list[dict]:
    return TOOL_DESCRIPTORS[:6]
