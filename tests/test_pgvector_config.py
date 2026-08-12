"""Unit tests for PGVector distance configuration (no database required)."""

import pytest

from toolrag.pgvector_store import (
    HNSW_OPS,
    PGVECTOR_OPERATORS,
    collection_name_for_metric,
    normalize_distance_metric,
)
from toolrag.vector_store_factory import VectorStoreFactory


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("l2", "l2"),
        ("L2", "l2"),
        ("euclidean", "l2"),
        ("cosine", "cosine"),
        ("inner_product", "inner_product"),
        ("inner", "inner_product"),
        ("ip", "inner_product"),
        ("l1", "l1"),
        ("manhattan", "l1"),
    ],
)
def test_normalize_distance_metric(raw: str, expected: str) -> None:
    assert normalize_distance_metric(raw) == expected


def test_unknown_distance_metric_raises() -> None:
    with pytest.raises(ValueError, match="Unknown pgvector distance metric"):
        normalize_distance_metric("hamming")


@pytest.mark.parametrize("metric", list(PGVECTOR_OPERATORS))
def test_operator_and_hnsw_ops_defined(metric: str) -> None:
    assert metric in HNSW_OPS
    assert PGVECTOR_OPERATORS[metric]


def test_collection_name_for_metric_suffix() -> None:
    name = collection_name_for_metric("tool_descriptors", "cosine")
    assert name == "tool_descriptors_cosine"


def test_pgvector_listed_in_factory() -> None:
    assert "pgvector" in VectorStoreFactory.list_providers()
