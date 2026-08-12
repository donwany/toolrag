"""
Integration tests: PGVector distance metrics for ablation studies.

Requires PostgreSQL + pgvector:
  docker compose -f docker-compose.pgvector.yml up -d

Run all metrics:
  pytest tests/test_pgvector_distance_ablation.py -v

Run ablation benchmark (writes JSON):
  python -m evaluation.pgvector_ablation
"""

import time

import pytest

from toolrag.pgvector_store import PGVECTOR_OPERATORS, normalize_distance_metric
from tests.embeddings_fixtures import DeterministicEmbeddings
from toolrag.vector_store_factory import VectorStoreFactory

DISTANCE_METRICS = list(PGVECTOR_OPERATORS.keys())


def _descriptor_texts(descriptors: list[dict]) -> tuple[list[str], list[dict]]:
    texts, metadatas = [], []
    for tool_desc in descriptors:
        text = f"""
            Tool: {tool_desc['tool_id']}
            Inputs: {tool_desc['inputs']}
            Description: {tool_desc['description']}
            When to use: {tool_desc['when_to_use']}
            Examples: {', '.join(tool_desc['examples'])}
            """
        texts.append(text)
        metadatas.append({"tool_id": tool_desc["tool_id"]})
    return texts, metadatas


@pytest.fixture
def test_embeddings():
    return DeterministicEmbeddings()


@pytest.mark.integration
@pytest.mark.parametrize("distance_metric", DISTANCE_METRICS)
def test_pgvector_factory_create_and_search(
    pgvector_available: str,
    sample_tool_descriptors: list[dict],
    test_embeddings,
    distance_metric: str,
) -> None:
    texts, metadatas = _descriptor_texts(sample_tool_descriptors)

    t0 = time.perf_counter()
    store = VectorStoreFactory.create(
        texts=texts,
        embeddings=test_embeddings,
        metadatas=metadatas,
        provider="pgvector",
        connection=pgvector_available,
        collection_name="ablation_test",
        distance_metric=distance_metric,
        pre_delete_collection=True,
    )
    index_ms = (time.perf_counter() - t0) * 1000

    query = "weather forecast for next week"
    t1 = time.perf_counter()
    results = store.similarity_search_with_score(query, k=3)
    search_ms = (time.perf_counter() - t1) * 1000

    assert len(results) >= 1
    assert all(score is not None for _, score in results)
    tool_ids = {doc.metadata.get("tool_id") for doc, _ in results}
    assert len(tool_ids) >= 1

    print(
        f"\n[{distance_metric}] op={PGVECTOR_OPERATORS[distance_metric]} "
        f"index+load={index_ms:.1f}ms search={search_ms:.1f}ms"
    )


@pytest.mark.integration
@pytest.mark.parametrize("distance_metric", DISTANCE_METRICS)
def test_pgvector_scored_search_returns_metadata(
    pgvector_available: str,
    sample_tool_descriptors: list[dict],
    test_embeddings,
    distance_metric: str,
) -> None:
    texts, metadatas = _descriptor_texts(sample_tool_descriptors)
    store = VectorStoreFactory.create(
        texts=texts,
        embeddings=test_embeddings,
        metadatas=metadatas,
        provider="pgvector",
        connection=pgvector_available,
        collection_name="scored_search_ablation",
        distance_metric=distance_metric,
        pre_delete_collection=True,
    )

    results = store.similarity_search_with_score("calendar events this week", k=2)
    assert len(results) >= 1
    assert normalize_distance_metric(distance_metric) == distance_metric
    doc, score = results[0]
    assert doc.metadata.get("tool_id")
    assert score is not None
