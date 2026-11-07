"""GraphData를 Neo4j에 넣는다."""

from __future__ import annotations

import os

from neo4j import Driver, GraphDatabase

from ..config import NEO4J_URI, NEO4J_USER
from ..schema import ALLOWED_ENDPOINTS, ConceptRelationKind
from .model import ENTITY_LABELS, NODE_LABELS, GraphData

_BATCH = 500

# 관계 타입 → (시작 노드 레이블, 끝 노드 레이블)
_CONCEPT_ENDPOINTS = {
    kind.value.upper(): (
        "Article" if source == "article" else ENTITY_LABELS[source],
        "Article" if target == "article" else ENTITY_LABELS[target],
    )
    for kind, (source, target) in ALLOWED_ENDPOINTS.items()
}
assert set(_CONCEPT_ENDPOINTS) == {k.value.upper() for k in ConceptRelationKind}


def connect() -> Driver:
    return GraphDatabase.driver(
        os.environ.get("NEO4J_URI", NEO4J_URI),
        auth=(os.environ.get("NEO4J_USER", NEO4J_USER), os.environ["NEO4J_PASSWORD"]),
    )


def _run(driver: Driver, query: str, rows: list[dict]) -> None:
    for start in range(0, len(rows), _BATCH):
        driver.execute_query(query, parameters_={"rows": rows[start : start + _BATCH]})


def load_graph(driver: Driver, data: GraphData, *, reset: bool = False) -> None:
    """데이터를 넣는다. 같은 데이터를 다시 넣어도 중복되지 않는다. reset이면 기존 노드를 모두 지운다."""
    if reset:
        driver.execute_query("MATCH (n) DETACH DELETE n")
    for label in NODE_LABELS:
        driver.execute_query(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")

    _run(driver, "UNWIND $rows AS row MERGE (n:Article {id: row.id}) SET n += row", data.articles)
    for label, rows in data.entities.items():
        _run(driver, f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n += row", rows)
    for label, rows in data.mentions.items():
        _run(
            driver,
            "UNWIND $rows AS row "
            f"MATCH (a:Article {{id: row.article_id}}) MATCH (e:{label} {{id: row.entity_id}}) "
            "MERGE (a)-[:MENTIONS]->(e)",
            rows,
        )
    for kind, rows in data.article_edges.items():
        _run(
            driver,
            "UNWIND $rows AS row "
            "MATCH (s:Article {id: row.source}) MATCH (t:Article {id: row.target}) "
            f"MERGE (s)-[x:{kind}]->(t) SET x.quote = row.quote",
            rows,
        )
    for kind, rows in data.concept_edges.items():
        source_label, target_label = _CONCEPT_ENDPOINTS[kind]
        _run(
            driver,
            "UNWIND $rows AS row "
            f"MATCH (s:{source_label} {{id: row.source}}) MATCH (t:{target_label} {{id: row.target}}) "
            f"MERGE (s)-[x:{kind} {{article_id: row.article_id}}]->(t) ON CREATE SET x.quote = row.quote",
            rows,
        )


def count_graph(driver: Driver) -> tuple[dict[str, int], dict[str, int]]:
    """(레이블별 노드 수, 관계 타입별 관계 수)."""
    nodes = driver.execute_query(
        "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY label"
    ).records
    edges = driver.execute_query(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY type"
    ).records
    return {r["label"]: r["count"] for r in nodes}, {r["type"]: r["count"] for r in edges}
