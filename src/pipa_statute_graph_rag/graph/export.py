"""Neo4j 그래프를 JSON 파일로 내보내고, 파일에서 읽는다. 파일로 LLM 추출 없이 같은 그래프를 다시 적재할 수 있다."""

from __future__ import annotations

from pathlib import Path

from neo4j import Driver

from ..schema import ConceptRelationKind, ReferenceKind
from .model import ENTITY_LABELS, GraphData


def read_graph(driver: Driver) -> GraphData:
    """Neo4j에 적재된 그래프를 적재 때 쓰는 행 형태로 읽는다. 같은 조문의 중복 관계는 적재 때 합쳐진 상태다."""

    def rows(query: str) -> list[dict]:
        return [dict(r) for r in driver.execute_query(query).records]

    data = GraphData()
    data.articles = [r["p"] for r in rows("MATCH (a:Article) RETURN properties(a) AS p ORDER BY a.id")]
    for label in ENTITY_LABELS.values():
        entities = [r["p"] for r in rows(f"MATCH (n:{label}) RETURN properties(n) AS p ORDER BY n.id")]
        if entities:
            data.entities[label] = entities
        mentions = rows(
            f"MATCH (a:Article)-[:MENTIONS]->(e:{label}) "
            "RETURN a.id AS article_id, e.id AS entity_id ORDER BY article_id, entity_id"
        )
        if mentions:
            data.mentions[label] = mentions
    for kind in ReferenceKind:
        edge_type = kind.value.upper()
        edges = rows(
            f"MATCH (s:Article)-[r:{edge_type}]->(t:Article) "
            "RETURN s.id AS source, t.id AS target, r.quote AS quote ORDER BY source, target"
        )
        if edges:
            data.article_edges[edge_type] = edges
    for kind in ConceptRelationKind:
        edge_type = kind.value.upper()
        edges = rows(
            f"MATCH (s)-[r:{edge_type}]->(t) "
            "RETURN s.id AS source, t.id AS target, r.article_id AS article_id, r.quote AS quote "
            "ORDER BY article_id, source, target"
        )
        if edges:
            data.concept_edges[edge_type] = edges
    return data


def subset(data: GraphData, article_ids: set[str]) -> GraphData:
    """지정한 조문과, 그 조문이 언급하는 개체, 그 조문 사이의 관계만 남긴다."""
    kept = GraphData()
    kept.articles = [a for a in data.articles if a["id"] in article_ids]
    kept_ids = {a["id"] for a in kept.articles}
    entity_ids: set[str] = set()
    for label, mentions in data.mentions.items():
        rows = [m for m in mentions if m["article_id"] in kept_ids]
        if rows:
            kept.mentions[label] = rows
            entity_ids |= {m["entity_id"] for m in rows}
    for label, entities in data.entities.items():
        rows = [e for e in entities if e["id"] in entity_ids]
        if rows:
            kept.entities[label] = rows
    nodes = kept_ids | entity_ids
    for kind, edges in data.article_edges.items():
        rows = [e for e in edges if e["source"] in kept_ids and e["target"] in kept_ids]
        if rows:
            kept.article_edges[kind] = rows
    for kind, edges in data.concept_edges.items():
        rows = [e for e in edges if e["article_id"] in kept_ids and e["source"] in nodes and e["target"] in nodes]
        if rows:
            kept.concept_edges[kind] = rows
    return kept


def save_graph(data: GraphData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data.model_dump_json(indent=1), encoding="utf-8")


def load_graph_file(path: Path) -> GraphData:
    return GraphData.model_validate_json(path.read_text(encoding="utf-8"))
