"""Neo4j에서 이웃 조문을 찾는다."""

from __future__ import annotations

from neo4j import Driver

from ..schema import ReferenceKind
from .expand import Edge, Neighbors

_REFERENCE_TYPES = "|".join(kind.value.upper() for kind in ReferenceKind)


def reference_neighbors(driver: Driver) -> Neighbors:
    """조문 사이 관계(참조, 준용, 예외, 위임)를 방향과 관계없이 따라간다. 예외 조문에서 원칙 조문으로만
    이어져 있어서, 원칙 조문에서 예외 조문을 찾으려면 거꾸로도 가야 한다."""

    def find(ids: list[str]) -> list[Edge]:
        records = driver.execute_query(
            f"MATCH (a:Article)-[r:{_REFERENCE_TYPES}]-(b:Article) WHERE a.id IN $ids "
            "RETURN a.id AS source, b.id AS target, type(r) AS via",
            parameters_={"ids": ids},
        ).records
        return [Edge(**dict(r)) for r in records]

    return find


def concept_neighbors(driver: Driver, max_entity_degree: int | None = None) -> Neighbors:
    """같은 개체를 언급하는 조문으로 넓힌다. max_entity_degree를 주면 그 수 이상의 조문이 언급하는 일반
    개체(`보호위원회` 등)는 건너뛴다."""
    limit = "" if max_entity_degree is None else "AND COUNT { (e)<-[:MENTIONS]-() } < $max_degree "

    def find(ids: list[str]) -> list[Edge]:
        records = driver.execute_query(
            "MATCH (a:Article)-[:MENTIONS]->(e)<-[:MENTIONS]-(b:Article) "
            f"WHERE a.id IN $ids AND a <> b {limit}"
            "RETURN a.id AS source, b.id AS target, '개체:' + e.name AS via",
            parameters_={"ids": ids, "max_degree": max_entity_degree},
        ).records
        return [Edge(**dict(r)) for r in records]

    return find
