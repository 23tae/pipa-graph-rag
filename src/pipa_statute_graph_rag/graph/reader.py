"""Neo4j에서 조문을 읽는다."""

from __future__ import annotations

from neo4j import Driver

from ..schema import Article


def read_articles(driver: Driver) -> list[Article]:
    records = driver.execute_query(
        "MATCH (a:Article) RETURN a.id AS id, a.law AS law, a.number AS number, a.branch AS branch, "
        "a.title AS title, a.text AS text, a.effective_date AS effective_date ORDER BY a.id"
    ).records
    return [Article(**dict(r)) for r in records]
