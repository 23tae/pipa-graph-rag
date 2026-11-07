"""조문, 조문 관계, 추출 결과를 Neo4j에 넣을 행으로 바꾼다. 데이터베이스를 쓰지 않는다."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..schema import Article, ArticleReference, ConceptExtraction

ENTITY_LABELS = {
    "role": "Role",
    "processing_activity": "ProcessingActivity",
    "data_category": "DataCategory",
    "purpose": "Purpose",
    "deontic": "Deontic",
}
NODE_LABELS = ["Article", *ENTITY_LABELS.values()]

_DELETED = re.compile(r"^제\d+조(?:의\d+)?\s*삭제")


def is_deleted(article: Article) -> bool:
    """본문이 "제N조 삭제 <날짜>"뿐인 조문."""
    return bool(_DELETED.match(article.text))


class GraphData(BaseModel):
    articles: list[dict] = Field(default_factory=list)
    entities: dict[str, list[dict]] = Field(default_factory=dict)  # 노드 레이블 → 행
    mentions: dict[str, list[dict]] = Field(default_factory=dict)  # 레이블 → {article_id, entity_id}
    article_edges: dict[str, list[dict]] = Field(default_factory=dict)  # 관계 타입 → 행
    concept_edges: dict[str, list[dict]] = Field(default_factory=dict)  # 관계 타입 → 행
    skipped: dict[str, int] = Field(default_factory=dict)


def _merge_entity(known: dict, new: dict) -> None:
    """같은 ID의 개체가 여러 조문에서 나오면 unknown이 아닌 값을 우선한다. 값이 서로 다르면 처음 값을 둔다."""
    for key, value in new.items():
        if known.get(key) in (None, "unknown") and value not in (None, "unknown"):
            known[key] = value


def build_graph(
    articles: list[Article],
    references: list[ArticleReference],
    extractions: list[ConceptExtraction],
) -> GraphData:
    kept = [a for a in articles if not is_deleted(a)]
    ids = {a.id for a in kept}
    data = GraphData(skipped={"deleted_articles": len(articles) - len(kept)})
    by_article = {e.article_id: e for e in extractions if e.article_id in ids}

    for article in kept:
        row = {
            "id": article.id,
            "law": article.law,
            "number": article.number,
            "branch": article.branch,
            "label": article.label,
            "title": article.title,
            "text": article.text,
            "effective_date": article.effective_date,
        }
        extraction = by_article.get(article.id)
        if extraction is not None and extraction.is_legal_basis is not None:
            row["is_legal_basis"] = extraction.is_legal_basis
        data.articles.append(row)

    for reference in references:
        if reference.source_id not in ids or reference.target_id not in ids:
            data.skipped["references_to_deleted"] = data.skipped.get("references_to_deleted", 0) + 1
            continue
        data.article_edges.setdefault(reference.kind.value.upper(), []).append(
            {
                "source": reference.source_id,
                "target": reference.target_id,
                "quote": reference.evidence.quote,
            }
        )

    merged: dict[str, dict] = {}
    labels: dict[str, str] = {}
    for extraction in by_article.values():
        mentioned: set[str] = set()
        for entity in extraction.entities:
            attributes = entity.model_dump(mode="json")
            attributes.pop("type")
            label = ENTITY_LABELS[entity.type]
            _merge_entity(merged.setdefault(entity.id, {}), attributes)
            labels[entity.id] = label
            mentioned.add(entity.id)
        for entity_id in sorted(mentioned):
            data.mentions.setdefault(labels[entity_id], []).append(
                {"article_id": extraction.article_id, "entity_id": entity_id}
            )
        for relation in extraction.relations:
            data.concept_edges.setdefault(relation.kind.value.upper(), []).append(
                {
                    "source": relation.source_id,
                    "target": relation.target_id,
                    "article_id": extraction.article_id,
                    "quote": relation.evidence.quote,
                }
            )

    for entity_id, attributes in merged.items():
        data.entities.setdefault(labels[entity_id], []).append(attributes)
    return data
