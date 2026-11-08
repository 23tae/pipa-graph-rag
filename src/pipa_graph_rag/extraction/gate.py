"""LLM 출력을 관계 하나씩 검증한다. 통과한 관계만 남기고 거부 이유를 기록한다."""

from __future__ import annotations

import re

from ..schema import Article, ConceptExtraction, ConceptRelation, Entity, Evidence
from .models import GateResult, RawExtraction, Rejection

_WHITESPACE = re.compile(r"\s+")


def _squash(text: str) -> str:
    return _WHITESPACE.sub("", text)


def quote_in_article(quote: str, article: Article) -> bool:
    """공백 차이를 무시하고 인용이 조문 본문의 부분 문자열인지 본다."""
    squashed = _squash(quote)
    return bool(squashed) and squashed in _squash(article.text)


def canonical_id(entity: Entity) -> str:
    return f"{entity.type}:{_squash(entity.name)}"


def apply_gate(article: Article, raw: RawExtraction) -> GateResult:
    """개체 ID를 `{type}:{이름}`으로 통일한 뒤 관계를 하나씩 검사한다."""
    canonical: dict[str, Entity] = {}
    renamed: dict[str, str] = {}
    for entity in raw.entities:
        new_id = canonical_id(entity)
        renamed[entity.id] = new_id
        canonical.setdefault(new_id, entity.model_copy(update={"id": new_id}))

    kept: list[ConceptRelation] = []
    rejected: list[Rejection] = []
    for relation in raw.relations:
        source_id = renamed.get(relation.source_id, relation.source_id)
        target_id = renamed.get(relation.target_id, relation.target_id)
        if not quote_in_article(relation.quote, article):
            rejected.append(Rejection(relation=relation, reason="인용이 조문 본문에 없다"))
            continue
        candidate = ConceptRelation(
            kind=relation.kind,
            source_id=source_id,
            target_id=target_id,
            evidence=Evidence(article_id=article.id, quote=relation.quote),
        )
        try:
            candidate.check_endpoints(canonical, article.id)
        except ValueError as error:
            rejected.append(Rejection(relation=relation, reason=str(error)))
            continue
        kept.append(candidate)

    used = {r.source_id for r in kept} | {r.target_id for r in kept}
    extraction = ConceptExtraction(
        article_id=article.id,
        entities=[e for i, e in canonical.items() if i in used],
        relations=kept,
        is_legal_basis=raw.is_legal_basis,
    )
    return GateResult(extraction=extraction, proposed=len(raw.relations), rejected=rejected)
