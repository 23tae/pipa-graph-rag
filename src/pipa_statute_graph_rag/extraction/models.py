"""LLM이 돌려주는 원본 출력과 검증 게이트의 결과."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..schema import ConceptExtraction, ConceptRelationKind, Entity


class RawRelation(BaseModel):
    kind: ConceptRelationKind
    source_id: str
    target_id: str
    quote: str


class RawExtraction(BaseModel):
    """조문 하나에서 LLM이 추출한 값. 검증하기 전이라 관계 하나가 틀려도 파싱은 성공한다."""

    entities: list[Entity] = Field(default_factory=list)
    relations: list[RawRelation] = Field(default_factory=list)
    is_legal_basis: bool | None = None


class ExtractorResult(BaseModel):
    raw: RawExtraction
    input_tokens: int
    output_tokens: int  # 추론 토큰을 포함한다
    reasoning_tokens: int


class Rejection(BaseModel):
    relation: RawRelation
    reason: str


class GateResult(BaseModel):
    extraction: ConceptExtraction
    proposed: int
    rejected: list[Rejection]

    @property
    def accepted(self) -> int:
        return len(self.extraction.relations)
