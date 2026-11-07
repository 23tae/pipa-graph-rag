"""판정기가 지키는 인터페이스. 구현체(Decisions API, Jev)는 이 뒤에 둔다."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from ..schema import Article


class SupportItem(BaseModel):
    """판정할 관계 하나. statement는 관계를 말로 푼 문장, quote는 근거 인용이다."""

    id: str
    statement: str
    quote: str


class JudgeResult(BaseModel):
    probabilities: list[float]  # items와 같은 순서. 인용이 주장을 뒷받침할 확률
    input_tokens: int
    model_version: str | None = None  # 응답이 알려 준 실제 모델. 별칭(latest)이 가리키는 버전을 남긴다


class JudgeFailed(Exception):
    """거절하거나 일부 질문에 답이 없어서 쓸 수 있는 결과가 없다."""


class SupportJudge(Protocol):
    name: str  # 예: decisions/gpt-6-luna
    prompt_id: str  # 예: classify/support/v1
    prompt_sha256: str

    def judge(self, article: Article, items: list[SupportItem]) -> JudgeResult: ...
