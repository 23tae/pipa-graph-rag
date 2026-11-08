"""질문에서 조문을 찾는다. 키워드로 시드 조문을 찾고, 그래프로 이웃 조문까지 넓힌다."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from ..graph.model import is_deleted
from ..schema import Article
from .bm25 import Bm25Index
from .expand import Expansion, Neighbors, expand


class Mode(StrEnum):
    KEYWORD = "keyword"  # 키워드 검색만
    REFERENCES = "references"  # 키워드 검색 + 조문 사이 관계로 확장
    REFERENCES_CONCEPTS = "references+concepts"  # 위에 개념 연결로도 확장


class Retrieval(BaseModel):
    question: str
    mode: Mode
    depth: int
    scores: dict[str, float]  # 시드 조문의 BM25 점수
    expansion: Expansion


def build_index(articles: list[Article]) -> Bm25Index:
    """삭제된 조문을 뺀 조문 본문으로 색인한다(조문 제목과 항목이 본문에 들어 있다)."""
    return Bm25Index({a.id: a.text for a in articles if not is_deleted(a)})


def retrieve(
    index: Bm25Index,
    question: str,
    *,
    mode: Mode,
    top_k: int,
    depth: int,
    references: Neighbors | None = None,
    concepts: Neighbors | None = None,
) -> Retrieval:
    ranked = index.search(question, top_k)
    seeds = [doc_id for doc_id, _ in ranked]
    neighbors: list[Neighbors] = []
    if mode in (Mode.REFERENCES, Mode.REFERENCES_CONCEPTS):
        neighbors.append(_required(references, "references"))
    if mode == Mode.REFERENCES_CONCEPTS:
        neighbors.append(_required(concepts, "concepts"))
    return Retrieval(
        question=question,
        mode=mode,
        depth=depth,
        scores=dict(ranked),
        expansion=expand(seeds, neighbors, depth if neighbors else 0),
    )


def _required(neighbors: Neighbors | None, name: str) -> Neighbors:
    if neighbors is None:
        raise ValueError(f"{name} 이웃 함수가 필요하다")
    return neighbors
