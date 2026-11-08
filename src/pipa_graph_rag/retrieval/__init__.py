"""질문에서 조문을 찾는다. 키워드로 시드 조문을 찾고, 그래프로 이웃 조문까지 넓힌다."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel

from ..graph.model import is_deleted
from ..schema import Article
from .bm25 import Bm25Index
from .embedding import EmbeddingCache, EmbeddingIndex
from .expand import Expansion, Neighbors, expand
from .hybrid import Searcher


class Mode(StrEnum):
    KEYWORD = "keyword"  # 시드 검색만(확장 없음). 시드는 Seed로 고른다
    REFERENCES = "references"  # 시드 검색 + 조문 사이 관계로 확장
    REFERENCES_CONCEPTS = "references+concepts"  # 위에 개념 연결로도 확장


class Seed(StrEnum):
    BM25 = "bm25"  # 문자 n-gram BM25
    EMBEDDING = "embedding"  # 임베딩 코사인 유사도
    HYBRID = "hybrid"  # 위 둘의 순위를 합산


class Retrieval(BaseModel):
    question: str
    mode: Mode
    depth: int
    scores: dict[str, float]  # 시드 조문의 점수. 시드 검색 방식마다 눈금이 다르다
    expansion: Expansion


def indexed_articles(articles: list[Article]) -> list[Article]:
    """삭제된 조문을 뺀 색인 대상 조문."""
    return [a for a in articles if not is_deleted(a)]


def build_index(articles: list[Article]) -> Bm25Index:
    """조문 본문으로 색인한다(조문 제목과 항목이 본문에 들어 있다)."""
    return Bm25Index({a.id: a.text for a in indexed_articles(articles)})


def build_embedding_index(
    articles: list[Article],
    cache: EmbeddingCache,
    format_query: Callable[[str], str] = lambda question: question,
) -> EmbeddingIndex:
    """캐시에 저장된 벡터로 색인한다. 질문 벡터도 캐시에서만 읽어서 API를 부르지 않는다.

    format_query는 질문을 임베딩할 때 쓴 형태(지시문을 붙인 텍스트)로 바꾼다. 조문에는 적용하지 않는다."""
    return EmbeddingIndex(
        {a.id: cache.get(a.text) for a in indexed_articles(articles)},
        lambda question: cache.get(format_query(question)),
    )


def retrieve(
    index: Searcher,
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
