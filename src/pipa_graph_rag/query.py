"""질문 하나를 처리하는 흐름. 키워드로 시드 조문을 찾고, 그래프로 넓히고, 가져온 조문만 근거로 답한다."""

from __future__ import annotations

from collections.abc import Callable

from neo4j import Driver
from pydantic import BaseModel

from .answer import Answer, Answerer, OpenAIAnswerer, ask
from .graph.reader import read_articles
from .retrieval import (
    Mode,
    Retrieval,
    Seed,
    build_embedding_index,
    build_index,
    retrieve,
)
from .retrieval.embedding import EmbeddingCache, EmbeddingIndex
from .retrieval.expand import Neighbors
from .retrieval.hybrid import HybridIndex, Searcher
from .retrieval.neo4j_neighbors import concept_neighbors, reference_neighbors
from .schema import Article


class QueryResult(BaseModel):
    retrieval: Retrieval
    answer: Answer | None  # 답변을 만들지 않았으면 None


class QueryEngine:
    """조문을 읽고 시드 검색 색인을 만드는 일을 만들 때 한 번만 하고, run을 부를 때마다 질문을 처리한다."""

    def __init__(
        self,
        articles: list[Article],
        *,
        references: Neighbors,
        concepts: Callable[[int | None], Neighbors],
        answerer: Answerer | None = None,
        embedding: EmbeddingIndex | None = None,
    ):
        self.articles = {a.id: a for a in articles}
        keyword = build_index(articles)
        self._searchers: dict[Seed, Searcher] = {Seed.BM25: keyword}
        if embedding is not None:
            self._searchers[Seed.EMBEDDING] = embedding
            self._searchers[Seed.HYBRID] = HybridIndex(keyword, embedding, len(embedding))
        self._references = references
        self._concepts = concepts
        self._answerer = answerer

    @classmethod
    def from_graph(
        cls,
        driver: Driver,
        answerer: Answerer | None = None,
        embedding_cache: EmbeddingCache | None = None,
    ) -> QueryEngine:
        articles = read_articles(driver)
        return cls(
            articles,
            references=reference_neighbors(driver),
            concepts=lambda max_entity_degree: concept_neighbors(driver, max_entity_degree),
            answerer=answerer,
            embedding=build_embedding_index(articles, embedding_cache) if embedding_cache is not None else None,
        )

    def run(
        self,
        question: str,
        *,
        mode: Mode,
        top_k: int,
        depth: int,
        max_entity_degree: int | None = None,
        answer: bool = True,
        seed: Seed = Seed.BM25,
    ) -> QueryResult:
        if seed not in self._searchers:
            raise ValueError(f"{seed.value} 시드 검색을 쓰려면 임베딩 색인이 필요하다")
        retrieval = retrieve(
            self._searchers[seed],
            question,
            mode=mode,
            top_k=top_k,
            depth=depth,
            references=self._references,
            concepts=self._concepts(max_entity_degree),
        )
        if not answer:
            return QueryResult(retrieval=retrieval, answer=None)
        if self._answerer is None:
            self._answerer = OpenAIAnswerer()
        retrieved = [self.articles[i] for i in retrieval.expansion.articles]
        return QueryResult(retrieval=retrieval, answer=ask(self._answerer, question, retrieved))
