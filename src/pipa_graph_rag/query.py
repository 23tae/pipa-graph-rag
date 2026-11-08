"""질문 하나를 처리하는 흐름. 키워드로 시드 조문을 찾고, 그래프로 넓히고, 가져온 조문만 근거로 답한다."""

from __future__ import annotations

from collections.abc import Callable

from neo4j import Driver
from pydantic import BaseModel

from .answer import Answer, Answerer, OpenAIAnswerer, ask
from .graph.reader import read_articles
from .retrieval import Mode, Retrieval, build_index, retrieve
from .retrieval.expand import Neighbors
from .retrieval.neo4j_neighbors import concept_neighbors, reference_neighbors
from .schema import Article


class QueryResult(BaseModel):
    retrieval: Retrieval
    answer: Answer | None  # 답변을 만들지 않았으면 None


class QueryEngine:
    """조문을 읽고 키워드 색인을 만드는 일을 만들 때 한 번만 하고, run을 부를 때마다 질문을 처리한다."""

    def __init__(
        self,
        articles: list[Article],
        *,
        references: Neighbors,
        concepts: Callable[[int | None], Neighbors],
        answerer: Answerer | None = None,
    ):
        self.articles = {a.id: a for a in articles}
        self._index = build_index(articles)
        self._references = references
        self._concepts = concepts
        self._answerer = answerer

    @classmethod
    def from_graph(cls, driver: Driver, answerer: Answerer | None = None) -> QueryEngine:
        return cls(
            read_articles(driver),
            references=reference_neighbors(driver),
            concepts=lambda max_entity_degree: concept_neighbors(driver, max_entity_degree),
            answerer=answerer,
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
    ) -> QueryResult:
        retrieval = retrieve(
            self._index,
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
