"""키워드(BM25) 순위와 임베딩 순위를 합쳐 시드 조문을 고른다."""

from __future__ import annotations

from typing import Protocol

# Reciprocal Rank Fusion에서 흔히 쓰는 값. 이 데이터에서 조정하지 않았다.
RRF_K = 60


class Searcher(Protocol):
    def search(self, query: str, k: int) -> list[tuple[str, float]]: ...


class HybridIndex:
    def __init__(self, keyword: Searcher, embedding: Searcher, candidates: int):
        """candidates: 두 검색에서 각각 가져와 합칠 순위의 수. 전체 조문 수로 둔다."""
        self._keyword = keyword
        self._embedding = embedding
        self._candidates = candidates

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """순위의 역수(1 / (RRF_K + 순위))를 두 검색에서 더한 값으로 정렬한다. 점수가 같으면 문서 ID 순이다."""
        scores: dict[str, float] = {}
        for ranked in (
            self._keyword.search(query, self._candidates),
            self._embedding.search(query, self._candidates),
        ):
            for rank, (doc_id, _) in enumerate(ranked, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (RRF_K + rank)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
