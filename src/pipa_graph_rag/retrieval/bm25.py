"""문자 n-gram(2~3) BM25 키워드 검색. 형태소 분석기 없이 한국어 조문을 검색한다."""

from __future__ import annotations

import math
import re
from collections import Counter

_WORD = re.compile(r"[0-9A-Za-z가-힣]+")
NGRAM_SIZES = (2, 3)

# BM25의 일반적인 기본값. 이 데이터에서 조정하지 않았다.
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """단어마다 2글자, 3글자 조각을 만든다. 한 글자 단어는 조각이 없어서 버려진다."""
    tokens = []
    for word in _WORD.findall(text):
        for size in NGRAM_SIZES:
            tokens.extend(word[i : i + size] for i in range(len(word) - size + 1))
    return tokens


class Bm25Index:
    def __init__(self, documents: dict[str, str], k1: float = K1, b: float = B):
        self._k1 = k1
        self._b = b
        self._counts = {doc_id: Counter(tokenize(text)) for doc_id, text in documents.items()}
        self._lengths = {doc_id: sum(c.values()) for doc_id, c in self._counts.items()}
        self._average_length = sum(self._lengths.values()) / max(1, len(self._lengths))
        document_frequency: Counter[str] = Counter()
        for counts in self._counts.values():
            document_frequency.update(counts.keys())
        total = len(self._counts)
        self._idf = {
            token: math.log((total - df + 0.5) / (df + 0.5) + 1) for token, df in document_frequency.items()
        }

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """점수가 0보다 큰 문서를 점수 순으로 최대 k개. 점수가 같으면 문서 ID 순이다."""
        query_tokens = Counter(tokenize(query))
        scores: dict[str, float] = {}
        for doc_id, counts in self._counts.items():
            length_norm = 1 - self._b + self._b * self._lengths[doc_id] / self._average_length
            score = 0.0
            for token in query_tokens:
                frequency = counts.get(token, 0)
                if frequency:
                    score += self._idf[token] * frequency * (self._k1 + 1) / (frequency + self._k1 * length_norm)
            if score > 0:
                scores[doc_id] = score
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]
