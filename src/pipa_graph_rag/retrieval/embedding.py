"""임베딩 검색. 조문과 질문을 벡터로 바꿔 두고, 질문과 코사인 유사도가 높은 조문을 시드로 고른다."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

BATCH_SIZE = 100  # 한 요청에 보내는 텍스트 수


class Embedder(Protocol):
    name: str  # 캐시 파일 이름에 쓰는 모델 식별자

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingFailed(Exception):
    """임베딩 응답이 오류이거나 요청한 텍스트 수와 맞지 않는다."""


class EmbeddingNotCached(Exception):
    """캐시에 없는 텍스트의 벡터를 호출 없이 읽으려고 했다."""


def cache_path(directory: Path, embedder_name: str) -> Path:
    return directory / f"{embedder_name.replace('/', '--')}.json"


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """텍스트별 벡터를 모델마다 JSON 파일 하나에 저장한다."""

    def __init__(self, path: Path):
        self.path = path
        self._vectors: dict[str, list[float]] = {}
        if path.exists():
            self._vectors = json.loads(path.read_text(encoding="utf-8"))["vectors"]

    def __contains__(self, text: str) -> bool:
        return _key(text) in self._vectors

    def get(self, text: str) -> list[float]:
        try:
            return self._vectors[_key(text)]
        except KeyError:
            raise EmbeddingNotCached(
                f"임베딩이 없는 텍스트: {text[:30]!r}. embed 명령으로 먼저 만든다"
            ) from None

    def put(self, text: str, vector: list[float]) -> None:
        sizes = {len(v) for v in self._vectors.values()}
        if sizes and len(vector) not in sizes:
            raise EmbeddingFailed(f"벡터 차원이 캐시와 다르다: {len(vector)} (캐시 {sizes.pop()})")
        self._vectors[_key(text)] = vector

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"vectors": self._vectors}, separators=(",", ":")), encoding="utf-8"
        )


@dataclass(frozen=True)
class EmbeddingPlan:
    missing: list[str]  # 캐시에 없는 텍스트. 중복은 한 번만 센다
    requests: int
    characters: int


def plan_embedding(texts: list[str], cache: EmbeddingCache) -> EmbeddingPlan:
    missing = list(dict.fromkeys(t for t in texts if t not in cache))
    return EmbeddingPlan(missing, math.ceil(len(missing) / BATCH_SIZE), sum(map(len, missing)))


def embed_missing(embedder: Embedder, cache: EmbeddingCache, texts: list[str]) -> int:
    """캐시에 없는 텍스트만 임베딩해 저장하고 보낸 요청 수를 돌려준다."""
    plan = plan_embedding(texts, cache)
    for start in range(0, len(plan.missing), BATCH_SIZE):
        batch = plan.missing[start : start + BATCH_SIZE]
        vectors = embedder.embed(batch)
        if len(vectors) != len(batch):
            raise EmbeddingFailed(f"텍스트 {len(batch)}개를 보냈는데 벡터 {len(vectors)}개가 왔다")
        for text, vector in zip(batch, vectors, strict=True):
            cache.put(text, vector)
        cache.save()  # 요청마다 저장해서 중간에 실패해도 받은 벡터는 남는다
    return plan.requests


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise EmbeddingFailed("길이가 0인 벡터")
    return [x / norm for x in vector]


class EmbeddingIndex:
    def __init__(self, vectors: dict[str, list[float]], embed_query: Callable[[str], list[float]]):
        self._vectors = {doc_id: _unit(vector) for doc_id, vector in vectors.items()}
        self._embed_query = embed_query

    def __len__(self) -> int:
        return len(self._vectors)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """코사인 유사도가 높은 순으로 최대 k개. 점수가 같으면 문서 ID 순이다."""
        unit = _unit(self._embed_query(query))
        scores = {
            doc_id: sum(a * b for a, b in zip(unit, vector, strict=True))
            for doc_id, vector in self._vectors.items()
        }
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
