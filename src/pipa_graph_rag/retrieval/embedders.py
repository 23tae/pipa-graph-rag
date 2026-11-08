"""임베딩 모델을 이름으로 고른다."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .embedding import Embedder, EmbeddingFailed

if TYPE_CHECKING:
    import httpx
    from openai import OpenAI

OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"


class OpenAIEmbedder:
    def __init__(self, model: str, client: OpenAI | None = None):
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.name = f"openai/{model}"
        self._model = model
        self._client = client

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


class OpenRouterEmbedder:
    def __init__(self, model: str, client: httpx.Client | None = None):
        if client is None:
            import httpx

            client = httpx.Client(timeout=120)
        self.name = f"openrouter/{model}"
        self._model = model
        self._client = client

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            json={"model": self._model, "input": texts, "encoding_format": "float"},
        )
        if response.status_code != 200:
            raise EmbeddingFailed(f"HTTP {response.status_code} {response.text[:200]}")
        return [item["embedding"] for item in sorted(response.json()["data"], key=lambda item: item["index"])]


@dataclass(frozen=True)
class EmbedderSpec:
    name: str  # 캐시 파일 이름. API 키 없이 알 수 있어야 해서 만들어진 임베더에서 읽지 않는다
    build: Callable[[], Embedder]


EMBEDDERS: dict[str, EmbedderSpec] = {
    "openai-small": EmbedderSpec(
        "openai/text-embedding-3-small", lambda: OpenAIEmbedder("text-embedding-3-small")
    ),
    "qwen3-8b": EmbedderSpec(
        "openrouter/qwen/qwen3-embedding-8b", lambda: OpenRouterEmbedder("qwen/qwen3-embedding-8b")
    ),
}
