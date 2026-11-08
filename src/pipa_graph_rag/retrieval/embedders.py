"""임베딩 모델을 이름으로 고른다."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .embedding import Embedder, EmbeddingFailed, EmbeddingResult

if TYPE_CHECKING:
    import httpx
    from openai import OpenAI

OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"

# Qwen3-Embedding 모델 카드가 질문 앞에 붙이라고 보여 주는 작업 설명 예시다. 이 데이터에 맞춰 고르지 않았다.
QWEN_QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


class OpenAIEmbedder:
    def __init__(self, model: str, client: OpenAI | None = None):
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.name = f"openai/{model}"
        self.request_model = model
        self._client = client

    def embed(self, texts: list[str]) -> EmbeddingResult:
        response = self._client.embeddings.create(model=self.request_model, input=texts)
        usage = getattr(response, "usage", None)
        return EmbeddingResult(
            vectors=[item.embedding for item in sorted(response.data, key=lambda item: item.index)],
            tokens=getattr(usage, "prompt_tokens", None),
            model=getattr(response, "model", None),
        )


class OpenRouterEmbedder:
    def __init__(self, model: str, variant: str | None = None, client: httpx.Client | None = None):
        """variant: 모델 ID 뒤에 `:변형`으로 붙는 라우팅 변형(예: floor). 없으면 기본 라우팅을 쓴다."""
        if client is None:
            import httpx

            client = httpx.Client(timeout=120)
        self.name = f"openrouter/{model}"
        self.request_model = f"{model}:{variant}" if variant else model
        self._client = client

    def embed(self, texts: list[str]) -> EmbeddingResult:
        response = self._client.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            json={"model": self.request_model, "input": texts, "encoding_format": "float"},
        )
        if response.status_code != 200:
            raise EmbeddingFailed(f"HTTP {response.status_code} {response.text[:200]}")
        body = response.json()
        return EmbeddingResult(
            vectors=[item["embedding"] for item in sorted(body["data"], key=lambda item: item["index"])],
            tokens=(body.get("usage") or {}).get("prompt_tokens"),
            model=body.get("model"),
        )


class MissingApiKey(Exception):
    """임베딩 API를 부르는 데 필요한 환경변수가 없다."""


@dataclass(frozen=True)
class EmbedderSpec:
    name: str  # 캐시 파일 이름. API 키 없이 알 수 있어야 해서 만들어진 임베더에서 읽지 않는다
    build: Callable[[], Embedder]
    api_key_env: str  # 호출에 필요한 API 키의 환경변수 이름
    query_instruction: str | None = None  # 질문 앞에 붙이는 작업 설명. 조문에는 붙이지 않는다

    def format_query(self, question: str) -> str:
        if self.query_instruction is None:
            return question
        return f"Instruct: {self.query_instruction}\nQuery:{question}"


EMBEDDERS: dict[str, EmbedderSpec] = {
    "openai-small": EmbedderSpec(
        "openai/text-embedding-3-small",
        lambda: OpenAIEmbedder("text-embedding-3-small"),
        "OPENAI_API_KEY",
    ),
    "qwen3-8b": EmbedderSpec(
        "openrouter/qwen/qwen3-embedding-8b",
        lambda: OpenRouterEmbedder("qwen/qwen3-embedding-8b", variant="floor"),
        "OPENROUTER_API_KEY",
        query_instruction=QWEN_QUERY_INSTRUCTION,
    ),
    # 질문에 지시문을 붙이지 않는 qwen3-8b. 같은 캐시 파일을 쓴다. 지시문의 효과를 비교하는 데 쓴다
    "qwen3-8b-plain": EmbedderSpec(
        "openrouter/qwen/qwen3-embedding-8b",
        lambda: OpenRouterEmbedder("qwen/qwen3-embedding-8b", variant="floor"),
        "OPENROUTER_API_KEY",
    ),
}


def live_query_embedder(spec: EmbedderSpec) -> Callable[[str], list[float]]:
    """질문을 호출할 때마다 임베딩 API로 임베딩하는 함수. API 키가 없으면 MissingApiKey를 낸다."""
    if not os.environ.get(spec.api_key_env):
        raise MissingApiKey(f"{spec.name} 임베딩에는 환경변수 {spec.api_key_env}가 필요하다(.env에 넣는다)")
    embedder = spec.build()
    return lambda question: embedder.embed([spec.format_query(question)]).vectors[0]
