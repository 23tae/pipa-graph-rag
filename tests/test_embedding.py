import json
import math

import httpx
import pytest

from pipa_graph_rag.retrieval import build_embedding_index
from pipa_graph_rag.retrieval import embedding as embedding_module
from pipa_graph_rag.retrieval.embedders import OPENROUTER_URL, OpenAIEmbedder, OpenRouterEmbedder
from pipa_graph_rag.retrieval.embedding import (
    EmbeddingCache,
    EmbeddingFailed,
    EmbeddingIndex,
    EmbeddingNotCached,
    cache_path,
    embed_missing,
    plan_embedding,
)
from pipa_graph_rag.retrieval.hybrid import RRF_K, HybridIndex
from pipa_graph_rag.schema import Article


def article(id: str, text: str) -> Article:
    law, rest = id.split(":")
    return Article(id=id, law=law, number=int(rest), title="제목", text=text, effective_date="20260911")


class FakeEmbedder:
    name = "fake/model"

    def __init__(self, fail_on_call: int | None = None):
        self.calls: list[list[str]] = []
        self._fail_on_call = fail_on_call

    def embed(self, texts):
        self.calls.append(list(texts))
        if self._fail_on_call == len(self.calls):
            raise RuntimeError("실패")
        return [[float(len(text)), 1.0] for text in texts]


def test_cache_path_replaces_slashes_in_the_model_name(tmp_path):
    assert cache_path(tmp_path, "openrouter/qwen/qwen3") == tmp_path / "openrouter--qwen--qwen3.json"


def test_cache_round_trips_through_the_file(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    cache.put("열람", [1.0, 2.0])
    cache.save()
    assert EmbeddingCache(tmp_path / "e.json").get("열람") == [1.0, 2.0]


def test_cache_get_raises_for_text_without_a_vector(tmp_path):
    with pytest.raises(EmbeddingNotCached):
        EmbeddingCache(tmp_path / "e.json").get("없는 글")


def test_cache_rejects_a_vector_with_a_different_dimension(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    cache.put("가", [1.0, 2.0])
    with pytest.raises(EmbeddingFailed):
        cache.put("나", [1.0, 2.0, 3.0])


def test_plan_counts_each_missing_text_once(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_module, "BATCH_SIZE", 2)
    cache = EmbeddingCache(tmp_path / "e.json")
    cache.put("있음", [1.0])
    plan = plan_embedding(["있음", "가", "가", "나나", "다다다"], cache)
    assert plan.missing == ["가", "나나", "다다다"]
    assert plan.requests == 2
    assert plan.characters == 6


def test_embed_missing_sends_only_missing_texts_in_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_module, "BATCH_SIZE", 2)
    cache = EmbeddingCache(tmp_path / "e.json")
    cache.put("있음", [9.0, 9.0])
    embedder = FakeEmbedder()
    assert embed_missing(embedder, cache, ["있음", "가", "나", "다"]) == 2
    assert embedder.calls == [["가", "나"], ["다"]]
    assert embed_missing(embedder, cache, ["있음", "가", "나", "다"]) == 0
    assert len(embedder.calls) == 2


def test_embed_missing_keeps_vectors_received_before_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_module, "BATCH_SIZE", 1)
    cache = EmbeddingCache(tmp_path / "e.json")
    with pytest.raises(RuntimeError):
        embed_missing(FakeEmbedder(fail_on_call=2), cache, ["가", "나"])
    saved = EmbeddingCache(tmp_path / "e.json")
    assert "가" in saved
    assert "나" not in saved


def test_embed_missing_rejects_a_response_with_the_wrong_vector_count(tmp_path):
    class Short:
        name = "short"

        def embed(self, texts):
            return [[1.0]]

    with pytest.raises(EmbeddingFailed):
        embed_missing(Short(), EmbeddingCache(tmp_path / "e.json"), ["가", "나"])


def test_search_orders_by_cosine_similarity_and_breaks_ties_by_id():
    vectors = {"far": [0.0, 1.0], "near": [1.0, 0.1], "b": [1.0, 0.0], "a": [2.0, 0.0]}
    index = EmbeddingIndex(vectors, lambda query: [1.0, 0.0])
    ranked = index.search("질문", 3)
    assert [doc_id for doc_id, _ in ranked] == ["a", "b", "near"]
    assert ranked[0][1] == pytest.approx(1.0)
    assert ranked[2][1] == pytest.approx(1.0 / math.sqrt(1.01))


def test_search_rejects_a_zero_vector():
    index = EmbeddingIndex({"a": [1.0]}, lambda query: [0.0])
    with pytest.raises(EmbeddingFailed):
        index.search("질문", 1)


def test_embedding_index_built_from_cache_never_calls_an_embedder(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.json")
    articles = [article("law:1", "열람"), article("law:2", "파기"), article("law:3", "제3조 삭제 <2020.1.1>")]
    cache.put("열람", [1.0, 0.0])
    cache.put("파기", [0.0, 1.0])
    cache.put("열람 질문", [1.0, 0.1])
    index = build_embedding_index(articles, cache)
    assert len(index) == 2
    assert index.search("열람 질문", 1)[0][0] == "law:1"
    with pytest.raises(EmbeddingNotCached):
        index.search("캐시에 없는 질문", 1)


def test_hybrid_sums_reciprocal_ranks_from_both_searches():
    class Ranked:
        def __init__(self, ids):
            self.ids = ids

        def search(self, query, k):
            return [(doc_id, 0.0) for doc_id in self.ids][:k]

    hybrid = HybridIndex(Ranked(["a", "b", "c"]), Ranked(["c", "b", "x"]), candidates=3)
    ranked = hybrid.search("질문", 4)
    assert [doc_id for doc_id, _ in ranked] == ["c", "b", "a", "x"]
    assert ranked[0][1] == pytest.approx(1 / (RRF_K + 3) + 1 / (RRF_K + 1))
    assert ranked[1][1] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 2))
    assert ranked[2][1] == pytest.approx(1 / (RRF_K + 1))
    assert ranked[3][1] == pytest.approx(1 / (RRF_K + 3))


def test_hybrid_returns_a_document_found_by_only_one_search():
    class Ranked:
        def __init__(self, ids):
            self.ids = ids

        def search(self, query, k):
            return [(doc_id, 0.0) for doc_id in self.ids][:k]

    hybrid = HybridIndex(Ranked([]), Ranked(["only"]), candidates=5)
    assert hybrid.search("질문", 3) == [("only", pytest.approx(1 / (RRF_K + 1)))]


def test_openai_embedder_returns_vectors_in_input_order():
    class Item:
        def __init__(self, index, embedding):
            self.index, self.embedding = index, embedding

    class Embeddings:
        def create(self, model, input):
            assert model == "text-embedding-3-small" and input == ["가", "나"]
            return type("R", (), {"data": [Item(1, [2.0]), Item(0, [1.0])]})()

    client = type("C", (), {"embeddings": Embeddings()})()
    embedder = OpenAIEmbedder("text-embedding-3-small", client=client)
    assert embedder.name == "openai/text-embedding-3-small"
    assert embedder.embed(["가", "나"]) == [[1.0], [2.0]]


def openrouter_embedder(handler) -> OpenRouterEmbedder:
    return OpenRouterEmbedder(
        "vendor/model", client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_openrouter_embedder_posts_the_texts_and_orders_by_index(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]},
        )

    embedder = openrouter_embedder(handler)
    assert embedder.name == "openrouter/vendor/model"
    assert embedder.embed(["가", "나"]) == [[1.0], [2.0]]
    assert seen == {
        "url": OPENROUTER_URL,
        "auth": "Bearer test-key",
        "body": {"model": "vendor/model", "input": ["가", "나"]},
    }


def test_openrouter_embedder_raises_on_an_http_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    embedder = openrouter_embedder(lambda request: httpx.Response(401, text="no"))
    with pytest.raises(EmbeddingFailed):
        embedder.embed(["가"])
