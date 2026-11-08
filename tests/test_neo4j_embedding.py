import pytest

from pipa_graph_rag.retrieval import embedders
from pipa_graph_rag.retrieval import neo4j_embedding as module
from pipa_graph_rag.retrieval.embedders import EMBEDDERS, EmbedderSpec, MissingApiKey, live_query_embedder
from pipa_graph_rag.retrieval.embedding import EmbeddingResult
from pipa_graph_rag.retrieval.neo4j_embedding import (
    INDEX_NAME,
    EmbeddingIndexMissing,
    Neo4jEmbeddingIndex,
    load_embeddings,
    reset_embeddings,
)


class Result:
    def __init__(self, records):
        self.records = records


class FakeDriver:
    """질의 문자열에 들어 있는 조각으로 응답을 고른다. 값이 함수면 질의 인자로 응답을 만든다."""

    def __init__(self, responses=None):
        self.calls: list[tuple[str, dict | None]] = []
        self._responses = responses or {}

    def execute_query(self, query, parameters_=None, **kwargs):
        self.calls.append((query, parameters_))
        for fragment, response in self._responses.items():
            if fragment in query:
                return Result(response(parameters_) if callable(response) else response)
        return Result([])


def index_with(model="m", n=3, query_records=None, embed=lambda question: [1.0, 2.0]):
    driver = FakeDriver(
        {
            "a.embedding IS NOT NULL": [{"model": model, "n": n}],
            "queryNodes": query_records or [],
        }
    )
    return driver, Neo4jEmbeddingIndex(driver, "m", embed)


def test_search_sends_index_name_k_and_the_query_vector_and_orders_by_score_then_id():
    records = [{"id": "b", "score": 0.9}, {"id": "a", "score": 0.9}, {"id": "c", "score": 0.95}]
    driver, index = index_with(query_records=records)
    assert index.search("질문", 3) == [("c", 0.95), ("a", 0.9), ("b", 0.9)]
    query, params = driver.calls[-1]
    assert "db.index.vector.queryNodes" in query
    assert params == {"index": INDEX_NAME, "k": 3, "vector": [1.0, 2.0]}


def test_len_is_the_number_of_articles_with_a_vector():
    assert len(index_with(n=251)[1]) == 251


def test_missing_vectors_ask_for_load_embeddings():
    driver = FakeDriver({"a.embedding IS NOT NULL": []})
    with pytest.raises(EmbeddingIndexMissing, match="load-embeddings"):
        Neo4jEmbeddingIndex(driver, "m", lambda q: [1.0])


def test_vectors_from_another_model_are_rejected():
    driver = FakeDriver({"a.embedding IS NOT NULL": [{"model": "다른/모델", "n": 3}]})
    with pytest.raises(EmbeddingIndexMissing, match="다른/모델"):
        Neo4jEmbeddingIndex(driver, "m", lambda q: [1.0])


def online_driver(loaded=lambda params: [{"n": len(params["rows"])}], states=("POPULATING", "ONLINE")):
    sequence = iter(states)
    return FakeDriver(
        {
            "UNWIND": loaded,
            "SHOW VECTOR INDEXES": lambda params: [{"state": next(sequence)}],
        }
    )


def test_load_embeddings_drops_the_old_index_loads_in_batches_then_creates_the_cosine_index(monkeypatch):
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    driver = online_driver()
    vectors = {f"law:{i}": [0.1, 0.2, 0.3] for i in range(120)}
    assert load_embeddings(driver, vectors, "model/x") == 120
    kinds = [query.split()[0] for query, _ in driver.calls]
    assert kinds == ["DROP", "UNWIND", "UNWIND", "UNWIND", "CREATE", "SHOW", "SHOW"]
    batches = [len(params["rows"]) for query, params in driver.calls if query.startswith("UNWIND")]
    assert batches == [50, 50, 20]
    assert all(params["model"] == "model/x" for query, params in driver.calls if query.startswith("UNWIND"))
    create = next(query for query, _ in driver.calls if query.startswith("CREATE"))
    assert "`vector.dimensions`: 3" in create and "'cosine'" in create and INDEX_NAME in create


def test_load_embeddings_rejects_vectors_of_different_dimensions_before_touching_neo4j():
    driver = online_driver()
    with pytest.raises(ValueError, match="차원"):
        load_embeddings(driver, {"law:1": [1.0], "law:2": [1.0, 2.0]}, "m")
    assert driver.calls == []


def test_load_embeddings_fails_without_creating_the_index_when_articles_are_missing(monkeypatch):
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    driver = online_driver(loaded=lambda params: [{"n": len(params["rows"]) - 1}])
    with pytest.raises(EmbeddingIndexMissing):
        load_embeddings(driver, {"law:1": [1.0], "law:2": [2.0]}, "m")
    assert not any(query.startswith("CREATE") for query, _ in driver.calls)


def test_load_embeddings_gives_up_when_the_index_never_comes_online(monkeypatch):
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(module, "ONLINE_TIMEOUT_SECONDS", -1)
    driver = online_driver(states=("POPULATING",) * 5)
    with pytest.raises(EmbeddingIndexMissing, match="ONLINE"):
        load_embeddings(driver, {"law:1": [1.0]}, "m")


def test_reset_drops_the_index_and_removes_the_vector_properties():
    driver = FakeDriver()
    reset_embeddings(driver)
    assert [query.split()[0] for query, _ in driver.calls] == ["DROP", "MATCH"]
    assert "REMOVE a.embedding, a.embeddingModel" in driver.calls[1][0]


class FakeEmbedder:
    name = "fake"
    request_model = "fake"

    def __init__(self):
        self.texts: list[list[str]] = []

    def embed(self, texts):
        self.texts.append(list(texts))
        return EmbeddingResult([[float(len(texts[0])), 1.0]], tokens=1, model="fake")


def test_live_query_embedder_formats_the_question_and_calls_the_embedder_once(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    fake = FakeEmbedder()
    spec = EmbedderSpec("fake/model", lambda: fake, "FAKE_KEY", query_instruction="지시")
    vector = live_query_embedder(spec)("질문")
    assert fake.texts == [["Instruct: 지시\nQuery:질문"]]
    assert vector == [float(len("Instruct: 지시\nQuery:질문")), 1.0]


def test_live_query_embedder_names_the_missing_api_key_variable(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(MissingApiKey, match="OPENROUTER_API_KEY"):
        live_query_embedder(EMBEDDERS["qwen3-8b"])
    assert embedders.EMBEDDERS["openai-small"].api_key_env == "OPENAI_API_KEY"
