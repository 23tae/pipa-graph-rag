import pytest

from pipa_graph_rag.config import QUERY_TOP_K
from pipa_graph_rag.evaluation import Condition, Question, evaluate_question, with_seeds
from pipa_graph_rag.query import QueryEngine
from pipa_graph_rag.retrieval import Mode, Seed
from pipa_graph_rag.retrieval.embedding import EmbeddingIndex
from pipa_graph_rag.schema import Article


def article(id: str, text: str) -> Article:
    law, rest = id.split(":")
    return Article(id=id, law=law, number=int(rest), title="제목", text=text, effective_date="20260911")


ARTICLES = [
    article("law:1", "열람을 요구할 수 있다"),
    article("law:2", "해외로 옮기는 국외 이전"),
    article("law:3", "개인정보를 파기하여야 한다"),
]
QUESTION = Question(id="q", type="단일", question="외국으로 반출", required=["law:2"], related=[])


def engine(with_embedding: bool = True) -> QueryEngine:
    embedding = None
    if with_embedding:
        vectors = {"law:1": [1.0, 0.0, 0.0], "law:2": [0.0, 1.0, 0.0], "law:3": [0.0, 0.0, 1.0]}
        embedding = EmbeddingIndex(vectors, lambda query: [0.0, 1.0, 0.0])
    return QueryEngine(
        ARTICLES,
        references=lambda ids: [],
        concepts=lambda degree: lambda ids: [],
        embedding=embedding,
    )


def test_condition_name_marks_a_non_default_seed_and_seed_count():
    assert Condition(mode=Mode.KEYWORD, depth=0).name == "keyword"
    assert Condition(mode=Mode.KEYWORD, depth=0, seed=Seed.HYBRID).name == "keyword:hybrid"
    assert Condition(mode=Mode.KEYWORD, depth=0, top_k=10).name == "keyword:k10"
    assert (
        Condition(mode=Mode.REFERENCES_CONCEPTS, depth=1, max_entity_degree=50, seed=Seed.EMBEDDING, top_k=5).name
        == "references+concepts/d1/skip50:embedding:k5"
    )
    assert Condition(mode=Mode.KEYWORD, depth=0, top_k=QUERY_TOP_K).name == "keyword"


def test_with_seeds_expands_each_condition_to_every_seed_and_count():
    base = [Condition(mode=Mode.KEYWORD, depth=0), Condition(mode=Mode.REFERENCES, depth=1)]
    expanded = with_seeds(base, [Seed.BM25, Seed.HYBRID], [3, 5])
    assert [c.name for c in expanded] == [
        "keyword",
        "keyword:k5",
        "keyword:hybrid",
        "keyword:hybrid:k5",
        "references/d1",
        "references/d1:k5",
        "references/d1:hybrid",
        "references/d1:hybrid:k5",
    ]


def test_embedding_seed_finds_the_article_that_keywords_miss():
    keyword = evaluate_question(engine(), QUESTION, Condition(mode=Mode.KEYWORD, depth=0, top_k=1), answer=False)
    embedding = evaluate_question(
        engine(), QUESTION, Condition(mode=Mode.KEYWORD, depth=0, seed=Seed.EMBEDDING, top_k=1), answer=False
    )
    assert keyword.retrieved == []
    assert embedding.retrieved == ["law:2"]
    assert embedding.recall_required == 1.0


def test_hybrid_seed_merges_keyword_and_embedding_results():
    result = evaluate_question(
        engine(), Question(id="q", type="단일", question="열람", required=["law:1"], related=[]),
        Condition(mode=Mode.KEYWORD, depth=0, seed=Seed.HYBRID, top_k=2), answer=False,
    )
    assert result.retrieved == ["law:1", "law:2"]


def test_a_searcher_that_only_returns_top_k_supports_the_embedding_seed_but_not_hybrid():
    class TopKOnly:
        def search(self, query, k):
            return [("law:2", 0.9)][:k]

    engine_with_neo4j = QueryEngine(
        ARTICLES,
        references=lambda ids: [],
        concepts=lambda degree: lambda ids: [],
        embedding=TopKOnly(),
    )
    found = evaluate_question(
        engine_with_neo4j, QUESTION, Condition(mode=Mode.KEYWORD, depth=0, seed=Seed.EMBEDDING, top_k=1), answer=False
    )
    assert found.retrieved == ["law:2"]
    with pytest.raises(ValueError, match="임베딩 색인"):
        evaluate_question(
            engine_with_neo4j, QUESTION, Condition(mode=Mode.KEYWORD, depth=0, seed=Seed.HYBRID), answer=False
        )


def test_embedding_seed_without_an_embedding_index_is_an_error():
    with pytest.raises(ValueError, match="임베딩 색인"):
        evaluate_question(
            engine(with_embedding=False),
            QUESTION,
            Condition(mode=Mode.KEYWORD, depth=0, seed=Seed.HYBRID),
            answer=False,
        )
