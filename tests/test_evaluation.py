import math

from pipa_statute_graph_rag.answer import AnswerOutput
from pipa_statute_graph_rag.evaluation import (
    Condition,
    Question,
    bootstrap_difference,
    conditions,
    evaluate_all,
    evaluate_question,
    summarize,
    summarize_by_type,
)
from pipa_statute_graph_rag.query import QueryEngine
from pipa_statute_graph_rag.retrieval import Mode
from pipa_statute_graph_rag.retrieval.expand import Edge
from pipa_statute_graph_rag.schema import Article


def article(id: str, text: str) -> Article:
    law, rest = id.split(":")
    return Article(id=id, law=law, number=int(rest), title="제목", text=text, effective_date="20260911")


ARTICLES = [
    article("law:1", "정보주체는 열람을 요구할 수 있다"),
    article("decree:2", "방법과 절차는 대통령령으로 정한다"),
    article("law:3", "개인정보를 파기하여야 한다"),
]


class FakeAnswerer:
    model = "fake"
    prompt_id = "answer/vtest"
    prompt_sha256 = "abc"

    def answer(self, question, articles):
        return AnswerOutput(answer="답", citations=["law:1", "law:3"]), 1, 1


def engine() -> QueryEngine:
    def references(ids):
        return [Edge(source="law:1", target="decree:2", via="DELEGATES_TO")] if "law:1" in ids else []

    return QueryEngine(
        ARTICLES, references=references, concepts=lambda degree: lambda ids: [], answerer=FakeAnswerer()
    )


QUESTION = Question(id="q", type="위임", question="열람 요구", required=["law:1", "decree:2"], related=["law:9"])


def test_keyword_retrieval_metrics_count_required_and_related():
    result = evaluate_question(engine(), QUESTION, Condition(mode=Mode.KEYWORD, depth=0), answer=False)
    assert result.retrieved == ["law:1"]
    assert result.recall_required == 0.5
    assert result.recall_all == 1 / 3
    assert result.precision == 1.0
    assert result.citations is None


def test_reference_expansion_finds_the_delegated_decree():
    result = evaluate_question(engine(), QUESTION, Condition(mode=Mode.REFERENCES, depth=1), answer=False)
    assert result.retrieved == ["law:1", "decree:2"]
    assert result.recall_required == 1.0


def test_citation_metrics_use_the_answer_citations():
    result = evaluate_question(engine(), QUESTION, Condition(mode=Mode.REFERENCES, depth=1), answer=True)
    assert result.citations == ["law:1", "law:3"]
    assert result.citation_precision == 0.5
    assert result.citation_recall == 0.5


def test_summarize_averages_per_condition_and_marks_missing_citation_metrics():
    results = evaluate_all(
        engine(),
        [QUESTION],
        [Condition(mode=Mode.KEYWORD, depth=0), Condition(mode=Mode.REFERENCES, depth=1)],
        answer=False,
        workers=2,
    )
    summary = summarize(results)
    assert summary["keyword"]["size"] == 1 and summary["references/d1"]["size"] == 2
    assert summary["references/d1"]["recall_required"] == 1.0
    assert math.isnan(summary["keyword"]["citation_precision"])


def test_summarize_by_type_averages_each_type_separately():
    other = Question(id="p", type="단일", question="파기", required=["law:3"], related=[])
    conds = [Condition(mode=Mode.KEYWORD, depth=0)]
    results = evaluate_all(engine(), [QUESTION, other], conds, answer=False, workers=1)
    by_type = summarize_by_type(results, [QUESTION, other], "recall_required")
    assert by_type["keyword"] == {"위임": 0.5, "단일": 1.0}


def test_conditions_cover_keyword_references_and_concepts_with_each_skip_value():
    names = [c.name for c in conditions()]
    assert names[0] == "keyword"
    assert "references/d2" in names
    assert "references+concepts/d1" in names and "references+concepts/d2/skip50" in names
    assert len(names) == len(set(names)) == 1 + 2 + 2 * 5


def test_bootstrap_difference_is_zero_for_equal_scores_and_positive_for_better_ones():
    assert bootstrap_difference([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], draws=100) == (0.0, 0.0, 0.0)
    value, low, high = bootstrap_difference([0.0] * 10, [1.0] * 10, draws=100)
    assert (value, low, high) == (1.0, 1.0, 1.0)
