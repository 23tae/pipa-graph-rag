from pipa_statute_graph_rag.answer import AnswerOutput
from pipa_statute_graph_rag.graph.reader import read_articles
from pipa_statute_graph_rag.query import QueryEngine
from pipa_statute_graph_rag.retrieval import Mode
from pipa_statute_graph_rag.retrieval.expand import Edge
from pipa_statute_graph_rag.schema import Article


def article(id: str, text: str) -> Article:
    law, rest = id.split(":")
    return Article(id=id, law=law, number=int(rest), title=f"제목 {id}", text=text, effective_date="20260911")


ARTICLES = [
    article("law:1", "정보주체는 열람을 요구할 수 있다"),
    article("law:2", "개인정보처리자는 개인정보를 파기하여야 한다"),
    article("decree:3", "열람 요구의 절차는 대통령령으로 정한다"),
]


class FakeAnswerer:
    model = "fake"
    prompt_id = "answer/vtest"
    prompt_sha256 = "abc"

    def __init__(self):
        self.seen: list[str] = []

    def answer(self, question, articles):
        self.seen = [a.id for a in articles]
        return AnswerOutput(answer="답", citations=[articles[0].id]), 5, 3


def engine(answerer=None) -> QueryEngine:
    def references(ids):
        return [Edge(source="law:1", target="decree:3", via="DELEGATES_TO")] if "law:1" in ids else []

    return QueryEngine(
        ARTICLES,
        references=references,
        concepts=lambda max_entity_degree: lambda ids: [],
        answerer=answerer,
    )


def test_run_retrieves_then_answers_with_only_the_retrieved_articles():
    answerer = FakeAnswerer()
    result = engine(answerer).run("열람 요구", mode=Mode.REFERENCES, top_k=1, depth=1)
    assert result.retrieval.expansion.articles == ["law:1", "decree:3"]
    assert answerer.seen == ["law:1", "decree:3"]
    assert result.answer is not None and result.answer.citations == ["law:1"]


def test_run_without_answer_does_not_call_the_answerer():
    answerer = FakeAnswerer()
    result = engine(answerer).run("열람 요구", mode=Mode.KEYWORD, top_k=1, depth=1, answer=False)
    assert result.answer is None
    assert answerer.seen == []


def test_run_passes_max_entity_degree_to_the_concept_neighbors():
    seen: list[int | None] = []

    def concepts(max_entity_degree):
        seen.append(max_entity_degree)
        return lambda ids: []

    query = QueryEngine(ARTICLES, references=lambda ids: [], concepts=concepts)
    query.run("열람", mode=Mode.REFERENCES_CONCEPTS, top_k=1, depth=1, max_entity_degree=20, answer=False)
    assert seen == [20]


class FakeResult:
    def __init__(self, records):
        self.records = records


class FakeDriver:
    def __init__(self, records):
        self._records = records

    def execute_query(self, query, **kwargs):
        return FakeResult(self._records)


def test_read_articles_builds_articles_from_graph_nodes_with_a_missing_branch():
    node = {
        "id": "law:7-11",
        "law": "law",
        "number": 7,
        "branch": 11,
        "title": "제목",
        "text": "본문",
        "effective_date": "20260911",
    }
    plain = {**node, "id": "law:8", "number": 8, "branch": None}
    articles = read_articles(FakeDriver([node, plain]))
    assert [(a.id, a.branch) for a in articles] == [("law:7-11", 11), ("law:8", None)]
