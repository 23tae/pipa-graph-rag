import json

import httpx
import pytest

from pipa_statute_graph_rag.classify.base import JudgeFailed, SupportItem
from pipa_statute_graph_rag.classify.jev import MAX_QUESTIONS, JevJudge
from pipa_statute_graph_rag.schema import Article


def article() -> Article:
    return Article(
        id="law:9",
        law="law",
        number=9,
        title="제목",
        text="제9조(제목)\n① 본문이다.",
        effective_date="20260911",
    )


def items(count: int) -> list[SupportItem]:
    return [SupportItem(id=f"r{i}", statement=f"주장{i}", quote=f"인용{i}") for i in range(count)]


def judge_with(handler) -> JevJudge:
    return JevJudge(client=httpx.Client(transport=httpx.MockTransport(handler)))


def ok(request: httpx.Request, value: float = 0.8) -> httpx.Response:
    names = json.loads(request.content)["questions"]
    return httpx.Response(
        200,
        json={
            "model": "typesafe/jev-test",
            "answers": {name: {"type": "noul", "noul": value} for name in names},
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
    )


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def test_request_uses_state_and_noul_questions_keyed_by_item_id():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return ok(request)

    result = judge_with(handler).judge(article(), items(2))
    body = seen["body"]
    assert body["state"] == article().text
    assert set(body["questions"]) == {"r0", "r1"}
    assert body["questions"]["r0"]["type"] == "noul"
    assert "주장0" in body["questions"]["r0"]["instructions"]
    assert "인용0" in body["questions"]["r0"]["instructions"]
    assert seen["auth"] == "Bearer test-key"
    assert result.probabilities == [0.8, 0.8]
    assert (result.input_tokens, result.model_version) == (10, "typesafe/jev-test")


def test_questions_are_split_into_chunks_and_tokens_add_up():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(len(json.loads(request.content)["questions"]))
        return ok(request)

    result = judge_with(handler).judge(article(), items(MAX_QUESTIONS + 3))
    assert calls == [MAX_QUESTIONS, 3]
    assert len(result.probabilities) == MAX_QUESTIONS + 3
    assert result.input_tokens == 20


def test_http_error_raises_judge_failed():
    handler = lambda request: httpx.Response(429, text="rate limited")  # noqa: E731
    with pytest.raises(JudgeFailed, match="429"):
        judge_with(handler).judge(article(), items(1))


def test_missing_answer_raises_judge_failed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "m", "answers": {}, "usage": {"input_tokens": 1, "output_tokens": 0}},
        )

    with pytest.raises(JudgeFailed, match="답이 없는"):
        judge_with(handler).judge(article(), items(1))
