import hashlib

import pytest

from pipa_graph_rag.answer import AnswerOutput, ask, check_citations
from pipa_graph_rag.answer.prompt import (
    available_versions,
    build_user_message,
    load_prompt,
)
from pipa_graph_rag.config import ANSWER_PROMPT_VERSION
from pipa_graph_rag.schema import Article

# 만든 프롬프트 버전의 본문은 고치지 않는다. 바꾸려면 새 버전을 추가하고 해시를 더한다.
PINNED_SHA256 = {"v1": "22dc86ba22b656a9aa70c59b3260d7f8ed3117ee1dc8d4a77ccf3e5f3591c809"}


def article(id: str, text: str = "본문") -> Article:
    law, rest = id.split(":")
    return Article(id=id, law=law, number=int(rest), title="제목", text=text, effective_date="20260911")


class FakeAnswerer:
    model = "fake"
    prompt_id = "answer/vtest"
    prompt_sha256 = "abc"

    def __init__(self, citations: list[str]):
        self.citations = citations
        self.seen: list[str] = []

    def answer(self, question, articles):
        self.seen = [a.id for a in articles]
        return AnswerOutput(answer="답", citations=self.citations), 11, 7


def test_citation_of_an_article_that_was_not_retrieved_is_reported_once():
    assert check_citations(["law:1", "law:9", "law:9", "decree:2"], ["law:1", "decree:2"]) == ["law:9"]


def test_ask_strips_brackets_from_citations_before_checking_them():
    result = ask(FakeAnswerer(["[law:1]", " [decree:2] ", "[law:99]"]), "질문", [article("law:1"), article("decree:2")])
    assert result.citations == ["law:1", "decree:2", "law:99"]
    assert result.invalid_citations == ["law:99"]


def test_all_retrieved_citations_are_valid():
    assert check_citations(["law:1"], ["law:1", "law:2"]) == []


def test_ask_passes_only_the_retrieved_articles_and_reports_invalid_citations():
    answerer = FakeAnswerer(["law:1", "law:99"])
    result = ask(answerer, "질문", [article("law:1"), article("decree:2")])
    assert answerer.seen == ["law:1", "decree:2"]
    assert result.citations == ["law:1", "law:99"]
    assert result.invalid_citations == ["law:99"]
    assert (result.input_tokens, result.output_tokens, result.prompt_id) == (11, 7, "answer/vtest")


def test_user_message_labels_each_article_with_its_id_and_law_name():
    message = build_user_message("질문?", [article("law:1", "법 본문"), article("decree:2", "령 본문")])
    assert "[law:1] 개인정보 보호법\n법 본문" in message
    assert "[decree:2] 개인정보 보호법 시행령\n령 본문" in message
    assert message.endswith("## 질문\n\n질문?")


def test_prompt_versions_are_pinned():
    assert set(available_versions()) == set(PINNED_SHA256)
    assert ANSWER_PROMPT_VERSION in PINNED_SHA256
    for version, expected in PINNED_SHA256.items():
        assert hashlib.sha256(load_prompt(version).encode("utf-8")).hexdigest() == expected


def test_unknown_prompt_version_names_available_ones():
    with pytest.raises(ValueError, match="v1"):
        load_prompt("v99")
