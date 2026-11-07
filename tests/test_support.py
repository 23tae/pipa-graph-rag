import hashlib
from pathlib import Path

import pytest

from pipa_statute_graph_rag.classify.base import JudgeResult, SupportItem
from pipa_statute_graph_rag.classify.prompt import available_versions, load_prompt
from pipa_statute_graph_rag.classify.support import (
    SupportRecord,
    build_items,
    filter_by_support,
    load_support,
    run_judgment,
)
from pipa_statute_graph_rag.schema import (
    Article,
    ConceptExtraction,
    ConceptRelation,
    ConceptRelationKind,
    Deontic,
    DeonticKind,
    Evidence,
    Role,
)

# 만든 프롬프트 버전의 본문은 고치지 않는다. 바꾸려면 새 버전을 추가하고 해시를 더한다.
PINNED = {"support/v1": "4a6adbb4eddcc2fae75373eac662c5811d3c31d5878734064f38f06879c88b5c"}


def article(number: int) -> Article:
    return Article(
        id=f"law:{number}",
        law="law",
        number=number,
        title="제목",
        text=f"제{number}조(제목)\n① 개인정보처리자는 열람하게 하여야 한다.",
        effective_date="20260911",
    )


def extraction(article_id: str = "law:9", count: int = 2) -> ConceptExtraction:
    role = Role(id="role:개인정보처리자", name="개인정보처리자")
    duties = [
        Deontic(id=f"deontic:의무{i}", name=f"의무{i}", kind=DeonticKind.OBLIGATION) for i in range(count)
    ]
    return ConceptExtraction(
        article_id=article_id,
        entities=[role, *duties],
        relations=[
            ConceptRelation(
                kind=ConceptRelationKind.BEARS,
                source_id=role.id,
                target_id=duty.id,
                evidence=Evidence(article_id=article_id, quote="열람하게 하여야 한다"),
            )
            for duty in duties
        ],
    )


class FakeJudge:
    name = "fake"
    prompt_id = "classify/support/vtest"
    prompt_sha256 = "abc"

    def __init__(self, failing: set[str] | None = None, probability: float = 0.9):
        self.calls: list[str] = []
        self.failing = failing or set()
        self.probability = probability

    def judge(self, article: Article, items: list[SupportItem]) -> JudgeResult:
        self.calls.append(article.id)
        if article.id in self.failing:
            raise RuntimeError("호출 실패")
        return JudgeResult(probabilities=[self.probability] * len(items), input_tokens=7)


def test_items_describe_each_relation_with_entity_names_and_its_quote():
    items = build_items(extraction())
    assert [i.id for i in items] == ["r0", "r1"]
    assert items[0].statement == "역할 '개인정보처리자'은(는) 권리·의무 '의무0'의 주체이다."
    assert items[0].quote == "열람하게 하여야 한다"


def test_run_judgment_skips_articles_without_relations_and_saved_ones(tmp_path: Path):
    out = tmp_path / "support.jsonl"
    articles = [article(9), article(10), article(11)]
    extractions = {"law:9": extraction("law:9"), "law:10": extraction("law:10", count=0)}
    judge = FakeJudge()
    saved, failures = run_judgment(articles, extractions, judge, out, workers=1)
    assert (saved, failures) == (1, [])
    assert judge.calls == ["law:9"]
    run_judgment(articles, extractions, judge, out, workers=1)
    assert judge.calls == ["law:9"]
    record = load_support(out)[0]
    assert (record.probabilities, record.input_tokens, record.prompt_id) == (
        [0.9, 0.9],
        7,
        "classify/support/vtest",
    )


def test_failure_of_one_article_does_not_stop_the_others(tmp_path: Path):
    out = tmp_path / "support.jsonl"
    articles = [article(9), article(10)]
    extractions = {"law:9": extraction("law:9"), "law:10": extraction("law:10")}
    saved, failures = run_judgment(articles, extractions, FakeJudge({"law:9"}), out, workers=1)
    assert saved == 1 and [f[0] for f in failures] == ["law:9"]


def record(article_id: str, probabilities: list[float]) -> SupportRecord:
    return SupportRecord(
        article_id=article_id,
        judge="fake",
        prompt_id="x",
        prompt_sha256="y",
        probabilities=probabilities,
        input_tokens=0,
    )


def test_filter_keeps_relations_at_or_above_threshold_and_drops_unused_entities():
    [result] = filter_by_support([extraction()], [record("law:9", [0.95, 0.2])], 0.5)
    assert [r.target_id for r in result.relations] == ["deontic:의무0"]
    assert {e.id for e in result.entities} == {"role:개인정보처리자", "deontic:의무0"}


def test_filter_threshold_is_inclusive():
    [result] = filter_by_support([extraction()], [record("law:9", [0.5, 0.49])], 0.5)
    assert len(result.relations) == 1


def test_filter_leaves_extractions_without_relations_alone():
    empty = extraction(count=0)
    assert filter_by_support([empty], [], 0.5) == [empty]


def test_filter_fails_when_record_is_missing_or_misaligned():
    with pytest.raises(ValueError):
        filter_by_support([extraction()], [], 0.5)
    with pytest.raises(ValueError):
        filter_by_support([extraction()], [record("law:9", [0.9])], 0.5)


def test_prompt_versions_are_pinned():
    assert available_versions("support") == ["v1"]
    for key, expected in PINNED.items():
        task, version = key.split("/")
        text = load_prompt(task, version)
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == expected
