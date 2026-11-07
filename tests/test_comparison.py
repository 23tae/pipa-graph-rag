import math

import pytest

from pipa_statute_graph_rag.classify.base import JudgeResult
from pipa_statute_graph_rag.classify.comparison import (
    Relation,
    compare_judges,
    evaluate,
    run_judge,
    stratified_sample,
    stratum,
)
from pipa_statute_graph_rag.schema import Article


def relation(key: str) -> Relation:
    return Relation(key=key, article_id=key.split("#")[0], statement=f"주장 {key}", quote=f"인용 {key}")


def pool(low: int, mid: int, high: int):
    probabilities, relations = {}, []
    for name, count, value in (("a", low, 0.02), ("b", mid, 0.5), ("c", high, 0.97)):
        for i in range(count):
            r = relation(f"law:{name}#{i}")
            relations.append(r)
            probabilities[r.key] = value
    return relations, probabilities


def test_stratum_boundaries():
    assert [stratum(p) for p in (0.0, 0.099, 0.1, 0.5, 0.899, 0.9, 1.0)] == [
        "low", "low", "mid", "mid", "mid", "high", "high",
    ]


def test_sample_takes_the_same_number_from_each_stratum():
    relations, probabilities = pool(10, 10, 10)
    sample, strata = stratified_sample(relations, probabilities, 4, seed=1)
    assert len(sample) == 12
    assert sorted(strata.values()) == ["high"] * 4 + ["low"] * 4 + ["mid"] * 4


def test_sample_is_reproducible_for_a_seed_and_differs_for_another():
    relations, probabilities = pool(10, 10, 10)
    first = [r.key for r in stratified_sample(relations, probabilities, 4, seed=1)[0]]
    again = [r.key for r in stratified_sample(list(reversed(relations)), probabilities, 4, seed=1)[0]]
    other = [r.key for r in stratified_sample(relations, probabilities, 4, seed=2)[0]]
    assert first == again
    assert first != other


def test_sample_fails_when_a_stratum_is_too_small():
    relations, probabilities = pool(10, 2, 10)
    with pytest.raises(ValueError, match="mid"):
        stratified_sample(relations, probabilities, 4, seed=1)


class FakeJudge:
    name = "fake"
    prompt_id = "classify/support/vtest"
    prompt_sha256 = "abc"

    def __init__(self):
        self.calls: list[list[str]] = []

    def judge(self, article, items):
        self.calls.append([i.id for i in items])
        return JudgeResult(probabilities=[0.8 for _ in items], input_tokens=100)


def test_run_judge_calls_once_per_article_with_only_the_sampled_relations():
    articles = {
        a: Article(id=a, law="law", number=n, title="제목", text="본문", effective_date="20260911")
        for a, n in (("law:1", 1), ("law:2", 2))
    }
    judge = FakeJudge()
    sample = [relation("law:2#0"), relation("law:1#3"), relation("law:1#0")]
    judgments = run_judge(judge, articles, sample)
    assert judge.calls == [["law:1#3", "law:1#0"], ["law:2#0"]]
    assert {j.key for j in judgments} == {"law:1#3", "law:1#0", "law:2#0"}
    assert all(j.probability == 0.8 and j.input_tokens == 100 for j in judgments)


def test_evaluate_computes_accuracy_precision_recall_and_auc():
    labels = {"a": True, "b": True, "c": False, "d": False}
    probabilities = {"a": 0.9, "b": 0.3, "c": 0.6, "d": 0.05}
    strata = {"a": "high", "b": "mid", "c": "mid", "d": "low"}
    metrics = evaluate(labels, probabilities, strata)
    at_half = metrics.by_threshold[0.5]
    assert (at_half["accuracy"], at_half["precision"], at_half["recall"]) == (0.5, 0.5, 0.5)
    at_low = metrics.by_threshold[0.1]
    assert (at_low["accuracy"], at_low["precision"], at_low["recall"]) == (0.75, 2 / 3, 1.0)
    assert metrics.auc == 0.75
    assert metrics.by_stratum["mid"][0.5] == 0.0
    assert metrics.by_bin[0] == (0.0, 0.1, 1, 0.0)


def test_compare_judges_gives_zero_for_identical_judges_and_positive_for_a_better_one():
    keys = [f"k{i}" for i in range(40)]
    labels = {k: i % 2 == 0 for i, k in enumerate(keys)}
    good = {k: 0.9 if labels[k] else 0.1 for k in keys}
    bad = {k: 0.5 for k in keys}
    same = compare_judges(labels, good, good, draws=200)
    assert same["AUC"] == (0.0, 0.0, 0.0)
    better = compare_judges(labels, bad, good, draws=200)
    assert better["AUC"][0] == 0.5 and better["AUC"][1] > 0
    assert better["정확도 0.5"][0] == 0.5


def test_auc_is_nan_without_both_classes():
    metrics = evaluate({"a": True}, {"a": 0.9}, {"a": "high"})
    assert math.isnan(metrics.auc)
