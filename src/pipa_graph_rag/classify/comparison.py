"""두 판정기를 라벨이 붙은 관계 표본에서 비교한다."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

from pydantic import BaseModel

from ..schema import Article, ConceptExtraction
from .base import SupportItem, SupportJudge
from .support import SupportRecord, build_items

LOW, HIGH = 0.1, 0.9  # 표본을 나누는 Decisions 확률 경계
STRATA = ("low", "mid", "high")
THRESHOLDS = (0.1, 0.5)
BINS = ((0.0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.01))


class Relation(BaseModel):
    key: str  # <조문 ID>#<관계 순번>
    article_id: str
    statement: str
    quote: str


class Label(BaseModel):
    key: str
    supported: bool
    reason: str


class Judgment(BaseModel):
    key: str
    judge: str
    probability: float
    input_tokens: int  # 이 관계가 속한 조문 호출 전체의 값
    seconds: float  # 이 관계가 속한 조문 호출 전체의 걸린 시간


def flatten(extractions: list[ConceptExtraction]) -> list[Relation]:
    relations = []
    for extraction in extractions:
        for index, item in enumerate(build_items(extraction)):
            relations.append(
                Relation(
                    key=f"{extraction.article_id}#{index}",
                    article_id=extraction.article_id,
                    statement=item.statement,
                    quote=item.quote,
                )
            )
    return relations


def probabilities_by_key(extractions: list[ConceptExtraction], records: list[SupportRecord]) -> dict[str, float]:
    by_article = {r.article_id: r.probabilities for r in records}
    result = {}
    for extraction in extractions:
        scores = by_article.get(extraction.article_id, [])
        if len(scores) != len(extraction.relations):
            raise ValueError(f"{extraction.article_id}: 판정 결과가 없거나 관계 수와 다르다")
        for index, score in enumerate(scores):
            result[f"{extraction.article_id}#{index}"] = score
    return result


def stratum(probability: float) -> str:
    if probability < LOW:
        return "low"
    return "high" if probability >= HIGH else "mid"


def stratified_sample(
    relations: list[Relation], probabilities: dict[str, float], per_stratum: int, seed: int
) -> tuple[list[Relation], dict[str, str]]:
    """확률 구간마다 per_stratum개를 무작위로 뽑는다. 표본은 구간을 알 수 없게 섞어서 돌려주고,
    구간은 따로 돌려준다."""
    rng = random.Random(seed)
    pools: dict[str, list[Relation]] = {s: [] for s in STRATA}
    for relation in sorted(relations, key=lambda r: r.key):
        pools[stratum(probabilities[relation.key])].append(relation)
    picked: list[Relation] = []
    for name in STRATA:
        if len(pools[name]) < per_stratum:
            raise ValueError(f"{name} 구간에 관계가 {len(pools[name])}개뿐이다")
        picked.extend(rng.sample(pools[name], per_stratum))
    rng.shuffle(picked)
    return picked, {r.key: stratum(probabilities[r.key]) for r in picked}


def run_judge(
    judge: SupportJudge, articles: dict[str, Article], sample: list[Relation]
) -> list[Judgment]:
    """표본의 관계를 조문별로 묶어 판정기를 부른다."""
    by_article: dict[str, list[Relation]] = {}
    for relation in sample:
        by_article.setdefault(relation.article_id, []).append(relation)
    judgments = []
    for article_id, relations in sorted(by_article.items()):
        items = [SupportItem(id=r.key, statement=r.statement, quote=r.quote) for r in relations]
        start = time.perf_counter()
        result = judge.judge(articles[article_id], items)
        seconds = time.perf_counter() - start
        for relation, probability in zip(relations, result.probabilities, strict=True):
            judgments.append(
                Judgment(
                    key=relation.key,
                    judge=judge.name,
                    probability=probability,
                    input_tokens=result.input_tokens,
                    seconds=seconds,
                )
            )
    return judgments


class Metrics(BaseModel):
    count: int
    auc: float
    by_threshold: dict[float, dict[str, float]]  # 기준값 → accuracy, precision, recall
    by_stratum: dict[str, dict[float, float]]  # 구간 → 기준값 → accuracy
    by_bin: list[tuple[float, float, int, float | None]]  # 확률 구간, 관계 수, 뒷받침 비율


def evaluate(labels: dict[str, bool], probabilities: dict[str, float], strata: dict[str, str]) -> Metrics:
    keys = sorted(labels)
    by_threshold = {}
    for t in THRESHOLDS:
        tp = sum(labels[k] and probabilities[k] >= t for k in keys)
        fp = sum(not labels[k] and probabilities[k] >= t for k in keys)
        fn = sum(labels[k] and probabilities[k] < t for k in keys)
        tn = len(keys) - tp - fp - fn
        by_threshold[t] = {
            "accuracy": (tp + tn) / len(keys),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
        }
    by_stratum = {}
    for name in STRATA:
        members = [k for k in keys if strata[k] == name]
        if not members:
            continue
        by_stratum[name] = {
            t: sum((probabilities[k] >= t) == labels[k] for k in members) / len(members) for t in THRESHOLDS
        }
    by_bin = []
    for low, high in BINS:
        members = [k for k in keys if low <= probabilities[k] < high]
        rate = sum(labels[k] for k in members) / len(members) if members else None
        by_bin.append((low, high, len(members), rate))
    return Metrics(
        count=len(keys),
        auc=_auc(labels, probabilities),
        by_threshold=by_threshold,
        by_stratum=by_stratum,
        by_bin=by_bin,
    )


def _auc(labels: dict[str, bool], probabilities: dict[str, float]) -> float:
    return _auc_of([(labels[k], probabilities[k]) for k in labels])


def _auc_of(pairs: list[tuple[bool, float]]) -> float:
    """뒷받침되는 관계가 뒷받침되지 않는 관계보다 확률이 높은 쌍의 비율(같으면 0.5)."""
    positives = [p for label, p in pairs if label]
    negatives = [p for label, p in pairs if not label]
    if not positives or not negatives:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def _accuracy_of(pairs: list[tuple[bool, float]], threshold: float) -> float:
    return sum((p >= threshold) == label for label, p in pairs) / len(pairs)


def compare_judges(
    labels: dict[str, bool],
    probabilities_a: dict[str, float],
    probabilities_b: dict[str, float],
    *,
    draws: int = 2000,
    seed: int = 20261107,
) -> dict[str, tuple[float, float, float]]:
    """지표별 b − a의 (표본 값, 95% 구간 아래, 위). 관계를 복원 추출해 다시 뽑고(부트스트랩) 두 판정기를
    같은 표본으로 잰다. 구간이 0을 포함하면 표본 90개로는 차이를 구분하지 못한다."""
    keys = sorted(labels)
    pairs_a = [(labels[k], probabilities_a[k]) for k in keys]
    pairs_b = [(labels[k], probabilities_b[k]) for k in keys]
    metrics = {
        "AUC": _auc_of,
        **{f"정확도 {t}": (lambda pairs, t=t: _accuracy_of(pairs, t)) for t in THRESHOLDS},
    }
    rng = random.Random(seed)
    resamples = [[rng.randrange(len(keys)) for _ in keys] for _ in range(draws)]
    result = {}
    for name, metric in metrics.items():
        diffs = sorted(
            d
            for d in (
                metric([pairs_b[i] for i in idx]) - metric([pairs_a[i] for i in idx]) for idx in resamples
            )
            if not math.isnan(d)
        )
        point = metric(pairs_b) - metric(pairs_a)
        result[name] = (point, diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1])
    return result


def format_difference(name_a: str, name_b: str, difference: dict[str, tuple[float, float, float]]) -> str:
    lines = [f"## {name_b} − {name_a} (표본 90개 부트스트랩 95% 구간)"]
    for metric, (point, low, high) in difference.items():
        lines.append(f"  {metric}: {point:+.3f} [{low:+.3f}, {high:+.3f}]")
    return "\n".join(lines)


def cost(judgments: list[Judgment]) -> tuple[int, float, int]:
    """(입력 토큰, 걸린 시간 합, 호출 수). 같은 조문의 관계는 한 호출이라 조문마다 한 번만 센다."""
    calls = {j.key.split("#")[0]: j for j in judgments}
    return (
        sum(j.input_tokens for j in calls.values()),
        sum(j.seconds for j in calls.values()),
        len(calls),
    )


def format_metrics(name: str, metrics: Metrics, judgments: list[Judgment]) -> str:
    tokens, seconds, calls = cost(judgments)
    lines = [f"## {name} (관계 {metrics.count}개, AUC {metrics.auc:.3f})"]
    for t, values in metrics.by_threshold.items():
        lines.append(
            f"  기준 {t}: 정확도 {values['accuracy']:.3f} 정밀도 {values['precision']:.3f} "
            f"재현율 {values['recall']:.3f}"
        )
    for stratum_name, values in metrics.by_stratum.items():
        text = ", ".join(f"기준 {t} {v:.3f}" for t, v in values.items())
        lines.append(f"  Decisions 구간 {stratum_name} 정확도: {text}")
    for low, high, count, rate in metrics.by_bin:
        shown = "-" if rate is None else f"{rate:.3f}"
        lines.append(f"  확률 {low:.1f}~{min(high, 1.0):.1f}: 관계 {count}개, 뒷받침 비율 {shown}")
    lines.append(f"  호출 {calls}회, 입력 토큰 {tokens}, 걸린 시간 합 {seconds:.1f}초")
    return "\n".join(lines)


def read_jsonl(path: Path, model: type[BaseModel]) -> list:
    return [model.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
