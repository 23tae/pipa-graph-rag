"""질문과 정답 조문으로 검색 설정을 비교한다."""

from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from statistics import mean

from pydantic import BaseModel

from .config import QUERY_TOP_K
from .query import QueryEngine
from .retrieval import Mode, Seed

DEPTHS = (1, 2)
SKIP_VALUES = (None, 5, 10, 20, 50)  # 개념 확장에서 건너뛸 개체의 조문 수 기준. None이면 건너뛰지 않는다


class Question(BaseModel):
    id: str
    type: str  # 단일, 위임, 예외, 준용
    question: str
    required: list[str]  # 답하려면 꼭 읽어야 하는 조문
    related: list[str]  # 있으면 좋은 조문


class Condition(BaseModel):
    mode: Mode
    depth: int
    max_entity_degree: int | None = None
    seed: Seed = Seed.BM25
    top_k: int = QUERY_TOP_K

    @property
    def name(self) -> str:
        """시드 검색이 BM25가 아니면 `:시드`를, 시드 수가 기본값이 아니면 `:k시드수`를 뒤에 붙인다."""
        if self.mode == Mode.KEYWORD:
            base = "keyword"
        else:
            skip = "" if self.max_entity_degree is None else f"/skip{self.max_entity_degree}"
            base = f"{self.mode.value}/d{self.depth}{skip}"
        seed = "" if self.seed == Seed.BM25 else f":{self.seed.value}"
        top_k = "" if self.top_k == QUERY_TOP_K else f":k{self.top_k}"
        return base + seed + top_k


class QuestionResult(BaseModel):
    question_id: str
    condition: str
    retrieved: list[str]
    recall_required: float
    recall_all: float  # 필수와 참고를 합친 정답 중 가져온 비율
    precision: float  # 가져온 조문 중 필수·참고 정답의 비율
    answer: str | None = None
    citations: list[str] | None = None
    citation_precision: float | None = None  # 인용한 조문 중 정답(필수·참고)의 비율
    citation_recall: float | None = None  # 필수 정답 중 인용한 비율
    input_tokens: int | None = None
    output_tokens: int | None = None


def conditions() -> list[Condition]:
    result = [Condition(mode=Mode.KEYWORD, depth=0)]
    result += [Condition(mode=Mode.REFERENCES, depth=d) for d in DEPTHS]
    result += [
        Condition(mode=Mode.REFERENCES_CONCEPTS, depth=d, max_entity_degree=skip)
        for d, skip in product(DEPTHS, SKIP_VALUES)
    ]
    return result


def with_seeds(base: list[Condition], seeds: list[Seed], top_ks: list[int]) -> list[Condition]:
    """각 조건을 시드 검색 방식과 시드 수의 모든 조합으로 늘린다."""
    return [c.model_copy(update={"seed": s, "top_k": k}) for c in base for s in seeds for k in top_ks]


def load_questions(path: Path) -> list[Question]:
    return [Question.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ratio(found: set[str], expected: set[str]) -> float:
    return len(found & expected) / len(expected) if expected else float("nan")


def evaluate_question(
    engine: QueryEngine, question: Question, condition: Condition, *, answer: bool
) -> QuestionResult:
    result = engine.run(
        question.question,
        mode=condition.mode,
        top_k=condition.top_k,
        depth=max(condition.depth, 1),
        max_entity_degree=condition.max_entity_degree,
        answer=answer,
        seed=condition.seed,
    )
    retrieved = result.retrieval.expansion.articles
    required, everything = set(question.required), set(question.required) | set(question.related)
    found = set(retrieved)
    row = QuestionResult(
        question_id=question.id,
        condition=condition.name,
        retrieved=retrieved,
        recall_required=_ratio(found, required),
        recall_all=_ratio(found, everything),
        precision=len(found & everything) / len(found) if found else 0.0,
    )
    if result.answer is not None:
        cited = set(result.answer.citations)
        row.answer = result.answer.answer
        row.citations = result.answer.citations
        row.citation_precision = len(cited & everything) / len(cited) if cited else 0.0
        row.citation_recall = _ratio(cited, required)
        row.input_tokens = result.answer.input_tokens
        row.output_tokens = result.answer.output_tokens
    return row


def evaluate_all(
    engine: QueryEngine,
    questions: list[Question],
    conds: list[Condition],
    *,
    answer: bool,
    workers: int = 4,
) -> list[QuestionResult]:
    jobs = [(q, c) for c in conds for q in questions]
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda job: evaluate_question(engine, job[0], job[1], answer=answer), jobs))


METRICS = ("recall_required", "recall_all", "precision", "citation_precision", "citation_recall")


def summarize(results: list[QuestionResult]) -> dict[str, dict[str, float]]:
    """조건별 평균. 크기는 가져온 조문 수의 평균이다."""
    by_condition: dict[str, list[QuestionResult]] = {}
    for r in results:
        by_condition.setdefault(r.condition, []).append(r)
    summary = {}
    for name, rows in by_condition.items():
        values = {"size": mean(len(r.retrieved) for r in rows)}
        for metric in METRICS:
            scores = [getattr(r, metric) for r in rows if getattr(r, metric) is not None]
            values[metric] = mean(scores) if scores else float("nan")
        summary[name] = values
    return summary


def summarize_by_type(
    results: list[QuestionResult], questions: list[Question], metric: str
) -> dict[str, dict[str, float]]:
    """조건별, 질문 유형별 지표 평균. {조건: {유형: 평균}}."""
    types = {q.id: q.type for q in questions}
    table: dict[str, dict[str, list[float]]] = {}
    for r in results:
        value = getattr(r, metric)
        if value is not None:
            table.setdefault(r.condition, {}).setdefault(types[r.question_id], []).append(value)
    return {c: {t: mean(v) for t, v in by_type.items()} for c, by_type in table.items()}


def format_summary(summary: dict[str, dict[str, float]]) -> str:
    header = "조건 | 평균 크기 | 재현율(필수) | 재현율(전체) | 정밀도 | 인용 정밀도 | 인용 재현율"
    lines = [header]
    for name, v in summary.items():
        cells = [f"{v['size']:.1f}"] + [
            "-" if math.isnan(v[m]) else f"{v[m]:.3f}" for m in METRICS
        ]
        lines.append(f"{name} | " + " | ".join(cells))
    return "\n".join(lines)


def bootstrap_difference(
    a: list[float], b: list[float], *, draws: int = 2000, seed: int = 20261107
) -> tuple[float, float, float]:
    """같은 질문 순서의 두 점수 목록에서 b − a 평균의 (값, 95% 구간 아래, 위). 질문을 복원 추출한다."""
    if len(a) != len(b):
        raise ValueError("점수 수가 다르다")
    rng = random.Random(seed)
    diffs = sorted(
        mean(b[i] - a[i] for i in idx)
        for idx in ([rng.randrange(len(a)) for _ in a] for _ in range(draws))
    )
    return mean(y - x for x, y in zip(a, b, strict=True)), diffs[int(0.025 * draws)], diffs[int(0.975 * draws) - 1]
