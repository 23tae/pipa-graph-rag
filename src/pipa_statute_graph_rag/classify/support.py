"""추출한 관계를 판정기에 맡기고, 판정 결과로 관계를 거른다."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel

from ..schema import Article, ConceptExtraction, ConceptRelationKind
from .base import SupportItem, SupportJudge

_STATEMENTS = {
    ConceptRelationKind.PERFORMS: "역할 '{s}'은(는) 처리 활동 '{t}'을(를) 수행한다.",
    ConceptRelationKind.ACTS_ON: "처리 활동 '{s}'의 대상은 개인정보 범주 '{t}'이다.",
    ConceptRelationKind.HAS_PURPOSE: "처리 활동 '{s}'은(는) 목적 '{t}'을(를) 위한 것이다.",
    ConceptRelationKind.BEARS: "역할 '{s}'은(는) 권리·의무 '{t}'의 주체이다.",
    ConceptRelationKind.COUNTERPARTY: "권리·의무 '{s}'의 상대방은 역할 '{t}'이다.",
    ConceptRelationKind.CONCERNS_ACTIVITY: "권리·의무 '{s}'은(는) 처리 활동 '{t}'에 관한 것이다.",
    ConceptRelationKind.CONCERNS_DATA: "권리·의무 '{s}'은(는) 개인정보 범주 '{t}'에 관한 것이다.",
    ConceptRelationKind.DESCRIBED_BY: "목적 '{s}'은(는) 이 조문에 서술되어 있다.",
}
assert set(_STATEMENTS) == set(ConceptRelationKind)


def build_items(extraction: ConceptExtraction) -> list[SupportItem]:
    """관계마다 판정할 주장(statement)과 인용을 만든다. 관계 순서와 같은 순서다."""
    names = {e.id: e.name for e in extraction.entities}
    return [
        SupportItem(
            id=f"r{index}",
            statement=_STATEMENTS[relation.kind].format(
                s=names.get(relation.source_id, relation.source_id),
                t=names.get(relation.target_id, relation.target_id),
            ),
            quote=relation.evidence.quote,
        )
        for index, relation in enumerate(extraction.relations)
    ]


class SupportRecord(BaseModel):
    article_id: str
    judge: str
    prompt_id: str
    prompt_sha256: str
    probabilities: list[float]  # extraction.relations와 같은 순서
    input_tokens: int


def load_support(path: Path) -> list[SupportRecord]:
    if not path.exists():
        return []
    return [
        SupportRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _judge_one(article: Article, extraction: ConceptExtraction, judge: SupportJudge) -> SupportRecord:
    result = judge.judge(article, build_items(extraction))
    return SupportRecord(
        article_id=article.id,
        judge=judge.name,
        prompt_id=judge.prompt_id,
        prompt_sha256=judge.prompt_sha256,
        probabilities=result.probabilities,
        input_tokens=result.input_tokens,
    )


def run_judgment(
    articles: list[Article],
    extractions: dict[str, ConceptExtraction],
    judge: SupportJudge,
    out_path: Path,
    *,
    workers: int = 4,
) -> tuple[int, list[tuple[str, str]]]:
    """관계가 있는 조문 중 아직 저장되지 않은 것만 판정한다. (새로 저장한 수, 실패 목록)."""
    done = {r.article_id for r in load_support(out_path)}
    todo = [a for a in articles if a.id not in done and extractions.get(a.id) and extractions[a.id].relations]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved, failures = 0, []
    with out_path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(_judge_one, a, extractions[a.id], judge): a for a in todo}
        for future in as_completed(futures):
            article = futures[future]
            try:
                record = future.result()
            except Exception as error:  # 한 조문의 실패가 나머지 실행을 막지 않게 한다
                failures.append((article.id, f"{type(error).__name__}: {error}"[:300]))
                continue
            out.write(record.model_dump_json() + "\n")
            out.flush()
            saved += 1
    return saved, failures


def filter_by_support(
    extractions: list[ConceptExtraction], records: list[SupportRecord], threshold: float
) -> list[ConceptExtraction]:
    """확률이 threshold 이상인 관계만 남기고, 남은 관계에 쓰이지 않는 개체는 뺀다."""
    probabilities = {r.article_id: r.probabilities for r in records}
    filtered = []
    for extraction in extractions:
        if not extraction.relations:
            filtered.append(extraction)
            continue
        scores = probabilities.get(extraction.article_id)
        if scores is None or len(scores) != len(extraction.relations):
            raise ValueError(f"{extraction.article_id}: 판정 결과가 없거나 관계 수와 다르다")
        kept = [r for r, p in zip(extraction.relations, scores, strict=True) if p >= threshold]
        used = {r.source_id for r in kept} | {r.target_id for r in kept}
        filtered.append(
            extraction.model_copy(
                update={
                    "relations": kept,
                    "entities": [e for e in extraction.entities if e.id in used],
                }
            )
        )
    return filtered
