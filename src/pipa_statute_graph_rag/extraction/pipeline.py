"""조문별 추출을 실행하고 결과를 JSONL로 이어서 저장한다. 중단했다가 다시 실행하면 이어서 한다."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from ..schema import Article, ConceptExtraction
from .gate import apply_gate
from .models import ExtractorResult, Rejection


class Extractor(Protocol):
    model: str
    prompt_version: str
    prompt_sha256: str

    def extract(self, article: Article) -> ExtractorResult: ...


class ExtractionRecord(BaseModel):
    article_id: str
    model: str
    prompt_version: str
    prompt_sha256: str | None = None
    extraction: ConceptExtraction
    proposed: int
    rejected: list[Rejection]
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None = None


def load_records(path: Path) -> list[ExtractionRecord]:
    if not path.exists():
        return []
    return [
        ExtractionRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _extract_one(article: Article, extractor: Extractor) -> ExtractionRecord:
    result = extractor.extract(article)
    gate = apply_gate(article, result.raw)
    return ExtractionRecord(
        article_id=article.id,
        model=extractor.model,
        prompt_version=f"extraction/{extractor.prompt_version}",
        prompt_sha256=extractor.prompt_sha256,
        extraction=gate.extraction,
        proposed=gate.proposed,
        rejected=gate.rejected,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        reasoning_tokens=result.reasoning_tokens,
    )


def run_extraction(
    articles: list[Article],
    extractor: Extractor,
    out_path: Path,
    *,
    workers: int = 4,
) -> tuple[int, list[tuple[str, str]]]:
    """아직 저장되지 않은 조문만 추출한다. (새로 저장한 수, 실패한 (조문, 오류) 목록)을 돌려준다."""
    done = {r.article_id for r in load_records(out_path)}
    todo = [a for a in articles if a.id not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved, failures = 0, []
    with out_path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(_extract_one, a, extractor): a for a in todo}
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
