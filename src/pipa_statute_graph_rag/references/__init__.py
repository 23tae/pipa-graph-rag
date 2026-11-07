"""조문 참조(참조, 준용, 예외, 위임)를 추출해 저장한다."""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import Article, ArticleReference
from .delegation import derive_delegations
from .extract import extract_references


def build_references(articles: list[Article], *, quoted_only: bool = False) -> list[ArticleReference]:
    by_id = {a.id: a for a in articles}
    references = [r for a in articles for r in extract_references(a, set(by_id))]
    return references + derive_delegations(references, by_id, quoted_only=quoted_only)


def load_references(path: Path) -> list[ArticleReference]:
    return [ArticleReference.model_validate(r) for r in json.loads(path.read_text(encoding="utf-8"))]


def save_references(references: list[ArticleReference], path: Path) -> None:
    payload = json.dumps([r.model_dump(mode="json") for r in references], ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")
