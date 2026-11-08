"""법제처 OPEN API로 본법과 시행령 본문을 받아 조문 단위로 저장한다."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..schema import Article
from .client import fetch_law
from .parser import assert_no_secret, parse_articles


class LawSource(BaseModel):
    key: Literal["law", "decree"]
    name: str
    mst: str  # 법령일련번호


LAWS = (
    LawSource(key="law", name="개인정보 보호법", mst="283839"),
    LawSource(key="decree", name="개인정보 보호법 시행령", mst="289537"),
)


def collect(out_dir: Path, oc: str | None = None) -> list[Article]:
    """본법과 시행령을 받아 out_dir/articles.json에 조문 단위로 저장한다."""
    oc = oc or os.environ["LAW_OC"]
    articles: list[Article] = []
    for law in LAWS:
        articles.extend(parse_articles(fetch_law(law.mst, oc), law.key))
    payload = json.dumps([a.model_dump() for a in articles], ensure_ascii=False, indent=2)
    assert_no_secret(payload, oc)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "articles.json").write_text(payload + "\n", encoding="utf-8")
    return articles


def load_articles(path: Path) -> list[Article]:
    return [Article.model_validate(a) for a in json.loads(path.read_text(encoding="utf-8"))]
