"""법제처 OPEN API로 법령 본문을 받아 조문 단위로 정리한다."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from .schema import Article

BASE_URL = "https://www.law.go.kr/DRF"

# 검색 응답의 이 필드에 인증키가 들어 있다.
SECRET_FIELDS = frozenset({"법령상세링크"})


class LawSource(BaseModel):
    key: Literal["law", "decree"]
    name: str
    mst: str  # 법령일련번호


LAWS = (
    LawSource(key="law", name="개인정보 보호법", mst="283839"),
    LawSource(key="decree", name="개인정보 보호법 시행령", mst="289537"),
)


def strip_secrets(value: Any) -> Any:
    """응답에서 인증키가 들어 있는 필드를 재귀적으로 제거한다."""
    if isinstance(value, dict):
        return {k: strip_secrets(v) for k, v in value.items() if k not in SECRET_FIELDS}
    if isinstance(value, list):
        return [strip_secrets(v) for v in value]
    return value


def _get_json(endpoint: str, oc: str, params: dict[str, str]) -> dict[str, Any]:
    response = httpx.get(
        f"{BASE_URL}/{endpoint}",
        params={"OC": oc, "type": "JSON", **params},
        timeout=30,
    )
    response.raise_for_status()
    return strip_secrets(response.json())


def find_mst(name: str, oc: str) -> str:
    """현행 법령 이름으로 법령일련번호를 찾는다."""
    data = _get_json("lawSearch.do", oc, {"target": "law", "query": name})
    laws = data["LawSearch"].get("law", [])
    if isinstance(laws, dict):
        laws = [laws]
    for law in laws:
        if law["법령명한글"] == name and law["현행연혁코드"] == "현행":
            return law["법령일련번호"]
    raise LookupError(f"현행 법령을 찾지 못했다: {name}")


def fetch_law(mst: str, oc: str) -> dict[str, Any]:
    return _get_json("lawService.do", oc, {"target": "law", "MST": mst})


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]


def _flatten(value: Any) -> list[str]:
    """내용 필드는 문자열이거나 (중첩된) 문자열 목록이다."""
    if isinstance(value, str):
        return [value]
    return [text for v in _as_list(value) for text in _flatten(v)]


def _clause_lines(clause: dict[str, Any]) -> list[str]:
    lines = _flatten(clause.get("항내용"))
    for item in _as_list(clause.get("호")):
        lines.extend(_flatten(item.get("호내용")))
        for sub in _as_list(item.get("목")):
            lines.extend(_flatten(sub.get("목내용")))
    return lines


def _normalize(lines: list[str]) -> str:
    cleaned = []
    for line in lines:
        text = " ".join(line.split())
        if text:
            cleaned.append(text)
    return "\n".join(cleaned)


def parse_articles(raw: dict[str, Any], law_key: Literal["law", "decree"]) -> list[Article]:
    units = raw["법령"]["조문"]["조문단위"]
    articles = []
    for unit in units:
        if unit.get("조문여부") != "조문":
            continue
        number = int(unit["조문번호"])
        branch = int(unit["조문가지번호"]) if unit.get("조문가지번호") else None
        lines = _flatten(unit.get("조문내용"))
        for clause in _as_list(unit.get("항")):
            lines.extend(_clause_lines(clause))
        article_id = f"{law_key}:{number}" + (f"-{branch}" if branch else "")
        articles.append(
            Article(
                id=article_id,
                law=law_key,
                number=number,
                branch=branch,
                title=unit.get("조문제목", ""),
                text=_normalize(lines),
                effective_date=unit["조문시행일자"],
            )
        )
    return articles


def assert_no_secret(payload: str, oc: str) -> None:
    if oc and oc in payload:
        raise RuntimeError("저장하려는 내용에 인증키가 들어 있다")


def collect(out_dir: Path, oc: str | None = None) -> list[Article]:
    """본법과 시행령을 받아 out_dir/articles.json에 조문 단위로 저장한다."""
    oc = oc or os.environ["LAW_OC"]
    articles: list[Article] = []
    for law in LAWS:
        articles.extend(parse_articles(fetch_law(law.mst, oc), law.key))
    payload = json.dumps(
        [a.model_dump() for a in articles], ensure_ascii=False, indent=2
    )
    assert_no_secret(payload, oc)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "articles.json").write_text(payload + "\n", encoding="utf-8")
    return articles
