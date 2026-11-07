"""법제처 응답을 조문 단위로 정리한다. 네트워크를 쓰지 않는다."""

from __future__ import annotations

from typing import Any, Literal

from ..schema import Article

# 검색 응답의 이 필드에 인증키가 들어 있다.
SECRET_FIELDS = frozenset({"법령상세링크"})


def strip_secrets(value: Any) -> Any:
    """응답에서 인증키가 들어 있는 필드를 재귀적으로 제거한다."""
    if isinstance(value, dict):
        return {k: strip_secrets(v) for k, v in value.items() if k not in SECRET_FIELDS}
    if isinstance(value, list):
        return [strip_secrets(v) for v in value]
    return value


def assert_no_secret(payload: str, oc: str) -> None:
    if oc and oc in payload:
        raise RuntimeError("저장하려는 내용에 인증키가 들어 있다")


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
