"""시행령 → 본법 `implements`를 뒤집어 본법 → 시행령 `delegates_to`를 만든다."""

from __future__ import annotations

import re
from collections import Counter

from ..schema import Article, ArticleReference, Evidence, ReferenceKind

_MARKER = "대통령령"
_QUOTED_MARKER = re.compile(r'"[^"]*대통령령[^"]*"')
_PAD = 30
_CIRCLED = {chr(0x2460 + i): i + 1 for i in range(20)}

# 본법 조문을 이만큼 넘게 인용하는 시행령 조문(업무 위탁처럼 여러 조문을 나열하는 조문)은 위임으로 보지 않는다.
# 표본에서 본 두 조문(인용 13개, 10개)을 걸러내려고 정한 값이고, 나머지 조문은 7개 이하다.
MAX_IMPLEMENTED_LAW_ARTICLES = 8


def _clauses(article: Article) -> dict[int, str]:
    """본문을 항 번호별 텍스트로 나눈다. 항 표시가 없는 줄은 앞선 항에 붙고, 항이 없으면 1항으로 본다."""
    clauses: dict[int, list[str]] = {}
    current = 1
    for line in article.text.split("\n")[1:] or [article.text]:
        if line[:1] in _CIRCLED:
            current = _CIRCLED[line[0]]
        clauses.setdefault(current, []).append(line)
    return {number: "\n".join(lines) for number, lines in clauses.items()}


def _cited_clauses(decree: Article, law: Article) -> set[int | None]:
    """시행령이 본법 조문을 인용한 항 번호. 항 없이 조문만 인용한 곳이 있으면 None이 들어간다."""
    branch = rf"의{law.branch}" if law.branch else ""
    pattern = re.compile(rf"법 제{law.number}조{branch}(?!의\d)(?:제(\d+)항)?")
    return {int(m[1]) if m[1] else None for m in pattern.finditer(decree.text)}


def _delegating_clause(decree: Article, law: Article) -> str | None:
    """시행령이 인용한 본법 항 중 "대통령령"이 있는 항의 줄. 없으면 None."""
    clauses = _clauses(law)
    cited = _cited_clauses(decree, law)
    candidates = clauses.values() if None in cited else (clauses.get(n, "") for n in cited)
    for text in candidates:
        for line in text.split("\n"):
            if _MARKER in line:
                at = line.index(_MARKER)
                return line[max(0, at - _PAD) : at + len(_MARKER) + _PAD].strip()
    return None


def derive_delegations(
    references: list[ArticleReference],
    articles: dict[str, Article],
    *,
    quoted_only: bool = False,
) -> list[ArticleReference]:
    """시행령 조문이 인용한 본법 항에 "대통령령"이 있으면 위임으로 본다.

    시행령이 `법 제N조제M항`처럼 항을 지정하면 그 항에 "대통령령"이 있어야 한다.
    quoted_only를 켜면 시행령 조문이 `"대통령령으로 정하는 …"`을 따옴표로 인용한 경우만 만든다.
    """
    implemented = Counter(r.source_id for r in references if r.kind == ReferenceKind.IMPLEMENTS)
    derived: dict[tuple[str, str], ArticleReference] = {}
    for ref in references:
        if ref.kind != ReferenceKind.IMPLEMENTS:
            continue
        if implemented[ref.source_id] > MAX_IMPLEMENTED_LAW_ARTICLES:
            continue
        law, decree = articles[ref.target_id], articles[ref.source_id]
        if quoted_only and not _QUOTED_MARKER.search(decree.text):
            continue
        quote = _delegating_clause(decree, law)
        if quote is None:
            continue
        derived.setdefault(
            (law.id, decree.id),
            ArticleReference(
                source_id=law.id,
                target_id=decree.id,
                kind=ReferenceKind.DELEGATES_TO,
                evidence=Evidence(article_id=law.id, quote=quote),
            ),
        )
    return list(derived.values())
