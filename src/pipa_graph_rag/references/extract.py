"""조문 본문에서 다른 조문을 가리키는 표현을 정규식으로 뽑는다. 외부 호출이 없다."""

from __future__ import annotations

import re

from ..schema import Article, ArticleReference, Evidence, ReferenceKind

_REF = re.compile(r"제(?P<n>\d+)조(?:의(?P<b>\d+))?")
_HEADING = re.compile(r"^제\d+조(?:의\d+)?\(")

# 참조 사이를 잇는 표현. 이 표현만 사이에 있으면 같은 대상 법령의 참조가 이어지는 것으로 본다.
_JOINER = r"(?:제\d+항|제\d+호(?:의\d+)?|제\d+목|제\d+조(?:의\d+)?|각 호|부터|까지|및|또는|[,，]|\s)*"
_GAP_ONLY_JOINERS = re.compile(rf"^{_JOINER}$")
_MUTATIS_MUTANDIS = re.compile(
    rf"^{_JOINER}(?:의\s*규정)?(?:을|를)?\s*(?:각각\s*)?(?:그대로\s*)?준용"
)
_NOTWITHSTANDING = re.compile(rf"^{_JOINER}에도\s*불구하고")

_RANGE_END = re.compile(r"^(?:제\d+항)?\s*부터\s*제(?P<n>\d+)조(?:의(?P<b>\d+))?(?:제\d+항)?\s*까지")

_QUOTES = '"“”'
_OTHER_LAW_PREFIXES = ("같은 법 ", "같은 법 시행령 ", "시행령 ")
_QUOTE_PAD = 20


def _quoted_spans(line: str) -> list[tuple[int, int]]:
    spans, start = [], None
    for i, ch in enumerate(line):
        if ch in _QUOTES:
            if start is None:
                start = i
            else:
                spans.append((start, i))
                start = None
    return spans


def _in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(s < pos < e for s, e in spans)


def _order(article_id: str) -> tuple[int, int]:
    """`law:7-11`을 (7, 11)로, 가지번호가 없으면 (7, 0)으로 바꿔 조문 순서를 비교한다."""
    number, _, branch = article_id.partition(":")[2].partition("-")
    return int(number), int(branch or 0)


def _ids_between(first: str, last: str, known_ids: set[str]) -> list[str]:
    """같은 법령에서 first부터 last까지(양 끝 포함) 순서에 있는 조문."""
    law = first.partition(":")[0]
    low, high = _order(first), _order(last)
    return sorted(
        (i for i in known_ids if i.startswith(f"{law}:") and low <= _order(i) <= high),
        key=_order,
    )


def _target_law(article: Article, prefix: str, previous: str | None, gap: str | None) -> str | None:
    """참조가 가리키는 법령. 이 프로젝트 밖의 법령이면 None."""
    if previous is not None and gap is not None and _GAP_ONLY_JOINERS.match(gap):
        return None if previous == "other" else previous
    if prefix.rstrip().endswith("」"):
        return None
    if prefix.endswith(_OTHER_LAW_PREFIXES):
        return None
    if prefix.endswith(("이 영 ", "영 ")):
        return "decree"
    if prefix.endswith("법 "):
        return "law"
    return article.law


def _kind(article: Article, target_law: str, rest: str) -> ReferenceKind:
    if _MUTATIS_MUTANDIS.match(rest):
        return ReferenceKind.APPLIES_MUTATIS_MUTANDIS
    if _NOTWITHSTANDING.match(rest):
        return ReferenceKind.EXCEPTION_TO
    if article.law == "decree" and target_law == "law":
        return ReferenceKind.IMPLEMENTS
    return ReferenceKind.REFERS_TO


def extract_references(article: Article, known_ids: set[str]) -> list[ArticleReference]:
    """한 조문에서 참조를 뽑는다. 같은 (대상, 종류)는 첫 번째만 남긴다."""
    found: dict[tuple[str, ReferenceKind], ArticleReference] = {}
    for index, line in enumerate(article.text.split("\n")):
        if index == 0 and _HEADING.match(line):
            heading_end = line.find(")") + 1
        else:
            heading_end = 0
        spans = _quoted_spans(line)
        previous: str | None = None
        previous_end = 0
        for match in _REF.finditer(line, heading_end):
            if _in_spans(match.start(), spans):
                continue
            prefix = line[max(0, match.start() - 12) : match.start()]
            gap = line[previous_end : match.start()] if previous is not None else None
            law = _target_law(article, prefix, previous, gap)
            previous, previous_end = law or "other", match.end()
            if law is None:
                continue
            target = f"{law}:{match['n']}" + (f"-{match['b']}" if match["b"] else "")
            rest = line[match.end() :]
            targets = [target]
            range_end = _RANGE_END.match(rest)
            if range_end:
                last = f"{law}:{range_end['n']}" + (f"-{range_end['b']}" if range_end["b"] else "")
                targets = _ids_between(target, last, known_ids) or [target]
            kind = _kind(article, law, rest)
            quote_start = max(0, match.start() - _QUOTE_PAD)
            quote_end = min(len(line), match.end() + _QUOTE_PAD)
            quote = line[quote_start:quote_end].strip()
            for each in targets:
                if each not in known_ids or each == article.id:
                    continue
                found.setdefault(
                    (each, kind),
                    ArticleReference(
                        source_id=article.id,
                        target_id=each,
                        kind=kind,
                        evidence=Evidence(article_id=article.id, quote=quote),
                    ),
                )
    return list(found.values())
