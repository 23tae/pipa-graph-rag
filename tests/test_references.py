from pipa_statute_graph_rag.references import build_references
from pipa_statute_graph_rag.references.extract import extract_references
from pipa_statute_graph_rag.schema import Article, ReferenceKind

# 실제 법령 문구가 아니라 표현 형태만 본뜬 합성 문장이다.


def article(id: str, text: str, title: str = "제목") -> Article:
    law, rest = id.split(":")
    number, _, branch = rest.partition("-")
    heading = f"제{number}조" + (f"의{branch}" if branch else "") + f"({title})"
    return Article(
        id=id,
        law=law,
        number=int(number),
        branch=int(branch) if branch else None,
        title=title,
        text=heading + "\n" + text,
        effective_date="20260911",
    )


KNOWN = {"law:2", "law:3", "law:4-2", "law:24", "decree:3", "decree:19", "decree:30"}


def kinds(art: Article) -> set[tuple[str, ReferenceKind]]:
    return {(r.target_id, r.kind) for r in extract_references(art, KNOWN)}


def test_bare_reference_stays_in_same_law():
    assert kinds(article("law:3", "① 제2조제1항에 따른다.")) == {("law:2", ReferenceKind.REFERS_TO)}


def test_branch_number_becomes_hyphen_id():
    assert kinds(article("law:3", "제4조의2를 본다.")) == {("law:4-2", ReferenceKind.REFERS_TO)}


def test_decree_reference_to_law_is_implements():
    result = kinds(article("decree:19", '① 법 제24조제1항에서 "대통령령으로 정하는 정보"란 다음과 같다.'))
    assert result == {("law:24", ReferenceKind.IMPLEMENTS)}


def test_list_after_law_marker_keeps_target_law():
    result = kinds(article("decree:3", "① 법 제2조 및 제3조에 따른다."))
    assert result == {("law:2", ReferenceKind.IMPLEMENTS), ("law:3", ReferenceKind.IMPLEMENTS)}


def test_decree_bare_reference_targets_decree():
    assert kinds(article("decree:19", "① 제3조에 따른다.")) == {("decree:3", ReferenceKind.REFERS_TO)}


def test_other_law_reference_is_skipped():
    assert kinds(article("law:3", "① 「다른법」 제2조 및 제3조에 따른다.")) == set()


def test_quoted_replacement_text_is_skipped():
    text = '① 제30조를 준용한다. 이 경우 "법 제2조"는 "법 제24조"로 본다.'
    assert kinds(article("decree:19", text)) == {("decree:30", ReferenceKind.APPLIES_MUTATIS_MUTANDIS)}


def test_mutatis_mutandis_applies_only_to_the_applied_provision():
    text = "① 법 제24조에 따른 조치에 관하여는 제30조를 준용한다."
    assert kinds(article("decree:19", text)) == {
        ("law:24", ReferenceKind.IMPLEMENTS),
        ("decree:30", ReferenceKind.APPLIES_MUTATIS_MUTANDIS),
    }


def test_notwithstanding_is_exception():
    assert kinds(article("law:3", "① 제2조에도 불구하고 할 수 있다.")) == {
        ("law:2", ReferenceKind.EXCEPTION_TO)
    }


def test_range_includes_articles_between_both_ends():
    known = {"law:2", "law:3", "law:3-2", "law:4", "law:5", "law:30"}
    art = article("law:30", "① 제2조부터 제4조까지를 준용한다.")
    result = {(r.target_id, r.kind) for r in extract_references(art, known)}
    assert result == {
        (target, ReferenceKind.APPLIES_MUTATIS_MUTANDIS)
        for target in ("law:2", "law:3", "law:3-2", "law:4")
    }


def test_own_heading_and_unknown_targets_are_skipped():
    assert kinds(article("law:3", "① 제99조를 본다.")) == set()


def test_evidence_quote_is_a_substring_of_the_article():
    art = article("decree:19", "① 법 제24조제1항에 따른다.")
    quote = extract_references(art, KNOWN)[0].evidence.quote
    assert quote in art.text


def test_delegation_is_derived_from_implements_when_law_mentions_decree():
    law = article("law:24", "① 대통령령으로 정하는 정보를 처리할 수 없다.")
    decree = article("decree:19", '① 법 제24조에서 "대통령령으로 정하는 정보"란 다음과 같다.')
    refs = build_references([law, decree])
    delegation = [r for r in refs if r.kind == ReferenceKind.DELEGATES_TO]
    assert [(r.source_id, r.target_id) for r in delegation] == [("law:24", "decree:19")]
    assert "대통령령" in delegation[0].evidence.quote


def test_no_delegation_when_law_does_not_mention_decree():
    law = article("law:24", "① 처리할 수 없다.")
    decree = article("decree:19", "① 법 제24조에 따른다.")
    assert all(r.kind != ReferenceKind.DELEGATES_TO for r in build_references([law, decree]))


def test_delegation_requires_the_cited_clause_to_mention_decree():
    law = article("law:24", "① 처리할 수 없다.\n② 대통령령으로 정한다.")
    cites_first = article("decree:19", "① 법 제24조제1항에 따른다.")
    cites_second = article("decree:3", "① 법 제24조제2항에 따른다.")
    refs = build_references([law, cites_first, cites_second])
    pairs = [(r.source_id, r.target_id) for r in refs if r.kind == ReferenceKind.DELEGATES_TO]
    assert pairs == [("law:24", "decree:3")]


def test_decree_citing_many_law_articles_is_not_delegation():
    laws = [article(f"law:{n}", "① 대통령령으로 정한다.") for n in range(2, 12)]
    listing = article("decree:19", "① " + " ".join(f"법 제{n}조에 따른다." for n in range(2, 12)))
    refs = build_references(laws + [listing])
    assert all(r.kind != ReferenceKind.DELEGATES_TO for r in refs)


def test_quoted_only_drops_delegation_without_quoted_phrase():
    law = article("law:24", "① 대통령령으로 정하는 정보를 처리할 수 없다.")
    decree = article("decree:19", "① 법 제24조에 따른다.")
    refs = build_references([law, decree], quoted_only=True)
    assert all(r.kind != ReferenceKind.DELEGATES_TO for r in refs)
