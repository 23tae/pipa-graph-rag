from pipa_statute_graph_rag.extraction.gate import apply_gate, quote_in_article
from pipa_statute_graph_rag.extraction.models import RawExtraction, RawRelation
from pipa_statute_graph_rag.schema import (
    Article,
    ConceptRelationKind,
    Deontic,
    DeonticKind,
    ProcessingActivity,
    Role,
)

# 실제 법령 문구가 아니라 형태만 본뜬 합성 조문이다.
ARTICLE = Article(
    id="law:99",
    law="law",
    number=99,
    title="가나다",
    text="제99조(가나다)\n① 개인정보처리자는 정보주체의 열람 요구를\n받으면 열람하게 하여야 한다.",
    effective_date="20260911",
)

ROLE = Role(id="r1", name="개인정보처리자")
ACTIVITY = ProcessingActivity(id="a1", name="열람")
DEONTIC = Deontic(id="d1", name="열람 의무", kind=DeonticKind.OBLIGATION)


def relation(kind, source, target, quote="열람하게 하여야 한다"):
    return RawRelation(kind=kind, source_id=source, target_id=target, quote=quote)


def test_quote_matching_ignores_whitespace_and_line_breaks():
    assert quote_in_article("열람 요구를 받으면", ARTICLE)
    assert not quote_in_article("삭제 요구를 받으면", ARTICLE)
    assert not quote_in_article("   ", ARTICLE)


def test_valid_relation_passes_and_entity_ids_become_canonical():
    raw = RawExtraction(
        entities=[ROLE, DEONTIC],
        relations=[relation(ConceptRelationKind.BEARS, "r1", "d1")],
    )
    result = apply_gate(ARTICLE, raw)
    assert result.accepted == 1 and result.rejected == []
    assert result.extraction.relations[0].source_id == "role:개인정보처리자"
    assert {e.id for e in result.extraction.entities} == {"role:개인정보처리자", "deontic:열람의무"}


def test_one_bad_relation_does_not_discard_the_others():
    raw = RawExtraction(
        entities=[ROLE, ACTIVITY, DEONTIC],
        relations=[
            relation(ConceptRelationKind.BEARS, "r1", "d1"),
            relation(ConceptRelationKind.COUNTERPARTY, "r1", "r1"),
        ],
    )
    result = apply_gate(ARTICLE, raw)
    assert (result.proposed, result.accepted, len(result.rejected)) == (2, 1, 1)
    assert "deontic→role" in result.rejected[0].reason


def test_relation_with_quote_not_in_article_is_rejected():
    raw = RawExtraction(
        entities=[ROLE, ACTIVITY],
        relations=[relation(ConceptRelationKind.PERFORMS, "r1", "a1", quote="조문에 없는 문장")],
    )
    result = apply_gate(ARTICLE, raw)
    assert result.accepted == 0
    assert result.rejected[0].reason == "인용이 조문 본문에 없다"


def test_relation_with_unknown_entity_is_rejected():
    raw = RawExtraction(
        entities=[ROLE],
        relations=[relation(ConceptRelationKind.PERFORMS, "r1", "missing")],
    )
    assert apply_gate(ARTICLE, raw).accepted == 0


def test_entities_without_accepted_relation_are_dropped():
    raw = RawExtraction(entities=[ROLE, ACTIVITY], relations=[])
    assert apply_gate(ARTICLE, raw).extraction.entities == []


def test_described_by_must_point_to_the_article():
    purpose = {"type": "purpose", "id": "p1", "name": "열람 목적"}
    ok = RawExtraction.model_validate(
        {
            "entities": [purpose],
            "relations": [
                {"kind": "described_by", "source_id": "p1", "target_id": "law:99", "quote": "열람하게 하여야 한다"}
            ],
        }
    )
    bad = ok.model_copy(
        update={"relations": [ok.relations[0].model_copy(update={"target_id": "law:1"})]}
    )
    assert apply_gate(ARTICLE, ok).accepted == 1
    assert apply_gate(ARTICLE, bad).accepted == 0


def test_is_legal_basis_passes_through():
    assert apply_gate(ARTICLE, RawExtraction(is_legal_basis=True)).extraction.is_legal_basis is True
    assert apply_gate(ARTICLE, RawExtraction()).extraction.is_legal_basis is None
