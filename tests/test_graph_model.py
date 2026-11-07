from pipa_statute_graph_rag.graph.model import build_graph, is_deleted
from pipa_statute_graph_rag.schema import (
    Article,
    ArticleReference,
    ConceptExtraction,
    ConceptRelation,
    ConceptRelationKind,
    Deontic,
    DeonticKind,
    Evidence,
    ProcessingActivity,
    ReferenceKind,
    Role,
    RoleKind,
)


def article(id: str, text: str | None = None) -> Article:
    law, rest = id.split(":")
    number, _, branch = rest.partition("-")
    heading = f"제{number}조" + (f"의{branch}" if branch else "")
    return Article(
        id=id,
        law=law,
        number=int(number),
        branch=int(branch) if branch else None,
        title="제목",
        text=text or f"{heading}(제목)\n① 본문이다.",
        effective_date="20260911",
    )


def reference(source: str, target: str, kind=ReferenceKind.REFERS_TO) -> ArticleReference:
    return ArticleReference(
        source_id=source,
        target_id=target,
        kind=kind,
        evidence=Evidence(article_id=source, quote="인용"),
    )


def extraction(article_id: str, role_kind=RoleKind.UNKNOWN, legal_basis=None) -> ConceptExtraction:
    role = Role(id="role:개인정보처리자", name="개인정보처리자", kind=role_kind)
    duty = Deontic(id="deontic:열람의무", name="열람 의무", kind=DeonticKind.OBLIGATION)
    return ConceptExtraction(
        article_id=article_id,
        entities=[role, duty],
        relations=[
            ConceptRelation(
                kind=ConceptRelationKind.BEARS,
                source_id=role.id,
                target_id=duty.id,
                evidence=Evidence(article_id=article_id, quote="열람하게 하여야 한다"),
            )
        ],
        is_legal_basis=legal_basis,
    )


def test_deleted_article_detection():
    assert is_deleted(article("law:8", "제8조 삭제 <2020.2.4>"))
    assert is_deleted(article("law:28-6", "제28조의6 삭제 <2023.3.14>"))
    assert not is_deleted(article("law:9"))


def test_deleted_articles_and_their_references_are_left_out():
    data = build_graph(
        [article("law:8", "제8조 삭제 <2020.2.4>"), article("law:9"), article("law:10")],
        [reference("law:9", "law:8"), reference("law:9", "law:10")],
        [],
    )
    assert [a["id"] for a in data.articles] == ["law:9", "law:10"]
    assert data.article_edges == {"REFERS_TO": [{"source": "law:9", "target": "law:10", "quote": "인용"}]}
    assert data.skipped == {"deleted_articles": 1, "references_to_deleted": 1}


def test_relation_types_are_upper_case_enum_values():
    data = build_graph(
        [article("law:9"), article("decree:3")],
        [reference("decree:3", "law:9", ReferenceKind.IMPLEMENTS)],
        [],
    )
    assert list(data.article_edges) == ["IMPLEMENTS"]


def test_same_entity_from_two_articles_becomes_one_node_with_two_mentions_and_edges():
    data = build_graph(
        [article("law:9"), article("law:10")],
        [],
        [extraction("law:9"), extraction("law:10")],
    )
    assert [e["id"] for e in data.entities["Role"]] == ["role:개인정보처리자"]
    assert len(data.mentions["Role"]) == 2
    assert [e["article_id"] for e in data.concept_edges["BEARS"]] == ["law:9", "law:10"]


def test_known_attribute_replaces_unknown_but_not_the_reverse():
    data = build_graph(
        [article("law:9"), article("law:10"), article("law:11")],
        [],
        [
            extraction("law:9", RoleKind.UNKNOWN),
            extraction("law:10", RoleKind.CONTROLLER),
            extraction("law:11", RoleKind.UNKNOWN),
        ],
    )
    assert data.entities["Role"][0]["kind"] == "controller"


def test_entity_rows_have_no_type_field_and_legal_basis_only_when_decided():
    data = build_graph(
        [article("law:9"), article("law:10")],
        [],
        [extraction("law:9", legal_basis=True), extraction("law:10")],
    )
    assert "type" not in data.entities["Role"][0]
    rows = {a["id"]: a for a in data.articles}
    assert rows["law:9"]["is_legal_basis"] is True
    assert "is_legal_basis" not in rows["law:10"]


def test_extraction_of_deleted_article_is_ignored():
    data = build_graph([article("law:8", "제8조 삭제 <2020.2.4>")], [], [extraction("law:8")])
    assert data.entities == {} and data.concept_edges == {}


def test_mentions_are_grouped_by_entity_label():
    ent = ProcessingActivity(id="processing_activity:수집", name="수집")
    ex = ConceptExtraction(
        article_id="law:9",
        entities=[ent, Role(id="role:a", name="a")],
        relations=[
            ConceptRelation(
                kind=ConceptRelationKind.PERFORMS,
                source_id="role:a",
                target_id=ent.id,
                evidence=Evidence(article_id="law:9", quote="수집한다"),
            )
        ],
    )
    data = build_graph([article("law:9")], [], [ex])
    assert set(data.mentions) == {"Role", "ProcessingActivity"}
