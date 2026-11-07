import pytest
from pydantic import ValidationError

from pipa_statute_graph_rag.schema import (
    ArticleReference,
    ConceptExtraction,
    ConceptRelation,
    ConceptRelationKind,
    DataCategory,
    DataCategoryKind,
    Deontic,
    DeonticKind,
    Evidence,
    ReferenceKind,
    Role,
    RoleKind,
)


def evidence(article_id: str = "law:15", quote: str = "동의를 받은 경우") -> Evidence:
    return Evidence(article_id=article_id, quote=quote)


def test_entity_defaults_to_unknown_kind():
    assert Role(id="role:x", name="x").kind == RoleKind.UNKNOWN
    assert DataCategory(id="data_category:x", name="x").kind == DataCategoryKind.UNKNOWN


def test_evidence_requires_non_empty_quote():
    with pytest.raises(ValidationError):
        Evidence(article_id="law:15", quote="   ")


def test_article_reference_evidence_must_come_from_source():
    with pytest.raises(ValidationError):
        ArticleReference(
            source_id="decree:3",
            target_id="law:15",
            kind=ReferenceKind.IMPLEMENTS,
            evidence=evidence("law:15"),
        )


def test_extraction_parses_entities_by_type_discriminator():
    extraction = ConceptExtraction.model_validate(
        {
            "article_id": "law:15",
            "entities": [
                {"type": "role", "id": "role:개인정보처리자", "name": "개인정보처리자", "kind": "controller"},
                {"type": "processing_activity", "id": "processing_activity:수집", "name": "수집", "action": "수집"},
            ],
            "relations": [
                {
                    "kind": "performs",
                    "source_id": "role:개인정보처리자",
                    "target_id": "processing_activity:수집",
                    "evidence": {"article_id": "law:15", "quote": "수집할 수 있다"},
                }
            ],
        }
    )
    assert isinstance(extraction.entities[0], Role)


def test_extraction_rejects_wrong_endpoint_types():
    with pytest.raises(ValidationError):
        ConceptExtraction(
            article_id="law:15",
            entities=[
                Role(id="role:a", name="a"),
                DataCategory(id="data_category:b", name="b"),
            ],
            relations=[
                ConceptRelation(
                    kind=ConceptRelationKind.PERFORMS,
                    source_id="role:a",
                    target_id="data_category:b",
                    evidence=evidence(),
                )
            ],
        )


def test_extraction_rejects_unknown_endpoint():
    with pytest.raises(ValidationError):
        ConceptExtraction(
            article_id="law:15",
            entities=[Role(id="role:a", name="a")],
            relations=[
                ConceptRelation(
                    kind=ConceptRelationKind.BEARS,
                    source_id="role:a",
                    target_id="deontic:missing",
                    evidence=evidence(),
                )
            ],
        )


def test_extraction_rejects_evidence_from_other_article():
    with pytest.raises(ValidationError):
        ConceptExtraction(
            article_id="law:15",
            entities=[
                Role(id="role:a", name="a"),
                Deontic(id="deontic:b", name="b", kind=DeonticKind.OBLIGATION),
            ],
            relations=[
                ConceptRelation(
                    kind=ConceptRelationKind.BEARS,
                    source_id="role:a",
                    target_id="deontic:b",
                    evidence=evidence("law:17"),
                )
            ],
        )
