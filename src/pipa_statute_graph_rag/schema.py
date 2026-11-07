"""PrOnto 다섯 모듈에 맞춘 개체와 관계 정의.

모듈 대응: 문서=Article, 데이터=DataCategory, 행위자와 역할=Role,
처리=ProcessingActivity, 목적=Purpose, 의무 연산자=Deontic.
처리 실행, 위반·준수 판정, 유럽연합 고유 개념은 정의하지 않는다.
근거를 찾지 못한 값은 UNKNOWN으로 둔다.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Article(BaseModel):
    """조문 하나(PrOnto의 문서). 장·절 제목은 포함하지 않는다."""

    id: str  # 예: law:7-11 (본법 제7조의11), decree:17 (시행령 제17조)
    law: Literal["law", "decree"]
    number: int
    branch: int | None = None  # 가지번호. 제7조의11이면 11
    title: str
    text: str  # 조문 제목과 항·호·목을 이어 붙인 전체 본문
    effective_date: str

    @property
    def label(self) -> str:
        suffix = f"의{self.branch}" if self.branch else ""
        return f"제{self.number}조{suffix}"


class Evidence(BaseModel):
    """관계의 출처. 인용이 조문에 실제로 있는지는 추출 단계에서 검증한다."""

    article_id: NonEmptyStr
    quote: NonEmptyStr


class DataCategoryKind(StrEnum):
    PERSONAL = "personal"  # 개인정보
    PSEUDONYMIZED = "pseudonymized"  # 가명정보
    ANONYMOUS = "anonymous"  # 익명
    SENSITIVE = "sensitive"  # 민감정보
    UNIQUE_IDENTIFIER = "unique_identifier"  # 고유식별정보 (한국 고유)
    RESIDENT_NUMBER = "resident_number"  # 주민등록번호 (한국 고유)
    OTHER = "other"
    UNKNOWN = "unknown"


class RoleKind(StrEnum):
    CONTROLLER = "controller"  # 개인정보처리자
    PROCESSOR = "processor"  # 수탁자
    DATA_SUBJECT = "data_subject"  # 정보주체
    THIRD_PARTY = "third_party"  # 제3자
    RECIPIENT = "recipient"  # 제공받는 자
    DPO = "dpo"  # 개인정보 보호책임자
    REPRESENTATIVE = "representative"  # 국내대리인
    SUPERVISORY_AUTHORITY = "supervisory_authority"  # 개인정보 보호위원회
    HANDLER = "handler"  # 개인정보취급자 (한국 고유)
    PUBLIC_INSTITUTION = "public_institution"  # 공공기관 (한국 고유)
    OTHER = "other"
    UNKNOWN = "unknown"


class ProcessingAction(StrEnum):
    """제2조 2호가 정의한 처리의 유형."""

    COLLECT = "수집"
    GENERATE = "생성"
    LINK = "연계"
    INTERWORK = "연동"
    RECORD = "기록"
    STORE = "저장"
    RETAIN = "보유"
    PROCESS = "가공"
    EDIT = "편집"
    SEARCH = "검색"
    OUTPUT = "출력"
    CORRECT = "정정"
    RESTORE = "복구"
    USE = "이용"
    PROVIDE = "제공"
    DISCLOSE = "공개"
    DESTROY = "파기"
    PSEUDONYMIZE = "가명처리"
    OTHER = "기타"
    UNKNOWN = "unknown"


class PurposeCategory(StrEnum):
    """제28조의2·3의 가명정보 처리 목적만 분류한다. 나머지는 UNKNOWN."""

    STATISTICS = "statistics"
    RESEARCH = "research"
    PUBLIC_INTEREST_ARCHIVING = "public_interest_archiving"
    OTHER = "other"
    UNKNOWN = "unknown"


class DeonticKind(StrEnum):
    RIGHT = "right"
    OBLIGATION = "obligation"
    PERMISSION = "permission"
    PROHIBITION = "prohibition"
    SANCTION = "sanction"


class DeonticSubtype(StrEnum):
    """PrOnto의 권리·의무 하위 분류와 한국 고유 의무 유형."""

    ACCESS = "access"  # 열람
    RECTIFICATION = "rectification"  # 정정
    ERASURE = "erasure"  # 삭제·파기
    RESTRICT_PROCESSING = "restrict_processing"  # 처리정지
    PORTABILITY = "portability"  # 전송 요구
    AUTOMATED_DECISION = "automated_decision"  # 자동화된 결정
    INFORM = "inform"  # 통지
    NOTIFY_BREACH = "notify_breach"  # 유출 통지·신고
    PREPARE_DOCUMENT = "prepare_document"  # 처리방침, 파일 등록
    SAFEGUARD = "safeguard"  # 안전조치 (한국 고유)
    IMPACT_ASSESSMENT = "impact_assessment"  # 영향평가 (한국 고유)
    DESIGNATE_DPO = "designate_dpo"  # 보호책임자 지정 (한국 고유)
    DESIGNATE_REPRESENTATIVE = "designate_representative"  # 국내대리인 지정 (한국 고유)
    SUPERVISE_PROCESSOR = "supervise_processor"  # 위탁 감독 (한국 고유)
    OTHER = "other"
    UNKNOWN = "unknown"


class _Entity(BaseModel):
    id: NonEmptyStr  # `{type}:{name}` 형태. 같은 이름은 같은 개체로 합친다
    name: NonEmptyStr


class DataCategory(_Entity):
    type: Literal["data_category"] = "data_category"
    kind: DataCategoryKind = DataCategoryKind.UNKNOWN


class Role(_Entity):
    type: Literal["role"] = "role"
    kind: RoleKind = RoleKind.UNKNOWN


class ProcessingActivity(_Entity):
    type: Literal["processing_activity"] = "processing_activity"
    action: ProcessingAction = ProcessingAction.UNKNOWN


class Purpose(_Entity):
    type: Literal["purpose"] = "purpose"
    category: PurposeCategory = PurposeCategory.UNKNOWN


class Deontic(_Entity):
    type: Literal["deontic"] = "deontic"
    kind: DeonticKind
    subtype: DeonticSubtype = DeonticSubtype.UNKNOWN


Entity = Annotated[
    DataCategory | Role | ProcessingActivity | Purpose | Deontic,
    Field(discriminator="type"),
]

EntityType = Literal["data_category", "role", "processing_activity", "purpose", "deontic"]


class ReferenceKind(StrEnum):
    """조문 사이의 관계 (정규식으로 추출)."""

    REFERS_TO = "refers_to"  # 참조
    APPLIES_MUTATIS_MUTANDIS = "applies_mutatis_mutandis"  # 준용
    EXCEPTION_TO = "exception_to"  # 예외 ("다만", "에도 불구하고")
    DELEGATES_TO = "delegates_to"  # 위임 (본법 → 시행령, 대통령령으로 정하는)
    IMPLEMENTS = "implements"  # 시행령 → 본법 ("법 제N조")


class ArticleReference(BaseModel):
    source_id: NonEmptyStr
    target_id: NonEmptyStr
    kind: ReferenceKind
    evidence: Evidence

    @model_validator(mode="after")
    def _source_matches_evidence(self) -> ArticleReference:
        if self.evidence.article_id != self.source_id:
            raise ValueError("evidence.article_id는 source_id와 같아야 한다")
        return self


class ConceptRelationKind(StrEnum):
    """개념 사이의 관계 (LLM이 추출). 허용하는 개체 유형 쌍은 ALLOWED_ENDPOINTS에 있다."""

    PERFORMS = "performs"  # Role → ProcessingActivity
    ACTS_ON = "acts_on"  # ProcessingActivity → DataCategory
    HAS_PURPOSE = "has_purpose"  # ProcessingActivity → Purpose
    BEARS = "bears"  # Role → Deontic (의무·금지의 주체, 권리의 보유자)
    COUNTERPARTY = "counterparty"  # Deontic → Role (상대방)
    CONCERNS_ACTIVITY = "concerns_activity"  # Deontic → ProcessingActivity
    CONCERNS_DATA = "concerns_data"  # Deontic → DataCategory


ALLOWED_ENDPOINTS: dict[ConceptRelationKind, tuple[EntityType, EntityType]] = {
    ConceptRelationKind.PERFORMS: ("role", "processing_activity"),
    ConceptRelationKind.ACTS_ON: ("processing_activity", "data_category"),
    ConceptRelationKind.HAS_PURPOSE: ("processing_activity", "purpose"),
    ConceptRelationKind.BEARS: ("role", "deontic"),
    ConceptRelationKind.COUNTERPARTY: ("deontic", "role"),
    ConceptRelationKind.CONCERNS_ACTIVITY: ("deontic", "processing_activity"),
    ConceptRelationKind.CONCERNS_DATA: ("deontic", "data_category"),
}


class ConceptRelation(BaseModel):
    kind: ConceptRelationKind
    source_id: NonEmptyStr
    target_id: NonEmptyStr
    evidence: Evidence

    def check_endpoints(self, entities: dict[str, Entity]) -> None:
        """두 끝점이 존재하고 관계가 허용하는 유형인지 확인한다. 어긋나면 ValueError."""
        expected_source, expected_target = ALLOWED_ENDPOINTS[self.kind]
        source, target = entities.get(self.source_id), entities.get(self.target_id)
        if source is None or target is None:
            raise ValueError("관계의 끝점이 개체 목록에 없다")
        if (source.type, target.type) != (expected_source, expected_target):
            raise ValueError(
                f"{self.kind}는 {expected_source}→{expected_target}만 허용한다 "
                f"(받은 값: {source.type}→{target.type})"
            )


class ConceptExtraction(BaseModel):
    """조문 하나에서 추출한 개체와 관계. 관계마다 evidence가 필수다."""

    article_id: NonEmptyStr
    entities: list[Entity] = Field(default_factory=list)
    relations: list[ConceptRelation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _relations_are_consistent(self) -> ConceptExtraction:
        entities = {e.id: e for e in self.entities}
        for relation in self.relations:
            if relation.evidence.article_id != self.article_id:
                raise ValueError("evidence.article_id는 article_id와 같아야 한다")
            relation.check_endpoints(entities)
        return self
