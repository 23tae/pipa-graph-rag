# 개념 대응표

개인정보 온톨로지 [PrOnto](https://link.springer.com/chapter/10.1007/978-3-319-98349-3_11)의 각 모듈이 이 프로젝트의 스키마(`src/pipa_statute_graph_rag/schema.py`)와 개인정보 보호법·시행령의 개념에 어떻게 대응하는지 정리한 표이다. 법령 개념의 근거 조문은 본법 기준이다.

| PrOnto 모듈 | 스키마 | 한국 법령 |
| --- | --- | --- |
| 문서 | `Article` | 조문 |
| 데이터 | `DataCategory` | 개인정보의 범주 |
| 행위자와 역할 | `Role` | 처리자, 정보주체 등 |
| 처리 | `ProcessingActivity` | 개인정보의 처리 |
| 목적과 법적 근거 | `Purpose`, `Article.is_legal_basis` | 처리 목적, 처리의 근거 조항 |
| 의무 연산자(deontic operators) | `Deontic` | 권리, 의무, 허용, 금지, 제재 |

## 데이터 범주 (`DataCategory.kind`)

| 값 | 한국 법령 | 근거 |
| --- | --- | --- |
| `personal` | 개인정보 | 제2조 제1호 |
| `pseudonymized` | 가명정보 | 제2조 제1호 다목, 제28조의2~5 |
| `sensitive` | 민감정보 | 제23조 |
| `anonymous` | 익명 | 정의 조항은 없고 제3조에 등장한다. 약한 대응이다 |
| `unique_identifier` | 고유식별정보 | 제24조. PrOnto에는 없고 한국 법에만 있는 개념 |
| `resident_number` | 주민등록번호 | 제24조의2. PrOnto에는 없고 한국 법에만 있는 개념 |

## 역할 (`Role.kind`)

| 값 | 한국 법령 | 근거 |
| --- | --- | --- |
| `controller` | 개인정보처리자 | 제2조 제5호 |
| `processor` | 수탁자 | 제26조 |
| `data_subject` | 정보주체 | 제2조 제3호 |
| `third_party` | 제3자 | 제17조 등 |
| `recipient` | 제공받는 자 | 제17조, 제18조 |
| `dpo` | 개인정보 보호책임자 | 제31조 |
| `representative` | 국내대리인 | 제31조의2 |
| `supervisory_authority` | 개인정보 보호위원회 | 제7조 |
| `handler` | 개인정보취급자 | 제28조. 한국 법에만 있는 개념 |
| `public_institution` | 공공기관 | 제2조 제6호. 한국 법에만 있는 개념 |

## 처리 (`ProcessingActivity.action`)

값은 제2조 제2호가 정의한 처리의 유형을 그대로 쓴다: 수집, 생성, 연계, 연동, 기록, 저장, 보유, 가공, 편집, 검색, 출력, 정정, 복구, 이용, 제공, 공개, 파기, 가명처리. PrOnto의 행위 클래스(`Observe`, `Store`, `Provide`, `Delete` 등)와 일대일로 대응하는지는 확정하지 않았다.

## 목적과 법적 근거

- `Purpose.category`는 가명정보 처리 목적인 통계작성(`statistics`), 과학적 연구(`research`), 공익적 기록보존(`public_interest_archiving`)만 분류하고(제28조의2·3) 나머지는 `unknown`이다.
- PrOnto는 법적 근거를 별도 클래스가 아니라 목적을 서술하는 문서의 속성(`isLegalBasis`)으로 둔다. 이 프로젝트도 같다. 조문이 처리의 근거 규정이면 `Article.is_legal_basis`가 참이고, `described_by` 관계가 목적과 그 목적을 서술하는 조문을 잇는다.

## 권리와 의무 (`Deontic`)

`Deontic.kind`는 권리(`right`), 의무(`obligation`), 허용(`permission`), 금지(`prohibition`), 제재(`sanction`)이다. `subtype`은 다음과 같다.

| 값 | 한국 법령 | 근거 |
| --- | --- | --- |
| `access` | 열람권 | 제35조 |
| `rectification` | 정정 요구권 | 제36조 |
| `erasure` | 삭제 요구권, 파기 의무 | 제36조, 제21조 |
| `restrict_processing` | 처리정지 요구권 | 제37조 |
| `portability` | 개인정보 전송 요구권 | 제35조의2 |
| `automated_decision` | 자동화된 결정에 대한 권리 | 제37조의2 |
| `inform` | 수집 출처·이용내역 통지 | 제20조, 제20조의2 |
| `notify_breach` | 유출 통지·신고 | 제34조 |
| `prepare_document` | 처리방침 수립·공개, 개인정보파일 등록 | 제30조, 제32조 |
| `safeguard` | 안전조치 | 제29조. 한국 법에만 있는 의무 |
| `impact_assessment` | 영향평가 | 제33조. 한국 법에만 있는 의무 |
| `designate_dpo` | 보호책임자 지정 | 제31조. 한국 법에만 있는 의무 |
| `designate_representative` | 국내대리인 지정 | 제31조의2. 한국 법에만 있는 의무 |
| `supervise_processor` | 위탁 감독 | 제26조, 제28조. 한국 법에만 있는 의무 |

제재(`sanction`)는 벌칙, 과태료, 과징금(제70~73조, 제75조, 제64조의2)에 대응한다.

## 관계

| 관계 | 끝점 | 설명 |
| --- | --- | --- |
| `bears` | `Role` → `Deontic` | 의무·금지의 주체, 권리의 보유자. PrOnto의 `Bearer` 역할에 대응한다 |
| `counterparty` | `Deontic` → `Role` | 상대방. PrOnto의 `CounterParty` 역할에 대응한다 |
| `performs` | `Role` → `ProcessingActivity` | 역할이 수행하는 처리 |
| `acts_on` | `ProcessingActivity` → `DataCategory` | 처리의 대상 |
| `has_purpose` | `ProcessingActivity` → `Purpose` | 처리의 목적 |
| `concerns_activity` | `Deontic` → `ProcessingActivity` | 권리·의무가 관한 처리 |
| `concerns_data` | `Deontic` → `DataCategory` | 권리·의무가 관한 데이터 |
| `described_by` | `Purpose` → `Article` | 목적을 서술하는 조문. PrOnto의 `isDescribedBy`에 대응한다 |

조문 사이의 관계는 정규식으로 추출하며 PrOnto에는 없다: 참조(`refers_to`), 준용(`applies_mutatis_mutandis`), 예외(`exception_to`), 위임(`delegates_to`), 시행령이 본법 조문을 인용한 관계(`implements`).

## 스키마에서 뺀 PrOnto 개념

| PrOnto 개념 | 이유 |
| --- | --- |
| 처리 실행, 단계별 위험·조치의 값 | 법령 텍스트는 규범이라 실제 처리 사건이 없다 |
| 투명성·공정성·적법성, 위반(`Violation`), 준수(`Compliance`) | 법 위반 여부를 판정하는 값이다. 이 프로젝트의 범위가 아니다 |
| 회원국, 관할권, 장소 | 유럽연합에만 있는 개념이다 |
| FRBR 기반 문서 표현 | 법제처 API가 법령, 조, 항, 호의 계층을 이미 준다 |

## 한계

- 한국 법령과의 대응은 용어와 조문 제목을 대조한 수준이다. 조문 본문의 의미가 PrOnto 클래스와 같은지는 확인하지 않았다.
- 대응 근거는 본법의 조문이다. 시행령의 용어는 대조하지 않았다.
