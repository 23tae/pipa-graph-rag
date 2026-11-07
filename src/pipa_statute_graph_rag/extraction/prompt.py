"""추출 프롬프트. 순수 문자열 조립이라 외부 호출이 없다."""

from __future__ import annotations

from ..schema import ALLOWED_ENDPOINTS, Article

PROMPT_VERSION = "v3"

_ENDPOINT_NAMES = {
    "role": "역할(role)",
    "processing_activity": "처리 활동(processing_activity)",
    "data_category": "개인정보 범주(data_category)",
    "purpose": "목적(purpose)",
    "deontic": "권리·의무(deontic)",
    "article": "이 조문",
}


def _relation_lines() -> str:
    return "\n".join(
        f"- {kind.value}: {_ENDPOINT_NAMES[source]} → {_ENDPOINT_NAMES[target]}"
        for kind, (source, target) in ALLOWED_ENDPOINTS.items()
    )


SYSTEM_PROMPT = f"""개인정보 보호법과 시행령의 조문 한 개에서 개체와 관계를 추출한다.

## 개체
- role: 법이 의무나 권리를 정하는 주체(개인정보처리자, 정보주체, 수탁자, 보호위원회 등)
- processing_activity: 개인정보의 처리 유형. 수집, 생성, 연계, 연동, 기록, 저장, 보유, 가공, 편집, 검색, 출력, 정정, 복구, 이용, 제공, 공개, 파기, 가명처리만 해당하고 action에 그 값을 쓴다. 서류 제출, 계획 수립, 조사, 회의, 지정, 신고처럼 개인정보를 처리하는 행위가 아닌 것은 processing_activity로 만들지 않는다.
- data_category: 개인정보의 범주(개인정보, 가명정보, 민감정보, 고유식별정보 등)
- purpose: 개인정보를 처리하는 이유나 목적을 서술한 구절("…하기 위하여", "…을 목적으로")만 해당한다. 사항의 이름이나 조문 제목에 "목적"이라는 말이 있는 것은 purpose가 아니다.
- deontic: 권리, 의무, 허용, 금지, 제재 가운데 조문이 정하는 것. kind에 종류를 쓴다
- name은 개체를 가리키는 짧은 명사구다. 조문에 나온 용어를 그대로 쓰되("개인정보처리자", "열람 요구권"), 문장의 일부를 잘라 쓰지 않는다("수집할 수 있으며"는 안 된다). 같은 개체는 같은 name을 쓴다.
- id는 이 응답 안에서만 쓰는 임시 이름이다.
- 각 개체의 kind, action, category, subtype은 근거가 없으면 unknown으로 둔다.

## 관계
허용하는 관계와 방향은 아래뿐이다. 방향이 다르면 쓰지 않는다.
{_relation_lines()}

## 규칙
- 모든 관계에 quote를 쓴다. quote는 조문 본문의 한 문장 안에서 이어진 구절을 글자 그대로 복사한 것이다. 요약하거나 고쳐 쓰지 않고, 여러 문장이나 떨어진 부분을 이어 붙이지 않는다.
- quote에는 그 관계를 보여 주는 주어와 서술어가 모두 들어가야 한다. "보호위원회는"처럼 주어만 있는 짧은 구절은 쓰지 않는다.
- 조문에 근거가 없는 관계는 만들지 않는다. 법 위반 여부는 판단하지 않는다.
- described_by의 target_id에는 주어진 조문 id를 쓴다. 목적이 이 조문에 서술되어 있을 때만 쓴다.
- is_legal_basis는 이 조문이 개인정보 처리를 할 수 있는 근거(동의, 법률의 특별한 규정 등)를 정하면 true, 아니면 false, 판단하기 어려우면 null이다.
- 개체와 관계가 없으면 빈 목록을 돌려준다.
"""


def build_user_message(article: Article) -> str:
    return f"조문 id: {article.id}\n\n{article.text}"
