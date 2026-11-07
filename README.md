# PIPA Statute Graph RAG

개인정보 보호법과 시행령의 조문 관계(참조, 준용, 예외, 위임)를 그래프로 구성하고, 연결된 조문까지 근거로 삼아 답하는 그래프 RAG. 스키마는 개인정보 온톨로지 PrOnto의 모듈 구조를 기반으로 한다.

## 요구 사항

- Python 3.12+
- uv
- Docker

## 설정

```bash
cp .env.example .env
uv sync
docker compose up -d
```

`.env`에서 채울 값

| 변수 | 용도 |
| --- | --- |
| `LAW_OC` | 법제처 OPEN API 인증키 |
| `NEO4J_PASSWORD` | Neo4j 비밀번호 |

Neo4j Browser: http://localhost:7474

## 사용법

조문 수집: 개인정보 보호법과 시행령의 현행 본문을 `data/processed/articles.json`에 조문 단위로 저장한다.

```bash
uv run pipa-statute-graph-rag collect
```

테스트

```bash
uv run pytest
```

## 출처

법령 본문은 [법제처 국가법령정보 공동활용](https://open.law.go.kr) OPEN API에서 받는다. 코드는 MIT 라이선스이고 법령 데이터는 해당하지 않는다.

## 참고 문헌

- Monica Palmirani, Michele Martoni, Arianna Rossi, Cesare Bartolini, Livio Robaldo. "PrOnto: Privacy Ontology for Legal Reasoning." Springer, 2018. [(링크)](https://link.springer.com/chapter/10.1007/978-3-319-98349-3_11)
