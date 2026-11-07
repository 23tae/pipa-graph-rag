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
| `OPENAI_API_KEY` | OpenAI API 키 |
| `OPENROUTER_API_KEY` | OpenRouter API 키(Jev 판정에만 필요) |
| `NEO4J_PASSWORD` | Neo4j 비밀번호 |

Neo4j Browser: http://localhost:7474

## 사용법

아래 순서대로 실행한다. 각 단계의 결과는 `data/processed/`에 저장되고, 다시 실행하면 저장되지 않은 조문만 처리한다.

### 1. 조문 수집

개인정보 보호법과 시행령의 현행 본문을 [법제처 국가법령정보 공동활용](https://open.law.go.kr) OPEN API에서 받아 조문 단위로 저장한다.

```bash
uv run pipa-statute-graph-rag collect
```

- 결과: `articles.json`

### 2. 조문 관계 추출

조문 본문에서 참조, 준용, 예외, 위임 관계를 정규식으로 추출한다. 위임은 시행령이 본법 조문을 인용한 관계(`IMPLEMENTS`)를 뒤집어서 만든다.

```bash
uv run pipa-statute-graph-rag references
```

- 결과: `references.json`

### 3. 개념 관계 추출

LLM으로 조문마다 역할, 처리 활동, 목적, 권리·의무 같은 개체와 그 관계를 추출한다. 관계마다 원문 인용을 요구하고, 인용이 조문 본문에 없거나 개체 유형이 맞지 않는 관계는 버린다.

```bash
uv run pipa-statute-graph-rag extract --model gpt-6-luna
```

- 결과: `extractions/<모델>-<프롬프트 버전>.jsonl`
- `--ids law:15 law:35`: 지정한 조문만 추출한다.

### 4. 지지 여부 판정

추출한 관계마다 인용이 그 관계를 뒷받침하는지 판정해 확률을 저장한다.

```bash
uv run pipa-statute-graph-rag judge --model gpt-6-luna
```

- 결과: `support/<모델>-<프롬프트 버전>-<판정기>.jsonl`
- `--judge decisions`(OpenAI Decisions API, 기본), `--judge jev`(Jev)

### 5. 그래프 적재

조문, 조문 관계, 개념 관계를 Neo4j에 넣는다. 본문이 "삭제"뿐인 조문은 넣지 않는다.

```bash
uv run pipa-statute-graph-rag load-graph --model gpt-6-luna --reset --min-support 0.1
```

- `--reset`: 적재 전에 Neo4j의 모든 노드를 지운다. 없으면 같은 데이터를 다시 넣어도 중복되지 않는다.
- `--min-support`: 판정 확률이 이 값 이상인 개념 관계만 넣는다.

### 6. 질의

Neo4j에서 읽은 조문에서 질문과 비슷한 조문을 키워드(문자 n-gram BM25)로 찾고, 그래프로 이웃 조문까지 넓혀서 가져온 조문만 근거로 답한다. 답에 인용한 조문이 가져온 조문에 없으면 알려 준다.

```bash
uv run pipa-statute-graph-rag ask "개인정보 유출 사실을 알게 되면 정보주체에게 언제까지 통지해야 하나요?"
```

| 옵션 | 뜻 | 기본값 |
| --- | --- | --- |
| `--mode` | `keyword`(키워드 검색만), `references`(조문 사이 관계로 확장), `references+concepts`(개념 연결로도 확장) | `references+concepts` |
| `--depth` | 확장 단계 수 | 1 |
| `--top-k` | 키워드 검색으로 가져올 시드 조문 수 | 3 |
| `--max-entity-degree` | 개념 연결 확장에서 N개 이상의 조문이 언급하는 개체는 건너뜀. 0이면 건너뛰지 않음 | 50 |
| `--no-answer` | 조문만 찾고 답변은 만들지 않음 | |

## 그래프 구조

- 노드: `Article`(조문), `Role`, `ProcessingActivity`, `DataCategory`, `Purpose`, `Deontic`(개체)
- 조문 사이 관계: `REFERS_TO`, `IMPLEMENTS`, `DELEGATES_TO`, `APPLIES_MUTATIS_MUTANDIS`, `EXCEPTION_TO`
- 개념 관계: `PERFORMS`, `ACTS_ON`, `HAS_PURPOSE`, `BEARS`, `COUNTERPARTY`, `CONCERNS_ACTIVITY`, `CONCERNS_DATA`, `DESCRIBED_BY`
- `MENTIONS`: 조문이 개체를 언급한다.
- 모든 관계에 원문 인용 `quote`가 있고, 개념 관계에는 출처 조문 `article_id`도 있다.

Neo4j Browser에서 본법 제24조가 위임한 시행령 조문을 조회하는 예:

```cypher
MATCH (:Article {id: 'law:24'})-[:DELEGATES_TO]->(d) RETURN d.id, d.title
```

## 테스트

```bash
uv run pytest
```

## 참고 문헌

- Monica Palmirani, Michele Martoni, Arianna Rossi, Cesare Bartolini, Livio Robaldo. "PrOnto: Privacy Ontology for Legal Reasoning." Springer, 2018. [(링크)](https://link.springer.com/chapter/10.1007/978-3-319-98349-3_11)
