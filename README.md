# PIPA Graph RAG

개인정보 보호법(PIPA, Personal Information Protection Act)과 시행령의 조문 관계(참조, 준용, 예외, 위임)를 그래프로 구성하고, 연결된 조문까지 근거로 삼아 답하는 그래프 RAG. 스키마는 개인정보 온톨로지 PrOnto의 모듈 구조를 기반으로 한다.

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
| `LAW_OC` | 법제처 OPEN API 인증키([국가법령정보 공동활용](https://open.law.go.kr)에서 신청) |
| `OPENAI_API_KEY` | OpenAI API 키 |
| `OPENROUTER_API_KEY` | OpenRouter API 키(`--judge jev`, `--judge pplx`에만 필요) |
| `NEO4J_PASSWORD` | Neo4j 비밀번호 |
| `NEO4J_URI`, `NEO4J_USER` | 선택. 기본은 `bolt://localhost:7687`, `neo4j` |

Neo4j Browser: http://localhost:7474

사용하는 모델, 프롬프트 버전, 기본 판정기, 질의 기본값, 병렬 작업 수, 데이터 폴더는 `src/pipa_graph_rag/config.py`에서 바꾼다. 각 설정의 선택지와 제약은 그 파일의 주석에 있고, 명령줄 옵션을 주면 그 값이 우선한다.

## 샘플로 바로 써 보기

추출과 판정(LLM 호출) 없이, 저장소의 샘플 그래프(유출 통지 조문 7개)로 질의를 해볼 수 있다. `--reset`은 Neo4j의 모든 노드를 지우고 적재한다. `--no-answer`를 주면 OpenAI API 키 없이 가져온 조문만 본다.

```bash
uv run pipa-graph-rag import-graph --file data/sample/graph-sample.json --reset
uv run pipa-graph-rag ask "개인정보 유출 사실을 알게 되면 정보주체에게 언제까지 통지해야 하나요?" --no-answer
```

전체 그래프는 `export-graph`로 JSON 파일로 내보내고, `import-graph --file`로 적재한다.

## 사용법

아래 순서대로 실행한다. 각 단계의 결과는 `data/processed/`에 저장되고, 다시 실행하면 저장되지 않은 조문만 처리한다. 3, 4단계는 API 비용이 든다. 명령마다 옵션은 `uv run pipa-graph-rag <명령> --help`로 본다.

### 1. 조문 수집

개인정보 보호법과 시행령의 현행 본문을 [법제처 국가법령정보 공동활용](https://open.law.go.kr) OPEN API에서 받아 조문 단위로 저장한다.

```bash
uv run pipa-graph-rag collect
```

- 결과: `articles.json`

### 2. 조문 관계 추출

조문 본문에서 참조, 준용, 예외, 위임 관계를 정규식으로 추출한다. 위임은 시행령이 본법 조문을 인용한 관계(`IMPLEMENTS`)를 뒤집어서 만든다.

```bash
uv run pipa-graph-rag references
```

- 결과: `references.json`

### 3. 개념 관계 추출

LLM으로 조문마다 역할, 처리 활동, 목적, 권리·의무 같은 개체와 그 관계를 추출한다. 관계마다 원문 인용을 요구하고, 인용이 조문 본문에 없거나 개체 유형이 맞지 않는 관계는 버린다.

```bash
uv run pipa-graph-rag extract
```

- 결과: `extractions/<모델>-<프롬프트 버전>.jsonl`
- `--model`: 추출에 쓸 OpenAI 모델. 기본값은 `src/pipa_graph_rag/config.py`의 `EXTRACTION_MODEL`이다.
- `--ids law:15 law:35`: 지정한 조문만 추출한다.

### 4. 지지 여부 판정

추출한 관계마다 인용이 그 관계를 뒷받침하는지 판정해 확률을 저장한다.

```bash
uv run pipa-graph-rag judge
```

- 결과: `support/<모델>-<프롬프트 버전>-<판정기>.jsonl`
- `--judge`: `decisions`(OpenAI Decisions API, 기본), `jev`(Jev), `pplx`(Perplexity의 pplx-decider). `jev`와 `pplx`는 OpenRouter로 호출한다.
- `--extraction-model`: 판정할 추출 결과의 모델. 없으면 최신 프롬프트 버전의 추출 결과가 하나뿐일 때 그 파일을, 여럿이면 `EXTRACTION_MODEL`의 파일을 쓴다. 파일이 없으면 오류를 낸다. `load-graph`도 같다.

### 5. 그래프 적재

조문, 조문 관계, 개념 관계를 Neo4j에 넣는다. 본문이 "삭제"뿐인 조문은 넣지 않는다.

```bash
uv run pipa-graph-rag load-graph --reset
```

- `--reset`: 적재 전에 Neo4j의 모든 노드를 지운다. 없으면 같은 데이터를 다시 넣어도 중복되지 않는다.
- `--judge`: 어느 판정기의 결과로 거를지 고른다. 기본은 `decisions`이다.
- `--min-support`: 판정 확률이 이 값 이상인 개념 관계만 넣는다. 0이면 판정으로 거르지 않고 모두 넣는다. 판정기마다 확률의 눈금이 달라서 기본값도 다르다. 기본값은 `load-graph --help`에 나온다.

### 6. 질의

Neo4j에서 읽은 조문에서 질문과 비슷한 조문을 키워드(문자 n-gram BM25)로 찾고, 그래프로 이웃 조문까지 넓혀서 가져온 조문만 근거로 답한다. 답에 인용한 조문이 가져온 조문에 없으면 알려 준다.

```bash
uv run pipa-graph-rag ask "개인정보 유출 사실을 알게 되면 정보주체에게 언제까지 통지해야 하나요?"
```

| 옵션 | 뜻 | 기본값 |
| --- | --- | --- |
| `--mode` | `keyword`(키워드 검색만), `references`(조문 사이 관계로 확장), `references+concepts`(개념 연결로도 확장) | `references+concepts` |
| `--depth` | 확장 단계 수 | 1 |
| `--top-k` | 키워드 검색으로 가져올 시드 조문 수 | 3 |
| `--max-entity-degree` | 개념 연결 확장에서 N개 이상의 조문이 언급하는 개체는 건너뜀. 0이면 건너뛰지 않음 | 50 |
| `--no-answer` | 조문만 찾고 답변은 만들지 않음 | |

## 평가

질문 23개(단일 조문, 위임, 예외, 준용)와 질문마다 정답 조문을 `data/evaluation/questions.jsonl`에 두었다. 검색 설정별로 정답 조문을 얼마나 가져오는지 비교한다.

```bash
uv run pipa-graph-rag evaluate-search
uv run pipa-graph-rag evaluate-search --answer --conditions keyword references/d1
```

`--answer`를 주면 답변도 만들어 인용한 조문을 정답과 대조한다. `--conditions`는 돌릴 조건을 고른다.

| 조건 | 가져온 조문 수 | 필수 정답 조문 재현율 |
| --- | --- | --- |
| `keyword` | 3.0 | 0.659 |
| `references`, 깊이 1 | 20.8 | 0.913 |
| `references+concepts`, 깊이 1, `--max-entity-degree 50` (기본) | 52.5 | 1.000 |

판정기 두 개(Decisions API, Jev)의 정확도 비교는 `judge-sample`과 `judge-compare`로 한다. `judge-compare`는 이 두 판정기만 비교한다. 라벨 90개로 쟀을 때 정확도는 둘 다 약 78%였고 차이를 구분할 수 없었다.

## 그래프 구조

개념 대응은 [개념 대응표](docs/concept-mapping.md)에 있다.

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

## 한계

- 본법과 시행령(현행)만 다룬다. 판례, 법령해석례, 위원회 결정은 없다.
- 시행령이 가리키는 별표의 본문은 수집하지 않는다.
- 개념 관계는 LLM이 추출한다. 지지 여부 판정으로 일부를 걸러도 틀린 관계가 남는다.
- 법 위반 여부는 판정하지 않는다.
- 평가는 질문 23개 규모라 작은 차이는 구분하지 못한다.

## 출처와 유의 사항

법령 본문의 출처는 법제처이다. 이 저장소의 조문 데이터는 가공본이라 공식 본문이 아니며, 정확한 내용은 [국가법령정보센터](https://www.law.go.kr)에서 확인한다. 질의 결과는 법률 자문이 아니다.

## 참고 문헌

- Monica Palmirani, Michele Martoni, Arianna Rossi, Cesare Bartolini, Livio Robaldo. "PrOnto: Privacy Ontology for Legal Reasoning." Springer, 2018. [(링크)](https://link.springer.com/chapter/10.1007/978-3-319-98349-3_11)
