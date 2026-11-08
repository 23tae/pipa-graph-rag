"""설정. 명령줄 옵션을 주지 않으면 여기의 값을 쓴다."""

from pathlib import Path

EXTRACTION_MODEL = "gpt-6-luna"  # OpenAI Responses API로 호출하고 구조화 출력을 지원하는 모델이어야 한다
ANSWER_MODEL = "gpt-6-luna"  # 위와 같다
DECISIONS_MODEL = "gpt-6-luna"  # OpenAI Decisions API가 지원하는 유일한 모델
JEV_MODEL = "~typesafe/jev-latest"  # OpenRouter 모델 ID. decisions 엔드포인트가 noul 응답을 내는 모델이어야 한다
PPLX_MODEL = "perplexity/pplx-decider-v1.2"  # 위와 같다

EXTRACTION_PROMPT_VERSION = "v3"  # extraction/prompts에 있는 버전: v2 | v3
SUPPORT_PROMPT_VERSION = "v1"  # classify/prompts/support에 있는 버전: v1
ANSWER_PROMPT_VERSION = "v1"  # answer/prompts에 있는 버전: v1

DEFAULT_JUDGE = "decisions"  # decisions | jev | pplx

QUERY_MODE = "references+concepts"  # keyword | references | references+concepts
QUERY_DEPTH = 1  # 시작 조문에서 그래프로 넓히는 단계 수
QUERY_TOP_K = 3  # 키워드 검색으로 고르는 시작 조문 수
QUERY_MAX_ENTITY_DEGREE = 50  # 이 수 이상의 조문이 언급하는 개체는 개념 확장에서 건너뛴다. 0이면 건너뛰지 않는다

DEFAULT_WORKERS = 4  # 동시에 보내는 API 요청 수
# 추론 토큰과 출력 토큰을 합친 한도. 추론 모델 가이드가 최소 25,000토큰을 확보하라고 권한다.
MAX_OUTPUT_TOKENS = 25_000

NEO4J_URI = "bolt://localhost:7687"  # 환경변수 NEO4J_URI로 덮어쓴다
NEO4J_USER = "neo4j"  # 환경변수 NEO4J_USER로 덮어쓴다

DATA_DIR = Path("data/processed")  # 수집, 추출, 판정 결과
COMPARISON_DIR = Path("data/judge-comparison")  # 판정기 비교용 표본, 라벨, 판정 결과
EVALUATION_DIR = Path("data/evaluation")  # 검색 평가 질문과 결과
GRAPH_DIR = Path("data/graph")  # export-graph로 내보낸 그래프 JSON
