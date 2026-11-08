"""설정. 명령줄 옵션을 주지 않으면 여기의 값을 쓴다."""

from pathlib import Path

EXTRACTION_MODEL = "gpt-6-luna"
ANSWER_MODEL = "gpt-6-luna"
DECISIONS_MODEL = "gpt-6-luna"  # Decisions API가 지원하는 유일한 모델
JEV_MODEL = "~typesafe/jev-latest"

EXTRACTION_PROMPT_VERSION = "v3"
SUPPORT_PROMPT_VERSION = "v1"
ANSWER_PROMPT_VERSION = "v1"

DEFAULT_JUDGE = "decisions"
MIN_SUPPORT = 0.1  # 이 확률 이상인 개념 관계만 그래프에 넣는다. 0이면 모두 넣는다

QUERY_MODE = "references+concepts"
QUERY_DEPTH = 1
QUERY_TOP_K = 3
QUERY_MAX_ENTITY_DEGREE = 50  # 이 수 이상의 조문이 언급하는 개체는 개념 확장에서 건너뛴다. 0이면 건너뛰지 않는다

DEFAULT_WORKERS = 4
# 추론 토큰과 출력 토큰을 합친 한도. 추론 모델 가이드가 최소 25,000토큰을 확보하라고 권한다.
MAX_OUTPUT_TOKENS = 25_000

NEO4J_URI = "bolt://localhost:7687"  # 환경변수 NEO4J_URI로 덮어쓴다
NEO4J_USER = "neo4j"  # 환경변수 NEO4J_USER로 덮어쓴다

DATA_DIR = Path("data/processed")
COMPARISON_DIR = Path("data/judge-comparison")
EVALUATION_DIR = Path("data/evaluation")
GRAPH_DIR = Path("data/graph")
