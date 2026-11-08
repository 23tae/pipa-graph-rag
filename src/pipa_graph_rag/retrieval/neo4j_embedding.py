"""Neo4j 벡터 인덱스로 임베딩 시드 검색을 한다. 조문 벡터는 `Article.embedding`에 저장한다."""

from __future__ import annotations

import time
from collections.abc import Callable

from neo4j import Driver

INDEX_NAME = "article_embedding"
ONLINE_TIMEOUT_SECONDS = 120  # 인덱스가 만들어진 직후에는 POPULATING 상태라 쓸 수 없다
_BATCH = 50


class EmbeddingIndexMissing(Exception):
    """Neo4j에 쓸 수 있는 벡터 인덱스가 없거나 설정된 모델의 벡터가 아니다."""


def load_embeddings(driver: Driver, vectors: dict[str, list[float]], model_name: str) -> int:
    """조문 벡터를 Article.embedding에 넣고 코사인 벡터 인덱스를 만든다. 넣은 조문 수를 돌려준다.

    이미 있는 인덱스는 지우고 새로 만든다. 모델이 바뀌면 차원도 바뀔 수 있기 때문이다."""
    dimensions = {len(v) for v in vectors.values()}
    if len(dimensions) != 1:
        raise ValueError(f"벡터 차원이 하나가 아니다: {sorted(dimensions)}")
    dimension = dimensions.pop()
    driver.execute_query(f"DROP INDEX {INDEX_NAME} IF EXISTS")
    rows = [{"id": article_id, "vector": vector} for article_id, vector in vectors.items()]
    loaded = 0
    for start in range(0, len(rows), _BATCH):
        records = driver.execute_query(
            "UNWIND $rows AS row MATCH (a:Article {id: row.id}) "
            "CALL db.create.setNodeVectorProperty(a, 'embedding', row.vector) "
            "SET a.embeddingModel = $model RETURN count(a) AS n",
            parameters_={"rows": rows[start : start + _BATCH], "model": model_name},
        ).records
        loaded += records[0]["n"]
    if loaded != len(rows):
        raise EmbeddingIndexMissing(f"Neo4j에서 찾은 조문 {loaded}개가 벡터 {len(rows)}개와 다르다")
    driver.execute_query(
        f"CREATE VECTOR INDEX {INDEX_NAME} FOR (a:Article) ON a.embedding "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, `vector.similarity_function`: 'cosine'}}}}"
    )
    _wait_until_online(driver)
    return loaded


def reset_embeddings(driver: Driver) -> None:
    driver.execute_query(f"DROP INDEX {INDEX_NAME} IF EXISTS")
    driver.execute_query("MATCH (a:Article) REMOVE a.embedding, a.embeddingModel")


def _wait_until_online(driver: Driver) -> None:
    deadline = time.monotonic() + ONLINE_TIMEOUT_SECONDS
    while True:
        records = driver.execute_query(
            "SHOW VECTOR INDEXES YIELD name, state WHERE name = $name RETURN state",
            parameters_={"name": INDEX_NAME},
        ).records
        state = records[0]["state"] if records else "없음"
        if state == "ONLINE":
            return
        if time.monotonic() > deadline:
            raise EmbeddingIndexMissing(f"벡터 인덱스가 {ONLINE_TIMEOUT_SECONDS}초 안에 ONLINE이 되지 않았다: {state}")
        time.sleep(1)


class Neo4jEmbeddingIndex:
    def __init__(self, driver: Driver, model_name: str, embed_query: Callable[[str], list[float]]):
        """model_name: 설정된 임베딩 모델. Neo4j에 저장된 벡터를 만든 모델과 다르면 오류를 낸다."""
        self._driver = driver
        self._embed_query = embed_query
        records = driver.execute_query(
            "MATCH (a:Article) WHERE a.embedding IS NOT NULL RETURN a.embeddingModel AS model, count(*) AS n"
        ).records
        if not records:
            raise EmbeddingIndexMissing("Neo4j에 조문 벡터가 없다. load-embeddings를 먼저 실행한다")
        models = {r["model"] for r in records}
        if models != {model_name}:
            raise EmbeddingIndexMissing(
                f"Neo4j의 벡터는 {sorted(map(str, models))}로 만든 것인데 설정된 모델은 {model_name}이다. "
                "load-embeddings로 다시 넣는다"
            )
        self._size = sum(r["n"] for r in records)

    def __len__(self) -> int:
        return self._size

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """벡터 인덱스에서 가까운 조문 최대 k개. 근사 검색이라 정확한 k개가 아닐 수 있다. 점수는 0~1이다."""
        records = self._driver.execute_query(
            "CALL db.index.vector.queryNodes($index, $k, $vector) YIELD node, score "
            "RETURN node.id AS id, score",
            parameters_={"index": INDEX_NAME, "k": k, "vector": self._embed_query(query)},
        ).records
        return sorted(((r["id"], r["score"]) for r in records), key=lambda item: (-item[1], item[0]))
