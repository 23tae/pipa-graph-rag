"""검색된 조문에서 이웃 조문으로 넓힌다. 이웃을 찾는 방법은 함수로 받아서 데이터베이스를 쓰지 않는다."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel


class Edge(BaseModel):
    source: str  # 이미 가져온 조문
    target: str  # 새로 가져올 조문
    via: str  # 어떻게 이어졌는가. 조문 관계 타입(IMPLEMENTS 등) 또는 `개체:<이름>`


class Reach(BaseModel):
    source: str
    via: str
    depth: int


Neighbors = Callable[[list[str]], list[Edge]]


class Expansion(BaseModel):
    seeds: list[str]
    articles: list[str]  # 시드, 이어서 깊이 순으로 새로 가져온 조문
    reached: dict[str, Reach]  # 시드가 아닌 조문이 어디서 어떻게 왔는가


def expand(seeds: list[str], neighbors: list[Neighbors], depth: int) -> Expansion:
    """시드에서 depth 단계까지 이웃 조문을 가져온다. 먼저 닿은 경로를 남기고 같은 조문은 한 번만 가져온다."""
    visited = list(dict.fromkeys(seeds))
    known = set(visited)
    reached: dict[str, Reach] = {}
    frontier = list(visited)
    for level in range(1, depth + 1):
        edges = [edge for find in neighbors for edge in find(frontier)]
        next_frontier: list[str] = []
        for edge in sorted(edges, key=lambda e: (e.target, e.source, e.via)):
            if edge.target in known:
                continue
            known.add(edge.target)
            reached[edge.target] = Reach(source=edge.source, via=edge.via, depth=level)
            next_frontier.append(edge.target)
        visited.extend(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return Expansion(seeds=list(dict.fromkeys(seeds)), articles=visited, reached=reached)
