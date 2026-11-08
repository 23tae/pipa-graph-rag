"""판정기를 이름으로 고른다. 판정기마다 확률의 눈금이 달라서 적재 기준(`--min-support`)의 기본값도 다르다."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..config import JEV_MODEL, PPLX_MODEL
from .base import SupportJudge


@dataclass(frozen=True)
class JudgeSpec:
    min_support: float  # 판정 확률이 이 값 이상인 관계만 적재하는 기본 기준
    build: Callable[[], SupportJudge]


def _decisions() -> SupportJudge:
    from .decisions import DecisionsJudge

    return DecisionsJudge()


def _openrouter(name: str, model: str) -> Callable[[], SupportJudge]:
    def build() -> SupportJudge:
        from .openrouter import OpenRouterJudge

        return OpenRouterJudge(name, model)

    return build


# decisions의 0.1은 확률 구간별로 표본을 읽어 정했다. jev와 pplx는 라벨 90개에서 decisions 0.1과
# 같은 비율(뒷받침되는 관계 62개 중 51개)을 남기는 값이다.
JUDGES: dict[str, JudgeSpec] = {
    "decisions": JudgeSpec(0.1, _decisions),
    "jev": JudgeSpec(0.54, _openrouter("jev", JEV_MODEL)),
    "pplx": JudgeSpec(0.92, _openrouter("pplx", PPLX_MODEL)),
}


def resolve_min_support(judge: str, min_support: float | None) -> float:
    """명령줄 값이 있으면 그 값을, 없으면 판정기의 기본 기준을 쓴다."""
    return JUDGES[judge].min_support if min_support is None else min_support
