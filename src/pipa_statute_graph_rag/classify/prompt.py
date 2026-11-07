"""분류 단계의 프롬프트. 작업마다 `prompts/<작업>/<버전>.txt`에 본문을 두고, 만든 버전은 고치지 않는다."""

from __future__ import annotations

from importlib import resources

LATEST_SUPPORT_PROMPT_VERSION = "v1"


def available_versions(task: str) -> list[str]:
    files = resources.files(__package__).joinpath("prompts", task)
    return sorted(p.name.removesuffix(".txt") for p in files.iterdir() if p.name.endswith(".txt"))


def load_prompt(task: str, version: str) -> str:
    path = resources.files(__package__).joinpath("prompts", task, f"{version}.txt")
    if not path.is_file():
        raise ValueError(
            f"없는 프롬프트 버전이다: {task}/{version} (있는 버전: {', '.join(available_versions(task))})"
        )
    return path.read_text(encoding="utf-8")
