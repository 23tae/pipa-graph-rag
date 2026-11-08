"""질의 답변 프롬프트. `prompts/<버전>.txt`에 본문을 두고, 만든 버전은 고치지 않는다."""

from __future__ import annotations

from importlib import resources

from ..schema import Article

_LAW_NAMES = {"law": "개인정보 보호법", "decree": "개인정보 보호법 시행령"}


def available_versions() -> list[str]:
    files = resources.files(__package__).joinpath("prompts")
    return sorted(p.name.removesuffix(".txt") for p in files.iterdir() if p.name.endswith(".txt"))


def load_prompt(version: str) -> str:
    path = resources.files(__package__).joinpath("prompts", f"{version}.txt")
    if not path.is_file():
        raise ValueError(f"없는 프롬프트 버전이다: answer/{version} (있는 버전: {', '.join(available_versions())})")
    return path.read_text(encoding="utf-8")


def build_user_message(question: str, articles: list[Article]) -> str:
    context = "\n\n".join(f"[{a.id}] {_LAW_NAMES[a.law]}\n{a.text}" for a in articles)
    return f"## 조문\n\n{context}\n\n## 질문\n\n{question}"
