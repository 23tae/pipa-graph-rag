"""추출 프롬프트. 버전마다 `prompts/<버전>.txt`에 본문을 두고, 만든 버전은 고치지 않는다."""

from __future__ import annotations

from importlib import resources

from ..schema import Article

def available_versions() -> list[str]:
    files = resources.files(__package__).joinpath("prompts")
    return sorted(p.name.removesuffix(".txt") for p in files.iterdir() if p.name.endswith(".txt"))


def load_prompt(version: str) -> str:
    path = resources.files(__package__).joinpath("prompts", f"{version}.txt")
    if not path.is_file():
        raise ValueError(f"없는 프롬프트 버전이다: {version} (있는 버전: {', '.join(available_versions())})")
    return path.read_text(encoding="utf-8")


def build_user_message(article: Article) -> str:
    return f"조문 id: {article.id}\n\n{article.text}"
