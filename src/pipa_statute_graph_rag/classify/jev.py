"""TypeSafe의 Jev로 인용이 주장을 뒷받침할 확률을 구한다. OpenRouter의 decisions 엔드포인트(alpha)를 쓴다."""

from __future__ import annotations

import hashlib
import os

import httpx

from ..config import JEV_MODEL, SUPPORT_PROMPT_VERSION
from ..schema import Article
from .base import JudgeFailed, JudgeResult, SupportItem
from .prompt import load_prompt

URL = "https://openrouter.ai/api/alpha/decisions"
MAX_QUESTIONS = 50  # 문서에 한도가 없어서 보수적으로 나눈다


class JevJudge:
    def __init__(
        self,
        prompt_version: str = SUPPORT_PROMPT_VERSION,
        client: httpx.Client | None = None,
    ):
        self.name = f"jev/{JEV_MODEL}"
        self.prompt_id = f"classify/support/{prompt_version}"
        self._template = load_prompt("support", prompt_version)
        self.prompt_sha256 = hashlib.sha256(self._template.encode("utf-8")).hexdigest()
        self._client = client or httpx.Client(timeout=120)

    def judge(self, article: Article, items: list[SupportItem]) -> JudgeResult:
        probabilities: dict[str, float] = {}
        input_tokens = 0
        model_version = None
        for start in range(0, len(items), MAX_QUESTIONS):
            chunk = items[start : start + MAX_QUESTIONS]
            response = self._client.post(
                URL,
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                json={
                    "model": JEV_MODEL,
                    "state": article.text,
                    "questions": {
                        item.id: {
                            "type": "noul",
                            "instructions": self._template.format(
                                statement=item.statement, quote=item.quote
                            ),
                        }
                        for item in chunk
                    },
                },
            )
            if response.status_code != 200:
                raise JudgeFailed(f"{article.id}: HTTP {response.status_code} {response.text[:200]}")
            body = response.json()
            input_tokens += body["usage"]["input_tokens"]
            model_version = body.get("model", model_version)
            for name, answer in body["answers"].items():
                if answer.get("type") != "noul":
                    raise JudgeFailed(f"{article.id}: {answer.get('type')} 응답 ({name})")
                probabilities[name] = answer["noul"]
        missing = [item.id for item in items if item.id not in probabilities]
        if missing:
            raise JudgeFailed(f"{article.id}: 답이 없는 질문 {missing[:3]}")
        return JudgeResult(
            probabilities=[probabilities[item.id] for item in items],
            input_tokens=input_tokens,
            model_version=model_version,
        )
