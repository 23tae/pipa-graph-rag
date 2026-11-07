"""OpenAI Decisions API로 인용이 주장을 뒷받침할 확률을 구한다."""

from __future__ import annotations

import hashlib

from openai import OpenAI

from ..config import DECISIONS_MODEL
from ..schema import Article
from .base import JudgeFailed, JudgeResult, SupportItem
from .prompt import LATEST_SUPPORT_PROMPT_VERSION, load_prompt

MAX_QUESTIONS = 400  # 요청 하나에 넣을 수 있는 질문 수


class DecisionsJudge:
    def __init__(self, prompt_version: str = LATEST_SUPPORT_PROMPT_VERSION, client: OpenAI | None = None):
        self.name = f"decisions/{DECISIONS_MODEL}"
        self.prompt_id = f"classify/support/{prompt_version}"
        self._template = load_prompt("support", prompt_version)
        self.prompt_sha256 = hashlib.sha256(self._template.encode("utf-8")).hexdigest()
        self._client = client or OpenAI()

    def judge(self, article: Article, items: list[SupportItem]) -> JudgeResult:
        probabilities: dict[str, float] = {}
        input_tokens = 0
        model_version = None
        for start in range(0, len(items), MAX_QUESTIONS):
            chunk = items[start : start + MAX_QUESTIONS]
            decision = self._client.decisions.create(
                model=DECISIONS_MODEL,
                input=article.text,
                questions=[
                    {
                        "type": "predicate",
                        "name": item.id,
                        "instructions": self._template.format(statement=item.statement, quote=item.quote),
                    }
                    for item in chunk
                ],
            )
            input_tokens += decision.usage.input_tokens
            model_version = getattr(decision, "model", None) or model_version
            for answer in decision.answers:
                if answer.type != "predicate":
                    raise JudgeFailed(f"{article.id}: {answer.type} 응답 ({answer.name})")
                probabilities[answer.name] = answer.probability
        missing = [item.id for item in items if item.id not in probabilities]
        if missing:
            raise JudgeFailed(f"{article.id}: 답이 없는 질문 {missing[:3]}")
        return JudgeResult(
            probabilities=[probabilities[item.id] for item in items],
            input_tokens=input_tokens,
            model_version=model_version,
        )
