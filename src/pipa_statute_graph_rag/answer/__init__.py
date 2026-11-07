"""가져온 조문만 근거로 질문에 답하고, 인용한 조문을 검증한다."""

from __future__ import annotations

import hashlib
from typing import Protocol

from pydantic import BaseModel

from ..config import ANSWER_MODEL
from ..schema import Article
from .prompt import LATEST_PROMPT_VERSION, build_user_message, load_prompt

# 추론 토큰과 출력 토큰을 합친 한도. 추론 모델 가이드가 최소 25,000토큰을 확보하라고 권한다.
MAX_OUTPUT_TOKENS = 25_000


class AnswerOutput(BaseModel):
    answer: str
    citations: list[str]  # 답의 근거로 든 조문 ID


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[str]
    invalid_citations: list[str]  # 가져온 조문에 없는 인용
    model: str
    prompt_id: str
    prompt_sha256: str
    input_tokens: int
    output_tokens: int


class Answerer(Protocol):
    model: str
    prompt_id: str
    prompt_sha256: str

    def answer(self, question: str, articles: list[Article]) -> tuple[AnswerOutput, int, int]: ...


def normalize_citation(citation: str) -> str:
    """모델이 조문 ID를 `[law:31]`처럼 대괄호로 감싸 내는 경우가 있어서 벗긴다. 프롬프트는 입력 조문 머리글을
    `[조문 ID]`로 보여 준다."""
    return citation.strip().strip("[]").strip()


def check_citations(citations: list[str], retrieved_ids: list[str]) -> list[str]:
    """가져오지 않은 조문을 인용한 것을 돌려준다. 중복은 한 번만 센다."""
    known = set(retrieved_ids)
    return list(dict.fromkeys(c for c in citations if c not in known))


def ask(answerer: Answerer, question: str, articles: list[Article]) -> Answer:
    output, input_tokens, output_tokens = answerer.answer(question, articles)
    citations = [normalize_citation(c) for c in output.citations]
    return Answer(
        question=question,
        answer=output.answer,
        citations=citations,
        invalid_citations=check_citations(citations, [a.id for a in articles]),
        model=answerer.model,
        prompt_id=answerer.prompt_id,
        prompt_sha256=answerer.prompt_sha256,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class OpenAIAnswerer:
    def __init__(self, model: str = ANSWER_MODEL, prompt_version: str = LATEST_PROMPT_VERSION, client=None):
        from openai import OpenAI

        self.model = model
        self.prompt_id = f"answer/{prompt_version}"
        self._prompt = load_prompt(prompt_version)
        self.prompt_sha256 = hashlib.sha256(self._prompt.encode("utf-8")).hexdigest()
        self._client = client or OpenAI()

    def answer(self, question: str, articles: list[Article]) -> tuple[AnswerOutput, int, int]:
        response = self._client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": build_user_message(question, articles)},
            ],
            text_format=AnswerOutput,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        if response.status == "incomplete":
            reason = response.incomplete_details.reason if response.incomplete_details else "알 수 없음"
            raise RuntimeError(f"응답이 끝나지 않았다: {reason}")
        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if part.type == "refusal":
                        raise RuntimeError(f"모델이 거절했다: {part.refusal[:200]}")
        if response.output_parsed is None:
            raise RuntimeError("파싱된 결과가 없다")
        usage = response.usage
        return response.output_parsed, usage.input_tokens, usage.output_tokens
