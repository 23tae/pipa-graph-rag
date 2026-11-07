"""OpenAI Responses API로 조문에서 개체와 관계를 추출한다."""

from __future__ import annotations

from openai import OpenAI

from ..extraction.models import ExtractorResult, RawExtraction
from ..extraction.prompt import SYSTEM_PROMPT, build_user_message
from ..schema import Article

# 추론 토큰과 출력 토큰을 합친 한도. 추론 모델 가이드가 최소 25,000토큰을 확보하라고 권한다.
MAX_OUTPUT_TOKENS = 25_000


class ExtractionFailed(Exception):
    """모델이 거절했거나 응답이 끝까지 생성되지 않아 쓸 수 있는 결과가 없다."""


class OpenAIExtractor:
    def __init__(self, model: str, client: OpenAI | None = None) -> None:
        self.model = model
        self._client = client or OpenAI()

    def extract(self, article: Article) -> ExtractorResult:
        response = self._client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(article)},
            ],
            text_format=RawExtraction,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        if response.status == "incomplete":
            reason = response.incomplete_details.reason if response.incomplete_details else "알 수 없음"
            raise ExtractionFailed(f"응답이 끝나지 않았다: {reason}")
        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if part.type == "refusal":
                        raise ExtractionFailed(f"모델이 거절했다: {part.refusal[:200]}")
        if response.output_parsed is None:
            raise ExtractionFailed("파싱된 결과가 없다")
        usage = response.usage
        return ExtractorResult(
            raw=response.output_parsed,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
        )
