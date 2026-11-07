from pathlib import Path

from pipa_statute_graph_rag.extraction.models import ExtractorResult, RawExtraction
from pipa_statute_graph_rag.extraction.pipeline import load_records, run_extraction
from pipa_statute_graph_rag.schema import Article


def make_article(number: int) -> Article:
    return Article(
        id=f"law:{number}",
        law="law",
        number=number,
        title="가나다",
        text=f"제{number}조(가나다)\n① 본문이다.",
        effective_date="20260911",
    )


class FakeExtractor:
    model = "fake"
    prompt_version = "vtest"
    prompt_sha256 = "abc"

    def __init__(self, failing: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.failing = failing or set()

    def extract(self, article: Article) -> ExtractorResult:
        self.calls.append(article.id)
        if article.id in self.failing:
            raise RuntimeError("호출 실패")
        return ExtractorResult(
            raw=RawExtraction(), input_tokens=10, output_tokens=5, reasoning_tokens=2
        )


def test_failure_of_one_article_does_not_stop_the_others(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    articles = [make_article(n) for n in (1, 2, 3)]
    saved, failures = run_extraction(articles, FakeExtractor({"law:2"}), out, workers=1)
    assert saved == 2
    assert [f[0] for f in failures] == ["law:2"]
    assert {r.article_id for r in load_records(out)} == {"law:1", "law:3"}


def test_rerun_processes_only_unsaved_articles(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    articles = [make_article(n) for n in (1, 2)]
    run_extraction(articles, FakeExtractor({"law:2"}), out, workers=1)
    retry = FakeExtractor()
    saved, failures = run_extraction(articles, retry, out, workers=1)
    assert retry.calls == ["law:2"]
    assert (saved, failures) == (1, [])
    assert len(load_records(out)) == 2


def test_record_keeps_token_usage(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    run_extraction([make_article(1)], FakeExtractor(), out, workers=1)
    record = load_records(out)[0]
    assert (record.input_tokens, record.output_tokens, record.reasoning_tokens) == (10, 5, 2)


def test_record_keeps_prompt_version_and_hash(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    run_extraction([make_article(1)], FakeExtractor(), out, workers=1)
    record = load_records(out)[0]
    assert (record.prompt_version, record.prompt_sha256) == ("extraction/vtest", "abc")
