import pytest

from pipa_graph_rag.classify.judges import JUDGES, resolve_min_support
from pipa_graph_rag.config import DECISIONS_MODEL, DEFAULT_JUDGE, JEV_MODEL, PPLX_MODEL


@pytest.fixture(autouse=True)
def api_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def test_default_judge_is_registered():
    assert DEFAULT_JUDGE in JUDGES


@pytest.mark.parametrize("name", list(JUDGES))
def test_missing_min_support_uses_the_judge_default(name):
    assert resolve_min_support(name, None) == JUDGES[name].min_support


@pytest.mark.parametrize("name", list(JUDGES))
def test_explicit_min_support_wins_including_zero(name):
    assert resolve_min_support(name, 0.3) == 0.3
    assert resolve_min_support(name, 0) == 0


def test_defaults_differ_by_judge_because_score_scales_differ():
    assert len({spec.min_support for spec in JUDGES.values()}) == len(JUDGES)


@pytest.mark.parametrize(
    ("name", "judge_name"),
    [
        ("decisions", f"decisions/{DECISIONS_MODEL}"),
        ("jev", f"jev/{JEV_MODEL}"),
        ("pplx", f"pplx/{PPLX_MODEL}"),
    ],
)
def test_build_creates_the_named_judge(name, judge_name):
    assert JUDGES[name].build().name == judge_name


def test_openrouter_judges_need_the_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    judge = JUDGES["jev"].build()
    with pytest.raises(KeyError, match="OPENROUTER_API_KEY"):
        judge.judge(_article(), [_item()])


def _article():
    from pipa_graph_rag.schema import Article

    return Article(
        id="law:9", law="law", number=9, title="제목", text="제9조(제목)\n① 본문이다.", effective_date="20260911"
    )


def _item():
    from pipa_graph_rag.classify.base import SupportItem

    return SupportItem(id="r0", statement="주장", quote="인용")
