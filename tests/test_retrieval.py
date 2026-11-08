import pytest

from pipa_graph_rag.retrieval import Mode, build_index, retrieve
from pipa_graph_rag.retrieval.bm25 import Bm25Index, tokenize
from pipa_graph_rag.retrieval.expand import Edge, expand
from pipa_graph_rag.schema import Article


def article(id: str, text: str) -> Article:
    law, rest = id.split(":")
    number, _, branch = rest.partition("-")
    return Article(
        id=id,
        law=law,
        number=int(number),
        branch=int(branch) if branch else None,
        title="제목",
        text=text,
        effective_date="20260911",
    )


def test_tokenize_makes_two_and_three_character_pieces_per_word():
    assert tokenize("개인정보 처리") == ["개인", "인정", "정보", "개인정", "인정보", "처리"]


def test_tokenize_drops_one_character_words_and_punctuation():
    assert tokenize("법, 는 (정보)") == ["정보"]


def test_bm25_ranks_the_article_that_shares_more_rare_pieces_first():
    index = Bm25Index(
        {
            "a": "정보주체는 열람을 요구할 수 있다",
            "b": "개인정보처리자는 개인정보를 파기하여야 한다",
            "c": "분쟁조정위원회의 구성",
        }
    )
    assert [doc for doc, _ in index.search("열람 요구", 3)] == ["a"]
    assert index.search("파기", 3)[0][0] == "b"


def test_bm25_returns_nothing_when_no_piece_matches():
    assert Bm25Index({"a": "개인정보"}).search("zzz", 3) == []


def test_bm25_limits_results_and_breaks_ties_by_id():
    index = Bm25Index({"b": "열람 요구", "a": "열람 요구"})
    assert [doc for doc, _ in index.search("열람", 1)] == ["a"]


def test_bm25_prefers_shorter_document_when_matches_equal():
    index = Bm25Index({"short": "열람 요구", "long": "열람 요구 " + "다른 내용 " * 20})
    assert index.search("열람", 2)[0][0] == "short"


def graph(edges: list[tuple[str, str, str]]):
    def find(ids: list[str]) -> list[Edge]:
        return [Edge(source=s, target=t, via=v) for s, t, v in edges if s in ids]

    return find


def test_expand_adds_neighbors_with_the_path_that_reached_them():
    result = expand(["a"], [graph([("a", "b", "IMPLEMENTS")])], depth=1)
    assert result.articles == ["a", "b"]
    assert result.reached["b"].source == "a"
    assert (result.reached["b"].via, result.reached["b"].depth) == ("IMPLEMENTS", 1)


def test_expand_stops_at_the_given_depth():
    find = graph([("a", "b", "X"), ("b", "c", "X"), ("c", "d", "X")])
    assert expand(["a"], [find], depth=1).articles == ["a", "b"]
    assert expand(["a"], [find], depth=2).articles == ["a", "b", "c"]


def test_expand_with_depth_zero_returns_only_seeds():
    assert expand(["a"], [graph([("a", "b", "X")])], depth=0).articles == ["a"]


def test_expand_does_not_revisit_and_keeps_the_first_path():
    find = graph([("a", "c", "FIRST"), ("a", "b", "X"), ("b", "c", "SECOND")])
    result = expand(["a"], [find], depth=2)
    assert result.reached["c"].via == "FIRST"
    assert result.articles.count("c") == 1


def test_expand_does_not_bring_back_seeds():
    result = expand(["a", "b"], [graph([("a", "b", "X")])], depth=1)
    assert result.articles == ["a", "b"] and result.reached == {}


def test_expand_combines_several_neighbor_functions():
    result = expand(["a"], [graph([("a", "b", "REF")]), graph([("a", "c", "개체:x")])], depth=1)
    assert result.articles == ["a", "b", "c"]


def test_retrieve_keyword_mode_does_not_expand():
    index = build_index([article("law:1", "열람 요구"), article("law:2", "파기")])
    result = retrieve(index, "열람", mode=Mode.KEYWORD, top_k=1, depth=2)
    assert result.expansion.articles == ["law:1"]


def test_retrieve_references_mode_expands_with_references_only():
    index = build_index([article("law:1", "열람 요구"), article("law:2", "파기")])
    result = retrieve(
        index,
        "열람",
        mode=Mode.REFERENCES,
        top_k=1,
        depth=1,
        references=graph([("law:1", "law:2", "REFERS_TO")]),
        concepts=graph([("law:1", "law:9", "개체:x")]),
    )
    assert result.expansion.articles == ["law:1", "law:2"]


def test_retrieve_concepts_mode_uses_both():
    index = build_index([article("law:1", "열람 요구")])
    result = retrieve(
        index,
        "열람",
        mode=Mode.REFERENCES_CONCEPTS,
        top_k=1,
        depth=1,
        references=graph([("law:1", "law:2", "REFERS_TO")]),
        concepts=graph([("law:1", "law:3", "개체:x")]),
    )
    assert result.expansion.articles == ["law:1", "law:2", "law:3"]


def test_retrieve_requires_the_neighbor_function_for_the_mode():
    index = build_index([article("law:1", "열람 요구")])
    with pytest.raises(ValueError):
        retrieve(index, "열람", mode=Mode.REFERENCES, top_k=1, depth=1)


def test_index_leaves_out_deleted_articles():
    index = build_index([article("law:8", "제8조 삭제 <2020.2.4>"), article("law:9", "삭제 요구")])
    assert [doc for doc, _ in index.search("삭제", 5)] == ["law:9"]
