from pipa_statute_graph_rag.graph.export import load_graph_file, save_graph, subset
from pipa_statute_graph_rag.graph.model import GraphData


def graph() -> GraphData:
    return GraphData(
        articles=[{"id": "law:1", "title": "가"}, {"id": "decree:2", "title": "나"}, {"id": "law:3", "title": "다"}],
        entities={
            "Role": [{"id": "r1", "name": "처리자"}, {"id": "r2", "name": "위원회"}],
            "Deontic": [{"id": "d1", "name": "의무"}],
        },
        mentions={
            "Role": [
                {"article_id": "law:1", "entity_id": "r1"},
                {"article_id": "law:3", "entity_id": "r2"},
            ],
            "Deontic": [{"article_id": "law:1", "entity_id": "d1"}],
        },
        article_edges={
            "DELEGATES_TO": [
                {"source": "law:1", "target": "decree:2", "quote": "대통령령"},
                {"source": "law:1", "target": "law:3", "quote": "제3조"},
            ]
        },
        concept_edges={
            "BEARS": [
                {"source": "r1", "target": "d1", "article_id": "law:1", "quote": "하여야 한다"},
                {"source": "r2", "target": "d1", "article_id": "law:3", "quote": "해야 한다"},
            ]
        },
    )


def test_subset_keeps_only_the_given_articles_and_what_they_mention():
    kept = subset(graph(), {"law:1", "decree:2"})
    assert [a["id"] for a in kept.articles] == ["law:1", "decree:2"]
    assert kept.mentions == {
        "Role": [{"article_id": "law:1", "entity_id": "r1"}],
        "Deontic": [{"article_id": "law:1", "entity_id": "d1"}],
    }
    assert kept.entities == {"Role": [{"id": "r1", "name": "처리자"}], "Deontic": [{"id": "d1", "name": "의무"}]}


def test_subset_drops_edges_that_leave_the_given_articles():
    kept = subset(graph(), {"law:1", "decree:2"})
    assert kept.article_edges == {"DELEGATES_TO": [{"source": "law:1", "target": "decree:2", "quote": "대통령령"}]}
    assert [e["article_id"] for e in kept.concept_edges["BEARS"]] == ["law:1"]


def test_subset_of_no_known_article_is_empty():
    kept = subset(graph(), {"law:99"})
    assert (kept.articles, kept.entities, kept.mentions, kept.article_edges, kept.concept_edges) == ([], {}, {}, {}, {})


def test_save_and_load_round_trip_keeps_korean_text(tmp_path):
    path = tmp_path / "out" / "graph.json"
    save_graph(graph(), path)
    assert "처리자" in path.read_text(encoding="utf-8")
    assert load_graph_file(path) == graph()
