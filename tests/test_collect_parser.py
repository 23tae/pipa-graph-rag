from pipa_graph_rag.collect.parser import parse_articles, strip_secrets

# API 응답 구조를 본뜬 합성 샘플이다. 실제 법령 문구가 아니다.
RAW = {
    "법령": {
        "조문": {
            "조문단위": [
                {"조문번호": "1", "조문내용": "제1장 총칙", "조문여부": "전문"},
                {
                    "조문번호": "2",
                    "조문시행일자": "20260911",
                    "조문내용": "제2조(정의)",
                    "조문제목": "정의",
                    "조문여부": "조문",
                    "항": {
                        "항내용": "① 이 법에서 사용하는 용어의 뜻은 다음과 같다.",
                        "호": [
                            {
                                "호내용": "1.   가나다",
                                "목": [{"목내용": [["가. 첫째", "   1) 둘째"]]}],
                            }
                        ],
                    },
                },
                {
                    "조문번호": "7",
                    "조문가지번호": "11",
                    "조문시행일자": "20260911",
                    "조문내용": "제7조의11(제척)",
                    "조문제목": "제척",
                    "조문여부": "조문",
                    "항": [{"항내용": "① 첫째 항"}, {"항내용": "② 둘째 항"}],
                },
                {
                    "조문번호": "9",
                    "조문시행일자": "20260911",
                    "조문내용": "제9조(항 없음) 본문이 조문내용에 있다.",
                    "조문제목": "항 없음",
                    "조문여부": "조문",
                },
            ]
        }
    }
}


def test_parse_skips_chapter_headings():
    articles = parse_articles(RAW, "law")
    assert [a.id for a in articles] == ["law:2", "law:7-11", "law:9"]


def test_parse_handles_dict_clause_and_nested_list_content():
    article = parse_articles(RAW, "law")[0]
    assert article.text.splitlines() == [
        "제2조(정의)",
        "① 이 법에서 사용하는 용어의 뜻은 다음과 같다.",
        "1. 가나다",
        "가. 첫째",
        "1) 둘째",
    ]


def test_parse_branch_number_and_label():
    article = parse_articles(RAW, "decree")[1]
    assert (article.id, article.branch, article.label) == ("decree:7-11", 11, "제7조의11")
    assert "② 둘째 항" in article.text


def test_parse_article_without_clauses_uses_heading_text():
    assert "본문이 조문내용에 있다" in parse_articles(RAW, "law")[2].text


def test_strip_secrets_removes_link_field_recursively():
    data = {"law": [{"법령명한글": "가", "법령상세링크": "/x?OC=secret"}], "법령상세링크": "y"}
    assert strip_secrets(data) == {"law": [{"법령명한글": "가"}]}
