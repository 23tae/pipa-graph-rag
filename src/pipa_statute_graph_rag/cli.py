import argparse
from pathlib import Path

from dotenv import load_dotenv

DATA_DIR = Path("data/processed")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="pipa-statute-graph-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("collect", help="법제처 API로 본법과 시행령 조문을 받는다")

    references_cmd = sub.add_parser("references", help="조문 사이의 참조, 준용, 예외, 위임을 추출한다")
    references_cmd.add_argument(
        "--quoted-only",
        action="store_true",
        help='시행령이 "대통령령으로 정하는 …"을 따옴표로 인용한 경우에만 위임을 만든다',
    )

    extract_cmd = sub.add_parser("extract", help="LLM으로 조문의 개체와 관계를 추출한다")
    extract_cmd.add_argument("--model", required=True, help="OpenAI 모델 ID")
    extract_cmd.add_argument("--ids", nargs="+", help="추출할 조문 ID. 없으면 전체")
    extract_cmd.add_argument("--workers", type=int, default=4)
    extract_cmd.add_argument(
        "--prompt-version", default=None, help="extraction/prompts의 버전. 없으면 최신 버전"
    )

    judge_cmd = sub.add_parser("judge", help="추출한 관계마다 인용이 뒷받침하는지 판정한다")
    judge_cmd.add_argument("--model", required=True, help="판정할 추출 결과의 모델 ID")
    judge_cmd.add_argument(
        "--extraction-prompt-version", default=None, help="판정할 추출 결과의 프롬프트 버전. 없으면 최신 버전"
    )
    judge_cmd.add_argument("--workers", type=int, default=4)
    judge_cmd.add_argument(
        "--judge", choices=["decisions", "jev"], default="decisions", help="판정기. 기본은 OpenAI Decisions API"
    )

    graph_cmd = sub.add_parser("load-graph", help="조문, 조문 관계, 추출 결과를 Neo4j에 적재한다")
    graph_cmd.add_argument("--model", required=True, help="적재할 추출 결과의 모델 ID")
    graph_cmd.add_argument(
        "--prompt-version", default=None, help="적재할 추출 결과의 프롬프트 버전. 없으면 최신 버전"
    )
    graph_cmd.add_argument("--reset", action="store_true", help="적재 전에 Neo4j의 모든 노드를 지운다")
    graph_cmd.add_argument(
        "--min-support",
        type=float,
        default=None,
        help="판정 확률이 이 값 이상인 개념 관계만 적재한다. 없으면 모두 적재한다",
    )
    graph_cmd.add_argument(
        "--judge", choices=["decisions", "jev"], default="decisions", help="--min-support에 쓸 판정 결과"
    )

    args = parser.parse_args()
    if args.command == "collect":
        from .collect import collect

        articles = collect(DATA_DIR)
        print(f"조문 {len(articles)}개를 {DATA_DIR}/articles.json에 저장했다")
    elif args.command == "references":
        from .collect import load_articles
        from .references import build_references, save_references

        references = build_references(
            load_articles(DATA_DIR / "articles.json"), quoted_only=args.quoted_only
        )
        save_references(references, DATA_DIR / "references.json")
        print(f"조문 관계 {len(references)}개를 {DATA_DIR}/references.json에 저장했다")
    elif args.command == "extract":
        from .collect import load_articles
        from .extraction.pipeline import load_records, run_extraction
        from .extraction.prompt import LATEST_PROMPT_VERSION
        from .llm.openai_client import OpenAIExtractor

        articles = load_articles(DATA_DIR / "articles.json")
        if args.ids:
            articles = [a for a in articles if a.id in set(args.ids)]
        extractor = OpenAIExtractor(args.model, args.prompt_version or LATEST_PROMPT_VERSION)
        out_path = DATA_DIR / "extractions" / f"{args.model}-{extractor.prompt_version}.jsonl"
        saved, failures = run_extraction(articles, extractor, out_path, workers=args.workers)
        records = load_records(out_path)
        proposed = sum(r.proposed for r in records)
        accepted = sum(len(r.extraction.relations) for r in records)
        print(
            f"새로 저장 {saved}, 실패 {len(failures)}, 누적 조문 {len(records)}, "
            f"관계 제안 {proposed} / 통과 {accepted} / 거부 {proposed - accepted}, "
            f"토큰 입력 {sum(r.input_tokens for r in records)} / 출력 {sum(r.output_tokens for r in records)}"
        )
        for article_id, error in failures:
            print(f"실패 {article_id}: {error}")
    elif args.command == "judge":
        from .classify.support import load_support, run_judgment
        from .collect import load_articles
        from .extraction.pipeline import load_records
        from .extraction.prompt import LATEST_PROMPT_VERSION

        version = args.extraction_prompt_version or LATEST_PROMPT_VERSION
        extractions = {
            r.article_id: r.extraction
            for r in load_records(DATA_DIR / "extractions" / f"{args.model}-{version}.jsonl")
        }
        if args.judge == "jev":
            from .classify.jev import JevJudge

            judge = JevJudge()
        else:
            from .classify.decisions import DecisionsJudge

            judge = DecisionsJudge()
        out_path = DATA_DIR / "support" / f"{args.model}-{version}-{args.judge}.jsonl"
        saved, failures = run_judgment(
            load_articles(DATA_DIR / "articles.json"), extractions, judge, out_path, workers=args.workers
        )
        records = load_support(out_path)
        scores = [p for r in records for p in r.probabilities]
        print(
            f"새로 저장 {saved}, 실패 {len(failures)}, 누적 조문 {len(records)}, 관계 {len(scores)}, "
            f"입력 토큰 {sum(r.input_tokens for r in records)}"
        )
        for edge in (0.1, 0.5, 0.9):
            print(f"  확률 {edge} 미만: {sum(p < edge for p in scores)}")
        for article_id, error in failures:
            print(f"실패 {article_id}: {error}")
    elif args.command == "load-graph":
        from .collect import load_articles
        from .extraction.pipeline import load_records
        from .extraction.prompt import LATEST_PROMPT_VERSION
        from .graph.loader import connect, count_graph, load_graph
        from .graph.model import build_graph
        from .references import load_references

        version = args.prompt_version or LATEST_PROMPT_VERSION
        records = load_records(DATA_DIR / "extractions" / f"{args.model}-{version}.jsonl")
        extractions = [r.extraction for r in records]
        if args.min_support is not None:
            from .classify.support import filter_by_support, load_support

            support = load_support(DATA_DIR / "support" / f"{args.model}-{version}-{args.judge}.jsonl")
            before = sum(len(e.relations) for e in extractions)
            extractions = filter_by_support(extractions, support, args.min_support)
            print(f"판정 확률 {args.min_support} 이상만 적재: 개념 관계 {before} → "
                  f"{sum(len(e.relations) for e in extractions)}")
        data = build_graph(
            load_articles(DATA_DIR / "articles.json"),
            load_references(DATA_DIR / "references.json"),
            extractions,
        )
        driver = connect()
        try:
            load_graph(driver, data, reset=args.reset)
            nodes, edges = count_graph(driver)
        finally:
            driver.close()
        print(f"제외 {data.skipped}")
        print(f"노드 {nodes}")
        print(f"관계 {edges}")
