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
