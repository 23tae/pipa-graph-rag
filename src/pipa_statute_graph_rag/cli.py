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
