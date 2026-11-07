import argparse
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="pipa-statute-graph-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    collect_cmd = sub.add_parser("collect", help="법제처 API로 본법과 시행령 조문을 받는다")
    collect_cmd.add_argument("--out", type=Path, default=Path("data/processed"))

    args = parser.parse_args()
    if args.command == "collect":
        from .collect import collect

        articles = collect(args.out)
        print(f"조문 {len(articles)}개를 {args.out}/articles.json에 저장했다")
