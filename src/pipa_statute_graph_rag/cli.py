import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

DATA_DIR = Path("data/processed")
COMPARISON_DIR = Path("data/judge-comparison")
EVALUATION_DIR = Path("data/evaluation")


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

    ask_cmd = sub.add_parser("ask", help="질문에 가져온 조문만 근거로 답한다")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument(
        "--mode",
        choices=["keyword", "references", "references+concepts"],
        default="references+concepts",
        help="keyword: 키워드 검색만, references: 조문 사이 관계로 확장, references+concepts: 개념 연결로도 확장",
    )
    ask_cmd.add_argument("--depth", type=int, default=1, help="확장 단계 수")
    ask_cmd.add_argument("--top-k", type=int, default=3, help="키워드 검색으로 가져올 시드 조문 수")
    ask_cmd.add_argument(
        "--max-entity-degree",
        type=int,
        default=None,
        help="개념 연결 확장에서, 이 수 이상의 조문이 언급하는 개체는 건너뛴다. 없으면 건너뛰지 않는다",
    )
    ask_cmd.add_argument("--no-answer", action="store_true", help="조문만 찾고 답변은 만들지 않는다")

    sample_cmd = sub.add_parser("judge-sample", help="판정기 비교용 관계 표본을 판정 확률 구간별로 뽑는다")
    sample_cmd.add_argument("--model", required=True, help="추출 결과의 모델 ID")
    sample_cmd.add_argument("--per-stratum", type=int, default=30, help="구간(0.1 미만, 0.1~0.9, 0.9 이상)마다 뽑을 수")
    sample_cmd.add_argument("--seed", type=int, default=20261107)

    sub.add_parser("judge-compare", help="표본에 두 판정기를 돌려 라벨과 대조한다")

    eval_cmd = sub.add_parser("evaluate-search", help="질문과 정답 조문으로 검색 설정을 비교한다")
    eval_cmd.add_argument("--answer", action="store_true", help="답변도 만들어 인용을 정답과 대조한다")
    eval_cmd.add_argument("--conditions", nargs="+", help="돌릴 조건 이름. 없으면 모든 조건")
    eval_cmd.add_argument("--workers", type=int, default=4)

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
    elif args.command == "ask":
        from .graph.loader import connect
        from .query import QueryEngine
        from .retrieval import Mode

        driver = connect()
        try:
            engine = QueryEngine.from_graph(driver)
            result = engine.run(
                args.question,
                mode=Mode(args.mode),
                top_k=args.top_k,
                depth=args.depth,
                max_entity_degree=args.max_entity_degree,
                answer=not args.no_answer,
            )
        finally:
            driver.close()
        retrieval = result.retrieval
        expansion = retrieval.expansion
        print(f"가져온 조문 {len(expansion.articles)}개 (모드 {retrieval.mode.value}, 깊이 {retrieval.depth})")
        for article_id in expansion.articles:
            if article_id in retrieval.scores:
                how = f"키워드 검색 {retrieval.scores[article_id]:.1f}"
            else:
                reach = expansion.reached[article_id]
                how = f"{reach.source}에서 {reach.via} (깊이 {reach.depth})"
            print(f"  {article_id} {engine.articles[article_id].title} <- {how}")
        if result.answer:
            answer = result.answer
            print(f"\n답변: {answer.answer}")
            print(f"근거 조문: {', '.join(answer.citations) or '없음'}")
            if answer.invalid_citations:
                print(f"가져오지 않은 조문을 인용함: {', '.join(answer.invalid_citations)}")
            print(f"토큰 입력 {answer.input_tokens} / 출력 {answer.output_tokens}")
    elif args.command == "judge-sample":
        from .classify.comparison import flatten, probabilities_by_key, stratified_sample, write_jsonl
        from .classify.support import load_support
        from .extraction.pipeline import load_records
        from .extraction.prompt import LATEST_PROMPT_VERSION

        version = LATEST_PROMPT_VERSION
        extractions = [
            r.extraction for r in load_records(DATA_DIR / "extractions" / f"{args.model}-{version}.jsonl")
        ]
        support = load_support(DATA_DIR / "support" / f"{args.model}-{version}-decisions.jsonl")
        sample, strata = stratified_sample(
            flatten(extractions), probabilities_by_key(extractions, support), args.per_stratum, args.seed
        )
        write_jsonl(COMPARISON_DIR / "sample.jsonl", sample)
        (COMPARISON_DIR / "strata.json").write_text(
            json.dumps(strata, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"표본 {len(sample)}개를 {COMPARISON_DIR}에 저장했다(구간 정보는 strata.json)")
    elif args.command == "judge-compare":
        from .classify.comparison import (
            Judgment,
            Label,
            Relation,
            compare_judges,
            evaluate,
            format_difference,
            format_metrics,
            read_jsonl,
            run_judge,
            write_jsonl,
        )
        from .collect import load_articles

        sample = read_jsonl(COMPARISON_DIR / "sample.jsonl", Relation)
        strata = json.loads((COMPARISON_DIR / "strata.json").read_text(encoding="utf-8"))
        labels = {l.key: l.supported for l in read_jsonl(COMPARISON_DIR / "labels.jsonl", Label)}
        missing = [r.key for r in sample if r.key not in labels]
        if missing:
            raise SystemExit(f"라벨이 없는 관계 {len(missing)}개: {missing[:5]}")
        articles = {a.id: a for a in load_articles(DATA_DIR / "articles.json")}
        by_judge = {}
        for name in ("decisions", "jev"):
            path = COMPARISON_DIR / f"judgments-{name}.jsonl"
            if not path.exists():
                if name == "jev":
                    from .classify.jev import JevJudge

                    judge = JevJudge()
                else:
                    from .classify.decisions import DecisionsJudge

                    judge = DecisionsJudge()
                write_jsonl(path, run_judge(judge, articles, sample))
            judgments = read_jsonl(path, Judgment)
            by_judge[name] = {j.key: j.probability for j in judgments}
            print(format_metrics(name, evaluate(labels, by_judge[name], strata), judgments))
        print(format_difference("decisions", "jev", compare_judges(labels, by_judge["decisions"], by_judge["jev"])))
    elif args.command == "evaluate-search":
        from .evaluation import (
            bootstrap_difference,
            conditions,
            evaluate_all,
            format_summary,
            load_questions,
            summarize,
        )
        from .graph.loader import connect
        from .query import QueryEngine

        questions = load_questions(EVALUATION_DIR / "questions.jsonl")
        selected = [c for c in conditions() if not args.conditions or c.name in args.conditions]
        driver = connect()
        try:
            results = evaluate_all(
                QueryEngine.from_graph(driver), questions, selected, answer=args.answer, workers=args.workers
            )
        finally:
            driver.close()
        name = "results-answer.jsonl" if args.answer else "results.jsonl"
        (EVALUATION_DIR / name).write_text(
            "".join(r.model_dump_json() + "\n" for r in results), encoding="utf-8"
        )
        print(format_summary(summarize(results)))
        by_condition = {}
        for r in results:
            by_condition.setdefault(r.condition, {})[r.question_id] = r
        for base, other in (("keyword", "references/d1"), ("references/d1", "references+concepts/d1")):
            for metric in ("recall_required", "precision"):
                a = [getattr(by_condition[base][q.id], metric) for q in questions]
                b = [getattr(by_condition[other][q.id], metric) for q in questions]
                value, low, high = bootstrap_difference(a, b)
                print(f"{other} − {base} {metric}: {value:+.3f} [{low:+.3f}, {high:+.3f}]")
