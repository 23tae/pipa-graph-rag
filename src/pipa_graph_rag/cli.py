import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .classify.judges import JUDGES, resolve_min_support
from .config import (
    COMPARISON_DIR,
    DATA_DIR,
    DEFAULT_JUDGE,
    DEFAULT_WORKERS,
    EMBEDDING_DIR,
    EMBEDDING_MODEL,
    EVALUATION_DIR,
    EXTRACTION_MODEL,
    EXTRACTION_PROMPT_VERSION,
    GRAPH_DIR,
    QUERY_DEPTH,
    QUERY_MAX_ENTITY_DEGREE,
    QUERY_MODE,
    QUERY_SEED,
    QUERY_TOP_K,
)
from .retrieval import Seed
from .retrieval.embedders import EMBEDDERS

EXTRACTION_MODEL_HELP = (
    "읽을 추출 결과의 모델 ID. 없으면 그 프롬프트 버전의 파일이 하나뿐일 때 그 파일을, 여럿이면 "
    f"{EXTRACTION_MODEL}의 파일을 쓴다"
)


def _resolve_extraction(model: str | None, version: str):
    from .extraction.pipeline import resolve_extraction_path

    try:
        return resolve_extraction_path(DATA_DIR / "extractions", version, model, EXTRACTION_MODEL)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="pipa-graph-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("collect", help="법제처 API로 본법과 시행령 조문을 받는다")

    references_cmd = sub.add_parser("references", help="조문 사이의 참조, 준용, 예외, 위임을 추출한다")
    references_cmd.add_argument(
        "--quoted-only",
        action="store_true",
        help='시행령이 "대통령령으로 정하는 …"을 따옴표로 인용한 경우에만 위임을 만든다',
    )

    extract_cmd = sub.add_parser("extract", help="LLM으로 조문의 개체와 관계를 추출한다")
    extract_cmd.add_argument(
        "--model", default=EXTRACTION_MODEL, help=f"OpenAI 모델 ID. 기본 {EXTRACTION_MODEL}"
    )
    extract_cmd.add_argument("--ids", nargs="+", help="추출할 조문 ID. 없으면 전체")
    extract_cmd.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    extract_cmd.add_argument(
        "--prompt-version",
        default=EXTRACTION_PROMPT_VERSION,
        help=f"extraction/prompts의 버전. 기본 {EXTRACTION_PROMPT_VERSION}",
    )

    judge_cmd = sub.add_parser("judge", help="추출한 관계마다 인용이 뒷받침하는지 판정한다")
    judge_cmd.add_argument("--extraction-model", default=None, help=EXTRACTION_MODEL_HELP)
    judge_cmd.add_argument(
        "--extraction-prompt-version",
        default=EXTRACTION_PROMPT_VERSION,
        help=f"판정할 추출 결과의 프롬프트 버전. 기본 {EXTRACTION_PROMPT_VERSION}",
    )
    judge_cmd.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    judge_cmd.add_argument(
        "--judge", choices=list(JUDGES), default=DEFAULT_JUDGE, help=f"판정기. 기본 {DEFAULT_JUDGE}"
    )

    graph_cmd = sub.add_parser("load-graph", help="조문, 조문 관계, 추출 결과를 Neo4j에 적재한다")
    graph_cmd.add_argument("--extraction-model", default=None, help=EXTRACTION_MODEL_HELP)
    graph_cmd.add_argument(
        "--prompt-version",
        default=EXTRACTION_PROMPT_VERSION,
        help=f"적재할 추출 결과의 프롬프트 버전. 기본 {EXTRACTION_PROMPT_VERSION}",
    )
    graph_cmd.add_argument("--reset", action="store_true", help="적재 전에 Neo4j의 모든 노드를 지운다")
    defaults = ", ".join(f"{name} {spec.min_support}" for name, spec in JUDGES.items())
    graph_cmd.add_argument(
        "--min-support",
        type=float,
        default=None,
        help=f"판정 확률이 이 값 이상인 개념 관계만 적재한다. 0이면 모두 적재한다. 없으면 판정기별 기본값({defaults})",
    )
    graph_cmd.add_argument(
        "--judge", choices=list(JUDGES), default=DEFAULT_JUDGE, help=f"--min-support에 쓸 판정 결과. 기본 {DEFAULT_JUDGE}"
    )

    export_cmd = sub.add_parser("export-graph", help="Neo4j 그래프를 JSON 파일로 내보낸다")
    export_cmd.add_argument("--out", type=Path, default=GRAPH_DIR / "graph.json")
    export_cmd.add_argument("--articles", nargs="+", help="이 조문과 관련 개체·관계만 내보낸다. 없으면 전체")

    import_cmd = sub.add_parser("import-graph", help="JSON 파일의 그래프를 Neo4j에 적재한다")
    import_cmd.add_argument("--file", type=Path, default=GRAPH_DIR / "graph.json")
    import_cmd.add_argument("--reset", action="store_true", help="적재 전에 Neo4j의 모든 노드를 지운다")

    ask_cmd = sub.add_parser("ask", help="질문에 가져온 조문만 근거로 답한다")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument(
        "--mode",
        choices=["keyword", "references", "references+concepts"],
        default=QUERY_MODE,
        help="keyword: 키워드 검색만, references: 조문 사이 관계로 확장, references+concepts: 개념 연결로도 확장",
    )
    ask_cmd.add_argument("--depth", type=int, default=QUERY_DEPTH, help="확장 단계 수")
    ask_cmd.add_argument("--top-k", type=int, default=QUERY_TOP_K, help="시드 조문 수")
    ask_cmd.add_argument(
        "--seed",
        choices=["bm25", "embedding"],
        default=QUERY_SEED,
        help="시드 조문을 고르는 방식. embedding은 질의마다 임베딩 API를 부르고 load-embeddings가 필요하다",
    )
    ask_cmd.add_argument(
        "--embedder",
        choices=list(EMBEDDERS),
        default=EMBEDDING_MODEL,
        help="--seed embedding에 쓸 임베딩 모델. load-embeddings로 넣은 모델과 같아야 한다",
    )
    ask_cmd.add_argument(
        "--max-entity-degree",
        type=int,
        default=QUERY_MAX_ENTITY_DEGREE,
        help="개념 연결 확장에서, 이 수 이상의 조문이 언급하는 개체는 건너뛴다. 0이면 건너뛰지 않는다",
    )
    ask_cmd.add_argument("--no-answer", action="store_true", help="조문만 찾고 답변은 만들지 않는다")

    sample_cmd = sub.add_parser("judge-sample", help="판정기 비교용 관계 표본을 판정 확률 구간별로 뽑는다")
    sample_cmd.add_argument("--extraction-model", default=None, help=EXTRACTION_MODEL_HELP)
    sample_cmd.add_argument("--per-stratum", type=int, default=30, help="구간(0.1 미만, 0.1~0.9, 0.9 이상)마다 뽑을 수")
    sample_cmd.add_argument("--seed", type=int, default=20261107)

    sub.add_parser("judge-compare", help="표본에 두 판정기를 돌려 라벨과 대조한다")

    embed_cmd = sub.add_parser("embed", help="조문과 평가 질문을 임베딩해 캐시에 저장한다(API를 부른다)")
    embed_cmd.add_argument("--embedder", choices=list(EMBEDDERS), default=EMBEDDING_MODEL)
    embed_cmd.add_argument(
        "--dry-run", action="store_true", help="API를 부르지 않고 보낼 요청 수와 글자 수만 출력한다"
    )

    load_embeddings_cmd = sub.add_parser(
        "load-embeddings", help="embed로 만든 조문 벡터를 Neo4j에 넣고 벡터 인덱스를 만든다(API를 부르지 않는다)"
    )
    load_embeddings_cmd.add_argument("--embedder", choices=list(EMBEDDERS), default=EMBEDDING_MODEL)
    load_embeddings_cmd.add_argument(
        "--reset", action="store_true", help="벡터를 넣지 않고 Neo4j의 벡터 인덱스와 조문 벡터를 지운다"
    )

    eval_cmd = sub.add_parser("evaluate-search", help="질문과 정답 조문으로 검색 설정을 비교한다")
    eval_cmd.add_argument("--answer", action="store_true", help="답변도 만들어 인용을 정답과 대조한다")
    eval_cmd.add_argument(
        "--conditions",
        nargs="+",
        help="돌릴 조건 이름. 모드/깊이/건너뛰기 기준 순서이고 예는 keyword, references/d1, "
        "references+concepts/d1/skip50이다. 없으면 모든 조건. --seeds나 --top-k를 주면 "
        "조건마다 그 조합으로 늘려 돌리고, 이름 뒤에 :시드(bm25는 생략)와 :k시드수(기본값이면 생략)가 붙는다",
    )
    eval_cmd.add_argument(
        "--seeds",
        nargs="+",
        choices=[seed.value for seed in Seed],
        default=[Seed.BM25.value],
        help="시드 조문을 고르는 방식. embedding과 hybrid는 embed로 만든 벡터가 필요하다",
    )
    eval_cmd.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[QUERY_TOP_K],
        help="시드 조문 수. 여러 값을 주면 각각 돌린다",
    )
    eval_cmd.add_argument(
        "--embedder", choices=list(EMBEDDERS), default=EMBEDDING_MODEL, help="embedding, hybrid 시드에 쓸 임베딩 모델"
    )
    eval_cmd.add_argument(
        "--vectors",
        choices=["memory", "neo4j"],
        default="memory",
        help="임베딩 시드를 계산하는 곳. memory는 캐시의 벡터로 메모리에서 정확히 계산하고, neo4j는 load-embeddings로 넣은 "
        "벡터 인덱스를 쓴다(질문 벡터는 캐시에서 읽는다). neo4j는 hybrid를 지원하지 않는다",
    )
    eval_cmd.add_argument("--questions", default="questions.jsonl", help=f"{EVALUATION_DIR}의 질문 파일 이름")
    eval_cmd.add_argument(
        "--out",
        help=f"{EVALUATION_DIR}에 저장할 결과 파일 이름. 질문 파일이나 시드 설정을 바꿀 때는 기존 결과를 덮어쓰지 않도록 꼭 준다",
    )
    eval_cmd.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

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
        from .llm.openai_client import OpenAIExtractor

        articles = load_articles(DATA_DIR / "articles.json")
        if args.ids:
            articles = [a for a in articles if a.id in set(args.ids)]
        extractor = OpenAIExtractor(args.model, args.prompt_version)
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

        version = args.extraction_prompt_version
        extraction_path, model = _resolve_extraction(args.extraction_model, version)
        extractions = {r.article_id: r.extraction for r in load_records(extraction_path)}
        judge = JUDGES[args.judge].build()
        out_path = DATA_DIR / "support" / f"{model}-{version}-{args.judge}.jsonl"
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
        from .graph.loader import connect, count_graph, load_graph
        from .graph.model import build_graph
        from .references import load_references

        version = args.prompt_version
        extraction_path, model = _resolve_extraction(args.extraction_model, version)
        extractions = [r.extraction for r in load_records(extraction_path)]
        min_support = resolve_min_support(args.judge, args.min_support)
        if min_support:
            from .classify.support import filter_by_support, load_support

            support = load_support(DATA_DIR / "support" / f"{model}-{version}-{args.judge}.jsonl")
            before = sum(len(e.relations) for e in extractions)
            extractions = filter_by_support(extractions, support, min_support)
            print(f"판정기 {args.judge}의 확률 {min_support} 이상만 적재: 개념 관계 {before} → "
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
        from .retrieval.embedders import MissingApiKey, live_query_embedder
        from .retrieval.neo4j_embedding import EmbeddingIndexMissing, Neo4jEmbeddingIndex

        seed = Seed(args.seed)
        driver = connect()
        try:
            embedding = None
            if seed == Seed.EMBEDDING:
                spec = EMBEDDERS[args.embedder]
                try:
                    embedding = Neo4jEmbeddingIndex(driver, spec.name, live_query_embedder(spec))
                except (MissingApiKey, EmbeddingIndexMissing) as error:
                    raise SystemExit(str(error)) from error
            engine = QueryEngine.from_graph(driver, embedding=embedding)
            result = engine.run(
                args.question,
                mode=Mode(args.mode),
                top_k=args.top_k,
                depth=args.depth,
                max_entity_degree=args.max_entity_degree or None,
                answer=not args.no_answer,
                seed=seed,
            )
        finally:
            driver.close()
        retrieval = result.retrieval
        expansion = retrieval.expansion
        print(f"가져온 조문 {len(expansion.articles)}개 (모드 {retrieval.mode.value}, 깊이 {retrieval.depth})")
        for article_id in expansion.articles:
            if article_id in retrieval.scores:
                how = (
                    f"임베딩 검색 {retrieval.scores[article_id]:.3f}"
                    if seed == Seed.EMBEDDING
                    else f"키워드 검색 {retrieval.scores[article_id]:.1f}"
                )
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

        version = EXTRACTION_PROMPT_VERSION
        extraction_path, model = _resolve_extraction(args.extraction_model, version)
        extractions = [r.extraction for r in load_records(extraction_path)]
        support = load_support(DATA_DIR / "support" / f"{model}-{version}-decisions.jsonl")
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
                write_jsonl(path, run_judge(JUDGES[name].build(), articles, sample))
            judgments = read_jsonl(path, Judgment)
            by_judge[name] = {j.key: j.probability for j in judgments}
            print(format_metrics(name, evaluate(labels, by_judge[name], strata), judgments))
        print(format_difference("decisions", "jev", compare_judges(labels, by_judge["decisions"], by_judge["jev"])))
    elif args.command == "embed":
        from .collect import load_articles
        from .evaluation import load_questions
        from .retrieval import indexed_articles
        from .retrieval.embedding import EmbeddingCache, cache_path, embed_missing, plan_embedding

        spec = EMBEDDERS[args.embedder]
        cache = EmbeddingCache(cache_path(EMBEDDING_DIR, spec.name))
        texts = [a.text for a in indexed_articles(load_articles(DATA_DIR / "articles.json"))]
        texts += [
            spec.format_query(q.question)
            for path in sorted(EVALUATION_DIR.glob("questions*.jsonl"))
            for q in load_questions(path)
        ]
        plan = plan_embedding(texts, cache)
        print(
            f"{spec.name}: 텍스트 {len(set(texts))}개 중 캐시에 없는 것 {len(plan.missing)}개, "
            f"요청 {plan.requests}번, {plan.characters:,}자"
        )
        if plan.missing and not args.dry_run:
            summary = embed_missing(spec.build(), cache, texts)
            tokens = "응답에 없음" if summary.tokens is None else f"{summary.tokens:,}토큰"
            print(f"요청 {summary.requests}번, 응답이 알려 준 입력 {tokens}")
            print(f"{cache.path}에 저장했다")
    elif args.command == "load-embeddings":
        from .graph.loader import connect
        from .graph.reader import read_articles
        from .retrieval import indexed_articles
        from .retrieval.embedding import EmbeddingCache, EmbeddingNotCached, cache_path
        from .retrieval.neo4j_embedding import load_embeddings, reset_embeddings

        driver = connect()
        try:
            if args.reset:
                reset_embeddings(driver)
                print("벡터 인덱스와 조문 벡터를 지웠다")
            else:
                spec = EMBEDDERS[args.embedder]
                cache = EmbeddingCache(cache_path(EMBEDDING_DIR, spec.name))
                try:
                    vectors = {a.id: cache.get(a.text) for a in indexed_articles(read_articles(driver))}
                except EmbeddingNotCached as error:
                    raise SystemExit(str(error)) from error
                loaded = load_embeddings(driver, vectors, spec.name)
                dimension = len(next(iter(vectors.values())))
                print(f"조문 {loaded}개의 벡터({dimension}차원, {spec.name})를 넣고 벡터 인덱스를 만들었다")
        finally:
            driver.close()
    elif args.command == "evaluate-search":
        from .evaluation import (
            bootstrap_difference,
            conditions,
            evaluate_all,
            format_summary,
            load_questions,
            summarize,
            summarize_by_type,
            with_seeds,
        )
        from .graph.loader import connect
        from .query import QueryEngine
        from .retrieval.embedding import EmbeddingCache, cache_path
        from .retrieval.neo4j_embedding import EmbeddingIndexMissing, Neo4jEmbeddingIndex

        seeds = [Seed(value) for value in args.seeds]
        default_run = args.questions == "questions.jsonl" and seeds == [Seed.BM25] and args.top_k == [QUERY_TOP_K]
        if args.out is None and not default_run:
            raise SystemExit("질문 파일이나 시드 설정을 바꾼 결과는 기존 결과를 덮어쓰지 않도록 --out으로 파일 이름을 준다")
        questions = load_questions(EVALUATION_DIR / args.questions)
        base_conditions = [c for c in conditions() if not args.conditions or c.name in args.conditions]
        selected = with_seeds(base_conditions, seeds, args.top_k)
        cache = (
            EmbeddingCache(cache_path(EMBEDDING_DIR, EMBEDDERS[args.embedder].name))
            if any(seed != Seed.BM25 for seed in seeds)
            else None
        )
        spec = EMBEDDERS[args.embedder]
        if args.vectors == "neo4j" and Seed.HYBRID in seeds:
            raise SystemExit("--vectors neo4j는 hybrid를 지원하지 않는다(벡터 인덱스는 상위 k개만 돌려준다)")
        driver = connect()
        try:
            embedding = None
            if cache is not None and args.vectors == "neo4j":
                try:
                    embedding = Neo4jEmbeddingIndex(driver, spec.name, lambda q: cache.get(spec.format_query(q)))
                except EmbeddingIndexMissing as error:
                    raise SystemExit(str(error)) from error
            results = evaluate_all(
                QueryEngine.from_graph(
                    driver, embedding_cache=cache, format_query=spec.format_query, embedding=embedding
                ),
                questions,
                selected,
                answer=args.answer,
                workers=args.workers,
            )
        finally:
            driver.close()
        name = args.out or ("results-answer.jsonl" if args.answer else "results.jsonl")
        (EVALUATION_DIR / name).write_text(
            "".join(r.model_dump_json() + "\n" for r in results), encoding="utf-8"
        )
        print(format_summary(summarize(results)))
        by_condition = {}
        for r in results:
            by_condition.setdefault(r.condition, {})[r.question_id] = r
        pairs = [
            ("keyword", "references/d1"),
            ("references/d1", "references+concepts/d1"),
            ("references/d1", "references+concepts/d1/skip50"),
            ("references+concepts/d1/skip50", "references+concepts/d1"),
        ]
        pairs += [  # 같은 조건과 시드 수에서 BM25 대비 다른 시드 검색의 차이
            (c.model_copy(update={"seed": Seed.BM25, "top_k": k}).name, c.model_copy(update={"seed": seed, "top_k": k}).name)
            for c in base_conditions
            for seed in seeds
            if seed != Seed.BM25
            for k in args.top_k
        ]
        for base_name, other in pairs:
            if base_name not in by_condition or other not in by_condition:
                continue
            metrics = ("recall_required", "precision") + (("citation_recall",) if args.answer else ())
            for metric in metrics:
                a = [getattr(by_condition[base_name][q.id], metric) for q in questions]
                b = [getattr(by_condition[other][q.id], metric) for q in questions]
                value, low, high = bootstrap_difference(a, b)
                print(f"{other} − {base_name} {metric}: {value:+.3f} [{low:+.3f}, {high:+.3f}]")
        for metric in ("recall_required",) + (("citation_recall",) if args.answer else ()):
            print(f"## 유형별 {metric}")
            for condition, by_type in summarize_by_type(results, questions, metric).items():
                print(f"{condition} | " + " | ".join(f"{t} {v:.3f}" for t, v in sorted(by_type.items())))
    elif args.command == "export-graph":
        from .graph.export import read_graph, save_graph, subset
        from .graph.loader import connect

        driver = connect()
        try:
            data = read_graph(driver)
        finally:
            driver.close()
        if args.articles:
            data = subset(data, set(args.articles))
        save_graph(data, args.out)
        print(
            f"{args.out}: 조문 {len(data.articles)}, 개체 {sum(map(len, data.entities.values()))}, "
            f"조문 관계 {sum(map(len, data.article_edges.values()))}, "
            f"개념 관계 {sum(map(len, data.concept_edges.values()))}"
        )
    elif args.command == "import-graph":
        from .graph.export import load_graph_file
        from .graph.loader import connect, count_graph, load_graph

        data = load_graph_file(args.file)
        driver = connect()
        try:
            load_graph(driver, data, reset=args.reset)
            nodes, edges = count_graph(driver)
        finally:
            driver.close()
        print(f"노드 {nodes}")
        print(f"관계 {edges}")
