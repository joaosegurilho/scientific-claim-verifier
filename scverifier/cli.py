"""
Command line inteface for scverifier.

scverifier verify "claim" [--max-papers N] [--kb-only] [--skip-extraction-eval] [--use-all-propositions]
scverifier extract [paths...] (no args → demo paper)
scverifier webapp [--port N]
scverifier query "question"
scverifier benchmark run [dataset] [--max-items N]

"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scverifier.config.settings import Config
from scverifier.core.benchmarking import VerificationMethod
from scverifier.utils.logging_config import configure_logging


def main():
    parser = argparse.ArgumentParser(description="SciVerifier Command Line Interface")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug-level logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ## Verification Parser
    verify_parser = subparsers.add_parser("verify", help="Run the claim verification pipeline")
    verify_parser.add_argument("claim", help="Scientific claim to verify")
    verify_parser.add_argument(
        "--max-papers",
        type=int,
        default=30,
        help="Maximum number of papers to search for (default: 30)",
    )
    verify_parser.add_argument(
        "--kb-only",
        action="store_true",
        help="Use only existing knowledge base data (don't search for new papers)",
    )
    verify_parser.add_argument(
        "--skip-extraction-eval",
        action="store_true",
        help="Skip quality evaluation during proposition extraction (faster, accepts all propositions). Only applies when searching for new papers.",
    )
    verify_parser.add_argument(
        "--use-all-propositions",
        action="store_true",
        help="Use all propositions instead of only quality ones during claim verification. Useful with --kb-only when quality evaluation was skipped during extraction.",
    )

    ## Extraction Parser
    extract_parser = subparsers.add_parser("extract", help="Run the proposition extraction pipeline")
    extract_parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="File(s) or folder(s) to process (default: data/demo_paper.pdf)",
    )

    ## Web App Parser
    webapp_parser = subparsers.add_parser("webapp", help="Run the web application")
    webapp_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the web application on (default: 8000)",
    )
    webapp_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")

    ## Dataset Parser
    dataset_parser = subparsers.add_parser("dataset", help="Run a dataset through the claim verification pipeline")
    dataset_parser.add_argument("file", type=Path, help="Dataset to verify")
    dataset_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output of verified dataset. (default: [dataset_path]_verified.[dataset_ext])",
    )
    dataset_parser.add_argument(
        "--claim-col",
        type=int,
        default="1",
        help='The index of the claims column in the dataset (Default: "Claims")',
    )
    dataset_parser.add_argument(
        "--id-col",
        type=int,
        default=0,
        help="The index of the id column in the dataset (Default: None -> use 0)",
    )
    dataset_parser.add_argument(
        "--method",
        choices=list(VerificationMethod),
        default=VerificationMethod.AGENTLESS_WITH_SEARCH,
        help="Verification method to employ.",
    )
    dataset_parser.add_argument(
        "--kb-only",
        action="store_true",
        help="Use only existing knowledge base data (don't search for new papers)",
    )
    dataset_parser.add_argument(
        "--skip-extraction-eval",
        action="store_true",
        help="Skip quality evaluation during proposition extraction (faster, accepts all propositions). Only applies when searching for new papers.",
    )
    dataset_parser.add_argument(
        "--use-all-propositions",
        action="store_true",
        help="Use all propositions instead of only quality ones during claim verification. Useful with --kb-only when quality evaluation was skipped during extraction.",
    )
    dataset_parser.add_argument(
        "--max-papers",
        type=int,
        default=10,
        help="Maximum number of papers to search for (default: 10)",
    )
    dataset_parser.add_argument("--max-items", type=int, help="Max items from the dataset to verify.")
    dataset_parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Resume a previous dataset verification pipeline. Requires --output to be the output of a previous interrupted run.",
    )

    ## Query Parser
    # query_parser = subparsers.add_parser("query", help="Query the knowledge base")

    ## Benchmark Parser
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks from YAML config")
    benchmark_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to YAML experiment config file",
    )
    benchmark_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print plan without running",
    )

    args = parser.parse_args()

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    # cofigures logging for the entire script, pipelines can log to the same file with different loggers
    configure_logging(verbose=args.verbose, log_file=logs_dir / f"{args.command}.log")

    if args.command == "verify":
        print("\n" + "=" * 70)
        print(" Starting Verification.")
        print("=" * 70)
        from scverifier.core.knowledge.knowledge_base import KnowledgeBase
        from scverifier.pipelines.verification_pipeline import VerificationPipeline

        print("\n Initializing knowledge base...")
        kb = KnowledgeBase()

        verification_pipeline = VerificationPipeline(kb=kb)
        verification_pipeline(
            args.claim,
            args.max_papers,
            args.kb_only,
            args.skip_extraction_eval,
            args.use_all_propositions,
        )

    elif args.command == "extract":
        print("\n" + "=" * 70)
        print(" Starting Extraction.")
        print("=" * 70)

        from scverifier.pipelines.extraction_pipeline import ExtractionPipeline

        paths_to_process = args.files if args.files else Path("data/demo_paper.pdf")

        extraction_pipeline = ExtractionPipeline(arg_path=paths_to_process)
        extraction_pipeline(paths_to_process)

    elif args.command == "webapp":
        import uvicorn

        print("\n" + "=" * 70)
        print(" Starting Scientific Claim Verification Web Application")
        print("=" * 70)
        # print(f"\n Knowledge Base: {len(kb.papers)} papers loaded")
        print(" Server: http://localhost:8000")
        print(" Documentation: http://localhost:8000/docs")
        print("\n" + "=" * 70 + "\n")

        uvicorn.run("scverifier.webapp.main:app", host=args.host, port=args.port, reload=True)
    elif args.command == "dataset":
        from scverifier.core.dataset_verifier import DatasetVerifier

        print("\n" + "=" * 70)
        print(f" {'Resuming' if args.resume else 'Starting'} Dataset Verification")
        print(f" Dataset at: {args.file}")
        print("=" * 70)

        d_verifier = DatasetVerifier(
            dataset_path=args.file,
            output_path=args.output,
            claim_column=args.claim_col,
            id_column=args.id_col,
            method=args.method,
        )
        d_verifier.run(
            kb_only=args.kb_only,
            skip_extraction_eval=args.skip_extraction_eval,
            use_all_propositions=args.use_all_propositions,
            max_papers=args.max_papers,
            max_items=args.max_items,
            resume=args.resume,
        )

        print("=" * 70)
        print("Verification done.")
        print(f" Saving at {d_verifier.output_path}")
        print("=" * 70)

    elif args.command == "query":
        raise NotImplementedError("Query interface not implemented yet")

    elif args.command == "benchmark":
        if not args.config:
            print("Error: --config is required for the benchmark command")
            sys.exit(1)

        import yaml

        from scverifier.config.override import apply_experiment_config
        from scverifier.config.yaml_config import build_experiment_config
        from scverifier.core.benchmarking import get_benchmark
        from scverifier.core.benchmarking.run_benchmark import BenchmarkRunner

        config_path = Path(args.config)
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        parsed_config = build_experiment_config(raw_config)
        unknown = set(parsed_config.features) - Config.KNOWN_FEAUTURES
        if unknown:
            raise ValueError(f"Unknown features: {unknown}. Known: {Config.KNOWN_FEAUTURES}")

        if args.dry_run:
            from datetime import datetime

            from scverifier.core.benchmarking import BENCHMARK_MAP, METHOD_MAP

            counter_file = Path("experiments/.counter")
            next_counter = int(counter_file.read_text().strip()) + 1 if counter_file.exists() else 1
            run_name = parsed_config.experiment.name or "unnamed"
            output_dir = f"experiments/results/exp_{next_counter:03d}_{run_name}_{datetime.now().strftime('%Y%m%d')}"

            print("Dry run — experiment plan:")
            print(f"  Name:        {parsed_config.experiment.name or 'Unnamed'}")
            print(f"  Description: {parsed_config.experiment.description or '-'}")
            print(f"  Agent model: {parsed_config.model.agent_model or '(default)'}")
            print(f"  Embedding:   {parsed_config.model.embedding_model or '(default)'}")
            print(f"  KB:          {parsed_config.kb or '(default)'}")
            print(f"  Features:    {parsed_config.features or 'none'}")
            print(f"  Output:      {output_dir}")
            print(f"  Benchmarks:  {len(parsed_config.benchmark)}")
            for item in parsed_config.benchmark:
                print(f"    - {item.dataset}/{item.method} split={item.split or 'all'}")
                benchmark_cls = BENCHMARK_MAP[item.dataset]
                vm = METHOD_MAP[item.method]
                try:
                    if item.dataset == "scifact":
                        bm = benchmark_cls(split=item.split, verification_method=vm)
                    else:
                        bm = benchmark_cls(verification_method=vm)
                    bm.load(max_items=1)
                    print("      ✓ data available")
                except FileNotFoundError as e:
                    print(f"      ✗ data not found: {e}")
                except Exception as e:
                    print(f"      ! load error: {e}")
            sys.exit(0)

        experiments_dir = Path("experiments")
        counter_file = experiments_dir / ".counter"
        counter_file.parent.mkdir(parents=True, exist_ok=True)
        if not counter_file.exists():
            counter_file.write_text("0")
        counter = int(counter_file.read_text().strip()) + 1
        counter_file.write_text(str(counter))

        apply_experiment_config(parsed_config)

        from datetime import datetime

        run_name = parsed_config.experiment.name or "unnamed"
        date_str = datetime.now().strftime("%Y%m%d")
        output_dir = experiments_dir / "results" / f"exp_{counter:03d}_{run_name}_{date_str}"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(raw_config, f)

        with open(output_dir / "settings_snapshot.yaml", "w") as f:
            yaml.dump(
                {k: getattr(Config, k) for k in dir(Config) if k.isupper() and not k.startswith("_")},
                f,
            )

        for bm_item in parsed_config.benchmark:
            print(f"\nBenchmark: {bm_item.dataset}/{bm_item.method} (split={bm_item.split or 'all'})")
            benchmark = get_benchmark(bm_item.dataset, bm_item.method, bm_item.split)
            runner = BenchmarkRunner(
                benchmark=benchmark,
                results_dir=output_dir,
                method=METHOD_MAP[bm_item.method],
                run_dir=output_dir / f"{bm_item.dataset}_{bm_item.method}",
            )
            runner.run(max_papers=bm_item.max_papers or 10)
    else:
        raise ValueError(f"Missing command. Choose one of [{', '.join(list(subparsers.choices.keys()))}]")


if __name__ == "__main__":
    main()
