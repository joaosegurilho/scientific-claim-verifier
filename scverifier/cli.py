"""
Command line inteface for scverifier.

scverifier verify "claim" [--max-papers N] [--kb-only] [--skip-extraction-eval] [--use-all-propositions]
scverifier extract [paths...] (no args → demo paper)
scverifier webapp [--port N]
scverifier query "question"
scverifier bulk <file> [--output FILE] [--method ...]

"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="SciVerifier Command Line Interface")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug-level logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress all logging")

    parser.add_argument(
        "--kdb",
        type=Path,
        help="Set a new knowledge base location. (default: 'data/kb_all/')",
    )
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

    ## Bulk Parser
    bulk_parser = subparsers.add_parser("bulk", help="Verify claims from a CSV/JSONL file")
    bulk_parser.add_argument("file", type=Path, help="CSV or JSONL file with claims")
    bulk_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path (default: [file]_verified.jsonl)",
    )
    bulk_parser.add_argument(
        "--claim-col",
        type=int,
        default="1",
        help='The index of the claims column in the dataset (Default: "Claims")',
    )
    bulk_parser.add_argument(
        "--id-col",
        type=int,
        default=0,
        help="The index of the id column in the dataset (Default: None -> use 0)",
    )
    bulk_parser.add_argument(
        "--method",
        choices=["agent", "agentless", "agent_with_search", "agentless_with_search"],
        default="agentless_with_search",
        help="Verification method to employ.",
    )
    bulk_parser.add_argument(
        "--kb-only",
        action="store_true",
        help="Use only existing knowledge base data (don't search for new papers)",
    )
    bulk_parser.add_argument(
        "--skip-extraction-eval",
        action="store_true",
        help="Skip quality evaluation during proposition extraction (faster, accepts all propositions). Only applies when searching for new papers.",
    )
    bulk_parser.add_argument(
        "--use-all-propositions",
        action="store_true",
        help="Use all propositions instead of only quality ones during claim verification. Useful with --kb-only when quality evaluation was skipped during extraction.",
    )
    bulk_parser.add_argument(
        "--max-papers",
        type=int,
        default=10,
        help="Maximum number of papers to search for (default: 10)",
    )
    bulk_parser.add_argument("--max-items", type=int, help="Max items from the dataset to verify.")
    bulk_parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Resume a previous verification run. Requires --output pointing to previous output.",
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

    from scverifier.utils.logging_config import configure_logging

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    # cofigures logging for the entire script, pipelines can log to the same file with different loggers
    if args.verbose:
        verbose = 1
    elif args.quiet:
        verbose = -1
    else:
        verbose = 0
    configure_logging(verbose=verbose, log_file=logs_dir / f"{args.command}.log")

    from scverifier.config.settings import Config

    if args.kdb:
        Config.DB_NAME = str(args.kdb)

    print("\n" + "=" * 70)
    print(f"Doing {args.command.capitalize()}")
    if args.command != "benchmark":
        print(f"\tModel: {Config.LLM_MODEL}")
        print(f"\tKnowledge Base: {Config.DB_NAME}")
    print("=" * 70)

    if args.command == "verify":
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
        from scverifier.pipelines.extraction_pipeline import ExtractionPipeline

        paths_to_process = args.files if args.files else Path("data/demo_paper.pdf")
        print()
        print(f"Processing: {paths_to_process}")

        extraction_pipeline = ExtractionPipeline(arg_path=paths_to_process)
        extraction_pipeline(paths_to_process)

    elif args.command == "webapp":
        import uvicorn

        print(" Server: http://localhost:8000")
        print(" Documentation: http://localhost:8000/docs")
        print("\n" + "=" * 70 + "\n")

        uvicorn.run("scverifier.webapp.main:app", host=args.host, port=args.port, reload=True)

    elif args.command == "bulk":
        from scverifier.core.bulk_verifier import BulkVerifier

        print("\n" + "=" * 70)
        print(f" {'Resuming' if args.resume else 'Starting'} Bulk Verification")
        print(f" File: {args.file}")
        print("=" * 70)

        bulk = BulkVerifier(
            dataset_path=args.file,
            output_path=args.output,
            claim_column=args.claim_col,
            id_column=args.id_col,
            method=args.method,
        )
        bulk.run(
            kb_only=args.kb_only,
            skip_extraction_eval=args.skip_extraction_eval,
            use_all_propositions=args.use_all_propositions,
            max_papers=args.max_papers,
            max_items=args.max_items,
            resume=args.resume,
            pbar=True if args.quiet else False,
        )

        print("=" * 70)
        print("Verification done.")
        print(f" Saving at {bulk.output_path}")
        print("=" * 70)

    elif args.command == "query":
        raise NotImplementedError("Query interface not implemented yet")

    elif args.command == "benchmark":
        if not args.config:
            print("Error: --config is required for the benchmark command")
            sys.exit(1)

        from scverifier.pipelines.benchmark_pipeline import BenchmarkPipeline

        benchmark_pipeline = BenchmarkPipeline(Path(args.config), pbar=True if args.quiet else False)

        if args.dry_run:
            benchmark_pipeline.dry_run()
            sys.exit(0)

        benchmark_pipeline()

        print("=" * 70)
        print("Benchmark done.")
        print("=" * 70)

    else:
        raise ValueError(f"Missing command. Choose one of [{', '.join(list(subparsers.choices.keys()))}]")


if __name__ == "__main__":
    main()
