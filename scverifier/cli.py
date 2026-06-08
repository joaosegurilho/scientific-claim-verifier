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

    ## Query Parser
    # query_parser = subparsers.add_parser("query", help="Query the knowledge base")

    ## Benchmark Parser
    # benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks")

    args = parser.parse_args()

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    # cofigures logging for the entire script, pipelines can log to the same file with different loggers
    configure_logging(verbose=args.verbose, log_file=logs_dir / f"{args.command}.log")

    if args.command == "verify":
        print("Verify command selected")
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
        print("Extract command selected")
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
    elif args.command == "query":
        raise NotImplementedError("Query interface not implemented yet")
    elif args.command == "benchmark":
        raise NotImplementedError("Benchmark is not implemented yet")


if __name__ == "__main__":
    main()
