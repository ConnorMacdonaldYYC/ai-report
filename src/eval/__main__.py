"""CLI entry point for the evaluation framework.

Usage:
    python -m src.eval \
        --generators gpt-4o,claude-sonnet-4-6 \
        --evaluators gpt-4o-mini,claude-sonnet-4-6 \
        --seeds 3
"""

import argparse
import asyncio
import logging

from src.config import get_settings
from src.eval.matrix import run_matrix
from src.eval.results import MatrixResult

logger = logging.getLogger(__name__)


def _split_models(value: str) -> list[str]:
    """Split a comma-separated model list, stripping whitespace.

    Args:
        value: Comma-separated model names (e.g. "gpt-4o, claude-sonnet").

    Returns:
        List of non-empty model names.
    """
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> None:
    """Run the model evaluation matrix from the command line.

    Args:
        argv: Command-line arguments. Defaults to sys.argv.
    """
    parser = argparse.ArgumentParser(
        description="Run the model evaluation matrix (generators x evaluators x seeds)."
    )
    parser.add_argument(
        "--generators",
        required=True,
        help="Comma-separated generator model names to compare",
    )
    parser.add_argument(
        "--evaluators",
        required=True,
        help="Comma-separated evaluator model names (the judge panel)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Repeated runs per generator (default: 3)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Maximum concurrent agent runs (default: 3)",
    )
    parser.add_argument(
        "--date-range",
        default=None,
        help="Fixed date range label for all reports (default: last 7 days)",
    )
    parser.add_argument(
        "--output-dir",
        default="./eval_results",
        help="Root directory for results (default: ./eval_results)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    settings = get_settings()
    settings.configure()

    generators = _split_models(args.generators)
    evaluators = _split_models(args.evaluators)

    matrix: MatrixResult = asyncio.run(
        run_matrix(
            settings,
            generators,
            evaluators,
            args.seeds,
            concurrency=args.concurrency,
            output_dir=args.output_dir,
            date_range=args.date_range,
        )
    )

    print(f"\n{'='*60}")
    print(f"Matrix complete: {len(matrix.runs)} runs, {len(matrix.evals)} evals")
    print(f"Generators: {', '.join(matrix.generators)}")
    print(f"Evaluators: {', '.join(matrix.evaluators)}")
    print(f"Results in {args.output_dir}/<timestamp>/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
