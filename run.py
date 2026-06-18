"""CLI entry point for the AI newsletter agent."""

import asyncio
import logging

from src.config import get_settings
from src.orchestrator import run_report


def main() -> None:
    """Run the AI newsletter report pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    settings = get_settings()
    settings.configure()
    result = asyncio.run(run_report(settings))

    print(f"\n{'='*60}")
    print(f"Report: {result.date_range}")
    print(f"Eval score: {result.eval_score:.2f} ({'PASS' if result.eval_passed else 'FAIL'})")
    print(f"Revisions: {result.revision_count}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()