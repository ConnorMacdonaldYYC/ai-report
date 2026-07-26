"""CLI entry point for the AI newsletter agent."""

import argparse
import asyncio
import logging

from src.config import get_settings
from src.orchestrator import run_report


def main() -> None:
    """Run the AI newsletter report pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the AI newsletter report pipeline."
    )
    parser.add_argument(
        "--send-email",
        action="store_true",
        help="Email the report after generating (requires SMTP config in .env)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    settings = get_settings()
    settings.configure()
    if args.send_email:
        settings.email_enabled = True

    result = asyncio.run(run_report(settings))

    print(f"\n{'='*60}")
    print(f"Report: {result.date_range}")
    print(f"Eval score: {result.eval_score:.2f} ({'PASS' if result.eval_passed else 'FAIL'})")
    print(f"Revisions: {result.revision_count}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()