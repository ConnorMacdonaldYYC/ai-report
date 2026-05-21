"""Report formatting and file writing for the AI newsletter."""

import os
from datetime import date

from src.config import Settings
from src.schemas import Source


def render_sources(sources: list[Source]) -> str:
    """Render a numbered sources section as markdown.

    Args:
        sources: List of Source objects to render.

    Returns:
        Markdown string with numbered source links.
    """
    if not sources:
        return ""

    lines = ["\n---\n\n## Sources\n"]
    for i, source in enumerate(sources, 1):
        lines.append(f"{i}. [{source.title}]({source.url})")
    return "\n".join(lines)


def format_report(
    report_markdown: str, date_range: str, sources: list[Source] | None = None
) -> str:
    """Format the final report with header, metadata, and sources.

    Args:
        report_markdown: The raw markdown content from the manager agent.
        date_range: The date range string for the report.
        sources: Optional list of Source objects to append as a sources section.

    Returns:
        Formatted markdown report string.
    """
    # Ensure the report starts with the correct header
    header = f"# AI Industry Weekly - {date_range}"

    lines = report_markdown.strip().split("\n")
    if lines and lines[0].startswith("# AI Industry Weekly"):
        lines[0] = header
        report_body = "\n".join(lines)
    else:
        report_body = f"{header}\n\n{report_markdown.strip()}"

    # Append sources section if provided
    if sources:
        report_body += render_sources(sources)

    return report_body


def write_report(
    report_markdown: str, date_range: str, settings: Settings,
    sources: list[Source] | None = None,
) -> str:
    """Write the report to a markdown file in the output directory.

    Args:
        report_markdown: The markdown content to write.
        date_range: The date range string for the report.
        settings: Application settings with output_dir configuration.
        sources: Optional list of Source objects to append as a sources section.

    Returns:
        The path to the written report file.
    """
    formatted = format_report(report_markdown, date_range, sources)

    # Ensure output directory exists
    os.makedirs(settings.output_dir, exist_ok=True)

    # Generate filename from date
    today = date.today()
    filename = f"ai-weekly-{today.strftime('%Y-%m-%d')}.md"
    filepath = os.path.join(settings.output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(formatted)

    return filepath