"""Tests for the output module."""

import os

from src.config import Settings
from src.output import format_report, render_sources, write_report
from src.schemas import Source


class TestRenderSources:
    """Tests for the render_sources function."""

    def test_render_sources_empty(self) -> None:
        """Should return empty string for no sources."""
        assert render_sources([]) == ""

    def test_render_sources_returns_markdown(self) -> None:
        """Should render sources as numbered markdown links."""
        sources = [
            Source(url="https://example.com/1", title="Source One", source_type="web_search"),
            Source(url="https://example.com/2", title="Source Two", source_type="arxiv"),
        ]
        result = render_sources(sources)
        assert "## Sources" in result
        assert "1. [Source One](https://example.com/1)" in result
        assert "2. [Source Two](https://example.com/2)" in result

    def test_render_sources_single(self) -> None:
        """Should handle a single source."""
        sources = [
            Source(url="https://example.com", title="Only Source", source_type="web_search"),
        ]
        result = render_sources(sources)
        assert "1. [Only Source](https://example.com)" in result


class TestFormatReport:
    """Tests for report formatting."""

    def test_format_report_adds_header(self) -> None:
        """Should add the correct header to the report."""
        markdown = "## Industry Overview\n\nSome content here."
        date_range = "May 11 - May 18, 2026"
        result = format_report(markdown, date_range)
        assert result.startswith(f"# AI Industry Weekly - {date_range}")

    def test_format_report_replaces_existing_header(self) -> None:
        """Should replace an existing header."""
        markdown = "# AI Industry Weekly - Old Date\n\n## Industry Overview\n\nContent."
        date_range = "May 11 - May 18, 2026"
        result = format_report(markdown, date_range)
        assert f"# AI Industry Weekly - {date_range}" in result
        assert "Old Date" not in result

    def test_format_report_preserves_content(self) -> None:
        """Should preserve the report content."""
        markdown = "## Industry Overview\n\n- Google announced Gemini 3.0"
        result = format_report(markdown, "May 11 - May 18, 2026")
        assert "Google announced Gemini 3.0" in result

    def test_format_report_with_sources(self) -> None:
        """Should append sources section when sources provided."""
        markdown = "## Industry Overview\n\nContent with citation [1]."
        date_range = "May 11 - May 18, 2026"
        sources = [
            Source(url="https://example.com", title="Example Article", source_type="web_search"),
        ]
        result = format_report(markdown, date_range, sources)
        assert "## Sources" in result
        assert "1. [Example Article](https://example.com)" in result

    def test_format_report_without_sources(self) -> None:
        """Should not include sources section when sources is None."""
        markdown = "## Industry Overview\n\nContent."
        result = format_report(markdown, "May 11 - May 18, 2026")
        assert "## Sources" not in result


class TestWriteReport:
    """Tests for report file writing."""

    def test_write_report_creates_file(self, tmp_path: str) -> None:
        """Should create a markdown file in the output directory."""
        settings = Settings(output_dir=str(tmp_path))
        markdown = "## Industry Overview\n\nContent here."
        date_range = "May 11 - May 18, 2026"

        filepath = write_report(markdown, date_range, settings)

        assert os.path.exists(filepath)
        assert filepath.endswith(".md")

        with open(filepath) as f:
            content = f.read()
        assert "AI Industry Weekly" in content

    def test_write_report_creates_output_dir(self, tmp_path: str) -> None:
        """Should create the output directory if it doesn't exist."""
        output_dir = os.path.join(str(tmp_path), "nested", "output")
        settings = Settings(output_dir=output_dir)
        markdown = "## Content"
        date_range = "May 11 - May 18, 2026"

        filepath = write_report(markdown, date_range, settings)

        assert os.path.exists(filepath)

    def test_write_report_with_sources(self, tmp_path: str) -> None:
        """Should include sources section when sources provided."""
        settings = Settings(output_dir=str(tmp_path))
        markdown = "## Industry Overview\n\nContent with citation [1]."
        date_range = "May 11 - May 18, 2026"
        sources = [
            Source(url="https://example.com", title="Example Source", source_type="web_search"),
        ]
        filepath = write_report(markdown, date_range, settings, sources)

        assert os.path.exists(filepath)
        with open(filepath) as f:
            content = f.read()
        assert "## Sources" in content
        assert "1. [Example Source](https://example.com)" in content

    def test_write_report_without_sources(self, tmp_path: str) -> None:
        """Should not include sources section when sources is None."""
        settings = Settings(output_dir=str(tmp_path))
        markdown = "## Content"
        date_range = "May 11 - May 18, 2026"
        filepath = write_report(markdown, date_range, settings)

        assert os.path.exists(filepath)
        with open(filepath) as f:
            content = f.read()
        assert "## Sources" not in content