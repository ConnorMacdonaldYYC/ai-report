"""Tests for the RSS reader tool."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.tools.rss_reader import _parse_entry_date, rss_reader


class TestParseEntryDate:
    """Tests for parsing feed entry dates."""

    def test_parse_published_parsed(self) -> None:
        """Should parse published_parsed date."""
        from time import struct_time

        parsed = struct_time((2026, 5, 15, 0, 0, 0, 0, 0, 0))
        entry = {"published_parsed": parsed}
        result = _parse_entry_date(entry)
        assert result == date(2026, 5, 15)

    def test_parse_updated_parsed(self) -> None:
        """Should fall back to updated_parsed."""
        from time import struct_time

        parsed = struct_time((2026, 5, 16, 0, 0, 0, 0, 0, 0))
        entry = {"updated_parsed": parsed}
        result = _parse_entry_date(entry)
        assert result == date(2026, 5, 16)

    def test_parse_no_date(self) -> None:
        """Should return None when no date is available."""
        entry: dict[str, object] = {}
        result = _parse_entry_date(entry)
        assert result is None


class TestRssReader:
    """Tests for the RSS reader function."""

    @pytest.mark.asyncio
    async def test_rss_reader_with_mock_feed(self) -> None:
        """Should read and format RSS feed entries."""
        from datetime import date, timedelta
        from time import struct_time

        recent = date.today() - timedelta(days=2)
        recent_struct = struct_time(
            (recent.year, recent.month, recent.day, 0, 0, 0, 0, 0, 0)
        )

        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.feed = {"title": "Test Blog"}
        mock_feed.entries = [
            {
                "title": "AI Breakthrough",
                "link": "https://example.com/breakthrough",
                "summary": "A major AI breakthrough was announced.",
                "published_parsed": recent_struct,
            },
        ]

        with patch("src.tools.rss_reader.feedparser.parse", return_value=mock_feed):
            result = await rss_reader(
                feeds=["https://example.com/feed.xml"],
                days_back=7,
            )

        assert "AI Breakthrough" in result
        assert "https://example.com/breakthrough" in result

    @pytest.mark.asyncio
    async def test_rss_reader_empty_feed(self) -> None:
        """Should handle empty feeds gracefully."""
        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.feed = {"title": "Empty Blog"}
        mock_feed.entries = []

        with patch("src.tools.rss_reader.feedparser.parse", return_value=mock_feed):
            result = await rss_reader(
                feeds=["https://example.com/empty.xml"],
                days_back=7,
            )

        assert "No recent posts found" in result

    @pytest.mark.asyncio
    async def test_rss_reader_filters_old_entries(self) -> None:
        """Should filter out entries older than days_back."""
        from time import struct_time

        old_date = date.today() - timedelta(days=30)
        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.feed = {"title": "Old Blog"}
        mock_feed.entries = [
            {
                "title": "Old Post",
                "link": "https://example.com/old",
                "summary": "An old post.",
                "published_parsed": struct_time(
                    (old_date.year, old_date.month, old_date.day, 0, 0, 0, 0, 0, 0)
                ),
            },
        ]

        with patch("src.tools.rss_reader.feedparser.parse", return_value=mock_feed):
            result = await rss_reader(
                feeds=["https://example.com/feed.xml"],
                days_back=7,
            )

        assert "Old Post" not in result