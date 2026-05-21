"""Tests for the HackerNews search tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.hackernews import hackernews_search, hackernews_top_comments


class TestHackernewsSearch:
    """Tests for the HackerNews search function."""

    @pytest.mark.asyncio
    async def test_search_returns_formatted_results(self) -> None:
        """Should format HN search results correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hits": [
                {
                    "title": "New GPT-5 Release",
                    "url": "https://openai.com/gpt5",
                    "points": 150,
                    "num_comments": 75,
                    "objectID": "12345",
                },
                {
                    "title": "AI Regulation Debate",
                    "url": "https://example.com/regulation",
                    "points": 80,
                    "num_comments": 40,
                    "objectID": "67890",
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.hackernews.httpx.AsyncClient", return_value=mock_client):
            result = await hackernews_search("GPT-5", min_score=50)

        assert "New GPT-5 Release" in result
        assert "150" in result
        assert "75" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self) -> None:
        """Should handle no results gracefully."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"hits": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.hackernews.httpx.AsyncClient", return_value=mock_client):
            result = await hackernews_search("nonexistent topic")

        assert "No HackerNews stories found" in result


class TestHackernewsTopComments:
    """Tests for the HackerNews top comments function."""

    @pytest.mark.asyncio
    async def test_top_comments_returns_formatted(self) -> None:
        """Should format top comments correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "children": [
                {
                    "author": "user1",
                    "text": "<p>Great article about AI</p>",
                },
                {
                    "author": "user2",
                    "text": "<p>Interesting perspective</p>",
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.tools.hackernews.httpx.AsyncClient", return_value=mock_client):
            result = await hackernews_top_comments("12345")

        assert "user1" in result
        assert "Great article about AI" in result