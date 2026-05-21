"""Tests for the arXiv search tool."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.tools.arxiv_search import _parse_arxiv_response, arxiv_search

VALID_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
    <entry>
        <title>Test Paper</title>
        <summary>A test abstract.</summary>
        <published>2026-05-15T00:00:00Z</published>
        <id>http://arxiv.org/abs/0001.0001</id>
        <author><name>Author A</name></author>
    </entry>
</feed>"""


def _make_response(status_code: int, text: str = "") -> httpx.Response:
    """Build a fake httpx.Response with the given status and body."""
    return httpx.Response(
        status_code=status_code,
        request=httpx.Request("GET", "https://export.arxiv.org/api/query"),
        text=text,
    )


class TestParseArxivResponse:
    """Tests for parsing arXiv API XML responses."""

    def test_parse_valid_response(self) -> None:
        """Should parse a valid arXiv XML response."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Attention Is All You Need</title>
                <summary>We propose a new network architecture, the Transformer.</summary>
                <published>2026-05-15T00:00:00Z</published>
                <id>http://arxiv.org/abs/1706.03762v7</id>
                <author><name>Vaswani, Ashish</name></author>
            </entry>
        </feed>"""
        result = _parse_arxiv_response(xml)
        assert "Attention Is All You Need" in result
        assert "Vaswani, Ashish" in result
        assert "http://arxiv.org/abs/1706.03762v7" in result

    def test_parse_empty_response(self) -> None:
        """Should handle an empty feed."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>"""
        result = _parse_arxiv_response(xml)
        assert "No papers found" in result

    def test_parse_invalid_xml(self) -> None:
        """Should handle invalid XML gracefully."""
        result = _parse_arxiv_response("not xml at all")
        assert "Error" in result

    def test_parse_multiple_entries(self) -> None:
        """Should parse multiple entries and separate them."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Paper One</title>
                <summary>First paper abstract.</summary>
                <published>2026-05-15T00:00:00Z</published>
                <id>http://arxiv.org/abs/0001.0001</id>
                <author><name>Author A</name></author>
            </entry>
            <entry>
                <title>Paper Two</title>
                <summary>Second paper abstract.</summary>
                <published>2026-05-16T00:00:00Z</published>
                <id>http://arxiv.org/abs/0002.0002</id>
                <author><name>Author B</name></author>
            </entry>
        </feed>"""
        result = _parse_arxiv_response(xml)
        assert "Paper One" in result
        assert "Paper Two" in result
        assert "---" in result


class TestArxivSearchRetry:
    """Tests for retry/backoff logic in arxiv_search."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self) -> None:
        """Should return results immediately when the first request succeeds."""
        mock_get = AsyncMock(return_value=_make_response(200, VALID_XML))
        with (
            patch("httpx.AsyncClient.get", mock_get),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await arxiv_search("test query", max_results=5)

        assert "Test Paper" in result
        mock_get.assert_called_once()
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_retries_on_429_then_succeeds(self) -> None:
        """Should retry with backoff on 429 and return results once it succeeds."""
        responses = [
            _make_response(429),
            _make_response(200, VALID_XML),
        ]
        mock_get = AsyncMock(side_effect=responses)
        with (
            patch("httpx.AsyncClient.get", mock_get),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await arxiv_search("test query", max_results=5)

        assert "Test Paper" in result
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_on_500_then_succeeds(self) -> None:
        """Should retry with backoff on 5xx and return results once it succeeds."""
        responses = [
            _make_response(503),
            _make_response(200, VALID_XML),
        ]
        mock_get = AsyncMock(side_effect=responses)
        with (
            patch("httpx.AsyncClient.get", mock_get),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await arxiv_search("test query", max_results=5)

        assert "Test Paper" in result
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_runtime_error_on_exhausted_429_retries(self) -> None:
        """Should raise RuntimeError when all retries are exhausted due to 429s."""
        mock_get = AsyncMock(return_value=_make_response(429))
        with (
            patch("httpx.AsyncClient.get", mock_get),
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError, match="rate-limited"),
        ):
            await arxiv_search("test query", max_results=5)

    @pytest.mark.asyncio
    async def test_raises_http_error_on_exhausted_5xx_retries(self) -> None:
        """Should raise HTTPStatusError when all retries are exhausted due to 5xx."""
        mock_get = AsyncMock(return_value=_make_response(500))
        with (
            patch("httpx.AsyncClient.get", mock_get),
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await arxiv_search("test query", max_results=5)

    @pytest.mark.asyncio
    async def test_raises_immediately_on_4xx(self) -> None:
        """Should raise immediately on non-429/5xx client errors (e.g. 403)."""
        mock_get = AsyncMock(return_value=_make_response(403))
        with (
            patch("httpx.AsyncClient.get", mock_get),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            pytest.raises(httpx.HTTPStatusError),
        ):
            await arxiv_search("test query", max_results=5)

        mock_get.assert_called_once()
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_backoff_sleeps_increase_exponentially(self) -> None:
        """Should sleep with exponentially increasing backoff on repeated 429s."""
        responses = [
            _make_response(429),
            _make_response(429),
            _make_response(429),
            _make_response(200, VALID_XML),
        ]
        mock_get = AsyncMock(side_effect=responses)
        with (
            patch("httpx.AsyncClient.get", mock_get),
            patch("src.tools.arxiv_search.random.uniform", return_value=1.0),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await arxiv_search("test query", max_results=5)

        assert "Test Paper" in result
        # With jitter fixed at 1.0, backoff = 3.0 * 2^attempt * 1.0
        sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
        assert sleep_calls == [3.0, 6.0, 12.0]

    @pytest.mark.asyncio
    async def test_no_sleep_on_final_429_attempt(self) -> None:
        """Should not sleep after the final 429 — just raise immediately."""
        mock_get = AsyncMock(return_value=_make_response(429))
        with (
            patch("httpx.AsyncClient.get", mock_get),
            patch("src.tools.arxiv_search.random.uniform", return_value=1.0),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            pytest.raises(RuntimeError, match="rate-limited"),
        ):
            await arxiv_search("test query", max_results=5)

        # Should sleep MAX_RETRIES - 1 times (not on the final attempt)
        assert mock_sleep.call_count == 4