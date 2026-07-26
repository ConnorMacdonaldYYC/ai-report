"""Live integration tests for the HackerNews tool against the real Algolia API.

These tests call the actual HN Search API to verify that:
- The request parameters are accepted (no 400 errors)
- The response shape matches what the tool expects
- The formatted output is usable

Marked with `@pytest.mark.integration` so they can be excluded from fast test runs
with: pytest -m "not integration"
"""

import httpx
import pytest

from src.tools.hackernews import hackernews_search, hackernews_top_comments

# A popular story with many comments — stable enough for integration tests.
WELL_KNOWN_STORY_ID = "48489163"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_returns_results() -> None:
    """Searching for a broad query should return formatted stories from the real API."""
    result = await hackernews_search("Python", min_score=10, days_back=365)

    assert "No HackerNews stories found" not in result
    assert "Title:" in result
    assert "Score:" in result
    assert "URL:" in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_numeric_filters_accepted() -> None:
    """The numericFilters parameter should not cause a 400 Bad Request.

    Regression test: the /api/v1/search endpoint rejects numericFilters,
    while /api/v1/search_by_date accepts them.
    """
    result = await hackernews_search("obscure_topic_xyz", min_score=9999, days_back=1)

    # Either results or the no-results message — both are fine.
    # The key assertion is that no httpx.HTTPStatusError was raised.
    assert isinstance(result, str)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_no_results() -> None:
    """A nonsense query should return the no-results message, not an error."""
    result = await hackernews_search(
        "zzzzz_nonexistent_topic_xyzzy", min_score=50, days_back=1
    )

    assert "No HackerNews stories found" in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_top_comments_returns_comments() -> None:
    """Fetching comments for a well-known story should return formatted text."""
    result = await hackernews_top_comments(WELL_KNOWN_STORY_ID, max_comments=3)

    assert result != "No comments found."
    assert result.startswith("- ")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_top_comments_nonexistent_story() -> None:
    """Fetching comments for a nonexistent story ID should raise an HTTP error."""
    with pytest.raises(httpx.HTTPStatusError):
        await hackernews_top_comments("00000000", max_comments=1)