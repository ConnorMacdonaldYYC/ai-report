"""HackerNews API tool for searching trending AI stories."""

import re
from datetime import date, datetime, timedelta

import httpx

from src.config import get_settings

_TAG_RE = re.compile(r"<[^>]+>")


async def hackernews_search(
    query: str,
    min_score: int | None = None,
    days_back: int = 7,
) -> str:
    """Search HackerNews for AI-related stories and discussions.

    Uses the Algolia HN Search API to find trending stories.

    Args:
        query: Search query string.
        min_score: Minimum story score threshold. Defaults to settings.hn_min_score.
        days_back: How many days back to search.

    Returns:
        Formatted string with story titles, URLs, scores, and top comments.
    """
    settings = get_settings()
    if min_score is None:
        min_score = settings.hn_min_score

    start_date = date.today() - timedelta(days=days_back)
    start_timestamp = int(datetime(start_date.year, start_date.month, start_date.day).timestamp())

    url = "https://hn.algolia.com/api/v1/search"
    params: dict[str, str | int] = {
        "query": query,
        "tags": "story",
        "numericFilters": f"points>{min_score},created_at_i>{start_timestamp}",
        "hitsPerPage": 20,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()

    data = response.json()
    hits = data.get("hits", [])

    if not hits:
        return f"No HackerNews stories found for '{query}'."

    results: list[str] = []
    for hit in hits[:10]:
        title = hit.get("title", "Unknown")
        url_link = hit.get("url", "")
        points = hit.get("points", 0)
        num_comments = hit.get("num_comments", 0)
        object_id = hit.get("objectID", "")
        hn_link = f"https://news.ycombinator.com/item?id={object_id}"

        url_display = url_link or hn_link
        entry = (
            f"Title: {title}\n"
            f"Score: {points} | Comments: {num_comments}\n"
            f"URL: {url_display}"
        )
        results.append(entry)

    return "\n---\n".join(results)


async def hackernews_top_comments(story_id: str, max_comments: int = 5) -> str:
    """Fetch top comments for a HackerNews story.

    Args:
        story_id: The HN story object ID.
        max_comments: Maximum number of comments to return.

    Returns:
        Formatted string with top comments.
    """
    url = f"https://hn.algolia.com/api/v1/items/{story_id}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()

    data = response.json()
    children = data.get("children", [])

    if not children:
        return "No comments found."

    comments: list[str] = []
    for child in children[:max_comments]:
        author = child.get("author", "Unknown")
        text = child.get("text", "")
        # Strip HTML tags for readability
        text = _TAG_RE.sub("", text).strip()
        if text:
            comments.append(f"- {author}: {text[:300]}")

    return "\n".join(comments) if comments else "No readable comments found."