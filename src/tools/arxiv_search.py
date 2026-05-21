"""arXiv API tool for searching recent AI/ML papers."""

import asyncio
import logging
import random
from datetime import date, timedelta

import httpx

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 3.0


async def arxiv_search(
    query: str,
    max_results: int = 10,
    days_back: int = 7,
    categories: list[str] | None = None,
) -> str:
    """Search arXiv for recent AI/ML papers.

    Args:
        query: Search query string (e.g. "transformer", "reinforcement learning").
        max_results: Maximum number of results to return.
        days_back: How many days back to search.
        categories: arXiv categories to filter by. Defaults to cs.AI, cs.LG, cs.CL.

    Returns:
        Formatted string with paper titles, authors, abstracts, and links.
    """
    if categories is None:
        categories = ["cs.AI", "cs.LG", "cs.CL"]

    # Build the search query with category filter
    cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
    full_query = f"({cat_query}) AND ({query})"

    # Calculate date range
    start_date = date.today() - timedelta(days=days_back)
    today_str = date.today().strftime("%Y%m%d")
    start_str = start_date.strftime("%Y%m%d")
    date_filter = f"submittedDate:[{start_str}0000 TO {today_str}2359]"

    url = "https://export.arxiv.org/api/query"
    params: dict[str, str | int] = {
        "search_query": f"{full_query} AND {date_filter}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    last_exception: Exception | None = None
    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(MAX_RETRIES):
            response = await client.get(url, params=params)

            if response.status_code == 429:
                if attempt == MAX_RETRIES - 1:
                    msg = "arXiv API rate-limited: all retries exhausted"
                    raise RuntimeError(msg)
                backoff = INITIAL_BACKOFF_SECONDS * (2 ** attempt) * random.uniform(0.5, 1.5)
                logger.warning(
                    "arXiv API rate-limited (429), retrying in %.1fs (attempt %d/%d)",
                    backoff,
                    attempt + 1,
                    MAX_RETRIES,
                )
                await asyncio.sleep(backoff)
                continue

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                last_exception = exc
                if exc.response.status_code >= 500:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    backoff = INITIAL_BACKOFF_SECONDS * (2 ** attempt) * random.uniform(0.5, 1.5)
                    logger.warning(
                        "arXiv API server error (%d), retrying in %.1fs (attempt %d/%d)",
                        exc.response.status_code,
                        backoff,
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise

            return _parse_arxiv_response(response.text)

    # All retries exhausted for 5xx — raise the last error
    if last_exception:
        raise last_exception
    msg = "arXiv API: all retries exhausted"
    raise RuntimeError(msg)


def _parse_arxiv_response(xml: str) -> str:
    """Parse arXiv API XML response into a readable string.

    Args:
        xml: Raw XML response from arXiv API.

    Returns:
        Formatted string with paper details.
    """
    import xml.etree.ElementTree as ET

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return "Error: Could not parse arXiv response."

    entries = root.findall("atom:entry", ns)
    if not entries:
        return "No papers found matching the query."

    results: list[str] = []
    for entry in entries:
        title = entry.find("atom:title", ns)
        summary = entry.find("atom:summary", ns)
        published = entry.find("atom:published", ns)
        link = entry.find("atom:id", ns)
        authors = entry.findall("atom:author/atom:name", ns)

        title_text = (
            (title.text or "").strip().replace("\n", " ")
            if title is not None
            else "Unknown"
        )
        summary_text = (
            (summary.text or "").strip().replace("\n", " ")
            if summary is not None
            else ""
        )
        published_text = (
            (published.text or "")[:10] if published is not None else "Unknown"
        )
        link_text = (link.text or "").strip() if link is not None else ""
        author_names = ", ".join(a.text for a in authors if a.text) if authors else "Unknown"

        results.append(
            f"Title: {title_text}\n"
            f"Authors: {author_names}\n"
            f"Published: {published_text}\n"
            f"URL: {link_text}\n"
            f"Abstract: {summary_text}\n"
        )

    return "\n---\n".join(results)