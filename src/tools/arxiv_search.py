"""arXiv API tool for searching recent AI/ML papers."""

import asyncio
import logging
import random
import time
from datetime import date, timedelta

import httpx

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 3.0

# arXiv's Terms of Use (https://info.arxiv.org/help/api/tou.html) require
# "no more than one request every three seconds, and limit requests to a single
# connection at a time." Enforced via this lock + the timestamp below.
_ARXIV_MIN_INTERVAL_SEC = 3.0
_arxiv_lock = asyncio.Lock()
_arxiv_last_call_at: float = 0.0

# arXiv's API uses both 429 and 406 as rate-limit responses, and additionally
# returns HTTP 200 with body "Rate exceeded." for a soft rate-limit. The values
# below centralize the detection.
_RATE_LIMIT_STATUS_CODES: frozenset[int] = frozenset({429, 406})
_SOFT_RATE_LIMIT_BODY_PREFIX = "rate exceeded"


async def _pace_arxiv_call() -> None:
    """Block until at least ``_ARXIV_MIN_INTERVAL_SEC`` has passed since the
    last arXiv call. Honors the arXiv TOU's single-connection, 3-second rule.
    """
    global _arxiv_last_call_at
    async with _arxiv_lock:
        now = time.monotonic()
        wait = _ARXIV_MIN_INTERVAL_SEC - (now - _arxiv_last_call_at)
        if wait > 0:
            await asyncio.sleep(wait)
        _arxiv_last_call_at = time.monotonic()


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

    # arXiv's API is picky about Accept-Encoding (it only advertises gzip/identity
    # and returns 406 for the multi-encoding value httpx sends by default) and about
    # User-Agent (it throttles unidentified clients). Pin both explicitly.
    headers = {
        "Accept-Encoding": "gzip",
        "User-Agent": "ai-report/1.0 (https://github.com/connormacdonald/ai-report)",
    }

    last_exception: Exception | None = None
    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        for attempt in range(MAX_RETRIES):
            await _pace_arxiv_call()
            response = await client.get(url, params=params)

            # Hard rate-limit: arXiv returns 429 (and sometimes 406) when throttled.
            if response.status_code in _RATE_LIMIT_STATUS_CODES:
                if attempt == MAX_RETRIES - 1:
                    msg = "arXiv API rate-limited: all retries exhausted"
                    raise RuntimeError(msg)
                backoff = INITIAL_BACKOFF_SECONDS * (2 ** attempt) * random.uniform(0.5, 1.5)
                logger.warning(
                    "arXiv API rate-limited (%d), retrying in %.1fs (attempt %d/%d)",
                    response.status_code,
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

            # Soft rate-limit: arXiv returns HTTP 200 with body "Rate exceeded."
            # when a secondary rate limiter trips. Treat it as retryable.
            body_head = (response.text or "")[:64].strip().lower()
            if body_head.startswith(_SOFT_RATE_LIMIT_BODY_PREFIX):
                if attempt == MAX_RETRIES - 1:
                    msg = "arXiv API rate-limited (soft): all retries exhausted"
                    raise RuntimeError(msg)
                backoff = INITIAL_BACKOFF_SECONDS * (2 ** attempt) * random.uniform(1.0, 2.0)
                logger.warning(
                    "arXiv API soft rate-limited, retrying in %.1fs (attempt %d/%d)",
                    backoff,
                    attempt + 1,
                    MAX_RETRIES,
                )
                await asyncio.sleep(backoff)
                continue

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
