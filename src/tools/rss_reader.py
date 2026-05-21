"""RSS/Atom feed reader tool for AI blogs and news sources."""

import asyncio
import re
from datetime import date, timedelta
from time import mktime

import feedparser  # type: ignore[import-untyped]

from src.config import get_settings

_RE = re.compile(r"<[^>]+>")


async def rss_reader(
    feeds: list[str] | None = None,
    days_back: int = 7,
) -> str:
    """Read recent posts from configured RSS/Atom feeds.

    Args:
        feeds: List of feed URLs to read. Defaults to settings.rss_feeds.
        days_back: How many days back to include posts from.

    Returns:
        Formatted string with article titles, URLs, dates, and summaries.
    """
    settings = get_settings()
    if feeds is None:
        feeds = settings.rss_feeds

    cutoff_date = date.today() - timedelta(days=days_back)

    all_entries: list[str] = []

    for feed_url in feeds:
        try:
            # feedparser.parse is synchronous and makes HTTP requests,
            # so run it in a thread to avoid blocking the event loop.
            feed = await asyncio.to_thread(feedparser.parse, feed_url)
            if getattr(feed, "bozo", False) and not feed.entries:
                all_entries.append(f"Feed error ({feed_url}): Could not parse feed.")
                continue

            feed_title = feed.feed.get("title", feed_url)
            entries_text: list[str] = []

            for entry in feed.entries:
                published = _parse_entry_date(entry)
                if published is None or published < cutoff_date:
                    continue

                title = entry.get("title", "Untitled")
                link = entry.get("link", "")
                summary = entry.get("summary", "")
                # Strip HTML tags from summary
                summary = _RE.sub("", summary).strip()
                summary = summary[:500] if len(summary) > 500 else summary

                entries_text.append(f"- **{title}** ({published})\n  {link}\n  {summary}")

            if entries_text:
                all_entries.append(f"## {feed_title}\n" + "\n\n".join(entries_text))

        except Exception as e:
            all_entries.append(f"Feed error ({feed_url}): {e}")

    return "\n\n---\n\n".join(all_entries) if all_entries else "No recent posts found in any feeds."


def _parse_entry_date(entry: feedparser.FeedParserDict) -> date | None:
    """Parse the publication date from a feed entry.

    Args:
        entry: A feedparser entry dict.

    Returns:
        The publication date, or None if parsing fails.
    """
    for attr in ("published_parsed", "updated_parsed"):
        parsed = entry.get(attr)
        if parsed:
            try:
                return date.fromtimestamp(mktime(parsed))
            except (TypeError, ValueError, OverflowError):
                continue
    return None