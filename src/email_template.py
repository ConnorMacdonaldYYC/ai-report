"""Hacker News-inspired HTML email template for the AI weekly report.

This module converts a Markdown report into a compact, email-safe HTML document
with a subtle Hacker News visual treatment: the classic orange header band,
upvote-triangle bullets, orange citation markers, and a source list styled like
HN story metadata.
"""

from __future__ import annotations

import html
import logging
import re

import markdown  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_HN_ORANGE = "#ff6600"
_HN_GRAY = "#828282"
_TEXT = "#1a1a1a"
_BORDERS = "#e6e6e0"
_BG = "#ffffff"

_UPVOTE = "▴"

_BODY_STYLE = (
    "font-family:Verdana,Geneva,sans-serif; font-size:14px; line-height:1.6; "
    f"color:{_TEXT}"
)
_HEADER_STYLE = (
    "font-family:Verdana,Geneva,sans-serif; font-size:20px; font-weight:bold; "
    "color:#ffffff"
)
_SUBTITLE_STYLE = (
    "font-family:Verdana,Geneva,sans-serif; font-size:13px; color:#ffeecc; "
    "margin-top:4px"
)
_FOOTER_STYLE = (
    "font-family:Verdana,Geneva,sans-serif; font-size:12px; "
    f"color:{_HN_GRAY}"
)


def render_html(
    report_markdown: str,
    date_range: str,
    eval_score: float,
    eval_passed: bool,
) -> str:
    """Render the report as an HN-themed HTML email body."""
    safe_date_range = html.escape(date_range)
    safe_eval_score = html.escape(f"{eval_score:.2f}")
    eval_label = "PASS" if eval_passed else "FAIL"

    body_html = markdown.markdown(
        report_markdown,
        extensions=["extra", "sane_lists"],
    )

    body_html = _strip_lead_heading(body_html)
    body_html = _convert_raw_source_refs(body_html)
    body_html = _style_headings_and_rules(body_html)
    body_html = _convert_source_list(body_html)
    body_html = _dedup_sources(body_html)
    body_html = _mark_citations(body_html)
    body_html = _style_bullet_lists(body_html)
    body_html = _style_links(body_html)

    return _wrap_document(body_html, safe_date_range, safe_eval_score, eval_label)


def _strip_lead_heading(body_html: str) -> str:
    """Remove the first <h1> title; the email header already shows it."""
    return re.sub(r"<h1>.*?</h1>", "", body_html, count=1, flags=re.DOTALL)


def _convert_raw_source_refs(body_html: str) -> str:
    """Turn [n] https://... paragraphs into compact source items.

    Raw refs have no title (the link text is the URL itself), so we render the
    URL once as a link without repeating it in gray monospace.
    """

    def _replace(match: re.Match[str]) -> str:
        num = html.escape(match.group(1))
        url = html.escape(match.group(2))
        link_style = f"color:{_HN_ORANGE}; text-decoration:underline;"
        num_style = f"color:{_HN_ORANGE}; font-weight:bold;"
        url_style = f"color:{_HN_GRAY}; font-family:monospace; font-size:12px;"
        return (
            f'<p class="hn-source-compact" style="margin:4px 0; font-size:13px;">'
            f'<span class="hn-source-num" style="{num_style}">{num}.</span> '
            f'<a href="{url}" class="hn-link" style="{link_style}">'
            f'<span class="hn-source-url" style="{url_style}">{url}</span></a>'
            f"</p>"
        )

    # Paragraph-style raw references: [1] https://example.com
    body_html = re.sub(
        r"<p>\[(\d{1,3})\]\s+(https?://[^\s<]+)</p>",
        _replace,
        body_html,
        flags=re.DOTALL,
    )
    # Stray heading-style raw reference (occasionally emitted by markdown).
    body_html = re.sub(
        r"<h2>\[(\d{1,3})\]\s+(https?://[^\s<]+)</h2>",
        _replace,
        body_html,
        flags=re.DOTALL,
    )
    return body_html


def _style_headings_and_rules(body_html: str) -> str:
    """Add HN section/sub-section styling and tame horizontal rules."""

    def _section(match: re.Match[str]) -> str:
        # markdown already escapes entities; do not re-escape (avoids &amp;amp;)
        text = match.group(1)
        style = (
            f"font-size:18px; font-weight:bold; color:{_TEXT}; "
            "text-transform:uppercase; "
            f"border-left:3px solid {_HN_ORANGE}; "
            f"padding-left:8px; border-bottom:1px solid {_BORDERS}; "
            "margin:24px 0 12px 0;"
        )
        return f'<h2 class="hn-section" style="{style}">{text}</h2>'

    def _subsection(match: re.Match[str]) -> str:
        # markdown already escapes entities; do not re-escape
        text = match.group(1)
        style = (
            f"font-size:15px; font-weight:bold; color:{_TEXT}; "
            "margin:18px 0 8px 0;"
        )
        return f'<h3 class="hn-subsection" style="{style}">{text}</h3>'

    body_html = re.sub(r"<h2>([^<]+)</h2>", _section, body_html, flags=re.DOTALL)
    body_html = re.sub(r"<h3>([^<]+)</h3>", _subsection, body_html, flags=re.DOTALL)
    hr = (
        f'<hr style="border:0; border-top:1px solid {_BORDERS}; '
        f'margin:16px 0; height:0;" />'
    )
    body_html = re.sub(r"<hr\s*/?>", hr, body_html)
    return body_html


def _convert_source_list(body_html: str) -> str:
    """Append the source URL in gray monospace after each titled source link."""

    link_style = f"color:{_HN_ORANGE}; text-decoration:underline;"
    url_style = f"color:{_HN_GRAY}; font-family:monospace; font-size:12px;"

    def _items(block: str) -> str:
        return re.sub(
            r'<li><a href="([^"]+)">([^<]+)</a></li>',
            lambda m: (
                f'<li style="margin:4px 0;">'
                f'<a href="{html.escape(m.group(1))}" class="hn-link" '
                f'style="{link_style}">'
                f"{html.escape(m.group(2))}</a>"
                f'<span class="hn-source-url" style="{url_style}">'
                f" {html.escape(m.group(1))}</span>"
                f"</li>"
            ),
            block,
            flags=re.DOTALL,
        )

    return re.sub(
        r'(<h2 class="hn-section"[^>]*>Sources</h2>)\s*<ol>(.*?)</ol>',
        lambda m: (
            f'{m.group(1)}<ol class="hn-sources" style="margin:0; padding-left:20px;"'
            f">{_items(m.group(2))}</ol>"
        ),
        body_html,
        flags=re.DOTALL,
    )


def _dedup_sources(body_html: str) -> str:
    """Drop a raw-reference Sources block when a titled Sources list also exists.

    The manager agent occasionally emits sources twice: once as raw
    ``[n] https://...`` reference paragraphs and once as a titled numbered
    list. The titled list is more useful, so when both are present we remove
    the raw-ref block (heading + compact paragraphs) to avoid duplication.
    """
    sources_heading = r'<h2 class="hn-section"[^>]*>Sources</h2>'
    # Split the document at each Sources heading.
    parts = re.split(rf"({sources_heading})", body_html, flags=re.DOTALL)
    # parts: [pre, heading, content, heading, content, ...]
    sources_indices = [
        i for i, p in enumerate(parts) if re.match(sources_heading, p)
    ]
    if len(sources_indices) < 2:
        return body_html

    # Identify which content block has a titled <ol> list.
    titled_index = None
    raw_index = None
    for idx in sources_indices:
        content = parts[idx + 1] if idx + 1 < len(parts) else ""
        if "<ol" in content:
            titled_index = idx
        elif 'hn-source-compact' in content:
            raw_index = idx

    # Remove the raw-ref block (heading + its content) if a titled list exists.
    if titled_index is not None and raw_index is not None:
        parts[raw_index] = ""
        parts[raw_index + 1] = ""
        return "".join(parts)
    return body_html


def _mark_citations(body_html: str) -> str:
    """Style inline citation markers like [1] as small orange badges."""
    return re.sub(
        r"\[(\d{1,3})\]",
        lambda m: (
            f'<sup class="hn-cite" style="color:{_HN_ORANGE}; font-weight:bold; '
            f'font-size:0.85em;">[{m.group(1)}]</sup>'
        ),
        body_html,
    )


def _style_bullet_lists(body_html: str) -> str:
    """Replace bullet markers with the HN upvote triangle and highlight lead-ins."""

    def _list(inner: str) -> str:
        items = re.sub(
            r"<li>(.*?)</li>",
            lambda m: (
                f'<li style="list-style-type:none; margin:0 0 6px 0; padding-left:0;">'
                f'<span class="hn-upvote" style="color:{_HN_ORANGE}; font-size:14px; '
                f'margin-right:6px; line-height:1;">{_UPVOTE}</span>'
                f"{_highlight_lead_in(m.group(1))}</li>"
            ),
            inner,
            flags=re.DOTALL,
        )
        return (
            f'<ul class="hn-list" style="list-style-type:none; margin:0; padding:0;">'
            f"{items}</ul>"
        )

    return re.sub(r"<ul>(.*?)</ul>", lambda m: _list(m.group(1)), body_html, flags=re.DOTALL)


def _highlight_lead_in(item_html: str) -> str:
    """Color the first bold lead-in (e.g. <strong>Google</strong>:) orange."""
    return re.sub(
        r"^(\s*)<strong>([^<]+)</strong>:",
        lambda m: (
            f'{m.group(1)}<strong class="hn-lead" style="color:{_HN_ORANGE};">'
            f"{html.escape(m.group(2))}</strong>:"
        ),
        item_html,
        count=1,
    )


def _style_links(body_html: str) -> str:
    """Color all remaining links with the HN orange accent."""
    return re.sub(
        r'<a href="([^"]+)">',
        lambda m: (
            f'<a href="{html.escape(m.group(1))}" class="hn-link" '
            f'style="color:{_HN_ORANGE}; text-decoration:underline;">'
        ),
        body_html,
    )


def _wrap_document(
    body_html: str,
    date_range: str,
    eval_score: str,
    eval_label: str,
) -> str:
    """Assemble the final HTML document with an email-safe table layout."""
    header_title = html.escape("AI Industry Weekly")
    footer_text = (
        f"Generated by AIReport · {date_range} · eval {eval_score}/1.0 "
        f"({eval_label})"
    )

    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '  <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0" />',
            f"  <title>{header_title} - {date_range}</title>",
            '  <style type="text/css">',
            "    body {",
            "      margin: 0;",
            "      padding: 0;",
            f"      background-color: {_BG};",
            "      font-family: Verdana, Geneva, sans-serif;",
            f"      color: {_TEXT};",
            "    }",
            "    .hn-body {",
            "      font-family: Verdana, Geneva, sans-serif;",
            "      font-size: 14px;",
            "      line-height: 1.6;",
            f"      color: {_TEXT};",
            "    }",
            "    .hn-body p {",
            "      margin: 0 0 12px 0;",
            "    }",
            "    .hn-section {",
            "      font-size: 18px;",
            "      font-weight: bold;",
            f"      color: {_TEXT};",
            "      text-transform: uppercase;",
            f"      border-left: 3px solid {_HN_ORANGE};",
            "      padding-left: 8px;",
            f"      border-bottom: 1px solid {_BORDERS};",
            "      margin: 24px 0 12px 0;",
            "    }",
            "    .hn-subsection {",
            "      font-size: 15px;",
            "      font-weight: bold;",
            f"      color: {_TEXT};",
            "      margin: 18px 0 8px 0;",
            "    }",
            "    .hn-link {",
            f"      color: {_HN_ORANGE};",
            "      text-decoration: underline;",
            "    }",
            "    .hn-cite {",
            f"      color: {_HN_ORANGE};",
            "      font-weight: bold;",
            "      font-size: 0.85em;",
            "    }",
            "    .hn-upvote {",
            f"      color: {_HN_ORANGE};",
            "    }",
            "    .hn-lead {",
            f"      color: {_HN_ORANGE};",
            "    }",
            "    .hn-source-url {",
            f"      color: {_HN_GRAY};",
            "      font-family: monospace;",
            "      font-size: 12px;",
            "    }",
            "    .hn-source-num {",
            f"      color: {_HN_ORANGE};",
            "      font-weight: bold;",
            "    }",
            "    .hn-sources li {",
            "      margin: 4px 0;",
            "    }",
            "  </style>",
            "</head>",
            "<body>",
            '  <table role="presentation" width="100%" cellspacing="0"',
            f'    cellpadding="0" border="0" bgcolor="{_BG}">',
            "    <tr>",
            '      <td align="center">',
            '        <table role="presentation" width="600" cellspacing="0"',
            '          cellpadding="0" border="0"',
            '          style="max-width:600px; width:100%;">',
            "          <tr>",
            "            <td>",
            '              <table role="presentation" width="100%" cellspacing="0"',
            f'                cellpadding="0" border="0" bgcolor="{_HN_ORANGE}">',
            "                <tr>",
            '                  <td style="padding:18px 20px;">',
            f'                    <div style="{_HEADER_STYLE}">',
            f"                      {header_title}",
            "                    </div>",
            f'                    <div style="{_SUBTITLE_STYLE}">',
            f"                      {date_range}",
            "                    </div>",
            "                  </td>",
            "                </tr>",
            "              </table>",
            "            </td>",
            "          </tr>",
            "          <tr>",
            '            <td style="padding:20px;">',
            f'              <div class="hn-body" style="{_BODY_STYLE}">',
            f"                {body_html}",
            "              </div>",
            "            </td>",
            "          </tr>",
            "          <tr>",
            '            <td style="padding:12px 20px; background-color:#f5f5f0;',
            f'              border-top:1px solid {_BORDERS};">',
            f'              <div style="{_FOOTER_STYLE}">',
            f"                {footer_text}",
            "              </div>",
            "            </td>",
            "          </tr>",
            "        </table>",
            "      </td>",
            "    </tr>",
            "  </table>",
            "</body>",
            "</html>",
        ]
    )
