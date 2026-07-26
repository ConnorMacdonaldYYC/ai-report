"""Email delivery for the AI weekly report via SMTP.

Best-effort delivery: failures are captured in the returned EmailResult and
logged, but never raised, so the report file (already on disk) is never lost.
Supports a dry-run mode that renders the HTML to the output directory for
in-browser preview without SMTP creds.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from dataclasses import dataclass
from datetime import date
from email.message import EmailMessage
from enum import StrEnum

from src.config import Settings
from src.email_template import render_html
from src.email_text import render_text

logger = logging.getLogger(__name__)


class EmailStatus(StrEnum):
    """Outcome of an email delivery attempt."""

    DISABLED = "disabled"  # email_enabled is False
    SKIPPED = "skipped"  # enabled but email_to/email_from missing
    DRY_RUN = "dry_run"  # HTML written to disk, no SMTP connection
    SENT = "sent"  # SMTP send succeeded
    FAILED = "failed"  # SMTP send raised; error captured in EmailResult.error


@dataclass(frozen=True)
class EmailResult:
    """Result of an email delivery attempt.

    Attributes:
        status: Outcome category (see EmailStatus).
        html_path: Path to the written HTML file (set only when DRY_RUN).
        error: Error message (set only when FAILED).
    """

    status: EmailStatus
    html_path: str | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """True if the email was sent or written as dry-run HTML."""
        return self.status in (EmailStatus.SENT, EmailStatus.DRY_RUN)


def send_report_email(
    report_markdown: str,
    date_range: str,
    eval_score: float,
    eval_passed: bool,
    settings: Settings,
) -> EmailResult:
    """Send the report via SMTP, or write HTML to disk in dry-run mode.

    Args:
        report_markdown: The raw markdown content from the manager agent.
        date_range: The date range string for the report.
        eval_score: Final evaluation score (0.0-1.0).
        eval_passed: Whether the report passed evaluation.
        settings: Application settings with email/SMTP configuration.

    Returns:
        EmailResult with status and (for dry-run) the HTML path, or (for
        failure) the error message. Never raises.
    """
    if not settings.email_enabled:
        logger.info("Email disabled; skipping send")
        return EmailResult(status=EmailStatus.DISABLED)

    if not settings.email_to or not settings.email_from:
        logger.warning(
            "Email enabled but email_to/email_from not configured; skipping send"
        )
        return EmailResult(status=EmailStatus.SKIPPED)

    html_body = render_html(report_markdown, date_range, eval_score, eval_passed)
    text_body = render_text(report_markdown, date_range, eval_score, eval_passed)
    subject = _build_subject(date_range, eval_score, eval_passed, settings)

    if settings.email_dry_run:
        path = _write_dry_run(html_body, settings)
        return EmailResult(status=EmailStatus.DRY_RUN, html_path=path)

    try:
        _send_smtp(html_body, text_body, subject, settings)
    except Exception as exc:  # best-effort, never raise
        logger.error("Email send failed: %s", exc)
        return EmailResult(status=EmailStatus.FAILED, error=str(exc))

    logger.info("Report emailed to %s", settings.email_to)
    return EmailResult(status=EmailStatus.SENT)


def _build_subject(
    date_range: str, eval_score: float, eval_passed: bool, settings: Settings
) -> str:
    """Build the email subject line.

    On eval failure, appends the score so the inbox signals a problem at a
    glance; on pass, keeps the subject clean.

    Args:
        date_range: The date range string for the report.
        eval_score: Final evaluation score (0.0-1.0).
        eval_passed: Whether the report passed evaluation.
        settings: Application settings with the subject prefix.

    Returns:
        Formatted subject string.
    """
    subject = f"{settings.email_subject_prefix} — {date_range}"
    if not eval_passed:
        subject += f" (eval {eval_score:.2f}, needs revision)"
    return subject


def _send_smtp(
    html_body: str,
    text_body: str,
    subject: str,
    settings: Settings,
) -> None:
    """Send a multipart email via SMTP.

    Args:
        html_body: HTML content for the email body.
        text_body: Plain-text alternative content.
        subject: Email subject line.
        settings: Application settings with SMTP configuration.

    Raises:
        smtplib.SMTPException or OSError on connection/auth/send failure.
        Callers are expected to catch and log (send_report_email does).
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.email_from
    msg["To"] = settings.email_to
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
        if settings.smtp_use_tls:
            server.starttls(context=ssl.create_default_context())
        if settings.smtp_username:
            server.login(settings.smtp_username, settings.smtp_password)
        server.send_message(msg)


def _write_dry_run(html_body: str, settings: Settings) -> str:
    """Write the rendered HTML to the output directory for preview.

    Args:
        html_body: Rendered HTML email content.
        settings: Application settings with output_dir configuration.

    Returns:
        The path to the written HTML file.
    """
    os.makedirs(settings.output_dir, exist_ok=True)
    filename = f"ai-weekly-{date.today().strftime('%Y-%m-%d')}.html"
    path = os.path.join(settings.output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_body)
    logger.info("Dry-run HTML written to %s", path)
    return path