"""Live integration test that sends a real email via SMTP.

This test exercises the full email delivery path (smtplib) using real SMTP
credentials from the .env file. It is marked with @pytest.mark.live_email and
is SKIPPED by default. To run it, pass --live-email to pytest:

    pytest tests/integration/test_email_live.py --live-email -v -s

If the required env vars (AI_REPORT_EMAIL_TO, AI_REPORT_SMTP_PASSWORD,
AI_REPORT_EMAIL_FROM) are not set, the test skips gracefully with a clear
message rather than failing.
"""

from __future__ import annotations

import datetime

import pytest

from src.config import Settings
from src.email_report import EmailStatus, send_report_email


@pytest.mark.live_email
class TestLiveEmail:
    """Tests that send a real email using SMTP credentials from .env."""

    def test_sends_real_email(self) -> None:
        """Send a real email via SMTP using live config.

        Requires --live-email flag and real SMTP credentials in .env.
        """
        settings = Settings()

        # Guard: skip gracefully if live email config is missing
        if not settings.email_to or not settings.smtp_password or not settings.email_from:
            pytest.skip(
                "Live email config not set in .env "
                "(AI_REPORT_EMAIL_TO / AI_REPORT_SMTP_PASSWORD / "
                "AI_REPORT_EMAIL_FROM)"
            )

        # Force enable sending (in case .env has email_enabled=False or dry_run=True)
        settings.email_enabled = True
        settings.email_dry_run = False

        today = datetime.date.today()
        date_range = f"Live Test - {today}"
        report_md = (
            "# AI Industry Weekly - Live Test\n"
            "\n"
            "## Industry Overview\n"
            "\n"
            "- **Test**: This is a live email test from the AIReport test suite [1]\n"
            "\n"
            "## Sources\n"
            "\n"
            "1. [Test source](https://example.com)\n"
        )

        result = send_report_email(report_md, date_range, 0.78, True, settings)

        # send_report_email now returns an EmailResult. On success the status is
        # SENT; on failure the status is FAILED with the error message captured.
        assert result.status == EmailStatus.SENT, (
            f"Email send failed: {result.error}"
        )
        print(f"Live email sent to {settings.email_to}")
