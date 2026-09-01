"""Tests for the email report delivery module."""

import logging
import os
import smtplib
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.email_report import EmailStatus, _build_subject, _write_dry_run, send_report_email


class TestBuildSubject:
    """Tests for the _build_subject function."""

    def test_pass_subject(self) -> None:
        """Should return clean subject when eval passes."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        subject = _build_subject("May 11 - May 18, 2026", 0.85, True, settings)
        assert subject == "AI Industry Weekly — May 11 - May 18, 2026"

    def test_fail_subject_appends_score(self) -> None:
        """Should append eval score when eval fails."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        subject = _build_subject("May 11 - May 18, 2026", 0.45, False, settings)
        assert subject == "AI Industry Weekly — May 11 - May 18, 2026 (eval 0.45, needs revision)"

    def test_custom_prefix(self) -> None:
        """Should use custom subject prefix from settings."""
        settings = Settings(_env_file=None, email_subject_prefix="AI Report")  # type: ignore[call-arg]
        subject = _build_subject("May 11 - May 18, 2026", 0.9, True, settings)
        assert subject == "AI Report — May 11 - May 18, 2026"

    def test_score_rounded_to_two_decimals(self) -> None:
        """Should format score to 2 decimal places in fail subject."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        subject = _build_subject("Test", 0.333333, False, settings)
        assert "eval 0.33" in subject


class TestWriteDryRun:
    """Tests for the _write_dry_run function."""

    def test_writes_html_file(self, tmp_path: str) -> None:
        """Should write HTML content to the output directory."""
        settings = Settings(_env_file=None, output_dir=str(tmp_path))  # type: ignore[call-arg]
        html_content = "<html><body><h1>Test</h1></body></html>"
        path = _write_dry_run(html_content, settings)

        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert content == html_content

    def test_returns_path(self, tmp_path: str) -> None:
        """Should return the full path to the written file."""
        settings = Settings(_env_file=None, output_dir=str(tmp_path))  # type: ignore[call-arg]
        path = _write_dry_run("<html></html>", settings)
        expected_date = date.today().strftime("%Y-%m-%d")
        assert path == os.path.join(
            str(tmp_path), f"ai-weekly-{expected_date}.html"
        )

    def test_creates_nested_output_directory(self, tmp_path: str) -> None:
        """Should create nested output directories if they don't exist."""
        nested_dir = os.path.join(str(tmp_path), "nested", "email")
        settings = Settings(_env_file=None, output_dir=nested_dir)  # type: ignore[call-arg]
        path = _write_dry_run("<html></html>", settings)
        assert os.path.exists(path)

    def test_filename_format(self, tmp_path: str) -> None:
        """Should use the correct filename pattern."""
        settings = Settings(_env_file=None, output_dir=str(tmp_path))  # type: ignore[call-arg]
        path = _write_dry_run("<html></html>", settings)
        expected_date = date.today().strftime("%Y-%m-%d")
        assert os.path.basename(path) == f"ai-weekly-{expected_date}.html"

    def test_file_is_html(self, tmp_path: str) -> None:
        """The written file should have .html extension."""
        settings = Settings(_env_file=None, output_dir=str(tmp_path))  # type: ignore[call-arg]
        path = _write_dry_run("<html></html>", settings)
        assert path.endswith(".html")


class TestSendReportEmail:
    """Tests for the send_report_email function."""

    def test_email_disabled_returns_none(
        self, sample_report_markdown: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should return DISABLED status when email is disabled."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            output_dir="/tmp",
        )
        with caplog.at_level(logging.INFO):
            result = send_report_email(
                sample_report_markdown, "Test Range", 0.85, True, settings
            )
        assert result.status == EmailStatus.DISABLED
        assert "Email disabled" in caplog.text

    def test_missing_email_to_returns_none(
        self, sample_report_markdown: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should return SKIPPED status when email_to is an empty list."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            email_enabled=True,
            email_from="sender@example.com",
            email_to=[],
        )
        with caplog.at_level(logging.WARNING):
            result = send_report_email(
                sample_report_markdown, "Test Range", 0.85, True, settings
            )
        assert result.status == EmailStatus.SKIPPED
        assert "email_to/email_from not configured" in caplog.text

    def test_missing_email_from_returns_none(
        self, sample_report_markdown: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should return SKIPPED status when email_from is empty."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            email_enabled=True,
            email_to=["recipient@example.com"],
        )
        with caplog.at_level(logging.WARNING):
            result = send_report_email(
                sample_report_markdown, "Test Range", 0.85, True, settings
            )
        assert result.status == EmailStatus.SKIPPED
        assert "email_to/email_from not configured" in caplog.text

    def test_dry_run_writes_html(
        self, sample_report_markdown: str, tmp_path: str
    ) -> None:
        """Should write an HTML file in dry-run mode (no SMTP)."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            output_dir=str(tmp_path),
            email_enabled=True,
            email_dry_run=True,
            email_to=["test@example.com"],
            email_from="sender@example.com",
        )
        result = send_report_email(
            sample_report_markdown, "Test Range", 0.85, True, settings
        )
        assert result.status == EmailStatus.DRY_RUN
        assert result.html_path is not None
        assert os.path.exists(result.html_path)
        assert result.html_path.endswith(".html")

    def test_dry_run_returns_path(
        self, sample_report_markdown: str, tmp_path: str
    ) -> None:
        """Should return an EmailResult with the path in dry-run mode."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            output_dir=str(tmp_path),
            email_enabled=True,
            email_dry_run=True,
            email_to=["test@example.com"],
            email_from="sender@example.com",
        )
        result = send_report_email(
            sample_report_markdown, "Test Range", 0.85, True, settings
        )
        expected_date = date.today().strftime("%Y-%m-%d")
        expected = os.path.join(str(tmp_path), f"ai-weekly-{expected_date}.html")
        assert result.status == EmailStatus.DRY_RUN
        assert result.html_path == expected

    def test_dry_run_no_smtp_connection(
        self, sample_report_markdown: str, tmp_path: str
    ) -> None:
        """Should NOT try to connect to SMTP in dry-run mode."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            output_dir=str(tmp_path),
            email_enabled=True,
            email_dry_run=True,
            email_to=["test@example.com"],
            email_from="sender@example.com",
        )
        with patch("src.email_report.smtplib.SMTP") as mock_smtp:
            result = send_report_email(
                sample_report_markdown, "Test Range", 0.85, True, settings
            )
        assert result.status == EmailStatus.DRY_RUN
        mock_smtp.assert_not_called()

    def test_successful_send(
        self, sample_report_markdown: str, tmp_path: str
    ) -> None:
        """Should send email via SMTP and return SENT status."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            output_dir=str(tmp_path),
            email_enabled=True,
            email_to=["test@example.com"],
            email_from="sender@example.com",
            smtp_host="smtp.example.com",
            smtp_port=587,
            smtp_username="user",
            smtp_password="pass",
            smtp_use_tls=True,
        )
        with patch("src.email_report.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            result = send_report_email(
                sample_report_markdown, "Test Range", 0.85, True, settings
            )

        assert result.status == EmailStatus.SENT
        mock_smtp.assert_called_once_with("smtp.example.com", 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user", "pass")
        mock_server.send_message.assert_called_once()

    def test_send_without_username(
        self, sample_report_markdown: str, tmp_path: str
    ) -> None:
        """Should skip login when no smtp_username is set."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            output_dir=str(tmp_path),
            email_enabled=True,
            email_to=["test@example.com"],
            email_from="sender@example.com",
            smtp_host="smtp.example.com",
            smtp_use_tls=True,
        )
        with patch("src.email_report.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            result = send_report_email(
                sample_report_markdown, "Test Range", 0.85, True, settings
            )

        assert result.status == EmailStatus.SENT
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_not_called()
        mock_server.send_message.assert_called_once()

    def test_tls_disabled(
        self, sample_report_markdown: str, tmp_path: str
    ) -> None:
        """Should not call starttls when smtp_use_tls is False."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            output_dir=str(tmp_path),
            email_enabled=True,
            email_to=["test@example.com"],
            email_from="sender@example.com",
            smtp_host="smtp.example.com",
            smtp_username="user",
            smtp_password="pass",
            smtp_use_tls=False,
        )
        with patch("src.email_report.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            result = send_report_email(
                sample_report_markdown, "Test Range", 0.85, True, settings
            )

        assert result.status == EmailStatus.SENT
        mock_server.starttls.assert_not_called()
        mock_server.login.assert_called_once_with("user", "pass")
        mock_server.send_message.assert_called_once()

    def test_smtp_failure_returns_none(
        self, sample_report_markdown: str, tmp_path: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should catch SMTP errors, log error, and return FAILED status (never raise)."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            output_dir=str(tmp_path),
            email_enabled=True,
            email_to=["test@example.com"],
            email_from="sender@example.com",
            smtp_host="smtp.example.com",
            smtp_use_tls=True,
        )
        with (
            patch("src.email_report.smtplib.SMTP") as mock_smtp,
            caplog.at_level(logging.WARNING),
        ):
            mock_smtp.side_effect = smtplib.SMTPException("Connection refused")
            # Should NOT raise
            result = send_report_email(
                sample_report_markdown, "Test Range", 0.85, True, settings
            )

        assert result.status == EmailStatus.FAILED
        assert result.error is not None
        assert "Email send failed" in caplog.text

    def test_send_to_multiple_recipients(
        self, sample_report_markdown: str, tmp_path: str
    ) -> None:
        """Should send email to all recipients when email_to has multiple addresses."""
        recipients = [
            "alice@example.com",
            "bob@example.com",
            "carol@example.com",
        ]
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            output_dir=str(tmp_path),
            email_enabled=True,
            email_to=recipients,
            email_from="sender@example.com",
            smtp_host="smtp.example.com",
            smtp_port=587,
            smtp_username="user",
            smtp_password="pass",
            smtp_use_tls=True,
        )
        with patch("src.email_report.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            result = send_report_email(
                sample_report_markdown, "Test Range", 0.85, True, settings
            )

        assert result.status == EmailStatus.SENT
        mock_server.send_message.assert_called_once()
        # Extract the EmailMessage that was sent and verify the To header
        sent_msg = mock_server.send_message.call_args.args[0]
        assert sent_msg["To"] == ", ".join(recipients)

    def test_send_to_two_recipients_logs_all(
        self,
        sample_report_markdown: str,
        tmp_path: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Should log all recipients on success."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            output_dir=str(tmp_path),
            email_enabled=True,
            email_to=["alice@example.com", "bob@example.com"],
            email_from="sender@example.com",
            smtp_host="smtp.example.com",
        )
        with (
            patch("src.email_report.smtplib.SMTP") as mock_smtp,
            caplog.at_level(logging.INFO),
        ):
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            result = send_report_email(
                sample_report_markdown, "Test Range", 0.85, True, settings
            )

        assert result.status == EmailStatus.SENT
        assert "alice@example.com" in caplog.text
        assert "bob@example.com" in caplog.text

    def test_smtp_send_called_once_for_multiple_recipients(
        self, sample_report_markdown: str, tmp_path: str
    ) -> None:
        """Should call server.send_message exactly once even with multiple recipients."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            model_provider="openai",
            openai_api_key="test-key",
            output_dir=str(tmp_path),
            email_enabled=True,
            email_to=["a@example.com", "b@example.com", "c@example.com"],
            email_from="sender@example.com",
            smtp_host="smtp.example.com",
        )
        with patch("src.email_report.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            send_report_email(
                sample_report_markdown, "Test Range", 0.85, True, settings
            )

        # One SMTP send_message call covering all recipients in the To header
        mock_server.send_message.assert_called_once()
