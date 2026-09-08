"""Tests for the eval framework CLI (python -m src.eval)."""

from typing import Any

import pytest

from src.config import Settings
from src.eval.results import MatrixResult
from tests.utils import build_test_settings


class TestMain:
    """Tests for CLI argument parsing and run_matrix delegation."""

    def test_main_parses_args_and_runs_matrix(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should parse comma-separated models and forward them to run_matrix."""
        from src.eval.__main__ import main

        captured: dict[str, Any] = {}

        async def fake_run_matrix(
            settings: Settings,
            generators: list[str],
            evaluators: list[str],
            seeds: int,
            **kwargs: Any,
        ) -> MatrixResult:
            captured.update(
                generators=generators,
                evaluators=evaluators,
                seeds=seeds,
                **kwargs,
            )
            return MatrixResult(
                generators=generators,
                evaluators=evaluators,
                seeds=seeds,
                runs=[],
                evals=[],
                created_at="2026-01-01T00:00:00Z",
            )

        monkeypatch.setattr("src.eval.__main__.run_matrix", fake_run_matrix)
        monkeypatch.setattr(
            "src.eval.__main__.get_settings", lambda: build_test_settings()
        )
        monkeypatch.setattr(Settings, "configure", lambda self: None)

        main(
            [
                "--generators", "g1, g2",
                "--evaluators", "e1,e2",
                "--seeds", "2",
                "--concurrency", "4",
                "--output-dir", str(tmp_path),
                "--date-range", "Jan 1 - Jan 8, 2026",
            ]
        )

        assert captured["generators"] == ["g1", "g2"]
        assert captured["evaluators"] == ["e1", "e2"]
        assert captured["seeds"] == 2
        assert captured["concurrency"] == 4
        assert captured["output_dir"] == str(tmp_path)
        assert captured["date_range"] == "Jan 1 - Jan 8, 2026"

    def test_main_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should default seeds=3, concurrency=3, ./eval_results, no date range."""
        from src.eval.__main__ import main

        captured: dict[str, Any] = {}

        async def fake_run_matrix(
            settings: Settings,
            generators: list[str],
            evaluators: list[str],
            seeds: int,
            **kwargs: Any,
        ) -> MatrixResult:
            captured.update(
                generators=generators,
                evaluators=evaluators,
                seeds=seeds,
                **kwargs,
            )
            return MatrixResult(
                generators=generators,
                evaluators=evaluators,
                seeds=seeds,
                runs=[],
                evals=[],
                created_at="2026-01-01T00:00:00Z",
            )

        monkeypatch.setattr("src.eval.__main__.run_matrix", fake_run_matrix)
        monkeypatch.setattr(
            "src.eval.__main__.get_settings", lambda: build_test_settings()
        )
        monkeypatch.setattr(Settings, "configure", lambda self: None)

        main(["--generators", "g1", "--evaluators", "e1"])

        assert captured["seeds"] == 3
        assert captured["concurrency"] == 3
        assert captured["output_dir"] == "./eval_results"
        assert captured["date_range"] is None

    def test_main_requires_generators_and_evaluators(self) -> None:
        """Should exit when --generators or --evaluators is missing."""
        from src.eval.__main__ import main

        with pytest.raises(SystemExit):
            main(["--evaluators", "e1"])
        with pytest.raises(SystemExit):
            main(["--generators", "g1"])
