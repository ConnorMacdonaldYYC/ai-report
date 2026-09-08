"""Result schemas for the evaluation framework.

TokenUsage — re-exported from src.schemas (core schema, shared with the
report pipeline).
RunResult — one report generation run (a generator model + seed).
EvalRun — one evaluation of a fixed report by one evaluator model.
MatrixResult — the full generators x evaluators x seeds matrix.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from src.schemas import DimensionScore, Source, TokenUsage

__all__ = ["EvalRun", "MatrixResult", "RunResult", "TokenUsage", "aggregate_usage"]


class RunResult(BaseModel):
    """One report generation run: a generator model at a given seed.

    eval_passed / eval_score come from the pipeline's in-pipeline
    evaluation (used for the revision loop), not from the external
    evaluator panel.
    """

    generator_model: str
    seed: int
    date_range: str
    report_markdown: str
    sources: list[Source] = Field(default_factory=list)
    tokens: TokenUsage
    duration_seconds: float
    revision_count: int
    eval_passed: bool
    eval_score: float
    timestamp: str


class EvalRun(BaseModel):
    """One evaluation of a fixed report by one evaluator model."""

    evaluator_model: str
    generator_model: str
    seed: int
    score: float
    eval_passed: bool
    dimensions: list[DimensionScore]
    tokens: TokenUsage
    duration_seconds: float
    timestamp: str


class MatrixResult(BaseModel):
    """The full evaluation matrix: generators x evaluators x seeds."""

    generators: list[str]
    evaluators: list[str]
    seeds: int
    runs: list[RunResult]
    evals: list[EvalRun]
    created_at: str

    def save(self, path: str | Path) -> Path:
        """Save the matrix to a JSON file, creating parent directories.

        Args:
            path: Destination file path.

        Returns:
            The resolved path that was written.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(self.model_dump_json(indent=2))
        return dest

    @classmethod
    def load(cls, path: str | Path) -> "MatrixResult":
        """Load a matrix from a JSON file.

        Args:
            path: Path to a matrix.json file written by save().

        Returns:
            The loaded MatrixResult.
        """
        return cls.model_validate_json(Path(path).read_text())


def aggregate_usage(usages: list[TokenUsage]) -> TokenUsage:
    """Sum a list of TokenUsage objects into one.

    Args:
        usages: Token usage objects to aggregate.

    Returns:
        A single TokenUsage with summed counts.
    """
    return TokenUsage(
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
    )
