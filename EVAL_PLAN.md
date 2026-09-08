# Evaluation Framework — Execution Plan

> **Status: IMPLEMENTED** (all phases complete, 178 tests green, lint clean).
> As-built notes:
> - `TokenUsage` lives in `src/schemas.py` (core) and is re-exported from
>   `src/eval/results.py`; `ReportResult.tokens` is populated by the pipeline.
> - `run_report` / `run_single` / `run_matrix` accept an optional fixed
>   `date_range` so every report in a matrix shares the same window label.
> - `run_matrix` fixes the in-pipeline evaluator (revision-loop referee) to
>   `evaluators[0]` for fairness across generators, and skips failed runs
>   with a warning instead of aborting the matrix.
> - Shared test helpers live in `tests/utils.py` (override_all_agents,
>   build_test_settings, model fn factories, result builders).
> - Makefile target: `make eval GENERATORS=... EVALUATORS=... SEEDS=...`

Compare report-generation models on **token usage, time, and evaluation score**,
with each report judged by a **separate panel of evaluator models** (cross-evaluation
matrix) and each generator run **repeated N seeds** to handle sampling variance.

Cost tracking is **out of scope** — token usage only.

---

## Design decisions (locked)

| Decision | Choice |
|---|---|
| Run isolation | Refactor `agents.py` → `build_agents(settings)` factory returning an `AgentBundle` |
| Input reproducibility | No control — each run fetches live data; per-seed stddev absorbs variance |
| Evaluator panel | Separate from generators (disjoint sets, no self-bias) |
| Repeated trials | N seeds per generator, report mean ± stddev |
| Metrics | Token usage (input/output/total), wall-clock time, eval score, revisions |

---

## Architecture overview

```
Generators  G = {gpt-4o, claude-sonnet-4-6, deepseek-v4-flash, ...}
Evaluators  E = {gpt-4o-mini, claude-sonnet-4-6, ...}   (disjoint from G)
Seeds       N (default 3)

For each g in G, seed in 1..N:
    report_g_seed = generate(generator=g)          # captures tokens + time
    For each e in E:
        score[g][seed][e] = evaluate(report_g_seed, evaluator=e)   # captures tokens

Aggregate per generator:
    score  = mean over (seeds × evaluators) ± stddev
    tokens = mean over seeds (gen tokens only; eval tokens tracked separately)
    time   = mean over seeds (gen wall-clock only)
```

---

## Module layout (all additive under `src/eval/`)

```
src/eval/
  __init__.py
  results.py     # RunResult, EvalRun, MatrixResult schemas + JSON save/load
  runner.py      # generate one report with a model config; return RunResult
  evaluator.py   # evaluate a fixed report with a given evaluator model; return EvalRun
  matrix.py      # orchestrate G × E × seeds with concurrency
  report.py      # render matrix -> markdown + CSV
  __main__.py    # CLI entry: python -m src.eval ...
```

Refactor (existing files):
- `src/agents.py` → add `build_agents(settings) -> AgentBundle`; keep module-level singletons for backward compat
- `src/orchestrator.py` → `run_report` accepts an optional `AgentBundle`

---

## Red/Green TDD rules for this plan

Every step follows this cycle:
1. **RED** — write a failing test (run `make test`, confirm it fails for the right reason)
2. **GREEN** — write the minimum code to make it pass (run `make test`, confirm green)
3. **REFACTOR** — clean up if needed (run `make lint && make test`, confirm still green)

Tests use the existing `FunctionModel` pattern from `tests/integration/test_orchestrator.py`
and `tests/utils.py`. No real API calls in tests. Test files live in `tests/` matching
source modules (per AGENTS.md).

---

## Phase 0 — Prerequisites check

Before starting, confirm the baseline is green:
```bash
make lint && make test
```
If anything fails, fix it first. Do not build on a red baseline.

---

## Phase 1 — Agent factory refactor (the enabler)

**Goal:** allow building agents from arbitrary settings in-process, so the eval
framework can spin up isolated agent bundles for different models.

### Step 1.1 — `AgentBundle` dataclass

**RED:** `tests/test_agents.py`
```python
class TestAgentBundle:
    def test_build_agents_returns_bundle_with_all_agents(self, sample_settings):
        bundle = build_agents(sample_settings)
        assert bundle.industry_overview is not None
        assert bundle.research is not None
        assert bundle.community_news is not None
        assert bundle.coding_agents is not None
        assert bundle.manager is not None
        assert bundle.evaluator is not None

    def test_bundle_agents_are_distinct_instances(self, sample_settings):
        b1 = build_agents(sample_settings)
        b2 = build_agents(sample_settings)
        assert b1.manager is not b2.manager  # fresh instances per call
```
Run `make test` → fails (no `build_agents`, no `AgentBundle`).

**GREEN:** `src/agents.py`
- Add `AgentBundle` dataclass with 6 fields (4 sub-agents + manager + evaluator)
- Add `build_agents(settings) -> AgentBundle` that constructs all 6 agents with their
  tools attached, exactly as the module-level singletons do today
- Keep the module-level singletons (`industry_overview_agent`, etc.) but build them
  via `build_agents(get_settings())` so existing imports keep working

Run `make test` → green.

### Step 1.2 — `run_report` accepts an `AgentBundle`

**RED:** `tests/integration/test_orchestrator.py` — add:
```python
class TestRunReportWithBundle:
    async def test_run_report_with_explicit_bundle(self, tmp_path, sample_settings):
        # build a bundle with FunctionModel-backed agents
        # call run_report(settings, bundle=bundle)
        # assert it works identically to the override path
```
Run `make test` → fails (`run_report` doesn't accept `bundle`).

**GREEN:** `src/orchestrator.py`
- Change signature: `async def run_report(settings=None, *, bundle=None) -> ReportResult`
- If `bundle is None`, build one: `bundle = build_agents(get_settings() if settings is None else settings)`
- Replace all module-level agent references (`industry_overview_agent`, etc.) with
  `bundle.industry_overview`, etc.
- Existing tests using `agent.override(...)` on module-level singletons still work because
  those singletons are still the same objects (built once at import)

Run `make test` → green (all old tests + new test pass).

### Step 1.3 — Verify `make run` unchanged

```bash
make run  # (or `uv run ai-report --help` to avoid real API calls)
```
Confirm CLI still works. No behavior change for normal use.

---

## Phase 2 — Token usage capture

**Goal:** capture input/output/total token counts per generation run and per
evaluation run.

### Step 2.1 — `TokenUsage` schema

**RED:** `tests/test_eval_results.py`
```python
class TestTokenUsage:
    def test_token_usage_defaults_to_zero(self):
        u = TokenUsage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.total_tokens == 0

    def test_total_is_sum_of_input_and_output(self):
        u = TokenUsage(input_tokens=100, output_tokens=50)
        assert u.total_tokens == 150
```
Run `make test` → fails.

**GREEN:** `src/eval/results.py`
```python
class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
```
Run `make test` → green.

### Step 2.2 — Extract usage from a pydantic-ai result

**RED:** `tests/test_runner.py`
```python
class TestExtractUsage:
    async def test_extract_usage_from_agent_result(self):
        # run an agent with FunctionModel, get result
        # call extract_usage(result)
        # assert TokenUsage with expected counts (FunctionModel reports known counts)
```
Run `make test` → fails (no `extract_usage`).

**GREEN:** `src/eval/runner.py`
```python
def extract_usage(result) -> TokenUsage:
    usage = result.usage()
    return TokenUsage(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
    )
```
Run `make test` → green.

---

## Phase 3 — Result schemas

**Goal:** define the data structures for a single run, a single evaluation, and
the full matrix.

### Step 3.1 — `RunResult` (one generation)

**RED:** `tests/test_eval_results.py`
```python
class TestRunResult:
    def test_run_result_holds_report_and_metrics(self):
        r = RunResult(
            generator_model="gpt-4o",
            seed=1,
            date_range="Jul 19 - Jul 26, 2026",
            report_markdown="# ...",
            sources=[],
            tokens=TokenUsage(input_tokens=1000, output_tokens=500),
            duration_seconds=12.5,
            revision_count=0,
            eval_passed=True,   # from the in-pipeline self-eval (if any)
            eval_score=0.85,
            timestamp="2026-07-28T10:00:00Z",
        )
        assert r.generator_model == "gpt-4o"
        assert r.tokens.total_tokens == 1500
```
Run `make test` → fails.

**GREEN:** add `RunResult` to `src/eval/results.py`. Run `make test` → green.

### Step 3.2 — `EvalRun` (one evaluation of a fixed report)

**RED:** `tests/test_eval_results.py`
```python
class TestEvalRun:
    def test_eval_run_holds_score_and_tokens(self):
        e = EvalRun(
            evaluator_model="claude-sonnet-4-6",
            generator_model="gpt-4o",
            seed=1,
            score=0.82,
            eval_passed=True,
            dimensions=[...],   # list of DimensionScore
            tokens=TokenUsage(input_tokens=200, output_tokens=100),
            duration_seconds=3.2,
            timestamp="2026-07-28T10:01:00Z",
        )
        assert e.score == 0.82
        assert e.tokens.total_tokens == 300
```
Run `make test` → fails.

**GREEN:** add `EvalRun` to `src/eval/results.py`. Run `make test` → green.

### Step 3.3 — `MatrixResult` (full matrix) + JSON persistence

**RED:** `tests/test_eval_results.py`
```python
class TestMatrixResult:
    def test_matrix_result_round_trips_json(self, tmp_path):
        matrix = MatrixResult(
            generators=["gpt-4o"], evaluators=["claude-sonnet-4-6"],
            seeds=2, runs=[...], evals=[...],
        )
        path = matrix.save(tmp_path / "matrix.json")
        loaded = MatrixResult.load(path)
        assert loaded.generators == ["gpt-4o"]
        assert len(loaded.runs) == len(matrix.runs)
```
Run `make test` → fails.

**GREEN:** add `MatrixResult` with `save(path) -> Path` and `load(path) -> MatrixResult`
classmethods using `.model_dump_json()` / `.model_validate_json()`. Run `make test` → green.

---

## Phase 4 — Single generation runner

**Goal:** generate one report with a given generator model config and return a
`RunResult` with tokens + time captured.

### Step 4.1 — `run_single` with FunctionModel

**RED:** `tests/test_runner.py`
```python
class TestRunSingle:
    async def test_run_single_returns_run_result(self, tmp_path, sample_settings):
        # build a bundle with FunctionModel-backed agents (reuse test_orchestrator helpers)
        # override generator model in settings
        # call run_single(settings, generator_model="test-model", seed=1)
        # assert RunResult with report_markdown, tokens, duration_seconds > 0
```
Run `make test` → fails.

**GREEN:** `src/eval/runner.py`
```python
async def run_single(
    settings: Settings,
    generator_model: str,
    seed: int,
    *,
    bundle: AgentBundle | None = None,
) -> RunResult:
    # 1. Clone settings, set sub_agent_model + report_manager_model = generator_model
    # 2. Build bundle (or use provided) via build_agents(cloned_settings)
    # 3. time.perf_counter() before/after
    # 4. call run_report(cloned_settings, bundle=bundle)
    # 5. extract tokens from the run (sum across all agent usages)
    # 6. return RunResult(...)
```
Run `make test` → green.

### Step 4.2 — Token aggregation across the run

**RED:** `tests/test_runner.py`
```python
    async def test_run_single_aggregates_tokens_across_agents(self, ...):
        # use FunctionModels that report known token counts
        # assert RunResult.tokens.total_tokens == sum of all agent usages
```
Run `make test` → fails.

**GREEN:** In `run_single`, accumulate usage from each agent call. The orchestrator's
shared `RunUsage` already tracks this — expose it or sum per-agent. Run `make test` → green.

---

## Phase 5 — Single evaluation runner

**Goal:** evaluate a fixed report string with a given evaluator model and return
an `EvalRun` with score + tokens + time.

### Step 5.1 — `evaluate_single` with FunctionModel

**RED:** `tests/test_evaluator.py`
```python
class TestEvaluateSingle:
    async def test_evaluate_single_returns_eval_run(self, sample_settings, sample_report_markdown):
        # build an evaluator agent with FunctionModel returning a known EvalResult
        # call evaluate_single(settings, report_markdown, evaluator_model="test-eval", generator_model="gpt-4o", seed=1)
        # assert EvalRun with score, tokens, duration_seconds > 0
```
Run `make test` → fails.

**GREEN:** `src/eval/evaluator.py`
```python
async def evaluate_single(
    settings: Settings,
    report_markdown: str,
    evaluator_model: str,
    generator_model: str,
    seed: int,
) -> EvalRun:
    # 1. Clone settings, set evaluator_model = evaluator_model
    # 2. Build only the evaluator agent via build_agents(cloned_settings).evaluator
    #    (or a lighter build_evaluator(settings) helper)
    # 3. time.perf_counter() before/after
    # 4. run evaluator_agent.run(f"Evaluate this newsletter report:\n\n{report_markdown}")
    # 5. extract EvalResult + tokens
    # 6. return EvalRun(...)
```
Run `make test` → green.

---

## Phase 6 — Matrix orchestrator

**Goal:** run the full G × E × seeds matrix with bounded concurrency.

### Step 6.1 — `run_matrix` produces a `MatrixResult`

**RED:** `tests/test_matrix.py`
```python
class TestRunMatrix:
    async def test_run_matrix_small(self, tmp_path, sample_settings):
        # 2 generators, 2 evaluators, 1 seed, FunctionModel-backed
        # call run_matrix(settings, generators=["g1","g2"], evaluators=["e1","e2"], seeds=1, concurrency=2, output_dir=tmp_path)
        # assert MatrixResult with 2 runs and 4 evals (2 runs × 2 evaluators)
        # assert matrix.json exists in output_dir
```
Run `make test` → fails.

**GREEN:** `src/eval/matrix.py`
```python
async def run_matrix(
    settings: Settings,
    generators: list[str],
    evaluators: list[str],
    seeds: int,
    *,
    concurrency: int = 3,
    output_dir: str = "./eval_results",
) -> MatrixResult:
    # 1. For each (generator, seed): run_single -> RunResult (bounded concurrency)
    # 2. For each RunResult, for each evaluator: evaluate_single -> EvalRun (bounded)
    # 3. Assemble MatrixResult, save to output_dir/<timestamp>/matrix.json
```
Run `make test` → green.

### Step 6.2 — Concurrency limit

**RED:** `tests/test_matrix.py`
```python
    async def test_run_matrix_respects_concurrency_limit(self, ...):
        # use a semaphore-tracking FunctionModel that records max concurrent calls
        # assert max concurrent <= concurrency arg
```
Run `make test` → fails.

**GREEN:** wrap generation/evaluation calls in `asyncio.Semaphore(concurrency)`.
Run `make test` → green.

### Step 6.3 — Saves generated reports to disk

**RED:** `tests/test_matrix.py`
```python
    async def test_run_matrix_saves_reports(self, tmp_path, ...):
        # after run_matrix, assert reports/<generator>__seed<n>.md files exist
```
Run `make test` → fails.

**GREEN:** write each `RunResult.report_markdown` to `output_dir/<timestamp>/reports/`.
Run `make test` → green.

---

## Phase 7 — Report rendering

**Goal:** render the matrix into a human-readable summary + CSV.

### Step 7.1 — Markdown summary

**RED:** `tests/test_report.py`
```python
class TestRenderSummary:
    def test_render_summary_produces_ranked_table(self, tmp_path):
        matrix = ...  # build a small MatrixResult
        md = render_summary_markdown(matrix)
        assert "Generator" in md
        assert "Mean Score" in md
        assert "±" in md  # stddev notation
        assert "Mean Tokens" in md
        assert "Mean Time (s)" in md
```
Run `make test` → fails.

**GREEN:** `src/eval/report.py`
```python
def render_summary_markdown(matrix: MatrixResult) -> str:
    # aggregate per generator: mean score ± stddev, mean tokens, mean time
    # sort by mean score descending
    # return markdown table
```
Run `make test` → green.

### Step 7.2 — CSV export

**RED:** `tests/test_report.py`
```python
    def test_render_csv_has_one_row_per_generator(self, ...):
        csv = render_summary_csv(matrix)
        lines = csv.strip().split("\n")
        assert len(lines) == 1 + len(matrix.generators)  # header + rows
```
Run `make test` → fails.

**GREEN:** add `render_summary_csv(matrix) -> str`. Run `make test` → green.

### Step 7.3 — Evaluator agreement table

**RED:** `tests/test_report.py`
```python
    def test_render_evaluator_agreement_shows_per_evaluator_scores(self, ...):
        md = render_evaluator_agreement(matrix)
        for e in matrix.evaluators:
            assert e in md
```
Run `make test` → fails.

**GREEN:** add `render_evaluator_agreement(matrix) -> str` — a table with generators
as rows, evaluators as columns, mean score per cell. Run `make test` → green.

---

## Phase 8 — CLI

**Goal:** `python -m src.eval` runs the full matrix from the command line.

### Step 8.1 — CLI parses args and runs

**RED:** `tests/test_cli.py`
```python
class TestCli:
    def test_cli_parses_args(self, monkeypatch):
        # monkeypatch run_matrix to capture args
        # invoke main(["--generators", "g1,g2", "--evaluators", "e1", "--seeds", "2"])
        # assert run_matrix called with generators=["g1","g2"], evaluators=["e1"], seeds=2
```
Run `make test` → fails.

**GREEN:** `src/eval/__main__.py`
```python
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--generators", required=True)   # comma-separated
    parser.add_argument("--evaluators", required=True)   # comma-separated
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--date-range", default=None)     # defaults to get_date_range()
    parser.add_argument("--output-dir", default="./eval_results")
    args = parser.parse_args(argv)
    settings = get_settings(); settings.configure()
    asyncio.run(run_matrix(settings, ..., output_dir=args.output_dir))
    # print summary path
```
Run `make test` → green.

### Step 8.2 — Add Makefile target

Add to `Makefile`:
```makefile
eval:
	uv run python -m src.eval --generators $(GENERATORS) --evaluators $(EVALUATORS) --seeds $(SEEDS)
```
Run `make eval GENERATORS=gpt-4o EVALUATORS=claude-sonnet-4-6 SEEDS=1` → confirm it
runs (will hit real APIs; use `--seeds 1` for a smoke test).

---

## Phase 9 — Final verification

```bash
make lint && make test
```
Everything green. The existing `make run` path is unchanged.

---

## Output structure

```
eval_results/<timestamp>/
  matrix.json              # full raw data: every run + every eval cell
  summary.md               # ranked table: generator | mean score ± std | mean tokens | mean time
  summary.csv              # flat rows for spreadsheet analysis
  evaluator_agreement.md   # per-evaluator × per-generator score table
  reports/
    gpt-4o__seed1.md
    gpt-4o__seed2.md
    claude-sonnet-4-6__seed1.md
    ...
```

---

## CLI usage

```bash
# Full run
uv run python -m src.eval \
  --generators gpt-4o,claude-sonnet-4-6,deepseek-v4-flash \
  --evaluators gpt-4o-mini,claude-sonnet-4-6 \
  --seeds 3 \
  --concurrency 3

# Smoke test (1 generator, 1 evaluator, 1 seed)
uv run python -m src.eval \
  --generators gpt-4o \
  --evaluators claude-sonnet-4-6 \
  --seeds 1
```

---

## Summary table of phases

| Phase | What | Files touched | Tests added |
|---|---|---|---|
| 0 | Baseline green | — | — |
| 1 | Agent factory refactor | `src/agents.py`, `src/orchestrator.py` | `test_agents.py`, `test_orchestrator.py` |
| 2 | Token usage capture | `src/eval/results.py`, `src/eval/runner.py` | `test_eval_results.py`, `test_runner.py` |
| 3 | Result schemas | `src/eval/results.py` | `test_eval_results.py` |
| 4 | Single generation runner | `src/eval/runner.py` | `test_runner.py` |
| 5 | Single evaluation runner | `src/eval/evaluator.py` | `test_evaluator.py` |
| 6 | Matrix orchestrator | `src/eval/matrix.py` | `test_matrix.py` |
| 7 | Report rendering | `src/eval/report.py` | `test_report.py` |
| 8 | CLI | `src/eval/__main__.py` | `test_cli.py` |
| 9 | Final verification | — | — |

Each phase is independently verifiable. Do not proceed to the next phase until
the current one is green (`make lint && make test`).