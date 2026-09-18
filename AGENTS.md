# AGENTS.md — AI Report Agent System
## Running
Run the project with `make run`

## Deployment (connorspi)
The report runs weekly on the Raspberry Pi via systemd — unit files are versioned in `scripts/`:

- `ai-report.timer` (Mon 06:00, Persistent) → triggers `ai-report-run.service`
- `ai-report-update.service` pulls `main` from GitHub and runs `uv sync` as `connormacdonald`
- `ai-report-run.service` executes the report as the dedicated `aireport` service user

**Code flows via GitHub only**: push to `main`, and the Pi pulls before each run. Never bundle or scp code.
**Secrets flow via `make deploy`**: it syncs `.env` to the Pi and fixes ownership to `root:aireport 640` (the `aireport` user can read it; `connormacdonald`/opencode cannot — by design). The Pi's `.env` is not directly writable by `connormacdonald`, so deploy scps via `/tmp` + `sudo mv`.
The Pi's Python 3.12 lives in `/opt/uv-python` (shared, so the service user can execute the venv) — the update unit sets `UV_PYTHON_INSTALL_DIR` accordingly.

## Observability Strategy

| Environment | Backend | Implementation |
|-------------|---------|----------------|
| Local dev | Logfire | `logfire.instrument_pydantic_ai()` — uses cached `logfire auth` credentials |
| Pi (weekly cron) | Logfire | Same instrumentation, headless auth via `LOGFIRE_TOKEN` (write token) in `.env` — blank token falls back to dev credentials |

## Code Style Guidelines
Always make sure the .venv is active before running commands
### Linting & Testing
Use the Makefile for all linting and testing — do not run `ruff`, `mypy`, or `pytest` directly:

| Command | What it does |
|---------|--------------|
| `make lint` | Runs `ruff check` + `mypy` |
| `make test` | Runs `pytest` |

Always run `make lint` after making changes.

### Python Version & Syntax
- Target Python 3.12+ (`requires-python` in `pyproject.toml` and `.python-version`)
- Use modern union syntax: `str | None` not `Optional[str]`, `list[str]` not `List[str]`

### Type Annotations
- **All functions must have full type annotations** (mypy `disallow_untyped_defs=true`)
- Return types mandatory, including `-> None`
- Use `TypedDict` for state schemas (`src/state.py`)

### Testing Patterns
- Tests should be robust and focus on testing functionality not internal details. Prefer stable integration tests over brittle unit tests that break every change. 
- Use pydantic ai testing best practices.
- Mock as little as possible
  - ONLY if absolutly NEEDED Mock external deps. Always chekc the `conftest.py` to see if mock exists
- It is preferred to use pytest fixutes to make reusable where possible
- If you are writing the same fixture twice put it in the `conftest.py`
- Use `conftest.py` fixtures (`sample_state`) and `tmp_path` for filesystem tests
- Test files: `test_<module>.py` matching source module, one class per feature area
- ALWAYS write tests after adding new logic
