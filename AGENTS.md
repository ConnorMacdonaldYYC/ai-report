# AGENTS.md — AI Report Agent System
## Running
Run the project with `make run`

## Observability Strategy

| Environment | Backend | Implementation |
|-------------|---------|----------------|
| Local dev | Logfire | `logfire.instrument_pydantic_ai()` — requires logfire auth |

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
- Target Python 3.11+ (`.python-version` and `pyproject.toml`)
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
