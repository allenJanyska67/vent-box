# Repository Guidelines

## Project Structure & Module Organization
- `app.py` is the Streamlit entry point that wires user chat events to the LangGraph workflow.
- `graph/main.py` defines the `ChatState`, tool bindings, and compiled graph exported as `graph`; extend this file when adding nodes or tools.
- `venting_list.txt` supplies canned prompts for the `parse_vent_list` tool—keep updates idempotent and one entry per line.
- `env.example` lists required environment keys (`OPENAI_API_KEY`); copy it to `.env` for local secrets. Assets and additional configs belong beside their consuming modules to keep navigation predictable.

## Build, Test, and Development Commands
- `uv venv` creates the project virtual environment; run once per machine.
- `source .venv/bin/activate` (macOS/Linux) activates the environment; adapt for Windows as needed.
- `uv sync` installs all runtime and dev dependencies declared in `pyproject.toml`.
- `uv run streamlit run app.py` launches the chat UI at `http://localhost:8501`.
- `uv run langgraph playground graph/main.py` opens an interactive graph preview useful for debugging node flows.

## Coding Style & Naming Conventions
- Follow standard Python 3.10+ conventions: 4-space indentation, type hints for new public functions, and descriptive snake_case names.
- Run `uv run ruff check --fix` before pushing; configure ignores in `pyproject.toml` if necessary.
- Keep Streamlit widget keys stable and prefer pure functions for graph nodes to simplify state reasoning.

## Testing Guidelines
- Add tests under `tests/` naming files `test_<feature>.py`; target pytest parametrization for tool behaviors and graph transitions.
- Execute `uv run pytest` locally; aim for meaningful coverage on graph state changes and Streamlit helpers.
- Mock external APIs (e.g., OpenAI) via fixtures to keep runs deterministic.

## Commit & Pull Request Guidelines
- Match existing history with short, imperative commit titles (`add tool`, `run langgraph`) and scoped changes per commit.
- Fill PR descriptions with context: problem statement, solution notes, manual test steps, and screenshots of UI tweaks.
- Link related issues and call out migration steps or new environment keys so reviewers can validate deployment impact.

## Security & Configuration Tips
- Never commit `.env` or `.streamlit/secrets.toml`; rely on the provided template and document new keys.
- Validate that `venting_list.txt` additions contain non-sensitive phrasing, as the tool reads it verbatim into responses.
