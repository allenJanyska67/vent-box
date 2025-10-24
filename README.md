# Chatbot Base

This is a uv-managed Python project scaffold using Streamlit and LangGraph.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) installed

## Quickstart

1. Create the virtual environment (uv uses venvs):

```
uv venv
```

2. Activate it:

- macOS/Linux (bash/zsh): `source .venv/bin/activate`
- Windows (PowerShell): `.venv\\Scripts\\Activate.ps1`

3. Install dependencies:

```
uv sync
```

4. Run Streamlit app (example):

```
streamlit run app.py
```

## CLI submission

Use the utility script to file a grievance without the Streamlit UI:

```
uv run python scripts/add_grievance.py 3 Build pipeline flakes on Mondays
```

The first argument is the level (1-5) and everything after it is treated as the message. Ensure `.env` contains a valid `OPENAI_API_KEY` before running the script so the LangGraph workflow can call OpenAI.

## Notes

- uv manages dependencies via `pyproject.toml` and creates a local `.venv/` by default when you run `uv venv`.
- Secrets for Streamlit should go in `.streamlit/secrets.toml` (already gitignored).
