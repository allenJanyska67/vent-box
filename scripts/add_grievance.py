#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Ensure project root is on sys.path so `graph` resolves when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.main import graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a grievance via the LangGraph workflow."
    )
    parser.add_argument(
        "level",
        type=int,
        help="Severity level from 1 (minor) to 5 (critical).",
    )
    parser.add_argument(
        "message",
        nargs=argparse.REMAINDER,
        help="Grievance text. Everything after the level is treated as the message.",
    )
    return parser.parse_args()


def ensure_message(parts: list[str]) -> str:
    message = " ".join(parts).strip()
    if not message:
        print(
            "Error: message is required. Provide text after the level argument.",
            file=sys.stderr,
        )
        sys.exit(2)
    return message


def ensure_level(level: int) -> int:
    if 1 <= level <= 5:
        return level
    print("Error: level must be an integer between 1 and 5.", file=sys.stderr)
    sys.exit(2)


def ensure_api_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    print(
        "Error: OPENAI_API_KEY is not set. Copy env.example to .env and fill it in.",
        file=sys.stderr,
    )
    sys.exit(2)


def pretty_print(result: Dict[str, Any]) -> None:
    decision = result.get("decision") or "unknown"
    print(f"\nModeration decision: {decision}")
    reason = result.get("reason")
    if reason:
        print(f"Reason: {reason}")

    sanitized = result.get("sanitized_message")
    if sanitized:
        print(f"Sanitized message: {sanitized}")

    response = result.get("response")
    if response:
        print(f"\nWorkflow response:\n{response}")

    summary = result.get("summary")
    if summary:
        print(f"\nSummary: {summary}")

    topic_key = result.get("topic_key")
    if topic_key:
        print(f"Topic key: {topic_key}")

    next_steps = result.get("next_steps")
    if next_steps:
        print(f"Suggested next step: {next_steps}")

    occurrences = result.get("occurrences")
    if occurrences is not None:
        print(f"Occurrences recorded: {occurrences}")

    storage_path = result.get("storage_path")
    if storage_path:
        print(f"Saved file: {storage_path}")

    ticket_path = result.get("ticket_path")
    if ticket_path:
        print(f"Ticket: {ticket_path}")


def main() -> None:
    load_dotenv()
    args = parse_args()

    level = ensure_level(args.level)
    message = ensure_message(args.message)
    ensure_api_key()

    print(f"Submitting level {level} grievance...")
    print(f"Message: {message}\n")

    try:
        result = graph.invoke({"request": {"level": level, "message": message}})
    except Exception as exc:  # pragma: no cover - surface errors to the CLI
        print(f"Graph invocation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    pretty_print(result)


if __name__ == "__main__":
    main()
