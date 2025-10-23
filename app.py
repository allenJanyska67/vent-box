import json
from json import JSONDecodeError
from typing import Any, Dict

import streamlit as st

from graph.main import graph


st.set_page_config(page_title="Grievance Intake", page_icon="📥")
st.title("Grievance Intake")
st.caption("Provide grievances as JSON. Messages are moderated and stored when approved.")


def render_chat_log():
    for entry in st.session_state.chat_log:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])


def record_message(role: str, content: str) -> None:
    st.session_state.chat_log.append({"role": role, "content": content})


def validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    errors = []
    level = payload.get("level")
    message = payload.get("message")

    if not isinstance(level, int) or not 1 <= level <= 5:
        errors.append("`level` must be an integer between 1 and 5.")

    if not isinstance(message, str) or not message.strip():
        errors.append("`message` must be a non-empty string.")

    if errors:
        raise ValueError(" ".join(errors))

    return {"level": level, "message": message.strip()}


if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

with st.sidebar:
    st.markdown(
        "Example payload:\n"
        "```json\n"
        '{\n  "level": 3,\n  "message": "Build pipeline flakes on Mondays."\n}\n'
        "```"
    )

prompt = st.chat_input('Enter grievance JSON e.g. {"level": 2, "message": "..."}')
if prompt:
    record_message("user", prompt)
    try:
        payload = json.loads(prompt)
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a JSON object with `level` and `message`.")
        cleaned = validate_payload(payload)
    except (JSONDecodeError, ValueError) as exc:
        record_message("assistant", f"Submission rejected: {exc}")
    else:
        result = graph.invoke({"request": cleaned})
        response = result.get("response", "Workflow completed.")

        if result.get("decision") == "approved":
            storage_path = result.get("storage_path")
            if storage_path:
                response += f"\nSaved to `{storage_path}`."
        else:
            reason = result.get("reason")
            if reason:
                response += f"\nReason: {reason}"

        record_message("assistant", response)

render_chat_log()
