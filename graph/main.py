import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage


class ChatState(TypedDict, total=False):
    request: Dict[str, Any]
    decision: str
    reason: str
    sanitized_message: str
    storage_path: str
    response: str


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRIEVANCE_DIR = PROJECT_ROOT / "grievances"

# Dedicated moderation model; streaming unnecessary for this workflow.
moderation_llm = ChatOpenAI(model="gpt-4.1", temperature=0).with_config(
    {"tags": ["moderation"]}
)


def moderate_grievance(state: ChatState) -> ChatState:
    request = state.get("request") or {}
    level = request.get("level")
    raw_message = request.get("message", "")

    if not isinstance(level, int) or not 1 <= level <= 5:
        reason = "Level must be an integer between 1 and 5."
        return {
            "decision": "rejected",
            "reason": reason,
            "sanitized_message": "",
            "response": f"Submission rejected: {reason}",
        }

    if not isinstance(raw_message, str) or not raw_message.strip():
        reason = "Message must be a non-empty string."
        return {
            "decision": "rejected",
            "reason": reason,
            "sanitized_message": "",
            "response": f"Submission rejected: {reason}",
        }

    prompt = (
        "You moderate employee grievances for inclusion in an internal repository. "
        "Assess whether the message contains professional, work-appropriate language. "
        "Reject content that includes harassment, hate speech, personal attacks, "
        "or sensitive personal data. Respond with JSON containing keys:\n"
        '- "decision": "approved" or "rejected"\n'
        '- "reason": short string explaining the decision\n'
        '- "sanitized_message": a cleaned version safe for storage (redact names/personal data)\n'
        "Only return JSON."
    )

    payload = {
        "level": level,
        "message": raw_message.strip(),
    }
    llm_response = moderation_llm.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=json.dumps(payload)),
        ]
    )

    content = llm_response.content
    if isinstance(content, list):
        content = "".join(str(part) for part in content if part is not None)

    decision = "rejected"
    reason = "Moderation failed to return a clear decision."
    sanitized_message = raw_message.strip()

    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            decision = str(parsed.get("decision", "rejected")).strip().lower()
            reason = str(parsed.get("reason", reason)).strip() or reason
            sanitized_message = str(
                parsed.get("sanitized_message", sanitized_message)
            ).strip() or sanitized_message
        except json.JSONDecodeError:
            reason = "Unable to parse moderation response."

    if decision != "approved":
        return {
            "decision": "rejected",
            "reason": reason,
            "sanitized_message": sanitized_message,
            "response": f"Submission rejected: {reason}",
        }

    return {
        "decision": "approved",
        "reason": reason,
        "sanitized_message": sanitized_message,
        "response": "Message approved. Preparing to store the grievance.",
    }


@tool
def save_grievance_tool(level: int, sanitized_message: str) -> str:
    """
    Save a grievance to the filesystem.
    """
    sanitized = sanitized_message.strip() or "*Redacted by moderation*"
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    GRIEVANCE_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{timestamp}-level{level}.md"
    file_path = GRIEVANCE_DIR / filename

    content_lines = [
        "# Grievance",
        "",
        f"- Level: {level}",
        f"- Submitted: {timestamp} UTC",
        "",
        sanitized,
        "",
    ]

    file_path.write_text("\n".join(content_lines), encoding="utf-8")

    relative_path = file_path.relative_to(PROJECT_ROOT)
    return str(relative_path)


def routing_after_moderation(state: ChatState) -> str:
    return "store" if state.get("decision") == "approved" else "end"


def store_node(state: ChatState) -> ChatState:
    request = state.get("request") or {}
    level = int(request.get("level"))
    sanitized = state.get("sanitized_message", "")
    sanitized = sanitized.strip() or "*Redacted by moderation*"
    relative_path = save_grievance_tool.invoke(
        {"level": level, "sanitized_message": sanitized}
    )
    return {
        "storage_path": relative_path,
        "response": f"Grievance saved to {relative_path}",
    }


def build_graph():
    workflow = StateGraph(ChatState)
    workflow.add_node("moderate", moderate_grievance)
    workflow.add_node("store", store_node)

    workflow.add_edge(START, "moderate")
    workflow.add_conditional_edges(
        "moderate",
        routing_after_moderation,
        {
            "store": "store",
            "end": END,
        },
    )
    workflow.add_edge("store", END)

    return workflow.compile()


graph = build_graph()
