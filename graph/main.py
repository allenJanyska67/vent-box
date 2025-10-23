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
    attempts: int
    next_action: str


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRIEVANCE_DIR = PROJECT_ROOT / "grievances"
MAX_SANITIZE_ATTEMPTS = 1

# Dedicated moderation model; streaming unnecessary for this workflow.
moderation_llm = ChatOpenAI(model="gpt-4.1", temperature=0).with_config(
    {"tags": ["moderation"]}
)

sanitize_llm = ChatOpenAI(model="gpt-4.1", temperature=0).with_config(
    {"tags": ["sanitize"]}
)


def moderate_grievance(state: ChatState) -> ChatState:
    request = state.get("request") or {}
    level = request.get("level")
    raw_message = request.get("message", "")
    attempts = state.get("attempts", 0)

    if not isinstance(level, int) or not 1 <= level <= 5:
        reason = "Level must be an integer between 1 and 5."
        return {
            "decision": "rejected",
            "reason": reason,
            "sanitized_message": "",
            "response": f"Submission rejected: {reason}",
            "attempts": attempts,
            "next_action": "end",
        }

    if not isinstance(raw_message, str) or not raw_message.strip():
        reason = "Message must be a non-empty string."
        return {
            "decision": "rejected",
            "reason": reason,
            "sanitized_message": "",
            "response": f"Submission rejected: {reason}",
            "attempts": attempts,
            "next_action": "end",
        }

    prompt = (
        "You moderate employee grievances for inclusion in an internal repository. "
        "Assess whether the message contains professional, work-appropriate language. "
        "Reject content that includes harassment, hate speech, personal attacks, or sensitive personal data. "
        "Respond with JSON containing keys:\n"
        '- "decision": either "approved" or "rejected"\n'
        '- "reason": short string explaining the decision\n'
        '- "sanitized_message": professional wording that is safe for storage; if rejected, leave as an empty string\n'
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
        next_action = "sanitize" if attempts < MAX_SANITIZE_ATTEMPTS else "end"
        response_parts = [f"Submission rejected: {reason}"]
        if next_action == "sanitize":
            response_parts.append("Attempting automated sanitization.")
        return {
            "decision": "rejected",
            "reason": reason,
            "sanitized_message": "",
            "response": " ".join(response_parts),
            "attempts": attempts,
            "next_action": next_action,
        }

    sanitized_message = sanitized_message or raw_message.strip()
    return {
        "decision": "approved",
        "reason": reason,
        "sanitized_message": sanitized_message,
        "response": "Message approved. Preparing to store the grievance.",
        "attempts": attempts,
        "next_action": "store",
    }


def sanitize_grievance(state: ChatState) -> ChatState:
    request = state.get("request") or {}
    level = request.get("level")
    raw_message = request.get("message", "")
    attempts = state.get("attempts", 0) + 1

    prompt = (
        "Rewrite the following grievance so it is professional, concise, and safe to store internally. "
        "Remove personal data (names, emails, phone numbers) and any inappropriate language while keeping the core issue. "
        "Respond with JSON containing a single key:\n"
        '- "sanitized_message": the cleaned message\n'
        "Only return JSON."
    )

    llm_response = sanitize_llm.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=raw_message),
        ]
    )

    content = llm_response.content
    if isinstance(content, list):
        content = "".join(str(part) for part in content if part is not None)

    sanitized_text = ""
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            sanitized_text = str(parsed.get("sanitized_message", "")).strip()
        except json.JSONDecodeError:
            sanitized_text = ""

    sanitized_text = sanitized_text or "*Content redacted for policy compliance.*"
    new_request = {"level": level, "message": sanitized_text}

    return {
        "request": new_request,
        "attempts": attempts,
        "response": "Message sanitized. Re-running moderation.",
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
    next_action = state.get("next_action")
    if next_action == "store":
        return "store"
    if next_action == "sanitize":
        return "sanitize"
    return "end"


def store_node(state: ChatState) -> ChatState:
    request = state.get("request") or {}
    level = int(request.get("level"))
    sanitized = state.get("sanitized_message", request.get("message", ""))
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
    workflow.add_node("sanitize", sanitize_grievance)
    workflow.add_node("store", store_node)

    workflow.add_edge(START, "moderate")
    workflow.add_conditional_edges(
        "moderate",
        routing_after_moderation,
        {
            "sanitize": "sanitize",
            "store": "store",
            "end": END,
        },
    )
    workflow.add_edge("sanitize", "moderate")
    workflow.add_edge("store", END)

    return workflow.compile()


graph = build_graph()
