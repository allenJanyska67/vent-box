import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from .vector_store import find_similar_topic, upsert_grievance_embedding


class ChatState(TypedDict, total=False):
    request: Dict[str, Any]
    decision: str
    reason: str
    sanitized_message: str
    storage_path: str
    raw_message: str
    raw_storage_path: str
    response: str
    attempts: int
    next_action: str
    summary: str
    topic_key: str
    occurrences: int
    ticket_path: str
    next_steps: str
    similarity: float
    matched_metadata: Dict[str, Any]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRIEVANCE_DIR = PROJECT_ROOT / "grievances"
RAW_GRIEVANCE_DIR = GRIEVANCE_DIR / "raw_submissions"
TICKETS_DIR = GRIEVANCE_DIR / "tickets"
INDEX_PATH = GRIEVANCE_DIR / "index.json"
MAX_SANITIZE_ATTEMPTS = 2
MIN_TICKET_THRESHOLD = 1
MAX_TICKET_THRESHOLD = 5

# Dedicated moderation model; streaming unnecessary for this workflow.
moderation_llm = ChatOpenAI(model="gpt-4.1", temperature=0).with_config(
    {"tags": ["moderation"]}
)

sanitize_llm = ChatOpenAI(model="gpt-4.1", temperature=0).with_config(
    {"tags": ["sanitize"]}
)

analysis_llm = ChatOpenAI(model="gpt-4.1", temperature=0).with_config(
    {"tags": ["analysis"]}
)


def normalize_topic_key(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    slug = "-".join(part for part in cleaned.split("-") if part)
    return slug or "general-issue"


def render_ticket_content(topic_key: str, entry: Dict[str, Any]) -> str:
    summary = entry.get("summary", "").strip() or "Summary pending."
    next_steps = entry.get("next_steps", "").strip()
    occurrences = entry.get("occurrences", 0)
    highest_level = entry.get("highest_level", 0)
    grievances = entry.get("grievances", [])

    lines = [
        f"# Ticket: {topic_key}",
        "",
        f"- Summary: {summary}",
        f"- Occurrences: {occurrences}",
        f"- Highest Level: {highest_level}",
        "",
    ]

    if next_steps:
        lines.extend(["## Next Steps", next_steps, ""])

    lines.append("## Related Grievances")
    if grievances:
        for item in grievances:
            file_ref = item.get("file", "unknown")
            level = item.get("level", "n/a")
            lines.append(f"- Level {level} → `{file_ref}`")
    else:
        lines.append("- None recorded yet.")

    lines.extend(["", "## Recent Notes"])
    if grievances:
        for item in grievances[-5:]:
            note = item.get("message", "").strip() or "*No message provided.*"
            lines.append(f"- {note}")
    else:
        lines.append("- No additional context captured.")

    lines.append("")
    return "\n".join(lines)


def calculate_ticket_threshold(grievances: list[Dict[str, Any]]) -> int:
    """Derive how many reports are needed before creating a ticket."""
    if not grievances:
        return MAX_TICKET_THRESHOLD

    levels: list[int] = []
    for item in grievances:
        try:
            level = int(item.get("level", 1))
        except (TypeError, ValueError):
            continue
        if 1 <= level <= 5:
            levels.append(level)
    if not levels:
        return MAX_TICKET_THRESHOLD

    avg_level = sum(levels) / len(levels)
    normalized = (avg_level - 1) / 4  # maps 1-5 → 0-1
    threshold_span = MAX_TICKET_THRESHOLD - MIN_TICKET_THRESHOLD
    dynamic_threshold = MAX_TICKET_THRESHOLD - (normalized * threshold_span)
    return max(
        MIN_TICKET_THRESHOLD,
        min(MAX_TICKET_THRESHOLD, round(dynamic_threshold)),
    )


def moderate_grievance(state: ChatState) -> ChatState:
    request = state.get("request") or {}
    level = request.get("level")
    raw_message = request.get("message", "")
    attempts = state.get("attempts", 0)
    original_message = state.get("raw_message") or raw_message

    if not isinstance(level, int) or not 1 <= level <= 5:
        reason = "Level must be an integer between 1 and 5."
        return {
            "decision": "rejected",
            "reason": reason,
            "sanitized_message": "",
            "response": f"Submission rejected: {reason}",
            "attempts": attempts,
            "next_action": "end",
            "raw_message": original_message,
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
            "raw_message": original_message,
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
            "raw_message": original_message,
        }

    sanitized_message = sanitized_message or raw_message.strip()
    return {
        "decision": "approved",
        "reason": reason,
        "sanitized_message": sanitized_message,
        "response": "Message approved. Preparing to store the grievance.",
        "attempts": attempts,
        "raw_message": original_message,
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


def summarize_grievance(state: ChatState) -> ChatState:
    request = state.get("request") or {}
    level = request.get("level", 1)
    sanitized = state.get("sanitized_message") or request.get("message", "")
    sanitized = (sanitized or "").strip()

    payload = {
        "message": sanitized,
        "level": level,
    }
    prompt = (
        "Summarize the following grievance for a tracking index. "
        "Return JSON with keys:\n"
        '- "summary": concise sentence (<25 words) capturing the core issue\n'
        '- "topic_key": lowercase slug (letters, numbers, hyphens) suitable as an identifier\n'
        '- "next_steps": short recommended follow-up action (single sentence)\n'
        "Only return JSON."
    )

    llm_response = analysis_llm.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=json.dumps(payload)),
        ]
    )

    content = llm_response.content
    if isinstance(content, list):
        content = "".join(str(part) for part in content if part is not None)

    summary = ""
    topic_key = ""
    next_steps = ""
    match_similarity = 0.0
    matched_metadata: Dict[str, Any] = {}
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            summary = str(parsed.get("summary", "")).strip()
            topic_key = str(parsed.get("topic_key", "")).strip()
            next_steps = str(parsed.get("next_steps", "")).strip()
        except json.JSONDecodeError:
            summary = ""

    summary = summary or (sanitized[:120] + ("…" if len(sanitized) > 120 else ""))
    topic_key = normalize_topic_key(topic_key or summary)
    next_steps = next_steps or "Review and prioritise with the project owner."

    match = find_similar_topic(sanitized)

    if match:
        match_topic = normalize_topic_key(str(match.get("topic_key", "")))
        if match_topic:
            topic_key = match_topic
            match_similarity = float(match.get("similarity", 0.0))
            matched_metadata = dict(match.get("metadata") or {})

    existing = state.get("response")
    response_lines = [existing] if existing else []
    response_lines.append(f"Summary: {summary}")
    response_lines.append(f"Topic key: {topic_key}")
    if match_similarity:
        response_lines.append(
            f"Existing topic match similarity: {match_similarity:.2f}"
        )

    return {
        "summary": summary,
        "topic_key": topic_key,
        "next_steps": next_steps,
        "similarity": match_similarity,
        "matched_metadata": matched_metadata,
        "response": "\n".join(response_lines),
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


def save_raw_grievance(raw_message: str, sanitized_relative_path: str, level: int) -> str:
    """
    Save the raw grievance submission to a gitignored directory.
    """
    if not isinstance(raw_message, str):
        return ""
    if not raw_message.strip():
        return ""

    RAW_GRIEVANCE_DIR.mkdir(parents=True, exist_ok=True)
    base_name = Path(sanitized_relative_path).stem if sanitized_relative_path else ""
    if not base_name:
        base_name = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    raw_filename = f"{base_name}-raw.txt"
    raw_file = RAW_GRIEVANCE_DIR / raw_filename

    content_lines = [
        "# Raw Grievance Submission",
        "",
        f"- Level: {level}",
        f"- Sanitized File: {sanitized_relative_path}",
        "",
        raw_message,
        "",
    ]

    raw_file.write_text("\n".join(content_lines), encoding="utf-8")
    return str(raw_file.relative_to(PROJECT_ROOT))


def catalog_grievance(state: ChatState) -> ChatState:
    request = state.get("request") or {}
    level = int(request.get("level", 1))
    sanitized = state.get("sanitized_message") or request.get("message", "")
    sanitized = sanitized.strip() or "*Content redacted for policy compliance.*"
    summary = state.get("summary", "").strip()
    topic_key = state.get("topic_key", "general-issue").strip()
    topic_key = normalize_topic_key(topic_key)
    next_steps = state.get("next_steps", "").strip()
    storage_path = state.get("storage_path", "")
    raw_storage_path = state.get("raw_storage_path", "")

    index_data: Dict[str, Any] = {"topics": {}}
    if INDEX_PATH.exists():
        try:
            with INDEX_PATH.open("r", encoding="utf-8") as fh:
                index_data = json.load(fh) or {"topics": {}}
        except (json.JSONDecodeError, OSError):
            index_data = {"topics": {}}

    topics = index_data.setdefault("topics", {})
    entry = topics.get(topic_key, {})

    grievances = entry.get("grievances", [])
    if storage_path and not any(item.get("file") == storage_path for item in grievances):
        entry_data = {
            "file": storage_path,
            "level": level,
            "message": sanitized,
        }
        if raw_storage_path:
            entry_data["raw_file"] = raw_storage_path
        grievances.append(entry_data)

    occurrences = len(grievances)
    highest_level = max([level] + [item.get("level", 0) for item in grievances])

    entry.update(
        {
            "summary": summary,
            "next_steps": next_steps,
            "grievances": grievances,
            "occurrences": occurrences,
            "highest_level": highest_level,
        }
    )

    ticket_path = entry.get("ticket_path", "")
    ticket_threshold = calculate_ticket_threshold(grievances)
    TICKETS_DIR.mkdir(parents=True, exist_ok=True)

    if occurrences >= ticket_threshold:
        ticket_filename = f"{topic_key}.md"
        ticket_file = TICKETS_DIR / ticket_filename
        ticket_content = render_ticket_content(topic_key, entry)
        ticket_file.write_text(ticket_content, encoding="utf-8")
        ticket_path = str(ticket_file.relative_to(PROJECT_ROOT))
        entry["ticket_path"] = ticket_path
    else:
        entry.setdefault("ticket_path", "")
    entry["ticket_threshold"] = ticket_threshold

    topics[topic_key] = entry

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INDEX_PATH.open("w", encoding="utf-8") as fh:
        json.dump(index_data, fh, indent=2, ensure_ascii=False)

    document_id = storage_path or f"{topic_key}-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
    upsert_grievance_embedding(
        document_id=document_id,
        message=sanitized,
        topic_key=topic_key,
        summary=summary,
        extra_metadata={
            "level": level,
            "storage_path": storage_path,
            "raw_storage_path": raw_storage_path,
            "ticket_path": ticket_path,
        },
    )

    response_lines = []
    if state.get("response"):
        response_lines.append(state["response"])
    response_lines.append(
        f"Occurrences for {topic_key}: {occurrences} (threshold {ticket_threshold})"
    )
    if occurrences >= ticket_threshold:
        response_lines.append(f"Recurring issue tracked at `{ticket_path}`.")
    else:
        response_lines.append(
            "Report logged; monitoring until it meets the escalation threshold."
        )
    if next_steps:
        response_lines.append(f"Suggested next step: {next_steps}")

    return {
        "occurrences": occurrences,
        "ticket_path": ticket_path,
        "response": "\n".join(response_lines),
    }


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
    raw_message = state.get("raw_message", "")
    raw_storage_path = save_raw_grievance(raw_message, relative_path, level)
    response_lines = [f"Grievance saved to {relative_path}"]
    if raw_storage_path:
        response_lines.append(f"Raw submission stored at {raw_storage_path}")
    return {
        "storage_path": relative_path,
        "raw_storage_path": raw_storage_path,
        "response": "\n".join(response_lines),
    }


def build_graph():
    workflow = StateGraph(ChatState)
    workflow.add_node("moderate", moderate_grievance)
    workflow.add_node("sanitize", sanitize_grievance)
    workflow.add_node("store", store_node)
    workflow.add_node("summarize", summarize_grievance)
    workflow.add_node("catalog", catalog_grievance)

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
    workflow.add_edge("store", "summarize")
    workflow.add_edge("summarize", "catalog")
    workflow.add_edge("catalog", END)

    return workflow.compile()


graph = build_graph()
