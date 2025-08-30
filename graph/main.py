from typing import TypedDict, List

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


class ChatState(TypedDict):
    messages: List[BaseMessage]


# LLM configured to use OpenAI with the provided environment variable OPENAI_API_KEY
llm = ChatOpenAI(model="gpt-5-mini")


def generate_reply(state: ChatState) -> ChatState:
    # Invoke the LLM with current conversation to generate the next message
    ai_msg = llm.invoke(state["messages"])  # returns an AIMessage
    return {"messages": [ai_msg]}


def build_graph():
    workflow = StateGraph(ChatState)

    # Single LLM node that appends an AI response
    workflow.add_node("llm", generate_reply)
    workflow.add_edge(START, "llm")
    workflow.add_edge("llm", END)

    return workflow.compile()


# Export a compiled graph instance for convenience
graph = build_graph()

