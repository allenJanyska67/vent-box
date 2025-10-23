from typing import TypedDict, List

from typing_extensions import Annotated
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv(".env")

class ChatState(TypedDict):
    # Use add_messages reducer so nodes can append messages
    messages: Annotated[List[BaseMessage], add_messages]
    vent_list: List[str]


# LLM configured to use OpenAI with the provided environment variable OPENAI_API_KEY
# Enable streaming and tag events to identify user message token stream in the UI
llm = ChatOpenAI(model="gpt-5-mini", streaming=True).with_config({"tags": ["output"]})

@tool
def parse_vent_list(state: ChatState) -> ChatState:
    with open("venting_list.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            state["vent_list"].append(line.strip())
    return {"vent_list": state["vent_list"]}

model_with_tools = llm.bind_tools([parse_vent_list], tool_choice="any")

def chain(state: ChatState) -> ChatState:
    messages = state["messages"]
    ai_msg = model_with_tools.invoke(messages)
    return {"messages": [ai_msg], "vent_list": state["vent_list"]}


def build_graph():
    workflow = StateGraph(ChatState)

    # Single LLM node that appends an AI response (stream-enabled)
    workflow.add_node("llm", chain)
    workflow.add_edge(START, "llm")
    workflow.add_edge("llm", END)

    return workflow.compile()


# Export a compiled graph instance for convenience
graph = build_graph()