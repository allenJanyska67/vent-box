import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from graph.main import graph


st.set_page_config(page_title="Structural Therapy Chat", page_icon="💬")
st.title("Structural Therapy Chat")
st.caption("Powered by LangGraph + OpenAI (gpt-5-mini)")


if "messages" not in st.session_state:
    st.session_state.messages = []  # list[BaseMessage]


# Render existing conversation
for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)


prompt = st.chat_input("Ask something...")
if prompt:
    # Add user message
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run LangGraph to get assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            state_in = {"messages": st.session_state.messages}
            state_out = graph.invoke(state_in)
            # graph returns partial updates; append new AI message(s)
            new_messages = state_out.get("messages", [])
            # Keep only newly generated messages (AI)
            for m in new_messages:
                if isinstance(m, AIMessage):
                    st.session_state.messages.append(m)
                    st.markdown(m.content)

