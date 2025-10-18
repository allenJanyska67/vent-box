import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk

from graph.main import graph


st.set_page_config(page_title="Chat", page_icon="💬")
st.title("Chat")
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

    # Run LangGraph; stream message updates and try to surface token chunks
    with st.chat_message("assistant"):
        state_in = {"messages": st.session_state.messages}

        # Placeholders for indicator and streaming text
        thinking = st.empty()
        thinking.markdown("⌛ Thinking...")
        output_placeholder = st.empty()

        streamed_any = False
        collected = ""
        final_text = None

        # Primary: stream messages to capture output-tagged chunks
        try:
            for item in graph.stream(state_in, stream_mode="messages"):
                if not isinstance(item, tuple) or len(item) != 2:
                    continue
                msg, meta = item
                tags = (meta or {}).get("tags", []) or []
                # Accumulate only output-tagged chunks
                if isinstance(msg, AIMessageChunk) and "output" in tags:
                    text = msg.content
                    if isinstance(text, list):
                        text = "".join(str(t) for t in text)
                    if text:
                        if not streamed_any:
                            streamed_any = True
                            thinking.empty()
                        collected += text
                        output_placeholder.markdown(collected)
                elif isinstance(msg, AIMessage) and "output" in tags:
                    # Final message (if chunks weren't emitted)
                    final_text = msg.content or ""
        except Exception:
            # Fallback: stream updates and accumulate message chunks if present
            for update in graph.stream(state_in, stream_mode="updates"):
                msgs = update.get("messages")
                if not msgs:
                    continue
                for m in msgs:
                    if isinstance(m, AIMessageChunk):
                        text = m.content
                        if isinstance(text, list):
                            text = "".join(str(t) for t in text)
                        if text:
                            if not streamed_any:
                                streamed_any = True
                                thinking.empty()
                            collected += text
                            output_placeholder.markdown(collected)
                    elif isinstance(m, AIMessage):
                        final_text = m.content or ""

        # If no chunk streaming occurred, show the final message fallback
        thinking.empty()
        if not streamed_any:
            if final_text is None:
                final_text = ""
            collected = final_text
            output_placeholder.markdown(collected)

        # Persist the AI message in the session
        if collected:
            st.session_state.messages.append(AIMessage(content=collected))
