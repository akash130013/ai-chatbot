"""Streamlit web UI for the local Ollama chatbot."""
import os

import ollama
import streamlit as st

MODEL = os.getenv("MODEL", "qwen3:4b")
SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. If the user's question is broad "
    "or could mean several things, ask 1-2 short clarifying questions before "
    "giving a full answer. Otherwise answer directly."
)

st.set_page_config(page_title="AI Chatbot", page_icon="💬")
st.title("💬 AI Chatbot")
st.caption(f"Running locally with Ollama ({MODEL})")

# Streamlit re-runs this script on every interaction, so history lives in session_state.
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

if st.sidebar.button("New chat"):
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.rerun()

for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            stream = ollama.chat(model=MODEL, messages=st.session_state.messages, stream=True)
            reply = st.write_stream(chunk["message"]["content"] for chunk in stream)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as exc:  # e.g. Ollama not running / model missing
            st.session_state.messages.pop()
            st.error(f"{exc}\n\nIs Ollama running? Try: brew services start ollama")
