"""
Day 4 exercise: AI Tools (browser version, using Streamlit).

The problem this solves: the model has no clock and no internet, so it
cannot correctly answer "what time is it?" -- it can only predict text
from patterns it learned during training (see Day 1 notes). We proved
this earlier: asked directly, it honestly said "I don't have access to
real-time data."

The fix is NOT to train it, and NOT just a cleverer system prompt
(Day 3). It is to give the model a TOOL: a real Python function it can
ask us to run, whose real result we hand back to it.

The flow, step by step (this is the whole concept of "tool calling"):
  1. We describe our Python function to the model (name + what it does)
     as a small JSON "schema" -- see tool_manager.TOOLS.
  2. We send the conversation PLUS that list of available tools.
  3. The model does NOT answer directly if it needs the tool. Instead it
     replies with a special "tool_calls" field saying, in effect,
     "please run get_current_time() for me".
  4. OUR code -- never the model -- actually runs that Python function.
     The model can only ask; it has no way to execute code itself.
  5. We send the function's real return value back to the model as a
     new message with role "tool".
  6. ONLY NOW does the model write its final, correct answer to you.

MERN comparison: step 3 is like a controller that, instead of writing
the HTTP response itself, calls `await userService.getUser(id)` first,
then uses that real data to build the response.

This file ONLY talks to the model and draws the chat UI. Every actual
tool (what it does, how it's run) now lives in tool_manager.py -- this
is the same split you'd make between a controller (this file) and a
services folder (tool_manager.py), so this file doesn't grow every time
a new tool is added.
"""
import os

import ollama
import streamlit as st

from tool_manager import TOOLS, run_tool_call

# A smaller/faster model is a good fit here: tool-calling adds an extra
# back-and-forth round trip (step 3 -> 4 -> 5 -> 6 above), so speed
# matters more than usual. See Day 3 notes on qwen3:4b vs qwen3:1.7b.
MODEL = os.getenv("MODEL", "qwen3:1.7b")

SYSTEM_PROMPT = (
    "You are a helpful assistant. You have access to tools for real-world "
    "information (like the current date/time) that you cannot know on your "
    "own. Use a tool whenever the user's question needs that real data, "
    "instead of guessing."
)


def run_with_tools(messages: list[dict]) -> str:
    """Send the conversation to the model, handle any tool call it asks
    for, and return the final text answer. This is steps 2-6 above."""

    # STEP 2: send the conversation + the list of tools we offer.
    # (TOOLS comes from tool_manager.py -- add a tool there and it
    # automatically becomes available here, with no change to this file.)
    response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
    message = response["message"]

    tool_calls = message.get("tool_calls")
    if not tool_calls:
        # The model answered directly -- no real-world data was needed.
        return message["content"]

    # STEP 3 already happened: `tool_calls` tells us what the model wants
    # to run. Save the model's own request into the conversation too,
    # so it remembers (Day 2 concept) that it asked for a tool.
    messages.append(message)

    # STEPS 4-5: run_tool_call (in tool_manager.py) does the actual
    # work -- looks up the real Python function, runs it with the
    # arguments the model extracted, and returns the result as text.
    for call in tool_calls:
        result = run_tool_call(call)
        messages.append({"role": "tool", "content": result})

    # STEP 6: ask the model again, now that it has the real data, to
    # write its final answer.
    final_response = ollama.chat(model=MODEL, messages=messages)
    return final_response["message"]["content"]


# -------------------------------------------------------------------
# Everything below is the same Streamlit chat-window pattern as
# ../app.py -- see that file's comments for how session_state works.
# -------------------------------------------------------------------
st.set_page_config(page_title="AI Tools Demo", page_icon="🛠️")
st.title("🛠️ AI Tools Demo")
st.caption(f"Running locally with Ollama ({MODEL}) — try asking the time")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

if st.sidebar.button("New chat"):
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.rerun()

st.sidebar.markdown(
    "**Try it:**\n- What time is it?\n- What's the weather in Mumbai?\n"
    "- What is 2 + 2? *(should answer directly, no tool needed)*"
)

for msg in st.session_state.messages:
    # "tool" and the model's own tool-call messages are internal
    # plumbing -- we only show the human-readable user/assistant turns.
    if msg["role"] in ("user", "assistant") and msg.get("content"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = run_with_tools(st.session_state.messages)
                st.markdown(reply)
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply}
                )
            except Exception as exc:  # e.g. Ollama not running / model missing
                st.error(f"{exc}\n\nIs Ollama running? Try: brew services start ollama")
