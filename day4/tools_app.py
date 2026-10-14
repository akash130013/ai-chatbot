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
     as a small JSON "schema", in the `TOOLS` list below.
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
"""
import os
from datetime import datetime

import ollama
import requests
import streamlit as st

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


# --- STEP 1: the real Python function (the "tool" itself) --------------
# This is a completely ordinary function. Nothing about it is "AI" --
# it's exactly the kind of helper you'd write in any Python project.
def get_current_time() -> str:
    return datetime.now().strftime("%I:%M %p on %A, %B %d, %Y")


# A second tool, this time one that calls a REAL external API over the
# internet (Open-Meteo -- free, no API key needed, unlike most weather
# APIs). The pattern is identical to get_current_time(): an ordinary
# Python function, with no AI involved in fetching the data itself.
#
# Note the `city: str` parameter. Unlike get_current_time (which needs
# no input), the model must extract "city" from the user's sentence
# ("weather in Mumbai" -> city="Mumbai") and pass it to this function.
# That's why TOOLS below describes a `parameters` schema for it.
def get_weather(city: str) -> str:
    # Step A: turn a city name into latitude/longitude (geocoding).
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=10,
    ).json()
    if not geo.get("results"):
        return f"Could not find a location named '{city}'."
    place = geo["results"][0]
    lat, lon = place["latitude"], place["longitude"]

    # Step B: ask for the current weather at that latitude/longitude.
    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current_weather": True},
        timeout=10,
    ).json()["current_weather"]

    return (
        f"{place['name']}: {weather['temperature']}°C, "
        f"wind {weather['windspeed']} km/h"
    )


# A lookup table so that when the model says "run get_current_time" (or
# "run get_weather"), our code knows which real Python function that
# name refers to. (This is the beginning of what the course later calls
# a "Tool Manager".)
AVAILABLE_FUNCTIONS = {
    "get_current_time": get_current_time,
    "get_weather": get_weather,
}

# --- STEP 1 (continued): describe the function to the model ------------
# This JSON-like schema is NOT code the model runs. It is just a
# description -- "here is a tool named X, it does Y, it takes these
# arguments" -- so the model knows the tool exists and when to ask for it.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current real-world date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current real-world weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Mumbai' or 'London'",
                    },
                },
                "required": ["city"],
            },
        },
    },
]


def run_with_tools(messages: list[dict]) -> str:
    """Send the conversation to the model, handle any tool call it asks
    for, and return the final text answer. This is steps 2-6 above."""

    # STEP 2: send the conversation + the list of tools we offer.
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

    # STEP 4: WE run the real Python function. The model never executes
    # anything itself -- it only ever asks.
    for call in tool_calls:
        func_name = call["function"]["name"]
        # The model fills in `arguments` itself, e.g. {"city": "Mumbai"},
        # by reading the `parameters` schema we gave it for this tool.
        arguments = call["function"].get("arguments", {})
        func = AVAILABLE_FUNCTIONS.get(func_name)
        result = func(**arguments) if func else f"Error: unknown tool '{func_name}'"

        # STEP 5: hand the real result back as a "tool" message.
        messages.append(
            {
                "role": "tool",
                "content": str(result),
            }
        )

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
    "**Try it:**\n- What time is it?\n- What is today's date?\n"
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
