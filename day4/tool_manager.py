"""
Tool Manager (Day 4 refactor, previews course lecture 56).

Why this file exists: as you add more tools, tools_app.py would keep
growing -- every new function, every new schema entry, every new
if/elif to run the right one. That mixes two different concerns in one
file: "how do I talk to the model / draw the UI" vs "what tools exist
and how do I run them". This file is only the second concern.

Everything here is PLAIN PYTHON. Nothing in this file is "AI" --
these are the same kinds of functions you'd write in any project.
The AI part (deciding WHEN to call them) lives in tools_app.py, which
only ever imports from here.
"""
from datetime import datetime

import requests


# ------------------------------------------------------------------
# 1) THE REAL FUNCTIONS -- ordinary Python, nothing AI-specific.
# ------------------------------------------------------------------
def get_current_time() -> str:
    """Reads this computer's own clock. No AI involved."""
    return datetime.now().strftime("%I:%M %p on %A, %B %d, %Y")


def get_weather(city: str) -> str:
    """Calls the free Open-Meteo API (no key needed) for real weather."""
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=10,
    ).json()
    if not geo.get("results"):
        return f"Could not find a location named '{city}'."
    place = geo["results"][0]
    lat, lon = place["latitude"], place["longitude"]

    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current_weather": True},
        timeout=10,
    ).json()["current_weather"]

    return (
        f"{place['name']}: {weather['temperature']}°C, "
        f"wind {weather['windspeed']} km/h"
    )


# ------------------------------------------------------------------
# 2) THE SCHEMAS -- plain-English descriptions the MODEL reads to
#    decide which tool (if any) fits the user's question. See Day 4
#    notes: this matching is semantic ("what's the clock showing?"
#    still matched get_current_time, even with no shared keywords).
#
#    TO ADD A NEW TOOL: write the function above, add one entry here,
#    and one entry in AVAILABLE_FUNCTIONS below. Nothing else changes.
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 3) THE LOOKUP TABLE -- maps a tool's NAME (a string, as the model
#    sends it back) to the REAL Python function it refers to.
# ------------------------------------------------------------------
AVAILABLE_FUNCTIONS = {
    "get_current_time": get_current_time,
    "get_weather": get_weather,
}


def run_tool_call(call: dict) -> str:
    """Given one tool_call from the model's response, actually run the
    matching Python function and return its result as a string.

    This is the one place that does steps 4-5 of the tool-calling flow
    (see day4/tools_app.py's docstring for the full 6-step picture):
    the model only ever ASKS; this function is what actually RUNS code.
    """
    func_name = call["function"]["name"]
    arguments = call["function"].get("arguments", {})
    func = AVAILABLE_FUNCTIONS.get(func_name)

    if func is None:
        return f"Error: unknown tool '{func_name}'"

    try:
        return str(func(**arguments))
    except Exception as exc:  # e.g. a bad/missing argument, API failure
        return f"Error running '{func_name}': {exc}"
