"""
Day 7 exercise: a real planning AGENT, vs Day 4's one-shot tool use.

Day 4's run_with_tools() (see ../day4/tools_app.py) does exactly ONE
round: ask the model -> maybe call ONE tool -> answer. That's fine for
a question that needs a single fact. It breaks down for a question
that needs MULTIPLE facts, one at a time, where the second tool call
might even depend on knowing the first result.

An AGENT is that same tool-calling mechanism, wrapped in a LOOP:
    ask -> call a tool -> look at the result -> decide: done, or need
    another tool? -> call another tool if so -> ... -> repeat until
    the model finally answers with no more tool_calls.

This file reuses the exact tools from Day 4 (get_current_time,
get_weather) -- nothing new to learn about tools themselves. The only
new thing is the LOOP.
"""
import os
import sys

import ollama

# Reuse Day 4's tools instead of rewriting them -- same TOOLS schema,
# same run_tool_call(), proven to work already.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "day4"))
from tool_manager import TOOLS, run_tool_call  # noqa: E402

MODEL = os.getenv("MODEL", "qwen3:1.7b")
MAX_STEPS = 5  # a safety limit -- never loop forever if something goes wrong

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. The user's question "
    "may need MORE THAN ONE tool call to fully answer -- for example, two "
    "separate pieces of information. Call tools one at a time as needed. "
    "Only give your final answer once you have everything required."
)


def run_agent(question: str) -> tuple[str, int]:
    """The agent loop. Returns (final_answer, number_of_steps_taken)."""
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(1, MAX_STEPS + 1):
        response = ollama.chat(model=MODEL, messages=history, tools=TOOLS)
        message = response["message"]

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            # The model decided it has everything it needs -- stop looping.
            return message["content"], step

        # The agent "planned" to use a tool. Log it so you can SEE the
        # planning happen, not just the final answer.
        for call in tool_calls:
            print(f"  [step {step}] agent calls: {call['function']['name']}"
                  f"({call['function'].get('arguments', {})})")

        history.append(message)
        for call in tool_calls:
            result = run_tool_call(call)
            history.append({"role": "tool", "content": result})

    return "Agent stopped: too many steps without a final answer.", MAX_STEPS


if __name__ == "__main__":
    question = "What time is it, and what is the weather in Mumbai?"
    print(f"Q: {question}\n")
    answer, steps = run_agent(question)
    print(f"\nA: {answer}")
    print(f"\n(Took {steps} loop step(s) to reach a final answer.)")
