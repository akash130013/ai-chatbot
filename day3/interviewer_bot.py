"""
Day 3 exercise: Prompt Engineering.

This file is a COPY of ../chatbot.py with only ONE real change: the
SYSTEM_PROMPT text. Nothing about the code's *logic* changes -- no new
loops, no new libraries, no new functions. This is the whole point of
prompt engineering: you reshape the assistant's behaviour just by
rewriting the instructions you hand it, the same way you'd rewrite a
job description without retraining the employee.

Role for this file: "Interviewer" -- a mock technical interview bot that
asks exactly one question at a time and waits for your answer.
"""
import os
import sys

import ollama

# Same as chatbot.py: which local Ollama model to talk to.
MODEL = os.getenv("MODEL", "qwen3:4b")

# --- THE ONLY THING THAT CHANGED VS. chatbot.py -----------------------
# This is a system prompt: a message with role "system" that sets the
# assistant's behaviour before the user says anything. It is NOT the
# same as training -- the model's weights are untouched. We are only
# changing what instructions get sent alongside every request.
#
# Notice the structure (this is a reusable pattern for ANY custom role):
#   1. Role     -> who the assistant is
#   2. Rules    -> the strict behaviour we require (numbered, explicit)
#   3. Style    -> how long/short replies should be
#   4. Stop     -> what it must NOT do (e.g. give away answers)
INTERVIEWER_PROMPT = """You are a technical interviewer conducting a mock interview.

Rules you must always follow:
1. Ask exactly ONE question at a time. Never ask multiple questions in one message.
2. Wait for the candidate's answer before asking the next question.
3. Do not give the answer or any hints unless the candidate explicitly asks for a hint.
4. After the candidate answers, give brief feedback (1-2 sentences: correct,
   partially correct, or wrong and why), then ask the next question.
5. Topic: JavaScript and the MERN stack. Start easy, then increase difficulty gradually.
6. Never break character or explain that you are an AI.

Begin the interview now with your first question."""


def main() -> None:
    # `history` is our "notebook" -- see Day 2 notes. The model itself
    # remembers nothing; we resend this whole list on every turn.
    history = [{"role": "system", "content": INTERVIEWER_PROMPT}]

    print(f"Interviewer bot ready (model: {MODEL}). Type 'exit' to quit.\n")

    # The very first turn: let the model speak first (it was told to
    # "begin the interview now"), so we ask Ollama to reply BEFORE we
    # read any input from the user.
    reply = _ask(history)
    history.append({"role": "assistant", "content": reply})

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        # Save your answer to the notebook (Day 2 concept).
        history.append({"role": "user", "content": user_input})

        reply = _ask(history)
        # Save the assistant's reply too -- skipping this would make the
        # bot "forget" its own earlier questions (see Day 2, Q4).
        history.append({"role": "assistant", "content": reply})


def _ask(history: list[dict]) -> str:
    """Send the whole conversation so far to Ollama and stream the reply."""
    print("Interviewer: ", end="", flush=True)
    reply = ""
    for chunk in ollama.chat(model=MODEL, messages=history, stream=True):
        piece = chunk["message"]["content"]
        reply += piece
        print(piece, end="", flush=True)
    print()
    return reply


if __name__ == "__main__":
    sys.exit(main())
