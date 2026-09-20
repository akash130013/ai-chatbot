"""Minimal terminal chatbot using a local Ollama model."""
import os
import sys

import ollama

MODEL = os.getenv("MODEL", "qwen3:4b")
SYSTEM_PROMPT = "You are a helpful, concise assistant."


def main() -> None:
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"Chatbot ready (model: {MODEL}). Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        history.append({"role": "user", "content": user_input})
        print("Bot: ", end="", flush=True)
        reply = ""
        try:
            for chunk in ollama.chat(model=MODEL, messages=history, stream=True):
                piece = chunk["message"]["content"]
                reply += piece
                print(piece, end="", flush=True)
        except Exception as exc:  # e.g. Ollama not running / model missing
            print(f"\n[error] {exc}\nIs Ollama running? Try: brew services start ollama")
            history.pop()
            continue
        print("\n")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    sys.exit(main())
