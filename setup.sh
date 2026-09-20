#!/usr/bin/env bash
# One-time setup for the AI chatbot project (macOS, Apple Silicon)
# Usage:  cd ~/Documents/Project-2026/ai-chatbot && bash setup.sh
set -e
cd "$(dirname "$0")"

MODEL="${MODEL:-llama3.2}"

echo "==> Checking Homebrew"
if ! command -v brew >/dev/null 2>&1; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi

echo "==> Installing Python 3.12 and Ollama"
brew install python@3.12 ollama

echo "==> Creating project virtualenv in ./.venv"
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo "==> Starting Ollama service"
brew services start ollama
sleep 3

echo "==> Pulling model: $MODEL"
ollama pull "$MODEL"

echo
echo "Done. Run the chatbot with:"
echo "  source .venv/bin/activate && python chatbot.py"
