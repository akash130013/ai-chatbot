# AI Chatbot (Python + Ollama)

Local chatbot running on your Mac with Ollama — no API keys, no cloud.

## Setup (one time)
    cd ~/Documents/Project-2026/ai-chatbot
    bash setup.sh

Installs Python 3.12 + Ollama via Homebrew, creates `.venv/` in this folder,
installs the `ollama` Python package, and pulls `llama3.2` (~2 GB).
Use a different model: `MODEL=qwen2.5:7b bash setup.sh`

## Run
    source .venv/bin/activate
    python chatbot.py
