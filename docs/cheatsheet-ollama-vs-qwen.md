# Cheat Sheet: Ollama vs Qwen

## The rule
> **Ollama = the app that runs models. Qwen = one model that Ollama runs.**

Test: *can I swap it out?* You can swap the model (song, movie, file) but keep the runner (player, Netflix, Node.js).

| Real-world pair | Ollama (the runner) | Qwen (the model) |
|---|---|---|
| Music | Spotify | One song |
| Movies | Netflix / VLC | One movie |
| Code | Node.js | One `.js` file |
| MERN | MongoDB server (`mongod`) | One collection of data |
| Gaming | The console | One game disc |

Say it out loud: **"I run Qwen with Ollama."** Ollama is the engine, Qwen is the fuel.

## The chain
**Company → Model → Runner**

| Company | Model | Runner |
|---|---|---|
| Alibaba | Qwen | Ollama |
| Meta | Llama | Ollama |
| Google | Gemma | Ollama |
| Mistral AI | Mistral | Ollama |

## Three exercises that prove it

**1. See both with one command**
```
ollama list
```
The command is `ollama` (the tool). The rows it prints (`qwen3:4b`, ...) are the models it holds. Ollama is the shelf, models are the books.

**2. Swap the model, keep Ollama**
```
ollama pull llama3.2
MODEL=llama3.2 python chatbot.py
```
The chatbot code is unchanged and Ollama is still running. Only the model changed, so the writing style changes.

**3. Remove a model, keep Ollama**
```
ollama rm qwen3
ollama list
```
The 5.2 GB model is gone, but the `ollama` command still works.

## Useful Ollama commands
| Command | What it does |
|---|---|
| `ollama pull <model>` | Download a model |
| `ollama run <model>` | Chat with a model in the terminal |
| `ollama list` | Show downloaded models |
| `ollama rm <model>` | Delete a model |
| `brew services start ollama` | Start the Ollama server in the background (macOS, Homebrew) |

## Model size and RAM (learned the hard way)
The model must fit in RAM together with everything else. On my 8 GB MacBook Air, the default `qwen3` (5.2 GB) timed out while loading. `qwen3:4b` (2.5 GB) works.
The tag after the colon (`:4b`) is the size: 4 billion parameters. Bigger is smarter but needs more memory.

## Interview one-liner
"Ollama is a local runtime that downloads and serves models over an API on `localhost:11434`. Qwen is an open-weight model family from Alibaba that Ollama can run. The runtime and the model are independent, so I can swap models without changing my application code."
