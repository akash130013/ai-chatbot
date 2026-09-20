# Day 1: Build Your First AI Chatbot with Python

## Q1. What is a language model (LLM)?
A file of billions of learned numbers (parameters). It was trained on huge amounts of text to do one job: predict the next word.
It does not look facts up. It answers from patterns it learned.

**Example:** ask "Where is India?" and it writes "India is a country in South Asia" one word at a time. No database is queried.
**MERN comparison:** MongoDB returns stored records. An LLM is like a developer who has read thousands of tutorials and writes code from memory.

## Q2. What is a hallucination?
When the model confidently says something false, because it predicts likely-sounding words, not verified facts.
**Example:** asking about a tiny npm package and getting functions that do not exist. Always verify important answers.

## Q3. What is a knowledge cutoff?
The model only knows what was in its training data. It cannot know today's news unless you give it a search tool (agents do this later).

## Q4. What are OpenAI, ChatGPT and the `openai` package?
- **OpenAI** is a company. **ChatGPT** is their chat website.
- Behind ChatGPT are models (GPT family). OpenAI also sells access to them through an **API**.
- The **`openai` Python package** (`pip install openai`) is a helper library, like `axios`, that sends requests to that API.
- The API is **paid** and needs an **API key** (a password). Keep it in a `.env` file, never in code or GitHub.

**Example (from the course):**
```python
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from the environment
reply = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
print(reply.choices[0].message.content)
```

## Q5. Can I follow the course without paying OpenAI?
Yes. Ollama exposes an **OpenAI-compatible** API at `http://localhost:11434/v1`. The same `openai` package works. Only two things change:
```python
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # key is ignored
reply = client.chat.completions.create(model="qwen3:4b", messages=[...])
```
**MERN comparison:** like pointing your app from MongoDB Atlas to a local MongoDB. Same driver, different connection string.
**Trade-off:** small local models are less smart and less reliable at tool use than GPT models. Fine for learning.

## Q6. Why does a chatbot "remember" the conversation?
It doesn't. Models have no memory. The code re-sends the whole conversation (`history` list) on every request.
**Example:** remove the `history.append(...)` lines and the bot forgets your name immediately.

## Q7. What is a system prompt?
The first message, with role `system`. It sets behaviour before the user speaks, for example "You are a friendly senior MERN developer."

## Q8. What is streaming?
Receiving the answer piece by piece as it is generated (`stream=True`), so the user sees words appear instead of waiting.

## Q9. What are Ollama and Qwen3?
- **Ollama** runs models locally and serves an API. **Qwen3** is one model (made by Alibaba) that Ollama can run.
- Analogy: Ollama is the music player, Qwen3 is the song.
- Model size must fit in RAM. On an 8 GB Mac use `qwen3:4b` (2.5 GB), not the default 5.2 GB version (it timed out loading).

## Q10. Local model vs ChatGPT: what is the difference?
| | ChatGPT / OpenAI API | Ollama + local model |
|---|---|---|
| Runs on | OpenAI servers | Your laptop |
| Cost | Paid per use | Free |
| Privacy | Data sent to a third party | Stays on your machine |
| Quality | Strongest | Smaller, less capable |
| Needs internet | Yes | No |

## Interview one-liners
- "An LLM predicts the next token; it has no built-in memory, so the application sends conversation history each time."
- "A system prompt sets behaviour; the messages list carries the conversation."
- "I can develop against a local model through an OpenAI-compatible endpoint, and switch to a hosted model by changing base URL, model name and key."

## My project
Terminal bot: [`../chatbot.py`](../chatbot.py). Web UI (Streamlit): [`../app.py`](../app.py).
