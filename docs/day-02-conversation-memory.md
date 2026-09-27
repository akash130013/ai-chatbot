# Day 2: Build an AI Assistant That Remembers (Conversation Memory)

## Q1. Does the model remember my earlier messages?
No. LLMs are **stateless**. The application remembers, by re-sending the whole conversation to the model on every request.

**Analogy:** a person who forgets everything as soon as you stop talking. You give them a notebook, and before every reply they read it from page 1.
**MERN comparison:** HTTP is stateless. To remember a user you use a session or JWT that the client sends each time. The model is the stateless server, and the `history` list is the session data.

## Q2. Where is the memory in my code?
```python
history.append({"role": "user", "content": user_input})     # write your message
reply = ollama.chat(model=MODEL, messages=history)           # the model reads ALL of it
history.append({"role": "assistant", "content": reply})      # write its reply too
```
In the Streamlit app the list is `st.session_state.messages`. Both are the same idea.

**Real test:** I said "my name is akash patel", then asked "what is my name?". The bot answered correctly only because the first message was still in the list. After clicking **New chat** (list cleared) it no longer knew.

## Q3. What are the three roles in the messages list?
| Role | Meaning |
|---|---|
| `system` | Instructions that set behaviour (written once, at the start) |
| `user` | What the person typed |
| `assistant` | What the model replied (must be saved too, or the model loses its own side of the chat) |

## Q4. Why must I store the assistant's replies as well?
The model reads the conversation as a script. If only the user lines are saved, the script has questions with no answers, and the model gets confused about what it already said.

## Q5. Where does this simple memory fall short?
| Weakness | Fix (later in the course) |
|---|---|
| Lost when the page refreshes or the app restarts | Save history to a file or database |
| A long chat grows without limit and slows down; models have a **context window** limit | Keep the last N messages (sliding window) or summarize old ones |
| Only remembers inside one chat | Long-term memory: store key facts and retrieve them later |

**Analogy:** current memory is a whiteboard, erased when everyone leaves. Later we add a notebook in a drawer (a database).

## Q6. What is the context window?
The maximum amount of text (measured in tokens, roughly words) the model can read in one request, including the whole history. Beyond it, older messages must be dropped or shortened.

## Q7. Mechanically, how does conversation memory actually work? (Lecture 20)
It's just a Python list of dictionaries, nothing more:
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Akash"},
    {"role": "assistant", "content": "Nice to meet you, Akash!"},
    {"role": "user", "content": "What's my name?"},
]
```
**MERN comparison:** exactly like an array of `{ role, content }` documents you'd store in MongoDB.

Every call to `ollama.chat(model=MODEL, messages=messages)` is a **fresh, independent request** — like a brand-new HTTP call. The model doesn't "stay connected" between your messages. It has zero memory of the previous call unless the old conversation is pasted back into this new one.

**Analogy:** calling customer support where every agent is new. To get help, you re-read your whole case history to them each time, from line one.

The model reads the list top to bottom and predicts what comes next. Given the list above, "My name is Akash" → "Nice to meet you, Akash!" → "What's my name?" makes "Your name is Akash" the most likely next words. It's pattern-completion, not a lookup.

Each turn adds **two** entries: one `user`, one `assistant`. This is why both must be saved (Q4) — otherwise the "script" the model reads has questions with no answers.

## Q8. What is "my chat" precisely, and what happens once it gets too long?
"Chat" = the `messages` list for **one session** (one browser tab, or one terminal run). Every message sent, in that one growing list, gets **resent in full** on every new request. Message #1 is still inside request #200.

Models can only read a limited amount of text at once — the **context window** (e.g. ~4K–128K tokens depending on the model). Once the list is too big to fit:
- **If nothing is done:** the request either **fails** with an error like "context length exceeded", or some tools silently **cut off the oldest messages**, sometimes even losing the `system` prompt — this can garble the conversation unpredictably.
- **The simple fix — a sliding window:** keep the system prompt, plus only the most recent N messages, and drop the rest:
```python
def trim_history(history, keep_last=20):
    system_msgs = [m for m in history if m["role"] == "system"]
    other_msgs = [m for m in history if m["role"] != "system"]
    return system_msgs + other_msgs[-keep_last:]
```
Call this before sending to the model. It's like tearing out the oldest pages of a notebook once it's full — old facts beyond the window are simply forgotten.

**Trade-off:** something asked 150 messages ago won't be remembered anymore. Fine for casual chat; not enough for facts that must persist, which is what long-term memory and RAG solve later in the course.

**Better (more advanced) fixes:** summarize old messages into one short paragraph instead of deleting them outright, and always protect the system prompt from being trimmed.

## Interview one-liners
- "LLMs are stateless; conversational memory is implemented in the application by re-sending message history."
- "History is bounded by the context window, so production systems trim, summarize, or retrieve selectively."
- "Short-term memory lives in the session; long-term memory is persisted in a store and retrieved when relevant."
- "Each chat request is stateless and independent; memory is simulated by resending the full message list, bounded by the model's context window."
