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

## Interview one-liners
- "LLMs are stateless; conversational memory is implemented in the application by re-sending message history."
- "History is bounded by the context window, so production systems trim, summarize, or retrieve selectively."
- "Short-term memory lives in the session; long-term memory is persisted in a store and retrieved when relevant."
