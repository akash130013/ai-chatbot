# Concepts: The Model Is Already Trained — So What Are We Building?

## Q1. If Qwen3 already gives correct answers, what am I actually building?
Nothing you build makes the model smarter. Qwen3 (or GPT, or any LLM) was already fully trained by its maker before you ever touched it. Training is done. You are building the **application layer around it**: memory, instructions, data access, and the ability to take actions.

**Analogy:** the model is a well-read employee who already knows grammar, facts, code, and math. You didn't teach them anything. You gave them a desk, a notepad, and a job description.

## Q2. What does my current code (Day 1–3) actually add?
| Piece | What it does | Without it |
|---|---|---|
| `history` list | Lets the model "remember" earlier turns in a conversation | It forgets your name every message |
| System prompt | Sets behaviour ("be concise", "you are a MERN mentor") | It answers however it likes |
| Streamlit UI ([app.py](app.py)) | A chat window instead of a raw terminal | You'd type commands with no UI |

None of this changes the model's knowledge. It only organizes how you talk to it.

## Q3. What are the model's two big limits, and how does the rest of the course fix them?

**Limit 1: it doesn't know your private/local data.**
Ask it about a PDF on your laptop or your company's internal docs — it has never seen them, and it will guess or refuse.
**Fix — RAG (Day 6, 12):** your code searches your own documents for the relevant piece of text and pastes it into the prompt before asking the model. The model isn't retrained; you hand it a "cheat sheet" for that one question.
**MERN analogy:** like an Express route that does `Model.find(...)` in MongoDB first, then puts those results into the response, instead of letting the client guess.

**Limit 2: it can only talk, it can't act.**
It can't check today's weather, query a live database, or send an email by itself.
**Fix — Tools & Agents (Day 4, 7+):** your code lets the model say "call this function with these arguments." Your code actually runs that function (an API call, a DB query, a file read) and feeds the result back to the model. This loop — think, call a tool, read the result, respond — is what "agent" means.
**MERN analogy:** the model becomes like a controller that decides *which* service to call; you still write the services.

## Q4. So what is "agentic AI" in one sentence?
**Agentic AI is the software you build around a pretrained model — memory, instructions, data retrieval, and tools — that turns a smart text predictor into an application that can answer from your data and take real actions.** That surrounding software is what this whole course teaches, section by section.

## Interview one-liners
- "The LLM itself is static after training; applications add memory, retrieval (RAG), and tool-calling on top of it."
- "RAG doesn't retrain the model — it injects relevant retrieved text into the prompt at request time."
- "An agent is a loop: the model proposes an action (a tool call), the application executes it, and the result is fed back into the conversation."
