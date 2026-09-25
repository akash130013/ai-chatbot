# Day 3: Prompt Engineering — Teach Your AI Assistant How to Behave

## Q1. Why does ChatGPT ask me follow-up questions, but my local bot just answers and stops?
Two reasons:
1. **Instructions.** ChatGPT has hidden instructions (a system prompt) that encourage asking clarifying questions. My bot only had "You are a helpful, concise assistant."
2. **Training style.** ChatGPT got extra training to be conversational. A small local model (qwen3:4b) is less tuned for that, so it defaults to "answer and stop."

**Analogy:** two support agents get "How do I fix my printer?". One was trained to ask the printer model first. The other was only told to answer. Same intelligence, different instructions.
**MERN comparison:** middleware. One server validates the request and replies "400, missing fields". The other just does its best with what it got.

## Q2. What is prompt engineering?
Changing the **instructions you give the model** to change how it behaves, without changing the model itself. It is like rewriting a job description, not retraining the employee.

## Q3. How did I make my bot ask follow-up questions?
By editing the system prompt in `chatbot.py` and `app.py`:
```python
SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. If the user's question is broad "
    "or could mean several things, ask 1-2 short clarifying questions before "
    "giving a full answer. Otherwise answer directly."
)
```

## Q4. What did I observe? (my real test)
| Step | Before the change | After the change |
|---|---|---|
| I asked "what is python?" | Plain definition, then stopped | Bot asked: "Are you asking about Python the programming language or the animal?" |
| I replied "language" | (no follow-up) | Bot gave a full answer about the programming language |

**What this shows:**
- The system prompt changed behaviour with **no change to the model**.
- The second answer worked because of **conversation history**: the bot re-read "what is python?" from the `history` list, so the one-word reply "language" made sense. (Models have no memory of their own.)

## Q5. Did the prompt work perfectly?
Not perfectly. The question about "the animal" was a bit silly, since almost nobody means the snake here. The rule was followed, but a better prompt would say *which* ambiguities are worth asking about.
**Ways to improve it:**
- Give an example inside the prompt of a good clarifying question (called *few-shot prompting*).
- Narrow the rule, e.g. "ask about the user's goal or experience level, not about unlikely meanings."
- Use a bigger model. Small models follow instructions less precisely.

## Q6. What are the parts of a good system prompt?
1. **Role:** who the assistant is ("a senior MERN developer mentor").
2. **Rules:** what to do and not do ("ask 1-2 clarifying questions if the request is broad").
3. **Style:** length, tone, format ("concise, use bullet points").
4. **Examples:** one or two sample exchanges, when the model needs to see the pattern.

## Interview one-liners
- "A system prompt shapes model behaviour without changing the model; it is the cheapest way to customize an LLM."
- "The model is stateless; conversational context comes from re-sending the message history on each request."
- "Prompt behaviour depends on model size and tuning. Smaller models follow instructions less reliably, so I test prompts on the real model."
