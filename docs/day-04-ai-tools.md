# Day 4: Teach Your AI Assistant to Use Tools

## Q1. Why can't the model answer "what time is it?" correctly?
The model only knows patterns from training. It has no clock, no internet connection, and no live connection to anything. I tested this directly:

```
Q: What time is it right now?
A: I don't have access to real-time data, so I can't provide the exact current time...
```

At best it admits this honestly (as above). At worst, a less careful model will **hallucinate** a wrong time with full confidence. Either way, it cannot be fixed by a better system prompt (Day 3) — the model simply has no access to that information. It needs a **tool**.

## Q2. What is a tool, mechanically?
A tool is an ordinary Python function, plus a small JSON description of it that gets sent to the model alongside your messages. The model never runs the function itself — it only ever *asks* for it to be run.

**The real flow (tested and confirmed working, see `day4/tools_app.py`):**
1. We describe the function to the model: a `name`, a `description`, and what `parameters` it takes.
2. We send the conversation **plus** that list of tools.
3. If the model decides it needs real data, it replies with `tool_calls` instead of an answer — e.g. "please run `get_current_time()`".
4. **Our code** (never the model) actually runs that Python function.
5. We send the function's real return value back to the model as a new `role: "tool"` message.
6. *Only now* does the model write its final answer, using the real data.

**MERN comparison:** exactly like a controller that calls `await userService.getUser(id)` before building its response — the controller (model) decides *which* service to call; the service itself (our Python function) does the real work.

## Q3. How does the model know *which* tool to call?
It reads the tool's `description` field and matches it against your question — plain pattern-matching, the same thing it does for everything else. I tested this with two tools available at once:

| Question | Tool chosen |
|---|---|
| "What time is it?" | `get_current_time()` |
| "What is 482 plus 319?" | `add_numbers(a=482, b=319)` — correctly extracted both numbers |
| "What is Python?" | *(no tool — answered directly)* |

**The `description` field is doing all the work.** This is prompt engineering again, just applied to tool descriptions instead of a system prompt. A vague description ("does time stuff") makes tool selection unreliable; a precise one makes it reliable.

**MERN comparison:** like an API gateway routing a request to the right microservice by reading each service's registered description — except the "router" here is the LLM, reasoning in English instead of matching a URL path.

## Q4. How does a tool receive arguments, like a city name?
The `parameters` part of the schema tells the model what to extract from the sentence:
```python
"parameters": {
    "properties": {"city": {"type": "string", "description": "The city name..."}},
    "required": ["city"],
}
```
Ask "weather in Mumbai" and the model pulls out `{"city": "Mumbai"}` on its own and sends it back with the tool call. Our code then does:
```python
func(**arguments)   # becomes get_weather(city="Mumbai")
```
This is genuinely new: the model is **parsing structured data out of free text** for us, not just deciding yes/no on a tool.

## Q5. Real example built and tested: `get_weather`
`day4/tools_app.py` has two tools: `get_current_time()` (reads the computer's own clock — no AI involved) and `get_weather(city)` (calls the free Open-Meteo API — no API key needed). Both tested and confirmed working:
```
Q: What is the weather in Mumbai?
A: The current weather in Mumbai is 27.5°C with a wind speed of 8.8 km/h...

Q: What time is it?
A: The current time is 07:43 AM on Wednesday, October 14, 2026.
```
Note: most real weather APIs *do* require an API key (stored in `.env`, same pattern as the tutor's `assistant.py` from Day 2) — Open-Meteo was picked specifically to avoid that for this first exercise.

## Q6. Does ChatGPT have all these functions written already?
**Yes.** This is the key realization tying back to [concepts-why-this-course.md](concepts-why-this-course.md): when you use chat.openai.com, you're not talking to the raw model — you're talking to **the model plus a large collection of tools OpenAI already built**, with exactly this mechanism:
- A **web search tool** (today's news, prices).
- A **code execution tool** ("Code Interpreter" — a real Python sandbox).
- An **image generation tool**, and others, hidden from the user.

**Proof:** call OpenAI's raw API with no `tools=` attached, and it has the *same* limitation your bot has — it can't know the time or browse. The website only feels smarter because OpenAI wired a large toolbox around the same kind of model. The model itself has the same hard limit.

## Interview one-liners
- "An LLM has no access to real-time or external data on its own; any such capability comes from tools — functions described to the model that the application executes on its behalf."
- "Tool selection is driven entirely by the natural-language `description` in the schema — writing good tool descriptions is a form of prompt engineering."
- "A production assistant's 'intelligence' is often mostly its tool layer — code execution, search, retrieval — not the base model's raw knowledge."
