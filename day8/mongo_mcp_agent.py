"""
Day 8: an AI agent whose tools come from an MCP server, now needing a
REAL multi-step plan: discover which collection to use, THEN query it.

Unlike the single-round version we tried first, this one wraps the
same request/response pattern in a LOOP (the Day 7 agent concept),
because the second tool call's choice (which collection) depends on
what the FIRST tool call (list_collections) returned. This is the
genuine sequential-dependency case Day 7's single test didn't prove.
"""
import asyncio
import os
import sys

import ollama
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

MODEL = os.getenv("MODEL", "qwen3:1.7b")
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mongo_mcp_server.py")
MAX_STEPS = 5

SYSTEM_PROMPT = (
    "You are a debugging assistant for a MongoDB database with MANY "
    "collections -- you do not know their names or fields in advance. "
    "Always follow these steps in order: "
    "1) list_collections() to see what's available, "
    "2) describe_collection() on the one that looks relevant, to get its "
    "real field names, "
    "3) query_collection() using those exact field names. "
    "Never guess a collection name or field name without checking first."
)


def mcp_tool_to_ollama_schema(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


async def main() -> None:
    question = "Which products are out of stock?"
    print(f"Q: {question}\n")

    server_params = StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = (await session.list_tools()).tools
            ollama_tools = [mcp_tool_to_ollama_schema(t) for t in mcp_tools]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]

            for step in range(1, MAX_STEPS + 1):
                response = ollama.chat(model=MODEL, messages=messages, tools=ollama_tools)
                message = response["message"]

                if not message.get("tool_calls"):
                    print("A:", message["content"])
                    return

                messages.append(message)
                for call in message["tool_calls"]:
                    name = call["function"]["name"]
                    args = call["function"].get("arguments", {})
                    print(f"  [step {step}] agent calls (via MCP): {name}({args})")

                    result = await session.call_tool(name, args)
                    result_text = result.content[0].text
                    print(f"  [step {step}] MCP server returned: {result_text}\n")
                    messages.append({"role": "tool", "content": result_text})

            print("Agent stopped: too many steps without a final answer.")


if __name__ == "__main__":
    asyncio.run(main())
