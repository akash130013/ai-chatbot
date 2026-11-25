"""
Day 8: an AI agent whose tools come from an MCP server (not hard-coded
in this file at all), backed by a MongoDB-style database.

This proves the exact mechanism asked about: "how does the agent pass
a query to MongoDB -- through MCP?" Yes. Here is the real path a
question takes, step by step:

  1. You ask a question in plain English.
  2. Ollama reads the tool's SCHEMA (fetched from the MCP server below,
     not written by hand in this file) and decides it needs
     `query_orders`, composing a `filter` dict itself from your words
     and the schema in the tool's description.
  3. That decision -- just a tool name + a dict of arguments, no code,
     no actual MongoDB syntax -- is sent over MCP to mongo_mcp_server.py.
  4. THAT server (a separate process) is the only place that touches
     the database: it runs `db.orders.find(filter)` for real.
  5. The real matching documents come back, through MCP, to this agent.
  6. Ollama reads those real rows and writes your final answer.

The model never "connects to MongoDB." It only ever asks a question in
a structured shape; the MCP server is what actually talks to the
database.
"""
import asyncio
import os
import sys

import ollama
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

MODEL = os.getenv("MODEL", "qwen3:1.7b")
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mongo_mcp_server.py")

SYSTEM_PROMPT = (
    "You are a debugging assistant with access to a tool that queries a "
    "MongoDB 'orders' collection. Use it to answer questions about orders. "
    "Compose the MongoDB filter yourself based on the schema in the tool's "
    "description."
)


def mcp_tool_to_ollama_schema(tool) -> dict:
    """Convert an MCP tool's schema into the shape Ollama's tools= param
    expects. Both use JSON Schema underneath, so this is mostly just
    relabeling -- proof that MCP and Ollama tool-calling are the same
    underlying idea, just two different pieces of plumbing around it."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


async def main() -> None:
    question = "Which customer has the most failed orders, and what were the error messages?"
    print(f"Q: {question}\n")

    server_params = StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover tools from the MCP server -- NOT hard-coded here.
            mcp_tools = (await session.list_tools()).tools
            ollama_tools = [mcp_tool_to_ollama_schema(t) for t in mcp_tools]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]

            response = ollama.chat(model=MODEL, messages=messages, tools=ollama_tools)
            message = response["message"]

            if message.get("tool_calls"):
                messages.append(message)
                for call in message["tool_calls"]:
                    name = call["function"]["name"]
                    args = call["function"].get("arguments", {})
                    print(f"  agent calls (via MCP): {name}({args})")

                    # THE key line: the call actually goes over MCP to the
                    # separate server process, which is the only thing
                    # that touches the database.
                    result = await session.call_tool(name, args)
                    result_text = result.content[0].text
                    print(f"  MCP server returned: {result_text}\n")
                    messages.append({"role": "tool", "content": result_text})

                final = ollama.chat(model=MODEL, messages=messages)
                print("A:", final["message"]["content"])
            else:
                print("A:", message["content"])


if __name__ == "__main__":
    asyncio.run(main())
