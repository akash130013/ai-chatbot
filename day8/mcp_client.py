"""
Day 8: your first MCP CLIENT.

This file does NOT contain any tool logic at all -- notice there's no
get_current_time() or add_numbers() defined here. It only knows how to:
  1. Start mcp_server.py as its own separate process.
  2. Ask it, over the MCP protocol, "what tools do you have?"
  3. Ask it to run one, with some arguments.
  4. Read back the real result.

This is the real point of MCP: the client never needs to know HOW a
tool works internally, only that it exists and what it needs. The
server could be rewritten in a different language tomorrow and this
client's code wouldn't need to change at all.
"""
import asyncio
import os
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_server.py")


async def main() -> None:
    # This tells the client HOW to start the server: run this exact
    # Python interpreter (sys.executable -- our venv's own python, not
    # a bare "python" which may not even exist on PATH), on this script.
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT],
    )

    # stdio_client actually launches mcp_server.py as a subprocess and
    # wires up two pipes (read/write) to talk to it.
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # The MCP "handshake" -- client and server agree on protocol
            # version and capabilities before anything else can happen.
            await session.initialize()

            # STEP 1: discover what tools this server offers. The client
            # did not hard-code these names anywhere -- it asked.
            tools_result = await session.list_tools()
            print("Tools this server offers:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")

            # STEP 2: call one tool with no arguments.
            print("\nCalling get_current_time()...")
            result = await session.call_tool("get_current_time", {})
            print("Result:", result.content[0].text)

            # STEP 3: call another tool, this time with arguments.
            print("\nCalling add_numbers(a=482, b=319)...")
            result = await session.call_tool("add_numbers", {"a": 482, "b": 319})
            print("Result:", result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
