"""
Day 8: your first MCP SERVER.

Every tool you've built so far (get_weather, get_current_time, read_file)
lived in the SAME Python file as your chatbot. That's fine for one
project, but it doesn't scale: if you wanted five different AI apps to
all use "get_current_time", you'd copy-paste the function into every one
of them, and fix bugs in five places.

MCP (Model Context Protocol) fixes this by putting tools behind a
STANDARD SERVER, running as its own separate process. ANY MCP-compatible
client (Claude Desktop, your own Python script, a teammate's app) can
connect to this one server and use its tools -- no copy-pasting.

MERN comparison: this is exactly the difference between writing a
database query directly inside one Express route, versus standing up a
separate microservice that any app can call over HTTP. The tool's LOGIC
doesn't change; WHO can reach it, and how, does.

This file is the SERVER half: it defines tools and exposes them. It does
nothing by itself -- a CLIENT (mcp_client.py) has to connect to it.
"""
from datetime import datetime

from mcp.server.mcpserver import MCPServer

# "my-first-server" is this server's name, shown to any client that connects.
server = MCPServer("my-first-server")


# The @server.tool() decorator is doing the SAME job as the TOOLS schema
# list in day4/tool_manager.py -- describing a function so something
# else can discover and call it. The difference: here, that "something
# else" talks to this function over the MCP protocol, potentially from
# a completely different program entirely.
@server.tool()
def get_current_time() -> str:
    """Get the current real-world date and time."""
    return datetime.now().strftime("%I:%M %p on %A, %B %d, %Y")


@server.tool()
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together and return the exact sum."""
    return a + b


if __name__ == "__main__":
    # "stdio" means this server talks to its client over standard
    # input/output -- the simplest MCP transport, good for local tools
    # run as a subprocess (as mcp_client.py will do).
    server.run(transport="stdio")
