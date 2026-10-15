"""
Day 5 tools: Document AI (reading files).

Same problem shape as Day 4's weather tool: the model cannot know what's
inside YOUR files -- they were never in its training data (see Day 1:
the model only knows patterns from training, nothing you wrote after
that, and definitely not a private file sitting on your laptop).

The fix is the SAME mechanism as Day 4, just a different function:
  1. Write an ordinary Python function that reads a file.
  2. Describe it to the model as a tool.
  3. The model asks us to run it ("please read company_policy.txt").
  4. OUR code reads the real file and sends the real text back.
  5. The model answers using that real content, not a guess.

This file follows the exact same shape as ../day4/tool_manager.py on
purpose -- same TOOLS list, same AVAILABLE_FUNCTIONS dict, same
run_tool_call() -- so you can compare the two side by side.
"""
import os

# All readable files live in this folder, and ONLY this folder. This is
# a basic but important safety rule: never let the model (or a user
# message) pass an arbitrary path like "../../.env" and have your code
# read it. We only ever look inside SAMPLE_DOCS_DIR.
SAMPLE_DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")


def list_available_documents() -> list[str]:
    """Plain Python, not a tool itself -- used to show the user (and to
    build the system prompt) which filenames actually exist."""
    return sorted(os.listdir(SAMPLE_DOCS_DIR))


def read_file(filename: str) -> str:
    """THE TOOL. Reads one file's full text content and returns it.

    Notice what this function does NOT do: it doesn't understand the
    file, summarize it, or answer anything. It just reads bytes off
    disk, exactly like `open(...).read()` in any ordinary script. All
    of the "understanding" happens later, when the model reads this
    returned text for itself.
    """
    # Safety: strip any directory parts so "../../.env" can't escape
    # SAMPLE_DOCS_DIR. Only a bare filename inside that folder is ever read.
    safe_name = os.path.basename(filename)
    path = os.path.join(SAMPLE_DOCS_DIR, safe_name)

    if not os.path.isfile(path):
        available = ", ".join(list_available_documents())
        return f"File '{filename}' not found. Available files: {available}"

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# --- Schema: tells the model the tool exists and what argument it needs.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the full text content of a document by filename, "
                "so you can answer questions about what it contains."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The exact filename to read, e.g. 'company_policy.txt'",
                    },
                },
                "required": ["filename"],
            },
        },
    },
]

# --- Lookup table: tool name (string) -> real Python function.
AVAILABLE_FUNCTIONS = {
    "read_file": read_file,
}


def run_tool_call(call: dict) -> str:
    """Identical pattern to day4/tool_manager.py's run_tool_call: look
    up the real function by name, run it with the model's arguments,
    and return the result as text."""
    func_name = call["function"]["name"]
    arguments = call["function"].get("arguments", {})
    func = AVAILABLE_FUNCTIONS.get(func_name)

    if func is None:
        return f"Error: unknown tool '{func_name}'"

    try:
        return str(func(**arguments))
    except Exception as exc:
        return f"Error running '{func_name}': {exc}"
