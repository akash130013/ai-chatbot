"""
Day 8: an MCP server backed by a (simulated) MongoDB.

This answers the real question: "I don't know what query I'll need
until I'm debugging -- how does the agent write it, and how does that
query actually reach MongoDB?"

The answer: the "query" is NOT a string of code the model writes and
somehow executes. It's a structured ARGUMENT (a filter dict), exactly
like `city` was an argument for get_weather, or `filename` for
read_file. The model fills in that argument based on the SCHEMA we
describe in the tool below; our Python code is what actually calls
pymongo's `.find(filter)` against the real database. MCP's only job is
carrying that argument from the model's decision to this function.

Using mongomock here (an in-memory, pymongo-compatible fake) so this
runs without a real MongoDB server installed -- swap mongomock.MongoClient
for pymongo.MongoClient and a real connection string, and NOTHING else
in this file changes.
"""
import mongomock
from mcp.server.mcpserver import MCPServer

server = MCPServer("mongo-debug-server")

# --- seed some realistic sample data, as if this were a real app's DB ---
client = mongomock.MongoClient()
db = client["shop"]
db.orders.insert_many([
    {"order_id": 1, "customer": "Ravi", "status": "failed", "error_message": "payment timeout", "amount": 1200},
    {"order_id": 2, "customer": "Priya", "status": "completed", "error_message": None, "amount": 450},
    {"order_id": 3, "customer": "Ravi", "status": "failed", "error_message": "invalid address", "amount": 800},
    {"order_id": 4, "customer": "Aisha", "status": "completed", "error_message": None, "amount": 2200},
    {"order_id": 5, "customer": "Ravi", "status": "completed", "error_message": None, "amount": 300},
])


# The docstring here is the SCHEMA the model reads to know what fields
# exist and what values they can hold -- this is what lets it compose a
# filter for a question nobody pre-wrote a function for.
@server.tool()
def query_orders(filter: dict, limit: int = 20) -> str:
    """Query the 'orders' collection in MongoDB and return matching documents.

    Collection schema:
      - order_id (int)
      - customer (string, e.g. "Ravi", "Priya", "Aisha")
      - status (string: "failed" or "completed")
      - error_message (string or null -- the reason a failed order failed)
      - amount (number, in rupees)

    `filter` must be a valid MongoDB filter object, e.g. {"status": "failed"}
    or {"customer": "Ravi", "status": "failed"}. Read-only: this only runs
    `.find()`, it can never insert, update, or delete anything.
    """
    try:
        results = list(db.orders.find(filter, {"_id": 0}).limit(limit))
        return str(results) if results else "No matching orders found."
    except Exception as exc:
        return f"Error running query: {exc}"


if __name__ == "__main__":
    server.run(transport="stdio")
