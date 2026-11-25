"""
Day 8: an MCP server backed by a (simulated) MongoDB -- now with MULTIPLE
collections, answering: "how does the agent know which collection to use
when there are 1000 of them?"

The earlier version of this file hard-coded `db.orders.find(...)` -- there
was no way to even NAME a different collection. That's the real bug this
version fixes, with TWO tools instead of one:

  1. list_collections() -- a short, one-line description per collection.
     This is deliberately NOT the full schema -- with 1000 real
     collections, cramming every field of every one into a single tool
     description would be far too much text (and would blow the
     context window, same lesson as Day 5/6). The model only sees
     enough to pick the RIGHT collection first.

  2. query_collection(collection, filter) -- now that a specific
     collection has been chosen, THIS tool returns its full schema (so
     the model finally sees field names) and runs the actual query.

This makes the agent do a real TWO-STEP plan: discover, then query --
exactly the agent-loop idea from Day 7, now driven by what an MCP
server tells it is available, not by what's hard-coded in one function.
"""
import mongomock
from mcp.server.mcpserver import MCPServer

server = MCPServer("mongo-debug-server")

client = mongomock.MongoClient()
db = client["shop"]

db.orders.insert_many([
    {"order_id": 1, "customer": "Ravi", "status": "failed", "error_message": "payment timeout", "amount": 1200},
    {"order_id": 2, "customer": "Priya", "status": "completed", "error_message": None, "amount": 450},
    {"order_id": 3, "customer": "Ravi", "status": "failed", "error_message": "invalid address", "amount": 800},
    {"order_id": 4, "customer": "Aisha", "status": "completed", "error_message": None, "amount": 2200},
    {"order_id": 5, "customer": "Ravi", "status": "completed", "error_message": None, "amount": 300},
])
db.customers.insert_many([
    {"customer": "Ravi", "email": "ravi@example.com", "phone": "9876500001", "city": "Pune"},
    {"customer": "Priya", "email": "priya@example.com", "phone": "9876500002", "city": "Delhi"},
    {"customer": "Aisha", "email": "aisha@example.com", "phone": "9876500003", "city": "Bengaluru"},
])

# In a real 1000-collection system, this dict (and the schemas below)
# would likely come from reading the DB's actual metadata at startup,
# not be hand-typed -- but the SHAPE of the solution is identical.
COLLECTION_SUMMARIES = {
    "orders": "Order records: who placed them, status, and failure reasons.",
    "customers": "Customer contact info: email, phone, city.",
}

COLLECTION_SCHEMAS = {
    "orders": (
        "orders: order_id (int), customer (string), status ('failed' or "
        "'completed'), error_message (string or null), amount (number)"
    ),
    "customers": (
        "customers: customer (string), email (string), phone (string), city (string)"
    ),
}


@server.tool()
def list_collections() -> str:
    """List every collection available in this database, with a short
    one-line description of each. ALWAYS call this first if you are not
    sure which collection contains the information you need."""
    return str(COLLECTION_SUMMARIES)


@server.tool()
def describe_collection(collection: str) -> str:
    """Get the exact field names and types for ONE specific collection.
    ALWAYS call this AFTER list_collections() and BEFORE query_collection()
    -- never guess a field name (e.g. assuming "name" instead of the
    collection's real field "customer"); check it here first."""
    if collection not in COLLECTION_SCHEMAS:
        return f"Error: unknown collection '{collection}'."
    return COLLECTION_SCHEMAS[collection]


@server.tool()
def query_collection(collection: str, filter: dict, limit: int = 20) -> str:
    """Query a specific MongoDB collection and return matching documents.
    Call describe_collection() first to get the real field names --
    guessing them will likely produce a wrong or empty filter."""
    if collection not in COLLECTION_SUMMARIES:
        return (
            f"Error: unknown collection '{collection}'. "
            f"Valid collections: {list(COLLECTION_SUMMARIES)}"
        )
    try:
        results = list(db[collection].find(filter, {"_id": 0}).limit(limit))
        return str(results) if results else "No matching documents found."
    except Exception as exc:
        return f"Error running query: {exc}"


if __name__ == "__main__":
    server.run(transport="stdio")
