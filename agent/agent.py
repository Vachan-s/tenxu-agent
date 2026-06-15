import os
import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from agent.search import hybrid_search, get_multiple_products

load_dotenv(Path(__file__).parent.parent / ".env")

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = Path(__file__).with_name("system_prompt.txt").read_text(encoding="utf-8")

# ── Tool implementations ──────────────────────────────────────────────────────

_SKIP_FIELDS = {"embedding"}  # fields to omit from formatted output


def _format_product(product: dict, table: str = "", similarity: float | None = None) -> str:
    """Format a single product dict as a human-readable text block."""
    name = product.get("product_name") or product.get("style_id", "Unknown")
    header_parts = [name]
    if table:
        header_parts.append(f"[{table.title()}]")
    if similarity is not None:
        header_parts.append(f"(match: {similarity:.0%})")
    lines = [" ".join(header_parts)]

    for key, value in product.items():
        if key in _SKIP_FIELDS or value is None:
            continue
        label = key.replace("_", " ").title()
        lines.append(f"  {label}: {value}")

    return "\n".join(lines)


def search_products(query: str, filters: dict = {}) -> str:
    """
    Run hybrid_search then fetch and format full details for the top 5 results.
    """
    try:
        results = hybrid_search(query=query, filters=filters)
    except Exception as exc:
        return f"Search error: {exc}"

    if not results:
        return "No products found matching your search criteria."

    top5 = results[:5]

    # Build a quick lookup for similarity + table, keyed by style_id
    meta = {
        r["style_id"]: {"table": r["table"], "similarity": r.get("similarity")}
        for r in top5
    }

    try:
        products = get_multiple_products(
            [{"style_id": r["style_id"], "table": r["table"]} for r in top5]
        )
    except Exception as exc:
        return f"Error fetching product details: {exc}"

    if not products:
        return "No products found matching your search criteria."

    blocks = []
    for product in products:
        m = meta.get(product.get("style_id"), {})
        blocks.append(_format_product(product, table=m.get("table", ""), similarity=m.get("similarity")))

    return "\n\n---\n\n".join(blocks)


def get_product_details(style_ids: list) -> str:
    """
    Fetch and format complete details for a list of style IDs.

    Accepts either plain style_id strings (the tool schema sends these) or
    dicts with {style_id, table}.  For plain strings the product is looked
    up across all three tables and the first match is used.
    """
    if not style_ids:
        return "No products found."

    items = []
    for item in style_ids:
        if isinstance(item, dict):
            items.append(item)
        else:
            # Plain string — try every table; get_multiple_products skips Nones
            for table in ("apparel", "footwear", "accessories"):
                items.append({"style_id": str(item), "table": table})

    try:
        products = get_multiple_products(items)
    except Exception as exc:
        return f"Error fetching product details: {exc}"

    # Deduplicate: a string-based lookup may return the same product from
    # multiple table attempts; keep only the first occurrence per style_id.
    seen: set[str] = set()
    unique: list[dict] = []
    for product in products:
        sid = product.get("style_id", "")
        if sid not in seen:
            seen.add(sid)
            unique.append(product)

    if not unique:
        return "No products found."

    blocks = [_format_product(p) for p in unique]
    return "\n\n---\n\n".join(blocks)

# ── Claude tools schema ───────────────────────────────────────────────────────

_TOOLS = [
    {
        "name": "search_products",
        "description": (
            "Search for products using semantic search and structured filters. "
            "Use this for any product query."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "filters": {
                    "type": "object",
                    "description": "Optional structured filters",
                    "properties": {
                        "color":    {"type": "string"},
                        "size":     {"type": "string"},
                        "mrp_max":  {"type": "number"},
                        "mrp_min":  {"type": "number"},
                        "gender":   {"type": "string"},
                        "activity": {"type": "string"},
                    },
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_product_details",
        "description": (
            "Get full details for specific products by their style IDs. "
            "Use this after search to get complete product info."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "style_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of style IDs to retrieve details for",
                },
            },
            "required": ["style_ids"],
        },
    },
]

# ── Tool dispatcher ───────────────────────────────────────────────────────────

def _dispatch_tool(name: str, tool_input: dict) -> str:
    if name == "search_products":
        return search_products(
            query=tool_input["query"],
            filters=tool_input.get("filters", {}),
        )
    if name == "get_product_details":
        return get_product_details(style_ids=tool_input["style_ids"])
    return f"Unknown tool: {name}"

# ── Agent entry point ─────────────────────────────────────────────────────────

def stream_agent(user_query: str, chat_history: list[dict]):
    """
    Generator: runs the tool-use loop, then streams Claude's final text response.

    Tool-use rounds execute synchronously without yielding (spinner phase in UI).
    Once Claude is ready to reply, text chunks are yielded as they arrive.
    The very last yielded value is the complete assembled response string, so
    callers that don't use st.write_stream() can still capture the full text.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    messages = list(chat_history) + [{"role": "user", "content": user_query}]

    try:
        while True:
            # Use streaming for every API call; text chunks are only yielded
            # to the caller when stop_reason is end_turn (final response).
            with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                tools=_TOOLS,
                messages=messages,
            ) as stream:
                # Collect text as it arrives; tool-use rounds produce no text
                text_chunks: list[str] = []
                for chunk in stream.text_stream:
                    text_chunks.append(chunk)

                final_message = stream.get_final_message()

            # Append the full assistant turn (preserves tool_use blocks)
            messages.append({"role": "assistant", "content": final_message.content})

            if final_message.stop_reason == "end_turn":
                # Final response: yield each chunk, then yield the complete text
                for chunk in text_chunks:
                    yield chunk
                yield "".join(text_chunks)  # sentinel — full response for history
                return

            if final_message.stop_reason == "tool_use":
                tool_results = []
                for block in final_message.content:
                    if block.type == "tool_use":
                        result = _dispatch_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})
                continue

            # Any other stop reason — yield whatever text we have
            full_text = "".join(text_chunks)
            yield full_text
            yield full_text
            return

    except anthropic.APIError as e:
        error_msg = f"API error: {e}"
        yield error_msg
        yield error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        yield error_msg
        yield error_msg
