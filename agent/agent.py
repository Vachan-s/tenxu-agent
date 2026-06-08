import os
import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = Path(__file__).with_name("system_prompt.txt").read_text(encoding="utf-8")

# ── Placeholder tool functions ────────────────────────────────────────────────

def search_products(query: str, filters: dict = {}) -> str:
    return "Search results placeholder - database not connected yet"

def get_product_details(style_ids: list[str]) -> str:
    return "Product details placeholder - database not connected yet"

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

def run_agent(user_query: str, chat_history: list[dict]) -> str:
    """
    Run the product intelligence agent for a single user turn.

    Args:
        user_query:   The latest message from the user.
        chat_history: Prior turns as [{"role": "user"|"assistant", "content": ...}].

    Returns:
        Claude's final text response as a string.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    messages = list(chat_history) + [{"role": "user", "content": user_query}]

    try:
        while True:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                tools=_TOOLS,
                messages=messages,
            )

            # Append the full assistant turn (preserves tool_use blocks)
            messages.append({"role": "assistant", "content": response.content})

            # If Claude is done, return its text
            if response.stop_reason == "end_turn":
                text_blocks = [b.text for b in response.content if b.type == "text"]
                return "\n".join(text_blocks) if text_blocks else ""

            # If Claude wants to call tools, execute them and feed results back
            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = _dispatch_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})
                continue

            # Any other stop reason — return whatever text is available
            text_blocks = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_blocks) if text_blocks else ""

    except anthropic.APIError as e:
        return f"API error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"
