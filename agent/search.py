from sentence_transformers import SentenceTransformer
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
import anthropic
import json
import os

# Load .env once at import time so os.getenv() works throughout this module
load_dotenv(Path(__file__).parent.parent / ".env")


@lru_cache(maxsize=1)
def load_embedding_model():
    """Load embedding model once and cache it."""
    return SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

def embed_query(text: str) -> list:
    """Convert a text query into a 384-dimension embedding vector."""
    model = load_embedding_model()
    return model.encode(text).tolist()


# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """
    Return a Supabase client, creating it only once per process via lru_cache.
    Credentials are read from .env which is loaded at module level.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_KEY must be set in .env"
        )
    return create_client(url, key)


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------

# Maps table name → name of the matching RPC function in Supabase
_RPC_FUNCTIONS = {
    "apparel":     "match_apparel",
    "footwear":    "match_footwear",
    "accessories": "match_accessories",
}


def semantic_search(
    query: str,
    tables: list = ["apparel", "footwear", "accessories"],
    match_count: int = 20,
    threshold: float = 0.3,
) -> list:
    """
    Embed the query and run vector similarity search across the given tables.

    Calls each table's RPC function (match_apparel / match_footwear /
    match_accessories) and combines the results.

    Returns a list of dicts sorted by similarity descending:
        [{style_id, product_name, similarity, table}, ...]
    Only results at or above threshold are included.
    Errors on individual tables are logged and skipped.
    """
    query_embedding = embed_query(query)
    client = get_supabase_client()
    results = []

    for table in tables:
        rpc_name = _RPC_FUNCTIONS.get(table)
        if not rpc_name:
            print(f"semantic_search: unknown table '{table}', skipping.")
            continue
        try:
            response = client.rpc(
                rpc_name,
                {
                    "query_embedding": query_embedding,
                    "match_threshold": threshold,
                    "match_count": match_count,
                },
            ).execute()
            for row in (response.data or []):
                results.append({
                    "style_id":     row["style_id"],
                    "product_name": row["product_name"],
                    "similarity":   row["similarity"],
                    "table":        table,
                })
        except Exception as exc:
            print(f"semantic_search: error querying '{table}': {exc}")

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results


def structured_filter(
    filters: dict,
    tables: list = ["apparel", "footwear", "accessories"],
) -> list:
    """
    Query tables with SQL-style filters (no vector search).

    Supported filter keys:
        gender, activity, category, breathability,
        available_colours, available_sizes  → exact match (eq)
        mrp_max                             → mrp <= value (lte)
        mrp_min                             → mrp >= value (gte)

    Returns [{style_id, product_name, table}, ...].
    Errors on individual tables are logged and skipped.
    """
    client = get_supabase_client()
    results = []

    # Restrict to a single table if the caller specified one
    table_filter = filters.get("table")
    if table_filter and table_filter in ("apparel", "footwear", "accessories"):
        tables = [table_filter]

    # Filters that map directly to column equality checks
    eq_filters = {
        "gender":            "gender",
        "activity":          "activity",
        "category":          "category",
        "breathability":     "breathability",
        "available_colours": "available_colours",
        "available_sizes":   "available_sizes",
    }

    for table in tables:
        try:
            query = client.table(table).select("style_id, product_name")

            for filter_key, column in eq_filters.items():
                if filter_key in filters and filters[filter_key] is not None:
                    query = query.eq(column, filters[filter_key])

            if "mrp_max" in filters and filters["mrp_max"] is not None:
                query = query.lte("mrp", filters["mrp_max"])

            if "mrp_min" in filters and filters["mrp_min"] is not None:
                query = query.gte("mrp", filters["mrp_min"])

            response = query.execute()
            for row in (response.data or []):
                results.append({
                    "style_id":     row["style_id"],
                    "product_name": row["product_name"],
                    "table":        table,
                })
        except Exception as exc:
            print(f"structured_filter: error querying '{table}': {exc}")

    return results


# Fields included in the product summary sent to the reranker
_RERANK_FIELDS = (
    "style_id", "product_name", "category", "gender", "activity",
    "description", "fabric", "breathability", "fit", "material",
    "functional_features", "upper_features", "cushioning",
)


def rerank_results(query: str, results: list) -> list:
    """
    Re-order search results by relevance using a fast claude-haiku-3-5 call.

    Fetches full product details for all results, asks Claude to rank them by
    query relevance, and returns the results in the new order.
    Falls back to the original order on any error or if ≤3 results.
    """
    if len(results) <= 3:
        return results

    # Fetch full product details to give the ranker richer context
    items = [{"style_id": r["style_id"], "table": r["table"]} for r in results]
    products = get_multiple_products(items)

    if not products:
        return results

    # Build a compact summary per product (key fields only)
    product_blocks = []
    for p in products:
        lines = [f"Style ID: {p.get('style_id', '')}"]
        for field in _RERANK_FIELDS:
            if field == "style_id":
                continue
            val = p.get(field)
            if val is not None:
                label = field.replace("_", " ").title()
                lines.append(f"{label}: {val}")
        product_blocks.append("\n".join(lines))

    formatted_products = "\n\n".join(product_blocks)

    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-haiku-3-5-20241022",
            max_tokens=256,
            system=(
                "You are a product relevance ranker for an athleisurewear brand. "
                "Given a search query and list of products, return the style_ids in "
                "order of relevance to the query. Return only a JSON array of "
                'style_ids, most relevant first. Example: ["XM1000", "XB2001"]'
            ),
            messages=[{
                "role": "user",
                "content": f"Query: {query}\n\nProducts:\n{formatted_products}",
            }],
        )
        ranked_ids: list[str] = json.loads(response.content[0].text.strip())

        # Reorder original results to match Claude's ranking;
        # any style_id not mentioned by Claude is appended at the end.
        id_to_result = {r["style_id"]: r for r in results}
        reranked = [id_to_result[sid] for sid in ranked_ids if sid in id_to_result]
        mentioned = set(ranked_ids)
        reranked += [r for r in results if r["style_id"] not in mentioned]
        return reranked

    except Exception as exc:
        print(f"rerank_results: failed, returning original order: {exc}")
        return results


def hybrid_search(
    query: str = "",
    filters: dict = {},
    tables: list = ["apparel", "footwear", "accessories"],
    match_count: int = 20,
) -> list:
    """
    Combine semantic and structured search:
      - query only   → semantic_search results
      - filters only → structured_filter results
      - both         → intersection by style_id (semantic scores preserved)
      - neither      → empty list

    Every result dict includes a 'table' field.
    Returns [{style_id, product_name, table}, ...].
    """
    has_query = bool(query and query.strip())
    has_filters = bool(filters)

    if not has_query and not has_filters:
        return []

    if has_query and not has_filters:
        results = semantic_search(query, tables=tables, match_count=match_count)
        return rerank_results(query, results)

    if has_filters and not has_query:
        return structured_filter(filters, tables=tables)

    # Both: run each search and return rows present in both result sets
    semantic_results = semantic_search(query, tables=tables, match_count=match_count)
    filter_results = structured_filter(filters, tables=tables)

    # Build a set of (style_id, table) pairs from the filter results
    filter_ids = {(r["style_id"], r["table"]) for r in filter_results}

    # Keep semantic results that also appear in the filter set
    intersection = [
        r for r in semantic_results
        if (r["style_id"], r["table"]) in filter_ids
    ]
    return rerank_results(query, intersection)


# ---------------------------------------------------------------------------
# Product detail fetchers
# ---------------------------------------------------------------------------

def get_product_details(style_id: str, table: str) -> dict | None:
    """
    Fetch the complete row for a single product from the given table.
    Returns a dict of all product fields, or None if not found or on error.
    """
    client = get_supabase_client()
    try:
        response = (
            client.table(table)
            .select("*")
            .eq("style_id", style_id)
            .limit(1)
            .execute()
        )
        if response.data:
            return response.data[0]
        return None
    except Exception as exc:
        print(f"get_product_details: error fetching '{style_id}' from '{table}': {exc}")
        return None


def get_all_products(table: str) -> list:
    """Fetch all products from a specific table."""
    client = get_supabase_client()
    try:
        response = client.table(table).select(
            "style_id, product_name, category"
        ).execute()
        return [
            {**row, "table": table}
            for row in (response.data or [])
        ]
    except Exception as exc:
        print(f"get_all_products: error fetching from '{table}': {exc}")
        return []


def get_multiple_products(items: list) -> list:
    """
    Fetch complete product details for a list of {style_id, table} dicts.
    Skips items where get_product_details returns None.
    Returns a list of complete product dicts.
    """
    results = []
    for item in items:
        details = get_product_details(item["style_id"], item["table"])
        if details is not None:
            results.append(details)
    return results