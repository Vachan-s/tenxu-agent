"""
sync_script.py

Reads product data from Excel, generates sentence-transformer embeddings,
and upserts to Supabase (PostgreSQL + pgvector).

Run from the project root:
    python sync/sync_script.py
"""

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

# Allow imports from project root so config/ is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.sheet_config import SHEET_CONFIG

# ---------------------------------------------------------------------------
# Embedding label map: renamed column → human-readable label for embed text
# Covers all three sheets; missing keys fall back to title-cased column name.
# ---------------------------------------------------------------------------
EMBEDDING_LABEL_MAP: dict[str, str] = {
    "product_name":                "Product",
    "website_name":                "Display Name",
    "tagline":                     "Tagline",
    "category":                    "Category",
    "sub_category":                "Sub-Category",
    "apparel_type":                "Apparel Type",
    "gender":                      "Gender",
    "activity":                    "Activity",
    "best_suited_for":             "Best Suited For",
    "collection":                  "Collection",
    "description":                 "Description",
    "expert_review":               "Expert Review",
    "fit":                         "Fit",
    # Apparel
    "fabric":                      "Fabric",
    "fabric_description":          "Fabric Description",
    "fabric_feel":                 "Fabric Feel",
    "breathability":               "Breathability",
    "stretch":                     "Stretch",
    "pattern_texture":             "Pattern",
    "pattern_texture_description": "Pattern Description",
    "stitching_description":       "Stitching",
    "pockets":                     "Pockets",
    "waistband":                   "Waistband",
    "neck_type":                   "Neck Type",
    "sleeve_type":                 "Sleeve Type",
    # Footwear
    "upper_material":              "Upper Material",
    "upper_features":              "Upper Features",
    "tongue_features":             "Tongue Features",
    "heel_features":               "Heel Features",
    "midsole_material":            "Midsole Material",
    "midsole_features":            "Midsole Features",
    "outsole_material":            "Outsole Material",
    "outsole_features":            "Outsole Features",
    "insole_features":             "Insole Features",
    "cushioning":                  "Cushioning",
    "stability":                   "Stability",
    "waterproof":                  "Waterproof",
    "lacing_closure":              "Lacing/Closure",
    # Accessories
    "material":                    "Material",
    "functional_features":         "Features",
    "ventilation":                 "Ventilation",
    "structure_type":              "Structure",
    "closure_type":                "Closure Type",
}

# Columns excluded from embedding text (IDs, numeric, the vector itself)
_NON_TEXT_COLS = {"style_id", "mrp", "selling_price", "embedding"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_null(value) -> bool:
    """Return True for None, NaN, or NaT without raising on non-scalars."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def build_embedding_text(row: dict) -> str:
    """
    Concatenate all non-null text fields into a single embedding string.
    Format: "Label: value | Label: value | ..."
    """
    parts = []
    for col, value in row.items():
        if col in _NON_TEXT_COLS:
            continue
        if _is_null(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        label = EMBEDDING_LABEL_MAP.get(col, col.replace("_", " ").title())
        parts.append(f"{label}: {text}")
    return " | ".join(parts)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps:
      - strip whitespace from text columns
      - numeric coercion for mrp / selling_price
      - empty strings → NaN
      - drop fully-empty rows and rows with null style_id
    """
    # Strip whitespace from object columns without converting NaN to the string "nan"
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Convert price columns to numeric; non-numeric values become NaN
    for price_col in ("mrp", "selling_price"):
        if price_col in df.columns:
            df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # Replace empty strings with NaN so they serialize to NULL in Supabase
    df = df.replace("", np.nan)

    # Drop rows that are entirely empty, then rows with no style_id
    df = df.dropna(how="all")
    df = df.dropna(subset=["style_id"])

    return df.reset_index(drop=True)


def row_to_upsert_dict(row: dict, embedding: list) -> dict:
    """
    Convert a pandas row dict + embedding vector to a JSON-safe dict for
    Supabase upsert.  numpy scalars and NaN values are normalised.
    """
    result: dict = {}
    for key, value in row.items():
        if _is_null(value):
            result[key] = None
        elif isinstance(value, np.integer):
            result[key] = int(value)
        elif isinstance(value, np.floating):
            result[key] = float(value)
        elif isinstance(value, np.bool_):
            result[key] = bool(value)
        else:
            result[key] = value
    result["embedding"] = embedding  # list[float], accepted by supabase-py
    return result


# ---------------------------------------------------------------------------
# Per-sheet processing
# ---------------------------------------------------------------------------

def process_sheet(
    sheet_name: str,
    config: dict,
    excel_path: str,
    model: SentenceTransformer,
    supabase: Client,
) -> dict:
    """
    Full pipeline for one sheet: read → clean → embed → upsert.
    Returns a summary dict with row counts and success status.
    """
    table_name = sheet_name.lower()

    # 1. Read sheet
    print(f"\nReading {sheet_name}...", end=" ", flush=True)
    try:
        df = pd.read_excel(
            excel_path,
            sheet_name=config["sheet_name"],
            header=config["header_row"],
        )
    except Exception as exc:
        print(f"FAILED")
        print(f"  Could not read sheet '{sheet_name}': {exc}")
        return {"sheet": sheet_name, "rows": 0, "loaded": 0, "failed": 0,
                "success": False, "error": str(exc)}

    # Strip whitespace from column headers before renaming
    df.columns = df.columns.str.strip()

    # 2. Rename columns per mapping; drop columns not in the mapping
    df = df.rename(columns=config["column_map"])
    mapped_cols = [c for c in config["column_map"].values() if c in df.columns]
    df = df[mapped_cols]

    # 3. Clean data
    df = clean_dataframe(df)
    row_count = len(df)
    print(f"{row_count} rows")

    if row_count == 0:
        print(f"  No valid rows found — skipping.")
        return {"sheet": sheet_name, "rows": 0, "loaded": 0, "failed": 0,
                "success": True, "error": None}

    # 4. Build embedding texts from all non-null text fields per row
    print("Generating embeddings...", flush=True)
    rows_as_dicts = df.to_dict(orient="records")
    embedding_texts = [build_embedding_text(r) for r in rows_as_dicts]
    # encode() returns a numpy ndarray; .tolist() converts to plain Python lists
    embeddings: list[list[float]] = model.encode(
        embedding_texts, show_progress_bar=False
    ).tolist()

    # 5. Upsert row-by-row so a single bad row doesn't abort the sheet
    print(f"Upserting to Supabase...", end=" ", flush=True)
    success_count = 0
    fail_count = 0

    for i, (row_dict, embedding) in enumerate(zip(rows_as_dicts, embeddings)):
        try:
            upsert_data = row_to_upsert_dict(row_dict, embedding)
            supabase.table(table_name).upsert(
                upsert_data, on_conflict="style_id"
            ).execute()
            success_count += 1
        except Exception as exc:
            fail_count += 1
            style_id = row_dict.get("style_id", f"index {i}")
            print(f"\n  Error on style_id '{style_id}': {exc}")

    status = "done" if fail_count == 0 else f"{fail_count} error(s)"
    print(f"{status}. {success_count} rows loaded.")

    return {
        "sheet": sheet_name,
        "rows": row_count,
        "loaded": success_count,
        "failed": fail_count,
        "success": fail_count == 0,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Load credentials from .env in the project root, resolved relative to
    # this file so the path is correct regardless of where the script is run from
    load_dotenv(Path(__file__).parent.parent / ".env")
    supabase_url = os.getenv("SUPABASE_URL")
    # Use the service role key so the sync script can bypass RLS and write data
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

    if not supabase_url or not supabase_key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
        sys.exit(1)

    # Resolve Excel path relative to project root (one level above sync/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # All sheets share the same file; read it from the first config entry
    first_config = next(iter(SHEET_CONFIG.values()))
    excel_path = os.path.join(project_root, first_config["file"])

    if not os.path.exists(excel_path):
        print(f"ERROR: Excel file not found:\n  {excel_path}")
        sys.exit(1)

    # Load embedding model once — reused across all sheets
    print("Loading embedding model (multi-qa-MiniLM-L6-cos-v1)...")
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    print("Connecting to Supabase...")
    client: Client = create_client(supabase_url, supabase_key)

    # Process each sheet, catching unexpected top-level errors per sheet
    results: list[dict] = []
    for sheet_name, config in SHEET_CONFIG.items():
        try:
            result = process_sheet(sheet_name, config, excel_path, model, client)
        except Exception as exc:
            print(f"\nUnexpected error processing '{sheet_name}':")
            traceback.print_exc()
            result = {"sheet": sheet_name, "rows": 0, "loaded": 0, "failed": 0,
                      "success": False, "error": str(exc)}
        results.append(result)

    # Print summary
    print("\n--- Sync Summary ---")
    all_ok = True
    for r in results:
        if r["success"]:
            extra = f", {r['failed']} failed" if r["failed"] else ""
            line = f"{r['loaded']} rows loaded{extra}"
        else:
            line = f"FAILED — {r.get('error', 'unknown error')}"
            all_ok = False
        print(f"  {r['sheet']:15s}: {line}")

    print("\nSync complete." if all_ok else "\nSync finished with errors.")


if __name__ == "__main__":
    main()
