import pandas as pd

# Read everything raw with no header
df_raw = pd.read_excel(
    'Catalog Master - Ten x You.xlsx',
    sheet_name='Catalog - SKU level - D2C',
    header=None
)

# Row 3 (index 2) has the actual column names
# Extract column names from row 3
column_names = df_raw.iloc[2].tolist()

# Data starts from row 4 (index 3)
# Skip rows 0, 1, 2, 3 (parent headers, notes, column names, and use from row 4)
df = df_raw.iloc[3:].copy()

# Assign the column names
df.columns = column_names

# Reset index
df = df.reset_index(drop=True)

# Remove completely empty rows
df = df.dropna(how='all')

print("=== COLUMN NAMES ===")
for i, col in enumerate(df.columns):
    print(f"Col {i}: {col}")

print(f"\nTotal rows: {len(df)}")
print("\nFirst row of actual data:")
print(df.iloc[0].to_string())