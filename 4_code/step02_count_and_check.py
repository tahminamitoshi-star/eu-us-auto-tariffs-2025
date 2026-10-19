import os
from pathlib import Path
import polars as pl

RAW = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper")) / "2_raw_data" / "eurostat_comext_extracted"
f = RAW / "eurostat_exports_imports_202301.csv"

# Read 5 columns, all as text, then make VALUE_EUR a number
df = (pl.scan_csv(f, infer_schema=False)
        .select(["REPORTER", "PARTNER", "PRODUCT_NC", "FLOW", "VALUE_EUR"])
        .with_columns(pl.col("VALUE_EUR").cast(pl.Float64))
        .collect())

print("Total rows:", df.height)
print("\nRows by FLOW (1 = import, 2 = export):")
print(df.group_by("FLOW").len().sort("FLOW"))
print("\nNumber of reporting countries:", df["REPORTER"].n_unique())

# Product codes that are NOT exactly 8 digits
bad = df.filter(~pl.col("PRODUCT_NC").str.contains(r"^\d{8}$"))
print("\nRows without a normal 8-digit code:", bad.height)
print(bad.group_by("PRODUCT_NC")
         .agg(pl.len().alias("rows"), (pl.col("VALUE_EUR").sum() / 1e9).round(2).alias("bn_eur"))
         .sort("rows", descending=True).head(15))

# Sanity check: EU exports to the US this month
us = df.filter((pl.col("FLOW") == "2") & (pl.col("PARTNER") == "US")
               & pl.col("PRODUCT_NC").str.contains(r"^\d{8}$"))
print("\nEU exports to the US, Jan 2023 (bn EUR):", round(us["VALUE_EUR"].sum() / 1e9, 1))
