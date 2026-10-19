"""Step 1.5: Build the working dataset from the 42 raw files.

Keeps: EU exports (FLOW 2), normal 8-digit codes, chapters 84-87, Jan 2023 - Jun 2026.
Keeps the full 8-digit code (needed later to match the exact US list of tariffed car parts)
and the number of units (QUANTITY_SUPPL_UNIT; for cars this is the number of vehicles).

Destinations are grouped:
  EU  = another EU member state (intra-EU trade)
  US, GB, CH, CN, JP, KR, CA, MX, TR = kept separately (large markets)
  XX  = all other non-EU countries

Rows that differ only by customs procedure (STAT_PROCEDURE) are added together.
Output: 7_processed_data/exports_ch84_87.parquet, one row per
        exporter x CN8 product x destination group x month.
"""
import os
from pathlib import Path
import time
import polars as pl

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
RAW = BASE / "2_raw_data" / "eurostat_comext_extracted"
OUT = BASE / "7_processed_data"
OUT.mkdir(parents=True, exist_ok=True)

KEEP = ["US", "GB", "CH", "CN", "JP", "KR", "CA", "MX", "TR"]
CHAPTERS = ["84", "85", "86", "87"]

files = [f for f in sorted(RAW.glob("eurostat_exports_imports_*.csv")) if f.stem[-6:] <= "202606"]
print("Files to read:", len(files))

parts = []
start = time.time()
for f in files:
    lf = pl.scan_csv(f, infer_schema=False)
    # The 27 reporters in the file are the EU members; their codes are also used as partner codes
    eu = lf.select(pl.col("REPORTER").unique()).collect()["REPORTER"].to_list()
    df = (lf.select(["REPORTER", "PARTNER", "PRODUCT_NC", "FLOW", "SUPPL_UNIT", "PERIOD",
                     "VALUE_EUR", "QUANTITY_KG", "QUANTITY_SUPPL_UNIT"])
            .filter((pl.col("FLOW") == "2")
                    & pl.col("PRODUCT_NC").str.contains(r"^\d{8}$")
                    & pl.col("PRODUCT_NC").str.slice(0, 2).is_in(CHAPTERS))
            .with_columns(
                pl.col("VALUE_EUR").cast(pl.Float64),
                pl.col("QUANTITY_KG").cast(pl.Float64),
                pl.col("QUANTITY_SUPPL_UNIT").cast(pl.Float64),
                pl.when(pl.col("PARTNER").is_in(eu)).then(pl.lit("EU"))
                  .when(pl.col("PARTNER").is_in(KEEP)).then(pl.col("PARTNER"))
                  .otherwise(pl.lit("XX")).alias("dest"))
            .group_by(["REPORTER", "PRODUCT_NC", "dest", "PERIOD", "SUPPL_UNIT"])
            .agg(pl.col("VALUE_EUR").sum().alias("value_eur"),
                 pl.col("QUANTITY_KG").sum().alias("qty_kg"),
                 pl.col("QUANTITY_SUPPL_UNIT").sum().alias("qty_units"))
            .rename({"REPORTER": "exporter", "PRODUCT_NC": "cn8", "PERIOD": "period",
                     "SUPPL_UNIT": "unit"})
            .collect())
    parts.append(df)
    print(f.stem[-6:], f"{df.height:,} rows", f"({time.time() - start:.0f} s)")

panel = pl.concat(parts).sort(["exporter", "cn8", "dest", "period"])
panel.write_parquet(OUT / "exports_ch84_87.parquet")

print("\nTotal rows:", f"{panel.height:,}")
print("Exporters:", panel["exporter"].n_unique(), " CN8 products:", panel["cn8"].n_unique(),
      " Months:", panel["period"].n_unique())
print("\nExports by destination group, 2024 total, EUR billion:")
print(panel.filter(pl.col("period").str.starts_with("2024"))
           .group_by("dest").agg((pl.col("value_eur").sum() / 1e9).round(1).alias("bn_eur"))
           .sort("bn_eur", descending=True))
print("\nSaved:", OUT / "exports_ch84_87.parquet")
