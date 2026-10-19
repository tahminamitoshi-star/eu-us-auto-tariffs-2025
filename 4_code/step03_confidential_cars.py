"""Step 1.3: How much EU export trade to the US in chapters 84-87 is hidden as confidential?

Chapters: 84 machinery and engines, 85 electrical equipment, 86 railway, 87 vehicles and parts.
A confidential row shows only the chapter (e.g. 87XXXXXX); the 8-digit product is hidden.
If much car trade were hidden, our car and parts groups would miss it.
"""
import os
from pathlib import Path
import polars as pl

RAW = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper")) / "2_raw_data" / "eurostat_comext_extracted"
f = RAW / "eurostat_exports_imports_202301.csv"

df = (pl.scan_csv(f, infer_schema=False)
        .select(["REPORTER", "PARTNER", "PRODUCT_NC", "FLOW", "VALUE_EUR"])
        .filter((pl.col("FLOW") == "2") & (pl.col("PARTNER") == "US"))   # exports to the US only
        .filter(pl.col("PRODUCT_NC") != "TOTAL")                          # drop sum rows
        .with_columns(
            pl.col("VALUE_EUR").cast(pl.Float64),
            pl.col("PRODUCT_NC").str.slice(0, 2).alias("chapter"),        # first 2 digits
            pl.col("PRODUCT_NC").str.contains(r"^\d{8}$").alias("normal_code"))
        .filter(pl.col("chapter").is_in(["84", "85", "86", "87"]))
        .collect())

# Value in normal 8-digit rows vs confidential rows, by chapter
t = (df.group_by("chapter")
       .agg((pl.col("VALUE_EUR").filter(pl.col("normal_code")).sum() / 1e6).round(1).alias("normal_m_eur"),
            (pl.col("VALUE_EUR").filter(~pl.col("normal_code")).sum() / 1e6).round(1).alias("confidential_m_eur"))
       .with_columns((100 * pl.col("confidential_m_eur") / (pl.col("normal_m_eur") + pl.col("confidential_m_eur")))
                     .round(1).alias("confidential_pct"))
       .sort("chapter"))
print("EU exports to the US, January 2023, EUR million")
print(t)

# Which countries report confidential trade in chapter 87?
c87 = (df.filter((pl.col("chapter") == "87") & ~pl.col("normal_code"))
         .group_by("REPORTER").agg((pl.col("VALUE_EUR").sum() / 1e6).round(1).alias("confidential_m_eur"))
         .sort("confidential_m_eur", descending=True))
print("\nChapter 87 confidential exports to the US by reporter, EUR million")
print(c87)
