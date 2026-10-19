"""Step 1.21 (follow-up): list every HS6 code in the HEAVY_VEHICLES group with its truck-tariff status.

Reads 7_processed_data/hs6_groups_v3_truckflags.csv (made by step16) and the export data.
Prints each code with: truck-list status, 2024 EU exports to the US (EUR m),
and 2024 EU exports to other non-EU markets (EUR m). Changes nothing.
"""
import os
from pathlib import Path
import polars as pl

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
P = pl.read_parquet(BASE / "7_processed_data" / "exports_ch84_87.parquet")
F = pl.read_csv(BASE / "7_processed_data" / "hs6_groups_v3_truckflags.csv", schema_overrides={"hs6": pl.Utf8})

y24 = (P.filter(pl.col("period").str.starts_with("2024") & (pl.col("dest") != "EU"))
         .with_columns(pl.col("cn8").str.slice(0, 6).alias("hs6"),
                       pl.when(pl.col("dest") == "US").then(pl.lit("US")).otherwise(pl.lit("Other")).alias("market"))
         .group_by(["hs6", "market"]).agg((pl.col("value_eur").sum() / 1e6).alias("m_eur"))
         .pivot(on="market", index="hs6", values="m_eur"))

H = (F.filter(pl.col("group_v3") == "6 HEAVY_VEHICLES")
      .join(y24, on="hs6", how="left")
      .with_columns(pl.col("US").fill_null(0).round(1).alias("US_2024_m_eur"),
                    pl.col("Other").fill_null(0).round(1).alias("Other_2024_m_eur"))
      .select(["hs6", "truck_bus_vehicle", "truck_part", "US_2024_m_eur", "Other_2024_m_eur"])
      .sort(["truck_bus_vehicle", "US_2024_m_eur"], descending=[False, True]))

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_width_chars(120)
print(H)
