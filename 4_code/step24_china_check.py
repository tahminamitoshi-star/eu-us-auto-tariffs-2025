"""Step 3.5: Were EU car exports to China falling for reasons of their own?

Step 3.4 showed that the later car estimates are larger without China. The reason would be that EU exports
of cars to China fell, which makes the US look relatively better in a US-vs-other comparison.
This script shows EU exports by destination and period, for cars, core parts and the clean control:
monthly average in EUR bn, and an index (2023 = 100). Vehicles (units) are shown for cars.
Periods: 2023, 2024, Apr-Dec 2025, Jan-Jun 2026 (monthly averages, so periods of different length compare).
Nothing is saved; output only.
"""
import os
from pathlib import Path
import polars as pl

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
P = pl.read_parquet(BASE / "7_processed_data" / "exports_ch84_87.parquet")
G = pl.read_csv(BASE / "7_processed_data" / "hs6_groups_v4.csv", schema_overrides={"hs6": pl.Utf8})

D = (P.with_columns(pl.col("cn8").str.slice(0, 6).alias("hs6"))
      .join(G.select(["hs6", "group_v4"]), on="hs6")
      .filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", "4 CONTROL"]))
      .with_columns(pl.when(pl.col("period") < "202401").then(pl.lit("2023"))
                      .when(pl.col("period") < "202501").then(pl.lit("2024"))
                      .when(pl.col("period") < "202504").then(pl.lit("JanMar25"))
                      .when(pl.col("period") < "202601").then(pl.lit("AprDec25"))
                      .otherwise(pl.lit("JanJun26")).alias("per"))
      .filter(pl.col("per") != "JanMar25"))

nmonths = D.group_by("per").agg(pl.col("period").n_unique().alias("n"))
A = (D.group_by(["group_v4", "dest", "per"])
      .agg((pl.col("value_eur").sum() / 1e9).alias("bn"), pl.col("qty_units").sum().alias("units"))
      .join(nmonths, on="per")
      .with_columns((pl.col("bn") / pl.col("n")).alias("bn_m"), (pl.col("units") / pl.col("n")).alias("units_m")))

order = ["2023", "2024", "AprDec25", "JanJun26"]
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(150)
for g in ["1 CARS", "2 PARTS_CORE", "4 CONTROL"]:
    a = A.filter(pl.col("group_v4") == g)
    base = a.filter(pl.col("per") == "2023").select(["dest", pl.col("bn_m").alias("b23")])
    t = (a.join(base, on="dest").with_columns((100 * pl.col("bn_m") / pl.col("b23")).round(0).alias("idx"))
          .pivot(on="per", index="dest", values="idx").select(["dest"] + order))
    lv = (a.pivot(on="per", index="dest", values="bn_m").select(["dest", pl.col("2024").round(2).alias("EURbn_per_month_2024")]))
    print(f"\n{g}: index of monthly exports (2023 = 100), and 2024 level")
    print(t.join(lv, on="dest").sort("EURbn_per_month_2024", descending=True))
    if g == "1 CARS":
        u = (a.pivot(on="per", index="dest", values="units_m").select(["dest"] + order)
              .with_columns(pl.col(order).round(0)).sort("2024", descending=True))
        print("\nCars: vehicles per month")
        print(u)
