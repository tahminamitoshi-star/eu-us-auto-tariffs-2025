"""Step 1.8: Is the car price fall real, or a hidden mix change inside each code?

Step 1.7 compared prices within each 8-digit code. But one code can still hold very different cars
(87032319 covers every new petrol car of 1,500-3,000 cc, from a small hatchback to a luxury SUV).
Two checks:
  1. Compare prices within exporter x code (e.g. German 87032319 vs German 87032319).
     If the fall disappears, the mix of exporting countries changed, not prices.
  2. Weight per car (kg per vehicle) within exporter x code.
     If cars sent to the US became lighter, firms likely sent smaller models: a mix change.
     If weight per car is stable but price per car fell, it looks like a real price change.
Base: 2024. Comparison: Apr-Sep 2025. Averages per month.
"""
import os
from pathlib import Path
import polars as pl

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
P = pl.read_parquet(BASE / "7_processed_data" / "exports_ch84_87.parquet")

cars = (P.filter(pl.col("cn8").str.starts_with("8703")
                 | pl.col("cn8").str.slice(0, 6).is_in(["870421", "870431"]))
          .filter(pl.col("dest") != "EU")
          .with_columns(pl.when(pl.col("dest") == "US").then(pl.lit("US"))
                          .otherwise(pl.lit("Other non-EU")).alias("market")))

KEYS = ["market", "exporter", "cn8"]

def period_sum(first, last, s):
    return (cars.filter(pl.col("period").is_between(pl.lit(first), pl.lit(last)))
                .group_by(KEYS)
                .agg(pl.col("value_eur").sum().alias(f"v{s}"),
                     pl.col("qty_units").sum().alias(f"q{s}"),
                     pl.col("qty_kg").sum().alias(f"kg{s}")))

m = (period_sum("202401", "202412", "0")
       .join(period_sum("202504", "202509", "1"), on=KEYS, how="inner")
       .filter((pl.col("q0") > 0) & (pl.col("q1") > 0) & (pl.col("v0") > 0) & (pl.col("v1") > 0)
               & (pl.col("kg0") > 0) & (pl.col("kg1") > 0))
       .with_columns(((pl.col("v1") / pl.col("q1")) / (pl.col("v0") / pl.col("q0"))).log().alias("dlog_price"),
                     ((pl.col("kg1") / pl.col("q1")) / (pl.col("kg0") / pl.col("q0"))).log().alias("dlog_kg_per_car"),
                     ((pl.col("v1") / pl.col("kg1")) / (pl.col("v0") / pl.col("kg0"))).log().alias("dlog_eur_per_kg"),
                     (pl.col("v0") / pl.col("v0").sum().over("market")).alias("w")))

summary = (m.group_by("market")
             .agg(pl.len().alias("exporter_code_pairs"),
                  (100 * ((pl.col("w") * pl.col("dlog_price")).sum().exp() - 1)).alias("price_per_car_chg_pct"),
                  (100 * ((pl.col("w") * pl.col("dlog_kg_per_car")).sum().exp() - 1)).alias("kg_per_car_chg_pct"),
                  (100 * ((pl.col("w") * pl.col("dlog_eur_per_kg")).sum().exp() - 1)).alias("eur_per_kg_chg_pct"))
             .with_columns(pl.col(pl.Float64).round(1))
             .sort("market", descending=True))
pl.Config.set_tbl_width_chars(140)
print("Within exporter x code, 2024 vs Apr-Sep 2025 (weighted by 2024 export value), % change:")
print(summary)

# The biggest code, by exporting country, US market only
top = (m.filter((pl.col("market") == "US") & (pl.col("cn8") == "87032319"))
         .sort("v0", descending=True).head(8)
         .with_columns((pl.col("q0") / 12).round(0).alias("cars_pm_2024"),
                       (pl.col("q1") / 6).round(0).alias("cars_pm_post"),
                       (pl.col("v0") / pl.col("q0")).round(0).alias("eur_car_2024"),
                       (pl.col("v1") / pl.col("q1")).round(0).alias("eur_car_post"),
                       (pl.col("kg0") / pl.col("q0")).round(0).alias("kg_car_2024"),
                       (pl.col("kg1") / pl.col("q1")).round(0).alias("kg_car_post"))
         .select(["exporter", "cars_pm_2024", "cars_pm_post", "eur_car_2024", "eur_car_post",
                  "kg_car_2024", "kg_car_post"]))
print("\nCode 87032319 to the US by exporter (cars per month, EUR per car, kg per car):")
print(top)
