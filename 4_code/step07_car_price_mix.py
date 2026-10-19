"""Step 1.7: Did EU firms cut car prices, or did they ship a cheaper mix of cars?

The average price per car can fall for two reasons:
  (a) price cut: the same model type sells for fewer euros
  (b) mix shift: firms send more cheap types and fewer expensive types
To separate them we compare prices WITHIN each 8-digit car code (same type of car).

Within-code price change = average of each code's price change, weighted by the code's share
of export value in 2024. The rest of the average price change is the mix effect.

Values are at the EU border (before the US tariff is paid), in euros.
Periods: 2024 (base) vs Apr-Sep 2025 and vs Jan-Jun 2026; averages per month.
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

def period_avg(df, first, last, name):
    """Monthly average value and vehicles per market x code between two months."""
    n = df.filter(pl.col("period").is_between(pl.lit(first), pl.lit(last)))["period"].n_unique()
    return (df.filter(pl.col("period").is_between(pl.lit(first), pl.lit(last)))
              .group_by(["market", "cn8"])
              .agg((pl.col("value_eur").sum() / n).alias(f"v_{name}"),
                   (pl.col("qty_units").sum() / n).alias(f"q_{name}")))

base = period_avg(cars, "202401", "202412", "0")
pl.Config.set_tbl_width_chars(140)

for first, last, label in [("202504", "202509", "Apr-Sep 2025"), ("202601", "202606", "Jan-Jun 2026")]:
    post = period_avg(cars, first, last, "1")
    m = base.join(post, on=["market", "cn8"], how="full", coalesce=True).fill_null(0)

    # Average price per car over all codes
    tot = (m.group_by("market")
             .agg(pl.col("v_0").sum(), pl.col("q_0").sum(), pl.col("v_1").sum(), pl.col("q_1").sum())
             .with_columns((100 * ((pl.col("v_1") / pl.col("q_1")) / (pl.col("v_0") / pl.col("q_0")) - 1)).alias("avg_price_chg_pct"),
                           (100 * (pl.col("q_1") / pl.col("q_0") - 1)).alias("vehicles_chg_pct"),
                           (100 * (pl.col("v_1") / pl.col("v_0") - 1)).alias("value_chg_pct")))

    # Within-code price change: codes sold in both periods, weighted by 2024 value share
    both = m.filter((pl.col("q_0") > 0) & (pl.col("q_1") > 0) & (pl.col("v_0") > 0) & (pl.col("v_1") > 0))
    within = (both.with_columns(((pl.col("v_1") / pl.col("q_1")) / (pl.col("v_0") / pl.col("q_0"))).log().alias("dlogp"),
                                (pl.col("v_0") / pl.col("v_0").sum().over("market")).alias("w"))
                  .group_by("market")
                  .agg((100 * ((pl.col("w") * pl.col("dlogp")).sum().exp() - 1)).alias("within_code_price_chg_pct")))

    out = (tot.join(within, on="market")
              .with_columns((pl.col("avg_price_chg_pct") - pl.col("within_code_price_chg_pct")).alias("mix_effect_pct_points"))
              .select(["market", "value_chg_pct", "vehicles_chg_pct", "avg_price_chg_pct",
                       "within_code_price_chg_pct", "mix_effect_pct_points"])
              .with_columns(pl.col(pl.Float64).round(1))
              .sort("market", descending=True))
    print(f"\n=== 2024 vs {label}: changes in %, per month averages ===")
    print(out)

    # Top US codes in detail
    top = (m.filter(pl.col("market") == "US").sort("v_0", descending=True).head(6)
             .with_columns((pl.col("q_0")).round(0).alias("cars_2024"),
                           (pl.col("q_1")).round(0).alias("cars_post"),
                           (pl.col("v_0") / pl.col("q_0")).round(0).alias("price_2024"),
                           (pl.col("v_1") / pl.col("q_1")).round(0).alias("price_post"))
             .with_columns((100 * (pl.col("price_post") / pl.col("price_2024") - 1)).round(1).alias("price_chg_pct"))
             .select(["cn8", "cars_2024", "cars_post", "price_2024", "price_post", "price_chg_pct"]))
    print(f"Six biggest car codes to the US (cars per month, EUR per car), 2024 vs {label}:")
    print(top)
