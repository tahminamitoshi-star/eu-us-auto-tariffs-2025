"""Step 1.10: Which car flows to the US carry the price fall?

Step 1.9: for cars to the US, the value-weighted price change was -15%, but the median was about -1%.
So the fall sits in a few large exporter x code flows, not in the typical one.
This step lists the largest flows and measures how concentrated the fall is.
Price = euros per car. 2024 vs Apr-Sep 2025.
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
                .agg(pl.col("value_eur").sum().alias(f"v{s}"), pl.col("qty_units").sum().alias(f"q{s}")))

m = (period_sum("202401", "202412", "0")
       .join(period_sum("202504", "202509", "1"), on=KEYS, how="inner")
       .filter((pl.col("q0") > 0) & (pl.col("q1") > 0) & (pl.col("v0") > 0) & (pl.col("v1") > 0))
       .with_columns((pl.col("v0") / pl.col("q0")).alias("p0"), (pl.col("v1") / pl.col("q1")).alias("p1"))
       .with_columns((100 * (pl.col("p1") / pl.col("p0") - 1)).alias("price_chg_pct"),
                     (100 * pl.col("v0") / pl.col("v0").sum().over("market")).alias("share_2024_pct")))

pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_width_chars(140)

print("A. The 15 largest car flows to the US (2024 value), price per car 2024 vs Apr-Sep 2025:")
print(m.filter(pl.col("market") == "US").sort("v0", descending=True).head(15)
       .select(["exporter", "cn8",
                (pl.col("v0") / 12e6).round(0).alias("m_eur_per_month_2024"),
                pl.col("share_2024_pct").round(1),
                pl.col("p0").round(0).alias("eur_car_2024"),
                pl.col("p1").round(0).alias("eur_car_post"),
                pl.col("price_chg_pct").round(1)]))

print("\nB. Share of 2024 car export value in flows whose price per car fell more than 10%:")
print(m.group_by("market")
       .agg(pl.len().alias("flows"),
            pl.col("share_2024_pct").filter(pl.col("price_chg_pct") < -10).sum().round(1).alias("value_share_fall_gt10_pct"),
            (100 * (pl.col("price_chg_pct") < -10).mean()).round(1).alias("flow_share_fall_gt10_pct"))
       .sort("market", descending=True))

print("\nC. Weighted price change for cars to the US, with and without Germany:")
for label, d in [("All exporters", m), ("Without DE", m.filter(pl.col("exporter") != "DE"))]:
    d = d.with_columns((pl.col("v0") / pl.col("v0").sum().over("market")).alias("w"))
    r = (d.group_by("market")
          .agg((100 * ((pl.col("w") * (pl.col("p1") / pl.col("p0")).log()).sum().exp() - 1)).round(1).alias("pct"))
          .sort("market", descending=True))
    print(label, dict(zip(r["market"], r["pct"])))
