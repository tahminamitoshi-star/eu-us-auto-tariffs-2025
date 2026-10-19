"""Step 1.9: Did euro prices fall only for cars, or for everything sold to the US?

If the euro price of ALL goods sent to the US fell (machinery, parts, not only cars), the cause is
probably common to the US market, such as the stronger euro with prices set in dollars.
If only cars fell, the cause is more likely the car tariff itself.

Price = euros per kg (available for every product). Compared within exporter x 8-digit code,
2024 vs Apr-Sep 2025, for three groups:
  CARS  = 8703, 870421, 870431
  PARTS = 8708, 8409, 840733, 840734, 840820
  OTHER = all other products in chapters 84-87
Two summaries per group: value-weighted mean (big flows count more) and median (each pair counts once).
"""
import os
from pathlib import Path
import polars as pl

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
P = pl.read_parquet(BASE / "7_processed_data" / "exports_ch84_87.parquet")

hs4 = pl.col("cn8").str.slice(0, 4)
hs6 = pl.col("cn8").str.slice(0, 6)
group = (pl.when((hs4 == "8703") | hs6.is_in(["870421", "870431"])).then(pl.lit("1 CARS"))
           .when(hs4.is_in(["8708", "8409"]) | hs6.is_in(["840733", "840734", "840820"])).then(pl.lit("2 PARTS"))
           .otherwise(pl.lit("3 OTHER")))

D = (P.filter(pl.col("dest") != "EU")
      .with_columns(group.alias("group"),
                    pl.when(pl.col("dest") == "US").then(pl.lit("US")).otherwise(pl.lit("Other non-EU")).alias("market")))

KEYS = ["group", "market", "exporter", "cn8"]

def period_sum(first, last, s):
    return (D.filter(pl.col("period").is_between(pl.lit(first), pl.lit(last)))
             .group_by(KEYS)
             .agg(pl.col("value_eur").sum().alias(f"v{s}"), pl.col("qty_kg").sum().alias(f"kg{s}")))

m = (period_sum("202401", "202412", "0")
       .join(period_sum("202504", "202509", "1"), on=KEYS, how="inner")
       .filter((pl.col("v0") > 0) & (pl.col("v1") > 0) & (pl.col("kg0") > 0) & (pl.col("kg1") > 0))
       .with_columns(((pl.col("v1") / pl.col("kg1")) / (pl.col("v0") / pl.col("kg0"))).log().alias("dlog"),
                     (pl.col("v0") / pl.col("v0").sum().over(["group", "market"])).alias("w")))

out = (m.group_by(["group", "market"])
         .agg(pl.len().alias("pairs"),
              (pl.col("v0").sum() / 12e9).round(2).alias("bn_eur_per_month_2024"),
              (100 * ((pl.col("w") * pl.col("dlog")).sum().exp() - 1)).round(1).alias("weighted_eur_per_kg_chg_pct"),
              (100 * (pl.col("dlog").median().exp() - 1)).round(1).alias("median_eur_per_kg_chg_pct"))
         .sort(["group", "market"], descending=[False, True]))
pl.Config.set_tbl_width_chars(140)
print("Euro price per kg, within exporter x code, 2024 vs Apr-Sep 2025:")
print(out)
