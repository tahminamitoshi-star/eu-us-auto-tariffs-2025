"""Step 1.6: Cars in vehicles, not only euros.

Cars = CN codes starting 8703 (passenger cars), plus 870421 and 870431 (small goods vehicles).
For cars, Eurostat reports a second quantity: number of vehicles (unit NAR = number of articles).

Part A: check that every car code reports NAR.
Part B: EU car exports to the US vs to other non-EU markets, average per month in three periods:
        2024 (before the tariff), Apr-Sep 2025 (25% tariff), Jan-Jun 2026 (15% deal rate).
        Value, number of vehicles, and average price per vehicle.
Part C: the ten biggest car codes exported to the US in 2024.
"""
import os
from pathlib import Path
import polars as pl

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
P = pl.read_parquet(BASE / "7_processed_data" / "exports_ch84_87.parquet")

cars = P.filter(pl.col("cn8").str.starts_with("8703")
                | pl.col("cn8").str.slice(0, 6).is_in(["870421", "870431"]))

# Part A
print("Part A. Units reported for car codes (rows):")
print(cars.group_by("unit").len())

# Part B
per = (pl.when(pl.col("period").str.starts_with("2024")).then(pl.lit("1) 2024"))
         .when(pl.col("period").is_between(pl.lit("202504"), pl.lit("202509"))).then(pl.lit("2) Apr-Sep 2025"))
         .when(pl.col("period").is_between(pl.lit("202601"), pl.lit("202606"))).then(pl.lit("3) Jan-Jun 2026"))
         .otherwise(None))
market = (pl.when(pl.col("dest") == "US").then(pl.lit("US"))
            .when(pl.col("dest") != "EU").then(pl.lit("Other non-EU"))
            .otherwise(None))
b = (cars.with_columns(per.alias("per"), market.alias("market"))
         .drop_nulls(["per", "market"])
         .group_by(["market", "per"])
         .agg((pl.col("value_eur").sum() / 1e9).alias("bn_eur"),
              pl.col("qty_units").sum().alias("vehicles"),
              pl.col("period").n_unique().alias("months"))
         .with_columns((pl.col("bn_eur") / pl.col("months")).round(2).alias("bn_eur_per_month"),
                       (pl.col("vehicles") / pl.col("months")).round(0).alias("vehicles_per_month"),
                       (pl.col("bn_eur") * 1e9 / pl.col("vehicles")).round(0).alias("eur_per_vehicle"))
         .select(["market", "per", "bn_eur_per_month", "vehicles_per_month", "eur_per_vehicle"])
         .sort(["market", "per"], descending=[True, False]))
print("\nPart B. EU car exports, average per month:")
pl.Config.set_tbl_width_chars(120)
print(b)

# Part C
c = (cars.filter((pl.col("dest") == "US") & pl.col("period").str.starts_with("2024"))
         .group_by("cn8")
         .agg(pl.col("value_eur").sum().alias("v"),
              pl.col("qty_units").sum().alias("vehicles_2024"))
         .with_columns((pl.col("v") / 1e9).round(2).alias("bn_eur_2024"),
                       (pl.col("v") / pl.col("vehicles_2024")).round(0).alias("eur_per_vehicle"))
         .sort("v", descending=True).drop("v")
         .head(10))
print("\nPart C. Top 10 car codes exported to the US in 2024:")
print(c)
