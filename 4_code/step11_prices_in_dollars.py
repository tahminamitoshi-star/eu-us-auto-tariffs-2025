"""Step 1.11: Car prices in US dollars.

The euro rose against the dollar in 2025. If carmakers keep their US prices fixed in dollars,
each car brings in fewer euros even though nothing changed for US buyers.
Here every monthly euro value is converted to dollars with that month's ECB rate
(USD per EUR, monthly average, series EXR.M.USD.EUR.SP00.A), and the price changes are recomputed.

Source: ECB Data Portal, saved as 2_raw_data/ecb_exchange_rates/ecb_eur_usd_monthly.csv
"""
import os
from pathlib import Path
import polars as pl

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
P = pl.read_parquet(BASE / "7_processed_data" / "exports_ch84_87.parquet")

# Exchange rate: columns are DATE, TIME PERIOD, and the rate; keep YYYYMM and the rate
fx = pl.read_csv(BASE / "2_raw_data" / "ecb_exchange_rates" / "ecb_eur_usd_monthly.csv", infer_schema=False)
fx = fx.select(pl.col(fx.columns[0]).str.slice(0, 7).str.replace("-", "").alias("period"),
               pl.col(fx.columns[2]).cast(pl.Float64).alias("usd_per_eur"))

def avg_rate(first, last):
    return fx.filter(pl.col("period").is_between(pl.lit(first), pl.lit(last)))["usd_per_eur"].mean()

r0, r1, r2 = avg_rate("202401", "202412"), avg_rate("202504", "202509"), avg_rate("202601", "202606")
print(f"Average USD per EUR: 2024 {r0:.4f} | Apr-Sep 2025 {r1:.4f} ({100*(r1/r0-1):+.1f}%) "
      f"| Jan-Jun 2026 {r2:.4f} ({100*(r2/r0-1):+.1f}%)")

hs4 = pl.col("cn8").str.slice(0, 4)
hs6 = pl.col("cn8").str.slice(0, 6)
D = (P.filter(pl.col("dest") == "US")
      .join(fx, on="period", how="left")
      .with_columns((pl.col("value_eur") * pl.col("usd_per_eur")).alias("value_usd"),
                    pl.when((hs4 == "8703") | hs6.is_in(["870421", "870431"])).then(pl.lit("1 CARS"))
                      .when(hs4.is_in(["8708", "8409"]) | hs6.is_in(["840733", "840734", "840820"])).then(pl.lit("2 PARTS"))
                      .otherwise(pl.lit("3 OTHER")).alias("group")))
print("Months without an exchange rate:", D.filter(pl.col("usd_per_eur").is_null())["period"].unique().to_list())

KEYS = ["group", "exporter", "cn8"]

def period_sum(first, last, s):
    return (D.filter(pl.col("period").is_between(pl.lit(first), pl.lit(last)))
             .group_by(KEYS)
             .agg(pl.col("value_eur").sum().alias(f"eur{s}"), pl.col("value_usd").sum().alias(f"usd{s}"),
                  pl.col("qty_units").sum().alias(f"q{s}"), pl.col("qty_kg").sum().alias(f"kg{s}")))

base = period_sum("202401", "202412", "0")
pl.Config.set_tbl_width_chars(140)
for first, last, label in [("202504", "202509", "Apr-Sep 2025"), ("202601", "202606", "Jan-Jun 2026")]:
    m = (base.join(period_sum(first, last, "1"), on=KEYS, how="inner")
             .filter((pl.col("eur0") > 0) & (pl.col("eur1") > 0) & (pl.col("kg0") > 0) & (pl.col("kg1") > 0))
             .with_columns((pl.col("eur0") / pl.col("eur0").sum().over("group")).alias("w")))
    # Cars: price per car; other groups: price per kg (they have no common unit)
    qty0 = pl.when(pl.col("group") == "1 CARS").then(pl.col("q0")).otherwise(pl.col("kg0"))
    qty1 = pl.when(pl.col("group") == "1 CARS").then(pl.col("q1")).otherwise(pl.col("kg1"))
    m = m.with_columns(qty0.alias("u0"), qty1.alias("u1")).filter((pl.col("u0") > 0) & (pl.col("u1") > 0))
    out = (m.group_by("group")
             .agg(pl.len().alias("pairs"),
                  (100 * ((pl.col("w") * ((pl.col("eur1") / pl.col("u1")) / (pl.col("eur0") / pl.col("u0"))).log()).sum().exp() - 1))
                     .round(1).alias("price_chg_in_EUR_pct"),
                  (100 * ((pl.col("w") * ((pl.col("usd1") / pl.col("u1")) / (pl.col("usd0") / pl.col("u0"))).log()).sum().exp() - 1))
                     .round(1).alias("price_chg_in_USD_pct"))
             .sort("group"))
    print(f"\nExports to the US, within exporter x code, 2024 vs {label} (cars: per car; parts and other: per kg):")
    print(out)
