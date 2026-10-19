"""Step 1.23: Groups v4 (after the truck tariff and the steel/aluminium derivative checks), and the raw comparison.

Starts from hs6_groups_v3_metalflags.csv (step18), which carries the v3 group, the truck flags (step16)
and the metal-derivative flags (step18). Changes:
  4  CONTROL        codes that paid only the general tariff in every period: v3 CONTROL codes that were never a
                    steel/aluminium derivative (2025 lists) and have no 2026 tier, plus HEAVY_VEHICLES codes that are
                    on neither the truck list nor any metal list.
  4m CONTROL_METAL  v3 CONTROL codes (and unlisted HEAVY_VEHICLES codes) that were metal derivatives at some point:
                    general tariff plus a metal duty (on metal content in 2025; on full value from 6 Apr 2026).
  6  HEAVY_VEHICLES only codes on the truck list (vehicles or parts), tariffed as trucks from 1 Nov 2025.
  Other groups unchanged. Flags kept: aircraft_dual_use, ag_tractor (870191-870195), metal_2025_from, metal_2026_tier.
Saves 7_processed_data/hs6_groups_v4.csv

Raw comparison: EU exports per month, US vs other non-EU markets, change against 2024 average.
Periods: Apr-Jul 2025 (before the August metal additions and the EU 15% deal), Aug-Sep 2025,
Jan-Mar 2026, Apr-Jun 2026 (after the 2026 metal regime). Gap = change to the US minus change to other markets.
"""
import os
from pathlib import Path
import polars as pl

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
P = pl.read_parquet(BASE / "7_processed_data" / "exports_ch84_87.parquet")
F = pl.read_csv(BASE / "7_processed_data" / "hs6_groups_v3_metalflags.csv", schema_overrides={"hs6": pl.Utf8})

metal = (pl.col("metal_2025_from") != "never") | (pl.col("metal_2026_tier") != "none")
truck = (pl.col("truck_bus_vehicle") != "none") | (pl.col("truck_part") != "none")
F = F.rename({"deriv_2025_from": "metal_2025_from", "deriv_2026_tier": "metal_2026_tier"})
G = F.with_columns(
        pl.when(pl.col("group_v3") == "4 CONTROL")
          .then(pl.when(metal).then(pl.lit("4m CONTROL_METAL")).otherwise(pl.lit("4 CONTROL")))
          .when(pl.col("group_v3") == "6 HEAVY_VEHICLES")
          .then(pl.when(truck).then(pl.lit("6 HEAVY_VEHICLES"))
                  .when(metal).then(pl.lit("4m CONTROL_METAL")).otherwise(pl.lit("4 CONTROL")))
          .otherwise(pl.col("group_v3")).alias("group_v4"),
        pl.col("hs6").is_in(["870191", "870192", "870193", "870194", "870195"]).alias("ag_tractor"))
G.select(["hs6", "annex_status", "group_v3", "group_v4", "aircraft_dual_use", "ag_tractor",
          "truck_bus_vehicle", "truck_part", "metal_2025_from", "metal_2026_tier"]).sort("hs6") \
 .write_csv(BASE / "7_processed_data" / "hs6_groups_v4.csv")

print("Moves from v3 to v4 (HS6 codes):")
print(G.group_by(["group_v3", "group_v4"]).len().sort(["group_v3", "group_v4"]))

D = (P.with_columns(pl.col("cn8").str.slice(0, 6).alias("hs6"))
      .join(G.select(["hs6", "group_v4"]), on="hs6")
      .filter((pl.col("dest") != "EU") & (pl.col("group_v4") != "9 DROPPED"))
      .with_columns(pl.when(pl.col("dest") == "US").then(pl.lit("US")).otherwise(pl.lit("Other")).alias("market")))

def avg(first, last, name):
    d = D.filter(pl.col("period").is_between(pl.lit(first), pl.lit(last)))
    n = d["period"].n_unique()
    return d.group_by(["group_v4", "market"]).agg((pl.col("value_eur").sum() / n / 1e9).alias(name))

periods = [("202504", "202507", "AprJul25"), ("202508", "202509", "AugSep25"),
           ("202601", "202603", "JanMar26"), ("202604", "202606", "AprJun26")]
T = avg("202401", "202412", "y2024")
for a, b, n in periods:
    T = T.join(avg(a, b, n), on=["group_v4", "market"], how="left")
T = T.with_columns([(100 * (pl.col(n) / pl.col("y2024") - 1)).alias(n) for _, _, n in periods])
W = T.pivot(on="market", index="group_v4", values=["y2024"] + [n for _, _, n in periods])
W = W.with_columns([(pl.col(f"{n}_US") - pl.col(f"{n}_Other")).alias(f"gap_{n}") for _, _, n in periods])
W = (W.select(["group_v4", "y2024_US"] + [f"gap_{n}" for _, _, n in periods])
      .with_columns(pl.col(pl.Float64).round(1)).sort("group_v4"))
pl.Config.set_tbl_width_chars(160)
pl.Config.set_tbl_rows(20)
print("\n2024 exports to the US per month (EUR bn) and gap (US minus other markets, % change vs 2024) by period")
print(W)
