"""Step 1.19: Final product groups for Step 3, and the raw comparison again.

Starts from Step 1.18 (hs6_groups_v2_flags.csv). Changes to the control group:
  5 EXEMPT_ELECTRONICS  control codes exempt from the Section 122 surcharge for any use (note 2(aa)(ii):
                        computers, chip-making machines, chips, phones, network equipment). These goods
                        may not have paid the general tariff in 2025 either, so they cannot stand for
                        "general tariff only". Moved out of the control group.
  6 HEAVY_VEHICLES      tractors, buses, heavy trucks, special vehicles, trailers (8701, 8702, 8704 except light
                        trucks, 8705, 8709, 8716): possibly under the Section 232 heavy-vehicle tariff from
                        late 2025. Moved out of the control group as a precaution.
  4 CONTROL             the rest. Codes exempt only for civil-aircraft use stay in, with a flag
                        (aircraft_dual_use) so Step 3 can test the result without them.
Saves 7_processed_data/hs6_groups_v3.csv
"""
import os
from pathlib import Path
import polars as pl

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
P = pl.read_parquet(BASE / "7_processed_data" / "exports_ch84_87.parquet")
F = pl.read_csv(BASE / "7_processed_data" / "hs6_groups_v2_flags.csv", schema_overrides={"hs6": pl.Utf8})

G = F.with_columns(
        pl.when((pl.col("group") == "4 CONTROL") & (pl.col("s122_exempt_any_use") != "none")).then(pl.lit("5 EXEMPT_ELECTRONICS"))
          .when((pl.col("group") == "4 CONTROL") & pl.col("possible_heavy_vehicle")).then(pl.lit("6 HEAVY_VEHICLES"))
          .otherwise(pl.col("group")).alias("group_v3"),
        (pl.col("s122_exempt_aircraft_use") != "none").alias("aircraft_dual_use"))
G.select(["hs6", "annex_status", "group", "group_v3", "aircraft_dual_use"]).sort("hs6") \
 .write_csv(BASE / "7_processed_data" / "hs6_groups_v3.csv")
print("HS6 codes per group:")
print(G.group_by("group_v3").len().sort("group_v3"))

D = (P.with_columns(pl.col("cn8").str.slice(0, 6).alias("hs6"))
      .join(G.select(["hs6", "group_v3", "aircraft_dual_use"]), on="hs6")
      .filter((pl.col("dest") != "EU") & (pl.col("group_v3") != "9 DROPPED"))
      .with_columns(pl.when(pl.col("dest") == "US").then(pl.lit("US")).otherwise(pl.lit("Other")).alias("market")))
# Extra row: control without aircraft dual-use codes
D = pl.concat([D, D.filter((pl.col("group_v3") == "4 CONTROL") & ~pl.col("aircraft_dual_use"))
                   .with_columns(pl.lit("4b CONTROL_no_aircraft").alias("group_v3"))])

def avg(first, last, name):
    d = D.filter(pl.col("period").is_between(pl.lit(first), pl.lit(last)))
    n = d["period"].n_unique()
    return d.group_by(["group_v3", "market"]).agg((pl.col("value_eur").sum() / n / 1e9).alias(name))

T = (avg("202401", "202412", "y2024").join(avg("202504", "202509", "s2025"), on=["group_v3", "market"])
       .join(avg("202601", "202606", "h2026"), on=["group_v3", "market"])
       .with_columns((100 * (pl.col("s2025") / pl.col("y2024") - 1)).alias("chg_AprSep25"),
                     (100 * (pl.col("h2026") / pl.col("y2024") - 1)).alias("chg_JanJun26")))
W = (T.pivot(on="market", index="group_v3", values=["y2024", "chg_AprSep25", "chg_JanJun26"])
       .with_columns((pl.col("chg_AprSep25_US") - pl.col("chg_AprSep25_Other")).alias("gap_AprSep25"),
                     (pl.col("chg_JanJun26_US") - pl.col("chg_JanJun26_Other")).alias("gap_JanJun26"))
       .select(["group_v3", "y2024_US", "chg_AprSep25_US", "chg_AprSep25_Other", "gap_AprSep25",
                "chg_JanJun26_US", "chg_JanJun26_Other", "gap_JanJun26"])
       .with_columns(pl.col(pl.Float64).round(1)).sort("group_v3"))
pl.Config.set_tbl_width_chars(170)
pl.Config.set_tbl_rows(20)
print("\nEU exports per month: 2024 level to the US (EUR bn) and % change vs 2024; gap = US minus other markets")
print(W)
