"""Step 1.13: New product groups based on the legal list (three tiers), and a first raw comparison.

Groups (HS6 level, chapters 84-87):
  CARS        vehicle codes in note 33(b): 8703.22-8703.90 and light trucks 870421, 870431, 870441, 870451, 870460
  PARTS_CORE  fully listed AND car-only by definition: parts and bodies in 8706, 8707, 8708, and vehicle
              engines 840731-840734
  UNCERTAIN   everything else that is listed in full or in part (dual-use goods such as batteries, bearings,
              control panels, computers; partly listed car parts such as gear boxes 870840), plus the two
              unlisted car codes 870310 and 870321. Kept out of the control group and studied separately.
  CONTROL     nothing listed. Aircraft engines and parts (8411, 8412) are dropped: they had their own
              tariff treatment (duty-free for the EU from September 2025).

Raw comparison: EU exports per month, US vs other non-EU markets, 2024 vs Apr-Sep 2025 and Jan-Jun 2026.
"Gap" = change to the US minus change to other markets (a first, uncontrolled difference-in-differences).
Saves 7_processed_data/hs6_groups_v2.csv
"""
import os
from pathlib import Path
import polars as pl

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
P = pl.read_parquet(BASE / "7_processed_data" / "exports_ch84_87.parquet")
S = pl.read_csv(BASE / "7_processed_data" / "hs6_annex_status.csv", schema_overrides={"hs6": pl.Utf8})

CARS = ["870322", "870323", "870324", "870331", "870332", "870333", "870340", "870350", "870360",
        "870370", "870380", "870390", "870421", "870431", "870441", "870451", "870460"]
hs4 = pl.col("hs6").str.slice(0, 4)

# All HS6 codes in the dataset (some have no US exports in 2024 and are missing from S)
allhs6 = (P.select(pl.col("cn8").str.slice(0, 6).alias("hs6")).unique()
           .join(S.select(["hs6", "annex_status"]), on="hs6", how="left")
           .with_columns(pl.col("annex_status").fill_null("none")))

G = allhs6.with_columns(
    pl.when(pl.col("hs6").is_in(CARS)).then(pl.lit("1 CARS"))
      .when((pl.col("annex_status") == "full")
            & (hs4.is_in(["8706", "8707", "8708"]) | pl.col("hs6").is_in(["840731", "840732", "840733", "840734"])))
        .then(pl.lit("2 PARTS_CORE"))
      .when((pl.col("annex_status") != "none") | pl.col("hs6").is_in(["870310", "870321"])).then(pl.lit("3 UNCERTAIN"))
      .when(hs4.is_in(["8411", "8412"])).then(pl.lit("9 DROPPED"))
      .otherwise(pl.lit("4 CONTROL")).alias("group"))
G.sort("hs6").write_csv(BASE / "7_processed_data" / "hs6_groups_v2.csv")
print("HS6 codes per group:")
print(G.group_by("group").len().sort("group"))

D = (P.with_columns(pl.col("cn8").str.slice(0, 6).alias("hs6"))
      .join(G.select(["hs6", "group"]), on="hs6")
      .filter((pl.col("dest") != "EU") & (pl.col("group") != "9 DROPPED"))
      .with_columns(pl.when(pl.col("dest") == "US").then(pl.lit("US")).otherwise(pl.lit("Other")).alias("market")))

def avg(first, last, name):
    d = D.filter(pl.col("period").is_between(pl.lit(first), pl.lit(last)))
    n = d["period"].n_unique()
    return d.group_by(["group", "market"]).agg((pl.col("value_eur").sum() / n / 1e9).alias(name))

T = (avg("202401", "202412", "y2024")
       .join(avg("202504", "202509", "s2025"), on=["group", "market"])
       .join(avg("202601", "202606", "h2026"), on=["group", "market"])
       .with_columns((100 * (pl.col("s2025") / pl.col("y2024") - 1)).alias("chg_AprSep25"),
                     (100 * (pl.col("h2026") / pl.col("y2024") - 1)).alias("chg_JanJun26")))
W = (T.pivot(on="market", index="group", values=["y2024", "chg_AprSep25", "chg_JanJun26"])
       .with_columns((pl.col("chg_AprSep25_US") - pl.col("chg_AprSep25_Other")).alias("gap_AprSep25"),
                     (pl.col("chg_JanJun26_US") - pl.col("chg_JanJun26_Other")).alias("gap_JanJun26"))
       .select(["group", "y2024_US", "chg_AprSep25_US", "chg_AprSep25_Other", "gap_AprSep25",
                "chg_JanJun26_US", "chg_JanJun26_Other", "gap_JanJun26"])
       .with_columns(pl.col(pl.Float64).round(1))
       .sort("group"))
pl.Config.set_tbl_width_chars(160)
print("\nEU exports per month: 2024 level to the US (EUR bn) and % change vs 2024")
print(W)
