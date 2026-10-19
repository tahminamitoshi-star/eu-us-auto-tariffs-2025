"""Step 3.8: Why do car sales to the US stay down? Germany against the other EU exporters.

Idea: German carmakers (BMW, Mercedes-Benz, Volkswagen) have large US assembly plants. If they moved production for
the US market there, German car exports to the US should stay down after the tariff fell to 15%, and German exports
of core parts to the US (which feed those plants) should hold up or rise. Exporters without big US plants have
less reason to stay out of the US once the tariff falls.

Part 1 (descriptive): each EU country's share of EU car and core-parts exports to the US in 2024.
Part 2 (PPML, main specification of step25): the car and parts effects split by exporter, DE vs REST of the EU,
for two periods: HIGH = Apr-Sep 2025 (27.5% until the deal was applied) and LOW = Oct 2025 - Jun 2026 (15%).
Same fixed effects, seasonal effects, China out of OTHER, Jan-Mar 2025 out, clustered by exporter x HS6.
Prints % effects and the test that DE and REST are equal. Saves 5_results/table_by_exporter.csv
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
OUT = BASE / "5_results"
RAW = pl.read_parquet(BASE / "7_processed_data" / "panel_v1.parquet",
                      columns=["exporter", "hs6", "dest", "us", "period", "group_v4", "value_eur", "frontload"])
RAW = RAW.filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", "4 CONTROL"]))

# Part 1: who exports cars and parts to the US
sh = (RAW.filter(pl.col("us") & pl.col("period").str.starts_with("2024")
                 & pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE"]))
         .group_by(["group_v4", "exporter"]).agg((pl.col("value_eur").sum() / 1e9).alias("bn_2024"))
         .with_columns((100 * pl.col("bn_2024") / pl.col("bn_2024").sum().over("group_v4")).round(1).alias("share_pct"))
         .sort(["group_v4", "bn_2024"], descending=[False, True]))
pl.Config.set_tbl_rows(20)
for g in ["1 CARS", "2 PARTS_CORE"]:
    print(f"\n{g}: largest EU exporters to the US, 2024 (EUR bn, % of EU total)")
    print(sh.filter(pl.col("group_v4") == g).head(10).select(["exporter", pl.col("bn_2024").round(2), "share_pct"]))

# Part 2: PPML split by exporter
d = RAW.filter((pl.col("dest") != "CN") & ~pl.col("frontload"))
P = (d.with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
      .group_by(["exporter", "hs6", "market", "period", "group_v4"])
      .agg((pl.col("value_eur").sum() / 1e6).alias("v")))
periods = [("202504", "202509", "HIGH"), ("202510", "202606", "LOW")]
cols = []
for g, short in [("1 CARS", "CARS"), ("2 PARTS_CORE", "PARTS")]:
    for who, cond in [("DE", pl.col("exporter") == "DE"), ("REST", pl.col("exporter") != "DE")]:
        for a, b, per in periods:
            c = f"{short}_{who}_{per}"
            P = P.with_columns(((pl.col("group_v4") == g) & (pl.col("market") == "US") & cond
                                & pl.col("period").is_between(pl.lit(a), pl.lit(b))).cast(pl.Float64).alias(c))
            cols.append(c)
P = P.with_columns(
    pl.concat_str(["exporter", "hs6", "market"], separator="_").cast(pl.Categorical).to_physical().alias("fe_flow"),
    pl.concat_str(["exporter", "hs6", "period"], separator="_").cast(pl.Categorical).to_physical().alias("fe_prod_month"),
    pl.concat_str(["market", "period"], separator="_").cast(pl.Categorical).to_physical().alias("fe_mkt_month"),
    pl.concat_str([pl.col("group_v4"), pl.col("market"), pl.col("period").str.slice(4, 2)], separator="_")
      .cast(pl.Categorical).to_physical().alias("fe_season"),
    pl.concat_str(["exporter", "hs6"], separator="_").cast(pl.Categorical).to_physical().alias("cluster"))
df = P.select(["v", "cluster", "fe_flow", "fe_prod_month", "fe_mkt_month", "fe_season"] + cols).to_pandas()
fit = pf.fepois(f"v ~ {' + '.join(cols)} | fe_flow + fe_prod_month + fe_mkt_month + fe_season",
                data=df, vcov={"CRV1": "cluster"})
T = fit.tidy().reset_index().rename(columns={"Coefficient": "term"})
parts = T["term"].str.split("_", expand=True)
T["group"], T["exporter"], T["period"] = parts[0], parts[1], parts[2]
for a, b in [("Estimate", "pct"), ("2.5%", "pct_low"), ("97.5%", "pct_high")]:
    T[b] = 100 * (np.exp(T[a]) - 1)
T.to_csv(OUT / "table_by_exporter.csv", index=False)

print(f"\nPPML: {fit._N:,} observations. Effect on exports to the US, in % (95% interval):")
for g in ["CARS", "PARTS"]:
    for per in ["HIGH", "LOW"]:
        row = []
        for who in ["DE", "REST"]:
            r = T[(T["group"] == g) & (T["exporter"] == who) & (T["period"] == per)].iloc[0]
            row.append(f"{who} {r['pct']:+.1f} [{r['pct_low']:+.0f}, {r['pct_high']:+.0f}] p={r['Pr(>|t|)']:.3f}")
        R = np.zeros((1, len(cols)))
        R[0, cols.index(f"{g}_DE_{per}")], R[0, cols.index(f"{g}_REST_{per}")] = 1.0, -1.0
        w = fit.wald_test(R=R)
        print(f"  {g:5s} {per:4s}: " + " | ".join(row) + f" | DE = REST: p = {float(w['pvalue']):.3f}")
