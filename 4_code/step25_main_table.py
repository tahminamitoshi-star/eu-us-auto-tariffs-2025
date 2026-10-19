"""Step 3.6: Main results table.

Main specification (decided 13 Oct 2026, Step 3.6 of the log):
  PPML, exporter x HS6 x market (US vs OTHER) x month, groups CARS and PARTS_CORE against the clean CONTROL,
  OTHER = all non-EU destinations except the US and China, seasonal fixed effects (group x market x calendar month),
  Jan-Mar 2025 left out, five post periods, clustered by exporter x HS6.
Columns:
  (1) MAIN
  (2) China kept in OTHER
  (3) no seasonal fixed effects
  (4) Jan-Mar 2025 kept, with its own coefficient (front-loading)
Each cell: % effect = 100 x (exp(b) - 1), 95% interval, stars (* 10%, ** 5%, *** 1%).
Saves 5_results/table_main.csv (all numbers) and 5_results/table_main.txt (formatted table).
"""
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
OUT = BASE / "5_results"
OUT.mkdir(exist_ok=True)
RAW = pl.read_parquet(BASE / "7_processed_data" / "panel_v1.parquet",
                      columns=["exporter", "hs6", "dest", "us", "period", "group_v4", "value_eur"])
RAW = RAW.filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", "4 CONTROL"]))
GROUPS = [("1 CARS", "CARS"), ("2 PARTS_CORE", "PARTS")]
PERIODS = [("202504", "202507", "AprJul25"), ("202508", "202509", "AugSep25"), ("202510", "202512", "OctDec25"),
           ("202601", "202603", "JanMar26"), ("202604", "202606", "AprJun26")]
FRONT = ("202501", "202503", "JanMar25")


def estimate(name, drop_china=True, seasonal=True, keep_front=False):
    d = RAW.filter(pl.col("dest") != "CN") if drop_china else RAW
    if not keep_front:
        d = d.filter(~pl.col("period").is_between(pl.lit("202501"), pl.lit("202503")))
    P = (d.with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
          .group_by(["exporter", "hs6", "market", "period", "group_v4"])
          .agg((pl.col("value_eur").sum() / 1e6).alias("v")))
    periods = ([FRONT] if keep_front else []) + PERIODS
    cols = []
    for g, short in GROUPS:
        for a, b, per in periods:
            c = f"{short}_{per}"
            P = P.with_columns(((pl.col("group_v4") == g) & (pl.col("market") == "US")
                                & pl.col("period").is_between(pl.lit(a), pl.lit(b))).cast(pl.Float64).alias(c))
            cols.append(c)
    fes = ["fe_flow", "fe_prod_month", "fe_mkt_month"]
    P = P.with_columns(
        pl.concat_str(["exporter", "hs6", "market"], separator="_").cast(pl.Categorical).to_physical().alias("fe_flow"),
        pl.concat_str(["exporter", "hs6", "period"], separator="_").cast(pl.Categorical).to_physical().alias("fe_prod_month"),
        pl.concat_str(["market", "period"], separator="_").cast(pl.Categorical).to_physical().alias("fe_mkt_month"),
        pl.concat_str(["exporter", "hs6"], separator="_").cast(pl.Categorical).to_physical().alias("cluster"))
    if seasonal:
        P = P.with_columns(pl.concat_str([pl.col("group_v4"), pl.col("market"), pl.col("period").str.slice(4, 2)],
                                         separator="_").cast(pl.Categorical).to_physical().alias("fe_season"))
        fes.append("fe_season")
    df = P.select(["v", "cluster"] + fes + cols).to_pandas()
    start = time.time()
    fit = pf.fepois(f"v ~ {' + '.join(cols)} | {' + '.join(fes)}", data=df, vcov={"CRV1": "cluster"})
    T = fit.tidy().reset_index().rename(columns={"Coefficient": "term"})
    T["spec"] = name
    T["group"] = T["term"].str.split("_").str[0]
    T["period"] = T["term"].str.split("_").str[1]
    T["pct"] = 100 * (np.exp(T["Estimate"]) - 1)
    T["pct_low"] = 100 * (np.exp(T["2.5%"]) - 1)
    T["pct_high"] = 100 * (np.exp(T["97.5%"]) - 1)
    T["nobs"] = fit._N
    print(f"{name}: done in {time.time() - start:.0f} s, {fit._N:,} observations")
    return T


specs = [("(1) MAIN", {}), ("(2) with China", {"drop_china": False}),
         ("(3) no seasonal FE", {"seasonal": False}), ("(4) Jan-Mar 2025 kept", {"keep_front": True})]
R = pd.concat([estimate(n, **kw) for n, kw in specs], ignore_index=True)
R.to_csv(OUT / "table_main.csv", index=False)


def cell(r):
    p = r["Pr(>|t|)"]
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    return f"{r['pct']:+.1f}{stars} [{r['pct_low']:+.0f}, {r['pct_high']:+.0f}]"


R["cell"] = R.apply(cell, axis=1)
order = ["JanMar25"] + [p for _, _, p in PERIODS]
lines = ["Effect on EU exports to the US relative to other non-EU markets, relative to the clean control, in %",
         "PPML; 95% intervals in brackets; * p<0.10, ** p<0.05, *** p<0.01; clustered by exporter x HS6.", ""]
for g in ["CARS", "PARTS"]:
    t = R[R["group"] == g].pivot(index="period", columns="spec", values="cell")
    t = t.reindex([p for p in order if p in t.index])[[n for n, _ in specs]].fillna("")
    lines.append(f"--- {'Cars' if g == 'CARS' else 'Core car parts'} ---")
    lines.append(t.to_string())
    lines.append("")
nobs = R.groupby("spec")["nobs"].first().reindex([n for n, _ in specs])
lines.append("Observations: " + "; ".join(f"{k} {v:,}" for k, v in nobs.items()))
text = "\n".join(lines)
(OUT / "table_main.txt").write_text(text, encoding="utf-8")
print("\n" + text)
