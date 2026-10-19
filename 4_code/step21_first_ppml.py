"""Step 3.2: First PPML estimate. Cars and core parts against the clean control, US against other non-EU markets.

Data: panel_v1.parquet (step20), summed to exporter x HS6 x market x month, where market = US or OTHER
(all other non-EU destinations together). HS6 because CN8 codes change between years (Step 3.1).
Sample: groups 1 CARS, 2 PARTS_CORE, 4 CONTROL (clean); Jan 2023 - Jun 2026 without Jan-Mar 2025.

Model (PPML):
  exports[e,p,m,t] = exp( sum over g in {CARS, PARTS_CORE} and period k of  b[g,k] * 1[group g] * 1[US] * 1[t in k]
                          + FE(e,p,m) + FE(e,p,t) + FE(m,t) )
  e = exporter, p = HS6, m = market, t = month.
  FE(e,p,m): each flow's normal level. FE(e,p,t): anything that hits one exporter's product in a month in all
  markets (supply shocks, model changes, the euro). FE(m,t): anything that hits all products in the US market in
  a month, including the general tariff paid by the control group.
  So b[g,k] = how much more group g's exports to the US fell (or rose) than its exports to other markets,
  compared with the same difference for the clean control. 100*(exp(b)-1) is the % effect.
Periods: Apr-Jul 2025 | Aug-Sep 2025 | Oct-Dec 2025 | Jan-Mar 2026 | Apr-Jun 2026 (pre-period: 2023-2024).
Standard errors clustered by exporter x HS6.
Saves 5_results/step21_first_ppml.csv
"""
import os
from pathlib import Path
import time
import numpy as np
import polars as pl
import pyfixest as pf

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
OUT = BASE / "5_results"
OUT.mkdir(exist_ok=True)

GROUPS = {"1 CARS": "CARS", "2 PARTS_CORE": "PARTS"}
PERIODS = [("202504", "202507", "AprJul25"), ("202508", "202509", "AugSep25"), ("202510", "202512", "OctDec25"),
           ("202601", "202603", "JanMar26"), ("202604", "202606", "AprJun26")]

P = (pl.scan_parquet(BASE / "7_processed_data" / "panel_v1.parquet")
       .filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", "4 CONTROL"]) & ~pl.col("frontload"))
       .with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
       .group_by(["exporter", "hs6", "market", "period", "group_v4"])
       .agg((pl.col("value_eur").sum() / 1e6).alias("v"))          # EUR million
       .collect())

cols = []
for g, short in GROUPS.items():
    for a, b, name in PERIODS:
        c = f"{short}_{name}"
        P = P.with_columns(((pl.col("group_v4") == g) & (pl.col("market") == "US")
                            & pl.col("period").is_between(pl.lit(a), pl.lit(b))).cast(pl.Float64).alias(c))
        cols.append(c)
P = P.with_columns(
    pl.concat_str(["exporter", "hs6", "market"], separator="_").cast(pl.Categorical).to_physical().alias("fe_flow"),
    pl.concat_str(["exporter", "hs6", "period"], separator="_").cast(pl.Categorical).to_physical().alias("fe_prod_month"),
    pl.concat_str(["market", "period"], separator="_").cast(pl.Categorical).to_physical().alias("fe_mkt_month"),
    pl.concat_str(["exporter", "hs6"], separator="_").cast(pl.Categorical).to_physical().alias("cluster"))
print(f"Estimation sample: {P.height:,} rows, {P['cluster'].n_unique():,} exporter x HS6 pairs, "
      f"{P['period'].n_unique()} months")

df = P.select(["v", "fe_flow", "fe_prod_month", "fe_mkt_month", "cluster"] + cols).to_pandas()
start = time.time()
fit = pf.fepois(f"v ~ {' + '.join(cols)} | fe_flow + fe_prod_month + fe_mkt_month",
                data=df, vcov={"CRV1": "cluster"})
print(f"PPML done in {time.time() - start:.0f} s; observations used: {fit._N:,}")

T = fit.tidy().reset_index().rename(columns={"Coefficient": "term"})
T["group"] = T["term"].str.split("_").str[0]
T["period"] = T["term"].str.split("_").str[1]
T["pct_effect"] = 100 * (np.exp(T["Estimate"]) - 1)
T["pct_low"] = 100 * (np.exp(T["2.5%"]) - 1)
T["pct_high"] = 100 * (np.exp(T["97.5%"]) - 1)
T.to_csv(OUT / "step21_first_ppml.csv", index=False)

print("\nEffect on exports to the US relative to other markets, relative to the clean control (in %):")
show = T[["group", "period", "Estimate", "Std. Error", "Pr(>|t|)", "pct_effect", "pct_low", "pct_high"]].round(3)
print(show.to_string(index=False))
print("\nIn words:")
for _, r in T.iterrows():
    print(f"  {r['group']}, {r['period']}: exports to the US {r['pct_effect']:+.1f}% "
          f"(95% CI {r['pct_low']:+.1f}% to {r['pct_high']:+.1f}%) relative to the control, p = {r['Pr(>|t|)']:.3f}")
