"""Step 3.9: Price cuts by exporter: Germany against the rest of the EU (copy of step26 with a DE/REST split).

Step 3.8: German car exports to the US fell less (-22%) than the rest of the EU (-45%) in Apr-Sep 2025.
Question: did German exporters cut their US prices more, absorbing the tariff and keeping volume (as dominant
suppliers do in Freund, L16a)? Periods: HIGH = Apr-Sep 2025, LOW = Oct 2025 - Jun 2026.

Original description of step26 follows.
Step 3.7: Did EU exporters cut their prices for the US?

Price = unit value of each flow in each month: EUR per vehicle for cars (supplementary units), EUR per kg for core
parts and the clean control. Prices are exporters' prices at the EU border (exports are valued free on board,
without the US tariff), so a fall means the EU exporter absorbed part of the tariff.

Data: panel_v1.parquet at CN8 level (units are reported per CN8), summed to exporter x CN8 x market x month,
market = US or OTHER (other non-EU destinations without China, as in the main specification). Months with trade only.
Cleaning (stated in the log): drop flow-months with fewer than 5 vehicles (cars) or less than 100 kg (others),
and flow-months whose log price is more than log(5) away from the flow's own 2023-2024 median.
Jan-Mar 2025 left out.

Model (OLS on log price):
  log p[e,c,m,t] = sum over g in {CARS, PARTS} and period k of b[g,k] * 1[group g] * 1[US] * 1[t in k]
                   + FE(e,c,m) + FE(e,c,t) + FE(m,t) + FE(group x market x calendar month)
  b[g,k]: change in the US price relative to the OTHER price of the same exporter, product and month,
  relative to the same difference for the clean control. 100*(exp(b)-1) = % price change.
Two versions, the point of the Freund vs Hinz/Amiti debate (L16a, L18, L19):
  (1) unweighted: every flow counts the same
  (2) weighted by the flow's 2023-2024 export value: large flows count more
Clustered by exporter x CN8. Saves 5_results/table_prices.csv and table_prices.txt
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
GROUPS = [("1 CARS", "CARS"), ("2 PARTS_CORE", "PARTS")]
PERIODS = [("202504", "202509", "HIGH"), ("202510", "202606", "LOW")]
WHO = [("DE", pl.col("exporter") == "DE"), ("REST", pl.col("exporter") != "DE")]

P = (pl.scan_parquet(BASE / "7_processed_data" / "panel_v1.parquet")
       .filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", "4 CONTROL"]) & (pl.col("dest") != "CN")
               & ~pl.col("frontload"))
       .with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
       .group_by(["exporter", "cn8", "market", "period", "group_v4"])
       .agg(pl.col("value_eur").sum(), pl.col("qty_units").sum(), pl.col("qty_kg").sum())
       .collect())
is_car = pl.col("group_v4") == "1 CARS"
P = (P.with_columns(pl.when(is_car).then(pl.col("qty_units")).otherwise(pl.col("qty_kg")).alias("q"))
      .filter((pl.col("value_eur") > 0)
              & pl.when(is_car).then(pl.col("q") >= 5).otherwise(pl.col("q") >= 100))
      .with_columns((pl.col("value_eur") / pl.col("q")).log().alias("lp"),
                    pl.concat_str(["exporter", "cn8", "market"], separator="_").alias("flow")))
n0 = P.height
base = (P.filter(pl.col("period") < "202501").group_by("flow")
         .agg(pl.col("lp").median().alias("lp_med"), pl.col("value_eur").sum().alias("w")))
P = (P.join(base, on="flow", how="inner")
      .filter((pl.col("lp") - pl.col("lp_med")).abs() <= np.log(5)))
print(f"Price observations: {n0:,} with trade; {P.height:,} after cleaning "
      f"(flows without 2023-2024 trade or with extreme prices dropped)")

cols = []
for g, short in GROUPS:
    for who, cond in WHO:
        for a, b, per in PERIODS:
            c = f"{short}{who}_{per}"
            P = P.with_columns(((pl.col("group_v4") == g) & (pl.col("market") == "US") & cond
                                & pl.col("period").is_between(pl.lit(a), pl.lit(b))).cast(pl.Float64).alias(c))
            cols.append(c)
P = P.with_columns(
    pl.col("flow").cast(pl.Categorical).to_physical().alias("fe_flow"),
    pl.concat_str(["exporter", "cn8", "period"], separator="_").cast(pl.Categorical).to_physical().alias("fe_prod_month"),
    pl.concat_str(["market", "period"], separator="_").cast(pl.Categorical).to_physical().alias("fe_mkt_month"),
    pl.concat_str([pl.col("group_v4"), pl.col("market"), pl.col("period").str.slice(4, 2)], separator="_")
      .cast(pl.Categorical).to_physical().alias("fe_season"),
    pl.concat_str(["exporter", "cn8"], separator="_").cast(pl.Categorical).to_physical().alias("cluster"),
    (pl.col("w") / pl.col("w").mean()).alias("weight"))
df = P.select(["lp", "weight", "cluster", "fe_flow", "fe_prod_month", "fe_mkt_month", "fe_season"] + cols).to_pandas()

fml = f"lp ~ {' + '.join(cols)} | fe_flow + fe_prod_month + fe_mkt_month + fe_season"
res = []
for name, kw in [("(1) unweighted", {}), ("(2) weighted by 2023-24 value", {"weights": "weight"})]:
    start = time.time()
    fit = pf.feols(fml, data=df, vcov={"CRV1": "cluster"}, **kw)
    T = fit.tidy().reset_index().rename(columns={"Coefficient": "term"})
    T["spec"] = name
    T["nobs"] = fit._N
    res.append(T)
    print(f"{name}: done in {time.time() - start:.0f} s, {fit._N:,} observations")
R = pd.concat(res, ignore_index=True)
R["group"] = R["term"].str.split("_").str[0]
R["period"] = R["term"].str.split("_").str[1]
for a, b in [("Estimate", "pct"), ("2.5%", "pct_low"), ("97.5%", "pct_high")]:
    R[b] = 100 * (np.exp(R[a]) - 1)
R.to_csv(OUT / "table_prices_by_exporter.csv", index=False)


def cell(r):
    p = r["Pr(>|t|)"]
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    return f"{r['pct']:+.1f}{stars} [{r['pct_low']:+.1f}, {r['pct_high']:+.1f}]"


R["cell"] = R.apply(cell, axis=1)
lines = ["Change in EU export prices (EUR, before the US tariff) for the US relative to other non-EU markets "
         "(without China), relative to the clean control, in %",
         "OLS on log unit value; 95% intervals; * p<0.10, ** p<0.05, *** p<0.01; clustered by exporter x CN8.", ""]
for g, label in [("CARSDE", "Cars, Germany (EUR per vehicle)"), ("CARSREST", "Cars, rest of EU"),
                 ("PARTSDE", "Core parts, Germany (EUR per kg)"), ("PARTSREST", "Core parts, rest of EU")]:
    t = R[R["group"] == g].pivot(index="period", columns="spec", values="cell")
    t = t.reindex([p for _, _, p in PERIODS])
    lines += [f"--- {label} ---", t.to_string(), ""]
text = "\n".join(lines)
(OUT / "table_prices_by_exporter.txt").write_text(text, encoding="utf-8")
print("\n" + text)
