"""Step 6.5: two open reviewer items (points 4 and 23).

Part A. Table 1: number of observations and number of clusters for all four columns.
  The four models are re-estimated exactly as in step31 (col. 1), step32 Part 3 (col. 2), step32 Part 2a (col. 3)
  and step33 Part 1 (col. 4). The script prints the estimate next to the saved one, so you can see that nothing
  changed, and then N (observations used, after PPML drops singletons and separated flows) and the number of
  exporter x HS6 clusters in the estimation sample.
Part B. Weight per car (step37 E part 3, which failed with a ValueError).
  Cause: step37 ran the kg model on cars only but kept the market x month fixed effect from the price model. With
  cars only, "US x period" is the same as market x month, so pyfixest dropped every variable as collinear
  (ValueError). The price model avoided this because control products share the market x month effect.
  Fix: cars only, comparing the same exporter x CN8 code in the same month across the US and other markets:
  fixed effects flow (exporter x CN8 x market), exporter x CN8 x month, and market x calendar month (season).
  Same price sample, weights and clustering as step37 E. Flow-months with 0 kg are dropped (log of 0).
  Outcome: log kg per car. A shift toward lighter (smaller, cheaper) models inside a CN8 code would show up as a
  negative US coefficient.
Saves 5_results/step39_table1_counts.csv and 5_results/step39_kg_per_car.csv
"""
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf

warnings.filterwarnings("ignore")
BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
OUT, DATA = BASE / "5_results", BASE / "7_processed_data"
HIGH, LOW = ("202504", "202509"), ("202510", "202606")
inp = lambda p: pl.col("period").is_between(pl.lit(p[0]), pl.lit(p[1]))
cat = lambda cs, n: pl.concat_str(cs, separator="_").cast(pl.Categorical).to_physical().alias(n)
pct = lambda x: 100 * (np.exp(x) - 1)
G = pl.read_csv(DATA / "hs6_groups_v4.csv", schema_overrides={"hs6": pl.Utf8})
KEEP = ["1 CARS", "2 PARTS_CORE", "4 CONTROL"]


def fit(Q, terms, fes, label, saved=None):
    Q = Q.with_columns([c.cast(pl.Float64).alias(n) for n, c in terms.items()])
    names = [n for n in terms if Q[n].sum() > 0]
    df = Q.select(["v", "cl"] + fes + names).to_pandas()
    m = pf.fepois(f"v ~ {' + '.join(names)} | {' + '.join(fes)}", data=df, vcov={"CRV1": "cl"})
    T = m.tidy()
    n_cl = int(m._G[0]) if hasattr(m, "_G") else np.nan
    print(f"\n{label}: N = {m._N:,} observations, {n_cl:,} exporter x HS6 clusters")
    for t in ["CARS_HIGH", "CARS_LOW", "PARTS_HIGH", "PARTS_LOW"]:
        b = T.loc[t, "Estimate"]
        old = ""
        if saved is not None and t in saved.index:
            old = f"   saved: {pct(saved.loc[t, 'Estimate']):+.1f}%"
        print(f"  {t:11s} {pct(b):+6.1f}%  p = {T.loc[t, 'Pr(>|t|)']:.3f}{old}")
    return dict(column=label, N=m._N, clusters=n_cl)


panel = pl.read_parquet(DATA / "panel_v1.parquet",
                        columns=["exporter", "hs6", "dest", "us", "period", "group_v4", "value_eur", "frontload"])
panel = panel.filter(pl.col("group_v4").is_in(KEEP) & ~pl.col("frontload") & (pl.col("dest") != "CN"))
rows = []
s32 = pd.read_csv(OUT / "step32_deflection_ppml.csv")
s33 = pd.read_csv(OUT / "step33_intra_eu.csv")
s31 = pd.read_csv(OUT / "step31_clusters.csv")

# ---- column 1: US vs other non-EU markets (main model, as step31 (a))
M = (panel.with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
          .group_by(["exporter", "hs6", "market", "period", "group_v4"]).agg((pl.col("value_eur").sum() / 1e6).alias("v"))
          .with_columns(cat(["exporter", "hs6", "market"], "f1"), cat(["exporter", "hs6", "period"], "f2"),
                        cat(["market", "period"], "f3"), cat(["group_v4", "market", pl.col("period").str.slice(4, 2)], "f4"),
                        cat(["exporter", "hs6"], "cl")))
cars, parts = pl.col("group_v4") == "1 CARS", pl.col("group_v4") == "2 PARTS_CORE"
us = pl.col("market") == "US"
rows.append(fit(M, {"CARS_HIGH": cars & us & inp(HIGH), "CARS_LOW": cars & us & inp(LOW),
                    "PARTS_HIGH": parts & us & inp(HIGH), "PARTS_LOW": parts & us & inp(LOW)},
                ["f1", "f2", "f3", "f4"], "(1) US vs other non-EU markets",
                s31[s31["clustering"].str.startswith("(a)")].set_index("term")))

# ---- columns 2 and 3: as step32
P = (panel.group_by(["exporter", "hs6", "dest", "period", "group_v4"]).agg((pl.col("value_eur").sum() / 1e6).alias("v"))
          .with_columns(pl.col("period").str.slice(4, 2).alias("cm")))
std = lambda: {"CARS_HIGH": cars & inp(HIGH), "CARS_LOW": cars & inp(LOW),
               "PARTS_HIGH": parts & inp(HIGH), "PARTS_LOW": parts & inp(LOW)}
U = (P.filter(pl.col("dest") == "US")
      .with_columns(cat(["exporter", "hs6"], "f_flow"), cat(["exporter", "period"], "f_em"),
                    cat(["group_v4", "cm"], "f_season"), cat(["exporter", "hs6"], "cl")))
rows.append(fit(U, std(), ["f_flow", "f_em", "f_season"], "(2) US only",
                s32[s32["model"].str.startswith("3.")].set_index("term")))
O = (P.filter(pl.col("dest") != "US")
      .with_columns(cat(["exporter", "hs6", "dest"], "f_flow"), cat(["dest", "period"], "f_dm"),
                    cat(["exporter", "period"], "f_em"), cat(["group_v4", "dest", "cm"], "f_season"),
                    cat(["exporter", "hs6"], "cl")))
rows.append(fit(O, std(), ["f_flow", "f_dm", "f_em", "f_season"], "(3) Other non-EU markets",
                s32[s32["model"].str.startswith("2a")].set_index("term")))

# ---- column 4: intra-EU, as step33 Part 1
X = (pl.read_parquet(DATA / "exports_ch84_87.parquet", columns=["exporter", "cn8", "dest", "period", "value_eur"])
       .with_columns(pl.col("cn8").str.slice(0, 6).alias("hs6"))
       .join(G.select(["hs6", "group_v4"]), on="hs6", how="left").filter(pl.col("group_v4").is_in(KEEP)))
E = (X.filter(pl.col("dest") == "EU")
       .group_by(["exporter", "hs6", "group_v4", "period"]).agg((pl.col("value_eur").sum() / 1e6).alias("v")))
flows = (E.filter(pl.col("period") < "202501").group_by(["exporter", "hs6", "group_v4"])
          .agg(pl.col("v").sum().alias("pre")).filter(pl.col("pre") > 0).drop("pre"))
months = E.select("period").unique().sort("period")
I = (flows.join(months, how="cross")
          .join(E.drop("group_v4"), on=["exporter", "hs6", "period"], how="left")
          .with_columns(pl.col("v").fill_null(0.0))
          .filter(~pl.col("period").is_between(pl.lit("202501"), pl.lit("202503")))
          .with_columns(pl.col("period").str.slice(4, 2).alias("cm"))
          .with_columns(cat(["exporter", "hs6"], "f_flow"), cat(["exporter", "period"], "f_em"),
                        cat(["group_v4", "cm"], "f_season"), cat(["exporter", "hs6"], "cl")))
rows.append(fit(I, std(), ["f_flow", "f_em", "f_season"], "(4) Intra-EU",
                s33[s33["model"].str.startswith("1.")].set_index("term")))
pd.DataFrame(rows).to_csv(OUT / "step39_table1_counts.csv", index=False)

# ---- Part B: kg per car
PER = [("202504", "202507", "AprJul25"), ("202508", "202509", "AugSep25"), ("202510", "202512", "OctDec25"),
       ("202601", "202603", "JanMar26"), ("202604", "202606", "AprJun26")]
R = pl.read_parquet(DATA / "panel_v1.parquet",
                    columns=["exporter", "cn8", "dest", "us", "period", "group_v4", "value_eur", "qty_units", "qty_kg",
                             "frontload"])
C = (R.filter(~pl.col("frontload") & (pl.col("group_v4") == "1 CARS") & (pl.col("dest") != "CN"))
       .with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
       .group_by(["exporter", "cn8", "market", "period", "group_v4"])
       .agg(pl.col("value_eur").sum(), pl.col("qty_units").sum(), pl.col("qty_kg").sum())
       .filter((pl.col("value_eur") > 0) & (pl.col("qty_units") >= 5))          # same price sample as step37 E
       .with_columns((pl.col("value_eur") / pl.col("qty_units")).log().alias("lp"),
                     pl.concat_str(["exporter", "cn8", "market"], separator="_").alias("flow")))
base = (C.filter(pl.col("period") < "202501").group_by("flow")
         .agg(pl.col("lp").median().alias("lp_med"), pl.col("value_eur").sum().alias("w")))
C = C.join(base, on="flow", how="inner").filter((pl.col("lp") - pl.col("lp_med")).abs() <= np.log(5))
n_zero = C.filter(pl.col("qty_kg") <= 0).height
C = (C.filter(pl.col("qty_kg") > 0)                                            # the fix
       .with_columns((pl.col("qty_kg") / pl.col("qty_units")).log().alias("lkg"))
       .with_columns(cat(["flow"], "f1"), cat(["exporter", "cn8", "period"], "f2"),
                     cat(["market", pl.col("period").str.slice(4, 2)], "f4"),
                     cat(["exporter", "cn8"], "cl"), (pl.col("w") / pl.col("w").mean()).alias("wt")))
terms = {f"CARS_{per}": ((pl.col("market") == "US") & pl.col("period").is_between(pl.lit(a), pl.lit(b))).cast(pl.Float64)
         for a, b, per in PER}
C = C.with_columns([e.alias(n) for n, e in terms.items()])
print(f"\nPart B. Log kg per car. Flow-months dropped because kg = 0: {n_zero:,}.")
out = []
for wlab, w in [("weighted", True), ("unweighted", False)]:
    df = C.select(["lkg", "wt", "cl", "f1", "f2", "f4"] + list(terms)).to_pandas()
    m = pf.feols(f"lkg ~ {' + '.join(terms)} | f1 + f2 + f4", data=df, vcov={"CRV1": "cl"},
                 **({"weights": "wt"} if w else {}))
    T = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
    T["spec"], T["nobs"] = f"log kg per car, {wlab}", m._N
    out.append(T)
    print(f"  {wlab} ({m._N:,} obs.): % change in kg per car, US relative to other markets")
    for t in terms:
        r = T.set_index("term").loc[t]
        print(f"    {t:14s} {pct(r['Estimate']):+5.1f}%  [{pct(r['2.5%']):+.1f}, {pct(r['97.5%']):+.1f}]  p = {r['Pr(>|t|)']:.3f}")
pd.concat(out).to_csv(OUT / "step39_kg_per_car.csv", index=False)
print("\nSaved 5_results/step39_table1_counts.csv and 5_results/step39_kg_per_car.csv")
