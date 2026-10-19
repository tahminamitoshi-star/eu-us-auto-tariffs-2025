"""Step 3.14: Did EU exporters sell the cars inside the EU instead? (decision register D23)

Step 3.13: intra-EU car trade rose by EUR 1.46 bn a month in Apr-Sep 2025 (+7.4% value, +3.1% cars), more than
the EUR 1.1 bn lost in the US. The question is whether any of this is redirected US volume.

Data: exports_ch84_87.parquet, destination "EU" (each member state's exports to the other member states).
Unit: exporter x HS6 x month, flows with intra-EU exports in 2023-2024, months without trade set to 0.
Jan-Mar 2025 left out, as in the main model.

Part 1. Cars and core parts against control products in intra-EU trade (PPML).
        FE: flow (exporter x HS6), exporter x month, group x calendar month (seasonality).
        Clustered by exporter x HS6. Pooled: HIGH (Apr-Sep 2025), LOW (Oct 2025 - Jun 2026).
Part 2. Pre-trend: the same model with one car term per quarter (2023 Q1 - 2026 Q2), 2024 as the reference
        year. If cars were already growing faster than control products inside the EU in 2023, the 2025 rise
        is an old trend, not redirection.
Part 3. Exposure test: redirection should be strongest where the US mattered most. For each exporter x HS6 car
        product, US share = 2024 exports to the US / 2024 exports to all destinations (including intra-EU).
        The model adds CARS_HIGH x US share and CARS_LOW x US share. A positive interaction means products that
        depended more on the US gained more inside the EU, as redirection predicts. Zero means the intra-EU rise
        is unrelated to US exposure.
Saves 5_results/step33_intra_eu.csv
"""
import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf

warnings.filterwarnings("ignore")
BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
OUT = BASE / "5_results"
G = pl.read_csv(BASE / "7_processed_data" / "hs6_groups_v4.csv", schema_overrides={"hs6": pl.Utf8})
HIGH, LOW = ("202504", "202509"), ("202510", "202606")
inp = lambda p: pl.col("period").is_between(pl.lit(p[0]), pl.lit(p[1]))
cat = lambda cols, name: pl.concat_str(cols, separator="_").cast(pl.Categorical).to_physical().alias(name)

X = (pl.read_parquet(BASE / "7_processed_data" / "exports_ch84_87.parquet",
                     columns=["exporter", "cn8", "dest", "period", "value_eur"])
       .with_columns(pl.col("cn8").str.slice(0, 6).alias("hs6"))
       .join(G.select(["hs6", "group_v4"]), on="hs6", how="left")
       .filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", "4 CONTROL"])))

# US share of each exporter x HS6 in 2024 (all destinations, including intra-EU)
sh = (X.filter(pl.col("period").str.starts_with("2024"))
        .group_by(["exporter", "hs6"])
        .agg((pl.col("value_eur").filter(pl.col("dest") == "US").sum() / pl.col("value_eur").sum()).alias("us_share")))

E = (X.filter(pl.col("dest") == "EU")
       .group_by(["exporter", "hs6", "group_v4", "period"]).agg((pl.col("value_eur").sum() / 1e6).alias("v")))
flows = (E.filter(pl.col("period") < "202501").group_by(["exporter", "hs6", "group_v4"])
          .agg(pl.col("v").sum().alias("pre")).filter(pl.col("pre") > 0).drop("pre"))
months = E.select("period").unique().sort("period")
P = (flows.join(months, how="cross")
          .join(E.drop("group_v4"), on=["exporter", "hs6", "period"], how="left")
          .with_columns(pl.col("v").fill_null(0.0))
          .filter(~pl.col("period").is_between(pl.lit("202501"), pl.lit("202503")))
          .join(sh, on=["exporter", "hs6"], how="left").with_columns(pl.col("us_share").fill_null(0.0))
          .with_columns(pl.col("period").str.slice(4, 2).alias("cm"),
                        (pl.col("period").str.slice(0, 4) + "Q"
                         + ((pl.col("period").str.slice(4, 2).cast(pl.Int32) - 1) // 3 + 1).cast(pl.Utf8)).alias("q"))
          .with_columns(cat(["exporter", "hs6"], "f_flow"), cat(["exporter", "period"], "f_em"),
                        cat(["group_v4", "cm"], "f_season"), cat(["exporter", "hs6"], "cl")))
cars, parts = pl.col("group_v4") == "1 CARS", pl.col("group_v4") == "2 PARTS_CORE"
print(f"Intra-EU panel: {P.height:,} rows, {flows.height:,} flows; 2024 intra-EU value by group (EUR bn):")
print(P.filter(pl.col("period").str.starts_with("2024")).group_by("group_v4")
       .agg((pl.col("v").sum() / 1e3).round(1).alias("bn_2024")).sort("group_v4"))
c24 = P.filter(cars & pl.col("period").str.starts_with("2024"))
print(f"US share of car products (2024, weighted by intra-EU value): "
      f"{100 * (c24['us_share'] * c24['v']).sum() / c24['v'].sum():.1f}%; "
      f"25th / 50th / 75th percentile across car flows: "
      + " / ".join(f"{100 * c24.group_by(['exporter', 'hs6']).agg(pl.col('us_share').first())['us_share'].quantile(q):.1f}%"
                   for q in [0.25, 0.5, 0.75]))
FE = "f_flow + f_em + f_season"
out = []


def fit(Q, terms, label, show=None):
    Q = Q.with_columns([c.cast(pl.Float64).alias(n) if not isinstance(c, str) else pl.col(c).alias(n)
                        for n, c in terms.items()])
    names = [n for n in terms if Q[n].abs().sum() > 0]
    df = Q.select(["v", "cl", "f_flow", "f_em", "f_season"] + names).to_pandas()
    m = pf.fepois(f"v ~ {' + '.join(names)} | {FE}", data=df, vcov={"CRV1": "cl"})
    T = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
    T["model"] = label
    print(f"\n{label} ({m._N:,} obs.)")
    for _, r in T.iterrows():
        if show and not r["term"].startswith(show):
            continue
        if "_x_" in r["term"]:
            print(f"  {r['term']:22s} coef {r['Estimate']:+.3f} (s.e. {r['Std. Error']:.3f})  p = {r['Pr(>|t|)']:.3f}")
        else:
            print(f"  {r['term']:22s} {100 * (np.exp(r['Estimate']) - 1):+6.1f}%  "
                  f"[{100 * (np.exp(r['2.5%']) - 1):+.0f}, {100 * (np.exp(r['97.5%']) - 1):+.0f}]  p = {r['Pr(>|t|)']:.3f}")
    return T


print("\nPart 1. Intra-EU exports: cars and core parts against control products")
out.append(fit(P, {"CARS_HIGH": cars & inp(HIGH), "CARS_LOW": cars & inp(LOW),
                   "PARTS_HIGH": parts & inp(HIGH), "PARTS_LOW": parts & inp(LOW)}, "1. Pooled"))

qs = [q for q in sorted(P["q"].unique().to_list()) if not q.startswith("2024") and q != "2025Q1"]
terms = {f"CARS_{q}": cars & (pl.col("q") == q) for q in qs}
terms.update({"PARTS_HIGH": parts & inp(HIGH), "PARTS_LOW": parts & inp(LOW)})
print("\nPart 2. Pre-trend: car effect by quarter, 2024 = reference (2025 Q1 left out)")
out.append(fit(P, terms, "2. Quarters", show="CARS_"))

P3 = P.with_columns((cars & inp(HIGH)).cast(pl.Float64).alias("_h"), (cars & inp(LOW)).cast(pl.Float64).alias("_l"))
P3 = P3.with_columns((pl.col("_h") * pl.col("us_share")).alias("CARS_HIGH_x_USSHARE"),
                     (pl.col("_l") * pl.col("us_share")).alias("CARS_LOW_x_USSHARE"))
print("\nPart 3. Exposure: does the intra-EU car change rise with the product's 2024 US share?")
print("        CARS_HIGH / CARS_LOW = effect at US share 0. Interaction = change in log exports per 1.0 of US share")
print("        (for a product with a 20% US share, multiply the coefficient by 0.2).")
out.append(fit(P3, {"CARS_HIGH": "_h", "CARS_LOW": "_l", "CARS_HIGH_x_USSHARE": "CARS_HIGH_x_USSHARE",
                    "CARS_LOW_x_USSHARE": "CARS_LOW_x_USSHARE",
                    "PARTS_HIGH": parts & inp(HIGH), "PARTS_LOW": parts & inp(LOW)}, "3. Exposure", show="CARS"))
pd.concat(out, ignore_index=True).to_csv(OUT / "step33_intra_eu.csv", index=False)
