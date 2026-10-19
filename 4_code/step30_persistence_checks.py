"""Step 3.11: Three checks on the open questions D20 (sales stay down) and D21 (Germany fell less).

A  US demand shock for electric cars. The US federal clean-vehicle tax credits ended for vehicles acquired after
   30 Sep 2025 (One Big Beautiful Bill Act). That hits US demand for electric and plug-in hybrid cars in my LOW period
   for reasons unrelated to the tariff. Re-estimate the main model without 870360, 870370 (plug-in hybrids) and
   870380 (electric). If the LOW-period car effect shrinks a lot, part of the persistence is the end of the credit.
B  Exits. For car flows (exporter x HS6) that existed in 2024: how many stopped completely in Oct 2025 - Jun 2026,
   to the US and to other markets, and how much of the fall in value comes from flows that stopped.
C  Which countries drive the rest of the EU? Main model with the car effect split by exporter:
   DE, SE, IT, SK, HU, BE, AT, and the remaining countries together, for HIGH (Apr-Sep 2025) and LOW (Oct 2025 -
   Jun 2026).
Main specification as in step25 (China out of OTHER, seasonal effects, Jan-Mar 2025 out, clustered by exporter x HS6).
Saves 5_results/step30_checks.csv
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
PLUGIN = ["870360", "870370", "870380"]
HIGH, LOW = ("202504", "202509"), ("202510", "202606")


def panel(data):
    return (data.filter((pl.col("dest") != "CN") & ~pl.col("frontload"))
                .with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
                .group_by(["exporter", "hs6", "market", "period", "group_v4"])
                .agg((pl.col("value_eur").sum() / 1e6).alias("v")))


def fit(P, terms):
    """terms: dict name -> polars boolean expression (already including market == US and the period)."""
    for name, cond in terms.items():
        P = P.with_columns(cond.cast(pl.Float64).alias(name))
    P = P.with_columns(
        pl.concat_str(["exporter", "hs6", "market"], separator="_").cast(pl.Categorical).to_physical().alias("f1"),
        pl.concat_str(["exporter", "hs6", "period"], separator="_").cast(pl.Categorical).to_physical().alias("f2"),
        pl.concat_str(["market", "period"], separator="_").cast(pl.Categorical).to_physical().alias("f3"),
        pl.concat_str([pl.col("group_v4"), pl.col("market"), pl.col("period").str.slice(4, 2)], separator="_")
          .cast(pl.Categorical).to_physical().alias("f4"),
        pl.concat_str(["exporter", "hs6"], separator="_").cast(pl.Categorical).to_physical().alias("cl"))
    names = [n for n in terms if P[n].sum() > 0]
    df = P.select(["v", "cl", "f1", "f2", "f3", "f4"] + names).to_pandas()
    m = pf.fepois(f"v ~ {' + '.join(names)} | f1 + f2 + f3 + f4", data=df, vcov={"CRV1": "cl"})
    T = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
    for a, b in [("Estimate", "pct"), ("2.5%", "lo"), ("97.5%", "hi")]:
        T[b] = 100 * (np.exp(T[a]) - 1)
    return T


def show(T, label):
    print(f"\n{label}")
    for _, r in T.iterrows():
        print(f"  {r['term']:16s} {r['pct']:+6.1f}%  [{r['lo']:+.0f}, {r['hi']:+.0f}]  p = {r['Pr(>|t|)']:.3f}")


us, cars, parts = pl.col("market") == "US", pl.col("group_v4") == "1 CARS", pl.col("group_v4") == "2 PARTS_CORE"
inp = lambda p: pl.col("period").is_between(pl.lit(p[0]), pl.lit(p[1]))
out = []

# ---- A: without plug-in and electric cars
sh = (RAW.filter(cars & pl.col("us") & pl.col("period").str.starts_with("2024"))
         .group_by(pl.col("hs6").is_in(PLUGIN).alias("plugin")).agg(pl.col("value_eur").sum()))
tot = sh["value_eur"].sum()
print("A. Share of 2024 EU car exports to the US in plug-in hybrid and electric codes: "
      f"{100 * sh.filter(pl.col('plugin'))['value_eur'].sum() / tot:.1f}%")
base_terms = {"CARS_HIGH": cars & us & inp(HIGH), "CARS_LOW": cars & us & inp(LOW),
              "PARTS_HIGH": parts & us & inp(HIGH), "PARTS_LOW": parts & us & inp(LOW)}
for label, data in [("A1. All cars (main model, two periods)", RAW),
                    ("A2. Without plug-in hybrid and electric cars", RAW.filter(~pl.col("hs6").is_in(PLUGIN)))]:
    T = fit(panel(data), base_terms)
    T["check"] = label
    out.append(T)
    show(T[T["term"].str.startswith("CARS")], label)

# ---- B: exits of car flows
c = (RAW.filter(cars & (pl.col("dest") != "CN"))
        .with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"),
                      pl.when(pl.col("period").str.starts_with("2024")).then(pl.lit("y2024"))
                        .when(pl.col("period").is_between(pl.lit(LOW[0]), pl.lit(LOW[1]))).then(pl.lit("low"))
                        .otherwise(pl.lit(None)).alias("per"))
        .filter(pl.col("per").is_not_null())
        .group_by(["exporter", "hs6", "market", "per"]).agg(pl.col("value_eur").sum())
        .pivot(on="per", index=["exporter", "hs6", "market"], values="value_eur")
        .with_columns(pl.col(["y2024", "low"]).fill_null(0.0))
        .filter(pl.col("y2024") > 0)
        .with_columns((pl.col("y2024") / 12).alias("m24"), (pl.col("low") / 9).alias("mlow"),
                      (pl.col("low") == 0).alias("exit")))
print("\nB. Car flows (exporter x HS6) with exports in 2024: exits in Oct 2025 - Jun 2026 (no shipment at all)")
b = (c.group_by("market").agg(pl.len().alias("flows_2024"), pl.col("exit").sum().alias("exits"),
                              (100 * pl.col("m24").filter(pl.col("exit")).sum() / pl.col("m24").sum()).round(1)
                              .alias("pct_2024_value_in_exits"),
                              (100 * (pl.col("mlow").sum() / pl.col("m24").sum() - 1)).round(1).alias("pct_change_value"),
                              (100 * pl.col("m24").filter(pl.col("exit")).sum()
                               / (pl.col("m24").sum() - pl.col("mlow").sum())).round(1).alias("pct_of_fall_from_exits")))
print(b)
print("   pct_of_fall_from_exits = share of the fall in monthly value that comes from flows that stopped completely")

# ---- C: car effect by exporter
who = {"DE": ["DE"], "SE": ["SE"], "IT": ["IT"], "SK": ["SK"], "HU": ["HU"], "BE": ["BE"], "AT": ["AT"]}
terms = {}
for k, v in who.items():
    for p, nm in [(HIGH, "HIGH"), (LOW, "LOW")]:
        terms[f"CARS{k}_{nm}"] = cars & us & inp(p) & pl.col("exporter").is_in(v)
listed = sum(who.values(), [])
for p, nm in [(HIGH, "HIGH"), (LOW, "LOW")]:
    terms[f"CARSOTH_{nm}"] = cars & us & inp(p) & ~pl.col("exporter").is_in(listed)
    terms[f"PARTS_{nm}"] = parts & us & inp(p)
T = fit(panel(RAW), terms)
T["check"] = "C. by exporter"
out.append(T)
show(T[T["term"].str.startswith("CARS")], "C. Car effect by exporter (2024 share of EU car exports to the US in Step 3.8)")
pd.concat(out, ignore_index=True).to_csv(OUT / "step30_checks.csv", index=False)
