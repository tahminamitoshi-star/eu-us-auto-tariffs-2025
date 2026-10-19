"""Step 3.10: Premium against mass-market cars (decision register D21).

Step 3.8-3.9: German car exports to the US fell half as much as those of the rest of the EU, with the same price cut.
Hypothesis: US demand for expensive (premium) cars reacts less to price, and German exports are mostly premium.
Test: split all EU car flows by their price per car before the tariff, not by country.

Price per car: for each exporter x HS6 car product, EUR per vehicle over 2023-2024 across all non-EU destinations
(value / number of vehicles). Defined before the tariff and the same for the US and other markets.
Part 1: 2024 EU car exports to the US by price band and by Germany / rest.
Part 2: main specification (step25) with the car effect split into PREMIUM (price per car >= threshold) and MASS,
for HIGH (Apr-Sep 2025) and LOW (Oct 2025 - Jun 2026); thresholds EUR 40,000, 50,000 (main) and 60,000.
Prints % effects and the test PREMIUM = MASS. Saves 5_results/table_premium.csv
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
                      columns=["exporter", "hs6", "dest", "us", "period", "group_v4", "value_eur", "qty_units",
                               "frontload"])
RAW = RAW.filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", "4 CONTROL"]))

price = (RAW.filter((pl.col("group_v4") == "1 CARS") & (pl.col("period") < "202501"))
            .group_by(["exporter", "hs6"])
            .agg(pl.col("value_eur").sum().alias("v"), pl.col("qty_units").sum().alias("n"))
            .filter(pl.col("n") >= 50)
            .with_columns((pl.col("v") / pl.col("n")).alias("eur_per_car"))
            .select(["exporter", "hs6", "eur_per_car"]))

us24 = (RAW.filter((pl.col("group_v4") == "1 CARS") & pl.col("us") & pl.col("period").str.starts_with("2024"))
           .group_by(["exporter", "hs6"]).agg((pl.col("value_eur").sum() / 1e9).alias("bn"))
           .join(price, on=["exporter", "hs6"], how="left")
           .with_columns(pl.col("eur_per_car").cut([30000, 50000, 80000], labels=["<30k", "30-50k", "50-80k", ">=80k"]).alias("band"),
                         pl.when(pl.col("exporter") == "DE").then(pl.lit("DE")).otherwise(pl.lit("REST")).alias("who")))
print("2024 EU car exports to the US (EUR bn) by price per car (2023-24, all non-EU markets) and exporter:")
print(us24.group_by(["band", "who"]).agg(pl.col("bn").sum().round(2)).pivot(on="who", index="band", values="bn")
          .sort("band"))
print(f"Car exports to the US without a price (fewer than 50 vehicles in 2023-24): "
      f"EUR {us24.filter(pl.col('eur_per_car').is_null())['bn'].sum():.2f} bn")

d = (RAW.filter((pl.col("dest") != "CN") & ~pl.col("frontload"))
        .with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
        .group_by(["exporter", "hs6", "market", "period", "group_v4"])
        .agg((pl.col("value_eur").sum() / 1e6).alias("v"))
        .join(price, on=["exporter", "hs6"], how="left"))
d = d.with_columns(
    pl.concat_str(["exporter", "hs6", "market"], separator="_").cast(pl.Categorical).to_physical().alias("fe_flow"),
    pl.concat_str(["exporter", "hs6", "period"], separator="_").cast(pl.Categorical).to_physical().alias("fe_prod_month"),
    pl.concat_str(["market", "period"], separator="_").cast(pl.Categorical).to_physical().alias("fe_mkt_month"),
    pl.concat_str([pl.col("group_v4"), pl.col("market"), pl.col("period").str.slice(4, 2)], separator="_")
      .cast(pl.Categorical).to_physical().alias("fe_season"),
    pl.concat_str(["exporter", "hs6"], separator="_").cast(pl.Categorical).to_physical().alias("cluster"))
periods = [("202504", "202509", "HIGH"), ("202510", "202606", "LOW")]
rows = []
for thr in [40000, 50000, 60000]:
    P, cols = d, []
    seg = {"PREM": pl.col("eur_per_car") >= thr, "MASS": pl.col("eur_per_car") < thr}
    for a, b, per in periods:
        inper = pl.col("period").is_between(pl.lit(a), pl.lit(b)) & (pl.col("market") == "US")
        for s, cond in seg.items():
            c = f"CARS{s}_{per}"
            P = P.with_columns(((pl.col("group_v4") == "1 CARS") & cond.fill_null(False) & inper)
                               .cast(pl.Float64).alias(c))
            cols.append(c)
        c = f"CARSNOPRICE_{per}"
        P = P.with_columns(((pl.col("group_v4") == "1 CARS") & pl.col("eur_per_car").is_null() & inper)
                           .cast(pl.Float64).alias(c))
        cols.append(c)
        c = f"PARTS_{per}"
        P = P.with_columns(((pl.col("group_v4") == "2 PARTS_CORE") & inper).cast(pl.Float64).alias(c))
        cols.append(c)
    df = P.select(["v", "cluster", "fe_flow", "fe_prod_month", "fe_mkt_month", "fe_season"] + cols).to_pandas()
    cols = [c for c in cols if df[c].sum() > 0]          # drop empty indicators (e.g. no cars without a price)
    fit = pf.fepois(f"v ~ {' + '.join(cols)} | fe_flow + fe_prod_month + fe_mkt_month + fe_season",
                    data=df, vcov={"CRV1": "cluster"})
    T = fit.tidy().reset_index().rename(columns={"Coefficient": "term"})
    T["threshold"] = thr
    rows.append(T)
    print(f"\nThreshold EUR {thr:,} per car ({fit._N:,} obs). Effect on car exports to the US, %:")
    for per in ["HIGH", "LOW"]:
        if not {f"CARSPREM_{per}", f"CARSMASS_{per}"} <= set(T["term"]):
            print(f"  {per}: one segment has no car flows at this threshold; skipped")
            continue
        out = []
        for s in ["PREM", "MASS"]:
            r = T[T["term"] == f"CARS{s}_{per}"].iloc[0]
            out.append(f"{s} {100 * (np.exp(r['Estimate']) - 1):+.1f} "
                       f"[{100 * (np.exp(r['2.5%']) - 1):+.0f}, {100 * (np.exp(r['97.5%']) - 1):+.0f}]")
        names = list(T["term"])
        R = np.zeros((1, len(names)))
        R[0, names.index(f"CARSPREM_{per}")], R[0, names.index(f"CARSMASS_{per}")] = 1.0, -1.0
        w = fit.wald_test(R=R)
        print(f"  {per}: " + " | ".join(out) + f" | PREMIUM = MASS: p = {float(w['pvalue']):.3f}")
R = pd.concat(rows, ignore_index=True)
for a, b in [("Estimate", "pct"), ("2.5%", "pct_low"), ("97.5%", "pct_high")]:
    R[b] = 100 * (np.exp(R[a]) - 1)
R.to_csv(OUT / "table_premium.csv", index=False)
