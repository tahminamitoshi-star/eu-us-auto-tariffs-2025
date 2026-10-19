"""Step 3.4: Checks on the event study (step22 showed a pre-trend for cars in 2023).

Three versions, each for cars and core parts (same fixed effects and clustering as step21/22):
  A  Reference = Oct-Dec 2024 (not all of 2024), so the 2024 months are shown too.
     Question: did the 2023 rise in car exports to the US (relative to other markets) continue in 2024,
     or had it levelled off before the tariff?
  B  As A, but China is left out of OTHER.
     Question: is the 2023 pattern driven by falling EU car sales in China?
  C  Seasonal version: dummies only for Jan 2025 - Jun 2026, reference = all of 2023-2024, plus
     group x US x calendar-month fixed effects (each group's normal US-vs-other pattern in Jan, Feb, ...).
     Question: are the 2025 effects still there when each month is compared with the same month before?
Prints for each version: joint test that all pre-tariff months are zero (A, B), mean pre and mean Apr-Jul 2025
effect, and the monthly table. Saves 5_results/step23_<A|B|C>.csv and .png
"""
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
OUT = BASE / "5_results"
OUT.mkdir(exist_ok=True)
RAW = pl.read_parquet(BASE / "7_processed_data" / "panel_v1.parquet",
                      columns=["exporter", "hs6", "dest", "us", "period", "group_v4", "value_eur"])
RAW = RAW.filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", "4 CONTROL"]))
GROUPS = [("1 CARS", "CARS"), ("2 PARTS_CORE", "PARTS")]


def build(drop_china):
    d = RAW.filter(pl.col("dest") != "CN") if drop_china else RAW
    return (d.with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
             .group_by(["exporter", "hs6", "market", "period", "group_v4"])
             .agg((pl.col("value_eur").sum() / 1e6).alias("v")))


def run(name, drop_china, dummy_months, seasonal):
    P = build(drop_china)
    cols = []
    for g, short in GROUPS:
        for m in dummy_months:
            c = f"{short}_{m}"
            P = P.with_columns(((pl.col("group_v4") == g) & (pl.col("market") == "US") & (pl.col("period") == m))
                               .cast(pl.Float64).alias(c))
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
    T["group"] = T["term"].str.split("_").str[0]
    T["month"] = T["term"].str.split("_").str[1]
    for a, b in [("Estimate", "pct"), ("2.5%", "pct_low"), ("97.5%", "pct_high")]:
        T[b] = 100 * (np.exp(T[a]) - 1)
    T.to_csv(OUT / f"step23_{name}.csv", index=False)
    print(f"\n=== Version {name} ({time.time() - start:.0f} s, {fit._N:,} obs) ===")
    for _, short in GROUPS:
        t = T[T["group"] == short]
        pre = [m for m in dummy_months if m < "202501"]
        if pre:
            R = np.zeros((len(pre), len(cols)))
            for i, m in enumerate(pre):
                R[i, cols.index(f"{short}_{m}")] = 1.0
            w = fit.wald_test(R=R)
            pre_mean = t[t["month"].isin(pre)]["Estimate"].mean()
            print(f"{short}: joint test that all pre-tariff months are zero: p = {float(w['pvalue']):.3f}; "
                  f"mean pre-tariff effect {100 * (np.exp(pre_mean) - 1):+.1f}%")
        post = t[t["month"].isin(["202504", "202505", "202506", "202507"])]["Estimate"].mean()
        print(f"{short}: mean effect Apr-Jul 2025 {100 * (np.exp(post) - 1):+.1f}%")
    print(T.pivot(index="month", columns="group", values="pct").round(1).to_string())
    plot(T, dummy_months, name)


def plot(T, months, name):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    xpos = {m: i for i, m in enumerate(months)}
    for ax, (short, title) in zip(axes, [("CARS", "Cars"), ("PARTS", "Core car parts")]):
        d = T[T["group"] == short].sort_values("month")
        ax.axhline(0, color="#6b6b6b", linewidth=1)
        if "202501" in xpos:
            ax.axvspan(xpos["202501"] - 0.5, xpos["202503"] + 0.5, color="#e6e6e6", zorder=0)
        for part in [d[d["month"] < "202501"], d[d["month"] >= "202501"]]:
            if len(part):
                xp = [xpos[m] for m in part["month"]]
                ax.fill_between(xp, part["pct_low"], part["pct_high"], color="#2a5d9f", alpha=0.18, linewidth=0)
                ax.plot(xp, part["pct"], color="#2a5d9f", linewidth=2, marker="o", markersize=3.5)
        for m in ["202504", "202508", "202602"]:
            if m in xpos:
                ax.axvline(xpos[m] - 0.5, color="#9a9a9a", linewidth=1, linestyle="--")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("%")
        ax.grid(axis="y", color="#ececec")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[1].set_xticks(range(len(months)))
    axes[1].set_xticklabels([f"{m[4:]}/{m[2:4]}" for m in months], rotation=90, fontsize=6)
    notes = {"A": "Reference: Oct-Dec 2024.", "B": "Reference: Oct-Dec 2024. China left out of other markets.",
             "C": "Reference: 2023-2024; group x US x calendar-month fixed effects."}
    fig.text(0.01, 0.005, notes[name] + " Dashed lines: 3 Apr 2025, 1 Aug 2025, 24 Feb 2026. Grey: Jan-Mar 2025. "
             "PPML, 95% intervals, clustered by exporter x HS6.", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / f"step23_{name}.png", dpi=200)
    plt.close(fig)


all_months = sorted(RAW["period"].unique().to_list())
ref_q4 = ["202410", "202411", "202412"]
run("A", False, [m for m in all_months if m not in ref_q4], False)
run("B", True, [m for m in all_months if m not in ref_q4], False)
run("C", False, [m for m in all_months if m >= "202501"], True)
print(f"\nFigures saved in {OUT}")
