"""Step 3.3: Monthly event study. Same model as step21, but one coefficient per month instead of per period.

Why: before reading step21 as a tariff effect, the months BEFORE the tariff must show no gap. If cars to the US
were already falling relative to other markets in 2023, the 2025 estimate would mix the tariff with an old trend.

Model (PPML), exporter x HS6 x market (US / OTHER) x month, groups CARS, PARTS_CORE and the clean CONTROL:
  one dummy per month for 1[group g] x 1[US] x 1[month], for every month of 2023 and of Jan 2025 - Jun 2026.
  Reference period: the 12 months of 2024 (no dummies), so every coefficient is measured against 2024.
  Jan-Mar 2025 are kept here (and shaded in the figure) to show the front-loading.
Same fixed effects and clustering as step21.
Saves 5_results/step22_event_study.csv and 5_results/step22_event_study.png
"""
import os
from pathlib import Path
import time
import numpy as np
import polars as pl
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
OUT = BASE / "5_results"
OUT.mkdir(exist_ok=True)

P = (pl.scan_parquet(BASE / "7_processed_data" / "panel_v1.parquet")
       .filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", "4 CONTROL"]))
       .with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
       .group_by(["exporter", "hs6", "market", "period", "group_v4"])
       .agg((pl.col("value_eur").sum() / 1e6).alias("v"))
       .collect())

months = sorted(m for m in P["period"].unique().to_list() if not m.startswith("2024"))
cols = []
for g, short in [("1 CARS", "CARS"), ("2 PARTS_CORE", "PARTS")]:
    for m in months:
        c = f"{short}_{m}"
        P = P.with_columns(((pl.col("group_v4") == g) & (pl.col("market") == "US") & (pl.col("period") == m))
                           .cast(pl.Float64).alias(c))
        cols.append(c)
P = P.with_columns(
    pl.concat_str(["exporter", "hs6", "market"], separator="_").cast(pl.Categorical).to_physical().alias("fe_flow"),
    pl.concat_str(["exporter", "hs6", "period"], separator="_").cast(pl.Categorical).to_physical().alias("fe_prod_month"),
    pl.concat_str(["market", "period"], separator="_").cast(pl.Categorical).to_physical().alias("fe_mkt_month"),
    pl.concat_str(["exporter", "hs6"], separator="_").cast(pl.Categorical).to_physical().alias("cluster"))
df = P.select(["v", "fe_flow", "fe_prod_month", "fe_mkt_month", "cluster"] + cols).to_pandas()

start = time.time()
fit = pf.fepois(f"v ~ {' + '.join(cols)} | fe_flow + fe_prod_month + fe_mkt_month",
                data=df, vcov={"CRV1": "cluster"})
print(f"PPML done in {time.time() - start:.0f} s; observations used: {fit._N:,}")

T = fit.tidy().reset_index().rename(columns={"Coefficient": "term"})
T["group"] = T["term"].str.split("_").str[0]
T["month"] = T["term"].str.split("_").str[1]
for a, b in [("Estimate", "pct"), ("2.5%", "pct_low"), ("97.5%", "pct_high")]:
    T[b] = 100 * (np.exp(T[a]) - 1)
T.to_csv(OUT / "step22_event_study.csv", index=False)

# Pre-trend test: are the 12 monthly coefficients of 2023 jointly zero?
for short in ["CARS", "PARTS"]:
    pre = [f"{short}_{m}" for m in months if m.startswith("2023")]
    R = np.zeros((len(pre), len(cols)))
    for i, c in enumerate(pre):
        R[i, cols.index(c)] = 1.0
    w = fit.wald_test(R=R)
    print(f"{short}: joint test that all 2023 months are zero: statistic {float(w['statistic']):.2f}, "
          f"p = {float(w['pvalue']):.3f}")

print("\nMonthly effects (% change of exports to the US vs other markets, relative to the control, vs 2024):")
show = T.pivot(index="month", columns="group", values="pct").round(1)
print(show.to_string())

# Figure: one panel per group, coefficient and 95% band, tariff dates marked.
fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
x_labels = months
xpos = {m: i for i, m in enumerate(x_labels)}
events = [("202504", "car tariff 3 Apr"), ("202508", "EU 15% from 1 Aug"), ("202602", "Section 122, 24 Feb")]
for ax, (short, title) in zip(axes, [("CARS", "Cars"), ("PARTS", "Core car parts")]):
    d = T[T["group"] == short].sort_values("month")
    x = [xpos[m] for m in d["month"]]
    ax.axhline(0, color="#6b6b6b", linewidth=1)
    ax.axvspan(xpos["202501"] - 0.5, xpos["202503"] + 0.5, color="#e6e6e6", zorder=0)
    for part in [d[d["month"] < "202401"], d[d["month"] > "202412"]]:   # no line across the omitted 2024
        xp = [xpos[m] for m in part["month"]]
        ax.fill_between(xp, part["pct_low"], part["pct_high"], color="#2a5d9f", alpha=0.18, linewidth=0)
        ax.plot(xp, part["pct"], color="#2a5d9f", linewidth=2, marker="o", markersize=4)
    for m, lab in events:
        ax.axvline(xpos[m] - 0.5, color="#9a9a9a", linewidth=1, linestyle="--")
    if short == "PARTS":   # the parts tariff started a month later than the car tariff
        ax.axvline(xpos["202505"] - 0.5, color="#9a9a9a", linewidth=1, linestyle=":")
        ax.text(xpos["202505"] - 0.3, ax.get_ylim()[1], "parts tariff 3 May", fontsize=7, va="top", color="#555555")
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel("% vs 2024")
    ax.grid(axis="y", color="#ececec")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
top = axes[0].get_ylim()[1]
axes[0].text(xpos["202501"] - 0.4, top, "front-\nloading", fontsize=7, va="top", color="#555555")
for m, lab in events:
    axes[0].text(xpos[m] - 0.3, top, lab, fontsize=7, va="top", ha="left", color="#555555")
axes[1].set_xticks(range(len(x_labels)))
axes[1].set_xticklabels([f"{m[4:]}/{m[2:4]}" for m in x_labels], rotation=90, fontsize=7)
fig.text(0.01, 0.005, "2024 is the reference period (not shown); grey band: Jan-Mar 2025. PPML; 95% intervals; "
         "clustered by exporter x HS6.", fontsize=7, color="#555555")
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(OUT / "step22_event_study.png", dpi=200)
print(f"\nFigure saved: {OUT / 'step22_event_study.png'}")
