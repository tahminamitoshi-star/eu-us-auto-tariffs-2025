"""Step 5.2: Is the car fall in 2026 also electric and plug-in cars?

Figure 1 shows EU car exports to the US still 35-37% below 2024 in May-June 2026. Step 3.11 found that most of the
Oct 2025 - Jun 2026 effect comes from plug-in hybrid and electric cars (HS 870360, 870370, 870380), whose US demand
lost the federal clean-vehicle credits after 30 Sep 2025.

This script re-estimates the Figure 1 event study (main specification) with cars split into two groups:
  CARSICE = cars without the three plug-in and electric codes
  CARSEV  = plug-in hybrid and electric cars (870360, 870370, 870380)
Same fixed effects, 2024 = reference year, clustered by exporter x HS6. Prints both monthly series and the averages
for Apr-Sep 2025, Oct 2025 - Mar 2026 and Apr-Jun 2026, and saves a figure with both lines.
Saves 5_results/step35_event_study_no_ev.csv and 5_figures/fig1b_event_study_ice_ev.png
"""
import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
RES, FIG = BASE / "5_results", BASE / "5_figures"
EV = ["870360", "870370", "870380"]
pct = lambda x: 100 * (np.exp(x) - 1)

RAW = pl.read_parquet(BASE / "7_processed_data" / "panel_v1.parquet",
                      columns=["exporter", "hs6", "dest", "us", "period", "group_v4", "value_eur"])
P = (RAW.filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", "4 CONTROL"]) & (pl.col("dest") != "CN"))
        .with_columns(pl.when((pl.col("group_v4") == "1 CARS") & pl.col("hs6").is_in(EV)).then(pl.lit("1b CARS_EV"))
                        .otherwise(pl.col("group_v4")).alias("grp"),
                      pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
        .group_by(["exporter", "hs6", "market", "period", "grp"]).agg((pl.col("value_eur").sum() / 1e6).alias("v")))
sh = (RAW.filter(pl.col("us") & (pl.col("group_v4") == "1 CARS"))
         .with_columns(pl.col("hs6").is_in(EV).alias("ev"), pl.col("period").str.slice(0, 4).alias("y"))
         .group_by(["y", "ev"]).agg(pl.col("value_eur").sum()).pivot(on="ev", index="y", values="value_eur").sort("y"))
print("Share of plug-in and electric cars in EU car exports to the US, by year (%):")
for r in sh.iter_rows(named=True):
    print(f"  {r['y']}: {100 * r['true'] / (r['true'] + r['false']):.1f}%")

months = [m for m in sorted(P["period"].unique().to_list()) if not m.startswith("2024")]
cols = []
for g, short in [("1 CARS", "CARSICE"), ("1b CARS_EV", "CARSEV"), ("2 PARTS_CORE", "PARTS")]:
    for m in months:
        c = f"{short}_{m}"
        P = P.with_columns(((pl.col("grp") == g) & (pl.col("market") == "US") & (pl.col("period") == m))
                           .cast(pl.Float64).alias(c))
        cols.append(c)
cat = lambda cs, n: pl.concat_str(cs, separator="_").cast(pl.Categorical).to_physical().alias(n)
P = P.with_columns(cat(["exporter", "hs6", "market"], "f1"), cat(["exporter", "hs6", "period"], "f2"),
                   cat(["market", "period"], "f3"),
                   pl.concat_str([pl.col("grp"), pl.col("market"), pl.col("period").str.slice(4, 2)], separator="_")
                     .cast(pl.Categorical).to_physical().alias("f4"),
                   cat(["exporter", "hs6"], "cl"))
cols = [c for c in cols if P[c].sum() > 0]
print("\nEstimating...")
m = pf.fepois(f"v ~ {' + '.join(cols)} | f1 + f2 + f3 + f4",
              data=P.select(["v", "cl", "f1", "f2", "f3", "f4"] + cols).to_pandas(), vcov={"CRV1": "cl"})
E = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
E["group"], E["month"] = E["term"].str.split("_").str[0], E["term"].str.split("_").str[1]
for a, b in [("Estimate", "pct"), ("2.5%", "lo"), ("97.5%", "hi")]:
    E[b] = pct(E[a])
E.to_csv(RES / "step35_event_study_no_ev.csv", index=False)
print(E.pivot(index="month", columns="group", values="pct").round(1).to_string())

print("\nAverage monthly effect (mean of log coefficients, shown in %):")
for lab, a, b in [("Apr-Sep 2025", "202504", "202509"), ("Oct 2025 - Mar 2026", "202510", "202603"),
                  ("Apr-Jun 2026", "202604", "202606")]:
    out = []
    for g, name in [("CARSICE", "cars without plug-in/electric"), ("CARSEV", "plug-in/electric"), ("PARTS", "parts")]:
        e = E[(E["group"] == g) & E["month"].between(a, b)]["Estimate"]
        out.append(f"{name} {pct(e.mean()):+.1f}" if len(e) else f"{name} n.a.")
    print(f"  {lab}: " + " | ".join(out))

allm = sorted(RAW["period"].unique().to_list())
x = {mm: i for i, mm in enumerate(allm)}
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.axhline(0, color="#8a8a8a", linewidth=0.8)
ax.axvspan(x["202401"] - 0.5, x["202412"] + 0.5, color="#f2f2f2", zorder=0)
for g, colr, lab in [("CARSICE", "#2a5d9f", "Cars without plug-in and electric"),
                     ("CARSEV", "#c26a1b", "Plug-in hybrid and electric cars")]:
    d = E[E["group"] == g].sort_values("month")
    for k, seg in enumerate([d[d["month"] < "202401"], d[d["month"] >= "202501"]]):
        xs = [x[mm] for mm in seg["month"]]
        if g == "CARSICE":
            ax.fill_between(xs, seg["lo"], seg["hi"], color=colr, alpha=0.13, linewidth=0)
        ax.plot(xs, seg["pct"], color=colr, linewidth=1.8, marker="o", markersize=2.5, label=lab if k == 1 else None)
for mm in ["202504", "202508", "202510"]:
    ax.axvline(x[mm] - 0.5, color="#8a8a8a", linewidth=0.9, linestyle="--")
ax.text(x["202510"] - 0.3, 0.04, "US EV credits end", transform=ax.get_xaxis_transform(), fontsize=7, color="#444444")
ticks = [mm for mm in allm if mm[4:] in ("01", "07")]
ax.set_xticks([x[mm] for mm in ticks])
ax.set_xticklabels([f"{mm[:4]}-{mm[4:]}" for mm in ticks], fontsize=7)
ax.set_ylabel("% change")
ax.grid(axis="y", color="#ececec")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "fig1b_event_study_ice_ev.png", dpi=300)
print("\nSaved 5_figures/fig1b_event_study_ice_ev.png")
