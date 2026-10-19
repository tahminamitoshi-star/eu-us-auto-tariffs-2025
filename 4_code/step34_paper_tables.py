"""Step 5.1: Final tables and figures for the paper.

No new results except the event-study figure, which is re-estimated so that it uses exactly the main specification.
Everything else is read from the CSV files that steps 25-33 saved in 5_results.

Part 1. Event study for the paper (Figure 1). Main specification: US vs other non-EU markets without China, flow,
        exporter x HS6 x month, market x month and group x market x calendar-month fixed effects, clustered by
        exporter x HS6. One coefficient per month for cars and for core parts, 2023 and 2025-2026; 2024 is the
        reference year (so each month is compared with the same calendar month of 2024). Jan-Mar 2025 is shown
        (front-loading) but marked. Takes a few minutes.
Part 2. Figure 2: placebo histogram (500 random groups of 17 control products), restyled for the paper.
Part 3. Tables 1-5 from the saved CSV files, written as Markdown tables into 5_results/paper_tables.md.

Saves 5_figures/fig1_event_study.png, 5_figures/fig2_placebo.png, 5_results/fig1_event_study.csv,
5_results/paper_tables.md
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
RES = BASE / "5_results"
FIG = BASE / "5_figures"
FIG.mkdir(exist_ok=True)
BLUE, GREY, RED = "#2a5d9f", "#8a8a8a", "#b42318"
pct = lambda x: 100 * (np.exp(x) - 1)


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def cell(r, ci=True):
    s = f"{pct(r['Estimate']):+.1f}{stars(r['Pr(>|t|)'])}"
    return s + (f" [{pct(r['2.5%']):+.0f}, {pct(r['97.5%']):+.0f}]" if ci else "")


# ======================= Part 1: event study, main specification
RAW = pl.read_parquet(BASE / "7_processed_data" / "panel_v1.parquet",
                      columns=["exporter", "hs6", "dest", "us", "period", "group_v4", "value_eur"])
P = (RAW.filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", "4 CONTROL"]) & (pl.col("dest") != "CN"))
        .with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
        .group_by(["exporter", "hs6", "market", "period", "group_v4"]).agg((pl.col("value_eur").sum() / 1e6).alias("v")))
months = [m for m in sorted(P["period"].unique().to_list()) if not m.startswith("2024")]
cols = []
for g, short in [("1 CARS", "CARS"), ("2 PARTS_CORE", "PARTS")]:
    for m in months:
        c = f"{short}_{m}"
        P = P.with_columns(((pl.col("group_v4") == g) & (pl.col("market") == "US") & (pl.col("period") == m))
                           .cast(pl.Float64).alias(c))
        cols.append(c)
cat = lambda cs, n: pl.concat_str(cs, separator="_").cast(pl.Categorical).to_physical().alias(n)
P = P.with_columns(cat(["exporter", "hs6", "market"], "f1"), cat(["exporter", "hs6", "period"], "f2"),
                   cat(["market", "period"], "f3"),
                   pl.concat_str([pl.col("group_v4"), pl.col("market"), pl.col("period").str.slice(4, 2)],
                                 separator="_").cast(pl.Categorical).to_physical().alias("f4"),
                   cat(["exporter", "hs6"], "cl"))
print("Part 1. Estimating the event study (main specification)...")
m = pf.fepois(f"v ~ {' + '.join(cols)} | f1 + f2 + f3 + f4", data=P.select(["v", "cl", "f1", "f2", "f3", "f4"] + cols)
              .to_pandas(), vcov={"CRV1": "cl"})
E = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
E["group"], E["month"] = E["term"].str.split("_").str[0], E["term"].str.split("_").str[1]
for a, b in [("Estimate", "pct"), ("2.5%", "lo"), ("97.5%", "hi")]:
    E[b] = pct(E[a])
E.to_csv(RES / "fig1_event_study.csv", index=False)
print(E.pivot(index="month", columns="group", values="pct").round(1).to_string())

allm = sorted(RAW["period"].unique().to_list())
x = {mm: i for i, mm in enumerate(allm)}
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, sharey=True)
for ax, (short, title) in zip(axes, [("CARS", "A. Finished cars"), ("PARTS", "B. Core car parts")]):
    d = E[E["group"] == short].sort_values("month")
    ax.axhline(0, color=GREY, linewidth=0.8)
    ax.axvspan(x["202401"] - 0.5, x["202412"] + 0.5, color="#f2f2f2", zorder=0)
    ax.axvspan(x["202501"] - 0.5, x["202503"] + 0.5, color="#fbe9d7", zorder=0)
    for seg in [d[d["month"] < "202401"], d[d["month"] >= "202501"]]:
        xs = [x[mm] for mm in seg["month"]]
        ax.fill_between(xs, seg["lo"], seg["hi"], color=BLUE, alpha=0.15, linewidth=0)
        ax.plot(xs, seg["pct"], color=BLUE, linewidth=1.8, marker="o", markersize=3)
    for mm, lab in [("202504", "27.5%"), ("202508", "15%")]:
        ax.axvline(x[mm] - 0.5, color=GREY, linewidth=0.9, linestyle="--")
        if short == "CARS":
            ax.text(x[mm] - 0.3, 0.97, lab, transform=ax.get_xaxis_transform(), fontsize=8, color="#444444", va="top")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_ylabel("% change")
    ax.grid(axis="y", color="#ececec")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

ticks = [mm for mm in allm if mm[4:] in ("01", "04", "07", "10")]
axes[1].set_xticks([x[mm] for mm in ticks])
axes[1].set_xticklabels([f"{mm[:4]}-{mm[4:]}" for mm in ticks], rotation=45, fontsize=7)
axes[0].text(x["202406"], 0.95, "2024 = reference", fontsize=7, color="#666666", ha="center", va="top", transform=axes[0].get_xaxis_transform())
axes[0].text(x["202502"], 0.95, "front-\nloading", fontsize=7, color="#9a5b1e", ha="center", va="top", transform=axes[0].get_xaxis_transform())
fig.tight_layout()
fig.savefig(FIG / "fig1_event_study.png", dpi=300)
plt.close(fig)

# ======================= Part 2: placebo figure
F = pd.read_csv(RES / "step31b_placebo_500.csv")
real = float(pd.read_csv(RES / "step31_clusters.csv").query("term == 'CARS_HIGH'")["Estimate"].iloc[0])
f = F["fake_cars_high"].dropna().to_numpy()
p_ri = (np.sum(f <= real) + 1) / (len(f) + 1)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.hist(pct(f), bins=30, color="#b9c3cf", edgecolor="white")
ax.axvline(pct(real), color=RED, linewidth=2)
ax.text(pct(real) + 1, ax.get_ylim()[1] * 0.92, f"EU cars\n{pct(real):+.1f}%", color=RED, fontsize=9, va="top")
ax.set_xlabel("Estimated effect on exports to the US, Apr-Sep 2025 (%)")
ax.set_ylabel("Number of placebo groups")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig2_placebo.png", dpi=300)
plt.close(fig)
print(f"\nPart 2. Placebo: {len(f)} draws, randomization p = {p_ri:.3f}")

# ======================= Part 3: tables
L = []
terms = [("CARS_HIGH", "Cars, Apr-Sep 2025"), ("CARS_LOW", "Cars, Oct 2025-Jun 2026"),
         ("PARTS_HIGH", "Core parts, Apr-Sep 2025"), ("PARTS_LOW", "Core parts, Oct 2025-Jun 2026")]
C31 = pd.read_csv(RES / "step31_clusters.csv")
D32 = pd.read_csv(RES / "step32_deflection_ppml.csv")
I33 = pd.read_csv(RES / "step33_intra_eu.csv")
cols1 = [("(1) US vs other markets", C31[C31["clustering"].str.startswith("(a)")]),
         ("(2) US only", D32[D32["model"].str.startswith("3.")]),
         ("(3) Other non-EU markets", D32[D32["model"].str.startswith("2a")]),
         ("(4) Intra-EU", I33[I33["model"].str.startswith("1.")])]
L += ["## Table 1. Effect of the 2025 US tariffs on EU exports of cars and core parts", "",
      "| | " + " | ".join(n for n, _ in cols1) + " |", "| --- |" + " --- |" * len(cols1)]
for t, lab in terms:
    row = [cell(T.set_index("term").loc[t].to_dict() | {}) if t in set(T["term"]) else "" for _, T in cols1]
    L.append(f"| {lab} | " + " | ".join(row) + " |")
L += ["", f"Notes: PPML; % change relative to control products (clean control, 629 HS6 codes); 95% intervals in brackets; "
      f"clustered by exporter x HS6; * p<0.10, ** p<0.05, *** p<0.01. (1) EU exports to the US relative to other non-EU "
      f"markets except China (main specification). (2) US market only, cars and parts against control products in the US. "
      f"(3) Other non-EU markets except China, against control products in the same markets. (4) Intra-EU exports. "
      f"Jan-Mar 2025 excluded. Randomization inference for (1), cars Apr-Sep 2025: p = {p_ri:.3f} ({len(f)} placebo groups "
      f"of 17 control products).", ""]

# Table 2: robustness of the car effect
M25 = pd.read_csv(RES / "table_main.csv")
C30 = pd.read_csv(RES / "step30_checks.csv")
rows2 = [("Main specification", "Apr-Sep 2025", C31[C31["clustering"].str.startswith("(a)")]),
         ("Clustered by HS6", "Apr-Sep 2025", C31[C31["clustering"].str.startswith("(b)")]),
         ("Two-way clustered, exporter and HS6", "Apr-Sep 2025", C31[C31["clustering"].str.startswith("(c)")]),
         ("Without plug-in hybrid and electric cars", "Apr-Sep 2025", C30[C30["check"].str.startswith("A2")])]
L += ["## Table 2. Robustness of the car effect", "", "| Specification | Period | Cars, high-tariff period | Cars, Oct 2025-Jun 2026 |",
      "| --- | --- | --- | --- |"]
for lab, per, T in rows2:
    T = T.set_index("term")
    L.append(f"| {lab} | {per} | {cell(T.loc['CARS_HIGH'])} | {cell(T.loc['CARS_LOW'])} |")
for spec, lab in [("(1) MAIN", "Five periods (main fixed effects)"), ("(2) with China", "China in the comparison markets"),
                  ("(3) no seasonal FE", "No seasonal fixed effects"), ("(4) Jan-Mar 2025 kept", "Jan-Mar 2025 kept")]:
    T = M25[M25["spec"] == spec].set_index("term")
    L.append(f"| {lab} | Apr-Jul 2025 | {cell(T.loc['CARS_AprJul25'])} | |")
L.append(f"| Randomization inference, {len(f)} placebo groups | Apr-Sep 2025 | p = {p_ri:.3f} | |")
L += ["", "Notes: PPML, % change, 95% intervals in brackets. Rows 1-4 use two periods (Apr-Sep 2025, Oct 2025-Jun 2026); "
      "rows 5-8 use five periods, of which Apr-Jul 2025 is shown. Unless stated, clustered by exporter x HS6.", ""]

# Table 3: prices
PR = pd.read_csv(RES / "table_prices.csv")
pers = [("AprJul25", "Apr-Jul 2025"), ("AugSep25", "Aug-Sep 2025"), ("OctDec25", "Oct-Dec 2025"),
        ("JanMar26", "Jan-Mar 2026"), ("AprJun26", "Apr-Jun 2026")]
L += ["## Table 3. EU export prices to the US (border prices, before the US tariff)", "",
      "| Period | Cars, unweighted | Cars, weighted | Core parts, unweighted | Core parts, weighted |",
      "| --- | --- | --- | --- | --- |"]
for p, lab in pers:
    out = []
    for g in ["CARS", "PARTS"]:
        for s in ["(1) unweighted", "(2) weighted by 2023-24 value"]:
            r = PR[(PR["spec"] == s) & (PR["term"] == f"{g}_{p}")]
            out.append(cell(r.iloc[0]) if len(r) else "")
    L.append(f"| {lab} | " + " | ".join(out) + " |")
L += ["", "Notes: OLS on log unit value (EUR per vehicle for cars, EUR per kg for parts) at CN8 level; US relative to "
      "other non-EU markets except China, relative to control products. Weighted: by the flow's 2023-24 export value. "
      "Fixed effects: flow, exporter x CN8 x month, market x month, group x market x calendar month. Clustered by "
      "exporter x CN8. 95% intervals in brackets.", ""]

# Table 4: heterogeneity
T30 = C30[C30["check"].str.startswith("C")].set_index("term")
L += ["## Table 4. Which car exports fell", "", "| | Apr-Sep 2025 | Oct 2025-Jun 2026 |", "| --- | --- | --- |",
      "| *A. By exporter* | | |"]
for k, lab in [("DE", "Germany"), ("SE", "Sweden"), ("IT", "Italy"), ("SK", "Slovakia"), ("HU", "Hungary"),
               ("BE", "Belgium"), ("AT", "Austria"), ("OTH", "Other EU")]:
    if f"CARS{k}_HIGH" in T30.index:
        L.append(f"| {lab} | {cell(T30.loc[f'CARS{k}_HIGH'])} | {cell(T30.loc[f'CARS{k}_LOW'])} |")
PM = pd.read_csv(RES / "table_premium.csv")
PM = PM[PM["threshold"] == 50000].set_index("term")
L.append("| *B. By pre-tariff price per car* | | |")
for k, lab in [("PREM", "EUR 50,000 or more"), ("MASS", "Below EUR 50,000")]:
    if f"CARS{k}_HIGH" in PM.index:
        L.append(f"| {lab} | {cell(PM.loc[f'CARS{k}_HIGH'])} | {cell(PM.loc[f'CARS{k}_LOW'])} |")
L += ["", "Notes: main specification, PPML, % change, 95% intervals in brackets. Panel B: price per car = 2023-24 EUR per "
      "vehicle of each exporter x HS6 across non-EU markets; car products with fewer than 50 vehicles in 2023-24 are "
      "estimated separately and not shown.", ""]

# Table 5: destinations (descriptive)
DS = pd.read_csv(RES / "step32_deflection_desc.csv")
names = {"EU": "Intra-EU", "US": "United States", "XX": "Rest of world", "GB": "United Kingdom", "CN": "China",
         "TR": "Turkey", "CH": "Switzerland", "KR": "South Korea", "JP": "Japan", "CA": "Canada", "MX": "Mexico"}
c = DS[(DS["period"] == "HIGH") & (DS["group_v4"] == "1 CARS")].sort_values("v_base", ascending=False)
k = DS[(DS["period"] == "HIGH") & (DS["group_v4"] == "4 CONTROL")].set_index("dest")["pct_value"]
L += ["## Table 5. EU car exports by destination, Apr-Sep 2025 against Apr-Sep 2024", "",
      "| Destination | 2024, EUR m per month | Change, EUR m per month | Cars, % value | Cars, % units | Control products, % value |",
      "| --- | --- | --- | --- | --- | --- |"]
for _, r in c.iterrows():
    L.append(f"| {names.get(r['dest'], r['dest'])} | {r['v_base'] / 1e6:,.0f} | {r['chg_eur_m_per_month']:+,.0f} | "
             f"{r['pct_value']:+.1f} | {r['pct_cars']:+.1f} | {k.get(r['dest'], np.nan):+.1f} |")
L += ["", "Notes: all EU exports in Comext (including flows that started after 2024), monthly averages.", ""]

(RES / "paper_tables.md").write_text("\n".join(L), encoding="utf-8")
print("\nPart 3. Tables written to 5_results/paper_tables.md")
print("\n".join(L))
print("\nFigures saved: 5_figures/fig1_event_study.png, 5_figures/fig2_placebo.png")
