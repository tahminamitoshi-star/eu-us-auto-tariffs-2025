"""Step 6.10: redraw Figure 1 from the saved event-study estimates (5_results/fig1_event_study.csv, written by step34).

No re-estimation. Changes from step34: the second dashed line is labelled as the retroactive effective date of the
agreed 15% rate (published 25 September 2025), and the January-March 2025 shading is labelled "not in main
estimates" rather than "front-loading", because the estimates show no clear front-loading for cars and the months are
only left out of the main model.
Saves 5_figures/fig1_event_study.png and .pdf
"""
import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
E = pd.read_csv(BASE / "5_results" / "fig1_event_study.csv", dtype={"month": str})
FIG = BASE / "5_figures"
FIG.mkdir(exist_ok=True)
BLUE, GREY = "#2a5d9f", "#8a8a8a"

allm = [f"{y}{m:02d}" for y in (2023, 2024, 2025, 2026) for m in range(1, 13)]
allm = [mm for mm in allm if mm <= "202606"]
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
    for mm, lab in [("202504", "3 Apr 2025:\n27.5%"), ("202508", "1 Aug 2025: 15%\n(retroactive)")]:
        ax.axvline(x[mm] - 0.5, color=GREY, linewidth=0.9, linestyle="--")
        if short == "CARS":
            ax.text(x[mm] - 0.3, 0.03, lab, transform=ax.get_xaxis_transform(), fontsize=7, color="#444444", va="bottom")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_ylabel("% change")
    ax.grid(axis="y", color="#ececec")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

ticks = [mm for mm in allm if mm[4:] in ("01", "04", "07", "10")]
axes[1].set_xticks([x[mm] for mm in ticks])
axes[1].set_xticklabels([f"{mm[:4]}-{mm[4:]}" for mm in ticks], rotation=45, fontsize=7)
axes[0].text(x["202406"], 0.95, "2024 = reference", fontsize=7, color="#666666", ha="center", va="top",
             transform=axes[0].get_xaxis_transform())
axes[0].text(x["202502"], 0.95, "not in\nmain\nestimates", fontsize=6.5, color="#9a5b1e", ha="center", va="top",
             transform=axes[0].get_xaxis_transform())
fig.tight_layout()
fig.savefig(FIG / "fig1_event_study.png", dpi=300)
fig.savefig(FIG / "fig1_event_study.pdf")
print("Saved 5_figures/fig1_event_study.png and .pdf")
