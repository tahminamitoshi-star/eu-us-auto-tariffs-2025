"""Step 3.12b: placebo test with 500 draws (final version for the paper).

Same placebo as step31 Part 2, but with 500 draws instead of 100. With 100 draws the smallest possible p-value is
0.010, so step31 could only show p <= 0.010. With 500 draws the floor is 0.002.
The seed is the same (2027), so the first 100 draws are exactly the step31 draws; this run adds 400 more.

Changes against step31:
- only the placebo (the clustering checks are done);
- pyfixest warnings are switched off, so the screen shows only progress lines;
- results are saved every 50 draws, so a crash or a closed window does not lose the work done;
- a histogram of the fake effects with the real car effect marked is saved for the paper's appendix.

Takes about 2.5 hours. Start it and leave the PowerShell window open (the computer must not go to sleep).
Saves 5_results/step31b_placebo_500.csv and 5_results/step31b_placebo_hist.png
"""
import os
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

N_REPS = 500
SEED = 2027

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
OUT = BASE / "5_results"
RAW = pl.read_parquet(BASE / "7_processed_data" / "panel_v1.parquet",
                      columns=["exporter", "hs6", "dest", "us", "period", "group_v4", "value_eur", "frontload"])
P0 = (RAW.filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", "4 CONTROL"]) & (pl.col("dest") != "CN")
                 & ~pl.col("frontload"))
         .with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
         .group_by(["exporter", "hs6", "market", "period", "group_v4"])
         .agg((pl.col("value_eur").sum() / 1e6).alias("v"))
         .with_columns(
             pl.concat_str(["exporter", "hs6", "market"], separator="_").cast(pl.Categorical).to_physical().alias("f1"),
             pl.concat_str(["exporter", "hs6", "period"], separator="_").cast(pl.Categorical).to_physical().alias("f2"),
             pl.concat_str(["market", "period"], separator="_").cast(pl.Categorical).to_physical().alias("f3"),
             (pl.col("market") == "US").alias("is_us"),
             pl.col("period").is_between(pl.lit("202504"), pl.lit("202509")).alias("high"),
             pl.col("period").is_between(pl.lit("202510"), pl.lit("202606")).alias("low")))


def with_season(P, label_col):
    return P.with_columns(pl.concat_str([pl.col(label_col), pl.col("market"), pl.col("period").str.slice(4, 2)],
                                        separator="_").cast(pl.Categorical).to_physical().alias("f4"))


# ---------------- real car effect (main model, same as step31 Part 1)
P = with_season(P0, "group_v4")
cars, parts = pl.col("group_v4") == "1 CARS", pl.col("group_v4") == "2 PARTS_CORE"
terms = {"CARS_HIGH": cars & pl.col("is_us") & pl.col("high"), "CARS_LOW": cars & pl.col("is_us") & pl.col("low"),
         "PARTS_HIGH": parts & pl.col("is_us") & pl.col("high"), "PARTS_LOW": parts & pl.col("is_us") & pl.col("low")}
P = P.with_columns([c.cast(pl.Float64).alias(n) for n, c in terms.items()])
df = P.select(["v", "f1", "f2", "f3", "f4"] + list(terms)).to_pandas()
real = float(pf.fepois(f"v ~ {' + '.join(terms)} | f1 + f2 + f3 + f4", data=df, vcov="iid").coef()["CARS_HIGH"])
print(f"Real car effect in HIGH: {100 * (np.exp(real) - 1):+.1f}%")

# ---------------- placebo
C = P0.filter(pl.col("group_v4") == "4 CONTROL")
codes = sorted(C["hs6"].unique().to_list())
rng = np.random.default_rng(SEED)
fake = []
start = time.time()


def save():
    pd.DataFrame({"draw": range(1, len(fake) + 1), "fake_cars_high": fake}).to_csv(
        OUT / "step31b_placebo_500.csv", index=False)


print(f"Placebo: {N_REPS} draws of 17 random control HS6 codes (progress every 25 draws)")
for i in range(N_REPS):
    pick = list(rng.choice(codes, size=17, replace=False))
    Q = (C.with_columns(pl.when(pl.col("hs6").is_in(pick)).then(pl.lit("FAKE")).otherwise(pl.lit("CTRL")).alias("lab"))
          .pipe(with_season, "lab")
          .with_columns(((pl.col("lab") == "FAKE") & pl.col("is_us") & pl.col("high")).cast(pl.Float64).alias("FAKE_HIGH"),
                        ((pl.col("lab") == "FAKE") & pl.col("is_us") & pl.col("low")).cast(pl.Float64).alias("FAKE_LOW")))
    d = Q.select(["v", "f1", "f2", "f3", "f4", "FAKE_HIGH", "FAKE_LOW"]).to_pandas()
    try:
        m = pf.fepois("v ~ FAKE_HIGH + FAKE_LOW | f1 + f2 + f3 + f4", data=d, vcov="iid")
        fake.append(float(m.coef()["FAKE_HIGH"]))
    except Exception as e:
        fake.append(np.nan)
        print(f"  draw {i + 1}: failed ({type(e).__name__}); recorded as missing")
    if (i + 1) % 25 == 0:
        el = time.time() - start
        print(f"  {i + 1} draws done ({el / 60:.0f} min; about {el / (i + 1) * (N_REPS - i - 1) / 60:.0f} min left)")
    if (i + 1) % 50 == 0:
        save()
save()

f = np.array(fake)
f = f[~np.isnan(f)]
n = len(f)
p1 = (np.sum(f <= real) + 1) / (n + 1)
p2 = (np.sum(np.abs(f) >= abs(real)) + 1) / (n + 1)
pct = lambda x: 100 * (np.exp(x) - 1)
print(f"\nPlacebo results ({n} usable draws): fake effects from {pct(f.min()):+.1f}% to {pct(f.max()):+.1f}%; "
      f"median {pct(np.median(f)):+.1f}%; 2.5th-97.5th percentile {pct(np.percentile(f, 2.5)):+.1f}% to "
      f"{pct(np.percentile(f, 97.5)):+.1f}%.")
print(f"Real car effect {pct(real):+.1f}%.")
print(f"Fake effects at least as negative: {np.sum(f <= real)} of {n}; one-sided randomization p = {p1:.4f}")
print(f"Fake effects at least as large in absolute value: {np.sum(np.abs(f) >= abs(real))} of {n}; "
      f"two-sided randomization p = {p2:.4f}")
print(f"First 100 draws (should match step31): min {pct(f[:100].min()):+.1f}%, max {pct(f[:100].max()):+.1f}%")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(pct(f), bins=30, color="#9aa5b1", edgecolor="white")
ax.axvline(pct(real), color="#b42318", linewidth=2)
ax.text(pct(real), ax.get_ylim()[1] * 0.95, f"  EU cars: {pct(real):+.1f}%", color="#b42318", va="top", ha="left")
ax.set_xlabel("Estimated effect on exports to the US, Apr-Sep 2025 (%)")
ax.set_ylabel("Number of random product groups")
ax.set_title(f"Placebo: {n} random groups of 17 control products", fontsize=11)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "step31b_placebo_hist.png", dpi=200)
print("Saved 5_results/step31b_placebo_500.csv and 5_results/step31b_placebo_hist.png")
