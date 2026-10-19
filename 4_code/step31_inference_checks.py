"""Step 3.12: Are the main results still significant with more cautious standard errors?

Only 17 HS6 codes are cars, so standard errors clustered by exporter x HS6 may be too small (Step 3.3 simulation).
Part 1: the main model (HIGH = Apr-Sep 2025, LOW = Oct 2025 - Jun 2026; cars and core parts) with three ways of
        clustering: (a) exporter x HS6 (main), (b) HS6 only, (c) two-way, exporter and HS6.
Part 2: placebo test (randomization inference). Take only the clean control, pick 17 HS6 codes at random, call them
        "fake cars", and estimate their US effect in HIGH exactly like the real one. Repeat N_REPS times. If the real
        car effect is larger (more negative) than almost all fake ones, it is not chance. p = share of fake effects at
        least as negative as the real one. No standard errors needed, so the few-clusters problem does not apply.
Saves 5_results/step31_clusters.csv and 5_results/step31_placebo.csv. Part 2 prints progress; with N_REPS = 100
it takes roughly 15-30 minutes. Set N_REPS lower to test quickly.
"""
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf

N_REPS = 100
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
             pl.concat_str(["exporter", "hs6"], separator="_").cast(pl.Categorical).to_physical().alias("cl_pair"),
             pl.col("hs6").cast(pl.Categorical).to_physical().alias("cl_hs6"),
             pl.col("exporter").cast(pl.Categorical).to_physical().alias("cl_exp"),
             (pl.col("market") == "US").alias("is_us"),
             pl.col("period").is_between(pl.lit("202504"), pl.lit("202509")).alias("high"),
             pl.col("period").is_between(pl.lit("202510"), pl.lit("202606")).alias("low")))


def with_season(P, label_col):
    return P.with_columns(pl.concat_str([pl.col(label_col), pl.col("market"), pl.col("period").str.slice(4, 2)],
                                        separator="_").cast(pl.Categorical).to_physical().alias("f4"))


# ---------------- Part 1: clustering
P = with_season(P0, "group_v4")
cars, parts = pl.col("group_v4") == "1 CARS", pl.col("group_v4") == "2 PARTS_CORE"
terms = {"CARS_HIGH": cars & pl.col("is_us") & pl.col("high"), "CARS_LOW": cars & pl.col("is_us") & pl.col("low"),
         "PARTS_HIGH": parts & pl.col("is_us") & pl.col("high"), "PARTS_LOW": parts & pl.col("is_us") & pl.col("low")}
P = P.with_columns([c.cast(pl.Float64).alias(n) for n, c in terms.items()])
df = P.select(["v", "f1", "f2", "f3", "f4", "cl_pair", "cl_hs6", "cl_exp"] + list(terms)).to_pandas()
fml = f"v ~ {' + '.join(terms)} | f1 + f2 + f3 + f4"
rows = []
print("Part 1. Main model with three ways of clustering (% effect, 95% interval, p):")
for label, vc in [("(a) exporter x HS6", {"CRV1": "cl_pair"}), ("(b) HS6", {"CRV1": "cl_hs6"}),
                  ("(c) two-way exporter and HS6", {"CRV1": "cl_exp+cl_hs6"})]:
    m = pf.fepois(fml, data=df, vcov=vc)
    T = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
    T["clustering"] = label
    rows.append(T)
    print(f"  {label}")
    for _, r in T.iterrows():
        print(f"    {r['term']:11s} {100 * (np.exp(r['Estimate']) - 1):+6.1f}%  "
              f"[{100 * (np.exp(r['2.5%']) - 1):+.0f}, {100 * (np.exp(r['97.5%']) - 1):+.0f}]  p = {r['Pr(>|t|)']:.3f}")
pd.concat(rows, ignore_index=True).to_csv(OUT / "step31_clusters.csv", index=False)
real = float(rows[0].set_index("term").loc["CARS_HIGH", "Estimate"])

# ---------------- Part 2: placebo
C = P0.filter(pl.col("group_v4") == "4 CONTROL")
codes = sorted(C["hs6"].unique().to_list())
rng = np.random.default_rng(SEED)
fake = []
start = time.time()
print(f"\nPart 2. Placebo: {N_REPS} draws of 17 random control HS6 codes as fake cars (real car effect in HIGH: "
      f"{100 * (np.exp(real) - 1):+.1f}%)")
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
        print(f"  draw {i + 1}: failed ({type(e).__name__}); skipped")
    if (i + 1) % 10 == 0:
        print(f"  {i + 1} draws done ({time.time() - start:.0f} s)")
fake = np.array(fake)
pd.DataFrame({"fake_cars_high": fake}).to_csv(OUT / "step31_placebo.csv", index=False)
p = (np.sum(fake <= real) + 1) / (len(fake) + 1)
print(f"\nPlacebo results ({len(fake)} usable draws): fake effects range from {100 * (np.exp(fake.min()) - 1):+.1f}% "
      f"to {100 * (np.exp(fake.max()) - 1):+.1f}%; median {100 * (np.exp(np.median(fake)) - 1):+.1f}%.")
print(f"Real car effect {100 * (np.exp(real) - 1):+.1f}%. Share of fake effects at least as negative: "
      f"{np.mean(fake <= real):.3f}; randomization p-value = {p:.3f}")
