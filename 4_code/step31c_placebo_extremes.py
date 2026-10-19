"""Step 3.12c: which products are behind the most negative placebo draws?

step31b: 6 of 500 random groups of 17 control products fell at least as much as EU cars (-30.3%), the most negative
by -45.1%. PPML weights by value, so one large product with its own US shock can drag a whole fake group down.
If the same few products sit in all the extreme draws, the extreme draws are driven by those products, not by chance
across many products.

This script repeats the random picks of step31b (same seed, same order, so the picks are identical; no estimation,
it takes seconds), reads the 500 fake effects from step31b_placebo_500.csv, and for the 10 most negative draws lists
the products that appear in them most often, with each product's 2024 EU exports to the US.
Saves 5_results/step31c_extremes.csv
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

N_REPS = 500
SEED = 2027
N_WORST = 10

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
OUT = BASE / "5_results"
RAW = pl.read_parquet(BASE / "7_processed_data" / "panel_v1.parquet",
                      columns=["exporter", "hs6", "dest", "us", "period", "group_v4", "value_eur", "frontload"])
C = RAW.filter((pl.col("group_v4") == "4 CONTROL") & (pl.col("dest") != "CN") & ~pl.col("frontload"))
codes = sorted(C["hs6"].unique().to_list())
rng = np.random.default_rng(SEED)
picks = [list(rng.choice(codes, size=17, replace=False)) for _ in range(N_REPS)]

F = pd.read_csv(OUT / "step31b_placebo_500.csv")
assert len(F) == N_REPS, "step31b_placebo_500.csv should have 500 rows"
F["pct"] = 100 * (np.exp(F["fake_cars_high"]) - 1)
worst = F.nsmallest(N_WORST, "fake_cars_high")

us24 = (C.filter(pl.col("us") & pl.col("period").str.starts_with("2024"))
         .group_by("hs6").agg((pl.col("value_eur").sum() / 1e9).alias("us_2024_bn")).to_pandas().set_index("hs6"))
tot = us24["us_2024_bn"].sum()

print(f"{len(codes)} control HS6 codes; each code is expected in {17 / len(codes) * N_WORST:.1f} of {N_WORST} draws "
      f"by chance.\n")
print(f"The {N_WORST} most negative placebo draws:")
for _, r in worst.iterrows():
    g = picks[int(r["draw"]) - 1]
    v = us24.reindex(g)["us_2024_bn"].fillna(0)
    top = v.sort_values(ascending=False)
    print(f"  draw {int(r['draw']):3d}: {r['pct']:+6.1f}%   largest products in the group (2024 EU exports to US, "
          f"EUR bn): " + ", ".join(f"{c} ({x:.2f})" for c, x in top.head(3).items())
          + f"   share of group's US value in top product: {100 * top.iloc[0] / max(top.sum(), 1e-9):.0f}%")

cnt = pd.Series([c for d in worst["draw"] for c in picks[int(d) - 1]]).value_counts()
res = (pd.DataFrame({"times_in_worst": cnt})
         .join(us24, how="left").fillna({"us_2024_bn": 0.0}))
res["share_of_control_us_2024_pct"] = 100 * res["us_2024_bn"] / tot
res["share_of_all_draws_with_code"] = [np.mean([c in p for p in picks]) for c in res.index]
res["mean_fake_effect_when_in_pct"] = [F.loc[[c in p for p in picks], "pct"].mean() for c in res.index]
res["mean_fake_effect_when_out_pct"] = [F.loc[[c not in p for p in picks], "pct"].mean() for c in res.index]
res = res.sort_values(["times_in_worst", "us_2024_bn"], ascending=False)
res.index.name = "hs6"
res.round(3).to_csv(OUT / "step31c_extremes.csv")

print(f"\nProducts that appear most often in the {N_WORST} most negative draws:")
print(res.head(12).round(2).to_string())
print("\nmean_fake_effect_when_in/out: average fake effect over all 500 draws with and without that product. "
      "A product with a large gap pulls any group it joins downward.")
