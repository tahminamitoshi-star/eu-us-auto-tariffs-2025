"""Step 6.2: Placebo (permutation) test with more draws, using all processor cores.

Reviewer request: more draws, a documented sampling rule and seed, and a version where placebo products resemble
the car products. Two sampling rules, each with N_DRAWS draws:
  random   17 HS6 codes drawn without replacement from the 629 clean control codes, all equally likely.
  matched  size-matched: control codes are sorted into ten groups (deciles) by 2024 EU exports to the US; each of
           the 17 car codes is matched to its decile, and each draw takes one control code from the decile of each
           car code (without replacement), so a placebo group has the same size profile as the car group.
Each draw is estimated with the main specification (as in step31b). The placebo label is not a random
assignment of the real tariff, so the result is a permutation-based diagnostic: it shows how often a group of 17
products that did not face the car tariff shows a US effect as negative as cars.
Empirical p-value (one-sided) = (1 + number of draws at least as negative as cars) / (N + 1).
Two-sided version uses absolute values. Seed 2027 (random) and 2028 (matched).
Results are saved every 100 draws with the drawn codes, so draws containing any product can be excluded later.

N_DRAWS = 2000 per rule. The script prints the expected run time after the first draws; on 8 cores expect
roughly 1.5 hours per rule. Lower N_DRAWS to run faster. Keep the PowerShell window open and the PC awake.
Saves 5_results/step38_placebo_random.csv, step38_placebo_matched.csv, 5_figures/fig2_placebo_v2.png
"""
from pathlib import Path
import os
import time
import warnings
import multiprocessing as mp
import numpy as np
import pandas as pd

N_DRAWS = 2000
RULES = [("random", 2027), ("matched", 2028)]
BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
OUT = BASE / "5_results"
FIG = BASE / "5_figures"
C = None


def load_control():
    import polars as pl
    raw = pl.read_parquet(BASE / "7_processed_data" / "panel_v1.parquet",
                          columns=["exporter", "hs6", "dest", "us", "period", "group_v4", "value_eur", "frontload"])
    c = (raw.filter((pl.col("group_v4") == "4 CONTROL") & (pl.col("dest") != "CN") & ~pl.col("frontload"))
            .with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
            .group_by(["exporter", "hs6", "market", "period"]).agg((pl.col("value_eur").sum() / 1e6).alias("v")))
    cat = lambda cs, n: pl.concat_str(cs, separator="_").cast(pl.Categorical).to_physical().alias(n)
    c = c.with_columns(cat(["exporter", "hs6", "market"], "f1"), cat(["exporter", "hs6", "period"], "f2"),
                       cat(["market", "period"], "f3"), cat(["exporter", "hs6"], "cl"),
                       (pl.col("market") == "US").alias("is_us"),
                       pl.col("period").is_between(pl.lit("202504"), pl.lit("202509")).alias("high"),
                       pl.col("period").is_between(pl.lit("202510"), pl.lit("202606")).alias("low"),
                       pl.col("period").str.slice(4, 2).alias("cm"))
    return raw, c


def init_worker():
    global C
    warnings.filterwarnings("ignore")
    _, C = load_control()


def one_draw(task):
    import polars as pl
    import pyfixest as pf
    warnings.filterwarnings("ignore")
    i, pick = task
    q = (C.with_columns(pl.col("hs6").is_in(pick).alias("fake"))
          .with_columns(pl.concat_str([pl.col("fake").cast(pl.Utf8), pl.col("market"), pl.col("cm")], separator="_")
                          .cast(pl.Categorical).to_physical().alias("f4"),
                        (pl.col("fake") & pl.col("is_us") & pl.col("high")).cast(pl.Float64).alias("FAKE_HIGH"),
                        (pl.col("fake") & pl.col("is_us") & pl.col("low")).cast(pl.Float64).alias("FAKE_LOW")))
    try:
        m = pf.fepois("v ~ FAKE_HIGH + FAKE_LOW | f1 + f2 + f3 + f4",
                      data=q.select(["v", "f1", "f2", "f3", "f4", "FAKE_HIGH", "FAKE_LOW"]).to_pandas(), vcov="iid")
        return i, float(m.coef()["FAKE_HIGH"])
    except Exception:
        return i, np.nan


def main():
    import polars as pl
    warnings.filterwarnings("ignore")
    raw, c = load_control()
    codes = sorted(c["hs6"].unique().to_list())
    us24 = (raw.filter(pl.col("us") & pl.col("period").str.starts_with("2024"))
               .group_by(["hs6", "group_v4"]).agg(pl.col("value_eur").sum().alias("u")))
    ctl = us24.filter((pl.col("group_v4") == "4 CONTROL") & pl.col("hs6").is_in(codes)).to_pandas()
    ctl = pd.concat([ctl, pd.DataFrame({"hs6": [x for x in codes if x not in set(ctl["hs6"])], "u": 0.0})])
    edges = np.quantile(ctl["u"], np.linspace(0, 1, 11))
    ctl["dec"] = np.clip(np.searchsorted(edges, ctl["u"], side="right") - 1, 0, 9)
    car = us24.filter(pl.col("group_v4") == "1 CARS").to_pandas()
    car["dec"] = np.clip(np.searchsorted(edges, car["u"], side="right") - 1, 0, 9)
    need = car["dec"].value_counts().to_dict()
    print(f"Control codes: {len(codes)}. Car codes by control-size decile (0 = smallest): {dict(sorted(need.items()))}")
    pools = {d: ctl.loc[ctl["dec"] == d, "hs6"].tolist() for d in range(10)}
    real = float(pd.read_csv(OUT / "step31_clusters.csv").query("term == 'CARS_HIGH'")["Estimate"].iloc[0])
    print(f"Real car effect (main model, Apr-Sep 2025): {100 * (np.exp(real) - 1):+.1f}%")
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Using {workers} worker processes.")
    for rule, seed in RULES:
        rng = np.random.default_rng(seed)
        tasks = []
        for i in range(N_DRAWS):
            if rule == "random":
                pick = list(rng.choice(codes, size=17, replace=False))
            else:
                pick = []
                for d, k in need.items():
                    pool = [x for x in pools[d] if x not in pick]
                    pick += list(rng.choice(pool, size=min(k, len(pool)), replace=False))
            tasks.append((i, pick))
        out = {}
        start = time.time()
        print(f"\nRule '{rule}': {N_DRAWS} draws, seed {seed}")
        with mp.get_context("spawn").Pool(workers, initializer=init_worker) as pool:
            for n, (i, b) in enumerate(pool.imap_unordered(one_draw, tasks, chunksize=2), 1):
                out[i] = b
                if n % 100 == 0 or n == 20:
                    el = time.time() - start
                    print(f"  {n} draws done ({el / 60:.0f} min; about {el / n * (N_DRAWS - n) / 60:.0f} min left)")
                if n % 100 == 0 or n == N_DRAWS:
                    pd.DataFrame({"draw": list(out.keys()), "fake_high": list(out.values()),
                                  "codes": [";".join(tasks[k][1]) for k in out.keys()]}).sort_values("draw") \
                      .to_csv(OUT / f"step38_placebo_{rule}.csv", index=False)
        f = np.array([out[k] for k in sorted(out)])
        f = f[~np.isnan(f)]
        n = len(f)
        p1 = (1 + np.sum(f <= real)) / (n + 1)
        p2 = (1 + np.sum(np.abs(f) >= abs(real))) / (n + 1)
        pc = lambda x: 100 * (np.exp(x) - 1)
        print(f"  Rule '{rule}': {n} usable draws; fake effects {pc(f.min()):+.1f}% to {pc(f.max()):+.1f}%, middle 95% "
              f"{pc(np.percentile(f, 2.5)):+.1f}% to {pc(np.percentile(f, 97.5)):+.1f}%")
        print(f"  At least as negative as cars: {np.sum(f <= real)}; one-sided p = {p1:.4f}; two-sided p = {p2:.4f}")
        d = pd.read_csv(OUT / f"step38_placebo_{rule}.csv").dropna()
        top = d[d["fake_high"] <= real]["codes"].str.split(";").explode().value_counts().head(5)
        print(f"  Products most often in the draws at least as negative as cars: {top.to_dict()}")
        if len(top):
            x = top.index[0]
            dd = d[~d["codes"].str.contains(x)]
            print(f"  Sensitivity only: without draws containing {x}: {len(dd)} draws, one-sided p = "
                  f"{(1 + np.sum(dd['fake_high'] <= real)) / (len(dd) + 1):.4f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, (rule, _) in zip(axes, RULES):
        d = pd.read_csv(OUT / f"step38_placebo_{rule}.csv").dropna()
        ax.hist(100 * (np.exp(d["fake_high"]) - 1), bins=40, color="#b9c3cf", edgecolor="white")
        ax.axvline(100 * (np.exp(real) - 1), color="#b42318", linewidth=2)
        ax.set_title({"random": "A. Random control groups", "matched": "B. Size-matched control groups"}[rule],
                     loc="left", fontsize=10)
        ax.set_xlabel("Placebo effect, Apr-Sep 2025 (%)")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Number of placebo groups")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_placebo_v2.png", dpi=300)
    print("\nSaved 5_figures/fig2_placebo_v2.png")


if __name__ == "__main__":
    mp.freeze_support()
    main()
