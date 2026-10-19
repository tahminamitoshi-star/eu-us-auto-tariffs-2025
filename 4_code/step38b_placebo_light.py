"""Step 6.3: lighter placebo (permutation) run that resumes from saved draws.

Why a new version. step38_placebo_parallel.py froze the PC on 18 Oct 2026 after 1,100 random draws. Each worker
process read the full panel (8.9 million rows) and pyfixest/numba in every worker started one thread per core, so
7 workers x 8 threads competed for 8 cores and the memory. The 104 failed draws came in runs of consecutive draw
numbers (139-174, 686-704, 1084-1095), which points to the PC running out of resources, not to the data.

What changes here (the model and the sampling rules are the same as step38):
  1. The control-group data are built once and saved as 7_processed_data/placebo_control.parquet; workers read
     only that small file.
  2. Every worker uses one thread (NUMBA, OpenMP, MKL, OpenBLAS, polars) and runs at below-normal priority, so
     the PC stays usable. Default: 2 workers. Change WORKERS below if the PC copes well.
  3. Draws are saved every 10 draws. If the run stops, start it again: finished draws are skipped.
  4. Failed draws are retried once, and the error text is saved in the column "error".
  5. On Windows the script asks the PC not to go to sleep while it runs.

Sampling rules (same as step38, same seeds; the random draws are the same codes as in step38):
  random   17 HS6 codes drawn without replacement from the 629 clean control codes; seed 2027.
  matched  control codes sorted into ten deciles by 2024 EU exports to the US; each draw takes, for every car code,
           one control code from that car code's decile (without replacement); seed 2028. Correction: step38
           matched only the 15 car codes with 2024 US exports; here all 17 are matched (870370 and 870451 had no
           2024 US exports and are matched in the decile that holds zero-export codes).
Target: draws 0-1,999 of each sequence (2,000 per rule, as promised to the reviewer). The random rule tops up the
draws saved by step38 and reruns its 104 failures.
p-value (one-sided) = (1 + draws at least as negative as cars) / (usable draws + 1); two-sided uses absolute values.

How to run (PowerShell, in 4_code\\rebuild):
  py step38b_placebo_light.py matched 20     a short test: 20 new matched draws, then stops (resumable)
  py step38b_placebo_light.py matched        all matched draws still missing
  py step38b_placebo_light.py random         tops up the random rule to draws 0-1,999
  py step38b_placebo_light.py summary        summary table and Figure 2 from whatever has been saved
Saves 5_results/step38_placebo_random.csv, step38_placebo_matched.csv, step38_placebo_summary.csv,
5_figures/fig2_placebo_v2.png and .pdf
"""
import os
for _k in ["NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[_k] = "1"
import sys
import time
import shutil
import warnings
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd

WORKERS = 2
TARGET = {"random": 2000, "matched": 2000}
SEEDS = {"random": 2027, "matched": 2028}
BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
OUT, FIG, DATA = BASE / "5_results", BASE / "5_figures", BASE / "7_processed_data"
CTRL_FILE = DATA / "placebo_control.parquet"
D = None  # worker data
pct = lambda x: 100 * (np.exp(x) - 1)
pfmt = lambda p: f"{p:.4f}" if p < 0.001 else f"{p:.3f}"  # 0.0005, not 0.000


# ---------------------------------------------------------------- data, built once
def build_control():
    """Clean control, China out, Jan-Mar 2025 out; exporter x HS6 x market (US / OTHER) x month, EUR million."""
    import polars as pl
    raw = pl.read_parquet(DATA / "panel_v1.parquet",
                          columns=["exporter", "hs6", "dest", "us", "period", "group_v4", "value_eur", "frontload"])
    # 2024 exports to the US per HS6, EUR million; every HS6 in the panel is listed (0 if none), so car codes
    # with no 2024 US exports are matched too (step38 left them out, giving matched groups of 15 codes)
    us24 = (raw.with_columns(pl.when(pl.col("us") & pl.col("period").str.starts_with("2024"))
                               .then(pl.col("value_eur")).otherwise(0.0).alias("x"))
               .group_by(["hs6", "group_v4"]).agg((pl.col("x").sum() / 1e6).alias("u")).to_pandas())
    c = (raw.filter((pl.col("group_v4") == "4 CONTROL") & (pl.col("dest") != "CN") & ~pl.col("frontload"))
            .with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
            .group_by(["exporter", "hs6", "market", "period"]).agg((pl.col("value_eur").sum() / 1e6).alias("v")))
    del raw
    code = lambda cs, n: pl.concat_str(cs, separator="_").cast(pl.Categorical).to_physical().cast(pl.Int64).alias(n)
    c = c.with_columns(code(["exporter", "hs6", "market"], "f1"), code(["exporter", "hs6", "period"], "f2"),
                       code(["market", "period"], "f3"),
                       code(["market", pl.col("period").str.slice(4, 2)], "mcm"),
                       (pl.col("market") == "US").alias("is_us"),
                       pl.col("period").is_between(pl.lit("202504"), pl.lit("202509")).alias("high"),
                       pl.col("period").is_between(pl.lit("202510"), pl.lit("202606")).alias("low"))
    c.select(["v", "hs6", "f1", "f2", "f3", "mcm", "is_us", "high", "low"]).write_parquet(CTRL_FILE)
    us24.to_csv(DATA / "placebo_us2024_by_hs6.csv", index=False)
    print(f"Built {CTRL_FILE.name}: {c.height:,} rows")


def make_draws(rule, n):
    """Same draw sequence as step38_placebo_parallel.py (same seed, same order of random calls)."""
    codes = sorted(pd.read_parquet(CTRL_FILE, columns=["hs6"])["hs6"].unique().tolist())
    us = pd.read_csv(DATA / "placebo_us2024_by_hs6.csv", dtype={"hs6": str})
    ctl = us[(us["group_v4"] == "4 CONTROL") & us["hs6"].isin(codes)][["hs6", "u"]].copy()
    edges = np.quantile(ctl["u"], np.linspace(0, 1, 11))
    ctl["dec"] = np.clip(np.searchsorted(edges, ctl["u"], side="right") - 1, 0, 9)
    car = us[us["group_v4"] == "1 CARS"].copy()
    car["dec"] = np.clip(np.searchsorted(edges, car["u"], side="right") - 1, 0, 9)
    need = car["dec"].value_counts().to_dict()
    pools = {d: ctl.loc[ctl["dec"] == d, "hs6"].tolist() for d in range(10)}
    rng = np.random.default_rng(SEEDS[rule])
    draws = []
    for _ in range(n):
        if rule == "random":
            pick = list(rng.choice(codes, size=17, replace=False))
        else:
            pick = []
            for d, k in need.items():
                pool = [x for x in pools[d] if x not in pick]
                pick += list(rng.choice(pool, size=min(k, len(pool)), replace=False))
        draws.append([str(x) for x in pick])
    return draws, need, ctl.set_index("hs6")["u"], car


# ---------------------------------------------------------------- workers
def init_worker():
    global D
    warnings.filterwarnings("ignore")
    if os.name == "nt":  # below-normal priority, so the PC stays responsive
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    D = pd.read_parquet(CTRL_FILE)
    D["hs6"] = D["hs6"].astype("category")


def one_draw(task):
    import pyfixest as pf
    warnings.filterwarnings("ignore")
    i, pick = task
    try:
        fake = D["hs6"].isin(pick).to_numpy()
        q = pd.DataFrame({"v": D["v"], "f1": D["f1"], "f2": D["f2"], "f3": D["f3"],
                          "f4": D["mcm"].to_numpy() * 2 + fake,
                          "FAKE_HIGH": (fake & D["is_us"].to_numpy() & D["high"].to_numpy()).astype(float),
                          "FAKE_LOW": (fake & D["is_us"].to_numpy() & D["low"].to_numpy()).astype(float)})
        m = pf.fepois("v ~ FAKE_HIGH + FAKE_LOW | f1 + f2 + f3 + f4", data=q, vcov="iid")
        return i, float(m.coef()["FAKE_HIGH"]), ""
    except Exception as e:
        return i, np.nan, f"{type(e).__name__}: {e}"[:200]


# ---------------------------------------------------------------- run one rule
def run(rule, max_new=None):
    path = OUT / f"step38_placebo_{rule}.csv"
    draws, *_ = make_draws(rule, TARGET[rule])
    done = pd.DataFrame(columns=["draw", "fake_high", "codes", "error"])
    if path.exists():
        backup = OUT / f"step38_placebo_{rule}_backup.csv"
        if not backup.exists():
            shutil.copy(path, backup)
        done = pd.read_csv(path, dtype={"codes": str})
        if "error" not in done:
            done["error"] = ""
        # check that saved draws are the same codes as the regenerated sequence
        chk = done[done["draw"] < len(draws)]
        bad = [int(r.draw) for r in chk.itertuples() if r.codes != ";".join(draws[int(r.draw)])]
        if bad:
            sys.exit(f"Saved draws {bad[:5]} do not match the seed-{SEEDS[rule]} sequence. Stopping.")
    ok = set(done.loc[done["fake_high"].notna(), "draw"].astype(int))
    todo = [i for i in range(len(draws)) if i not in ok]
    if max_new:
        todo = todo[:max_new]
    print(f"\nRule '{rule}': {len(ok)} usable draws saved; {len(todo)} to run with {WORKERS} workers.")
    if not todo:
        return
    res = {int(r.draw): (r.fake_high, r.codes, r.error if isinstance(r.error, str) else "")
           for r in done.itertuples()}

    def save():
        out = pd.DataFrame([(k, *v) for k, v in res.items()], columns=["draw", "fake_high", "codes", "error"])
        out.sort_values("draw").to_csv(path, index=False)

    if os.name == "nt":  # keep the PC awake while this runs
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    start = time.time()
    n = 0
    for attempt in (1, 2):
        tasks = [(i, draws[i]) for i in todo]
        with mp.get_context("spawn").Pool(WORKERS, initializer=init_worker, maxtasksperchild=50) as pool:
            for i, b, err in pool.imap_unordered(one_draw, tasks, chunksize=1):
                res[i] = (b, ";".join(draws[i]), err)
                n += 1
                if n % 10 == 0:
                    save()
                if n in (5, 20) or n % 50 == 0:
                    el = time.time() - start
                    left = len(todo) - n if attempt == 1 else 0
                    print(f"  {n} draws done ({el / 60:.0f} min; about {el / n * max(left, 0) / 60:.0f} min left)")
        save()
        todo = [i for i in todo if np.isnan(res[i][0])]
        if not todo:
            break
        print(f"  {len(todo)} draws failed; retrying once. First error: {res[todo[0]][2]}")
    print(f"  Finished in {(time.time() - start) / 60:.0f} min; still failed: {len(todo)}")


# ---------------------------------------------------------------- summary and Figure 2
def summary():
    real = float(pd.read_csv(OUT / "step31_clusters.csv").query("term == 'CARS_HIGH'")["Estimate"].iloc[0])
    print(f"\nReal car effect (main model, Apr-Sep 2025): b = {real:.4f}, {pct(real):+.1f}%")
    rows, hists = [], {}
    for rule in ["random", "matched"]:
        path = OUT / f"step38_placebo_{rule}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path, dtype={"codes": str})
        d = d[d["draw"] < TARGET[rule]]
        draws, need, u, car = make_draws(rule, TARGET[rule])
        f = d["fake_high"].dropna().to_numpy()
        n = len(f)
        k1, k2 = int(np.sum(f <= real)), int(np.sum(np.abs(f) >= abs(real)))
        ext = d[d["fake_high"] <= real]
        top = ext["codes"].str.split(";").explode().value_counts()
        x = top.index[0] if len(top) else ""
        dd = d.dropna(subset=["fake_high"])
        dd = dd[~dd["codes"].str.contains(x)] if x else dd
        has_x = d["codes"].str.contains("841989").mean()
        size = d["codes"].str.split(";").apply(lambda s: u.reindex(s).fillna(0).sum()).median()
        rows.append(dict(rule=rule, draws_attempted=len(d), usable=n, failed=len(d) - n,
                         min_pct=pct(f.min()), p2_5_pct=pct(np.percentile(f, 2.5)), median_pct=pct(np.median(f)),
                         p97_5_pct=pct(np.percentile(f, 97.5)), max_pct=pct(f.max()),
                         n_as_negative=k1, p_one_sided=(1 + k1) / (n + 1), p_two_sided=(1 + k2) / (n + 1),
                         top_product=x, top_count=int(top.iloc[0]) if len(top) else 0,
                         usable_without_top=len(dd), n_as_negative_without_top=int(np.sum(dd["fake_high"] <= real)),
                         p_without_top=(1 + np.sum(dd["fake_high"] <= real)) / (len(dd) + 1),
                         share_draws_with_841989=has_x, median_group_us2024_eur_bn=size / 1e3,
                         cars_us2024_eur_bn=car["u"].sum() / 1e3))
        hists[rule] = f
        r = rows[-1]
        print(f"\nRule '{rule}': {n} usable of {len(d)} attempted (target {TARGET[rule]}); "
              f"fake effects {r['min_pct']:+.1f}% to {r['max_pct']:+.1f}%, middle 95% {r['p2_5_pct']:+.1f}% to "
              f"{r['p97_5_pct']:+.1f}%, median {r['median_pct']:+.1f}%")
        print(f"  At least as negative as cars: {k1}; one-sided p = {r['p_one_sided']:.4f}; "
              f"two-sided p = {r['p_two_sided']:.4f}")
        print(f"  Most frequent product in those draws: {x} ({r['top_count']} of {k1}); without draws containing it: "
              f"{len(dd)} draws, {r['n_as_negative_without_top']} as negative, p = {r['p_without_top']:.4f}")
        print(f"  Draws containing 841989: {100 * has_x:.1f}%. Median 2024 US exports of a placebo group: "
              f"EUR {size / 1e3:.2f} bn (cars: EUR {car['u'].sum() / 1e3:.2f} bn)")
        if rule == "matched":
            print(f"  Car codes per control-size decile (0 = smallest): {dict(sorted(need.items()))}")
    pd.DataFrame(rows).to_csv(OUT / "step38_placebo_summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    titles = {"random": "A. Random groups of 17 control products",
              "matched": "B. Size-matched groups of 17 control products"}
    fig, axes = plt.subplots(1, len(hists), figsize=(5 * len(hists), 3.6), squeeze=False)
    lo = min(pct(min(v.min() for v in hists.values())), pct(real)) - 3
    hi = pct(max(v.max() for v in hists.values())) + 3
    bins = np.arange(np.floor(lo / 2.5) * 2.5, hi + 2.5, 2.5)
    for ax, (rule, f), r in zip(axes[0], hists.items(), rows):
        ax.hist(pct(f), bins=bins, color="#b9c3cf", edgecolor="white")
        ax.axvline(pct(real), color="#b42318", linewidth=2)
        ax.text(pct(real) + 1.5, ax.get_ylim()[1] * 0.95,
                f"EU cars {pct(real):+.1f}%\n{r['n_as_negative']} of {r['usable']:,} groups\nas negative; p = {pfmt(r['p_one_sided'])}",
                color="#b42318", fontsize=8, va="top")
        ax.set_title(titles[rule], loc="left", fontsize=10)
        ax.set_xlabel("Estimated effect on exports to the US, Apr-Sep 2025 (%)", fontsize=9)
        ax.set_xlim(lo, hi)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0][0].set_ylabel("Number of placebo groups")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG / f"fig2_placebo_v2.{ext}", dpi=300)
    print("\nSaved 5_results/step38_placebo_summary.csv and 5_figures/fig2_placebo_v2.png / .pdf")


if __name__ == "__main__":
    mp.freeze_support()
    warnings.filterwarnings("ignore")
    arg = sys.argv[1] if len(sys.argv) > 1 else "matched"
    if not CTRL_FILE.exists() or not (DATA / "placebo_us2024_by_hs6.csv").exists():
        build_control()
    if arg in ("random", "matched"):
        run(arg, int(sys.argv[2]) if len(sys.argv) > 2 else None)
    summary()
