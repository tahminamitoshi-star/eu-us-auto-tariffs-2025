"""Step 5.4: Do car exports follow the tariff gap between cars and other goods?

Three tariff regimes for EU goods (Steps 1.12-1.18 of the log):
  R1  Apr-Jul 2025       cars 27.5%; control products normal duty + 10%      -> cars pay much more
  R2  Aug 2025-Feb 2026  cars 15%;   control products 15% all-in (EU deal)    -> about the same
  R3  Mar-Jun 2026       cars 15%;   control products normal duty + 10%      -> cars pay a little more
      (the Supreme Court struck down the IEEPA tariffs on 20 Feb 2026; the Section 122 surcharge from
       24 Feb 2026 is 10% and does not apply to cars under Section 232)
If the car effect comes from the tariff gap, petrol and diesel cars should fall a lot in R1, little in R2 and
somewhat in R3. Plug-in and electric cars had a second shock (US credits ended 30 Sep 2025), so they are shown
separately.

Main specification (China out, Jan-Mar 2025 out, flow, exporter x HS6 x month, market x month and
group x market x calendar month fixed effects, clustered by exporter x HS6), with these terms:
  CARSICE x R1, R2, R3; CARSEV x R1, R2, R3; PARTS x R1, R2, R3.
Prints % effects, the approximate log tariff gap (cars minus control, assuming a 2% normal duty on control
products) and the implied elasticity for petrol and diesel cars where the gap is not zero.
Saves 5_results/step36_tariff_regimes.csv
"""
import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf

warnings.filterwarnings("ignore")
BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
RES = BASE / "5_results"
EV = ["870360", "870370", "870380"]
pct = lambda x: 100 * (np.exp(x) - 1)
REG = [("R1", "202504", "202507", np.log(1.275 / 1.12), "Apr-Jul 2025: cars 27.5%, control normal duty + 10%"),
       ("R2", "202508", "202602", np.log(1.15 / 1.15), "Aug 2025-Feb 2026: both 15%"),
       ("R3", "202603", "202606", np.log(1.15 / 1.12), "Mar-Jun 2026: cars 15%, control normal duty + 10%")]

RAW = pl.read_parquet(BASE / "7_processed_data" / "panel_v1.parquet",
                      columns=["exporter", "hs6", "dest", "us", "period", "group_v4", "value_eur", "frontload"])
P = (RAW.filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", "4 CONTROL"]) & (pl.col("dest") != "CN")
                & ~pl.col("frontload"))
        .with_columns(pl.when((pl.col("group_v4") == "1 CARS") & pl.col("hs6").is_in(EV)).then(pl.lit("1b CARS_EV"))
                        .otherwise(pl.col("group_v4")).alias("grp"),
                      pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
        .group_by(["exporter", "hs6", "market", "period", "grp"]).agg((pl.col("value_eur").sum() / 1e6).alias("v")))
cols = []
for g, short in [("1 CARS", "CARSICE"), ("1b CARS_EV", "CARSEV"), ("2 PARTS_CORE", "PARTS")]:
    for r, a, b, _, _ in REG:
        c = f"{short}_{r}"
        P = P.with_columns(((pl.col("grp") == g) & (pl.col("market") == "US") & pl.col("period").is_between(pl.lit(a), pl.lit(b)))
                           .cast(pl.Float64).alias(c))
        cols.append(c)
cat = lambda cs, n: pl.concat_str(cs, separator="_").cast(pl.Categorical).to_physical().alias(n)
P = P.with_columns(cat(["exporter", "hs6", "market"], "f1"), cat(["exporter", "hs6", "period"], "f2"),
                   cat(["market", "period"], "f3"),
                   pl.concat_str([pl.col("grp"), pl.col("market"), pl.col("period").str.slice(4, 2)], separator="_")
                     .cast(pl.Categorical).to_physical().alias("f4"),
                   cat(["exporter", "hs6"], "cl"))
cols = [c for c in cols if P[c].sum() > 0]
m = pf.fepois(f"v ~ {' + '.join(cols)} | f1 + f2 + f3 + f4",
              data=P.select(["v", "cl", "f1", "f2", "f3", "f4"] + cols).to_pandas(), vcov={"CRV1": "cl"})
T = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
T.to_csv(RES / "step36_tariff_regimes.csv", index=False)
T = T.set_index("term")


def show(t):
    if t not in T.index:
        return "n.a."
    r = T.loc[t]
    s = "***" if r["Pr(>|t|)"] < 0.01 else "**" if r["Pr(>|t|)"] < 0.05 else "*" if r["Pr(>|t|)"] < 0.1 else ""
    return f"{pct(r['Estimate']):+6.1f}{s} [{pct(r['2.5%']):+.0f}, {pct(r['97.5%']):+.0f}] p={r['Pr(>|t|)']:.3f}"


print(f"Main specification, {m._N:,} obs. Effect on EU exports to the US, % (95% interval):\n")
for r, a, b, gap, lab in REG:
    print(f"{r}  {lab}  (approx. log tariff gap, cars minus control: {gap:.3f})")
    print(f"    Cars without plug-in/electric: {show(f'CARSICE_{r}')}")
    if gap > 0.01 and f"CARSICE_{r}" in T.index:
        print(f"        implied elasticity of exports to the tariff factor: {T.loc[f'CARSICE_{r}', 'Estimate'] / gap:.1f}")
    print(f"    Plug-in hybrid and electric:   {show(f'CARSEV_{r}')}")
    print(f"    Core parts:                    {show(f'PARTS_{r}')}\n")
names = list(T.index)
if {"CARSICE_R1", "CARSICE_R2"} <= set(names):
    R = np.zeros((1, len(names)))
    R[0, names.index("CARSICE_R1")], R[0, names.index("CARSICE_R2")] = 1.0, -1.0
    print(f"Test cars without plug-in/electric R1 = R2: p = {float(m.wald_test(R=R)['pvalue']):.3f}")
