"""Step 6.9: checks requested in the second reviewer report (v4).

Part A. Direct test that the car and core-parts responses differ (main model, same sample and fixed effects as
        Table 2, column 1). Difference CARS - PARTS with its standard error from the full covariance matrix,
        under (a) exporter x HS6 clustering (main) and (b) HS6 clustering. Two-period model and a three-period
        model that shows April-July 2025 (largest tariff gap) separately from August-September 2025.
Part B. Cars without light trucks (870421, 870431, 870441, 870451, 870460): these codes leave the sample.
Part C. Leave-one-car-code-out: the main model re-estimated 17 times, each time without one car HS6 code;
        range of the April-September 2025 car estimate and a jackknife standard error over the car codes
        that had 2024 exports to the US.
Part D. Exporter-country estimates for October 2025 - June 2026 with Holm-adjusted p-values (from step30), and the
        Germany-versus-rest contrast (from step27), printed for the tables.
Part E. Full product crosswalk for Appendix B: every HS6 code in chapters 84-87 with its group and the reason for
        it. Corrects the annex status of 870370 and 870451, which are listed in Annex I of Proclamation 10908
        (8703.70.00, 8704.51.00) but were marked "none" in appendix_crosswalk.csv.
Saves 5_results/step41_contrasts.csv, step41_no_light_trucks.csv, step41_leave_one_code_out.csv,
      step41_countries_low_holm.csv, appendix_crosswalk_full.csv
"""
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf
from scipy import stats

warnings.filterwarnings("ignore")
BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
OUT = BASE / "5_results"
pct = lambda x: 100 * (np.exp(x) - 1)
inp = lambda a, b: pl.col("period").is_between(pl.lit(a), pl.lit(b))
cat = lambda cs, n: pl.concat_str(cs, separator="_").cast(pl.Categorical).to_physical().alias(n)
LIGHT_TRUCKS = ["870421", "870431", "870441", "870451", "870460"]

RAW = pl.read_parquet(BASE / "7_processed_data" / "panel_v1.parquet",
                      columns=["exporter", "hs6", "dest", "us", "period", "group_v4", "value_eur", "frontload"])
RAW = RAW.filter(~pl.col("frontload") & pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", "4 CONTROL"])
                 & (pl.col("dest") != "CN"))
M = (RAW.with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
        .group_by(["exporter", "hs6", "market", "period", "group_v4"]).agg((pl.col("value_eur").sum() / 1e6).alias("v"))
        .with_columns(cat(["exporter", "hs6", "market"], "f1"), cat(["exporter", "hs6", "period"], "f2"),
                      cat(["market", "period"], "f3"), cat(["group_v4", "market", pl.col("period").str.slice(4, 2)], "f4"),
                      cat(["exporter", "hs6"], "cl_pair"), pl.col("hs6").cast(pl.Categorical).to_physical().alias("cl_hs6")))
cars, parts, us = pl.col("group_v4") == "1 CARS", pl.col("group_v4") == "2 PARTS_CORE", pl.col("market") == "US"
TWO = {"CARS_HIGH": cars & us & inp("202504", "202509"), "CARS_LOW": cars & us & inp("202510", "202606"),
       "PARTS_HIGH": parts & us & inp("202504", "202509"), "PARTS_LOW": parts & us & inp("202510", "202606")}
THREE = {"CARS_AprJul": cars & us & inp("202504", "202507"), "CARS_AugSep": cars & us & inp("202508", "202509"),
         "CARS_LOW": cars & us & inp("202510", "202606"),
         "PARTS_AprJul": parts & us & inp("202504", "202507"), "PARTS_AugSep": parts & us & inp("202508", "202509"),
         "PARTS_LOW": parts & us & inp("202510", "202606")}


def fit(P, terms, vcov):
    P = P.with_columns([c.cast(pl.Float64).alias(n) for n, c in terms.items()])
    names = [n for n in terms if P[n].sum() > 0]
    df = P.select(["v", "f1", "f2", "f3", "f4", "cl_pair", "cl_hs6"] + names).to_pandas()
    return pf.fepois(f"v ~ {' + '.join(names)} | f1 + f2 + f3 + f4", data=df, vcov=vcov)


# ---------------------------------------------------------------- A
rows = []
print("Part A. Cars minus core parts (log points; % = ratio of the two relative changes minus 1)")
for model, terms, pairs in [("two periods", TWO, [("CARS_HIGH", "PARTS_HIGH"), ("CARS_LOW", "PARTS_LOW")]),
                            ("three periods", THREE, [("CARS_AprJul", "PARTS_AprJul"), ("CARS_AugSep", "PARTS_AugSep"),
                                                      ("CARS_LOW", "PARTS_LOW")])]:
    for vlab, vc in [("exporter x HS6", {"CRV1": "cl_pair"}), ("HS6", {"CRV1": "cl_hs6"})]:
        m = fit(M, terms, vc)
        b, V = m.coef(), m._vcov
        names = list(b.index)
        G = int(m._G[0]) if hasattr(m, "_G") else None
        print(f"\n  {model}, clustered by {vlab} ({m._N:,} obs.; {G:,} clusters)")
        for c, p in pairs:
            i, j = names.index(c), names.index(p)
            d = b[c] - b[p]
            se = float(np.sqrt(V[i, i] + V[j, j] - 2 * V[i, j]))
            z = d / se
            pv = 2 * stats.norm.sf(abs(z))
            pv_t = 2 * stats.t.sf(abs(z), df=G - 1) if G else np.nan
            for t in (c, p):
                k = names.index(t)
                se_t = float(np.sqrt(V[k, k]))
                rows.append(dict(model=model, clustering=vlab, term=t, estimate=b[t], se=se_t,
                                 pct=pct(b[t]), lo=pct(b[t] - 1.96 * se_t), hi=pct(b[t] + 1.96 * se_t),
                                 p=2 * stats.norm.sf(abs(b[t] / se_t)), nobs=m._N, clusters=G))
            rows.append(dict(model=model, clustering=vlab, term=f"{c} - {p}", estimate=d, se=se, pct=pct(d),
                             lo=pct(d - 1.96 * se), hi=pct(d + 1.96 * se), p=pv, p_t=pv_t, nobs=m._N, clusters=G))
            print(f"    {c:12s} {pct(b[c]):+6.1f}%   {p:13s} {pct(b[p]):+6.1f}%   difference {d:+.3f} log points "
                  f"(se {se:.3f}), {pct(d):+.1f}% [{pct(d - 1.96 * se):+.1f}, {pct(d + 1.96 * se):+.1f}], p = {pv:.4f}"
                  f"{f' (t with {G - 1} df: {pv_t:.4f})' if G else ''}")
pd.DataFrame(rows).to_csv(OUT / "step41_contrasts.csv", index=False)

# ---------------------------------------------------------------- B
m = fit(M.filter(~pl.col("hs6").is_in(LIGHT_TRUCKS)), TWO, {"CRV1": "cl_pair"})
T = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
T.to_csv(OUT / "step41_no_light_trucks.csv", index=False)
print(f"\nPart B. Without light trucks ({m._N:,} obs.)")
for _, r in T.iterrows():
    print(f"  {r['term']:11s} {pct(r['Estimate']):+6.1f}%  [{pct(r['2.5%']):+.0f}, {pct(r['97.5%']):+.0f}]  p = {r['Pr(>|t|)']:.3f}")

# ---------------------------------------------------------------- C
us24 = (RAW.filter(cars & pl.col("us") & pl.col("period").str.starts_with("2024"))
           .group_by("hs6").agg((pl.col("value_eur").sum() / 1e6).alias("us2024")).sort("us2024", descending=True))
codes = us24.filter(pl.col("us2024") > 0)["hs6"].to_list()
loo = []
for c in codes:
    m = fit(M.filter(pl.col("hs6") != c), TWO, {"CRV1": "cl_pair"})
    b = m.coef()
    loo.append(dict(dropped=c, us2024_eur_m=float(us24.filter(pl.col("hs6") == c)["us2024"][0]),
                    cars_high=b["CARS_HIGH"], cars_low=b.get("CARS_LOW", np.nan)))
    print(f"  without {c}: cars Apr-Sep 2025 {pct(b['CARS_HIGH']):+.1f}%")
L = pd.DataFrame(loo)
g = len(L)
for col in ["cars_high", "cars_low"]:
    th = L[col].to_numpy()
    se_j = float(np.sqrt((g - 1) / g * np.sum((th - th.mean()) ** 2)))
    L[f"jackknife_se_{col}"] = se_j
L.to_csv(OUT / "step41_leave_one_code_out.csv", index=False)
full = fit(M, TWO, {"CRV1": "cl_pair"}).coef()
print(f"\nPart C. Leave one car code out ({g} codes with 2024 US exports): Apr-Sep 2025 estimate from "
      f"{pct(L['cars_high'].min()):+.1f}% (without {L.loc[L['cars_high'].idxmin(), 'dropped']}) to "
      f"{pct(L['cars_high'].max()):+.1f}% (without {L.loc[L['cars_high'].idxmax(), 'dropped']})")
for col, t in [("cars_high", "CARS_HIGH"), ("cars_low", "CARS_LOW")]:
    se_j = L[f"jackknife_se_{col}"].iloc[0]
    b = full[t]
    p_t = 2 * stats.t.sf(abs(b / se_j), df=g - 1)
    print(f"  {t}: estimate {pct(b):+.1f}%, jackknife se over car codes {se_j:.3f}, "
          f"interval [{pct(b - stats.t.ppf(0.975, g - 1) * se_j):+.1f}, {pct(b + stats.t.ppf(0.975, g - 1) * se_j):+.1f}], "
          f"p (t, {g - 1} df) = {p_t:.4f}")

# ---------------------------------------------------------------- D
S = pd.read_csv(OUT / "step30_checks.csv")
out = []
for per in ["HIGH", "LOW"]:
    s = S[S["check"].str.startswith("C") & S["term"].str.endswith(f"_{per}") & S["term"].str.startswith("CARS")].copy()
    s = s.sort_values("Pr(>|t|)")
    k = len(s)
    s["p_holm"] = np.minimum(1, np.maximum.accumulate((k - np.arange(k)) * s["Pr(>|t|)"].to_numpy()))
    s["period"] = per
    out.append(s)
D = pd.concat(out)
D.to_csv(OUT / "step41_countries_low_holm.csv", index=False)
print("\nPart D. Exporter estimates with Holm-adjusted p-values (8 tests per period)")
for _, r in D.iterrows():
    print(f"  {r['period']:4s} {r['term']:14s} {pct(r['Estimate']):+6.1f}%  [{pct(r['2.5%']):+.0f}, {pct(r['97.5%']):+.0f}]  "
          f"p = {r['Pr(>|t|)']:.3f}  Holm p = {r['p_holm']:.3f}")
E = pd.read_csv(OUT / "table_by_exporter.csv")
print("\nGermany against the rest of the EU (step27):")
print(E.to_string(index=False)[:1500])

# ---------------------------------------------------------------- E
G = pd.read_csv(BASE / "7_processed_data" / "hs6_groups_v4.csv", dtype=str)
X = pd.read_csv(OUT / "appendix_crosswalk.csv", dtype=str)[["hs6", "listed_us_codes", "n_listed_lines", "cn8_codes"]]
G.loc[G["hs6"].isin(["870370", "870451"]), "annex_status"] = "full"
v = (pl.read_parquet(BASE / "7_processed_data" / "panel_v1.parquet", columns=["hs6", "us", "period", "value_eur"])
       .filter(pl.col("us") & pl.col("period").str.starts_with("2024")).group_by("hs6")
       .agg((pl.col("value_eur").sum() / 1e6).round(1).alias("eu_exports_to_us_2024_eur_m")).to_pandas())


def reason(r):
    g = r["group_v4"]
    if g == "1 CARS":
        return "Vehicle code listed in Annex I of Proclamation 10908 (note 33(b))"
    if g == "2 PARTS_CORE":
        return "Listed in full in Annex I (note 33(g)); used only in vehicles"
    if g == "3 UNCERTAIN":
        if r["annex_status"] == "partial":
            return "Only some US tariff lines listed in Annex I; exposure cannot be assigned at HS6"
        if r["annex_status"] == "full":
            return "Listed in full, but also used outside vehicles; the parts tariff applies only to vehicle use"
        return "Car code not listed in Annex I"
    if g == "4 CONTROL":
        return "Not listed; general tariff only" + (" (also a civil-aircraft good)" if r["aircraft_dual_use"] == "true" else "")
    if g == "4m CONTROL_METAL":
        w = r["metal_2025_from"]
        return "Steel or aluminium derivative under Section 232 " + (f"from {w}" if w != "never" else "from 6 Apr 2026")
    if g == "5 EXEMPT_ELECTRONICS":
        return "Exempt from the general tariff for any use"
    if g == "6 HEAVY_VEHICLES":
        return "Truck, tractor, bus or truck part listed in Proclamation 10984"
    if g == "9 DROPPED":
        return "Aircraft engines and parts (8411, 8412): separate tariff treatment"
    return ""


G["reason"] = G.apply(reason, axis=1)
F = G.merge(X, on="hs6", how="left").merge(v, on="hs6", how="left").fillna({"eu_exports_to_us_2024_eur_m": 0.0})
F.sort_values(["group_v4", "hs6"]).to_csv(OUT / "appendix_crosswalk_full.csv", index=False)
print(f"\nPart E. Full crosswalk: {len(F)} HS6 codes")
print(pd.crosstab(F["group_v4"], F["annex_status"], margins=True))
print("\nSaved 5_results/step41_*.csv and appendix_crosswalk_full.csv")
