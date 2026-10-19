"""Step 6.1: Checks requested by the reviewer report (18 Oct 2026).

Each part is independent: if one fails, the script prints the error and continues.
  A  Variance estimators side by side (main model): exporter x HS6 pairs (main), HS6, exporter, two-way
     (exporter and HS6), heteroskedasticity-robust without clustering. Prints the variance pieces that explain
     why two-way intervals can be narrower than one-way intervals.
  B  Pre-trends: (1) linear pre-trend test on 2023-2024 only; (2) quarterly event study with 2023 quarters and
     a joint test that the four 2023 quarters are zero; (3) appendix table of the monthly Figure 1 coefficients.
  C  Comparison markets: leave each comparison market out in turn; all non-EU markets including China;
     high-income markets only (UK, Switzerland, Japan, Korea, Canada).
  D  Control groups: clean control (main); clean + metal-derivative codes; without civil-aircraft dual-use codes;
     only control products above the median 2024 US export value; without HS 841989.
  E  Prices: (1) direct size test, car price effect interacted with a predetermined large-flow indicator (top
     half of 2023-24 value among car flows) and with log 2023-24 flow value; (2) flows observed in most months;
     (3) kilograms per car as outcome (does the car mix shift toward lighter or heavier models?).
  F  Volume effect by pre-tariff unit value: continuous interaction with log EUR per car; distribution of
     EUR per car; number of car flows by segment.
  G  Exporter countries: number of car flows and 2024 value per country; Holm-adjusted p-values for the
     country estimates of step30.
  H  Exits among regular flows (flows that shipped in at least 6 months of 2024).
  I  Tariff regimes with Aug-Sep 2025 separate: Apr-Jul 2025, Aug-Sep 2025, Oct 2025-Feb 2026, Mar-Jun 2026.
  J  Panel coverage of Apr 2025-Jun 2026 export value, by group and market.
  K  Product crosswalk appendix: every HS6 in the cars, core parts, uncertain and control groups with legal
     status, listed US codes, CN8 codes and 2024 EU exports to the US.
Main specification throughout: US vs other non-EU markets without China, Jan-Mar 2025 out, flow,
exporter x HS6 x month, market x month and group x market x calendar-month fixed effects.
Saves 5_results/step37_*.csv and 5_results/appendix_*.csv. Run time: roughly 45-90 minutes.
"""
import os
from pathlib import Path
import time
import traceback
import warnings
import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf

warnings.filterwarnings("ignore")
BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
OUT = BASE / "5_results"
EV = ["870360", "870370", "870380"]
HIGH, LOW = ("202504", "202509"), ("202510", "202606")
pct = lambda x: 100 * (np.exp(x) - 1)
inp = lambda p: pl.col("period").is_between(pl.lit(p[0]), pl.lit(p[1]))
cat = lambda cs, n: pl.concat_str(cs, separator="_").cast(pl.Categorical).to_physical().alias(n)
START = time.time()

RAW = pl.read_parquet(BASE / "7_processed_data" / "panel_v1.parquet",
                      columns=["exporter", "cn8", "hs6", "dest", "us", "period", "group_v4", "value_eur",
                               "qty_units", "qty_kg", "frontload", "aircraft_dual_use"])
RAW = RAW.filter(~pl.col("frontload"))


def section(name):
    print("\n" + "=" * 100 + f"\n{name}   [{(time.time() - START) / 60:.0f} min]\n" + "=" * 100)


def build(data, dests=None, control=("4 CONTROL",)):
    d = data.filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", *control]))
    d = d.filter(pl.col("dest") != "CN") if dests is None else d.filter(pl.col("dest").is_in(["US", *dests]))
    return (d.with_columns(pl.when(pl.col("group_v4").is_in(list(control))).then(pl.lit("4 CONTROL"))
                             .otherwise(pl.col("group_v4")).alias("g"),
                           pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
             .group_by(["exporter", "hs6", "market", "period", "g"]).agg((pl.col("value_eur").sum() / 1e6).alias("v")))


def add_fe(P):
    return P.with_columns(cat(["exporter", "hs6", "market"], "f1"), cat(["exporter", "hs6", "period"], "f2"),
                          cat(["market", "period"], "f3"),
                          pl.concat_str([pl.col("g"), pl.col("market"), pl.col("period").str.slice(4, 2)], separator="_")
                            .cast(pl.Categorical).to_physical().alias("f4"),
                          cat(["exporter", "hs6"], "cl_pair"), pl.col("hs6").cast(pl.Categorical).to_physical().alias("cl_hs6"),
                          pl.col("exporter").cast(pl.Categorical).to_physical().alias("cl_exp"))


cars, parts, us = pl.col("g") == "1 CARS", pl.col("g") == "2 PARTS_CORE", pl.col("market") == "US"
MAIN_TERMS = {"CARS_HIGH": cars & us & inp(HIGH), "CARS_LOW": cars & us & inp(LOW),
              "PARTS_HIGH": parts & us & inp(HIGH), "PARTS_LOW": parts & us & inp(LOW)}


def fit(P, terms, vcov=None, extra=None):
    P = P.with_columns([(c.cast(pl.Float64) if isinstance(c, pl.Expr) else c).alias(n) for n, c in terms.items()])
    names = [n for n in terms if P[n].abs().sum() > 0]
    keep = ["v", "f1", "f2", "f3", "f4", "cl_pair", "cl_hs6", "cl_exp"] + names
    m = pf.fepois(f"v ~ {' + '.join(names)} | f1 + f2 + f3 + f4", data=P.select(keep).to_pandas(),
                  vcov=vcov or {"CRV1": "cl_pair"})
    T = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
    T["nobs"] = m._N
    return m, T


def line(T, t, label=None):
    r = T.set_index("term").loc[t]
    s = "***" if r["Pr(>|t|)"] < 0.01 else "**" if r["Pr(>|t|)"] < 0.05 else "*" if r["Pr(>|t|)"] < 0.1 else ""
    return (f"  {label or t:34s} {pct(r['Estimate']):+6.1f}{s:3s} [{pct(r['2.5%']):+4.0f}, {pct(r['97.5%']):+4.0f}]"
            f"  b={r['Estimate']:+.3f} se={r['Std. Error']:.3f} p={r['Pr(>|t|)']:.3f}")


def run(name, f):
    section(name)
    try:
        f()
    except Exception:
        print("  FAILED:\n" + traceback.format_exc())


MAIN = add_fe(build(RAW))
res = []


# ---------------------------------------------------------------- A
def part_a():
    rows = []
    for lab, vc in [("(a) exporter x HS6 pairs (main)", {"CRV1": "cl_pair"}), ("(b) HS6", {"CRV1": "cl_hs6"}),
                    ("(c) exporter", {"CRV1": "cl_exp"}), ("(d) two-way, exporter and HS6", {"CRV1": "cl_exp+cl_hs6"}),
                    ("(e) heteroskedasticity-robust, no clustering", "hetero")]:
        m, T = fit(MAIN, MAIN_TERMS, vc)
        T["spec"] = lab
        rows.append(T)
        print(f"{lab}  ({m._N:,} obs.)")
        for t in MAIN_TERMS:
            print(line(T, t))
    R = pd.concat(rows)
    R.to_csv(OUT / "step37_A_variance.csv", index=False)
    se = R.pivot(index="term", columns="spec", values="Std. Error")
    print("\nTwo-way variance = V(exporter) + V(HS6) - V(exporter x HS6). If V(exporter) is small and V(HS6) is below")
    print("V(exporter x HS6), the two-way variance is smaller than the pair-clustered one. Variance pieces:")
    for t in MAIN_TERMS:
        a, b, c, d = (se.loc[t, k] ** 2 for k in ["(c) exporter", "(b) HS6", "(a) exporter x HS6 pairs (main)",
                                                  "(d) two-way, exporter and HS6"])
        print(f"  {t:11s} V_exp={a:.4f}  V_hs6={b:.4f}  V_pair={c:.4f}  V_exp+V_hs6-V_pair={a + b - c:.4f}  "
              f"reported two-way={d:.4f}")


# ---------------------------------------------------------------- B
def part_b():
    pre = MAIN.filter(pl.col("period") < "202501").with_columns(
        ((pl.col("period").str.slice(0, 4).cast(pl.Int32) - 2023) * 12 + pl.col("period").str.slice(4, 2).cast(pl.Int32)
         - 12.5).alias("t"))
    m, T = fit(pre, {"CARS_TREND": (cars & us).cast(pl.Float64) * pl.col("t"),
                     "PARTS_TREND": (parts & us).cast(pl.Float64) * pl.col("t")})
    print("(1) Linear pre-trend, 2023-2024 only (coefficient per month; x12 = per year):")
    for t in ["CARS_TREND", "PARTS_TREND"]:
        r = T.set_index("term").loc[t]
        print(f"  {t}: {r['Estimate']:+.4f} per month ({100 * 12 * r['Estimate']:+.1f}% per year), "
              f"se {r['Std. Error']:.4f}, p = {r['Pr(>|t|)']:.3f}")
    T.to_csv(OUT / "step37_B1_pretrend.csv", index=False)
    q = pl.col("period").str.slice(0, 4) + "Q" + ((pl.col("period").str.slice(4, 2).cast(pl.Int32) - 1) // 3 + 1).cast(pl.Utf8)
    Pq = MAIN.with_columns(q.alias("q"))
    terms = {}
    for g, short in [(cars, "CARS"), (parts, "PARTS")]:
        for qq in ["2023Q1", "2023Q2", "2023Q3", "2023Q4"]:
            terms[f"{short}_{qq}"] = g & us & (pl.col("q") == qq)
        terms[f"{short}_HIGH"], terms[f"{short}_LOW"] = g & us & inp(HIGH), g & us & inp(LOW)
    m, T = fit(Pq, terms)
    T.to_csv(OUT / "step37_B2_quarters.csv", index=False)
    print("\n(2) 2023 quarters relative to 2024, with the two post periods in the same model:")
    for t in terms:
        print(line(T, t))
    names = list(T["term"])
    for short in ["CARS", "PARTS"]:
        idx = [names.index(f"{short}_{qq}") for qq in ["2023Q1", "2023Q2", "2023Q3", "2023Q4"]]
        R = np.zeros((4, len(names)))
        for i, j in enumerate(idx):
            R[i, j] = 1.0
        print(f"  Joint test, {short}, four 2023 quarters = 0: p = {float(m.wald_test(R=R)['pvalue']):.3f}"
              " (cluster-robust Wald tests over-reject with few treated clusters; Step 3.3)")
    E = pd.read_csv(OUT / "fig1_event_study.csv")
    E = E[["group", "month", "Estimate", "Std. Error", "Pr(>|t|)", "2.5%", "97.5%"]]
    for a, b in [("Estimate", "pct"), ("2.5%", "pct_low"), ("97.5%", "pct_high")]:
        E[b] = pct(E[a]).round(1)
    E.to_csv(OUT / "appendix_event_study.csv", index=False)
    print("\n(3) Appendix table of Figure 1 coefficients saved: 5_results/appendix_event_study.csv "
          "(reference: same calendar month of 2024)")


# ---------------------------------------------------------------- C
def part_c():
    alld = [d for d in RAW["dest"].unique().to_list() if d not in ("US", "CN")]
    specs = [("main: all except China", None)] + [(f"without {d}", [x for x in alld if x != d]) for d in sorted(alld)]
    specs += [("all non-EU markets incl. China", alld + ["CN"]), ("high-income only: GB CH JP KR CA",
                                                                 [d for d in ["GB", "CH", "JP", "KR", "CA"] if d in alld])]
    rows = []
    for lab, ds in specs:
        m, T = fit(add_fe(build(RAW, ds)), MAIN_TERMS)
        T["spec"] = lab
        rows.append(T)
        print(f"{lab} ({m._N:,} obs.)")
        print(line(T, "CARS_HIGH") + "\n" + line(T, "PARTS_HIGH"))
    pd.concat(rows).to_csv(OUT / "step37_C_markets.csv", index=False)


# ---------------------------------------------------------------- D
def part_d():
    us24 = (RAW.filter(pl.col("us") & pl.col("period").str.starts_with("2024") & (pl.col("group_v4") == "4 CONTROL"))
               .group_by("hs6").agg(pl.col("value_eur").sum().alias("u")))
    big = us24.filter(pl.col("u") >= us24["u"].median())["hs6"].to_list()
    air = RAW.filter(pl.col("aircraft_dual_use").cast(pl.Utf8).is_in(["true", "True", "1", "full", "partial"]))["hs6"].unique().to_list()
    specs = [("clean control (main)", RAW, ("4 CONTROL",)),
             ("clean + metal-derivative codes", RAW, ("4 CONTROL", "4m CONTROL_METAL")),
             ("without civil-aircraft dual-use codes", RAW.filter(~((pl.col("group_v4") == "4 CONTROL") & pl.col("hs6").is_in(air))), ("4 CONTROL",)),
             ("control above median 2024 US value", RAW.filter(~((pl.col("group_v4") == "4 CONTROL") & ~pl.col("hs6").is_in(big))), ("4 CONTROL",)),
             ("without HS 841989", RAW.filter(pl.col("hs6") != "841989"), ("4 CONTROL",))]
    print(f"Civil-aircraft dual-use HS6 codes found: {len(air)}; control codes above median US value: {len(big)}")
    rows = []
    for lab, d, ctl in specs:
        m, T = fit(add_fe(build(d, None, ctl)), MAIN_TERMS)
        T["spec"] = lab
        rows.append(T)
        print(f"{lab} ({m._N:,} obs.)")
        print(line(T, "CARS_HIGH") + "\n" + line(T, "PARTS_HIGH"))
    pd.concat(rows).to_csv(OUT / "step37_D_controls.csv", index=False)


# ---------------------------------------------------------------- E
def part_e():
    PER = [("202504", "202507", "AprJul25"), ("202508", "202509", "AugSep25"), ("202510", "202512", "OctDec25"),
           ("202601", "202603", "JanMar26"), ("202604", "202606", "AprJun26")]
    P = (RAW.filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", "4 CONTROL"]) & (pl.col("dest") != "CN"))
            .with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
            .group_by(["exporter", "cn8", "market", "period", "group_v4"])
            .agg(pl.col("value_eur").sum(), pl.col("qty_units").sum(), pl.col("qty_kg").sum()))
    car = pl.col("group_v4") == "1 CARS"
    P = (P.with_columns(pl.when(car).then(pl.col("qty_units")).otherwise(pl.col("qty_kg")).alias("q"))
          .filter((pl.col("value_eur") > 0) & pl.when(car).then(pl.col("q") >= 5).otherwise(pl.col("q") >= 100))
          .with_columns((pl.col("value_eur") / pl.col("q")).log().alias("lp"),
                        pl.when(car & (pl.col("qty_units") > 0)).then((pl.col("qty_kg") / pl.col("qty_units")).log())
                          .otherwise(None).alias("lkg"),
                        pl.concat_str(["exporter", "cn8", "market"], separator="_").alias("flow")))
    base = (P.filter(pl.col("period") < "202501").group_by("flow")
             .agg(pl.col("lp").median().alias("lp_med"), pl.col("value_eur").sum().alias("w")))
    P = P.join(base, on="flow", how="inner").filter((pl.col("lp") - pl.col("lp_med")).abs() <= np.log(5))
    medw = P.filter(car).group_by("flow").agg(pl.col("w").first())["w"].median()
    mlog = float(np.log(P.filter(car).group_by("flow").agg(pl.col("w").first())["w"].to_numpy()).mean())
    P = P.with_columns((pl.col("w") > medw).cast(pl.Float64).alias("large"),
                       (pl.col("w").log() - mlog).alias("lsize"))
    nper = P.group_by("flow").agg(pl.col("period").n_unique().alias("n"))
    P = P.join(nper, on="flow")
    P = P.with_columns(cat(["flow"], "f1"), cat(["exporter", "cn8", "period"], "f2"), cat(["market", "period"], "f3"),
                       pl.concat_str([pl.col("group_v4"), pl.col("market"), pl.col("period").str.slice(4, 2)], separator="_")
                         .cast(pl.Categorical).to_physical().alias("f4"),
                       cat(["exporter", "cn8"], "cl"), (pl.col("w") / pl.col("w").mean()).alias("wt"))
    is_us = pl.col("market") == "US"
    base_terms = {}
    for g, short in [("1 CARS", "CARS"), ("2 PARTS_CORE", "PARTS")]:
        for a, b, per in PER:
            base_terms[f"{short}_{per}"] = ((pl.col("group_v4") == g) & is_us & pl.col("period").is_between(pl.lit(a), pl.lit(b))).cast(pl.Float64)

    def ols(D, terms, y="lp", weights=None):
        D = D.with_columns([e.alias(n) for n, e in terms.items()])
        names = [n for n in terms if D[n].abs().sum() > 0]
        df = D.select([y, "wt", "cl", "f1", "f2", "f3", "f4"] + names).drop_nulls().to_pandas()
        m = pf.feols(f"{y} ~ {' + '.join(names)} | f1 + f2 + f3 + f4", data=df, vcov={"CRV1": "cl"},
                     **({"weights": "wt"} if weights else {}))
        T = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
        return m, T

    rows = []
    t1 = dict(base_terms)
    for a, b, per in PER:
        t1[f"CARS_{per}_xLARGE"] = base_terms[f"CARS_{per}"] * pl.col("large")
    m, T = ols(P, t1)
    T["spec"] = "unweighted, x large-flow indicator"
    rows.append(T)
    print(f"(1a) Unweighted price model with CARS x period x LARGE (large = 2023-24 flow value above the car-flow "
          f"median, EUR {medw / 1e6:.1f} m). Effect for small flows and the extra effect for large flows (log points):")
    for a, b, per in PER:
        r0, r1 = T.set_index("term").loc[f"CARS_{per}"], T.set_index("term").loc[f"CARS_{per}_xLARGE"]
        print(f"  {per}: small {r0['Estimate']:+.3f} (p={r0['Pr(>|t|)']:.3f}); large minus small {r1['Estimate']:+.3f} "
              f"(se {r1['Std. Error']:.3f}, p={r1['Pr(>|t|)']:.3f})")
    t2 = dict(base_terms)
    for a, b, per in PER:
        t2[f"CARS_{per}_xLSIZE"] = base_terms[f"CARS_{per}"] * pl.col("lsize")
    m, T = ols(P, t2)
    T["spec"] = "unweighted, x log flow size"
    rows.append(T)
    print("(1b) Continuous: CARS x period x log 2023-24 flow value (centred). Extra effect per log point of size:")
    for a, b, per in PER:
        r1 = T.set_index("term").loc[f"CARS_{per}_xLSIZE"]
        print(f"  {per}: {r1['Estimate']:+.4f} (se {r1['Std. Error']:.4f}, p={r1['Pr(>|t|)']:.3f})")
    Pb = P.filter(pl.col("n") >= 30)
    for wlab, w in [("unweighted", False), ("weighted", True)]:
        m, T = ols(Pb, base_terms, weights=w)
        T["spec"] = f"flows with prices in at least 30 of 39 months, {wlab}"
        rows.append(T)
        print(f"(2) Flows with prices in at least 30 of the 39 months, {wlab} ({m._N:,} obs.):")
        for a, b, per in PER[:3]:
            print(line(T, f"CARS_{per}"))
    pd.concat(rows).to_csv(OUT / "step37_E_prices.csv", index=False)
    try:
        m, T = ols(P.filter(car), {k: v for k, v in base_terms.items() if k.startswith("CARS")}, y="lkg", weights=True)
        T["spec"] = "kg per car (weighted)"
        rows.append(T)
        print("(3) Log kilograms per car, weighted (a shift to heavier or lighter models would show up here):")
        for a, b, per in PER:
            print(line(T, f"CARS_{per}"))
        pd.concat(rows).to_csv(OUT / "step37_E_prices.csv", index=False)
    except Exception as e:
        print(f"(3) kg per car: failed ({type(e).__name__})")


# ---------------------------------------------------------------- F
def part_f():
    price = (RAW.filter((pl.col("group_v4") == "1 CARS") & (pl.col("period") < "202501") & (pl.col("dest") != "CN"))
                .group_by(["exporter", "hs6"]).agg(pl.col("value_eur").sum().alias("val"), pl.col("qty_units").sum().alias("n"))
                .filter(pl.col("n") >= 50).with_columns((pl.col("val") / pl.col("n")).alias("eur_per_car")))
    us24 = (RAW.filter((pl.col("group_v4") == "1 CARS") & pl.col("us") & pl.col("period").str.starts_with("2024"))
               .group_by(["exporter", "hs6"]).agg(pl.col("value_eur").sum().alias("us24")))
    D = price.join(us24, on=["exporter", "hs6"], how="left").with_columns(pl.col("us24").fill_null(0))
    w = D.filter(pl.col("us24") > 0).sort("eur_per_car")
    cw = (w["us24"].cum_sum() / w["us24"].sum()).to_numpy()
    print("Distribution of pre-tariff EUR per car across car flows with US exports, weighted by 2024 US value:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"  {int(q * 100)}th percentile: EUR {w['eur_per_car'][int(np.searchsorted(cw, q))]:,.0f}")
    for thr in [40000, 50000, 60000]:
        a = w.filter(pl.col("eur_per_car") >= thr)
        print(f"  Threshold EUR {thr:,}: {a.height} flows at or above ({100 * a['us24'].sum() / w['us24'].sum():.0f}% of US value), "
              f"{w.height - a.height} below")
    mu = np.log(w["eur_per_car"]).mean()
    P = add_fe(build(RAW)).join(price.select(["exporter", "hs6", "eur_per_car"]), on=["exporter", "hs6"], how="left")
    lpc = (pl.col("eur_per_car").log() - mu).fill_null(0.0)
    has = pl.col("eur_per_car").is_not_null().cast(pl.Float64)
    terms = {"CARS_HIGH": cars & us & inp(HIGH), "CARS_LOW": cars & us & inp(LOW),
             "CARS_HIGH_xLOGPRICE": (cars & us & inp(HIGH)).cast(pl.Float64) * lpc * has,
             "CARS_LOW_xLOGPRICE": (cars & us & inp(LOW)).cast(pl.Float64) * lpc * has,
             "PARTS_HIGH": parts & us & inp(HIGH), "PARTS_LOW": parts & us & inp(LOW)}
    m, T = fit(P, terms)
    T.to_csv(OUT / "step37_F_unitvalue.csv", index=False)
    print("Main model with CARS x period x log pre-tariff EUR per car (centred at the US-weighted mean log price):")
    for t in ["CARS_HIGH", "CARS_LOW"]:
        print(line(T, t, t + " at mean log price"))
    for t in [x for x in ["CARS_HIGH_xLOGPRICE", "CARS_LOW_xLOGPRICE"] if x in set(T["term"])]:
        r = T.set_index("term").loc[t]
        print(f"  {t}: {r['Estimate']:+.3f} per log point of price (se {r['Std. Error']:.3f}, p = {r['Pr(>|t|)']:.3f}); "
              f"a car 50% dearer: {pct(r['Estimate'] * np.log(1.5)):+.1f}% relative to the mean")


# ---------------------------------------------------------------- G
def part_g():
    c = (RAW.filter((pl.col("group_v4") == "1 CARS") & pl.col("us") & pl.col("period").str.starts_with("2024"))
            .group_by(["exporter", "hs6"]).agg(pl.col("value_eur").sum().alias("v")).filter(pl.col("v") > 0))
    t = (c.group_by("exporter").agg(pl.len().alias("car_flows_2024"), (pl.col("v").sum() / 1e9).round(2).alias("bn_2024"))
          .sort("bn_2024", descending=True))
    t.write_csv(OUT / "step37_G_countries.csv")
    print("Car flows (exporter x HS6) with 2024 exports to the US, by exporter:")
    print(t.head(12))
    S = pd.read_csv(OUT / "step30_checks.csv")
    S = S[S["check"].str.startswith("C") & S["term"].str.endswith("_HIGH") & S["term"].str.startswith("CARS")]
    p = S.sort_values("Pr(>|t|)")["Pr(>|t|)"].to_numpy()
    k = len(p)
    holm = np.minimum(1, np.maximum.accumulate((k - np.arange(k)) * p))
    S = S.sort_values("Pr(>|t|)").assign(p_holm=holm)
    print("\nCountry estimates Apr-Sep 2025 (step30), unadjusted and Holm-adjusted p-values (8 tests):")
    for _, r in S.iterrows():
        print(f"  {r['term']:14s} {pct(r['Estimate']):+6.1f}%  p = {r['Pr(>|t|)']:.3f}  Holm p = {r['p_holm']:.3f}")


# ---------------------------------------------------------------- H
def part_h():
    c = (RAW.filter((pl.col("group_v4") == "1 CARS") & (pl.col("dest") != "CN"))
            .with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
            .group_by(["exporter", "hs6", "market", "period"]).agg(pl.col("value_eur").sum()))
    y24 = (c.filter(pl.col("period").str.starts_with("2024"))
            .group_by(["exporter", "hs6", "market"]).agg(pl.col("value_eur").sum().alias("v24"),
                                                          (pl.col("value_eur") > 0).sum().alias("months24")))
    low = c.filter(inp(LOW)).group_by(["exporter", "hs6", "market"]).agg(pl.col("value_eur").sum().alias("vlow"))
    D = y24.filter(pl.col("v24") > 0).join(low, on=["exporter", "hs6", "market"], how="left").with_columns(
        pl.col("vlow").fill_null(0), (pl.col("vlow").fill_null(0) == 0).alias("exit"))
    for lab, d in [("all flows with 2024 exports", D), ("regular flows: exports in at least 6 months of 2024",
                                                         D.filter(pl.col("months24") >= 6))]:
        t = d.group_by("market").agg(pl.len().alias("flows"), pl.col("exit").sum().alias("exits"),
                                     (100 * pl.col("exit").mean()).round(1).alias("pct_exit"),
                                     (100 * pl.col("v24").filter(pl.col("exit")).sum() / pl.col("v24").sum()).round(2)
                                     .alias("pct_2024_value_in_exits")).sort("market")
        print(lab)
        print(t)


# ---------------------------------------------------------------- I
def part_i():
    P = (RAW.filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", "4 CONTROL"]) & (pl.col("dest") != "CN"))
            .with_columns(pl.when((pl.col("group_v4") == "1 CARS") & pl.col("hs6").is_in(EV)).then(pl.lit("1b EV"))
                            .otherwise(pl.col("group_v4")).alias("g"),
                          pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market"))
            .group_by(["exporter", "hs6", "market", "period", "g"]).agg((pl.col("value_eur").sum() / 1e6).alias("v")))
    P = add_fe(P)
    REG = [("R1", "202504", "202507"), ("R1b", "202508", "202509"), ("R2", "202510", "202602"), ("R3", "202603", "202606")]
    terms = {}
    for g, short in [("1 CARS", "ICE"), ("1b EV", "EV"), ("2 PARTS_CORE", "PARTS")]:
        for r, a, b in REG:
            terms[f"{short}_{r}"] = (pl.col("g") == g) & us & pl.col("period").is_between(pl.lit(a), pl.lit(b))
    m, T = fit(P, terms)
    T.to_csv(OUT / "step37_I_regimes.csv", index=False)
    labels = {"R1": "Apr-Jul 2025 (cars 27.5%)", "R1b": "Aug-Sep 2025 (15% agreed, applied 25 Sep)",
              "R2": "Oct 2025-Feb 2026 (both 15%)", "R3": "Mar-Jun 2026 (cars 15%, control duty + 10%)"}
    for short, name in [("ICE", "cars without plug-in hybrid and battery-electric"), ("EV", "plug-in hybrid and battery-electric"),
                        ("PARTS", "core parts")]:
        print(name)
        for r, a, b in REG:
            if f"{short}_{r}" in set(T["term"]):
                print(line(T, f"{short}_{r}", labels[r]))
    names = list(T["term"])
    R = np.zeros((1, len(names)))
    R[0, names.index("ICE_R1")], R[0, names.index("ICE_R2")] = 1.0, -1.0
    print(f"Test cars without plug-in/electric, Apr-Jul 2025 = Oct 2025-Feb 2026: p = {float(m.wald_test(R=R)['pvalue']):.3f}")


# ---------------------------------------------------------------- J
def part_j():
    X = (pl.read_parquet(BASE / "7_processed_data" / "exports_ch84_87.parquet",
                         columns=["exporter", "cn8", "dest", "period", "value_eur"])
           .filter((pl.col("dest") != "EU") & (pl.col("period") >= "202504")))
    flows = RAW.select(["exporter", "cn8", "dest", "group_v4"]).unique()
    G = pl.read_csv(BASE / "7_processed_data" / "hs6_groups_v4.csv", schema_overrides={"hs6": pl.Utf8})
    X = (X.with_columns(pl.col("cn8").str.slice(0, 6).alias("hs6")).join(G.select(["hs6", "group_v4"]), on="hs6", how="left")
           .join(flows.select(["exporter", "cn8", "dest"]).with_columns(pl.lit(True).alias("kept")),
                 on=["exporter", "cn8", "dest"], how="left")
           .with_columns(pl.col("kept").fill_null(False),
                         pl.when(pl.col("dest") == "US").then(pl.lit("US")).otherwise(pl.lit("OTHER")).alias("market")))
    t = (X.filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", "4 CONTROL"]))
           .group_by(["group_v4", "market"]).agg((100 * pl.col("value_eur").filter(pl.col("kept")).sum()
                                                  / pl.col("value_eur").sum()).round(1).alias("pct_value_kept"))
           .sort(["group_v4", "market"]))
    t.write_csv(OUT / "step37_J_coverage.csv")
    print("Share of Apr 2025-Jun 2026 export value covered by flows that existed in 2023-2024:")
    print(t)


# ---------------------------------------------------------------- K
def part_k():
    G = pl.read_csv(BASE / "7_processed_data" / "hs6_groups_v4.csv", schema_overrides={"hs6": pl.Utf8})
    A = pl.read_csv(BASE / "3_tariff_documents" / "annex_I_codes_proclamation10908.csv",
                    schema_overrides={"hs4": pl.Utf8, "hs6": pl.Utf8, "us_code": pl.Utf8})
    a6 = A.filter(pl.col("hs6").is_not_null()).group_by("hs6").agg(
        pl.col("us_code").sort().str.join("; ").alias("listed_us_codes"), pl.len().alias("n_listed_lines"),
        pl.col("list").unique().sort().str.join("/").alias("annex_list"))
    a4 = (A.filter(pl.col("digits") == 4).group_by("hs4").agg(pl.col("us_code").sort().str.join("; ").alias("listed_hs4")))
    X = pl.read_parquet(BASE / "7_processed_data" / "exports_ch84_87.parquet", columns=["cn8", "dest", "period", "value_eur"])
    v = (X.filter((pl.col("dest") == "US") & pl.col("period").str.starts_with("2024"))
           .with_columns(pl.col("cn8").str.slice(0, 6).alias("hs6"))
           .group_by("hs6").agg((pl.col("value_eur").sum() / 1e6).round(1).alias("eu_exports_to_us_2024_eur_m"),
                                pl.col("cn8").unique().sort().str.join("; ").alias("cn8_codes")))
    K = (G.join(a6, on="hs6", how="left").with_columns(pl.col("hs6").str.slice(0, 4).alias("hs4"))
          .join(a4, on="hs4", how="left").join(v, on="hs6", how="left")
          .filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE", "3 UNCERTAIN", "4 CONTROL"]))
          .sort(["group_v4", "hs6"]))
    K.write_csv(OUT / "appendix_crosswalk.csv")
    print(f"Crosswalk saved: 5_results/appendix_crosswalk.csv ({K.height} HS6 codes). Columns: {K.columns}")
    print(K.filter(pl.col("group_v4").is_in(["1 CARS", "2 PARTS_CORE"]))
           .select(["group_v4", "hs6", "annex_status", "n_listed_lines", "eu_exports_to_us_2024_eur_m"]))


for name, f in [("A. Variance estimators", part_a), ("B. Pre-trends", part_b), ("C. Comparison markets", part_c),
                ("D. Control groups", part_d), ("E. Prices: size, balanced flows, weight per car", part_e),
                ("F. Volume effect by pre-tariff unit value", part_f), ("G. Exporter countries", part_g),
                ("H. Exits among regular flows", part_h), ("I. Tariff regimes, Aug-Sep separate", part_i),
                ("J. Panel coverage by group", part_j), ("K. Product crosswalk", part_k)]:
    run(name, f)
print(f"\nDone in {(time.time() - START) / 60:.0f} minutes.")
