"""Step 3.13: Deflection. Where did the EU cars that no longer went to the US go?

Two questions in one step.
(1) Mechanism: when the US tariff cut EU car exports to the US, did EU exporters sell more cars in other markets
    (the UK, Switzerland, Turkey, Japan, Korea, Canada, Mexico, the rest of the world, or inside the EU)?
(2) Identification: the main model compares the US with other non-EU markets. If cars were redirected to those
    markets, the comparison group rose because of the tariff, and the main US effect (-30%) is too large.

Part 1 (descriptive, all trade in exports_ch84_87.parquet, so new destinations count too):
    EU car exports by destination, monthly average, against the same calendar months of 2024:
    HIGH = Apr-Sep 2025, LOW = Oct 2025 - Jun 2026. Value (EUR bn) and number of cars. Control products shown
    next to cars, so a destination where everything grew is not read as deflection. Intra-EU trade is included.
Part 2 (PPML, panel, non-EU destinations except the US and China): did car exports to other markets rise relative
    to control products in the same markets after April 2025? FE: flow (exporter x HS6 x destination),
    destination x month, exporter x month, and group x destination x calendar month (seasonality).
    Clustered by exporter x HS6. Then the same split by destination.
Part 3 (PPML, US only): cars against control products within the US market, without any other market as the
    comparison. FE: flow, exporter x month, group x calendar month. If this is close to the main -30%, the main
    result does not come from deflection.
Saves 5_results/step32_deflection_desc.csv and 5_results/step32_deflection_ppml.csv
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf
import warnings

warnings.filterwarnings("ignore")
BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
OUT = BASE / "5_results"
G = pl.read_csv(BASE / "7_processed_data" / "hs6_groups_v4.csv", schema_overrides={"hs6": pl.Utf8})
KEEP = ["1 CARS", "2 PARTS_CORE", "4 CONTROL"]
HIGH, LOW = ("202504", "202509"), ("202510", "202606")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(160)
inp = lambda p: pl.col("period").is_between(pl.lit(p[0]), pl.lit(p[1]))

# ---------------- Part 1: descriptive
X = (pl.read_parquet(BASE / "7_processed_data" / "exports_ch84_87.parquet")
       .with_columns(pl.col("cn8").str.slice(0, 6).alias("hs6"))
       .join(G.select(["hs6", "group_v4"]), on="hs6", how="left")
       .filter(pl.col("group_v4").is_in(["1 CARS", "4 CONTROL"]))
       .with_columns(pl.col("period").str.slice(4, 2).alias("cm"),
                     pl.when(pl.col("period").str.starts_with("2024")).then(pl.lit("base"))
                       .when(inp(HIGH)).then(pl.lit("HIGH")).when(inp(LOW)).then(pl.lit("LOW"))
                       .otherwise(None).alias("per"))
       .filter(pl.col("per").is_not_null()))
# monthly totals by group x destination x period x calendar month
M = (X.group_by(["group_v4", "dest", "per", "cm"])
      .agg(pl.col("value_eur").sum().alias("v"), pl.col("qty_units").sum().alias("n")))
rows = []
CMS = {"HIGH": ["04", "05", "06", "07", "08", "09"],
       "LOW": ["10", "11", "12", "01", "02", "03", "04", "05", "06"]}
for per in ["HIGH", "LOW"]:
    cms = CMS[per]
    nmon = len(cms)
    base = (M.filter((pl.col("per") == "base") & pl.col("cm").is_in(cms))
             .group_by(["group_v4", "dest"]).agg((pl.col("v").sum() / len(cms)).alias("v_base"),
                                                 (pl.col("n").sum() / len(cms)).alias("n_base")))
    post = (M.filter(pl.col("per") == per)
             .group_by(["group_v4", "dest"]).agg((pl.col("v").sum() / nmon).alias("v_post"),
                                                 (pl.col("n").sum() / nmon).alias("n_post")))
    rows.append(base.join(post, on=["group_v4", "dest"], how="full", coalesce=True).fill_null(0.0)
                    .with_columns(pl.lit(per).alias("period")))
D = (pl.concat(rows)
       .with_columns(((pl.col("v_post") - pl.col("v_base")) / 1e6).alias("chg_eur_m_per_month"),
                     (100 * (pl.col("v_post") / pl.col("v_base") - 1)).alias("pct_value"),
                     (100 * (pl.col("n_post") / pl.col("n_base") - 1)).alias("pct_cars"))
       .sort(["period", "group_v4", "v_base"], descending=[False, False, True]))
D.write_csv(OUT / "step32_deflection_desc.csv")

print("Part 1. EU exports by destination, monthly average, % change against the same calendar months of 2024.")
print("        Change in value is in EUR million per month. EU = intra-EU trade. XX = rest of the world.\n")
for per in ["HIGH", "LOW"]:
    c = D.filter((pl.col("period") == per) & (pl.col("group_v4") == "1 CARS"))
    k = D.filter((pl.col("period") == per) & (pl.col("group_v4") == "4 CONTROL")).select(
        "dest", pl.col("pct_value").alias("control_pct_value"))
    t = (c.join(k, on="dest", how="left")
          .select("dest", (pl.col("v_base") / 1e6).round(0).alias("cars_2024_eur_m_month"),
                  pl.col("chg_eur_m_per_month").round(0).alias("cars_chg_eur_m_month"),
                  pl.col("pct_value").round(1).alias("cars_pct_value"), pl.col("pct_cars").round(1).alias("cars_pct_units"),
                  pl.col("control_pct_value").round(1)))
    lost = -c.filter(pl.col("dest") == "US")["chg_eur_m_per_month"].sum()
    gain_noneu = c.filter(~pl.col("dest").is_in(["US", "EU"]))["chg_eur_m_per_month"].sum()
    gain_eu = c.filter(pl.col("dest") == "EU")["chg_eur_m_per_month"].sum()
    print(f"{per} ({'Apr-Sep 2025' if per == 'HIGH' else 'Oct 2025 - Jun 2026'}): cars")
    print(t)
    print(f"  Car exports to the US: {-lost:+.0f} EUR m per month. Other non-EU markets together: {gain_noneu:+.0f}. "
          f"Intra-EU: {gain_eu:+.0f}.\n")

# ---------------- PPML helpers
P = pl.read_parquet(BASE / "7_processed_data" / "panel_v1.parquet",
                    columns=["exporter", "hs6", "dest", "us", "period", "group_v4", "value_eur", "frontload"])
P = (P.filter(pl.col("group_v4").is_in(KEEP) & ~pl.col("frontload") & (pl.col("dest") != "CN"))
      .group_by(["exporter", "hs6", "dest", "period", "group_v4"]).agg((pl.col("value_eur").sum() / 1e6).alias("v"))
      .with_columns(pl.col("period").str.slice(4, 2).alias("cm")))
cars, parts = pl.col("group_v4") == "1 CARS", pl.col("group_v4") == "2 PARTS_CORE"
cat = lambda cols, name: pl.concat_str(cols, separator="_").cast(pl.Categorical).to_physical().alias(name)


def fit(Q, terms, fes, label):
    Q = Q.with_columns([c.cast(pl.Float64).alias(n) for n, c in terms.items()])
    names = [n for n in terms if Q[n].sum() > 0]
    df = Q.select(["v", "cl"] + [f for f in fes] + names).to_pandas()
    m = pf.fepois(f"v ~ {' + '.join(names)} | {' + '.join(fes)}", data=df, vcov={"CRV1": "cl"})
    T = m.tidy().reset_index().rename(columns={"Coefficient": "term"})
    for a, b in [("Estimate", "pct"), ("2.5%", "lo"), ("97.5%", "hi")]:
        T[b] = 100 * (np.exp(T[a]) - 1)
    T["model"] = label
    print(f"\n{label} ({m._N:,} obs.)")
    for _, r in T.iterrows():
        print(f"  {r['term']:16s} {r['pct']:+6.1f}%  [{r['lo']:+.0f}, {r['hi']:+.0f}]  p = {r['Pr(>|t|)']:.3f}")
    return T


out = []
# ---------------- Part 2: other non-EU markets (no US, no China)
O = (P.filter(pl.col("dest") != "US")
      .with_columns(cat(["exporter", "hs6", "dest"], "f_flow"), cat(["dest", "period"], "f_dm"),
                    cat(["exporter", "period"], "f_em"), cat(["group_v4", "dest", "cm"], "f_season"),
                    cat(["exporter", "hs6"], "cl")))
fes = ["f_flow", "f_dm", "f_em", "f_season"]
print("\nPart 2. Exports to other non-EU markets (not the US, not China): cars and core parts against control "
      "products in the same markets.")
out.append(fit(O, {"CARS_HIGH": cars & inp(HIGH), "CARS_LOW": cars & inp(LOW),
                   "PARTS_HIGH": parts & inp(HIGH), "PARTS_LOW": parts & inp(LOW)}, fes,
               "2a. All other non-EU markets together"))
dests = [d for d in ["GB", "CH", "TR", "JP", "KR", "CA", "MX", "XX"] if d in O["dest"].unique().to_list()]
terms = {}
for d in dests:
    terms[f"CARS{d}_HIGH"] = cars & inp(HIGH) & (pl.col("dest") == d)
    terms[f"CARS{d}_LOW"] = cars & inp(LOW) & (pl.col("dest") == d)
terms["PARTS_HIGH"], terms["PARTS_LOW"] = parts & inp(HIGH), parts & inp(LOW)
T = fit(O, terms, fes, "2b. Car effect by destination")
out.append(T)

# ---------------- Part 3: US only
U = (P.filter(pl.col("dest") == "US")
      .with_columns(cat(["exporter", "hs6"], "f_flow"), cat(["exporter", "period"], "f_em"),
                    cat(["group_v4", "cm"], "f_season"), cat(["exporter", "hs6"], "cl")))
print("\nPart 3. US market only: cars and core parts against control products in the US (no other market used).")
out.append(fit(U, {"CARS_HIGH": cars & inp(HIGH), "CARS_LOW": cars & inp(LOW),
                   "PARTS_HIGH": parts & inp(HIGH), "PARTS_LOW": parts & inp(LOW)},
               ["f_flow", "f_em", "f_season"], "3. US only (compare with main model: cars HIGH -30.3%)"))
pd.concat(out, ignore_index=True).to_csv(OUT / "step32_deflection_ppml.csv", index=False)
