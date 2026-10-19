"""Step 6.7: US normal (MFN) duties for the control products and the tariff wedge (reviewer point 13).

Until now the tariff wedge between cars and control products (Table 6, the -2.6 response) assumed a 2% US normal
duty for every control product. This script uses the actual US "General" duty from the US tariff schedule.

Data: the USITC HTS export for chapters 84-87 (CSV), saved in 2_raw_data/usitc_hts/ (the newest CSV there is used).
  The General rate sits on the 8-digit tariff line (or on its 10-digit lines); 10-digit statistical lines with an
  empty rate take the rate of their 8-digit line. "Free" = 0. One line in 84-87 has a compound rate
  (cents per unit + ad valorem); only its ad valorem part is used, and the script prints it.
Steps:
  1. US General rate for every 8-digit tariff line; simple average across the 8-digit lines of each HS6 code.
  2. For cars and the clean control, the total US tariff in each period (statutory rates, Table A1):
       baseline 2024             normal duty (m)
       Apr-Jul 2025              cars: m + 25% (Section 232);   control: m + 10% (reciprocal)
       Oct 2025 - Feb 2026       cars: 15%, or m if higher;     control: 15%, or m if higher
       Mar - Jun 2026            cars: 15%, or m if higher;     control: m + 10% (Section 122)
     Aug-Sep 2025 is shown twice: as agreed (both 15%) and as charged at entry (cars m + 25%, control 15%).
  3. Log change of (1 + tariff) against 2024 for each HS6 code, averaged with 2024 EU exports to the US as
     weights; wedge = cars minus control. Cars are the group "cars other than plug-in hybrid and battery-electric"
     used in Table 6 (all car codes except 870360, 870370, 870380); the all-cars wedge is printed too.
  4. Implied reduced-form response = Table 6 estimate (step37 part I, log points) / wedge, with the interval
     divided the same way (wedge treated as known).
Saves 5_results/step40_control_mfn_by_hs6.csv and 5_results/step40_tariff_wedge.csv
"""
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
OUT, DATA = BASE / "5_results", BASE / "7_processed_data"
EV = ["870360", "870370", "870380"]

# ---------------------------------------------------------------- 1. HTS rates
files = sorted((BASE / "2_raw_data" / "usitc_hts").glob("*.csv"), key=lambda p: p.stat().st_mtime)
if not files:
    raise SystemExit("No CSV file in 2_raw_data/usitc_hts")
H = pd.read_csv(files[-1], dtype=str, encoding="utf-8-sig").fillna("")
print(f"HTS file: {files[-1].name}, {len(H):,} rows")
H["digits"] = H["HTS Number"].str.replace(".", "", regex=False)
H = H[H["digits"].str.len().isin([8, 10])].copy()
H["line8"] = H["digits"].str[:8]
H["rate_txt"] = H["General Rate of Duty"].str.strip()


def parse(txt):
    if txt == "":
        return np.nan, ""
    if txt.lower().startswith("free"):
        return 0.0, ""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)%", txt)
    if m:
        return float(m.group(1)) / 100, ""
    pct = re.findall(r"(\d+(?:\.\d+)?)%", txt)
    return (float(pct[-1]) / 100 if pct else np.nan), txt  # compound or specific: ad valorem part only


H[["rate", "nonadval"]] = H["rate_txt"].apply(lambda t: pd.Series(parse(t)))
L8 = H.dropna(subset=["rate"]).groupby("line8").agg(rate=("rate", "first"), n_rates=("rate", "nunique"),
                                                     nonadval=("nonadval", "max")).reset_index()
if (L8["n_rates"] > 1).any():
    print("Warning: 8-digit lines with more than one rate:", L8.loc[L8["n_rates"] > 1, "line8"].tolist())
odd = L8[L8["nonadval"] != ""]
print(f"8-digit tariff lines with a rate: {len(L8):,}; lines with a specific or compound rate: {len(odd)}")
for _, r in odd.iterrows():
    print(f"  {r['line8']}: '{r['nonadval']}' -> ad valorem part {100 * r['rate']:.1f}% used")
L8["hs6"] = L8["line8"].str[:6]
R6 = L8.groupby("hs6").agg(mfn=("rate", "mean"), mfn_min=("rate", "min"), mfn_max=("rate", "max"),
                           n_lines=("rate", "size")).reset_index()

# ---------------------------------------------------------------- 2. groups and weights
P = pl.read_parquet(DATA / "panel_v1.parquet", columns=["hs6", "us", "period", "group_v4", "value_eur"])
W = (P.filter(pl.col("us") & pl.col("period").str.starts_with("2024") & pl.col("group_v4").is_in(["1 CARS", "4 CONTROL"]))
      .group_by(["hs6", "group_v4"]).agg((pl.col("value_eur").sum() / 1e6).alias("us2024_eur_m")).to_pandas())
G = pl.read_csv(DATA / "hs6_groups_v4.csv", schema_overrides={"hs6": pl.Utf8}).to_pandas()
G = G[G["group_v4"].isin(["1 CARS", "4 CONTROL"])][["hs6", "group_v4"]]
D = G.merge(W, on=["hs6", "group_v4"], how="left").fillna({"us2024_eur_m": 0.0}).merge(R6, on="hs6", how="left")
miss = D[D["mfn"].isna()]
print(f"\nHS6 codes: cars {int((D['group_v4'] == '1 CARS').sum())}, control {int((D['group_v4'] == '4 CONTROL').sum())}; "
      f"without a rate in the HTS file: {len(miss)} (2024 US exports EUR {miss['us2024_eur_m'].sum():,.1f} m)")
if len(miss):
    print("  ", miss["hs6"].tolist()[:20])
D = D.dropna(subset=["mfn"])
D.sort_values(["group_v4", "hs6"]).to_csv(OUT / "step40_control_mfn_by_hs6.csv", index=False)

C = D[D["group_v4"] == "4 CONTROL"]
w = C["us2024_eur_m"]
print("\nUS normal duty of control products (2024 EU exports to the US as weights):")
print(f"  weighted mean {100 * np.average(C['mfn'], weights=w):.2f}%, simple mean across HS6 {100 * C['mfn'].mean():.2f}%, "
      f"weighted median {100 * C.sort_values('mfn').assign(c=lambda x: x['us2024_eur_m'].cumsum() / x['us2024_eur_m'].sum()).query('c >= 0.5')['mfn'].iloc[0]:.2f}%")
bands = pd.cut(C["mfn"], [-0.001, 0.0, 0.02, 0.04, 0.15, 1.0], labels=["free", "0-2%", "2-4%", "4-15%", "15%+"])
print("  share of control US exports by duty band:",
      {k: f"{100 * v:.1f}%" for k, v in (w.groupby(bands, observed=False).sum() / w.sum()).items()})
K = D[D["group_v4"] == "1 CARS"]
print("Car codes: US normal duty by HS6 (2024 US exports, EUR m):")
for _, r in K.sort_values("us2024_eur_m", ascending=False).iterrows():
    print(f"  {r['hs6']}  {100 * r['mfn']:5.1f}%  {r['us2024_eur_m']:10,.1f}{'  (plug-in/electric)' if r['hs6'] in EV else ''}")

# ---------------------------------------------------------------- 3. wedges
ln = np.log


def tariffs(m, car):
    """Total US tariff by period for normal duty m (vectorised)."""
    agreed = np.maximum(0.15, m)
    if car:
        return {"AprJul25": m + 0.25, "AugSep25_agreed": agreed, "AugSep25_at_entry": m + 0.25,
                "OctFeb": agreed, "MarJun26": agreed}
    return {"AprJul25": m + 0.10, "AugSep25_agreed": agreed, "AugSep25_at_entry": agreed,
            "OctFeb": agreed, "MarJun26": m + 0.10}


def change(df, car):
    t = tariffs(df["mfn"].to_numpy(), car)
    return {k: float(np.average(ln(1 + v) - ln(1 + df["mfn"].to_numpy()), weights=df["us2024_eur_m"]))
            for k, v in t.items()}


ICE = K[~K["hs6"].isin(EV)]
dc, dk, dall = change(C, False), change(ICE, True), change(K, True)
dc_unw = {k: float(np.mean(ln(1 + v) - ln(1 + C["mfn"].to_numpy()))) for k, v in tariffs(C["mfn"].to_numpy(), False).items()}
old_c = {k: float(ln(1 + v) - ln(1.02)) for k, v in tariffs(np.array(0.02), False).items()}
old_k = {k: float(ln(1 + v) - ln(1.025)) for k, v in tariffs(np.array(0.025), True).items()}

# ---------------------------------------------------------------- 4. implied responses
I = pd.read_csv(OUT / "step37_I_regimes.csv").set_index("term")
est = {"AprJul25": "ICE_R1", "AugSep25_agreed": "ICE_R1b", "AugSep25_at_entry": "ICE_R1b", "OctFeb": "ICE_R2", "MarJun26": "ICE_R3"}
rows = []
print("\nLog change of (1 + tariff) against 2024, and the wedge (cars other than plug-in/electric minus control):")
print(f"{'period':18s} {'cars':>7s} {'control':>8s} {'wedge':>7s} {'old wedge (2%)':>15s} {'all cars':>9s} "
      f"{'control unweighted':>19s}  implied response [95% interval]")
for k in dc:
    wedge = dk[k] - dc[k]
    b, lo, hi = I.loc[est[k], "Estimate"], I.loc[est[k], "2.5%"], I.loc[est[k], "97.5%"]
    resp = b / wedge if abs(wedge) >= 0.02 else np.nan
    ci = sorted([lo / wedge, hi / wedge]) if abs(wedge) >= 0.02 else [np.nan, np.nan]
    rows.append(dict(period=k, cars_dlog=dk[k], control_dlog=dc[k], wedge=wedge, wedge_all_cars=dall[k] - dc[k],
                     wedge_control_unweighted=dk[k] - dc_unw[k], wedge_old_2pct=old_k[k] - old_c[k],
                     estimate=b, implied=resp, implied_lo=ci[0], implied_hi=ci[1]))
    txt = f"{resp:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]" if not np.isnan(resp) else "not computed (wedge below 0.02)"
    print(f"{k:18s} {dk[k]:7.3f} {dc[k]:8.3f} {wedge:7.3f} {old_k[k] - old_c[k]:15.3f} {dall[k] - dc[k]:9.3f} "
          f"{dk[k] - dc_unw[k]:19.3f}  {txt}")
pd.DataFrame(rows).to_csv(OUT / "step40_tariff_wedge.csv", index=False)
print("\nSaved 5_results/step40_control_mfn_by_hs6.csv and 5_results/step40_tariff_wedge.csv")
