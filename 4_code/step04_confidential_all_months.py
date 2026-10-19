"""Step 1.4: Repeat the confidentiality check for every month, Jan 2023 - Jun 2026.

For EU exports to the US, per month:
  ch87_normal_m_eur   chapter 87 (vehicles and parts) in normal 8-digit rows, EUR million
  ch87_conf_m_eur     chapter 87 hidden as 87XXXXXX, EUR million
  all_normal_bn_eur   all products in normal 8-digit rows, EUR billion
  all_conf_bn_eur     all confidential rows (any chapter), EUR billion
July 2026 is skipped because that file is incomplete.
Result is saved to 6_results_and_notes/rebuild/step04_confidential_by_month.csv
"""
import os
from pathlib import Path
import time
import polars as pl

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
RAW = BASE / "2_raw_data" / "eurostat_comext_extracted"
OUT = BASE / "6_results_and_notes" / "rebuild"
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(RAW.glob("eurostat_exports_imports_*.csv"))
files = [f for f in files if f.stem[-6:] <= "202606"]          # skip July 2026
print("Files to read:", len(files))

rows = []
start = time.time()
for f in files:
    month = f.stem[-6:]                                          # e.g. 202301
    df = (pl.scan_csv(f, infer_schema=False)
            .select(["PARTNER", "PRODUCT_NC", "FLOW", "VALUE_EUR"])
            .filter((pl.col("FLOW") == "2") & (pl.col("PARTNER") == "US")
                    & (pl.col("PRODUCT_NC") != "TOTAL"))
            .with_columns(pl.col("VALUE_EUR").cast(pl.Float64),
                          pl.col("PRODUCT_NC").str.contains(r"^\d{8}$").alias("normal"),
                          (pl.col("PRODUCT_NC").str.slice(0, 2) == "87").alias("ch87"))
            .collect())
    v = pl.col("VALUE_EUR")
    r = df.select(
        (v.filter(pl.col("ch87") & pl.col("normal")).sum() / 1e6).alias("ch87_normal_m_eur"),
        (v.filter(pl.col("ch87") & ~pl.col("normal")).sum() / 1e6).alias("ch87_conf_m_eur"),
        (v.filter(pl.col("normal")).sum() / 1e9).alias("all_normal_bn_eur"),
        (v.filter(~pl.col("normal")).sum() / 1e9).alias("all_conf_bn_eur"),
    ).to_dicts()[0]
    r["month"] = month
    rows.append(r)
    print(month, "done", f"({time.time() - start:.0f} s)")

res = (pl.DataFrame(rows)
         .select(["month", "ch87_normal_m_eur", "ch87_conf_m_eur", "all_normal_bn_eur", "all_conf_bn_eur"])
         .with_columns((100 * pl.col("ch87_conf_m_eur") / (pl.col("ch87_normal_m_eur") + pl.col("ch87_conf_m_eur"))).alias("ch87_conf_pct"),
                       (100 * pl.col("all_conf_bn_eur") / (pl.col("all_normal_bn_eur") + pl.col("all_conf_bn_eur"))).alias("all_conf_pct"))
         .with_columns(pl.col(pl.Float64).round(2)))

pl.Config.set_tbl_rows(50)
print(res)
print("\nLargest chapter 87 confidential share in any month:", res["ch87_conf_pct"].max(), "%")
res.write_csv(OUT / "step04_confidential_by_month.csv")
print("Saved:", OUT / "step04_confidential_by_month.csv")
