"""Step 1.18: Is the control group really "general tariff only"?

Checks the CONTROL group from Step 1.13 against two things found in Proclamation 11012 (Section 122, Feb 2026):
  1. Goods exempt from the 10% surcharge for any use (note 2(aa)(ii)): computers 8471, parts 8473.30,
     chip-making machines 8486, phones 8517.13/8517.62, storage 8523.51, displays 8524, monitors 8528.52,
     diodes and chips 8541/8542. Many of these were also exempt from the 2025 reciprocal tariff, so they
     may never have paid the general tariff at all.
  2. Goods exempt only when used in civil aircraft (note 2(aa)(iv)). Dual-use, like the car-parts list.
Also flags products that may fall under the Section 232 tariff on medium and heavy-duty vehicles and buses
(Proclamation 10984 of 17 Oct 2025, cited in Proclamation 11012): 8701, 8702, 8704 except light trucks,
8705, 8709, 8716. This list is a candidate list and must be checked against Proclamation 10984 itself.

Coverage rule as in Step 1.12: full = whole HS6 listed (6-digit code, or 8-digit ending .00, 10-digit ending .0000);
partial = only some US sub-lines listed.
Output: 2024 EU exports to the US in the control group, by flag. Saves 7_processed_data/hs6_groups_v2_flags.csv
"""
import os
from pathlib import Path
import polars as pl

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
P = pl.read_parquet(BASE / "7_processed_data" / "exports_ch84_87.parquet")
G = pl.read_csv(BASE / "7_processed_data" / "hs6_groups_v2.csv", schema_overrides={"hs6": pl.Utf8})
E = pl.read_csv(BASE / "3_tariff_documents" / "section122_exemptions_ch84_87.csv",
                schema_overrides={"hs6": pl.Utf8, "us_code": pl.Utf8})

code = pl.col("us_code").str.replace_all(r"\.", "")
E = E.with_columns(((pl.col("digits") == 6) | ((pl.col("digits") == 8) & code.str.ends_with("00"))
                    | ((pl.col("digits") == 10) & code.str.ends_with("0000"))).alias("whole"))

def status(lst):
    full = set(E.filter((pl.col("exemption") == lst) & pl.col("whole"))["hs6"].to_list())
    part = set(E.filter((pl.col("exemption") == lst) & ~pl.col("whole"))["hs6"].to_list()) - full
    return (pl.when(pl.col("hs6").is_in(list(full))).then(pl.lit("full"))
              .when(pl.col("hs6").is_in(list(part))).then(pl.lit("partial"))
              .otherwise(pl.lit("none")))

LIGHT = ["870421", "870431", "870441", "870451", "870460"]
hs4 = pl.col("hs6").str.slice(0, 4)
F = G.with_columns(status("ii").alias("s122_exempt_any_use"),
                   status("iv").alias("s122_exempt_aircraft_use"),
                   (hs4.is_in(["8701", "8702", "8705", "8709", "8716"])
                    | ((hs4 == "8704") & ~pl.col("hs6").is_in(LIGHT))).alias("possible_heavy_vehicle"))
F.sort("hs6").write_csv(BASE / "7_processed_data" / "hs6_groups_v2_flags.csv")

us24 = (P.filter((pl.col("dest") == "US") & pl.col("period").str.starts_with("2024"))
          .with_columns(pl.col("cn8").str.slice(0, 6).alias("hs6"))
          .group_by("hs6").agg((pl.col("value_eur").sum() / 1e9).alias("bn_eur_2024")))
C = F.filter(pl.col("group") == "4 CONTROL").join(us24, on="hs6", how="left").with_columns(pl.col("bn_eur_2024").fill_null(0))
total = C["bn_eur_2024"].sum()
print(f"Control group: {C.height} HS6 codes, EUR {total:.1f} bn of exports to the US in 2024\n")

pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_width_chars(140)
for flag in ["s122_exempt_any_use", "s122_exempt_aircraft_use", "possible_heavy_vehicle"]:
    t = (C.group_by(flag).agg(pl.len().alias("hs6_codes"), pl.col("bn_eur_2024").sum().round(2))
          .with_columns((100 * pl.col("bn_eur_2024") / total).round(1).alias("pct_of_control")).sort(flag))
    print(flag); print(t); print()

print("Largest control codes exempt for any use, or possible heavy vehicles:")
print(C.filter((pl.col("s122_exempt_any_use") != "none") | pl.col("possible_heavy_vehicle"))
       .sort("bn_eur_2024", descending=True).head(15)
       .select(["hs6", "s122_exempt_any_use", "possible_heavy_vehicle", pl.col("bn_eur_2024").round(3)]))
