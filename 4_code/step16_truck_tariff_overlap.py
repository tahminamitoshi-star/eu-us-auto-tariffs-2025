"""Step 1.21: Which of my product groups does the truck tariff (Proclamation 10984) touch?

Input: 3_tariff_documents/annex_I_codes_proclamation10984.csv, the codes in Annex I of Proclamation 10984
(signed 17 Oct 2025, 90 FR 48451, effective 1 Nov 2025):
  MHDV_vehicles  trucks and tractors, U.S. note 38(b), +25%   (25 codes)
  buses          heading 8702, U.S. note 38(c), +10%           (10 codes)
  MHDV_parts     truck parts, U.S. note 38(i), +25%            (182 codes)
Same coverage rule as Step 1.12 (only HS6 matches between the US and EU code lists):
  full    = whole HS6 listed (6-digit code, 8-digit ending .00, or 10-digit ending .0000)
  partial = only some US sub-lines inside the HS6 are listed
  none    = nothing listed
Truck parts pay only when used in trucks (note 38(l)), like car parts (Step 1.12).

Output: for each group in hs6_groups_v3, how many HS6 codes and how much of 2024 EU exports to the US
are touched by the truck tariff. Saves 7_processed_data/hs6_groups_v3_truckflags.csv
Nothing in hs6_groups_v3.csv is changed.
"""
import os
from pathlib import Path
import polars as pl

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
P = pl.read_parquet(BASE / "7_processed_data" / "exports_ch84_87.parquet")
G = pl.read_csv(BASE / "7_processed_data" / "hs6_groups_v3.csv", schema_overrides={"hs6": pl.Utf8})
A = pl.read_csv(BASE / "3_tariff_documents" / "annex_I_codes_proclamation10984.csv",
                schema_overrides={"us_code": pl.Utf8, "hs4": pl.Utf8, "hs6": pl.Utf8})
print("Annex I codes of Proclamation 10984 by list:")
print(A.group_by("list").len().sort("list"))

code = pl.col("us_code").str.replace_all(r"\.", "")
A = A.with_columns(((pl.col("digits") == 6) | ((pl.col("digits") == 8) & code.str.ends_with("00"))
                    | ((pl.col("digits") == 10) & code.str.ends_with("0000"))).alias("whole"))

def status(lists):
    a = A.filter(pl.col("list").is_in(lists))
    full = set(a.filter(pl.col("whole"))["hs6"].to_list())
    part = set(a.filter(~pl.col("whole"))["hs6"].to_list()) - full
    return (pl.when(pl.col("hs6").is_in(list(full))).then(pl.lit("full"))
              .when(pl.col("hs6").is_in(list(part))).then(pl.lit("partial"))
              .otherwise(pl.lit("none")))

F = G.with_columns(status(["MHDV_vehicles", "buses"]).alias("truck_bus_vehicle"),
                   status(["MHDV_parts"]).alias("truck_part"))
F.sort("hs6").write_csv(BASE / "7_processed_data" / "hs6_groups_v3_truckflags.csv")

us24 = (P.filter((pl.col("dest") == "US") & pl.col("period").str.starts_with("2024"))
          .with_columns(pl.col("cn8").str.slice(0, 6).alias("hs6"))
          .group_by("hs6").agg((pl.col("value_eur").sum() / 1e9).alias("bn_eur_2024")))
F = F.join(us24, on="hs6", how="left").with_columns(pl.col("bn_eur_2024").fill_null(0))

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(150)
for flag in ["truck_bus_vehicle", "truck_part"]:
    print(f"\n{flag}: HS6 codes and 2024 EU exports to the US (EUR bn), by group")
    print(F.group_by(["group_v3", flag])
           .agg(pl.len().alias("hs6_codes"), pl.col("bn_eur_2024").sum().round(2))
           .sort(["group_v3", flag]))

print("\nLargest CONTROL codes listed as truck parts (top 15):")
print(F.filter((pl.col("group_v3") == "4 CONTROL") & (pl.col("truck_part") != "none"))
       .sort("bn_eur_2024", descending=True).head(15)
       .select(["hs6", "truck_part", "aircraft_dual_use", pl.col("bn_eur_2024").round(3)]))

print("\nHEAVY_VEHICLES codes and their truck-tariff status:")
print(F.filter(pl.col("group_v3") == "6 HEAVY_VEHICLES")
       .group_by("truck_bus_vehicle").agg(pl.len().alias("hs6_codes"), pl.col("bn_eur_2024").sum().round(2))
       .sort("truck_bus_vehicle"))
