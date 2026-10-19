"""Step 1.12: Which products did the US car and parts tariff actually cover?

Input: 3_tariff_documents/annex_I_codes_proclamation10908.csv, the codes listed in Annex I of
Proclamation 10908 (vehicles: note 33(b), from 3 Apr 2025; parts: note 33(g), from 3 May 2025).

Important: US codes have 4, 6, 8 or 10 digits. Only the first 6 digits (HS6) are the same worldwide.
Digits 7-10 are US-only and do not match the EU's 8-digit CN codes. So we can match exactly only at HS6:
  full    = the whole HS6 is listed: a 4- or 6-digit US code, a vehicle code, or an 8-digit code ending
            in .00 / a 10-digit code ending in .0000 (the US uses these when an HS6 is not split further)
  partial = only some US 8- or 10-digit lines inside the HS6 are listed
  none    = nothing in the HS6 is listed
Also, parts are tariffed only when they are parts of passenger vehicles or light trucks
(note 33(h)(ii)). A bearing for a car pays; the same bearing for a washing machine does not.
So we flag codes that are auto-only by definition: 8703, 8704, 8706, 8707, 8708.

Compares this with the groups used in the first draft (CARS / PARTS / CTRL), using 2024 EU exports to the US.
Saves 7_processed_data/hs6_annex_status.csv
"""
import os
from pathlib import Path
import polars as pl

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
P = pl.read_parquet(BASE / "7_processed_data" / "exports_ch84_87.parquet")
A = pl.read_csv(BASE / "3_tariff_documents" / "annex_I_codes_proclamation10908.csv",
                schema_overrides={"hs4": pl.Utf8, "hs6": pl.Utf8, "us_code": pl.Utf8})
print("Annex codes:", A.height, "| by list and digits:")
print(A.group_by(["list", "digits"]).len().sort(["list", "digits"]))

full4 = set(A.filter(pl.col("digits") == 4)["hs4"].to_list())
code = pl.col("us_code").str.replace_all(r"\.", "")
whole = ((pl.col("digits") == 6) | (pl.col("list") == "vehicles")
         | ((pl.col("digits") == 8) & code.str.ends_with("00"))
         | ((pl.col("digits") == 10) & code.str.ends_with("0000")))
full6 = set(A.filter(whole)["hs6"].to_list())
part6 = set(A.filter(~whole & (pl.col("digits") > 6))["hs6"].to_list()) - full6

us24 = (P.filter((pl.col("dest") == "US") & pl.col("period").str.starts_with("2024"))
          .with_columns(pl.col("cn8").str.slice(0, 6).alias("hs6"))
          .group_by("hs6").agg((pl.col("value_eur").sum() / 1e9).alias("bn_eur_2024")))

hs4 = pl.col("hs6").str.slice(0, 4)
S = (us24.with_columns(
        pl.when(hs4.is_in(list(full4)) | pl.col("hs6").is_in(list(full6))).then(pl.lit("full"))
          .when(pl.col("hs6").is_in(list(part6))).then(pl.lit("partial"))
          .otherwise(pl.lit("none")).alias("annex_status"),
        hs4.is_in(["8703", "8704", "8706", "8707", "8708"]).alias("auto_only"),
        pl.when((hs4 == "8703") | pl.col("hs6").is_in(["870421", "870431"])).then(pl.lit("CARS"))
          .when(hs4.is_in(["8708", "8409"]) | pl.col("hs6").is_in(["840733", "840734", "840820"])).then(pl.lit("PARTS"))
          .otherwise(pl.lit("CTRL")).alias("draft_group")))
S.sort("hs6").write_csv(BASE / "7_processed_data" / "hs6_annex_status.csv")

pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_width_chars(140)
print("\nEU exports to the US in 2024 (EUR bn) by first-draft group and annex status:")
print(S.group_by(["draft_group", "annex_status"])
       .agg(pl.len().alias("hs6_codes"), pl.col("bn_eur_2024").sum().round(2))
       .sort(["draft_group", "annex_status"]))

print("\nFirst-draft PARTS codes and their annex status:")
print(S.filter(pl.col("draft_group") == "PARTS").sort("bn_eur_2024", descending=True)
       .select(["hs6", "annex_status", pl.col("bn_eur_2024").round(3)]))

print("\nLargest annex-listed codes inside the first-draft CONTROL group (top 15):")
print(S.filter((pl.col("draft_group") == "CTRL") & (pl.col("annex_status") != "none"))
       .sort("bn_eur_2024", descending=True).head(15)
       .select(["hs6", "annex_status", "auto_only", pl.col("bn_eur_2024").round(3)]))
