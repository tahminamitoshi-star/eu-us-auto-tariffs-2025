"""Step 1.22: Which of my product groups were steel or aluminium "derivatives" under Section 232, and from when?

Input: 3_tariff_documents/metal_derivatives_ch84_87.csv, built from the legal texts (codes in chapters 84-87 only):
  2025-03-12  Proclamation 10896 (steel) and 10895 (aluminium), Annex I       duty on the metal content (25%, 50% from 4 Jun 2025)
  2025-06-23  BIS notice 2025-11067 (household appliances, steel)             metal content
  2025-08-18  BIS notice 2025-15819 (407 codes, steel and aluminium)          metal content
  2026-04-06  Proclamation 11021, Annex IV note 16(c)                         full value if metal >= 15% of weight;
              tier I-B = 25%, tier III = 15% all-in (temporary, to end-2027)
Coverage rule as in Step 1.12: full = whole HS6 listed (6-digit code, 8-digit ending 00, 10-digit ending 0000);
partial = only some US sub-lines in the HS6 are listed. Car and truck goods under the auto/truck 232 tariffs
do not pay the metal duties (U.S. notes 33 and 38), so flags on CARS and PARTS_CORE matter little.

Output: 2024 EU exports to the US by group and derivative status, for 2025 (by first date) and for April 2026.
Saves 7_processed_data/hs6_groups_v3_metalflags.csv. Nothing else is changed.
"""
import os
from pathlib import Path
import polars as pl

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
P = pl.read_parquet(BASE / "7_processed_data" / "exports_ch84_87.parquet")
F = pl.read_csv(BASE / "7_processed_data" / "hs6_groups_v3_truckflags.csv", schema_overrides={"hs6": pl.Utf8})
M = pl.read_csv(BASE / "3_tariff_documents" / "metal_derivatives_ch84_87.csv",
                schema_overrides={"us_code": pl.Utf8, "hs6": pl.Utf8, "tier_2026": pl.Utf8})
code = pl.col("us_code").str.replace_all(r"\.", "")
M = M.with_columns(((pl.col("digits") == 6) | ((pl.col("digits") == 8) & code.str.ends_with("00"))
                    | ((pl.col("digits") == 10) & code.str.ends_with("0000"))).alias("whole"))

# 2025: earliest date each HS6 became a derivative, and whether the whole HS6 was listed by that date
m25 = M.filter(pl.col("in_force_from") < "2026-01-01")
d25 = (m25.group_by("hs6").agg(pl.col("in_force_from").min().alias("deriv_2025_from"),
                               pl.col("whole").any().alias("deriv_2025_whole")))
# 2026: tier under Proclamation 11021 (if listed in more than one tier, keep the highest)
m26 = M.filter(pl.col("in_force_from") == "2026-04-06")
d26 = (m26.with_columns(pl.when(pl.col("tier_2026").str.starts_with("I-B")).then(2).otherwise(1).alias("rank"))
          .sort("rank", descending=True)
          .group_by("hs6").agg(pl.col("tier_2026").first().alias("deriv_2026_tier"),
                               pl.col("whole").any().alias("deriv_2026_whole")))

G = (F.join(d25, on="hs6", how="left").join(d26, on="hs6", how="left")
      .with_columns(pl.col("deriv_2025_from").fill_null("never"),
                    pl.col("deriv_2026_tier").fill_null("none")))
G.sort("hs6").write_csv(BASE / "7_processed_data" / "hs6_groups_v3_metalflags.csv")

us24 = (P.filter((pl.col("dest") == "US") & pl.col("period").str.starts_with("2024"))
          .with_columns(pl.col("cn8").str.slice(0, 6).alias("hs6"))
          .group_by("hs6").agg((pl.col("value_eur").sum() / 1e9).alias("bn_eur_2024")))
G = G.join(us24, on="hs6", how="left").with_columns(pl.col("bn_eur_2024").fill_null(0))

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(160)
print("2025: first date the HS6 became a metal derivative (2024 EU exports to the US, EUR bn)")
print(G.group_by(["group_v3", "deriv_2025_from"])
       .agg(pl.len().alias("hs6_codes"), pl.col("bn_eur_2024").sum().round(2),
            pl.col("deriv_2025_whole").sum().alias("of_which_whole_hs6"))
       .sort(["group_v3", "deriv_2025_from"]))

print("\nFrom 6 Apr 2026: tier under Proclamation 11021")
print(G.group_by(["group_v3", "deriv_2026_tier"])
       .agg(pl.len().alias("hs6_codes"), pl.col("bn_eur_2024").sum().round(2))
       .sort(["group_v3", "deriv_2026_tier"]))

print("\nLargest CONTROL codes that were derivatives in 2025 (top 20):")
print(G.filter((pl.col("group_v3") == "4 CONTROL") & (pl.col("deriv_2025_from") != "never"))
       .sort("bn_eur_2024", descending=True).head(20)
       .select(["hs6", "deriv_2025_from", "deriv_2025_whole", "deriv_2026_tier", pl.col("bn_eur_2024").round(3)]))

print("\nHEAVY_VEHICLES codes not on the truck list, and their derivative status:")
print(G.filter((pl.col("group_v3") == "6 HEAVY_VEHICLES") & (pl.col("truck_bus_vehicle") == "none")
               & (pl.col("truck_part") == "none"))
       .sort("bn_eur_2024", descending=True)
       .select(["hs6", "deriv_2025_from", "deriv_2025_whole", "deriv_2026_tier", pl.col("bn_eur_2024").round(3)]))
