"""Step 3.1: Build the estimation panel.

Decisions (Tahmina, 13 Oct 2026):
  - unit: EU exporter country x CN8 product x destination x month, Jan 2023 - Jun 2026
  - comparison: the US against all other non-EU destinations (GB, CH, CN, JP, KR, CA, MX, TR, XX);
    intra-EU trade is not in the panel (used later for deflection)
  - treatment: group x period, groups from hs6_groups_v4.csv
  - window: Jan-Mar 2025 kept in the panel but flagged (frontload = True), dropped in the main estimates

Rules:
  - A "flow" is one exporter x CN8 x destination. Keep flows with positive exports in the pre-period
    (2023-2024). Flows that start only in 2025-2026 are left out (their entry is a separate question),
    and the script reports how much 2025-26 value this leaves out.
  - Every kept flow gets all 42 months; months with no trade are set to 0 (PPML uses the zeros).
  - Values summed over the supplementary-unit field (one CN8 has one unit).
Checks printed: flows and value by group and market, share of zero months, coverage of 2025-26 trade,
and CN8 codes that appear or disappear at a year change (a sign of CN nomenclature changes).
Saves 7_processed_data/panel_v1.parquet
"""
import os
from pathlib import Path
import polars as pl

BASE = Path(os.environ.get("FIW_BASE", Path.home() / "Downloads" / "FIW_2027_paper"))
P = pl.read_parquet(BASE / "7_processed_data" / "exports_ch84_87.parquet")
G = pl.read_csv(BASE / "7_processed_data" / "hs6_groups_v4.csv", schema_overrides={"hs6": pl.Utf8})

X = (P.filter(pl.col("dest") != "EU")
      .group_by(["exporter", "cn8", "dest", "period"])
      .agg(pl.col("value_eur").sum(), pl.col("qty_kg").sum(), pl.col("qty_units").sum()))

pre = X.filter(pl.col("period") < "202501")
flows = (pre.group_by(["exporter", "cn8", "dest"]).agg(pl.col("value_eur").sum().alias("v_pre"))
            .filter(pl.col("v_pre") > 0).drop("v_pre"))
months = X.select("period").unique().sort("period")
print(f"Months in data: {months.height} ({months['period'][0]} to {months['period'][-1]})")

panel = (flows.join(months, how="cross")
              .join(X, on=["exporter", "cn8", "dest", "period"], how="left")
              .with_columns(pl.col(["value_eur", "qty_kg", "qty_units"]).fill_null(0.0),
                            pl.col("cn8").str.slice(0, 6).alias("hs6"))
              .join(G.select(["hs6", "group_v4", "ag_tractor", "aircraft_dual_use", "metal_2025_from"]),
                    on="hs6", how="left")
              .with_columns((pl.col("dest") == "US").alias("us"),
                            pl.col("period").is_between(pl.lit("202501"), pl.lit("202503")).alias("frontload"),
                            ((pl.col("period").str.slice(0, 4).cast(pl.Int32) - 2023) * 12
                             + pl.col("period").str.slice(4, 2).cast(pl.Int32)).alias("t")))
missing = panel.filter(pl.col("group_v4").is_null())["hs6"].n_unique()
print(f"HS6 codes without a group (should be 0): {missing}")
panel.write_parquet(BASE / "7_processed_data" / "panel_v1.parquet")

pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_width_chars(150)
print(f"\nPanel: {panel.height:,} rows, {flows.height:,} flows, {panel['exporter'].n_unique()} exporters, "
      f"{panel['cn8'].n_unique():,} CN8 codes")
print("\nFlows by group and market, 2024 value (EUR bn) and share of zero months:")
print(panel.with_columns(pl.when(pl.col("us")).then(pl.lit("US")).otherwise(pl.lit("Other")).alias("market"))
      .group_by(["group_v4", "market"])
      .agg((pl.len() / months.height).cast(pl.Int64).alias("flows"),
           (pl.col("value_eur").filter(pl.col("period").str.starts_with("2024")).sum() / 1e9).round(2).alias("bn_2024"),
           (100 * (pl.col("value_eur") == 0).mean()).round(1).alias("pct_zero_months"))
      .sort(["group_v4", "market"]))

late = X.filter(pl.col("period") >= "202504")
cov = (late.join(flows.with_columns(pl.lit(True).alias("kept")), on=["exporter", "cn8", "dest"], how="left")
           .with_columns(pl.col("kept").fill_null(False), (pl.col("dest") == "US").alias("us"))
           .group_by("us").agg((100 * pl.col("value_eur").filter(pl.col("kept")).sum()
                                / pl.col("value_eur").sum()).round(1).alias("pct_of_AprJun25_26_value_kept")))
print("\nShare of Apr 2025 - Jun 2026 export value covered by flows that existed in 2023-2024:")
print(cov.sort("us"))

yr = X.with_columns(pl.col("period").str.slice(0, 4).alias("year")).group_by(["cn8", "year"]).agg(
        pl.col("value_eur").sum())
codes = {y: set(yr.filter(pl.col("year") == y)["cn8"].to_list()) for y in ["2023", "2024", "2025", "2026"]}
for a, b in [("2023", "2024"), ("2024", "2025"), ("2025", "2026")]:
    print(f"CN8 codes in {a} but not {b}: {len(codes[a] - codes[b])}; new in {b}: {len(codes[b] - codes[a])}")
