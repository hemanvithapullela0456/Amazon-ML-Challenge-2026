"""run31 France: new students (retrained on run30's cleaned pseudo-labels) everywhere EXCEPT pairs whose copy carries a
different house number than the S1 (same-street shifted numbers are the France decoy pattern; train P(true) for
small shifts ~0.34-0.51, below the F0.5 break-even). Those pairs keep run30's scores.

    python src/merge_students_shift.py --new _c_all_u_own2.parquet --old test_scored_unseen_decoy_all_u_own2.parquet --out <f>
"""
import argparse
import os

import polars as pl

import config as C
from moves import move_features
from prep import load_norm

W = C.WORK_DIR

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    new = pl.read_parquet(os.path.join(W, a.new))
    old = pl.read_parquet(os.path.join(W, a.old)).rename({"prob": "p_old"})
    x = new.join(old, on=["i", "j"], how="left")
    cand = x.filter((pl.col("prob") >= 0.3) | (pl.col("p_old") >= 0.3)).select("i", "j", "prob")
    norm = load_norm("test", columns=["name_core", "addr_core", "addr_nums", "business_name", "business_address", "source"]).with_row_index("r")
    m = move_features(cand, norm)
    shift = m.filter((pl.col("mv_addr_dropped") == 0) & pl.col("mv_hn_delta").is_not_null() & (pl.col("mv_hn_equal") == 0))
    shift = shift.select("i", "j", pl.lit(True).alias("shift"))
    x = x.join(shift, on=["i", "j"], how="left").with_columns(pl.col("shift").fill_null(False))
    x = x.with_columns(pl.when(pl.col("shift")).then(pl.col("p_old")).otherwise(pl.col("prob")).cast(new.schema["prob"]).alias("prob"))
    print(f"pairs scored: {x.height} | shifted-number pairs kept at run30 scores: {int(x['shift'].sum())}")
    x.select("i", "j", "prob").write_parquet(os.path.join(W, a.out))
    print(f"-> work/{a.out}")
