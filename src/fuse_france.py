"""Fuse the unseen-country matcher (LightGBM) with the cross-encoder scores on its uncertain band.

Validated on the France-like simulation (Latin-only India, true labels): F0.5 0.9044 (LightGBM alone) ->
0.9325 with ranks fused 0.2 LightGBM + 0.6 teacher-student cross-encoder + 0.2 plain cross-encoder
(0.3 / 0.7 without the plain one: 0.9278). The fused ranks are mapped back onto LightGBM's own probability values
inside the band, so the band keeps its probability distribution and only the ORDER of pairs changes -
decision thresholds stay comparable.

    python src/fuse_france.py --lgbm work/test_scored_unseen_roles.parquet \
        --student work/tgt_scores_france_ts.parquet [--plain work/tgt_scores_france.parquet]
-> work/test_scored_unseen_fused.parquet   (then run_pipeline.py --reuse ... --unseen_scores <it>)
"""
import argparse
import os

import numpy as np
import polars as pl
from scipy.stats import rankdata

import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgbm", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--plain", default="")
    ap.add_argument("--band", default=os.path.join(C.WORK_DIR, "da_france", "tgt_pairs.parquet"))
    ap.add_argument("--out", default=os.path.join(C.WORK_DIR, "test_scored_unseen_fused.parquet"))
    a = ap.parse_args()
    lg = pl.read_parquet(a.lgbm)
    band = pl.read_parquet(a.band).select("ka", "kb", "i", "j").with_columns(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32))
    b = (band.join(lg, on=["i", "j"], how="inner")
             .join(pl.read_parquet(a.student).rename({"ce": "ce_ts"}), on=["ka", "kb"], how="inner"))
    if a.plain:
        b = b.join(pl.read_parquet(a.plain).rename({"ce": "ce_plain"}), on=["ka", "kb"], how="inner")
    n = b.height
    R = lambda x: rankdata(x) / len(x)
    if a.plain:
        fused = 0.2 * R(b["prob"].to_numpy()) + 0.6 * R(b["ce_ts"].to_numpy()) + 0.2 * R(b["ce_plain"].to_numpy())
    else:
        fused = 0.3 * R(b["prob"].to_numpy()) + 0.7 * R(b["ce_ts"].to_numpy())
    qs = np.sort(b["prob"].to_numpy())
    newp = qs[np.clip((rankdata(fused) - 1).astype(int), 0, n - 1)].astype(np.float32)
    upd = b.select("i", "j").with_columns(pl.Series("p_new", newp))
    out = lg.join(upd, on=["i", "j"], how="left").with_columns(pl.coalesce("p_new", "prob").alias("prob")).drop("p_new")
    out.write_parquet(a.out)
    moved = (np.abs(newp - b["prob"].to_numpy()) > 0.2).mean()
    print(f"band pairs fused: {n} of {lg.height} France pairs ({'3-way' if a.plain else '2-way'}) | "
          f"pairs whose probability moved by > 0.2: {moved:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
