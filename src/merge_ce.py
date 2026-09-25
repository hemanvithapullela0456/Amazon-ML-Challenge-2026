"""Combine per-fold cross-encoder outputs (one process / Kaggle account per fold) into the two files stack.py reads.

    python src/merge_ce.py --tag _xlmr
reads  work/ce_train_xlmr_f{0,1,2}.parquet and work/ce_test_xlmr_f{0,1,2}.parquet
writes work/ce_train.parquet (each bundle pair scored out-of-fold by exactly one fold model)
       work/ce_test.parquet  (mean of the three fold models' test scores)
"""
import argparse
import os
import sys

import numpy as np
import polars as pl

import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--dir", default=C.WORK_DIR)
    ap.add_argument("--out_tag", default="", help="suffix for the merged files (default none -> ce_train.parquet)")
    a = ap.parse_args()

    tr, te = [], []
    for f in range(a.folds):
        p_tr = os.path.join(a.dir, f"ce_train{a.tag}_f{f}.parquet")
        p_te = os.path.join(a.dir, f"ce_test{a.tag}_f{f}.parquet")
        for p in (p_tr, p_te):
            if not os.path.exists(p):
                sys.exit(f"missing {p}")
        tr.append(pl.read_parquet(p_tr))
        te.append(pl.read_parquet(p_te))

    base = tr[0].select("i", "j")
    for t in tr[1:]:
        assert t.select("i", "j").equals(base), "train parts list different pairs"
    ce = np.stack([t["ce"].to_numpy() for t in tr])          # folds x pairs; NaN outside a fold's validation part
    n_scored = (~np.isnan(ce)).sum(axis=0)
    assert (n_scored == 1).all(), f"{int((n_scored != 1).sum())} pairs were not scored by exactly one fold"
    out_tr = base.with_columns(pl.Series("ce", np.nansum(ce, axis=0).astype(np.float32)))

    base_te = te[0].select("i", "j")
    for t in te[1:]:
        assert t.select("i", "j").equals(base_te), "test parts list different pairs"
    out_te = base_te.with_columns(
        pl.Series("ce", np.mean(np.stack([t["ce"].to_numpy() for t in te]), axis=0).astype(np.float32)))

    out_tr.write_parquet(os.path.join(a.dir, f"ce_train{a.out_tag}.parquet"))
    out_te.write_parquet(os.path.join(a.dir, f"ce_test{a.out_tag}.parquet"))
    print(f"merged {a.folds} folds -> ce_train{a.out_tag}.parquet ({out_tr.height} pairs, all scored), "
          f"ce_test{a.out_tag}.parquet ({out_te.height} pairs)")


if __name__ == "__main__":
    main()
